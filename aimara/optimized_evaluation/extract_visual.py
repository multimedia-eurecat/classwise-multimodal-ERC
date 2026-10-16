"""Batched visual embeddings using repository source and explicit local assets."""

import os
from .eval_metrics import phase, check_embeddings
from .visual_cache import _load_or_build_visual_cache, _read_unique_crops
import sys
import argparse
import hashlib
import json
import pickle
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

CSV_PATH = os.environ["EVAL_CSV_PATH"]
VIDEO_DIR = os.environ["EVAL_VIDEO_DIR"]
OUT_DIR = os.environ["EVAL_EMB_DIR"]
ASD_WEIGHTS = os.environ["EVAL_ASD_WEIGHTS"]
VIT_MODEL_ID = os.environ["EVAL_VIT_MODEL"]
DEVICE = os.environ.get("EVAL_DEVICE", "auto")
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16
FACE_MARGIN = 0.2
from unimodal.visual.models import VisualConfig, _pool_landmarks
from .active_speaker import LightASDWrapper


def load_visual_models(cfg):
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=True, device=cfg.device)
    if cfg.visual_mode == "landmarks":
        from .landmarks import SparseLandmarkModel
        processor = None
        model = SparseLandmarkModel(os.environ["EVAL_LANDMARK_MODEL"], os.environ["EVAL_LANDMARK_BFM"], os.environ["EVAL_LANDMARK_NORMALIZATION"])
        dimension = 408
    else:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        processor = AutoImageProcessor.from_pretrained(VIT_MODEL_ID, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_ID, local_files_only=True).to(cfg.device).eval()
        dimension = model.config.hidden_size
    asd = LightASDWrapper(ASD_WEIGHTS, cfg.device)
    return mtcnn, processor, model, dimension, asd


VIT_BATCH_SIZE = int(os.environ.get("EVAL_VIT_BATCH_SIZE", "32"))
if VIT_BATCH_SIZE < 1:
    raise ValueError("EVAL_VIT_BATCH_SIZE must be positive")
FACE_BATCH_SIZE = int(os.environ.get("EVAL_FACE_BATCH_SIZE", "8"))


@torch.no_grad()
def _extract_video_vit(crops, processor, model, device, embed_dim, utterance_ids):
    vectors = {uid: [] for uid in utterance_ids}
    for offset in range(0, len(crops), VIT_BATCH_SIZE):
        batch = crops[offset:offset + VIT_BATCH_SIZE]
        inputs = processor(
            images=[Image.fromarray(crop) for _, crop in batch],
            return_tensors="pt",
        ).to(device)
        output = model(**inputs, output_hidden_states=True)
        cls = output.hidden_states[-1][:, 0, :].detach().cpu().numpy()
        for (uids, _), vector in zip(batch, cls):
            for uid in uids:
                vectors[uid].append(vector)
    return {
        uid: (np.mean(np.stack(values), axis=0).astype(np.float32)
              if values else np.zeros(embed_dim, dtype=np.float32))
        for uid, values in vectors.items()
    }


def _extract_video_landmarks(crops, tddfa, embed_dim, utterance_ids):
    landmarks = {uid: [] for uid in utterance_ids}
    for uids, crop in crops:
        points = tddfa.extract(crop)
        if points is not None:
            for uid in uids:
                landmarks[uid].append(points)
    return {uid: _pool_landmarks(values, embed_dim) for uid, values in landmarks.items()}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["vit", "landmarks", "action_units"],
                    required=True, help="Which visual encoder to use.")
    ap.add_argument("--output", default=None,
                    help="Output PKL path. Defaults to embeddings/visual_{encoder}.pkl")
    args = ap.parse_args()

    out_pkl = args.output or os.path.join(OUT_DIR, f"visual_{args.encoder}.pkl")
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)

    # Load CSV
    df = pd.read_csv(CSV_PATH, sep=";")
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} utterances from {df['video_file'].nunique()} videos")

    # Assign dialogue_id and utterance_id (same scheme as audio/text)
    video_files  = sorted(df["video_file"].unique())
    video_to_did = {v: i for i, v in enumerate(video_files)}
    df["dialogue_id"]  = df["video_file"].map(video_to_did)
    df["utterance_id"] = df.groupby("dialogue_id").cumcount()

    if args.encoder == "action_units":
        raise ValueError("Use the separate AU worker")
    cfg = VisualConfig(device=DEVICE, visual_mode=args.encoder,
                       model_id=VIT_MODEL_ID, num_frames=NUM_FRAMES,
                       face_margin=FACE_MARGIN, use_fullframe_fallback=False,
                       asd_weights=ASD_WEIGHTS, asd_threshold=0.5,
                       asd_iou_threshold=0.5, asd_max_seconds=60.0)

    # Load models
    print(f"\nLoading visual models (encoder={args.encoder}) ...")
    stage_start = time.perf_counter()
    with phase("model_load"):
        mtcnn, processor, model, embed_dim, asd = load_visual_models(cfg)
    print(f"Models loaded. embed_dim={embed_dim}")

    # Detect/track each video once. The cache is shared by ViT and landmarks.
    with phase("visual_preprocessing"):
        cache = _load_or_build_visual_cache(df, mtcnn, asd, cfg)

    # Decode sampled frames once per video, then batch/reuse per-frame features.
    embeddings = {}
    t_start = stage_start
    for vfile, rows in df.groupby("video_file", sort=True):
        did = int(rows.iloc[0]["dialogue_id"])
        utterance_ids = [int(uid) for uid in rows["utterance_id"]]
        video_path = os.path.join(VIDEO_DIR, vfile)
        t0 = time.perf_counter()
        with phase("crop_decode"):
            crops = _read_unique_crops(
                video_path, cache["indices"][vfile],
                cache["speaker_boxes"][vfile], cfg.face_margin,
            )
        with phase("encoder_inference"):
            if cfg.visual_mode == "landmarks":
                video_embeddings = _extract_video_landmarks(
                    crops, model, embed_dim, utterance_ids
                )
            else:  # vit
                video_embeddings = _extract_video_vit(
                    crops, processor, model, cfg.device, embed_dim, utterance_ids
                )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        embeddings[did] = video_embeddings
        print(
            f"  [{vfile}] {len(utterance_ids)} utterances, "
            f"{len(crops)} unique crops ({elapsed_ms:.0f}ms)",
            flush=True,
        )

    total_s = time.perf_counter() - t_start

    check_embeddings(embeddings, embed_dim)

    # Save
    with open(out_pkl, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_utts = sum(len(v) for v in embeddings.values())
    print(f"\nSaved: {out_pkl}")
    print(f"  dialogues={len(embeddings)}  utterances={n_utts}  "
          f"dim={embed_dim}  total={total_s:.1f}s")


if __name__ == "__main__":
    main()
