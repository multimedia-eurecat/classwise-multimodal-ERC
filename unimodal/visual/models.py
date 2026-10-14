import os
import re
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from facenet_pytorch import MTCNN
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .utils import (
    sample_window_frame_indices,
    read_frame_at,
    pick_face_index,
    apply_margin_xyxy,
)

IEMOCAP_TRAIN_SESSIONS = [1, 2, 3, 4]
IEMOCAP_TEST_SESSIONS  = [5]
_TRANS_LINE_RE = re.compile(r"^(\S+)\s+\[\d+\.\d+-\d+\.\d+\]:\s*(.*)")
_EMO_LINE_RE   = re.compile(r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)")


@dataclass
class VisualConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    visual_mode: str = "vit"

    model_id: str = "dima806/facial_emotions_image_detection"

    tddfa_root: str = ""
    tddfa_onnx_path: str = ""

    # ── Shared settings ───────────────────────────────────────────────────────
    num_frames: int = 16
    keep_all: bool = True
    face_margin: float = 0.2
    use_fullframe_fallback: bool = False
    scene_cut_threshold: float = 0.3

    # ── Active Speaker Detection ──────────────────────────────────────────────
    asd_weights: Optional[str] = None
    asd_threshold: float = 0.5
    asd_iou_threshold: float = 0.5
    asd_max_seconds: Optional[float] = 60.0


def load_visual_models(cfg: VisualConfig):
    """
    Returns: (mtcnn, processor, model, embed_dim, asd)

    vit mode:       processor = AutoImageProcessor, model = ViT, embed_dim = hidden_size
    landmarks mode: processor = None,               model = TDDFA_ONNX,  embed_dim = 408
                    (408 = 68 landmarks × 3 coords × 2 stats [mean, std])
    """
    device = cfg.device
    mtcnn  = MTCNN(keep_all=cfg.keep_all, device=device)

    if cfg.visual_mode == "landmarks":
        tddfa    = _load_tddfa(cfg)
        embed_dim = 68 * 3 * 2  # mean + std over 68×3 normalized coordinates → 408
        processor = None
        model     = tddfa
    elif cfg.visual_mode == "landmarks_exp":
        tddfa    = _load_tddfa(cfg)
        # 10 expression coefficients + 3 pose angles = 13-d per frame
        # pooled with mean + std + delta → 39-d per utterance
        embed_dim = 13 * 3
        processor = None
        model     = tddfa
    else:
        processor = AutoImageProcessor.from_pretrained(cfg.model_id)
        model     = AutoModelForImageClassification.from_pretrained(
            cfg.model_id, low_cpu_mem_usage=True
        ).to(device).eval()
        embed_dim = getattr(model.config, "hidden_size", None) or getattr(model.config, "dim", None)
        if embed_dim is None:
            dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            with torch.no_grad():
                inputs = processor(images=dummy, return_tensors="pt").to(device)
                out    = model(**inputs, output_hidden_states=True)
                embed_dim = int(out.hidden_states[-1].shape[-1])
        else:
            embed_dim = int(embed_dim)

    asd = None
    if cfg.asd_weights is not None:
        from .asd import LightASDWrapper
        print(f"  Loading Light-ASD from: {cfg.asd_weights}")
        asd = LightASDWrapper(weights_path=cfg.asd_weights, device=device)
        print("  Light-ASD loaded ✓")

    return mtcnn, processor, model, embed_dim, asd


# ── 3DDFA-V2 landmark helpers ─────────────────────────────────────────────────

def _load_tddfa(cfg: VisualConfig):
    import sys
    import yaml

    if not cfg.tddfa_root:
        raise ValueError("--tddfa_root must point to the cloned 3DDFA_V2 directory")
    if not cfg.tddfa_onnx_path:
        raise ValueError("--tddfa_onnx_path must point to the .onnx weights file")
    if not os.path.isdir(cfg.tddfa_root):
        raise FileNotFoundError(f"tddfa_root not found: {cfg.tddfa_root}")
    if not os.path.isfile(cfg.tddfa_onnx_path):
        raise FileNotFoundError(f"tddfa_onnx_path not found: {cfg.tddfa_onnx_path}")

    if cfg.tddfa_root not in sys.path:
        sys.path.insert(0, cfg.tddfa_root)

    try:
        from TDDFA_ONNX import TDDFA_ONNX
    except ImportError:
        raise ImportError(
            "Could not import TDDFA_ONNX. Make sure:\n"
            "  1. cfg.tddfa_root points to the cloned 3DDFA_V2 repo\n"
            "  2. onnxruntime is installed: pip install onnxruntime\n"
        )

    onnx_name = os.path.basename(cfg.tddfa_onnx_path)
    if onnx_name.startswith("mb1"):
        yml_name = "mb1_120x120.yml"
    elif onnx_name.startswith("mb05"):
        yml_name = "mb05_120x120.yml"
    elif onnx_name.startswith("resnet22"):
        yml_name = "resnet22.yml"
    else:
        yml_name = "mb1_120x120.yml"

    cfg_path = os.path.join(cfg.tddfa_root, "configs", yml_name)
    tddfa_cfg = yaml.load(open(cfg_path), Loader=yaml.SafeLoader)
    tddfa_cfg["checkpoint_fp"] = cfg.tddfa_onnx_path

    # 3DDFA_V2 uses relative paths internally (e.g. 'configs/bfm_noneck_v3.pkl')
    # so we must cd into the repo before instantiating
    orig_dir = os.getcwd()
    try:
        os.chdir(cfg.tddfa_root)
        tddfa = TDDFA_ONNX(**tddfa_cfg)
    finally:
        os.chdir(orig_dir)

    print(f"  3DDFA-V2 ONNX loaded: {onnx_name}")
    return tddfa


# 68-point landmark indices used for normalization
_NOSE_TIP_IDX       = 30   # nose tip — used as coordinate origin
_LEFT_EYE_OUTER_IDX = 36   # left eye outer corner
_RIGHT_EYE_OUTER_IDX = 45  # right eye outer corner


def _normalize_landmarks(lmks: np.ndarray) -> np.ndarray:
    """
    Pose- and scale-normalize 68 3D landmarks.
      - Center: subtract nose tip (landmark 30)
      - Scale:  divide by inter-ocular distance (landmarks 36 ↔ 45)

    lmks: (68, 3) float32
    Returns: (68, 3) float32
    """
    nose      = lmks[_NOSE_TIP_IDX]
    left_eye  = lmks[_LEFT_EYE_OUTER_IDX]
    right_eye = lmks[_RIGHT_EYE_OUTER_IDX]

    centered  = lmks - nose
    iol_dist  = float(np.linalg.norm(left_eye - right_eye)) + 1e-6
    return (centered / iol_dist).astype(np.float32)


# ── Expression coefficient helpers (landmarks_exp mode) ──────────────────────
# 62-d param vector layout for mb1_120x120 config:
#   params[  0:12] → camera matrix (rotation + translation) → used for pose
#   params[ 12:52] → shape coefficients (40-d) — face identity, NOT used
#   params[ 52:62] → expression coefficients (10-d) — facial deformation only
_EXP_START = 52
_EXP_END   = 62


def _extract_pose_angles(params: np.ndarray) -> np.ndarray:
    """
    Extract [pitch, yaw, roll] in degrees from 3DMM params using 3DDFA-V2's
    own calc_pose (handles P2sRt decomposition correctly).
    Returns (3,) float32.
    """
    from utils.pose import calc_pose
    _, pose_deg = calc_pose(params)   # [pitch, yaw, roll] in degrees
    return np.array(pose_deg, dtype=np.float32)


def _extract_exp_frame(crop_rgb: np.ndarray, tddfa) -> Optional[np.ndarray]:
    """
    Run 3DDFA-V2 and return a 13-d feature vector per frame:
      params[52:62] — 10 expression coefficients (deformation only,
                      identity and pose factored out by the 3DMM)
      pose angles   — 3 values [pitch, yaw, roll] in degrees

    WHY THIS IS BETTER THAN RAW LANDMARKS:
    recon_vers() reconstructs: landmarks = u_base + w_shp@shape + w_exp@exp
    This mixes identity (w_shp@shape) into coordinates, so two people making
    the same smile look different. Expression coefficients are extracted before
    that mixing — same expression from different speakers → similar vectors.
    """
    h, w = crop_rgb.shape[:2]
    if h < 16 or w < 16:
        return None
    box = [0.0, 0.0, float(w - 1), float(h - 1), 1.0]
    try:
        param_lst, _ = tddfa(crop_rgb, [box])
        params = param_lst[0]
        exp  = params[_EXP_START:_EXP_END]      # (10,)
        pose = _extract_pose_angles(params)      # (3,)
        return np.concatenate([exp, pose]).astype(np.float32)  # (13,)
    except Exception:
        return None


def _pool_exp(feats_list: List[np.ndarray], embed_dim: int) -> np.ndarray:
    """
    Aggregate per-frame 13-d vectors → 39-d utterance embedding:
      mean  (13,) — average expression/pose over the utterance
      std   (13,) — expressivity: how much the face moved
      delta (13,) — mean frame-to-frame change: direction of movement

    Mean+std+delta captures both WHAT expression was held and HOW it changed,
    so neutral→angry looks different from angry→neutral (delta sign differs).
    """
    if not feats_list:
        return np.zeros((embed_dim,), dtype=np.float32)
    stack = np.stack([f.flatten() for f in feats_list], axis=0)  # (N, 13)
    mean  = stack.mean(axis=0)                        # (13,)
    std   = stack.std(axis=0)                         # (13,)
    if len(stack) > 1:
        delta = np.diff(stack, axis=0).mean(axis=0)  # (13,)
    else:
        delta = np.zeros(stack.shape[1], dtype=np.float32)  # (13,)
    return np.concatenate([mean, std, delta]).astype(np.float32)  # (39,)


def _extract_landmark_frame(crop_rgb: np.ndarray, tddfa) -> Optional[np.ndarray]:
    """
    Run 3DDFA-V2 on an already-cropped face image.
    Returns normalized (68, 3) landmarks, or None on failure.

    crop_rgb: (H, W, 3) uint8 — face crop from MTCNN or ASD
    """
    h, w = crop_rgb.shape[:2]
    if h < 16 or w < 16:
        return None

    # 3DDFA-V2 expects the full image + bounding box list.
    # Since we already have the face crop, use a synthetic box covering the crop.
    box = [0.0, 0.0, float(w - 1), float(h - 1), 1.0]

    try:
        param_lst, roi_box_lst = tddfa(crop_rgb, [box])
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
        lmks = ver_lst[0].T  # (3, 68) → (68, 3)
        return _normalize_landmarks(lmks)
    except Exception:
        return None


@torch.no_grad()
def extract_utterance_landmark_embedding(
    video_path: str, mtcnn: MTCNN, tddfa,
    embed_dim: int, num_frames: int = 16,
    face_margin: float = 0.2, use_fullframe_fallback: bool = False,
    scene_cut_threshold: float = 0.3, asd=None, asd_threshold: float = 0.5,
    asd_iou_threshold: float = 0.5, asd_max_seconds: Optional[float] = 60.0,
    start_sec: Optional[float] = None, end_sec: Optional[float] = None,
    visual_mode: str = "landmarks",
) -> np.ndarray:
    """
    Returns landmark embedding for one utterance.

    visual_mode="landmarks"    : 68 normalized 3D landmarks, mean+std → (408,)
    visual_mode="landmarks_exp": expression coefficients + pose, mean+std+delta → (39,)

    Face selection logic (MTCNN → IoU heuristic / ASD) is identical in both modes.
    The only difference is the per-crop feature extraction step.
    """
    # Select per-frame extractor and pooling function based on mode
    if visual_mode == "landmarks_exp":
        frame_fn = _extract_exp_frame
        pool_fn  = _pool_exp
    else:
        frame_fn = _extract_landmark_frame
        pool_fn  = _pool_landmarks
    zero = np.zeros((embed_dim,), dtype=np.float32)

    if not os.path.exists(video_path):
        return zero

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return zero

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    idxs = sample_window_frame_indices(total, fps, num_frames, start_sec, end_sec)
    if idxs.size == 0:
        cap.release()
        return zero

    # ── ASD path ──────────────────────────────────────────────────────────────
    if asd is not None:
        cap.release()
        speaker_boxes = asd.get_speaker_boxes(
            video_path=video_path, frame_indices=idxs, mtcnn=mtcnn,
            iou_threshold=asd_iou_threshold, asd_threshold=asd_threshold,
            max_seconds=None if end_sec is not None else asd_max_seconds,
            start_sec=start_sec or 0.0,
            end_sec=end_sec,
        )
        return _landmarks_with_boxes(
            video_path=video_path, frame_indices=idxs,
            speaker_boxes=speaker_boxes, tddfa=tddfa,
            embed_dim=embed_dim, face_margin=face_margin,
            use_fullframe_fallback=use_fullframe_fallback,
            frame_fn=frame_fn, pool_fn=pool_fn,
            frame_transform=None,
        )

    # ── IoU heuristic path ────────────────────────────────────────────────────
    lmks_list: List[np.ndarray] = []
    prev_box:  Optional[np.ndarray] = None
    prev_hist: Optional[np.ndarray] = None

    for fi in idxs:
        frame_rgb = read_frame_at(cap, int(fi))
        if frame_rgb is None:
            continue

        curr_hist = _frame_histogram(frame_rgb)
        if prev_hist is not None and _is_scene_cut(prev_hist, curr_hist):
            prev_box = None
        prev_hist = curr_hist

        h, w  = frame_rgb.shape[:2]
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is None or len(boxes) == 0:
            if use_fullframe_fallback:
                lmk = frame_fn(frame_rgb, tddfa)
                if lmk is not None:
                    lmks_list.append(lmk)
                prev_box = None
            continue

        sel = pick_face_index(boxes, prev_box)
        if sel < 0:
            if use_fullframe_fallback:
                lmk = frame_fn(frame_rgb, tddfa)
                if lmk is not None:
                    lmks_list.append(lmk)
                prev_box = None
            continue

        prev_box = boxes[sel]
        x1, y1, x2, y2 = apply_margin_xyxy(prev_box, face_margin, w=w, h=h)
        crop = frame_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            if use_fullframe_fallback:
                lmk = frame_fn(frame_rgb, tddfa)
                if lmk is not None:
                    lmks_list.append(lmk)
                prev_box = None
            continue

        lmk = frame_fn(crop, tddfa)
        if lmk is not None:
            lmks_list.append(lmk)

    cap.release()
    return pool_fn(lmks_list, embed_dim)


def _landmarks_with_boxes(
    video_path: str, frame_indices: np.ndarray, speaker_boxes: dict, tddfa,
    embed_dim: int, face_margin: float, use_fullframe_fallback: bool,
    frame_fn=None, pool_fn=None, frame_transform=None,
) -> np.ndarray:
    """ASD path: given pre-computed speaker boxes, extract landmarks per frame."""
    if frame_fn is None:
        frame_fn = _extract_landmark_frame
    if pool_fn is None:
        pool_fn = _pool_landmarks

    cap = cv2.VideoCapture(video_path)
    lmks_list: List[np.ndarray] = []

    for fi in frame_indices:
        fi        = int(fi)
        frame_rgb = read_frame_at(cap, fi)
        if frame_rgb is None:
            continue
        if frame_transform is not None:
            frame_rgb = frame_transform(frame_rgb)

        box  = speaker_boxes.get(fi)
        h, w = frame_rgb.shape[:2]

        if box is not None:
            x1, y1, x2, y2 = apply_margin_xyxy(box, face_margin, w=w, h=h)
            crop = frame_rgb[y1:y2, x1:x2]
            src  = crop if crop.size > 0 else (frame_rgb if use_fullframe_fallback else None)
        else:
            src = frame_rgb if use_fullframe_fallback else None

        if src is None:
            continue

        lmk = frame_fn(src, tddfa)
        if lmk is not None:
            lmks_list.append(lmk)

    cap.release()
    return pool_fn(lmks_list, embed_dim)


def _pool_landmarks(lmks_list: List[np.ndarray], embed_dim: int) -> np.ndarray:
    """
    Mean + std pool a list of (68, 3) landmark arrays → (408,).
    Returns zeros if no valid frames.
    """
    if not lmks_list:
        return np.zeros((embed_dim,), dtype=np.float32)
    stack = np.stack(lmks_list, axis=0)          # (N, 68, 3)
    mean  = stack.mean(axis=0).flatten()          # (204,)
    std   = stack.std(axis=0).flatten()           # (204,)
    return np.concatenate([mean, std]).astype(np.float32)  # (408,)


# ── Scene cut detection ───────────────────────────────────────────────────────

def _frame_histogram(frame_rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    """Computes a normalised HSV histogram for scene cut detection."""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None, [bins, bins],
        [0, 180, 0, 256],
    ).flatten().astype(np.float32)
    total = hist.sum()
    return hist / (total + 1e-9)


def _is_scene_cut(hist_a: np.ndarray, hist_b: np.ndarray) -> bool:
    """Chi-squared distance between two normalised histograms."""
    diff = hist_a - hist_b
    denom = hist_a + hist_b + 1e-9
    chi2 = float(np.sum(diff ** 2 / denom))
    return chi2 > 0.3   # conservative threshold for Friends clips


# ── Embedding helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def _embed_image(img: Image.Image, processor, model, device: str) -> Optional[np.ndarray]:
    """Runs the emotion ViT on a PIL image, returns CLS embedding (D,)."""
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True)
    last = out.hidden_states[-1]   # (1, tokens, D)
    cls  = last[:, 0, :]           # (1, D)
    return cls.squeeze(0).detach().cpu().numpy()


# ── Main extraction ───────────────────────────────────────────────────────────

@torch.no_grad()
def extract_utterance_visual_embedding(
    video_path: str, mtcnn: MTCNN, processor, model,
    device: str, embed_dim: int, num_frames: int = 16,
    face_margin: float = 0.2, use_fullframe_fallback: bool = False,
    scene_cut_threshold: float = 0.3, asd=None, asd_threshold: float = 0.5,
    asd_iou_threshold: float = 0.5, asd_max_seconds: Optional[float] = 60.0,
    start_sec: Optional[float] = None, end_sec: Optional[float] = None,
    frame_transform=None,
    ) -> np.ndarray:

    """
    Returns float32 (D,) ViT embedding for one utterance clip.
    start_sec/end_sec: sample only timestamps in [start_sec, end_sec).
    """
    if not os.path.exists(video_path):
        return np.zeros((embed_dim,), dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((embed_dim,), dtype=np.float32)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    idxs = sample_window_frame_indices(total, fps, num_frames, start_sec, end_sec)
    if idxs.size == 0:
        cap.release()
        return np.zeros((embed_dim,), dtype=np.float32)

    # ── ASD path ──────────────────────────────────────────────────────────────
    if asd is not None:
        cap.release()
        speaker_boxes = asd.get_speaker_boxes(
            video_path=video_path,
            frame_indices=idxs,
            mtcnn=mtcnn,
            iou_threshold=asd_iou_threshold,
            asd_threshold=asd_threshold,
            max_seconds=None if end_sec is not None else asd_max_seconds,
            start_sec=start_sec or 0.0,
            end_sec=end_sec,
        )
        return _embed_with_boxes(
            video_path=video_path,
            frame_indices=idxs,
            speaker_boxes=speaker_boxes,
            processor=processor,
            model=model,
            device=device,
            embed_dim=embed_dim,
            face_margin=face_margin,
            use_fullframe_fallback=use_fullframe_fallback,
            frame_transform=frame_transform,
        )

    # ── IoU heuristic path with scene cut detection ───────────────────────────
    embs: List[np.ndarray] = []
    prev_box:  Optional[np.ndarray] = None
    prev_hist: Optional[np.ndarray] = None

    for fi in idxs:
        frame_rgb = read_frame_at(cap, int(fi))
        if frame_rgb is None:
            continue
        if frame_transform is not None:
            frame_rgb = frame_transform(frame_rgb)

        # Scene cut detection: reset tracking if histogram distance is large
        curr_hist = _frame_histogram(frame_rgb)
        if prev_hist is not None and _is_scene_cut(prev_hist, curr_hist):
            prev_box = None   # drop tracking — new scene, unknown face
        prev_hist = curr_hist

        h, w = frame_rgb.shape[:2]
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is None or len(boxes) == 0:
            if use_fullframe_fallback:
                emb = _embed_image(Image.fromarray(frame_rgb), processor, model, device)
                if emb is not None:
                    embs.append(emb)
                prev_box = None
            # else: skip this frame entirely
            continue

        sel = pick_face_index(boxes, prev_box)
        if sel < 0:
            if use_fullframe_fallback:
                emb = _embed_image(Image.fromarray(frame_rgb), processor, model, device)
                if emb is not None:
                    embs.append(emb)
                prev_box = None
            continue

        prev_box = boxes[sel]
        x1, y1, x2, y2 = apply_margin_xyxy(prev_box, face_margin, w=w, h=h)
        crop = frame_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            if use_fullframe_fallback:
                emb = _embed_image(Image.fromarray(frame_rgb), processor, model, device)
                if emb is not None:
                    embs.append(emb)
                prev_box = None
            continue

        emb = _embed_image(Image.fromarray(crop), processor, model, device)
        if emb is not None:
            embs.append(emb)

    cap.release()

    if not embs:
        return np.zeros((embed_dim,), dtype=np.float32)
    return np.mean(np.stack(embs), axis=0).astype(np.float32)


def _embed_with_boxes(
    video_path: str, frame_indices: np.ndarray, speaker_boxes: dict, processor,
    model, device: str, embed_dim: int, face_margin: float,
    use_fullframe_fallback: bool, frame_transform=None,
    ) -> np.ndarray:
    """Given pre-computed ASD speaker boxes, crops and embeds each frame."""
    cap  = cv2.VideoCapture(video_path)
    embs: List[np.ndarray] = []

    for fi in frame_indices:
        fi        = int(fi)
        frame_rgb = read_frame_at(cap, fi)
        if frame_rgb is None:
            continue
        if frame_transform is not None:
            frame_rgb = frame_transform(frame_rgb)

        box   = speaker_boxes.get(fi)
        h, w  = frame_rgb.shape[:2]

        if box is not None:
            x1, y1, x2, y2 = apply_margin_xyxy(box, face_margin, w=w, h=h)
            crop = frame_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                if use_fullframe_fallback:
                    img = Image.fromarray(frame_rgb)
                else:
                    continue
            else:
                img = Image.fromarray(crop)
        else:
            if use_fullframe_fallback:
                img = Image.fromarray(frame_rgb)
            else:
                continue

        emb = _embed_image(img, processor, model, device)
        if emb is not None:
            embs.append(emb)

    cap.release()

    if not embs:
        return np.zeros((embed_dim,), dtype=np.float32)
    return np.mean(np.stack(embs), axis=0).astype(np.float32)

# ── IEMOCAP loader ────────────────────────────────────────────────────────────

import pandas as pd

def _parse_transcription_visual(path: Path) -> list[tuple[str, str]]:
    entries = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            m = _TRANS_LINE_RE.match(line)
            if m:
                entries.append((m.group(1), m.group(2)))
    return entries


def _parse_emo_eval_visual(path: Path) -> dict[str, tuple[float, float]]:
    times: dict[str, tuple[float, float]] = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _EMO_LINE_RE.match(line.strip())
            if m:
                uid = m.group(3)
                if uid not in times:
                    times[uid] = (float(m.group(1)), float(m.group(2)))
    return times


def _speaker_from_utt_id_visual(utt_id: str) -> str:
    part = utt_id.rsplit("_", 1)[-1]
    return part[0] if part and part[0] in ("M", "F") else "U"


def load_iemocap_visual_dataframes(
    iemocap_root: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """
    Parse IEMOCAP into (train_df, empty_dev_df, test_df, id_map).
    DataFrame columns: Dialogue_ID, Utterance_ID, Video_Path, Start_Time,
                       End_Time, Speaker, Utt_Str_ID
    Sessions 1-4 → train, Session 5 → test.
    """
    root = Path(iemocap_root)
    rows: list[dict] = []
    id_map: dict[int, str] = {}
    dialogue_counter = 0

    for split_sessions in [IEMOCAP_TRAIN_SESSIONS, IEMOCAP_TEST_SESSIONS]:
        for sess in split_sessions:
            trans_dir    = root / f"Session{sess}" / "dialog" / "transcriptions"
            emo_eval_dir = root / f"Session{sess}" / "dialog" / "EmoEvaluation"
            avi_dir      = root / f"Session{sess}" / "dialog" / "avi" / "DivX"

            for trans_file in sorted(
                f for f in trans_dir.glob("*.txt") if not f.name.startswith("._")
            ):
                dialog_name = trans_file.stem
                emo_file    = emo_eval_dir / f"{dialog_name}.txt"
                timestamps  = _parse_emo_eval_visual(emo_file) if emo_file.exists() else {}
                entries     = _parse_transcription_visual(trans_file)
                if not entries:
                    continue

                entries.sort(key=lambda x: timestamps.get(x[0], (float("inf"), 0))[0])
                avi_path = avi_dir / f"{dialog_name}.avi"
                d_int = dialogue_counter
                id_map[d_int] = dialog_name
                dialogue_counter += 1

                for u_idx, (utt_str_id, _) in enumerate(entries):
                    start, end = timestamps.get(utt_str_id, (0.0, 0.0))
                    rows.append({
                        "Dialogue_ID":  d_int,
                        "Utterance_ID": u_idx,
                        "Video_Path":   str(avi_path),
                        "Start_Time":   start,
                        "End_Time":     end,
                        "Speaker":      _speaker_from_utt_id_visual(utt_str_id),
                        "Utt_Str_ID":   utt_str_id,
                    })

    df_all = pd.DataFrame(rows)
    train_mask   = df_all["Utt_Str_ID"].str.startswith(
        tuple(f"Ses0{s}" for s in IEMOCAP_TRAIN_SESSIONS)
    )
    train_df     = df_all[train_mask].reset_index(drop=True)
    test_df      = df_all[~train_mask].reset_index(drop=True)
    empty_dev_df = pd.DataFrame(columns=df_all.columns)

    print(f"IEMOCAP loaded: {len(train_df)} train utterances, "
          f"{len(test_df)} test utterances across {dialogue_counter} dialogues.")
    return train_df, empty_dev_df, test_df, id_map
