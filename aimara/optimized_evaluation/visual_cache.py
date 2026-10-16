"""Shared per-video detection cache and crop decoding for ViT, landmarks and AU."""
import hashlib
import json
import math
import os
import pickle
import cv2
import numpy as np
from .eval_metrics import phase
from unimodal.visual.asd import build_face_tracks, audio_to_mfcc_lightasd
from unimodal.visual.utils import apply_margin_xyxy, extract_raw_audio, sample_frame_indices

VIDEO_DIR = os.environ["EVAL_VIDEO_DIR"]
OUT_DIR = os.environ["EVAL_EMB_DIR"]
NUM_FRAMES = 16
FACE_BATCH_SIZE = int(os.environ.get("EVAL_FACE_BATCH_SIZE", "8"))
if FACE_BATCH_SIZE < 1:
    raise ValueError("EVAL_FACE_BATCH_SIZE must be positive")
VISUAL_CACHE_VERSION = 3
ASD_LOOKAHEAD = float(os.environ.get("EVAL_ASD_LOOKAHEAD", "2.0"))


def aligned_audio_window(audio, frame_offset, frame_count, fps, sr=16000):
    """Match the decoded video span; retain silence when audio ends early."""
    if fps <= 0 or frame_offset < 0 or frame_count <= 0:
        raise ValueError("Invalid video window")
    start = int(round(frame_offset * sr / fps))
    end = int(round((frame_offset + frame_count) * sr / fps))
    result = np.zeros(end - start, dtype=np.float32)
    values = audio[start:end]
    result[:len(values)] = values
    return result


def _cache_signature(df):
    videos = []
    for vfile in sorted(df["video_file"].unique()):
        path = os.path.join(VIDEO_DIR, vfile)
        stat = os.stat(path)
        rows = df.loc[df["video_file"] == vfile, ["start", "end"]]
        videos.append({
            "name": vfile,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "intervals": rows.astype(float).values.tolist(),
        })
    payload = {"version": VISUAL_CACHE_VERSION, "num_frames": NUM_FRAMES, "videos": videos, "fingerprint": os.environ.get("EVAL_FINGERPRINT"), "lookahead": ASD_LOOKAHEAD}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _frame_indices(total, fps, start, end):
    frame_start = max(0, min(int(start * fps), total - 1))
    frame_end = max(frame_start + 1, min(int(end * fps), total))
    return frame_start + sample_frame_indices(frame_end - frame_start, NUM_FRAMES)


def window_bounds(total, fps, start, end, lookahead):
    offset = max(0, int(start * fps))
    stop = min(total, max(offset + 1, int(math.ceil((end + lookahead) * fps))))
    return offset, stop


def _load_or_build_visual_cache(df, mtcnn, asd, cfg):
    cache_path = os.path.join(OUT_DIR, "visual_preprocessing_cache.pkl")
    signature = _cache_signature(df)
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as handle:
            cache = pickle.load(handle)
        if cache.get("signature") == signature:
            print(f"Reusing per-video face/ASD cache: {cache_path}")
            return cache

    cache = {"signature": signature, "indices": {}, "speaker_boxes": {}}
    for vfile, rows in df.groupby("video_file", sort=True):
        video_path = os.path.join(VIDEO_DIR, vfile)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        with phase("video_decode"):
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        total = len(frames)
        if not total or not math.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"No valid video frames/FPS: {video_path}")

        all_boxes = []
        with phase("face_detection"):
            for offset in range(0, len(frames), FACE_BATCH_SIZE):
                detected, _ = mtcnn.detect(frames[offset:offset + FACE_BATCH_SIZE])
                for boxes in detected:
                    all_boxes.append(boxes if boxes is not None and len(boxes) else None)
        with phase("audio_decode"):
            audio = extract_raw_audio(video_path, sr=16000)
        if audio is None or not len(audio):
            raise RuntimeError(f"No audio decoded for active-speaker detection: {video_path}")
        by_uid = {}
        boxes_by_uid = {}
        for _, row in rows.iterrows():
            uid = int(row["utterance_id"])
            start = float(row["start"])
            end = float(row["end"])
            indices = _frame_indices(total, fps, start, end)
            by_uid[uid] = indices

            frame_offset, stop = window_bounds(total, fps, start, end, ASD_LOOKAHEAD)
            segment_frames = frames[frame_offset:stop]
            segment_boxes = all_boxes[frame_offset:stop]
            tracks = build_face_tracks(
                segment_boxes,
                iou_threshold=cfg.asd_iou_threshold,
                max_gap=1,
            )

            result = {int(frame_index): None for frame_index in indices}
            if tracks and audio is not None and len(audio):
                audio_slice = aligned_audio_window(audio, frame_offset, len(segment_frames), fps)
                with phase("active_speaker_detection"):
                    mfcc = audio_to_mfcc_lightasd(
                        audio_slice, sr=16000, fps=fps, n_frames=len(segment_frames),
                    )
                    tracks = asd.score_tracks(tracks, segment_frames, mfcc, fps)
                for frame_index in indices:
                    frame_index = int(frame_index)
                    relative = frame_index - frame_offset
                    candidates = [
                        (track.score, track.boxes[track.frames.index(relative)])
                        for track in tracks
                        if relative in track.frames and track.score >= cfg.asd_threshold
                    ]
                    if candidates:
                        result[frame_index] = max(candidates, key=lambda item: item[0])[1]
            boxes_by_uid[uid] = result

        print(
            f"  [{vfile}] decoded {len(frames)} frames; detected faces once; "
            f"preserved {len(by_uid)} utterance ASD windows",
            flush=True,
        )
        cache["indices"][vfile] = by_uid
        cache["speaker_boxes"][vfile] = boxes_by_uid

    temp_path = cache_path + ".tmp"
    with open(temp_path, "wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, cache_path)
    print(f"Saved reusable face/ASD cache: {cache_path}")
    return cache


def _read_unique_crops(video_path, by_uid, boxes_by_uid, face_margin):
    frame_to_uids = {}
    for uid, indices in by_uid.items():
        for frame_index in indices:
            frame_to_uids.setdefault(int(frame_index), []).append(uid)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for crop extraction: {video_path}")
    crops = []
    wanted = set(frame_to_uids)
    last = max(wanted, default=-1)
    frame_index = 0
    while frame_index <= last:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_index in wanted:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            grouped = {}
            for uid in frame_to_uids[frame_index]:
                box = boxes_by_uid[uid].get(frame_index)
                if box is not None:
                    grouped.setdefault(tuple(float(x) for x in box), []).append(uid)
            for box_values, uids in grouped.items():
                box = np.asarray(box_values)
                h, w = frame_rgb.shape[:2]
                x1, y1, x2, y2 = apply_margin_xyxy(box, face_margin, w=w, h=h)
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size:
                    crops.append((uids, crop))
        frame_index += 1
    cap.release()
    if frame_index <= last:
        raise RuntimeError(f"Video ended before the requested sampled frame: {video_path}")
    return crops


