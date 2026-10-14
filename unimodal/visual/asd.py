"""
asd.py — Active Speaker Detection wrapper using Light-ASD
==========================================================

Light-ASD paper: "A Light Weight Model for Active Speaker Detection" (CVPR 2023)
Repo: https://github.com/Junhua-Liao/Light-ASD

Installation
------------
    git clone https://github.com/Junhua-Liao/Light-ASD
    # Download pretrained weights:
    #   Columbia dataset:  col_model.model
    #   AVA-ActiveSpeaker: ava_model.model  ← recommended for general use
    # Place the .model file anywhere and pass the path via --asd_weights

What this module does
---------------------
Given a video clip (one MELD utterance), it:
  1. Detects all faces in every frame (MTCNN)
  2. Tracks faces across frames building face tracks
  3. Runs Light-ASD to score each track: P(speaking | audio+video)
  4. Returns, for each sampled frame index, the bounding box of the
     highest-scoring (active) speaker face

If ASD fails or finds no active speaker, falls back to the largest face
(same behaviour as the original pipeline).

Interface
---------
    asd = LightASDWrapper(weights_path, device)
    # Per utterance:
    speaker_boxes = asd.get_speaker_boxes(video_path, frame_indices)
    # speaker_boxes: dict {frame_idx -> np.ndarray([x1,y1,x2,y2]) or None}
"""

import os
import sys
import math
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample

from .utils import (
    read_all_frames,
    extract_raw_audio,
    frame_index_at_or_after,
    iou_xyxy,
    apply_margin_xyxy,
)


# ─── Face track dataclass ────────────────────────────────────────────────────

class FaceTrack:
    """
    A face track is a sequence of (frame_idx, box) pairs for the same identity
    across consecutive frames, linked by IoU overlap.
    """
    def __init__(self, track_id: int, frame_idx: int, box: np.ndarray):
        self.track_id = track_id
        self.frames: List[int] = [frame_idx]
        self.boxes: List[np.ndarray] = [box]
        self.last_box = box
        self.last_frame = frame_idx
        self.score: float = 0.0   # ASD score assigned after inference

    def update(self, frame_idx: int, box: np.ndarray):
        self.frames.append(frame_idx)
        self.boxes.append(box)
        self.last_box = box
        self.last_frame = frame_idx

    def __len__(self):
        return len(self.frames)


def build_face_tracks(all_boxes: List[Optional[np.ndarray]], iou_threshold: float = 0.5,
                      max_gap: int = 5) -> List[FaceTrack]:
    """
    Simple greedy IoU tracker: links face detections across frames into tracks.

    all_boxes[i] = np.ndarray (N_i, 4) of boxes detected in frame i, or None.
    Returns list of FaceTrack objects.
    """
    active_tracks: List[FaceTrack] = []
    finished_tracks: List[FaceTrack] = []
    next_id = 0

    for fi, boxes in enumerate(all_boxes):
        if boxes is None or len(boxes) == 0:
            # Close tracks that have been inactive too long
            still_active = []
            for t in active_tracks:
                if fi - t.last_frame <= max_gap:
                    still_active.append(t)
                else:
                    finished_tracks.append(t)
            active_tracks = still_active
            continue

        matched_track_ids = set()
        matched_box_ids = set()

        # Match each active track to the best IoU box
        for t in active_tracks:
            best_iou, best_bi = 0.0, -1
            for bi, box in enumerate(boxes):
                if bi in matched_box_ids:
                    continue
                iou = iou_xyxy(t.last_box, box)
                if iou > best_iou:
                    best_iou, best_bi = iou, bi
            if best_iou >= iou_threshold and best_bi >= 0:
                t.update(fi, boxes[best_bi])
                matched_track_ids.add(id(t))
                matched_box_ids.add(best_bi)

        # Close stale tracks
        still_active = []
        for t in active_tracks:
            if id(t) in matched_track_ids:
                still_active.append(t)
            elif fi - t.last_frame <= max_gap:
                still_active.append(t)
            else:
                finished_tracks.append(t)
        active_tracks = still_active

        # Start new tracks for unmatched boxes
        for bi, box in enumerate(boxes):
            if bi not in matched_box_ids:
                active_tracks.append(FaceTrack(next_id, fi, box))
                next_id += 1

    finished_tracks.extend(active_tracks)
    
    return finished_tracks


# ─── Audio preprocessing ─────────────────────────────────────────────────────

def audio_to_mfcc_lightasd(audio: np.ndarray, sr: int, fps: float, n_frames: int, 
                           n_mfcc: int = 13) -> np.ndarray:
    """
    Converts raw waveform to MFCC aligned to Light-ASD training format.
    Returns (n_frames * 4, n_mfcc) float32.

    Light-ASD dataLoader uses winstep = 0.010 * 25/fps → 4 MFCC frames
    per video frame at 25fps. Output must have exactly n_frames * 4 rows.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("pip install librosa")

    fps = fps if fps > 0 else 25.0
    winlen  = 0.025 * 25 / fps
    winstep = 0.010 * 25 / fps

    win_length = int(round(winlen  * sr))
    hop_length = int(round(winstep * sr))

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=max(win_length, 512),
    )  # (n_mfcc, T_mfcc)

    target = n_frames * 4
    mfcc_resampled = resample(mfcc, target, axis=1)
    return mfcc_resampled.T.astype(np.float32)  # (n_frames*4, n_mfcc)


def crop_face_sequence(frames: List[np.ndarray], track: FaceTrack, size: int = 112, 
                       margin: float = 0.2) -> np.ndarray:
    """
    Crops and resizes face crops for a track.
    Returns (T, H, W, 3) uint8 where T = len(track).
    """
    crops = []
    for fi, box in zip(track.frames, track.boxes):
        frame = frames[fi]
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = apply_margin_xyxy(box, margin, w, h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        crop = cv2.resize(crop, (size, size))
        crops.append(crop)
    return np.stack(crops, axis=0)  # (T, size, size, 3)


# ─── Light-ASD wrapper ───────────────────────────────────────────────────────

class LightASDWrapper:
    """
    Wraps the Light-ASD model for active speaker detection.

    Light-ASD repo must be cloned and its root added to sys.path,
    OR the model class imported via the weights path parent directory.

    Usage:
        asd = LightASDWrapper(weights_path="ava_model.model", device="cuda")
        speaker_boxes = asd.get_speaker_boxes(video_path, frame_indices, mtcnn)
    """

    # Minimum track length to run ASD inference (shorter tracks are unreliable)
    MIN_TRACK_LEN = 5

    def __init__(self, weights_path: str, device: str = "cuda"):
        self.device = device
        self.weights_path = weights_path
        self.model = self._load_model(weights_path, device)

    def _load_model(self, weights_path: str, device: str):
        """
        Loads Light-ASD for inference using ASD_Model + lossAV directly.
        Automatically adds Light-ASD/ (next to the weights file) to sys.path.
        """
        weights_dir = os.path.dirname(os.path.abspath(weights_path))
        candidate = os.path.join(weights_dir, "Light-ASD")
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)

        try:
            from model.Model import ASD_Model
            from loss import lossAV
        except ImportError as e:
            raise ImportError(
                f"Light-ASD modules not found.\n"
                f"Tried: {candidate}\n"
                f"Make sure the repo is cloned there:\n"
                f"  cd {weights_dir} && git clone https://github.com/Junhua-Liao/Light-ASD\n"
                f"Original error: {e}"
            )

        backbone = ASD_Model().to(device)
        classifier = lossAV().to(device)

        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        backbone_state   = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}
        classifier_state = {k.replace("lossAV.", "", 1): v for k, v in state.items() if k.startswith("lossAV.")}

        backbone.load_state_dict(backbone_state)
        classifier.load_state_dict(classifier_state)
        backbone.eval()
        classifier.eval()

        return {"backbone": backbone, "classifier": classifier}

    @torch.no_grad()
    def score_tracks(self, tracks: List[FaceTrack], frames: List[np.ndarray], 
                     audio_mfcc: np.ndarray, fps: float) -> List[FaceTrack]:
        """
        Runs Light-ASD inference on each track and assigns track.score.
        Score = mean P(speaking) over the track frames.

        ASD_Model expects:
          audio:  (1, T*4, 13)     — 4 MFCC frames per video frame
          visual: (1, T, 112, 112) — grayscale uint8 face crops
        """
        backbone   = self.model["backbone"]
        classifier = self.model["classifier"]

        for track in tracks:
            if len(track) < self.MIN_TRACK_LEN:
                track.score = -1.0
                continue

            # ── Visual input: (1, T, 112, 112) ───────────────────────────
            crops = crop_face_sequence(frames, track, size=112, margin=0.2)
            gray = np.mean(crops, axis=-1).astype(np.float32)  # (T, 112, 112)
            v = torch.from_numpy(gray).unsqueeze(0).to(self.device)  # (1, T, 112, 112)

            # ── Audio input: (1, T*4, 13) ─────────────────────────────────
            T_track = len(track)
            T_audio = T_track * 4

            t_start = track.frames[0]
            t_end   = track.frames[-1] + 1
            a_slice = audio_mfcc[t_start * 4 : t_end * 4]

            if len(a_slice) < 1:
                a_slice = np.zeros((T_audio, audio_mfcc.shape[1]), dtype=np.float32)
            else:
                a_slice = resample(a_slice, T_audio, axis=0)

            a = torch.from_numpy(a_slice.astype(np.float32)).unsqueeze(0).to(self.device)

            try:
                outsAV, _ = backbone(a, v)
                logits = classifier.FC(outsAV)
                prob = F.softmax(logits, dim=-1)[:, 1]
                track.score = float(prob.mean().cpu())
                # print(f"    track {track.track_id}: len={len(track)} score={track.score:.3f}")

            except Exception as e:
                warnings.warn(f"Light-ASD inference failed for track {track.track_id}: {e}")
                track.score = 0.0

        return tracks

    def get_speaker_boxes(self, video_path: str, frame_indices: np.ndarray, mtcnn, 
                          iou_threshold: float = 0.5, asd_threshold: float = 0.5, 
                          max_seconds: Optional[float] = None,
                          start_sec: float = 0.0,
                          end_sec: Optional[float] = None,
                          ) -> Dict[int, Optional[np.ndarray]]:
        """
        Main entry point. Given a video and the frame indices to embed,
        returns the bounding box of the active speaker for each frame.

        start_sec/end_sec: restrict both source frames and audio to this window.
                   frame_indices are still absolute — they are offset internally.
                   Default 0.0 → MELD behaviour completely unchanged.

        Returns:
            {frame_idx: np.ndarray([x1,y1,x2,y2]) or None}
            None means no active speaker detected → caller should use fallback.
        """
        # 1. Read frames starting from start_sec up to max_seconds
        frames, fps = read_all_frames(video_path, max_seconds=max_seconds,
                                      start_sec=start_sec, end_sec=end_sec)
        if not frames:
            return {int(fi): None for fi in frame_indices}

        fps = fps if fps > 0 else 25.0
        n_frames = len(frames)

        # Use the same frame seek as read_all_frames, including fractional starts.
        frame_offset = frame_index_at_or_after(start_sec, fps)

        # 2. Detect faces in every frame
        all_boxes: List[Optional[np.ndarray]] = []
        for frame in frames:
            boxes, _ = mtcnn.detect(frame)
            all_boxes.append(boxes if boxes is not None and len(boxes) > 0 else None)

        # 3. Build face tracks
        tracks = build_face_tracks(all_boxes, iou_threshold=iou_threshold, max_gap=1)

        if not tracks:
            return {int(fi): None for fi in frame_indices}

        # 4. Extract audio MFCC aligned to video frames
        # Trim audio to match the video window starting at start_sec
        audio = extract_raw_audio(video_path, sr=16000)
        if audio is None or len(audio) == 0:
            return {int(fi): None for fi in frame_indices}

        audio_start_sample = round(frame_offset / fps * 16000)
        audio_end_sec = (frame_offset + n_frames) / fps
        if end_sec is not None:
            audio_end_sec = min(audio_end_sec, end_sec)
        audio = audio[audio_start_sample:round(audio_end_sec * 16000)]
        if audio.size == 0:
            return {int(fi): None for fi in frame_indices}

        audio_mfcc = audio_to_mfcc_lightasd(audio, sr=16000, fps=fps, n_frames=n_frames)

        # 5. Score each track with Light-ASD
        tracks = self.score_tracks(tracks, frames, audio_mfcc, fps)

        # 6. For each sampled frame index, find the box of the highest-scoring
        #    track that is active at that frame
        result: Dict[int, Optional[np.ndarray]] = {}

        for fi in frame_indices:
            fi     = int(fi)
            fi_rel = fi - frame_offset   # relative index into frames list

            if fi_rel < 0 or fi_rel >= n_frames:
                result[fi] = None
                continue

            candidates = [
                (t.score, t.boxes[t.frames.index(fi_rel)])
                for t in tracks
                if fi_rel in t.frames
                and t.score >= asd_threshold
            ]

            result[fi] = max(candidates, key=lambda x: x[0])[1] if candidates else None

        winning_scores = [t.score for t in tracks]
        # print(f"  [ASD] {len(tracks)} tracks, scores: {[round(s,3) for s in winning_scores]}")
        return result
