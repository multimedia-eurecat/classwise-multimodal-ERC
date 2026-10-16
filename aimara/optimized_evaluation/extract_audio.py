"""
Extract emotion2vec audio embeddings from videos.

Reads inference.csv, slices each utterance's audio from the corresponding
MP4 using ffmpeg, then embeds with emotion2vec_plus_large.
"""

import os
import torch
from .eval_metrics import phase, check_embeddings
import sys
import pickle
import subprocess
import tempfile

import numpy as np
import pandas as pd

CSV_PATH = os.environ["EVAL_CSV_PATH"]
VIDEO_DIR = os.environ["EVAL_VIDEO_DIR"]
OUTPUT_PKL = os.environ["EVAL_AUDIO_PKL"]
MODEL_ID = os.environ["EVAL_AUDIO_MODEL"]
TARGET_SR = 16000
DEVICE = os.environ.get("EVAL_DEVICE", "auto")
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def slice_audio_from_video(video_path: str, start: float, end: float,
                           target_sr: int) -> np.ndarray:
    """
    Extracts audio from [start, end] seconds of a video using ffmpeg.
    Returns a float32 mono waveform at target_sr. Returns zeros on failure.
    """
    duration = end - start
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start),
        "-t",  str(duration),
        "-i",  video_path,
        "-ac", "1",                    # mono
        "-ar", str(target_sr),         # resample
        "-sample_fmt", "s16",
        tmp_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        import soundfile as sf
        wav, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
        assert sr == target_sr
        return wav
    except Exception as e:
        raise RuntimeError(f"Audio decode failed for {video_path} [{start}-{end}]") from e
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def embed_wav(wav: np.ndarray, model, target_sr: int) -> np.ndarray:
    """Embeds a single waveform with emotion2vec via FunASR."""
    import soundfile as sf

    # Check for silence
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < 1e-4:
        print("    [INFO] Silent utterance → zero vector")
        return np.zeros((1024,), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, wav, target_sr)
        res = model.generate(tmp_path, granularity="utterance", extract_embedding=True)
        # extract embedding from FunASR result
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
        for key in ("embedding", "emb", "feats", "feat", "vector"):
            if key in res and res[key] is not None:
                arr = np.asarray(res[key])
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr[0]
                return arr.astype(np.float32)
        raise RuntimeError(f"No embedding found in FunASR output: {list(res.keys())}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



def main():
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)

    # Load CSV
    df = pd.read_csv(CSV_PATH, sep=";")
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} utterances from {df['video_file'].nunique()} videos")

    # Assign integer dialogue_id per video (sorted for reproducibility)
    video_files  = sorted(df["video_file"].unique())
    video_to_did = {v: i for i, v in enumerate(video_files)}
    df["dialogue_id"] = df["video_file"].map(video_to_did)

    # Assign utterance_id as 0-based position within each dialogue
    df["utterance_id"] = df.groupby("dialogue_id").cumcount()

    # Load emotion2vec
    print(f"\nLoading {MODEL_ID} ...")
    from funasr import AutoModel as FunASRAutoModel
    with phase("model_load"):
        model = FunASRAutoModel(model=MODEL_ID, hub="ms", device=DEVICE, disable_update=True)
    print("Model loaded.")

    # Extract embeddings
    embeddings = {}  # {dialogue_id: {utterance_id: np.ndarray}}

    for _, row in df.iterrows():
        did  = int(row["dialogue_id"])
        uid  = int(row["utterance_id"])
        vfile = row["video_file"]
        start = float(row["start"])
        end   = float(row["end"])

        video_path = os.path.join(VIDEO_DIR, vfile)
        if not os.path.exists(video_path):
            print(f"  [WARNING] Video not found: {video_path} → zero vector")
            embeddings.setdefault(did, {})[uid] = np.zeros((1024,), dtype=np.float32)
            continue

        print(f"  [{vfile}] utt {uid}  [{start:.3f}-{end:.3f}s]", end=" ... ", flush=True)
        with phase("audio_decode"):
            wav = slice_audio_from_video(video_path, start, end, TARGET_SR)
        with phase("encoder_inference"):
            emb = embed_wav(wav, model, TARGET_SR)
        embeddings.setdefault(did, {})[uid] = emb
        print(f"dim={emb.shape[0]}")

    # Save
    check_embeddings(embeddings, 1024)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Summary
    n_dialogues = len(embeddings)
    n_utts = sum(len(v) for v in embeddings.values())
    any_d = next(iter(embeddings))
    any_u = next(iter(embeddings[any_d]))
    dim   = embeddings[any_d][any_u].shape[0]
    print(f"\nSaved: {OUTPUT_PKL}")
    print(f"  dialogues={n_dialogues}  utterances={n_utts}  dim={dim}")

    # Save dialogue mapping for merge step
    map_path = OUTPUT_PKL.replace(".pkl", "_video_map.pkl")
    with open(map_path, "wb") as f:
        pickle.dump(video_to_did, f)
    print(f"  video→id map: {map_path}")


if __name__ == "__main__":
    main()
