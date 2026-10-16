"""Run IEMOCAP/MELD checkpoints on saved ROB segments without speech inference.

The two models receive the same text/audio/visual embeddings. Only the trained
ERC checkpoint, class inventory and speaker-mask width differ. Inputs and
outputs are fingerprinted; a supplied prediction CSV remains an independent
baseline whose model provenance is not inferred from its labels alone.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .domain import SpeechSegment
from .erc_features import ConversationFeatures, ERCFeatureBridge
from .erc_model import ERCEmotionModel, ERCModelProfile, IEMOCAP_EMOTIONS
from .speech_metadata import ASRMetadata


LOGGER = logging.getLogger(__name__)
MELD_EMOTIONS = ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger")
DATASETS = {"iemocap": ("IEMOCAP", IEMOCAP_EMOTIONS, 2), "meld": ("MELD", MELD_EMOTIONS, 9)}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_inputs(input_dir: Path):
    """Require JSON and CSV to describe exactly the same saved segments."""
    with (input_dir / "rob_native.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_file"]].append(row)
    conversations, baseline, identities = {}, {}, {}
    for video_file, csv_rows in grouped.items():
        directory = input_dir / Path(video_file).stem
        prediction_file = directory / "predictions.json"
        records = json.loads(prediction_file.read_text())
        if len(records) != len(csv_rows):
            raise ValueError(f"CSV/JSON segment count differs: {video_file}")
        segments = []
        for index, (record, row) in enumerate(zip(records, csv_rows)):
            if not (
                record["segment_id"] == index
                and record["transcript"] == row["text"]
                and record["speaker_id"] == row["speaker"]
                and record["start_time"] == float(row["start"])
                and record["end_time"] == float(row["end"])
                and record["emotion"] == row["emotion"]
            ):
                raise ValueError(f"CSV/JSON segment differs: {video_file}, {index}")
            segments.append(SpeechSegment(
                conversation_id=record["conversation_id"],
                segment_id=record["segment_id"],
                source_video=directory / "source.mp4",
                audio_path=directory / "audio" / f"segment_{index:04d}.wav",
                start_time=record["start_time"], end_time=record["end_time"],
                speaker_id=record["speaker_id"], transcript=record["transcript"],
                language=record.get("language"),
                asr_metadata=ASRMetadata(
                    backend=record.get("asr_backend"), model=record.get("asr_model"),
                    transcript_confidence=record.get("transcript_confidence"),
                    confidence_method=record.get("transcript_confidence_method"),
                    language_source=record.get("language_source"),
                    device=record.get("asr_device"), compute_type=record.get("asr_compute_type"),
                ),
                speaker_id_confidence=record.get("speaker_id_confidence", -1.0),
                asr_input_start_time=record.get("asr_input_start_time"),
                asr_input_end_time=record.get("asr_input_end_time"),
            ))
        paths = [prediction_file, directory / "source.mp4"] + [s.audio_path for s in segments]
        identities[video_file] = {str(p.resolve()): digest(p) for p in paths}
        conversations[video_file], baseline[video_file] = tuple(segments), records
    if not conversations:
        raise ValueError("input has no conversations")
    return conversations, baseline, identities


def serialize_prediction(prediction, profile: ERCModelProfile) -> dict:
    """Serialize using the actual checkpoint inventory (including MELD class 7)."""
    row = prediction.to_dict()
    row["probabilities"] = dict(zip(profile.emotions, prediction.probabilities))
    if len(prediction.probabilities) != len(profile.emotions):
        raise ValueError("prediction class count differs from checkpoint inventory")
    probabilities = np.asarray(prediction.probabilities)
    if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1, atol=1e-5):
        raise ValueError("invalid prediction probabilities")
    return row


def expand_speaker_mask(mask: np.ndarray, speaker_slots: int) -> np.ndarray:
    if mask.shape[1] > speaker_slots:
        raise ValueError("requested checkpoint has too few speaker slots")
    return np.pad(mask, ((0, 0), (0, speaker_slots - mask.shape[1])))


def cached_snapshot(cache_root: Path, repository: str) -> Path:
    directory = cache_root / "huggingface" / "hub" / ("models--" + repository.replace("/", "--"))
    revision = (directory / "refs" / "main").read_text().strip()
    snapshot = directory / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing local snapshot: {snapshot}")
    return snapshot


def encoder_paths(cache_root: Path) -> dict:
    return {
        "text": cached_snapshot(cache_root, "sentence-transformers/all-roberta-large-v1"),
        "audio": cache_root / "modelscope" / "iic" / "emotion2vec_plus_large",
        "visual": cached_snapshot(cache_root, "dima806/facial_emotions_image_detection"),
    }


def build_features(conversations, cache_root: Path, output: Path, device: str):
    """Load cached encoders on the requested device and never call ASR/VAD."""
    from funasr import AutoModel as FunASRAutoModel
    from unimodal.text.models import TextConfig
    from unimodal.audio.models import AudioConfig, _infer_funasr_embed_dim
    from unimodal.visual.models import VisualConfig
    from .erc_encoders import OriolTextEncoder, OriolAudioEncoder, OriolVisualEncoder

    paths = encoder_paths(cache_root)
    text_config = TextConfig(device=device, model_id=str(paths["text"]), use_sentiment_signal=False)
    audio_config = AudioConfig(device=device, model_id=str(paths["audio"]))
    visual_config = VisualConfig(device=device, model_id=str(paths["visual"]))
    LOGGER.info("Loading local text encoder")
    text_encoder = OriolTextEncoder.load(text_config)
    LOGGER.info("Loading local audio encoder on %s", device)
    audio_model = FunASRAutoModel(model=str(paths["audio"]), hub="ms", device=device,
                                disable_update=True, disable_pbar=True)
    audio_encoder = OriolAudioEncoder(audio_config, None, audio_model,
                                     _infer_funasr_embed_dim(audio_model, 16000))
    LOGGER.info("Loading local visual encoder")
    visual_encoder = OriolVisualEncoder.load(visual_config)
    bridge = ERCFeatureBridge(text_encoder, audio_encoder, visual_encoder, speaker_slots=2)
    features = {}
    for video_file, segments in conversations.items():
        started = time.perf_counter()
        LOGGER.info("Encoding %s (%d saved segments)", video_file, len(segments))
        value = bridge.build(segments)
        features[video_file] = value
        target = output / "features" / (Path(video_file).stem + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, text=value.text, audio=value.audio,
                            visual=value.visual, speaker_mask=value.speaker_mask)
        LOGGER.info("Encoded %s in %.1f seconds", video_file, time.perf_counter() - started)
    fingerprints = {}
    for modality, directory in paths.items():
        fingerprints[modality] = {
            "path": str(directory),
            "files_sha256": {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()},
        }
    configuration = {"text": asdict(text_config), "audio": asdict(audio_config), "visual": asdict(visual_config)}
    return features, {"encoders": fingerprints, "configuration": configuration}


def compare_baseline(predictions, baseline):
    """Observed reproduction is evidence, not a claim of original provenance."""
    labels_equal = sum(a["emotion"] == b["emotion"] for a, b in zip(predictions, baseline))
    comparable = [
        max(abs(a["probabilities"][key] - b["probabilities"][key]) for key in a["probabilities"])
        for a, b in zip(predictions, baseline)
        if a["probabilities"].keys() == b["probabilities"].keys()
    ]
    return {
        "segments": len(predictions), "labels_equal": labels_equal,
        "comparable_probability_rows": len(comparable),
        "max_absolute_probability_difference": max(comparable) if comparable else None,
    }


def run_checkpoint(dataset, checkpoint, conversations, baseline, features, output, provenance, device):
    started = time.perf_counter()
    dataset_name, emotions, speakers = DATASETS[dataset]
    profile = ERCModelProfile(dataset=dataset_name, emotions=emotions, n_speakers=speakers)
    target = output / dataset
    if target.exists():
        raise FileExistsError(f"refusing to overwrite model run: {target}")
    target.mkdir(parents=True)
    report = {
        "schema_version": 1, "status": "running", "model_id": dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(), "device": device,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest(checkpoint),
        "profile": asdict(profile), "label_inventory_source": "multimodal/inference.py",
        "dataset": dataset_name, "label_inventory": emotions,
        "source_prediction_baseline": "supplied rob_native.csv; original checkpoint not recorded in run_report",
        "reuse": "Exact saved segment IDs, transcripts, speakers, WAVs and video intervals; no ASR/VAD/diarization calls",
        "shared_feature_manifest": "../shared_features.json", "source": provenance, "videos": [],
    }
    write_json(target / "run_report.json", report)
    try:
        model = ERCEmotionModel.from_checkpoint(checkpoint, profile=profile, device=device)
        all_predictions, csv_rows = [], []
        for video_file, original in features.items():
            LOGGER.info("%s inference: %s", dataset_name, video_file)
            value = ConversationFeatures(segments=original.segments, text=original.text,
                audio=original.audio, visual=original.visual,
                speaker_mask=expand_speaker_mask(original.speaker_mask, speakers))
            predictions = [serialize_prediction(p, profile) for p in model.predict(value)]
            for row in predictions:
                row.update(video_file=video_file, model_id=dataset,
                           checkpoint=str(checkpoint.resolve()), checkpoint_sha256=report["checkpoint_sha256"])
                csv_rows.append({"video_file": video_file, "start": row["start_time"],
                    "end": row["end_time"], "speaker": row["speaker_id"],
                    "text": row["transcript"], "emotion": row["emotion"]})
            write_json(target / Path(video_file).stem / "predictions.json", predictions)
            all_predictions.extend(predictions)
            report["videos"].append({"video_file": video_file, "segment_count": len(predictions),
                "baseline_comparison": compare_baseline(predictions, baseline[video_file])})
            write_json(target / "run_report.json", report)
        with (target / "predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("video_file", "start", "end", "speaker", "text", "emotion"))
            writer.writeheader()
            writer.writerows(csv_rows)
        write_json(target / "predictions.json", all_predictions)
        report.update(status="passed", segment_count=len(all_predictions), video_count=len(features),
                      predictions_csv_sha256=digest(target / "predictions.csv"))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["runtime_seconds"] = time.perf_counter() - started
        write_json(target / "run_report.json", report)
        write_json(target / "run_manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/media/ssd2/oriol/checkpoints"))
    parser.add_argument("--dataset", choices=("iemocap", "meld"), action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reuse-features", action="store_true", help="Reuse fingerprint-validated features from this output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Local snapshots prevent network downloads and writes to existing model caches.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    datasets = args.dataset or list(DATASETS)
    for dataset in datasets:
        if (args.output_dir / dataset).exists():
            raise FileExistsError(f"refusing to overwrite {args.output_dir / dataset}")
    conversations, baseline, identities = load_inputs(args.input_dir)
    provenance = {"input_dir": str(args.input_dir.resolve()),
        "input_csv_sha256": digest(args.input_dir / "rob_native.csv"), "input_sha256": identities,
        "runner_sha256": digest(Path(__file__)), "python": sys.executable,
        "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "threads": args.threads}
    shared_manifest = args.output_dir / "shared_features.json"
    if args.reuse_features:
        manifest = json.loads(shared_manifest.read_text())
        if manifest["source"]["input_sha256"] != identities:
            raise ValueError("feature cache input fingerprints differ")
        if manifest["source"]["runner_sha256"] != provenance["runner_sha256"]:
            raise ValueError("feature cache runner fingerprint differs")
        features = {}
        for video_file, segments in conversations.items():
            path = args.output_dir / "features" / (Path(video_file).stem + ".npz")
            if digest(path) != manifest["features_sha256"][video_file]:
                raise ValueError(f"feature cache fingerprint differs: {video_file}")
            with np.load(path, allow_pickle=False) as arrays:
                features[video_file] = ConversationFeatures(segments=segments, **dict(arrays))
    else:
        if shared_manifest.exists():
            raise FileExistsError("shared feature manifest exists; use --reuse-features")
        features, encoder_provenance = build_features(conversations, args.model_cache, args.output_dir, args.device)
        manifest = {"schema_version": 1, "source": provenance, **encoder_provenance,
            "features_sha256": {video_file: digest(args.output_dir / "features" / (Path(video_file).stem + ".npz")) for video_file in features},
            "note": "CPU float32 encoders can differ from historical CUDA float16 text embeddings; shared identically between these two runs."}
        write_json(shared_manifest, manifest)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for dataset in datasets:
        checkpoint = args.checkpoint_root / DATASETS[dataset][0] / "final" / "vit_softmax.pt"
        run_checkpoint(dataset, checkpoint, conversations, baseline, features, args.output_dir, provenance, args.device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    LOGGER.info("Completed %s on %d conversations and %d unchanged segments", datasets, len(conversations), sum(map(len, conversations.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
