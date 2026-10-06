"""Resumable batch execution for the AIMARA speech-to-emotion pipeline."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable

from .pipeline import AIMARAPipeline


LOGGER = logging.getLogger(__name__)
BASE_COLUMNS = (
    "video_file",
    "expected_emotion",
    "conversation_id",
    "segment_id",
    "start_time",
    "end_time",
    "speaker_id",
    "transcript",
    "language",
    "emotion",
    "confidence",
)


def _read_predictions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_combined(output_dir: Path, rows: list[dict], failures: list[dict]) -> None:
    _atomic_json(output_dir / "predictions.json", rows)
    _atomic_json(output_dir / "failures.json", failures)

    probability_names = sorted(
        {
            name
            for row in rows
            for name in row.get("probabilities", {}).keys()
        }
    )
    fieldnames = [*BASE_COLUMNS, *(f"probability_{name}" for name in probability_names)]
    csv_path = output_dir / "predictions.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {name: row.get(name) for name in BASE_COLUMNS}
            for name in probability_names:
                flat[f"probability_{name}"] = row.get("probabilities", {}).get(name)
            writer.writerow(flat)
    temporary.replace(csv_path)


def run_batch(
    pipeline: AIMARAPipeline,
    video_dir: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    extensions: Iterable[str] = (".mp4",),
) -> tuple[list[dict], list[dict]]:
    """Process all matching videos while preserving successful partial results."""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    if not video_dir.is_dir():
        raise NotADirectoryError(f"video directory not found: {video_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed = {extension.lower() for extension in extensions}
    videos = sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    )
    if not videos:
        raise ValueError(f"no supported videos found in {video_dir}")

    all_rows: list[dict] = []
    failures: list[dict] = []
    total = len(videos)
    for index, video_path in enumerate(videos, start=1):
        video_output = output_dir / video_path.stem
        result_path = video_output / "predictions.json"
        try:
            if resume and result_path.is_file():
                try:
                    rows = _read_predictions(result_path)
                    LOGGER.info("[%d/%d] Resuming %s", index, total, video_path.name)
                except (OSError, ValueError, json.JSONDecodeError):
                    LOGGER.warning(
                        "[%d/%d] Invalid cached result; rerunning %s",
                        index,
                        total,
                        video_path.name,
                    )
                    predictions = pipeline.process_video(video_path, video_output)
                    rows = [prediction.to_dict() for prediction in predictions]
            else:
                LOGGER.info("[%d/%d] Processing %s", index, total, video_path.name)
                predictions = pipeline.process_video(video_path, video_output)
                rows = [prediction.to_dict() for prediction in predictions]

            for row in rows:
                enriched = dict(row)
                enriched["video_file"] = video_path.name
                enriched["expected_emotion"] = video_path.stem.lower()
                all_rows.append(enriched)
        except Exception as exc:
            LOGGER.exception("[%d/%d] Failed %s", index, total, video_path.name)
            failures.append(
                {
                    "video_file": video_path.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        _write_combined(output_dir, all_rows, failures)

    return all_rows, failures
