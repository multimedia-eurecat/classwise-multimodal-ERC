"""Resumable batch execution for the AIMARA speech-to-emotion pipeline."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

from .pipeline import AIMARAPipeline
from .batch_resume import (
    CACHE_VERSION,
    MANIFEST_NAME,
    configuration_digest,
    file_digest,
    snapshot_configuration,
    source_identity,
)


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
    "asr_backend",
    "asr_model",
    "transcript_confidence",
    "transcript_confidence_method",
    "speaker_id_confidence",
    "language_source",
    "asr_device",
    "asr_compute_type",
    "asr_input_start_time",
    "asr_input_end_time",
    "emotion",
    "confidence",
)


def _read_predictions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    required = set(BASE_COLUMNS) - {"video_file", "expected_emotion"}
    required.add("probabilities")
    if any(
        not isinstance(row, dict)
        or not required.issubset(row)
        or not isinstance(row["probabilities"], dict)
        for row in data
    ):
        raise ValueError(f"{path} has an incompatible prediction schema")
    return data


def _read_cached_predictions(
    result_path: Path,
    manifest_path: Path,
    configuration: dict,
    fingerprint: str,
    source: dict,
) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("cache_version") != CACHE_VERSION:
        raise ValueError("missing or incompatible resume manifest")
    if (
        manifest.get("configuration_fingerprint") != fingerprint
        or manifest.get("configuration") != configuration
    ):
        raise ValueError("pipeline configuration changed")
    if manifest.get("source") != source:
        raise ValueError("source video changed")
    if manifest.get("predictions_sha256") != file_digest(result_path):
        raise ValueError("cached predictions changed or are incomplete")
    return _read_predictions(result_path)


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_combined(output_dir: Path, rows: list[dict], failures: list[dict]) -> None:
    _atomic_json(output_dir / "predictions.json", rows)
    _atomic_json(output_dir / "failures.json", failures)

    probability_names = sorted(
        {name for row in rows for name in row.get("probabilities", {}).keys()}
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
    run_config: Mapping | None = None,
) -> tuple[list[dict], list[dict]]:
    """Resume only matching, intact results with known configuration provenance.

    CLI pipelines supply resume_config. Custom pipelines can pass run_config;
    without either, processing works but cached results cannot be trusted.
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    if not video_dir.is_dir():
        raise NotADirectoryError(f"video directory not found: {video_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed = {extension.lower() for extension in extensions}
    videos = sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    )
    if not videos:
        raise ValueError(f"no supported videos found in {video_dir}")

    config = (
        run_config
        if run_config is not None
        else getattr(pipeline, "resume_config", None)
    )
    configuration = snapshot_configuration(config) if config is not None else None
    fingerprint = (
        configuration_digest(configuration) if configuration is not None else None
    )
    if resume and configuration is None:
        LOGGER.warning("No resume configuration supplied; rerunning cached videos")

    all_rows: list[dict] = []
    failures: list[dict] = []
    total = len(videos)
    for index, video_path in enumerate(videos, start=1):
        video_output = output_dir / video_path.stem
        result_path = video_output / "predictions.json"
        manifest_path = video_output / MANIFEST_NAME
        try:
            source = source_identity(video_path)
            rows = None
            if resume and configuration is not None and result_path.is_file():
                try:
                    rows = _read_cached_predictions(
                        result_path, manifest_path, configuration, fingerprint, source
                    )
                    LOGGER.info("[%d/%d] Resuming %s", index, total, video_path.name)
                except (OSError, ValueError) as exc:
                    LOGGER.warning(
                        "[%d/%d] Incompatible or invalid cache; rerunning %s: %s",
                        index,
                        total,
                        video_path.name,
                        exc,
                    )
            if rows is None:
                LOGGER.info("[%d/%d] Processing %s", index, total, video_path.name)
                # A failed rerun must never leave old provenance attached to new results.
                manifest_path.unlink(missing_ok=True)
                predictions = pipeline.process_video(video_path, video_output)
                rows = [prediction.to_dict() for prediction in predictions]
                if source_identity(video_path) != source:
                    raise RuntimeError("source video changed during processing")
                video_output.mkdir(parents=True, exist_ok=True)
                _atomic_json(result_path, rows)
                if configuration is not None:
                    _read_predictions(result_path)
                    _atomic_json(
                        manifest_path,
                        {
                            "cache_version": CACHE_VERSION,
                            "configuration": configuration,
                            "configuration_fingerprint": fingerprint,
                            "source": source,
                            "predictions_sha256": file_digest(result_path),
                        },
                    )

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
