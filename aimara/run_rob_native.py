"""Run actual ROB engines in real time, then consume their exports with ERC."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

from .batch import _atomic_json
from .rob_native import ROB_REVISION, inspect_checkout
from .rob_native_export import ROBExportProcessor


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rob-repo", type=Path, required=True)
    parser.add_argument("--rob-revision", default=ROB_REVISION)
    parser.add_argument("--video", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        help="Optional reference turns for the review timelines",
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="Writable ROB model cache"
    )
    parser.add_argument("--asr-backend", choices=("whisper", "parakeet"))
    parser.add_argument(
        "--asr-model", help="Omit to use ROB's configured model (turbo)"
    )
    parser.add_argument(
        "--language", help="Omit to use ROB's configured language allow-list"
    )
    parser.add_argument("--compute-type", help="Omit to use ROB's configured precision")
    parser.add_argument(
        "--batched-inference", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--pre-roll-seconds", type=float, default=2.0)
    parser.add_argument("--drain-timeout", type=float, default=120.0)
    parser.add_argument("--speech-only", action="store_true")
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="Consume existing successful captures; do not replay",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--audio-model", default="iic/emotion2vec_plus_large")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="ERC device; ROB selects its own device",
    )
    parser.add_argument(
        "--unknown-speaker-policy",
        choices=("omit_embedding", "reserve_slot"),
        default="omit_embedding",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def run(args):
    if args.worker:
        from .rob_native import replay_video

        overrides = {
            target: value
            for target, value in (
                ("asr_backend", args.asr_backend),
                ("model_size", args.asr_model),
                ("language", args.language),
                ("compute_type", args.compute_type),
                ("use_batched_inference", args.batched_inference),
            )
            if value is not None
        }
        replay_video(
            args.rob_repo,
            args.video[0],
            args.output_dir,
            weights=args.weights,
            overrides=overrides,
            revision=args.rob_revision,
            tail_seconds=args.tail_seconds,
            pre_roll_seconds=args.pre_roll_seconds,
            drain_timeout=args.drain_timeout,
        )
        return

    if not args.speech_only and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --speech-only is selected")
    if args.import_only and any(
        value is not None
        for value in (
            args.asr_backend,
            args.asr_model,
            args.language,
            args.compute_type,
            args.batched_inference,
        )
    ):
        raise ValueError(
            "--import-only cannot change the ASR configuration of saved captures"
        )
    if len({video.stem for video in args.video}) != len(args.video):
        raise ValueError("Video basenames must be unique within a replay batch")
    inspect_checkout(args.rob_repo, args.rob_revision)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "running",
        "transport": "direct_callbacks_no_ros",
        "videos": [],
    }
    _atomic_json(args.output_dir / "run_report.json", summary)
    try:
        # Each source gets a fresh process: VAD recurrence, identities, ASR timers,
        # GPU models and language history cannot leak into the next recording.
        for video in args.video:
            target = args.output_dir / video.stem
            if not args.import_only:
                target.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "-m",
                    "aimara.run_rob_native",
                    "--worker",
                    "--rob-repo",
                    str(args.rob_repo),
                    "--rob-revision",
                    args.rob_revision,
                    "--video",
                    str(video),
                    "--output-dir",
                    str(target),
                    "--weights",
                    str(args.weights),
                    "--tail-seconds",
                    str(args.tail_seconds),
                    "--pre-roll-seconds",
                    str(args.pre_roll_seconds),
                    "--drain-timeout",
                    str(args.drain_timeout),
                    "--speech-only",
                ]
                for flag, value in (
                    ("--asr-backend", args.asr_backend),
                    ("--asr-model", args.asr_model),
                    ("--language", args.language),
                    ("--compute-type", args.compute_type),
                ):
                    if value is not None:
                        command.extend([flag, value])
                if args.batched_inference is not None:
                    command.append(
                        "--batched-inference"
                        if args.batched_inference
                        else "--no-batched-inference"
                    )
                with (target / "rob.log").open("w") as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT
                    )
                if result.returncode:
                    raise RuntimeError(
                        f"ROB replay failed for {video}; see {target / 'rob.log'}"
                    )
            processor = ROBExportProcessor(target / "rob_capture.json")
            segments = processor.process_video(video, target)
            capture = json.loads(processor.capture_path.read_text())
            summary["videos"].append(
                {
                    "video": str(video.resolve()),
                    "segment_count": len(segments),
                    "rob_revision": capture["rob"]["revision"],
                    "max_replay_lateness_seconds": capture[
                        "replay_max_lateness_seconds"
                    ],
                    "erc_status": "not_requested" if args.speech_only else "pending",
                }
            )
        bridge = model = None
        if not args.speech_only:
            from .run_pipeline import build_erc_components

            bridge, model, _ = build_erc_components(args)
        rows = []
        for video, record in zip(args.video, summary["videos"]):
            target = args.output_dir / video.stem
            processor = ROBExportProcessor(target / "rob_capture.json")
            segments = processor.process_video(video, target)
            predictions = []
            if model is not None:
                try:
                    predictions = (
                        model.predict(bridge.build(segments)) if segments else []
                    )
                    record["erc_status"] = "passed"
                except Exception as error:
                    # Keep ROB's groups intact even if it discovers more known
                    # voices than this ERC checkpoint supports.
                    record["erc_status"] = "failed"
                    record["erc_failure"] = f"{type(error).__name__}: {error}"
                    logging.getLogger(__name__).error(
                        "ERC failed for %s: %s", video, error
                    )
                _atomic_json(
                    target / "predictions.json",
                    [item.to_dict() for item in predictions],
                )
                _atomic_json(target / "erc_result.json", record)
            emotions = {item.segment.segment_id: item.emotion for item in predictions}
            for segment in segments:
                rows.append(
                    {
                        "video_file": video.name,
                        "start": segment.start_time,
                        "end": segment.end_time,
                        "speaker": segment.speaker_id,
                        "text": segment.transcript,
                        "emotion": emotions.get(segment.segment_id, ""),
                    }
                )
        with (args.output_dir / "rob_native.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("video_file", "start", "end", "speaker", "text", "emotion"),
            )
            writer.writeheader()
            writer.writerows(rows)
        from .rob_native_review import write_review

        write_review(args.output_dir, summary, reference_csv=args.reference_csv)
        summary["status"] = (
            "failed"
            if any(item["erc_status"] == "failed" for item in summary["videos"])
            else "passed"
        )
    except Exception as error:
        summary["status"] = "failed"
        summary["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _atomic_json(args.output_dir / "run_report.json", summary)
    return summary


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    result = run(build_parser().parse_args())
    return 0 if result is None or result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
