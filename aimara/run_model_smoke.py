"""Run pretrained speech-to-ERC smoke checks and preserve model observations."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import math
import platform
import sys
import time
import traceback
import wave
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .batch import _atomic_json, run_batch
from .batch_resume import file_digest, snapshot_configuration, source_identity
from .prepare_manifest import read_reference_csv
from .run_pipeline import add_runtime_arguments, build_pipeline


LOGGER = logging.getLogger(__name__)


def package_versions() -> dict:
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "scipy",
        "transformers",
        "huggingface-hub",
        "funasr",
        "modelscope",
        "faster-whisper",
        "ctranslate2",
        "nemo-toolkit",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_transcript(
    transcript, duration: float, tolerance: float = 0.2
) -> list[str]:
    issues = []
    words = transcript.words
    if transcript.text.strip() and not words:
        issues.append("nonempty transcript has no word timestamps")
    previous_start = -1.0
    for word in words:
        start, end = word.start_time, word.end_time
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and 0 <= start <= end <= duration + tolerance
        ):
            issues.append(f"word outside ASR input: {start}-{end}; duration={duration}")
        if start < previous_start:
            issues.append("word timestamps are out of order")
        previous_start = start
    return issues


def validate_predictions(
    predictions, audio: np.ndarray, rate: int = 16000
) -> list[str]:
    issues = []
    previous_end = 0
    for index, prediction in enumerate(predictions):
        segment = prediction.segment
        start, end = round(segment.start_time * rate), round(segment.end_time * rate)
        if not 0 <= start < end <= len(audio) or start < previous_end:
            issues.append(f"segment {index}: invalid or overlapping source bounds")
        previous_end = end
        if segment.segment_id != index:
            issues.append(f"segment {index}: nonsequential ID")
        probabilities = np.asarray(prediction.probabilities)
        if (
            probabilities.shape != (6,)
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-6)
        ):
            issues.append(f"segment {index}: invalid ERC probabilities")
        elif not math.isclose(
            prediction.confidence, float(probabilities.max()), abs_tol=1e-6
        ):
            issues.append(
                f"segment {index}: emotion confidence differs from maximum probability"
            )
        with wave.open(str(segment.audio_path)) as wav:
            if (
                wav.getframerate(),
                wav.getnchannels(),
                wav.getsampwidth(),
                wav.getnframes(),
            ) != (rate, 1, 2, end - start):
                issues.append(
                    f"segment {index}: WAV format/length differs from source interval"
                )
            else:
                actual = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
                expected = np.clip(audio[start:end] * 32768, -32768, 32767).astype(
                    "<i2"
                )
                if not np.array_equal(actual, expected):
                    issues.append(
                        f"segment {index}: WAV content differs from source interval"
                    )
    return issues


def speaker_diagnostics(timeline, references, rate: int = 16000) -> dict:
    labels = []
    for reference in references:
        overlaps = {}
        for interval in timeline:
            length = max(
                0,
                min(round(reference.end * rate), interval.end_sample)
                - max(round(reference.start * rate), interval.start_sample),
            )
            if length:
                overlaps[interval.speaker_id] = (
                    overlaps.get(interval.speaker_id, 0) + length
                )
        labels.append(max(overlaps, key=overlaps.get) if overlaps else "UNKNOWN")
    returned = None
    if timeline and (
        len(references) >= 3
        and references[0].speaker == references[2].speaker != references[1].speaker
    ):
        returned = labels[0] == labels[2] != labels[1] and "UNKNOWN" not in labels[:3]
    return {
        "reference_speakers": [r.speaker for r in references],
        "dominant_predicted_speakers": labels,
        "a_b_a_pattern_recovered": returned,
    }


def state_matches(first: list[dict], repeated: list[dict]) -> bool:
    if len(first) != len(repeated):
        return False
    for left, right in zip(first, repeated):
        if any(
            left[key] != right[key]
            for key in (
                "segment_id",
                "speaker_id",
                "transcript",
                "language",
                "emotion",
                "start_time",
                "end_time",
            )
        ):
            return False
        if left["probabilities"].keys() != right["probabilities"].keys():
            return False
        if any(
            abs(left["probabilities"][key] - right["probabilities"][key]) > 1e-5
            for key in left["probabilities"]
        ):
            return False
    return True


class ObservedPipeline:
    """Observe real model calls without substituting their decisions or embeddings."""

    def __init__(self, pipeline, report, references, report_path):
        self.pipeline, self.report = pipeline, report
        self.references, self.report_path = references, report_path
        self.resume_config = pipeline.resume_config
        self.probe = False
        self.current = {}
        processor = pipeline.speech_processor
        decode = processor._audio_decoder

        def observed_decode(path, *, sample_rate):
            audio = decode(path, sample_rate=sample_rate)
            self.current["audio"] = audio
            return audio

        processor._audio_decoder = observed_decode
        transcriber = processor.transcriber
        observer = self
        identifier = getattr(transcriber, "language_identifier", None)
        if identifier is not None:

            class LanguageIdentifier:
                def probabilities(self, audio, languages):
                    started = time.perf_counter()
                    scores = identifier.probabilities(audio, languages)
                    observer.current["language_id_calls"].append(
                        {
                            "input_duration": len(audio) / 16000,
                            "seconds": time.perf_counter() - started,
                            "probabilities": scores,
                        }
                    )
                    return scores

            transcriber.language_identifier = LanguageIdentifier()

        class ASR:
            def transcribe(self, audio, sample_rate):
                started = time.perf_counter()
                transcript = transcriber.transcribe(audio, sample_rate)
                duration = len(audio) / sample_rate
                observer.current["asr_calls"].append(
                    {
                        "input_duration": duration,
                        "seconds": time.perf_counter() - started,
                        "text": transcript.text,
                        "language": transcript.language,
                        "metadata": transcript.metadata.to_dict(),
                        "words": [asdict(word) for word in transcript.words],
                        "segments": [
                            asdict(segment) for segment in transcript.segments
                        ],
                        "issues": validate_transcript(transcript, duration),
                    }
                )
                return transcript

        processor.transcriber = ASR()
        if processor.diarizer is not None:
            diarizer = processor.diarizer

            class Diarizer:
                def diarize(self, audio, sample_rate, regions):
                    timeline = diarizer.diarize(audio, sample_rate, regions)
                    observer.current["timeline"] = timeline
                    return timeline

            processor.diarizer = Diarizer()

    def process_video(self, video, output):
        started = time.perf_counter()
        self.current = {"asr_calls": [], "language_id_calls": [], "timeline": []}
        item = {
            "video_file": video.name,
            "source": source_identity(video),
            "source_sha256": file_digest(video),
        }
        try:
            predictions = self.pipeline.process_video(video, output)
            issues = validate_predictions(predictions, self.current["audio"])
            issues.extend(
                issue for call in self.current["asr_calls"] for issue in call["issues"]
            )
            references = [r for r in self.references if r.video_file == video.name]
            if references and not any(
                p.segment.transcript.strip() for p in predictions
            ):
                issues.append("annotated speech produced no nonempty predictions")
            item.update(
                status="passed" if not issues else "failed",
                issues=issues,
                segment_count=len(predictions),
                predictions=[p.to_dict() for p in predictions],
                audio_duration=len(self.current["audio"]) / 16000,
                speaker_diagnostics=speaker_diagnostics(
                    self.current["timeline"], references
                ),
            )
            if issues:
                raise ValueError("; ".join(issues))
            return predictions
        except Exception as exc:
            item.update(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            item.update(
                seconds=time.perf_counter() - started,
                asr_calls=self.current["asr_calls"],
                speaker_timeline=[
                    asdict(interval) for interval in self.current["timeline"]
                ],
            )
            item["language_id_calls"] = self.current["language_id_calls"]
            if self.probe:
                self.report["state_reset_observation"] = item
            else:
                self.report["videos"].append(item)
            output.mkdir(parents=True, exist_ok=True)
            _atomic_json(output / "model_observations.json", item)
            _atomic_json(self.report_path, self.report)


def model_artifacts(args) -> dict:
    paths = [args.checkpoint]
    paths.extend(args.vad_weights.rglob("*.jit"))
    audio_model = Path(args.audio_model)
    if audio_model.is_dir():
        paths.extend(
            audio_model / name
            for name in ("model.pt", "config.yaml", "configuration.json")
        )
    if args.asr_backend == "parakeet":
        from .rob.parakeet import ParakeetTranscriber
        from .run_pipeline import build_parakeet_config

        paths.append(
            ParakeetTranscriber._resolve_checkpoint(
                build_parakeet_config(args), args.asr_weights
            )
        )
        language_id = Path(args.parakeet_language_id_model)
        if language_id.is_file():
            paths.append(language_id)
        elif args.parakeet_language_id_model:
            paths.extend(
                (args.asr_weights / "nemo" / args.parakeet_language_id_model).glob(
                    "*.nemo"
                )
            )
    else:
        from .rob import ROBTranscriber

        snapshot = ROBTranscriber._resolve_local_snapshot(
            args.asr_weights, args.asr_model
        )
        if snapshot:
            paths.append(snapshot / "model.bin")
    if args.diarization_backend != "none":
        paths.extend((args.diarization_weights / "checkpoints").glob("*.pt"))
    return {str(path.resolve()): file_digest(path) for path in paths if path.is_file()}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Select a video filename; repeatable",
    )
    parser.add_argument("--require-language-id", action="store_true")
    parser.add_argument(
        "--check-state-reset", action=argparse.BooleanOptionalAction, default=True
    )
    add_runtime_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    videos = sorted(
        p
        for p in args.video_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".mp4"
    )
    selected = set(args.video)
    if selected:
        missing = selected - {p.name for p in videos}
        if missing:
            raise ValueError(f"selected videos not found: {sorted(missing)}")
        videos = [p for p in videos if p.name in selected]
    if not videos:
        raise ValueError("no MP4 videos selected")
    # run_batch operates on one directory; stage symlinks for a selected subset.
    video_dir = args.video_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if selected:
        video_dir = args.output_dir / "selected_inputs"
        video_dir.mkdir(exist_ok=True)
        for video in videos:
            link = video_dir / video.name
            if link.is_symlink():
                link.unlink()
            link.symlink_to(video.resolve())
        if {p.name for p in video_dir.iterdir()} != {p.name for p in videos}:
            raise ValueError(
                "output contains a different selected input set; use a fresh directory"
            )
    references = read_reference_csv(args.reference_csv) if args.reference_csv else []
    report_path = args.output_dir / "smoke_report.json"
    report = {
        "schema_version": 1,
        "validation_kind": "pretrained_model_smoke",
        "status": "running",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "device": args.device,
        "asr_backend": args.asr_backend,
        "diarization_backend": args.diarization_backend,
        "selected_videos": [p.name for p in videos],
        "videos": [],
        "reference_sha256": file_digest(args.reference_csv)
        if args.reference_csv
        else None,
        "versions": package_versions(),
        "requested_options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runner_sha256": file_digest(Path(__file__)),
    }
    _atomic_json(report_path, report)
    try:
        started = time.perf_counter()
        pipeline = build_pipeline(args)
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
        }
        if args.device == "cuda" and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            report["cuda"].update(
                name=properties.name,
                compute_capability=[properties.major, properties.minor],
                total_memory_bytes=properties.total_memory,
            )
        report["model_load_seconds"] = time.perf_counter() - started
        report["configuration"] = snapshot_configuration(pipeline.resume_config)
        report["artifact_sha256"] = model_artifacts(args)
        transcriber = pipeline.speech_processor.transcriber
        identifier = getattr(transcriber, "language_identifier", None)
        report["language_identifier_loaded"] = identifier is not None
        if args.require_language_id and identifier is None:
            raise RuntimeError("required AmberNet identifier did not load")
        observed = ObservedPipeline(pipeline, report, references, report_path)
        rows, failures = run_batch(observed, video_dir, args.output_dir)
        report["segment_count"], report["failures"] = len(rows), failures
        if args.check_state_reset:
            first = next((v for v in report["videos"] if v["status"] == "passed"), None)
            if first:
                observed.probe = True
                try:
                    repeated = observed.process_video(
                        video_dir / first["video_file"], args.output_dir / "state_reset"
                    )
                    passed = state_matches(
                        first["predictions"], [p.to_dict() for p in repeated]
                    )
                    report["state_reset"] = {
                        "passed": passed,
                        "video_file": first["video_file"],
                    }
                except Exception as exc:
                    report["state_reset"] = {"passed": False, "error": str(exc)}
                finally:
                    observed.probe = False
            else:
                report["state_reset"] = {
                    "passed": False,
                    "error": "no successful video to repeat",
                }
        passed = not failures and all(v["status"] == "passed" for v in report["videos"])
        passed = passed and report.get("state_reset", {"passed": True})["passed"]
        language_calls = [
            call for video in report["videos"] for call in video["language_id_calls"]
        ]
        report["language_identifier_call_count"] = len(language_calls)
        if args.require_language_id:
            passed = (
                passed
                and bool(language_calls)
                and all(
                    call["probabilities"]
                    and all(
                        math.isfinite(score) and 0 <= score <= 1
                        for score in call["probabilities"].values()
                    )
                    for call in language_calls
                )
            )
        report.update(status="passed" if passed else "failed", passed=passed)
    except Exception as exc:
        LOGGER.exception("Model smoke setup failed")
        report.update(
            status="failed",
            passed=False,
            setup_error=str(exc),
            traceback=traceback.format_exc(),
        )
    _atomic_json(report_path, report)
    print(f"Model smoke {report['status'].upper()}: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
