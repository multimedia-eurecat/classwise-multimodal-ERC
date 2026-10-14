"""Measure one pretrained configuration with sequential warm pipeline runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .audio_io import decode_audio
from .batch import _atomic_json
from .batch_resume import file_digest, snapshot_configuration
from .run_model_smoke import (
    model_artifacts,
    package_versions,
    state_matches,
    validate_predictions,
)
from .run_pipeline import build_pipeline


def summarize(samples: list[dict]) -> dict:
    if not samples or any(
        not math.isfinite(s["seconds"])
        or not math.isfinite(s["audio_duration"])
        or s["seconds"] <= 0
        or s["audio_duration"] <= 0
        for s in samples
    ):
        raise ValueError("Benchmark requires positive timing and duration samples")
    seconds = [s["seconds"] for s in samples]
    total_time = sum(seconds)
    duration = sum(s["audio_duration"] for s in samples)
    return dict(
        samples=len(samples),
        median_seconds=statistics.median(seconds),
        min_seconds=min(seconds),
        max_seconds=max(seconds),
        total_seconds=total_time,
        total_audio_seconds=duration,
        real_time_factor=total_time / duration,
    )


class MemorySampler:
    """Sample parent-process RSS and driver-reported memory including CTranslate2.

    Torch allocator peaks omit non-Torch allocations. nvidia-smi observes the
    process across visible GPUs; a short-lived spike can fall between samples.
    """

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.rss_peak_bytes = 0
        self.gpu_peak_bytes = None
        self.errors = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            try:
                status = Path("/proc/self/status").read_text()
                rss = next(
                    line for line in status.splitlines() if line.startswith("VmRSS:")
                )
                self.rss_peak_bytes = max(
                    self.rss_peak_bytes, int(rss.split()[1]) * 1024
                )
                if self.cuda:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5,
                    )
                    usages = []
                    for line in result.stdout.splitlines():
                        pid, memory = line.split(",")
                        if int(pid.strip()) == os.getpid():
                            usages.append(int(memory.strip()) * 1024 * 1024)
                    if usages:
                        self.gpu_peak_bytes = max(self.gpu_peak_bytes or 0, sum(usages))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if message not in self.errors:
                    self.errors.append(message)
            self.stop.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-report",
        type=Path,
        required=True,
        help="Passed report whose runtime options will be rebuilt",
    )
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        help="Repeat to select filenames from that report",
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("--repeat must be at least 2")
    if args.output_dir.exists():
        parser.error("Use a fresh output directory to exclude resume caches")
    source = json.loads(args.smoke_report.read_text())
    if not source.get("passed"):
        parser.error("The source smoke report must have passed")
    selected = sorted(set(args.video))
    cases = {v["video_file"]: v for v in source["videos"]}
    if set(selected) - cases.keys():
        parser.error("Selected videos must exist in the source report")
    options = dict(source["requested_options"])
    for key in (
        "video_dir",
        "output_dir",
        "reference_csv",
        "checkpoint",
        "vad_weights",
        "asr_weights",
        "diarization_weights",
    ):
        if options.get(key) is not None:
            options[key] = Path(options[key])
    runtime = SimpleNamespace(**options)
    videos = {name: runtime.video_dir / name for name in selected}
    for name, path in videos.items():
        if file_digest(path) != cases[name]["source_sha256"]:
            parser.error(f"Source changed: {name}")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "benchmark_report.json"
    report = dict(
        schema_version=1,
        validation_kind="sequential_pipeline_benchmark",
        status="running",
        generated_at=datetime.now(timezone.utc).isoformat(),
        runner_sha256=file_digest(Path(__file__)),
        platform=platform.platform(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        source_report_sha256=file_digest(args.smoke_report),
        executable=sys.executable,
        versions=package_versions(),
        selected_videos=selected,
        repeats=args.repeat,
        environment={
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
        },
        methods=dict(
            timing="perf_counter around process_video, with CUDA synchronization; includes decoding, models, WAV/JSON writes; checks outside timer",
            startup="Model construction in a fresh process; local caches warm; includes loader-internal warmups and lazy imports; excludes initial CLI imports",
            warmup="One full inference per selected video excluded; round-robin sequential measured repeats; no resume",
            memory="Parent ru_maxrss includes imports/startup; 250 ms sampled driver GPU memory includes non-Torch models; Torch allocator peaks separately; FFmpeg child RSS excluded",
            scope="Small clip subset, two repeats by default; descriptive ranges, no tail-latency claim",
        ),
        samples=[],
    )
    _atomic_json(report_path, report)
    try:
        import torch

        cuda = runtime.device == "cuda" or (
            runtime.device is None and torch.cuda.is_available()
        )
        if cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        def synchronize():
            if cuda:
                torch.cuda.synchronize()

        if cuda:
            torch.cuda.reset_peak_memory_stats()
        with MemorySampler(cuda) as memory:
            started = time.perf_counter()
            pipeline = build_pipeline(runtime)
            synchronize()
            report["model_load_seconds"] = time.perf_counter() - started
            encoder_devices = {
                name: getattr(
                    getattr(pipeline.feature_bridge, f"{name}_encoder").config,
                    "device",
                    None,
                )
                for name in ("text", "audio", "visual")
            }
            report["encoder_devices"] = encoder_devices
            if runtime.device == "cpu" and any(
                str(device).startswith("cuda") for device in encoder_devices.values()
            ):
                raise ValueError(
                    "CPU benchmark has CUDA encoders; use a CPU-only profile or hide GPUs"
                )
            report["configuration"] = snapshot_configuration(pipeline.resume_config)
            report["artifact_sha256"] = model_artifacts(runtime)
            if report["artifact_sha256"] != source["artifact_sha256"]:
                # Older smoke reports lacked the Silero JIT entries. Every
                # artifact they did record must still match exactly.
                for path, digest in source["artifact_sha256"].items():
                    if report["artifact_sha256"].get(path) != digest:
                        raise ValueError(f"Model artifact changed: {path}")
            audio = {name: decode_audio(path) for name, path in videos.items()}
            warmup = {}
            for name in selected:
                synchronize()
                started = time.perf_counter()
                predictions = pipeline.process_video(
                    videos[name], args.output_dir / "warmup" / Path(name).stem
                )
                synchronize()
                elapsed = time.perf_counter() - started
                issues = validate_predictions(predictions, audio[name])
                if issues or not predictions:
                    raise ValueError(f"{name}: {issues or 'No predictions'}")
                warmup[name] = [p.to_dict() for p in predictions]
                report.setdefault("warmup_seconds", {})[name] = elapsed
            for repeat in range(args.repeat):
                for name in selected:
                    synchronize()
                    started = time.perf_counter()
                    predictions = pipeline.process_video(
                        videos[name],
                        args.output_dir / f"repeat-{repeat}" / Path(name).stem,
                    )
                    synchronize()
                    elapsed = time.perf_counter() - started
                    issues = validate_predictions(predictions, audio[name])
                    stable = state_matches(
                        warmup[name], [p.to_dict() for p in predictions]
                    )
                    report["samples"].append(
                        dict(
                            video_file=name,
                            repeat=repeat,
                            seconds=elapsed,
                            audio_duration=len(audio[name]) / 16000,
                            segment_count=len(predictions),
                            stable=stable,
                            issues=issues,
                        )
                    )
                    _atomic_json(report_path, report)
                    if issues or not stable:
                        raise ValueError(f"{name}: invalid or unstable timed inference")
        report["summary"] = summarize(report["samples"])
        report["memory"] = dict(
            parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            sampled_parent_peak_rss_bytes=memory.rss_peak_bytes,
            sampled_gpu_peak_bytes=memory.gpu_peak_bytes,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated()
            if cuda
            else None,
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved()
            if cuda
            else None,
            sampler_errors=memory.errors,
        )
        if memory.errors or (cuda and memory.gpu_peak_bytes is None):
            raise RuntimeError("Memory sampling incomplete; inspect report")
        if cuda:
            properties = torch.cuda.get_device_properties(0)
            report["cuda"] = dict(
                name=properties.name,
                compute_capability=[properties.major, properties.minor],
                total_memory_bytes=properties.total_memory,
                runtime=torch.version.cuda,
            )
        report.update(status="passed", passed=True)
    except Exception as exc:
        report.update(
            status="failed", passed=False, error=f"{type(exc).__name__}: {exc}"
        )
        _atomic_json(report_path, report)
        raise
    _atomic_json(report_path, report)
    print(f"Benchmark PASSED: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
