"""Real-time file replay using the pinned ROB engines, without ROS transport.

The speech-activity callbacks are compiled directly from ROB's node source to
avoid maintaining a second implementation of its node-level segmentation rules.
Only message delivery and ROS publisher plumbing are replaced with direct calls.
"""

from __future__ import annotations

import ast
import __future__
import importlib
import inspect
import logging
import math
import multiprocessing
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np

from .audio_io import decode_audio
from .batch import _atomic_json
from .batch_resume import file_digest
from .rob_native_export import observe_groups


ROB_REVISION = "fd89b8b537c17b7ccd103310039ddcf6faf7aa4e"
LOGGER = logging.getLogger(__name__)


def inspect_checkout(repo: Path, revision: str = ROB_REVISION) -> dict:
    """Reject unreviewed source changes; fingerprint the actual imported sources."""
    repo = repo.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != revision:
        raise ValueError(f"ROB revision mismatch: expected {revision}, found {head}")
    paths = ["speech_recognition/speech_recognition", "speech_recognition/config"]
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet", revision, "--", *paths]
    )
    if diff.returncode:
        raise ValueError(
            "ROB speech source/config has local changes; use a clean checkout"
        )
    files = sorted((repo / paths[0]).glob("*.py")) + sorted(
        (repo / paths[1]).glob("*params.yaml")
    )
    if not files:
        raise ValueError("ROB speech source files are missing")
    # Reject untracked Python modules too: git diff alone cannot detect those.
    hashes = {}
    for path in files:
        relative = str(path.relative_to(repo))
        original = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:{relative}"]
        )
        if path.read_bytes() != original:
            raise ValueError(f"ROB source differs from pinned revision: {relative}")
        hashes[relative] = file_digest(path)
    return {"path": str(repo), "revision": head, "files_sha256": hashes}


def load_settings(repo: Path, overrides: dict | None = None) -> dict:
    import yaml

    settings = {}
    for name in ("asr", "diarization", "vad"):
        document = yaml.safe_load(
            (repo / "speech_recognition" / "config" / f"{name}_params.yaml").read_text()
        )
        matches = [
            value["ros__parameters"]
            for value in document.values()
            if "ros__parameters" in value
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one ROB parameter block in {name}_params.yaml")
        settings[name] = dict(matches[0])
    for key, value in (overrides or {}).items():
        if key not in settings["asr"]:
            raise ValueError(f"Unknown ROB ASR parameter: {key}")
        settings["asr"][key] = value
    if settings["diarization"]["diarization_backend"] != "redimnet2":
        raise ValueError("Native replay currently supports ROB's redimnet2 backend")
    return settings


def import_engines(repo: Path, backend: str):
    package_root = repo / "speech_recognition"
    if "speech_recognition" in sys.modules:
        existing = Path(sys.modules["speech_recognition"].__file__).resolve()
        if not existing.is_relative_to(package_root.resolve()):
            raise RuntimeError(
                "A different speech_recognition package is already imported"
            )
    sys.path.insert(0, str(package_root))
    asr = importlib.import_module("speech_recognition.asr_engine")
    redi = importlib.import_module("speech_recognition.redi_voice_engine")
    vad = importlib.import_module("speech_recognition.vad_engine")
    if backend == "parakeet":
        engine_class = importlib.import_module(
            "speech_recognition.parakeet_asr_engine"
        ).ParakeetASREngine
    elif backend == "whisper":
        engine_class = asr.ASREngine
    else:
        raise ValueError("ROB ASR backend must be whisper or parakeet")
    return asr.ASREngine, engine_class, redi.RediVoiceEngine, vad.VADEngine


def node_callback_class(repo: Path):
    """Load unchanged non-ROS method bodies from the pinned DiarizationNode."""
    path = repo / "speech_recognition/speech_recognition/diarization.py"
    tree = ast.parse(path.read_text())
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "DiarizationNode"
    )
    names = {"_vad_callback", "_on_eut_speaker_changed"}
    methods = [
        item
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    if {item.name for item in methods} != names:
        raise ValueError("ROB diarization callback contract changed")
    # Read the node's buffer settings from its constructor, rather than silently
    # substituting defaults from our earlier offline adaptation.
    initial = next(
        item
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    )
    constants = {}
    for item in initial.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Attribute):
            if item.target.attr in {"_vad_buffer_size", "_vad_rms_threshold"}:
                constants[item.target.attr] = ast.literal_eval(item.value)
    if len(constants) != 2:
        raise ValueError("ROB diarization state initialization changed")
    selected = ast.ClassDef(
        name="ROBActivityCallbacks",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    namespace = {"np": np, "time": time}
    exec(
        compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace["ROBActivityCallbacks"], constants


class ReplayLogger:
    """Preserve ROB logging while making swallowed engine failures fail the run."""

    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(str(message))
        LOGGER.error(message)

    def __getattr__(self, name):
        return getattr(LOGGER, name)


def _vad_process(repo, settings, weights, requests, responses):
    """A separate model process, as in ROB's actual VAD ROS node."""
    try:
        sys.path.insert(0, str(Path(repo) / "speech_recognition"))
        from speech_recognition.vad_engine import VADEngine

        vad = VADEngine(
            repo_model=settings["repo_model"],
            model_name=settings["model_name"],
            weights_dir=str(weights),
            logger=logging.getLogger(__name__),
        )
        responses.put(("ready", None))
        while True:
            chunk = requests.get()
            if chunk is None:
                return
            responses.put(("probability", vad.predict(chunk, 16000)))
    except Exception as error:
        responses.put(("error", f"{type(error).__name__}: {error}"))


class ProcessVAD:
    """Transport audio to the original VADEngine in an independent process."""

    independent_process = True

    def __init__(self, repo, settings, weights, timeout=120):
        context = multiprocessing.get_context("spawn")
        self.requests, self.responses = context.Queue(), context.Queue()
        self.timeout = timeout
        self.process = context.Process(
            target=_vad_process,
            args=(str(repo), settings, str(weights), self.requests, self.responses),
            daemon=True,
        )
        self.process.start()
        try:
            kind, message = self.responses.get(timeout=timeout)
            if kind != "ready":
                raise RuntimeError(f"ROB VAD initialization failed: {message}")
        except Exception:
            self.stop()
            raise

    def predict(self, audio, sample_rate):
        if sample_rate != 16000:
            raise ValueError("ROB native replay requires 16 kHz audio")
        self.requests.put(audio)
        kind, message = self.responses.get(timeout=self.timeout)
        if kind != "probability":
            raise RuntimeError(f"ROB VAD failed: {message}")
        return message

    def stop(self):
        self.requests.put(None)
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.requests.close()
        self.responses.close()


class VADDelivery:
    """Keep the audio publisher independent of ROB's VAD inference latency.

    ROS normally runs VAD in its own process. A separate CUDA stream avoids
    artificially serializing it behind REDI on Torch's shared default stream.
    Queue depth matches the VAD node subscription; overflow invalidates a run.
    """

    def __init__(self, vad, wiring, asr, observations, logger):
        self.queue = queue.Queue(maxsize=10)
        self.max_backlog = 0

        def worker():
            import torch

            stream = (
                torch.cuda.Stream()
                if torch.cuda.is_available()
                and not getattr(vad, "independent_process", False)
                else None
            )
            if stream is not None:
                stream.wait_stream(torch.cuda.default_stream())
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                while True:
                    item = self.queue.get()
                    try:
                        if item is None:
                            return
                        chunk, source_end = item
                        probability = vad.predict(chunk, 16000)
                        wiring._vad_callback(
                            SimpleNamespace(vad_probability=probability)
                        )
                        asr.update_vad(probability)
                        observations.append(
                            {"source_end": source_end, "probability": probability}
                        )
                    except Exception as error:
                        logger.error(f"ROB VAD delivery failed: {error}")
                    finally:
                        self.queue.task_done()

        self.thread = threading.Thread(
            target=worker, name="rob-vad-delivery", daemon=True
        )
        self.thread.start()

    def push(self, chunk, source_end):
        try:
            self.queue.put_nowait((chunk, source_end))
            self.max_backlog = max(self.max_backlog, self.queue.qsize())
        except queue.Full as error:
            raise RuntimeError(
                "ROB VAD delivery exceeded its 10-chunk queue; replay is invalid"
            ) from error

    def stop(self, timeout):
        self.queue.put(None, timeout=timeout)
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise TimeoutError("ROB VAD did not finish before the drain timeout")


def build_engines(repo, settings, weights, on_group, activity, logger):
    base, asr_class, redi_class, _vad_class = import_engines(
        repo, settings["asr"]["asr_backend"]
    )
    parameters = settings["asr"]
    options = {
        name: parameters[name]
        for name in inspect.signature(base.__init__).parameters
        if name in parameters
    }
    if parameters["asr_backend"] == "parakeet":
        options.update(
            parakeet_model_name=parameters["parakeet_model_name"],
            language_id_model=parameters["parakeet_language_id_model"],
            language_id_min_confidence=parameters[
                "parakeet_language_id_min_confidence"
            ],
        )
    asr = asr_class(
        **options,
        weights_dir=str(weights),
        on_transcript_ready=lambda *args: None,
        logger=logger,
    )
    asr.sample_rate = 16000
    observe_groups(asr, on_group)
    callback_class, constants = node_callback_class(repo)

    class Activity(callback_class):
        def get_logger(self):
            return logger

        def _publish_speech_activity(self, speaker, active, stamp=None):
            stamp = time.time() if stamp is None else stamp
            confidence = self.engine.speaker_confidence
            speaker = speaker.replace("EUT_", "")
            activity.append(
                {
                    "stamp_unix": stamp,
                    "speaker": speaker,
                    "active": active,
                    "confidence": confidence,
                }
            )
            asr.update_speaker(
                speaker, active=active, confidence=confidence, stamp=stamp
            )

    wiring = Activity()
    wiring._eut_speaker_id = None
    wiring._speaker_activated = False
    wiring._last_voiced_time = None
    wiring._vad_buffer = []
    wiring.__dict__.update(constants)
    diar = settings["diarization"]
    wiring.vad_threshold = diar["vad_threshold"]
    turn_keys = (
        "turn_silence_seconds",
        "min_embed_seconds",
        "embed_interval_seconds",
        "max_embed_seconds",
        "probe_seconds",
    )
    identity_keys = (
        "similarity_threshold",
        "young_threshold",
        "match_margin",
        "stickiness_margin",
        "merge_threshold",
        "short_window_seconds",
        "short_window_threshold",
    )
    identities = {key: diar[f"redi_identity_{key}"] for key in identity_keys}
    identities["young_identity_threshold"] = identities.pop("young_threshold")
    # Match ROB's node arguments. Session-only is explicit; no shared gallery is
    # opened while evaluating independent files on this developer machine.
    redi = redi_class(
        vad_threshold=diar["vad_threshold"],
        use_database=False,
        ros4hri_enabled=False,
        on_eut_speaker_changed=wiring._on_eut_speaker_changed,
        on_voice_update=lambda *args: None,
        logger=logger,
        **{
            key: diar[key]
            for key in (
                "redi_repository",
                "redi_model_name",
                "redi_train_type",
                "redi_dataset",
            )
        },
        min_create_seconds=diar["redi_min_create_seconds"],
        change_threshold=diar["redi_change_threshold"],
        create_on_ambiguous_probe=diar["redi_create_on_ambiguous_probe"],
        turn_options={key: diar[f"redi_{key}"] for key in turn_keys},
        identity_options=identities,
    )
    wiring.engine = redi
    original_handle = redi._handle
    redi._aimara_handle_active = False

    def handle(observation):
        redi._aimara_handle_active = True
        try:
            return original_handle(observation)
        finally:
            redi._aimara_handle_active = False

    redi._handle = handle
    import torch

    torch.hub.set_dir(str(weights))
    if not redi.initialize(16000):
        raise RuntimeError("ROB ReDimNet2 initialization failed")
    vad = ProcessVAD(repo, settings["vad"], weights)
    return asr, redi, vad, wiring


def wait_for_engines(asr, redi, timeout):
    """Drain actual engine workers, including earlier concurrent ASR threads."""
    deadline = time.monotonic() + timeout
    quiet_since = None
    while True:
        asr_threads = [
            thread
            for thread in threading.enumerate()
            if thread.is_alive()
            and any(
                name in thread.name
                for name in ("_process_speech_end", "_transcribe_with_data")
            )
        ]
        # ROB's queue consumer does not call task_done(). Allow a quiet interval
        # to cover the gap between queue.get() and the start of _handle().
        if not asr_threads and redi._queue.empty() and not redi._aimara_handle_active:
            quiet_since = time.monotonic() if quiet_since is None else quiet_since
            if time.monotonic() - quiet_since >= 0.3:
                return
        else:
            quiet_since = None
        if time.monotonic() > deadline:
            raise TimeoutError("ROB workers did not finish before the drain timeout")
        time.sleep(0.05)


def replay_video(
    repo,
    video,
    output_dir,
    *,
    weights,
    overrides=None,
    revision=ROB_REVISION,
    tail_seconds=2.0,
    pre_roll_seconds=2.0,
    drain_timeout=120.0,
):
    """Replay at 512 samples / 16 kHz with real wall clocks and fresh engines."""
    if not math.isfinite(tail_seconds) or tail_seconds < 2:
        raise ValueError("Replay needs at least two seconds of tail silence")
    if not math.isfinite(drain_timeout) or drain_timeout <= 0:
        raise ValueError("Drain timeout must be finite and positive")
    if not math.isfinite(pre_roll_seconds) or pre_roll_seconds < 0:
        raise ValueError("Pre-roll must be finite and nonnegative")
    repo, video, weights = (
        Path(repo).resolve(),
        Path(video).resolve(),
        Path(weights).resolve(),
    )
    provenance = inspect_checkout(repo, revision)
    settings = load_settings(repo, overrides)
    audio = decode_audio(video, sample_rate=16000)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "running",
        "validation_kind": "rob_native_engine_replay",
        "transport": "direct_callbacks_no_ros",
        "rob": provenance,
        "source_video": str(video),
        "source_video_sha256": file_digest(video),
        "sample_rate": 16000,
        "chunk_size": 512,
        "source_samples": len(audio),
        "tail_seconds": tail_seconds,
        "pre_roll_seconds": math.ceil(pre_roll_seconds * 16000 / 512) * 512 / 16000,
        "session_only": True,
        "robot_voice": False,
        "vad_in_separate_process": True,
        "asr_settings": settings["asr"],
        "diarization_settings": {
            key: value
            for key, value in settings["diarization"].items()
            if "uri" not in key
        },
        "groups": [],
        "activity": [],
        "vad": [],
        "timing_basis": "ROB input-start wall clock + final group offsets; not human-verified word times",
        "runner_sha256": {
            name: file_digest(Path(__file__).with_name(name))
            for name in ("rob_native.py", "rob_native_export.py", "run_rob_native.py")
        },
    }
    path = output_dir / "rob_capture.json"
    _atomic_json(path, report)
    logger = ReplayLogger()
    asr = redi = vad = delivery = None
    try:
        asr, redi, vad, wiring = build_engines(
            repo, settings, weights, report["groups"].append, report["activity"], logger
        )
        model = getattr(asr, "model", None)
        actual = getattr(model, "model", None)
        report["asr"] = {
            "backend": settings["asr"]["asr_backend"],
            "model": settings["asr"]["parakeet_model_name"]
            if settings["asr"]["asr_backend"] == "parakeet"
            else settings["asr"]["model_size"],
            "device": str(
                getattr(actual, "device", getattr(asr, "_device", "unknown"))
            ),
            "compute_type": str(getattr(actual, "compute_type", "float32")),
        }
        delivery = VADDelivery(vad, wiring, asr, report["vad"], logger)
        # Prime the first CUDA/JIT call before starting the replay clock. This
        # is explicit leading silence, not discarded speech or an engine reset.
        report["vad_warmup_samples"] = 512
        delivery.push(np.zeros(512, dtype=np.float32), -report["pre_roll_seconds"])
        warmup_deadline = time.monotonic() + drain_timeout
        while delivery.queue.unfinished_tasks:
            if time.monotonic() >= warmup_deadline:
                raise TimeoutError("ROB VAD warmup did not complete")
            time.sleep(0.01)
        start = time.monotonic()
        pre_samples = round(report["pre_roll_seconds"] * 16000)
        report["source_start_unix"] = time.time() + pre_samples / 16000
        late = []
        count = len(audio) + round(tail_seconds * 16000)
        for first in range(-pre_samples, count, 512):
            chunk = np.zeros(512, dtype=np.float32)
            if first >= 0:
                available = audio[first : min(first + 512, len(audio))]
                chunk[: len(available)] = available
            due = start + (first + pre_samples + 512) / 16000
            time.sleep(max(0.0, due - time.monotonic()))
            if first >= 0:
                late.append(max(0.0, time.monotonic() - due))
            # Audio consumers receive the chunk before the VAD result for it,
            # as in ROB's separate audio and VAD ROS subscriptions.
            asr.push_audio(chunk)
            redi.push_audio(chunk)
            delivery.push(chunk, (first + 512) / 16000)
        delivery.stop(drain_timeout)
        report["vad_delivery_max_backlog"] = delivery.max_backlog
        delivery = None
        wait_for_engines(asr, redi, drain_timeout)
        report["replay_max_lateness_seconds"] = max(late, default=0)
        report["replay_late_chunks_over_32ms"] = sum(value > 0.032 for value in late)
        if logger.errors:
            raise RuntimeError("ROB logged engine errors; see capture errors")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if delivery is not None:
            delivery.stop(drain_timeout)
        if vad is not None:
            vad.stop()
        if redi is not None:
            redi.stop()
        if asr is not None:
            asr.stop()
        report["errors"] = logger.errors
        _atomic_json(path, report)
    return path
