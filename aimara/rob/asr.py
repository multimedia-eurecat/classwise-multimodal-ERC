"""Synchronous Faster-Whisper ASR adapted from eut_speech_audio_processing."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from ..speech_interfaces import Transcript, TranscriptSegment, TranscriptWord

if TYPE_CHECKING:
    import numpy as np


WHISPER_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


WHISPER_COMPUTE_TYPES = (
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
)


@dataclass(frozen=True)
class WhisperInferenceConfig:
    """Whisper loading and decoding options, preserving Aimara's defaults."""

    compute_type: str = "float16"
    cpu_compute_type: str = "int8"
    use_batched_inference: bool = False
    batch_size: int = 16
    beam_size: int = 5

    def __post_init__(self) -> None:
        if self.compute_type not in WHISPER_COMPUTE_TYPES:
            raise ValueError(f"compute_type must be one of {WHISPER_COMPUTE_TYPES}")
        if self.cpu_compute_type not in ("int8", "float32"):
            raise ValueError("cpu_compute_type must be int8 or float32")
        if not isinstance(self.use_batched_inference, bool):
            raise ValueError("use_batched_inference must be boolean")
        for name in ("batch_size", "beam_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class ROBTranscriber:
    """Transcribe one finalized 16 kHz speech region with ROB's ASR model."""

    MIN_TRANSCRIPTION_DURATION = 0.01
    DEFAULT_ALLOWED_LANGUAGES = ("en", "es", "ca")
    MAX_BATCHED_CHUNK_SAMPLES = 30 * 16000

    def __init__(
        self,
        *,
        model: Any,
        model_size: str,
        language: str = "auto",
        allowed_languages: Sequence[str] = DEFAULT_ALLOWED_LANGUAGES,
        inference_config: Optional[WhisperInferenceConfig] = None,
        batched_model: Any = None,
        device: Optional[str] = None,
        effective_compute_type: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.validate_model_size(model_size)
        self._model = model
        self.model_size = model_size
        self.language = language
        self.allowed_languages = tuple(allowed_languages)
        self.inference_config = inference_config or WhisperInferenceConfig()
        self.device = device
        self.effective_compute_type = effective_compute_type
        self._logger = logger or logging.getLogger(__name__)
        self._batched_model = None
        if self.inference_config.use_batched_inference:
            if batched_model is None:
                try:
                    from faster_whisper import BatchedInferencePipeline
                except ImportError as exc:
                    raise RuntimeError(
                        "faster-whisper is required for batched ASR"
                    ) from exc
                batched_model = BatchedInferencePipeline(model=model)
            self._batched_model = batched_model

    @classmethod
    def from_faster_whisper(
        cls,
        *,
        model_size: str,
        weights_dir: Union[str, Path],
        compute_type: str = "float16",
        cpu_compute_type: str = "int8",
        use_batched_inference: bool = False,
        batch_size: int = 16,
        beam_size: int = 5,
        language: str = "auto",
        allowed_languages: Sequence[str] = DEFAULT_ALLOWED_LANGUAGES,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "ROBTranscriber":
        """Load Faster-Whisper with ROB's registry and CPU fallback rules."""
        cls.validate_model_size(model_size)
        inference_config = WhisperInferenceConfig(
            compute_type=compute_type,
            cpu_compute_type=cpu_compute_type,
            use_batched_inference=use_batched_inference,
            batch_size=batch_size,
            beam_size=beam_size,
        )
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper and ctranslate2 are required to load ROB ASR"
            ) from exc

        selected_logger = logger or logging.getLogger(__name__)
        selected_device = device or (
            "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        )
        effective_compute_type = compute_type
        if selected_device == "cpu" and compute_type in {"float16", "int8_float16"}:
            effective_compute_type = cpu_compute_type
            selected_logger.warning(
                "ASR compute type %s is not supported on CPU; using %s",
                compute_type,
                effective_compute_type,
            )

        model_dir = Path(weights_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        local_snapshot = cls._resolve_local_snapshot(model_dir, model_size)

        def build_model(selected_compute_type: str):
            if local_snapshot is not None:
                return WhisperModel(
                    str(local_snapshot),
                    device=selected_device,
                    compute_type=selected_compute_type,
                )
            return WhisperModel(
                model_size,
                device=selected_device,
                compute_type=selected_compute_type,
                download_root=str(model_dir),
            )

        try:
            model = build_model(effective_compute_type)
        except ValueError as exc:
            if effective_compute_type == "float32":
                raise
            selected_logger.warning(
                "ASR loading failed with compute type %s (%s); retrying with float32",
                effective_compute_type,
                exc,
            )
            model = build_model("float32")
            effective_compute_type = "float32"

        selected_logger.info(
            "Whisper %s: device=%s, compute_type=%s, batched=%s, batch_size=%s, beam_size=%s",
            model_size,
            selected_device,
            effective_compute_type,
            use_batched_inference,
            batch_size,
            beam_size,
        )

        return cls(
            model=model,
            model_size=model_size,
            language=language,
            allowed_languages=allowed_languages,
            inference_config=inference_config,
            device=selected_device,
            effective_compute_type=effective_compute_type,
            logger=selected_logger,
        )

    @staticmethod
    def validate_model_size(model_size: str) -> str:
        try:
            return WHISPER_MODELS[model_size]
        except KeyError as exc:
            raise ValueError(
                f"invalid model_size {model_size!r}; available: {sorted(WHISPER_MODELS)}"
            ) from exc

    @classmethod
    def _resolve_local_snapshot(
        cls, weights_dir: Path, model_size: str
    ) -> Optional[Path]:
        repository = cls.validate_model_size(model_size)
        candidate = weights_dir / ("models--" + repository.replace("/", "--"))
        if not candidate.is_dir():
            return None
        return next(
            (path.parent for path in sorted(candidate.rglob("model.bin"))),
            None,
        )

    def transcribe(self, audio: "np.ndarray", sample_rate: int) -> Transcript:
        """Transcribe a mono float32 interval, retaining input-relative timings."""
        if getattr(audio, "ndim", None) != 1:
            raise ValueError("audio must be a one-dimensional array")
        if str(getattr(audio, "dtype", "")) != "float32":
            raise TypeError("audio must use float32 samples")
        if sample_rate != 16000:
            raise ValueError("Faster-Whisper audio must use a 16000 Hz sample rate")

        if audio.size < int(sample_rate * self.MIN_TRANSCRIPTION_DURATION):
            return Transcript(text="", language=None)

        transcription_language = self._resolve_language(audio)
        try:
            kwargs = {
                "vad_filter": False,
                "word_timestamps": True,
                "language": transcription_language,
                "beam_size": self.inference_config.beam_size,
            }
            backend = self._model
            if self._batched_model is not None:
                backend = self._batched_model
                # Explicit clips bypass internal VAD and cover >=30s inputs
                # without Faster-Whisper's implicit clip rejection/truncation.
                kwargs["clip_timestamps"] = self._batched_clips(audio.size, sample_rate)
                kwargs["batch_size"] = self.inference_config.batch_size
                backend.last_speech_timestamp = 0.0
            segments, info = backend.transcribe(audio, **kwargs)
            # Materialize the lazy backend generator once so text and timings
            # describe the same decode, including errors raised during iteration.
            timed_segments = tuple(
                TranscriptSegment(
                    start_time=float(segment.start),
                    end_time=float(segment.end),
                    text=segment.text.strip(),
                    words=tuple(
                        TranscriptWord(
                            start_time=float(word.start),
                            end_time=float(word.end),
                            text=word.word,
                        )
                        for word in (getattr(segment, "words", None) or ())
                        if word.word.strip()
                    ),
                )
                for segment in segments
                if segment.text.strip()
            )
            text = " ".join(segment.text for segment in timed_segments)
        except Exception as exc:
            raise RuntimeError("Faster-Whisper transcription failed") from exc
        finally:
            if self._batched_model is not None:
                # Its lazy generator normally resets on exhaustion, but errors
                # can leave alignment state behind for the next region/video.
                self._batched_model.last_speech_timestamp = 0.0

        detected_language = getattr(info, "language", transcription_language)
        return Transcript(
            text=text, language=detected_language, segments=timed_segments
        )

    @classmethod
    def _batched_clips(
        cls, total_samples: int, sample_rate: int
    ) -> list[dict[str, float]]:
        def timestamp(sample: int) -> float:
            value = sample / sample_rate
            # The backend converts seconds with int(time * sample_rate). Keep
            # that round trip exact even when float division rounds downward.
            if int(value * sample_rate) < sample:
                value = math.nextafter(value, math.inf)
            return value

        return [
            {
                "start": timestamp(start),
                "end": timestamp(
                    min(start + cls.MAX_BATCHED_CHUNK_SAMPLES, total_samples)
                ),
            }
            for start in range(0, total_samples, cls.MAX_BATCHED_CHUNK_SAMPLES)
        ]

    def _resolve_language(self, audio: "np.ndarray") -> str:
        if self.model_size.endswith(".en"):
            return "en"
        if self.language == "auto":
            return self._detect_language(audio, self.allowed_languages)
        if "," in self.language:
            requested = tuple(
                language.strip()
                for language in self.language.split(",")
                if language.strip()
            )
            return self._detect_language(audio, requested)
        return self.language

    def _detect_language(
        self, audio: "np.ndarray", allowed_languages: Sequence[str]
    ) -> str:
        fallback = "en"
        try:
            detected, _, probabilities = self._model.detect_language(audio)
            allowed = set(allowed_languages)
            matches = [
                (language, probability)
                for language, probability in probabilities
                if language in allowed
            ]
            if matches:
                return max(matches, key=lambda item: item[1])[0]
            return detected if not allowed else fallback
        except Exception as exc:
            self._logger.warning(
                "ASR language detection failed; using %s: %s", fallback, exc
            )
            return fallback
