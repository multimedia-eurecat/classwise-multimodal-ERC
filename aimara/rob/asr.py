"""Synchronous Faster-Whisper ASR adapted from eut_speech_audio_processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from ..speech_interfaces import Transcript

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


class ROBTranscriber:
    """Transcribe one finalized 16 kHz speech region with ROB's ASR model."""

    MIN_TRANSCRIPTION_DURATION = 0.01
    DEFAULT_ALLOWED_LANGUAGES = ("en", "es", "ca")

    def __init__(
        self,
        *,
        model: Any,
        model_size: str,
        language: str = "auto",
        allowed_languages: Sequence[str] = DEFAULT_ALLOWED_LANGUAGES,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.validate_model_size(model_size)
        self._model = model
        self.model_size = model_size
        self.language = language
        self.allowed_languages = tuple(allowed_languages)
        self._logger = logger or logging.getLogger(__name__)

    @classmethod
    def from_faster_whisper(
        cls,
        *,
        model_size: str,
        weights_dir: Union[str, Path],
        compute_type: str = "float16",
        language: str = "auto",
        allowed_languages: Sequence[str] = DEFAULT_ALLOWED_LANGUAGES,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "ROBTranscriber":
        """Load Faster-Whisper with ROB's registry and CPU fallback rules."""
        cls.validate_model_size(model_size)
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
            effective_compute_type = "int8"
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

        return cls(
            model=model,
            model_size=model_size,
            language=language,
            allowed_languages=allowed_languages,
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
        """Synchronously transcribe one mono float32 speech interval."""
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
            segments, info = self._model.transcribe(
                audio,
                vad_filter=False,
                word_timestamps=False,
                language=transcription_language,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            raise RuntimeError("Faster-Whisper transcription failed") from exc

        detected_language = getattr(info, "language", transcription_language)
        return Transcript(text=text, language=detected_language)

    def _resolve_language(self, audio: "np.ndarray") -> str:
        if self.model_size.endswith(".en"):
            return "en"
        if self.language == "auto":
            return self._detect_language(audio, self.allowed_languages)
        if "," in self.language:
            requested = tuple(
                language.strip() for language in self.language.split(",") if language.strip()
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
            self._logger.warning("ASR language detection failed; using %s: %s", fallback, exc)
            return fallback
