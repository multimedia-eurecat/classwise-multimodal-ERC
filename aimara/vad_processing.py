"""Offline chunking and speech-region detection using ROB VAD."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from .segmentation import SpeechRegion, VADSegmenter
from .speech_interfaces import ROBProcessorConfig, VoiceActivityDetector

if TYPE_CHECKING:
    import numpy as np


class OfflineVADProcessor:
    """Run chunk-level VAD over a complete decoded recording."""

    def __init__(
        self,
        detector: VoiceActivityDetector,
        config: ROBProcessorConfig,
    ) -> None:
        expected_size = getattr(detector, "EXPECTED_CHUNK_SIZE", config.chunk_size)
        if expected_size != config.chunk_size:
            raise ValueError(
                f"detector expects {expected_size} samples but config uses "
                f"{config.chunk_size}"
            )
        self.detector = detector
        self.config = config
        self.segmenter = VADSegmenter(config)

    def predict_probabilities(self, audio: "np.ndarray") -> List[float]:
        """Return one probability per chunk, padding only the final chunk."""
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required for offline VAD processing") from exc

        if getattr(audio, "ndim", None) != 1:
            raise ValueError("audio must be a one-dimensional array")
        if str(getattr(audio, "dtype", "")) != "float32":
            raise TypeError("audio must use float32 samples")

        probabilities: List[float] = []
        for start in range(0, int(audio.size), self.config.chunk_size):
            source = audio[start : start + self.config.chunk_size]
            if source.size == self.config.chunk_size:
                chunk = np.ascontiguousarray(source)
            else:
                chunk = np.zeros(self.config.chunk_size, dtype=np.float32)
                chunk[: source.size] = source
            probabilities.append(
                float(self.detector.predict(chunk, self.config.sample_rate))
            )
        return probabilities

    def process(self, audio: "np.ndarray") -> List[SpeechRegion]:
        probabilities = self.predict_probabilities(audio)
        return self.segmenter.segment(
            probabilities,
            total_samples=int(audio.size),
        )
