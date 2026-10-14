"""Speech diagnostics kept separate from ERC emotion probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ASRMetadata:
    """Metadata for one complete ASR input; confidence is not calibrated accuracy."""

    backend: Optional[str] = None
    model: Optional[str] = None
    transcript_confidence: Optional[float] = None
    confidence_method: Optional[str] = None
    language_source: Optional[str] = None
    device: Optional[str] = None
    compute_type: Optional[str] = None

    def __post_init__(self) -> None:
        value = self.transcript_confidence
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(
                "transcript_confidence must be None or finite and between 0 and 1"
            )
        if self.language_source not in (
            None,
            "configured",
            "detected",
            "fallback",
            "model",
        ):
            raise ValueError(
                "language_source must be configured, detected, fallback, model, or None"
            )
        for name in ("backend", "model", "confidence_method", "device", "compute_type"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"ASR {name} must be non-empty when supplied")

    def to_dict(self) -> dict:
        return {
            "asr_backend": self.backend,
            "asr_model": self.model,
            "transcript_confidence": self.transcript_confidence,
            "transcript_confidence_method": self.confidence_method,
            "language_source": self.language_source,
            "asr_device": self.device,
            "asr_compute_type": self.compute_type,
        }
