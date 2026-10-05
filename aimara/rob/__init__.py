"""Offline adapters for the ROB speech models."""

from .asr import ROBTranscriber
from .vad import ROBVoiceActivityDetector

__all__ = ["ROBTranscriber", "ROBVoiceActivityDetector"]
