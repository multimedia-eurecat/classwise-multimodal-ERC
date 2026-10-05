"""AIMARA integration utilities."""

from .domain import SpeechSegment
from .segmentation import SpeechRegion, VADSegmenter
from .speech_interfaces import (
    ROBProcessorConfig,
    Transcript,
    Transcriber,
    VoiceActivityDetector,
)

__all__ = [
    "ROBProcessorConfig",
    "SpeechRegion",
    "SpeechSegment",
    "Transcript",
    "Transcriber",
    "VADSegmenter",
    "VoiceActivityDetector",
]
