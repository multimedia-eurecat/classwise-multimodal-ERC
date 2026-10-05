"""AIMARA integration utilities."""

from .audio_io import decode_audio
from .domain import SpeechSegment
from .segmentation import SpeechRegion, VADSegmenter
from .speech_interfaces import (
    ROBProcessorConfig,
    Transcript,
    Transcriber,
    VoiceActivityDetector,
)
from .vad_processing import OfflineVADProcessor

__all__ = [
    "decode_audio",
    "OfflineVADProcessor",
    "ROBProcessorConfig",
    "SpeechRegion",
    "SpeechSegment",
    "Transcript",
    "Transcriber",
    "VADSegmenter",
    "VoiceActivityDetector",
]
