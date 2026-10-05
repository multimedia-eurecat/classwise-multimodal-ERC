"""AIMARA integration utilities."""

from .audio_io import decode_audio, write_wav
from .domain import SpeechSegment
from .offline_processor import OfflineROBProcessor
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
    "OfflineROBProcessor",
    "OfflineVADProcessor",
    "ROBProcessorConfig",
    "SpeechRegion",
    "SpeechSegment",
    "Transcript",
    "Transcriber",
    "VADSegmenter",
    "VoiceActivityDetector",
    "write_wav",
]
