"""AIMARA integration utilities."""

from .audio_io import decode_audio, write_wav
from .domain import SpeechSegment
from .diarization import SpeakerInterval
from .erc_features import ConversationFeatures, ERCFeatureBridge, SegmentFeatureEncoder
from .erc_model import ERCEmotionModel, ERCModelProfile, EmotionPrediction
from .offline_processor import OfflineROBProcessor
from .segmentation import SpeechRegion, VADSegmenter
from .speech_interfaces import (
    ROBProcessorConfig,
    SpeakerDiarizer,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    Transcriber,
    VoiceActivityDetector,
)
from .vad_processing import OfflineVADProcessor

__all__ = [
    "decode_audio",
    "ConversationFeatures",
    "ERCFeatureBridge",
    "ERCEmotionModel",
    "ERCModelProfile",
    "EmotionPrediction",
    "OfflineROBProcessor",
    "OfflineVADProcessor",
    "ROBProcessorConfig",
    "SpeechRegion",
    "SpeechSegment",
    "SpeakerDiarizer",
    "SpeakerInterval",
    "SegmentFeatureEncoder",
    "Transcript",
    "TranscriptSegment",
    "TranscriptWord",
    "Transcriber",
    "VADSegmenter",
    "VoiceActivityDetector",
    "write_wav",
]
