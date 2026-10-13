"""Model-agnostic contracts for offline speech processing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Sequence

if TYPE_CHECKING:
    import numpy as np

    from .diarization import SpeakerInterval
    from .segmentation import SpeechRegion


@dataclass(frozen=True)
class TranscriptWord:
    """One token with estimated times in seconds relative to the ASR input.

    Text retains backend whitespace and punctuation for later speaker grouping.
    Zero-duration tokens are valid when the backend cannot resolve their span.
    """

    start_time: float
    end_time: float
    text: str

    def __post_init__(self) -> None:
        _validate_transcript_times(self.start_time, self.end_time)
        if not self.text.strip():
            raise ValueError("word text must not be empty")


@dataclass(frozen=True)
class TranscriptSegment:
    """One backend segment with times relative to the ASR input, not the video."""

    start_time: float
    end_time: float
    text: str
    words: tuple[TranscriptWord, ...] = ()

    def __post_init__(self) -> None:
        _validate_transcript_times(self.start_time, self.end_time)
        words = tuple(self.words)
        if any(not isinstance(word, TranscriptWord) for word in words):
            raise TypeError("words must contain TranscriptWord objects")
        object.__setattr__(self, "words", words)


def _validate_transcript_times(start_time: float, end_time: float) -> None:
    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise ValueError("transcript times must be finite")
    if start_time < 0 or end_time < start_time:
        raise ValueError("transcript times must satisfy 0 <= start_time <= end_time")


@dataclass(frozen=True)
class Transcript:
    """Joined text and optional timed backend segments for one ASR input."""

    text: str
    language: Optional[str] = None
    segments: tuple[TranscriptSegment, ...] = ()

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if any(not isinstance(segment, TranscriptSegment) for segment in segments):
            raise TypeError("segments must contain TranscriptSegment objects")
        object.__setattr__(self, "segments", segments)

    @property
    def words(self) -> tuple[TranscriptWord, ...]:
        """Flatten timed words in the backend's segment/token order."""
        return tuple(word for segment in self.segments for word in segment.words)


@dataclass(frozen=True)
class ROBProcessorConfig:
    """Audio and segmentation settings for offline ROB processing."""

    sample_rate: int = 16000
    chunk_size: int = 512
    vad_threshold: float = 0.5
    min_speech_duration: float = 0.2
    min_silence_duration: float = 0.5
    pre_buffer_duration: float = 0.2
    max_segment_duration: float = 30.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 <= self.vad_threshold <= 1.0:
            raise ValueError("vad_threshold must be between 0 and 1")
        if self.min_speech_duration < 0:
            raise ValueError("min_speech_duration must be non-negative")
        if self.min_silence_duration < 0:
            raise ValueError("min_silence_duration must be non-negative")
        if self.pre_buffer_duration < 0:
            raise ValueError("pre_buffer_duration must be non-negative")
        if self.max_segment_duration <= 0:
            raise ValueError("max_segment_duration must be positive")
        if self.min_speech_duration > self.max_segment_duration:
            raise ValueError("min_speech_duration must not exceed max_segment_duration")


class VoiceActivityDetector(Protocol):
    """Return a speech probability for one audio chunk."""

    def predict(self, audio: np.ndarray, sample_rate: int) -> float: ...


class Transcriber(Protocol):
    """Transcribe one finalized speech interval."""

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript: ...


class SpeakerDiarizer(Protocol):
    """Return a source-aligned speaker timeline for one complete recording."""

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int,
        regions: Sequence[SpeechRegion],
    ) -> Sequence[SpeakerInterval]: ...
