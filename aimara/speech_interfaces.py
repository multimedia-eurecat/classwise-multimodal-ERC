"""Model-agnostic contracts for offline speech processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Sequence

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Transcript:
    """Text returned by ASR for one finalized speech interval."""

    text: str
    language: Optional[str] = None


@dataclass(frozen=True)
class SpeakerTurn:
    """A diarized speaker assignment over an audio interval."""

    start_time: float
    end_time: float
    speaker_id: str

    def __post_init__(self) -> None:
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if not self.speaker_id.strip():
            raise ValueError("speaker_id must not be empty")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


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
            raise ValueError(
                "min_speech_duration must not exceed max_segment_duration"
            )


class VoiceActivityDetector(Protocol):
    """Return a speech probability for one audio chunk."""

    def predict(self, audio: np.ndarray, sample_rate: int) -> float:
        ...


class Transcriber(Protocol):
    """Transcribe one finalized speech interval."""

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        ...


class Diarizer(Protocol):
    """Return speaker turns for a complete audio recording."""

    def diarize(self, audio: np.ndarray, sample_rate: int) -> Sequence[SpeakerTurn]:
        ...
