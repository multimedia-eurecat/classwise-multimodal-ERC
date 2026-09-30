"""Runtime data contracts shared by the speech and emotion pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SpeechSegment:
    """One finalized ROB speech segment ready for emotion processing."""

    conversation_id: str
    segment_id: int
    source_video: Path
    audio_path: Path
    start_time: float
    end_time: float
    speaker_id: str
    transcript: str
    language: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_video", Path(self.source_video))
        object.__setattr__(self, "audio_path", Path(self.audio_path))

        if not self.conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        if self.segment_id < 0:
            raise ValueError("segment_id must be non-negative")
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if not self.speaker_id.strip():
            raise ValueError("speaker_id must not be empty")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
