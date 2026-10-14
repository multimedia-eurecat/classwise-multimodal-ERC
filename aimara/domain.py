"""Runtime data contracts shared by the speech and emotion pipelines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .speaker_ids import normalize_speaker_id
from .speech_metadata import ASRMetadata


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
    asr_metadata: ASRMetadata = field(default_factory=ASRMetadata)
    speaker_id_confidence: float = -1.0
    asr_input_start_time: Optional[float] = None
    asr_input_end_time: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_video", Path(self.source_video))
        object.__setattr__(self, "audio_path", Path(self.audio_path))
        object.__setattr__(self, "speaker_id", normalize_speaker_id(self.speaker_id))

        if not self.conversation_id.strip():
            raise ValueError("conversation_id must not be empty")
        if self.segment_id < 0:
            raise ValueError("segment_id must be non-negative")
        if not math.isfinite(self.start_time) or not math.isfinite(self.end_time):
            raise ValueError("segment times must be finite")
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if not isinstance(self.asr_metadata, ASRMetadata):
            raise TypeError("asr_metadata must be ASRMetadata")
        if not math.isfinite(self.speaker_id_confidence) or not (
            self.speaker_id_confidence == -1 or 0 <= self.speaker_id_confidence <= 1
        ):
            raise ValueError(
                "speaker_id_confidence must be -1 (unavailable) or between 0 and 1"
            )
        start, end = self.asr_input_start_time, self.asr_input_end_time
        if (start is None) != (end is None):
            raise ValueError("ASR input times must be supplied together")
        if start is not None and (
            not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start <= self.start_time < self.end_time <= end
        ):
            raise ValueError(
                "ASR input times must be finite and contain the speech segment"
            )

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
