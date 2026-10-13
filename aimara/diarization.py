"""Speaker timelines aligned to decoded audio sample offsets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .segmentation import SpeechRegion


@dataclass(frozen=True)
class SpeakerInterval(SpeechRegion):
    """One speaker label for a half-open interval of the source audio."""

    speaker_id: str
    confidence: float = -1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.speaker_id.strip():
            raise ValueError("speaker_id must not be empty")
        if not math.isfinite(self.confidence) or not (
            self.confidence == -1.0 or 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be -1 (unavailable) or between 0 and 1")


def speaker_for_region(
    region: SpeechRegion,
    timeline: Sequence[SpeakerInterval],
    *,
    unknown_speaker_id: str = "UNKNOWN",
) -> str:
    """Choose the speaker covering most of a region; ties stay unresolved."""
    coverage: dict[str, int] = {}
    covered = 0
    for interval in timeline:
        overlap = min(region.end_sample, interval.end_sample) - max(
            region.start_sample, interval.start_sample
        )
        if overlap <= 0:
            continue
        speaker = interval.speaker_id.strip()
        if speaker.casefold() == "unknown":
            speaker = unknown_speaker_id
        coverage[speaker] = coverage.get(speaker, 0) + overlap
        covered += overlap

    # Gaps are unknown, rather than evidence for whichever label arrived last.
    uncovered = region.duration_samples - covered
    if uncovered > 0:
        coverage[unknown_speaker_id] = coverage.get(unknown_speaker_id, 0) + uncovered
    if not coverage:
        return unknown_speaker_id
    maximum = max(coverage.values())
    winners = [speaker for speaker, samples in coverage.items() if samples == maximum]
    return winners[0] if len(winners) == 1 else unknown_speaker_id
