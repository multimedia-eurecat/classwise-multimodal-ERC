"""Speaker timelines aligned to decoded audio sample offsets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .segmentation import SpeechRegion
from .speaker_ids import normalize_speaker_id


def _validate_confidence(confidence: float) -> None:
    if not math.isfinite(confidence) or not (
        confidence == -1.0 or 0.0 <= confidence <= 1.0
    ):
        raise ValueError("confidence must be -1 (unavailable) or between 0 and 1")


@dataclass(frozen=True)
class SpeakerConfidenceSpan(SpeechRegion):
    """Source sample interval carrying one identity-match score."""

    confidence: float

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_confidence(self.confidence)


@dataclass(frozen=True)
class SpeakerInterval(SpeechRegion):
    """One speaker label for a half-open interval of the source audio."""

    speaker_id: str
    confidence: float = -1.0
    confidence_spans: tuple[SpeakerConfidenceSpan, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "speaker_id", normalize_speaker_id(self.speaker_id))
        _validate_confidence(self.confidence)
        spans = tuple(self.confidence_spans)
        previous_end = self.start_sample
        for span in spans:
            if not isinstance(span, SpeakerConfidenceSpan):
                raise TypeError("confidence_spans must contain SpeakerConfidenceSpan")
            if span.start_sample < previous_end or span.end_sample > self.end_sample:
                raise ValueError(
                    "confidence spans must be ordered, non-overlapping, and inside their speaker interval"
                )
            previous_end = span.end_sample
        object.__setattr__(self, "confidence_spans", spans)


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
        speaker = normalize_speaker_id(
            interval.speaker_id, unknown_speaker_id=unknown_speaker_id
        )
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


def speaker_confidence_for_region(
    region: SpeechRegion,
    speaker_id: str,
    timeline: Sequence[SpeakerInterval],
) -> float:
    """Weight identity scores by their coverage of the full emitted interval.

    Gaps, other voices, and unscored stretches contribute no scored coverage.
    -1 means unavailable, while zero is a valid reported match score.
    """
    speaker_id = normalize_speaker_id(speaker_id)
    if speaker_id == "UNKNOWN":
        return -1.0
    weighted = 0.0
    scored_samples = 0
    for interval in timeline:
        if interval.speaker_id != speaker_id:
            continue
        evidence = interval.confidence_spans or (
            SpeakerConfidenceSpan(
                interval.start_sample, interval.end_sample, interval.confidence
            ),
        )
        for span in evidence:
            if span.confidence < 0:
                continue
            overlap = min(region.end_sample, span.end_sample) - max(
                region.start_sample, span.start_sample
            )
            if overlap > 0:
                weighted += overlap * span.confidence
                scored_samples += overlap
    return min(1.0, weighted / region.duration_samples) if scored_samples else -1.0
