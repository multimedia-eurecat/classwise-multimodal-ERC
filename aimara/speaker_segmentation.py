"""Group timed transcripts into source-aligned speaker utterances."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .diarization import SpeakerInterval, speaker_for_region
from .segmentation import SpeechRegion
from .speech_interfaces import Transcript


@dataclass(frozen=True)
class SpeakerSegmentationConfig:
    """ROB's sentence grouping and short speaker-run smoothing settings."""

    snap_splits_to_sentences: bool = True
    min_speaker_run_tokens: int = 2
    min_speaker_run_duration: float = 0.4

    def __post_init__(self) -> None:
        if self.min_speaker_run_tokens < 1:
            raise ValueError("min_speaker_run_tokens must be positive")
        if (
            not math.isfinite(self.min_speaker_run_duration)
            or self.min_speaker_run_duration < 0
        ):
            raise ValueError("min_speaker_run_duration must be finite and non-negative")


@dataclass(frozen=True)
class SpeakerUtterance:
    """A transcript group with an exact half-open source audio interval."""

    region: SpeechRegion
    speaker_id: str
    text: str


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    text: str
    speaker: str


class SpeakerSegmenter:
    """Assign timed words to speakers, then partition the original VAD region."""

    def __init__(self, config: SpeakerSegmentationConfig | None = None) -> None:
        self.config = config or SpeakerSegmentationConfig()

    def split(
        self,
        region: SpeechRegion,
        transcript: Transcript,
        sample_rate: int,
        timeline: Sequence[SpeakerInterval],
    ) -> list[SpeakerUtterance]:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        def normalize(text: str) -> str:
            return " ".join(text.split())

        if normalize(
            " ".join(segment.text for segment in transcript.segments)
        ) != normalize(transcript.text):
            return [
                SpeakerUtterance(
                    region, speaker_for_region(region, timeline), transcript.text
                )
            ]
        tokens = []
        for segment in transcript.segments:
            # A backend segment without word alignment is an indivisible token.
            complete_words = segment.words and normalize(
                "".join(word.text for word in segment.words)
            ) == normalize(segment.text)
            spans = segment.words if complete_words else (segment,)
            for span in spans:
                if not span.text.strip():
                    continue
                start = min(
                    region.end_sample,
                    region.start_sample + round(span.start_time * sample_rate),
                )
                end = min(
                    region.end_sample,
                    region.start_sample + round(span.end_time * sample_rate),
                )
                if tokens and start < tokens[-1].start:
                    raise ValueError(
                        "transcript timings must be in chronological order"
                    )
                # A zero-duration token is attributed at its source sample.
                point = min(start, region.end_sample - 1)
                attribution = SpeechRegion(point, max(point + 1, end))
                text = span.text if complete_words else " " + span.text.strip()
                tokens.append(
                    _Token(start, end, text, speaker_for_region(attribution, timeline))
                )

        if not tokens:
            return [
                SpeakerUtterance(
                    region, speaker_for_region(region, timeline), transcript.text
                )
            ]
        if self.config.snap_splits_to_sentences:
            groups = self._sentences(tokens, sample_rate)
        else:
            groups = self._smooth(self._runs(tokens), sample_rate)
        if len(groups) == 1:
            return [SpeakerUtterance(region, groups[0][0], transcript.text)]
        if len(groups) > region.duration_samples:
            raise ValueError("too many speaker groups for the source audio samples")

        # Cut between neighbouring words (the midpoint of a gap or overlap).
        # Prefix/tail audio stays with the outer groups. Every source sample is
        # written exactly once even when model word timestamps overlap.
        boundaries = [region.start_sample]
        for index in range(1, len(groups)):
            midpoint = (groups[index - 1][1][-1].end + groups[index][1][0].start) // 2
            latest = region.end_sample - (len(groups) - index)
            boundaries.append(max(boundaries[-1] + 1, min(midpoint, latest)))
        boundaries.append(region.end_sample)
        return [
            SpeakerUtterance(
                SpeechRegion(boundaries[index], boundaries[index + 1]),
                speaker,
                "".join(token.text for token in spans).strip(),
            )
            for index, (speaker, spans) in enumerate(groups)
        ]

    @staticmethod
    def _runs(tokens: Sequence[_Token]) -> list[tuple[str, list[_Token]]]:
        groups = []
        for token in tokens:
            if groups and groups[-1][0] == token.speaker:
                groups[-1][1].append(token)
            else:
                groups.append((token.speaker, [token]))
        return groups

    def _credible(self, spans: Sequence[_Token], sample_rate: int) -> bool:
        return (
            len(spans) >= self.config.min_speaker_run_tokens
            and (spans[-1].end - spans[0].start) / sample_rate
            >= self.config.min_speaker_run_duration
        )

    def _smooth(
        self, groups: list[tuple[str, list[_Token]]], sample_rate: int
    ) -> list[tuple[str, list[_Token]]]:
        smoothed = []
        for speaker, spans in groups:
            if (
                smoothed
                and speaker != smoothed[-1][0]
                and not self._credible(spans, sample_rate)
                and len(smoothed[-1][1]) >= self.config.min_speaker_run_tokens
            ):
                smoothed[-1][1].extend(spans)
            elif smoothed and speaker == smoothed[-1][0]:
                smoothed[-1][1].extend(spans)
            else:
                smoothed.append((speaker, spans))
        return smoothed

    def _sentences(
        self, tokens: Sequence[_Token], sample_rate: int
    ) -> list[tuple[str, list[_Token]]]:
        units, unit = [], []
        for token in tokens:
            unit.append(token)
            if token.text.strip().endswith((".", "!", "?", "…")):
                units.append(unit)
                unit = []
        if unit:
            units.append(unit)
        groups = []
        for spans in units:
            runs = self._runs(spans)
            if len(runs) > 1 and self._credible(runs[-1][1], sample_rate):
                speaker = runs[-1][0]
            else:
                durations = {}
                for token in spans:
                    durations[token.speaker] = (
                        durations.get(token.speaker, 0) + token.end - token.start
                    )
                maximum = max(durations.values())
                winners = [
                    speaker
                    for speaker, duration in durations.items()
                    if duration == maximum
                ]
                speaker = winners[0] if len(winners) == 1 else "UNKNOWN"
            if groups and groups[-1][0] == speaker:
                groups[-1][1].extend(spans)
            else:
                groups.append((speaker, list(spans)))
        return groups
