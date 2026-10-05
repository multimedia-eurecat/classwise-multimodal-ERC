"""Deterministic speech segmentation from chunk-level VAD probabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .speech_interfaces import ROBProcessorConfig


@dataclass(frozen=True)
class SpeechRegion:
    """Half-open speech interval represented by exact audio sample offsets."""

    start_sample: int
    end_sample: int

    def __post_init__(self) -> None:
        if self.start_sample < 0:
            raise ValueError("start_sample must be non-negative")
        if self.end_sample <= self.start_sample:
            raise ValueError("end_sample must be greater than start_sample")

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    def start_time(self, sample_rate: int) -> float:
        self._validate_sample_rate(sample_rate)
        return self.start_sample / sample_rate

    def end_time(self, sample_rate: int) -> float:
        self._validate_sample_rate(sample_rate)
        return self.end_sample / sample_rate

    @staticmethod
    def _validate_sample_rate(sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")


class VADSegmenter:
    """Turn fixed-size VAD probabilities into speech regions."""

    def __init__(self, config: ROBProcessorConfig) -> None:
        self.config = config

    def segment(
        self,
        probabilities: Iterable[float],
        *,
        total_samples: int,
    ) -> List[SpeechRegion]:
        """Segment one recording using one probability per audio chunk."""
        if total_samples < 0:
            raise ValueError("total_samples must be non-negative")

        values = list(probabilities)
        expected = (
            math.ceil(total_samples / self.config.chunk_size) if total_samples else 0
        )
        if len(values) != expected:
            raise ValueError(
                f"expected {expected} VAD probabilities for {total_samples} samples, "
                f"got {len(values)}"
            )

        for probability in values:
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("VAD probabilities must be finite values between 0 and 1")

        regions: List[SpeechRegion] = []
        speech_start: Optional[int] = None
        region_start: Optional[int] = None
        silence_start: Optional[int] = None

        pre_buffer_samples = round(
            self.config.pre_buffer_duration * self.config.sample_rate
        )
        min_speech_samples = round(
            self.config.min_speech_duration * self.config.sample_rate
        )
        min_silence_samples = round(
            self.config.min_silence_duration * self.config.sample_rate
        )
        max_segment_samples = max(
            1,
            round(self.config.max_segment_duration * self.config.sample_rate),
        )

        def append_region(end_sample: int) -> None:
            nonlocal speech_start, region_start
            if speech_start is None or region_start is None:
                return
            if end_sample <= speech_start:
                return
            if end_sample - speech_start >= min_speech_samples:
                regions.append(SpeechRegion(region_start, end_sample))

        def reset() -> None:
            nonlocal speech_start, region_start, silence_start
            speech_start = None
            region_start = None
            silence_start = None

        for index, probability in enumerate(values):
            chunk_start = index * self.config.chunk_size
            chunk_end = min(total_samples, chunk_start + self.config.chunk_size)
            is_speech = probability >= self.config.vad_threshold

            if is_speech:
                if speech_start is None:
                    speech_start = chunk_start
                    previous_end = regions[-1].end_sample if regions else 0
                    region_start = max(previous_end, chunk_start - pre_buffer_samples)
                silence_start = None

                while chunk_end - speech_start >= max_segment_samples:
                    split_end = speech_start + max_segment_samples
                    append_region(split_end)
                    speech_start = split_end
                    region_start = split_end
                continue

            if speech_start is None:
                continue
            if silence_start is None:
                silence_start = chunk_start
            if chunk_end - silence_start >= min_silence_samples:
                append_region(silence_start)
                reset()

        if speech_start is not None:
            append_region(silence_start if silence_start is not None else total_samples)

        return regions
