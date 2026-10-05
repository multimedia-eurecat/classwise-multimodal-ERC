"""Feature contracts connecting ROB speech segments to Oriol's ERC model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .domain import SpeechSegment


class SegmentFeatureEncoder(Protocol):
    """An encoder that returns one feature vector per speech segment."""

    def encode(self, segments: Sequence[SpeechSegment]) -> np.ndarray:
        """Return a float array shaped ``(segments, feature_dimension)``."""


@dataclass(frozen=True)
class ConversationFeatures:
    """Aligned multimodal features for one ERC conversation."""

    segments: tuple[SpeechSegment, ...]
    text: np.ndarray
    audio: np.ndarray
    visual: np.ndarray
    speaker_mask: np.ndarray

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("segments must not be empty")

        expected_rows = len(self.segments)
        for name in ("text", "audio", "visual", "speaker_mask"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.ndim != 2:
                raise ValueError(f"{name} must be a two-dimensional array")
            if value.shape[0] != expected_rows:
                raise ValueError(
                    f"{name} has {value.shape[0]} rows; expected {expected_rows}"
                )
            if value.shape[1] == 0:
                raise ValueError(f"{name} must have at least one feature column")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, value)

        row_totals = self.speaker_mask.sum(axis=1)
        if not np.allclose(row_totals, 1.0):
            raise ValueError("speaker_mask must contain one active speaker per segment")

    @property
    def conversation_id(self) -> str:
        return self.segments[0].conversation_id

    def to_torch_batch(self, device: str = "cpu") -> dict:
        """Build the single-conversation tensor layout expected by the ERC model."""
        import torch

        return {
            "textf": torch.from_numpy(self.text).unsqueeze(1).to(device),
            "acouf": torch.from_numpy(self.audio).unsqueeze(1).to(device),
            "visuf": torch.from_numpy(self.visual).unsqueeze(1).to(device),
            "qmask": torch.from_numpy(self.speaker_mask).unsqueeze(0).to(device),
            "u_mask": torch.ones(
                (1, len(self.segments)), dtype=torch.float32, device=device
            ),
            "dia_len": [len(self.segments)],
        }


class ERCFeatureBridge:
    """Run aligned feature encoders over a conversation of ROB segments."""

    def __init__(
        self,
        text_encoder: SegmentFeatureEncoder,
        audio_encoder: SegmentFeatureEncoder,
        visual_encoder: SegmentFeatureEncoder,
        *,
        speaker_slots: int = 2,
        unknown_speaker_id: str = "UNKNOWN",
    ) -> None:
        if speaker_slots < 1:
            raise ValueError("speaker_slots must be positive")
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder
        self.speaker_slots = speaker_slots
        self.unknown_speaker_id = unknown_speaker_id

    def build(self, segments: Sequence[SpeechSegment]) -> ConversationFeatures:
        ordered = self._validate_and_order(segments)
        return ConversationFeatures(
            segments=ordered,
            text=self.text_encoder.encode(ordered),
            audio=self.audio_encoder.encode(ordered),
            visual=self.visual_encoder.encode(ordered),
            speaker_mask=self._build_speaker_mask(ordered),
        )

    def _validate_and_order(
        self, segments: Sequence[SpeechSegment]
    ) -> tuple[SpeechSegment, ...]:
        if not segments:
            raise ValueError("segments must not be empty")

        conversation_ids = {segment.conversation_id for segment in segments}
        if len(conversation_ids) != 1:
            raise ValueError("all segments must belong to one conversation")

        segment_ids = [segment.segment_id for segment in segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("segment_id values must be unique within a conversation")

        return tuple(sorted(segments, key=lambda item: (item.start_time, item.segment_id)))

    def _build_speaker_mask(
        self, segments: Sequence[SpeechSegment]
    ) -> np.ndarray:
        mask = np.zeros((len(segments), self.speaker_slots), dtype=np.float32)
        speaker_slots: dict[str, int] = {}

        for row, segment in enumerate(segments):
            if segment.speaker_id == self.unknown_speaker_id:
                slot = 0
            else:
                if segment.speaker_id not in speaker_slots:
                    used_slots = set(speaker_slots.values())
                    if any(s.speaker_id == self.unknown_speaker_id for s in segments):
                        used_slots.add(0)
                    available = next(
                        (i for i in range(self.speaker_slots) if i not in used_slots),
                        None,
                    )
                    if available is None:
                        raise ValueError(
                            f"conversation has more than {self.speaker_slots} speaker slots"
                        )
                    speaker_slots[segment.speaker_id] = available
                slot = speaker_slots[segment.speaker_id]
            mask[row, slot] = 1.0

        return mask
