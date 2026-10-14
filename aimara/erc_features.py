"""Feature contracts connecting ROB speech segments to Oriol's ERC model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

import numpy as np

from .domain import SpeechSegment
from .speaker_ids import normalize_speaker_id


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

        if not np.isin(self.speaker_mask, (0.0, 1.0)).all():
            raise ValueError("speaker_mask must contain only zero and one")
        row_totals = self.speaker_mask.sum(axis=1)
        if not np.isin(row_totals, (0.0, 1.0)).all():
            raise ValueError(
                "speaker_mask must contain at most one active speaker per segment"
            )
        for segment, total in zip(self.segments, row_totals):
            if total == 0 and segment.speaker_id != "UNKNOWN":
                raise ValueError("only UNKNOWN segments may omit a speaker embedding")

    @property
    def conversation_id(self) -> str:
        return self.segments[0].conversation_id

    def to_torch_batch(self, device: str = "cpu") -> dict:
        """Build the single-conversation tensor layout expected by the ERC model."""
        import torch

        batch = {
            "textf": torch.from_numpy(self.text).unsqueeze(1).to(device),
            "acouf": torch.from_numpy(self.audio).unsqueeze(1).to(device),
            "visuf": torch.from_numpy(self.visual).unsqueeze(1).to(device),
            "qmask": torch.from_numpy(self.speaker_mask).unsqueeze(0).to(device),
            "u_mask": torch.ones(
                (1, len(self.segments)), dtype=torch.float32, device=device
            ),
            "dia_len": [len(self.segments)],
        }
        # An explicit flag keeps legacy training/inference calls unchanged. A
        # zero qmask row alone would otherwise become speaker zero via argmax.
        if np.any(self.speaker_mask.sum(axis=1) == 0):
            batch["speaker_known_mask"] = (
                torch.from_numpy(self.speaker_mask.sum(axis=1) == 1)
                .unsqueeze(0)
                .to(device)
            )
        return batch


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
        unknown_speaker_policy: str = "omit_embedding",
    ) -> None:
        if speaker_slots < 1:
            raise ValueError("speaker_slots must be positive")
        if unknown_speaker_policy not in ("omit_embedding", "reserve_slot"):
            raise ValueError(
                "unknown_speaker_policy must be omit_embedding or reserve_slot"
            )
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder
        self.speaker_slots = speaker_slots
        self.unknown_speaker_id = normalize_speaker_id(unknown_speaker_id)
        self.unknown_speaker_policy = unknown_speaker_policy

    def build(self, segments: Sequence[SpeechSegment]) -> ConversationFeatures:
        ordered = self._validate_and_order(segments)
        speaker_mask = self._build_speaker_mask(ordered)
        return ConversationFeatures(
            segments=ordered,
            text=self.text_encoder.encode(ordered),
            audio=self.audio_encoder.encode(ordered),
            visual=self.visual_encoder.encode(ordered),
            speaker_mask=speaker_mask,
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

        normalized = tuple(
            replace(segment, speaker_id="UNKNOWN")
            if segment.speaker_id.casefold() == self.unknown_speaker_id.casefold()
            else segment
            for segment in segments
        )
        return tuple(
            sorted(normalized, key=lambda item: (item.start_time, item.segment_id))
        )

    def _build_speaker_mask(self, segments: Sequence[SpeechSegment]) -> np.ndarray:
        mask = np.zeros((len(segments), self.speaker_slots), dtype=np.float32)
        known_speakers = list(
            dict.fromkeys(
                segment.speaker_id
                for segment in segments
                if segment.speaker_id != "UNKNOWN"
            )
        )
        reserve_unknown = self.unknown_speaker_policy == "reserve_slot" and any(
            segment.speaker_id == "UNKNOWN" for segment in segments
        )
        required = len(known_speakers) + int(reserve_unknown)
        if required > self.speaker_slots:
            raise ValueError(
                f"conversation requires {required} speaker slots; checkpoint supports "
                f"{self.speaker_slots} (known speakers: {known_speakers}, "
                f"unknown policy: {self.unknown_speaker_policy})"
            )
        speaker_slots = {
            speaker: slot + int(reserve_unknown)
            for slot, speaker in enumerate(known_speakers)
        }
        for row, segment in enumerate(segments):
            if segment.speaker_id == "UNKNOWN":
                if reserve_unknown:
                    mask[row, 0] = 1.0
            else:
                mask[row, speaker_slots[segment.speaker_id]] = 1.0

        return mask
