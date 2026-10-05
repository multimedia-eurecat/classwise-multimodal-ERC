"""Checkpoint loading and inference for Oriol's ERC transformer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from .domain import SpeechSegment
from .erc_features import ConversationFeatures


IEMOCAP_EMOTIONS = ("happy", "sad", "neutral", "angry", "excited", "frustrated")


@dataclass(frozen=True)
class ERCModelProfile:
    """Architecture settings needed to reconstruct a trained ERC model."""

    dataset: str = "IEMOCAP"
    text_dim: int = 1024
    audio_dim: int = 1024
    visual_dim: int = 768
    hidden_dim: int = 1024
    n_head: int = 8
    n_speakers: int = 2
    emotions: tuple[str, ...] = IEMOCAP_EMOTIONS
    dropout: float = 0.3
    temperature: int = 1
    fusion_mode: str = "softmax"


@dataclass(frozen=True)
class EmotionPrediction:
    """One emotion prediction aligned with one speech segment."""

    segment: SpeechSegment
    emotion: str
    confidence: float
    probabilities: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.segment.conversation_id,
            "segment_id": self.segment.segment_id,
            "start_time": self.segment.start_time,
            "end_time": self.segment.end_time,
            "speaker_id": self.segment.speaker_id,
            "transcript": self.segment.transcript,
            "language": self.segment.language,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "probabilities": dict(zip(IEMOCAP_EMOTIONS, self.probabilities)),
        }


class ERCEmotionModel:
    """Loaded ERC transformer with a small runtime prediction interface."""

    def __init__(self, model, profile: ERCModelProfile, device: str) -> None:
        self.model = model
        self.profile = profile
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        profile: ERCModelProfile | None = None,
        device: str | None = None,
    ) -> "ERCEmotionModel":
        import torch

        from multimodal.model import Transformer_Based_Model

        profile = profile or ERCModelProfile()
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"ERC checkpoint not found: {checkpoint_path}")

        try:
            state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
        cls._validate_checkpoint(state, profile)

        model = Transformer_Based_Model(
            profile.dataset,
            profile.temperature,
            profile.text_dim,
            profile.visual_dim,
            profile.audio_dim,
            profile.n_head,
            n_classes=len(profile.emotions),
            hidden_dim=profile.hidden_dim,
            n_speakers=profile.n_speakers,
            dropout=profile.dropout,
            fusion_mode=profile.fusion_mode,
        )
        model.load_state_dict(state)
        model.to(selected_device).eval()
        return cls(model, profile, selected_device)

    @staticmethod
    def _validate_checkpoint(state: dict, profile: ERCModelProfile) -> None:
        expected = {
            "textf_input.weight": (profile.hidden_dim, profile.text_dim, 1),
            "acouf_input.weight": (profile.hidden_dim, profile.audio_dim, 1),
            "visuf_input.weight": (profile.hidden_dim, profile.visual_dim, 1),
            "speaker_embeddings.weight": (
                profile.n_speakers + 1,
                profile.hidden_dim,
            ),
            "all_output_layer.weight": (len(profile.emotions), profile.hidden_dim),
        }
        for name, shape in expected.items():
            if name not in state:
                raise ValueError(f"checkpoint is missing {name}")
            actual = tuple(state[name].shape)
            if actual != shape:
                raise ValueError(
                    f"checkpoint {name} has shape {actual}; expected {shape}"
                )

    def predict(self, features: ConversationFeatures) -> list[EmotionPrediction]:
        import torch

        self._validate_features(features)
        batch = features.to_torch_batch(self.device)
        with torch.no_grad():
            output = self.model(**batch)
        probabilities = output[4][0].detach().cpu().float().numpy()

        predictions = []
        for segment, row in zip(features.segments, probabilities):
            class_index = int(np.argmax(row))
            predictions.append(
                EmotionPrediction(
                    segment=segment,
                    emotion=self.profile.emotions[class_index],
                    confidence=float(row[class_index]),
                    probabilities=tuple(float(value) for value in row),
                )
            )
        return predictions

    def _validate_features(self, features: ConversationFeatures) -> None:
        expected = {
            "text": self.profile.text_dim,
            "audio": self.profile.audio_dim,
            "visual": self.profile.visual_dim,
            "speaker_mask": self.profile.n_speakers,
        }
        for name, width in expected.items():
            actual = getattr(features, name).shape[1]
            if actual != width:
                raise ValueError(f"{name} features have width {actual}; expected {width}")
