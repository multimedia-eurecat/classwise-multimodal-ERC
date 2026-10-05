"""Adapters around the repository's existing unimodal feature extractors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .domain import SpeechSegment


def _contextual_transcripts(
    segments: Sequence[SpeechSegment],
    *,
    context_window: int,
    context_sep: str,
    include_speaker: bool,
    task_prefix: str,
) -> list[str]:
    """Build causal text context without using labels or sentiment metadata."""
    texts: list[str] = []
    for target in range(len(segments)):
        start = max(0, target - context_window)
        parts = []
        for segment in segments[start : target + 1]:
            text = segment.transcript
            if include_speaker:
                text = f"{segment.speaker_id}: {text}"
            parts.append(text)
        body = context_sep.join(parts)
        texts.append(f"{task_prefix}{body}" if task_prefix else body)
    return texts


@dataclass
class OriolTextEncoder:
    """Text encoder using the model and pooling configured by Oriol's pipeline."""

    config: object
    tokenizer: object
    model: object

    @classmethod
    def load(cls, config=None) -> "OriolTextEncoder":
        from unimodal.text.models import TextConfig, load_text_model

        config = config or TextConfig(use_sentiment_signal=False)
        tokenizer, model, _ = load_text_model(config)
        return cls(config=config, tokenizer=tokenizer, model=model)

    def encode(self, segments: Sequence[SpeechSegment]) -> np.ndarray:
        from unimodal.text.models import embed_text_batch

        texts = _contextual_transcripts(
            segments,
            context_window=self.config.context_window,
            context_sep=self.config.context_sep,
            include_speaker=self.config.include_speaker,
            task_prefix=self.config.task_prefix,
        )
        batches = []
        for start in range(0, len(texts), self.config.batch_size):
            batches.append(
                embed_text_batch(
                    texts[start : start + self.config.batch_size],
                    self.tokenizer,
                    self.model,
                    self.config.device,
                    self.config.max_length,
                    self.config.pooling,
                )
            )
        return np.concatenate(batches).astype(np.float32, copy=False)


@dataclass
class OriolAudioEncoder:
    """Audio encoder using emotion2vec through Oriol's FunASR helpers."""

    config: object
    processor: object
    model: object
    embed_dim: int

    @classmethod
    def load(cls, config=None) -> "OriolAudioEncoder":
        from unimodal.audio.models import AudioConfig, load_audio_model

        config = config or AudioConfig()
        processor, model, embed_dim = load_audio_model(config)
        return cls(config, processor, model, embed_dim)

    def encode(self, segments: Sequence[SpeechSegment]) -> np.ndarray:
        from unimodal.audio.models import embed_audio_batch, load_wav

        waves = []
        silent_mask = []
        for segment in segments:
            wave, _, is_silent = load_wav(
                str(segment.audio_path),
                self.config.target_sr,
                self.config.max_seconds,
            )
            waves.append(wave)
            silent_mask.append(is_silent)

        return embed_audio_batch(
            waves,
            self.processor,
            self.model,
            self.config.device,
            self.config.target_sr,
            self.config.pooling,
            silent_mask,
            self.embed_dim,
        ).astype(np.float32, copy=False)


@dataclass
class OriolVisualEncoder:
    """Visual encoder sampling each segment's time window in its source video."""

    config: object
    mtcnn: object
    processor: object
    model: object
    embed_dim: int
    asd: object = None

    @classmethod
    def load(cls, config=None) -> "OriolVisualEncoder":
        from unimodal.visual.models import VisualConfig, load_visual_models

        config = config or VisualConfig()
        mtcnn, processor, model, embed_dim, asd = load_visual_models(config)
        return cls(config, mtcnn, processor, model, embed_dim, asd)

    def encode(self, segments: Sequence[SpeechSegment]) -> np.ndarray:
        from unimodal.visual.models import (
            extract_utterance_landmark_embedding,
            extract_utterance_visual_embedding,
        )

        rows = []
        for segment in segments:
            common = {
                "video_path": str(segment.source_video),
                "mtcnn": self.mtcnn,
                "model": self.model,
                "embed_dim": self.embed_dim,
                "num_frames": self.config.num_frames,
                "face_margin": self.config.face_margin,
                "use_fullframe_fallback": self.config.use_fullframe_fallback,
                "scene_cut_threshold": self.config.scene_cut_threshold,
                "asd": self.asd,
                "asd_threshold": self.config.asd_threshold,
                "asd_iou_threshold": self.config.asd_iou_threshold,
                "asd_max_seconds": self.config.asd_max_seconds,
                "start_sec": segment.start_time,
                "end_sec": segment.end_time,
            }
            if self.config.visual_mode.startswith("landmarks"):
                common["tddfa"] = common.pop("model")
                common["visual_mode"] = self.config.visual_mode
                row = extract_utterance_landmark_embedding(**common)
            else:
                common["processor"] = self.processor
                common["device"] = self.config.device
                row = extract_utterance_visual_embedding(**common)
            rows.append(row)
        return np.stack(rows).astype(np.float32, copy=False)
