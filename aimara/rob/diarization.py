"""Synchronous ReDimNet2 speaker attribution for offline VAD regions."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

from ..diarization import SpeakerInterval
from ..segmentation import SpeechRegion

if TYPE_CHECKING:
    import numpy as np


class ROBSpeakerDiarizer:
    """Embed VAD regions and resolve session-local identities using ROB's matcher.

    Each recording has its own identity manager. This first offline adapter
    labels existing regions; speaker changes inside a region require subsequent
    word-timed segmentation rather than changing the region's boundaries here.
    """

    DEFAULT_REPOSITORY = "PalabraAI/redimnet2:c5bbe0b76e37df698c403f8844e41304ceab6307"
    DEFAULT_MODEL_NAME = "b6"
    DEFAULT_TRAIN_TYPE = "lm"
    DEFAULT_DATASET = "vb2+vox2+cnc2_v0"

    def __init__(
        self,
        *,
        embedder: Callable[["np.ndarray"], "np.ndarray"],
        min_embed_duration: float = 0.8,
        min_create_duration: float = 1.5,
        max_embed_duration: float = 2.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if (
            not all(
                math.isfinite(value) and value > 0
                for value in (
                    min_embed_duration,
                    min_create_duration,
                    max_embed_duration,
                )
            )
            or not min_embed_duration <= min_create_duration <= max_embed_duration
        ):
            raise ValueError(
                "embedding durations must be finite, positive and ordered: "
                "min_embed_duration <= min_create_duration <= max_embed_duration"
            )
        self._embedder = embedder
        self.min_embed_duration = min_embed_duration
        self.min_create_duration = min_create_duration
        self.max_embed_duration = max_embed_duration
        self._logger = logger or logging.getLogger(__name__)

    @classmethod
    def from_torch_hub(
        cls,
        *,
        weights_dir: Union[str, Path],
        repository: str = DEFAULT_REPOSITORY,
        model_name: str = DEFAULT_MODEL_NAME,
        train_type: str = DEFAULT_TRAIN_TYPE,
        dataset: str = DEFAULT_DATASET,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "ROBSpeakerDiarizer":
        """Explicitly load ROB's pinned ReDimNet2 model without ROS or MongoDB."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to load ReDimNet2 diarization"
            ) from exc

        model_dir = Path(weights_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(model_dir))
        selected_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = torch.hub.load(
            repo_or_dir=repository,
            model="redimnet2",
            model_name=model_name,
            train_type=train_type,
            dataset=dataset,
            pretrained=True,
            trust_repo=True,
        )
        model.eval().to(selected_device)

        def embed(audio: "np.ndarray") -> "np.ndarray":
            import numpy as np

            with torch.no_grad():
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
                )
                vector = model(tensor.to(selected_device)).squeeze(0)
            return vector.detach().float().cpu().numpy()

        return cls(embedder=embed, logger=logger)

    def diarize(
        self,
        audio: "np.ndarray",
        sample_rate: int,
        regions: Sequence[SpeechRegion],
    ) -> list[SpeakerInterval]:
        """Label each existing region; short unmatched audio remains UNKNOWN."""
        import numpy as np

        from ._voice_identity import VoiceIdentityManager, normalize_embedding

        if getattr(audio, "ndim", None) != 1:
            raise ValueError("audio must be a one-dimensional array")
        if str(getattr(audio, "dtype", "")) != "float32":
            raise TypeError("audio must use float32 samples")
        if sample_rate != 16000:
            raise ValueError("ReDimNet2 audio must use a 16000 Hz sample rate")
        if not np.isfinite(audio).all():
            raise ValueError("audio must contain finite samples")

        ordered = sorted(regions, key=lambda region: region.start_sample)
        previous_end = 0
        for region in ordered:
            if region.end_sample > audio.size:
                raise ValueError("speech region exceeds the source audio")
            if region.start_sample < previous_end:
                raise ValueError("speech regions must not overlap")
            previous_end = region.end_sample

        manager = VoiceIdentityManager(
            logger=self._logger,
            short_window_seconds=1.5,
            short_window_threshold=0.40,
            # Offline compute time must not expire a voice in a long recording.
            identity_timeout=float("inf"),
        )
        observations = []
        embedding_size = None
        for index, region in enumerate(ordered):
            track = f"region{index}"
            seconds = min(
                region.duration_samples / sample_rate, self.max_embed_duration
            )
            embedding = None
            result = None
            if seconds >= self.min_embed_duration:
                start = max(
                    region.start_sample,
                    region.end_sample - round(self.max_embed_duration * sample_rate),
                )
                window = np.ascontiguousarray(audio[start : region.end_sample])
                # Digital silence cannot establish or update an identity.
                if np.any(window):
                    embedding = normalize_embedding(self._embedder(window))
                    if embedding_size is not None and embedding.size != embedding_size:
                        raise ValueError(
                            "speaker embedding dimension changed within a recording"
                        )
                    embedding_size = embedding.size
                    result = manager.process_new_embedding_batch(
                        {track: embedding},
                        speech_seconds=seconds,
                        learn=True,
                        allow_create=seconds >= self.min_create_duration,
                        allow_create_when_ambiguous=False,
                    ).get(track)
            observations.append((region, track, embedding, seconds, result))

        # Offline attribution can resolve an earlier short turn once a later
        # region has established that voice. This pass never creates or learns.
        resolved = []
        for region, track, embedding, seconds, result in observations:
            if result is None and embedding is not None:
                result = manager.process_new_embedding_batch(
                    {track: embedding},
                    speech_seconds=seconds,
                    learn=False,
                    allow_create=False,
                ).get(track)
            resolved.append((region, track, result))

        timeline = []
        for region, track, result in resolved:
            speaker = manager.track_to_identity.get(track, "UNKNOWN")
            confidence = (
                -1.0 if result is None or result[1] < 0 else min(1.0, float(result[1]))
            )
            timeline.append(
                SpeakerInterval(
                    region.start_sample,
                    region.end_sample,
                    speaker.removeprefix("EUT_"),
                    confidence,
                )
            )
        return timeline
