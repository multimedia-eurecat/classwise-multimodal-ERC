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
    """Resolve session-local voices and probe speaker changes within VAD regions.

    Stable long windows establish the gallery; short centred probes label the
    source timeline without learning from windows across a speaker change.
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
        probe_duration: float = 1.0,
        step_duration: float = 0.5,
        change_threshold: float = 0.35,
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
        if (
            not math.isfinite(probe_duration)
            or not min_embed_duration <= probe_duration <= max_embed_duration
        ):
            raise ValueError(
                "probe_duration must be between minimum and maximum embedding duration"
            )
        if not math.isfinite(step_duration) or not 0 < step_duration <= probe_duration:
            raise ValueError(
                "step_duration must be positive and no larger than probe_duration"
            )
        if not math.isfinite(change_threshold) or not 0 <= change_threshold <= 1:
            raise ValueError("change_threshold must be between 0 and 1")
        self.probe_duration = probe_duration
        self.step_duration = step_duration
        self.change_threshold = change_threshold
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
        """Build a non-overlapping source timeline, including mid-region changes."""
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
        cache = {}
        embedding_size = None

        def embed(start: int, end: int):
            nonlocal embedding_size
            key = (start, end)
            if key not in cache:
                window = np.ascontiguousarray(audio[start:end])
                vector = None
                if len(window) / sample_rate >= self.min_embed_duration and np.any(
                    window
                ):
                    vector = normalize_embedding(self._embedder(window))
                    if embedding_size is not None and vector.size != embedding_size:
                        raise ValueError(
                            "speaker embedding dimension changed within a recording"
                        )
                    embedding_size = vector.size
                cache[key] = vector
            return cache[key]

        step = max(1, round(self.step_duration * sample_rate))
        minimum = math.ceil(self.min_embed_duration * sample_rate)
        maximum = round(self.max_embed_duration * sample_rate)
        learned_until = {}
        for region in ordered:
            length = min(maximum, region.duration_samples)
            if length / sample_rate < self.min_create_duration:
                continue
            last_start = region.end_sample - length
            starts = list(range(region.start_sample, last_start + 1, step))
            if starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                end = start + length
                # An embedding spanning unlike voices must not seed or train a
                # speaker. ROB's change score is applied to the two end probes.
                head, tail = embed(start, start + minimum), embed(end - minimum, end)
                if (
                    head is None
                    or tail is None
                    or float(head @ tail) < self.change_threshold
                ):
                    continue
                vector = embed(start, end)
                if vector is None:
                    continue
                track = f"gallery:{start}:{end}"
                known = set(manager.identities)
                result = manager.process_new_embedding_batch(
                    {track: vector},
                    speech_seconds=length / sample_rate,
                    learn=False,
                    allow_create=True,
                    allow_create_when_ambiguous=False,
                ).get(track)
                if result is None:
                    continue
                speaker = result[0]
                last_learning = max(
                    (
                        until
                        for prior_track, until in learned_until.items()
                        if manager.track_to_identity.get(prior_track) == speaker
                    ),
                    default=0,
                )
                if speaker not in known:
                    learned_until[track] = end
                elif start >= last_learning:
                    # Overlapping windows must not count the same speech twice
                    # towards confirmation or repeatedly move the gallery mean.
                    manager.process_new_embedding_batch(
                        {track: vector},
                        speech_seconds=length / sample_rate,
                        learn=True,
                        allow_create=False,
                    )
                    learned_until[track] = end

        # Resolve probes after the gallery is established, so an early short
        # turn can use a voice learned later in the recording. Probes never learn.
        records = []
        for region_index, region in enumerate(ordered):
            width = min(
                round(self.probe_duration * sample_rate), region.duration_samples
            )
            for start in range(region.start_sample, region.end_sample, step):
                end = min(start + step, region.end_sample)
                middle = (start + end) // 2
                probe_start = max(
                    region.start_sample,
                    min(middle - width // 2, region.end_sample - width),
                )
                vector = embed(probe_start, probe_start + width)
                track = f"probe:{region_index}:{start}"
                result = (
                    None
                    if vector is None
                    else manager.process_new_embedding_batch(
                        {track: vector},
                        speech_seconds=width / sample_rate,
                        learn=False,
                        allow_create=False,
                    ).get(track)
                )
                records.append((region_index, start, end, track, result))

        timeline = []
        previous_region = None
        for region_index, start, end, track, result in records:
            speaker = manager.track_to_identity.get(track, "UNKNOWN").removeprefix(
                "EUT_"
            )
            confidence = (
                -1.0 if result is None or result[1] < 0 else min(1.0, float(result[1]))
            )
            if (
                timeline
                and region_index == previous_region
                and timeline[-1].speaker_id == speaker
            ):
                previous = timeline.pop()
                total = end - previous.start_sample
                confidence = (
                    (
                        previous.confidence * previous.duration_samples
                        + confidence * (end - start)
                    )
                    / total
                    if confidence >= 0 and previous.confidence >= 0
                    else -1.0
                )
                start = previous.start_sample
            timeline.append(SpeakerInterval(start, end, speaker, confidence))
            previous_region = region_index
        return timeline
