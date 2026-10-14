"""Optional offline NeMo Parakeet adapter using ROB's decoding conventions."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

from ..speech_interfaces import Transcript, TranscriptSegment, TranscriptWord
from ..speech_metadata import ASRMetadata

DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_LANGUAGES = frozenset(
    "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split()
)


@dataclass(frozen=True)
class ParakeetConfig:
    model_name: str = DEFAULT_PARAKEET_MODEL
    language: str = "auto"
    language_id_model: str = "langid_ambernet"
    language_id_min_confidence: float = 0.9
    local_files_only: bool = False
    warm_up: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("parakeet_model must not be empty")
        if self.model_name.endswith(".nemo"):
            if not Path(self.model_name).is_file():
                raise FileNotFoundError(
                    f"Parakeet checkpoint not found: {self.model_name}"
                )
        elif len(self.model_name.split("/")) != 2 or not all(
            self.model_name.split("/")
        ):
            raise ValueError(
                "parakeet_model must be a Hugging Face repo ID or a .nemo file"
            )
        if (
            not math.isfinite(self.language_id_min_confidence)
            or not 0 <= self.language_id_min_confidence <= 1
        ):
            raise ValueError(
                "parakeet_language_id_min_confidence must be between 0 and 1"
            )
        self.candidate_languages()

    def candidate_languages(self) -> tuple[str, ...]:
        if self.language.strip().casefold() == "auto":
            return tuple(sorted(PARAKEET_LANGUAGES))
        languages = tuple(
            dict.fromkeys(
                code.strip().casefold()
                for code in self.language.split(",")
                if code.strip()
            )
        )
        if not languages:
            raise ValueError(
                "Parakeet language must be auto or a supported language list"
            )
        unsupported = set(languages) - PARAKEET_LANGUAGES
        if unsupported:
            raise ValueError(
                f"Parakeet v3 does not support {sorted(unsupported)}; use --asr-backend whisper for those languages"
            )
        return languages


class ParakeetTranscriber:
    """Return the same timed Transcript as Whisper, without importing NeMo eagerly."""

    SAMPLE_RATE = 16000
    MIN_TRANSCRIPTION_DURATION = 0.01
    MIN_LANGUAGE_ID_DURATION = 0.25
    CONFIDENCE_AGGREGATION_ERROR = "word-level confidence aggregation"

    def __init__(
        self,
        *,
        model: Any,
        config: Optional[ParakeetConfig] = None,
        language_identifier: Any = None,
        device: str = "cpu",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.config = config or ParakeetConfig()
        self.device = device
        self.effective_compute_type = "float32"
        self.language_identifier = language_identifier
        self._logger = logger or logging.getLogger(__name__)

    @classmethod
    def from_nemo(
        cls,
        *,
        weights_dir: str | Path,
        config: Optional[ParakeetConfig] = None,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "ParakeetTranscriber":
        config = config or ParakeetConfig()
        try:
            import torch
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise RuntimeError(
                "Parakeet requires the optional nemo_toolkit[asr]==2.4.0 dependencies; see aimara/requirements-parakeet.txt"
            ) from exc
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        selected_logger = logger or logging.getLogger(__name__)
        checkpoint = cls._resolve_checkpoint(config, Path(weights_dir))
        model = nemo_asr.models.ASRModel.restore_from(
            str(checkpoint), map_location=selected_device
        )
        model.to(selected_device)
        model.float()
        model.eval()
        cls._configure_decoding(model)
        transcriber = cls(
            model=model, config=config, device=selected_device, logger=selected_logger
        )
        if config.warm_up:
            import numpy as np

            try:
                transcriber._transcribe_hypotheses(
                    np.zeros(cls.SAMPLE_RATE, dtype=np.float32)
                )
            except Exception as exc:
                selected_logger.warning("Parakeet warm-up failed: %s", exc)
        candidates = config.candidate_languages()
        if len(candidates) > 1 and config.language_id_model:
            try:
                transcriber.language_identifier = ParakeetLanguageIdentifier.load(
                    config.language_id_model,
                    Path(weights_dir),
                    selected_device,
                    local_files_only=config.local_files_only,
                )
            except Exception as exc:
                selected_logger.warning(
                    "Parakeet language identification unavailable; language will be unset: %s",
                    exc,
                )
        selected_logger.info(
            "Parakeet %s: device=%s, compute_type=float32, languages=%s",
            config.model_name,
            selected_device,
            candidates,
        )
        return transcriber

    @staticmethod
    def _resolve_checkpoint(config: ParakeetConfig, weights_dir: Path) -> Path:
        if config.model_name.endswith(".nemo"):
            return Path(config.model_name)
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        filename = config.model_name.split("/")[-1] + ".nemo"
        cached = try_to_load_from_cache(
            repo_id=config.model_name, filename=filename, cache_dir=str(weights_dir)
        )
        if isinstance(cached, str) and Path(cached).is_file():
            return Path(cached)
        return Path(
            hf_hub_download(
                repo_id=config.model_name,
                filename=filename,
                cache_dir=str(weights_dir),
                local_files_only=config.local_files_only,
            )
        )

    @staticmethod
    def _configure_decoding(model: Any, *, word_confidence: bool = True) -> None:
        from omegaconf import open_dict

        with open_dict(model.cfg.decoding):
            model.cfg.decoding.compute_timestamps = True
            model.cfg.decoding.preserve_alignments = True
            model.cfg.decoding.confidence_cfg = {
                "preserve_word_confidence": word_confidence,
                "aggregation": "mean",
            }
        model.change_decoding_strategy(model.cfg.decoding, verbose=False)

    def _transcribe_hypotheses(self, audio: Any) -> tuple[Sequence[Any], bool]:
        import torch

        with torch.inference_mode():
            try:
                return self.model.transcribe(
                    [audio], batch_size=1, return_hypotheses=True, verbose=False
                ), True
            except (ValueError, RuntimeError) as exc:
                if self.CONFIDENCE_AGGREGATION_ERROR not in str(exc):
                    raise
                self._logger.warning(
                    "NeMo word-confidence aggregation failed; retrying without confidence: %s",
                    exc,
                )
                self._configure_decoding(self.model, word_confidence=False)
                try:
                    return self.model.transcribe(
                        [audio], batch_size=1, return_hypotheses=True, verbose=False
                    ), False
                finally:
                    self._configure_decoding(self.model, word_confidence=True)

    def transcribe(self, audio: Any, sample_rate: int) -> Transcript:
        if getattr(audio, "ndim", None) != 1:
            raise ValueError("audio must be a one-dimensional array")
        if str(getattr(audio, "dtype", "")) != "float32":
            raise TypeError("audio must use float32 samples")
        if sample_rate != self.SAMPLE_RATE:
            raise ValueError("Parakeet audio must use a 16000 Hz sample rate")
        if audio.size < round(sample_rate * self.MIN_TRANSCRIPTION_DURATION):
            return Transcript("", metadata=self._metadata())
        try:
            hypotheses, confidence_available = self._transcribe_hypotheses(audio)
            if not hypotheses:
                return Transcript("", metadata=self._metadata())
            if len(hypotheses) != 1:
                raise ValueError(
                    "Parakeet must return one hypothesis for one audio region"
                )
            result = self._to_transcript(hypotheses[0], audio.size / sample_rate)
            if not confidence_available:
                result = replace(result, metadata=self._metadata())
        except Exception as exc:
            raise RuntimeError("Parakeet transcription failed") from exc
        if not result.text:
            return result
        language, language_source = self._identify_language(audio)
        return replace(
            result,
            language=language,
            metadata=replace(result.metadata, language_source=language_source),
        )

    def _metadata(self, confidence: Optional[float] = None) -> ASRMetadata:
        return ASRMetadata(
            backend="parakeet",
            model=self.config.model_name,
            transcript_confidence=confidence,
            confidence_method="mean_word_confidence"
            if confidence is not None
            else None,
            device=self.device,
            compute_type=self.effective_compute_type,
        )

    def _hypothesis_confidence(self, hypothesis: Any) -> Optional[float]:
        scores = getattr(hypothesis, "word_confidence", None)
        if scores is None:
            return None
        try:
            values = [float(value) for value in scores]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError("word confidences must be finite and between 0 and 1")
            return math.fsum(values) / len(values) if values else None
        except (ValueError, TypeError) as exc:
            self._logger.warning("Parakeet confidence unavailable: %s", exc)
            return None

    def _identify_language(self, audio: Any) -> tuple[Optional[str], Optional[str]]:
        candidates = self.config.candidate_languages()
        if len(candidates) == 1:
            # A configured output label, not a detected or forced language.
            return candidates[0], "configured"
        if (
            self.language_identifier is None
            or audio.size < self.MIN_LANGUAGE_ID_DURATION * self.SAMPLE_RATE
        ):
            return None, None
        try:
            probabilities = self.language_identifier.probabilities(audio, candidates)
            valid = {
                code: float(score)
                for code, score in probabilities.items()
                if code in candidates and math.isfinite(score) and 0 <= score <= 1
            }
            if valid:
                best = max(valid, key=valid.get)
                if valid[best] >= self.config.language_id_min_confidence:
                    return best, "detected"
        except Exception as exc:
            self._logger.warning("Parakeet language identification failed: %s", exc)
        return None, None

    def _entry_times(self, entry: dict) -> tuple[float, float]:
        if "start" in entry and "end" in entry:
            return float(entry["start"]), float(entry["end"])
        # NeMo 2.4 raw offsets use encoder subsampling * preprocessor stride.
        scale = float(self.model.encoder.subsampling_factor) * float(
            self.model.cfg.preprocessor.window_stride
        )
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("NeMo timestamp scale must be finite and positive")
        return float(entry["start_offset"]) * scale, float(entry["end_offset"]) * scale

    def _to_transcript(self, hypothesis: Any, duration: float) -> Transcript:
        text = (getattr(hypothesis, "text", None) or "").strip()
        if not text:
            return Transcript("", metadata=self._metadata())
        stamps = getattr(hypothesis, "timestamp", None)
        stamps = stamps if isinstance(stamps, dict) else {}
        words = tuple(
            TranscriptWord(*self._entry_times(entry), " " + entry["word"].strip())
            for entry in (stamps.get("word") or [])
            if (entry.get("word") or "").strip()
        )
        if any(
            right.start_time < left.start_time for left, right in zip(words, words[1:])
        ):
            raise ValueError("Parakeet word times must be chronological")
        segments = []
        for entry in stamps.get("segment") or []:
            segment_text = (entry.get("segment") or "").strip()
            if segment_text:
                segments.append(
                    TranscriptSegment(*self._entry_times(entry), segment_text)
                )

        def normalize(value: str) -> str:
            return " ".join(value.split())

        if segments and normalize(" ".join(s.text for s in segments)) == normalize(
            text
        ):
            if any(
                right.start_time < left.start_time or right.start_time < left.end_time
                for left, right in zip(segments, segments[1:])
            ):
                segments = []
            else:
                # Assign each word to at most one sentence, including a word
                # whose start equals two neighbouring sentence boundaries.
                assigned: list[list[TranscriptWord]] = [[] for _ in segments]
                for word in words:
                    for index, segment in enumerate(segments):
                        if segment.start_time <= word.start_time < segment.end_time or (
                            index == len(segments) - 1
                            and word.start_time == segment.end_time
                        ):
                            assigned[index].append(word)
                            break
                if sum(map(len, assigned)) != len(words):
                    segments = []
                else:
                    segments = [
                        TranscriptSegment(
                            s.start_time, s.end_time, s.text, tuple(group)
                        )
                        for s, group in zip(segments, assigned)
                    ]
        else:
            segments = []
        if not segments:
            # Keep every character when sentence metadata is incomplete. The
            # shared speaker grouper handles complete words or text-only spans.
            start = words[0].start_time if words else 0.0
            end = max(w.end_time for w in words) if words else duration
            segments = [TranscriptSegment(start, end, text, words)]
        return Transcript(
            text,
            segments=tuple(segments),
            metadata=self._metadata(self._hypothesis_confidence(hypothesis)),
        )


class ParakeetLanguageIdentifier:
    """Optional AmberNet labels for the same 16 kHz audio region."""

    def __init__(self, model: Any, device: str) -> None:
        self.model, self.device = model, device
        self.labels = tuple(model.cfg.train_ds.labels)

    @classmethod
    def load(
        cls,
        model_name: str,
        weights_dir: Path,
        device: str,
        *,
        local_files_only: bool = False,
    ) -> "ParakeetLanguageIdentifier":
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        if model_name.endswith(".nemo"):
            checkpoint = Path(model_name)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Language ID checkpoint not found: {checkpoint}"
                )
        else:
            available = {
                info.pretrained_model_name: info.location
                for info in EncDecSpeakerLabelModel.list_available_models() or []
            }
            if model_name not in available:
                raise ValueError(
                    f"Unknown NeMo language identification model: {model_name}"
                )
            url = available[model_name]
            filename = url.split("/")[-1]
            checkpoint = weights_dir / "nemo" / model_name / filename
            if not checkpoint.is_file():
                if local_files_only:
                    raise FileNotFoundError(
                        f"Language ID checkpoint is not cached: {checkpoint}"
                    )
                from nemo.utils.cloud import maybe_download_from_cloud

                checkpoint = Path(
                    maybe_download_from_cloud(
                        url=url[: -len(filename)],
                        filename=filename,
                        cache_dir=weights_dir / "nemo",
                        subfolder=model_name,
                    )
                )
        model = EncDecSpeakerLabelModel.restore_from(
            str(checkpoint), map_location=device
        )
        model.to(device)
        model.float()
        model.eval()
        return cls(model, device)

    def probabilities(self, audio: Any, languages: Sequence[str]) -> dict[str, float]:
        import torch

        codes = [code for code in languages if code in self.labels]
        if not codes:
            return {}
        with torch.inference_mode():
            logits, _ = self.model.forward(
                input_signal=torch.as_tensor(
                    audio, dtype=torch.float32, device=self.device
                )[None],
                input_signal_length=torch.tensor([len(audio)], device=self.device),
            )
            selected = logits[0, [self.labels.index(code) for code in codes]].float()
            probabilities = torch.softmax(selected, dim=-1).cpu().tolist()
        return dict(zip(codes, probabilities))
