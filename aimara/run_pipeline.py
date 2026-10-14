"""Command-line runner for the ROB-to-ERC AIMARA pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .erc_encoders import OriolAudioEncoder, OriolTextEncoder, OriolVisualEncoder
from .erc_features import ERCFeatureBridge
from .erc_model import ERCEmotionModel, ERCModelProfile
from .offline_processor import OfflineROBProcessor
from .pipeline import AIMARAPipeline
from .rob import (
    ParakeetConfig,
    ParakeetTranscriber,
    ROBSpeakerDiarizer,
    ROBTranscriber,
    ROBVoiceActivityDetector,
    WhisperInferenceConfig,
)
from .rob.asr import WHISPER_COMPUTE_TYPES
from .speaker_segmentation import SpeakerSegmentationConfig, SpeakerSegmenter
from .speech_interfaces import ROBProcessorConfig
from .vad_processing import OfflineVADProcessor


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vad-weights", type=Path, required=True)
    parser.add_argument(
        "--asr-weights", type=Path, required=True, help="ASR model cache directory"
    )
    parser.add_argument(
        "--asr-backend", choices=("whisper", "parakeet"), default="whisper"
    )
    parser.add_argument(
        "--asr-model",
        default="tiny.en",
        help="Whisper model alias (Whisper backend only)",
    )
    parser.add_argument(
        "--diarization-backend", choices=("none", "redimnet2"), default="none"
    )
    parser.add_argument("--diarization-weights", type=Path)
    parser.add_argument(
        "--unknown-speaker-policy",
        choices=("omit_embedding", "reserve_slot"),
        default="omit_embedding",
        help="Omit UNKNOWN speaker embeddings, or reserve a speaker slot for legacy runs",
    )
    parser.add_argument(
        "--audio-model",
        default="iic/emotion2vec_plus_large",
        help="FunASR model ID or local emotion2vec model directory",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)

    asr_defaults = WhisperInferenceConfig()
    asr = parser.add_argument_group("Whisper inference (Whisper backend only)")
    asr.add_argument(
        "--asr-compute-type",
        choices=WHISPER_COMPUTE_TYPES,
        default=asr_defaults.compute_type,
        help="Requested CTranslate2 precision (default: %(default)s)",
    )
    asr.add_argument(
        "--asr-cpu-compute-type",
        choices=("int8", "float32"),
        default=asr_defaults.cpu_compute_type,
        help="CPU fallback for float16 and int8_float16 (default: %(default)s)",
    )
    asr.add_argument(
        "--asr-batched-inference",
        action=argparse.BooleanOptionalAction,
        default=asr_defaults.use_batched_inference,
        help="Decode chunks within each VAD region in batches",
    )
    asr.add_argument(
        "--asr-batch-size",
        type=int,
        default=asr_defaults.batch_size,
        help="Maximum Whisper chunks decoded together when batching is enabled (default: %(default)s)",
    )
    asr.add_argument(
        "--asr-beam-size",
        type=int,
        default=asr_defaults.beam_size,
        help="Whisper beam search size (default: %(default)s)",
    )

    parakeet_defaults = ParakeetConfig()
    parakeet = parser.add_argument_group("Parakeet inference (optional NeMo backend)")
    parakeet.add_argument(
        "--parakeet-model",
        default=parakeet_defaults.model_name,
        help="Hugging Face repo ID or local .nemo checkpoint (default: %(default)s)",
    )
    parakeet.add_argument(
        "--parakeet-language-id-model",
        default=parakeet_defaults.language_id_model,
        help="NeMo language identifier or .nemo path; empty disables it (default: %(default)s)",
    )
    parakeet.add_argument(
        "--parakeet-language-id-min-confidence",
        type=float,
        default=parakeet_defaults.language_id_min_confidence,
        help="Minimum confidence for automatic language metadata (default: %(default)s)",
    )
    parakeet.add_argument(
        "--parakeet-local-files-only",
        action="store_true",
        help="Use cached/local Parakeet and language-ID checkpoints",
    )
    parakeet.add_argument(
        "--parakeet-warmup",
        action=argparse.BooleanOptionalAction,
        default=parakeet_defaults.warm_up,
        help="Warm up the Parakeet model once at startup",
    )

    defaults = ROBProcessorConfig()
    speech = parser.add_argument_group("VAD segmentation")
    speech.add_argument(
        "--vad-threshold",
        type=float,
        default=defaults.vad_threshold,
        help="Speech probability threshold, 0 to 1 (default: %(default)s)",
    )
    speech.add_argument(
        "--min-speech-duration",
        type=float,
        default=defaults.min_speech_duration,
        help="Minimum detected speech duration in seconds (default: %(default)s)",
    )
    speech.add_argument(
        "--min-silence-duration",
        type=float,
        default=defaults.min_silence_duration,
        help="Silence in seconds needed to close a VAD region (default: %(default)s)",
    )
    speech.add_argument(
        "--pre-buffer-duration",
        type=float,
        default=defaults.pre_buffer_duration,
        help="Source audio in seconds retained before speech onset (default: %(default)s)",
    )
    speech.add_argument(
        "--max-segment-duration",
        type=float,
        default=defaults.max_segment_duration,
        help="Maximum detected speech span before splitting, excluding onset pre-buffer (seconds; default: %(default)s)",
    )
    speaker_defaults = SpeakerSegmentationConfig()
    speakers = parser.add_argument_group("Speaker grouping (requires diarization)")
    speakers.add_argument(
        "--snap-splits-to-sentences",
        action=argparse.BooleanOptionalAction,
        default=speaker_defaults.snap_splits_to_sentences,
        help="Keep sentences intact; disable for raw word runs",
    )
    speakers.add_argument(
        "--min-speaker-run-tokens",
        type=int,
        default=speaker_defaults.min_speaker_run_tokens,
        help="Minimum words for a credible speaker change (default: %(default)s)",
    )
    speakers.add_argument(
        "--min-speaker-run-duration",
        type=float,
        default=speaker_defaults.min_speaker_run_duration,
        help="Minimum duration in seconds for a credible speaker change (default: %(default)s)",
    )


def build_segmentation_configs(
    args: argparse.Namespace,
) -> tuple[ROBProcessorConfig, SpeakerSegmentationConfig]:
    """Validate runtime settings before loading models, including legacy namespaces."""
    speech_defaults = ROBProcessorConfig()
    speaker_defaults = SpeakerSegmentationConfig()
    speech = ROBProcessorConfig(
        **{
            name: getattr(args, name, getattr(speech_defaults, name))
            for name in (
                "vad_threshold",
                "min_speech_duration",
                "min_silence_duration",
                "pre_buffer_duration",
                "max_segment_duration",
            )
        }
    )
    speakers = SpeakerSegmentationConfig(
        **{
            name: getattr(args, name, getattr(speaker_defaults, name))
            for name in (
                "snap_splits_to_sentences",
                "min_speaker_run_tokens",
                "min_speaker_run_duration",
            )
        }
    )
    return speech, speakers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_runtime_arguments(parser)
    return parser


def build_asr_config(args: argparse.Namespace) -> WhisperInferenceConfig:
    defaults = WhisperInferenceConfig()
    return WhisperInferenceConfig(
        compute_type=getattr(args, "asr_compute_type", defaults.compute_type),
        cpu_compute_type=getattr(
            args, "asr_cpu_compute_type", defaults.cpu_compute_type
        ),
        use_batched_inference=getattr(
            args, "asr_batched_inference", defaults.use_batched_inference
        ),
        batch_size=getattr(args, "asr_batch_size", defaults.batch_size),
        beam_size=getattr(args, "asr_beam_size", defaults.beam_size),
    )


def build_parakeet_config(args: argparse.Namespace) -> ParakeetConfig:
    defaults = ParakeetConfig()
    return ParakeetConfig(
        model_name=getattr(args, "parakeet_model", defaults.model_name),
        language=args.language,
        language_id_model=getattr(
            args, "parakeet_language_id_model", defaults.language_id_model
        ),
        language_id_min_confidence=getattr(
            args,
            "parakeet_language_id_min_confidence",
            defaults.language_id_min_confidence,
        ),
        local_files_only=getattr(
            args, "parakeet_local_files_only", defaults.local_files_only
        ),
        warm_up=getattr(args, "parakeet_warmup", defaults.warm_up),
    )


def build_pipeline(args: argparse.Namespace) -> AIMARAPipeline:
    config, speaker_config = build_segmentation_configs(args)
    asr_backend = getattr(args, "asr_backend", "whisper")
    if asr_backend == "whisper":
        asr_config = build_asr_config(args)
        ROBTranscriber.validate_model_size(args.asr_model)
    elif asr_backend == "parakeet":
        parakeet_config = build_parakeet_config(args)
    else:
        raise ValueError("asr_backend must be whisper or parakeet")
    diarization_backend = getattr(args, "diarization_backend", "none")
    diarization_weights = getattr(args, "diarization_weights", None)
    if diarization_backend == "redimnet2" and diarization_weights is None:
        raise ValueError(
            "--diarization-weights is required with --diarization-backend redimnet2"
        )
    logging.getLogger(__name__).info("VAD segmentation: %s", config)
    logging.getLogger(__name__).info(
        "Speaker grouping (%s): %s", diarization_backend, speaker_config
    )
    if asr_backend == "parakeet":
        transcriber = ParakeetTranscriber.from_nemo(
            weights_dir=args.asr_weights,
            config=parakeet_config,
            device=args.device,
        )
    else:
        transcriber = ROBTranscriber.from_faster_whisper(
            model_size=args.asr_model,
            weights_dir=args.asr_weights,
            language=args.language,
            device=args.device,
            compute_type=asr_config.compute_type,
            cpu_compute_type=asr_config.cpu_compute_type,
            use_batched_inference=asr_config.use_batched_inference,
            batch_size=asr_config.batch_size,
            beam_size=asr_config.beam_size,
        )
    vad = ROBVoiceActivityDetector.from_torch_hub(
        weights_dir=args.vad_weights,
        device=args.device,
    )
    speech_processor = OfflineROBProcessor(
        vad_processor=OfflineVADProcessor(vad, config),
        transcriber=transcriber,
        speaker_segmenter=SpeakerSegmenter(speaker_config),
        diarizer=(
            ROBSpeakerDiarizer.from_torch_hub(
                weights_dir=diarization_weights, device=args.device
            )
            if diarization_backend == "redimnet2"
            else None
        ),
    )

    profile = ERCModelProfile()
    emotion_model = ERCEmotionModel.from_checkpoint(
        args.checkpoint,
        profile=profile,
        device=args.device,
    )

    text_encoder = OriolTextEncoder.load()
    from unimodal.audio.models import AudioConfig

    audio_encoder = OriolAudioEncoder.load(AudioConfig(model_id=args.audio_model))
    visual_encoder = OriolVisualEncoder.load()
    feature_bridge = ERCFeatureBridge(
        text_encoder,
        audio_encoder,
        visual_encoder,
        speaker_slots=profile.n_speakers,
        unknown_speaker_policy=getattr(
            args, "unknown_speaker_policy", "omit_embedding"
        ),
    )
    return AIMARAPipeline(speech_processor, feature_bridge, emotion_model)


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pipeline = build_pipeline(args)
    predictions = pipeline.process_video(args.video, args.output_dir)
    for item in predictions:
        print(
            f"{item.segment.start_time:.3f}-{item.segment.end_time:.3f} "
            f"{item.emotion} ({item.confidence:.3f}): {item.segment.transcript}"
        )
    print(f"Wrote {args.output_dir / 'predictions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
