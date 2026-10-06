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
from .rob import ROBTranscriber, ROBVoiceActivityDetector
from .speech_interfaces import ROBProcessorConfig
from .vad_processing import OfflineVADProcessor


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vad-weights", type=Path, required=True)
    parser.add_argument("--asr-weights", type=Path, required=True)
    parser.add_argument("--asr-model", default="tiny.en")
    parser.add_argument(
        "--audio-model",
        default="iic/emotion2vec_plus_large",
        help="FunASR model ID or local emotion2vec model directory",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_runtime_arguments(parser)
    return parser


def build_pipeline(args: argparse.Namespace) -> AIMARAPipeline:
    config = ROBProcessorConfig()
    vad = ROBVoiceActivityDetector.from_torch_hub(
        weights_dir=args.vad_weights,
        device=args.device,
    )
    transcriber = ROBTranscriber.from_faster_whisper(
        model_size=args.asr_model,
        weights_dir=args.asr_weights,
        language=args.language,
        device=args.device,
    )
    speech_processor = OfflineROBProcessor(
        vad_processor=OfflineVADProcessor(vad, config),
        transcriber=transcriber,
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
