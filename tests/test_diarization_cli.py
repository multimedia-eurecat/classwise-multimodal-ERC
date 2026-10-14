import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from aimara.run_batch import build_parser as batch_parser
from aimara.run_pipeline import build_parser, build_pipeline


class DiarizationRunnerTests(unittest.TestCase):
    def arguments(self, *extra):
        return build_parser().parse_args(
            [
                "--video",
                "video.mp4",
                "--output-dir",
                "output",
                "--checkpoint",
                "model.pt",
                "--vad-weights",
                "vad",
                "--asr-weights",
                "asr",
                *extra,
            ]
        )

    def test_missing_weights_fail_before_any_model_load(self):
        args = self.arguments("--diarization-backend", "redimnet2")
        with patch(
            "aimara.run_pipeline.ROBVoiceActivityDetector.from_torch_hub"
        ) as load:
            with self.assertRaisesRegex(ValueError, "diarization-weights"):
                build_pipeline(args)
        load.assert_not_called()

    def test_batch_parser_accepts_the_same_diarization_selection(self):
        args = batch_parser().parse_args(
            [
                "--video-dir",
                "videos",
                "--output-dir",
                "output",
                "--checkpoint",
                "model.pt",
                "--vad-weights",
                "vad",
                "--asr-weights",
                "asr",
                "--diarization-backend",
                "redimnet2",
                "--diarization-weights",
                "speakers",
            ]
        )
        self.assertEqual(args.diarization_weights, Path("speakers"))
        self.assertEqual(args.unknown_speaker_policy, "omit_embedding")

    def test_pipeline_wires_the_selected_diarizer_and_default_does_not_load_it(self):
        for enabled in (False, True):
            extra = [
                "--diarization-backend",
                "redimnet2",
                "--diarization-weights",
                "speakers",
            ]
            args = self.arguments(
                *(extra if enabled else []),
                "--unknown-speaker-policy",
                "reserve_slot" if enabled else "omit_embedding",
            )
            diarizer = object()
            # Avoid loading the ERC encoders; exercise real pipeline/processor construction.
            import sys
            import types

            audio_models = types.ModuleType("unimodal.audio.models")
            audio_models.AudioConfig = Mock()
            with (
                patch.dict(sys.modules, {"unimodal.audio.models": audio_models}),
                patch("aimara.run_pipeline.ROBVoiceActivityDetector.from_torch_hub"),
                patch("aimara.run_pipeline.ROBTranscriber.from_faster_whisper"),
                patch("aimara.run_pipeline.OfflineVADProcessor"),
                patch("aimara.run_pipeline.ERCEmotionModel.from_checkpoint"),
                patch("aimara.run_pipeline.OriolTextEncoder.load"),
                patch("aimara.run_pipeline.OriolAudioEncoder.load"),
                patch("aimara.run_pipeline.OriolVisualEncoder.load"),
                patch(
                    "aimara.run_pipeline.ROBSpeakerDiarizer.from_torch_hub",
                    return_value=diarizer,
                ) as load,
            ):
                pipeline = build_pipeline(args)
            self.assertIs(
                pipeline.speech_processor.diarizer, diarizer if enabled else None
            )
            self.assertEqual(
                pipeline.feature_bridge.unknown_speaker_policy,
                "reserve_slot" if enabled else "omit_embedding",
            )
            if enabled:
                load.assert_called_once_with(weights_dir=Path("speakers"), device=None)
            else:
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
