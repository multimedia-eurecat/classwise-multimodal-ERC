"""Verify both runners configure the real Whisper adapter before model loading."""

import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aimara.rob import WhisperInferenceConfig
from aimara.run_batch import build_parser as batch_parser
from aimara.run_pipeline import build_asr_config, build_parser, build_pipeline


def arguments(batch=False, extra=()):
    parser = batch_parser() if batch else build_parser()
    return parser.parse_args(
        [
            "--video-dir" if batch else "--video",
            "video",
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


class WhisperRunnerTests(unittest.TestCase):
    def test_both_runners_keep_whisper_defaults_and_old_namespaces(self):
        self.assertEqual(build_asr_config(SimpleNamespace()), WhisperInferenceConfig())
        for batch in (False, True):
            with self.subTest(batch=batch):
                self.assertEqual(
                    build_asr_config(arguments(batch)), WhisperInferenceConfig()
                )

    def test_both_runners_wire_precision_batching_and_beam_size_to_real_adapter(self):
        audio_models = types.ModuleType("unimodal.audio.models")
        audio_models.AudioConfig = Mock()
        for batch in (False, True):
            for enabled in (False, True):
                with (
                    self.subTest(batch=batch, enabled=enabled),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    args = arguments(
                        batch,
                        (
                            "--asr-weights",
                            directory,
                            "--asr-compute-type",
                            "float16",
                            "--asr-cpu-compute-type",
                            "float32",
                            "--asr-batched-inference"
                            if enabled
                            else "--no-asr-batched-inference",
                            "--asr-batch-size",
                            "3",
                            "--asr-beam-size",
                            "2",
                        ),
                    )
                    model, batched = object(), object()
                    loader = Mock(return_value=model)
                    batch_wrapper = Mock(return_value=batched)
                    with (
                        patch.dict(
                            sys.modules,
                            {
                                "ctranslate2": SimpleNamespace(
                                    get_cuda_device_count=lambda: 0
                                ),
                                "faster_whisper": SimpleNamespace(
                                    WhisperModel=loader,
                                    BatchedInferencePipeline=batch_wrapper,
                                ),
                                "unimodal.audio.models": audio_models,
                            },
                        ),
                        patch(
                            "aimara.run_pipeline.ROBVoiceActivityDetector.from_torch_hub",
                            return_value=object(),
                        ),
                        patch("aimara.run_pipeline.ERCEmotionModel.from_checkpoint"),
                        patch("aimara.run_pipeline.OriolTextEncoder.load"),
                        patch("aimara.run_pipeline.OriolAudioEncoder.load"),
                        patch("aimara.run_pipeline.OriolVisualEncoder.load"),
                    ):
                        pipeline = build_pipeline(args)
                    asr = pipeline.speech_processor.transcriber
                    self.assertEqual(
                        asr.inference_config,
                        WhisperInferenceConfig(
                            compute_type="float16",
                            cpu_compute_type="float32",
                            use_batched_inference=enabled,
                            batch_size=3,
                            beam_size=2,
                        ),
                    )
                    self.assertEqual(asr.effective_compute_type, "float32")
                    loader.assert_called_once_with(
                        "tiny.en",
                        device="cpu",
                        compute_type="float32",
                        download_root=directory,
                    )
                    if enabled:
                        batch_wrapper.assert_called_once_with(model=model)
                        self.assertIs(asr._batched_model, batched)
                    else:
                        batch_wrapper.assert_not_called()

    def test_invalid_whisper_settings_fail_before_any_model_load(self):
        for batch in (False, True):
            for flag, value in (
                ("--asr-batch-size", "0"),
                ("--asr-beam-size", "-1"),
                ("--asr-model", "unknown"),
            ):
                with (
                    self.subTest(batch=batch, flag=flag),
                    patch(
                        "aimara.run_pipeline.ROBVoiceActivityDetector.from_torch_hub"
                    ) as load,
                ):
                    with self.assertRaises(ValueError):
                        build_pipeline(arguments(batch, (flag, value)))
                    load.assert_not_called()
        args = arguments()
        args.asr_compute_type = "fp17"
        with patch(
            "aimara.run_pipeline.ROBVoiceActivityDetector.from_torch_hub"
        ) as load:
            with self.assertRaisesRegex(ValueError, "compute_type"):
                build_pipeline(args)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
