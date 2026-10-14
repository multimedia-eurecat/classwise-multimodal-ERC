import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from aimara.rob import ParakeetConfig
from aimara.run_batch import build_parser as batch_parser
from aimara.run_pipeline import build_parser, build_pipeline


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


class ParakeetRunnerTests(unittest.TestCase):
    def test_both_runners_select_only_the_requested_backend(self):
        audio_models = types.ModuleType("unimodal.audio.models")
        audio_models.AudioConfig = Mock()
        for batch in (False, True):
            for backend in ("whisper", "parakeet"):
                with self.subTest(batch=batch, backend=backend), ExitStack() as stack:
                    stack.enter_context(
                        patch.dict(sys.modules, {"unimodal.audio.models": audio_models})
                    )
                    for target in (
                        "ERCEmotionModel.from_checkpoint",
                        "OriolTextEncoder.load",
                        "OriolAudioEncoder.load",
                        "OriolVisualEncoder.load",
                    ):
                        stack.enter_context(patch("aimara.run_pipeline." + target))
                    stack.enter_context(
                        patch(
                            "aimara.run_pipeline.ROBVoiceActivityDetector.from_torch_hub",
                            return_value=object(),
                        )
                    )
                    whisper = stack.enter_context(
                        patch(
                            "aimara.run_pipeline.ROBTranscriber.from_faster_whisper",
                            return_value=object(),
                        )
                    )
                    parakeet = stack.enter_context(
                        patch(
                            "aimara.run_pipeline.ParakeetTranscriber.from_nemo",
                            return_value=object(),
                        )
                    )
                    extra = (
                        ()
                        if backend == "whisper"
                        else (
                            "--asr-backend",
                            "parakeet",
                            "--language",
                            "en,es",
                            "--parakeet-language-id-min-confidence",
                            "0.8",
                            "--parakeet-language-id-model",
                            "langid_ambernet",
                            "--parakeet-local-files-only",
                            "--no-parakeet-warmup",
                            # Whisper-only options have no effect on Parakeet.
                            "--asr-model",
                            "not-a-whisper-model",
                            "--asr-batch-size",
                            "0",
                        )
                    )
                    args = arguments(batch, extra)
                    pipeline = build_pipeline(args)
                    if backend == "parakeet":
                        whisper.assert_not_called()
                        parakeet.assert_called_once_with(
                            weights_dir=Path("asr"),
                            config=ParakeetConfig(
                                language="en,es",
                                language_id_min_confidence=0.8,
                                local_files_only=True,
                                warm_up=False,
                            ),
                            device=None,
                        )
                        self.assertIs(
                            pipeline.speech_processor.transcriber, parakeet.return_value
                        )
                    else:
                        parakeet.assert_not_called()
                        whisper.assert_called_once()
                        self.assertIs(
                            pipeline.speech_processor.transcriber, whisper.return_value
                        )

    def test_invalid_parakeet_settings_fail_before_any_model_load(self):
        cases = (
            ("--language", "ca"),
            ("--language", "en,ca"),
            ("--parakeet-language-id-min-confidence", "nan"),
            ("--parakeet-language-id-min-confidence", "1.1"),
            ("--parakeet-model", ""),
            ("--parakeet-model", "/tmp/aimara-missing-model.nemo"),
        )
        for batch in (False, True):
            for extra in cases:
                with (
                    self.subTest(batch=batch, extra=extra),
                    patch(
                        "aimara.run_pipeline.ROBVoiceActivityDetector.from_torch_hub"
                    ) as vad,
                    patch(
                        "aimara.run_pipeline.ParakeetTranscriber.from_nemo"
                    ) as parakeet,
                ):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        build_pipeline(
                            arguments(batch, ("--asr-backend", "parakeet", *extra))
                        )
                    vad.assert_not_called()
                    parakeet.assert_not_called()

    def test_missing_nemo_fails_before_vad_loading(self):
        for batch in (False, True):
            with (
                self.subTest(batch=batch),
                patch.dict(sys.modules, {"nemo": None}),
                patch(
                    "aimara.run_pipeline.ROBVoiceActivityDetector.from_torch_hub"
                ) as vad,
            ):
                with self.assertRaisesRegex(RuntimeError, "optional nemo_toolkit"):
                    build_pipeline(arguments(batch, ("--asr-backend", "parakeet")))
                vad.assert_not_called()


if __name__ == "__main__":
    unittest.main()
