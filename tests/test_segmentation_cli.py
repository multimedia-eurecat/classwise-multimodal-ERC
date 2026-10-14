"""Check that runner options reach actual VAD and speaker grouping behavior."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from aimara import (
    ROBProcessorConfig,
    SpeakerInterval,
    SpeakerSegmentationConfig,
    SpeechRegion,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    VADSegmenter,
)
from aimara.run_batch import build_parser as batch_parser
from aimara.run_pipeline import build_parser, build_pipeline, build_segmentation_configs


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


class SegmentationRunnerTests(unittest.TestCase):
    def test_both_runners_retain_existing_defaults_and_fixed_audio_format(self):
        for batch in (False, True):
            with self.subTest(batch=batch):
                speech, speakers = build_segmentation_configs(arguments(batch))
                self.assertEqual(speech, ROBProcessorConfig())
                self.assertEqual(speakers, SpeakerSegmentationConfig())
                self.assertEqual(speech.sample_rate, 16000)
                self.assertEqual(speech.chunk_size, 512)

    def test_legacy_namespaces_keep_the_same_segmentation_defaults(self):
        self.assertEqual(
            build_segmentation_configs(SimpleNamespace()),
            (ROBProcessorConfig(), SpeakerSegmentationConfig()),
        )

    def test_custom_vad_options_change_sample_boundaries_and_threshold(self):
        options = (
            "--vad-threshold",
            "0.6",
            "--min-speech-duration",
            "0.064",
            "--min-silence-duration",
            "0.032",
            "--pre-buffer-duration",
            "0.032",
            "--max-segment-duration",
            "0.096",
        )
        probabilities = [0.1, 0.7, 0.7, 0.7, 0.1, 0.7, 0.7, 0.1]
        for batch in (False, True):
            with self.subTest(batch=batch):
                config, _ = build_segmentation_configs(arguments(batch, options))
                regions = VADSegmenter(config).segment(
                    probabilities, total_samples=4096
                )
                self.assertEqual(
                    regions, [SpeechRegion(0, 2048), SpeechRegion(2048, 3584)]
                )
                high_threshold, _ = build_segmentation_configs(
                    arguments(batch, (*options, "--vad-threshold", "0.8"))
                )
                self.assertEqual(
                    VADSegmenter(high_threshold).segment(
                        probabilities, total_samples=4096
                    ),
                    [],
                )

    def test_selected_grouping_policy_reaches_offline_processor_and_changes_splits(
        self,
    ):
        words = tuple(
            TranscriptWord(*args)
            for args in (
                (0, 0.5, " The"),
                (0.5, 1.5, " speaker"),
                (1.5, 1.8, " changed"),
                (1.8, 2, " now."),
            )
        )
        transcript = Transcript(
            "The speaker changed now.",
            "en",
            (TranscriptSegment(0, 2, "The speaker changed now.", words),),
        )
        timeline = [SpeakerInterval(0, 24000, "A"), SpeakerInterval(24000, 32000, "B")]
        audio_models = types.ModuleType("unimodal.audio.models")
        audio_models.AudioConfig = Mock()
        for batch in (False, True):
            for snap in (False, True):
                with self.subTest(batch=batch, snap=snap):
                    args = arguments(
                        batch,
                        (
                            "--diarization-backend",
                            "redimnet2",
                            "--diarization-weights",
                            "speakers",
                            "--snap-splits-to-sentences"
                            if snap
                            else "--no-snap-splits-to-sentences",
                            "--min-speaker-run-tokens",
                            "2",
                            "--min-speaker-run-duration",
                            "0.4",
                            "--min-speech-duration",
                            "0",
                            "--min-silence-duration",
                            "0.25",
                            "--pre-buffer-duration",
                            "0.5",
                            "--max-segment-duration",
                            "5",
                        ),
                    )
                    detector = SimpleNamespace(
                        EXPECTED_CHUNK_SIZE=512, predict=lambda samples, rate: 1.0
                    )
                    transcriber = SimpleNamespace(
                        transcribe=lambda samples, rate: transcript
                    )
                    diarizer = SimpleNamespace(
                        diarize=lambda audio, rate, regions: timeline
                    )
                    with (
                        patch.dict(
                            sys.modules, {"unimodal.audio.models": audio_models}
                        ),
                        patch(
                            "aimara.run_pipeline.ROBVoiceActivityDetector.from_torch_hub",
                            return_value=detector,
                        ),
                        patch(
                            "aimara.run_pipeline.ROBTranscriber.from_faster_whisper",
                            return_value=transcriber,
                        ),
                        patch(
                            "aimara.run_pipeline.ROBSpeakerDiarizer.from_torch_hub",
                            return_value=diarizer,
                        ),
                        patch("aimara.run_pipeline.ERCEmotionModel.from_checkpoint"),
                        patch("aimara.run_pipeline.OriolTextEncoder.load"),
                        patch("aimara.run_pipeline.OriolAudioEncoder.load"),
                        patch("aimara.run_pipeline.OriolVisualEncoder.load"),
                    ):
                        pipeline = build_pipeline(args)
                    processor = pipeline.speech_processor
                    self.assertEqual(
                        processor.vad_processor.config.min_silence_duration, 0.25
                    )
                    self.assertEqual(
                        processor.vad_processor.config.pre_buffer_duration, 0.5
                    )
                    self.assertEqual(
                        processor.vad_processor.config.max_segment_duration, 5
                    )
                    processor._audio_decoder = lambda path, sample_rate: np.ones(
                        32000, dtype=np.float32
                    )
                    with tempfile.TemporaryDirectory() as directory:
                        segments = processor.process_video(
                            Path(directory) / "video.mp4", directory
                        )
                    self.assertEqual(
                        [s.transcript for s in segments],
                        [transcript.text] if snap else ["The speaker", "changed now."],
                    )
                    self.assertEqual(
                        [s.speaker_id for s in segments], ["B"] if snap else ["A", "B"]
                    )

    def test_smoothing_thresholds_change_credible_trailing_speaker_switch(self):
        words = tuple(
            TranscriptWord(*args)
            for args in (
                (0, 0.5, " The"),
                (0.5, 1.5, " speaker"),
                (1.5, 1.8, " changed"),
                (1.8, 2, " now."),
            )
        )
        transcript = Transcript(
            "The speaker changed now.",
            "en",
            (TranscriptSegment(0, 2, "The speaker changed now.", words),),
        )
        from aimara import SpeakerSegmenter

        for flag, value in (
            ("--min-speaker-run-tokens", "3"),
            ("--min-speaker-run-duration", "0.6"),
        ):
            with self.subTest(flag=flag):
                _, config = build_segmentation_configs(arguments(extra=(flag, value)))
                groups = SpeakerSegmenter(config).split(
                    SpeechRegion(0, 32000),
                    transcript,
                    16000,
                    [
                        SpeakerInterval(0, 24000, "A"),
                        SpeakerInterval(24000, 32000, "B"),
                    ],
                )
                self.assertEqual([g.speaker_id for g in groups], ["A"])
                self.assertEqual([g.text for g in groups], [transcript.text])

    def test_invalid_runtime_values_fail_before_loading_any_models(self):
        invalid = (
            ("--vad-threshold", "1.1"),
            ("--vad-threshold", "nan"),
            ("--min-speech-duration", "-0.1"),
            ("--min-speech-duration", "inf"),
            ("--min-silence-duration", "-0.1"),
            ("--min-silence-duration", "nan"),
            ("--pre-buffer-duration", "-0.1"),
            ("--pre-buffer-duration", "inf"),
            ("--max-segment-duration", "0"),
            ("--max-segment-duration", "nan"),
            ("--min-speaker-run-tokens", "0"),
            ("--min-speaker-run-duration", "-0.1"),
            ("--min-speaker-run-duration", "nan"),
        )
        for batch in (False, True):
            for option, value in invalid:
                with (
                    self.subTest(batch=batch, option=option, value=value),
                    patch(
                        "aimara.run_pipeline.ROBVoiceActivityDetector.from_torch_hub"
                    ) as load,
                ):
                    with self.assertRaisesRegex(
                        ValueError, option[2:].replace("-", "_")
                    ):
                        build_pipeline(arguments(batch, (option, value)))
                    load.assert_not_called()
        with patch(
            "aimara.run_pipeline.ROBVoiceActivityDetector.from_torch_hub"
        ) as load:
            with self.assertRaisesRegex(ValueError, "min_speech_duration"):
                build_pipeline(
                    arguments(
                        extra=(
                            "--min-speech-duration",
                            "2",
                            "--max-segment-duration",
                            "1",
                        )
                    )
                )
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
