import unittest
from dataclasses import FrozenInstanceError

from aimara import ROBProcessorConfig, SpeakerTurn, Transcript


class SpeechInterfaceValueTests(unittest.TestCase):
    def test_default_processor_config_matches_rob_vad_input(self):
        config = ROBProcessorConfig()

        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.vad_threshold, 0.5)

    def test_processor_config_rejects_invalid_values(self):
        invalid_values = [
            ({"sample_rate": 0}, "sample_rate"),
            ({"chunk_size": 0}, "chunk_size"),
            ({"vad_threshold": -0.1}, "vad_threshold"),
            ({"vad_threshold": 1.1}, "vad_threshold"),
            ({"min_speech_duration": -0.1}, "min_speech_duration"),
            ({"min_silence_duration": -0.1}, "min_silence_duration"),
            ({"pre_buffer_duration": -0.1}, "pre_buffer_duration"),
            ({"max_segment_duration": 0}, "max_segment_duration"),
            (
                {"min_speech_duration": 2.0, "max_segment_duration": 1.0},
                "min_speech_duration",
            ),
        ]

        for overrides, message in invalid_values:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    ROBProcessorConfig(**overrides)

    def test_speaker_turn_validates_interval_and_speaker(self):
        turn = SpeakerTurn(1.25, 2.75, "EUT_SPEAKER_1")

        self.assertEqual(turn.duration, 1.5)
        for values, message in [
            ((-0.1, 1.0, "speaker"), "start_time"),
            ((1.0, 1.0, "speaker"), "end_time"),
            ((0.0, 1.0, " "), "speaker_id"),
        ]:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    SpeakerTurn(*values)

    def test_transcript_allows_empty_asr_result_and_is_immutable(self):
        transcript = Transcript(text="", language=None)

        self.assertEqual(transcript.text, "")
        with self.assertRaises(FrozenInstanceError):
            transcript.text = "changed"


if __name__ == "__main__":
    unittest.main()
