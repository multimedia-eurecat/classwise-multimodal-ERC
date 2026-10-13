import unittest
from dataclasses import FrozenInstanceError

from aimara import ROBProcessorConfig, Transcript, TranscriptSegment, TranscriptWord


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

    def test_transcript_allows_empty_asr_result_and_is_immutable(self):
        transcript = Transcript(text="", language=None)

        self.assertEqual(transcript.text, "")
        with self.assertRaises(FrozenInstanceError):
            transcript.text = "changed"

    def test_legacy_transcript_construction_still_has_empty_timing(self):
        transcript = Transcript("hello", "en")
        self.assertEqual(transcript.segments, ())
        self.assertEqual(transcript.words, ())

    def test_timed_transcript_snapshots_mutable_backend_collections(self):
        word = TranscriptWord(0.1, 0.5, " hello")
        words = [word]
        segment = TranscriptSegment(0, 0.5, "hello", words)
        segments = [segment]
        transcript = Transcript("hello", "en", segments)
        words.clear()
        segments.clear()
        self.assertEqual(transcript.segments, (segment,))
        self.assertEqual(transcript.words, (word,))
        with self.assertRaises(FrozenInstanceError):
            word.start_time = 1
        with self.assertRaises(FrozenInstanceError):
            segment.words = ()

    def test_timed_values_reject_invalid_offsets_and_allow_zero_duration(self):
        for value_type in (TranscriptWord, TranscriptSegment):
            for start, end in [(-0.1, 0), (1, 0), (0, float("inf")), (float("nan"), 1)]:
                with (
                    self.subTest(value_type=value_type, start=start, end=end),
                    self.assertRaises(ValueError),
                ):
                    value_type(start, end, "hi")
            self.assertEqual(value_type(0.5, 0.5, "hi").end_time, 0.5)

    def test_timed_contract_rejects_invalid_nested_values_and_blank_words(self):
        with self.assertRaisesRegex(TypeError, "TranscriptWord"):
            TranscriptSegment(0, 1, "hello", words=["hello"])
        with self.assertRaisesRegex(TypeError, "TranscriptSegment"):
            Transcript("hello", segments=["hello"])
        with self.assertRaisesRegex(ValueError, "word text"):
            TranscriptWord(0, 1, " ")


if __name__ == "__main__":
    unittest.main()
