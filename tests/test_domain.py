import unittest
from pathlib import Path

from aimara import SpeechSegment


class SpeechSegmentTests(unittest.TestCase):
    def make_segment(self, **overrides):
        values = {
            "conversation_id": "Anger",
            "segment_id": 0,
            "source_video": "/data/Anger.mp4",
            "audio_path": "/data/audio/dia0_utt0.wav",
            "start_time": 0.15,
            "end_time": 3.45,
            "speaker_id": "EUT_SPEAKER_1",
            "transcript": "This is completely unacceptable.",
            "language": "en",
        }
        values.update(overrides)
        return SpeechSegment(**values)

    def test_normalizes_paths_and_reports_duration(self):
        segment = self.make_segment()

        self.assertIsInstance(segment.source_video, Path)
        self.assertIsInstance(segment.audio_path, Path)
        self.assertAlmostEqual(segment.duration, 3.3)

    def test_allows_empty_transcript_when_asr_has_no_result(self):
        segment = self.make_segment(transcript="")

        self.assertEqual(segment.transcript, "")

    def test_rejects_invalid_identifiers_and_times(self):
        invalid_values = [
            ({"conversation_id": " "}, "conversation_id"),
            ({"segment_id": -1}, "segment_id"),
            ({"start_time": -0.1}, "start_time"),
            ({"end_time": 0.15}, "end_time"),
            ({"speaker_id": ""}, "speaker_id"),
        ]

        for overrides, message in invalid_values:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self.make_segment(**overrides)


if __name__ == "__main__":
    unittest.main()
