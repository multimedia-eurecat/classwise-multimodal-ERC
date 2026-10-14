import unittest

from aimara import SpeakerInterval, SpeechRegion
from aimara.diarization import speaker_for_region


class SpeakerTimelineTests(unittest.TestCase):
    def test_coverage_is_accumulated_across_runs_of_the_same_speaker(self):
        timeline = [
            SpeakerInterval(0, 3, "speaker1"),
            SpeakerInterval(3, 7, "speaker2"),
            SpeakerInterval(7, 10, "speaker1"),
        ]
        self.assertEqual(speaker_for_region(SpeechRegion(0, 10), timeline), "speaker1")

    def test_uncovered_audio_and_ties_are_unknown(self):
        for timeline in [
            [],
            [SpeakerInterval(0, 2, "speaker1")],
            [SpeakerInterval(0, 5, "speaker1"), SpeakerInterval(5, 10, "speaker2")],
        ]:
            with self.subTest(timeline=timeline):
                self.assertEqual(
                    speaker_for_region(SpeechRegion(0, 10), timeline), "UNKNOWN"
                )

    def test_only_the_requested_region_contributes_and_unknown_is_normalized(self):
        timeline = [
            SpeakerInterval(0, 10, "speaker1"),
            SpeakerInterval(10, 20, "unknown"),
        ]
        self.assertEqual(speaker_for_region(SpeechRegion(10, 20), timeline), "UNKNOWN")
        self.assertEqual(timeline[1].speaker_id, "UNKNOWN")

    def test_unknown_spellings_share_coverage_and_custom_output_alias(self):
        timeline = [
            SpeakerInterval(0, 3, " unknown "),
            SpeakerInterval(3, 6, "UnKnOwN"),
            SpeakerInterval(6, 10, " speaker1 "),
        ]
        self.assertEqual(speaker_for_region(SpeechRegion(0, 10), timeline), "UNKNOWN")
        self.assertEqual(
            speaker_for_region(
                SpeechRegion(0, 10), timeline, unknown_speaker_id="unresolved"
            ),
            "unresolved",
        )
        self.assertEqual(timeline[-1].speaker_id, "speaker1")

    def test_interval_validates_boundaries_speaker_and_score(self):
        for args in [
            (0, 0, "speaker1"),
            (0, 1, " "),
            (0, 1, "speaker1", -0.5),
            (0, 1, "speaker1", float("nan")),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                SpeakerInterval(*args)


if __name__ == "__main__":
    unittest.main()
