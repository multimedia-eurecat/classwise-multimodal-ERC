import unittest

from aimara import (
    SpeakerInterval,
    SpeakerSegmentationConfig,
    SpeakerSegmenter,
    SpeechRegion,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


def timed_transcript(words):
    tokens = tuple(TranscriptWord(*word) for word in words)
    text = "".join(word.text for word in tokens).strip()
    return Transcript(
        text,
        "en",
        (
            TranscriptSegment(
                tokens[0].start_time, max(w.end_time for w in tokens), text, tokens
            ),
        ),
    )


class SpeakerSegmentationTests(unittest.TestCase):
    def test_a_b_a_sentences_partition_the_source_region_and_preserve_every_word(self):
        transcript = timed_transcript(
            [
                (0.2, 0.6, " Hello"),
                (0.8, 1.8, " there."),
                (2.2, 2.8, " Goodbye"),
                (3.0, 3.8, " now!"),
                (4.2, 4.8, " Welcome"),
                (5.0, 5.8, " back."),
            ]
        )
        timeline = [
            SpeakerInterval(1000, 1200, "A"),
            SpeakerInterval(1200, 1400, "B"),
            SpeakerInterval(1400, 1600, "A"),
        ]
        groups = SpeakerSegmenter().split(
            SpeechRegion(1000, 1600), transcript, 100, timeline
        )
        self.assertEqual([g.speaker_id for g in groups], ["A", "B", "A"])
        self.assertEqual(
            [g.text for g in groups], ["Hello there.", "Goodbye now!", "Welcome back."]
        )
        self.assertEqual(
            [(g.region.start_sample, g.region.end_sample) for g in groups],
            [(1000, 1200), (1200, 1400), (1400, 1600)],
        )
        self.assertEqual(" ".join(g.text for g in groups), transcript.text)

    def test_credible_trailing_switch_labels_an_intact_sentence(self):
        transcript = timed_transcript(
            [
                (0, 0.5, " The"),
                (0.5, 1.5, " speaker"),
                (1.5, 1.8, " changed"),
                (1.8, 2, " now."),
            ]
        )
        timeline = [SpeakerInterval(0, 150, "A"), SpeakerInterval(150, 200, "B")]
        groups = SpeakerSegmenter().split(
            SpeechRegion(0, 200), transcript, 100, timeline
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].speaker_id, "B")
        self.assertEqual(groups[0].text, transcript.text)

    def test_raw_word_runs_can_split_the_same_sentence(self):
        transcript = timed_transcript(
            [
                (0, 0.5, " The"),
                (0.5, 1.5, " speaker"),
                (1.5, 1.8, " changed"),
                (1.8, 2, " now."),
            ]
        )
        segmenter = SpeakerSegmenter(
            SpeakerSegmentationConfig(snap_splits_to_sentences=False)
        )
        groups = segmenter.split(
            SpeechRegion(0, 200),
            transcript,
            100,
            [SpeakerInterval(0, 150, "A"), SpeakerInterval(150, 200, "B")],
        )
        self.assertEqual([g.text for g in groups], ["The speaker", "changed now."])

    def test_short_raw_speaker_flicker_is_absorbed_without_losing_its_word(self):
        transcript = timed_transcript(
            [
                (0, 0.6, " Hello"),
                (0.6, 1.2, " there"),
                (1.2, 1.3, " uh"),
                (1.3, 2, " again."),
            ]
        )
        groups = SpeakerSegmenter(
            SpeakerSegmentationConfig(snap_splits_to_sentences=False)
        ).split(
            SpeechRegion(0, 200),
            transcript,
            100,
            [
                SpeakerInterval(0, 120, "A"),
                SpeakerInterval(120, 130, "B"),
                SpeakerInterval(130, 200, "A"),
            ],
        )
        self.assertEqual(
            [(g.speaker_id, g.text) for g in groups], [("A", transcript.text)]
        )

    def test_backend_segment_times_are_used_when_word_times_are_missing(self):
        transcript = Transcript(
            "Hello there. Goodbye now.",
            "en",
            (
                TranscriptSegment(0, 1, "Hello there."),
                TranscriptSegment(1, 2, "Goodbye now."),
            ),
        )
        groups = SpeakerSegmenter().split(
            SpeechRegion(0, 200),
            transcript,
            100,
            [SpeakerInterval(0, 100, "A"), SpeakerInterval(100, 200, "B")],
        )
        self.assertEqual([g.text for g in groups], ["Hello there.", "Goodbye now."])

    def test_partial_timing_does_not_discard_untimed_text(self):
        transcript = Transcript(
            "Hello there. Goodbye now.",
            "en",
            (
                TranscriptSegment(
                    0, 1, "Hello there.", (TranscriptWord(0, 0.5, " Hello"),)
                ),
                TranscriptSegment(1, 2, "Goodbye now."),
            ),
        )
        groups = SpeakerSegmenter().split(
            SpeechRegion(0, 200),
            transcript,
            100,
            [SpeakerInterval(0, 100, "A"), SpeakerInterval(100, 200, "B")],
        )
        self.assertEqual([g.text for g in groups], ["Hello there.", "Goodbye now."])

    def test_text_only_transcript_keeps_one_whole_region(self):
        region = SpeechRegion(0, 200)
        groups = SpeakerSegmenter().split(
            region,
            Transcript("Hello", "en"),
            100,
            [SpeakerInterval(0, 150, "A"), SpeakerInterval(150, 200, "B")],
        )
        self.assertEqual(
            [(g.region, g.text, g.speaker_id) for g in groups], [(region, "Hello", "A")]
        )

    def test_a_timeline_gap_remains_unknown(self):
        transcript = timed_transcript([(0, 1, " Hello."), (1, 2, " Goodbye.")])
        groups = SpeakerSegmenter().split(
            SpeechRegion(0, 200), transcript, 100, [SpeakerInterval(0, 100, "A")]
        )
        self.assertEqual([g.speaker_id for g in groups], ["A", "UNKNOWN"])

    def test_overlapping_word_times_are_partitioned_without_duplicate_audio(self):
        transcript = timed_transcript([(0.2, 1.1, " Hello."), (0.9, 1.8, " Goodbye.")])
        groups = SpeakerSegmenter().split(
            SpeechRegion(0, 200),
            transcript,
            100,
            [SpeakerInterval(0, 100, "A"), SpeakerInterval(100, 200, "B")],
        )
        self.assertEqual(
            [(g.region.start_sample, g.region.end_sample) for g in groups],
            [(0, 100), (100, 200)],
        )

    def test_zero_duration_words_and_backend_overshoot_still_have_nonempty_clips(self):
        transcript = timed_transcript([(0, 0, " Hello."), (2, 2.01, " Goodbye.")])
        groups = SpeakerSegmenter().split(
            SpeechRegion(0, 200),
            transcript,
            100,
            [SpeakerInterval(0, 100, "A"), SpeakerInterval(100, 200, "B")],
        )
        self.assertEqual([g.speaker_id for g in groups], ["A", "B"])
        self.assertTrue(all(g.region.duration_samples > 0 for g in groups))
        self.assertEqual(sum(g.region.duration_samples for g in groups), 200)

    def test_invalid_timing_order_and_configuration_fail_explicitly(self):
        transcript = timed_transcript([(1, 1.5, " Bye."), (0, 1, " Hello.")])
        with self.assertRaisesRegex(ValueError, "chronological"):
            SpeakerSegmenter().split(SpeechRegion(0, 200), transcript, 100, [])
        for kwargs in (
            {"min_speaker_run_tokens": 0},
            {"min_speaker_run_duration": -1},
            {"min_speaker_run_duration": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpeakerSegmentationConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
