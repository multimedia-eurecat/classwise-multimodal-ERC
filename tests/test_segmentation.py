import math
import unittest

from aimara import ROBProcessorConfig, SpeechRegion, VADSegmenter


class VADSegmenterTests(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "sample_rate": 100,
            "chunk_size": 10,
            "vad_threshold": 0.5,
            "min_speech_duration": 0.1,
            "min_silence_duration": 0.2,
            "pre_buffer_duration": 0.1,
            "max_segment_duration": 30.0,
        }
        values.update(overrides)
        return ROBProcessorConfig(**values)

    def test_segments_speech_and_includes_pre_buffer(self):
        segmenter = VADSegmenter(self.config())

        regions = segmenter.segment(
            [0.1, 0.8, 0.9, 0.1, 0.1, 0.1], total_samples=60
        )

        self.assertEqual(regions, [SpeechRegion(0, 30)])

    def test_short_silence_does_not_split_speech(self):
        segmenter = VADSegmenter(self.config(pre_buffer_duration=0.0))

        regions = segmenter.segment(
            [0.8, 0.1, 0.8, 0.1, 0.1], total_samples=50
        )

        self.assertEqual(regions, [SpeechRegion(0, 30)])

    def test_discards_speech_below_minimum_duration(self):
        segmenter = VADSegmenter(self.config(min_speech_duration=0.2))

        regions = segmenter.segment([0.8, 0.1, 0.1], total_samples=30)

        self.assertEqual(regions, [])

    def test_closes_final_partial_chunk_at_end_of_file(self):
        segmenter = VADSegmenter(self.config(pre_buffer_duration=0.0))

        regions = segmenter.segment([0.1, 0.8, 0.8], total_samples=25)

        self.assertEqual(regions, [SpeechRegion(10, 25)])

    def test_splits_continuous_speech_at_maximum_duration(self):
        segmenter = VADSegmenter(
            self.config(
                sample_rate=10,
                chunk_size=2,
                min_speech_duration=0.0,
                min_silence_duration=0.2,
                pre_buffer_duration=0.0,
                max_segment_duration=0.4,
            )
        )

        regions = segmenter.segment([1.0, 1.0, 1.0], total_samples=6)

        self.assertEqual(regions, [SpeechRegion(0, 4), SpeechRegion(4, 6)])

    def test_exact_maximum_duration_has_no_empty_trailing_region(self):
        segmenter = VADSegmenter(
            self.config(
                sample_rate=10,
                chunk_size=2,
                min_speech_duration=0.0,
                pre_buffer_duration=0.0,
                max_segment_duration=0.4,
            )
        )

        regions = segmenter.segment([1.0, 1.0], total_samples=4)

        self.assertEqual(regions, [SpeechRegion(0, 4)])

    def test_validates_probability_count_and_values(self):
        segmenter = VADSegmenter(self.config())

        with self.assertRaisesRegex(ValueError, "expected 2"):
            segmenter.segment([0.5], total_samples=20)
        for probability in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    segmenter.segment([probability], total_samples=10)

    def test_speech_region_converts_samples_to_seconds(self):
        region = SpeechRegion(8000, 24000)

        self.assertEqual(region.duration_samples, 16000)
        self.assertEqual(region.start_time(16000), 0.5)
        self.assertEqual(region.end_time(16000), 1.5)


if __name__ == "__main__":
    unittest.main()
