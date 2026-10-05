import unittest

import numpy as np

from aimara import OfflineVADProcessor, ROBProcessorConfig, SpeechRegion


class FakeDetector:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)
        self.calls = []

    def predict(self, audio, sample_rate):
        self.calls.append((audio.copy(), sample_rate))
        return next(self.probabilities)


class OfflineVADProcessorTests(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "sample_rate": 4,
            "chunk_size": 4,
            "vad_threshold": 0.5,
            "min_speech_duration": 0.0,
            "min_silence_duration": 0.0,
            "pre_buffer_duration": 0.0,
            "max_segment_duration": 30.0,
        }
        values.update(overrides)
        return ROBProcessorConfig(**values)

    def test_runs_detector_and_pads_final_chunk(self):
        detector = FakeDetector([0.1, 0.9, 0.1])
        processor = OfflineVADProcessor(detector, self.config())
        audio = np.arange(10, dtype=np.float32)

        regions = processor.process(audio)

        self.assertEqual(regions, [SpeechRegion(4, 8)])
        self.assertEqual(len(detector.calls), 3)
        np.testing.assert_array_equal(
            detector.calls[-1][0], np.array([8, 9, 0, 0], dtype=np.float32)
        )
        self.assertTrue(all(call[0].size == 4 for call in detector.calls))
        self.assertTrue(all(call[1] == 4 for call in detector.calls))

    def test_empty_audio_produces_no_regions_or_detector_calls(self):
        detector = FakeDetector([])
        processor = OfflineVADProcessor(detector, self.config())

        regions = processor.process(np.array([], dtype=np.float32))

        self.assertEqual(regions, [])
        self.assertEqual(detector.calls, [])

    def test_rejects_invalid_audio_shape_and_dtype(self):
        processor = OfflineVADProcessor(FakeDetector([]), self.config())

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            processor.process(np.zeros((2, 2), dtype=np.float32))
        with self.assertRaisesRegex(TypeError, "float32"):
            processor.process(np.zeros(4, dtype=np.float64))

    def test_rejects_detector_chunk_size_mismatch(self):
        detector = FakeDetector([])
        detector.EXPECTED_CHUNK_SIZE = 512

        with self.assertRaisesRegex(ValueError, "expects 512"):
            OfflineVADProcessor(detector, self.config(chunk_size=256))


if __name__ == "__main__":
    unittest.main()
