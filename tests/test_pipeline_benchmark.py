"""Check benchmark aggregation and memory sampling without pretrained models."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aimara.benchmark_pipeline import MemorySampler, summarize


class PipelineBenchmarkTests(unittest.TestCase):
    def test_real_time_factor_weights_audio_duration(self):
        summary = summarize(
            [dict(seconds=2, audio_duration=1), dict(seconds=6, audio_duration=3)]
        )
        self.assertEqual(summary["median_seconds"], 4)
        self.assertEqual(summary["real_time_factor"], 2)
        with self.assertRaises(ValueError):
            summarize([])
        with self.assertRaises(ValueError):
            summarize([dict(seconds=0, audio_duration=1)])
        with self.assertRaises(ValueError):
            summarize([dict(seconds=float("nan"), audio_duration=1)])

    def test_driver_memory_includes_only_current_process_and_sums_devices(self):
        sampler = MemorySampler(True)

        def finish(*args, **kwargs):
            sampler.stop.set()
            return SimpleNamespace(stdout="123, 100\n123, 200\n456, 999\n")

        with (
            patch("aimara.benchmark_pipeline.os.getpid", return_value=123),
            patch("aimara.benchmark_pipeline.subprocess.run", side_effect=finish),
            patch(
                "aimara.benchmark_pipeline.Path.read_text",
                return_value="VmRSS: 2048 kB\n",
            ),
        ):
            sampler._sample()
        self.assertEqual(sampler.gpu_peak_bytes, 300 * 1024 * 1024)
        self.assertEqual(sampler.rss_peak_bytes, 2048 * 1024)
        self.assertEqual(sampler.errors, [])
