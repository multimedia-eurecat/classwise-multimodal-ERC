import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aimara.audio_io import decode_audio, write_wav


class DecodeAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.media_path = Path(self.temp_dir.name) / "video.mp4"
        self.media_path.touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("aimara.audio_io.subprocess.run")
    def test_decodes_mono_float32_audio_with_ffmpeg(self, run):
        expected = np.array([-0.5, 0.25], dtype=np.float32)
        run.return_value = SimpleNamespace(stdout=expected.astype("<f4").tobytes())

        audio = decode_audio(self.media_path, sample_rate=16000)

        self.assertEqual(audio.tolist(), expected.tolist())
        command = run.call_args.args[0]
        self.assertIn("16000", command)
        self.assertIn("pcm_f32le", command)
        self.assertEqual(command[-1], "pipe:1")
        run.assert_called_once_with(command, capture_output=True, check=True)

    @patch("aimara.audio_io.subprocess.run")
    def test_rejects_media_without_decoded_audio(self, run):
        run.return_value = SimpleNamespace(stdout=b"")

        with self.assertRaisesRegex(ValueError, "No audio samples"):
            decode_audio(self.media_path)

    @patch("aimara.audio_io.subprocess.run")
    def test_reports_ffmpeg_failure(self, run):
        run.side_effect = subprocess.CalledProcessError(
            1, ["ffmpeg"], stderr=b"invalid media"
        )

        with self.assertRaisesRegex(RuntimeError, "invalid media"):
            decode_audio(self.media_path)


class WriteWavTests(unittest.TestCase):
    def test_writes_mono_16_bit_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "segment.wav"

            result = write_wav(
                path,
                np.array([-1.0, 0.0, 1.0], dtype=np.float32),
                sample_rate=16000,
            )

            with wave.open(str(path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), 16000)
                samples = np.frombuffer(wav_file.readframes(3), dtype="<i2")
            self.assertEqual(result, path)
            self.assertEqual(samples.tolist(), [-32768, 0, 32767])
