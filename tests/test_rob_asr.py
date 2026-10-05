import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from aimara import Transcript
from aimara.rob import ROBTranscriber


class FakeModel:
    def __init__(self, texts=(" hello ", "world"), detected_language="en"):
        self.texts = texts
        self.detected_language = detected_language
        self.transcribe_calls = []
        self.detect_result = (
            detected_language,
            0.7,
            [("en", 0.2), ("es", 0.7), ("ca", 0.1)],
        )

    def transcribe(self, audio, **kwargs):
        self.transcribe_calls.append((audio, kwargs))
        segments = [SimpleNamespace(text=text) for text in self.texts]
        return iter(segments), SimpleNamespace(language=self.detected_language)

    def detect_language(self, audio):
        return self.detect_result


class ROBTranscriberTests(unittest.TestCase):
    def audio(self, seconds=1.0):
        return np.zeros(round(16000 * seconds), dtype=np.float32)

    def test_transcribes_and_joins_whisper_segments(self):
        model = FakeModel()
        transcriber = ROBTranscriber(
            model=model,
            model_size="tiny.en",
            language="auto",
        )

        result = transcriber.transcribe(self.audio(), 16000)

        self.assertEqual(result, Transcript(text="hello world", language="en"))
        kwargs = model.transcribe_calls[0][1]
        self.assertEqual(kwargs["language"], "en")
        self.assertFalse(kwargs["vad_filter"])
        self.assertFalse(kwargs["word_timestamps"])

    def test_restricts_automatic_language_detection(self):
        model = FakeModel(detected_language="es")
        transcriber = ROBTranscriber(
            model=model,
            model_size="tiny",
            language="es,ca",
        )

        transcriber.transcribe(self.audio(), 16000)

        self.assertEqual(model.transcribe_calls[0][1]["language"], "es")

    def test_short_audio_returns_empty_transcript_without_inference(self):
        model = FakeModel()
        transcriber = ROBTranscriber(model=model, model_size="tiny")

        result = transcriber.transcribe(self.audio(seconds=0.005), 16000)

        self.assertEqual(result, Transcript(text="", language=None))
        self.assertEqual(model.transcribe_calls, [])

    def test_validates_audio_and_model_size(self):
        transcriber = ROBTranscriber(model=FakeModel(), model_size="tiny")

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            transcriber.transcribe(np.zeros((2, 2), dtype=np.float32), 16000)
        with self.assertRaisesRegex(TypeError, "float32"):
            transcriber.transcribe(np.zeros(16000, dtype=np.float64), 16000)
        with self.assertRaisesRegex(ValueError, "16000"):
            transcriber.transcribe(self.audio(), 8000)
        with self.assertRaisesRegex(ValueError, "model_size"):
            ROBTranscriber(model=FakeModel(), model_size="unknown")

    def test_loads_model_with_cpu_compute_fallback(self):
        model = FakeModel()
        whisper_model = Mock(return_value=model)
        fake_faster_whisper = SimpleNamespace(WhisperModel=whisper_model)
        fake_ctranslate2 = SimpleNamespace(
            get_cuda_device_count=Mock(return_value=0)
        )

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                sys.modules,
                {
                    "ctranslate2": fake_ctranslate2,
                    "faster_whisper": fake_faster_whisper,
                },
            ):
                transcriber = ROBTranscriber.from_faster_whisper(
                    model_size="tiny",
                    weights_dir=directory,
                    compute_type="float16",
                )

        whisper_model.assert_called_once_with(
            "tiny",
            device="cpu",
            compute_type="int8",
            download_root=directory,
        )
        self.assertIs(transcriber._model, model)

    def test_uses_complete_local_snapshot(self):
        model = FakeModel()
        whisper_model = Mock(return_value=model)
        fake_faster_whisper = SimpleNamespace(WhisperModel=whisper_model)
        fake_ctranslate2 = SimpleNamespace(
            get_cuda_device_count=Mock(return_value=0)
        )

        with tempfile.TemporaryDirectory() as directory:
            snapshot = (
                Path(directory)
                / "models--Systran--faster-whisper-tiny"
                / "snapshots"
                / "revision"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "model.bin").touch()
            with patch.dict(
                sys.modules,
                {
                    "ctranslate2": fake_ctranslate2,
                    "faster_whisper": fake_faster_whisper,
                },
            ):
                ROBTranscriber.from_faster_whisper(
                    model_size="tiny",
                    weights_dir=directory,
                    compute_type="int8",
                )

        whisper_model.assert_called_once_with(
            str(snapshot),
            device="cpu",
            compute_type="int8",
        )


if __name__ == "__main__":
    unittest.main()
