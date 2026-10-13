import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from aimara import Transcript, TranscriptSegment, TranscriptWord
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
        segments = [
            SimpleNamespace(text=text, start=i * 0.5, end=(i + 1) * 0.5, words=None)
            for i, text in enumerate(self.texts)
        ]
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

        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.language, "en")
        self.assertEqual(
            result.segments,
            (
                TranscriptSegment(0.0, 0.5, "hello"),
                TranscriptSegment(0.5, 1.0, "world"),
            ),
        )
        kwargs = model.transcribe_calls[0][1]
        self.assertEqual(kwargs["language"], "en")
        self.assertFalse(kwargs["vad_filter"])
        self.assertTrue(kwargs["word_timestamps"])

    def test_preserves_word_punctuation_spacing_and_input_relative_offsets(self):
        model = FakeModel()
        model.transcribe = Mock(
            return_value=(
                iter(
                    [
                        SimpleNamespace(
                            start=0.2,
                            end=1.3,
                            text=" Hello, world!",
                            words=[
                                SimpleNamespace(start=0.2, end=0.7, word=" Hello,"),
                                SimpleNamespace(start=0.7, end=1.3, word=" world!"),
                            ],
                        ),
                        SimpleNamespace(
                            start=1.5,
                            end=2.0,
                            text=" Bye.",
                            words=[
                                SimpleNamespace(start=1.5, end=2.0, word=" Bye."),
                            ],
                        ),
                    ]
                ),
                SimpleNamespace(language="en"),
            )
        )
        result = ROBTranscriber(model=model, model_size="tiny.en").transcribe(
            self.audio(seconds=2), 16000
        )
        self.assertEqual(result.text, "Hello, world! Bye.")
        self.assertEqual(
            result.words,
            (
                TranscriptWord(0.2, 0.7, " Hello,"),
                TranscriptWord(0.7, 1.3, " world!"),
                TranscriptWord(1.5, 2.0, " Bye."),
            ),
        )
        self.assertEqual(result.segments[1].start_time, 1.5)

    def test_skips_blank_segments_and_words_without_dropping_other_timing(self):
        model = FakeModel()
        model.transcribe = Mock(
            return_value=(
                iter(
                    [
                        SimpleNamespace(start=0, end=0.2, text=" ", words=None),
                        SimpleNamespace(
                            start=0.3,
                            end=0.8,
                            text=" hi",
                            words=[
                                SimpleNamespace(start=0.3, end=0.3, word=" "),
                                SimpleNamespace(start=0.3, end=0.8, word=" hi"),
                            ],
                        ),
                    ]
                ),
                SimpleNamespace(language="en"),
            )
        )
        result = ROBTranscriber(model=model, model_size="tiny.en").transcribe(
            self.audio(), 16000
        )
        self.assertEqual(result.text, "hi")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.words, (TranscriptWord(0.3, 0.8, " hi"),))

    def test_empty_backend_output_returns_empty_text_and_timing(self):
        transcriber = ROBTranscriber(model=FakeModel(texts=()), model_size="tiny.en")
        self.assertEqual(
            transcriber.transcribe(self.audio(), 16000), Transcript("", "en")
        )

    def test_lazy_backend_failure_does_not_return_a_partial_transcript(self):
        def failing_segments():
            yield SimpleNamespace(start=0, end=0.5, text="hello", words=None)
            raise ValueError("decoder failed")

        model = FakeModel()
        model.transcribe = Mock(
            return_value=(failing_segments(), SimpleNamespace(language="en"))
        )
        with self.assertRaisesRegex(RuntimeError, "transcription failed") as caught:
            ROBTranscriber(model=model, model_size="tiny.en").transcribe(
                self.audio(), 16000
            )
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_invalid_backend_timestamp_fails_explicitly(self):
        model = FakeModel()
        model.transcribe = Mock(
            return_value=(
                iter(
                    [
                        SimpleNamespace(
                            start=float("nan"), end=1, text="hi", words=None
                        ),
                    ]
                ),
                SimpleNamespace(language="en"),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "transcription failed") as caught:
            ROBTranscriber(model=model, model_size="tiny.en").transcribe(
                self.audio(), 16000
            )
        self.assertIn("finite", str(caught.exception.__cause__))

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
        fake_ctranslate2 = SimpleNamespace(get_cuda_device_count=Mock(return_value=0))

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
        fake_ctranslate2 = SimpleNamespace(get_cuda_device_count=Mock(return_value=0))

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
