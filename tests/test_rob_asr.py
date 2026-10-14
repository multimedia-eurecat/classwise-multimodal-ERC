import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from aimara import ASRMetadata, Transcript, TranscriptSegment, TranscriptWord
from aimara.rob import ROBTranscriber, WhisperInferenceConfig


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
        self.assertEqual(kwargs["beam_size"], 5)
        self.assertNotIn("batch_size", kwargs)
        self.assertNotIn("clip_timestamps", kwargs)

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
            transcriber.transcribe(self.audio(), 16000),
            Transcript(
                "",
                "en",
                metadata=ASRMetadata(
                    backend="whisper", model="tiny.en", language_source="model"
                ),
            ),
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

        self.assertEqual(
            result,
            Transcript(text="", metadata=ASRMetadata(backend="whisper", model="tiny")),
        )
        self.assertEqual(model.transcribe_calls, [])

    def test_language_provenance_distinguishes_model_configuration_detection_and_fallback(
        self,
    ):
        for model_size, language, source in (
            ("tiny.en", "auto", "model"),
            ("tiny", "en", "configured"),
            ("tiny", "es,ca", "detected"),
            ("tiny", "auto", "detected"),
        ):
            with self.subTest(model_size=model_size, language=language):
                result = ROBTranscriber(
                    model=FakeModel(), model_size=model_size, language=language
                ).transcribe(self.audio(), 16000)
                self.assertEqual(result.metadata.language_source, source)
                self.assertIsNone(result.metadata.transcript_confidence)
                self.assertEqual(result.metadata.backend, "whisper")
                self.assertEqual(result.metadata.model, model_size)
        model = FakeModel()
        model.detect_language = Mock(side_effect=RuntimeError("no language result"))
        result = ROBTranscriber(model=model, model_size="tiny").transcribe(
            self.audio(), 16000
        )
        self.assertEqual(result.metadata.language_source, "fallback")

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
        self.assertEqual(transcriber.device, "cpu")
        self.assertEqual(transcriber.effective_compute_type, "int8")

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

    def test_batched_decode_uses_base_model_for_language_and_preserves_late_word_times(
        self,
    ):
        base = FakeModel(detected_language="es")
        batched = FakeModel(detected_language="es")
        batched.transcribe = Mock(
            return_value=(
                iter(
                    [
                        SimpleNamespace(
                            start=30.05,
                            end=30.5,
                            text=" Hola.",
                            words=[
                                SimpleNamespace(start=30.05, end=30.5, word=" Hola.")
                            ],
                        )
                    ]
                ),
                SimpleNamespace(language="es"),
            )
        )
        batched.last_speech_timestamp = 123
        transcriber = ROBTranscriber(
            model=base,
            model_size="tiny",
            language="es,ca",
            batched_model=batched,
            inference_config=WhisperInferenceConfig(
                use_batched_inference=True, batch_size=4, beam_size=3
            ),
        )
        result = transcriber.transcribe(self.audio(30.5), 16000)
        self.assertEqual(base.transcribe_calls, [])
        self.assertEqual(result.words, (TranscriptWord(30.05, 30.5, " Hola."),))
        kwargs = batched.transcribe.call_args.kwargs
        self.assertEqual(kwargs["language"], "es")
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertEqual(kwargs["beam_size"], 3)
        self.assertFalse(kwargs["vad_filter"])
        self.assertTrue(kwargs["word_timestamps"])
        self.assertEqual(
            kwargs["clip_timestamps"],
            [{"start": 0, "end": 30}, {"start": 30, "end": 30.5}],
        )
        self.assertEqual(batched.last_speech_timestamp, 0)

    def test_batched_clips_cover_every_sample_for_short_exact_and_long_regions(self):
        # 1001 / 16000 * 16000 rounds below 1001 without boundary correction.
        for samples in (160, 1001, 461203, 480000, 480001, 960000, 960013):
            with self.subTest(samples=samples):
                clips = ROBTranscriber._batched_clips(samples, 16000)
                boundaries = [
                    (int(c["start"] * 16000), int(c["end"] * 16000)) for c in clips
                ]
                self.assertEqual(boundaries[0][0], 0)
                self.assertEqual(boundaries[-1][1], samples)
                self.assertTrue(
                    all(0 < end - start <= 480000 for start, end in boundaries)
                )
                self.assertTrue(
                    all(
                        left[1] == right[0]
                        for left, right in zip(boundaries, boundaries[1:])
                    )
                )
                self.assertEqual(sum(end - start for start, end in boundaries), samples)

    def test_batched_alignment_state_resets_between_regions_even_after_lazy_failure(
        self,
    ):
        base, batched = FakeModel(), FakeModel()
        batched.last_speech_timestamp = 999
        starts = []

        def decode(audio, **kwargs):
            starts.append(batched.last_speech_timestamp)

            def segments():
                batched.last_speech_timestamp = 9
                yield SimpleNamespace(start=0, end=0.5, text="hello", words=None)
                if len(starts) == 1:
                    raise ValueError("lazy decoder error")

            return segments(), SimpleNamespace(language="en")

        batched.transcribe = decode
        transcriber = ROBTranscriber(
            model=base,
            model_size="tiny.en",
            batched_model=batched,
            inference_config=WhisperInferenceConfig(use_batched_inference=True),
        )
        with self.assertRaisesRegex(RuntimeError, "transcription failed"):
            transcriber.transcribe(self.audio(), 16000)
        self.assertEqual(batched.last_speech_timestamp, 0)
        self.assertEqual(transcriber.transcribe(self.audio(), 16000).text, "hello")
        self.assertEqual(starts, [0, 0])
        self.assertEqual(batched.last_speech_timestamp, 0)

    def test_batched_short_input_skips_inference(self):
        base, batched = FakeModel(), FakeModel()
        transcriber = ROBTranscriber(
            model=base,
            model_size="tiny.en",
            batched_model=batched,
            inference_config=WhisperInferenceConfig(use_batched_inference=True),
        )
        self.assertEqual(
            transcriber.transcribe(self.audio(0.005), 16000),
            Transcript("", metadata=ASRMetadata(backend="whisper", model="tiny.en")),
        )
        self.assertEqual(base.transcribe_calls, [])
        self.assertEqual(batched.transcribe_calls, [])

    def test_loader_selects_cpu_policy_and_optional_batched_wrapper(self):
        for fallback in ("int8", "float32"):
            for enabled in (False, True):
                with self.subTest(fallback=fallback, enabled=enabled):
                    base, batched = FakeModel(), FakeModel()
                    whisper_model = Mock(return_value=base)
                    batch_wrapper = Mock(return_value=batched)
                    backend = SimpleNamespace(
                        WhisperModel=whisper_model,
                        BatchedInferencePipeline=batch_wrapper,
                    )
                    with (
                        tempfile.TemporaryDirectory() as directory,
                        patch.dict(
                            sys.modules,
                            {
                                "ctranslate2": SimpleNamespace(
                                    get_cuda_device_count=lambda: 0
                                ),
                                "faster_whisper": backend,
                            },
                        ),
                    ):
                        transcriber = ROBTranscriber.from_faster_whisper(
                            model_size="tiny",
                            weights_dir=directory,
                            compute_type="int8_float16",
                            cpu_compute_type=fallback,
                            use_batched_inference=enabled,
                            batch_size=3,
                            beam_size=2,
                        )
                    self.assertEqual(
                        whisper_model.call_args.kwargs["compute_type"], fallback
                    )
                    self.assertEqual(transcriber.effective_compute_type, fallback)
                    self.assertEqual(transcriber.inference_config.batch_size, 3)
                    self.assertEqual(transcriber.inference_config.beam_size, 2)
                    if enabled:
                        batch_wrapper.assert_called_once_with(model=base)
                        self.assertIs(transcriber._batched_model, batched)
                    else:
                        batch_wrapper.assert_not_called()

    def test_loader_records_float32_retry_and_does_not_retry_float32_failure(self):
        for compute_type, succeeds in (("int8", True), ("float32", False)):
            with self.subTest(compute_type=compute_type):
                whisper_model = Mock(
                    side_effect=[ValueError("unsupported precision"), FakeModel()]
                )
                with (
                    tempfile.TemporaryDirectory() as directory,
                    patch.dict(
                        sys.modules,
                        {
                            "ctranslate2": SimpleNamespace(
                                get_cuda_device_count=lambda: 0
                            ),
                            "faster_whisper": SimpleNamespace(
                                WhisperModel=whisper_model
                            ),
                        },
                    ),
                ):
                    if succeeds:
                        transcriber = ROBTranscriber.from_faster_whisper(
                            model_size="tiny",
                            weights_dir=directory,
                            compute_type=compute_type,
                        )
                        self.assertEqual(transcriber.effective_compute_type, "float32")
                        self.assertEqual(
                            [
                                c.kwargs["compute_type"]
                                for c in whisper_model.call_args_list
                            ],
                            ["int8", "float32"],
                        )
                    else:
                        with self.assertRaisesRegex(
                            ValueError, "unsupported precision"
                        ):
                            ROBTranscriber.from_faster_whisper(
                                model_size="tiny",
                                weights_dir=directory,
                                compute_type=compute_type,
                            )
                        self.assertEqual(whisper_model.call_count, 1)

    def test_cuda_keeps_requested_precision(self):
        whisper_model = Mock(return_value=FakeModel())
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                sys.modules,
                {
                    "ctranslate2": SimpleNamespace(get_cuda_device_count=lambda: 1),
                    "faster_whisper": SimpleNamespace(WhisperModel=whisper_model),
                },
            ),
        ):
            transcriber = ROBTranscriber.from_faster_whisper(
                model_size="tiny",
                weights_dir=directory,
                compute_type="float16",
                cpu_compute_type="float32",
            )
        self.assertEqual(whisper_model.call_args.kwargs["device"], "cuda")
        self.assertEqual(whisper_model.call_args.kwargs["compute_type"], "float16")
        self.assertEqual(transcriber.effective_compute_type, "float16")

    def test_invalid_inference_settings_fail_before_dependencies_or_directory_creation(
        self,
    ):
        for kwargs in (
            {"compute_type": "fp17"},
            {"cpu_compute_type": "float16"},
            {"use_batched_inference": "false"},
            {"batch_size": 0},
            {"batch_size": 1.5},
            {"beam_size": -1},
            {"beam_size": True},
        ):
            with (
                self.subTest(kwargs=kwargs),
                tempfile.TemporaryDirectory() as directory,
            ):
                weights = Path(directory) / "new"
                with self.assertRaises(ValueError):
                    ROBTranscriber.from_faster_whisper(
                        model_size="tiny", weights_dir=weights, **kwargs
                    )
                self.assertFalse(weights.exists())


if __name__ == "__main__":
    unittest.main()
