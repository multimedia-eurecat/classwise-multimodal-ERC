import json
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf

from aimara import (
    ASRMetadata,
    EmotionPrediction,
    ERCFeatureBridge,
    OfflineROBProcessor,
    ROBProcessorConfig,
    SpeakerInterval,
    SpeechRegion,
    Transcript,
    TranscriptWord,
)
from aimara.pipeline import AIMARAPipeline
from aimara.rob import ParakeetConfig, ParakeetTranscriber
from aimara.rob.parakeet import ParakeetLanguageIdentifier


def hypothesis(raw=False):
    def entry(key, text, start, end):
        if raw:
            return {
                key: text,
                "start_offset": round(start * 10),
                "end_offset": round(end * 10),
            }
        return {key: text, "start": start, "end": end}

    return SimpleNamespace(
        text="Hello there. Goodbye now!",
        timestamp={
            "word": [
                entry("word", "Hello", 0.2, 0.7),
                entry("word", "there.", 0.7, 1.3),
                entry("word", "Goodbye", 2, 2.4),
                entry("word", "now!", 2.4, 2.9),
            ],
            "segment": [
                entry("segment", "Hello there.", 0.2, 1.3),
                entry("segment", "Goodbye now!", 2, 2.9),
            ],
        },
    )


class NemoModel:
    def __init__(self, outputs=None):
        self.cfg = OmegaConf.create(
            {
                "decoding": {
                    "compute_timestamps": False,
                    "preserve_alignments": False,
                    "confidence_cfg": {},
                },
                "preprocessor": {"window_stride": 0.025},
            }
        )
        self.encoder = SimpleNamespace(subsampling_factor=4)
        self.outputs = iter(outputs or [[hypothesis()]])
        self.calls, self.decoding_changes = [], []
        self.to, self.float, self.eval = Mock(), Mock(), Mock()

    def transcribe(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        result = next(self.outputs)
        if isinstance(result, Exception):
            raise result
        return result

    def change_decoding_strategy(self, config, verbose=False):
        self.decoding_changes.append(OmegaConf.to_container(config))


def nemo_modules(model):
    root, collections, asr = (
        types.ModuleType(name)
        for name in ("nemo", "nemo.collections", "nemo.collections.asr")
    )
    root.collections, collections.asr = collections, asr
    asr.models = SimpleNamespace(
        ASRModel=SimpleNamespace(restore_from=Mock(return_value=model))
    )
    return {
        "nemo": root,
        "nemo.collections": collections,
        "nemo.collections.asr": asr,
    }, asr.models.ASRModel.restore_from


class ParakeetTests(unittest.TestCase):
    def audio(self, seconds=3):
        return np.zeros(round(seconds * 16000), dtype=np.float32)

    def test_seconds_and_raw_offsets_produce_identical_timed_transcripts(self):
        results = []
        for raw in (False, True):
            model = NemoModel([[hypothesis(raw)]])
            transcriber = ParakeetTranscriber(
                model=model, config=ParakeetConfig(language="en")
            )
            results.append(transcriber.transcribe(self.audio(), 16000))
            self.assertEqual(len(model.calls), 1)
            self.assertEqual(
                model.calls[0][1],
                {"batch_size": 1, "return_hypotheses": True, "verbose": False},
            )
        self.assertEqual(results[0].text, "Hello there. Goodbye now!")
        self.assertEqual(results[0].language, "en")
        self.assertEqual(
            results[0].words,
            (
                TranscriptWord(0.2, 0.7, " Hello"),
                TranscriptWord(0.7, 1.3, " there."),
                TranscriptWord(2, 2.4, " Goodbye"),
                TranscriptWord(2.4, 2.9, " now!"),
            ),
        )
        np.testing.assert_allclose(
            [w.start_time for w in results[1].words],
            [w.start_time for w in results[0].words],
        )
        self.assertEqual(
            [s.text for s in results[1].segments], [s.text for s in results[0].segments]
        )

    def test_boundary_word_belongs_to_one_sentence_and_partial_metadata_keeps_text(
        self,
    ):
        hyp = SimpleNamespace(
            text="Hello. Goodbye.",
            timestamp={
                "word": [
                    {"word": "Hello.", "start": 0, "end": 1},
                    {"word": "Goodbye.", "start": 1, "end": 2},
                ],
                "segment": [
                    {"segment": "Hello.", "start": 0, "end": 1},
                    {"segment": "Goodbye.", "start": 1, "end": 2},
                ],
            },
        )
        transcriber = ParakeetTranscriber(model=NemoModel())
        result = transcriber._to_transcript(hyp, 2)
        self.assertEqual([len(s.words) for s in result.segments], [1, 1])
        self.assertEqual(len(result.words), 2)
        hyp.timestamp["segment"] = hyp.timestamp["segment"][:1]
        result = transcriber._to_transcript(hyp, 2)
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, hyp.text)
        self.assertEqual(len(result.words), 2)

    def test_text_without_alignment_retains_the_full_region(self):
        model = NemoModel([[SimpleNamespace(text="Hello.", timestamp=None)]])
        result = ParakeetTranscriber(model=model).transcribe(self.audio(2), 16000)
        self.assertEqual(result.text, "Hello.")
        self.assertEqual(result.words, ())
        self.assertEqual(
            (result.segments[0].start_time, result.segments[0].end_time), (0, 2)
        )

    def test_short_and_empty_results_do_not_run_language_identification(self):
        identifier = SimpleNamespace(probabilities=Mock())
        model = NemoModel([[], [SimpleNamespace(text="", timestamp={})]])
        transcriber = ParakeetTranscriber(model=model, language_identifier=identifier)
        empty = Transcript(
            "",
            metadata=ASRMetadata(
                backend="parakeet",
                model="nvidia/parakeet-tdt-0.6b-v3",
                device="cpu",
                compute_type="float32",
            ),
        )
        self.assertEqual(transcriber.transcribe(self.audio(0.005), 16000), empty)
        self.assertEqual(len(model.calls), 0)
        self.assertEqual(transcriber.transcribe(self.audio(), 16000), empty)
        self.assertEqual(transcriber.transcribe(self.audio(), 16000), empty)
        identifier.probabilities.assert_not_called()

    def test_confidence_failure_retries_and_restores_decoding_even_if_retry_fails(self):
        for error_type in (ValueError, RuntimeError):
            for retry_fails in (False, True):
                with self.subTest(error_type=error_type, retry_fails=retry_fails):
                    recovered = hypothesis()
                    recovered.word_confidence = [0.9] * 4
                    second = (
                        RuntimeError("retry failed") if retry_fails else [recovered]
                    )
                    model = NemoModel(
                        [
                            error_type(
                                "Something went wrong with word-level confidence aggregation"
                            ),
                            second,
                        ]
                    )
                    transcriber = ParakeetTranscriber(model=model)
                    if retry_fails:
                        with self.assertRaisesRegex(
                            RuntimeError, "Parakeet transcription failed"
                        ):
                            transcriber.transcribe(self.audio(), 16000)
                    else:
                        result = transcriber.transcribe(self.audio(), 16000)
                        self.assertEqual(result.text, hypothesis().text)
                        self.assertIsNone(result.metadata.transcript_confidence)
                    self.assertEqual(len(model.calls), 2)
                    self.assertEqual(
                        [
                            c["confidence_cfg"]["preserve_word_confidence"]
                            for c in model.decoding_changes
                        ],
                        [False, True],
                    )
                    self.assertTrue(model.cfg.decoding.compute_timestamps)
                    self.assertTrue(model.cfg.decoding.preserve_alignments)

    def test_other_decoder_failures_are_not_retried(self):
        model = NemoModel([RuntimeError("CUDA failed")])
        with self.assertRaisesRegex(
            RuntimeError, "Parakeet transcription failed"
        ) as caught:
            ParakeetTranscriber(model=model).transcribe(self.audio(), 16000)
        self.assertEqual(str(caught.exception.__cause__), "CUDA failed")
        self.assertEqual(len(model.calls), 1)

    def test_auto_language_has_no_stale_fallback_between_utterances(self):
        identifier = SimpleNamespace(
            probabilities=Mock(
                side_effect=[
                    {"en": 0.02, "es": 0.98},
                    {"en": 0.55, "es": 0.45},
                    RuntimeError("unavailable"),
                ]
            )
        )
        model = NemoModel([[hypothesis()]] * 4)
        transcriber = ParakeetTranscriber(
            model=model,
            config=ParakeetConfig(language="en,es"),
            language_identifier=identifier,
        )
        self.assertEqual(
            [transcriber.transcribe(self.audio(), 16000).language for _ in range(3)],
            ["es", None, None],
        )
        self.assertIsNone(transcriber.transcribe(self.audio(0.2), 16000).language)
        self.assertEqual(identifier.probabilities.call_count, 3)

    def test_confidence_is_mean_of_reported_word_scores_and_zero_is_available(self):
        for values, expected in (([0.9, 0.8, 0.7, 0.8], 0.8), ([0.0] * 4, 0.0)):
            with self.subTest(values=values):
                hyp = hypothesis()
                hyp.word_confidence = values
                result = ParakeetTranscriber(
                    model=NemoModel([[hyp]]), config=ParakeetConfig(language="en")
                ).transcribe(self.audio(), 16000)
                self.assertAlmostEqual(result.metadata.transcript_confidence, expected)
                self.assertEqual(
                    result.metadata.confidence_method, "mean_word_confidence"
                )
                self.assertEqual(result.metadata.language_source, "configured")
                self.assertEqual(result.metadata.backend, "parakeet")

    def test_missing_or_invalid_confidence_preserves_transcript_without_a_score(self):
        for values in (
            None,
            [],
            [float("nan")],
            [float("inf")],
            [-0.1],
            [1.1],
            ["invalid"],
        ):
            with self.subTest(values=values):
                hyp = hypothesis()
                hyp.word_confidence = values
                result = ParakeetTranscriber(model=NemoModel([[hyp]])).transcribe(
                    self.audio(), 16000
                )
                self.assertEqual(result.text, hyp.text)
                self.assertIsNone(result.metadata.transcript_confidence)
                self.assertIsNone(result.metadata.confidence_method)

    def test_invalid_languages_audio_and_timestamps_fail_explicitly(self):
        for language in ("ca", "en,ca", "", "auto,en"):
            with self.subTest(language=language), self.assertRaises(ValueError):
                ParakeetConfig(language=language)
        transcriber = ParakeetTranscriber(model=NemoModel())
        for audio, rate, error in (
            (np.zeros((2, 2), dtype=np.float32), 16000, ValueError),
            (np.zeros(20), 16000, TypeError),
            (self.audio(), 8000, ValueError),
        ):
            with self.subTest(rate=rate, dtype=audio.dtype), self.assertRaises(error):
                transcriber.transcribe(audio, rate)
        hyp = hypothesis()
        hyp.timestamp["word"][0]["start"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Parakeet transcription failed"):
            ParakeetTranscriber(model=NemoModel([[hyp]])).transcribe(
                self.audio(), 16000
            )

    def test_missing_optional_dependency_fails_without_creating_cache(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"nemo": None}),
        ):
            weights = Path(directory) / "new-cache"
            with self.assertRaisesRegex(RuntimeError, "optional nemo_toolkit"):
                ParakeetTranscriber.from_nemo(
                    weights_dir=weights, config=ParakeetConfig(language="en")
                )
            self.assertFalse(weights.exists())

    def test_loader_restores_local_model_float32_and_warms_up_once(self):
        model = NemoModel()
        modules, restore = nemo_modules(model)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.nemo"
            checkpoint.touch()
            config = ParakeetConfig(model_name=str(checkpoint), language="en")
            with (
                patch.dict(sys.modules, modules),
                patch("aimara.rob.parakeet.ParakeetLanguageIdentifier.load") as langid,
            ):
                transcriber = ParakeetTranscriber.from_nemo(
                    weights_dir=directory, config=config, device="cpu"
                )
            restore.assert_called_once_with(str(checkpoint), map_location="cpu")
            model.to.assert_called_once_with("cpu")
            model.float.assert_called_once()
            model.eval.assert_called_once()
            self.assertEqual(transcriber.effective_compute_type, "float32")
            self.assertEqual(len(model.calls), 1)
            np.testing.assert_array_equal(
                model.calls[0][0][0], np.zeros(16000, dtype=np.float32)
            )
            langid.assert_not_called()
            self.assertEqual(
                model.decoding_changes[0]["confidence_cfg"]["aggregation"], "mean"
            )

    def test_cached_hf_checkpoint_reuses_cache_and_offline_download_is_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.nemo"
            checkpoint.touch()
            backend = SimpleNamespace(
                try_to_load_from_cache=Mock(return_value=str(checkpoint)),
                hf_hub_download=Mock(return_value=str(checkpoint)),
            )
            with patch.dict(sys.modules, {"huggingface_hub": backend}):
                self.assertEqual(
                    ParakeetTranscriber._resolve_checkpoint(
                        ParakeetConfig(), Path(directory)
                    ),
                    checkpoint,
                )
                backend.hf_hub_download.assert_not_called()
                backend.try_to_load_from_cache.return_value = None
                config = ParakeetConfig(local_files_only=True)
                self.assertEqual(
                    ParakeetTranscriber._resolve_checkpoint(config, Path(directory)),
                    checkpoint,
                )
                backend.hf_hub_download.assert_called_once_with(
                    repo_id=config.model_name,
                    filename="parakeet-tdt-0.6b-v3.nemo",
                    cache_dir=directory,
                    local_files_only=True,
                )

    def test_optional_langid_load_failure_preserves_transcription(self):
        model = NemoModel([[hypothesis()]])
        modules, _ = nemo_modules(model)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, modules),
            patch(
                "aimara.rob.parakeet.ParakeetTranscriber._resolve_checkpoint",
                return_value=Path(directory) / "model.nemo",
            ),
            patch(
                "aimara.rob.parakeet.ParakeetLanguageIdentifier.load",
                side_effect=RuntimeError("missing weights"),
            ),
        ):
            transcriber = ParakeetTranscriber.from_nemo(
                weights_dir=directory,
                config=ParakeetConfig(warm_up=False),
                device="cpu",
            )
        result = transcriber.transcribe(self.audio(), 16000)
        self.assertEqual(result.text, hypothesis().text)
        self.assertIsNone(result.language)

    def test_ambernet_probabilities_are_restricted_to_configured_languages(self):
        import torch

        model = SimpleNamespace(
            cfg=SimpleNamespace(train_ds=SimpleNamespace(labels=["en", "es", "ca"])),
            forward=Mock(return_value=(torch.tensor([[0.0, 2.0, 10.0]]), None)),
        )
        identifier = ParakeetLanguageIdentifier(model, "cpu")
        probabilities = identifier.probabilities(self.audio(), ["en", "es", "xx"])
        self.assertEqual(set(probabilities), {"en", "es"})
        self.assertAlmostEqual(sum(probabilities.values()), 1, delta=1e-6)
        self.assertGreater(probabilities["es"], probabilities["en"])

    def test_parakeet_words_reach_speaker_split_wavs_and_prediction_json(self):
        source_audio = np.concatenate(
            (np.zeros(16000, dtype=np.float32), np.full(48000, 0.5, dtype=np.float32))
        )
        vad = SimpleNamespace(
            config=ROBProcessorConfig(),
            process=lambda audio: [SpeechRegion(16000, 64000)],
        )
        diarizer = SimpleNamespace(
            diarize=lambda audio, rate, regions: [
                SpeakerInterval(16000, 40000, "speaker1"),
                SpeakerInterval(40000, 64000, "speaker2"),
            ]
        )
        processor = OfflineROBProcessor(
            vad_processor=vad,
            diarizer=diarizer,
            transcriber=ParakeetTranscriber(
                model=NemoModel(), config=ParakeetConfig(language="en")
            ),
            audio_decoder=lambda path, sample_rate: source_audio,
        )

        class Encoder:
            def encode(self, segments):
                return np.ones((len(segments), 1), dtype=np.float32)

        class EmotionModel:
            def predict(self, features):
                np.testing.assert_array_equal(features.speaker_mask, [[1, 0], [0, 1]])
                return [
                    EmotionPrediction(s, "neutral", 1, (0, 0, 1, 0, 0, 0))
                    for s in features.segments
                ]

        encoder = Encoder()
        pipeline = AIMARAPipeline(
            processor, ERCFeatureBridge(encoder, encoder, encoder), EmotionModel()
        )
        with tempfile.TemporaryDirectory() as directory:
            predictions = pipeline.process_video("video.mp4", directory)
            saved = json.loads((Path(directory) / "predictions.json").read_text())
            for prediction, count in zip(predictions, (26400, 21600)):
                with wave.open(str(prediction.segment.audio_path)) as clip:
                    self.assertEqual(clip.getnframes(), count)
        self.assertEqual(
            [p["transcript"] for p in saved], ["Hello there.", "Goodbye now!"]
        )
        self.assertEqual([p["speaker_id"] for p in saved], ["speaker1", "speaker2"])
        self.assertEqual([p["language"] for p in saved], ["en", "en"])
        self.assertEqual(
            [(p["start_time"], p["end_time"]) for p in saved], [(1, 2.65), (2.65, 4)]
        )


if __name__ == "__main__":
    unittest.main()
