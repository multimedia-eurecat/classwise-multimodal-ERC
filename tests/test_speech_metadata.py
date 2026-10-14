import csv
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aimara import (
    ASRMetadata,
    EmotionPrediction,
    ERCFeatureBridge,
    OfflineROBProcessor,
    ROBProcessorConfig,
    SpeakerConfidenceSpan,
    SpeakerInterval,
    SpeechRegion,
    SpeechSegment,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)
from aimara.batch import run_batch
from aimara.diarization import speaker_confidence_for_region
from aimara.pipeline import AIMARAPipeline


class SpeechMetadataTests(unittest.TestCase):
    def test_legacy_values_keep_unavailable_metadata_and_new_metadata_is_immutable(
        self,
    ):
        transcript = Transcript("hello", "en")
        self.assertEqual(transcript.metadata, ASRMetadata())
        self.assertIsNone(transcript.metadata.transcript_confidence)
        with self.assertRaises(FrozenInstanceError):
            transcript.metadata.backend = "whisper"
        with self.assertRaisesRegex(TypeError, "ASRMetadata"):
            Transcript("hello", metadata={})

    def test_invalid_confidence_cannot_be_serialized_as_valid_metadata(self):
        for score in (-1, 1.1, float("nan"), float("inf")):
            with (
                self.subTest(score=score),
                self.assertRaisesRegex(ValueError, "transcript_confidence"),
            ):
                ASRMetadata(transcript_confidence=score)
        for score in (None, 0, 1):
            self.assertEqual(
                json.loads(
                    json.dumps(
                        ASRMetadata(transcript_confidence=score).to_dict(),
                        allow_nan=False,
                    )
                )["transcript_confidence"],
                score,
            )

    def test_speaker_confidence_accounts_for_gaps_other_speakers_and_unscored_samples(
        self,
    ):
        timeline = [
            SpeakerInterval(0, 20, "speaker1", 0.9),
            SpeakerInterval(20, 40, "speaker1", -1),
            SpeakerInterval(40, 70, "speaker2", 1),
            SpeakerInterval(70, 90, "speaker1", 0),
        ]
        region = SpeechRegion(0, 100)
        self.assertAlmostEqual(
            speaker_confidence_for_region(region, "speaker1", timeline), 0.18
        )
        self.assertAlmostEqual(
            speaker_confidence_for_region(region, "speaker2", timeline), 0.3
        )
        self.assertEqual(
            speaker_confidence_for_region(SpeechRegion(70, 90), "speaker1", timeline), 0
        )
        self.assertEqual(
            speaker_confidence_for_region(SpeechRegion(20, 40), "speaker1", timeline),
            -1,
        )
        self.assertEqual(speaker_confidence_for_region(region, "UNKNOWN", timeline), -1)

    def test_combined_interval_retains_confidence_variation_and_available_evidence(
        self,
    ):
        interval = SpeakerInterval(
            0,
            100,
            "speaker1",
            -1,
            (
                SpeakerConfidenceSpan(0, 20, 0.9),
                SpeakerConfidenceSpan(20, 40, -1),
                SpeakerConfidenceSpan(40, 100, 0.4),
            ),
        )
        self.assertAlmostEqual(
            speaker_confidence_for_region(SpeechRegion(0, 100), "speaker1", [interval]),
            0.42,
        )
        self.assertAlmostEqual(
            speaker_confidence_for_region(SpeechRegion(0, 20), "speaker1", [interval]),
            0.9,
        )
        self.assertEqual(
            speaker_confidence_for_region(SpeechRegion(20, 40), "speaker1", [interval]),
            -1,
        )
        self.assertAlmostEqual(
            speaker_confidence_for_region(
                SpeechRegion(40, 100), "speaker1", [interval]
            ),
            0.4,
        )

    def test_confidence_spans_must_be_inside_and_ordered(self):
        for spans in (
            (SpeakerConfidenceSpan(0, 20, 1), SpeakerConfidenceSpan(10, 30, 1)),
            (SpeakerConfidenceSpan(0, 110, 1),),
        ):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                SpeakerInterval(0, 100, "speaker1", confidence_spans=spans)

    def test_unavailable_transcript_confidence_is_null_in_json_and_blank_in_csv(self):
        segment = SpeechSegment(
            conversation_id="demo",
            segment_id=0,
            source_video="demo.mp4",
            audio_path="demo.wav",
            start_time=0,
            end_time=1,
            speaker_id="UNKNOWN",
            transcript="hello",
            asr_metadata=ASRMetadata(backend="whisper", model="tiny.en"),
        )
        prediction = EmotionPrediction(
            segment, "neutral", 0.75, (0, 0, 0.75, 0, 0.1, 0.15)
        )
        pipeline = SimpleNamespace(process_video=lambda video, output: [prediction])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            (videos / "demo.mp4").touch()
            run_batch(pipeline, videos, root / "output")
            saved = json.loads((root / "output" / "predictions.json").read_text())[0]
            with (root / "output" / "predictions.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertIsNone(saved["transcript_confidence"])
        self.assertEqual(row["transcript_confidence"], "")
        self.assertEqual(float(row["speaker_id_confidence"]), -1)
        self.assertEqual(float(row["confidence"]), 0.75)

    def test_source_interval_and_speaker_confidence_validate_at_segment_boundary(self):
        kwargs = dict(
            conversation_id="demo",
            segment_id=0,
            source_video="demo.mp4",
            audio_path="segment.wav",
            start_time=1,
            end_time=2,
            speaker_id="speaker1",
            transcript="hello",
        )
        invalid = (
            {"speaker_id_confidence": -0.5},
            {"speaker_id_confidence": float("nan")},
            {"asr_input_start_time": 0},
            {"asr_input_start_time": 1.5, "asr_input_end_time": 3},
            {"asr_input_start_time": 0, "asr_input_end_time": float("inf")},
            {"start_time": float("nan")},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                SpeechSegment(**(kwargs | override))
        segment = SpeechSegment(**kwargs)
        self.assertEqual(segment.speaker_id_confidence, -1)
        self.assertIsNone(segment.asr_input_start_time)

    def test_split_groups_keep_decode_confidence_and_export_distinct_scores_to_json_and_csv(
        self,
    ):
        metadata = ASRMetadata(
            backend="parakeet",
            model="test-model",
            transcript_confidence=0.8,
            confidence_method="mean_word_confidence",
            language_source="configured",
            device="cpu",
            compute_type="float32",
        )
        words = (
            TranscriptWord(0, 0.8, " First."),
            TranscriptWord(1.2, 2.0, " Second."),
            TranscriptWord(2.2, 2.8, " Third."),
        )
        transcript = Transcript(
            "First. Second. Third.",
            "en",
            (TranscriptSegment(0, 2.8, "First. Second. Third.", words),),
            metadata,
        )
        vad = SimpleNamespace(
            config=ROBProcessorConfig(),
            process=lambda audio: [SpeechRegion(16000, 64000)],
        )
        diarizer = SimpleNamespace(
            diarize=lambda audio, rate, regions: [
                SpeakerInterval(16000, 32000, "speaker1", 0),
                SpeakerInterval(32000, 48000, "speaker2", 1),
            ]
        )
        processor = OfflineROBProcessor(
            vad_processor=vad,
            diarizer=diarizer,
            transcriber=SimpleNamespace(transcribe=lambda audio, rate: transcript),
            audio_decoder=lambda path, sample_rate: np.zeros(64000, dtype=np.float32),
        )

        class Encoder:
            def encode(self, segments):
                return np.ones((len(segments), 1), dtype=np.float32)

        class EmotionModel:
            def predict(self, features):
                np.testing.assert_array_equal(
                    features.speaker_mask, [[1, 0], [0, 1], [0, 0]]
                )
                return [
                    EmotionPrediction(s, "neutral", 0.75, (0, 0, 0.75, 0, 0.1, 0.15))
                    for s in features.segments
                ]

        encoder = Encoder()
        pipeline = AIMARAPipeline(
            processor, ERCFeatureBridge(encoder, encoder, encoder), EmotionModel()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            (videos / "demo.mp4").touch()
            rows, failures = run_batch(pipeline, videos, root / "output")
            self.assertEqual(failures, [])
            per_video = json.loads(
                (root / "output" / "demo" / "predictions.json").read_text()
            )
            combined = json.loads((root / "output" / "predictions.json").read_text())
            with (root / "output" / "predictions.csv").open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
        self.assertEqual(
            [s["speaker_id"] for s in per_video], ["speaker1", "speaker2", "UNKNOWN"]
        )
        self.assertEqual([s["speaker_id_confidence"] for s in per_video][::2], [0, -1])
        self.assertAlmostEqual(per_video[1]["speaker_id_confidence"], 1 / 1.1)
        for index, row in enumerate(rows):
            self.assertEqual(row["asr_backend"], "parakeet")
            self.assertEqual(row["asr_model"], "test-model")
            self.assertEqual(row["transcript_confidence"], 0.8)
            self.assertEqual(row["confidence"], 0.75)
            self.assertEqual(
                (row["asr_input_start_time"], row["asr_input_end_time"]), (1, 4)
            )
            self.assertEqual(combined[index], row)
            self.assertEqual(float(csv_rows[index]["transcript_confidence"]), 0.8)
            self.assertEqual(float(csv_rows[index]["confidence"]), 0.75)
            self.assertEqual(
                float(csv_rows[index]["speaker_id_confidence"]),
                row["speaker_id_confidence"],
            )
            self.assertEqual(csv_rows[index]["language_source"], "configured")


if __name__ == "__main__":
    unittest.main()
