import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from aimara import (
    EmotionPrediction,
    ERCFeatureBridge,
    OfflineROBProcessor,
    ROBProcessorConfig,
    SpeechRegion,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    write_wav,
)
from aimara.pipeline import AIMARAPipeline
from aimara.rob import ROBSpeakerDiarizer


class OfflineDiarizationPipelineTests(unittest.TestCase):
    def test_one_vad_region_produces_aligned_a_b_a_emotion_inputs_and_wavs(self):
        source_audio = np.concatenate(
            (
                np.zeros(16000, dtype=np.float32),
                *(
                    np.full(32000, value, dtype=np.float32)
                    for value in (0.25, 0.75, 0.25)
                ),
            )
        )
        source_region = SpeechRegion(16000, len(source_audio))
        words = tuple(
            TranscriptWord(*args)
            for args in [
                (0.2, 0.6, " Hello"),
                (0.8, 1.8, " there."),
                (2.2, 2.8, " Goodbye"),
                (3.0, 3.8, " now!"),
                (4.2, 4.8, " Welcome"),
                (5.0, 5.8, " back."),
            ]
        )
        text = "".join(word.text for word in words).strip()

        class VAD:
            config = ROBProcessorConfig()

            def process(self, samples):
                return [source_region]

        class Transcriber:
            calls = 0

            def transcribe(self, samples, rate):
                self.calls += 1
                np.testing.assert_array_equal(samples, source_audio[16000:])
                return Transcript(
                    text, "en", (TranscriptSegment(0.2, 5.8, text, words),)
                )

        class Encoder:
            def encode(self, segments):
                self.seen = tuple(
                    (s.start_time, s.end_time, s.audio_path, s.source_video)
                    for s in segments
                )
                return np.ones((len(segments), 1), dtype=np.float32)

        class EmotionModel:
            def predict(self, features):
                np.testing.assert_array_equal(
                    features.speaker_mask, [[1, 0], [0, 1], [1, 0]]
                )
                return [
                    EmotionPrediction(s, "neutral", 1, (0, 0, 1, 0, 0, 0))
                    for s in features.segments
                ]

        transcriber, encoder = Transcriber(), Encoder()
        processor = OfflineROBProcessor(
            vad_processor=VAD(),
            transcriber=transcriber,
            diarizer=ROBSpeakerDiarizer(
                embedder=lambda samples: np.array(
                    [
                        np.count_nonzero(samples == 0.25),
                        np.count_nonzero(samples == 0.75),
                    ],
                    dtype=np.float32,
                )
            ),
            audio_decoder=lambda path, sample_rate: source_audio,
        )
        pipeline = AIMARAPipeline(
            processor, ERCFeatureBridge(encoder, encoder, encoder), EmotionModel()
        )
        with tempfile.TemporaryDirectory() as directory:
            predictions = pipeline.process_video("conversation.mp4", directory)
            saved = json.loads((Path(directory) / "predictions.json").read_text())
            frames = []
            for prediction in predictions:
                with wave.open(str(prediction.segment.audio_path)) as wav:
                    self.assertEqual(wav.getnframes(), 32000)
                    frames.append(wav.readframes(wav.getnframes()))
            reference = write_wav(
                Path(directory) / "reference.wav", source_audio[16000:]
            )
            with wave.open(str(reference)) as wav:
                self.assertEqual(b"".join(frames), wav.readframes(wav.getnframes()))
        self.assertEqual(transcriber.calls, 1)
        self.assertEqual(
            [p["speaker_id"] for p in saved], ["speaker1", "speaker2", "speaker1"]
        )
        self.assertEqual([p["segment_id"] for p in saved], [0, 1, 2])
        self.assertEqual(
            [(p["start_time"], p["end_time"]) for p in saved], [(1, 3), (3, 5), (5, 7)]
        )
        self.assertEqual(
            [p["transcript"] for p in saved],
            ["Hello there.", "Goodbye now!", "Welcome back."],
        )
        self.assertEqual(
            [(start, end) for start, end, _, _ in encoder.seen],
            [(1, 3), (3, 5), (5, 7)],
        )
        self.assertTrue(
            all(video.name == "conversation.mp4" for _, _, _, video in encoder.seen)
        )

    def test_two_voices_reach_speaker_mask_and_serialized_emotion_predictions(self):
        audio = np.concatenate(
            [np.full(32000, value, dtype=np.float32) for value in (1, 2, 1)]
        )

        class VAD:
            config = ROBProcessorConfig()

            def process(self, samples):
                return [SpeechRegion(i * 32000, (i + 1) * 32000) for i in range(3)]

        class Transcriber:
            def transcribe(self, samples, rate):
                return Transcript("hello" if samples[0] == 1 else "goodbye", "en")

        class Encoder:
            def encode(self, segments):
                return np.ones((len(segments), 1), dtype=np.float32)

        class EmotionModel:
            def predict(self, features):
                np.testing.assert_array_equal(
                    features.speaker_mask, [[1, 0], [0, 1], [1, 0]]
                )
                return [
                    EmotionPrediction(s, "neutral", 1.0, (0, 0, 1, 0, 0, 0))
                    for s in features.segments
                ]

        processor = OfflineROBProcessor(
            vad_processor=VAD(),
            transcriber=Transcriber(),
            diarizer=ROBSpeakerDiarizer(
                embedder=lambda samples: np.array(
                    [1, 0] if samples[0] == 1 else [0, 1], dtype=np.float32
                )
            ),
            audio_decoder=lambda path, sample_rate: audio,
        )
        encoder = Encoder()
        pipeline = AIMARAPipeline(
            processor, ERCFeatureBridge(encoder, encoder, encoder), EmotionModel()
        )
        with tempfile.TemporaryDirectory() as directory:
            pipeline.process_video("conversation.mp4", directory)
            predictions = json.loads((Path(directory) / "predictions.json").read_text())
        self.assertEqual(
            [p["speaker_id"] for p in predictions], ["speaker1", "speaker2", "speaker1"]
        )
        self.assertEqual(
            [(p["start_time"], p["end_time"]) for p in predictions],
            [(0, 2), (2, 4), (4, 6)],
        )
        self.assertEqual(
            [p["transcript"] for p in predictions], ["hello", "goodbye", "hello"]
        )


if __name__ == "__main__":
    unittest.main()
