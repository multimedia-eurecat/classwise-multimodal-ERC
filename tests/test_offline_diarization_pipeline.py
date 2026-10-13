import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aimara import (
    EmotionPrediction,
    ERCFeatureBridge,
    OfflineROBProcessor,
    ROBProcessorConfig,
    SpeechRegion,
    Transcript,
)
from aimara.pipeline import AIMARAPipeline
from aimara.rob import ROBSpeakerDiarizer


class OfflineDiarizationPipelineTests(unittest.TestCase):
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
