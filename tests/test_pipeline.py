import json
import tempfile
import unittest
from pathlib import Path

from aimara import EmotionPrediction, SpeechSegment
from aimara.pipeline import AIMARAPipeline


class StubSpeechProcessor:
    def __init__(self, segments):
        self.segments = segments

    def process_video(self, video_path, output_dir):
        return self.segments


class StubBridge:
    def build(self, segments):
        return "features"


class StubEmotionModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, features):
        return self.predictions


class AIMARAPipelineTests(unittest.TestCase):
    def test_runs_pipeline_and_writes_json(self):
        segment = SpeechSegment(
            conversation_id="demo",
            segment_id=0,
            source_video="demo.mp4",
            audio_path="demo.wav",
            start_time=0.0,
            end_time=1.0,
            speaker_id="UNKNOWN",
            transcript="hello",
            language="en",
        )
        prediction = EmotionPrediction(segment, "angry", 0.75, (0, 0, 0, 0.75, 0.1, 0.15))
        pipeline = AIMARAPipeline(
            StubSpeechProcessor([segment]),
            StubBridge(),
            StubEmotionModel([prediction]),
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.process_video("demo.mp4", tmp)
            saved = json.loads((Path(tmp) / "predictions.json").read_text())

        self.assertEqual(result, [prediction])
        self.assertEqual(saved[0]["emotion"], "angry")
        self.assertEqual(saved[0]["transcript"], "hello")

    def test_empty_speech_still_writes_results(self):
        pipeline = AIMARAPipeline(
            StubSpeechProcessor([]), StubBridge(), StubEmotionModel([])
        )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(pipeline.process_video("demo.mp4", tmp), [])
            self.assertEqual(json.loads((Path(tmp) / "predictions.json").read_text()), [])


if __name__ == "__main__":
    unittest.main()
