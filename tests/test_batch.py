import csv
import json
import tempfile
import unittest
from pathlib import Path

from aimara.batch import run_batch


class StubPrediction:
    def __init__(self, video_name):
        self.video_name = video_name

    def to_dict(self):
        return {
            "conversation_id": Path(self.video_name).stem,
            "segment_id": 0,
            "start_time": 0.0,
            "end_time": 1.0,
            "speaker_id": "UNKNOWN",
            "transcript": "hello",
            "language": "en",
            "emotion": "neutral",
            "confidence": 0.75,
            "probabilities": {"neutral": 0.75, "angry": 0.25},
        }


class StubPipeline:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def process_video(self, video_path, output_dir):
        self.calls.append(video_path.name)
        if video_path.name in self.fail_on:
            raise RuntimeError("intentional failure")
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction = StubPrediction(video_path.name)
        (output_dir / "predictions.json").write_text(
            json.dumps([prediction.to_dict()]), encoding="utf-8"
        )
        return [prediction]


class BatchRunnerTests(unittest.TestCase):
    def make_videos(self, root, names):
        video_dir = root / "videos"
        video_dir.mkdir()
        for name in names:
            (video_dir / name).touch()
        return video_dir

    def test_processes_sorted_videos_and_writes_combined_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_dir = self.make_videos(root, ["Sadness.mp4", "Anger.mp4", "ignore.txt"])
            output_dir = root / "output"
            pipeline = StubPipeline()

            rows, failures = run_batch(pipeline, video_dir, output_dir)

            self.assertEqual(pipeline.calls, ["Anger.mp4", "Sadness.mp4"])
            self.assertEqual(failures, [])
            self.assertEqual([row["expected_emotion"] for row in rows], ["anger", "sadness"])
            self.assertEqual(len(json.loads((output_dir / "predictions.json").read_text())), 2)
            with (output_dir / "predictions.csv").open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[0]["probability_angry"], "0.25")

    def test_continues_after_failure_and_records_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_dir = self.make_videos(root, ["Anger.mp4", "Joy.mp4"])
            output_dir = root / "output"

            rows, failures = run_batch(
                StubPipeline(fail_on={"Anger.mp4"}), video_dir, output_dir
            )

            self.assertEqual([row["video_file"] for row in rows], ["Joy.mp4"])
            self.assertEqual(failures[0]["video_file"], "Anger.mp4")
            self.assertEqual(failures[0]["error_type"], "RuntimeError")
            self.assertEqual(
                json.loads((output_dir / "failures.json").read_text()), failures
            )

    def test_resume_reuses_valid_per_video_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_dir = self.make_videos(root, ["Anger.mp4", "Joy.mp4"])
            output_dir = root / "output"
            first = StubPipeline()
            run_batch(first, video_dir, output_dir)

            resumed = StubPipeline()
            rows, failures = run_batch(resumed, video_dir, output_dir, resume=True)

            self.assertEqual(resumed.calls, [])
            self.assertEqual(len(rows), 2)
            self.assertEqual(failures, [])

    def test_resume_reruns_invalid_per_video_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_dir = self.make_videos(root, ["Anger.mp4"])
            output_dir = root / "output"
            result_dir = output_dir / "Anger"
            result_dir.mkdir(parents=True)
            (result_dir / "predictions.json").write_text("{truncated")
            pipeline = StubPipeline()

            rows, failures = run_batch(pipeline, video_dir, output_dir, resume=True)

            self.assertEqual(pipeline.calls, ["Anger.mp4"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
