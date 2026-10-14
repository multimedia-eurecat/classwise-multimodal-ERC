import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from aimara.batch import run_batch
from aimara.batch_resume import MANIFEST_NAME, file_digest


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
            **dict.fromkeys(
                (
                    "asr_backend",
                    "asr_model",
                    "transcript_confidence",
                    "transcript_confidence_method",
                    "speaker_id_confidence",
                    "language_source",
                    "asr_device",
                    "asr_compute_type",
                    "asr_input_start_time",
                    "asr_input_end_time",
                )
            ),
        }


class StubPipeline:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)
        self.resume_config = {"test_pipeline": "v1"}

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
            video_dir = self.make_videos(
                root, ["Sadness.mp4", "Anger.mp4", "ignore.txt"]
            )
            output_dir = root / "output"
            pipeline = StubPipeline()

            rows, failures = run_batch(pipeline, video_dir, output_dir)

            self.assertEqual(pipeline.calls, ["Anger.mp4", "Sadness.mp4"])
            self.assertEqual(failures, [])
            self.assertEqual(
                [row["expected_emotion"] for row in rows], ["anger", "sadness"]
            )
            self.assertEqual(
                len(json.loads((output_dir / "predictions.json").read_text())), 2
            )
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

    def test_changed_configuration_reruns_and_then_reuses_new_results(self):
        for name, value in (
            ("asr", {"backend": "parakeet"}),
            ("vad", {"threshold": 0.7}),
            ("diarization", {"backend": "redimnet2"}),
            ("erc", {"unknown_speaker_policy": "reserve_slot"}),
            ("encoders", {"audio": "different-model"}),
        ):
            with self.subTest(setting=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                videos = self.make_videos(root, ["Anger.mp4"])
                output = root / "output"
                run_batch(StubPipeline(), videos, output)
                changed = StubPipeline()
                changed.resume_config[name] = value
                run_batch(changed, videos, output, resume=True)
                self.assertEqual(changed.calls, ["Anger.mp4"])
                changed.calls.clear()
                run_batch(changed, videos, output, resume=True)
                self.assertEqual(changed.calls, [])

    def test_legacy_results_without_manifest_rerun_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = self.make_videos(root, ["Anger.mp4"])
            output = root / "output"
            StubPipeline().process_video(videos / "Anger.mp4", output / "Anger")
            pipeline = StubPipeline()
            run_batch(pipeline, videos, output, resume=True)
            self.assertEqual(pipeline.calls, ["Anger.mp4"])
            self.assertTrue((output / "Anger" / MANIFEST_NAME).is_file())
            pipeline.calls.clear()
            run_batch(pipeline, videos, output, resume=True)
            self.assertEqual(pipeline.calls, [])

    def test_damaged_or_incompatible_manifests_rerun(self):
        for contents in ("{truncated", "[]", '{"cache_version": 999}'):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                videos = self.make_videos(root, ["Anger.mp4"])
                output = root / "output"
                run_batch(StubPipeline(), videos, output)
                (output / "Anger" / MANIFEST_NAME).write_text(contents)
                pipeline = StubPipeline()
                _, failures = run_batch(pipeline, videos, output, resume=True)
                self.assertEqual(pipeline.calls, ["Anger.mp4"])
                self.assertEqual(failures, [])

    def test_modified_predictions_or_old_schema_rerun(self):
        for refresh_digest in (False, True):
            with (
                self.subTest(refresh_digest=refresh_digest),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                videos = self.make_videos(root, ["Anger.mp4"])
                output = root / "output"
                run_batch(StubPipeline(), videos, output)
                result = output / "Anger" / "predictions.json"
                result.write_text('[{"transcript": "edited"}]')
                if refresh_digest:
                    manifest_path = output / "Anger" / MANIFEST_NAME
                    manifest = json.loads(manifest_path.read_text())
                    manifest["predictions_sha256"] = file_digest(result)
                    manifest_path.write_text(json.dumps(manifest))
                pipeline = StubPipeline()
                run_batch(pipeline, videos, output, resume=True)
                self.assertEqual(pipeline.calls, ["Anger.mp4"])

    def test_source_change_reruns_only_changed_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = self.make_videos(root, ["Anger.mp4", "Joy.mp4"])
            output = root / "output"
            run_batch(StubPipeline(), videos, output)
            (videos / "Joy.mp4").write_bytes(b"new video")
            pipeline = StubPipeline()
            rows, failures = run_batch(pipeline, videos, output, resume=True)
            self.assertEqual(pipeline.calls, ["Joy.mp4"])
            self.assertEqual(len(rows), 2)
            self.assertEqual(failures, [])

    def test_checkpoint_replacement_at_same_path_is_detected_even_with_same_stat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = self.make_videos(root, ["Anger.mp4"])
            output = root / "output"
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"old weights")
            config = {"checkpoint": checkpoint}
            run_batch(StubPipeline(), videos, output, run_config=config)
            stat = checkpoint.stat()
            checkpoint.write_bytes(b"new weights")
            os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            pipeline = StubPipeline()
            run_batch(pipeline, videos, output, resume=True, run_config=config)
            self.assertEqual(pipeline.calls, ["Anger.mp4"])

    def test_model_cache_changes_are_detected_but_locks_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = self.make_videos(root, ["Anger.mp4"])
            output = root / "output"
            cache = root / "weights"
            cache.mkdir()
            weights = cache / "model.bin"
            weights.write_bytes(b"weights")
            config = {"weights": cache}
            run_batch(StubPipeline(), videos, output, run_config=config)
            (cache / "download.lock").touch()
            pipeline = StubPipeline()
            run_batch(pipeline, videos, output, resume=True, run_config=config)
            self.assertEqual(pipeline.calls, [])
            weights.write_bytes(b"changed weights")
            run_batch(pipeline, videos, output, resume=True, run_config=config)
            self.assertEqual(pipeline.calls, ["Anger.mp4"])

    def test_interrupted_configuration_change_preserves_only_completed_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = self.make_videos(root, ["Anger.mp4", "Joy.mp4"])
            output = root / "output"
            run_batch(StubPipeline(), videos, output)
            changed = {"test_pipeline": "v2"}
            _, failures = run_batch(
                StubPipeline(fail_on={"Anger.mp4"}),
                videos,
                output,
                resume=True,
                run_config=changed,
            )
            self.assertEqual(len(failures), 1)
            self.assertFalse((output / "Anger" / MANIFEST_NAME).exists())
            recovered = StubPipeline()
            rows, failures = run_batch(
                recovered, videos, output, resume=True, run_config=changed
            )
            self.assertEqual(recovered.calls, ["Anger.mp4"])
            self.assertEqual(len(rows), 2)
            self.assertEqual(failures, [])

    def test_unknown_configuration_cannot_reuse_or_keep_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = self.make_videos(root, ["Anger.mp4"])
            output = root / "output"
            run_batch(StubPipeline(), videos, output)
            pipeline = StubPipeline()
            pipeline.resume_config = None
            run_batch(pipeline, videos, output, resume=True)
            self.assertEqual(pipeline.calls, ["Anger.mp4"])
            self.assertFalse((output / "Anger" / MANIFEST_NAME).exists())

    def test_valid_empty_predictions_are_reused(self):
        class SilentPipeline(StubPipeline):
            def process_video(self, video_path, output_dir):
                self.calls.append(video_path.name)
                return []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = self.make_videos(root, ["Silence.mp4"])
            output = root / "output"
            pipeline = SilentPipeline()
            run_batch(pipeline, videos, output)
            pipeline.calls.clear()
            rows, failures = run_batch(pipeline, videos, output, resume=True)
            self.assertEqual(pipeline.calls, [])
            self.assertEqual((rows, failures), ([], []))


if __name__ == "__main__":
    unittest.main()
