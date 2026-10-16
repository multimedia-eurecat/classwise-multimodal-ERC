import json
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aimara.batch_resume import file_digest
from aimara.rob_native import (
    ProcessVAD,
    ReplayLogger,
    VADDelivery,
    load_settings,
    node_callback_class,
)
from aimara.rob_native_export import ROBExportProcessor, observe_groups
from aimara.run_rob_native import build_parser, run
from aimara.erc_features import ERCFeatureBridge
from aimara.erc_model import EmotionPrediction


class PublishingEngine:
    """Minimal publication contract, without copying ROB's segmentation logic."""

    def __init__(self):
        self.published = []
        self._on_transcript_ready = lambda *args: self.published.append(args)

    def _group_segments_by_speaker(self, segments, start, end):
        return segments

    def publish(self, groups, start=100.0):
        self._group_segments_by_speaker(groups, start, start + 10)
        for item in groups:
            self._on_transcript_ready(
                item["text"], item["speaker"], "en", 0.0, 12, 250, 1000, 0.012, 0.7
            )


def group(text="Hello.", speaker="speaker1", start=0.25, end=1.25):
    return {"text": text, "speaker": speaker, "start_offset": start, "end_offset": end}


class ROBNativeTests(unittest.TestCase):
    def test_vad_model_state_lives_in_a_separate_process_and_is_drained(self):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "speech_recognition/speech_recognition"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "vad_engine.py").write_text("""
class VADEngine:
    def __init__(self, **kwargs):
        self.count = 0
    def predict(self, audio, rate):
        self.count += 1
        return float(self.count)
""")
            proxy = ProcessVAD(
                root,
                {"repo_model": "fixture", "model_name": "fixture"},
                root,
                timeout=10,
            )
            try:
                self.assertNotEqual(proxy.process.pid, os.getpid())
                self.assertEqual(
                    proxy.predict(np.zeros(512, dtype=np.float32), 16000), 1
                )
                self.assertEqual(
                    proxy.predict(np.zeros(512, dtype=np.float32), 16000), 2
                )
            finally:
                proxy.stop()
            self.assertFalse(proxy.process.is_alive())

    def test_batch_preserves_three_rob_ids_and_continues_erc_for_other_videos(self):
        class Encoder:
            def encode(self, segments):
                return np.zeros((len(segments), 2), dtype=np.float32)

        class Model:
            def predict(self, features):
                return [
                    EmotionPrediction(segment, "neutral", 1, (0, 0, 1, 0, 0, 0))
                    for segment in features.segments
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = []
            output = root / "results"
            for name, speakers in (("crowded", 3), ("normal", 2)):
                folder = root / name
                folder.mkdir()
                source, capture = self.make_capture(folder)
                video = source.rename(folder / f"{name}.mp4")
                videos.append(video)
                capture.update(
                    rob={"revision": "fixture"}, replay_max_lateness_seconds=0.001
                )
                capture["groups"] = [
                    {
                        "start_unix": 100 + index,
                        "end_unix": 100.8 + index,
                        "speaker": f"speaker{index + 1}",
                        "text": f"turn {index}",
                        "language": "en",
                        "speaker_confidence": 0.7,
                        "transcript_confidence": 0,
                    }
                    for index in range(speakers)
                ]
                target = output / name
                target.mkdir(parents=True)
                (target / "rob_capture.json").write_text(json.dumps(capture))
            args = build_parser().parse_args(
                [
                    "--rob-repo",
                    str(root),
                    "--video",
                    *map(str, videos),
                    "--output-dir",
                    str(output),
                    "--weights",
                    str(root),
                    "--checkpoint",
                    str(root / "checkpoint"),
                    "--import-only",
                ]
            )
            bridge = ERCFeatureBridge(Encoder(), Encoder(), Encoder())
            with (
                patch("aimara.run_rob_native.inspect_checkout"),
                patch(
                    "aimara.rob_native_export.decode_audio",
                    return_value=np.zeros(64000, dtype=np.float32),
                ),
                patch(
                    "aimara.run_pipeline.build_erc_components",
                    return_value=(bridge, Model(), None),
                ),
                self.assertLogs("aimara.run_rob_native", level="ERROR"),
            ):
                report = run(args)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(
                [row["erc_status"] for row in report["videos"]], ["failed", "passed"]
            )
            self.assertIn("3 speaker slots", report["videos"][0]["erc_failure"])
            import csv

            with (output / "rob_native.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["speaker"] for row in rows[:3]],
                ["speaker1", "speaker2", "speaker3"],
            )
            self.assertEqual(
                [row["emotion"] for row in rows], ["", "", "", "neutral", "neutral"]
            )
            self.assertEqual(
                json.loads((output / "crowded/predictions.json").read_text()), []
            )

    def test_slow_vad_does_not_block_audio_delivery_and_drains_in_order(self):
        from types import SimpleNamespace

        entered, release = threading.Event(), threading.Event()
        updates, observations = [], []

        class VAD:
            def predict(self, chunk, rate):
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("test VAD was never released")
                return float(chunk[0])

        delivery = VADDelivery(
            VAD(),
            SimpleNamespace(_vad_callback=lambda message: None),
            SimpleNamespace(update_vad=updates.append),
            observations,
            ReplayLogger(),
        )
        try:
            delivery.push(np.ones(512, dtype=np.float32), 1)
            self.assertTrue(entered.wait(timeout=5))
            # Inference is blocked; audio publisher can still enqueue later chunks.
            delivery.push(np.zeros(512, dtype=np.float32), 2)
            self.assertEqual(observations, [])
        finally:
            release.set()
            delivery.stop(5)
        self.assertEqual(updates, [1.0, 0.0])
        self.assertEqual([row["source_end"] for row in observations], [1, 2])
        self.assertFalse(delivery.thread.is_alive())

    def test_vad_queue_overflow_is_explicit_rather_than_silently_dropping_audio(self):
        from types import SimpleNamespace

        entered, release = threading.Event(), threading.Event()

        class VAD:
            def predict(self, chunk, rate):
                entered.set()
                release.wait(timeout=5)
                return 0.0

        delivery = VADDelivery(
            VAD(),
            SimpleNamespace(_vad_callback=lambda message: None),
            SimpleNamespace(update_vad=lambda value: None),
            [],
            ReplayLogger(),
        )
        try:
            delivery.push(np.zeros(512, dtype=np.float32), 0)
            self.assertTrue(entered.wait(timeout=5))
            for index in range(10):
                delivery.push(np.zeros(512, dtype=np.float32), index + 1)
            with self.assertRaisesRegex(RuntimeError, "queue"):
                delivery.push(np.zeros(512, dtype=np.float32), 11)
        finally:
            release.set()
            delivery.stop(5)

    def test_observer_preserves_publication_and_exports_the_last_retry(self):
        engine = PublishingEngine()
        exported = []
        remove = observe_groups(engine, exported.append)
        engine._group_segments_by_speaker([group(speaker="unknown")], 95, 105)
        final = [group(), group("Reply.", "speaker2", 2, 3)]
        engine.publish(final)
        self.assertEqual([row[0] for row in engine.published], ["Hello.", "Reply."])
        self.assertEqual([row[1] for row in engine.published], ["speaker1", "speaker2"])
        self.assertEqual([row["start_unix"] for row in exported], [100.25, 102])
        self.assertEqual([row["end_unix"] for row in exported], [101.25, 103])
        remove()
        engine.publish([group()])
        self.assertEqual(len(exported), 2)
        self.assertEqual(len(engine.published), 3)

    def test_observer_does_not_mix_concurrent_callers(self):
        engine = PublishingEngine()
        exported = []
        observe_groups(engine, exported.append)
        barrier = threading.Barrier(2)
        errors = []

        def worker(text, clock):
            try:
                engine._group_segments_by_speaker([group(text=text)], clock, clock + 2)
                barrier.wait(timeout=2)
                engine._on_transcript_ready(
                    text, "speaker1", "en", 0, 0, 0, 1000, 0, -1
                )
            except Exception as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=("one", 10)),
            threading.Thread(target=worker, args=("two", 20)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        self.assertEqual(errors, [])
        self.assertEqual(
            {row["text"]: row["start_unix"] for row in exported},
            {"one": 10.25, "two": 20.25},
        )

    def test_observer_rejects_double_install_and_changed_publication(self):
        engine = PublishingEngine()
        observe_groups(engine, lambda row: None)
        with self.assertRaisesRegex(ValueError, "already"):
            observe_groups(engine, lambda row: None)
        engine._group_segments_by_speaker([group()], 100, 110)
        with self.assertRaisesRegex(RuntimeError, "differs"):
            engine._on_transcript_ready(
                "different", "speaker1", "en", 0, 0, 0, 0, 0, -1
            )
        self.assertEqual(engine.published, [])

    def make_capture(self, directory):
        source = directory / "video.mp4"
        source.write_bytes(b"source identity")
        capture = {
            "schema_version": 1,
            "status": "passed",
            "sample_rate": 16000,
            "source_samples": 64000,
            "source_start_unix": 100,
            "source_video_sha256": file_digest(source),
            "asr": {
                "backend": "whisper",
                "model": "turbo",
                "device": "cpu",
                "compute_type": "float32",
            },
            "groups": [],
        }
        return source, capture

    def test_import_preserves_groups_overlaps_and_unknown_and_audits_clipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, capture = self.make_capture(root)
            for text, speaker, start, end in (
                ("second", "unknown", 101, 102.5),
                ("first", "EUT_speaker1", 99.8, 101.5),
            ):
                capture["groups"].append(
                    {
                        "text": text,
                        "speaker": speaker,
                        "language": "en",
                        "speaker_confidence": -1,
                        "transcript_confidence": 0,
                        "start_unix": start,
                        "end_unix": end,
                    }
                )
            path = root / "capture.json"
            path.write_text(json.dumps(capture))
            audio = np.linspace(-0.5, 0.5, 64000, dtype=np.float32)
            with patch("aimara.rob_native_export.decode_audio", return_value=audio):
                segments = ROBExportProcessor(path).process_video(source, root / "out")
            self.assertEqual(
                [item.transcript for item in segments], ["first", "second"]
            )
            self.assertEqual(
                [item.speaker_id for item in segments], ["EUT_speaker1", "UNKNOWN"]
            )
            self.assertEqual(
                [(item.start_time, item.end_time) for item in segments],
                [(0, 1.5), (1, 2.5)],
            )
            self.assertIsNone(segments[0].asr_metadata.transcript_confidence)
            with wave.open(str(segments[0].audio_path)) as wav:
                self.assertEqual(wav.getnframes(), 24000)
                expected = (
                    np.clip(audio[:24000] * 32768, -32768, 32767)
                    .astype("<i2")
                    .tobytes()
                )
                self.assertEqual(wav.readframes(24000), expected)
            bounds = json.loads((root / "out/erc_bounds.json").read_text())
            self.assertTrue(bounds[0]["edge_clipped"])
            self.assertFalse(bounds[1]["edge_clipped"])

    def test_import_rejects_failed_wrong_source_and_out_of_media_captures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, capture = self.make_capture(root)
            path = root / "capture.json"
            processor = ROBExportProcessor(path)
            capture["status"] = "failed"
            path.write_text(json.dumps(capture))
            with self.assertRaisesRegex(ValueError, "successful"):
                processor.process_video(source, root)
            capture["status"] = "passed"
            capture["source_video_sha256"] = "different"
            path.write_text(json.dumps(capture))
            with self.assertRaisesRegex(ValueError, "source video"):
                processor.process_video(source, root)
            capture["source_video_sha256"] = file_digest(source)
            capture["groups"] = [{"start_unix": 110, "end_unix": 111}]
            path.write_text(json.dumps(capture))
            with patch(
                "aimara.rob_native_export.decode_audio",
                return_value=np.zeros(64000, dtype=np.float32),
            ):
                with self.assertRaisesRegex(ValueError, "outside"):
                    processor.process_video(source, root)

    def test_settings_use_checkout_values_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            config = repo / "speech_recognition/config"
            config.mkdir(parents=True)
            for name, values in (
                ("asr", {"model_size": "turbo", "language": "en, es"}),
                ("diarization", {"diarization_backend": "redimnet2"}),
                ("vad", {"model_name": "silero_vad"}),
            ):
                # JSON is also valid YAML.
                (config / f"{name}_params.yaml").write_text(
                    json.dumps({"/**": {"ros__parameters": values}})
                )
            settings = load_settings(repo, {"language": "en"})
            self.assertEqual(settings["asr"], {"model_size": "turbo", "language": "en"})
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_settings(repo, {"invented": 1})

    def test_node_methods_are_loaded_without_importing_ros(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            path = repo / "speech_recognition/speech_recognition/diarization.py"
            path.parent.mkdir(parents=True)
            path.write_text("""import unavailable_ros_dependency
class DiarizationNode:
    def __init__(self):
        self._vad_buffer_size: int = 7
        self._vad_rms_threshold: float = 0.2
    def _vad_callback(self, msg: SomeROSMessage):
        self.seen = msg.value
    def _on_eut_speaker_changed(self, speaker: Optional[str]):
        self.speaker = speaker
""")
            klass, constants = node_callback_class(repo)
            instance = klass()
            instance._on_eut_speaker_changed("speaker3")
            self.assertEqual(instance.speaker, "speaker3")
            self.assertEqual(
                constants, {"_vad_buffer_size": 7, "_vad_rms_threshold": 0.2}
            )


if __name__ == "__main__":
    unittest.main()
