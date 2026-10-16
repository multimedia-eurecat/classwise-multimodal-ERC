import json
import tempfile
import unittest
from pathlib import Path

from aimara.rob_native_review import activity_timeline, timeline_data, write_review


class NativeReviewTests(unittest.TestCase):
    def test_activity_preserves_backdated_updates_and_stops_without_filling_gaps(self):
        def event(at, speaker, active=True):
            return {"stamp_unix": 100 + at, "speaker": speaker, "active": active}

        rows = activity_timeline(
            [
                event(0, "speaker1"),
                event(2, "speaker1"),
                event(2.5, "speaker2"),
                event(4, "speaker2"),
                event(4, "speaker2", False),
                # A final label arrives later but refers to an earlier start.
                event(2.2, "speaker2"),
                event(4, "speaker2", False),
                event(5, "speaker1"),
                event(6, "speaker1", False),
            ],
            100,
            8,
        )
        self.assertEqual(
            [
                (round(row["start"], 6), round(row["end"], 6), row["speaker"])
                for row in rows
            ],
            [(0, 2.2, "speaker1"), (2.2, 4, "speaker2"), (5, 6, "speaker1")],
        )

    def test_raw_activity_does_not_inherit_final_erc_labels_or_extend_past_last_event(
        self,
    ):
        rows = activity_timeline(
            [
                {"stamp_unix": 99, "speaker": "speaker3", "active": True},
                {"stamp_unix": 102, "speaker": "speaker3", "active": True},
            ],
            100,
            8,
        )
        self.assertEqual(
            [(row["start"], row["end"], row["speaker"]) for row in rows],
            [(0, 2, "speaker3")],
        )
        self.assertEqual(activity_timeline([], 100, 8), [])

    def test_shared_asr_input_is_deduplicated_and_overlapping_inputs_are_retained(self):
        capture = {
            "sample_rate": 16000,
            "source_samples": 128000,
            "source_start_unix": 100,
            "activity": [],
            "groups": [
                {
                    "start_unix": 100.2,
                    "speaker": "speaker1",
                    "text": "one",
                    "input_start_unix": 99.5,
                    "input_end_unix": 104,
                },
                {
                    "start_unix": 102,
                    "speaker": "speaker2",
                    "text": "two",
                    "input_start_unix": 99.5,
                    "input_end_unix": 104,
                },
                {
                    "start_unix": 103.5,
                    "speaker": "unknown",
                    "text": "three",
                    "input_start_unix": 103,
                    "input_end_unix": 109,
                },
            ],
        }
        bounds = [
            {"erc_start": 0.2, "erc_end": 2},
            {"erc_start": 2, "erc_end": 3.8},
            {"erc_start": 3.5, "erc_end": 8},
        ]
        refs = [{"start": 0, "end": 8, "speaker": "0", "text": "reference"}]
        data = timeline_data(capture, bounds, refs)
        self.assertEqual(
            [(row["start"], row["end"]) for row in data["asr_inputs"]], [(0, 4), (3, 8)]
        )
        self.assertEqual(len(data["final_groups"]), 3)
        self.assertEqual(data["final_groups"][2]["speaker"], "UNKNOWN")
        self.assertEqual(data["raw_speakers"], [])
        self.assertEqual(data["reference"][0]["label"], "ref 0")

    def test_portable_review_rebuild_uses_saved_video_and_reference_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            target = output / "demo"
            target.mkdir()
            (target / "source.mp4").write_bytes(b"previously copied video")
            capture = {
                "source_start_unix": 100,
                "sample_rate": 16000,
                "source_samples": 128000,
                "activity": [
                    {"stamp_unix": 100, "speaker": "speaker2", "active": True},
                    {"stamp_unix": 102, "speaker": "speaker2", "active": False},
                ],
                "groups": [
                    {
                        "start_unix": 100,
                        "speaker": "speaker1",
                        "text": '<script>unsafe & "quoted"',
                        "input_start_unix": 99.8,
                        "input_end_unix": 102,
                    }
                ],
            }
            (target / "rob_capture.json").write_text(json.dumps(capture))
            (target / "erc_bounds.json").write_text('[{"erc_start":0,"erc_end":2}]')
            reference = output / "annotations.csv"
            reference.write_text(
                "video_file,start,end,speaker,text,emotion\ndemo.mp4,0,2,0,reference,neutral\n"
            )
            report = {
                "videos": [
                    {
                        "video": "/unavailable/original/demo.mp4",
                        "erc_status": "not_requested",
                        "max_replay_lateness_seconds": 0.001,
                    }
                ]
            }
            write_review(output, report, reference_csv=reference)
            first = (output / "index.html").read_text()
            for label in (
                "Reference turns",
                "ASR input regions",
                "Raw speaker timeline",
                "Final ERC groups",
            ):
                self.assertIn(label, first)
            self.assertIn("&lt;script&gt;unsafe &amp;", first)
            self.assertNotIn("<script>unsafe", first)
            data = json.loads((output / "review_timelines.json").read_text())
            self.assertEqual(data["reference_turns"], 1)
            self.assertEqual(
                data["videos"]["demo.mp4"]["raw_speakers"][0]["speaker"], "speaker2"
            )
            self.assertEqual(
                data["videos"]["demo.mp4"]["final_groups"][0]["speaker"], "speaker1"
            )
            # No dependency on the original CSV/video paths when regenerating.
            reference.unlink()
            write_review(output, report)
            self.assertEqual(first, (output / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
