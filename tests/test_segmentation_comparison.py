"""Controlled evidence attribution for the saved-observation segmentation audit."""

import unittest
from dataclasses import asdict

from aimara.compare_segmentation import (
    align,
    align_reference,
    audit_video,
    regions_for_calls,
    replay,
)
from aimara.diarization import SpeakerInterval
from aimara.prepare_manifest import ReferenceUtterance
from aimara.speaker_segmentation import SpeakerSegmentationConfig, SpeakerSegmenter


def call(segments, duration):
    return dict(
        input_duration=duration,
        language="en",
        text=" ".join(s["text"] for s in segments),
        segments=segments,
    )


def segment(words):
    return dict(
        start_time=words[0][0],
        end_time=words[-1][1],
        text="".join(w[2] for w in words).strip(),
        words=[dict(start_time=a, end_time=b, text=t) for a, b, t in words],
    )


def fixture(calls, regions, timeline):
    video = dict(
        video_file="demo.mp4",
        asr_calls=calls,
        speaker_timeline=[asdict(p) for p in timeline],
    )
    groups = replay(video, regions, SpeakerSegmenter(), timeline)
    for group in groups:
        start, end = next((a, b) for a, b in regions if a <= group["start_time"] < b)
        group.update(asr_input_start_time=start, asr_input_end_time=end)
    video["predictions"] = groups
    return video


class SegmentationComparisonTests(unittest.TestCase):
    def test_edit_alignment_and_numeric_alias_do_not_shift_a_turn_boundary(self):
        self.assertEqual(
            align(["a", "b"], ["a", "x", "b"]),
            [(0, 0, "exact"), (None, 1, "insertion"), (1, 2, "exact")],
        )
        reference = ["and", "four", "all", "right", "i'll"]
        hypothesis = ["and", "4", "alright", "i'll"]
        result = align_reference(reference, hypothesis)
        self.assertIn((1, 1, "numeric_equivalent"), result)
        self.assertIn((4, 3, "exact"), result)

    def test_exact_saved_replay_is_required_before_any_diagnosis(self):
        timeline = [SpeakerInterval(0, 32000, "speaker1")]
        video = fixture([call([segment([(0, 0.4, " Hello.")])], 2)], [(0, 2)], timeline)
        video["predictions"][0]["speaker_id"] = "incorrect_saved_label"
        references = [ReferenceUtterance("demo.mp4", 0, 2, "0", "Hello.", "neutral", 2)]
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            audit_video(video, references, SpeakerSegmentationConfig())

    def test_flat_diarization_loses_a_change_already_present_in_asr_segments(self):
        timeline = [SpeakerInterval(0, 32000, "speaker1")]
        calls = [
            call(
                [
                    segment([(0, 0.3, " Hello"), (0.3, 0.8, " there.")]),
                    segment([(1.2, 1.5, " Goodbye"), (1.5, 1.8, " now.")]),
                ],
                2,
            )
        ]
        video = fixture(calls, [(0, 2)], timeline)
        refs = [
            ReferenceUtterance("demo.mp4", 0, 1, "0", "Hello there.", "neutral", 2),
            ReferenceUtterance("demo.mp4", 1, 2, "1", "Goodbye now.", "neutral", 3),
        ]
        result = audit_video(video, refs, SpeakerSegmentationConfig())
        boundary = result["boundaries"][0]
        self.assertTrue(boundary["asr_backend_segments_separate"])
        self.assertFalse(boundary["actual_separates"])
        self.assertTrue(boundary["reference_timeline_separates"])
        self.assertEqual(boundary["diagnosis"], "raw_diarization_merges_boundary_words")

    def test_only_removing_trailing_override_restores_the_controlled_split(self):
        timeline = [
            SpeakerInterval(0, 10400, "speaker1"),
            SpeakerInterval(10400, 32000, "speaker2"),
        ]
        calls = [
            call(
                [
                    segment(
                        [
                            (0, 0.2, " I'm"),
                            (0.2, 0.6, " sorry,"),
                            (0.7, 0.95, " your"),
                            (0.95, 1.2, " loss."),
                        ]
                    ),
                    segment([(1.3, 1.6, " Thank"), (1.6, 1.9, " you.")]),
                ],
                2,
            )
        ]
        video = fixture(calls, [(0, 2)], timeline)
        refs = [
            ReferenceUtterance(
                "demo.mp4", 0, 1.25, "0", "I'm sorry, your loss.", "sadness", 2
            ),
            ReferenceUtterance("demo.mp4", 1.25, 2, "1", "Thank you.", "sadness", 3),
        ]
        result = audit_video(video, refs, SpeakerSegmentationConfig())
        boundary = result["boundaries"][0]
        self.assertFalse(boundary["actual_separates"])
        self.assertTrue(boundary["majority_only_separates"])
        self.assertEqual(
            boundary["diagnosis"], "grouping_trailing_override_removes_split"
        )
        self.assertEqual(len(result["variants"]["actual"]), 1)
        self.assertEqual(len(result["variants"]["majority_only"]), 2)
        self.assertEqual(result["words"][0]["actual_speaker"], "speaker2")

    def test_back_to_back_reference_in_a_vad_gap_is_not_a_missing_text_split(self):
        timeline = [
            SpeakerInterval(0, 11200, "speaker1"),
            SpeakerInterval(20800, 32000, "speaker2"),
        ]
        calls = [
            call([segment([(0, 0.5, " Hello.")])], 0.7),
            call([segment([(0, 0.5, " Goodbye.")])], 0.7),
        ]
        video = fixture(calls, [(0, 0.7), (1.3, 2)], timeline)
        refs = [
            ReferenceUtterance("demo.mp4", 0, 1, "0", "Hello.", "neutral", 2),
            ReferenceUtterance("demo.mp4", 1, 2, "1", "Goodbye.", "neutral", 3),
        ]
        result = audit_video(video, refs, SpeakerSegmentationConfig())
        self.assertTrue(result["boundaries"][0]["actual_separates"])
        self.assertTrue(result["boundaries"][0]["reference_inside_output_gap"])
        self.assertAlmostEqual(
            result["turns"][0]["reference_minus_vad_coverage_seconds"], 0.3
        )

    def test_ambiguous_or_inconsistent_asr_input_mapping_is_rejected(self):
        video = dict(
            predictions=[dict(asr_input_start_time=0, asr_input_end_time=2)],
            asr_calls=[dict(input_duration=3)],
        )
        with self.assertRaisesRegex(ValueError, "duration"):
            regions_for_calls(video)
        video["asr_calls"] = []
        with self.assertRaisesRegex(ValueError, "associate"):
            regions_for_calls(video)
