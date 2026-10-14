"""Known metric examples and provenance failures for annotated-turn scoring."""

import json
import tempfile
import unittest
from pathlib import Path

from aimara.batch_resume import file_digest
from aimara.evaluate_quality import (
    DEFAULT_MAPPING,
    aggregate,
    boundary_score,
    classification,
    evaluate_reports,
    evaluate_video,
    speaker_score,
    tokens,
    word_errors,
)
from aimara.prepare_manifest import ReferenceUtterance


def interval(start, end, speaker):
    return dict(start=start, end=end, speaker=speaker)


class QualityEvaluationTests(unittest.TestCase):
    def test_wer_normalization_and_insertions_deletions_substitutions(self):
        self.assertEqual(tokens("I WON’T, foo_bar!"), ["i", "won't", "foo", "bar"])
        self.assertEqual(word_errors("A b c", "A b c")["errors"], 0)
        self.assertEqual(word_errors("A b c", "A wrong c")["substitutions"], 1)
        self.assertEqual(word_errors("A b c", "A c")["deletions"], 1)
        self.assertEqual(word_errors("A b", "A b c")["insertions"], 1)
        self.assertEqual(word_errors("A", "b c d")["wer"], 3)
        self.assertIsNone(word_errors("", "speech")["wer"])

    def test_speakers_optimal_mapping_never_rewards_unknown_or_duplicate_voice(self):
        references = [interval(0, 2, "A"), interval(2, 4, "B"), interval(4, 6, "A")]
        swapped = [interval(0, 2, "two"), interval(2, 4, "one"), interval(4, 6, "two")]
        self.assertEqual(speaker_score(references, swapped)["accuracy"], 1)
        degraded = [
            interval(0, 2, "two"),
            interval(2, 3, "UNKNOWN"),
            interval(4, 6, "two"),
        ]
        scores = speaker_score(references, degraded)
        self.assertEqual(scores["correct_seconds"], 4)
        self.assertEqual(scores["unknown_seconds"], 1)
        self.assertEqual(scores["uncovered_seconds"], 1)
        self.assertEqual(
            speaker_score(references, [interval(0, 6, "same")])["confusion_seconds"], 2
        )

    def test_boundary_matching_is_one_to_one_and_maximizes_recovery(self):
        references = [interval(0, 1, "A"), interval(1, 1.4, "B"), interval(1.4, 3, "A")]
        predicted = [
            interval(0, 0.8, "x"),
            interval(0.8, 1.15, "y"),
            interval(1.15, 3, "x"),
        ]
        scores = boundary_score(references, predicted, 0.3)
        self.assertEqual(scores["matched"], 2)
        only_one = [interval(0, 1.2, "x"), interval(1.2, 3, "y")]
        self.assertEqual(boundary_score(references, only_one, 0.3)["matched"], 1)
        self.assertIsNone(boundary_score(references, [], 0.3)["precision"])

    def test_emotion_missing_predictions_count_and_exclusions_are_explicit(self):
        rows = [
            dict(target="happy", predicted="happy", reference_emotion="happy"),
            dict(target="sad", predicted=None, reference_emotion="sadness"),
            dict(target=None, predicted="happy", reference_emotion="joy"),
        ]
        score = classification(rows)
        self.assertEqual(score["accuracy"], 0.5)
        self.assertEqual(score["macro_f1"], 0.5)
        self.assertEqual(score["excluded_labels"], {"joy": 1})
        self.assertEqual(score["confusion"]["sad"], {"NO_PREDICTION": 1})

    def test_duration_weighted_turn_scoring_and_micro_wer(self):
        references = [
            ReferenceUtterance("demo.mp4", 0, 3, "A", "one two", "happy", 2),
            ReferenceUtterance("demo.mp4", 3, 4, "B", "three", "fear", 3),
        ]

        def prediction(start, end, label, text):
            return dict(
                start_time=start,
                end_time=end,
                speaker_id="UNKNOWN",
                transcript=text,
                probabilities={
                    emotion: int(emotion == label)
                    for emotion in (
                        "happy",
                        "sad",
                        "neutral",
                        "angry",
                        "excited",
                        "frustrated",
                    )
                },
            )

        video = dict(
            video_file="demo.mp4",
            source_sha256="abc",
            speaker_timeline=[],
            predictions=[
                prediction(0, 1, "sad", "one"),
                prediction(1, 3, "happy", "two"),
            ],
            asr_calls=[dict(text="one two")],
        )
        scored = evaluate_video(video, references, DEFAULT_MAPPING, 0.5)
        self.assertEqual(scored["emotion_turns"][0]["predicted"], "happy")
        self.assertIsNone(scored["emotion_turns"][1]["target"])
        combined = aggregate([scored, scored])
        self.assertEqual(combined["asr"]["wer"], 1 / 3)
        self.assertEqual(combined["emotion"]["scored_turns"], 2)
        video["predictions"][0]["end_time"] = 2
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            evaluate_video(video, references, DEFAULT_MAPPING, 0.5)

    def test_failed_incomplete_and_changed_source_reports_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.csv"
            reference.write_text(
                "video_file,start,end,speaker,text,emotion\ndemo.mp4,0,1,A,word,happy\n"
            )
            video = dict(
                video_file="demo.mp4",
                source_sha256="one",
                status="passed",
                predictions=[],
                speaker_timeline=[],
                asr_calls=[],
            )
            report = dict(
                passed=True,
                status="passed",
                reference_sha256=file_digest(reference),
                selected_videos=["demo.mp4"],
                videos=[video],
                configuration={},
                versions={},
            )
            paths = []
            for name in ("left", "right"):
                (root / name).mkdir()
                path = root / name / "smoke_report.json"
                path.write_text(json.dumps(report))
                paths.append(path)
            result = evaluate_reports(paths, reference, DEFAULT_MAPPING, 0.5)
            self.assertEqual(
                result["paired_comparisons"][0]["common_videos"], ["demo.mp4"]
            )
            for key, value in [
                ("passed", False),
                ("reference_sha256", "changed"),
                ("selected_videos", ["missing.mp4"]),
            ]:
                changed = dict(report, **{key: value})
                paths[1].write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    evaluate_reports(paths, reference, DEFAULT_MAPPING, 0.5)
            changed = dict(report, videos=[dict(video, source_sha256="two")])
            paths[1].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Source checksum"):
                evaluate_reports(paths, reference, DEFAULT_MAPPING, 0.5)
            with self.assertRaises(ValueError):
                evaluate_reports(paths[:1], reference, DEFAULT_MAPPING, float("nan"))
