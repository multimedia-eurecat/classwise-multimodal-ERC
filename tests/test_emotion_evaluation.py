"""Numerical and eligibility checks for transcript-aligned emotion evaluation."""

import unittest

from aimara.evaluate_emotions import (
    evaluate_model,
    load_mapping,
    score_emotions,
    summarize,
)


def row(text, emotion, video="sample.mp4"):
    return dict(video_file=video, text=text, emotion=emotion)


class EmotionScoringTests(unittest.TestCase):
    def setUp(self):
        self.mapping = load_mapping()

    def test_shortest_circular_distance_and_aliases(self):
        mapping = dict(
            self.mapping, angles=dict(self.mapping["angles"], happy=350, excited=10)
        )
        result = score_emotions("joy", "excited", mapping)
        self.assertEqual(result["angular_error_deg"], 20)
        self.assertAlmostEqual(result["score"], 8 / 9)
        self.assertEqual(score_emotions("anger", "angry", mapping)["score"], 1)
        mapping["angles"]["excited"] = 170
        self.assertEqual(score_emotions("happy", "excited", mapping)["score"], 0)

    def test_neutral_has_no_invented_angle(self):
        for reference, prediction, score in (
            ("neutral", "neutral", 1),
            ("happy", "neutral", 0),
            ("neutral", "happy", 0),
        ):
            result = score_emotions(reference, prediction, self.mapping)
            self.assertIsNone(result["angular_error_deg"])
            self.assertIsNone(result["angular_score"])
            self.assertEqual(result["score"], score)

    def test_split_and_merged_segments_weight_targets_equally(self):
        reference = [
            row("red green blue yellow", "happy"),
            row("phone speaker responds", "neutral"),
            row("silver gold", "happy"),
        ]
        prediction = [
            row("red", "happy"),
            row("green blue yellow", "sad"),
            row("phone speaker responds silver gold", "happy"),
        ]
        details, targets, unused = evaluate_model(
            reference,
            prediction,
            model="test",
            supported={"happy"},
            common={"happy"},
            mapping=self.mapping,
        )
        self.assertEqual(len(details), 3)
        self.assertEqual(len(targets), 2)
        first = next(t for t in targets if t["reference_turn"] == 1)
        second = next(t for t in targets if t["reference_turn"] == 3)
        sad_score = score_emotions("happy", "sad", self.mapping)["score"]
        self.assertAlmostEqual(first["score"], (1 + 3 * sad_score) / 4)
        self.assertEqual(second["score"], 1)
        self.assertTrue(second["prediction_spans_phone_response"])
        result = summarize(targets, "test", "supported")
        self.assertAlmostEqual(result["mean_score"], (first["score"] + 1) / 2)
        self.assertEqual(result["matched_segment_pairs"], 3)
        self.assertFalse(unused)

    def test_eligibility_depends_on_reference_and_inventory(self):
        reference = [
            row("first target phrase", "fear"),
            row("phone response words", "neutral"),
            row("last target phrase", "fear"),
        ]
        prediction = [row(r["text"], "angry") for r in reference]
        details, targets, unused = evaluate_model(
            reference,
            prediction,
            model="test",
            supported={"anger"},
            common={"anger"},
            mapping=self.mapping,
        )
        self.assertEqual(len(details), 2)
        self.assertTrue(all(not r["eligible_supported"] for r in targets))
        self.assertEqual(summarize(targets, "test", "supported")["eligible_targets"], 0)
        self.assertEqual(len(unused), 1)

    def test_missing_target_is_retained_with_zero_and_no_angle(self):
        reference = [
            row("alpha beta gamma", "happy"),
            row("phone responds now", "neutral"),
            row("delta epsilon zeta", "happy"),
        ]
        details, targets, _ = evaluate_model(
            reference,
            [],
            model="test",
            supported={"happy"},
            common={"happy"},
            mapping=self.mapping,
        )
        self.assertEqual(len(targets), 2)
        self.assertTrue(
            all(r["score"] == 0 and r["matched_tokens"] == 0 for r in targets)
        )
        self.assertTrue(all(r["angular_score"] is None for r in details))
        self.assertEqual(summarize(targets, "test", "supported")["matched_targets"], 0)

    def test_unknown_label_fails_instead_of_silently_omitting(self):
        with self.assertRaises(ValueError):
            score_emotions("happy", "unrecognized", self.mapping)


if __name__ == "__main__":
    unittest.main()
