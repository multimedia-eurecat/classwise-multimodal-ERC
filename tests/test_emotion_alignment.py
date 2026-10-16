"""Transcript alignment must survive segment errors without using emotion labels."""

import unittest

from aimara.emotion_alignment import align_video, normalized_tokens


def rows(*texts, **extra):
    return [{"text": text, "video_file": "example.mp4", **extra} for text in texts]


class EmotionAlignmentTests(unittest.TestCase):
    def test_split_target_contributes_each_segment_and_excludes_phone(self):
        reference = rows(
            "This is completely unacceptable and I will not tolerate it.",
            "Wait, I think you are misunderstanding.",
            "There is nothing to misunderstand. Fix it!",
        )
        prediction = rows(
            "This is completely unacceptable",
            "and I won't tolerate it.",
            "Wait, I think you're misunderstanding.",
            "There is nothing to misunderstand. Fix it!",
        )
        result = align_video(reference, prediction)
        self.assertEqual([(r["reference_turn"], r["prediction_segment"]) for r in result], [(1, 1), (1, 2), (3, 4)])
        self.assertTrue(all(r["reference_coverage"] == 1 for r in result))
        self.assertAlmostEqual(sum(r["segment_reference_coverage"] for r in result[:2]), 1)

    def test_all_turns_merged_into_one_prediction_have_two_fragment_records(self):
        reference = rows("We finally approved the project.", "We got funding too.", "That is incredible news!")
        prediction = rows("We finally approved the project! We got funding too! That is incredible news!")
        result = align_video(reference, prediction)
        self.assertEqual(len(result), 2)
        self.assertEqual([r["prediction_segment"] for r in result], [1, 1])
        self.assertTrue(all(r["alignment_status"] == "matched" for r in result))
        self.assertTrue(all(r["prediction_spans_phone_response"] for r in result))
        self.assertTrue(all(r["prediction_reference_turns"] == [1, 2, 3] for r in result))
        self.assertEqual(result[0]["matched_prediction_text"], "We finally approved the project")
        self.assertEqual(result[1]["matched_prediction_text"], "That is incredible news")

    def test_target_fragments_cross_prediction_and_phone_boundaries(self):
        reference = rows("alpha bravo charlie delta", "echo foxtrot", "golf hotel india juliet")
        prediction = rows("alpha bravo", "charlie delta echo", "foxtrot golf hotel", "india juliet")
        result = align_video(reference, prediction)
        self.assertEqual([(r["reference_turn"], r["prediction_segment"]) for r in result], [(1, 1), (1, 2), (3, 3), (3, 4)])
        self.assertEqual([r["prediction_spans_phone_response"] for r in result], [False, True, True, False])
        self.assertTrue(all(r["reference_coverage"] == 1 for r in result))
        self.assertEqual(result[2]["prediction_token_start"], 1)
        self.assertEqual(result[2]["prediction_token_end"], 3)

    def test_punctuation_unicode_contractions_and_numbers_normalize(self):
        reference = rows("I won’t do it. It's due between two and four!", "You're sure?", "All right, I've done it.")
        prediction = rows("I will not do it; it is due between 2 & 4.", "You are sure", "Alright I have done it")
        result = align_video(reference, prediction)
        # '&' is punctuation rather than a lexical 'and'; the missing word is
        # visible as partial coverage instead of being silently fabricated.
        self.assertEqual(result[0]["alignment_status"], "partial")
        self.assertEqual(result[1]["alignment_status"], "matched")
        self.assertEqual(normalized_tokens("won’t, can't; cannot; two 4"), ["will", "not", "can", "not", "can", "not", "2", "4"])
        exact_numbers = rows("I will not do it. It is due between 2 and 4.", "You are sure", "All right I have done it")
        self.assertTrue(all(r["reference_coverage"] == 1 for r in align_video(reference, exact_numbers)))

    def test_matching_ignores_timing_speaker_and_emotion(self):
        reference = rows("I feel delighted today", "Please tell me more", "Wonderful news indeed", start=50, end=0, speaker="wrong", emotion="anger")
        prediction = rows("I feel delighted today", "Please tell me more", "Wonderful news indeed", start=-1, end=9999, speaker="same", emotion="sad")
        result = align_video(reference, prediction)
        self.assertTrue(all(r["alignment_status"] == "matched" for r in result))

    def test_missing_target_does_not_borrow_phone_response(self):
        reference = rows("The project is wonderful", "Thanks for calling today", "We should celebrate tomorrow")
        result = align_video(reference, rows("Thanks for calling today"))
        self.assertEqual([r["alignment_status"] for r in result], ["unmatched", "unmatched"])
        self.assertTrue(all(r["prediction_segment"] is None for r in result))

    def test_hallucinated_extra_segment_is_not_a_target_overlap(self):
        reference = rows("We finally approved the project", "We got funding too", "That is incredible news")
        prediction = rows("We finally approved the project", "We got funding too", "That is incredible news", "Thanks for watching and please subscribe")
        result = align_video(reference, prediction)
        self.assertEqual([r["prediction_segment"] for r in result], [1, 3])

    def test_unrelated_speech_with_shared_function_words_is_rejected(self):
        reference = rows("It is wonderful that we finally approved the project", "A clear phone response", "Another beautiful summer morning")
        prediction = rows("It is time for the weather report", "A clear phone response", "An unrelated advertisement about cars")
        result = align_video(reference, prediction)
        self.assertEqual([r["alignment_status"] for r in result], ["unmatched", "unmatched"])

    def test_one_shared_function_word_does_not_match_a_short_target(self):
        result = align_video(rows("It was", "Please wait", "That is"), rows("It happens", "Please wait", "That never occurred"))
        self.assertEqual([r["alignment_status"] for r in result], ["unmatched", "unmatched"])

    def test_fuzzy_spelling_match_is_supported_but_confidence_is_lower(self):
        reference = rows("It was horrendous", "Please explain", "This is unacceptable")
        prediction = rows("It was horrendeous", "Please explain", "This is unacceptable")
        result = align_video(reference, prediction)
        self.assertEqual(result[0]["matched_tokens"], 3)
        self.assertEqual(result[0]["reference_coverage"], 1)
        self.assertLess(result[0]["text_similarity"], 1)
        self.assertGreater(result[0]["text_similarity"], 0.9)

    def test_missing_word_retains_partial_coverage(self):
        result = align_video(rows("It was so wonderful", "Tell me more", "Oh yes absolutely"), rows("It was so wonderful", "Tell me more", "Yes absolutely"))
        self.assertEqual(result[1]["alignment_status"], "partial")
        self.assertAlmostEqual(result[1]["reference_coverage"], 2 / 3)

    def test_single_word_fragments_are_kept_when_contiguous_with_other_matches(self):
        result = align_video(rows("I really hate this", "Please calm down", "That sounds reasonable"), rows("I", "really hate", "this", "Please calm down", "That sounds reasonable"))
        self.assertEqual([r["prediction_segment"] for r in result], [1, 2, 3, 5])
        self.assertEqual(result[0]["reference_coverage"], 1)

    def test_identical_repeated_targets_with_missing_context_are_ambiguous(self):
        result = align_video(rows("Yes absolutely", "Please wait", "Yes absolutely"), rows("Yes absolutely"))
        assigned = [r for r in result if r["prediction_segment"] is not None]
        self.assertEqual(len(assigned), 1)
        self.assertEqual(assigned[0]["alignment_status"], "ambiguous")
        self.assertIn("alternative_reference_turns", assigned[0]["ambiguity_reasons"])

    def test_repetition_is_resolved_by_complete_monotonic_context(self):
        reference = rows("Yes absolutely", "Please wait", "Yes absolutely")
        result = align_video(reference, rows("Yes absolutely", "Please wait", "Yes absolutely"))
        self.assertEqual([r["prediction_segment"] for r in result], [1, 3])
        self.assertTrue(all(r["alignment_status"] == "matched" for r in result))

    def test_empty_prediction_produces_two_unmatched_rows(self):
        result = align_video(rows("first target", "phone response", "final target"), [])
        self.assertEqual([r["reference_turn"] for r in result], [1, 3])
        self.assertTrue(all(r["matched_tokens"] == 0 for r in result))

    def test_single_word_target_and_original_character_spans(self):
        reference = rows("No!", "Are you sure?", "Yes.")
        result = align_video(reference, rows("NO", "Are you sure", "yes"))
        self.assertTrue(all(r["alignment_status"] == "matched" for r in result))
        for record in result:
            self.assertEqual(record["reference_text"][record["reference_char_start"] : record["reference_char_end"]], record["matched_reference_text"])

    def test_different_videos_are_rejected_before_matching(self):
        reference = rows("first target", "phone response", "final target")
        prediction = [{"video_file": "wrong.mp4", "text": "first target"}]
        with self.assertRaisesRegex(ValueError, "filename"):
            align_video(reference, prediction)

    def test_video_basename_comparison_accepts_paths_and_case(self):
        reference = rows("first target", "phone response", "final target")
        prediction = rows("first target", "phone response", "final target", video_file="/tmp/EXAMPLE.mp4")
        self.assertTrue(all(r["alignment_status"] == "matched" for r in align_video(reference, prediction)))


if __name__ == "__main__":
    unittest.main()
