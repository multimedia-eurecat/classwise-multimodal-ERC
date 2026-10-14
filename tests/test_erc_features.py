import unittest

import numpy as np

from aimara import ConversationFeatures, ERCFeatureBridge, SpeechSegment


class FakeEncoder:
    def __init__(self, width):
        self.width = width
        self.seen = None

    def encode(self, segments):
        self.seen = tuple(segment.segment_id for segment in segments)
        return np.full((len(segments), self.width), self.width, dtype=np.float64)


def make_segment(segment_id, start_time, speaker_id="UNKNOWN", conversation_id="demo"):
    return SpeechSegment(
        conversation_id=conversation_id,
        segment_id=segment_id,
        source_video="demo.mp4",
        audio_path=f"segment-{segment_id}.wav",
        start_time=start_time,
        end_time=start_time + 1.0,
        speaker_id=speaker_id,
        transcript=f"utterance {segment_id}",
    )


class ERCFeatureBridgeTests(unittest.TestCase):
    def test_orders_segments_and_aligns_all_modalities(self):
        text, audio, visual = FakeEncoder(2), FakeEncoder(3), FakeEncoder(4)
        bridge = ERCFeatureBridge(text, audio, visual)

        features = bridge.build([make_segment(1, 2.0), make_segment(0, 0.0)])

        self.assertEqual([s.segment_id for s in features.segments], [0, 1])
        self.assertEqual(text.seen, (0, 1))
        self.assertEqual(audio.seen, (0, 1))
        self.assertEqual(visual.seen, (0, 1))
        self.assertEqual(features.text.shape, (2, 2))
        self.assertEqual(features.audio.shape, (2, 3))
        self.assertEqual(features.visual.shape, (2, 4))
        self.assertEqual(features.text.dtype, np.float32)
        np.testing.assert_array_equal(features.speaker_mask, [[0, 0], [0, 0]])

    def test_assigns_known_speakers_to_stable_slots(self):
        encoder = FakeEncoder(1)
        bridge = ERCFeatureBridge(encoder, encoder, encoder)

        features = bridge.build(
            [make_segment(0, 0.0, "Alice"), make_segment(1, 1.0, "Bob")]
        )

        np.testing.assert_array_equal(features.speaker_mask, [[1, 0], [0, 1]])

    def test_unknown_turns_preserve_two_known_speaker_slots_and_all_rows(self):
        import torch

        encoder = FakeEncoder(1)
        bridge = ERCFeatureBridge(encoder, encoder, encoder)
        labels = (" unknown ", "Alice", "Bob", "UNKNOWN", "Alice", "Bob")
        features = bridge.build(
            [make_segment(i, float(i), label) for i, label in enumerate(labels)]
        )
        self.assertEqual(encoder.seen, tuple(range(6)))
        self.assertEqual(
            [s.speaker_id for s in features.segments],
            ["UNKNOWN", "Alice", "Bob", "UNKNOWN", "Alice", "Bob"],
        )
        np.testing.assert_array_equal(
            features.speaker_mask, [[0, 0], [1, 0], [0, 1], [0, 0], [1, 0], [0, 1]]
        )
        batch = features.to_torch_batch()
        self.assertEqual(batch["speaker_known_mask"].dtype, torch.bool)
        self.assertEqual(
            batch["speaker_known_mask"].tolist(),
            [[False, True, True, False, True, True]],
        )
        self.assertEqual(batch["u_mask"].tolist(), [[1] * 6])
        self.assertEqual(batch["dia_len"], [6])

    def test_aliases_normalize_before_feature_extraction(self):
        encoder = FakeEncoder(1)
        bridge = ERCFeatureBridge(
            encoder, encoder, encoder, unknown_speaker_id="unresolved"
        )
        features = bridge.build(
            [make_segment(0, 0, " UnResolved "), make_segment(1, 1, "unknown")]
        )
        self.assertEqual(
            [s.speaker_id for s in features.segments], ["UNKNOWN", "UNKNOWN"]
        )
        np.testing.assert_array_equal(features.speaker_mask, [[0, 0], [0, 0]])

    def test_capacity_errors_precede_encoders_under_both_policies(self):
        cases = (
            ("omit_embedding", ("Alice", "Bob", "Carol")),
            ("reserve_slot", ("unknown", "Alice", "Bob")),
        )
        for policy, labels in cases:
            with self.subTest(policy=policy):
                encoder = FakeEncoder(1)
                bridge = ERCFeatureBridge(
                    encoder, encoder, encoder, unknown_speaker_policy=policy
                )
                with self.assertRaisesRegex(
                    ValueError, "requires 3 speaker slots; checkpoint supports 2"
                ):
                    bridge.build(
                        [
                            make_segment(i, float(i), label)
                            for i, label in enumerate(labels)
                        ]
                    )
                self.assertIsNone(encoder.seen)

    def test_reserve_slot_preserves_the_legacy_unknown_policy(self):
        encoder = FakeEncoder(1)
        bridge = ERCFeatureBridge(
            encoder, encoder, encoder, unknown_speaker_policy="reserve_slot"
        )
        features = bridge.build(
            [
                make_segment(0, 0, "Alice"),
                make_segment(1, 1, "unknown"),
                make_segment(2, 2, "Alice"),
            ]
        )
        np.testing.assert_array_equal(features.speaker_mask, [[0, 1], [1, 0], [0, 1]])
        self.assertNotIn("speaker_known_mask", features.to_torch_batch())

    def test_rejects_invalid_speaker_policy(self):
        encoder = FakeEncoder(1)
        with self.assertRaisesRegex(ValueError, "unknown_speaker_policy"):
            ERCFeatureBridge(
                encoder, encoder, encoder, unknown_speaker_policy="assign_to_nearest"
            )

    def test_rejects_invalid_speaker_masks_and_zero_rows_for_known_speakers(self):
        for mask, speaker in (
            ([0.5, 0.5], "Alice"),
            ([-1, 2], "Alice"),
            ([1, 1], "Alice"),
            ([0, 0], "Alice"),
        ):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "speaker"):
                ConversationFeatures(
                    segments=(make_segment(0, 0, speaker),),
                    text=np.zeros((1, 1)),
                    audio=np.zeros((1, 1)),
                    visual=np.zeros((1, 1)),
                    speaker_mask=np.array([mask]),
                )

    def test_rejects_multiple_conversations_and_duplicate_segment_ids(self):
        encoder = FakeEncoder(1)
        bridge = ERCFeatureBridge(encoder, encoder, encoder)

        with self.assertRaisesRegex(ValueError, "one conversation"):
            bridge.build(
                [make_segment(0, 0.0), make_segment(1, 1.0, conversation_id="other")]
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            bridge.build([make_segment(0, 0.0), make_segment(0, 1.0)])

    def test_feature_contract_rejects_misaligned_rows(self):
        segments = (make_segment(0, 0.0),)
        with self.assertRaisesRegex(ValueError, "audio has 2 rows"):
            ConversationFeatures(
                segments=segments,
                text=np.zeros((1, 2)),
                audio=np.zeros((2, 2)),
                visual=np.zeros((1, 2)),
                speaker_mask=np.ones((1, 2)),
            )

    def test_builds_erc_tensor_layout(self):
        encoder = FakeEncoder(3)
        features = ERCFeatureBridge(encoder, encoder, encoder).build(
            [make_segment(0, 0.0), make_segment(1, 1.0)]
        )

        batch = features.to_torch_batch()

        self.assertEqual(tuple(batch["textf"].shape), (2, 1, 3))
        self.assertEqual(tuple(batch["acouf"].shape), (2, 1, 3))
        self.assertEqual(tuple(batch["visuf"].shape), (2, 1, 3))
        self.assertEqual(tuple(batch["qmask"].shape), (1, 2, 2))
        self.assertEqual(tuple(batch["u_mask"].shape), (1, 2))
        self.assertEqual(batch["dia_len"], [2])
        self.assertEqual(batch["speaker_known_mask"].tolist(), [[False, False]])


if __name__ == "__main__":
    unittest.main()
