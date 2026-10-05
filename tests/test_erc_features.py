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
        np.testing.assert_array_equal(features.speaker_mask, [[1, 0], [1, 0]])

    def test_assigns_known_speakers_to_stable_slots(self):
        encoder = FakeEncoder(1)
        bridge = ERCFeatureBridge(encoder, encoder, encoder)

        features = bridge.build(
            [make_segment(0, 0.0, "Alice"), make_segment(1, 1.0, "Bob")]
        )

        np.testing.assert_array_equal(features.speaker_mask, [[1, 0], [0, 1]])

    def test_rejects_multiple_conversations_and_duplicate_segment_ids(self):
        encoder = FakeEncoder(1)
        bridge = ERCFeatureBridge(encoder, encoder, encoder)

        with self.assertRaisesRegex(ValueError, "one conversation"):
            bridge.build([make_segment(0, 0.0), make_segment(1, 1.0, conversation_id="other")])
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


if __name__ == "__main__":
    unittest.main()
