"""Run the real ERC transformer with unresolved identities and two known voices."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aimara import ERCEmotionModel, ERCFeatureBridge, ERCModelProfile, SpeechSegment
from aimara.pipeline import AIMARAPipeline


class Encoder:
    def __init__(self, width):
        self.width = width

    def encode(self, segments):
        return np.array(
            [
                [(s.segment_id + i + 1) / 10 for i in range(self.width)]
                for s in segments
            ],
            dtype=np.float32,
        )


def segments_for(labels):
    return [
        SpeechSegment(
            conversation_id="demo",
            segment_id=i,
            source_video="demo.mp4",
            audio_path=f"segment-{i}.wav",
            start_time=float(i),
            end_time=float(i + 1),
            speaker_id=label,
            transcript=f"turn {i}",
        )
        for i, label in enumerate(labels)
    ]


class ERCSpeakerPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        # Tiny real models run more quickly with one CPU thread.
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)

    def setUp(self):
        import torch
        from multimodal.model import Transformer_Based_Model

        self.profile = ERCModelProfile(
            text_dim=2, audio_dim=3, visual_dim=4, hidden_dim=8, n_head=2, dropout=0
        )
        with torch.random.fork_rng():
            torch.manual_seed(7)
            self.model = Transformer_Based_Model(
                "IEMOCAP",
                1,
                2,
                4,
                3,
                2,
                n_classes=6,
                hidden_dim=8,
                n_speakers=2,
                dropout=0,
            ).eval()
        self.wrapper = ERCEmotionModel(self.model, self.profile, "cpu")
        self.bridge = ERCFeatureBridge(Encoder(2), Encoder(3), Encoder(4))

    def test_unknown_turns_have_zero_identity_signal_in_all_nine_transformers(self):
        import torch

        features = self.bridge.build(
            segments_for(("unknown", "speaker1", "speaker2", " UNKNOWN ", "speaker1"))
        )
        captured = {}

        def capture(name):
            def hook(module, inputs):
                captured[name] = (inputs[2].clone(), inputs[3].clone())

            return hook

        names = ("t_t", "a_t", "v_t", "a_a", "t_a", "v_a", "v_v", "t_v", "a_v")
        hooks = [
            getattr(self.model, name).register_forward_pre_hook(capture(name))
            for name in names
        ]
        try:
            predictions = self.wrapper.predict(features)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(predictions), 5)
        self.assertEqual(set(captured), set(names))
        for name, (utterance_mask, speaker_embeddings) in captured.items():
            with self.subTest(transformer=name):
                torch.testing.assert_close(utterance_mask, torch.ones((1, 5)))
                torch.testing.assert_close(
                    speaker_embeddings[0, [0, 3]], torch.zeros((2, 8))
                )
                torch.testing.assert_close(
                    speaker_embeddings[0, [1, 4]],
                    self.model.speaker_embeddings.weight[0].expand(2, -1),
                )
                torch.testing.assert_close(
                    speaker_embeddings[0, 2], self.model.speaker_embeddings.weight[1]
                )
        self.assertEqual(self.model.speaker_embeddings.num_embeddings, 3)
        self.assertEqual(self.model.speaker_embeddings.padding_idx, 2)
        probabilities = np.array([p.probabilities for p in predictions])
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)

    def test_unknown_only_inference_does_not_depend_on_any_speaker_embedding(self):
        import torch

        features = self.bridge.build(segments_for(("unknown", "UNKNOWN")))
        before = self.wrapper.predict(features)
        with torch.no_grad():
            self.model.speaker_embeddings.weight.fill_(1000)
        after = self.wrapper.predict(features)
        self.assertEqual(
            [p.probabilities for p in before], [p.probabilities for p in after]
        )

    def test_all_known_and_legacy_reserved_slot_calls_keep_original_model_behavior(
        self,
    ):
        import torch

        for policy, labels in (
            ("omit_embedding", ("speaker1", "speaker2")),
            ("reserve_slot", ("unknown", "speaker1")),
        ):
            with self.subTest(policy=policy):
                bridge = ERCFeatureBridge(
                    Encoder(2), Encoder(3), Encoder(4), unknown_speaker_policy=policy
                )
                batch = bridge.build(segments_for(labels)).to_torch_batch()
                self.assertNotIn("speaker_known_mask", batch)
                with torch.no_grad():
                    legacy = self.model(**batch)[4]
                    explicit = self.model(
                        **batch, speaker_known_mask=torch.ones((1, 2), dtype=torch.bool)
                    )[4]
                torch.testing.assert_close(legacy, explicit, rtol=0, atol=0)

    def test_checkpoint_reload_and_pipeline_preserve_all_turns_and_unknown_labels(self):
        import torch

        segments = segments_for(("unknown", "speaker1", "speaker2", "UnKnOwN"))

        class SpeechProcessor:
            def process_video(self, video, output):
                return segments

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(self.model.state_dict(), checkpoint)
            reloaded = ERCEmotionModel.from_checkpoint(
                checkpoint, profile=self.profile, device="cpu"
            )
            pipeline = AIMARAPipeline(SpeechProcessor(), self.bridge, reloaded)
            predictions = pipeline.process_video("demo.mp4", directory)
            saved = json.loads((Path(directory) / "predictions.json").read_text())
        self.assertEqual(len(predictions), 4)
        self.assertEqual([p["segment_id"] for p in saved], [0, 1, 2, 3])
        self.assertEqual(
            [p["speaker_id"] for p in saved],
            ["UNKNOWN", "speaker1", "speaker2", "UNKNOWN"],
        )
        self.assertTrue(all(len(p["probabilities"]) == 6 for p in saved))

    def test_model_rejects_malformed_identity_masks(self):
        import torch

        batch = self.bridge.build(
            segments_for(("speaker1", "speaker2"))
        ).to_torch_batch()
        for mask in (torch.ones((1, 1), dtype=torch.bool), torch.ones((1, 2))):
            with (
                self.subTest(shape=mask.shape, dtype=mask.dtype),
                self.assertRaisesRegex(ValueError, "speaker_known_mask"),
            ):
                self.model(**batch, speaker_known_mask=mask)


if __name__ == "__main__":
    unittest.main()
