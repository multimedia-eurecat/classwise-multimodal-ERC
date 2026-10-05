import unittest

import numpy as np

from aimara import ConversationFeatures, ERCEmotionModel, ERCModelProfile, SpeechSegment


def make_features(**widths):
    segment = SpeechSegment(
        conversation_id="demo",
        segment_id=0,
        source_video="demo.mp4",
        audio_path="demo.wav",
        start_time=0.0,
        end_time=1.0,
        speaker_id="UNKNOWN",
        transcript="hello",
    )
    return ConversationFeatures(
        segments=(segment,),
        text=np.zeros((1, widths.get("text", 2)), dtype=np.float32),
        audio=np.zeros((1, widths.get("audio", 3)), dtype=np.float32),
        visual=np.zeros((1, widths.get("visual", 4)), dtype=np.float32),
        speaker_mask=np.array([[1, 0]], dtype=np.float32),
    )


class ERCEmotionModelTests(unittest.TestCase):
    def test_validates_feature_widths_before_inference(self):
        profile = ERCModelProfile(text_dim=2, audio_dim=3, visual_dim=4)
        wrapper = ERCEmotionModel(model=None, profile=profile, device="cpu")

        wrapper._validate_features(make_features())
        with self.assertRaisesRegex(ValueError, "text features have width 5"):
            wrapper._validate_features(make_features(text=5))

    def test_validates_checkpoint_architecture(self):
        profile = ERCModelProfile(
            text_dim=2,
            audio_dim=3,
            visual_dim=4,
            hidden_dim=5,
            n_speakers=2,
            emotions=("a", "b"),
        )
        state = {
            "textf_input.weight": np.zeros((5, 2, 1)),
            "acouf_input.weight": np.zeros((5, 3, 1)),
            "visuf_input.weight": np.zeros((5, 4, 1)),
            "speaker_embeddings.weight": np.zeros((3, 5)),
            "all_output_layer.weight": np.zeros((2, 5)),
        }

        ERCEmotionModel._validate_checkpoint(state, profile)
        state["visuf_input.weight"] = np.zeros((5, 8, 1))
        with self.assertRaisesRegex(ValueError, "expected"):
            ERCEmotionModel._validate_checkpoint(state, profile)


if __name__ == "__main__":
    unittest.main()
