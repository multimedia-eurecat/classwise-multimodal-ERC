import unittest

from aimara.domain import SpeechSegment
from aimara.erc_encoders import _contextual_transcripts


def make_segment(segment_id, speaker, transcript):
    return SpeechSegment(
        conversation_id="demo",
        segment_id=segment_id,
        source_video="demo.mp4",
        audio_path=f"segment-{segment_id}.wav",
        start_time=float(segment_id),
        end_time=float(segment_id + 1),
        speaker_id=speaker,
        transcript=transcript,
    )


class ContextualTranscriptTests(unittest.TestCase):
    def test_builds_causal_context_without_emotion_labels(self):
        segments = [
            make_segment(0, "UNKNOWN", "hello"),
            make_segment(1, "UNKNOWN", "how are you"),
            make_segment(2, "UNKNOWN", "fine"),
        ]

        texts = _contextual_transcripts(
            segments,
            context_window=1,
            context_sep=" [SEP] ",
            include_speaker=True,
            task_prefix="",
        )

        self.assertEqual(texts[0], "UNKNOWN: hello")
        self.assertEqual(texts[1], "UNKNOWN: hello [SEP] UNKNOWN: how are you")
        self.assertEqual(texts[2], "UNKNOWN: how are you [SEP] UNKNOWN: fine")


if __name__ == "__main__":
    unittest.main()
