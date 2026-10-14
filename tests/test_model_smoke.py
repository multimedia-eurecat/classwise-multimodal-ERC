"""Validate the gates used by real-model smoke runs without pretrained downloads."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aimara import EmotionPrediction, SpeakerInterval, SpeechSegment, write_wav
from aimara.run_model_smoke import (
    speaker_diagnostics,
    state_matches,
    validate_predictions,
    validate_transcript,
)


class ModelSmokeTests(unittest.TestCase):
    def test_word_timestamp_gate_accepts_small_frame_tolerance_and_rejects_wrong_scale(
        self,
    ):
        def transcript(words, text="speech"):
            return SimpleNamespace(
                text=text,
                words=[SimpleNamespace(start_time=a, end_time=b) for a, b in words],
            )

        self.assertEqual(validate_transcript(transcript([(0, 1.08)]), 1), [])
        for words in ([(0, 8)], [(0.5, 0.8), (0.1, 0.3)], [(0, float("nan"))]):
            with self.subTest(words=words):
                self.assertTrue(validate_transcript(transcript(words), 1))
        self.assertTrue(validate_transcript(transcript([]), 1))
        self.assertEqual(validate_transcript(transcript([], text=""), 1), [])

    def test_source_audio_and_probability_gate_detects_changed_wav_and_invalid_scores(
        self,
    ):
        audio = np.array([0.25, -0.25, 0.5, -0.5], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = write_wav(Path(directory) / "clip.wav", audio)
            segment = SpeechSegment(
                conversation_id="demo",
                segment_id=0,
                source_video="demo.mp4",
                audio_path=path,
                start_time=0,
                end_time=len(audio) / 16000,
                speaker_id="UNKNOWN",
                transcript="speech",
            )
            prediction = EmotionPrediction(segment, "neutral", 1, (0, 0, 1, 0, 0, 0))
            self.assertEqual(validate_predictions([prediction], audio), [])
            invalid = EmotionPrediction(
                segment, "neutral", 1, (0, 0, float("nan"), 0, 0, 0)
            )
            self.assertTrue(
                any(
                    "probabilities" in issue
                    for issue in validate_predictions([invalid], audio)
                )
            )
            unnormalized = EmotionPrediction(
                segment, "neutral", 0.999998, (0, 0, 0.999998, 0, 0, 0)
            )
            self.assertTrue(
                any(
                    "probabilities" in issue
                    for issue in validate_predictions([unnormalized], audio)
                )
            )
            write_wav(path, -audio)
            self.assertTrue(
                any(
                    "content" in issue
                    for issue in validate_predictions([prediction], audio)
                )
            )

    def test_returning_speaker_diagnostic_uses_label_permutation_and_reports_unknown(
        self,
    ):
        references = [
            SimpleNamespace(start=i, end=i + 1, speaker=s)
            for i, s in enumerate(("0", "1", "0"))
        ]
        timeline = [
            SpeakerInterval(i * 16000, (i + 1) * 16000, s)
            for i, s in enumerate(("speaker2", "speaker1", "speaker2"))
        ]
        self.assertTrue(
            speaker_diagnostics(timeline, references)["a_b_a_pattern_recovered"]
        )
        timeline[1] = SpeakerInterval(16000, 32000, "UNKNOWN")
        self.assertFalse(
            speaker_diagnostics(timeline, references)["a_b_a_pattern_recovered"]
        )
        self.assertIsNone(
            speaker_diagnostics([], references)["a_b_a_pattern_recovered"]
        )

    def test_state_reset_compares_turns_and_allows_only_small_probability_noise(self):
        row = {
            "segment_id": 0,
            "speaker_id": "speaker1",
            "transcript": "speech",
            "language": "en",
            "emotion": "neutral",
            "start_time": 0,
            "end_time": 1,
            "probabilities": {"neutral": 1},
        }
        self.assertTrue(state_matches([row], [dict(row)]))
        self.assertTrue(
            state_matches([row], [dict(row, probabilities={"neutral": 1 - 1e-7})])
        )
        self.assertFalse(state_matches([row], [dict(row, speaker_id="speaker2")]))
        self.assertFalse(
            state_matches([row], [dict(row, probabilities={"neutral": 0.5})])
        )
        self.assertFalse(state_matches([row], []))
