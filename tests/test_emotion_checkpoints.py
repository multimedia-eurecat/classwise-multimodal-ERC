"""Contracts that matter when sharing saved speech inputs across ERC datasets."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aimara.erc_model import ERCModelProfile
from aimara.run_emotion_checkpoints import (
    MELD_EMOTIONS,
    expand_speaker_mask,
    load_inputs,
    serialize_prediction,
)


class EmotionCheckpointTests(unittest.TestCase):
    def test_meld_serialization_keeps_seventh_class_and_actual_names(self):
        probabilities = (0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.75)
        prediction = SimpleNamespace(
            probabilities=probabilities,
            to_dict=lambda: {"emotion": "anger", "probabilities": {"happy": 0.01}},
        )
        profile = ERCModelProfile(dataset="MELD", emotions=MELD_EMOTIONS, n_speakers=9)
        result = serialize_prediction(prediction, profile)
        self.assertEqual(list(result["probabilities"]), list(MELD_EMOTIONS))
        self.assertEqual(result["probabilities"]["anger"], 0.75)

    def test_speaker_width_expansion_preserves_known_and_unknown_assignments(self):
        original = np.asarray([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
        expanded = expand_speaker_mask(original, 9)
        self.assertEqual(expanded.shape, (3, 9))
        np.testing.assert_array_equal(expanded[:, :2], original)
        np.testing.assert_array_equal(expanded[:, 2:], 0)
        with self.assertRaises(ValueError):
            expand_speaker_mask(original, 1)

    def test_csv_json_mismatch_is_rejected_before_feature_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "Example").mkdir()
            row = {"video_file": "Example.mp4", "start": 0, "end": 1,
                   "speaker": "speaker1", "text": "actual supplied transcript", "emotion": "happy"}
            with (directory / "rob_native.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
            prediction = {"segment_id": 0, "start_time": 0, "end_time": 1,
                          "speaker_id": "speaker1", "transcript": "different transcript", "emotion": "happy"}
            (directory / "Example" / "predictions.json").write_text(json.dumps([prediction]))
            with self.assertRaisesRegex(ValueError, "CSV/JSON segment differs"):
                load_inputs(directory)


if __name__ == "__main__":
    unittest.main()
