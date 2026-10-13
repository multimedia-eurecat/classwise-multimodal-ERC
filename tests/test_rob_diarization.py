import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from aimara import SpeechRegion
from aimara.rob import ROBSpeakerDiarizer


SAMPLE_RATE = 16000


def recording(turns):
    chunks, regions = [], []
    offset = 0
    for speaker, seconds in turns:
        count = round(seconds * SAMPLE_RATE)
        chunks.append(np.full(count, speaker, dtype=np.float32))
        regions.append(SpeechRegion(offset, offset + count))
        offset += count
    return np.concatenate(chunks), regions


def speaker_embedding(audio):
    # Two independent synthetic voices; the real ROB identity manager is used.
    return np.array([1, 0] if audio[0] == 1 else [0, 1], dtype=np.float32)


class ROBSpeakerDiarizerTests(unittest.TestCase):
    def test_alternating_speakers_keep_ids_and_exact_source_boundaries(self):
        audio, regions = recording([(1, 2), (2, 2), (1, 2), (2, 2)])
        timeline = ROBSpeakerDiarizer(embedder=speaker_embedding).diarize(
            audio, SAMPLE_RATE, regions
        )
        self.assertEqual(
            [s.speaker_id for s in timeline],
            ["speaker1", "speaker2", "speaker1", "speaker2"],
        )
        self.assertEqual(
            [(s.start_sample, s.end_sample) for s in timeline],
            [(r.start_sample, r.end_sample) for r in regions],
        )
        self.assertEqual(timeline[0].confidence, -1.0)
        self.assertAlmostEqual(timeline[2].confidence, 1.0)

    def test_short_turn_can_match_a_voice_established_later_without_new_identity(self):
        audio, regions = recording([(1, 1), (2, 2), (1, 2)])
        timeline = ROBSpeakerDiarizer(embedder=speaker_embedding).diarize(
            audio, SAMPLE_RATE, regions
        )
        self.assertEqual(timeline[0].speaker_id, timeline[2].speaker_id)
        self.assertNotEqual(timeline[0].speaker_id, timeline[1].speaker_id)
        self.assertNotEqual(timeline[0].speaker_id, "UNKNOWN")

    def test_too_short_or_silent_turns_do_not_create_voices(self):
        embedder = Mock(side_effect=speaker_embedding)
        audio, regions = recording([(1, 0.2), (1, 1), (0, 2)])
        timeline = ROBSpeakerDiarizer(embedder=embedder).diarize(
            audio, SAMPLE_RATE, regions
        )
        self.assertEqual([s.speaker_id for s in timeline], ["UNKNOWN"] * 3)
        self.assertEqual(embedder.call_count, 1)

    def test_identity_state_is_reset_for_each_recording_while_embedder_is_reused(self):
        diarizer = ROBSpeakerDiarizer(embedder=speaker_embedding)
        first, first_regions = recording([(1, 2), (2, 2)])
        second, second_regions = recording([(2, 2)])
        diarizer.diarize(first, SAMPLE_RATE, first_regions)
        timeline = diarizer.diarize(second, SAMPLE_RATE, second_regions)
        self.assertEqual(timeline[0].speaker_id, "speaker1")

    def test_embedding_is_bounded_to_the_latest_two_seconds(self):
        embedder = Mock(side_effect=speaker_embedding)
        audio, regions = recording([(1, 5)])
        ROBSpeakerDiarizer(embedder=embedder).diarize(audio, SAMPLE_RATE, regions)
        np.testing.assert_array_equal(embedder.call_args.args[0], audio[-32000:])

    def test_ambiguous_match_does_not_invent_a_third_speaker(self):
        embeddings = iter([np.array([1, 0]), np.array([0, 1]), np.array([1, 1])])
        audio, regions = recording([(1, 2), (2, 2), (1, 2)])
        timeline = ROBSpeakerDiarizer(embedder=lambda audio: next(embeddings)).diarize(
            audio, SAMPLE_RATE, regions
        )
        self.assertEqual(
            [s.speaker_id for s in timeline], ["speaker1", "speaker2", "UNKNOWN"]
        )

    def test_invalid_audio_regions_and_embeddings_fail_explicitly(self):
        diarizer = ROBSpeakerDiarizer(embedder=speaker_embedding)
        audio = np.ones(32000, dtype=np.float32)
        for samples, rate, regions, error in [
            (audio.reshape(2, -1), SAMPLE_RATE, [], ValueError),
            (audio.astype(np.float64), SAMPLE_RATE, [], TypeError),
            (audio, 8000, [], ValueError),
            (np.array([np.nan], dtype=np.float32), SAMPLE_RATE, [], ValueError),
            (audio, SAMPLE_RATE, [SpeechRegion(0, 32001)], ValueError),
            (
                audio,
                SAMPLE_RATE,
                [SpeechRegion(0, 20000), SpeechRegion(10000, 32000)],
                ValueError,
            ),
        ]:
            with self.subTest(rate=rate, regions=regions), self.assertRaises(error):
                diarizer.diarize(samples, rate, regions)
        with self.assertRaises(ValueError):
            ROBSpeakerDiarizer(embedder=lambda audio: np.zeros(2)).diarize(
                audio, SAMPLE_RATE, [SpeechRegion(0, 32000)]
            )

    def test_empty_recording_does_not_run_the_embedder(self):
        embedder = Mock()
        self.assertEqual(
            ROBSpeakerDiarizer(embedder=embedder).diarize(
                np.empty(0, dtype=np.float32), SAMPLE_RATE, []
            ),
            [],
        )
        embedder.assert_not_called()

    def test_loader_uses_pinned_upstream_model_and_cpu_waveform_shape(self):
        import torch

        seen = []

        class Model(torch.nn.Module):
            def forward(self, waveform):
                seen.append(
                    (tuple(waveform.shape), waveform.dtype, waveform.device.type)
                )
                return torch.tensor([[1.0, 0.0]])

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("torch.hub.load", return_value=Model()) as load,
                patch("torch.hub.set_dir") as set_dir,
                patch("torch.cuda.is_available", return_value=False),
            ):
                diarizer = ROBSpeakerDiarizer.from_torch_hub(weights_dir=directory)
            audio, regions = recording([(1, 2)])
            result = diarizer.diarize(audio, SAMPLE_RATE, regions)

        load.assert_called_once_with(
            repo_or_dir=ROBSpeakerDiarizer.DEFAULT_REPOSITORY,
            model="redimnet2",
            model_name="b6",
            train_type="lm",
            dataset="vb2+vox2+cnc2_v0",
            pretrained=True,
            trust_repo=True,
        )
        set_dir.assert_called_once_with(directory)
        self.assertEqual(seen, [((1, 32000), torch.float32, "cpu")])
        self.assertEqual(result[0].speaker_id, "speaker1")


if __name__ == "__main__":
    unittest.main()
