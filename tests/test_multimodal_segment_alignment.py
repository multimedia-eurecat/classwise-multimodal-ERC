"""Exercise split clips against real PCM and lossless, numbered video frames."""

import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aimara import (
    ERCFeatureBridge,
    OfflineROBProcessor,
    ROBProcessorConfig,
    SpeechRegion,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    write_wav,
)
from aimara.audio_io import decode_audio
from aimara.diarization import SpeakerInterval
from aimara.erc_encoders import OriolAudioEncoder, OriolVisualEncoder


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for media alignment")
class MultimodalSegmentAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.video = cls.root / "numbered.avi"
        # One continuous VAD region, with sample-aligned cuts between video frames.
        cls.bounds = (2000, 11600, 21200, 30800)
        audio = np.zeros(32000, dtype=np.float32)
        for start, end, level in zip(cls.bounds, cls.bounds[1:], (0.25, 0.75, 0.25)):
            audio[start:end] = level
        source_wav = write_wav(cls.root / "source.wav", audio)
        frames = np.zeros((20, 16, 16, 3), dtype=np.uint8)
        for index in range(20):
            frames[index, :, :, 0] = index * 10
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                "16x16",
                "-framerate",
                "10",
                "-i",
                "pipe:0",
                "-i",
                str(source_wav),
                "-c:v",
                "ffv1",
                "-c:a",
                "pcm_s16le",
                str(cls.video),
            ],
            input=frames.tobytes(),
            capture_output=True,
            check=True,
        )

    def make_segments(self):
        bounds = self.bounds

        class VAD:
            config = ROBProcessorConfig()

            def process(self, audio):
                return [SpeechRegion(bounds[0], bounds[-1])]

        class Diarizer:
            def diarize(self, audio, sample_rate, regions):
                return [
                    SpeakerInterval(start, end, speaker)
                    for start, end, speaker in zip(
                        bounds, bounds[1:], ("speaker1", "speaker2", "speaker1")
                    )
                ]

        class ASR:
            calls = 0

            def transcribe(self, audio, sample_rate):
                self.calls += 1
                words = tuple(
                    TranscriptWord(*args)
                    for args in (
                        (0.1, 0.2, " Hello"),
                        (0.3, 0.5, " there."),
                        (0.7, 0.8, " Goodbye"),
                        (0.9, 1.1, " now!"),
                        (1.3, 1.4, " Welcome"),
                        (1.5, 1.7, " back."),
                    )
                )
                text = "".join(word.text for word in words).strip()
                return Transcript(
                    text, "en", (TranscriptSegment(0.1, 1.7, text, words),)
                )

        asr = ASR()
        processor = OfflineROBProcessor(
            vad_processor=VAD(), transcriber=asr, diarizer=Diarizer()
        )
        segments = processor.process_video(self.video, self.root / "output")
        self.assertEqual(asr.calls, 1)
        self.assertEqual(
            [(s.start_time, s.end_time) for s in segments],
            [(0.125, 0.725), (0.725, 1.325), (1.325, 1.925)],
        )
        return segments

    def visual_encoder(self, mode, asd=None):
        config = SimpleNamespace(
            visual_mode=mode,
            num_frames=100,
            device="cpu",
            face_margin=0.2,
            use_fullframe_fallback=True,
            scene_cut_threshold=0.3,
            asd_threshold=0.5,
            asd_iou_threshold=0.5,
            asd_max_seconds=60,
        )
        detector = SimpleNamespace(detect=lambda image: (None, None))
        return OriolVisualEncoder(
            config,
            detector,
            None,
            None,
            408 if mode == "landmarks" else 39 if mode == "landmarks_exp" else 1,
            asd,
        )

    def test_split_pcm_and_real_audio_encoder_receive_only_their_own_clip(self):
        from unimodal.audio import models as audio_models

        segments = self.make_segments()
        source = decode_audio(self.video)
        input_lengths = []

        def embed(
            waves, processor, model, device, rate, pooling, silent_mask, embed_dim
        ):
            # Existing training preprocessing adds synthetic silence and RMS gain.
            # Verify the model input still contains precisely one split's PCM.
            pad = int(audio_models._BOUNDARY_PAD_S * rate)
            for samples, start, end in zip(waves, self.bounds, self.bounds[1:]):
                self.assertEqual(len(samples), end - start + 2 * pad)
                np.testing.assert_array_equal(samples[:pad], 0)
                np.testing.assert_array_equal(samples[-pad:], 0)
                middle = samples[pad:-pad]
                ratio = middle[0] / source[start]
                np.testing.assert_allclose(middle, source[start:end] * ratio, atol=1e-7)
                input_lengths.append(len(middle))
            return np.ones((len(waves), embed_dim), dtype=np.float32)

        combined_pcm = []
        for segment, start, end in zip(segments, self.bounds, self.bounds[1:]):
            with wave.open(str(segment.audio_path)) as clip:
                self.assertEqual(clip.getnframes(), end - start)
                self.assertEqual(clip.getframerate(), 16000)
                combined_pcm.append(clip.readframes(clip.getnframes()))
        with wave.open(str(self.root / "source.wav")) as original:
            original.setpos(self.bounds[0])
            self.assertEqual(
                b"".join(combined_pcm),
                original.readframes(self.bounds[-1] - self.bounds[0]),
            )
        encoder = OriolAudioEncoder(
            audio_models.AudioConfig(device="cpu"), None, None, 1
        )
        with patch.object(audio_models, "embed_audio_batch", side_effect=embed):
            features = encoder.encode(segments)
        self.assertEqual(features.shape, (3, 1))
        self.assertEqual(input_lengths, [9600, 9600, 9600])

    def test_all_visual_modes_read_correct_source_frames_for_a_b_a_groups(self):
        from unimodal.visual import models

        segments = self.make_segments()
        expected_indices = list(range(2, 20))
        for mode in ("vit", "landmarks", "landmarks_exp"):
            with self.subTest(mode=mode):
                seen = []

                def frame_embedding(image, *args):
                    index = int(np.asarray(image)[0, 0, 0]) // 10
                    seen.append(index)
                    shape = (
                        (68, 3)
                        if mode == "landmarks"
                        else (13,)
                        if mode == "landmarks_exp"
                        else (1,)
                    )
                    return np.full(shape, index, dtype=np.float32)

                function = {
                    "vit": "_embed_image",
                    "landmarks": "_extract_landmark_frame",
                    "landmarks_exp": "_extract_exp_frame",
                }[mode]
                encoder = self.visual_encoder(mode)

                class TextEncoder:
                    def encode(self, items):
                        return np.ones((len(items), 1), dtype=np.float32)

                # The bridge must restore chronological order before encoding.
                with patch.object(models, function, side_effect=frame_embedding):
                    features = ERCFeatureBridge(
                        TextEncoder(), TextEncoder(), encoder
                    ).build(list(reversed(segments)))
                self.assertEqual(seen, expected_indices)
                self.assertEqual(features.visual.shape, (3, encoder.embed_dim))
                np.testing.assert_allclose(features.visual[:, 0], [4.5, 10.5, 16.5])
                np.testing.assert_array_equal(
                    features.speaker_mask, [[1, 0], [0, 1], [1, 0]]
                )

    def test_empty_visual_windows_return_zero_without_borrowing_neighbour_frames(self):
        from unimodal.visual import models

        for start, end in ((0.21, 0.29), (2.1, 2.2)):
            for mode in ("vit", "landmarks", "landmarks_exp"):
                with self.subTest(start=start, mode=mode):
                    encoder = self.visual_encoder(mode)
                    function = (
                        models.extract_utterance_visual_embedding
                        if mode == "vit"
                        else models.extract_utterance_landmark_embedding
                    )
                    kwargs = (
                        {"processor": None, "model": None, "device": "cpu"}
                        if mode == "vit"
                        else {"tddfa": None, "visual_mode": mode}
                    )
                    with (
                        patch.object(models, "read_frame_at") as read,
                        patch.object(models, "_embed_image") as embed,
                    ):
                        result = function(
                            str(self.video),
                            encoder.mtcnn,
                            embed_dim=encoder.embed_dim,
                            start_sec=start,
                            end_sec=end,
                            **kwargs,
                        )
                    np.testing.assert_array_equal(result, np.zeros(encoder.embed_dim))
                    read.assert_not_called()
                    embed.assert_not_called()

    def test_active_speaker_detection_matches_absolute_frames_and_bounds_audio(self):
        from unimodal.visual import asd

        wrapper = object.__new__(asd.LightASDWrapper)
        audio = np.arange(32000, dtype=np.float32)

        def detect(image):
            index = int(image[0, 0, 0]) // 10
            return np.array([[index, 0, index + 20, 20]], dtype=np.float32), None

        def score(tracks, frames, mfcc, fps):
            for track in tracks:
                track.score = 1.0
            return tracks

        for start, end, indices in (
            (0.125, 0.725, range(2, 8)),
            (0.725, 1.325, range(8, 14)),
        ):
            with (
                self.subTest(start=start),
                patch.object(asd, "extract_raw_audio", return_value=audio),
                patch.object(
                    asd, "audio_to_mfcc_lightasd", return_value=np.zeros((24, 13))
                ) as mfcc,
                patch.object(wrapper, "score_tracks", side_effect=score),
            ):
                boxes = wrapper.get_speaker_boxes(
                    str(self.video),
                    np.array(list(indices)),
                    SimpleNamespace(detect=detect),
                    start_sec=start,
                    end_sec=end,
                )
                self.assertEqual([int(boxes[i][0]) for i in indices], list(indices))
                np.testing.assert_array_equal(
                    mfcc.call_args.args[0],
                    audio[round(indices.start / 10 * 16000) : round(end * 16000)],
                )
                self.assertEqual(mfcc.call_args.kwargs["n_frames"], 6)

    def test_asd_visual_extractors_pass_the_split_end_without_extra_context(self):
        from unimodal.visual import models

        for mode in ("vit", "landmarks", "landmarks_exp"):
            with self.subTest(mode=mode):
                from unittest.mock import Mock

                detector = Mock()
                detector.get_speaker_boxes.return_value = {}
                encoder = self.visual_encoder(mode, detector)
                pooled = (
                    "_embed_with_boxes" if mode == "vit" else "_landmarks_with_boxes"
                )
                with patch.object(
                    models, pooled, return_value=np.zeros(encoder.embed_dim)
                ):
                    encoder.encode(self.make_segments())
                calls = detector.get_speaker_boxes.call_args_list
                self.assertEqual(
                    [(c.kwargs["start_sec"], c.kwargs["end_sec"]) for c in calls],
                    [(0.125, 0.725), (0.725, 1.325), (1.325, 1.925)],
                )
                self.assertTrue(all(c.kwargs["max_seconds"] is None for c in calls))


class FrameWindowTests(unittest.TestCase):
    def test_adjacent_windows_partition_source_frame_timestamps(self):
        from unimodal.visual.utils import sample_window_frame_indices

        for fps in (10, 25, 30000 / 1001):
            with self.subTest(fps=fps):
                cut = 0.725
                left = sample_window_frame_indices(60, fps, 100, 0, cut)
                right = sample_window_frame_indices(60, fps, 100, cut, 60 / fps)
                np.testing.assert_array_equal(
                    np.concatenate((left, right)), np.arange(60)
                )
                self.assertTrue(np.all(left / fps < cut))
                self.assertTrue(np.all(right / fps >= cut))

    def test_exact_frame_boundaries_and_full_video_sampling(self):
        from unimodal.visual.utils import sample_window_frame_indices

        np.testing.assert_array_equal(
            sample_window_frame_indices(60, 25, 100, 0.28, 0.56), np.arange(7, 14)
        )
        np.testing.assert_array_equal(
            sample_window_frame_indices(60, 25, 3), [0, 29, 59]
        )

    def test_invalid_windows_are_rejected(self):
        from unimodal.visual.utils import sample_window_frame_indices

        for start, end in ((-0.1, 1), (1, 0.5), (0, float("inf")), (float("nan"), 1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                sample_window_frame_indices(60, 25, 4, start, end)


if __name__ == "__main__":
    unittest.main()
