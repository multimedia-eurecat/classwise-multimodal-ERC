"""Generated lossless media through real orchestration, encoders and a tiny ERC.

Only the learned speech decisions and unimodal networks are controlled doubles.
These fixtures check contracts, not pretrained model accuracy.
"""

import csv
import json
import math
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
    ASRMetadata,
    ERCEmotionModel,
    ERCFeatureBridge,
    ERCModelProfile,
    OfflineROBProcessor,
    OfflineVADProcessor,
    ROBProcessorConfig,
    SpeakerInterval,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    write_wav,
)
from aimara.batch import run_batch
from aimara.batch_resume import MANIFEST_NAME
from aimara.erc_encoders import OriolAudioEncoder, OriolTextEncoder, OriolVisualEncoder
from aimara.pipeline import AIMARAPipeline


MANIFEST = Path(__file__).parent / "fixtures" / "acceptance_cases.json"
VOICE_LABELS = {0.25: "A", 0.75: "B", 0.5: "unresolved", 0.875: "C"}


def audio_runs(audio):
    levels = np.array([0, *VOICE_LABELS], dtype=np.float32)
    labels = levels[np.argmin(np.abs(audio[:, None] - levels), axis=1)]
    boundaries = [0, *(np.flatnonzero(np.diff(labels)) + 1), len(audio)]
    return [
        (int(start), int(end), float(labels[start]))
        for start, end in zip(boundaries, boundaries[1:])
    ]


class FixtureVAD:
    def predict(self, audio, sample_rate):
        return 0.99 if np.max(np.abs(audio)) > 0.1 else 0.01


class FixtureASR:
    def __init__(self):
        self.input_lengths = []

    def transcribe(self, audio, sample_rate):
        self.input_lengths.append(len(audio))
        words = []
        for start, end, value in audio_runs(audio):
            label = VOICE_LABELS[value]
            begin, finish = start / sample_rate, end / sample_rate
            middle = (begin + finish) / 2
            words.extend(
                (
                    TranscriptWord(begin + 0.1, middle - 0.05, " Voice"),
                    TranscriptWord(middle + 0.05, finish - 0.1, f" {label}."),
                )
            )
        text = "".join(word.text for word in words).strip()
        return Transcript(
            text,
            "en",
            (
                TranscriptSegment(
                    words[0].start_time, words[-1].end_time, text, tuple(words)
                ),
            ),
            ASRMetadata(
                backend="fixture",
                model="scripted-words",
                language_source="configured",
                device="cpu",
                compute_type="float32",
            ),
        )


class FixtureDiarizer:
    def diarize(self, audio, sample_rate, regions):
        return [
            SpeakerInterval(start, end, VOICE_LABELS[value], 0 if value == 0.25 else 1)
            for start, end, value in audio_runs(audio)
            if value in VOICE_LABELS and value != 0.5
        ]


class CapturingProcessor(OfflineROBProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def process_video(self, video, output, **kwargs):
        self.calls.append(Path(video).name)
        return super().process_video(video, output, **kwargs)


class CapturingBridge(ERCFeatureBridge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = []

    def build(self, segments):
        features = super().build(segments)
        self.features.append(features)
        return features


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for acceptance media")
class PipelineAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from multimodal.model import Transformer_Based_Model

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        manifest = json.loads(MANIFEST.read_text())
        cls.rate, cls.fps = manifest["sample_rate"], manifest["fps"]
        cls.cases = {case["id"]: case for case in manifest["cases"]}
        cls.audio = {}
        cls.videos = cls.root / "videos"
        cls.videos.mkdir()
        for name, case in cls.cases.items():
            blocks = []
            for index, (count, value) in enumerate(case["audio_blocks"]):
                samples = np.full(count, value, dtype=np.float32)
                if value:
                    # Phase/amplitude markers remain distinguishable after RMS normalization.
                    samples += ((np.arange(count) // 128 + index) % 4).astype(
                        np.float32
                    ) / 512
                blocks.append(samples)
            audio = np.concatenate(blocks)
            cls.audio[name] = audio
            wav = write_wav(cls.root / f"{name}.wav", audio, sample_rate=cls.rate)
            count = math.ceil(len(audio) * cls.fps / cls.rate)
            frames = np.zeros((count, 16, 16, 3), dtype=np.uint8)
            for index in range(count):
                frames[index, :, :, 0] = index % 256
                frames[index, :, :, 1] = index // 256
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
                    str(cls.fps),
                    "-i",
                    "pipe:0",
                    "-i",
                    str(wav),
                    "-c:v",
                    "ffv1",
                    "-pix_fmt",
                    "bgr0",
                    "-c:a",
                    "pcm_s16le",
                    str(cls.videos / f"{name}.avi"),
                ],
                input=frames.tobytes(),
                capture_output=True,
                check=True,
            )
        cls.profile = ERCModelProfile(
            text_dim=2, audio_dim=3, visual_dim=4, hidden_dim=8, n_head=2, dropout=0
        )
        with torch.random.fork_rng():
            torch.manual_seed(7)
            model = Transformer_Based_Model(
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
            )
        cls.checkpoint = cls.root / "tiny-erc.pt"
        torch.save(model.state_dict(), cls.checkpoint)

    def make_pipeline(self, case=None):
        for patcher in reversed(getattr(self, "encoder_patchers", [])):
            patcher.stop()
        self.encoder_patchers = []
        from unimodal.audio import models as audio_models
        from unimodal.text import models as text_models
        from unimodal.visual import models as visual_models

        case = case or {"diarization": True}
        self.asr = FixtureASR()
        self.text_inputs, self.audio_inputs, self.visual_windows = [], [], []
        seen_frames = []

        def embed_text(texts, *args):
            self.text_inputs.extend(texts)
            return np.array(
                [[len(text) / 100, text.count("Voice") / 10] for text in texts],
                dtype=np.float32,
            )

        def embed_audio(waves, *args):
            self.audio_inputs.extend(waves)
            return np.array(
                [
                    [np.mean(samples), np.std(samples), len(samples) / self.rate]
                    for samples in waves
                ],
                dtype=np.float32,
            )

        def detect(image):
            seen_frames.append(int(image[0, 0, 0]) + 256 * int(image[0, 0, 1]))
            return None, None

        def embed_image(image, *args):
            pixels = np.asarray(image)
            marker = int(pixels[0, 0, 0]) + 256 * int(pixels[0, 0, 1])
            return np.array([marker / 1000, 1, 0, 1], dtype=np.float32)

        extract_visual = visual_models.extract_utterance_visual_embedding

        def track_visual(**kwargs):
            begin = len(seen_frames)
            result = extract_visual(**kwargs)
            self.visual_windows.append(
                (kwargs["start_sec"], kwargs["end_sec"], seen_frames[begin:])
            )
            return result

        for module, name, function in (
            (text_models, "embed_text_batch", embed_text),
            (audio_models, "embed_audio_batch", embed_audio),
            (visual_models, "_embed_image", embed_image),
            (visual_models, "extract_utterance_visual_embedding", track_visual),
        ):
            patcher = patch.object(module, name, new=function)
            patcher.start()
            self.encoder_patchers.append(patcher)
            self.addCleanup(patcher.stop)
        speech_config = ROBProcessorConfig(
            min_speech_duration=0.064, min_silence_duration=0.064, pre_buffer_duration=0
        )
        processor = CapturingProcessor(
            vad_processor=OfflineVADProcessor(FixtureVAD(), speech_config),
            transcriber=self.asr,
            diarizer=FixtureDiarizer() if case["diarization"] else None,
        )
        bridge = CapturingBridge(
            OriolTextEncoder(
                text_models.TextConfig(device="cpu", use_sentiment_signal=False),
                None,
                None,
            ),
            OriolAudioEncoder(audio_models.AudioConfig(device="cpu"), None, None, 3),
            OriolVisualEncoder(
                visual_models.VisualConfig(
                    device="cpu",
                    use_fullframe_fallback=case.get("fullframe_fallback", True),
                ),
                SimpleNamespace(detect=detect),
                None,
                None,
                4,
            ),
        )
        emotion_model = ERCEmotionModel.from_checkpoint(
            self.checkpoint, profile=self.profile, device="cpu"
        )
        return AIMARAPipeline(
            processor,
            bridge,
            emotion_model,
            resume_config={"fixture_pipeline": 1, "checkpoint": self.checkpoint},
        )

    def check_case(self, name):
        case = self.cases[name]
        pipeline = self.make_pipeline(case)
        output = self.root / self._testMethodName / name
        predictions = pipeline.process_video(self.videos / f"{name}.avi", output)
        saved = json.loads(
            (output / "predictions.json").read_text(),
            parse_constant=lambda value: self.fail(f"Non-finite JSON: {value}"),
        )
        self.assertEqual(len(saved), len(case["segments"]))
        self.assertEqual([prediction.to_dict() for prediction in predictions], saved)
        self.assertEqual(
            self.asr.input_lengths, [end - start for start, end in case["regions"]]
        )
        source_pcm = (self.audio[name] * 32768).astype("<i2")
        for index, (row, expected) in enumerate(zip(saved, case["segments"])):
            start, end, speaker, text = expected
            self.assertEqual(row["conversation_id"], name)
            self.assertEqual(row["segment_id"], index)
            self.assertEqual(
                (row["start_time"], row["end_time"]),
                (start / self.rate, end / self.rate),
            )
            self.assertEqual((row["speaker_id"], row["transcript"]), (speaker, text))
            region = next((a, b) for a, b in case["regions"] if a <= start < end <= b)
            self.assertEqual(
                (row["asr_input_start_time"], row["asr_input_end_time"]),
                tuple(value / self.rate for value in region),
            )
            self.assertEqual(row["asr_backend"], "fixture")
            self.assertIsNone(row["transcript_confidence"])
            self.assertEqual(row["language_source"], "configured")
            with wave.open(str(predictions[index].segment.audio_path)) as wav:
                self.assertEqual(
                    (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()),
                    (self.rate, 1, 2),
                )
                self.assertEqual(wav.getnframes(), end - start)
                np.testing.assert_array_equal(
                    np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2"),
                    source_pcm[start:end],
                )
            probabilities = np.array(list(row["probabilities"].values()))
            self.assertEqual(len(probabilities), 6)
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(np.all((0 <= probabilities) & (probabilities <= 1)))
            self.assertAlmostEqual(float(probabilities.sum()), 1, places=6)
            self.assertEqual(row["confidence"], float(probabilities.max()))
            self.assertEqual(
                row["emotion"], max(row["probabilities"], key=row["probabilities"].get)
            )
            # The audio adapter retains exactly the clip plus its training-time boundary padding.
            self.assertEqual(
                len(self.audio_inputs[index]),
                end - start + 2 * round(audio_models_pad() * self.rate),
            )
            pad = round(audio_models_pad() * self.rate)
            encoded = self.audio_inputs[index]
            np.testing.assert_array_equal(encoded[:pad], 0)
            np.testing.assert_array_equal(encoded[-pad:], 0)
            middle, reference = encoded[pad:-pad], self.audio[name][start:end]
            gain = middle[0] / reference[0]
            np.testing.assert_allclose(middle, reference * gain, rtol=1e-6, atol=1e-7)
            expected_context = " [SEP] ".join(
                f"{turn[2]}: {turn[3]}"
                for turn in case["segments"][max(0, index - 2) : index + 1]
            )
            self.assertEqual(self.text_inputs[index], expected_context)
            window_start, window_end, frames = self.visual_windows[index]
            self.assertEqual(
                (window_start, window_end), (start / self.rate, end / self.rate)
            )
            self.assertTrue(frames, "Fixture window should contain source frames")
            self.assertEqual(frames, sorted(set(frames)))
            self.assertTrue(
                all(start <= frame * self.rate / self.fps < end for frame in frames)
            )
        if saved:
            features = pipeline.feature_bridge.features[-1]
            self.assertEqual(
                [segment.segment_id for segment in features.segments],
                list(range(len(saved))),
            )
            if not case.get("fullframe_fallback", True):
                np.testing.assert_array_equal(
                    features.visual, np.zeros((len(saved), 4))
                )
        return pipeline, saved

    def test_silence_produces_empty_json_without_asr_or_feature_calls(self):
        pipeline, saved = self.check_case("silence")
        self.assertEqual(saved, [])
        self.assertEqual(pipeline.feature_bridge.features, [])
        self.assertEqual(
            (self.text_inputs, self.audio_inputs, self.visual_windows), ([], [], [])
        )

    def test_source_boundaries_and_long_speech_preserve_every_speech_sample(self):
        for name in ("speech_at_boundaries", "long_speech"):
            with self.subTest(case=name):
                self.check_case(name)

    def test_speaker_return_and_unknown_turns_reach_real_erc_in_order(self):
        for name, mask in (
            ("speaker_returns", [[1, 0], [0, 1], [1, 0]]),
            ("two_known_plus_unknown", [[1, 0], [0, 1], [0, 0]]),
        ):
            with self.subTest(case=name):
                pipeline, saved = self.check_case(name)
                np.testing.assert_array_equal(
                    pipeline.feature_bridge.features[-1].speaker_mask, mask
                )
                self.assertEqual(saved[0]["speaker_id_confidence"], 0)
                if name == "two_known_plus_unknown":
                    self.assertEqual(saved[-1]["speaker_id_confidence"], -1)
                    self.assertIn("UNKNOWN: Voice unresolved.", self.text_inputs[-1])

    def test_missing_faces_keep_speech_and_produce_finite_erc_predictions(self):
        self.check_case("missing_faces")

    def test_excess_known_speakers_fail_before_emotion_results_are_written(self):
        case = self.cases["speaker_capacity"]
        pipeline = self.make_pipeline(case)
        output = self.root / self._testMethodName
        with self.assertRaisesRegex(ValueError, case["error"]):
            pipeline.process_video(self.videos / "speaker_capacity.avi", output)
        self.assertFalse((output / "predictions.json").exists())
        self.assertEqual(pipeline.feature_bridge.features, [])

    def test_real_media_batch_exports_recovers_and_checks_resume_configuration(self):
        root = self.root / self._testMethodName
        videos, output = root / "videos", root / "output"
        videos.mkdir(parents=True)
        for filename, fixture in (
            ("00_silence.avi", "silence"),
            ("01_turns.avi", "speaker_returns"),
            ("03_capacity.avi", "speaker_capacity"),
            ("04_valid.avi", "speech_at_boundaries"),
        ):
            shutil.copyfile(self.videos / f"{fixture}.avi", videos / filename)
        broken = videos / "02_broken.avi"
        broken.write_bytes(b"invalid media")
        pipeline = self.make_pipeline()

        def batch(**kwargs):
            return run_batch(pipeline, videos, output, extensions=(".avi",), **kwargs)

        rows, failures = batch()
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            [failure["error_type"] for failure in failures],
            ["RuntimeError", "ValueError"],
        )
        self.assertEqual(
            [failure["video_file"] for failure in failures],
            ["02_broken.avi", "03_capacity.avi"],
        )
        self.assertEqual(json.loads((output / "failures.json").read_text()), failures)
        self.assertTrue((output / "00_silence" / MANIFEST_NAME).is_file())
        self.assertFalse((output / "03_capacity" / MANIFEST_NAME).exists())
        self.assertEqual(json.loads((output / "predictions.json").read_text()), rows)
        with (output / "predictions.csv").open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), len(rows))
        for csv_row, row in zip(csv_rows, rows):
            self.assertEqual(csv_row["video_file"], row["video_file"])
            self.assertEqual(csv_row["transcript"], row["transcript"])
            self.assertEqual(csv_row["transcript_confidence"], "")
            self.assertAlmostEqual(float(csv_row["confidence"]), row["confidence"])
            for emotion, value in row["probabilities"].items():
                self.assertAlmostEqual(float(csv_row[f"probability_{emotion}"]), value)

        pipeline.speech_processor.calls.clear()
        reused, failures = batch(resume=True)
        self.assertEqual(reused, rows)
        self.assertEqual(
            pipeline.speech_processor.calls, ["02_broken.avi", "03_capacity.avi"]
        )
        shutil.copyfile(self.videos / "missing_faces.avi", broken)
        pipeline.speech_processor.calls.clear()
        recovered, failures = batch(resume=True)
        self.assertEqual(len(recovered), 6)
        self.assertEqual(
            [failure["video_file"] for failure in failures], ["03_capacity.avi"]
        )
        self.assertEqual(
            pipeline.speech_processor.calls, ["02_broken.avi", "03_capacity.avi"]
        )
        self.assertTrue((output / "02_broken" / MANIFEST_NAME).is_file())

        pipeline.speech_processor.calls.clear()
        changed, failures = batch(
            resume=True,
            run_config={"fixture_pipeline": 2, "checkpoint": self.checkpoint},
        )
        self.assertEqual(changed, recovered)
        self.assertEqual(
            pipeline.speech_processor.calls,
            sorted(path.name for path in videos.iterdir()),
        )
        self.assertEqual(
            [failure["video_file"] for failure in failures], ["03_capacity.avi"]
        )


def audio_models_pad():
    from unimodal.audio.models import _BOUNDARY_PAD_S

    return _BOUNDARY_PAD_S
