"""Check that runner configuration identities track active inference settings."""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from aimara.batch_resume import configuration_digest, snapshot_configuration
from aimara.erc_features import ERCFeatureBridge
from aimara.erc_model import ERCModelProfile
from aimara.rob import ROBSpeakerDiarizer
from aimara.run_batch import build_parser
from aimara.run_pipeline import (
    build_asr_config,
    build_parakeet_config,
    build_resume_config,
    build_segmentation_configs,
)


@dataclass
class EncoderConfig:
    model_id: str = "test/model"
    device: str = "cpu"
    batch_size: int = 16


class ResumeConfigurationTests(unittest.TestCase):
    def configuration(self, root, *extra):
        for name in ("vad", "asr", "diarization"):
            (root / name).mkdir(exist_ok=True)
        checkpoint = root / "model.pt"
        if not checkpoint.exists():
            checkpoint.write_bytes(b"weights")
        args = build_parser().parse_args(
            [
                "--video-dir",
                "videos",
                "--output-dir",
                "output",
                "--checkpoint",
                str(checkpoint),
                "--vad-weights",
                str(root / "vad"),
                "--asr-weights",
                str(root / "asr"),
                "--diarization-weights",
                str(root / "diarization"),
                *extra,
            ]
        )
        speech, speakers = build_segmentation_configs(args)

        def encoder(model_id="test/model"):
            return SimpleNamespace(
                config=EncoderConfig(model_id=model_id),
                model=SimpleNamespace(
                    config=SimpleNamespace(_commit_hash="revision-1")
                ),
            )

        bridge = ERCFeatureBridge(
            encoder(),
            encoder(args.audio_model),
            encoder(),
            unknown_speaker_policy=args.unknown_speaker_policy,
        )
        config = build_resume_config(
            args,
            speech_config=speech,
            speaker_config=speakers,
            asr_config=(
                build_asr_config(args)
                if args.asr_backend == "whisper"
                else build_parakeet_config(args)
            ),
            transcriber=SimpleNamespace(
                device="cpu",
                effective_compute_type="int8",
                language_identifier=None,
            ),
            feature_bridge=bridge,
            emotion_model=SimpleNamespace(device="cpu"),
            profile=ERCModelProfile(),
            diarizer=(
                ROBSpeakerDiarizer(embedder=lambda audio: audio)
                if args.diarization_backend == "redimnet2"
                else None
            ),
        )
        return snapshot_configuration(config)

    def test_whisper_runtime_changes_invalidate_identity(self):
        cases = (
            ("--asr-model", "small"),
            ("--language", "es"),
            ("--asr-compute-type", "float32"),
            ("--asr-cpu-compute-type", "float32"),
            ("--asr-batched-inference",),
            ("--asr-batch-size", "4"),
            ("--asr-beam-size", "2"),
            ("--vad-threshold", "0.7"),
            ("--min-speech-duration", "0.4"),
            ("--min-silence-duration", "0.3"),
            ("--pre-buffer-duration", "0.4"),
            ("--max-segment-duration", "20"),
            ("--diarization-backend", "redimnet2"),
            ("--unknown-speaker-policy", "reserve_slot"),
            ("--audio-model", "different/model"),
            ("--device", "cpu"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.configuration(root)
            for options in cases:
                with self.subTest(options=options):
                    self.assertNotEqual(baseline, self.configuration(root, *options))

    def test_speaker_grouping_matters_only_with_diarization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grouping = (
                "--no-snap-splits-to-sentences",
                "--min-speaker-run-tokens",
                "3",
                "--min-speaker-run-duration",
                "0.7",
            )
            self.assertEqual(
                self.configuration(root), self.configuration(root, *grouping)
            )
            enabled = ("--diarization-backend", "redimnet2")
            self.assertNotEqual(
                self.configuration(root, *enabled),
                self.configuration(root, *enabled, *grouping),
            )

    def test_inactive_backend_and_output_options_do_not_invalidate_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                self.configuration(root),
                self.configuration(
                    root,
                    "--parakeet-model",
                    "different/model",
                    "--video-dir",
                    "elsewhere",
                    "--output-dir",
                    "new",
                    "--resume",
                ),
            )
            parakeet = ("--asr-backend", "parakeet")
            self.assertEqual(
                self.configuration(root, *parakeet),
                self.configuration(
                    root,
                    *parakeet,
                    "--asr-model",
                    "small",
                    "--asr-batched-inference",
                    "--asr-beam-size",
                    "2",
                ),
            )

    def test_parakeet_runtime_changes_invalidate_identity(self):
        cases = (
            ("--parakeet-model", "different/model"),
            ("--parakeet-language-id-model", ""),
            ("--parakeet-language-id-min-confidence", "0.8"),
            ("--parakeet-local-files-only",),
            ("--no-parakeet-warmup",),
            ("--language", "es"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = ("--asr-backend", "parakeet")
            baseline = self.configuration(root, *backend)
            self.assertNotEqual(baseline, self.configuration(root))
            for options in cases:
                with self.subTest(options=options):
                    self.assertNotEqual(
                        baseline, self.configuration(root, *backend, *options)
                    )

    def test_dictionary_order_does_not_change_fingerprint_and_nan_is_rejected(self):
        self.assertEqual(
            configuration_digest(snapshot_configuration({"a": 1, "b": {"c": (2, 3)}})),
            configuration_digest(snapshot_configuration({"b": {"c": [2, 3]}, "a": 1})),
        )
        with self.assertRaises(ValueError):
            snapshot_configuration({"threshold": float("nan")})


if __name__ == "__main__":
    unittest.main()
