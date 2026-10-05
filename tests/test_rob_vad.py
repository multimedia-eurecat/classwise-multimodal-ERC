import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aimara.rob import ROBVoiceActivityDetector


class FakeAudio:
    def __init__(self, *, size=512, ndim=1, dtype="float32"):
        self.size = size
        self.ndim = ndim
        self.dtype = dtype


class Probability:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeModel:
    def __init__(self, probability=0.75):
        self.probability = probability
        self.calls = []
        self.device = None
        self.eval_called = False

    def __call__(self, tensor, *, sr):
        self.calls.append((tensor, sr))
        return Probability(self.probability)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class ROBVoiceActivityDetectorTests(unittest.TestCase):
    def test_predict_passes_chunk_and_sample_rate_to_model(self):
        model = FakeModel(probability=0.82)
        converted_tensor = object()
        detector = ROBVoiceActivityDetector(
            model=model,
            tensor_factory=lambda audio: converted_tensor,
            inference_context=nullcontext,
        )

        probability = detector.predict(FakeAudio(), sample_rate=16000)

        self.assertEqual(probability, 0.82)
        self.assertEqual(model.calls, [(converted_tensor, 16000)])

    def test_unexpected_chunk_size_returns_zero_without_inference(self):
        model = FakeModel()
        detector = ROBVoiceActivityDetector(
            model=model,
            tensor_factory=Mock(),
        )

        probability = detector.predict(FakeAudio(size=128), sample_rate=16000)

        self.assertEqual(probability, 0.0)
        self.assertEqual(model.calls, [])

    def test_predict_validates_shape_dtype_and_sample_rate(self):
        detector = ROBVoiceActivityDetector(
            model=FakeModel(),
            tensor_factory=lambda audio: audio,
        )

        for audio, sample_rate, error in [
            (FakeAudio(ndim=2), 16000, ValueError),
            (FakeAudio(dtype="float64"), 16000, TypeError),
            (FakeAudio(), 0, ValueError),
        ]:
            with self.subTest(audio=audio, sample_rate=sample_rate):
                with self.assertRaises(error):
                    detector.predict(audio, sample_rate)

    def test_from_torch_hub_uses_rob_defaults_and_cpu_fallback(self):
        model = FakeModel()
        tensor = SimpleNamespace(to=Mock(return_value="converted"))
        fake_torch = SimpleNamespace(
            hub=SimpleNamespace(
                set_dir=Mock(),
                load=Mock(return_value=(model, object())),
            ),
            cuda=SimpleNamespace(is_available=Mock(return_value=False)),
            device=Mock(side_effect=lambda value: value),
            from_numpy=Mock(return_value=tensor),
            no_grad=nullcontext,
        )

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"torch": fake_torch}):
                detector = ROBVoiceActivityDetector.from_torch_hub(
                    weights_dir=Path(directory) / "weights"
                )

            probability = detector.predict(FakeAudio(), 16000)

        fake_torch.hub.load.assert_called_once_with(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.assertEqual(model.device, "cpu")
        self.assertTrue(model.eval_called)
        self.assertEqual(probability, 0.75)


if __name__ == "__main__":
    unittest.main()
