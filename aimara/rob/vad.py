"""ROS-free Silero VAD adapted from eut_speech_audio_processing."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Optional, Union

if TYPE_CHECKING:
    import numpy as np


class ROBVoiceActivityDetector:
    """Load ROB's Silero model and infer speech probability for one chunk."""

    EXPECTED_CHUNK_SIZE = 512
    DEFAULT_REPO_MODEL = "snakers4/silero-vad"
    DEFAULT_MODEL_NAME = "silero_vad"

    def __init__(
        self,
        *,
        model: Any,
        tensor_factory: Callable[["np.ndarray"], Any],
        inference_context: Callable[[], ContextManager[Any]] = nullcontext,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._model = model
        self._tensor_factory = tensor_factory
        self._inference_context = inference_context
        self._logger = logger or logging.getLogger(__name__)

    @classmethod
    def from_torch_hub(
        cls,
        *,
        weights_dir: Union[str, Path],
        repo_model: str = DEFAULT_REPO_MODEL,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "ROBVoiceActivityDetector":
        """Load the same Torch Hub model and defaults used by ROB."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to load ROB VAD") from exc

        selected_logger = logger or logging.getLogger(__name__)
        model_dir = Path(weights_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(model_dir))

        selected_logger.info("Loading ROB VAD model from %s", model_dir)
        model, _ = torch.hub.load(
            repo_or_dir=repo_model,
            model=model_name,
            trust_repo=True,
        )
        selected_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.to(selected_device)
        model.eval()
        selected_logger.info("ROB VAD ready on %s", selected_device)

        return cls(
            model=model,
            tensor_factory=lambda audio: torch.from_numpy(audio).to(selected_device),
            inference_context=torch.no_grad,
            logger=selected_logger,
        )

    def predict(self, audio: "np.ndarray", sample_rate: int) -> float:
        """Return speech probability for one mono float32 audio chunk."""
        size = getattr(audio, "size", None)
        if size != self.EXPECTED_CHUNK_SIZE:
            self._logger.warning(
                "Unexpected audio chunk size: %s. Expected %s samples.",
                size,
                self.EXPECTED_CHUNK_SIZE,
            )
            return 0.0

        if getattr(audio, "ndim", None) != 1:
            raise ValueError("audio must be a one-dimensional array")
        if str(getattr(audio, "dtype", "")) != "float32":
            raise TypeError("audio must use float32 samples")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        audio_tensor = self._tensor_factory(audio)
        with self._inference_context():
            result = self._model(audio_tensor, sr=sample_rate)
        probability = result.item() if hasattr(result, "item") else result
        return float(probability)
