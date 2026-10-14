"""Check the offline runtime without loading or downloading model checkpoints."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys


IMPORT_PROBE = """
import numpy as np
import torch
import torchaudio
import torchvision
import cv2
from facenet_pytorch import MTCNN
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from funasr import AutoModel as FunASRAutoModel
from unimodal.audio.models import AudioConfig
from unimodal.text.models import TextConfig
from unimodal.visual.models import VisualConfig
from aimara.run_pipeline import build_parser

# Exercise NumPy/Torch and the packaged face-detector weights without a network call.
assert torch.from_numpy(np.zeros((2, 2), dtype=np.float32)).numpy().shape == (2, 2)
assert MTCNN(device='cpu').detect(np.zeros((160, 160, 3), dtype=np.uint8))[0] is None
assert cv2.imencode('.png', np.zeros((8, 8, 3), dtype=np.uint8))[0]

backend, device = __import__('sys').argv[1:]
if backend == 'parakeet':
    import nemo.collections.asr as nemo_asr
    assert hasattr(nemo_asr.models.ASRModel, 'restore_from')
    assert hasattr(nemo_asr.models.EncDecSpeakerLabelModel, 'restore_from')
else:
    import ctranslate2
    from faster_whisper import WhisperModel, BatchedInferencePipeline
if device == 'cuda':
    assert torch.cuda.is_available(), 'Torch cannot access a CUDA device'
    if backend == 'whisper':
        assert ctranslate2.get_cuda_device_count() > 0, 'CTranslate2 cannot access a CUDA device'
print('Runtime imports and local CPU operations passed')
"""


def check_environment(backend: str, device: str) -> dict:
    errors = []
    versions = {}
    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "opencv-python-headless",
        "facenet-pytorch",
        "transformers",
        "huggingface-hub",
        "funasr",
        "modelscope",
        "faster-whisper",
        "ctranslate2",
    ]
    if backend == "parakeet":
        packages.append("nemo-toolkit")
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing package: {package}")
    flavors = {
        versions[name].partition("+")[2]
        for name in ("torch", "torchvision", "torchaudio")
        if name in versions
    }
    if len(flavors) > 1:
        errors.append(
            "Torch, Torchvision and Torchaudio must use the same CPU/CUDA build"
        )
    opencv = []
    for package in (
        "opencv-python",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ):
        try:
            importlib.metadata.version(package)
            opencv.append(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if opencv:
        errors.append(
            f"Use only opencv-python-headless; conflicting cv2 packages: {opencv}"
        )
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            errors.append(f"Missing system executable: {executable}")

    checks = {}
    for name, command in (
        ("dependencies", [sys.executable, "-m", "pip", "--isolated", "check"]),
        ("runtime", [sys.executable, "-c", IMPORT_PROBE, backend, device]),
    ):
        completed = subprocess.run(command, capture_output=True, text=True)
        checks[name] = {
            "passed": completed.returncode == 0,
            "output": (completed.stdout + completed.stderr).strip(),
        }
        if completed.returncode:
            errors.append(f"{name} check failed; see its output")
    return {
        "passed": not errors,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "virtual_environment": sys.prefix != sys.base_prefix,
        "backend": backend,
        "device": device,
        "versions": versions,
        "checks": checks,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asr-backend", choices=("whisper", "parakeet"), default="whisper"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = check_environment(args.asr_backend, args.device)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
