"""Audio decoding helpers for offline AIMARA processing."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import numpy as np


def decode_audio(
    media_path: Union[str, Path],
    *,
    sample_rate: int = 16000,
    ffmpeg: str = "ffmpeg",
) -> "np.ndarray":
    """Decode the first audio stream as mono float32 PCM samples."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to decode audio") from exc

    path = Path(media_path)
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"FFmpeg executable not found: {ffmpeg}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg could not decode {path}: {detail}") from exc

    audio = np.frombuffer(completed.stdout, dtype="<f4").copy()
    if audio.size == 0:
        raise ValueError(f"No audio samples decoded from {path}")
    return audio
