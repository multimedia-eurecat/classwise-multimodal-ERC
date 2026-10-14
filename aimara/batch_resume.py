"""Stable identities for configuration-aware batch prediction caches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path


# Bump when prediction/manifest semantics change incompatibly.
CACHE_VERSION = 2
MANIFEST_NAME = "resume_manifest.json"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _artifact_identity(path: Path) -> dict:
    """Hash explicit files; inventory cache directories without reading all weights."""
    path = path.resolve()
    if path.is_file():
        return {"path": str(path), "sha256": file_digest(path)}
    if not path.is_dir():
        raise FileNotFoundError(f"resume configuration artifact not found: {path}")
    files = []
    for entry in sorted(path.rglob("*")):
        relative = entry.relative_to(path)
        # Locks, download temporaries and Python bytecode are not model inputs.
        if (
            any(
                part.startswith(".") or part == "__pycache__" for part in relative.parts
            )
            or entry.suffix in {".pyc", ".lock", ".tmp", ".incomplete"}
            or not entry.is_file()
        ):
            continue
        stat = entry.stat()
        files.append([str(relative), stat.st_size, stat.st_mtime_ns])
    return {"path": str(path), "files": files}


def snapshot_configuration(config: Mapping) -> dict:
    """Freeze configs and artifact identities once, before processing batch videos."""

    def normalize(value):
        if isinstance(value, Path):
            return _artifact_identity(value)
        if is_dataclass(value) and not isinstance(value, type):
            return normalize(asdict(value))
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("resume configuration keys must be strings")
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(
            f"unsupported resume configuration value: {type(value).__name__}"
        )

    normalized = normalize(config)
    # Reject non-finite numbers before any output is touched.
    configuration_digest(normalized)
    return normalized


def configuration_digest(config: dict) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
