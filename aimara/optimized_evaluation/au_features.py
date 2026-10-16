"""Adapt py-feat's path API to face crops; surface operational errors."""
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image


def extract_crop_aus(crop, detector, dimension=20):
    with tempfile.TemporaryDirectory(prefix="aimara-au-") as directory:
        path = Path(directory) / "face.png"
        Image.fromarray(crop).save(path)
        result = detector.detect_image([str(path)], batch_size=1)
    if result is None or len(result) == 0:
        return None
    columns = [c for c in result.columns if c.startswith("AU")]
    if len(columns) != dimension:
        raise ValueError(f"Expected {dimension} AU columns, got {columns}")
    if len(result) > 1:
        areas = result["FaceRectWidth"] * result["FaceRectHeight"]
        result = result.iloc[[int(np.argmax(areas.to_numpy()))]]
    values = result[columns].to_numpy(dtype=np.float32)[0]
    if np.isnan(values).all():
        return None
    if not np.isfinite(values).all():
        raise ValueError("Non-finite AU values")
    return values


def pool_aus(values, dimension=20):
    if not values:
        return np.zeros(dimension * 3, dtype=np.float32)
    stack = np.stack(values)
    if stack.ndim != 2 or stack.shape[1] != dimension:
        raise ValueError("Invalid per-frame AU dimensions")
    delta = np.diff(stack, axis=0).mean(axis=0) if len(stack) > 1 else np.zeros(dimension)
    return np.concatenate((stack.mean(axis=0), stack.std(axis=0), delta)).astype(np.float32)
