"""Stage instrumentation; the launcher measures process wall time."""
import atexit
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import sys
import time

PHASES = {}
DETAILS = {}


def synchronize():
    torch = sys.modules.get("torch")
    if os.environ.get("EVAL_DEVICE") != "cpu" and torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def phase(name):
    synchronize()
    started = time.perf_counter()
    try:
        yield
    finally:
        synchronize()
        PHASES[name] = PHASES.get(name, 0.0) + time.perf_counter() - started


def check_embeddings(embeddings, dimension):
    import numpy as np
    vectors = [v for turns in embeddings.values() for v in turns.values()]
    if not vectors:
        raise ValueError("No embeddings produced")
    for vector in vectors:
        if np.asarray(vector).shape != (dimension,) or not np.isfinite(vector).all():
            raise ValueError("Invalid embedding dimension or non-finite values")
    zeros = sum(not np.any(v) for v in vectors)
    DETAILS.update(utterances=len(vectors), zero_vectors=zeros, dimension=dimension)
    if zeros == len(vectors):
        raise ValueError("Every embedding is zero; extraction is not valid")


def write_metrics():
    target = os.environ.get("EVAL_STAGE_METRICS")
    if not target:
        return
    result = dict(phases_seconds=PHASES, details=DETAILS,
                  peak_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["cuda"] = dict(runtime=torch.version.cuda, device=torch.cuda.current_device(),
                              name=torch.cuda.get_device_name(), torch_version=torch.__version__)
    path = Path(target)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(path)


atexit.register(write_metrics)
