"""Run this checkout's evaluator while preserving caller-relative CLI paths."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aimara.optimized_evaluation.run_evaluation import main

if __name__ == "__main__":
    raise SystemExit(main())
