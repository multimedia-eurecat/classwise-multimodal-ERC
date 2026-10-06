"""Run ROB-to-ERC inference over every video in a directory."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .batch import run_batch
from .run_pipeline import add_runtime_arguments, build_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    add_runtime_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pipeline = build_pipeline(args)
    rows, failures = run_batch(
        pipeline,
        args.video_dir,
        args.output_dir,
        resume=args.resume,
    )
    print(
        f"Finished: {len(rows)} segment predictions, {len(failures)} failed videos. "
        f"Results: {args.output_dir / 'predictions.csv'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
