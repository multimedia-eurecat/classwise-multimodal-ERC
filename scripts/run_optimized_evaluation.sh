#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${EVAL_LAUNCHER_PYTHON:-python3}" "$REPO_ROOT/scripts/run_optimized_evaluation.py" "$@"
