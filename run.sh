#!/usr/bin/env bash
set -euo pipefail
ROBOHARNESS_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROBOHARNESS_PROJECT_ROOT"
# The evaluation runner lives in eval/runner (Python package `runner`).
export PYTHONPATH="$ROBOHARNESS_PROJECT_ROOT/eval${PYTHONPATH:+:$PYTHONPATH}"
exec "${ROBOHARNESS_PYTHON:-python3}" -m runner "$@"
