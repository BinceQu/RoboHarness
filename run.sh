#!/usr/bin/env bash
set -euo pipefail
ROBOHARNESS_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROBOHARNESS_PROJECT_ROOT"
exec "${ROBOHARNESS_PYTHON:-python3}" -m roboharness "$@"
