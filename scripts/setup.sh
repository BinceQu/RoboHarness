#!/usr/bin/env bash
# Create separate agent, interface and evaluator environments. Dataset access
# and the NVIDIA EULA are handled by the pinned upstream interactive installer.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_BIN=${ROBOHARNESS_SETUP_PYTHON:-python3.11}
mode=${1:-all}
if [[ "$mode" != all && "$mode" != agent ]]; then
  echo 'Usage: scripts/setup.sh [all|agent] [upstream setup flags...]' >&2
  exit 2
fi
if [[ $# -gt 0 ]]; then shift; fi
git submodule update --init --recursive
"$PYTHON_BIN" -m venv harness/claude_code/.venv
harness/claude_code/.venv/bin/python -m pip install -r requirements/agent.txt -e harness/claude_code -e harness/codex
if [[ "$mode" == agent ]]; then exit 0; fi
"$PYTHON_BIN" -m venv .venv-interface
.venv-interface/bin/python -m pip install -c requirements/interface.txt torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv-interface/bin/python -m pip install -r requirements/interface.txt
# The custom robot assets ship here; cuRobo's demo meshes and videos are not
# needed. Put the installed ninja executable on PATH for CUDA extension builds.
PATH="$ROOT/.venv-interface/bin:$PATH" GIT_LFS_SKIP_SMUDGE=1 \
  .venv-interface/bin/python -m pip install --no-build-isolation -c requirements/interface.txt 'nvidia-curobo @ git+https://github.com/NVlabs/curobo@cbaf7d32436160956dad190a9465360fad6aba73'
"$PYTHON_BIN" -m venv .venv-evaluator
.venv-evaluator/bin/python -m pip install 'setuptools>=71,<81' wheel
(
  source .venv-evaluator/bin/activate
  # Upstream uses CONDA_PREFIX for package cleanup. A venv must not inherit
  # the caller's conda prefix and accidentally edit that separate environment.
  unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
  cd BEHAVIOR
  bash setup.sh --bddl --omnigibson --joylo --eval --confirm-no-conda "$@"
)
.venv-evaluator/bin/python -m pip install warp-lang==1.12.0
"$PYTHON_BIN" -c 'from roboharness.assets import prepare_robot_asset; print(prepare_robot_asset())'
"$PYTHON_BIN" scripts/setup_perception.py
echo 'Environments ready. Configure data and model access as described in docs/setup.md.'
