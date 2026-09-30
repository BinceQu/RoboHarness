#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
INTERFACE_PYTHON=${ROBOHARNESS_INTERFACE_PYTHON:-$ROOT/.venv-interface/bin/python}
AGENT_PYTHON=${ROBOHARNESS_AGENT_PYTHON:-$ROOT/harness/claude_code/.venv/bin/python}
export PYTHONPATH="$ROOT:$ROOT/interface:$ROOT/harness/claude_code/src"
python3 -c 'from roboharness.assets import prepare_robot_asset; prepare_robot_asset()'
"$INTERFACE_PYTHON" -m unittest discover -s tests -v
"$INTERFACE_PYTHON" -m unittest behavior_interface_eval_test.test_official_protocol behavior_interface_eval_test.test_official_rgbd_wrapper behavior_interface_eval_test.test_custom_robot_profile
"$INTERFACE_PYTHON" -m unittest behavior_interface_eval_test.test_native_asset_watches behavior_interface_eval_test.test_official_evaluator_entrypoint
cd harness/claude_code
export PYTHONPATH="$PWD/src:$PWD/tests"
"$AGENT_PYTHON" -m unittest discover -s tests
cd "$ROOT/harness/codex"
export PYTHONPATH="$PWD/src:$PWD/tests"
"$AGENT_PYTHON" -m unittest discover -s tests
