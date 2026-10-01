#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
Usage: scripts/reproduce_task.sh taskNN [runner options]

Examples:
  scripts/reproduce_task.sh task01 --gpu 5
  scripts/reproduce_task.sh task08 --gpu 5 --instances 301,304

The wrapper uses ROOT/.local/session-config.json. Set task_ports there to
choose all three listeners. It never edits global Claude or Codex configuration.
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
if [[ $1 == '-h' || $1 == '--help' ]]; then
  usage
  exit 0
fi
TASK=$1
shift
if [[ ! $TASK =~ ^task([0-9]{2})$ ]]; then
  echo "Invalid task: $TASK (expected task00..task09)" >&2
  exit 2
fi
CONFIG=${ROBOHARNESS_SESSION_CONFIG:-$ROOT/.local/session-config.json}
mkdir -p "$(dirname "$CONFIG")"
if [[ ! -f $CONFIG ]]; then
  if [[ -f $ROOT/configs/local.json ]]; then
    cp --no-clobber "$ROOT/configs/local.json" "$CONFIG"
  else
    echo "Missing session config: $CONFIG" >&2
    exit 1
  fi
fi

exec "$ROOT/run.sh" --task "$TASK" --config "$CONFIG" "$@"
