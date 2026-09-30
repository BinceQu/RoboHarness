#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
Usage: scripts/reproduce_task.sh taskNN [runner options]

Examples:
  scripts/reproduce_task.sh task01 --gpu 5
  scripts/reproduce_task.sh task08 --gpu 5 --instances 301,304

The wrapper uses only ROOT/.local/session-config.json and defaults to the
1507* HTTP range. It never edits a global Claude or Codex configuration.
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
INDEX=$((10#${BASH_REMATCH[1]}))
PORT=$((15070 + INDEX))
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

has_port=0
for arg in "$@"; do
  [[ $arg == --port || $arg == --port=* ]] && has_port=1
done
if [[ $has_port -eq 0 ]]; then
  set -- --port "$PORT" "$@"
fi

exec "$ROOT/run.sh" --task "$TASK" --config "$CONFIG" "$@"
