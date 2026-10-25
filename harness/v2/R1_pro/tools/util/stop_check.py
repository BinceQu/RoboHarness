"""One native completion self-check per robot-call state and Claude session.

No robot or model calls, task judging, conversation edits, or restart loop.
Claude receives ordinary Stop feedback and decides whether work remains under
the parent task. A second stop without a new workspace call is always allowed.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import re
import sys

from .files import atomic_json, load_json
from .settings import settings


FEEDBACK = (
    "Before ending, check the parent task's completion and stop conditions "
    "against current observations. A statement of the next action is not its "
    "execution. If work remains and no permitted stop condition applies, carry "
    "out the next action with workspace tools. Otherwise finish with the "
    "observed outcome and reason. Do not reset, repeat a completed action, or "
    "make an unnecessary robot call merely to satisfy this check."
)


def process(event):
    if event.get("hook_event_name") != "Stop":
        return {}
    claude_session = event.get("session_id", "")
    if not isinstance(claude_session, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", claude_session):
        raise ValueError("invalid Claude session id")
    config = settings()
    state_dir = Path(config["state_dir"]).resolve()
    if not state_dir.is_relative_to(Path(config["workspace"]).resolve()):
        raise ValueError("stop-check state must be inside this workspace")
    state = load_json(state_dir / "state.json")
    if not isinstance(state, dict) or state.get("session_id") != config["session_id"]:
        raise ValueError("stop-check state belongs to another robot session")
    counter = state.get("counter")
    if type(counter) is not int or counter < 0:
        raise ValueError("invalid workspace call counter")
    stamp = {"robot_session": config["session_id"], "counter": counter}
    directory = state_dir / "native-stop"
    directory.mkdir(exist_ok=True)
    marker = directory / f"{claude_session}.json"
    # Stop hooks can be retried. Reserve the feedback before returning it so
    # the same robot-call state cannot cause an unbounded continuation loop.
    with (directory / f"{claude_session}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if load_json(marker) == stamp:
            return {}
        atomic_json(marker, stamp)
    return {"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext": FEEDBACK}}


def main():
    try:
        output = process(json.load(sys.stdin))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        # A broken optional self-check must not trap the native CLI in a loop.
        print(f"Workspace completion self-check unavailable: {exc}", file=sys.stderr)
        output = {}
    print(json.dumps(output))


if __name__ == "__main__":
    main()
