"""Restore current skill instructions through Claude's native SessionStart.

Read-only: no model call, robot call, lifecycle transition or transcript edit.
This retains the original plugin's startup/resume/compaction behavior while
loading the canonical files and state from the portable workspace.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

# Native hooks may run with the host Python, before the workspace venv is
# activated. Keep this read-only restore path independent of tools/__init__
# and all robot/validation dependencies.
from .settings import settings

PART_CHARACTERS = 8000
PART_SLOTS = 6


def current_state(environ=None):
    config = settings()
    state = json.loads((Path(config["state_dir"]) / "state.json").read_text())
    if state.get("session_id") != config["session_id"]:
        raise ValueError("skill state belongs to another session")
    active = state.get("active_skill")
    if active is not None and (not isinstance(active, str)
        or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", active)
        or active == "behavior-v2-baseline"):
        raise ValueError("invalid active skill")
    return state


def build_context(state, workspace=None):
    workspace = Path(workspace or settings()["workspace"])
    active = state.get("active_skill")
    notice = "Current task_skill_state=" + json.dumps({
        "baseline": "behavior-v2-baseline", "active_skill": active,
        "revision": state.get("skill_revision", 0)}, ensure_ascii=False)
    parts = [notice,
        "The current lifecycle state is authoritative. Historical skill text does not activate a skill. "
        "Resume the parent task using current observations; images and points in a summary are historical."]
    for name in ["behavior-v2-baseline"] + ([active] if active else []):
        text = (workspace / "skills" / name / "SKILL.md").read_text().strip()
        parts.append(f'<activated_skill name="{name}">\n{text}\n</activated_skill>')
    parts.append(notice)
    return "\n\n".join(parts)


def context_parts(context):
    """Keep each native hook below its observed 10,000-character spill cap.

    Split only between lines; concatenating the parts reproduces the exact
    instruction text. Each of the six project hooks emits one numbered part.
    This does not summarize instructions or edit the conversation.
    """
    parts, current = [], ""
    for line in context.splitlines(keepends=True):
        if len(line) > PART_CHARACTERS:
            raise ValueError("A skill instruction line exceeds the native hook inline budget")
        if len(current) + len(line) > PART_CHARACTERS:
            parts.append(current)
            current = ""
        current += line
    if current:
        parts.append(current)
    if len(parts) > PART_SLOTS:
        raise ValueError("Skill instructions exceed the configured native hook parts; read the full files")
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int, choices=range(1, PART_SLOTS + 1), required=True)
    args = parser.parse_args()
    # Consume the native hook input; BEHAVIOR_SESSION_ID, not Claude's UUID,
    # identifies the robot workflow that must be restored.
    json.load(sys.stdin)
    try:
        state = current_state()
        parts = context_parts(build_context(state))
        if args.part > len(parts):
            print("{}")
            return
        context = (
            f'<workspace_skill_context part="{args.part}" total="{len(parts)}">\n'
            + parts[args.part - 1] + "\n</workspace_skill_context>")
    except (OSError, ValueError, TypeError) as exc:
        context = (
            f"Workspace skill state could not be restored: {exc}. "
            "Do not infer an active skill from history. Recover the matching robot session state "
            "from skills/active.json and read the baseline and current skill files before acting.")
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "SessionStart", "additionalContext": context}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
