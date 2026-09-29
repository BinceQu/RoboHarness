#!/usr/bin/env python3
"""Commit native Skill activations after success; never infer task success."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill_router
from embodied_claude_code import skill_state


def handle(payload: dict[str, Any], plugin_root: Path) -> dict[str, Any] | None:
    event = payload.get("hook_event_name")
    if event == "UserPromptExpansion":
        if payload.get("expansion_type") != "slash_command":
            return None
        name = skill_router.extract_invoked_skill_name(
            {"tool_input": {"skill": payload.get("command_name")}}, plugin_root
        )
        if not name:
            return None
        # Expansion can still be blocked by another hook or fail. Ask for an
        # explicit successful MCP activation rather than committing prematurely.
        context = (
            f"The user requested task Skill {name}. Before following its steps, "
            f"call activate_skill with name=\"{name}\" to commit its activation. "
            "This expansion alone does not change the active task Skill."
        )
    elif event == "PostToolUse":
        if not skill_router.is_native_skill_tool(str(payload.get("tool_name") or "")):
            return None
        name = skill_router.extract_invoked_skill_name(payload, plugin_root)
        response = payload.get("tool_response")
        if not name or (isinstance(response, dict) and (
            response.get("isError") or response.get("is_error")
            or response.get("ok") is False or response.get("success") is False
        )):
            return None
        with skill_state.locked_state() as writer:
            state = writer.set_active(name)
            synced = skill_router.publish_user_prompt("", plugin_root=plugin_root, invoked=name)
            writer.mark_monitor_synced(synced)
        context = skill_state.state_notice(state)
    else:
        return None
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": context}}


def main() -> None:
    event = "PostToolUse"
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            return
        event = payload.get("hook_event_name", event)
        root = Path(os.environ.get("CLAUDE_PLUGIN_ROOT")
                    or os.environ.get("EMBODIED_PLUGIN_ROOT")
                    or Path(__file__).resolve().parents[1])
        result = handle(payload, root)
    except (OSError, ValueError) as exc:
        # The native document may have loaded, but workflow activation did not.
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": (
                "Task Skill activation could not be recorded. Before following its "
                "steps, call activate_skill and check its successful lifecycle result."
            ),
        }}))
        print(f"[embodied-claude-code] Skill lifecycle: {exc}", file=sys.stderr)
        return
    if result is not None:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
