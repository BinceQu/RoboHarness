#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any


EMBODIED_PREFIXES = (
    "mcp__behavior-v2__",
    "mcp__plugin_embodied-claude-code_behavior-v2__",
    "mcp__behavior-robot__",
    "mcp__plugin_embodied-claude-code_behavior-robot__",
)
TOOL_FUNCTION_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_HOOKS_DIR = str(Path(__file__).resolve().parent)
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


def is_embodied_tool(tool_name: str) -> bool:
    return any(
        tool_name.startswith(prefix)
        and TOOL_FUNCTION_RE.fullmatch(tool_name[len(prefix):]) is not None
        for prefix in EMBODIED_PREFIXES
    )


def decision(payload: dict[str, Any]) -> dict[str, Any] | None:
    import skill_router

    tool_name = str(payload.get("tool_name") or "")
    if is_embodied_tool(tool_name):
        return None
    if skill_router.is_native_skill_tool(tool_name):
        plugin_root = Path(
            os.environ.get("CLAUDE_PLUGIN_ROOT")
            or os.environ.get("EMBODIED_PLUGIN_ROOT")
            or os.environ.get("PLUGIN_ROOT")
            or Path(__file__).resolve().parents[1]
        )
        if skill_router.extract_invoked_skill_name(payload, plugin_root):
            return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "Embodied strict mode allows only the behavior-v2 MCP tools "
                "and this plugin's native task Skills."
            ),
        }
    }


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    if isinstance(payload, dict):
        result = decision(payload)
        if result is None and is_embodied_tool(str(payload.get("tool_name") or "")):
            import skill_router
            if os.environ.get("BEHAVIOR_SESSION_ID"):
                try:
                    with skill_router.skill_state.locked_state() as writer:
                        if not writer.state.monitor_synced:
                            synced = skill_router.publish_user_prompt(
                                "", invoked=writer.state.active_task_skill or ""
                            )
                            writer.mark_monitor_synced(synced)
                except (OSError, ValueError) as exc:
                    print(f"[embodied-claude-code] Skill monitor sync: {exc}", file=sys.stderr)
    else:
        result = decision({})
    if result is not None:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
