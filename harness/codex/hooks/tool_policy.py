#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any


EMBODIED_PREFIX = "mcp__behavior_v2__"
TOOL_FUNCTION_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_HOOKS_DIR = str(Path(__file__).resolve().parent)
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


def is_embodied_tool(tool_name: str) -> bool:
    if not tool_name.startswith(EMBODIED_PREFIX):
        return False
    function_name = tool_name[len(EMBODIED_PREFIX) :]
    return TOOL_FUNCTION_RE.fullmatch(function_name) is not None


def decision(payload: dict[str, Any]) -> dict[str, Any] | None:
    import skill_router

    tool_name = str(payload.get("tool_name") or "")
    if is_embodied_tool(tool_name) or skill_router.is_native_skill_tool(tool_name):
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "Embodied strict mode allows only the behavior-v2 MCP tools "
                "and Codex native skill tools."
            ),
        }
    }


def _publish_invoked_skill(payload: dict[str, Any]) -> None:
    """模型 invoke 了一个任务 skill 时，把名字打到 Agent Monitor。"""
    import skill_router

    tool_name = str(payload.get("tool_name") or "")
    if not skill_router.is_native_skill_tool(tool_name):
        return
    plugin_root = Path(
        os.environ.get("PLUGIN_ROOT", Path(__file__).resolve().parents[1])
    )
    invoked = skill_router.extract_invoked_skill_name(payload, plugin_root)
    if not invoked:
        return
    skill_router.save_stamped_active_task_skill(invoked)
    skill_router.publish_user_prompt(
        "",
        plugin_root=plugin_root,
        invoked=invoked,
    )


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    if isinstance(payload, dict):
        _publish_invoked_skill(payload)
        result = decision(payload)
    else:
        result = decision({})
    if result is not None:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
