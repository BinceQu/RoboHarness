"""Finish a task skill without loading an unrelated replacement."""

import json
from pathlib import Path
from tools.util.settings import write_json


def deactivate_skill(root, arguments):
    path = Path(root) / ".runtime/state.json"
    state = json.loads(path.read_text())
    name = arguments.get("name")
    if name and name != state["active_skill"]:
        return {
            "ok": False,
            "error": "Named skill is not active",
            "active_skill": state["active_skill"],
        }
    previous = state["active_skill"]
    state["active_skill"] = None
    write_json(path, state)
    return {
        "ok": True,
        "deactivated": previous,
        "active_skill": None,
        "baseline": "robodojo-baseline",
    }


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("deactivate_skill", deactivate_skill)
