"""Load one readable task skill; the tabletop baseline always remains active."""

import json
from pathlib import Path
from tools.util.settings import write_json


def activate_skill(root, arguments):
    name = arguments.get("name", "")
    available = sorted(
        p.parent.name for p in (Path(root) / "skills").glob("*/SKILL.md")
    )
    if not name:
        return {"ok": True, "available_skills": available}
    if name not in available:
        raise ValueError("Unknown skill; use one of " + ", ".join(available))
    path = Path(root) / "skills" / name / "SKILL.md"
    state_path = Path(root) / ".runtime/state.json"
    state = json.loads(state_path.read_text())
    state["active_skill"] = None if name == "robodojo-baseline" else name
    write_json(state_path, state)
    return {
        "ok": True,
        "baseline": "robodojo-baseline",
        "active_skill": state["active_skill"],
        "skill_path": str(path),
        "instructions": path.read_text(),
    }


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("activate_skill", activate_skill)
