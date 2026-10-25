"""Read and activate one task procedure: python tools/activate_skill.py '{"name":"pick-up-object"}'.

The full selected SKILL.md is returned. Activating a different task skill replaces
the previous one; behavior-v2-baseline remains active. No robot motion occurs.
"""
from tools.util.skills import main, skill_path, save_state

NAME = "activate_skill"


def activate(robot, state, arguments):
    """Return a procedure's actual text and replace the current task skill."""
    name = arguments.get("name")
    previous = state.get("active_skill")
    if not name:
        return {"active_skill": previous,
            "skills": [path.parent.name for path in sorted((robot.workspace / "skills").glob("*/SKILL.md"))
                       if path.parent.name != "behavior-v2-baseline"], "images": []}
    path = skill_path(robot, name)
    state["active_skill"] = name
    notice = save_state(robot, state)
    return {"previous_skill": previous, "task_skill_state": notice,
            "reason": arguments.get("reason", ""), "images": [],
            "skill_path": str(path), "instructions": path.read_text()}

if __name__ == "__main__":
    main(NAME, activate)
