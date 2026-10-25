"""Leave the named task procedure and return to the parent task under the baseline.

Example: python tools/deactivate_skill.py '{"name":"pick-up-object","reason":"object held and stowed"}'
Deactivation changes instructions only: it does not move or open a gripper.
"""
from tools.util.skills import main, skill_path, save_state

NAME = "deactivate_skill"


def deactivate(robot, state, arguments):
    """Leave this named procedure without changing any robot target/effort."""
    name = arguments.get("name")
    skill_path(robot, name)
    previous = state.get("active_skill")
    if previous not in (None, name):
        raise ValueError(f"Active skill is {previous}, not {name}")
    state["active_skill"] = None
    notice = save_state(robot, state)
    return {"previous_skill": previous, "task_skill_state": notice,
            "reason": arguments.get("reason", ""), "images": [],
            "instructions": "Continue the parent task under the baseline. Deactivation does not move the robot or release a grasp."}

if __name__ == "__main__":
    main(NAME, deactivate)
