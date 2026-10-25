"""Shared skill-file validation, state persistence and CLI/history handling."""
import argparse
import json
import re

from .files import atomic_json, load_json
from .history import CallHistory, record_failure
from .robot import Robot
from .receipt import model_receipt_text


def skill_path(robot, name):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("Provide an exact skill directory name")
    if name == "behavior-v2-baseline":
        raise ValueError("The baseline is always active")
    path = robot.workspace / "skills" / name / "SKILL.md"
    if not path.is_file():
        raise ValueError(f"Unknown skill: {name}")
    return path


def save_state(robot, state):
    state["skill_revision"] = state.get("skill_revision", 0) + 1
    atomic_json(robot.state_dir / "state.json", state)
    notice = {"baseline": "behavior-v2-baseline", "active_skill": state["active_skill"],
              "revision": state["skill_revision"]}
    atomic_json(robot.workspace / "skills/active.json", notice)
    return notice


def run(name, handler, arguments):
    robot = Robot()
    with robot.exclusive():
        history = CallHistory(robot, name, arguments)
        try:
            if set(arguments) - {"name", "reason"}:
                raise ValueError("Only name and reason are accepted")
            result = handler(robot, load_json(robot.state_dir / "state.json", {}), arguments)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}", "images": []}
        return history.finish(result, result)


def main(name, handler):
    parser = argparse.ArgumentParser(description=handler.__doc__)
    parser.add_argument("arguments", nargs="?", default="{}")
    options = parser.parse_args()
    arguments = options.arguments
    try:
        arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Skill input must be a JSON object")
        result = run(name, handler, arguments)
    except Exception as exc:
        result = record_failure(name, arguments, {"error": str(exc), "images": []})
    print(model_receipt_text(result), flush=True)
