"""Behavior Challenge 2026 task metadata and model-facing instructions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final


_TASK_DATA_PATH = Path(__file__).resolve().parents[1] / "docs" / "challenge" / "task_data.json"

BEHAVIOR_CHALLENGE_YEAR: Final[int] = 2026


def _format_task_instruction(task: dict[str, Any]) -> str:
    sections = [f"Task: {str(task['name']).strip()}."]
    objective = str(task.get("instruction") or "").strip()
    if objective and task.get("instruction_source") != "bddl_generated":
        sections.append(f"Task objective: {objective}")

    goals = [str(item).strip() for item in task["goal_conditions"]]
    sections.append(
        "Authoritative success conditions (all must be true; these come from the task BDDL):\n"
        + "\n".join(f"- {condition}" for condition in goals)
    )

    guidance = [str(item).strip() for item in task.get("execution_guidance", [])]
    if guidance:
        sections.append(
            "Execution guidance (task-specific hints, not additional success conditions):\n"
            + "\n".join(f"- {hint}" for hint in guidance)
        )
    return "\n\n".join(sections)


def _load_tasks() -> tuple[dict[str, object], ...]:
    data = json.loads(_TASK_DATA_PATH.read_text(encoding="utf-8"))
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, source in enumerate(data.get("tasks", [])):
        task_name = str(source.get("id") or "").strip()
        if not task_name:
            raise ValueError(f"Challenge task without an id in {_TASK_DATA_PATH}")
        if task_name in seen:
            raise ValueError(f"Duplicate challenge task id {task_name!r} in {_TASK_DATA_PATH}")
        seen.add(task_name)

        goals = source.get("goal_conditions")
        if not isinstance(goals, list) or not goals or not all(
            isinstance(item, str) and item.strip() for item in goals
        ):
            raise ValueError(
                f"Challenge task {task_name!r} is missing complete BDDL goal conditions; "
                "run scripts/sync_challenge_task_instructions.py"
            )
        guidance = source.get("execution_guidance", [])
        if not isinstance(guidance, list) or not all(
            isinstance(item, str) and item.strip() for item in guidance
        ):
            raise ValueError(f"Challenge task {task_name!r} has invalid execution guidance")

        task: dict[str, object] = {
            "id": index,
            "name": task_name,
            "display_name": str(source.get("name") or task_name.replace("_", " ").title()),
            "instruction": _format_task_instruction(source),
            "task_objective": str(source.get("instruction") or "").strip(),
            "instruction_source": str(source.get("instruction_source") or ""),
            "goal_conditions": tuple(goals),
            "execution_guidance": tuple(guidance),
            "rooms": list(source.get("rooms") or []),
            "duration": source.get("duration"),
            "thumbnail": source.get("thumbnail"),
            "video": source.get("video"),
            "scene": source.get("scene_model"),
            "scene_model": source.get("scene_model"),
        }
        result.append(task)

    if len(result) != 100:
        raise ValueError(f"Expected 100 challenge tasks, found {len(result)} in {_TASK_DATA_PATH}")
    return tuple(result)


BEHAVIOR_CHALLENGE_2026_TASKS: Final[tuple[dict[str, object], ...]] = _load_tasks()

# Backward-compatible alias for existing interface code and scripts.
BEHAVIOR_CHALLENGE_2025_TASKS: Final[tuple[dict[str, object], ...]] = BEHAVIOR_CHALLENGE_2026_TASKS

_TASK_BY_NAME = {
    str(task["name"]): task
    for task in BEHAVIOR_CHALLENGE_2026_TASKS
}
_TASK_BY_ID = {
    int(task["id"]): task
    for task in BEHAVIOR_CHALLENGE_2026_TASKS
}


def challenge_tasks() -> list[dict[str, object]]:
    return [dict(task) for task in BEHAVIOR_CHALLENGE_2026_TASKS]


def challenge_task_names() -> list[str]:
    return [str(task["name"]) for task in BEHAVIOR_CHALLENGE_2026_TASKS]


def challenge_task_id(task_name: str) -> int | None:
    task = _TASK_BY_NAME.get(str(task_name))
    return None if task is None else int(task["id"])


def challenge_task_name(task_id: int) -> str | None:
    task = _TASK_BY_ID.get(int(task_id))
    return None if task is None else str(task["name"])


def challenge_task_scene(task_name: str) -> str | None:
    task = challenge_task_by_name(task_name)
    return None if task is None else str(task["scene"])


def challenge_task_instruction(task_name: str) -> str | None:
    task = challenge_task_by_name(task_name)
    return None if task is None else str(task["instruction"])


def challenge_task_goal_conditions(task_name: str) -> tuple[str, ...]:
    task = _TASK_BY_NAME.get(str(task_name))
    if task is None:
        raise KeyError(f"Unknown 2026 challenge task: {task_name}")
    return tuple(str(item) for item in task["goal_conditions"])


def challenge_task_execution_guidance(task_name: str) -> tuple[str, ...]:
    task = _TASK_BY_NAME.get(str(task_name))
    if task is None:
        raise KeyError(f"Unknown 2026 challenge task: {task_name}")
    return tuple(str(item) for item in task["execution_guidance"])


def challenge_task_by_name(task_name: str) -> dict[str, object] | None:
    task = _TASK_BY_NAME.get(str(task_name))
    return None if task is None else dict(task)


def challenge_task_by_id(task_id: int) -> dict[str, object] | None:
    task = _TASK_BY_ID.get(int(task_id))
    return None if task is None else dict(task)
