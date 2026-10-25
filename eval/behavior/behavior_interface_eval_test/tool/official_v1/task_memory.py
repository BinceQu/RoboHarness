"""Static public task fields owned by the isolated official_v1 capture tools."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_TASK_DATA_PATH = (
    Path(__file__).resolve().parents[3] / "docs" / "challenge" / "task_data.json"
)


@lru_cache(maxsize=256)
def _task_definition(task_name: str) -> tuple[str, str, tuple[str, ...], str]:
    payload = json.loads(_TASK_DATA_PATH.read_text(encoding="utf-8"))
    matches = [
        source
        for source in payload.get("tasks", [])
        if isinstance(source, dict)
        and str(source.get("id") or "").strip() == task_name
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown public evaluator task {task_name!r}")
    source = matches[0]
    display_name = str(source.get("name") or "").strip()
    objective = str(source.get("instruction") or "").strip()
    raw_conditions = source.get("goal_conditions")
    if not objective:
        raise ValueError(f"public task {task_name!r} has no objective")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError(f"public task {task_name!r} has no BDDL conditions")
    conditions = tuple(str(item).strip() for item in raw_conditions)
    if not all(conditions):
        raise ValueError(f"public task {task_name!r} has an empty BDDL condition")
    instruction = "\n\n".join(
        (
            f"Task: {display_name or task_name.replace('_', ' ').title()}.",
            f"Task objective: {objective}",
            "Authoritative success conditions (all must be true; these come "
            "from the task BDDL):\n"
            + "\n".join(f"- {condition}" for condition in conditions),
        )
    )
    return objective, instruction, conditions, str(
        source.get("instruction_source") or ""
    ).strip()


def task_memory_fields(task_name: str) -> dict[str, Any]:
    normalized = str(task_name or "").strip()
    if not normalized:
        raise ValueError("task_name is required")
    objective, instruction, conditions, instruction_source = _task_definition(
        normalized
    )
    return {
        "task": normalized,
        "task_objective": objective,
        "instruction": instruction,
        "task_instruction": instruction,
        "bddl_conditions": list(conditions),
        "goal_conditions": list(conditions),
        "instruction_source": instruction_source,
        "memory_source": "official_public_task_metadata",
        "bddl_conditions_source": "public_metadata_synced_from_task_bddl",
        "bddl_live_state_available": False,
    }
