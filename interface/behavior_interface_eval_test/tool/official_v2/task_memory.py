"""Static public task memory for the isolated official_v2 profile."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


_TASK_DATA_ENV = "BEHAVIOR_EVAL_TEST_TASK_DATA_PATH"
_TASK_DATA_RELATIVE_PATH = Path("docs/challenge/task_data.json")
_MEMORY_SOURCE = "official_public_task_metadata"


def _default_task_data_path() -> Path:
    return Path(__file__).resolve().parents[3] / _TASK_DATA_RELATIVE_PATH


def _task_data_path(path: str | os.PathLike[str] | None = None) -> Path:
    configured = path or os.environ.get(_TASK_DATA_ENV)
    return (
        Path(configured).expanduser().resolve()
        if configured
        else _default_task_data_path().resolve()
    )


def _nonempty_text(value: Any, *, field: str, task_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(
            f"public task metadata for {task_name!r} has no {field}"
        )
    return text


@dataclass(frozen=True)
class OfficialTaskMemory:
    """Immutable public task definition exposed by the test interface."""

    task: str
    task_index: int
    display_name: str
    task_objective: str
    instruction: str
    bddl_conditions: tuple[str, ...]
    instruction_source: str

    def public_fields(self) -> dict[str, Any]:
        conditions = list(self.bddl_conditions)
        return {
            "task": self.task,
            "task_index": self.task_index,
            "display_name": self.display_name,
            "task_objective": self.task_objective,
            "instruction": self.instruction,
            "task_instruction": self.instruction,
            "bddl_conditions": conditions,
            "goal_conditions": list(conditions),
            "instruction_source": self.instruction_source,
            "memory_source": _MEMORY_SOURCE,
            "bddl_conditions_source": "public_metadata_synced_from_task_bddl",
            "bddl_live_state_available": False,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "task_definition_available": True,
            "bddl_conditions": len(self.bddl_conditions),
            "bddl_live_state_available": False,
            "source": _MEMORY_SOURCE,
        }

    def raw(self) -> dict[str, Any]:
        return {**self.public_fields(), "summary": self.summary()}

    def text(self) -> str:
        return (
            self.instruction
            + "\n\nBDDL condition status: definition only; the evaluator "
            "does not expose live predicate satisfaction to this interface."
        )


def _format_instruction(
    *,
    display_name: str,
    objective: str,
    conditions: tuple[str, ...],
) -> str:
    return "\n\n".join(
        (
            f"Task: {display_name}.",
            f"Task objective: {objective}",
            "Authoritative success conditions (all must be true; these come "
            "from the task BDDL):\n"
            + "\n".join(f"- {condition}" for condition in conditions),
        )
    )


@lru_cache(maxsize=256)
def _load_task_memory_cached(
    task_name: str,
    task_data_path: str,
) -> OfficialTaskMemory:
    path = Path(task_data_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"public task metadata is unavailable at {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"public task metadata is invalid JSON: {path}") from exc

    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        raise ValueError(f"public task metadata has no tasks list: {path}")

    matches: list[tuple[int, dict[str, Any]]] = []
    for index, source in enumerate(tasks):
        if not isinstance(source, dict):
            continue
        if str(source.get("id") or "").strip() == task_name:
            matches.append((index, source))
    if not matches:
        raise KeyError(f"unknown public evaluator task {task_name!r}")
    if len(matches) != 1:
        raise ValueError(f"duplicate public evaluator task {task_name!r}")

    task_index, source = matches[0]
    objective = _nonempty_text(
        source.get("instruction"),
        field="task objective",
        task_name=task_name,
    )
    display_name = str(source.get("name") or "").strip()
    if not display_name:
        display_name = task_name.replace("_", " ").title()

    raw_conditions = source.get("goal_conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError(
            f"public task metadata for {task_name!r} has no BDDL conditions"
        )
    conditions = tuple(
        _nonempty_text(
            condition,
            field="BDDL condition",
            task_name=task_name,
        )
        for condition in raw_conditions
    )
    return OfficialTaskMemory(
        task=task_name,
        task_index=task_index,
        display_name=display_name,
        task_objective=objective,
        instruction=_format_instruction(
            display_name=display_name,
            objective=objective,
            conditions=conditions,
        ),
        bddl_conditions=conditions,
        instruction_source=str(source.get("instruction_source") or "").strip(),
    )


def load_official_task_memory(
    task_name: str,
    *,
    task_data_path: str | os.PathLike[str] | None = None,
) -> OfficialTaskMemory:
    """Load one task definition from public static metadata."""
    normalized = str(task_name or "").strip()
    if not normalized:
        raise ValueError("task_name is required")
    return _load_task_memory_cached(
        normalized,
        str(_task_data_path(task_data_path)),
    )


def task_memory_fields(task_name: str) -> dict[str, Any]:
    """Return copy-safe task fields for capture-tool memory payloads."""
    return load_official_task_memory(task_name).public_fields()


def install_official_task_memory(
    server: Any,
    task_name: str,
    *,
    dynamic_memory: Any | None = None,
) -> OfficialTaskMemory:
    """Bind profile-owned memory accessors to the compatibility HTTP server."""
    memory = load_official_task_memory(task_name)
    return bind_official_task_memory(
        server,
        memory,
        dynamic_memory=dynamic_memory,
    )


def bind_official_task_memory(
    server: Any,
    memory: OfficialTaskMemory,
    *,
    dynamic_memory: Any | None = None,
) -> OfficialTaskMemory:
    """Bind an existing task-memory object without replacing its identity."""

    if dynamic_memory is None:
        base_raw, base_text = memory.raw, memory.text
        base_summary = memory.summary
    else:

        def base_raw() -> dict[str, Any]:
            dynamic_fields = dynamic_memory.memory_fields()
            return {
                **memory.public_fields(),
                **dynamic_fields,
                "summary": {
                    **memory.summary(),
                    **dynamic_memory.summary_fields(),
                },
            }

        def base_text() -> str:
            return memory.text() + "\n\n" + str(dynamic_memory.text())

        def base_summary() -> dict[str, Any]:
            return {**memory.summary(), **dynamic_memory.summary_fields()}

    # The official_v2 package is deliberately isolated from the production
    # ``behavior_interface`` package.  Dynamic map state, when needed by a
    # compatibility UI, is supplied by that outer layer through
    # ``dynamic_memory``; this module only exposes evaluator-safe metadata.
    server.get_memory = base_raw
    server.get_memory_text = base_text
    server.get_memory_summary = base_summary
    return memory
