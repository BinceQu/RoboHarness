"""Evaluator-only, prohibited ground truth for grasp-confirmation tests.

This module must never be imported by the policy process or included in a
challenge submission path. It reads OmniGibson's private assisted-grasp state
and writes it only to the evaluator-side audit trace.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping

from behavior_interface_eval_test.grasp_confirmation_audit import (
    AUDIT_SCHEMA,
    TRUTH_DEFINITION,
    _JSONLWriter,
    _jsonable,
    privileged_oracle_trace_path,
)


def _mapping_value(value: Any, arm: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(arm)
    try:
        return value[arm]
    except (KeyError, TypeError, IndexError):
        return None


def _object_identity(obj: Any) -> dict[str, str | None]:
    if obj is None:
        return {"name": None, "prim_path": None}
    return {
        "name": str(getattr(obj, "name", "") or "") or None,
        "prim_path": str(getattr(obj, "prim_path", "") or "") or None,
    }


def _constraint_prim_path(constraint: Any, params: Any) -> str | None:
    if isinstance(params, Mapping):
        path = str(params.get("ag_joint_prim_path") or "").strip()
        if path:
            return path
    path = str(getattr(constraint, "prim_path", "") or "").strip()
    if path:
        return path
    get_path = getattr(constraint, "GetPath", None)
    if callable(get_path):
        try:
            path = str(get_path()).strip()
        except Exception:
            path = ""
    return path or None


def _constraint_valid(constraint: Any) -> bool:
    if constraint is None:
        return False
    is_valid = getattr(constraint, "IsValid", None)
    if callable(is_valid):
        try:
            return bool(is_valid())
        except Exception:
            return False
    explicit = getattr(constraint, "valid", None)
    return True if explicit is None else bool(explicit)


def _constraint_enabled(constraint: Any) -> bool | None:
    """Return authored jointEnabled for diagnostics, not truth classification."""
    if constraint is None:
        return None
    explicit = getattr(constraint, "enabled", None)
    if explicit is not None:
        return bool(explicit)
    get_attribute = getattr(constraint, "GetAttribute", None)
    if not callable(get_attribute):
        return None
    try:
        attribute = get_attribute("physics:jointEnabled")
        if attribute is None:
            return None
        is_valid = getattr(attribute, "IsValid", None)
        if callable(is_valid) and not bool(is_valid()):
            return None
        getter = getattr(attribute, "Get", None)
        value = getter() if callable(getter) else None
        return None if value is None else bool(value)
    except Exception:
        return None


def privileged_arm_truth(robot: Any, arm: str) -> dict[str, Any]:
    """Read OmniGibson's canonical private assisted-grasp state for one arm."""
    obj = _mapping_value(getattr(robot, "_ag_obj_in_hand", None), arm)
    constraint = _mapping_value(
        getattr(robot, "_ag_obj_constraints", None), arm
    )
    params = _mapping_value(
        getattr(robot, "_ag_obj_constraint_params", None), arm
    )
    object_present = obj is not None
    constraint_present = constraint is not None
    constraint_valid = _constraint_valid(constraint)
    constraint_live = bool(constraint_present and constraint_valid)
    established = bool(object_present and constraint_live)
    if established:
        phase = "established"
    elif object_present and not constraint_live:
        phase = "release_window_or_stale_object_reference"
    elif constraint_live and not object_present:
        phase = "orphan_constraint"
    else:
        phase = "idle"
    identity = _object_identity(obj)
    return {
        "established": established,
        "phase": phase,
        "object_reference_present": object_present,
        "object_name": identity["name"],
        "object_prim_path": identity["prim_path"],
        "constraint_reference_present": constraint_present,
        "constraint_valid": constraint_valid,
        "constraint_enabled": _constraint_enabled(constraint),
        "constraint_live": constraint_live,
        "constraint_prim_path": _constraint_prim_path(constraint, params),
        "freeze_gripper": bool(
            _mapping_value(getattr(robot, "_ag_freeze_gripper", None), arm)
        ),
    }


def _robot_arms(robot: Any) -> list[str]:
    ordered: list[str] = []
    for arm in ("left", "right", *tuple(getattr(robot, "arm_names", ()) or ())):
        arm = str(arm)
        if arm not in ordered:
            ordered.append(arm)
    return ordered


class PrivilegedGraspOracleTrace:
    """Evaluator-only recorder; never returns labels to the policy path."""

    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        self._writer = _JSONLWriter(self.path)
        self._sequence = 0

    def record(
        self,
        robot: Any,
        *,
        event: str,
        action: Any = None,
    ) -> dict[str, Any]:
        self._sequence += 1
        record = {
            "schema": AUDIT_SCHEMA,
            "record_type": "privileged_oracle_sample",
            "privileged_test_only": True,
            "policy_visible": False,
            "truth_definition": TRUTH_DEFINITION,
            "sequence": self._sequence,
            "time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "event": str(event),
            "robot": str(getattr(robot, "name", "") or ""),
            "action": _jsonable(action),
            "arms": {
                arm: privileged_arm_truth(robot, arm)
                for arm in _robot_arms(robot)
            },
        }
        self._writer.append(record)
        return record

    def close(self) -> None:
        self._writer.close()


def privileged_grasp_oracle_from_env() -> PrivilegedGraspOracleTrace | None:
    path = privileged_oracle_trace_path()
    return None if path is None else PrivilegedGraspOracleTrace(path)
