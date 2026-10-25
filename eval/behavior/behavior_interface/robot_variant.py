"""Canonical R1Pro 7DOF / 8DOF variant selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG_DIR = REPO_ROOT / "configs"
ROBOT_VARIANTS = {
    7: {
        "name": "7dof",
        "robot_type": "R1Pro",
        "config_path": ROBOT_CONFIG_DIR / "r1pro_high_force.yaml",
    },
    8: {
        "name": "8dof",
        "robot_type": "R1Pro8DOF",
        "config_path": ROBOT_CONFIG_DIR / "r1pro_8dof_high_force.yaml",
    },
}


def normalize_robot_dof(value: Any = None, *, default: int = 8) -> int:
    raw = os.environ.get("BEHAVIOR_ROBOT_DOF") if value is None else value
    if raw is None or str(raw).strip() == "":
        raw = default
    text = str(raw).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "7": 7,
        "7dof": 7,
        "r1pro": 7,
        "8": 8,
        "8dof": 8,
        "r1pro8dof": 8,
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(f"robot DOF must be 7 or 8, got {value!r}") from exc


def robot_type_for_dof(value: Any = None) -> str:
    return str(ROBOT_VARIANTS[normalize_robot_dof(value)]["robot_type"])


def robot_config_path_for_dof(value: Any = None) -> Path:
    return Path(ROBOT_VARIANTS[normalize_robot_dof(value)]["config_path"]).resolve()


def robot_dof_from_type(robot_type: str) -> int:
    requested = str(robot_type).strip().lower()
    for dof, spec in ROBOT_VARIANTS.items():
        if requested == str(spec["robot_type"]).lower():
            return int(dof)
    raise ValueError(f"unsupported robot type {robot_type!r}")


def robot_has_tool_roll(robot: Optional[Any], arm: Optional[str] = None) -> bool:
    """Return whether the loaded robot has the complete independent J8 contract."""
    if robot is None:
        return normalize_robot_dof() == 8
    arms = (str(arm).lower().strip(),) if arm is not None else ("left", "right")
    controllers = getattr(robot, "controllers", {}) or {}
    joints = getattr(robot, "joints", {}) or {}
    action_idx = getattr(robot, "controller_action_idx", {}) or {}
    return all(
        f"{side}_arm_joint8" in joints
        and (
            f"tool_roll_{side}" in controllers
            if controllers
            else f"tool_roll_{side}" in action_idx
        )
        and f"tool_roll_{side}" in action_idx
        for side in arms
    )
