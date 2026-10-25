"""Action and proprioception contracts for official evaluator robot profiles."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
CUSTOM_PROFILE_NAME = "r1pro_8dof_hf250"
CUSTOM_CONTRACT_PATH = (
    PACKAGE_DIR
    / "robot_profiles"
    / CUSTOM_PROFILE_NAME
    / "robot_contract.json"
)

STOCK_R1PRO_CONTRACT = {
    "profile": "stock_r1pro",
    "model": "r1pro",
    "arm_dof": 7,
    "action_dim": 23,
    "proprio_dim": 61,
    "base_footprint_radius_m": 0.42,
    "action_slices": {
        "base": [0, 3],
        "trunk": [3, 7],
        "arm_left": [7, 14],
        "gripper_left": [14, 15],
        "arm_right": [15, 22],
        "gripper_right": [22, 23],
    },
    "proprio_slices": {
        "base_qvel": [0, 3],
        "arm_left_qpos": [3, 10],
        "arm_left_qvel": [10, 17],
        "eef_left_pos": [17, 20],
        "eef_left_quat": [20, 24],
        "gripper_left_qpos": [24, 26],
        "gripper_left_qvel": [26, 28],
        "arm_right_qpos": [28, 35],
        "arm_right_qvel": [35, 42],
        "eef_right_pos": [42, 45],
        "eef_right_quat": [45, 49],
        "gripper_right_qpos": [49, 51],
        "gripper_right_qvel": [51, 53],
        "trunk_qpos": [53, 57],
        "trunk_qvel": [57, 61],
    },
}


@dataclass(frozen=True)
class RobotContract:
    profile: str
    model: str
    arm_dof: int
    action_dim: int
    proprio_dim: int
    base_footprint_radius_m: float
    action_slices: Mapping[str, slice]
    proprio_slices: Mapping[str, slice]


def _slice_map(payload: Mapping[str, list[int]]) -> dict[str, slice]:
    return {
        str(name): np.s_[int(bounds[0]) : int(bounds[1])]
        for name, bounds in payload.items()
    }


def _from_payload(payload: Mapping[str, object]) -> RobotContract:
    footprint_radius = float(payload["base_footprint_radius_m"])
    if not np.isfinite(footprint_radius) or not 0.0 < footprint_radius <= 2.0:
        raise ValueError("base_footprint_radius_m must be finite and in (0, 2]")
    return RobotContract(
        profile=str(payload["profile"]),
        model=str(payload["model"]),
        arm_dof=int(payload["arm_dof"]),
        action_dim=int(payload["action_dim"]),
        proprio_dim=int(payload["proprio_dim"]),
        base_footprint_radius_m=footprint_radius,
        action_slices=_slice_map(payload["action_slices"]),
        proprio_slices=_slice_map(payload["proprio_slices"]),
    )


def load_robot_contract(profile: str | os.PathLike[str] | None = None) -> RobotContract:
    selected = str(profile or os.environ.get("BEHAVIOR_EVAL_TEST_ROBOT_PROFILE", "stock_r1pro"))
    if selected in {"stock_r1pro", "r1pro", "7"}:
        return _from_payload(STOCK_R1PRO_CONTRACT)
    if selected in {CUSTOM_PROFILE_NAME, "8"}:
        path = CUSTOM_CONTRACT_PATH
    else:
        path = Path(selected).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _from_payload(payload)


ACTIVE_ROBOT_CONTRACT = load_robot_contract()
ROBOT_PROFILE = ACTIVE_ROBOT_CONTRACT.profile
ROBOT_MODEL = ACTIVE_ROBOT_CONTRACT.model
ARM_DOF = ACTIVE_ROBOT_CONTRACT.arm_dof
ACTION_DIM = ACTIVE_ROBOT_CONTRACT.action_dim
PROPRIO_DIM = ACTIVE_ROBOT_CONTRACT.proprio_dim
BASE_FOOTPRINT_RADIUS_M = ACTIVE_ROBOT_CONTRACT.base_footprint_radius_m
ACTION_SLICES = ACTIVE_ROBOT_CONTRACT.action_slices
PROPRIO_SLICES = ACTIVE_ROBOT_CONTRACT.proprio_slices
LOCKED_ARM_JOINT_INDEX = 7 if ARM_DOF == 8 else None
LOCKED_ARM_JOINT_VALUE = 0.0
LOCKED_TRUNK_YAW_INDEX = 3
LOCKED_TRUNK_YAW_VALUE = 0.0


def enforce_locked_arm_joint(values, *, copy: bool = True) -> np.ndarray:
    """Return an arm vector with the custom profile's J8 fixed at zero."""
    arm = np.array(values, copy=copy).reshape(-1)
    if LOCKED_ARM_JOINT_INDEX is not None and arm.size > LOCKED_ARM_JOINT_INDEX:
        arm[LOCKED_ARM_JOINT_INDEX] = LOCKED_ARM_JOINT_VALUE
    return arm


def enforce_locked_trunk_joint(values, *, copy: bool = True) -> np.ndarray:
    """Return a trunk vector with the torso yaw joint fixed at zero."""
    trunk = np.array(values, copy=copy).reshape(-1)
    if trunk.size > LOCKED_TRUNK_YAW_INDEX:
        trunk[LOCKED_TRUNK_YAW_INDEX] = LOCKED_TRUNK_YAW_VALUE
    return trunk


def enforce_locked_action(values, *, copy: bool = True) -> np.ndarray:
    """Return an official action with the profile's locked joints zeroed."""
    action = np.array(values, copy=copy).reshape(-1)
    trunk_slice = ACTION_SLICES["trunk"]
    action[trunk_slice.start + LOCKED_TRUNK_YAW_INDEX] = (
        LOCKED_TRUNK_YAW_VALUE
    )
    if LOCKED_ARM_JOINT_INDEX is not None:
        for side in ("left", "right"):
            arm_slice = ACTION_SLICES[f"arm_{side}"]
            action[arm_slice.start + LOCKED_ARM_JOINT_INDEX] = (
                LOCKED_ARM_JOINT_VALUE
            )
    return action
