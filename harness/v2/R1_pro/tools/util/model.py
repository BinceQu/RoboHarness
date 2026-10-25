"""Static R1 Pro geometry and controller units; no simulator imports."""
from functools import lru_cache
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

URDF = Path(__file__).resolve().parents[2] / "robot_config/model.urdf"
ACTION_NAMES = (["base_forward", "base_left", "base_yaw"]
    + [f"torso_joint{i}" for i in range(1, 5)]
    + [f"left_arm_joint{i}" for i in range(1, 9)]
    + [f"left_gripper_finger_joint{i}" for i in (1, 2)]
    + [f"right_arm_joint{i}" for i in range(1, 9)]
    + [f"right_gripper_finger_joint{i}" for i in (1, 2)])
# Only torso yaw is locked by the evaluator's submitted controller limits.
# Baseline motion tools pin J8 by policy; explicit joint commands can move it.
LOCKED_INDICES = (6,)
FINGER_INDICES = (15, 16, 25, 26)
BASE_SCALE = np.array([0.75, 0.75, 1.0])
ACTION_SLICES = {"base": slice(0, 3), "trunk": slice(3, 7),
    "arm_left": slice(7, 15), "gripper_left": slice(15, 17),
    "arm_right": slice(17, 25), "gripper_right": slice(25, 27)}
PROPRIO_SLICES = {"base_qvel": slice(0, 3), "arm_left_qpos": slice(3, 11),
    "arm_left_qvel": slice(11, 19), "eef_left_pos": slice(19, 22), "eef_left_quat": slice(22, 26),
    "gripper_left_qpos": slice(26, 28), "gripper_left_qvel": slice(28, 30),
    "arm_right_qpos": slice(30, 38), "arm_right_qvel": slice(38, 46),
    "eef_right_pos": slice(46, 49), "eef_right_quat": slice(49, 53),
    "gripper_right_qpos": slice(53, 55), "gripper_right_qvel": slice(55, 57),
    "trunk_qpos": slice(57, 61), "trunk_qvel": slice(61, 65)}


def numbers(value):
    return np.array([float(x) for x in value.split()])


@lru_cache(maxsize=8)
def joints(urdf=URDF):
    result = {}
    for node in ET.parse(urdf).getroot().findall("joint"):
        origin, axis, limit = node.find("origin"), node.find("axis"), node.find("limit")
        result[node.get("name")] = {
            "type": node.get("type"), "parent": node.find("parent").get("link"),
            "child": node.find("child").get("link"),
            "xyz": numbers(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"),
            "rpy": numbers(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"),
            "axis": numbers(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0"),
            "limits": {k: float(v) for k, v in limit.attrib.items()} if limit is not None else {},
        }
    return result


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def origin(xyz, rpy):
    roll, pitch, yaw = rpy
    matrix = np.eye(4)
    matrix[:3, :3] = rotation((0, 0, 1), yaw) @ rotation((0, 1, 0), pitch) @ rotation((1, 0, 0), roll)
    matrix[:3, 3] = xyz
    return matrix


def forward(positions, urdf=URDF):
    frames, axes, pending = {"base_link": np.eye(4)}, {}, dict(joints(urdf))
    while pending:
        progressed = False
        for name, joint in list(pending.items()):
            if joint["parent"] not in frames:
                continue
            before = frames[joint["parent"]] @ origin(joint["xyz"], joint["rpy"])
            axes[name] = (before[:3, 3].copy(), before[:3, :3] @ joint["axis"])
            motion, value = np.eye(4), float(positions.get(name, 0))
            if joint["type"] in ("revolute", "continuous"):
                motion[:3, :3] = rotation(joint["axis"], value)
            elif joint["type"] == "prismatic":
                motion[:3, 3] = joint["axis"] * value
            frames[joint["child"]] = before @ motion
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError("URDF contains disconnected or cyclic joints")
    return frames, axes


def proprio_positions(proprio):
    vector = np.asarray(proprio, dtype=float)
    if vector.shape != (65,) or not np.isfinite(vector).all():
        raise ValueError("proprioception must have 65 finite values")
    positions = {}
    for group in ("trunk", "arm_left", "arm_right", "gripper_left", "gripper_right"):
        for name, value in zip(ACTION_NAMES[ACTION_SLICES[group]], vector[PROPRIO_SLICES[group + "_qpos"]]):
            positions[name] = float(value)
    return positions


@lru_cache(maxsize=1)
def position_command_bounds():
    """The evaluator's float32 position limits, excluding velocity/effort slots."""
    indices = [i for group in ("trunk", "arm_left", "arm_right")
               for i in range(ACTION_SLICES[group].start, ACTION_SLICES[group].stop)]
    limits = [[0., 0.] if i in LOCKED_INDICES else
              [joints()[ACTION_NAMES[i]]["limits"][key] for key in ("lower", "upper")]
              for i in indices]
    return np.array(indices), np.array(limits, dtype=np.float32).T


def clip_position_targets(action):
    """Bound internally constructed targets as the evaluator does at actuation.

    Measured joint positions can have physics-solver residuals outside nominal
    limits. Preserve those observations, but hold the nearest legal target.
    Velocity and finger-effort slots are unchanged. Explicit agent actions
    must pass validate_action *before* entering the local controller.
    """
    vector = np.asarray(action, dtype=np.float32).copy()
    if vector.shape != (27,) or not np.isfinite(vector).all():
        raise ValueError("A constructed action requires 27 finite values")
    indices, (lower, upper) = position_command_bounds()
    vector[indices] = np.clip(vector[indices], lower, upper)
    return vector


def validate_action(action):
    if not isinstance(action, (list, tuple)) or len(action) != 27:
        raise ValueError("action requires exactly 27 controller values")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in action):
        raise ValueError("controller values must be finite numbers, never booleans")
    for index, (name, value) in enumerate(zip(ACTION_NAMES, action)):
        if index in LOCKED_INDICES:
            low = high = 0.0
        elif index < 3:
            low, high = -1.0, 1.0
        elif index in FINGER_INDICES:
            low, high = -20.0, 20.0
        else:
            low, high = (joints()[name]["limits"][key] for key in ("lower", "upper"))
            # Accept both the documented URDF endpoint and its float32 wire
            # representation. Do not expand beyond this rounding interval or
            # clip explicit inputs: genuinely out-of-range targets still fail.
            low = min(low, float(np.float32(low)))
            high = max(high, float(np.float32(high)))
        if not low <= value <= high:
            raise ValueError(f"action[{index}] {name} must be within [{low}, {high}]; got {value}")
    return np.array(action, dtype=np.float32)
