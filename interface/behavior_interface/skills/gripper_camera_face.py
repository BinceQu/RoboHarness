"""Wrist-camera face disambiguation for grasp EEF poses.

The grasp search treats roll around the approach axis as mostly symmetric, but
the real wrist has a camera mounted on one side.  For execution we prefer the
camera face to look toward the robot/front direction, not back into the robot.
This module applies a final 180-degree roll flip when the camera-side normal is
opposite the robot forward direction.  It does not change the approach axis.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from behavior_interface.gripper_geometry_calibration import (
    R1PRO_GRIPPER_LINK_Z_EEF_M,
)

BUILD = "gripper_camera_face_v1_forward_dot_roll180"

_RY_PI_MAT = np.diag([-1.0, 1.0, -1.0])
_PALM_T_EEF = np.array(
    [0.0, 0.0, R1PRO_GRIPPER_LINK_Z_EEF_M],
    dtype=np.float64,
)
_RIGHT_REALSENSE_ORIGIN_GRIPPER = np.array(
    [0.05051, 0.0028934, 0.0051317], dtype=np.float64
)
_APPROACH_AXIS_EEF = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = [float(v) for v in _quat_normalize(q)]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _quat_mul_xyzw(q1, q2) -> np.ndarray:
    x1, y1, z1, w1 = [float(v) for v in _quat_normalize(q1)]
    x2, y2, z2, w2 = [float(v) for v in _quat_normalize(q2)]
    return _quat_normalize(np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64))


def _axis_angle_quat(axis, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / n
    half = float(angle_rad) * 0.5
    s = math.sin(half)
    return _quat_normalize(np.array([
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
        math.cos(half),
    ], dtype=np.float64))


def wrist_camera_normal_eef() -> np.ndarray:
    """Camera-side normal in EEF local frame, projected off the approach axis."""
    cam_eef = _RY_PI_MAT @ _RIGHT_REALSENSE_ORIGIN_GRIPPER + _PALM_T_EEF
    n = cam_eef - _APPROACH_AXIS_EEF * float(np.dot(cam_eef, _APPROACH_AXIS_EEF))
    if float(np.linalg.norm(n)) < 1e-9:
        n = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    return n / float(np.linalg.norm(n))


def _horizontal_unit(v) -> Optional[np.ndarray]:
    arr = np.asarray(v, dtype=np.float64).reshape(3).copy()
    arr[2] = 0.0
    n = float(np.linalg.norm(arr))
    if n < 1e-9:
        return None
    return arr / n


def robot_forward_horizontal(world) -> Optional[np.ndarray]:
    """Robot/chest forward projected to XY; returns None if unavailable."""
    if world is None:
        return None
    try:
        ch = world.chest_pose()
        f = _horizontal_unit(ch.get("forward", [0.0, 0.0, 0.0]))
        if f is not None:
            return f
    except Exception:
        pass
    try:
        yaw = float(world.robot_pose().yaw)
        return np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    except Exception:
        return None


def camera_normal_world_horizontal(quat) -> Optional[np.ndarray]:
    n = _quat_to_mat(quat) @ wrist_camera_normal_eef()
    return _horizontal_unit(n)


def approach_from_quat(quat) -> np.ndarray:
    approach = _quat_to_mat(quat) @ _APPROACH_AXIS_EEF
    n = float(np.linalg.norm(approach))
    if n < 1e-12:
        return _APPROACH_AXIS_EEF.copy()
    return approach / n


def flip_quat_across_middle_plane(quat) -> np.ndarray:
    """Preserve approach (+Z) and flip front/back by 180 deg roll."""
    return _quat_mul_xyzw(
        _quat_normalize(quat),
        _axis_angle_quat(_APPROACH_AXIS_EEF, math.pi),
    )


def ensure_camera_face_forward(
    quat,
    *,
    world=None,
    forward: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return possibly-flipped quat and an audit dict.

    The rule is dot(camera_side_normal_xy, robot_forward_xy) > 0, equivalent to
    an angle < 90 degrees.  If forward cannot be resolved, the pose is left
    unchanged and the audit marks skipped=True.
    """
    q0 = _quat_normalize(quat)
    fwd = _horizontal_unit(forward) if forward is not None else robot_forward_horizontal(world)
    n0 = camera_normal_world_horizontal(q0)
    if fwd is None or n0 is None:
        return q0, {
            "build": BUILD,
            "skipped": True,
            "flipped": False,
            "reason": "missing_forward_or_camera_normal",
        }
    dot0 = float(np.dot(n0, fwd))
    flipped = dot0 < 0.0
    q1 = flip_quat_across_middle_plane(q0) if flipped else q0
    n1 = camera_normal_world_horizontal(q1)
    dot1 = float(np.dot(n1, fwd)) if n1 is not None else float("nan")
    return q1, {
        "build": BUILD,
        "skipped": False,
        "flipped": bool(flipped),
        "dot_before": dot0,
        "dot_after": dot1,
        "angle_before_deg": float(math.degrees(math.acos(float(np.clip(dot0, -1.0, 1.0))))),
        "angle_after_deg": (
            float(math.degrees(math.acos(float(np.clip(dot1, -1.0, 1.0)))))
            if math.isfinite(dot1) else None
        ),
        "robot_forward_xy": fwd.tolist(),
        "camera_normal_xy_before": n0.tolist(),
        "camera_normal_xy_after": n1.tolist() if n1 is not None else None,
    }


def apply_camera_face_forward_to_pose(
    pos,
    quat,
    *,
    world=None,
    forward: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply final face flip to a pose; position is returned unchanged."""
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    q, audit = ensure_camera_face_forward(quat, world=world, forward=forward)
    return p, q, audit


def apply_camera_face_forward_to_grasp_dict(
    grasp: Dict[str, Any],
    *,
    world=None,
    forward: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return a copy of a best-grasp dict with the final face rule applied."""
    out = dict(grasp)
    p, q, audit = apply_camera_face_forward_to_pose(
        out["pos"], out["quat"], world=world, forward=forward,
    )
    out["pos"] = p.tolist()
    out["quat"] = q.tolist()
    out["approach"] = approach_from_quat(q).tolist()
    if audit.get("flipped") and "roll_deg" in out:
        out["roll_deg"] = (float(out.get("roll_deg", 0.0)) + 180.0) % 360.0
    out["camera_face"] = audit
    return out, audit
