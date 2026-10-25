"""Small, simulator-independent geometry helpers."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np


# RoboDojo exposes the USD camera Xform. Pinhole RGB-D coordinates use
# x-right, y-down, z-forward, while USD camera local coordinates use
# x-right, y-up, z-backward.
OPTICAL_TO_USD_AXES = np.array([1.0, -1.0, -1.0], dtype=np.float64)


def robodojo_base_rotation(yaw_deg: float | None = None) -> np.ndarray:
    """Map BEHAVIOR-style robot-base vectors into RoboDojo env vectors.

    RoboDojo's fixed rig uses +X right, +Y forward, +Z up.  The public
    official_v2 contract uses +X forward, +Y left, +Z up.  With the default
    90 degree fixed-base yaw this is ``env = [-base_y, base_x, base_z]``.
    """

    if yaw_deg is None:
        yaw_deg = float(os.environ.get("ROBODOJO_INTERFACE_BASE_YAW_DEG", "90"))
    yaw = math.radians(float(yaw_deg))
    return np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def robot_base_to_robodojo_env(
    value: Any,
    base_rotation: Any | None = None,
) -> np.ndarray:
    rotation = (
        robodojo_base_rotation()
        if base_rotation is None
        else np.asarray(base_rotation, dtype=np.float64).reshape(3, 3)
    )
    return rotation @ finite_vector(value, 3, "robot-base vector")


def robodojo_env_to_robot_base(
    value: Any,
    base_rotation: Any | None = None,
) -> np.ndarray:
    rotation = (
        robodojo_base_rotation()
        if base_rotation is None
        else np.asarray(base_rotation, dtype=np.float64).reshape(3, 3)
    )
    return rotation.T @ finite_vector(value, 3, "RoboDojo env vector")


def finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def normalize_quaternion(quaternion: Any) -> np.ndarray:
    result = finite_vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-10:
        raise ValueError("quaternion norm is zero")
    return result / norm


def quaternion_multiply(left: Any, right: Any) -> np.ndarray:
    lw, lx, ly, lz = normalize_quaternion(left)
    rw, rx, ry, rz = normalize_quaternion(right)
    return normalize_quaternion(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def quaternion_from_euler_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = np.radians([roll, pitch, yaw]) * 0.5
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return normalize_quaternion(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quaternion_angle_deg(left: Any, right: Any) -> float:
    dot = abs(float(np.dot(normalize_quaternion(left), normalize_quaternion(right))))
    return math.degrees(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))


def relative_uv_to_pixel(
    u: float, v: float, width: int, height: int
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    for name, value in (("u", u), ("v", v)):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1000.0:
            raise ValueError(f"{name} must be in 0..1000")
    px = int(round(float(u) * max(0, width - 1) / 1000.0))
    py = int(round(float(v) * max(0, height - 1) / 1000.0))
    return px, py


def optical_vector_to_world(extrinsic: Any, vector: Any) -> np.ndarray:
    transform = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    optical = finite_vector(vector, 3, "optical camera vector")
    if not np.all(np.isfinite(transform)):
        raise ValueError("camera extrinsic matrix must be finite")
    return transform[:3, :3] @ (optical * OPTICAL_TO_USD_AXES)


def unproject_relative_uv(
    depth: Any,
    intrinsic: Any,
    extrinsic: Any,
    u: float,
    v: float,
    *,
    depth_override: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth_array = np.asarray(depth, dtype=np.float64)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[:, :, 0]
    if depth_array.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth_array.shape}")
    height, width = depth_array.shape
    px, py = relative_uv_to_pixel(u, v, width, height)
    depth_m = float(depth_array[py, px] if depth_override is None else depth_override)
    if not math.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError(f"invalid depth at pixel ({px}, {py})")
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    transform = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(transform)):
        raise ValueError("camera matrices must be finite")
    fx, fy = float(k[0, 0]), float(k[1, 1])
    if abs(fx) <= 1e-10 or abs(fy) <= 1e-10:
        raise ValueError("camera focal length is zero")
    optical_point = np.array(
        [
            (px - float(k[0, 2])) * depth_m / fx,
            (py - float(k[1, 2])) * depth_m / fy,
            depth_m,
        ],
        dtype=np.float64,
    )
    camera_point = np.array(
        [*(optical_point * OPTICAL_TO_USD_AXES).tolist(), 1.0],
        dtype=np.float64,
    )
    world_point = transform @ camera_point
    return world_point[:3], {
        "pixel_uv": [px, py],
        "relative_uv": [float(u), float(v)],
        "depth_m": depth_m,
        "frame": "robodojo_env",
        "camera_extrinsic_convention": "usd_camera_xform",
        "depth_coordinate_convention": "optical_x_right_y_down_z_forward",
    }


def surface_normal_at_uv(
    depth: Any,
    intrinsic: Any,
    extrinsic: Any,
    u: float,
    v: float,
) -> np.ndarray:
    depth_array = np.asarray(depth, dtype=np.float64)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[:, :, 0]
    if depth_array.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth_array.shape}")
    height, width = depth_array.shape
    px, py = relative_uv_to_pixel(u, v, width, height)
    step = max(1, min(width, height) // 180)

    def point(x: int, y: int) -> np.ndarray:
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        ur = 1000.0 * x / max(1, width - 1)
        vr = 1000.0 * y / max(1, height - 1)
        return unproject_relative_uv(depth_array, intrinsic, extrinsic, ur, vr)[0]

    tangent_x = point(px + step, py) - point(px - step, py)
    tangent_y = point(px, py + step) - point(px, py - step)
    normal = np.cross(tangent_x, tangent_y)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        raise ValueError("cannot estimate a surface normal at the selected pixel")
    normal /= norm
    camera_position = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)[:3, 3]
    center = point(px, py)
    if float(np.dot(normal, camera_position - center)) < 0.0:
        normal = -normal
    return normal


def pose_with_delta(
    pose: Any,
    translation_world: Any,
    roll: float,
    pitch: float,
    yaw: float,
) -> list[float]:
    current = finite_vector(pose, 7, "EEF pose")
    translation = finite_vector(translation_world, 3, "translation")
    orientation = quaternion_multiply(
        current[3:7], quaternion_from_euler_deg(roll, pitch, yaw)
    )
    return [*(current[:3] + translation).tolist(), *orientation.tolist()]


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value
