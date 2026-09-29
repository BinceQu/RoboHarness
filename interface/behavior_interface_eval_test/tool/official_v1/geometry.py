"""Observation-only camera and static R1Pro geometry helpers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FrozenCapture:
    session_id: str
    image_id: str
    role: str
    depth: np.ndarray
    camera: dict[str, Any]


def quat_to_mat_xyzw(quat) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def relative_uv_to_pixel(
    u: float,
    v: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    u_f = float(u)
    v_f = float(v)
    if not (math.isfinite(u_f) and math.isfinite(v_f)):
        raise ValueError("u/v must be finite")
    if not (0.0 <= u_f <= 1000.0 and 0.0 <= v_f <= 1000.0):
        raise ValueError("u/v must use Qwen3-VL relative coordinates 0..1000")
    px = int(round(u_f / 1000.0 * max(0, int(width) - 1)))
    py = int(round(v_f / 1000.0 * max(0, int(height) - 1)))
    return px, py


def _depth_value(depth: np.ndarray, px: int, py: int, radius: int = 2) -> float:
    arr = np.asarray(depth, dtype=np.float64).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"depth must be 2-D, got shape {arr.shape}")
    height, width = arr.shape
    x0, x1 = max(0, px - radius), min(width, px + radius + 1)
    y0, y1 = max(0, py - radius), min(height, py + radius + 1)
    patch = arr[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 1e-5)]
    if valid.size == 0:
        raise ValueError(f"no valid depth near pixel ({px}, {py})")
    return float(np.median(valid))


def unproject_pixel(
    *,
    px: int,
    py: int,
    depth_m: float,
    camera: dict[str, Any],
) -> np.ndarray:
    """Unproject forward z-depth into the capture camera's declared frame."""
    z = float(depth_m)
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError(f"invalid forward depth {depth_m!r}")
    fx = float(camera["fx"])
    fy = float(camera.get("fy", fx))
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    point_camera = np.array(
        [
            (float(px) - cx) / fx * z,
            -(float(py) - cy) / fy * z,
            -z,
        ],
        dtype=np.float64,
    )
    camera_pos = np.asarray(camera["pos"], dtype=np.float64).reshape(3)
    camera_rot = quat_to_mat_xyzw(camera["quat"])
    return camera_pos + camera_rot @ point_camera


def point_from_relative_uv(
    capture: FrozenCapture,
    u: float,
    v: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = capture.depth.shape[:2]
    px, py = relative_uv_to_pixel(u, v, width, height)
    depth_m = _depth_value(capture.depth, px, py)
    point = unproject_pixel(
        px=px,
        py=py,
        depth_m=depth_m,
        camera=capture.camera,
    )
    return point, {
        "relative_uv": [float(u), float(v)],
        "pixel_uv": [int(px), int(py)],
        "depth_m": depth_m,
        "frame": str(capture.camera.get("frame", "local_command_odometry")),
    }


def surface_normal_from_relative_uv(
    capture: FrozenCapture,
    u: float,
    v: float,
    *,
    pixel_radius: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = capture.depth.shape[:2]
    px, py = relative_uv_to_pixel(u, v, width, height)
    points: dict[str, np.ndarray] = {}
    offsets = {
        "left": (-pixel_radius, 0),
        "right": (pixel_radius, 0),
        "up": (0, -pixel_radius),
        "down": (0, pixel_radius),
    }
    for name, (dx, dy) in offsets.items():
        x = int(np.clip(px + dx, 0, width - 1))
        y = int(np.clip(py + dy, 0, height - 1))
        z = _depth_value(capture.depth, x, y, radius=1)
        points[name] = unproject_pixel(
            px=x,
            py=y,
            depth_m=z,
            camera=capture.camera,
        )
    normal = np.cross(
        points["right"] - points["left"],
        points["down"] - points["up"],
    )
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        raise ValueError("local depth surface normal is degenerate")
    normal /= norm
    if normal[2] < 0.0:
        normal = -normal
    return normal, {
        "normal": normal.astype(float).tolist(),
        "normal_abs_z": abs(float(normal[2])),
        "pixel_radius": int(pixel_radius),
    }


def load_frozen_capture(
    session_id: str,
    image_id: str,
    *,
    role: str = "head",
) -> FrozenCapture:
    from behavior_interface import agent_runs

    session = str(session_id or "").strip()
    image = str(image_id or "").strip()
    meta = agent_runs.load_image_meta(session, image)
    camera = dict(meta.get("camera") or {})
    if not camera:
        raise ValueError(f"capture {image!r} has no camera metadata")
    modal = dict(meta.get("modalities") or {})
    depth_name = modal.get("depth_linear")
    if not depth_name:
        raise ValueError(f"capture {image!r} has no allowed depth_linear artifact")
    depth_path = (
        depth_name
        if os.path.isabs(depth_name)
        else os.path.join(agent_runs.images_dir(session), depth_name)
    )
    depth = np.asarray(np.load(depth_path), dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"capture depth must be 2-D, got {depth.shape}")
    return FrozenCapture(
        session_id=session,
        image_id=image,
        role=str(role),
        depth=depth,
        camera=camera,
    )


def point_in_robot_frame(world, point_local) -> np.ndarray:
    """Convert local command-odometry point to the current robot base frame."""
    point = np.asarray(point_local, dtype=np.float64).reshape(3)
    pose = world.robot_pose()
    base = np.asarray(pose.pos, dtype=np.float64).reshape(3)
    delta = point - base
    yaw = float(pose.yaw)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            cos_y * delta[0] + sin_y * delta[1],
            -sin_y * delta[0] + cos_y * delta[1],
            delta[2],
        ],
        dtype=np.float64,
    )


def point_from_robot_frame(world, point_robot) -> np.ndarray:
    point = np.asarray(point_robot, dtype=np.float64).reshape(3)
    pose = world.robot_pose()
    base = np.asarray(pose.pos, dtype=np.float64).reshape(3)
    yaw = float(pose.yaw)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return base + np.array(
        [
            cos_y * point[0] - sin_y * point[1],
            sin_y * point[0] + cos_y * point[1],
            point[2],
        ],
        dtype=np.float64,
    )


def r1pro_chest_pose_robot(trunk_q) -> tuple[np.ndarray, np.ndarray]:
    """Exact R1Pro torso_link4 FK in the robot base frame."""
    q1, q2, q3, q4 = np.asarray(trunk_q, dtype=np.float64).reshape(4)

    def rot_y(angle: float) -> np.ndarray:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return np.array(
            [
                [cos_a, 0.0, sin_a],
                [0.0, 1.0, 0.0],
                [-sin_a, 0.0, cos_a],
            ],
            dtype=np.float64,
        )

    def rot_z(angle: float) -> np.ndarray:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return np.array(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    position = np.array([-0.079032, 0.0, 0.34265], dtype=np.float64)
    rotation = rot_y(q1)
    position += rotation @ np.array([0.0, 0.0, 0.400])
    rotation = rotation @ rot_y(q2)
    position += rotation @ np.array([0.0, 0.0001005, 0.300])
    rotation = rotation @ rot_y(-q3)
    position += rotation @ np.array([0.0, -0.00010015, 0.09962])
    rotation = rotation @ rot_z(q4)
    return position, rotation


def r1pro_shoulder_positions_robot(trunk_q) -> dict[str, np.ndarray]:
    chest, rotation = r1pro_chest_pose_robot(trunk_q)
    return {
        "left": chest
        + rotation @ np.array([-0.00048618, 0.170734, 0.30302]),
        "right": chest
        + rotation @ np.array([-0.00048706, -0.170736, 0.30302]),
    }
