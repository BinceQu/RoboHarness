"""Submission-local geometry for positioning a chassis in front of a surface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


SURFACE_FACING_STANDOFF_M = 0.8
SURFACE_FACING_MIN_LONGEST_EDGE_M = 0.005
SURFACE_FACING_MIN_NORMALIZED_DOUBLE_AREA = 1.0e-3
SURFACE_FACING_MIN_CAMERA_NORMAL_COSINE = 1.0e-4
SURFACE_FACING_MIN_HORIZONTAL_NORMAL = 0.05
SURFACE_FACING_HEADING_TOLERANCE_DEG = 1.0


def _planar_rotation(yaw_rad: float) -> np.ndarray:
    cosine = math.cos(float(yaw_rad))
    sine = math.sin(float(yaw_rad))
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rebase_robot_points(
    points_source_robot_m,
    *,
    source_base_position_policy_m,
    source_base_yaw_rad: float,
    target_base_position_policy_m,
    target_base_yaw_rad: float,
) -> np.ndarray:
    """Convert robot-relative points through policy-local SE(2) odometry."""

    points = np.asarray(points_source_robot_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("robot-relative points must be a finite Nx3 array")
    source_position = np.asarray(
        source_base_position_policy_m,
        dtype=np.float64,
    ).reshape(-1)
    target_position = np.asarray(
        target_base_position_policy_m,
        dtype=np.float64,
    ).reshape(-1)
    if (
        source_position.size != 3
        or target_position.size != 3
        or not np.all(np.isfinite(source_position))
        or not np.all(np.isfinite(target_position))
        or not math.isfinite(float(source_base_yaw_rad))
        or not math.isfinite(float(target_base_yaw_rad))
    ):
        raise ValueError("policy-local base poses must be finite xyz+yaw values")
    source_rotation = _planar_rotation(source_base_yaw_rad)
    target_rotation = _planar_rotation(target_base_yaw_rad)
    policy_points = points @ source_rotation.T + source_position
    return (policy_points - target_position) @ target_rotation


def rebase_robot_vector(
    vector_source_robot,
    *,
    source_base_yaw_rad: float,
    target_base_yaw_rad: float,
) -> np.ndarray:
    """Rotate a direction between two robot frames in policy-local odometry."""

    vector = np.asarray(vector_source_robot, dtype=np.float64).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError("robot-relative vector must contain three finite values")
    if not (
        math.isfinite(float(source_base_yaw_rad))
        and math.isfinite(float(target_base_yaw_rad))
    ):
        raise ValueError("policy-local base yaws must be finite")
    return (
        _planar_rotation(target_base_yaw_rad).T
        @ _planar_rotation(source_base_yaw_rad)
        @ vector
    )


@dataclass(frozen=True)
class SurfaceFacingSolution:
    """One deterministic base target derived from three surface points."""

    points_robot_base_m: np.ndarray
    centroid_robot_base_m: np.ndarray
    normal_toward_robot: np.ndarray
    camera_forward_robot_base: np.ndarray
    stand_point_robot_base_m: np.ndarray
    desired_chassis_forward_robot_base: np.ndarray
    desired_spin_deg: float
    camera_normal_dot: float
    longest_edge_m: float
    normalized_double_area: float
    standoff_m: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "points_robot_base_m": self.points_robot_base_m.astype(float).tolist(),
            "centroid_robot_base_m": self.centroid_robot_base_m.astype(float).tolist(),
            "normal_toward_robot": self.normal_toward_robot.astype(float).tolist(),
            "camera_forward_robot_base": self.camera_forward_robot_base.astype(float).tolist(),
            "stand_point_robot_base_m": self.stand_point_robot_base_m.astype(float).tolist(),
            "desired_chassis_forward_robot_base": (
                self.desired_chassis_forward_robot_base.astype(float).tolist()
            ),
            "desired_spin_deg": float(self.desired_spin_deg),
            "camera_normal_dot": float(self.camera_normal_dot),
            "longest_edge_m": float(self.longest_edge_m),
            "normalized_double_area": float(self.normalized_double_area),
            "standoff_m": float(self.standoff_m),
            "ground_projection": "robot_base_z_equals_zero",
        }


def solve_surface_facing_target(
    points_robot_base_m,
    camera_forward_robot_base,
    *,
    standoff_m: float = SURFACE_FACING_STANDOFF_M,
) -> SurfaceFacingSolution:
    """Resolve the base-centre target and yaw in one robot-base frame.

    The triangle normal has two signs.  The selected sign must oppose the
    frozen head-camera viewing direction, so it points from the surface back
    toward the observing robot.  The desired chassis forward direction is the
    opposite horizontal direction, toward the surface.
    """

    points = np.asarray(points_robot_base_m, dtype=np.float64)
    if points.shape != (3, 3) or not np.all(np.isfinite(points)):
        raise ValueError("surface points must be a finite 3x3 array")

    camera_forward = np.asarray(
        camera_forward_robot_base,
        dtype=np.float64,
    ).reshape(-1)
    if camera_forward.size != 3 or not np.all(np.isfinite(camera_forward)):
        raise ValueError("head camera forward vector must contain three finite values")
    camera_norm = float(np.linalg.norm(camera_forward))
    if camera_norm <= 1.0e-12:
        raise ValueError("head camera forward vector is degenerate")
    camera_forward = camera_forward / camera_norm

    standoff = float(standoff_m)
    if not math.isfinite(standoff) or standoff <= 0.0:
        raise ValueError("surface standoff must be finite and positive")

    edge01 = points[1] - points[0]
    edge02 = points[2] - points[0]
    edge12 = points[2] - points[1]
    longest_edge = max(
        float(np.linalg.norm(edge01)),
        float(np.linalg.norm(edge02)),
        float(np.linalg.norm(edge12)),
    )
    raw_normal = np.cross(edge01, edge02)
    double_area = float(np.linalg.norm(raw_normal))
    normalized_double_area = double_area / max(longest_edge * longest_edge, 1.0e-18)
    if (
        longest_edge < SURFACE_FACING_MIN_LONGEST_EDGE_M
        or normalized_double_area < SURFACE_FACING_MIN_NORMALIZED_DOUBLE_AREA
    ):
        raise ValueError(
            "the three selected surface points are degenerate or nearly collinear"
        )

    normal = raw_normal / double_area
    raw_camera_dot = float(np.dot(normal, camera_forward))
    if abs(raw_camera_dot) < SURFACE_FACING_MIN_CAMERA_NORMAL_COSINE:
        raise ValueError(
            "surface-normal direction is ambiguous because it is perpendicular "
            "to the frozen head-camera viewing direction"
        )
    if raw_camera_dot > 0.0:
        normal = -normal
    camera_normal_dot = float(np.dot(normal, camera_forward))

    horizontal_normal_norm = float(np.linalg.norm(normal[:2]))
    if horizontal_normal_norm < SURFACE_FACING_MIN_HORIZONTAL_NORMAL:
        raise ValueError(
            "surface is too close to horizontal to define a chassis-facing yaw"
        )

    centroid = np.mean(points, axis=0)
    stand_point = centroid + standoff * normal
    stand_point[2] = 0.0
    desired_forward = np.array(
        [
            -float(normal[0]) / horizontal_normal_norm,
            -float(normal[1]) / horizontal_normal_norm,
            0.0,
        ],
        dtype=np.float64,
    )
    desired_spin_deg = math.degrees(
        math.atan2(float(desired_forward[1]), float(desired_forward[0]))
    )

    return SurfaceFacingSolution(
        points_robot_base_m=points.copy(),
        centroid_robot_base_m=centroid,
        normal_toward_robot=normal,
        camera_forward_robot_base=camera_forward,
        stand_point_robot_base_m=stand_point,
        desired_chassis_forward_robot_base=desired_forward,
        desired_spin_deg=float(desired_spin_deg),
        camera_normal_dot=camera_normal_dot,
        longest_edge_m=longest_edge,
        normalized_double_area=normalized_double_area,
        standoff_m=standoff,
    )


__all__ = [
    "SURFACE_FACING_HEADING_TOLERANCE_DEG",
    "SURFACE_FACING_STANDOFF_M",
    "SurfaceFacingSolution",
    "rebase_robot_points",
    "rebase_robot_vector",
    "solve_surface_facing_target",
]
