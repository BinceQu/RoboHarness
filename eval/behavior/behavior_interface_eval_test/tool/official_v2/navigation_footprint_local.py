"""Submission-local whole-body footprint bounds for map navigation.

The bound uses evaluator proprioception plus collision AABBs generated offline
from the submission's own R1Pro USD. Runtime code never reads USD or simulator
state. The bound deliberately does not claim to cover a held object.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from typing import Any, Mapping, Sequence

import numpy as np

from ...robot_contract import BASE_FOOTPRINT_RADIUS_M
from .grasp_kinematics_local import LocalRobotState, link_transforms


WHOLE_BODY_FOOTPRINT_SCHEMA = "official_v2_whole_body_usd_aabb_circle_v1"
BASE_NAVIGATION_FOOTPRINT_SCHEMA = "official_v2_base_footprint_circle_v1"
_GEOMETRY_ASSET_PATH = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "r1pro_8dof_hf250_collision_aabbs.json",
)
_KINEMATICS_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "r1pro_8dof_hf250_kinematics.urdf",
)
_GEOMETRY_ASSET_SCHEMA = "r1pro_usd_link_collision_aabbs_v1"
_GEOMETRY_ASSET_SHA256 = (
    "a7a7c099ef2d39a91918e6fc5ad6d4a8d4f341e07d834ebcc1b1cb200df34b0b"
)
_KINEMATICS_MODEL_SHA256 = (
    "e303a57e6d85cf054e390e0e645460b0e4be3c238f198289f9883e55f60ca4db"
)
_SOURCE_USD_SHA256 = (
    "56f12570594c80a317d6fb38c230238e666438cdbe9b035ee8f6cf942ddc395b"
)
BASE_POLYGON_SHA256 = "8c559e6f537b59e91fc0768aa11b97f0944579f41b8dac79b79eede890db57e9"


@lru_cache(maxsize=1)
def base_navigation_polygon() -> np.ndarray:
    """Convex base and wheel envelope, exported from submission-owned geometry."""
    path = os.path.join(os.path.dirname(__file__), "assets", "r1pro_base_footprint_polygon.json")
    with open(path, "rb") as stream:
        encoded = stream.read()
    if hashlib.sha256(encoded).hexdigest() != BASE_POLYGON_SHA256:
        raise RuntimeError("navigation base polygon digest mismatch")
    payload = json.loads(encoded)
    polygon = np.asarray(payload["polygon_xy_m"], dtype=np.float64)
    if (payload.get("schema") != "submission_static_base_convex_polygon_v1"
            or payload.get("source_sha256") != _SOURCE_USD_SHA256
            or polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.all(np.isfinite(polygon))):
        raise RuntimeError("navigation base polygon is invalid")
    polygon.setflags(write=False)
    return polygon


def base_navigation_footprint_envelope(
    *,
    base_radius_m: float = BASE_FOOTPRINT_RADIUS_M,
) -> dict[str, Any]:
    """Return the static base-only footprint used by chassis navigation."""

    radius_m = float(base_radius_m)
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("base_radius_m must be finite and positive")
    if abs(radius_m - BASE_FOOTPRINT_RADIUS_M) > 1.0e-9:
        raise ValueError("base_radius_m must match the active robot contract")
    z_min_m, z_max_m = base_navigation_vertical_bounds_m()
    return {
        "schema": BASE_NAVIGATION_FOOTPRINT_SCHEMA,
        "radius_m": radius_m,
        "base_radius_m": radius_m,
        "source": "robot_contract",
        "z_min_m": z_min_m,
        "z_max_m": z_max_m,
        "height_source": "submission_local_collision_aabbs",
        "arm_posture_dependent": False,
        "payload_envelope_included": False,
    }


def _finite_vector(raw: Any, size: int, *, label: str) -> np.ndarray:
    try:
        value = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain {size} finite values") from exc
    if value.size != int(size) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain {size} finite values")
    return value.copy()


def _aabb_corners(raw_lower: Any, raw_upper: Any, *, label: str) -> np.ndarray:
    lower = _finite_vector(raw_lower, 3, label=f"{label} lower corner")
    upper = _finite_vector(raw_upper, 3, label=f"{label} upper corner")
    if np.any(lower > upper):
        raise RuntimeError(f"{label} has an inverted AABB")
    corners = np.asarray(
        tuple(itertools.product(*zip(lower, upper))),
        dtype=np.float64,
    )
    corners.setflags(write=False)
    return corners


@lru_cache(maxsize=1)
def _collision_aabb_model() -> tuple[
    float,
    tuple[tuple[str, tuple[str, ...], np.ndarray], ...],
    int,
    str,
    str,
]:
    with open(_GEOMETRY_ASSET_PATH, "rb") as stream:
        encoded = stream.read()
    asset_digest = hashlib.sha256(encoded).hexdigest()
    if asset_digest != _GEOMETRY_ASSET_SHA256:
        raise RuntimeError("navigation collision geometry digest mismatch")
    with open(_KINEMATICS_MODEL_PATH, "rb") as stream:
        kinematics_digest = hashlib.sha256(stream.read()).hexdigest()
    if kinematics_digest != _KINEMATICS_MODEL_SHA256:
        raise RuntimeError("navigation kinematics model digest mismatch")
    try:
        payload = json.loads(encoded)
        source = payload["source"]
        contract = payload["navigation_contract"]
        links = payload["links"]
        base_radius_m = float(contract["base_and_wheels_radius_m"])
        dynamic_links = tuple(str(item) for item in contract["dynamic_aabb_links"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("navigation collision geometry is invalid") from exc
    if (
        not isinstance(payload, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(contract, Mapping)
        or payload.get("schema") != _GEOMETRY_ASSET_SCHEMA
        or source.get("usd_sha256") != _SOURCE_USD_SHA256
        or source.get("fk_urdf_sha256") != _KINEMATICS_MODEL_SHA256
        or not math.isfinite(base_radius_m)
        or abs(base_radius_m - BASE_FOOTPRINT_RADIUS_M) > 1.0e-9
        or not isinstance(links, Mapping)
        or not dynamic_links
    ):
        raise RuntimeError("navigation collision geometry contract is invalid")

    groups: list[tuple[str, tuple[str, ...], np.ndarray]] = []
    box_count = 0
    for link in dynamic_links:
        raw_boxes = links.get(link)
        if not isinstance(raw_boxes, Sequence) or isinstance(
            raw_boxes, (str, bytes, bytearray)
        ) or not raw_boxes:
            raise RuntimeError(
                f"navigation collision geometry has no AABBs for {link!r}"
            )
        names: list[str] = []
        corners: list[np.ndarray] = []
        for raw_box in raw_boxes:
            if not isinstance(raw_box, Mapping):
                raise RuntimeError("navigation collision geometry is invalid")
            name = str(raw_box.get("name") or "")
            if not name:
                raise RuntimeError("navigation collision geometry is invalid")
            names.append(name)
            corners.append(
                _aabb_corners(
                    raw_box.get("aabb_min_m"),
                    raw_box.get("aabb_max_m"),
                    label=f"{link}/{name}",
                )
            )
        grouped_corners = np.concatenate(corners, axis=0)
        grouped_corners.setflags(write=False)
        groups.append((link, tuple(names), grouped_corners))
        box_count += len(names)
    if box_count <= len(dynamic_links):
        raise RuntimeError("navigation collision geometry has too few AABBs")
    return (
        base_radius_m,
        tuple(groups),
        box_count,
        asset_digest,
        kinematics_digest,
    )


def whole_body_footprint_envelope(
    evaluator_qpos: Mapping[str, Any],
    *,
    arm_dof: int,
    base_radius_m: float = BASE_FOOTPRINT_RADIUS_M,
) -> dict[str, Any]:
    """Return a conservative circular XY sweep bound for the fixed posture."""

    if not isinstance(evaluator_qpos, Mapping):
        raise ValueError("evaluator_qpos must be an object")
    arm_dof = int(arm_dof)
    if arm_dof not in (7, 8):
        raise ValueError("arm_dof must be 7 or 8")
    if not math.isfinite(float(base_radius_m)) or float(base_radius_m) <= 0.0:
        raise ValueError("base_radius_m must be finite and positive")
    trunk = _finite_vector(evaluator_qpos.get("trunk"), 4, label="trunk qpos")
    arm_left = _finite_vector(
        evaluator_qpos.get("arm_left"), arm_dof, label="left arm qpos"
    )
    arm_right = _finite_vector(
        evaluator_qpos.get("arm_right"), arm_dof, label="right arm qpos"
    )
    gripper_left = _finite_vector(
        evaluator_qpos.get("gripper_left"), 2, label="left gripper qpos"
    )
    gripper_right = _finite_vector(
        evaluator_qpos.get("gripper_right"), 2, label="right gripper qpos"
    )
    if arm_dof == 8 and (
        abs(float(arm_left[7])) > 1.0e-6
        or abs(float(arm_right[7])) > 1.0e-6
    ):
        raise ValueError("navigation footprint requires both J8 joints locked to zero")

    state = LocalRobotState(
        arm_dof=arm_dof,
        base_pos=np.zeros(3, dtype=np.float64),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        trunk_q=trunk,
        arm_left_q=arm_left,
        arm_right_q=arm_right,
        gripper_left_q=gripper_left,
        gripper_right_q=gripper_right,
        robot_forward=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    transforms = link_transforms(state)
    (
        model_base_radius_m,
        groups,
        box_count,
        geometry_digest,
        kinematics_digest,
    ) = _collision_aabb_model()
    radius_m = max(float(base_radius_m), float(model_base_radius_m))
    limiting_link = "base_footprint"
    limiting_primitive = "base_and_wheels_contract"
    for link, names, corners in groups:
        transform = transforms.get(link)
        if transform is None:
            raise RuntimeError(
                f"navigation FK does not provide collision link {link!r}"
            )
        transformed = corners @ transform[:3, :3].T + transform[:3, 3]
        radial_squared = np.sum(np.square(transformed[:, :2]), axis=1)
        point_index = int(np.argmax(radial_squared))
        candidate_m = math.sqrt(float(radial_squared[point_index]))
        if candidate_m > radius_m:
            radius_m = candidate_m
            limiting_link = link
            limiting_primitive = names[point_index // 8]
    if not math.isfinite(radius_m) or radius_m > 2.0:
        raise RuntimeError("computed navigation whole-body footprint is invalid")
    return {
        "schema": WHOLE_BODY_FOOTPRINT_SCHEMA,
        "radius_m": float(radius_m),
        "base_radius_m": float(base_radius_m),
        "limiting_link": limiting_link,
        "limiting_primitive": limiting_primitive,
        "collision_aabb_count": box_count,
        "geometry_sha256": geometry_digest,
        "kinematics_sha256": kinematics_digest,
        "source_usd_sha256": _SOURCE_USD_SHA256,
        "payload_envelope_included": False,
    }


def _observed_navigation_transforms(
    evaluator_qpos: Mapping[str, Any],
    *,
    arm_dof: int,
) -> dict[str, np.ndarray]:
    state = LocalRobotState(
        arm_dof=int(arm_dof),
        base_pos=np.zeros(3),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        trunk_q=_finite_vector(evaluator_qpos.get("trunk"), 4, label="trunk"),
        arm_left_q=_finite_vector(evaluator_qpos.get("arm_left"), arm_dof, label="left arm"),
        arm_right_q=_finite_vector(evaluator_qpos.get("arm_right"), arm_dof, label="right arm"),
        gripper_left_q=_finite_vector(evaluator_qpos.get("gripper_left"), 2, label="left gripper"),
        gripper_right_q=_finite_vector(evaluator_qpos.get("gripper_right"), 2, label="right gripper"),
        robot_forward=np.array([1.0, 0.0, 0.0]),
    )
    return link_transforms(state)


@lru_cache(maxsize=1)
def base_navigation_vertical_bounds_m() -> tuple[float, float]:
    """Bound the fixed chassis and every wheel angle from the authored model."""

    _collision_aabb_model()
    with open(_GEOMETRY_ASSET_PATH, "rb") as stream:
        encoded = stream.read()
    if hashlib.sha256(encoded).hexdigest() != _GEOMETRY_ASSET_SHA256:
        raise RuntimeError("navigation collision geometry digest mismatch")
    model = json.loads(encoded)
    neutral = {"trunk": np.zeros(4), "arm_left": np.zeros(8),
               "arm_right": np.zeros(8), "gripper_left": np.zeros(2),
               "gripper_right": np.zeros(2)}
    transforms = _observed_navigation_transforms(neutral, arm_dof=8)
    lower, upper = math.inf, -math.inf
    for link in model["navigation_contract"]["base_and_wheels_links"]:
        transform = transforms[link]
        for box in model["links"][link]:
            corners = _aabb_corners(box["aabb_min_m"], box["aabb_max_m"], label=link)
            if link == "base_link":
                z = (corners @ transform[:3, :3].T + transform[:3, 3])[:, 2]
                low_z, high_z = float(z.min()), float(z.max())
            else:
                # Steering is about base Z; wheel rotation is bounded by the
                # sphere around its joint origin, for all wheel/steer angles.
                radius = float(np.linalg.norm(corners, axis=1).max())
                low_z, high_z = float(transform[2, 3] - radius), float(transform[2, 3] + radius)
            lower, upper = min(lower, low_z), max(upper, high_z)
    if not math.isfinite(lower + upper) or upper <= lower:
        raise RuntimeError("navigation base vertical geometry is invalid")
    return lower, upper


def filter_navigation_depth_obstacles(
    points_robot: np.ndarray,
    evaluator_qpos: Mapping[str, Any],
    *,
    arm_dof: int,
    self_margin_m: float = 0.03,
    vertical_margin_m: float = 0.08,
) -> np.ndarray:
    """Remove self, floor and above-robot returns using observed joint poses."""

    points = np.asarray(points_robot, dtype=np.float64).reshape(-1, 3)
    transforms = _observed_navigation_transforms(evaluator_qpos, arm_dof=arm_dof)
    keep = np.all(np.isfinite(points), axis=1) & (points[:, 2] >= 0.15)
    keep &= np.linalg.norm(points[:, :2], axis=1) <= 3.0
    height_m = 0.30
    for link, _names, corners in _collision_aabb_model()[1]:
        transform = transforms[link]
        transformed = corners @ transform[:3, :3].T + transform[:3, 3]
        height_m = max(height_m, float(transformed[:, 2].max()))
        local = (points - transform[:3, 3]) @ transform[:3, :3]
        for box in corners.reshape(-1, 8, 3):
            keep &= ~np.all(
                (local >= box.min(axis=0) - self_margin_m)
                & (local <= box.max(axis=0) + self_margin_m),
                axis=1,
            )
    keep &= points[:, 2] <= height_m + vertical_margin_m
    keep &= ~(
        (points[:, 2] < 0.30)
        & (np.linalg.norm(points[:, :2], axis=1) < BASE_FOOTPRINT_RADIUS_M)
    )
    return points[keep]


def _translated_box_sweep_distances(
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Exact point distances to an AABB swept through a straight translation."""

    # Point-to-box squared distance is quadratic between crossings of the six
    # box planes. Minimize each interval, including its endpoints, in parallel.
    crossings = np.zeros((len(points), 6), dtype=np.float64)
    for axis in range(3):
        if abs(translation[axis]) > 1.0e-15:
            crossings[:, 2 * axis] = (points[:, axis] - lower[axis]) / translation[axis]
            crossings[:, 2 * axis + 1] = (points[:, axis] - upper[axis]) / translation[axis]
    breaks = np.sort(np.column_stack((np.zeros(len(points)),
                                     np.clip(crossings, 0.0, 1.0),
                                     np.ones(len(points)))), axis=1)
    left, right = breaks[:, :-1], breaks[:, 1:]
    middle = points[:, None, :] - ((left + right) / 2.0)[..., None] * translation
    active = (middle < lower) | (middle > upper)
    boundary = np.where(middle < lower, lower, upper)
    numerator = np.sum(np.where(active, (points[:, None, :] - boundary) * translation, 0.0), axis=2)
    denominator = np.sum(active * np.square(translation), axis=2)
    stationary = np.divide(numerator, denominator, out=left.copy(), where=denominator > 0.0)
    times = np.clip(stationary, left, right)
    relative = points[:, None, :] - times[..., None] * translation
    gap = relative - np.clip(relative, lower, upper)
    return np.sqrt(np.min(np.sum(np.square(gap), axis=2), axis=1))


def navigation_upper_body_sweep_is_clear(
    points_robot: np.ndarray,
    evaluator_qpos: Mapping[str, Any],
    *,
    arm_dof: int,
    translation_xy: Sequence[float],
    yaw_delta_rad: float,
    margin_m: float,
    point_error_m: float = 0.0,
) -> bool:
    """Conservatively sweep observed link boxes, without a full-height cylinder.

    Straight translations use exact swept-box distances. A hull and arc-sagitta
    expansion enclose intermediate yaw, including collisions absent at endpoints.
    Only the robot's submission-owned geometry and evaluator observations enter.
    """

    points = np.asarray(points_robot, dtype=np.float64).reshape(-1, 3)
    translation = np.r_[_finite_vector(translation_xy, 2, label="translation"), 0.0]
    angle = float(yaw_delta_rad)
    margin = float(margin_m)
    point_error = float(point_error_m)
    if not math.isfinite(angle) or abs(angle) > math.pi:
        raise ValueError("navigation sweep yaw must be within [-pi, pi]")
    if (not math.isfinite(margin) or margin < 0.0
            or not math.isfinite(point_error) or not 0.0 <= point_error <= margin
            or not np.all(np.isfinite(points))):
        raise ValueError("navigation sweep points or margin are invalid")
    transforms = _observed_navigation_transforms(evaluator_qpos, arm_dof=arm_dof)
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                         [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    for link, _names, corners in _collision_aabb_model()[1]:
        transform = transforms[link]
        initial = corners @ transform[:3, :3].T + transform[:3, 3]
        final = initial @ rotation.T + translation
        sagitta = float(np.linalg.norm(initial[:, :2], axis=1).max()) * (
            1.0 - math.cos(angle / 2.0)
        )
        padding = margin + sagitta
        lower = np.minimum(initial.min(axis=0), final.min(axis=0)) - padding
        upper = np.maximum(initial.max(axis=0), final.max(axis=0)) + padding
        nearby = points[np.all((points >= lower) & (points <= upper), axis=1)]
        if not len(nearby):
            continue
        local_points = (nearby - transform[:3, 3]) @ transform[:3, :3]
        final_points = ((nearby - translation) @ rotation - transform[:3, 3]) @ transform[:3, :3]
        local_final = (final - transform[:3, 3]) @ transform[:3, :3]
        local_translation = translation @ transform[:3, :3]
        for before, after in zip(corners.reshape(-1, 8, 3), local_final.reshape(-1, 8, 3)):
            lower = np.minimum(before.min(axis=0), after.min(axis=0)) - sagitta
            upper = np.maximum(before.max(axis=0), after.max(axis=0)) + sagitta
            sweep_gap = np.linalg.norm(np.maximum(np.maximum(
                lower - local_points, local_points - upper), 0.0), axis=1)
            start_gap = np.linalg.norm(np.maximum(np.maximum(
                before.min(axis=0) - local_points, local_points - before.max(axis=0)), 0.0), axis=1)
            end_gap = np.linalg.norm(np.maximum(np.maximum(
                before.min(axis=0) - final_points, final_points - before.max(axis=0)), 0.0), axis=1)
            # A robot already inside the desired margin may retreat, but only
            # with proven non-decreasing separation and no geometric overlap.
            escaping = ((start_gap > point_error) & (sweep_gap >= start_gap - 1.0e-9)
                        & (end_gap > start_gap + 1.0e-6))
            if angle == 0.0:
                box_lower, box_upper = before.min(axis=0), before.max(axis=0)
                delta = local_points - np.clip(local_points, box_lower, box_upper)
                # Convex squared distance with a nonnegative initial derivative
                # cannot decrease. This also permits motion parallel to a wall.
                escaping = ((start_gap > point_error)
                            & (delta @ local_translation <= 1.0e-12))
                candidates = (sweep_gap <= margin) & ~escaping
                if np.any(candidates):
                    sweep_gap[candidates] = _translated_box_sweep_distances(
                        local_points[candidates], box_lower, box_upper, local_translation,
                    )
            if np.any((sweep_gap <= margin) & ~escaping):
                return False
    return True
