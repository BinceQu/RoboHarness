"""RGB-D-only grasp-point planning over a whole-view reconstructed scene mesh."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import math
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .grasp_geometry_local import (
    depth_backproject_uv as _depth_backproject_uv,
    eef_frame_from_approach_roll as _eef_frame_from_approach_roll,
    ensure_camera_face_forward,
    estimate_outward_normal_from_depth,
    gap_opening_strict_mask_from_lut,
    get_wrist_opening_lut,
    gripper_geometry,
    icosa_face_normals,
    mat_to_quat_xyzw as _local_mat_to_quat_xyzw,
    world_to_pixel as _world_to_pixel,
)
from .grasp_kinematics_local import (
    IK_FILTER_ORI_TOL_DEG,
    IK_FILTER_POS_TOL_M,
    LocalRobotState,
    arm_joint_limits as _local_arm_joint_limits,
    eef_pose as _local_eef_pose,
    filter_poses_dual_arm_ik as _local_filter_poses_dual_arm_ik,
    pose_error as _local_pose_error,
)


BUILD = "grasp_point_filter_rgbd_v48_submission_local_8dof_final_safe_fk_pre_ik_overlap_gate_unique_occupancy_metrics_paired_safe_final_ik_filter_exact_inflation_vectorized_cached_batch_shared_scene_click_ray_depth_column_r3mm_first_hit_camera10mm_finger_inward15mm_static_eef_calibrated_organized_shell_v8_wrist_j567_limit_margin_tier"
MESH_VERSION = "v8"
VOXEL_M = 0.003
DEPTH_HIT_COLUMN_RADIUS_M = 0.003
N_ROLL = 8
IK_INPUT_MAX = 120
IK_QUALITY_QUOTA = 90
IK_ALIGNMENT_QUOTA = 20
IK_ALIGNMENT_QUOTA_WEAK = 12
REFINEMENT_SEED_MAX = 32
REFINEMENT_TILT_DEG = (-10.0, -5.0, 0.0, 5.0, 10.0)
REFINEMENT_ROLL_DEG = (-18.0, -9.0, 0.0, 9.0, 18.0)
RESCUE_ORIENTATION_SEED_MAX = 20
RESCUE_TILT_DEG = (
    -20.0,
    -15.0,
    -10.0,
    -5.0,
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
)
RESCUE_ROLL_DEG = (-22.5, -15.0, -7.5, 0.0, 7.5, 15.0, 22.5)
RESCUE_SAMPLING_SEED_MAX = 48
CONTACT_AXIS_INTERPOLATION_FRACTIONS = (0.25, 0.50, 0.75)
PRE_IK_TRANSLATION_OFFSET_M = (
    -0.010,
    -0.008,
    -0.006,
    -0.004,
    -0.002,
    0.002,
    0.004,
    0.006,
    0.008,
    0.010,
)
PRE_IK_NORMAL_CLEARANCE_M = (0.002, 0.004, 0.006, 0.008, 0.010, 0.012)
RESCUE_OCCUPANCY_MARGIN_M = 0.030
INFLATED_GRIPPER_CACHE_MAX_ENTRIES = 8
INTERPOLATION_SEED_MAX = 96
INTERPOLATION_PAIR_MAX = 128
INTERPOLATION_NEIGHBORS_PER_SEED = 3
INTERPOLATION_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
INTERPOLATION_GRASP_RATIO = 0.60
INTERPOLATION_MIN_ANGLE_DEG = 6.0
INTERPOLATION_MAX_ANGLE_DEG = 50.0
PARETO_MICRO_SEED_MAX = 20
POST_IK_MICRO_SEED_MAX = 16
PARETO_MICRO_GRASP_RATIO = 0.50
PARETO_MICRO_TILT_DEG = (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
PARETO_MICRO_ROLL_DEG = (-6.0, -3.0, 0.0, 3.0, 6.0)
REACHABILITY_BRIDGE_TARGETS_PER_SEED = 4
REACHABILITY_BRIDGE_FRACTIONS = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)
REACHABILITY_BRIDGE_MIN_ANGLE_DEG = 2.0
REACHABILITY_BRIDGE_MAX_ANGLE_DEG = 90.0
REACHABLE_SE3_ENDPOINT_MAX = 20
REACHABLE_SE3_PAIR_MAX = 32
REACHABLE_SE3_FRACTIONS = tuple(
    float(index) / 20.0 for index in range(1, 20)
)
REACHABLE_SE3_ORIENTATION_FRACTIONS = tuple(
    float(index) / 20.0 for index in range(21)
)
REACHABLE_SE3_MIN_GRASP_RATIO = 0.75
REACHABLE_SE3_MIN_QUAT_ANGLE_DEG = 2.0
REACHABLE_SE3_MAX_QUAT_ANGLE_DEG = 60.0
REACHABLE_SE3_MAX_EEF_DISTANCE_M = 0.060
REACHABLE_SE3_MAX_ANCHOR_DISTANCE_M = 0.030
REACHABLE_TRANSLATION_SEED_MAX = 8
REACHABLE_TRANSLATION_OFFSET_M = (
    -0.010,
    -0.008,
    -0.006,
    -0.004,
    -0.002,
    0.002,
    0.004,
    0.006,
    0.008,
    0.010,
)
# Every pose in the strict IK pool has passed GPU IK and submission-local FK.
# Final occupancy scoring is cheap, so retain the complete pool rather than
# discarding potentially better grasps based on negligible IK-error ordering.
IK_TOP_N = IK_INPUT_MAX
CAMERA_INFLATE_M = 0.010
FINGER_INWARD_INFLATE_M = 0.015
# A 3 mm occupancy voxel is 0.027 cm3. The original gripper threshold admits
# at most five voxels (0.135 cm3), while the inflated threshold admits at most
# 59 voxels (1.593 cm3).
ORIGINAL_OVERLAP_MAX_CM3 = 0.15
INFLATED_OVERLAP_MAX_CM3 = 1.6
MIN_GRASP_VOL_CM3 = 5.0
MIN_ADAPTIVE_GRASP_VOL_CM3 = 1.0
MIN_NORMAL_GRASP_VOL_CM3 = 2.7
ADAPTIVE_GRASP_QUALITY_RATIO = 0.85
NORMAL_ADAPTIVE_GRASP_QUALITY_RATIO = 0.75
PREFILTER_ALIGNMENT_GRASP_RATIO = 0.85
PREFILTER_ALIGNMENT_GRASP_RATIO_WEAK = 0.50
FINAL_ALIGNMENT_GRASP_RATIO_STRONG = 0.75
FINAL_ALIGNMENT_GRASP_RATIO_WEAK = 0.94
FINAL_ALIGNMENT_ANGLE_SLACK_STRONG_DEG = 0.25
FINAL_ALIGNMENT_ANGLE_SLACK_WEAK_DEG = 0.20
GRASP_LIFT_M = 0.10
SAFE_PLAN_BACK_M = 0.10
SAFE_IK_POS_TOL_M = 0.030
SAFE_IK_ORI_TOL_DEG = 10.0
SAFE_FINAL_BRANCH_GAP_RAD = 0.85
# 腕部 J5/J6/J7 距关节限位的裕度分档：裕度越小档位越高，在最终选姿时越先被排除。
# Only J5/J6/J7 contribute to the established wrist-limit tier.
WRIST_LIMIT_JOINT_INDICES = (4, 5, 6)
WRIST_LIMIT_CRITICAL_RAD = math.radians(5.0)
WRIST_LIMIT_WARN_RAD = math.radians(15.0)


class GraspObjPlanningError(RuntimeError):
    """Planning failed without producing an executable candidate."""


def _locked_j8_list(values: Any) -> List[float]:
    q = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if q.size == 8:
        q[7] = 0.0
    return q.astype(float).tolist()


def _pose_ik_hard_constraint_summary(
    pose: Dict[str, Any],
    *,
    pos_tol_m: float = IK_FILTER_POS_TOL_M,
    ori_tol_deg: float = IK_FILTER_ORI_TOL_DEG,
) -> Dict[str, Any]:
    def finite_round(value: float, digits: int = 3):
        return (
            round(float(value), digits)
            if math.isfinite(float(value))
            else None
        )

    def arm_summary(arm: str):
        info = (pose.get("ik_filter") or {}).get(arm) or {}
        pos_m = float(info.get("pos_err_m", float("inf")))
        ori_deg = float(info.get("ori_err_deg", float("inf")))
        ok = bool(
            math.isfinite(pos_m)
            and math.isfinite(ori_deg)
            and pos_m <= float(pos_tol_m)
            and ori_deg <= float(ori_tol_deg)
        )
        q_arm = None
        if ok and info.get("q_arm") is not None:
            q_arm = [round(value, 8) for value in _locked_j8_list(info["q_arm"])]
            info["q_arm"] = list(q_arm)
        return ok, finite_round(pos_m * 1000.0), finite_round(ori_deg), q_arm

    left_ok, left_pos_mm, left_ori_deg, left_q = arm_summary("left")
    right_ok, right_pos_mm, right_ori_deg, right_q = arm_summary("right")
    solution = (
        "both"
        if left_ok and right_ok
        else "left"
        if left_ok
        else "right"
        if right_ok
        else "none"
    )
    return {
        "solution": solution,
        "left_ok": bool(left_ok),
        "right_ok": bool(right_ok),
        "both_ok": bool(left_ok and right_ok),
        "left_pos_mm": left_pos_mm,
        "left_ori_deg": left_ori_deg,
        "right_pos_mm": right_pos_mm,
        "right_ori_deg": right_ori_deg,
        "pos_tol_mm": finite_round(float(pos_tol_m) * 1000.0),
        "ori_tol_deg": finite_round(float(ori_tol_deg)),
        "hard_constraint": (
            f"pos<={float(pos_tol_m) * 1000.0:g}mm "
            f"&& ori<={float(ori_tol_deg):g}deg"
        ),
        "validated_by_local_fk": bool(
            pose.get("ik_validated_by_local_fk")
        ),
        "validated_by_og_fk": False,
        "ik_allerr": finite_round(
            float(pose.get("ik_allerr", float("inf")))
        ),
        "left_q_arm": left_q,
        "right_q_arm": right_q,
    }


def _selected_pose_ik_q_map(
    selected_pose_ik: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(selected_pose_ik, dict):
        return {}
    return {
        arm: _locked_j8_list(selected_pose_ik[f"{arm}_q_arm"])
        for arm in ("left", "right")
        if selected_pose_ik.get(f"{arm}_q_arm") is not None
    }


_INFLATED_GRIPPER_CACHE: OrderedDict[
    Tuple[Any, ...],
    Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]],
] = OrderedDict()
_INFLATED_GRIPPER_CACHE_LOCK = threading.Lock()
_INFLATED_GRIPPER_CACHE_HITS = 0
_INFLATED_GRIPPER_CACHE_MISSES = 0


def _fingertip_axial_samples(count: int = 7) -> np.ndarray:
    """Sample the mesh-derived continuous corridor through the fingertips."""
    samples = np.asarray(
        gripper_geometry(VOXEL_M)["axial_z_m"],
        dtype=np.float64,
    ).reshape(-1)
    if len(samples) != int(count):
        raise RuntimeError(
            f"frozen fingertip sample count {len(samples)} != requested {count}"
        )
    return samples.copy()


AXIAL_Z_M = _fingertip_axial_samples()

RGBD_RECONSTRUCTION_KWARGS: Dict[str, Any] = {
    "completion_method": "organized_shell",
    "pixel_stride": 3,
    "back_extrusion_mm": 60.0,
    "surface_edge_scale": 3.5,
    "surface_edge_slack_mm": 2.0,
    "max_depth_jump_mm": 35.0,
}


def _quat_to_mat_xyzw(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _mat_to_quat_xyzw(mat: np.ndarray) -> np.ndarray:
    quat = np.asarray(
        _local_mat_to_quat_xyzw(np.asarray(mat, dtype=np.float64)),
        dtype=np.float64,
    )
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return quat


def _quat_slerp_xyzw(
    quat_a: Sequence[float],
    quat_b: Sequence[float],
    fraction: float,
) -> np.ndarray:
    """Shortest-path quaternion interpolation in xyzw convention."""
    q0 = np.asarray(quat_a, dtype=np.float64).reshape(4)
    q1 = np.asarray(quat_b, dtype=np.float64).reshape(4)
    q0 /= max(float(np.linalg.norm(q0)), 1e-12)
    q1 /= max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.clip(q0 @ q1, -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    t = float(np.clip(float(fraction), 0.0, 1.0))
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        result /= max(float(np.linalg.norm(result)), 1e-12)
        return result
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    sin_theta = math.sin(theta)
    result = (
        math.sin((1.0 - t) * theta) / sin_theta * q0
        + math.sin(t * theta) / sin_theta * q1
    )
    result /= max(float(np.linalg.norm(result)), 1e-12)
    return result


def _quat_distance_deg(
    quat_a: Sequence[float],
    quat_b: Sequence[float],
) -> float:
    q0 = np.asarray(quat_a, dtype=np.float64).reshape(4)
    q1 = np.asarray(quat_b, dtype=np.float64).reshape(4)
    q0 /= max(float(np.linalg.norm(q0)), 1e-12)
    q1 /= max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.clip(abs(float(q0 @ q1)), 0.0, 1.0))
    return float(math.degrees(2.0 * math.acos(dot)))


def estimate_grasp_alignment_normal(
    depth: np.ndarray,
    *,
    u: int,
    v: int,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    image_width: int,
    image_height: int,
    focal_length: float,
    horizontal_aperture: float,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Reuse plan_press depth normals with a one-scale weak fallback."""
    point, normal, audit = estimate_outward_normal_from_depth(
        np.asarray(depth, dtype=np.float64),
        int(u),
        int(v),
        np.asarray(camera_pos, dtype=np.float64),
        np.asarray(camera_quat_xyzw, dtype=np.float64),
        int(image_width),
        int(image_height),
        float(focal_length),
        float(horizontal_aperture),
    )
    audit = dict(audit)
    audit["point_world"] = (
        np.asarray(point, dtype=np.float64).reshape(3).tolist()
        if point is not None
        else None
    )
    if normal is not None:
        selected = np.asarray(normal, dtype=np.float64).reshape(3)
        selected /= max(float(np.linalg.norm(selected)), 1e-12)
        audit.update({
            "usable": True,
            "confidence": "strong",
            "selection": "plan_press_strict_multiscale",
            "normal_world": selected.tolist(),
        })
        return selected, audit

    passing = [
        scale
        for scale in (audit.get("scales") or [])
        if scale.get("ok")
        and float(scale.get("planarity", float("inf")))
        <= float(audit.get("max_planarity", 0.08))
        and float(scale.get("rms_m", float("inf")))
        <= float(audit.get("max_rms_m", 0.003))
    ]
    if len(passing) == 1:
        scale = passing[0]
        selected = np.asarray(scale["normal_world"], dtype=np.float64).reshape(3)
        selected /= max(float(np.linalg.norm(selected)), 1e-12)
        audit.update({
            "usable": True,
            "confidence": "weak",
            "selection": "single_quality_scale_soft_preference",
            "normal_world": selected.tolist(),
            "selected_half_window_px": int(scale["half_window_px"]),
            "selected_support_points": int(scale["support_points"]),
            "selected_planarity": float(scale["planarity"]),
            "selected_rms_m": float(scale["rms_m"]),
        })
        return selected, audit

    audit.update({
        "usable": False,
        "confidence": "none",
        "selection": "disabled",
    })
    return None, audit


def prepare_rgbd_grasp_target(
    depth: np.ndarray,
    *,
    u: int,
    v: int,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
) -> Dict[str, Any]:
    """Prepare one click's depth hit and RGBD-only surface normal."""
    hit, hit_audit = depth_hit_from_pixel(
        depth,
        u=int(u),
        v=int(v),
        camera_pos=cam_pos,
        camera_quat_xyzw=cam_quat,
        focal_length=fl,
        horizontal_aperture=ha,
    )
    normal_u, normal_v = hit_audit["depth_pixel"]
    outward_normal, normal_audit = estimate_grasp_alignment_normal(
        depth,
        u=int(normal_u),
        v=int(normal_v),
        camera_pos=cam_pos,
        camera_quat_xyzw=cam_quat,
        image_width=int(w),
        image_height=int(h),
        focal_length=float(fl),
        horizontal_aperture=float(ha),
    )
    normal_audit = dict(normal_audit)
    normal_point = normal_audit.get("point_world")
    if normal_point is not None:
        normal_audit["hit_point_delta_m"] = float(
            np.linalg.norm(
                np.asarray(normal_point, dtype=np.float64).reshape(3) - hit
            )
        )
    return {
        "hit": np.asarray(hit, dtype=np.float64),
        "hit_audit": dict(hit_audit),
        "outward_normal": (
            None
            if outward_normal is None
            else np.asarray(outward_normal, dtype=np.float64)
        ),
        "normal_audit": normal_audit,
    }


def pose_gripper_vector_world(pose: Dict[str, Any]) -> np.ndarray:
    """Match plan_press: vector from the fingertip side toward O_EEF."""
    rotation = np.asarray(pose["R"], dtype=np.float64).reshape(3, 3)
    vector = -rotation[:, 2]
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return vector


def attach_normal_alignment(
    poses: Sequence[Dict[str, Any]],
    outward_normal_world: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Attach plan_press-compatible normal angles to grasp poses."""
    if outward_normal_world is None:
        return {"enabled": False, "pose_count": int(len(poses))}
    outward = np.asarray(outward_normal_world, dtype=np.float64).reshape(3)
    outward /= max(float(np.linalg.norm(outward)), 1e-12)
    angles = []
    for pose in poses:
        gripper_vector = pose_gripper_vector_world(pose)
        angle = float(
            math.degrees(
                math.acos(float(np.clip(outward @ gripper_vector, -1.0, 1.0)))
            )
        )
        pose["gripper_vector_world"] = gripper_vector
        pose["normal_alignment_deg"] = angle
        angles.append(angle)
    values = np.asarray(angles, dtype=np.float64)
    return {
        "enabled": True,
        "pose_count": int(len(poses)),
        "outward_normal_world": outward.tolist(),
        "angle_min_deg": float(values.min()) if len(values) else None,
        "angle_median_deg": float(np.median(values)) if len(values) else None,
        "angle_max_deg": float(values.max()) if len(values) else None,
    }


def depth_hit_from_pixel(
    depth: np.ndarray,
    *,
    u: int,
    v: int,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    column_radius_m: float = DEPTH_HIT_COLUMN_RADIUS_M,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select the first observed point in a fixed-radius click-ray column."""
    metric = np.asarray(depth, dtype=np.float64)
    if metric.ndim == 3:
        metric = metric[..., 0]
    if metric.ndim != 2:
        raise ValueError(f"depth must be HxW, got {metric.shape}")
    height, width = metric.shape
    u0 = int(np.clip(int(u), 0, width - 1))
    v0 = int(np.clip(int(v), 0, height - 1))
    focal_px = float(focal_length) / float(horizontal_aperture) * float(width)
    if not math.isfinite(focal_px) or focal_px <= 0.0:
        raise ValueError(f"invalid focal length in pixels: {focal_px}")
    radius_m = float(column_radius_m)
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError(f"column radius must be positive, got {radius_m}")

    click_ray = np.asarray(
        [
            (float(u0) - width / 2.0) / focal_px,
            -(float(v0) - height / 2.0) / focal_px,
            -1.0,
        ],
        dtype=np.float64,
    )
    click_ray /= max(float(np.linalg.norm(click_ray)), 1e-12)

    x_scale = (
        np.arange(width, dtype=np.float64) - float(width) / 2.0
    ) / focal_px
    y_scale = -(
        np.arange(height, dtype=np.float64) - float(height) / 2.0
    ) / focal_px
    camera_x = metric * x_scale[None, :]
    camera_y = metric * y_scale[:, None]
    camera_z = -metric
    axial = (
        camera_x * click_ray[0]
        + camera_y * click_ray[1]
        + camera_z * click_ray[2]
    )
    radial_sq = np.maximum(
        camera_x * camera_x
        + camera_y * camera_y
        + camera_z * camera_z
        - axial * axial,
        0.0,
    )
    candidate_mask = (
        np.isfinite(metric)
        & (metric > 0.0)
        & np.isfinite(axial)
        & (axial > 0.0)
        & (radial_sq <= radius_m * radius_m + 1e-15)
    )
    candidate_v, candidate_u = np.nonzero(candidate_mask)
    if candidate_u.size == 0:
        raise ValueError(
            f"depth column at pixel ({u0},{v0}) with radius "
            f"{radius_m * 1000.0:.1f}mm contains no valid observed depth point"
        )

    candidate_axial = axial[candidate_v, candidate_u]
    candidate_radial_sq = radial_sq[candidate_v, candidate_u]
    pixel_distance_sq = (
        (candidate_u.astype(np.int64) - int(u0)) ** 2
        + (candidate_v.astype(np.int64) - int(v0)) ** 2
    )
    # Axial distance is the first-hit criterion. The remaining keys only make
    # equal-distance samples deterministic and favor points close to the ray.
    order = np.lexsort(
        (
            candidate_u,
            candidate_v,
            pixel_distance_sq,
            candidate_radial_sq,
            candidate_axial,
        )
    )
    selected_index = int(order[0])
    chosen_u = int(candidate_u[selected_index])
    chosen_v = int(candidate_v[selected_index])
    value = float(metric[chosen_v, chosen_u])
    selected_axial = float(axial[chosen_v, chosen_u])
    selected_axis_offset = math.sqrt(float(radial_sq[chosen_v, chosen_u]))

    camera_point = np.asarray(
        [
            (float(chosen_u) - width / 2.0) / focal_px * value,
            -(float(chosen_v) - height / 2.0) / focal_px * value,
            -value,
        ],
        dtype=np.float64,
    )
    rotation = _quat_to_mat_xyzw(camera_quat_xyzw)
    hit = (
        np.asarray(camera_pos, dtype=np.float64).reshape(3)
        + rotation @ camera_point
    )
    center_value = float(metric[v0, u0])
    center_axial: Optional[float] = None
    if math.isfinite(center_value) and center_value > 0.0:
        center_point = np.asarray(
            [
                (float(u0) - width / 2.0) / focal_px * center_value,
                -(float(v0) - height / 2.0) / focal_px * center_value,
                -center_value,
            ],
            dtype=np.float64,
        )
        center_axial = float(center_point @ click_ray)
    return hit, {
        "method": "depth_column_first_observed_surface",
        "selection": "minimum_positive_axial_distance_in_click_ray_column",
        "source": "raw_observed_depth",
        "input_pixel": [u0, v0],
        "depth_pixel": [int(chosen_u), int(chosen_v)],
        "depth_m": float(value),
        "focal_px": float(focal_px),
        "used_segmentation": False,
        "column_radius_m": radius_m,
        "column_radius_mm": radius_m * 1000.0,
        "column_diameter_mm": radius_m * 2000.0,
        "candidate_count": int(candidate_u.size),
        "front_axial_distance_m": selected_axial,
        "selected_axial_distance_m": selected_axial,
        "selected_axis_offset_m": selected_axis_offset,
        "selected_axis_offset_mm": selected_axis_offset * 1000.0,
        "center_ray_axial_distance_m": center_axial,
        "selected_lead_vs_center_mm": (
            None
            if center_axial is None
            else (center_axial - selected_axial) * 1000.0
        ),
    }


def reconstruct_scene_mesh_from_session(
    session: Dict[str, Any],
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Run the frozen v8 whole-view reconstructor on the session RGB-D files."""
    from PIL import Image
    from .rgbd_scene_mesh_v8 import reconstruct_rgbd_scene_mesh

    init_dir = str(session.get("init_dir") or "")
    rgb_path = str(session.get("rgb_path") or os.path.join(init_dir, "rgb.png"))
    depth_path = str(
        session.get("depth_path") or os.path.join(init_dir, "depth.npy")
    )
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"missing frozen RGB: {rgb_path}")
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"missing frozen depth: {depth_path}")
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path)
    started = time.perf_counter()
    result = reconstruct_rgbd_scene_mesh(
        rgb,
        depth,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
        **RGBD_RECONSTRUCTION_KWARGS,
    )
    metadata = dict(result.metadata)
    metadata["elapsed_s"] = float(time.perf_counter() - started)
    metadata["mesh_version"] = MESH_VERSION
    metadata["reconstruction_kwargs"] = dict(RGBD_RECONSTRUCTION_KWARGS)
    return result.mesh, rgb, depth, metadata


def clicked_hit_anchor(
    hit: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Use the depth-unprojected click as the sole grasp surface anchor."""
    anchor = np.asarray(hit, dtype=np.float64).reshape(1, 3).copy()
    return anchor, {
        "source": "depth_hit",
        "radius_m": 0.0,
        "requested_max_anchors": 1,
        "candidate_count": 1,
        "anchor_count": 1,
        "anchor_distance_mm": [0.0],
    }


def generate_rgbd_filter_poses(
    anchors: np.ndarray,
    *,
    axial_z_m: np.ndarray = AXIAL_Z_M,
    n_roll: int = N_ROLL,
) -> List[Dict[str, Any]]:
    """Generate 20 approach x 8 roll x 7 axial placements per anchor."""
    normals = icosa_face_normals()
    rolls = 2.0 * np.pi * np.arange(int(n_roll), dtype=np.float64) / float(n_roll)
    poses: List[Dict[str, Any]] = []
    for pi, anchor in enumerate(np.asarray(anchors, dtype=np.float64).reshape(-1, 3)):
        for ni, normal in enumerate(normals):
            approach = -normal
            for ri, roll in enumerate(rolls):
                rotation = _eef_frame_from_approach_roll(approach, float(roll))
                quat = _mat_to_quat_xyzw(rotation)
                for ai, axial_z in enumerate(np.asarray(axial_z_m, dtype=np.float64)):
                    anchor_local = np.asarray([0.0, 0.0, axial_z], dtype=np.float64)
                    eef_pos = anchor - rotation @ anchor_local
                    poses.append(
                        {
                            "pi": int(pi),
                            "ni": int(ni),
                            "ri": int(ri),
                            "ai": int(ai),
                            "axial_point": f"p{ai + 1}",
                            "axial_z_m": float(axial_z),
                            "anchor_local": anchor_local,
                            "anchor": anchor.copy(),
                            "eef_pos": eef_pos,
                            "R": rotation,
                            "quat": quat.copy(),
                        }
                    )
    return poses


def apply_camera_face_preserving_anchor(
    poses: List[Dict[str, Any]],
    *,
    world,
    ctx=None,
) -> Dict[str, int]:
    """Apply camera-face roll while preserving each pose's selected P1-P7 anchor."""
    flipped = 0
    skipped = 0
    for pose in poses:
        forward = np.asarray(
            getattr(world, "robot_forward", [1.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        quat, audit = ensure_camera_face_forward(
            pose["quat"],
            forward=forward,
        )
        rotation = _quat_to_mat_xyzw(quat)
        anchor = np.asarray(pose["anchor"], dtype=np.float64)
        anchor_local = np.asarray(pose["anchor_local"], dtype=np.float64)
        pose["quat"] = quat
        pose["R"] = rotation
        pose["eef_pos"] = anchor - rotation @ anchor_local
        pose["camera_face"] = audit
        if audit.get("flipped"):
            pose.pop("ik_warm_start_q_by_arm", None)
        flipped += int(bool(audit.get("flipped")))
        skipped += int(bool(audit.get("skipped")))
    audit = {"pose_count": len(poses), "flipped": flipped, "skipped": skipped}
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] camera-face poses={len(poses)} "
            f"flipped={flipped} skipped={skipped}"
        )
    return audit


def inflated_gripper_voxels_eef(
    base_voxels: np.ndarray,
    *,
    component_voxels: Dict[str, np.ndarray],
    voxel_m: float = VOXEL_M,
    camera_inflate_m: float = CAMERA_INFLATE_M,
    finger_inward_inflate_m: float = FINGER_INWARD_INFLATE_M,
    return_regions: bool = False,
):
    """Build the component-aware inflated collision query in the EEF frame.

    The wrist camera receives a 10 mm isotropic shell. Each finger is swept
    15 mm only toward the center gap: the +Y finger toward -Y and the -Y
    finger toward +Y. The palm and all other gripper geometry are not expanded.
    """
    global _INFLATED_GRIPPER_CACHE_HITS
    global _INFLATED_GRIPPER_CACHE_MISSES

    voxel = float(voxel_m)

    def _keys(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not len(pts):
            return np.zeros((0, 3), dtype=np.int64)
        keys = np.rint(pts / voxel).astype(np.int64)
        return np.unique(keys, axis=0)

    def _points(keys: np.ndarray) -> np.ndarray:
        return np.asarray(keys, dtype=np.float64).reshape(-1, 3) * voxel

    def _difference(keys: np.ndarray, occupied: set) -> np.ndarray:
        if not len(keys):
            return np.zeros((0, 3), dtype=np.int64)
        return np.asarray(
            [key for key in map(tuple, keys.tolist()) if key not in occupied],
            dtype=np.int64,
        ).reshape(-1, 3)

    def _cache_key(
        arrays: Sequence[Tuple[str, np.ndarray]],
    ) -> Tuple[Any, ...]:
        digest = hashlib.blake2b(digest_size=20)
        for name, keys in arrays:
            contiguous = np.ascontiguousarray(
                np.asarray(keys, dtype=np.int64).reshape(-1, 3)
            )
            digest.update(name.encode("ascii"))
            digest.update(
                np.asarray(contiguous.shape, dtype=np.int64).tobytes()
            )
            digest.update(contiguous.tobytes())
        return (
            voxel.hex(),
            float(camera_inflate_m).hex(),
            float(finger_inward_inflate_m).hex(),
            digest.digest(),
        )

    def _copy_cached(
        cached: Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]],
    ):
        result, regions, audit = cached
        result_copy = result.copy()
        if return_regions:
            return (
                result_copy,
                {name: values.copy() for name, values in regions.items()},
                dict(audit),
            )
        return result_copy

    base_keys = _keys(base_voxels)
    camera_keys = _keys(component_voxels.get("camera", []))
    positive_finger_keys = _keys(
        component_voxels.get("finger_positive_y", [])
    )
    negative_finger_keys = _keys(
        component_voxels.get("finger_negative_y", [])
    )
    cache_key = _cache_key(
        (
            ("base", base_keys),
            ("camera", camera_keys),
            ("finger_positive_y", positive_finger_keys),
            ("finger_negative_y", negative_finger_keys),
        )
    )
    with _INFLATED_GRIPPER_CACHE_LOCK:
        cached = _INFLATED_GRIPPER_CACHE.get(cache_key)
        if cached is not None:
            _INFLATED_GRIPPER_CACHE.move_to_end(cache_key)
            _INFLATED_GRIPPER_CACHE_HITS += 1
            return _copy_cached(cached)
        _INFLATED_GRIPPER_CACHE_MISSES += 1

    sample_step_m = min(0.001, voxel)
    camera_radius_steps = int(
        math.ceil(float(camera_inflate_m) / sample_step_m)
    )
    camera_offsets = []
    for ix in range(-camera_radius_steps, camera_radius_steps + 1):
        for iy in range(-camera_radius_steps, camera_radius_steps + 1):
            for iz in range(-camera_radius_steps, camera_radius_steps + 1):
                offset = np.asarray([ix, iy, iz], dtype=np.float64)
                if (
                    np.linalg.norm(offset) * sample_step_m
                    <= float(camera_inflate_m) + 1e-12
                ):
                    camera_offsets.append(offset * sample_step_m)
    if len(camera_keys) and camera_offsets:
        camera_offsets_array = np.asarray(
            camera_offsets,
            dtype=np.float64,
        )
        scaled_camera_offsets = camera_offsets_array / voxel
        camera_offset_fraction = np.mod(
            np.abs(scaled_camera_offsets),
            1.0,
        )
        if np.any(
            np.isclose(camera_offset_fraction, 0.5, atol=1e-12)
        ):
            camera_points = _points(camera_keys)
            camera_dilated_keys = _keys(
                (
                    camera_points[:, None, :]
                    + camera_offsets_array[None, :, :]
                ).reshape(-1, 3)
            )
        else:
            quantized_camera_offsets = np.unique(
                np.rint(scaled_camera_offsets).astype(np.int64),
                axis=0,
            )
            camera_dilated_keys = np.unique(
                (
                    camera_keys[:, None, :]
                    + quantized_camera_offsets[None, :, :]
                ).reshape(-1, 3),
                axis=0,
            )
    else:
        camera_dilated_keys = camera_keys

    finger_steps = np.arange(
        0.0,
        float(finger_inward_inflate_m) + sample_step_m * 0.5,
        sample_step_m,
        dtype=np.float64,
    )

    def _swept_finger(keys: np.ndarray, inward_sign_y: float) -> np.ndarray:
        if not len(keys):
            return keys
        scaled_y_offsets = inward_sign_y * finger_steps / voxel
        y_offset_fraction = np.mod(np.abs(scaled_y_offsets), 1.0)
        if np.any(np.isclose(y_offset_fraction, 0.5, atol=1e-12)):
            offsets = np.zeros((len(finger_steps), 3), dtype=np.float64)
            offsets[:, 1] = inward_sign_y * finger_steps
            return _keys(
                (
                    _points(keys)[:, None, :]
                    + offsets[None, :, :]
                ).reshape(-1, 3)
            )
        quantized_y_offsets = np.unique(
            np.rint(scaled_y_offsets).astype(np.int64)
        )
        offsets = np.zeros(
            (len(quantized_y_offsets), 3),
            dtype=np.int64,
        )
        offsets[:, 1] = quantized_y_offsets
        return np.unique(
            (keys[:, None, :] + offsets[None, :, :]).reshape(-1, 3),
            axis=0,
        )

    positive_finger_swept = _swept_finger(
        positive_finger_keys,
        inward_sign_y=-1.0,
    )
    negative_finger_swept = _swept_finger(
        negative_finger_keys,
        inward_sign_y=1.0,
    )

    base_set = set(map(tuple, base_keys.tolist()))
    camera_body_added = _difference(camera_keys, base_set)
    occupied = base_set | set(map(tuple, camera_body_added.tolist()))
    camera_shell_added = _difference(camera_dilated_keys, occupied)
    occupied |= set(map(tuple, camera_shell_added.tolist()))
    positive_finger_added = _difference(positive_finger_swept, occupied)
    occupied |= set(map(tuple, positive_finger_added.tolist()))
    negative_finger_added = _difference(negative_finger_swept, occupied)
    occupied |= set(map(tuple, negative_finger_added.tolist()))

    union_keys = np.asarray(sorted(occupied), dtype=np.int64).reshape(-1, 3)
    regions = {
        "base_gripper": _points(base_keys),
        "camera_body": _points(camera_body_added),
        "camera_outward_10mm": _points(camera_shell_added),
        "finger_positive_y_inward_15mm": _points(positive_finger_added),
        "finger_negative_y_inward_15mm": _points(negative_finger_added),
    }
    audit = {
        "policy": "camera_outward_10mm_fingers_inward_15mm",
        "voxel_m": voxel,
        "sample_step_m": float(sample_step_m),
        "camera_inflate_m": float(camera_inflate_m),
        "finger_inward_inflate_m": float(finger_inward_inflate_m),
        "finger_positive_y_direction_eef": [0.0, -1.0, 0.0],
        "finger_negative_y_direction_eef": [0.0, 1.0, 0.0],
        "base_gripper_voxels": int(len(base_keys)),
        "camera_body_added_voxels": int(len(camera_body_added)),
        "camera_shell_added_voxels": int(len(camera_shell_added)),
        "finger_positive_y_added_voxels": int(len(positive_finger_added)),
        "finger_negative_y_added_voxels": int(len(negative_finger_added)),
        "inflated_union_voxels": int(len(union_keys)),
        "quantization": "1mm_sweep_then_round_to_3mm_eef_grid",
    }
    result = _points(union_keys)
    cache_value = (
        result.copy(),
        {name: values.copy() for name, values in regions.items()},
        dict(audit),
    )
    with _INFLATED_GRIPPER_CACHE_LOCK:
        _INFLATED_GRIPPER_CACHE[cache_key] = cache_value
        _INFLATED_GRIPPER_CACHE.move_to_end(cache_key)
        while (
            len(_INFLATED_GRIPPER_CACHE)
            > int(INFLATED_GRIPPER_CACHE_MAX_ENTRIES)
        ):
            _INFLATED_GRIPPER_CACHE.popitem(last=False)
    if return_regions:
        return result, regions, audit
    return result


def clear_inflated_gripper_voxel_cache() -> None:
    """Clear only the exact component-aware inflation cache."""
    global _INFLATED_GRIPPER_CACHE_HITS
    global _INFLATED_GRIPPER_CACHE_MISSES

    with _INFLATED_GRIPPER_CACHE_LOCK:
        _INFLATED_GRIPPER_CACHE.clear()
        _INFLATED_GRIPPER_CACHE_HITS = 0
        _INFLATED_GRIPPER_CACHE_MISSES = 0


def inflated_gripper_voxel_cache_info() -> Dict[str, int]:
    """Return cache counters for diagnostics and regression tests."""
    with _INFLATED_GRIPPER_CACHE_LOCK:
        return {
            "entries": int(len(_INFLATED_GRIPPER_CACHE)),
            "hits": int(_INFLATED_GRIPPER_CACHE_HITS),
            "misses": int(_INFLATED_GRIPPER_CACHE_MISSES),
            "max_entries": int(INFLATED_GRIPPER_CACHE_MAX_ENTRIES),
        }


def _mesh_topology_diagnostic(
    mesh: Any,
    reconstruction_metadata: Dict[str, Any],
) -> Tuple[Any, str]:
    """Read topology status without triggering deferred Trimesh work."""
    topology_validation = str(
        reconstruction_metadata.get("mesh_topology_validation") or ""
    )
    if topology_validation == "deferred_to_warp_cuda":
        return "deferred_to_warp_cuda", topology_validation
    return bool(mesh.is_watertight), topology_validation or "trimesh_cpu"


@dataclass
class DenseSceneOccupancy:
    origin: np.ndarray
    occupancy: np.ndarray
    voxel_m: float
    metadata: Dict[str, Any]
    _device_cache: Dict[str, Dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _offset_tensor_cache: Dict[Tuple[str, bytes], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _count_cache: Dict[Tuple[str, bytes], Dict[bytes, int]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @staticmethod
    def _pose_query_key(
        pose: Dict[str, Any],
        *,
        dtype,
    ) -> bytes:
        position = np.ascontiguousarray(
            np.asarray(pose["eef_pos"], dtype=dtype).reshape(3)
        )
        rotation = np.ascontiguousarray(
            np.asarray(pose["R"], dtype=dtype).reshape(3, 3)
        )
        return position.tobytes() + rotation.tobytes()

    @staticmethod
    def _offset_query_key(offsets: np.ndarray) -> bytes:
        digest = hashlib.sha1()
        digest.update(
            np.asarray(offsets.shape, dtype=np.int64).tobytes()
        )
        digest.update(np.ascontiguousarray(offsets).tobytes())
        return digest.digest()

    def counts_for_poses(
        self,
        poses: Sequence[Dict[str, Any]],
        offsets_eef: np.ndarray,
        *,
        batch_size: int = 32,
        prefer_cuda: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Count occupied transformed query voxels for every pose."""
        started = time.perf_counter()
        offsets = np.asarray(offsets_eef, dtype=np.float32).reshape(-1, 3)
        if not poses or not len(offsets):
            return np.zeros(len(poses), dtype=np.int32), {
                "device": "none",
                "elapsed_s": 0.0,
                "query_voxels_per_pose": int(len(offsets)),
            }
        try:
            import torch

            use_cuda = bool(prefer_cuda and torch.cuda.is_available())
        except Exception:
            torch = None
            use_cuda = False

        required_device_name = str(
            self.metadata.get("pose_count_device_required") or ""
        ).strip()
        if required_device_name:
            if not required_device_name.startswith("cuda"):
                raise RuntimeError(
                    "invalid pose-count device contract "
                    f"{required_device_name!r}"
                )
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError(
                    "projective occupancy requires CUDA pose counting; "
                    "CPU fallback is disabled"
                )
            try:
                required_device = torch.device(required_device_name)
                torch.empty(1, dtype=torch.uint8, device=required_device)
            except Exception as exc:
                raise RuntimeError(
                    "projective occupancy pose-count device is unavailable: "
                    f"{required_device_name}: {exc}"
                ) from exc
            use_cuda = True
        else:
            required_device = None

        backend = (
            str(required_device or "cuda:0")
            if torch is not None and use_cuda
            else ("torch:cpu" if torch is not None else "numpy")
        )
        pose_dtype = np.float32 if torch is not None else np.float64
        offset_key = self._offset_query_key(offsets)
        count_cache = self._count_cache.setdefault(
            (backend, offset_key),
            {},
        )
        pose_keys = [
            self._pose_query_key(pose, dtype=pose_dtype)
            for pose in poses
        ]
        missing_keys: List[bytes] = []
        missing_poses: List[Dict[str, Any]] = []
        pending = set()
        for key, pose in zip(pose_keys, poses):
            if key in count_cache or key in pending:
                continue
            pending.add(key)
            missing_keys.append(key)
            missing_poses.append(pose)

        if torch is not None:
            device = torch.device(
                required_device or ("cuda:0" if use_cuda else "cpu")
            )
            device_key = str(device)
            device_state = self._device_cache.get(device_key)
            if device_state is None:
                device_state = {
                    "dense": torch.as_tensor(
                        np.asarray(
                            self.occupancy,
                            dtype=np.bool_,
                        ).reshape(-1),
                        dtype=torch.bool,
                        device=device,
                    ),
                    "origin": torch.as_tensor(
                        np.asarray(self.origin, dtype=np.float32),
                        dtype=torch.float32,
                        device=device,
                    ),
                }
                self._device_cache[device_key] = device_state
            dense = device_state["dense"]
            origin = device_state["origin"]
            tensor_cache_key = (device_key, offset_key)
            local = self._offset_tensor_cache.get(tensor_cache_key)
            if local is None:
                local = torch.as_tensor(
                    offsets,
                    dtype=torch.float32,
                    device=device,
                )
                self._offset_tensor_cache[tensor_cache_key] = local
            shape = tuple(int(value) for value in self.occupancy.shape)
            counts: List[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, len(missing_poses), int(batch_size)):
                    batch = missing_poses[
                        start : start + int(batch_size)
                    ]
                    positions = torch.as_tensor(
                        np.stack([pose["eef_pos"] for pose in batch]).astype(np.float32),
                        dtype=torch.float32,
                        device=device,
                    )
                    rotations = torch.as_tensor(
                        np.stack([pose["R"] for pose in batch]).astype(np.float32),
                        dtype=torch.float32,
                        device=device,
                    )
                    world_points = (
                        torch.einsum("bij,nj->bni", rotations, local)
                        + positions[:, None, :]
                    )
                    index = torch.round(
                        (world_points - origin[None, None, :]) / float(self.voxel_m)
                    ).to(torch.long)
                    valid = (
                        (index[..., 0] >= 0)
                        & (index[..., 0] < shape[0])
                        & (index[..., 1] >= 0)
                        & (index[..., 1] < shape[1])
                        & (index[..., 2] >= 0)
                        & (index[..., 2] < shape[2])
                    )
                    flat_index = (
                        (index[..., 0] * shape[1] + index[..., 1]) * shape[2]
                        + index[..., 2]
                    )
                    flat_index = torch.clamp(flat_index, 0, dense.numel() - 1)
                    occupied = dense[flat_index] & valid
                    counts.append(
                        occupied.sum(dim=1).detach().cpu().numpy().astype(np.int32)
                    )
            missing_counts = (
                np.concatenate(counts, axis=0)
                if counts
                else np.zeros(0, dtype=np.int32)
            )
            for key, count in zip(missing_keys, missing_counts):
                count_cache[key] = int(count)
            result = np.asarray(
                [count_cache[key] for key in pose_keys],
                dtype=np.int32,
            )
            return result, {
                "device": str(device),
                "elapsed_s": float(time.perf_counter() - started),
                "query_voxels_per_pose": int(len(offsets)),
                "pose_count": int(len(poses)),
                "unique_query_pose_count": int(len(missing_poses)),
                "cache_hit_count": int(
                    len(poses) - len(missing_poses)
                ),
                "device_tensor_cached": True,
            }

        shape = np.asarray(self.occupancy.shape, dtype=np.int64)
        missing_counts = np.zeros(len(missing_poses), dtype=np.int32)
        for index, pose in enumerate(missing_poses):
            points = (
                np.asarray(pose["R"], dtype=np.float64) @ offsets.T
            ).T + np.asarray(pose["eef_pos"], dtype=np.float64)
            grid_index = np.rint(
                (points - self.origin.reshape(1, 3)) / float(self.voxel_m)
            ).astype(np.int64)
            valid = np.all(grid_index >= 0, axis=1) & np.all(
                grid_index < shape.reshape(1, 3),
                axis=1,
            )
            if np.any(valid):
                valid_index = grid_index[valid]
                missing_counts[index] = int(
                    self.occupancy[tuple(valid_index.T)].sum()
                )
        for key, count in zip(missing_keys, missing_counts):
            count_cache[key] = int(count)
        result = np.asarray(
            [count_cache[key] for key in pose_keys],
            dtype=np.int32,
        )
        return result, {
            "device": "numpy",
            "elapsed_s": float(time.perf_counter() - started),
            "query_voxels_per_pose": int(len(offsets)),
            "pose_count": int(len(poses)),
            "unique_query_pose_count": int(len(missing_poses)),
            "cache_hit_count": int(len(poses) - len(missing_poses)),
            "device_tensor_cached": False,
        }


def build_local_scene_occupancy(
    mesh,
    *,
    hit: Sequence[float],
    query_offsets: Iterable[np.ndarray],
    axial_z_m: np.ndarray = AXIAL_Z_M,
    anchor_radius_m: float = 0.0,
    voxel_m: float = VOXEL_M,
    contains_chunk: int = 200_000,
    ctx=None,
) -> DenseSceneOccupancy:
    """Sample exact v8 mesh occupancy in the complete candidate query envelope."""
    from .mesh_occupancy import MeshOccupancyEvaluator

    local_parts = [
        np.asarray(part, dtype=np.float64).reshape(-1, 3)
        for part in query_offsets
    ]
    maximum_radius = 0.0
    for axial_z in np.asarray(axial_z_m, dtype=np.float64):
        anchor_local = np.asarray([0.0, 0.0, axial_z], dtype=np.float64)
        for local in local_parts:
            if len(local):
                maximum_radius = max(
                    maximum_radius,
                    float(np.linalg.norm(local - anchor_local, axis=1).max()),
                )
    radius = float(anchor_radius_m) + maximum_radius + float(voxel_m) * 2.0
    center = np.asarray(hit, dtype=np.float64).reshape(3)
    origin = np.floor((center - radius) / float(voxel_m)) * float(voxel_m)
    upper = np.ceil((center + radius) / float(voxel_m)) * float(voxel_m)
    shape = np.rint((upper - origin) / float(voxel_m)).astype(np.int64) + 1
    total = int(np.prod(shape, dtype=np.int64))
    evaluator = MeshOccupancyEvaluator.build(
        mesh,
        voxel_m=float(voxel_m),
        query_bounds=np.stack((origin, upper), axis=0),
    )
    started = time.perf_counter()
    cancel_check = None
    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        cancel_check = lambda: ctx.raise_if_cancelled(
            "rgbd scene occupancy"
        )
    occupancy, grid_metadata = _contains_regular_grid(
        evaluator,
        origin=origin,
        shape=shape,
        chunk_size=int(contains_chunk),
        cancel_check=cancel_check,
    )
    metadata = {
        **evaluator.metadata(),
        **grid_metadata,
        "origin": origin.tolist(),
        "shape": [int(value) for value in shape],
        "grid_voxels": int(total),
        "occupied_voxels": int(occupancy.sum()),
        "candidate_envelope_radius_m": float(radius),
        "elapsed_s": float(time.perf_counter() - started),
    }
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] occupancy 3mm shape={metadata['shape']} "
            f"occupied={metadata['occupied_voxels']}/{total} "
            f"method={metadata['occupancy_method']} "
            f"elapsed={metadata['elapsed_s']:.2f}s"
        )
    return DenseSceneOccupancy(
        origin=origin,
        occupancy=occupancy,
        voxel_m=float(voxel_m),
        metadata=metadata,
    )


def _contains_regular_grid(
    evaluator,
    *,
    origin: np.ndarray,
    shape: np.ndarray,
    chunk_size: int,
    cancel_check=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Evaluate a regular grid while skipping component-external voxels."""
    grid_origin = np.asarray(origin, dtype=np.float64).reshape(3)
    grid_shape = np.asarray(shape, dtype=np.int64).reshape(3)
    if np.any(grid_shape <= 0):
        raise ValueError(f"invalid regular-grid shape {grid_shape.tolist()}")
    chunk = max(1, int(chunk_size))
    total = int(np.prod(grid_shape, dtype=np.int64))
    occupied = np.zeros(total, dtype=bool)

    if evaluator.method == "empty_components":
        return occupied.reshape(tuple(grid_shape)), {
            "regular_grid_strategy": "empty",
            "regular_grid_query_points": 0,
        }

    if evaluator.method in {"contains", "component_union_contains"}:
        if evaluator.method == "contains":
            components = (evaluator.mesh,)
            component_bounds = np.asarray(
                [evaluator.mesh.bounds],
                dtype=np.float64,
            )
        else:
            assert evaluator.components is not None
            component_bounds = np.asarray(
                [component.bounds for component in evaluator.components],
                dtype=np.float64,
            )
            components = evaluator.components

        query_points = 0
        for component, bounds in zip(components, component_bounds):
            lower = np.floor(
                (bounds[0] - 1e-12 - grid_origin)
                / float(evaluator.voxel_m)
            ).astype(np.int64)
            upper = np.ceil(
                (bounds[1] + 1e-12 - grid_origin)
                / float(evaluator.voxel_m)
            ).astype(np.int64)
            lower = np.maximum(lower, 0)
            upper = np.minimum(upper, grid_shape - 1)
            if np.any(lower > upper):
                continue

            local_shape = upper - lower + 1
            local_total = int(np.prod(local_shape, dtype=np.int64))
            local_yz = int(local_shape[1] * local_shape[2])
            for start in range(0, local_total, chunk):
                stop = min(local_total, start + chunk)
                linear = np.arange(start, stop, dtype=np.int64)
                ix = linear // local_yz
                remainder = linear - ix * local_yz
                iy = remainder // int(local_shape[2])
                iz = remainder - iy * int(local_shape[2])
                index = (
                    np.column_stack((ix, iy, iz))
                    + lower.reshape(1, 3)
                )
                flat_index = (
                    (index[:, 0] * grid_shape[1] + index[:, 1])
                    * grid_shape[2]
                    + index[:, 2]
                )
                points = (
                    grid_origin.reshape(1, 3)
                    + index * float(evaluator.voxel_m)
                )
                candidates = (
                    (~occupied[flat_index])
                    & np.all(
                        points >= bounds[0] - 1e-12,
                        axis=1,
                    )
                    & np.all(
                        points <= bounds[1] + 1e-12,
                        axis=1,
                    )
                )
                if np.any(candidates):
                    candidate_points = points[candidates]
                    candidate_flat_index = flat_index[candidates]
                    query_points += int(len(candidate_points))
                    occupied[candidate_flat_index] = np.asarray(
                        component.contains(candidate_points),
                        dtype=bool,
                    )
                if cancel_check is not None:
                    cancel_check()
        return occupied.reshape(tuple(grid_shape)), {
            "regular_grid_strategy": "component_aabb",
            "regular_grid_query_points": int(query_points),
        }

    yz = int(grid_shape[1] * grid_shape[2])
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        linear = np.arange(start, stop, dtype=np.int64)
        ix = linear // yz
        remainder = linear - ix * yz
        iy = remainder // int(grid_shape[2])
        iz = remainder - iy * int(grid_shape[2])
        points = (
            grid_origin.reshape(1, 3)
            + np.column_stack((ix, iy, iz))
            * float(evaluator.voxel_m)
        )
        occupied[start:stop] = evaluator.contains(points)
        if cancel_check is not None:
            cancel_check()
    return occupied.reshape(tuple(grid_shape)), {
        "regular_grid_strategy": "full_grid",
        "regular_grid_query_points": int(total),
    }


def pose_dedupe_key(pose: Dict[str, Any]) -> tuple:
    position = np.asarray(pose["eef_pos"], dtype=np.float64).reshape(3)
    quat = np.asarray(pose["quat"], dtype=np.float64).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    if quat[3] < 0.0:
        quat = -quat
    return tuple(np.round(position, 4).tolist()) + tuple(np.round(quat, 5).tolist())


def inherit_strict_ik_warm_start(
    pose: Dict[str, Any],
    parent: Dict[str, Any],
) -> Dict[str, List[float]]:
    """Attach strict parent arm solutions as optional per-pose IK seeds."""
    warm_start: Dict[str, List[float]] = {}
    parent_ik = parent.get("ik_filter") or {}
    for arm in ("left", "right"):
        if not bool(parent.get(f"{arm}_ik_ok", False)):
            continue
        q_arm = (parent_ik.get(arm) or {}).get("q_arm")
        if q_arm is None:
            continue
        q = np.asarray(q_arm, dtype=np.float64).reshape(-1)
        if len(q) not in (7, 8) or not np.all(np.isfinite(q)):
            continue
        warm_start[arm] = [float(value) for value in q]
    if warm_start:
        pose["ik_warm_start_q_by_arm"] = warm_start
    else:
        pose.pop("ik_warm_start_q_by_arm", None)
    return warm_start


def _prefilter_metric_arrays(
    poses: Sequence[Dict[str, Any]],
    overlap_counts: np.ndarray,
    grasp_counts: np.ndarray,
    inflated_counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlap = np.asarray(overlap_counts, dtype=np.int64)
    grasp = np.asarray(grasp_counts, dtype=np.int64)
    inflated = np.asarray(inflated_counts, dtype=np.int64)
    if not (len(poses) == len(overlap) == len(grasp) == len(inflated)):
        raise ValueError("pose and prefilter metric counts must have equal length")
    return overlap, grasp, inflated


def select_local_refinement_seeds(
    poses: List[Dict[str, Any]],
    overlap_counts: np.ndarray,
    grasp_counts: np.ndarray,
    inflated_counts: np.ndarray,
    *,
    limit: int = REFINEMENT_SEED_MAX,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample the grasp-collision boundary, including a zero-safe rescue."""
    overlap, grasp, inflated = _prefilter_metric_arrays(
        poses,
        overlap_counts,
        grasp_counts,
        inflated_counts,
    )
    voxel_cm3 = VOXEL_M ** 3 * 1e6
    original_limit_vox = int(
        math.floor(ORIGINAL_OVERLAP_MAX_CM3 / voxel_cm3 + 1e-12)
    )
    inflated_limit_vox = int(
        math.ceil(INFLATED_OVERLAP_MAX_CM3 / voxel_cm3 - 1e-12) - 1
    )
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for pose, original_vox, grasp_vox, inflated_vox in zip(
        poses,
        overlap,
        grasp,
        inflated,
    ):
        key = pose_dedupe_key(pose)
        if key in seen:
            continue
        seen.add(key)
        original_excess = max(int(original_vox) - original_limit_vox, 0)
        inflated_excess = max(int(inflated_vox) - inflated_limit_vox, 0)
        candidates.append(
            {
                "pose": pose,
                "grasp_vox": int(grasp_vox),
                "original_vox": int(original_vox),
                "inflated_vox": int(inflated_vox),
                "violation_vox": int(original_excess + inflated_excess),
                "strict_safe": bool(
                    original_excess == 0 and inflated_excess == 0
                ),
            }
        )

    selected_rows: List[Dict[str, Any]] = []
    selected_keys = set()

    def add_rows(rows: Iterable[Dict[str, Any]], count: int) -> None:
        if count <= 0:
            return
        for row in rows:
            key = pose_dedupe_key(row["pose"])
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected_rows.append(row)
            if len(selected_rows) >= int(limit) or count <= 1:
                return
            count -= 1

    strict = sorted(
        (row for row in candidates if row["strict_safe"]),
        key=lambda row: (
            -row["grasp_vox"],
            row["inflated_vox"],
            row["original_vox"],
        ),
    )
    add_rows(strict, min(8, int(limit)))

    near = [row for row in candidates if row["violation_vox"] <= 64]
    for penalty in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        ordered = sorted(
            near,
            key=lambda row: (
                -(row["grasp_vox"] - penalty * row["violation_vox"]),
                row["violation_vox"],
                -row["grasp_vox"],
            ),
        )
        add_rows(ordered, 4)
        if len(selected_rows) >= int(limit):
            break

    if len(selected_rows) < int(limit):
        add_rows(
            sorted(
                near,
                key=lambda row: (
                    -row["grasp_vox"],
                    row["violation_vox"],
                    row["inflated_vox"],
                    row["original_vox"],
                ),
            ),
            int(limit) - len(selected_rows),
        )

    fallback_used = False
    if not selected_rows and candidates:
        fallback_used = True
        rescue_limit = min(int(limit), int(RESCUE_ORIENTATION_SEED_MAX))
        minimum_violation = min(row["violation_vox"] for row in candidates)
        low_violation_band = [
            row
            for row in candidates
            if row["violation_vox"]
            <= minimum_violation + max(64, int(math.ceil(0.5 * minimum_violation)))
        ]
        add_rows(
            sorted(
                candidates,
                key=lambda row: (
                    row["violation_vox"],
                    row["inflated_vox"],
                    row["original_vox"],
                    -row["grasp_vox"],
                ),
            ),
            min(6, rescue_limit),
        )
        add_rows(
            sorted(
                low_violation_band,
                key=lambda row: (
                    -row["grasp_vox"],
                    row["violation_vox"],
                    row["inflated_vox"],
                ),
            ),
            min(4, max(0, rescue_limit - len(selected_rows))),
        )
        for penalty in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            add_rows(
                sorted(
                    candidates,
                    key=lambda row: (
                        -(
                            row["grasp_vox"]
                            - float(penalty) * row["violation_vox"]
                        ),
                        row["violation_vox"],
                        row["inflated_vox"],
                    ),
                ),
                min(2, max(0, rescue_limit - len(selected_rows))),
            )
            if len(selected_rows) >= rescue_limit:
                break
        if len(selected_rows) < rescue_limit:
            best_by_anchor: Dict[int, Dict[str, Any]] = {}
            for row in candidates:
                anchor_index = int(row["pose"].get("pi", -1))
                incumbent = best_by_anchor.get(anchor_index)
                if incumbent is None or (
                    row["violation_vox"],
                    -row["grasp_vox"],
                    row["inflated_vox"],
                ) < (
                    incumbent["violation_vox"],
                    -incumbent["grasp_vox"],
                    incumbent["inflated_vox"],
                ):
                    best_by_anchor[anchor_index] = row
            add_rows(
                sorted(
                    best_by_anchor.values(),
                    key=lambda row: (
                        row["violation_vox"],
                        -row["grasp_vox"],
                    ),
                ),
                rescue_limit - len(selected_rows),
            )

    seeds = [row["pose"] for row in selected_rows[: int(limit)]]
    return seeds, {
        "candidate_count": int(len(candidates)),
        "seed_count": int(len(seeds)),
        "zero_safe_rescue": bool(fallback_used),
        "strict_seed_count": int(
            sum(bool(row["strict_safe"]) for row in selected_rows[: int(limit)])
        ),
        "seed_grasp_max_cm3": (
            float(max(row["grasp_vox"] for row in selected_rows) * voxel_cm3)
            if selected_rows
            else 0.0
        ),
        "seed_violation_min_vox": (
            int(min(row["violation_vox"] for row in selected_rows))
            if selected_rows
            else None
        ),
        "seed_violation_max_vox": (
            int(max(row["violation_vox"] for row in selected_rows))
            if selected_rows
            else None
        ),
    }


def select_rescue_sampling_seeds(
    poses: Sequence[Dict[str, Any]],
    overlap_counts: np.ndarray,
    grasp_counts: np.ndarray,
    inflated_counts: np.ndarray,
    *,
    limit: int = RESCUE_SAMPLING_SEED_MAX,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Balance low violation, high grasp, and anchor diversity without gating."""
    overlap, grasp, inflated = _prefilter_metric_arrays(
        poses,
        overlap_counts,
        grasp_counts,
        inflated_counts,
    )
    voxel_cm3 = VOXEL_M ** 3 * 1e6
    original_limit_vox = int(
        math.floor(ORIGINAL_OVERLAP_MAX_CM3 / voxel_cm3 + 1e-12)
    )
    inflated_limit_vox = int(
        math.ceil(INFLATED_OVERLAP_MAX_CM3 / voxel_cm3 - 1e-12) - 1
    )
    rows: List[Dict[str, Any]] = []
    seen = set()
    for pose, original_vox, grasp_vox, inflated_vox in zip(
        poses,
        overlap,
        grasp,
        inflated,
    ):
        key = pose_dedupe_key(pose)
        if key in seen:
            continue
        seen.add(key)
        original_excess = max(int(original_vox) - original_limit_vox, 0)
        inflated_excess = max(int(inflated_vox) - inflated_limit_vox, 0)
        rows.append(
            {
                "pose": pose,
                "key": key,
                "pi": int(pose.get("pi", -1)),
                "original_vox": int(original_vox),
                "grasp_vox": int(grasp_vox),
                "inflated_vox": int(inflated_vox),
                "violation_vox": int(original_excess + inflated_excess),
                "strict_safe": bool(
                    original_excess == 0 and inflated_excess == 0
                ),
            }
        )

    selected: List[Dict[str, Any]] = []
    selected_keys = set()

    def add_rows(candidates: Iterable[Dict[str, Any]], count: int) -> None:
        for row in candidates:
            if len(selected) >= int(limit) or count <= 0:
                return
            if row["key"] in selected_keys:
                continue
            selected_keys.add(row["key"])
            selected.append(row)
            count -= 1

    strict = sorted(
        (row for row in rows if row["strict_safe"]),
        key=lambda row: (
            -row["grasp_vox"],
            row["inflated_vox"],
            row["original_vox"],
        ),
    )
    add_rows(strict, min(8, int(limit)))
    add_rows(
        sorted(
            rows,
            key=lambda row: (
                row["violation_vox"],
                row["inflated_vox"],
                row["original_vox"],
                -row["grasp_vox"],
            ),
        ),
        min(12, int(limit) - len(selected)),
    )
    for penalty in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        add_rows(
            sorted(
                rows,
                key=lambda row: (
                    -(
                        row["grasp_vox"]
                        - float(penalty) * row["violation_vox"]
                    ),
                    row["violation_vox"],
                    row["inflated_vox"],
                ),
            ),
            min(3, int(limit) - len(selected)),
        )
        if len(selected) >= int(limit):
            break

    if len(selected) < int(limit):
        best_by_anchor: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            incumbent = best_by_anchor.get(row["pi"])
            if incumbent is None or (
                row["violation_vox"],
                -row["grasp_vox"],
                row["inflated_vox"],
            ) < (
                incumbent["violation_vox"],
                -incumbent["grasp_vox"],
                incumbent["inflated_vox"],
            ):
                best_by_anchor[row["pi"]] = row
        add_rows(
            sorted(
                best_by_anchor.values(),
                key=lambda row: (
                    row["violation_vox"],
                    -row["grasp_vox"],
                ),
            ),
            int(limit) - len(selected),
        )
    if len(selected) < int(limit):
        add_rows(
            sorted(
                rows,
                key=lambda row: (
                    -row["grasp_vox"],
                    row["violation_vox"],
                    row["inflated_vox"],
                ),
            ),
            int(limit) - len(selected),
        )

    return [row["pose"] for row in selected], {
        "input_pose_count": int(len(poses)),
        "unique_candidate_count": int(len(rows)),
        "seed_count": int(len(selected)),
        "strict_seed_count": int(
            sum(bool(row["strict_safe"]) for row in selected)
        ),
        "selected_anchor_count": int(len({row["pi"] for row in selected})),
        "violation_min_vox": (
            int(min(row["violation_vox"] for row in selected))
            if selected
            else None
        ),
        "violation_max_vox": (
            int(max(row["violation_vox"] for row in selected))
            if selected
            else None
        ),
        "grasp_max_cm3": (
            float(max(row["grasp_vox"] for row in selected) * voxel_cm3)
            if selected
            else 0.0
        ),
    }


def generate_opening_contact_refinements(
    seeds: Sequence[Dict[str, Any]],
    *,
    fractions: Sequence[float] = CONTACT_AXIS_INTERPOLATION_FRACTIONS,
    axial_z_m: Sequence[float] = AXIAL_Z_M,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Densify contact depth between adjacent P1-P7 points on local Z."""
    lut = get_wrist_opening_lut((0.05, 0.05))
    axial_samples = np.asarray(axial_z_m, dtype=np.float64).reshape(-1)
    generated: List[Dict[str, Any]] = []
    generated_keys = set()
    contact_id = 0
    for seed_index, seed in enumerate(seeds):
        rotation = np.asarray(seed["R"], dtype=np.float64)
        anchor = np.asarray(seed["anchor"], dtype=np.float64)
        seed_local = np.asarray(seed["anchor_local"], dtype=np.float64)
        z = float(seed_local[2])
        nearest_index = int(np.argmin(np.abs(axial_samples - z)))
        neighbor_indices = [
            index
            for index in (nearest_index - 1, nearest_index + 1)
            if 0 <= index < len(axial_samples)
        ]
        local_specs: List[Tuple[float, str]] = []
        for neighbor_index in neighbor_indices:
            neighbor_z = float(axial_samples[neighbor_index])
            for fraction in fractions:
                t = float(fraction)
                if not (0.0 < t < 1.0):
                    continue
                interpolated_z = (1.0 - t) * z + t * neighbor_z
                local_specs.append(
                    (
                        float(interpolated_z),
                        f"axis_p{nearest_index + 1}_to_p"
                        f"{neighbor_index + 1}_t{t:.2f}",
                    )
                )
        local_points = np.asarray(
            [[0.0, 0.0, local_z] for local_z, _name in local_specs],
            dtype=np.float64,
        )
        if not len(local_points):
            continue
        inside = gap_opening_strict_mask_from_lut(local_points, lut)
        for (local_z, name), anchor_local, valid in zip(
            local_specs,
            local_points,
            inside,
        ):
            if not bool(valid):
                continue
            pose = dict(seed)
            pose["anchor"] = anchor.copy()
            pose["anchor_local"] = anchor_local.copy()
            pose["eef_pos"] = anchor - rotation @ anchor_local
            pose["contact_offset_refined"] = True
            pose["contact_offset_id"] = int(contact_id)
            pose["contact_seed_index"] = int(seed_index)
            pose["contact_offset_name"] = str(name)
            pose["contact_offset_local_m"] = [
                0.0,
                0.0,
                float(local_z),
            ]
            pose["axial_z_m"] = float(local_z)
            key = pose_dedupe_key(pose)
            if key in generated_keys:
                continue
            generated_keys.add(key)
            generated.append(pose)
            contact_id += 1
    return generated, {
        "seed_count": int(len(seeds)),
        "pose_count": int(len(generated)),
        "fractions": [float(value) for value in fractions],
        "coarse_axial_z_mm": [
            float(value) * 1000.0 for value in axial_samples
        ],
        "translation_axis_local": [0.0, 0.0, 1.0],
        "translation_axis_name": "gripper_symmetry_axis_local_z",
        "lateral_offsets_enabled": False,
        "contact_domain": "exact_strict_opening_center_axis",
    }


def generate_pre_ik_translation_refinements(
    seeds: Sequence[Dict[str, Any]],
    *,
    outward_normal_world: Optional[np.ndarray],
    local_offsets_m: Sequence[float] = PRE_IK_TRANSLATION_OFFSET_M,
    normal_clearance_m: Sequence[float] = PRE_IK_NORMAL_CLEARANCE_M,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Translate rescue poses before IK only along the local Z symmetry axis."""
    generated: List[Dict[str, Any]] = []
    generated_keys = set()
    translation_id = 0
    for seed_index, seed in enumerate(seeds):
        rotation = np.asarray(seed["R"], dtype=np.float64)
        base_pos = np.asarray(seed["eef_pos"], dtype=np.float64)
        anchor = np.asarray(seed["anchor"], dtype=np.float64)
        deltas: List[Tuple[np.ndarray, str, float]] = []
        for offset in local_offsets_m:
            local = np.asarray([0.0, 0.0, float(offset)], dtype=np.float64)
            deltas.append(
                (
                    rotation @ local,
                    "local_z",
                    float(offset),
                )
            )
        for delta_world, axis_name, offset in deltas:
            pose = dict(seed)
            eef_pos = base_pos + delta_world
            pose["eef_pos"] = eef_pos
            pose["anchor"] = anchor.copy()
            pose["anchor_local"] = rotation.T @ (anchor - eef_pos)
            pose["axial_z_m"] = float(pose["anchor_local"][2])
            pose["pre_ik_translation_refined"] = True
            pose["pre_ik_translation_id"] = int(translation_id)
            pose["pre_ik_translation_seed_index"] = int(seed_index)
            pose["pre_ik_translation_axis"] = str(axis_name)
            pose["pre_ik_translation_offset_m"] = float(offset)
            pose["pre_ik_translation_world_m"] = (
                np.asarray(delta_world, dtype=np.float64).tolist()
            )
            key = pose_dedupe_key(pose)
            if key in generated_keys:
                continue
            generated_keys.add(key)
            generated.append(pose)
            translation_id += 1
    return generated, {
        "seed_count": int(len(seeds)),
        "pose_count": int(len(generated)),
        "local_axes": ["z"],
        "local_offsets_mm": [
            float(value) * 1000.0 for value in local_offsets_m
        ],
        "normal_enabled": False,
        "outward_normal_ignored": bool(outward_normal_world is not None),
        "normal_clearance_mm": [
            float(value) * 1000.0 for value in normal_clearance_m
        ],
        "translation_axis_local": [0.0, 0.0, 1.0],
        "translation_axis_name": "gripper_symmetry_axis_local_z",
        "lateral_offsets_enabled": False,
    }


def generate_local_orientation_refinements(
    seeds: Sequence[Dict[str, Any]],
    *,
    tilt_deg: Sequence[float] = REFINEMENT_TILT_DEG,
    roll_deg: Sequence[float] = REFINEMENT_ROLL_DEG,
) -> List[Dict[str, Any]]:
    """Refine orientation while preserving the coarse anchor and P1-P7 point."""

    def rotation_x(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            dtype=np.float64,
        )

    def rotation_y(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=np.float64,
        )

    def rotation_z(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    refined: List[Dict[str, Any]] = []
    refine_id = 0
    for seed_index, seed in enumerate(seeds):
        base_rotation = np.asarray(seed["R"], dtype=np.float64)
        anchor = np.asarray(seed["anchor"], dtype=np.float64)
        anchor_local = np.asarray(seed["anchor_local"], dtype=np.float64)
        for x_deg in tilt_deg:
            for y_deg in tilt_deg:
                for z_deg in roll_deg:
                    if (
                        abs(float(x_deg)) < 1e-12
                        and abs(float(y_deg)) < 1e-12
                        and abs(float(z_deg)) < 1e-12
                    ):
                        continue
                    delta = (
                        rotation_z(math.radians(float(z_deg)))
                        @ rotation_y(math.radians(float(y_deg)))
                        @ rotation_x(math.radians(float(x_deg)))
                    )
                    rotation = base_rotation @ delta
                    pose = dict(seed)
                    pose["R"] = rotation
                    pose["quat"] = _mat_to_quat_xyzw(rotation)
                    pose["eef_pos"] = anchor - rotation @ anchor_local
                    pose["anchor"] = anchor.copy()
                    pose["anchor_local"] = anchor_local.copy()
                    pose["refined"] = True
                    pose["refine_id"] = int(refine_id)
                    pose["refine_seed_index"] = int(seed_index)
                    pose["refine_tilt_x_deg"] = float(x_deg)
                    pose["refine_tilt_y_deg"] = float(y_deg)
                    pose["refine_roll_deg"] = float(z_deg)
                    refined.append(pose)
                    refine_id += 1
    return refined


def generate_quality_pose_interpolations(
    poses: Sequence[Dict[str, Any]],
    overlap_counts: np.ndarray,
    grasp_counts: np.ndarray,
    inflated_counts: np.ndarray,
    *,
    seed_limit: int = INTERPOLATION_SEED_MAX,
    pair_limit: int = INTERPOLATION_PAIR_MAX,
    neighbors_per_seed: int = INTERPOLATION_NEIGHBORS_PER_SEED,
    fractions: Sequence[float] = INTERPOLATION_FRACTIONS,
    grasp_ratio: float = INTERPOLATION_GRASP_RATIO,
    min_angle_deg: float = INTERPOLATION_MIN_ANGLE_DEG,
    max_angle_deg: float = INTERPOLATION_MAX_ANGLE_DEG,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Bridge good same-contact poses with anchor-preserving SE(3) samples."""
    overlap, grasp, inflated = _prefilter_metric_arrays(
        poses,
        overlap_counts,
        grasp_counts,
        inflated_counts,
    )
    voxel_cm3 = VOXEL_M ** 3 * 1e6
    original_limit_vox = int(
        math.floor(ORIGINAL_OVERLAP_MAX_CM3 / voxel_cm3 + 1e-12)
    )
    inflated_limit_vox = int(
        math.ceil(INFLATED_OVERLAP_MAX_CM3 / voxel_cm3 - 1e-12) - 1
    )
    unique_rows: List[Dict[str, Any]] = []
    seen_pose_keys = set()
    for index, (pose, original_vox, grasp_vox, inflated_vox) in enumerate(
        zip(poses, overlap, grasp, inflated)
    ):
        if (
            int(original_vox) > original_limit_vox
            or int(inflated_vox) > inflated_limit_vox
        ):
            continue
        key = pose_dedupe_key(pose)
        if key in seen_pose_keys:
            continue
        seen_pose_keys.add(key)
        angle = float(pose.get("normal_alignment_deg", float("inf")))
        unique_rows.append(
            {
                "index": int(index),
                "pose": pose,
                "key": key,
                "grasp_vox": int(grasp_vox),
                "original_vox": int(original_vox),
                "inflated_vox": int(inflated_vox),
                "normal_alignment_deg": angle,
            }
        )

    best_grasp_vox = max(
        (row["grasp_vox"] for row in unique_rows),
        default=0,
    )
    grasp_floor_vox = int(
        math.ceil(float(grasp_ratio) * float(best_grasp_vox) - 1e-12)
    )
    eligible = [
        row
        for row in unique_rows
        if best_grasp_vox > 0 and row["grasp_vox"] >= grasp_floor_vox
    ]

    quality_order = sorted(
        eligible,
        key=lambda row: (
            -row["grasp_vox"],
            row["inflated_vox"],
            row["original_vox"],
            row["normal_alignment_deg"],
        ),
    )
    alignment_order = sorted(
        (
            row
            for row in eligible
            if math.isfinite(row["normal_alignment_deg"])
        ),
        key=lambda row: (
            row["normal_alignment_deg"],
            -row["grasp_vox"],
            row["inflated_vox"],
            row["original_vox"],
        ),
    )
    selected_rows: List[Dict[str, Any]] = []
    selected_keys = set()

    def add_rows(rows: Iterable[Dict[str, Any]], count: int) -> None:
        for row in rows:
            if len(selected_rows) >= int(seed_limit) or count <= 0:
                return
            if row["key"] in selected_keys:
                continue
            selected_keys.add(row["key"])
            selected_rows.append(row)
            count -= 1

    quality_quota = (int(seed_limit) + 1) // 2
    add_rows(quality_order, quality_quota)
    add_rows(alignment_order, int(seed_limit) - len(selected_rows))
    add_rows(quality_order, int(seed_limit) - len(selected_rows))

    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in selected_rows:
        pose = row["pose"]
        grouped.setdefault(
            (int(pose.get("pi", -1)), int(pose.get("ai", -1))),
            [],
        ).append(row)

    pair_rows: List[Dict[str, Any]] = []
    seen_pairs = set()
    for group_key, rows in grouped.items():
        if len(rows) < 2:
            continue
        for left_index, left in enumerate(rows):
            neighbors = []
            for right_index, right in enumerate(rows):
                if left_index == right_index:
                    continue
                angle_deg = _quat_distance_deg(
                    left["pose"]["quat"],
                    right["pose"]["quat"],
                )
                if (
                    float(min_angle_deg) - 1e-12
                    <= angle_deg
                    <= float(max_angle_deg) + 1e-12
                ):
                    neighbors.append((angle_deg, right_index, right))
            neighbors.sort(key=lambda item: (item[0], -item[2]["grasp_vox"]))
            if not neighbors:
                continue
            choice_indices = {0, len(neighbors) - 1}
            if int(neighbors_per_seed) >= 3:
                choice_indices.add(len(neighbors) // 2)
            if int(neighbors_per_seed) > 3 and len(neighbors) > 3:
                for rank in np.linspace(
                    0,
                    len(neighbors) - 1,
                    int(neighbors_per_seed),
                    dtype=np.int64,
                ):
                    choice_indices.add(int(rank))
            for choice_index in sorted(choice_indices)[
                : max(0, int(neighbors_per_seed))
            ]:
                angle_deg, _, right = neighbors[choice_index]
                pair_key = tuple(
                    sorted((int(left["index"]), int(right["index"])))
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pair_rows.append(
                    {
                        "group_key": group_key,
                        "left": left,
                        "right": right,
                        "angle_deg": float(angle_deg),
                        "min_grasp_vox": min(
                            left["grasp_vox"],
                            right["grasp_vox"],
                        ),
                        "max_inflated_vox": max(
                            left["inflated_vox"],
                            right["inflated_vox"],
                        ),
                        "max_original_vox": max(
                            left["original_vox"],
                            right["original_vox"],
                        ),
                    }
                )

    pair_rows.sort(
        key=lambda row: (
            -row["min_grasp_vox"],
            row["max_inflated_vox"],
            row["max_original_vox"],
            -row["angle_deg"],
            row["group_key"],
        )
    )
    selected_pairs = pair_rows[: max(0, int(pair_limit))]
    source_pose_keys = {pose_dedupe_key(pose) for pose in poses}
    generated_keys = set()
    interpolated: List[Dict[str, Any]] = []
    rejected_anchor_mismatch = 0
    interpolation_id = 0

    def parent_summary(row: Dict[str, Any]) -> Dict[str, Any]:
        pose = row["pose"]
        return {
            "source_index": int(row["index"]),
            "pi": int(pose.get("pi", -1)),
            "ni": int(pose.get("ni", -1)),
            "ri": int(pose.get("ri", -1)),
            "ai": int(pose.get("ai", -1)),
            "refined": bool(pose.get("refined", False)),
            "grasp_cm3": float(row["grasp_vox"] * voxel_cm3),
            "normal_alignment_deg": (
                float(row["normal_alignment_deg"])
                if math.isfinite(row["normal_alignment_deg"])
                else None
            ),
        }

    for pair in selected_pairs:
        left = pair["left"]
        right = pair["right"]
        left_pose = left["pose"]
        right_pose = right["pose"]
        anchor = np.asarray(left_pose["anchor"], dtype=np.float64).reshape(3)
        anchor_local = np.asarray(
            left_pose["anchor_local"],
            dtype=np.float64,
        ).reshape(3)
        if (
            not np.allclose(
                anchor,
                np.asarray(right_pose["anchor"], dtype=np.float64).reshape(3),
                atol=1e-9,
                rtol=0.0,
            )
            or not np.allclose(
                anchor_local,
                np.asarray(
                    right_pose["anchor_local"],
                    dtype=np.float64,
                ).reshape(3),
                atol=1e-9,
                rtol=0.0,
            )
        ):
            rejected_anchor_mismatch += 1
            continue
        for fraction in fractions:
            t = float(fraction)
            if not (0.0 < t < 1.0):
                continue
            quat = _quat_slerp_xyzw(
                left_pose["quat"],
                right_pose["quat"],
                t,
            )
            rotation = _quat_to_mat_xyzw(quat)
            pose = dict(left_pose)
            pose["R"] = rotation
            pose["quat"] = quat
            pose["eef_pos"] = anchor - rotation @ anchor_local
            pose["anchor"] = anchor.copy()
            pose["anchor_local"] = anchor_local.copy()
            pose["refined"] = False
            pose["interpolated"] = True
            pose["interpolation_id"] = int(interpolation_id)
            pose["interpolation_t"] = t
            pose["interpolation_parent_angle_deg"] = float(pair["angle_deg"])
            pose["interpolation_parent_a"] = parent_summary(left)
            pose["interpolation_parent_b"] = parent_summary(right)
            key = pose_dedupe_key(pose)
            if key in source_pose_keys or key in generated_keys:
                continue
            generated_keys.add(key)
            interpolated.append(pose)
            interpolation_id += 1

    return interpolated, {
        "strict_safe_unique_count": int(len(unique_rows)),
        "best_grasp_cm3": float(best_grasp_vox * voxel_cm3),
        "grasp_ratio": float(grasp_ratio),
        "grasp_floor_cm3": float(grasp_floor_vox * voxel_cm3),
        "eligible_count": int(len(eligible)),
        "seed_limit": int(seed_limit),
        "seed_count": int(len(selected_rows)),
        "group_count": int(len(grouped)),
        "pair_candidate_count": int(len(pair_rows)),
        "pair_limit": int(pair_limit),
        "pair_count": int(len(selected_pairs)),
        "neighbors_per_seed": int(neighbors_per_seed),
        "fractions": [float(value) for value in fractions],
        "min_angle_deg": float(min_angle_deg),
        "max_angle_deg": float(max_angle_deg),
        "selected_pair_angle_min_deg": (
            float(min(row["angle_deg"] for row in selected_pairs))
            if selected_pairs
            else None
        ),
        "selected_pair_angle_max_deg": (
            float(max(row["angle_deg"] for row in selected_pairs))
            if selected_pairs
            else None
        ),
        "rejected_anchor_mismatch_count": int(rejected_anchor_mismatch),
        "pose_count": int(len(interpolated)),
    }


def select_joint_pareto_refinement_seeds(
    poses: Sequence[Dict[str, Any]],
    overlap_counts: np.ndarray,
    grasp_counts: np.ndarray,
    inflated_counts: np.ndarray,
    *,
    limit: int = PARETO_MICRO_SEED_MAX,
    grasp_ratio: float = PARETO_MICRO_GRASP_RATIO,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select collision-safe seeds spanning the grasp/normal Pareto frontier."""
    overlap, grasp, inflated = _prefilter_metric_arrays(
        poses,
        overlap_counts,
        grasp_counts,
        inflated_counts,
    )
    voxel_cm3 = VOXEL_M ** 3 * 1e6
    original_limit_vox = int(
        math.floor(ORIGINAL_OVERLAP_MAX_CM3 / voxel_cm3 + 1e-12)
    )
    inflated_limit_vox = int(
        math.ceil(INFLATED_OVERLAP_MAX_CM3 / voxel_cm3 - 1e-12) - 1
    )
    rows: List[Dict[str, Any]] = []
    seen = set()
    for pose, original_vox, grasp_vox, inflated_vox in zip(
        poses,
        overlap,
        grasp,
        inflated,
    ):
        if (
            int(original_vox) > original_limit_vox
            or int(inflated_vox) > inflated_limit_vox
        ):
            continue
        angle = float(pose.get("normal_alignment_deg", float("inf")))
        if not math.isfinite(angle):
            continue
        key = pose_dedupe_key(pose)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "pose": pose,
                "key": key,
                "angle_deg": angle,
                "grasp_vox": int(grasp_vox),
                "original_vox": int(original_vox),
                "inflated_vox": int(inflated_vox),
            }
        )

    best_grasp_vox = max((row["grasp_vox"] for row in rows), default=0)
    grasp_floor_vox = int(
        math.ceil(float(grasp_ratio) * float(best_grasp_vox) - 1e-12)
    )
    eligible = [
        row
        for row in rows
        if best_grasp_vox > 0 and row["grasp_vox"] >= grasp_floor_vox
    ]
    by_angle = sorted(
        eligible,
        key=lambda row: (
            row["angle_deg"],
            -row["grasp_vox"],
            row["inflated_vox"],
            row["original_vox"],
        ),
    )
    frontier: List[Dict[str, Any]] = []
    best_seen_grasp = -1
    for row in by_angle:
        if row["grasp_vox"] <= best_seen_grasp:
            continue
        frontier.append(row)
        best_seen_grasp = row["grasp_vox"]

    selected_rows: List[Dict[str, Any]] = []
    selected_keys = set()

    def add_rows(candidates: Iterable[Dict[str, Any]], count: int) -> None:
        for row in candidates:
            if len(selected_rows) >= int(limit) or count <= 0:
                return
            if row["key"] in selected_keys:
                continue
            selected_keys.add(row["key"])
            selected_rows.append(row)
            count -= 1

    reserve = min(5, max(1, int(limit) // 4))
    add_rows(by_angle, reserve)
    add_rows(
        sorted(
            eligible,
            key=lambda row: (
                -row["grasp_vox"],
                row["angle_deg"],
                row["inflated_vox"],
            ),
        ),
        reserve,
    )
    remaining = int(limit) - len(selected_rows)
    if remaining > 0 and frontier:
        for index in np.linspace(
            0,
            len(frontier) - 1,
            min(remaining, len(frontier)),
            dtype=np.int64,
        ):
            add_rows((frontier[int(index)],), 1)
    if len(selected_rows) < int(limit):
        balanced = sorted(
            eligible,
            key=lambda row: (
                -(
                    float(row["grasp_vox"]) / max(float(best_grasp_vox), 1.0)
                    - float(row["angle_deg"]) / 90.0
                ),
                -row["grasp_vox"],
                row["angle_deg"],
            ),
        )
        add_rows(balanced, int(limit) - len(selected_rows))

    def row_summary(row: Dict[str, Any]) -> Dict[str, Any]:
        pose = row["pose"]
        return {
            "normal_alignment_deg": float(row["angle_deg"]),
            "grasp_cm3": float(row["grasp_vox"] * voxel_cm3),
            "overlap_cm3": float(row["original_vox"] * voxel_cm3),
            "inflated_overlap_cm3": float(
                row["inflated_vox"] * voxel_cm3
            ),
            "pi": int(pose.get("pi", -1)),
            "ai": int(pose.get("ai", -1)),
            "refined": bool(pose.get("refined", False)),
            "interpolated": bool(pose.get("interpolated", False)),
        }

    return [row["pose"] for row in selected_rows], {
        "strict_safe_finite_normal_count": int(len(rows)),
        "best_grasp_cm3": float(best_grasp_vox * voxel_cm3),
        "grasp_ratio": float(grasp_ratio),
        "grasp_floor_cm3": float(grasp_floor_vox * voxel_cm3),
        "eligible_count": int(len(eligible)),
        "pareto_frontier_count": int(len(frontier)),
        "seed_limit": int(limit),
        "seed_count": int(len(selected_rows)),
        "pareto_frontier": [row_summary(row) for row in frontier[:40]],
        "selected_seeds": [row_summary(row) for row in selected_rows],
    }


def generate_pareto_micro_refinements(
    seeds: Sequence[Dict[str, Any]],
    *,
    tilt_deg: Sequence[float] = PARETO_MICRO_TILT_DEG,
    roll_deg: Sequence[float] = PARETO_MICRO_ROLL_DEG,
) -> List[Dict[str, Any]]:
    refined = generate_local_orientation_refinements(
        seeds,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
    )
    for pose in refined:
        seed_index = int(pose["refine_seed_index"])
        parent = seeds[seed_index]
        inherit_strict_ik_warm_start(pose, parent)
        pose["micro_parent_refined"] = bool(parent.get("refined", False))
        pose["micro_parent_interpolated"] = bool(
            parent.get("interpolated", False)
        )
        pose["micro_refined"] = True
        pose["micro_refine_id"] = int(pose.pop("refine_id"))
        pose["micro_seed_index"] = int(pose.pop("refine_seed_index"))
        pose["micro_tilt_x_deg"] = float(pose.pop("refine_tilt_x_deg"))
        pose["micro_tilt_y_deg"] = float(pose.pop("refine_tilt_y_deg"))
        pose["micro_roll_deg"] = float(pose.pop("refine_roll_deg"))
        for key in tuple(pose):
            if key.startswith("interpolation_"):
                pose.pop(key)
        pose["refined"] = False
        pose["interpolated"] = False
    return refined


def generate_reachability_bridge_interpolations(
    reachable_seeds: Sequence[Dict[str, Any]],
    target_poses: Sequence[Dict[str, Any]],
    *,
    targets_per_seed: int = REACHABILITY_BRIDGE_TARGETS_PER_SEED,
    fractions: Sequence[float] = REACHABILITY_BRIDGE_FRACTIONS,
    min_angle_deg: float = REACHABILITY_BRIDGE_MIN_ANGLE_DEG,
    max_angle_deg: float = REACHABILITY_BRIDGE_MAX_ANGLE_DEG,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Bridge strict-reachable poses toward safer high-grasp orientations."""
    safe_targets: List[Dict[str, Any]] = []
    seen_targets = set()
    for pose in target_poses:
        if (
            float(pose.get("env_overlap_cm3", float("inf")))
            > ORIGINAL_OVERLAP_MAX_CM3 + 1e-12
            or float(
                pose.get(
                    "prefilter_inflated_overlap_cm3",
                    float("inf"),
                )
            )
            >= INFLATED_OVERLAP_MAX_CM3
        ):
            continue
        key = pose_dedupe_key(pose)
        if key in seen_targets:
            continue
        seen_targets.add(key)
        safe_targets.append(pose)

    grouped_targets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for pose in safe_targets:
        grouped_targets.setdefault(
            (int(pose.get("pi", -1)), int(pose.get("ai", -1))),
            [],
        ).append(pose)

    generated: List[Dict[str, Any]] = []
    generated_keys = set()
    selected_target_count = 0
    eligible_pair_count = 0
    bridge_id = 0
    selected_pair_rows: List[Dict[str, Any]] = []
    for seed_index, seed in enumerate(reachable_seeds):
        parent_grasp = float(
            seed.get(
                "grasp_vol_cm3",
                seed.get("prefilter_grasp_cm3", 0.0),
            )
        )
        anchor = np.asarray(seed["anchor"], dtype=np.float64).reshape(3)
        anchor_local = np.asarray(
            seed["anchor_local"],
            dtype=np.float64,
        ).reshape(3)
        candidates: List[Dict[str, Any]] = []
        for target in grouped_targets.get(
            (int(seed.get("pi", -1)), int(seed.get("ai", -1))),
            [],
        ):
            target_grasp = float(target.get("prefilter_grasp_cm3", 0.0))
            if target_grasp <= parent_grasp + 1e-12:
                continue
            if (
                not np.allclose(
                    anchor,
                    np.asarray(target["anchor"], dtype=np.float64).reshape(3),
                    atol=1e-9,
                    rtol=0.0,
                )
                or not np.allclose(
                    anchor_local,
                    np.asarray(
                        target["anchor_local"],
                        dtype=np.float64,
                    ).reshape(3),
                    atol=1e-9,
                    rtol=0.0,
                )
            ):
                continue
            angle_deg = _quat_distance_deg(seed["quat"], target["quat"])
            if not (
                float(min_angle_deg) - 1e-12
                <= angle_deg
                <= float(max_angle_deg) + 1e-12
            ):
                continue
            candidates.append(
                {
                    "target": target,
                    "angle_deg": float(angle_deg),
                    "target_grasp_cm3": target_grasp,
                    "target_normal_deg": float(
                        target.get("normal_alignment_deg", float("inf"))
                    ),
                }
            )
        eligible_pair_count += len(candidates)
        quality_order = sorted(
            candidates,
            key=lambda row: (
                -row["target_grasp_cm3"],
                row["angle_deg"],
                row["target_normal_deg"],
            ),
        )
        near_order = sorted(
            candidates,
            key=lambda row: (
                row["angle_deg"],
                -row["target_grasp_cm3"],
                row["target_normal_deg"],
            ),
        )
        selected_rows: List[Dict[str, Any]] = []
        selected_keys = set()

        def add_target_rows(
            rows: Iterable[Dict[str, Any]],
            count: int,
        ) -> None:
            for row in rows:
                if len(selected_rows) >= int(targets_per_seed) or count <= 0:
                    return
                key = pose_dedupe_key(row["target"])
                if key in selected_keys:
                    continue
                selected_keys.add(key)
                selected_rows.append(row)
                count -= 1

        quality_quota = (int(targets_per_seed) + 1) // 2
        add_target_rows(quality_order, quality_quota)
        add_target_rows(
            near_order,
            int(targets_per_seed) - len(selected_rows),
        )
        add_target_rows(
            quality_order,
            int(targets_per_seed) - len(selected_rows),
        )
        selected_target_count += len(selected_rows)
        for row in selected_rows:
            target = row["target"]
            selected_pair_rows.append(
                {
                    "seed_index": int(seed_index),
                    "parent_grasp_cm3": float(parent_grasp),
                    "target_grasp_cm3": float(row["target_grasp_cm3"]),
                    "parent_angle_deg": (
                        float(seed["normal_alignment_deg"])
                        if seed.get("normal_alignment_deg") is not None
                        else None
                    ),
                    "target_angle_deg": (
                        float(row["target_normal_deg"])
                        if math.isfinite(row["target_normal_deg"])
                        else None
                    ),
                    "quaternion_angle_deg": float(row["angle_deg"]),
                }
            )
            for fraction in fractions:
                t = float(fraction)
                if not (0.0 < t < 1.0):
                    continue
                quat = _quat_slerp_xyzw(seed["quat"], target["quat"], t)
                rotation = _quat_to_mat_xyzw(quat)
                pose = dict(seed)
                pose["R"] = rotation
                pose["quat"] = quat
                pose["eef_pos"] = anchor - rotation @ anchor_local
                pose["anchor"] = anchor.copy()
                pose["anchor_local"] = anchor_local.copy()
                pose["refined"] = False
                pose["interpolated"] = True
                pose["micro_refined"] = False
                pose["reachability_bridge"] = True
                pose["bridge_id"] = int(bridge_id)
                pose["bridge_seed_index"] = int(seed_index)
                pose["micro_seed_index"] = int(seed_index)
                pose["interpolation_t"] = t
                pose["interpolation_parent_angle_deg"] = float(
                    row["angle_deg"]
                )
                pose["bridge_parent_grasp_cm3"] = float(parent_grasp)
                pose["bridge_target_grasp_cm3"] = float(
                    row["target_grasp_cm3"]
                )
                inherit_strict_ik_warm_start(pose, seed)
                key = pose_dedupe_key(pose)
                if key in generated_keys:
                    continue
                generated_keys.add(key)
                generated.append(pose)
                bridge_id += 1

    return generated, {
        "reachable_seed_count": int(len(reachable_seeds)),
        "safe_target_count": int(len(safe_targets)),
        "eligible_pair_count": int(eligible_pair_count),
        "targets_per_seed": int(targets_per_seed),
        "selected_target_count": int(selected_target_count),
        "fractions": [float(value) for value in fractions],
        "min_angle_deg": float(min_angle_deg),
        "max_angle_deg": float(max_angle_deg),
        "pose_count": int(len(generated)),
        "selected_pairs": selected_pair_rows[:80],
    }


def generate_reachable_se3_interpolations(
    reachable_poses: Sequence[Dict[str, Any]],
    *,
    endpoint_max: int = REACHABLE_SE3_ENDPOINT_MAX,
    pair_max: int = REACHABLE_SE3_PAIR_MAX,
    fractions: Optional[Sequence[float]] = None,
    position_fractions: Sequence[float] = REACHABLE_SE3_FRACTIONS,
    orientation_fractions: Sequence[
        float
    ] = REACHABLE_SE3_ORIENTATION_FRACTIONS,
    min_grasp_ratio: float = REACHABLE_SE3_MIN_GRASP_RATIO,
    min_quat_angle_deg: float = REACHABLE_SE3_MIN_QUAT_ANGLE_DEG,
    max_quat_angle_deg: float = REACHABLE_SE3_MAX_QUAT_ANGLE_DEG,
    max_eef_distance_m: float = REACHABLE_SE3_MAX_EEF_DISTANCE_M,
    max_anchor_distance_m: float = REACHABLE_SE3_MAX_ANCHOR_DISTANCE_M,
    adjacent_frontier_only: bool = False,
    require_common_strict_arm: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Interpolate orientation and local-Z depth without moving off click axis."""
    if fractions is not None:
        interpolation_grid = [
            (float(value), float(value))
            for value in fractions
        ]
        coupled_fractions = [float(value) for value in fractions]
    else:
        interpolation_grid = [
            (float(position_t), float(orientation_t))
            for position_t in position_fractions
            for orientation_t in orientation_fractions
        ]
        coupled_fractions = None
    unique_rows: List[Dict[str, Any]] = []
    seen = set()
    for pose in reachable_poses:
        if (
            float(pose.get("overlap_vol_cm3", float("inf")))
            > ORIGINAL_OVERLAP_MAX_CM3 + 1e-12
            or float(
                pose.get("inflated_overlap_vol_cm3", float("inf"))
            )
            >= INFLATED_OVERLAP_MAX_CM3
        ):
            continue
        angle_deg = float(
            pose.get("normal_alignment_deg", float("inf"))
        )
        grasp_cm3 = float(pose.get("grasp_vol_cm3", 0.0))
        if not math.isfinite(angle_deg) or grasp_cm3 <= 0.0:
            continue
        strict_arms = tuple(
            arm
            for arm in ("left", "right")
            if bool(pose.get(f"{arm}_ik_ok", False))
            and (pose.get("ik_filter") or {}).get(arm, {}).get("q_arm")
            is not None
        )
        if not strict_arms:
            continue
        key = pose_dedupe_key(pose)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(
            {
                "pose": pose,
                "key": key,
                "angle_deg": angle_deg,
                "grasp_cm3": grasp_cm3,
                "strict_arms": strict_arms,
            }
        )

    best_grasp_cm3 = max(
        (row["grasp_cm3"] for row in unique_rows),
        default=0.0,
    )
    eligible = [
        row
        for row in unique_rows
        if row["grasp_cm3"]
        >= float(min_grasp_ratio) * best_grasp_cm3 - 1e-12
    ]
    by_angle = sorted(
        eligible,
        key=lambda row: (
            row["angle_deg"],
            -row["grasp_cm3"],
        ),
    )
    frontier: List[Dict[str, Any]] = []
    frontier_grasp = float("-inf")
    for row in by_angle:
        if row["grasp_cm3"] > frontier_grasp + 1e-12:
            frontier.append(row)
            frontier_grasp = row["grasp_cm3"]
    if len(frontier) > int(endpoint_max):
        keep_indices = np.linspace(
            0,
            len(frontier) - 1,
            int(endpoint_max),
            dtype=np.int64,
        )
        frontier = [frontier[int(index)] for index in keep_indices]

    pair_rows: List[Dict[str, Any]] = []
    rejected_anchor_mismatch = 0
    rejected_off_axis_endpoint = 0
    for left_index, left in enumerate(frontier):
        right_rows = frontier[left_index + 1 :]
        if adjacent_frontier_only:
            right_rows = right_rows[:1]
        for right in right_rows:
            common_arms = tuple(
                sorted(
                    set(left["strict_arms"]).intersection(
                        right["strict_arms"]
                    )
                )
            )
            if require_common_strict_arm and not common_arms:
                continue
            left_pose = left["pose"]
            right_pose = right["pose"]
            left_anchor = np.asarray(
                left_pose["anchor"],
                dtype=np.float64,
            ).reshape(3)
            right_anchor = np.asarray(
                right_pose["anchor"],
                dtype=np.float64,
            ).reshape(3)
            if not np.allclose(
                left_anchor,
                right_anchor,
                atol=1e-9,
                rtol=0.0,
            ):
                rejected_anchor_mismatch += 1
                continue
            left_anchor_local = np.asarray(
                left_pose["anchor_local"],
                dtype=np.float64,
            ).reshape(3)
            right_anchor_local = np.asarray(
                right_pose["anchor_local"],
                dtype=np.float64,
            ).reshape(3)
            if (
                float(np.linalg.norm(left_anchor_local[:2])) > 1e-9
                or float(np.linalg.norm(right_anchor_local[:2])) > 1e-9
            ):
                rejected_off_axis_endpoint += 1
                continue
            if int(left_pose.get("ai", -1)) != int(
                right_pose.get("ai", -1)
            ):
                continue
            quat_angle_deg = _quat_distance_deg(
                left_pose["quat"],
                right_pose["quat"],
            )
            if not (
                float(min_quat_angle_deg) - 1e-12
                <= quat_angle_deg
                <= float(max_quat_angle_deg) + 1e-12
            ):
                continue
            eef_distance_m = float(
                np.linalg.norm(
                    np.asarray(left_pose["eef_pos"], dtype=np.float64)
                    - np.asarray(right_pose["eef_pos"], dtype=np.float64)
                )
            )
            anchor_distance_m = float(
                np.linalg.norm(
                    np.asarray(left_pose["anchor"], dtype=np.float64)
                    - np.asarray(right_pose["anchor"], dtype=np.float64)
                )
            )
            if (
                eef_distance_m > float(max_eef_distance_m) + 1e-12
                or anchor_distance_m
                > float(max_anchor_distance_m) + 1e-12
            ):
                continue
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "common_arms": common_arms,
                    "quat_angle_deg": float(quat_angle_deg),
                    "eef_distance_m": eef_distance_m,
                    "anchor_distance_m": anchor_distance_m,
                    "grasp_span_cm3": abs(
                        left["grasp_cm3"] - right["grasp_cm3"]
                    ),
                    "angle_span_deg": abs(
                        left["angle_deg"] - right["angle_deg"]
                    ),
                }
            )
    pair_rows.sort(
        key=lambda row: (
            -min(
                row["left"]["grasp_cm3"],
                row["right"]["grasp_cm3"],
            ),
            -row["grasp_span_cm3"] * row["angle_span_deg"],
            row["quat_angle_deg"],
            row["eef_distance_m"],
        )
    )
    selected_pairs = pair_rows[: max(0, int(pair_max))]
    source_keys = {row["key"] for row in unique_rows}
    generated_keys = set()
    generated: List[Dict[str, Any]] = []
    pair_summaries: List[Dict[str, Any]] = []
    interpolation_id = 0
    for pair_id, pair in enumerate(selected_pairs):
        left = pair["left"]
        right = pair["right"]
        left_pose = left["pose"]
        right_pose = right["pose"]
        anchor = np.asarray(left_pose["anchor"], dtype=np.float64).reshape(3)
        left_anchor_local = np.asarray(
            left_pose["anchor_local"],
            dtype=np.float64,
        )
        right_anchor_local = np.asarray(
            right_pose["anchor_local"],
            dtype=np.float64,
        )
        pair_summaries.append(
            {
                "pair_id": int(pair_id),
                "common_arms": list(pair["common_arms"]),
                "left_grasp_cm3": float(left["grasp_cm3"]),
                "left_angle_deg": float(left["angle_deg"]),
                "right_grasp_cm3": float(right["grasp_cm3"]),
                "right_angle_deg": float(right["angle_deg"]),
                "quat_angle_deg": float(pair["quat_angle_deg"]),
                "eef_distance_mm": float(
                    pair["eef_distance_m"] * 1000.0
                ),
                "anchor_distance_mm": float(
                    pair["anchor_distance_m"] * 1000.0
                ),
            }
        )
        for position_t, orientation_t in interpolation_grid:
            if not (
                0.0 < float(position_t) < 1.0
                and 0.0 <= float(orientation_t) <= 1.0
            ):
                continue
            quat = _quat_slerp_xyzw(
                left_pose["quat"],
                right_pose["quat"],
                float(orientation_t),
            )
            rotation = _quat_to_mat_xyzw(quat)
            interpolated_z = (
                (1.0 - float(position_t)) * left_anchor_local
                + float(position_t) * right_anchor_local
            )[2]
            anchor_local = np.asarray(
                [0.0, 0.0, float(interpolated_z)],
                dtype=np.float64,
            )
            eef_pos = anchor - rotation @ anchor_local
            pose = dict(left_pose)
            pose["eef_pos"] = eef_pos
            pose["quat"] = quat
            pose["R"] = rotation
            pose["anchor_local"] = anchor_local
            pose["anchor"] = anchor.copy()
            pose["axial_z_m"] = float(interpolated_z)
            pose["refined"] = False
            pose["interpolated"] = True
            pose["micro_refined"] = False
            pose["reachability_bridge"] = False
            pose["reachable_se3_interpolated"] = True
            pose["reachable_se3_pair_id"] = int(pair_id)
            pose["reachable_se3_id"] = int(interpolation_id)
            pose["interpolation_t"] = float(orientation_t)
            pose["reachable_se3_position_t"] = float(position_t)
            pose["reachable_se3_orientation_t"] = float(orientation_t)
            pose["interpolation_parent_angle_deg"] = float(
                pair["quat_angle_deg"]
            )
            pose["reachable_se3_parent_a"] = {
                "grasp_cm3": float(left["grasp_cm3"]),
                "normal_alignment_deg": float(left["angle_deg"]),
            }
            pose["reachable_se3_parent_b"] = {
                "grasp_cm3": float(right["grasp_cm3"]),
                "normal_alignment_deg": float(right["angle_deg"]),
            }
            inherit_strict_ik_warm_start(
                pose,
                (
                    left_pose
                    if 0.5
                    * (float(position_t) + float(orientation_t))
                    <= 0.5
                    else right_pose
                ),
            )
            key = pose_dedupe_key(pose)
            if key in source_keys or key in generated_keys:
                continue
            generated_keys.add(key)
            generated.append(pose)
            interpolation_id += 1

    return generated, {
        "strict_safe_unique_count": int(len(unique_rows)),
        "best_grasp_cm3": float(best_grasp_cm3),
        "min_grasp_ratio": float(min_grasp_ratio),
        "eligible_count": int(len(eligible)),
        "pareto_endpoint_count": int(len(frontier)),
        "endpoint_max": int(endpoint_max),
        "pair_candidate_count": int(len(pair_rows)),
        "pair_count": int(len(selected_pairs)),
        "pair_max": int(pair_max),
        "adjacent_frontier_only": bool(adjacent_frontier_only),
        "require_common_strict_arm": bool(require_common_strict_arm),
        "min_quat_angle_deg": float(min_quat_angle_deg),
        "max_quat_angle_deg": float(max_quat_angle_deg),
        "max_eef_distance_m": float(max_eef_distance_m),
        "max_anchor_distance_m": float(max_anchor_distance_m),
        "rejected_anchor_mismatch_count": int(rejected_anchor_mismatch),
        "rejected_off_axis_endpoint_count": int(
            rejected_off_axis_endpoint
        ),
        "translation_axis_local": [0.0, 0.0, 1.0],
        "translation_axis_name": "gripper_symmetry_axis_local_z",
        "lateral_offsets_enabled": False,
        "coupled_fractions": coupled_fractions,
        "position_fractions": [
            float(value) for value in position_fractions
        ],
        "orientation_fractions": [
            float(value) for value in orientation_fractions
        ],
        "grid_count_per_pair": int(len(interpolation_grid)),
        "pose_count": int(len(generated)),
        "pairs": pair_summaries,
    }


def generate_reachable_translation_refinements(
    reachable_poses: Sequence[Dict[str, Any]],
    *,
    seed_max: int = REACHABLE_TRANSLATION_SEED_MAX,
    offset_values_m: Sequence[
        float
    ] = REACHABLE_TRANSLATION_OFFSET_M,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Search grasp depth along the gripper symmetry axis."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    for pose in reachable_poses:
        if (
            float(pose.get("overlap_vol_cm3", float("inf")))
            > ORIGINAL_OVERLAP_MAX_CM3 + 1e-12
            or float(
                pose.get("inflated_overlap_vol_cm3", float("inf"))
            )
            >= INFLATED_OVERLAP_MAX_CM3
        ):
            continue
        angle_deg = float(
            pose.get("normal_alignment_deg", float("inf"))
        )
        grasp_cm3 = float(pose.get("grasp_vol_cm3", 0.0))
        if not math.isfinite(angle_deg) or grasp_cm3 <= 0.0:
            continue
        if not any(
            bool(pose.get(f"{arm}_ik_ok", False))
            and (pose.get("ik_filter") or {})
            .get(arm, {})
            .get("q_arm")
            is not None
            for arm in ("left", "right")
        ):
            continue
        key = pose_dedupe_key(pose)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "pose": pose,
                "angle_deg": angle_deg,
                "grasp_cm3": grasp_cm3,
            }
        )

    best_grasp_cm3 = max(
        (row["grasp_cm3"] for row in rows),
        default=0.0,
    )
    angle_seed_grasp_floor_cm3 = (
        MIN_GRASP_VOL_CM3
        if best_grasp_cm3 >= MIN_GRASP_VOL_CM3 - 1e-12
        else 0.90 * best_grasp_cm3
    )
    angle_seed_rows = [
        row
        for row in rows
        if row["grasp_cm3"] >= angle_seed_grasp_floor_cm3 - 1e-12
    ]
    near_best_min_angle = min(
        (row["angle_deg"] for row in angle_seed_rows),
        default=float("inf"),
    )
    quality_order = sorted(
        rows,
        key=lambda row: (
            -row["grasp_cm3"],
            row["angle_deg"],
        ),
    )
    angle_order = sorted(
        angle_seed_rows,
        key=lambda row: (
            row["angle_deg"],
            -row["grasp_cm3"],
        ),
    )
    seeds: List[Dict[str, Any]] = []
    seed_keys = set()

    def add_seed_rows(
        candidates: Iterable[Dict[str, Any]],
        count: int,
    ) -> None:
        for row in candidates:
            if len(seeds) >= int(seed_max) or count <= 0:
                return
            key = pose_dedupe_key(row["pose"])
            if key in seed_keys:
                continue
            seed_keys.add(key)
            seeds.append(row)
            count -= 1

    quality_quota = (max(0, int(seed_max)) + 1) // 2
    add_seed_rows(quality_order, quality_quota)
    add_seed_rows(angle_order, max(0, int(seed_max)) - len(seeds))
    add_seed_rows(quality_order, max(0, int(seed_max)) - len(seeds))
    generated: List[Dict[str, Any]] = []
    generated_keys = set()
    translation_id = 0
    offsets = [
        np.asarray([0.0, 0.0, float(dz)], dtype=np.float64)
        for dz in offset_values_m
        if abs(float(dz)) >= 1e-12
    ]
    for seed_index, row in enumerate(seeds):
        seed = row["pose"]
        rotation = np.asarray(seed["R"], dtype=np.float64)
        base_pos = np.asarray(seed["eef_pos"], dtype=np.float64)
        for offset_local in offsets:
            pose = dict(seed)
            eef_pos = base_pos + rotation @ offset_local
            pose["eef_pos"] = eef_pos
            pose["anchor"] = np.asarray(seed["anchor"], dtype=np.float64).copy()
            pose["anchor_local"] = rotation.T @ (
                pose["anchor"] - eef_pos
            )
            pose["axial_z_m"] = float(pose["anchor_local"][2])
            pose["translation_refined"] = True
            pose["translation_refine_generation"] = 1
            pose["translation_refine_id"] = int(translation_id)
            pose["translation_seed_index"] = int(seed_index)
            pose["translation_offset_local_m"] = (
                offset_local.astype(np.float64).tolist()
            )
            inherit_strict_ik_warm_start(pose, seed)
            key = pose_dedupe_key(pose)
            if key in generated_keys:
                continue
            generated_keys.add(key)
            generated.append(pose)
            translation_id += 1
    return generated, {
        "strict_safe_unique_count": int(len(rows)),
        "best_grasp_cm3": float(best_grasp_cm3),
        "near_best_min_angle_deg": (
            float(near_best_min_angle)
            if math.isfinite(near_best_min_angle)
            else None
        ),
        "angle_seed_grasp_floor_cm3": float(
            angle_seed_grasp_floor_cm3
        ),
        "angle_seed_pool_count": int(len(angle_seed_rows)),
        "seed_count": int(len(seeds)),
        "seed_max": int(seed_max),
        "quality_seed_quota": int(quality_quota),
        "offset_values_mm": [
            float(value) * 1000.0 for value in offset_values_m
        ],
        "translation_axis_local": [0.0, 0.0, 1.0],
        "translation_axis_name": "gripper_symmetry_axis_local_z",
        "lateral_offsets_enabled": False,
        "offset_count_per_seed": int(len(offsets)),
        "pose_count": int(len(generated)),
        "seeds": [
            {
                "grasp_cm3": float(row["grasp_cm3"]),
                "normal_alignment_deg": float(row["angle_deg"]),
            }
            for row in seeds
        ],
    }


def rank_seed_balanced_micro_ik_input(
    poses: Sequence[Dict[str, Any]],
    overlap_counts: np.ndarray,
    grasp_counts: np.ndarray,
    inflated_counts: np.ndarray,
    *,
    limit: int = IK_INPUT_MAX,
    per_seed_quality: int = 2,
    per_seed_alignment: int = 2,
    per_seed_bridge_midpoint: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Retain quality and alignment variants from every reachable parent."""
    overlap, grasp, inflated = _prefilter_metric_arrays(
        poses,
        overlap_counts,
        grasp_counts,
        inflated_counts,
    )
    voxel_cm3 = VOXEL_M ** 3 * 1e6
    original_limit_vox = int(
        math.floor(ORIGINAL_OVERLAP_MAX_CM3 / voxel_cm3 + 1e-12)
    )
    inflated_limit_vox = int(
        math.ceil(INFLATED_OVERLAP_MAX_CM3 / voxel_cm3 - 1e-12) - 1
    )
    rows: List[Dict[str, Any]] = []
    seen = set()
    for pose, original_vox, grasp_vox, inflated_vox in zip(
        poses,
        overlap,
        grasp,
        inflated,
    ):
        pose["env_overlap_vox"] = int(original_vox)
        pose["env_overlap_cm3"] = float(original_vox * voxel_cm3)
        pose["fast_overlap"] = int(original_vox)
        pose["prefilter_grasp_vox"] = int(grasp_vox)
        pose["prefilter_grasp_cm3"] = float(grasp_vox * voxel_cm3)
        pose["prefilter_inflated_overlap_vox"] = int(inflated_vox)
        pose["prefilter_inflated_overlap_cm3"] = float(
            inflated_vox * voxel_cm3
        )
        if (
            int(original_vox) > original_limit_vox
            or int(inflated_vox) > inflated_limit_vox
        ):
            continue
        key = pose_dedupe_key(pose)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "pose": pose,
                "key": key,
                "seed_index": int(pose.get("micro_seed_index", -1)),
                "angle_deg": float(
                    pose.get("normal_alignment_deg", float("inf"))
                ),
                "grasp_vox": int(grasp_vox),
                "original_vox": int(original_vox),
                "inflated_vox": int(inflated_vox),
            }
        )

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["seed_index"], []).append(row)

    selected_rows: List[Dict[str, Any]] = []
    selected_keys = set()

    def add_rows(candidates: Iterable[Dict[str, Any]], count: int) -> None:
        for row in candidates:
            if len(selected_rows) >= int(limit) or count <= 0:
                return
            if row["key"] in selected_keys:
                continue
            selected_keys.add(row["key"])
            selected_rows.append(row)
            count -= 1

    for seed_index in sorted(grouped):
        seed_rows = grouped[seed_index]
        add_rows(
            sorted(
                seed_rows,
                key=lambda row: (
                    -row["grasp_vox"],
                    row["angle_deg"],
                    row["inflated_vox"],
                    row["original_vox"],
                ),
            ),
            int(per_seed_quality),
        )
        add_rows(
            sorted(
                seed_rows,
                key=lambda row: (
                    row["angle_deg"],
                    -row["grasp_vox"],
                    row["inflated_vox"],
                    row["original_vox"],
                ),
            ),
            int(per_seed_alignment),
        )
        add_rows(
            sorted(
                (
                    row
                    for row in seed_rows
                    if bool(row["pose"].get("reachability_bridge", False))
                ),
                key=lambda row: (
                    abs(float(row["pose"].get("interpolation_t", 0.5)) - 0.5),
                    -row["grasp_vox"],
                    row["angle_deg"],
                    row["inflated_vox"],
                ),
            ),
            int(per_seed_bridge_midpoint),
        )

    quality_order = sorted(
        rows,
        key=lambda row: (
            -row["grasp_vox"],
            row["angle_deg"],
            row["inflated_vox"],
            row["original_vox"],
        ),
    )
    alignment_order = sorted(
        rows,
        key=lambda row: (
            row["angle_deg"],
            -row["grasp_vox"],
            row["inflated_vox"],
            row["original_vox"],
        ),
    )
    while len(selected_rows) < int(limit):
        before = len(selected_rows)
        add_rows(quality_order, 1)
        add_rows(alignment_order, 1)
        if len(selected_rows) == before:
            break

    selected = [row["pose"] for row in selected_rows]
    return selected, {
        "input_pose_count": int(len(poses)),
        "strict_safe_unique_count": int(len(rows)),
        "parent_seed_count": int(len(grouped)),
        "limit": int(limit),
        "per_seed_quality": int(per_seed_quality),
        "per_seed_alignment": int(per_seed_alignment),
        "per_seed_bridge_midpoint": int(per_seed_bridge_midpoint),
        "selected_count": int(len(selected)),
        "selected_parent_seed_count": int(
            len({row["seed_index"] for row in selected_rows})
        ),
        "selected_grasp_max_cm3": (
            float(max(row["grasp_vox"] for row in selected_rows) * voxel_cm3)
            if selected_rows
            else 0.0
        ),
        "selected_angle_min_deg": (
            float(min(row["angle_deg"] for row in selected_rows))
            if selected_rows
            else None
        ),
        "selected_bridge_count": int(
            sum(
                bool(row["pose"].get("reachability_bridge", False))
                for row in selected_rows
            )
        ),
    }


def rank_dedupe_top_ik_input(
    poses: List[Dict[str, Any]],
    overlap_counts: np.ndarray,
    *,
    grasp_counts: Optional[np.ndarray] = None,
    inflated_counts: Optional[np.ndarray] = None,
    limit: int = IK_INPUT_MAX,
    quality_quota: int = IK_QUALITY_QUOTA,
    alignment_quota: int = IK_ALIGNMENT_QUOTA,
    normal_confidence: str = "none",
    weak_alignment_quota_cap: int = IK_ALIGNMENT_QUOTA_WEAK,
    alignment_grasp_ratio_override: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep high-quality safe poses plus a low-collision exploration reserve."""
    voxel_cm3 = VOXEL_M ** 3 * 1e6
    overlap_array = np.asarray(overlap_counts, dtype=np.int64)
    quality_enabled = grasp_counts is not None and inflated_counts is not None
    grasp_array = (
        np.asarray(grasp_counts, dtype=np.int64)
        if grasp_counts is not None
        else np.zeros(len(poses), dtype=np.int64)
    )
    inflated_array = (
        np.asarray(inflated_counts, dtype=np.int64)
        if inflated_counts is not None
        else np.full(len(poses), np.iinfo(np.int32).max, dtype=np.int64)
    )
    if not (
        len(poses)
        == len(overlap_array)
        == len(grasp_array)
        == len(inflated_array)
    ):
        raise ValueError("pose and prefilter metric counts must have equal length")

    for pose, overlap, grasp, inflated in zip(
        poses,
        overlap_array,
        grasp_array,
        inflated_array,
    ):
        pose["env_overlap_vox"] = int(overlap)
        pose["env_overlap_cm3"] = float(overlap * voxel_cm3)
        pose["fast_overlap"] = int(overlap)
        if quality_enabled:
            pose["prefilter_grasp_vox"] = int(grasp)
            pose["prefilter_grasp_cm3"] = float(grasp * voxel_cm3)
            pose["prefilter_inflated_overlap_vox"] = int(inflated)
            pose["prefilter_inflated_overlap_cm3"] = float(
                inflated * voxel_cm3
            )

    collision_ordered = sorted(
        poses,
        key=lambda pose: (
            int(pose["env_overlap_vox"]),
            int(pose["pi"]),
            int(pose["ni"]),
            int(pose["ri"]),
            int(pose["ai"]),
        ),
    )

    unique: List[Dict[str, Any]] = []
    seen = set()
    duplicate_count = 0
    for pose in collision_ordered:
        key = pose_dedupe_key(pose)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(pose)

    strict_safe: List[Dict[str, Any]] = []
    quality_selected: List[Dict[str, Any]] = []
    alignment_target: List[Dict[str, Any]] = []
    alignment_enabled = False
    alignment_pool: List[Dict[str, Any]] = []
    alignment_grasp_floor_vox = 0
    effective_alignment_ratio = PREFILTER_ALIGNMENT_GRASP_RATIO
    if quality_enabled:
        strict_safe = [
            pose
            for pose in unique
            if float(pose["env_overlap_cm3"])
            <= ORIGINAL_OVERLAP_MAX_CM3 + 1e-12
            and float(pose["prefilter_inflated_overlap_cm3"])
            < INFLATED_OVERLAP_MAX_CM3
        ]
        strict_safe.sort(
            key=lambda pose: (
                -int(pose["prefilter_grasp_vox"]),
                int(pose["prefilter_inflated_overlap_vox"]),
                int(pose["env_overlap_vox"]),
                int(pose["pi"]),
                int(pose["ni"]),
                int(pose["ri"]),
                int(pose["ai"]),
            )
        )
        quota = min(max(0, int(quality_quota)), int(limit))
        quality_selected = strict_safe[:quota]
        confidence = str(normal_confidence or "none")
        effective_alignment_quota = (
            min(int(alignment_quota), int(weak_alignment_quota_cap))
            if confidence == "weak"
            else int(alignment_quota)
        )
        effective_alignment_ratio = float(
            alignment_grasp_ratio_override
            if alignment_grasp_ratio_override is not None
            else (
                PREFILTER_ALIGNMENT_GRASP_RATIO_WEAK
                if confidence == "weak"
                else PREFILTER_ALIGNMENT_GRASP_RATIO
            )
        )
        alignment_enabled = bool(
            confidence in ("strong", "weak")
            and any(
                math.isfinite(
                    float(pose.get("normal_alignment_deg", float("inf")))
                )
                for pose in strict_safe
            )
        )
        if alignment_enabled and strict_safe:
            best_grasp_vox = max(
                int(pose["prefilter_grasp_vox"]) for pose in strict_safe
            )
            alignment_grasp_floor_vox = int(
                math.ceil(
                    effective_alignment_ratio * float(best_grasp_vox)
                    - 1e-12
                )
            )
            alignment_pool = [
                pose
                for pose in strict_safe
                if int(pose["prefilter_grasp_vox"]) >= alignment_grasp_floor_vox
                and math.isfinite(
                    float(pose.get("normal_alignment_deg", float("inf")))
                )
            ]
            alignment_pool.sort(
                key=lambda pose: (
                    float(pose["normal_alignment_deg"]),
                    -int(pose["prefilter_grasp_vox"]),
                    int(pose["prefilter_inflated_overlap_vox"]),
                    int(pose["env_overlap_vox"]),
                    int(pose["pi"]),
                    int(pose["ni"]),
                    int(pose["ri"]),
                    int(pose["ai"]),
                )
            )
            alignment_target = alignment_pool[
                : min(max(0, effective_alignment_quota), int(limit))
            ]
    else:
        effective_alignment_quota = 0

    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    for pose in quality_selected:
        if len(selected) >= int(limit):
            break
        key = pose_dedupe_key(pose)
        selected.append(pose)
        selected_keys.add(key)
    alignment_added_count = 0
    for pose in alignment_target:
        if len(selected) >= int(limit):
            break
        key = pose_dedupe_key(pose)
        if key in selected_keys:
            continue
        selected.append(pose)
        selected_keys.add(key)
        alignment_added_count += 1
    for pose in collision_ordered:
        if len(selected) >= int(limit):
            break
        key = pose_dedupe_key(pose)
        if key in selected_keys:
            continue
        selected_keys.add(key)
        selected.append(pose)

    selected_strict_safe = [
        pose
        for pose in selected
        if quality_enabled
        and float(pose["env_overlap_cm3"])
        <= ORIGINAL_OVERLAP_MAX_CM3 + 1e-12
        and float(pose["prefilter_inflated_overlap_cm3"])
        < INFLATED_OVERLAP_MAX_CM3
    ]

    def prefilter_candidate_summary(pose: Dict[str, Any]) -> Dict[str, Any]:
        angle = float(pose.get("normal_alignment_deg", float("inf")))
        return {
            "normal_alignment_deg": angle if math.isfinite(angle) else None,
            "grasp_cm3": float(pose.get("prefilter_grasp_cm3", 0.0)),
            "overlap_cm3": float(pose.get("env_overlap_cm3", float("inf"))),
            "inflated_overlap_cm3": float(
                pose.get("prefilter_inflated_overlap_cm3", float("inf"))
            ),
            "pi": int(pose.get("pi", -1)),
            "ni": int(pose.get("ni", -1)),
            "ri": int(pose.get("ri", -1)),
            "ai": int(pose.get("ai", -1)),
            "refined": bool(pose.get("refined", False)),
            "interpolated": bool(pose.get("interpolated", False)),
            "micro_refined": bool(pose.get("micro_refined", False)),
            "reachability_bridge": bool(
                pose.get("reachability_bridge", False)
            ),
            "reachable_se3_interpolated": bool(
                pose.get("reachable_se3_interpolated", False)
            ),
        }

    def source_counts(
        source_poses: Sequence[Dict[str, Any]],
    ) -> Dict[str, int]:
        interpolated_count = sum(
            bool(pose.get("interpolated", False)) for pose in source_poses
        )
        refined_count = sum(
            bool(pose.get("refined", False)) for pose in source_poses
        )
        micro_refined_count = sum(
            bool(pose.get("micro_refined", False)) for pose in source_poses
        )
        return {
            "raw": int(
                len(source_poses)
                - interpolated_count
                - refined_count
                - micro_refined_count
            ),
            "refined": int(refined_count),
            "interpolated": int(interpolated_count),
            "micro_refined": int(micro_refined_count),
        }

    strict_safe_by_angle = sorted(
        strict_safe,
        key=lambda pose: (
            float(pose.get("normal_alignment_deg", float("inf"))),
            -int(pose.get("prefilter_grasp_vox", 0)),
            int(pose.get("prefilter_inflated_overlap_vox", 0)),
        ),
    )
    return selected, {
        "input_pose_count": int(len(poses)),
        "unique_pose_count": int(len(unique)),
        "unique_selected_count": int(len(selected)),
        "duplicate_count": int(duplicate_count),
        "duplicate_count_before_limit": int(duplicate_count),
        "quality_enabled": bool(quality_enabled),
        "quality_quota": int(min(max(0, int(quality_quota)), int(limit))),
        "quality_selected_count": int(len(quality_selected)),
        "normal_confidence": str(normal_confidence),
        "alignment_enabled": bool(alignment_enabled),
        "alignment_quota": int(
            min(max(0, effective_alignment_quota), int(limit))
        ),
        "weak_alignment_quota_cap": int(weak_alignment_quota_cap),
        "alignment_target_count": int(len(alignment_target)),
        "alignment_added_count": int(alignment_added_count),
        "prefilter_alignment_grasp_ratio": float(effective_alignment_ratio),
        "alignment_grasp_floor_cm3": float(
            alignment_grasp_floor_vox * voxel_cm3
        ),
        "alignment_pool_count": int(len(alignment_pool)),
        "strict_safe_candidates_by_angle": [
            prefilter_candidate_summary(pose)
            for pose in strict_safe_by_angle[:20]
        ],
        "input_source_counts": source_counts(poses),
        "strict_safe_source_counts": source_counts(strict_safe),
        "quality_selected_source_counts": source_counts(quality_selected),
        "alignment_target_source_counts": source_counts(alignment_target),
        "selected_source_counts": source_counts(selected),
        "strict_safe_count": int(len(strict_safe)),
        "selected_strict_safe_count": int(len(selected_strict_safe)),
        "strict_safe_grasp_max_cm3": (
            float(strict_safe[0]["prefilter_grasp_cm3"])
            if strict_safe
            else 0.0
        ),
        "selected_strict_safe_grasp_max_cm3": (
            max(
                float(pose["prefilter_grasp_cm3"])
                for pose in selected_strict_safe
            )
            if selected_strict_safe
            else 0.0
        ),
        "limit": int(limit),
        "overlap_min_cm3": (
            float(collision_ordered[0]["env_overlap_cm3"])
            if collision_ordered
            else None
        ),
        "overlap_top_limit_max_cm3": (
            float(selected[-1]["env_overlap_cm3"]) if selected else None
        ),
    }


def arm_joint_limits_rad(
    world,
    arm: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read submission-local arm limits from the frozen custom URDF."""
    try:
        lower, upper = _local_arm_joint_limits(world, arm)
        lower = np.asarray(lower, dtype=np.float64).reshape(world.arm_dof)
        upper = np.asarray(upper, dtype=np.float64).reshape(world.arm_dof)
    except Exception:
        return None
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return None
    return lower, upper


def wrist_limit_margin_rad(
    q_arm: Optional[Sequence[float]],
    limits: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Optional[float]:
    """返回 J5/J6/J7 中距上下限最近的裕度（弧度）；解越限时为负。"""
    if q_arm is None or limits is None:
        return None
    try:
        q = np.asarray(q_arm, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    lower, upper = limits
    index = np.asarray(WRIST_LIMIT_JOINT_INDICES, dtype=int)
    margins = np.minimum(q[index] - lower[index], upper[index] - q[index])
    if not np.all(np.isfinite(margins)):
        return None
    return float(margins.min())


def wrist_limit_penalty_tier(margin_rad: Optional[float]) -> int:
    """裕度分档：0=充裕，1=偏近限位，2=紧贴或越过限位。裕度未知按 0 处理，不惩罚。"""
    if margin_rad is None or not math.isfinite(float(margin_rad)):
        return 0
    if float(margin_rad) < WRIST_LIMIT_CRITICAL_RAD:
        return 2
    if float(margin_rad) < WRIST_LIMIT_WARN_RAD:
        return 1
    return 0


def _paired_safe_target_for_pose(
    pose: Dict[str, Any],
    *,
    back_m: float = SAFE_PLAN_BACK_M,
) -> Dict[str, Any]:
    final_pos = np.asarray(pose["eef_pos"], dtype=np.float64).reshape(3)
    quat = np.asarray(pose["quat"], dtype=np.float64).reshape(4)
    rotation = np.asarray(
        pose.get("R", _quat_to_mat_xyzw(quat)),
        dtype=np.float64,
    ).reshape(3, 3)
    pointing = rotation[:, 2]
    pointing /= max(float(np.linalg.norm(pointing)), 1e-12)
    return {
        "eef_pos": (final_pos - float(back_m) * pointing).tolist(),
        "quat": quat.tolist(),
        "pointing": pointing.tolist(),
        "back_m": float(back_m),
        "pos_tol_m": float(SAFE_IK_POS_TOL_M),
        "ori_tol_deg": float(SAFE_IK_ORI_TOL_DEG),
        "final_branch_gap_rad": float(SAFE_FINAL_BRANCH_GAP_RAD),
    }


def _local_fk_validate_paired_safe_ik(
    world: LocalRobotState,
    ranked: List[Dict[str, Any]],
    *,
    active_arms: Tuple[str, ...] = ("left", "right"),
    ctx=None,
) -> Dict[str, Any]:
    """Validate final and safe endpoints without simulator reads or writes."""
    requested_arms = set(active_arms or ("left", "right"))
    active_arms = tuple(
        arm for arm in ("left", "right") if arm in requested_arms
    ) or ("left", "right")
    pair_counts = {"left": 0, "right": 0, "both": 0}
    failure_counts: Dict[str, int] = {}
    wrist_tier_counts: Dict[str, int] = {}
    elapsed_t0 = time.perf_counter()
    wrist_limits = {
        arm: arm_joint_limits_rad(world, arm)
        for arm in active_arms
    }
    for pose in ranked:
        final_pos = np.asarray(pose["eef_pos"], dtype=np.float64).reshape(3)
        final_quat = np.asarray(pose["quat"], dtype=np.float64).reshape(4)
        target = pose.get("ik_paired_safe_pose") or {}
        safe_pos = np.asarray(target.get("eef_pos"), dtype=np.float64).reshape(3)
        safe_quat = np.asarray(
            target.get("quat", pose["quat"]),
            dtype=np.float64,
        ).reshape(4)
        planned_by_arm: Dict[str, Dict[str, Any]] = {}
        arm_pair_ok: Dict[str, bool] = {}
        arm_wrist_margin: Dict[str, Optional[float]] = {}
        arm_wrist_tier: Dict[str, int] = {}
        active_allerr = 0.0
        for arm in active_arms:
            final_info = (pose.get("ik_filter") or {}).get(arm) or {}
            safe_info = final_info.get("paired_safe")
            safe_info = safe_info if isinstance(safe_info, dict) else {}
            q_safe_raw = safe_info.get("q_arm")
            q_final_raw = final_info.get("q_arm")
            final_ok = False
            final_pos_err = float("inf")
            final_ori_err = float("inf")
            final_approach_err = float("inf")
            if q_final_raw is not None:
                q_final = np.asarray(
                    q_final_raw,
                    dtype=np.float64,
                ).reshape(world.arm_dof).copy()
                if world.arm_dof == 8:
                    q_final[7] = 0.0
                    final_info["q_arm"] = q_final.astype(float).tolist()
                actual_pos, actual_quat = _local_eef_pose(
                    world,
                    arm,
                    q_final,
                )
                (
                    final_pos_err,
                    final_ori_err,
                    final_approach_err,
                ) = _local_pose_error(
                    actual_pos,
                    actual_quat,
                    final_pos,
                    final_quat,
                )
                final_ok = bool(
                    final_pos_err <= IK_FILTER_POS_TOL_M
                    and final_ori_err <= IK_FILTER_ORI_TOL_DEG
                )
            final_margin_rad = wrist_limit_margin_rad(
                q_final_raw,
                wrist_limits.get(arm),
            )
            final_wrist_tier = wrist_limit_penalty_tier(final_margin_rad)
            arm_wrist_margin[arm] = final_margin_rad
            arm_wrist_tier[arm] = int(final_wrist_tier)
            final_info.update(
                {
                    "wrist_limit_margin_rad": final_margin_rad,
                    "wrist_limit_tier": int(final_wrist_tier),
                    "local_fk_pos_err_m": float(final_pos_err),
                    "local_fk_ori_err_deg": float(final_ori_err),
                    "local_fk_approach_err_deg": float(final_approach_err),
                    "pos_err_m": float(final_pos_err),
                    "ori_err_deg": float(final_ori_err),
                    "approach_err_deg": float(final_approach_err),
                    "final_endpoint_ok": bool(final_ok),
                    "validated_by_local_fk": True,
                    "validated_by_og_fk": False,
                }
            )
            active_allerr += final_pos_err * 1000.0 + final_ori_err
            reason = str(
                safe_info.get("error")
                or (
                    "final_endpoint_ik_failed"
                    if not final_ok
                    else "safe_endpoint_ik_failed"
                )
            )
            safe_endpoint_ok = False
            branch_ok = False
            branch_gap = float("inf")
            if final_ok and q_safe_raw is not None and q_final_raw is not None:
                q_safe = np.asarray(
                    q_safe_raw,
                    dtype=np.float64,
                ).reshape(world.arm_dof).copy()
                q_final = np.asarray(
                    q_final_raw,
                    dtype=np.float64,
                ).reshape(world.arm_dof).copy()
                if world.arm_dof == 8:
                    q_safe[7] = 0.0
                    q_final[7] = 0.0
                    safe_info["q_arm"] = q_safe.astype(float).tolist()
                    final_info["q_arm"] = q_final.astype(float).tolist()
                actual_safe_pos, actual_safe_quat = _local_eef_pose(
                    world,
                    arm,
                    q_safe,
                )
                pos_err, ori_err, approach_err = _local_pose_error(
                    actual_safe_pos,
                    actual_safe_quat,
                    safe_pos,
                    safe_quat,
                )
                branch_gap = float(
                    np.linalg.norm(q_safe - q_final, ord=np.inf)
                )
                safe_endpoint_ok = bool(
                    pos_err <= SAFE_IK_POS_TOL_M
                    and ori_err <= SAFE_IK_ORI_TOL_DEG
                )
                branch_ok = bool(branch_gap <= SAFE_FINAL_BRANCH_GAP_RAD)
                safe_info.update(
                    {
                        "local_fk_pos_err_m": float(pos_err),
                        "local_fk_ori_err_deg": float(ori_err),
                        "local_fk_approach_err_deg": float(approach_err),
                        "pos_err_m": float(pos_err),
                        "ori_err_deg": float(ori_err),
                        "approach_err_deg": float(approach_err),
                        "endpoint_ok": safe_endpoint_ok,
                        "safe_to_final_joint_gap_rad": branch_gap,
                        "same_final_branch": branch_ok,
                        "validated_by_local_fk": True,
                        "validated_by_og_fk": False,
                    }
                )
                if not safe_endpoint_ok:
                    reason = "safe_endpoint_local_fk_failed"
                elif not branch_ok:
                    reason = "safe_final_branch_gap"
                else:
                    reason = "paired_safe_final_ik_filter"
            pair_ok = bool(final_ok and safe_endpoint_ok and branch_ok)
            safe_info["ok"] = pair_ok
            if not pair_ok:
                safe_info["error"] = reason
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            final_info["paired_safe"] = safe_info
            final_info["ok"] = pair_ok
            arm_pair_ok[arm] = pair_ok
            if pair_ok:
                pair_counts[arm] += 1
                planned_by_arm[arm] = {
                    "ok": True,
                    "source": "paired_safe_final_ik_filter",
                    "reason": "paired_safe_final_ik_filter",
                    "arm": arm,
                    "requested_back_m": float(SAFE_PLAN_BACK_M),
                    "active_back_m": float(SAFE_PLAN_BACK_M),
                    "safe_pos": safe_pos.tolist(),
                    "safe_quat": safe_quat.tolist(),
                    "safe_q": _locked_j8_list(q_safe),
                    "safe_endpoint_pos_err_m": float(
                        safe_info["pos_err_m"]
                    ),
                    "safe_endpoint_ori_err_deg": float(
                        safe_info["ori_err_deg"]
                    ),
                    "safe_to_final_joint_gap_rad": float(branch_gap),
                    "final_branch_gap_limit_rad": float(
                        SAFE_FINAL_BRANCH_GAP_RAD
                    ),
                    "wrist_limit_margin_rad": final_margin_rad,
                    "wrist_limit_tier": int(final_wrist_tier),
                    "to_safe_anchor_count": 0,
                    "validated_by": "submission_local_fk",
                }
        pose["left_ik_ok"] = bool(arm_pair_ok.get("left"))
        pose["right_ik_ok"] = bool(arm_pair_ok.get("right"))
        pose["dual_ik_ok"] = bool(
            arm_pair_ok.get("left") and arm_pair_ok.get("right")
        )
        pose["ik_allerr"] = float(active_allerr)
        pose["ik_validated_by_local_fk"] = True
        pose["safe_ik_validated_by_local_fk"] = True
        pose["ik_validated_by_og_fk"] = False
        pose["safe_ik_validated_by_og_fk"] = False
        pose["planned_safe_by_arm"] = planned_by_arm
        feasible_arms = [
            arm for arm in active_arms if arm_pair_ok.get(arm)
        ]
        feasible_margins = [
            arm_wrist_margin[arm]
            for arm in feasible_arms
            if arm_wrist_margin.get(arm) is not None
        ]
        pose["wrist_limit_margin_rad"] = (
            float(max(feasible_margins)) if feasible_margins else None
        )
        pose["wrist_limit_tier"] = (
            int(min(arm_wrist_tier[arm] for arm in feasible_arms))
            if feasible_arms
            else 0
        )
        if feasible_arms:
            tier_key = str(pose["wrist_limit_tier"])
            wrist_tier_counts[tier_key] = (
                wrist_tier_counts.get(tier_key, 0) + 1
            )
        if pose["dual_ik_ok"]:
            pair_counts["both"] += 1
    meta = {
        "n_input": int(len(ranked)),
        "n_left_pair_ok": int(pair_counts["left"]),
        "n_right_pair_ok": int(pair_counts["right"]),
        "n_both_pair_ok": int(pair_counts["both"]),
        "n_local_fk_validated": int(len(ranked)),
        "n_og_fk_validated": 0,
        "active_arms": list(active_arms),
        "failure_counts": failure_counts,
        "safe_back_m": float(SAFE_PLAN_BACK_M),
        "safe_pos_tol_m": float(SAFE_IK_POS_TOL_M),
        "safe_ori_tol_deg": float(SAFE_IK_ORI_TOL_DEG),
        "safe_final_branch_gap_rad": float(SAFE_FINAL_BRANCH_GAP_RAD),
        "wrist_limit_tier_counts": dict(wrist_tier_counts),
        "wrist_limit_critical_deg": float(
            math.degrees(WRIST_LIMIT_CRITICAL_RAD)
        ),
        "wrist_limit_warn_deg": float(math.degrees(WRIST_LIMIT_WARN_RAD)),
        "elapsed_s": float(time.perf_counter() - elapsed_t0),
        "joint_write_s": 0.0,
        "eef_error_s": 0.0,
        "restore_s": 0.0,
        "validation": "submission_local_fk",
    }
    if ctx is not None:
        ctx.log(
            "  [grasp_point_filter_rgbd] paired safe+final local FK "
            f"input={len(ranked)} left={pair_counts['left']} "
            f"right={pair_counts['right']} both={pair_counts['both']} "
            f"failures={failure_counts}"
        )
    return meta


def filter_poses_safe_final_dual_arm_ik(
    world,
    poses: List[Dict[str, Any]],
    *,
    plan_arm: str = "any",
    ik_worker_policy: Optional[str] = None,
    ctx=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Solve and hard-filter final and 10 cm safe IK as one candidate pair."""
    requested_arm = str(plan_arm or "any").strip().lower()
    if requested_arm not in ("left", "right", "any"):
        requested_arm = "any"
    active_arms = (
        ("left", "right")
        if requested_arm == "any"
        else (requested_arm,)
    )
    eligible: List[Dict[str, Any]] = []
    rejected_original = 0
    rejected_inflated = 0
    missing_metrics = 0
    for pose in poses:
        original = pose.get("env_overlap_cm3")
        inflated = pose.get("prefilter_inflated_overlap_cm3")
        if original is None or inflated is None:
            missing_metrics += 1
            eligible.append(pose)
            continue
        if float(original) > ORIGINAL_OVERLAP_MAX_CM3 + 1e-12:
            rejected_original += 1
            continue
        if float(inflated) >= INFLATED_OVERLAP_MAX_CM3:
            rejected_inflated += 1
            continue
        eligible.append(pose)
    overlap_gate_meta = {
        "n_input": int(len(poses)),
        "n_eligible": int(len(eligible)),
        "n_rejected": int(len(poses) - len(eligible)),
        "n_rejected_original": int(rejected_original),
        "n_rejected_inflated": int(rejected_inflated),
        "n_missing_metrics_kept": int(missing_metrics),
        "original_threshold_cm3": float(ORIGINAL_OVERLAP_MAX_CM3),
        "inflated_threshold_cm3": float(INFLATED_OVERLAP_MAX_CM3),
        "backfill": False,
    }
    if ctx is not None:
        ctx.log(
            "  [grasp_point_filter_rgbd] pre-IK hard overlap gate "
            f"input={len(poses)} eligible={len(eligible)} "
            f"reject_original={rejected_original} "
            f"reject_inflated={rejected_inflated} "
            f"missing_kept={missing_metrics} backfill=0"
        )
    if not eligible:
        return [], {
            "n_input": int(len(poses)),
            "n_solver_input": 0,
            "n_left_ok": 0,
            "n_right_ok": 0,
            "n_both_ok": 0,
            "n_ranked": 0,
            "active_arms": list(active_arms),
            "pre_ik_overlap_gate": overlap_gate_meta,
            "paired_safe_final": {
                "n_input": 0,
                "n_left_pair_ok": 0,
                "n_right_pair_ok": 0,
                "n_both_pair_ok": 0,
                "failure_counts": {},
            },
        }

    paired_input: List[Dict[str, Any]] = []
    for pose in eligible:
        item = dict(pose)
        item["ik_paired_safe_pose"] = _paired_safe_target_for_pose(item)
        if ik_worker_policy:
            item["_ik_worker_policy"] = str(ik_worker_policy)
        paired_input.append(item)
    ranked, meta = _local_filter_poses_dual_arm_ik(
        world,
        paired_input,
        keep_single_arm_results=True,
        active_arms=active_arms,
        ctx=ctx,
    )
    pair_meta = _local_fk_validate_paired_safe_ik(
        world,
        ranked,
        active_arms=active_arms,
        ctx=ctx,
    )
    ranked = [
        pose
        for pose in ranked
        if pose.get("left_ik_ok") or pose.get("right_ik_ok")
    ]
    ranked.sort(
        key=lambda pose: (
            float(pose.get("ik_allerr", float("inf"))),
            int(pose.get("fast_overlap", 0)),
        )
    )
    meta["paired_safe_final"] = pair_meta
    meta["n_left_ok"] = int(pair_meta["n_left_pair_ok"])
    meta["n_right_ok"] = int(pair_meta["n_right_pair_ok"])
    meta["n_both_ok"] = int(pair_meta["n_both_pair_ok"])
    meta["n_ranked"] = int(len(ranked))
    meta["n_local_fk_validated"] = int(
        pair_meta["n_local_fk_validated"]
    )
    meta["n_og_fk_validated"] = 0
    meta["active_arms"] = list(active_arms)
    meta["n_input_before_overlap_gate"] = int(len(poses))
    meta["n_solver_input"] = int(len(eligible))
    meta["pre_ik_overlap_gate"] = overlap_gate_meta
    return ranked, meta


def select_strict_ik_pool(
    ranked: List[Dict[str, Any]],
    *,
    plan_arm: str,
    top_n: int = IK_TOP_N,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply the established strict left/right/any pool policy."""
    requested = str(plan_arm or "any").strip().lower()
    if requested not in ("left", "right", "any"):
        requested = "any"
    both = [pose for pose in ranked if pose.get("dual_ik_ok")]
    left = [pose for pose in ranked if pose.get("left_ik_ok")]
    right = [pose for pose in ranked if pose.get("right_ik_ok")]
    if requested == "left":
        pool = left
        selected_arm = "left"
        selection = "forced_left_strict"
    elif requested == "right":
        pool = right
        selected_arm = "right"
        selection = "forced_right_strict"
    else:
        pool = [
            pose
            for pose in ranked
            if pose.get("left_ik_ok") or pose.get("right_ik_ok")
        ]
        selected_arm = None
        selection = "either_arm_strict"

    def arm_error(pose: Dict[str, Any], arm: str) -> float:
        info = (pose.get("ik_filter") or {}).get(arm) or {}
        return (
            float(info.get("pos_err_m", float("inf"))) * 1000.0
            + float(info.get("ori_err_deg", float("inf")))
        )

    if selected_arm in ("left", "right"):
        pool = sorted(pool, key=lambda pose: arm_error(pose, selected_arm))
    else:
        pool = sorted(
            pool,
            key=lambda pose: min(
                arm_error(pose, arm)
                for arm in ("left", "right")
                if pose.get(f"{arm}_ik_ok")
            ),
        )
    selected = pool[: min(int(top_n), len(pool))]
    return selected, {
        "plan_arm": requested,
        "selection": selection,
        "recommended_arm": selected_arm,
        "n_both_ok": int(len(both)),
        "n_left_ok": int(len(left)),
        "n_right_ok": int(len(right)),
        "n_selection_pool": int(len(pool)),
        "n_top_ik": int(len(selected)),
    }


def attach_final_volume_metrics(
    poses: List[Dict[str, Any]],
    *,
    occupancy: DenseSceneOccupancy,
    gripper_voxels: np.ndarray,
    opening_voxels: np.ndarray,
    inflated_voxels: np.ndarray,
) -> Dict[str, Any]:
    """Compute original overlap, grasp volume, and inflated overlap."""
    reuse_prefilter = bool(
        poses
        and all(
            pose.get("env_overlap_vox") is not None
            and pose.get("prefilter_grasp_vox") is not None
            and pose.get("prefilter_inflated_overlap_vox") is not None
            for pose in poses
        )
    )
    if reuse_prefilter:
        overlap_counts = np.asarray(
            [int(pose["env_overlap_vox"]) for pose in poses],
            dtype=np.int32,
        )
        grasp_counts = np.asarray(
            [int(pose["prefilter_grasp_vox"]) for pose in poses],
            dtype=np.int32,
        )
        inflated_counts = np.asarray(
            [
                int(pose["prefilter_inflated_overlap_vox"])
                for pose in poses
            ],
            dtype=np.int32,
        )
        query_meta = {
            "device": "prefilter_cache",
            "elapsed_s": 0.0,
            "pose_count": int(len(poses)),
            "cache_hit_count": int(len(poses)),
            "unique_query_pose_count": 0,
        }
        overlap_meta = dict(query_meta)
        grasp_meta = dict(query_meta)
        inflated_meta = dict(query_meta)
    else:
        overlap_counts, overlap_meta = occupancy.counts_for_poses(
            poses,
            gripper_voxels,
        )
        grasp_counts, grasp_meta = occupancy.counts_for_poses(
            poses,
            opening_voxels,
        )
        inflated_counts, inflated_meta = occupancy.counts_for_poses(
            poses,
            inflated_voxels,
        )
    voxel_cm3 = occupancy.voxel_m ** 3 * 1e6
    gripper_vol_cm3 = float(len(gripper_voxels) * voxel_cm3)
    for pose, overlap, grasp, inflated in zip(
        poses,
        overlap_counts,
        grasp_counts,
        inflated_counts,
    ):
        pose["overlap_vox"] = int(overlap)
        pose["overlap_vol_cm3"] = float(overlap * voxel_cm3)
        pose["grasp_vox"] = int(grasp)
        pose["grasp_vol_cm3"] = float(grasp * voxel_cm3)
        pose["inflated_overlap_vox"] = int(inflated)
        pose["inflated_overlap_vol_cm3"] = float(inflated * voxel_cm3)
        pose["gripper_vol_cm3"] = gripper_vol_cm3
        pose["overlap_frac"] = float(overlap / max(len(gripper_voxels), 1))
        pose["open_intersect_vox"] = int(grasp)
        pose["open_intersect_vol_cm3"] = float(grasp * voxel_cm3)
        pose["opening_voxel_total"] = int(len(opening_voxels))
        pose["open_vol_method"] = "rgbd_v8_dense_occupancy_gpu"
    inflated_values = np.asarray(
        [pose["inflated_overlap_vol_cm3"] for pose in poses],
        dtype=np.float64,
    )
    return {
        "original_overlap_query": overlap_meta,
        "grasp_query": grasp_meta,
        "inflated_overlap_query": inflated_meta,
        "gripper_voxels": int(len(gripper_voxels)),
        "opening_voxels": int(len(opening_voxels)),
        "inflated_gripper_voxels": int(len(inflated_voxels)),
        "reused_prefilter_counts": bool(reuse_prefilter),
        "inflated_overlap_min_cm3": (
            float(inflated_values.min()) if len(inflated_values) else None
        ),
        "inflated_overlap_median_cm3": (
            float(np.median(inflated_values)) if len(inflated_values) else None
        ),
        "inflated_overlap_max_cm3": (
            float(inflated_values.max()) if len(inflated_values) else None
        ),
    }


def select_final_pose(
    poses: List[Dict[str, Any]],
    *,
    original_threshold_cm3: float = ORIGINAL_OVERLAP_MAX_CM3,
    inflated_threshold_cm3: float = INFLATED_OVERLAP_MAX_CM3,
    grasp_threshold_cm3: float = MIN_GRASP_VOL_CM3,
    normal_confidence: str = "none",
    ranked_out: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Filter overlap and grasp quality, then maximize grasp volume."""
    original_passing = [
        pose
        for pose in poses
        if float(pose.get("overlap_vol_cm3", float("inf")))
        <= float(original_threshold_cm3) + 1e-12
    ]
    overlap_passing = [
        pose
        for pose in original_passing
        if float(pose.get("inflated_overlap_vol_cm3", float("inf")))
        < float(inflated_threshold_cm3)
    ]
    passing = [
        pose
        for pose in overlap_passing
        if float(pose.get("grasp_vol_cm3", float("-inf")))
        >= float(grasp_threshold_cm3) - 1e-12
    ]
    # 腕部限位档位前置：只在最好的档位内继续比抓取质量和法向对齐，
    # 避免后续 alignment 把候选池收窄后，限位惩罚失去作用。
    wrist_best_tier = min(
        (int(pose.get("wrist_limit_tier", 0)) for pose in passing),
        default=0,
    )
    wrist_passing = [
        pose
        for pose in passing
        if int(pose.get("wrist_limit_tier", 0)) <= int(wrist_best_tier)
    ]
    confidence = str(normal_confidence or "none")
    alignment_enabled = bool(
        confidence in ("strong", "weak")
        and any(
            math.isfinite(
                float(pose.get("normal_alignment_deg", float("inf")))
            )
            for pose in wrist_passing
        )
    )
    alignment_ratio = (
        FINAL_ALIGNMENT_GRASP_RATIO_STRONG
        if confidence == "strong"
        else FINAL_ALIGNMENT_GRASP_RATIO_WEAK
    )
    passing_grasp_max = max(
        (float(pose.get("grasp_vol_cm3", 0.0)) for pose in wrist_passing),
        default=0.0,
    )
    weak_absolute_target_active = bool(
        alignment_enabled
        and confidence == "weak"
        and float(grasp_threshold_cm3) >= MIN_GRASP_VOL_CM3 - 1e-12
    )
    alignment_quality_floor = (
        float(grasp_threshold_cm3)
        if weak_absolute_target_active
        else max(
            float(grasp_threshold_cm3),
            float(alignment_ratio) * passing_grasp_max,
        )
    )
    alignment_quality_pool = (
        [
            pose
            for pose in wrist_passing
            if float(pose.get("grasp_vol_cm3", 0.0))
            >= alignment_quality_floor - 1e-12
        ]
        if alignment_enabled
        else list(wrist_passing)
    )
    alignment_angle_slack_deg = (
        FINAL_ALIGNMENT_ANGLE_SLACK_STRONG_DEG
        if confidence == "strong"
        else FINAL_ALIGNMENT_ANGLE_SLACK_WEAK_DEG
    )
    alignment_angle_min_deg = (
        min(
            float(
                pose.get("normal_alignment_deg", float("inf"))
            )
            for pose in alignment_quality_pool
        )
        if alignment_enabled and alignment_quality_pool
        else None
    )
    selection_pool = (
        [
            pose
            for pose in alignment_quality_pool
            if float(
                pose.get("normal_alignment_deg", float("inf"))
            )
            <= float(alignment_angle_min_deg)
            + float(alignment_angle_slack_deg)
            + 1e-12
        ]
        if alignment_enabled and alignment_angle_min_deg is not None
        else list(alignment_quality_pool)
    )
    # 腕部限位档位前置于抓取质量：只有明显逼近 J5/J6/J7 限位时才会翻盘，
    # 同档内仍按原有的抓取体积/对齐/碰撞/IK 误差次序比较。
    if alignment_enabled:
        def selection_sort_key(pose: Dict[str, Any]) -> Tuple[float, ...]:
            return (
                float(pose.get("wrist_limit_tier", 0)),
                -float(pose.get("grasp_vol_cm3", 0.0)),
                float(pose.get("normal_alignment_deg", float("inf"))),
                float(pose.get("inflated_overlap_vol_cm3", float("inf"))),
                float(pose.get("overlap_vol_cm3", float("inf"))),
                float(pose.get("ik_allerr", float("inf"))),
            )
    else:
        def selection_sort_key(pose: Dict[str, Any]) -> Tuple[float, ...]:
            return (
                float(pose.get("wrist_limit_tier", 0)),
                -float(pose.get("grasp_vol_cm3", 0.0)),
                float(pose.get("inflated_overlap_vol_cm3", float("inf"))),
                float(pose.get("overlap_vol_cm3", float("inf"))),
                float(pose.get("ik_allerr", float("inf"))),
            )
    selection_pool.sort(key=selection_sort_key)
    safe_ranked_pool = list(selection_pool)
    safe_ranked_ids = {id(pose) for pose in safe_ranked_pool}
    for fallback_pool in (alignment_quality_pool, passing):
        for pose in sorted(fallback_pool, key=selection_sort_key):
            if id(pose) in safe_ranked_ids:
                continue
            safe_ranked_ids.add(id(pose))
            safe_ranked_pool.append(pose)
    if ranked_out is not None:
        ranked_out.extend(safe_ranked_pool)
    original_values = np.asarray(
        [float(pose.get("overlap_vol_cm3", float("inf"))) for pose in poses],
        dtype=np.float64,
    )
    inflated_values = np.asarray(
        [float(pose.get("inflated_overlap_vol_cm3", float("inf"))) for pose in poses],
        dtype=np.float64,
    )
    overlap_passing_grasp = np.asarray(
        [float(pose.get("grasp_vol_cm3", 0.0)) for pose in overlap_passing],
        dtype=np.float64,
    )

    def candidate_summary(pose: Dict[str, Any]) -> Dict[str, Any]:
        angle = float(pose.get("normal_alignment_deg", float("inf")))
        margin_rad = pose.get("wrist_limit_margin_rad")
        return {
            "normal_alignment_deg": angle if math.isfinite(angle) else None,
            "wrist_limit_tier": int(pose.get("wrist_limit_tier", 0)),
            "wrist_limit_margin_deg": (
                round(math.degrees(float(margin_rad)), 3)
                if margin_rad is not None
                else None
            ),
            "grasp_vol_cm3": float(pose.get("grasp_vol_cm3", 0.0)),
            "overlap_vol_cm3": float(
                pose.get("overlap_vol_cm3", float("inf"))
            ),
            "inflated_overlap_vol_cm3": float(
                pose.get("inflated_overlap_vol_cm3", float("inf"))
            ),
            "ik_allerr": float(pose.get("ik_allerr", float("inf"))),
            "pi": int(pose.get("pi", -1)),
            "ni": int(pose.get("ni", -1)),
            "ri": int(pose.get("ri", -1)),
            "ai": int(pose.get("ai", -1)),
            "axial_name": str(pose.get("axial_name", "")),
            "refined": bool(pose.get("refined", False)),
            "interpolated": bool(pose.get("interpolated", False)),
            "micro_refined": bool(pose.get("micro_refined", False)),
            "reachability_bridge": bool(
                pose.get("reachability_bridge", False)
            ),
            "reachable_se3_interpolated": bool(
                pose.get("reachable_se3_interpolated", False)
            ),
            "reachable_se3_generation": int(
                pose.get("reachable_se3_generation", 0)
            ),
            "translation_refined": bool(
                pose.get("translation_refined", False)
            ),
            "translation_refine_generation": int(
                pose.get("translation_refine_generation", 0)
            ),
        }

    by_angle = sorted(
        passing,
        key=lambda pose: (
            float(pose.get("normal_alignment_deg", float("inf"))),
            -float(pose.get("grasp_vol_cm3", 0.0)),
            float(pose.get("inflated_overlap_vol_cm3", float("inf"))),
        ),
    )
    by_grasp = sorted(
        passing,
        key=lambda pose: (
            -float(pose.get("grasp_vol_cm3", 0.0)),
            float(pose.get("normal_alignment_deg", float("inf"))),
            float(pose.get("inflated_overlap_vol_cm3", float("inf"))),
        ),
    )
    overlap_passing_by_angle = sorted(
        overlap_passing,
        key=lambda pose: (
            float(pose.get("normal_alignment_deg", float("inf"))),
            -float(pose.get("grasp_vol_cm3", 0.0)),
            float(pose.get("inflated_overlap_vol_cm3", float("inf"))),
        ),
    )
    joint_frontier: List[Dict[str, Any]] = []
    frontier_grasp = float("-inf")
    for pose in by_angle:
        grasp_cm3 = float(pose.get("grasp_vol_cm3", 0.0))
        if grasp_cm3 > frontier_grasp + 1e-12:
            joint_frontier.append(pose)
            frontier_grasp = grasp_cm3
    audit = {
        # Preserve the old inflated-only field names for existing consumers.
        "threshold_cm3": float(inflated_threshold_cm3),
        "comparison": "<",
        "input_count": int(len(poses)),
        "original_threshold_cm3": float(original_threshold_cm3),
        "original_comparison": "<=",
        "original_passing_count": int(len(original_passing)),
        "overlap_passing_count": int(len(overlap_passing)),
        "grasp_threshold_cm3": float(grasp_threshold_cm3),
        "grasp_comparison": ">=",
        "overlap_passing_grasp_max_cm3": (
            float(overlap_passing_grasp.max())
            if len(overlap_passing_grasp)
            else 0.0
        ),
        "passing_count": int(len(passing)),
        "passing_fraction": float(len(passing) / max(len(poses), 1)),
        "normal_confidence": confidence,
        "normal_alignment_enabled": bool(alignment_enabled),
        "selection_objective": (
            "max_grasp_within_near_best_band_and_normal_angle_slack"
            if alignment_enabled
            else "max_grasp_volume"
        ),
        "wrist_limit_prefilter": "min_wrist_limit_tier_before_selection_objective",
        "wrist_limit_critical_deg": float(
            math.degrees(WRIST_LIMIT_CRITICAL_RAD)
        ),
        "wrist_limit_warn_deg": float(math.degrees(WRIST_LIMIT_WARN_RAD)),
        "wrist_limit_best_tier": int(wrist_best_tier),
        "wrist_limit_passing_count": int(len(wrist_passing)),
        "wrist_limit_tier_counts": {
            str(tier): int(
                sum(
                    1
                    for pose in passing
                    if int(pose.get("wrist_limit_tier", 0)) == tier
                )
            )
            for tier in (0, 1, 2)
        },
        "selected_wrist_limit_tier": (
            int(selection_pool[0].get("wrist_limit_tier", 0))
            if selection_pool
            else None
        ),
        "selected_wrist_limit_margin_deg": (
            round(
                math.degrees(
                    float(selection_pool[0]["wrist_limit_margin_rad"])
                ),
                3,
            )
            if selection_pool
            and selection_pool[0].get("wrist_limit_margin_rad") is not None
            else None
        ),
        "alignment_grasp_ratio": (
            float(alignment_ratio) if alignment_enabled else None
        ),
        "weak_absolute_grasp_target_active": bool(
            weak_absolute_target_active
        ),
        "alignment_quality_floor_cm3": (
            float(alignment_quality_floor) if alignment_enabled else None
        ),
        "alignment_quality_pool_count": int(
            len(alignment_quality_pool)
        ),
        "alignment_angle_min_deg": (
            float(alignment_angle_min_deg)
            if alignment_angle_min_deg is not None
            else None
        ),
        "alignment_angle_slack_deg": (
            float(alignment_angle_slack_deg)
            if alignment_enabled
            else None
        ),
        "alignment_pool_count": int(len(selection_pool)),
        "safe_fallback_candidate_count": int(len(safe_ranked_pool)),
        "selected_normal_alignment_deg": (
            float(selection_pool[0]["normal_alignment_deg"])
            if selection_pool and alignment_enabled
            else None
        ),
        "passing_candidates_by_angle": [
            candidate_summary(pose) for pose in by_angle[:20]
        ],
        "passing_candidates_by_grasp": [
            candidate_summary(pose) for pose in by_grasp[:20]
        ],
        "passing_joint_pareto_frontier": [
            candidate_summary(pose) for pose in joint_frontier[:100]
        ],
        "overlap_passing_candidates_by_angle": [
            candidate_summary(pose) for pose in overlap_passing_by_angle[:20]
        ],
        "original_min_cm3": (
            float(original_values.min()) if len(original_values) else None
        ),
        "original_median_cm3": (
            float(np.median(original_values)) if len(original_values) else None
        ),
        "original_max_cm3": (
            float(original_values.max()) if len(original_values) else None
        ),
        "min_cm3": float(inflated_values.min()) if len(inflated_values) else None,
        "median_cm3": (
            float(np.median(inflated_values)) if len(inflated_values) else None
        ),
        "max_cm3": float(inflated_values.max()) if len(inflated_values) else None,
        "sorted_cm3": (
            np.sort(inflated_values).round(6).tolist()
            if len(inflated_values)
            else []
        ),
    }
    return (selection_pool[0] if selection_pool else None), audit


def execute_reachable_se3_stage(
    *,
    world,
    source_poses: Sequence[Dict[str, Any]],
    occupancy: DenseSceneOccupancy,
    gripper_voxels: np.ndarray,
    opening_voxels: np.ndarray,
    inflated_voxels: np.ndarray,
    outward_normal: Optional[np.ndarray],
    plan_arm: str,
    normal_confidence: str,
    generation: int,
    stage_name: str,
    generator_kwargs: Optional[Dict[str, Any]] = None,
    ranker=None,
    ik_worker_policy: Optional[str] = None,
    ctx=None,
) -> Dict[str, Any]:
    """Generate, score, and strictly validate one reachable SE(3) stage."""
    poses, meta = generate_reachable_se3_interpolations(
        source_poses,
        **dict(generator_kwargs or {}),
    )
    for pose in poses:
        pose["reachable_se3_generation"] = int(generation)
    alignment_meta = attach_normal_alignment(poses, outward_normal)
    overlap, overlap_meta = occupancy.counts_for_poses(
        poses,
        gripper_voxels,
    )
    grasp, grasp_meta = occupancy.counts_for_poses(
        poses,
        opening_voxels,
    )
    inflated, inflated_meta = occupancy.counts_for_poses(
        poses,
        inflated_voxels,
    )
    effective_ranker = ranker or rank_dedupe_top_ik_input
    ik_input, prefilter_meta = effective_ranker(
        poses,
        overlap,
        grasp_counts=grasp,
        inflated_counts=inflated,
        limit=IK_INPUT_MAX,
        quality_quota=60,
        alignment_quota=60,
        normal_confidence=normal_confidence,
        weak_alignment_quota_cap=60,
        alignment_grasp_ratio_override=0.95,
    )
    top_ik: List[Dict[str, Any]] = []
    ik_meta: Dict[str, Any] = {"skipped": not bool(ik_input)}
    pool_meta: Dict[str, Any] = {}
    volume_meta: Dict[str, Any] = {}
    if ik_input:
        if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
            ctx.raise_if_cancelled(stage_name)
        ranked, ik_meta = filter_poses_safe_final_dual_arm_ik(
            world,
            ik_input,
            plan_arm=plan_arm,
            ik_worker_policy=ik_worker_policy,
            ctx=ctx,
        )
        top_ik, pool_meta = select_strict_ik_pool(
            ranked,
            plan_arm=plan_arm,
            top_n=IK_TOP_N,
        )
        ik_meta.update(pool_meta)
        if top_ik:
            volume_meta = attach_final_volume_metrics(
                top_ik,
                occupancy=occupancy,
                gripper_voxels=gripper_voxels,
                opening_voxels=opening_voxels,
                inflated_voxels=inflated_voxels,
            )
    meta.update(
        {
            "normal_alignment": alignment_meta,
            "prefilter": prefilter_meta,
            "ik_filter": ik_meta,
            "strict_pool": pool_meta,
            "strict_reachable_count": int(len(top_ik)),
        }
    )
    return {
        "poses": poses,
        "ik_input": ik_input,
        "top_ik": top_ik,
        "meta": meta,
        "query": {
            "original_overlap": overlap_meta,
            "grasp": grasp_meta,
            "inflated_overlap": inflated_meta,
        },
        "volume": volume_meta,
        "alignment": alignment_meta,
    }


def execute_generated_reachable_pose_stage(
    *,
    world,
    poses: List[Dict[str, Any]],
    stage_meta: Dict[str, Any],
    occupancy: DenseSceneOccupancy,
    gripper_voxels: np.ndarray,
    opening_voxels: np.ndarray,
    inflated_voxels: np.ndarray,
    outward_normal: Optional[np.ndarray],
    plan_arm: str,
    normal_confidence: str,
    stage_name: str,
    ranker=None,
    ik_worker_policy: Optional[str] = None,
    ctx=None,
) -> Dict[str, Any]:
    """Score and strictly validate an already-generated reachable pose set."""
    alignment_meta = attach_normal_alignment(poses, outward_normal)
    overlap, overlap_meta = occupancy.counts_for_poses(
        poses,
        gripper_voxels,
    )
    grasp, grasp_meta = occupancy.counts_for_poses(
        poses,
        opening_voxels,
    )
    inflated, inflated_meta = occupancy.counts_for_poses(
        poses,
        inflated_voxels,
    )
    effective_ranker = ranker or rank_dedupe_top_ik_input
    ik_input, prefilter_meta = effective_ranker(
        poses,
        overlap,
        grasp_counts=grasp,
        inflated_counts=inflated,
        limit=IK_INPUT_MAX,
        quality_quota=60,
        alignment_quota=60,
        normal_confidence=normal_confidence,
        weak_alignment_quota_cap=60,
        alignment_grasp_ratio_override=0.95,
    )
    top_ik: List[Dict[str, Any]] = []
    ik_meta: Dict[str, Any] = {"skipped": not bool(ik_input)}
    pool_meta: Dict[str, Any] = {}
    volume_meta: Dict[str, Any] = {}
    if ik_input:
        if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
            ctx.raise_if_cancelled(stage_name)
        ranked, ik_meta = filter_poses_safe_final_dual_arm_ik(
            world,
            ik_input,
            plan_arm=plan_arm,
            ik_worker_policy=ik_worker_policy,
            ctx=ctx,
        )
        top_ik, pool_meta = select_strict_ik_pool(
            ranked,
            plan_arm=plan_arm,
            top_n=IK_TOP_N,
        )
        ik_meta.update(pool_meta)
        if top_ik:
            volume_meta = attach_final_volume_metrics(
                top_ik,
                occupancy=occupancy,
                gripper_voxels=gripper_voxels,
                opening_voxels=opening_voxels,
                inflated_voxels=inflated_voxels,
            )
    meta = dict(stage_meta)
    meta.update(
        {
            "normal_alignment": alignment_meta,
            "prefilter": prefilter_meta,
            "ik_filter": ik_meta,
            "strict_pool": pool_meta,
            "strict_reachable_count": int(len(top_ik)),
        }
    )
    return {
        "poses": poses,
        "ik_input": ik_input,
        "top_ik": top_ik,
        "meta": meta,
        "query": {
            "original_overlap": overlap_meta,
            "grasp": grasp_meta,
            "inflated_overlap": inflated_meta,
        },
        "volume": volume_meta,
        "alignment": alignment_meta,
    }


def adaptive_grasp_volume_floor(
    strict_safe_grasp_max_cm3: float,
    *,
    quality_ratio: float = ADAPTIVE_GRASP_QUALITY_RATIO,
    absolute_min_cm3: float = MIN_ADAPTIVE_GRASP_VOL_CM3,
) -> float:
    """Scale the grasp floor for small observed targets, capped at 5 cm3."""
    safe_max = max(0.0, float(strict_safe_grasp_max_cm3))
    ratio = float(np.clip(float(quality_ratio), 0.0, 1.0))
    absolute_min = max(0.0, float(absolute_min_cm3))
    if safe_max < absolute_min:
        return absolute_min
    return float(
        min(
            MIN_GRASP_VOL_CM3,
            safe_max,
            max(
                absolute_min,
                ratio * safe_max,
            ),
        )
    )


def overlap_safe_grasp_max_cm3(poses: Sequence[Dict[str, Any]]) -> float:
    """Return the best grasp among poses satisfying both collision limits."""
    values = [
        float(pose.get("grasp_vol_cm3", 0.0))
        for pose in poses
        if float(pose.get("overlap_vol_cm3", float("inf")))
        <= ORIGINAL_OVERLAP_MAX_CM3 + 1e-12
        and float(pose.get("inflated_overlap_vol_cm3", float("inf")))
        < INFLATED_OVERLAP_MAX_CM3
    ]
    return float(max(values)) if values else 0.0


def _best_arm_for_pose(pose: Dict[str, Any], preferred: Optional[str]) -> str:
    if preferred in ("left", "right"):
        return str(preferred)
    # 两臂都可行时先比腕部限位档位，同档再比 IK 误差
    scores = {}
    for arm in ("left", "right"):
        info = (pose.get("ik_filter") or {}).get(arm) or {}
        if pose.get(f"{arm}_ik_ok"):
            scores[arm] = (
                int(info.get("wrist_limit_tier", 0)),
                float(info.get("pos_err_m", float("inf"))) * 1000.0
                + float(info.get("ori_err_deg", float("inf"))),
            )
    return min(scores, key=scores.get) if scores else "right"


def render_local_three_views(
    mesh,
    *,
    hit: np.ndarray,
    anchors: np.ndarray,
    best_pose: Dict[str, Any],
    gripper_voxels: np.ndarray,
    inflation_regions: Dict[str, np.ndarray],
    threshold_audit: Dict[str, Any],
    outward_normal_world: Optional[np.ndarray],
    output_path: str,
) -> Optional[str]:
    """Render the local reconstructed mesh, selected gripper, and overlap distribution."""
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception:
        return None

    center = np.asarray(hit, dtype=np.float64).reshape(3)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if len(triangles):
        face_centers = triangles.mean(axis=1)
        local = triangles[np.linalg.norm(face_centers - center, axis=1) <= 0.20]
    else:
        local = np.zeros((0, 3, 3), dtype=np.float64)
    if len(local) > 12_000:
        stride = int(math.ceil(len(local) / 12_000))
        local = local[::stride]
    gripper_world = (
        np.asarray(best_pose["R"], dtype=np.float64)
        @ np.asarray(gripper_voxels, dtype=np.float64).T
    ).T + np.asarray(best_pose["eef_pos"], dtype=np.float64)
    if len(gripper_world) > 4_000:
        gripper_world = gripper_world[:: int(math.ceil(len(gripper_world) / 4_000))]
    region_styles = {
        "camera_body": ("#ff7f0e", "camera body", 3.0, 0.90),
        "camera_outward_10mm": (
            "#ffd400",
            "camera outward +10 mm",
            2.0,
            0.14,
        ),
        "finger_positive_y_inward_15mm": (
            "#00c9a7",
            "+Y finger inward 15 mm",
            4.0,
            0.90,
        ),
        "finger_negative_y_inward_15mm": (
            "#1976d2",
            "-Y finger inward 15 mm",
            4.0,
            0.90,
        ),
    }
    region_world: Dict[str, np.ndarray] = {}
    for name, points in inflation_regions.items():
        if name == "base_gripper":
            continue
        local_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        world_points = (
            np.asarray(best_pose["R"], dtype=np.float64) @ local_points.T
        ).T + np.asarray(best_pose["eef_pos"], dtype=np.float64)
        if len(world_points) > 4_000:
            world_points = world_points[
                :: int(math.ceil(len(world_points) / 4_000))
            ]
        region_world[name] = world_points

    local = (local - center.reshape(1, 1, 3)) * 1000.0
    gripper_world = (gripper_world - center.reshape(1, 3)) * 1000.0
    region_world = {
        name: (points - center.reshape(1, 3)) * 1000.0
        for name, points in region_world.items()
    }
    anchor_plot = (
        np.asarray(anchors, dtype=np.float64) - center.reshape(1, 3)
    ) * 1000.0
    hit_plot = np.zeros(3, dtype=np.float64)
    normal_plot = (
        np.asarray(outward_normal_world, dtype=np.float64).reshape(3) * 60.0
        if outward_normal_world is not None
        else None
    )
    gripper_vector_plot = pose_gripper_vector_world(best_pose) * 60.0

    fig = plt.figure(figsize=(18, 6), dpi=150)
    views = [(25, -70, "front"), (20, 20, "side"), (90, -90, "top")]
    geometry_point_groups = [
        gripper_world,
        anchor_plot,
        gripper_vector_plot.reshape(1, 3),
        (
            normal_plot.reshape(1, 3)
            if normal_plot is not None
            else hit_plot.reshape(1, 3)
        ),
    ]
    geometry_point_groups.extend(
        points for points in region_world.values() if len(points)
    )
    all_points = np.concatenate(geometry_point_groups, axis=0)
    lo = all_points.min(axis=0)
    hi = all_points.max(axis=0)
    span = max(float((hi - lo).max()) * 1.15, 180.0)
    plot_center = 0.5 * (lo + hi)
    for plot_index, (elev, azim, title) in enumerate(views, start=1):
        axis = fig.add_subplot(1, 4, plot_index, projection="3d")
        if len(local):
            collection = Poly3DCollection(
                local,
                facecolor=(0.35, 0.55, 0.70, 0.20),
                edgecolor=(0.25, 0.35, 0.45, 0.12),
                linewidth=0.15,
            )
            axis.add_collection3d(collection)
        axis.scatter(
            gripper_world[:, 0],
            gripper_world[:, 1],
            gripper_world[:, 2],
            s=1,
            c="#d62728",
            alpha=0.25,
        )
        for name in (
            "camera_outward_10mm",
            "camera_body",
            "finger_positive_y_inward_15mm",
            "finger_negative_y_inward_15mm",
        ):
            points = region_world.get(name)
            if points is None:
                continue
            if not len(points):
                continue
            color, _label, marker_size, alpha = region_styles[name]
            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=marker_size,
                c=color,
                alpha=alpha,
                depthshade=False,
            )
        axis.scatter(
            anchor_plot[:, 0],
            anchor_plot[:, 1],
            anchor_plot[:, 2],
            s=28,
            c="#ffd400",
            edgecolors="black",
            linewidths=0.3,
        )
        axis.scatter(
            [hit_plot[0]],
            [hit_plot[1]],
            [hit_plot[2]],
            s=55,
            c="#00ff70",
            edgecolors="black",
            linewidths=0.5,
        )
        if normal_plot is not None:
            axis.quiver(
                0.0,
                0.0,
                0.0,
                normal_plot[0],
                normal_plot[1],
                normal_plot[2],
                color="#39d353",
                linewidth=2.5,
                arrow_length_ratio=0.18,
            )
        axis.quiver(
            0.0,
            0.0,
            0.0,
            gripper_vector_plot[0],
            gripper_vector_plot[1],
            gripper_vector_plot[2],
            color="#ff00aa",
            linewidth=2.5,
            arrow_length_ratio=0.18,
        )
        for i, anchor in enumerate(anchor_plot):
            axis.text(anchor[0], anchor[1], anchor[2], str(i + 1), fontsize=6)
        axis.set_xlim(plot_center[0] - span / 2, plot_center[0] + span / 2)
        axis.set_ylim(plot_center[1] - span / 2, plot_center[1] + span / 2)
        axis.set_zlim(plot_center[2] - span / 2, plot_center[2] + span / 2)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(title)
        axis.set_xlabel("x (mm)", fontsize=8)
        axis.set_ylabel("y (mm)", fontsize=8)
        axis.set_zlabel("z (mm)", fontsize=8)
        axis.tick_params(labelsize=6, pad=0)
        if title == "top":
            axis.set_zticklabels([])

    histogram_axis = fig.add_subplot(1, 4, 4)
    values = np.asarray(threshold_audit.get("sorted_cm3") or [], dtype=np.float64)
    if len(values):
        histogram_axis.plot(np.arange(1, len(values) + 1), values, marker=".", ms=4)
    histogram_axis.axhline(
        float(threshold_audit["threshold_cm3"]),
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"< {threshold_audit['threshold_cm3']:.3f} cm3",
    )
    histogram_axis.set_xlabel("IK top pose sorted by inflated overlap")
    histogram_axis.set_ylabel("inflated overlap (cm3)")
    histogram_axis.grid(True, alpha=0.25)
    histogram_axis.legend(loc="best", fontsize=8)
    histogram_axis.set_title(
        f"final pass {threshold_audit['passing_count']}/"
        f"{threshold_audit['input_count']} | overlap pass "
        f"{threshold_audit['overlap_passing_count']}"
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#d62728",
            markersize=7,
            label="original palm + fingers (no expansion)",
        ),
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markersize=7,
            label=label,
        )
        for color, label, _marker_size, _alpha in region_styles.values()
    )
    legend_handles.extend(
        [
            Line2D([0], [0], color="#39d353", lw=2.5, label="outward normal"),
            Line2D([0], [0], color="#ff00aa", lw=2.5, label="gripper vector"),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        f"{BUILD} | best {best_pose['axial_point']} | "
        f"grasp={best_pose['grasp_vol_cm3']:.3f} cm3 | "
        f"overlap={best_pose['overlap_vol_cm3']:.3f} cm3 | "
        f"inflated={best_pose['inflated_overlap_vol_cm3']:.3f} cm3 | "
        f"normal angle={best_pose.get('normal_alignment_deg', float('nan')):.2f} deg"
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.93))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


RGBD_BATCH_GRIPPER_COLORS = (
    (230, 48, 48),
    (35, 170, 235),
    (245, 180, 35),
    (190, 75, 220),
    (45, 190, 105),
    (245, 115, 35),
    (65, 105, 225),
    (225, 80, 145),
    (0, 190, 190),
    (160, 120, 45),
    (115, 75, 220),
    (80, 165, 80),
    (235, 90, 65),
    (45, 125, 190),
    (205, 145, 20),
    (125, 125, 125),
)


def _draw_local_gripper_overlay(
    image: np.ndarray,
    *,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    color: Tuple[int, int, int],
) -> None:
    rotation = _quat_to_mat_xyzw(eef_quat)
    local_points = gripper_geometry(VOXEL_M)["gripper_voxels"][::12]
    world_points = (
        np.asarray(eef_pos, dtype=np.float64).reshape(1, 3)
        + (rotation @ local_points.T).T
    )
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    for point in world_points:
        pixel = _world_to_pixel(
            cam_pos,
            cam_quat,
            point,
            int(w),
            int(h),
            float(fl),
            float(ha),
        )
        if pixel is None:
            continue
        u, v = pixel
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            image[v, u] = bgr


def render_batch_gripper_overlay(
    session: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    *,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    output_path: str,
) -> Optional[str]:
    """Render all successful batch grippers on one frozen head image."""
    import shutil

    import cv2

    rgb_path = str(
        session.get("rgb_path")
        or os.path.join(str(session["init_dir"]), "rgb.png")
    )
    if not os.path.isfile(rgb_path):
        return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy2(rgb_path, output_path)
    depth_path = str(
        session.get("depth_path")
        or os.path.join(str(session["init_dir"]), "depth.npy")
    )
    scene_depth = None
    if os.path.isfile(depth_path):
        scene_depth = np.load(depth_path).astype(np.float32)
        if scene_depth.ndim == 3:
            scene_depth = scene_depth[..., 0]

    labels: List[Tuple[str, Tuple[int, int, int], Optional[Tuple[int, int]]]] = []
    for row_index, row in enumerate(rows):
        color = RGBD_BATCH_GRIPPER_COLORS[
            row_index % len(RGBD_BATCH_GRIPPER_COLORS)
        ]
        payload = row.get("payload")
        point = row.get("point") or {}
        if not isinstance(payload, dict) or not payload.get("ok"):
            reason = str(row.get("error") or "planning failed")
            labels.append(
                (
                    f"#{row_index + 1} {point.get('plan_arm', 'any')}: "
                    f"FAIL {reason[:52]}",
                    (155, 155, 155),
                    None,
                )
            )
            continue
        eef_pose = payload.get("eef_pose") or {}
        eef_pos = np.asarray(eef_pose.get("pos"), dtype=np.float64).reshape(3)
        eef_quat = np.asarray(eef_pose.get("quat"), dtype=np.float64).reshape(4)
        image = cv2.imread(output_path)
        if image is not None:
            _draw_local_gripper_overlay(
                image,
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            w=int(w),
            h=int(h),
            fl=float(fl),
            ha=float(ha),
                color=color,
            )
            cv2.imwrite(output_path, image)
        projected = _world_to_pixel(
            cam_pos,
            cam_quat,
            eef_pos,
            int(w),
            int(h),
            float(fl),
            float(ha),
        )
        arm = str(
            payload.get("recommended_arm")
            or payload.get("arm")
            or point.get("plan_arm")
            or "any"
        )
        plan_id = str(row.get("plan_id") or "")
        grasp = float(payload.get("grasp_vol_cm3") or 0.0)
        labels.append(
            (
                f"#{row_index + 1} {arm} {plan_id} grasp={grasp:.2f}cm3",
                color,
                (
                    (int(projected[0]), int(projected[1]))
                    if projected is not None
                    else None
                ),
            )
        )

    image = cv2.imread(output_path)
    if image is None:
        return output_path
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, (label, rgb, projected) in enumerate(labels):
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        y = 24 + index * 22
        cv2.putText(
            image,
            label,
            (12, y),
            font,
            0.52,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (12, y),
            font,
            0.52,
            bgr,
            1,
            cv2.LINE_AA,
        )
        if projected is not None:
            px = (
                int(np.clip(projected[0] + 8, 4, image.shape[1] - 32)),
                int(np.clip(projected[1] - 8, 18, image.shape[0] - 4)),
            )
            marker = f"#{index + 1}"
            cv2.putText(
                image,
                marker,
                px,
                font,
                0.62,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                marker,
                px,
                font,
                0.62,
                bgr,
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(output_path, image)
    return output_path


def plan_grasp_point_filter_rgbd(
    *,
    world,
    session: Dict[str, Any],
    u: int,
    v: int,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    plan_arm: str,
    seed: int,
    ctx=None,
    prepared_scene: Optional[
        Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]
    ] = None,
    prepared_target: Optional[Dict[str, Any]] = None,
    render_debug: bool = True,
) -> Dict[str, Any]:
    """Execute the complete RGB-D-only pipeline and return a plan payload."""
    del seed  # The finalized sampler is deterministic.
    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("rgbd reconstruction")
    if prepared_scene is None:
        mesh, _rgb, depth, reconstruction_meta = (
            reconstruct_scene_mesh_from_session(
                session,
                camera_pos=cam_pos,
                camera_quat_xyzw=cam_quat,
                focal_length=fl,
                horizontal_aperture=ha,
            )
        )
        reconstruction_meta = dict(reconstruction_meta)
        reconstruction_meta["shared_across_batch"] = False
    else:
        mesh, _rgb, depth, reconstruction_meta = prepared_scene
        reconstruction_meta = dict(reconstruction_meta)
        reconstruction_meta["shared_across_batch"] = True
    target = (
        prepare_rgbd_grasp_target(
            depth,
            u=int(u),
            v=int(v),
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            w=int(w),
            h=int(h),
            fl=float(fl),
            ha=float(ha),
        )
        if prepared_target is None
        else prepared_target
    )
    hit = np.asarray(target["hit"], dtype=np.float64).reshape(3)
    hit_audit = dict(target["hit_audit"])
    outward_normal = target.get("outward_normal")
    if outward_normal is not None:
        outward_normal = np.asarray(outward_normal, dtype=np.float64).reshape(3)
    normal_audit = dict(target["normal_audit"])
    normal_confidence = str(normal_audit.get("confidence", "none"))
    anchors, anchor_meta = clicked_hit_anchor(hit)
    if ctx is not None:
        try:
            reconstruction_elapsed_log = (
                f"{float(reconstruction_meta.get('elapsed_s')):.2f}s"
            )
        except (TypeError, ValueError):
            reconstruction_elapsed_log = "unknown"
        normal_log = (
            np.asarray(outward_normal).round(4).tolist()
            if outward_normal is not None
            else None
        )
        ctx.log(
            f"  [grasp_point_filter_rgbd] hit={hit.round(4).tolist()} "
            f"method={hit_audit['method']} mesh={reconstruction_meta.get('build')} "
            f"anchor={anchor_meta['source']} "
            f"reconstruct={reconstruction_elapsed_log}"
        )
        ctx.log(
            "  [grasp_point_filter_rgbd] normal "
            f"confidence={normal_confidence} "
            f"selection={normal_audit.get('selection')} "
            f"normal={normal_log}"
        )

    poses = generate_rgbd_filter_poses(anchors)
    camera_face_meta = apply_camera_face_preserving_anchor(
        poses,
        world=world,
        ctx=ctx,
    )
    raw_alignment_meta = attach_normal_alignment(poses, outward_normal)
    expected_per_anchor = 20 * N_ROLL * len(AXIAL_Z_M)
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] raw poses={len(poses)} "
            f"({len(anchors)} anchors x 20 approach x {N_ROLL} roll x "
            f"{len(AXIAL_Z_M)} axial; {expected_per_anchor}/anchor)"
        )

    frozen_geometry = gripper_geometry(VOXEL_M)
    gripper_voxels = frozen_geometry["gripper_voxels"]
    gripper_components = frozen_geometry["components"]
    opening_voxels = frozen_geometry["opening_voxels"]
    inflation_started = time.perf_counter()
    inflated_voxels, inflation_regions, inflation_meta = (
        inflated_gripper_voxels_eef(
            gripper_voxels,
            component_voxels=gripper_components,
            voxel_m=VOXEL_M,
            camera_inflate_m=CAMERA_INFLATE_M,
            finger_inward_inflate_m=FINGER_INWARD_INFLATE_M,
            return_regions=True,
        )
    )
    inflation_elapsed_s = float(time.perf_counter() - inflation_started)
    if len(inflated_voxels) < len(gripper_voxels):
        raise RuntimeError("inflated gripper voxel set is not a superset")
    if ctx is not None:
        inflation_cache = inflated_gripper_voxel_cache_info()
        ctx.log(
            "  [grasp_point_filter_rgbd] inflated geometry "
            f"policy={inflation_meta['policy']} "
            f"base={inflation_meta['base_gripper_voxels']} "
            f"camera_body+shell="
            f"{inflation_meta['camera_body_added_voxels']}+"
            f"{inflation_meta['camera_shell_added_voxels']} "
            f"finger_inward(+Y/-Y)="
            f"{inflation_meta['finger_positive_y_added_voxels']}/"
            f"{inflation_meta['finger_negative_y_added_voxels']} "
            f"union={inflation_meta['inflated_union_voxels']} "
            f"elapsed={inflation_elapsed_s:.3f}s "
            f"cache={inflation_cache['hits']}h/"
            f"{inflation_cache['misses']}m"
        )
    if ctx is not None and str(
        getattr(mesh, "geometry_kind", "")
    ).startswith("projective_cuda"):
        ctx.log(
            "  [grasp_point_filter_rgbd] occupancy_pre "
            f"representation={getattr(mesh, 'geometry_kind', 'unknown')} "
            f"depth_shape={list(np.asarray(depth).shape[:2])} "
            "mesh_built=False mesh_split=False trimesh_contains=False"
        )
    elif ctx is not None:
        try:
            import faulthandler

            faulthandler.enable()
            verts = np.asarray(mesh.vertices)
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            extents = bounds[1] - bounds[0]
            watertight_log, topology_validation = (
                _mesh_topology_diagnostic(mesh, reconstruction_meta)
            )
            ctx.log(
                "  [grasp_point_filter_rgbd] occupancy_pre "
                f"watertight={watertight_log} "
                f"topology={topology_validation} "
                f"verts={int(len(verts))} faces={int(len(mesh.faces))} "
                f"extents_m={extents.tolist()} "
                f"bounds_min={bounds[0].tolist()} "
                f"bounds_max={bounds[1].tolist()} "
                f"finite={bool(np.isfinite(verts).all())}"
            )
            dump_dir = os.environ.get(
                "BEHAVIOR_OCCUPANCY_MESH_DUMP_DIR",
                "",
            ).strip()
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                dump_path = os.path.join(dump_dir, "pre_occupancy.ply")
                mesh.export(dump_path)
                ctx.log(
                    "  [grasp_point_filter_rgbd] occupancy_pre dumped "
                    f"{dump_path}"
                )
        except Exception as exc:
            ctx.log(
                "  [grasp_point_filter_rgbd] occupancy_pre failed: "
                f"{type(exc).__name__}: {exc}"
            )
    try:
        occupancy = build_local_scene_occupancy(
            mesh,
            hit=hit,
            query_offsets=(gripper_voxels, opening_voxels, inflated_voxels),
            anchor_radius_m=RESCUE_OCCUPANCY_MARGIN_M,
            ctx=ctx,
        )
    except Exception as exc:
        if ctx is not None:
            ctx.log(
                "  [grasp_point_filter_rgbd] occupancy_exception "
                f"{type(exc).__name__}: {exc}"
            )
        raise
    raw_overlap, raw_overlap_meta = occupancy.counts_for_poses(
        poses,
        gripper_voxels,
    )
    raw_grasp, raw_grasp_meta = occupancy.counts_for_poses(
        poses,
        opening_voxels,
    )
    raw_inflated, raw_inflated_meta = occupancy.counts_for_poses(
        poses,
        inflated_voxels,
    )
    refinement_seeds, refinement_meta = select_local_refinement_seeds(
        poses,
        raw_overlap,
        raw_grasp,
        raw_inflated,
    )
    zero_safe_rescue = bool(refinement_meta.get("zero_safe_rescue", False))
    refined_poses = generate_local_orientation_refinements(
        refinement_seeds,
        tilt_deg=(
            RESCUE_TILT_DEG if zero_safe_rescue else REFINEMENT_TILT_DEG
        ),
        roll_deg=(
            RESCUE_ROLL_DEG if zero_safe_rescue else REFINEMENT_ROLL_DEG
        ),
    )
    refined_camera_face_meta = apply_camera_face_preserving_anchor(
        refined_poses,
        world=world,
        ctx=None,
    )
    refined_alignment_meta = attach_normal_alignment(
        refined_poses,
        outward_normal,
    )
    refined_overlap, refined_overlap_meta = occupancy.counts_for_poses(
        refined_poses,
        gripper_voxels,
    )
    refined_grasp, refined_grasp_meta = occupancy.counts_for_poses(
        refined_poses,
        opening_voxels,
    )
    refined_inflated, refined_inflated_meta = occupancy.counts_for_poses(
        refined_poses,
        inflated_voxels,
    )
    rescue_source_poses = poses + refined_poses
    rescue_source_overlap = np.concatenate((raw_overlap, refined_overlap))
    rescue_source_grasp = np.concatenate((raw_grasp, refined_grasp))
    rescue_source_inflated = np.concatenate((raw_inflated, refined_inflated))
    contact_seeds, contact_seed_meta = select_rescue_sampling_seeds(
        rescue_source_poses,
        rescue_source_overlap,
        rescue_source_grasp,
        rescue_source_inflated,
    )
    contact_poses, contact_meta = generate_opening_contact_refinements(
        contact_seeds
    )
    contact_alignment_meta = attach_normal_alignment(
        contact_poses,
        outward_normal,
    )
    contact_overlap, contact_overlap_meta = occupancy.counts_for_poses(
        contact_poses,
        gripper_voxels,
    )
    contact_grasp, contact_grasp_meta = occupancy.counts_for_poses(
        contact_poses,
        opening_voxels,
    )
    contact_inflated, contact_inflated_meta = occupancy.counts_for_poses(
        contact_poses,
        inflated_voxels,
    )
    translation_source_poses = rescue_source_poses + contact_poses
    translation_source_overlap = np.concatenate(
        (rescue_source_overlap, contact_overlap)
    )
    translation_source_grasp = np.concatenate(
        (rescue_source_grasp, contact_grasp)
    )
    translation_source_inflated = np.concatenate(
        (rescue_source_inflated, contact_inflated)
    )
    translation_seeds, translation_seed_meta = select_rescue_sampling_seeds(
        translation_source_poses,
        translation_source_overlap,
        translation_source_grasp,
        translation_source_inflated,
    )
    pre_ik_translation_poses, pre_ik_translation_meta = (
        generate_pre_ik_translation_refinements(
            translation_seeds,
            outward_normal_world=outward_normal,
        )
    )
    pre_ik_translation_alignment_meta = attach_normal_alignment(
        pre_ik_translation_poses,
        outward_normal,
    )
    pre_ik_translation_overlap, pre_ik_translation_overlap_meta = (
        occupancy.counts_for_poses(
            pre_ik_translation_poses,
            gripper_voxels,
        )
    )
    pre_ik_translation_grasp, pre_ik_translation_grasp_meta = (
        occupancy.counts_for_poses(
            pre_ik_translation_poses,
            opening_voxels,
        )
    )
    pre_ik_translation_inflated, pre_ik_translation_inflated_meta = (
        occupancy.counts_for_poses(
            pre_ik_translation_poses,
            inflated_voxels,
        )
    )
    contact_meta.update(
        {
            "seed_selection": contact_seed_meta,
            "normal_alignment": contact_alignment_meta,
        }
    )
    pre_ik_translation_meta.update(
        {
            "seed_selection": translation_seed_meta,
            "normal_alignment": pre_ik_translation_alignment_meta,
        }
    )
    interpolation_sources = rescue_source_poses + contact_poses
    interpolation_source_overlap = np.concatenate(
        (rescue_source_overlap, contact_overlap)
    )
    interpolation_source_grasp = np.concatenate(
        (rescue_source_grasp, contact_grasp)
    )
    interpolation_source_inflated = np.concatenate(
        (rescue_source_inflated, contact_inflated)
    )
    interpolated_poses, interpolation_meta = (
        generate_quality_pose_interpolations(
            interpolation_sources,
            interpolation_source_overlap,
            interpolation_source_grasp,
            interpolation_source_inflated,
        )
    )
    interpolated_camera_face_meta = apply_camera_face_preserving_anchor(
        interpolated_poses,
        world=world,
        ctx=None,
    )
    interpolated_alignment_meta = attach_normal_alignment(
        interpolated_poses,
        outward_normal,
    )
    interpolated_overlap, interpolated_overlap_meta = (
        occupancy.counts_for_poses(
            interpolated_poses,
            gripper_voxels,
        )
    )
    interpolated_grasp, interpolated_grasp_meta = occupancy.counts_for_poses(
        interpolated_poses,
        opening_voxels,
    )
    interpolated_inflated, interpolated_inflated_meta = (
        occupancy.counts_for_poses(
            interpolated_poses,
            inflated_voxels,
        )
    )
    interpolation_meta.update(
        {
            "camera_face": interpolated_camera_face_meta,
            "normal_alignment": interpolated_alignment_meta,
        }
    )
    pareto_sources = (
        interpolation_sources
        + interpolated_poses
        + pre_ik_translation_poses
    )
    pareto_source_overlap = np.concatenate(
        (
            interpolation_source_overlap,
            interpolated_overlap,
            pre_ik_translation_overlap,
        )
    )
    pareto_source_grasp = np.concatenate(
        (
            interpolation_source_grasp,
            interpolated_grasp,
            pre_ik_translation_grasp,
        )
    )
    pareto_source_inflated = np.concatenate(
        (
            interpolation_source_inflated,
            interpolated_inflated,
            pre_ik_translation_inflated,
        )
    )
    pareto_micro_poses: List[Dict[str, Any]] = []
    pareto_micro_meta: Dict[str, Any] = {
        "stage": "post_strict_ik",
        "seed_limit": int(POST_IK_MICRO_SEED_MAX),
        "pose_count": 0,
    }
    pareto_micro_alignment_meta: Dict[str, Any] = {
        "enabled": bool(outward_normal is not None),
        "pose_count": 0,
    }
    prefilter_poses = pareto_sources
    prefilter_overlap = pareto_source_overlap
    prefilter_grasp = pareto_source_grasp
    prefilter_inflated = pareto_source_inflated
    ik_input, prefilter_meta = rank_dedupe_top_ik_input(
        prefilter_poses,
        prefilter_overlap,
        grasp_counts=prefilter_grasp,
        inflated_counts=prefilter_inflated,
        limit=IK_INPUT_MAX,
        normal_confidence=normal_confidence,
    )
    refinement_meta.update(
        {
            "pose_count": int(len(refined_poses)),
            "camera_face": refined_camera_face_meta,
            "normal_alignment": refined_alignment_meta,
            "tilt_deg": [
                float(value)
                for value in (
                    RESCUE_TILT_DEG
                    if zero_safe_rescue
                    else REFINEMENT_TILT_DEG
                )
            ],
            "roll_deg": [
                float(value)
                for value in (
                    RESCUE_ROLL_DEG
                    if zero_safe_rescue
                    else REFINEMENT_ROLL_DEG
                )
            ],
        }
    )
    prefilter_meta["query"] = {
        "original_overlap": raw_overlap_meta,
        "grasp": raw_grasp_meta,
        "inflated_overlap": raw_inflated_meta,
        "refined_original_overlap": refined_overlap_meta,
        "refined_grasp": refined_grasp_meta,
        "refined_inflated_overlap": refined_inflated_meta,
        "interpolated_original_overlap": interpolated_overlap_meta,
        "interpolated_grasp": interpolated_grasp_meta,
        "interpolated_inflated_overlap": interpolated_inflated_meta,
        "contact_original_overlap": contact_overlap_meta,
        "contact_grasp": contact_grasp_meta,
        "contact_inflated_overlap": contact_inflated_meta,
        "pre_ik_translation_original_overlap": (
            pre_ik_translation_overlap_meta
        ),
        "pre_ik_translation_grasp": pre_ik_translation_grasp_meta,
        "pre_ik_translation_inflated_overlap": (
            pre_ik_translation_inflated_meta
        ),
    }
    prefilter_meta["refinement"] = refinement_meta
    prefilter_meta["contact_refinement"] = contact_meta
    prefilter_meta["pre_ik_translation_refinement"] = (
        pre_ik_translation_meta
    )
    prefilter_meta["interpolation"] = interpolation_meta
    prefilter_meta["pareto_micro_refinement"] = pareto_micro_meta
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] quality-aware prefilter "
            f"min={prefilter_meta['overlap_min_cm3']:.3f}cm3 "
            f"safe={prefilter_meta['strict_safe_count']} "
            f"safe_grasp_max={prefilter_meta['strict_safe_grasp_max_cm3']:.3f}cm3 "
            f"quality/explore={prefilter_meta['quality_selected_count']}/"
            f"{len(ik_input) - prefilter_meta['quality_selected_count']} "
            f"normal_target/added={prefilter_meta['alignment_target_count']}/"
            f"{prefilter_meta['alignment_added_count']} "
            f"refined={len(refined_poses)} "
            f"contact={len(contact_poses)} "
            f"preIK_translation={len(pre_ik_translation_poses)} "
            f"interpolated={len(interpolated_poses)} "
            f"top120_source={prefilter_meta['selected_source_counts']} "
            f"unique={len(ik_input)}/{len(prefilter_poses)}"
        )
    if not ik_input:
        raise GraspObjPlanningError("RGBD overlap prefilter produced no IK input pose")

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("rgbd dual-arm IK")
    ik_ranked, ik_meta = filter_poses_safe_final_dual_arm_ik(
        world,
        ik_input,
        plan_arm=plan_arm,
        ctx=ctx,
    )
    top_ik, pool_meta = select_strict_ik_pool(
        ik_ranked,
        plan_arm=plan_arm,
        top_n=IK_TOP_N,
    )
    ik_meta.update(pool_meta)
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] strict IK selection={pool_meta['selection']} "
            f"both/left/right={pool_meta['n_both_ok']}/"
            f"{pool_meta['n_left_ok']}/{pool_meta['n_right_ok']} "
            f"top={len(top_ik)}"
        )
    if not top_ik:
        raise GraspObjPlanningError(
            "RGBD planner has no strict IK-reachable pose "
            f"(plan_arm={plan_arm}, left={pool_meta['n_left_ok']}, "
            f"right={pool_meta['n_right_ok']}, both={pool_meta['n_both_ok']})"
        )

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("rgbd final volume")
    volume_meta = attach_final_volume_metrics(
        top_ik,
        occupancy=occupancy,
        gripper_voxels=gripper_voxels,
        opening_voxels=opening_voxels,
        inflated_voxels=inflated_voxels,
    )
    volume_meta["inflation_geometry"] = inflation_meta
    base_top_ik = list(top_ik)
    base_overlap = np.asarray(
        [int(pose["overlap_vox"]) for pose in base_top_ik],
        dtype=np.int64,
    )
    base_grasp = np.asarray(
        [int(pose["grasp_vox"]) for pose in base_top_ik],
        dtype=np.int64,
    )
    base_inflated = np.asarray(
        [int(pose["inflated_overlap_vox"]) for pose in base_top_ik],
        dtype=np.int64,
    )
    pareto_micro_seeds, pareto_micro_seed_meta = (
        select_joint_pareto_refinement_seeds(
            base_top_ik,
            base_overlap,
            base_grasp,
            base_inflated,
            limit=POST_IK_MICRO_SEED_MAX,
        )
    )
    pareto_micro_poses = generate_pareto_micro_refinements(
        pareto_micro_seeds
    )
    reachability_bridge_poses, reachability_bridge_meta = (
        generate_reachability_bridge_interpolations(
            pareto_micro_seeds,
            prefilter_poses,
        )
    )
    pareto_micro_camera_face_meta = apply_camera_face_preserving_anchor(
        pareto_micro_poses,
        world=world,
        ctx=None,
    )
    reachability_bridge_camera_face_meta = (
        apply_camera_face_preserving_anchor(
            reachability_bridge_poses,
            world=world,
            ctx=None,
        )
    )
    pareto_micro_alignment_meta = attach_normal_alignment(
        pareto_micro_poses,
        outward_normal,
    )
    reachability_bridge_alignment_meta = attach_normal_alignment(
        reachability_bridge_poses,
        outward_normal,
    )
    pareto_micro_overlap, pareto_micro_overlap_meta = (
        occupancy.counts_for_poses(
            pareto_micro_poses,
            gripper_voxels,
        )
    )
    pareto_micro_grasp, pareto_micro_grasp_meta = occupancy.counts_for_poses(
        pareto_micro_poses,
        opening_voxels,
    )
    pareto_micro_inflated, pareto_micro_inflated_meta = (
        occupancy.counts_for_poses(
            pareto_micro_poses,
            inflated_voxels,
        )
    )
    reachability_bridge_overlap, reachability_bridge_overlap_meta = (
        occupancy.counts_for_poses(
            reachability_bridge_poses,
            gripper_voxels,
        )
    )
    reachability_bridge_grasp, reachability_bridge_grasp_meta = (
        occupancy.counts_for_poses(
            reachability_bridge_poses,
            opening_voxels,
        )
    )
    reachability_bridge_inflated, reachability_bridge_inflated_meta = (
        occupancy.counts_for_poses(
            reachability_bridge_poses,
            inflated_voxels,
        )
    )
    post_ik_refinement_poses = (
        pareto_micro_poses + reachability_bridge_poses
    )
    post_ik_refinement_overlap = np.concatenate(
        (pareto_micro_overlap, reachability_bridge_overlap)
    )
    post_ik_refinement_grasp = np.concatenate(
        (pareto_micro_grasp, reachability_bridge_grasp)
    )
    post_ik_refinement_inflated = np.concatenate(
        (pareto_micro_inflated, reachability_bridge_inflated)
    )
    pareto_micro_ik_input, pareto_micro_prefilter_meta = (
        rank_seed_balanced_micro_ik_input(
            post_ik_refinement_poses,
            post_ik_refinement_overlap,
            post_ik_refinement_grasp,
            post_ik_refinement_inflated,
            limit=IK_INPUT_MAX,
        )
    )
    pareto_micro_top_ik: List[Dict[str, Any]] = []
    pareto_micro_ik_meta: Dict[str, Any] = {
        "skipped": not bool(pareto_micro_ik_input),
    }
    pareto_micro_pool_meta: Dict[str, Any] = {}
    pareto_micro_volume_meta: Dict[str, Any] = {}
    if pareto_micro_ik_input:
        if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
            ctx.raise_if_cancelled("rgbd post-IK Pareto refinement")
        pareto_micro_ranked, pareto_micro_ik_meta = (
            filter_poses_safe_final_dual_arm_ik(
                world,
                pareto_micro_ik_input,
                plan_arm=plan_arm,
                ctx=ctx,
            )
        )
        pareto_micro_top_ik, pareto_micro_pool_meta = select_strict_ik_pool(
            pareto_micro_ranked,
            plan_arm=plan_arm,
            top_n=IK_TOP_N,
        )
        pareto_micro_ik_meta.update(pareto_micro_pool_meta)
        if pareto_micro_top_ik:
            pareto_micro_volume_meta = attach_final_volume_metrics(
                pareto_micro_top_ik,
                occupancy=occupancy,
                gripper_voxels=gripper_voxels,
                opening_voxels=opening_voxels,
                inflated_voxels=inflated_voxels,
            )

    merged_top_ik: List[Dict[str, Any]] = []
    merged_keys = set()
    for pose in base_top_ik + pareto_micro_top_ik:
        key = pose_dedupe_key(pose)
        if key in merged_keys:
            continue
        merged_keys.add(key)
        merged_top_ik.append(pose)
    top_ik = merged_top_ik
    pareto_micro_meta = {
        **pareto_micro_seed_meta,
        "stage": "post_strict_ik",
        "seed_limit": int(POST_IK_MICRO_SEED_MAX),
        "pose_count": int(len(pareto_micro_poses)),
        "combined_pose_count": int(len(post_ik_refinement_poses)),
        "camera_face": pareto_micro_camera_face_meta,
        "normal_alignment": pareto_micro_alignment_meta,
        "tilt_deg": [float(value) for value in PARETO_MICRO_TILT_DEG],
        "roll_deg": [float(value) for value in PARETO_MICRO_ROLL_DEG],
        "prefilter": pareto_micro_prefilter_meta,
        "ik_filter": pareto_micro_ik_meta,
        "strict_pool": pareto_micro_pool_meta,
        "strict_reachable_count": int(len(pareto_micro_top_ik)),
        "merged_reachable_count": int(len(top_ik)),
        "reachability_bridge": {
            **reachability_bridge_meta,
            "camera_face": reachability_bridge_camera_face_meta,
            "normal_alignment": reachability_bridge_alignment_meta,
        },
    }
    prefilter_meta["pareto_micro_refinement"] = pareto_micro_meta
    prefilter_meta["query"].update(
        {
            "pareto_micro_original_overlap": pareto_micro_overlap_meta,
            "pareto_micro_grasp": pareto_micro_grasp_meta,
            "pareto_micro_inflated_overlap": pareto_micro_inflated_meta,
            "reachability_bridge_original_overlap": (
                reachability_bridge_overlap_meta
            ),
            "reachability_bridge_grasp": reachability_bridge_grasp_meta,
            "reachability_bridge_inflated_overlap": (
                reachability_bridge_inflated_meta
            ),
        }
    )
    ik_meta["post_ik_pareto_refinement"] = {
        "prefilter": pareto_micro_prefilter_meta,
        "ik_filter": pareto_micro_ik_meta,
        "strict_pool": pareto_micro_pool_meta,
    }
    volume_meta["base_reachable_count"] = int(len(base_top_ik))
    volume_meta["post_ik_pareto_reachable_count"] = int(
        len(pareto_micro_top_ik)
    )
    volume_meta["merged_reachable_count"] = int(len(top_ik))
    volume_meta["post_ik_pareto_volume"] = pareto_micro_volume_meta
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] post-IK Pareto seeds="
            f"{len(pareto_micro_seeds)} micro={len(pareto_micro_poses)} "
            f"bridge={len(reachability_bridge_poses)} "
            f"IK_input={len(pareto_micro_ik_input)} strict="
            f"{len(pareto_micro_top_ik)} merged={len(top_ik)}"
        )
    reachable_se3_poses, reachable_se3_meta = (
        generate_reachable_se3_interpolations(top_ik)
    )
    for pose in reachable_se3_poses:
        pose["reachable_se3_generation"] = 1
    reachable_se3_alignment_meta = attach_normal_alignment(
        reachable_se3_poses,
        outward_normal,
    )
    reachable_se3_overlap, reachable_se3_overlap_meta = (
        occupancy.counts_for_poses(
            reachable_se3_poses,
            gripper_voxels,
        )
    )
    reachable_se3_grasp, reachable_se3_grasp_meta = (
        occupancy.counts_for_poses(
            reachable_se3_poses,
            opening_voxels,
        )
    )
    reachable_se3_inflated, reachable_se3_inflated_meta = (
        occupancy.counts_for_poses(
            reachable_se3_poses,
            inflated_voxels,
        )
    )
    reachable_se3_ik_input, reachable_se3_prefilter_meta = (
        rank_dedupe_top_ik_input(
            reachable_se3_poses,
            reachable_se3_overlap,
            grasp_counts=reachable_se3_grasp,
            inflated_counts=reachable_se3_inflated,
            limit=IK_INPUT_MAX,
            quality_quota=60,
            alignment_quota=60,
            normal_confidence=normal_confidence,
            weak_alignment_quota_cap=60,
            alignment_grasp_ratio_override=0.95,
        )
    )
    reachable_se3_top_ik: List[Dict[str, Any]] = []
    reachable_se3_ik_meta: Dict[str, Any] = {
        "skipped": not bool(reachable_se3_ik_input),
    }
    reachable_se3_pool_meta: Dict[str, Any] = {}
    reachable_se3_volume_meta: Dict[str, Any] = {}
    if reachable_se3_ik_input:
        if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
            ctx.raise_if_cancelled("rgbd reachable SE3 interpolation")
        reachable_se3_ranked, reachable_se3_ik_meta = (
            filter_poses_safe_final_dual_arm_ik(
                world,
                reachable_se3_ik_input,
                plan_arm=plan_arm,
                ctx=ctx,
            )
        )
        reachable_se3_top_ik, reachable_se3_pool_meta = (
            select_strict_ik_pool(
                reachable_se3_ranked,
                plan_arm=plan_arm,
                top_n=IK_TOP_N,
            )
        )
        reachable_se3_ik_meta.update(reachable_se3_pool_meta)
        if reachable_se3_top_ik:
            reachable_se3_volume_meta = attach_final_volume_metrics(
                reachable_se3_top_ik,
                occupancy=occupancy,
                gripper_voxels=gripper_voxels,
                opening_voxels=opening_voxels,
                inflated_voxels=inflated_voxels,
            )
    for pose in reachable_se3_top_ik:
        key = pose_dedupe_key(pose)
        if key in merged_keys:
            continue
        merged_keys.add(key)
        top_ik.append(pose)
    reachable_se3_meta.update(
        {
            "normal_alignment": reachable_se3_alignment_meta,
            "prefilter": reachable_se3_prefilter_meta,
            "ik_filter": reachable_se3_ik_meta,
            "strict_pool": reachable_se3_pool_meta,
            "strict_reachable_count": int(len(reachable_se3_top_ik)),
            "merged_reachable_count": int(len(top_ik)),
        }
    )
    pareto_micro_meta["reachable_se3_interpolation"] = reachable_se3_meta
    prefilter_meta["query"].update(
        {
            "reachable_se3_original_overlap": (
                reachable_se3_overlap_meta
            ),
            "reachable_se3_grasp": reachable_se3_grasp_meta,
            "reachable_se3_inflated_overlap": (
                reachable_se3_inflated_meta
            ),
        }
    )
    ik_meta["reachable_se3_interpolation"] = {
        "prefilter": reachable_se3_prefilter_meta,
        "ik_filter": reachable_se3_ik_meta,
        "strict_pool": reachable_se3_pool_meta,
    }
    volume_meta["reachable_se3_reachable_count"] = int(
        len(reachable_se3_top_ik)
    )
    volume_meta["reachable_se3_volume"] = reachable_se3_volume_meta
    volume_meta["merged_reachable_count"] = int(len(top_ik))
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] reachable-SE3 endpoints="
            f"{reachable_se3_meta['pareto_endpoint_count']} pairs="
            f"{reachable_se3_meta['pair_count']} poses="
            f"{len(reachable_se3_poses)} IK_input="
            f"{len(reachable_se3_ik_input)} strict="
            f"{len(reachable_se3_top_ik)} merged={len(top_ik)}"
        )
    reachable_se3_closure_poses, reachable_se3_closure_meta = (
        generate_reachable_se3_interpolations(
            top_ik,
            adjacent_frontier_only=True,
            require_common_strict_arm=False,
            min_quat_angle_deg=0.01,
            max_quat_angle_deg=180.0,
        )
    )
    for pose in reachable_se3_closure_poses:
        pose["reachable_se3_generation"] = 2
    reachable_se3_closure_alignment_meta = attach_normal_alignment(
        reachable_se3_closure_poses,
        outward_normal,
    )
    (
        reachable_se3_closure_overlap,
        reachable_se3_closure_overlap_meta,
    ) = occupancy.counts_for_poses(
        reachable_se3_closure_poses,
        gripper_voxels,
    )
    (
        reachable_se3_closure_grasp,
        reachable_se3_closure_grasp_meta,
    ) = occupancy.counts_for_poses(
        reachable_se3_closure_poses,
        opening_voxels,
    )
    (
        reachable_se3_closure_inflated,
        reachable_se3_closure_inflated_meta,
    ) = occupancy.counts_for_poses(
        reachable_se3_closure_poses,
        inflated_voxels,
    )
    (
        reachable_se3_closure_ik_input,
        reachable_se3_closure_prefilter_meta,
    ) = rank_dedupe_top_ik_input(
        reachable_se3_closure_poses,
        reachable_se3_closure_overlap,
        grasp_counts=reachable_se3_closure_grasp,
        inflated_counts=reachable_se3_closure_inflated,
        limit=IK_INPUT_MAX,
        quality_quota=60,
        alignment_quota=60,
        normal_confidence=normal_confidence,
        weak_alignment_quota_cap=60,
        alignment_grasp_ratio_override=0.95,
    )
    reachable_se3_closure_top_ik: List[Dict[str, Any]] = []
    reachable_se3_closure_ik_meta: Dict[str, Any] = {
        "skipped": not bool(reachable_se3_closure_ik_input),
    }
    reachable_se3_closure_pool_meta: Dict[str, Any] = {}
    reachable_se3_closure_volume_meta: Dict[str, Any] = {}
    if reachable_se3_closure_ik_input:
        if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
            ctx.raise_if_cancelled("rgbd reachable SE3 closure")
        (
            reachable_se3_closure_ranked,
            reachable_se3_closure_ik_meta,
        ) = filter_poses_safe_final_dual_arm_ik(
            world,
            reachable_se3_closure_ik_input,
            plan_arm=plan_arm,
            ctx=ctx,
        )
        (
            reachable_se3_closure_top_ik,
            reachable_se3_closure_pool_meta,
        ) = select_strict_ik_pool(
            reachable_se3_closure_ranked,
            plan_arm=plan_arm,
            top_n=IK_TOP_N,
        )
        reachable_se3_closure_ik_meta.update(
            reachable_se3_closure_pool_meta
        )
        if reachable_se3_closure_top_ik:
            reachable_se3_closure_volume_meta = (
                attach_final_volume_metrics(
                    reachable_se3_closure_top_ik,
                    occupancy=occupancy,
                    gripper_voxels=gripper_voxels,
                    opening_voxels=opening_voxels,
                    inflated_voxels=inflated_voxels,
                )
            )
    for pose in reachable_se3_closure_top_ik:
        key = pose_dedupe_key(pose)
        if key in merged_keys:
            continue
        merged_keys.add(key)
        top_ik.append(pose)
    reachable_se3_closure_meta.update(
        {
            "normal_alignment": reachable_se3_closure_alignment_meta,
            "prefilter": reachable_se3_closure_prefilter_meta,
            "ik_filter": reachable_se3_closure_ik_meta,
            "strict_pool": reachable_se3_closure_pool_meta,
            "strict_reachable_count": int(
                len(reachable_se3_closure_top_ik)
            ),
            "merged_reachable_count": int(len(top_ik)),
        }
    )
    pareto_micro_meta["reachable_se3_closure"] = (
        reachable_se3_closure_meta
    )
    prefilter_meta["query"].update(
        {
            "reachable_se3_closure_original_overlap": (
                reachable_se3_closure_overlap_meta
            ),
            "reachable_se3_closure_grasp": (
                reachable_se3_closure_grasp_meta
            ),
            "reachable_se3_closure_inflated_overlap": (
                reachable_se3_closure_inflated_meta
            ),
        }
    )
    ik_meta["reachable_se3_closure"] = {
        "prefilter": reachable_se3_closure_prefilter_meta,
        "ik_filter": reachable_se3_closure_ik_meta,
        "strict_pool": reachable_se3_closure_pool_meta,
    }
    volume_meta["reachable_se3_closure_reachable_count"] = int(
        len(reachable_se3_closure_top_ik)
    )
    volume_meta["reachable_se3_closure_volume"] = (
        reachable_se3_closure_volume_meta
    )
    volume_meta["merged_reachable_count"] = int(len(top_ik))
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] reachable-SE3 closure endpoints="
            f"{reachable_se3_closure_meta['pareto_endpoint_count']} pairs="
            f"{reachable_se3_closure_meta['pair_count']} poses="
            f"{len(reachable_se3_closure_poses)} IK_input="
            f"{len(reachable_se3_closure_ik_input)} strict="
            f"{len(reachable_se3_closure_top_ik)} merged={len(top_ik)}"
        )
    reachable_se3_closure2_stage = execute_reachable_se3_stage(
        world=world,
        source_poses=top_ik,
        occupancy=occupancy,
        gripper_voxels=gripper_voxels,
        opening_voxels=opening_voxels,
        inflated_voxels=inflated_voxels,
        outward_normal=outward_normal,
        plan_arm=plan_arm,
        normal_confidence=normal_confidence,
        generation=3,
        stage_name="rgbd reachable SE3 closure 2",
        generator_kwargs={
            "adjacent_frontier_only": True,
            "require_common_strict_arm": False,
            "min_quat_angle_deg": 0.01,
            "max_quat_angle_deg": 180.0,
            "min_grasp_ratio": 0.95,
        },
        ctx=ctx,
    )
    reachable_se3_closure2_poses = (
        reachable_se3_closure2_stage["poses"]
    )
    reachable_se3_closure2_top_ik = (
        reachable_se3_closure2_stage["top_ik"]
    )
    reachable_se3_closure2_meta = (
        reachable_se3_closure2_stage["meta"]
    )
    for pose in reachable_se3_closure2_top_ik:
        key = pose_dedupe_key(pose)
        if key in merged_keys:
            continue
        merged_keys.add(key)
        top_ik.append(pose)
    reachable_se3_closure2_meta["merged_reachable_count"] = int(
        len(top_ik)
    )
    pareto_micro_meta["reachable_se3_closure2"] = (
        reachable_se3_closure2_meta
    )
    prefilter_meta["query"].update(
        {
            "reachable_se3_closure2_original_overlap": (
                reachable_se3_closure2_stage["query"][
                    "original_overlap"
                ]
            ),
            "reachable_se3_closure2_grasp": (
                reachable_se3_closure2_stage["query"]["grasp"]
            ),
            "reachable_se3_closure2_inflated_overlap": (
                reachable_se3_closure2_stage["query"][
                    "inflated_overlap"
                ]
            ),
        }
    )
    ik_meta["reachable_se3_closure2"] = {
        "prefilter": reachable_se3_closure2_meta["prefilter"],
        "ik_filter": reachable_se3_closure2_meta["ik_filter"],
        "strict_pool": reachable_se3_closure2_meta["strict_pool"],
    }
    volume_meta["reachable_se3_closure2_reachable_count"] = int(
        len(reachable_se3_closure2_top_ik)
    )
    volume_meta["reachable_se3_closure2_volume"] = (
        reachable_se3_closure2_stage["volume"]
    )
    volume_meta["merged_reachable_count"] = int(len(top_ik))
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] reachable-SE3 closure2 "
            f"endpoints="
            f"{reachable_se3_closure2_meta['pareto_endpoint_count']} "
            f"pairs={reachable_se3_closure2_meta['pair_count']} poses="
            f"{len(reachable_se3_closure2_poses)} IK_input="
            f"{len(reachable_se3_closure2_stage['ik_input'])} strict="
            f"{len(reachable_se3_closure2_top_ik)} merged={len(top_ik)}"
        )
    (
        reachable_translation_poses,
        reachable_translation_seed_meta,
    ) = generate_reachable_translation_refinements(top_ik)
    reachable_translation_stage = (
        execute_generated_reachable_pose_stage(
            world=world,
            poses=reachable_translation_poses,
            stage_meta=reachable_translation_seed_meta,
            occupancy=occupancy,
            gripper_voxels=gripper_voxels,
            opening_voxels=opening_voxels,
            inflated_voxels=inflated_voxels,
            outward_normal=outward_normal,
            plan_arm=plan_arm,
            normal_confidence=normal_confidence,
            stage_name="rgbd reachable translation refinement",
            ctx=ctx,
        )
    )
    reachable_translation_top_ik = reachable_translation_stage["top_ik"]
    reachable_translation_meta = reachable_translation_stage["meta"]
    for pose in reachable_translation_top_ik:
        key = pose_dedupe_key(pose)
        if key in merged_keys:
            continue
        merged_keys.add(key)
        top_ik.append(pose)
    reachable_translation_meta["merged_reachable_count"] = int(len(top_ik))
    pareto_micro_meta["reachable_translation_refinement"] = (
        reachable_translation_meta
    )
    prefilter_meta["query"].update(
        {
            "reachable_translation_original_overlap": (
                reachable_translation_stage["query"][
                    "original_overlap"
                ]
            ),
            "reachable_translation_grasp": (
                reachable_translation_stage["query"]["grasp"]
            ),
            "reachable_translation_inflated_overlap": (
                reachable_translation_stage["query"][
                    "inflated_overlap"
                ]
            ),
        }
    )
    ik_meta["reachable_translation_refinement"] = {
        "prefilter": reachable_translation_meta["prefilter"],
        "ik_filter": reachable_translation_meta["ik_filter"],
        "strict_pool": reachable_translation_meta["strict_pool"],
    }
    volume_meta["reachable_translation_reachable_count"] = int(
        len(reachable_translation_top_ik)
    )
    volume_meta["reachable_translation_volume"] = (
        reachable_translation_stage["volume"]
    )
    volume_meta["merged_reachable_count"] = int(len(top_ik))
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] reachable-translation seeds="
            f"{reachable_translation_meta['seed_count']} poses="
            f"{len(reachable_translation_poses)} IK_input="
            f"{len(reachable_translation_stage['ik_input'])} strict="
            f"{len(reachable_translation_top_ik)} merged={len(top_ik)}"
        )
    reachable_safe_grasp_max_cm3 = overlap_safe_grasp_max_cm3(top_ik)
    effective_grasp_quality_ratio = (
        NORMAL_ADAPTIVE_GRASP_QUALITY_RATIO
        if normal_confidence in ("strong", "weak")
        else ADAPTIVE_GRASP_QUALITY_RATIO
    )
    effective_grasp_absolute_min_cm3 = (
        MIN_NORMAL_GRASP_VOL_CM3
        if normal_confidence in ("strong", "weak")
        else MIN_ADAPTIVE_GRASP_VOL_CM3
    )
    effective_grasp_threshold_cm3 = adaptive_grasp_volume_floor(
        reachable_safe_grasp_max_cm3,
        quality_ratio=effective_grasp_quality_ratio,
        absolute_min_cm3=effective_grasp_absolute_min_cm3,
    )
    ranked_final_poses: List[Dict[str, Any]] = []
    best_pose, threshold_audit = select_final_pose(
        top_ik,
        original_threshold_cm3=ORIGINAL_OVERLAP_MAX_CM3,
        inflated_threshold_cm3=INFLATED_OVERLAP_MAX_CM3,
        grasp_threshold_cm3=effective_grasp_threshold_cm3,
        normal_confidence=normal_confidence,
        ranked_out=ranked_final_poses,
    )
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] final overlap original"
            f"<={ORIGINAL_OVERLAP_MAX_CM3:.3f}cm3 pass="
            f"{threshold_audit['original_passing_count']}/"
            f"{threshold_audit['input_count']}; inflated"
            f"<{INFLATED_OVERLAP_MAX_CM3:.3f}cm3 overlap_pass="
            f"{threshold_audit['overlap_passing_count']}/"
            f"{threshold_audit['input_count']}; grasp"
            f">={effective_grasp_threshold_cm3:.3f}cm3 "
            f"(cap={MIN_GRASP_VOL_CM3:.3f}, "
            f"reachable_safe_max={reachable_safe_grasp_max_cm3:.3f}, "
            f"global_safe_max="
            f"{prefilter_meta['strict_safe_grasp_max_cm3']:.3f}) "
            f"final_pass="
            f"{threshold_audit['passing_count']}/{threshold_audit['input_count']} "
            f"overlap_pass_grasp_max="
            f"{threshold_audit['overlap_passing_grasp_max_cm3']:.3f}cm3 "
            f"inflated_min/median/max={threshold_audit['min_cm3']:.3f}/"
            f"{threshold_audit['median_cm3']:.3f}/"
            f"{threshold_audit['max_cm3']:.3f}cm3 "
            f"normal={normal_confidence}/"
            f"{threshold_audit['selected_normal_alignment_deg']}deg "
            f"quality_floor={threshold_audit['alignment_quality_floor_cm3']} "
            f"wrist_limit tier={threshold_audit['selected_wrist_limit_tier']}/"
            f"margin={threshold_audit['selected_wrist_limit_margin_deg']}deg "
            f"tier_counts={threshold_audit['wrist_limit_tier_counts']}"
        )
    if best_pose is None:
        raise GraspObjPlanningError(
            "RGBD strict IK top poses all fail final overlap filters "
            f"(original<={ORIGINAL_OVERLAP_MAX_CM3:.3f}cm3, "
            f"inflated<{INFLATED_OVERLAP_MAX_CM3:.3f}cm3, "
            f"grasp>={effective_grasp_threshold_cm3:.3f}cm3 "
            f"(cap={MIN_GRASP_VOL_CM3:.3f}, "
            f"reachable_safe_max={reachable_safe_grasp_max_cm3:.3f}, "
            f"global_safe_max="
            f"{prefilter_meta['strict_safe_grasp_max_cm3']:.3f}); "
            f"original_pass={threshold_audit['original_passing_count']}, "
            f"overlap_pass={threshold_audit['overlap_passing_count']}, "
            f"overlap_pass_grasp_max="
            f"{threshold_audit['overlap_passing_grasp_max_cm3']:.3f}cm3, "
            f"final_pass={threshold_audit['passing_count']})"
        )

    selected_arm = _best_arm_for_pose(
        best_pose,
        pool_meta.get("recommended_arm"),
    )
    planned_safe = (
        best_pose.get("planned_safe_by_arm") or {}
    ).get(selected_arm)
    if not isinstance(planned_safe, dict) or not planned_safe.get("ok"):
        raise GraspObjPlanningError(
            "RGBD internal error: selected final pose has no same-arm "
            "safe+final IK pair"
        )
    best_pose["planned_safe"] = dict(planned_safe)
    selected_quality_rank = next(
        (
            rank
            for rank, pose in enumerate(ranked_final_poses)
            if pose is best_pose
        ),
        0,
    )
    safe_execution_audit = {
        "ok": True,
        "mode": "paired_safe_final_ik_filter",
        "requested_back_m": float(SAFE_PLAN_BACK_M),
        "ranked_candidate_count": int(len(ranked_final_poses)),
        "selected_quality_rank": int(selected_quality_rank),
        "selected_arm": selected_arm,
        "selected": dict(planned_safe),
        "post_selection_probe_count": 0,
    }
    threshold_audit["pre_safe_selected_grasp_vol_cm3"] = float(
        best_pose.get("grasp_vol_cm3", 0.0)
    )
    threshold_audit["pre_safe_selected_normal_alignment_deg"] = (
        float(best_pose["normal_alignment_deg"])
        if best_pose.get("normal_alignment_deg") is not None
        else None
    )
    threshold_audit["safe_selected_quality_rank"] = int(
        selected_quality_rank
    )
    threshold_audit["selected_normal_alignment_deg"] = (
        float(best_pose["normal_alignment_deg"])
        if best_pose.get("normal_alignment_deg") is not None
        else None
    )

    selected_pose_ik = _pose_ik_hard_constraint_summary(best_pose)
    selected_pose_ik_q = _selected_pose_ik_q_map(selected_pose_ik)
    eef_pos = np.asarray(best_pose["eef_pos"], dtype=np.float64)
    eef_quat = np.asarray(best_pose["quat"], dtype=np.float64)
    approach = np.asarray(best_pose["R"], dtype=np.float64)[:, 2]
    next_world = eef_pos + np.asarray([0.0, 0.0, GRASP_LIFT_M], dtype=np.float64)
    contact_pixel = _world_to_pixel(
        cam_pos,
        cam_quat,
        np.asarray(best_pose["anchor"], dtype=np.float64),
        w,
        h,
        fl,
        ha,
    ) or (int(u), int(v))

    grip_fit = {
        "build": BUILD,
        "volume_source": "rgbd_v8_dense_occupancy",
        "grasp_vol_cm3": float(best_pose["grasp_vol_cm3"]),
        "overlap_vol_cm3": float(best_pose["overlap_vol_cm3"]),
        "inflated_overlap_vol_cm3": float(
            best_pose["inflated_overlap_vol_cm3"]
        ),
        "overlap_frac": float(best_pose["overlap_frac"]),
        "gripper_vol_cm3": float(best_pose["gripper_vol_cm3"]),
        "gap_voxel_n": int(best_pose["grasp_vox"]),
        "grasp_vox": int(best_pose["grasp_vox"]),
        "overlap_vox": int(best_pose["overlap_vox"]),
        "inflated_overlap_vox": int(best_pose["inflated_overlap_vox"]),
        "original_overlap_threshold_cm3": float(ORIGINAL_OVERLAP_MAX_CM3),
        "inflated_overlap_threshold_cm3": float(INFLATED_OVERLAP_MAX_CM3),
        "grasp_vol_threshold_cm3": float(effective_grasp_threshold_cm3),
        "grasp_vol_threshold_cap_cm3": float(MIN_GRASP_VOL_CM3),
        "reachable_safe_grasp_max_cm3": float(
            reachable_safe_grasp_max_cm3
        ),
        "global_safe_grasp_max_cm3": float(
            prefilter_meta["strict_safe_grasp_max_cm3"]
        ),
        "adaptive_grasp_quality_ratio": float(effective_grasp_quality_ratio),
        "adaptive_grasp_absolute_min_cm3": float(
            effective_grasp_absolute_min_cm3
        ),
        "opening_voxel_total": int(best_pose["opening_voxel_total"]),
        "anchor_marker_pt": np.asarray(best_pose["anchor"]).tolist(),
        "anchor_dist_mm": float(
            np.linalg.norm(np.asarray(best_pose["anchor"]) - hit) * 1000.0
        ),
        "axial_point": best_pose["axial_point"],
        "axial_z_mm": float(best_pose["axial_z_m"] * 1000.0),
        "outward_normal_world": (
            np.asarray(outward_normal, dtype=np.float64).tolist()
            if outward_normal is not None
            else None
        ),
        "gripper_vector_world": pose_gripper_vector_world(best_pose).tolist(),
        "normal_alignment_deg": (
            float(best_pose["normal_alignment_deg"])
            if best_pose.get("normal_alignment_deg") is not None
            else None
        ),
        "normal_confidence": normal_confidence,
        "wrist_limit_tier": int(best_pose.get("wrist_limit_tier", 0)),
        "wrist_limit_margin_deg": (
            round(
                math.degrees(float(best_pose["wrist_limit_margin_rad"])),
                3,
            )
            if best_pose.get("wrist_limit_margin_rad") is not None
            else None
        ),
        "wrist_limit_critical_deg": float(
            math.degrees(WRIST_LIMIT_CRITICAL_RAD)
        ),
        "wrist_limit_warn_deg": float(math.degrees(WRIST_LIMIT_WARN_RAD)),
        "wrist_limit_joints": ["J5", "J6", "J7"],
        "inflation_geometry": inflation_meta,
    }
    plan_audit = {
        "build": BUILD,
        "allowed_observations": [
            "frozen_head_rgb",
            "frozen_head_metric_depth",
            "frozen_head_camera_intrinsics",
            "frozen_head_camera_extrinsics",
            "robot_kinematics_for_ik_fk",
        ],
        "forbidden_observations_used": [],
        "used_segmentation": False,
        "used_object_identity": False,
        "used_usd_scene_mesh": False,
        "used_collision_mesh": False,
        "hit": hit_audit,
        "surface_normal": normal_audit,
        "reconstruction": reconstruction_meta,
        "anchors": {
            **anchor_meta,
            "points": anchors.round(8).tolist(),
        },
        "sampling": {
            "anchor_count": int(len(anchors)),
            "approach_count": 20,
            "roll_count": int(N_ROLL),
            "axial_point_count": int(len(AXIAL_Z_M)),
            "poses_per_anchor": int(expected_per_anchor),
            "raw_pose_count": int(len(poses)),
            "refined_pose_count": int(len(refined_poses)),
            "contact_pose_count": int(len(contact_poses)),
            "pre_ik_translation_pose_count": int(
                len(pre_ik_translation_poses)
            ),
            "interpolated_pose_count": int(len(interpolated_poses)),
            "pareto_micro_pose_count": int(len(pareto_micro_poses)),
            "reachability_bridge_pose_count": int(
                len(reachability_bridge_poses)
            ),
            "reachable_se3_pose_count": int(len(reachable_se3_poses)),
            "reachable_se3_closure_pose_count": int(
                len(reachable_se3_closure_poses)
            ),
            "reachable_se3_closure2_pose_count": int(
                len(reachable_se3_closure2_poses)
            ),
            "reachable_translation_pose_count": int(
                len(reachable_translation_poses)
            ),
            "prefilter_pose_count": int(len(prefilter_poses)),
            "axial_z_mm": (AXIAL_Z_M * 1000.0).round(3).tolist(),
            "refinement": refinement_meta,
            "contact_refinement": contact_meta,
            "pre_ik_translation_refinement": pre_ik_translation_meta,
            "interpolation": interpolation_meta,
            "pareto_micro_refinement": pareto_micro_meta,
        },
        "camera_face": camera_face_meta,
        "normal_alignment": {
            "confidence": normal_confidence,
            "raw": raw_alignment_meta,
            "refined": refined_alignment_meta,
            "contact": contact_alignment_meta,
            "pre_ik_translation": pre_ik_translation_alignment_meta,
            "interpolated": interpolated_alignment_meta,
            "pareto_micro": pareto_micro_alignment_meta,
            "reachability_bridge": reachability_bridge_alignment_meta,
            "reachable_se3": reachable_se3_alignment_meta,
            "reachable_se3_closure": (
                reachable_se3_closure_alignment_meta
            ),
            "reachable_se3_closure2": (
                reachable_se3_closure2_stage["alignment"]
            ),
            "reachable_translation": (
                reachable_translation_stage["alignment"]
            ),
        },
        "occupancy": occupancy.metadata,
        "inflation_geometry": inflation_meta,
        "prefilter": prefilter_meta,
        "ik_filter": ik_meta,
        "volume": volume_meta,
        "inflated_overlap_filter": threshold_audit,
        "safe_execution_filter": safe_execution_audit,
        "pick": {
            "pi": int(best_pose["pi"]),
            "ni": int(best_pose["ni"]),
            "ri": int(best_pose["ri"]),
            "ai": int(best_pose["ai"]),
            "axial_point": best_pose["axial_point"],
            "refined": bool(best_pose.get("refined", False)),
            "interpolated": bool(best_pose.get("interpolated", False)),
            "micro_refined": bool(best_pose.get("micro_refined", False)),
            "reachability_bridge": bool(
                best_pose.get("reachability_bridge", False)
            ),
            "reachable_se3_interpolated": bool(
                best_pose.get("reachable_se3_interpolated", False)
            ),
            "reachable_se3_generation": int(
                best_pose.get("reachable_se3_generation", 0)
            ),
            "translation_refined": bool(
                best_pose.get("translation_refined", False)
            ),
            "translation_refine_generation": int(
                best_pose.get("translation_refine_generation", 0)
            ),
            "translation_offset_local_m": (
                [
                    float(value)
                    for value in best_pose["translation_offset_local_m"]
                ]
                if best_pose.get("translation_offset_local_m") is not None
                else None
            ),
            "interpolation_t": (
                float(best_pose["interpolation_t"])
                if best_pose.get("interpolation_t") is not None
                else None
            ),
            "reachable_se3_position_t": (
                float(best_pose["reachable_se3_position_t"])
                if best_pose.get("reachable_se3_position_t") is not None
                else None
            ),
            "reachable_se3_orientation_t": (
                float(best_pose["reachable_se3_orientation_t"])
                if best_pose.get("reachable_se3_orientation_t") is not None
                else None
            ),
            "interpolation_parent_angle_deg": (
                float(best_pose["interpolation_parent_angle_deg"])
                if best_pose.get("interpolation_parent_angle_deg") is not None
                else None
            ),
            "refine_tilt_x_deg": float(
                best_pose.get("refine_tilt_x_deg", 0.0)
            ),
            "refine_tilt_y_deg": float(
                best_pose.get("refine_tilt_y_deg", 0.0)
            ),
            "refine_roll_deg": float(best_pose.get("refine_roll_deg", 0.0)),
            "micro_tilt_x_deg": float(
                best_pose.get("micro_tilt_x_deg", 0.0)
            ),
            "micro_tilt_y_deg": float(
                best_pose.get("micro_tilt_y_deg", 0.0)
            ),
            "micro_roll_deg": float(
                best_pose.get("micro_roll_deg", 0.0)
            ),
            "outward_normal_world": (
                np.asarray(outward_normal, dtype=np.float64).tolist()
                if outward_normal is not None
                else None
            ),
            "gripper_vector_world": pose_gripper_vector_world(best_pose).tolist(),
            "normal_alignment_deg": (
                float(best_pose["normal_alignment_deg"])
                if best_pose.get("normal_alignment_deg") is not None
                else None
            ),
            "normal_confidence": normal_confidence,
            "selected_pose_ik": selected_pose_ik,
            "selected_pose_ik_q": selected_pose_ik_q,
            "planned_safe": best_pose.get("planned_safe"),
        },
        "recommended_arm": selected_arm,
    }

    debug_images: Dict[str, str] = {}
    if render_debug:
        import cv2
        import shutil

        rgb_path = str(
            session.get("rgb_path")
            or os.path.join(str(session["init_dir"]), "rgb.png")
        )
        overlay_path = os.path.abspath(
            os.path.join(
                str(session["init_dir"]),
                "..",
                "plan_eef_grasp_point_filter_rgbd_overlay.png",
            )
        )
        if os.path.isfile(rgb_path):
            os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
            shutil.copy2(rgb_path, overlay_path)
            overlay = cv2.imread(overlay_path)
            if overlay is not None:
                cv2.circle(overlay, (int(u), int(v)), 7, (0, 0, 255), 2)
                cv2.circle(
                    overlay,
                    (int(contact_pixel[0]), int(contact_pixel[1])),
                    6,
                    (0, 255, 0),
                    2,
                )
                _draw_local_gripper_overlay(
                    overlay,
                    eef_pos=eef_pos,
                    eef_quat=eef_quat,
                    cam_pos=cam_pos,
                    cam_quat=cam_quat,
                    w=int(w),
                    h=int(h),
                    fl=float(fl),
                    ha=float(ha),
                    color=(230, 48, 48),
                )
                cv2.imwrite(overlay_path, overlay)
                debug_images["frozen_rgb_overlay"] = overlay_path
        three_view_path = os.path.join(
            str(session["init_dir"]),
            "..",
            "plan_eef_grasp_point_filter_rgbd_three_views.png",
        )
        rendered = render_local_three_views(
            mesh,
            hit=hit,
            anchors=anchors,
            best_pose=best_pose,
            gripper_voxels=gripper_voxels,
            inflation_regions=inflation_regions,
            threshold_audit=threshold_audit,
            outward_normal_world=outward_normal,
            output_path=os.path.abspath(three_view_path),
        )
        if rendered:
            debug_images["rgbd_mesh_three_views"] = rendered

    next_delta = (next_world - eef_pos).tolist()
    candidate = {
        "id": 0,
        "target": "grasp",
        "arm": selected_arm,
        "label": "plan_eef_grasp_point_filter_rgbd",
        "eef_target": {
            "pos": eef_pos.tolist(),
            "quat": eef_quat.tolist(),
            "approach": approach.tolist(),
            "gripper_cmd": -1.0,
        },
        "next_eef_move": next_delta,
        "reachable": True,
        "reach_reason": "rgbd_dual_arm_gpu_ik_local_fk",
        "score": float(best_pose["grasp_vol_cm3"]),
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": selected_pose_ik_q,
        "ik_solution": selected_pose_ik.get("solution"),
        "meta": {
            "mode": "grasp_point_filter_rgbd",
            "exec_sequence": "move_then_close",
            "next_eef_move_world": next_world.tolist(),
            "hit_method": hit_audit["method"],
            "pixel": {"u": int(contact_pixel[0]), "v": int(contact_pixel[1])},
            "vlm_pixel": {"u": int(u), "v": int(v)},
            "recommended_arm": selected_arm,
            "selected_pose_ik": selected_pose_ik,
            "selected_pose_ik_q": selected_pose_ik_q,
            "planned_safe": best_pose.get("planned_safe"),
            "grip_fit": grip_fit,
            "plan_audit": plan_audit,
        },
    }
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] best={best_pose['axial_point']} "
            f"eef={eef_pos.round(4).tolist()} arm={selected_arm} "
            f"grasp={best_pose['grasp_vol_cm3']:.3f}cm3 "
            f"overlap={best_pose['overlap_vol_cm3']:.3f}cm3 "
            f"inflated={best_pose['inflated_overlap_vol_cm3']:.3f}cm3 "
            f"normal={normal_confidence}/"
            f"{best_pose.get('normal_alignment_deg')}deg"
        )
    return {
        "ok": True,
        "step": "plan",
        "mode": "grasp_point_filter_rgbd",
        "exec_sequence": "move_then_close",
        "target": "grasp",
        "session_id": session["session_id"],
        "object_name": "",
        "view": "head",
        "pixel": {"u": int(contact_pixel[0]), "v": int(contact_pixel[1])},
        "vlm_pixel": {"u": int(u), "v": int(v)},
        "hit_world": hit.tolist(),
        "hit_method": hit_audit["method"],
        "eef_pose": {
            "pos": eef_pos.tolist(),
            "quat": eef_quat.tolist(),
            "approach": approach.tolist(),
        },
        "next_eef_move": next_world.tolist(),
        "next_eef_move_delta": next_delta,
        "render_image": debug_images.get("grasp_move_vis"),
        "render_image_3d": debug_images.get("rgbd_mesh_three_views"),
        "debug_images": debug_images,
        "image": debug_images.get("image"),
        "point_on_image": debug_images.get("point_on_image"),
        "3d_point_render": debug_images.get("rgbd_mesh_three_views"),
        "grasp_move_vis": debug_images.get("grasp_move_vis"),
        "candidates": [candidate],
        "object": {"input": ""},
        "arm": selected_arm,
        "recommended_arm": selected_arm,
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": selected_pose_ik_q,
        "ik_solution": selected_pose_ik.get("solution"),
        "plan_audit": plan_audit,
        "grip_fit": grip_fit,
        "gap_center": np.asarray(best_pose["anchor"]).tolist(),
        "grasp_vol_cm3": float(best_pose["grasp_vol_cm3"]),
        "overlap_vol_cm3": float(best_pose["overlap_vol_cm3"]),
        "inflated_overlap_vol_cm3": float(
            best_pose["inflated_overlap_vol_cm3"]
        ),
        "overlap_frac": float(best_pose["overlap_frac"]),
        "candidate_count": int(len(prefilter_poses)),
    }
