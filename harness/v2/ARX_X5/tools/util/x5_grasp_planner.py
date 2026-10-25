"""RoboDojo X5 implementation of official_v2 RGB-D lite grasp planning.

The planner consumes one immutable evaluator RGB-D capture, capture-time
proprioception, and the policy-local official X5 model. It never imports the
simulator, segmentation, task objects, collision truth, or evaluator IK.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np

from tools.util.geometry import relative_uv_to_pixel, unproject_relative_uv
from tools.util.storage import Snapshot
from tools.util.x5_kinematics import (
    calibrated_pose_for_joints,
    matrix_to_quaternion,
    quaternion_to_matrix,
    solve_calibrated_x5_ik,
)
from tools.util.x5_grasp_geometry import (
    VOXEL_M,
    icosa_face_normals,
    x5_eef_frame_from_approach_roll,
    x5_gripper_geometry,
    x5_gripper_joint_position_m,
)
from tools.util.rgbd_scene_mesh_robodojo_v1 import (
    MESH_VERSION as RGBD_MESH_VERSION,
    MeshSceneOccupancy,
    RGBDSceneMeshResult,
)
from tools.util.rgbd_mesh_revision import reconstruct_scene_mesh_from_snapshot
from tools.util.x5_exact_opening import (
    OPENING_QUERY_VERSION,
    exact_x5_planning_geometry,
    points_inside_exact_x5_opening,
)
from tools.util.x5_self_collision import check_paired_path
from tools.util.x5_gripper_observation import (
    exclude_observed_gripper_material,
    points_on_withheld_rays,
)
from tools.util.x5_rgbd_path_guard import FrozenRGBDPathGuard


# Production plans are deliberately limited to a single near-vertical cone.
# Wider cones remain available only to the isolated offline audit helper, which
# passes an explicit max_downward_tilt_deg to the private kernel.
# v31 tightens the public contract after the old v26 runtime was observed
# widening to 15/30 degrees. The public route has one immutable cone; wider
# experiments remain available only through the read-only audit helper.
# v32 makes the public route a single five-degree cone.  The old v26 plans
# accepted 15/30-degree poses, and v31's 3-degree cap was still wider than
# the requested table-normal grasp contract.  Keep the numerical FK allowance
# below the cone rather than silently reopening the old fallback tiers.
# v33 relaxes to 5 degrees per user confirmation that valid solutions exist
# within this range; volume/overlap/IK thresholds remain unchanged.
# v34 keeps that cone. Centered samples were missing in-opening shifts, and
# the extra 3 mm collision-query halo was rejecting poses the 15 mm inflated
# finger shell already accepts.
# v35 raises the uninflated gripper overlap cap from 0.15 cm3 to 0.30 cm3.
BUILD = "robodojo_x5_rgbd_lite_strict_vertical_v36"
MAX_DOWNWARD_TILT_DEG = 5.0
DOWNWARD_CONE_TIERS_DEG = (MAX_DOWNWARD_TILT_DEG,)
# Capture-time FK can differ slightly from the ideal candidate.
DOWNWARD_FK_TILT_TOL_DEG = 0.25
# Private, unsigned cone audits may request the same upper range as the
# existing read-only experiments. The public wrapper never forwards these
# values and remains bounded by DOWNWARD_CONE_TIERS_DEG.
OFFLINE_MAX_DOWNWARD_TILT_DEG = 75.0
VERTICAL_TABLE_CLEARANCE_M = 0.0015
VERTICAL_MIN_OPENING_VOXELS = 12
VERTICAL_MIN_SURFACE_SPAN_M = 0.006
RAW_SURFACE_MIN_POINTS = 3
RAW_SURFACE_TRACKING_MARGIN_M = 0.003
VERTICAL_ANCHOR_TRUST_M = 0.025
MESH_ROUTE_ENV = "ROBODOJO_INTERFACE_RGBD_MESH_VERSION"
DEFAULT_MESH_ROUTE = "robodojo_v1"
N_ROLL = 8
CLOSURE_AXIS_MAX_DEG = 180.0 / N_ROLL
IK_INPUT_MAX = 120
IK_QUALITY_QUOTA = 90
IK_ALIGNMENT_QUOTA = 20
ORIGINAL_OVERLAP_MAX_CM3 = 0.30
INFLATED_OVERLAP_MAX_CM3 = 1.6
MIN_GRASP_VOL_CM3 = 5.0
MIN_ADAPTIVE_GRASP_VOL_CM3 = 1.0
MIN_NORMAL_GRASP_VOL_CM3 = 2.7
ADAPTIVE_GRASP_QUALITY_RATIO = 0.85
NORMAL_ADAPTIVE_GRASP_QUALITY_RATIO = 0.75
SAFE_PLAN_BACK_M = 0.10
SAFE_FINAL_BRANCH_GAP_RAD = 0.85
# IK tolerance is a numerical early-stop condition, not the grasp-safety
# contract. Every accepted final_q is additionally FK-scored against the
# frozen RGB-D occupancy below. The tighter tolerance avoids throwing away a
# cheap final DLS iteration when a target can reach sub-voxel accuracy.
FINAL_IK_POS_TOL_M = 0.003
FINAL_IK_ORI_TOL_DEG = 1.0
SAFE_IK_POS_TOL_M = 0.003
SAFE_IK_ORI_TOL_DEG = 1.0
IK_REFINEMENT_POS_TOL_M = 0.0005
IK_REFINEMENT_ORI_TOL_DEG = 0.15
BACK_EXTRUSION_M = 0.060
SURFACE_ANCHOR_DIAMETER_M = 0.030
SURFACE_ANCHOR_RADIUS_M = SURFACE_ANCHOR_DIAMETER_M * 0.5
# The organized shell used by BEHAVIOR deliberately includes the support
# surface.  For a thin garment that makes the table look like graspable
# volume, so the lite adapter estimates the dominant support height from the
# same frozen depth and uses it only for the opening (the solid collision
# metric still includes the support surface).
SUPPORT_PLANE_BIN_M = 0.001
SUPPORT_PLANE_MIN_Z_M = 0.55
SUPPORT_PLANE_MAX_Z_M = 1.20
OPENING_SUPPORT_MARGIN_M = 0.003
OPENING_TARGET_RADIUS_M = 0.110

# A local edge is useful when a click lands on a shirt hem/sleeve.  The
# multiscale PCA normal remains the first choice; this fallback is a generic
# depth discontinuity detector and does not classify the object.
EDGE_RAY_COUNT = 16
EDGE_SAMPLE_RADII_PX = (8, 12, 16, 20)
EDGE_TANGENTIAL_HALF_PX = 3
EDGE_MIN_CONTRAST_M = 0.004
EDGE_MIN_SIDE_SUPPORT = 5
EDGE_MIN_SUPPORT_PLANE_DELTA_M = 0.002
EDGE_APPROACH_PITCHES = (
    -0.60,
    -0.50,
    -0.40,
    -0.30,
    -0.20,
    -0.10,
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
)
EDGE_ANCHOR_INSET_M = 0.006
TABLE_CLEARANCE_M = 0.010
AXIAL_SPAN_MIN_M = 0.015
EDGE_MAX_TABLE_LIFT_M = 0.060
EDGE_MIN_OPENING_VOXELS = 24
JOINT_LIMIT_MARGIN_RAD = 0.18
X5_SHOULDERS = {
    "left": np.array([-0.30, -0.45, 0.765], dtype=np.float64),
    "right": np.array([0.30, -0.45, 0.765], dtype=np.float64),
}


class X5GraspPlanningError(RuntimeError):
    """Planning failed without producing an executable candidate."""

    def __init__(self, message: str, *, audit: dict[str, Any] | None = None):
        super().__init__(message)
        self.audit = audit or {}


class _CandidatePathGate:
    """Lazy, one-capture non-gripper path gate after paired IK/FK validation.

    The legacy route deliberately retains its previous behavior. Otherwise a
    missing or invalid guard is a terminal geometry error, never an invitation
    to widen the downward cone or silently use the old unguarded planner.
    """

    def __init__(self, snapshot, *, enabled):
        self.snapshot = snapshot
        self.guard = None
        self.audit = {
            "enabled": bool(enabled),
            "applied": False,
            "reason": (
                "awaiting_paired_ik_fk_valid_candidate"
                if enabled
                else "explicit_legacy_or_mesh_fallback_route"
            ),
            "checked_arm_candidates": 0,
            "passed_arm_candidates": 0,
            "rejected_arm_candidates": 0,
            "rejection_counts_by_body": {},
            "rejection_examples": [],
            "scene": None,
            "setup_s": 0.0,
            "query_s": 0.0,
            "path_scope": "current_to_safe_to_final",
            "gripper_volume_and_self_collision_gates_preserved": True,
            "geometry_error_is_terminal": True,
            "physical_grasp_success_proven": False,
        }

    def validate(self, arm, info):
        if not self.audit["enabled"] or not bool(info.get("ok")):
            return
        try:
            if not info.get("volume_validation", {}).get("ok"):
                raise ValueError(
                    "path gate requires prior paired IK/FK volume acceptance"
                )
            if self.guard is None:
                started = time.perf_counter()
                self.guard = FrozenRGBDPathGuard(self.snapshot)
                self.audit.update(
                    scene=self.guard.audit, setup_s=time.perf_counter() - started
                )
            started = time.perf_counter()
            result = self.guard.check(arm, info["safe_q"], info["final_q"])
            self.audit["query_s"] += time.perf_counter() - started
            if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                raise ValueError(
                    "path guard did not return an explicit boolean verdict"
                )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            reason = f"{type(error).__name__}: {error}"
            self.audit.update(reason="uncertain_path_geometry", error=reason)
            raise X5GraspPlanningError(
                f"geometry=uncertain_path_geometry; {reason}",
                audit={
                    "build": BUILD,
                    "geometry_failure_terminal": True,
                    "non_gripper_path_gate": self.audit,
                },
            ) from error
        self.audit.update(applied=True, reason="candidate_paths_checked")
        self.audit["checked_arm_candidates"] += 1
        info["rgbd_non_gripper_path_validation"] = result
        if result["ok"]:
            self.audit["passed_arm_candidates"] += 1
            return
        info.update(ok=False, reason="rgbd_non_gripper_path_envelope")
        self.audit["rejected_arm_candidates"] += 1
        body = str(result.get("body", "unspecified"))
        counts = self.audit["rejection_counts_by_body"]
        counts[body] = counts.get(body, 0) + 1
        if len(self.audit["rejection_examples"]) < 12:
            self.audit["rejection_examples"].append(result)


def _build_scene_occupancy(
    snapshot: Snapshot,
    *,
    support_z: float | None,
    target_anchor: np.ndarray,
) -> tuple[Any, RGBDSceneMeshResult | None, str, str | None]:
    """Build one reusable occupancy object for all fixed X5 candidates.

    ``v4`` is an explicit rollback route.  The new route is selected by
    ``ROBODOJO_INTERFACE_RGBD_MESH_VERSION=v1`` (the default) and remains
    policy-local: it consumes only the immutable snapshot RGB-D/calibration.
    ``v2_continuous`` is an explicit operator-side evaluation option, not a
    public tool argument. Failure retains the complete v1 geometry, never a
    partially voxelized result or a weakened collision/evidence gate.
    """
    requested = str(os.environ.get(MESH_ROUTE_ENV, DEFAULT_MESH_ROUTE)).strip().lower()
    if requested in {"v4", "legacy", "organized_shell", "robodojo_v4"}:
        return (
            DepthRayOccupancy(
                snapshot.depth,
                snapshot.intrinsic_matrix,
                snapshot.extrinsic_matrix,
                support_z=support_z,
                target_anchor=target_anchor,
                target_radius_m=OPENING_TARGET_RADIUS_M,
            ),
            None,
            "v4",
            None,
        )
    continuous_requested = requested in {"v2_continuous", "robodojo_v2_continuous"}
    if requested not in {"v1", "robodojo_v1", "latest"} and not continuous_requested:
        raise ValueError(
            f"unsupported {MESH_ROUTE_ENV}={requested!r}; use robodojo_v1, v4 or v2_continuous"
        )
    try:
        result = reconstruct_scene_mesh_from_snapshot(
            snapshot,
            pixel_stride=1,
            min_depth_m=0.03,
            max_depth_m=8.0,
            collision_extrusion_mm=12.0,
            analytic_support_plane=True,
        )
        effective_version = RGBD_MESH_VERSION
        fallback_reason = None
        if continuous_requested:
            # Full-view reconstruction happens before any click/robot material
            # partition. Surface scoring and all subsequent gates are intact.
            from tools.util.continuous_rgbd_voxelization import (
                continuous_collision_candidate,
            )

            try:
                candidate, continuous_audit = continuous_collision_candidate(
                    snapshot, result
                )
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                fallback_reason = f"continuous_collision_candidate_rejected: {type(error).__name__}: {error}"
                result.metadata.update(
                    requested_version="robodojo_v2_continuous",
                    effective_version=RGBD_MESH_VERSION,
                    fallback_reason=fallback_reason,
                    continuous_collision_evaluation={
                        "enabled": True,
                        "applied": False,
                        "reason": fallback_reason,
                        "fallback": "complete_original_v1_surface_and_collision_occupancy",
                    },
                )
            else:
                result = candidate
                effective_version = "robodojo_v2_continuous"
                result.metadata.update(
                    requested_version=effective_version,
                    effective_version=effective_version,
                    route="whole_view_observed_surface_continuous_collision_cell_cover",
                    fallback_reason=None,
                    continuous_collision_evaluation={
                        **continuous_audit,
                        "enabled": True,
                        "applied": True,
                        "operator_opt_in": True,
                        "default_route_changed": False,
                        "physical_grasp_success_proven": False,
                    },
                )
        occupancy = MeshSceneOccupancy(
            result,
            support_z=support_z,
            target_anchor=target_anchor,
            target_radius_m=OPENING_TARGET_RADIUS_M,
            support_margin_m=OPENING_SUPPORT_MARGIN_M,
            surface_query_radius_cells=0,
        )
        return occupancy, result, effective_version, fallback_reason
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        # Geometry failure is explicit and auditable; callers may select v4
        # deliberately, but a silent mixed-version result is unsafe.
        return (
            DepthRayOccupancy(
                snapshot.depth,
                snapshot.intrinsic_matrix,
                snapshot.extrinsic_matrix,
                support_z=support_z,
                target_anchor=target_anchor,
                target_radius_m=OPENING_TARGET_RADIUS_M,
            ),
            None,
            "v4_fallback",
            f"{type(error).__name__}: {error}",
        )


def _finite_depth(depth: Any) -> np.ndarray:
    result = np.asarray(depth, dtype=np.float64).squeeze()
    if result.ndim != 2:
        raise ValueError(f"frozen depth must be HxW, got {result.shape}")
    return result


def _backproject_pixels(
    u: np.ndarray,
    v: np.ndarray,
    distance: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> np.ndarray:
    optical = np.column_stack(
        (
            (u - float(intrinsic[0, 2])) * distance / float(intrinsic[0, 0]),
            (v - float(intrinsic[1, 2])) * distance / float(intrinsic[1, 1]),
            distance,
        )
    )
    points_usd = optical * np.array([1.0, -1.0, -1.0], dtype=np.float64)
    return points_usd @ extrinsic[:3, :3].T + extrinsic[:3, 3]


def estimate_support_plane_z(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[float | None, dict[str, Any]]:
    """Estimate the dominant horizontal support height from frozen RGB-D.

    RoboDojo's work surface is expressed in the evaluator world frame.  A
    quantized mode is intentionally used instead of a simulator/table handle:
    it remains valid for arbitrary scenes and is stable across cloth wrinkles
    and depth noise.  The estimate is a *filtering reference*, never an
    object or collision oracle.
    """
    metric = _finite_depth(depth)
    height, width = metric.shape
    stride = max(1, int(round(min(height, width) / 240.0)))
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    distances = metric[rows, columns]
    valid = np.isfinite(distances) & (distances > 0.02)
    audit: dict[str, Any] = {
        "source": "frozen_evaluator_depth_dominant_world_z_mode",
        "sample_stride_px": int(stride),
        "bin_m": SUPPORT_PLANE_BIN_M,
        "z_range_m": [SUPPORT_PLANE_MIN_Z_M, SUPPORT_PLANE_MAX_Z_M],
        "uses_segmentation": False,
        "uses_simulator_geometry": False,
    }
    if not np.any(valid):
        audit["reason"] = "no_valid_depth"
        return None, audit
    points = _backproject_pixels(
        columns[valid].astype(np.float64),
        rows[valid].astype(np.float64),
        distances[valid].astype(np.float64),
        np.asarray(intrinsic, dtype=np.float64),
        np.asarray(extrinsic, dtype=np.float64),
    )
    z_values = points[:, 2]
    z_values = z_values[
        np.isfinite(z_values)
        & (z_values >= SUPPORT_PLANE_MIN_Z_M)
        & (z_values <= SUPPORT_PLANE_MAX_Z_M)
    ]
    if len(z_values) < 32:
        audit.update(
            {"reason": "too_few_world_points", "point_count": int(len(z_values))}
        )
        return None, audit
    lower = math.floor(SUPPORT_PLANE_MIN_Z_M / SUPPORT_PLANE_BIN_M)
    upper = math.ceil(SUPPORT_PLANE_MAX_Z_M / SUPPORT_PLANE_BIN_M) + 1
    edges = np.arange(lower, upper + 1, dtype=np.float64) * SUPPORT_PLANE_BIN_M
    histogram, _ = np.histogram(z_values, bins=edges)
    if not np.any(histogram):
        audit["reason"] = "empty_z_histogram"
        return None, audit
    mode_index = int(np.argmax(histogram))
    mode_center = float((edges[mode_index] + edges[mode_index + 1]) * 0.5)
    in_mode = np.abs(z_values - mode_center) <= 1.5 * SUPPORT_PLANE_BIN_M
    support_z = float(np.median(z_values[in_mode])) if np.any(in_mode) else mode_center
    mode_count = int(np.count_nonzero(in_mode))
    plane_points = points[np.abs(points[:, 2] - support_z) <= 0.003]
    coefficients = np.array([0.0, 0.0, support_z])
    if len(plane_points) >= 32:
        design = np.column_stack((plane_points[:, :2], np.ones(len(plane_points))))
        coefficients, *_ = np.linalg.lstsq(design, plane_points[:, 2], rcond=None)
    support_normal = np.array([-coefficients[0], -coefficients[1], 1.0])
    support_normal /= np.linalg.norm(support_normal)
    audit.update(
        {
            "ok": True,
            "reason": "dominant_world_z_mode",
            "support_z_m": support_z,
            "normal_world": support_normal.tolist(),
            "plane_coefficients_z_from_xy": coefficients.tolist(),
            "mode_center_m": mode_center,
            "mode_count": mode_count,
            "point_count": int(len(z_values)),
            "mode_fraction": float(mode_count / max(1, len(z_values))),
            "mode_mad_m": float(np.median(np.abs(z_values[in_mode] - support_z)))
            if np.any(in_mode)
            else None,
        }
    )
    return support_z, audit


def estimate_depth_edge_frame(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    pixel_u: int,
    pixel_v: int,
    *,
    support_z: float | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Find a table-to-object direction around a click using RGB-D only.

    Each candidate image direction compares robust medians on opposite sides
    of the click.  The returned vector points from the lower side (normally
    the support) toward the higher visible surface, which is the direction in
    which an open finger corridor should enter a thin garment.  It is not a
    semantic edge detector and deliberately reports ``usable=false`` when no
    support/object height step is present.
    """
    metric = _finite_depth(depth)
    height, width = metric.shape
    u0, v0 = int(pixel_u), int(pixel_v)
    audit: dict[str, Any] = {
        "source": "organized_depth_opposite_side_edge_scan",
        "pixel": [u0, v0],
        "ray_count": EDGE_RAY_COUNT,
        "sample_radii_px": list(EDGE_SAMPLE_RADII_PX),
        "tangential_half_px": EDGE_TANGENTIAL_HALF_PX,
        "min_contrast_m": EDGE_MIN_CONTRAST_M,
        "support_z_m": None if support_z is None else float(support_z),
        "uses_segmentation": False,
        "uses_simulator_mesh": False,
        "rays": [],
    }
    if not (0 <= u0 < width and 0 <= v0 < height):
        audit["reason"] = "pixel_out_of_bounds"
        return None, audit
    records: list[dict[str, Any]] = []
    for ray_index in range(EDGE_RAY_COUNT):
        angle = 2.0 * math.pi * ray_index / EDGE_RAY_COUNT
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        tangent = np.array([-direction[1], direction[0]], dtype=np.float64)
        side_points: list[np.ndarray] = []
        side_depths: list[np.ndarray] = []
        side_uv: list[list[tuple[int, int]]] = []
        for sign in (-1.0, 1.0):
            uv_samples: list[tuple[int, int]] = []
            for radius in EDGE_SAMPLE_RADII_PX:
                for tangent_offset in range(
                    -EDGE_TANGENTIAL_HALF_PX,
                    EDGE_TANGENTIAL_HALF_PX + 1,
                ):
                    uv = np.array([u0, v0], dtype=np.float64)
                    uv += sign * float(radius) * direction
                    uv += float(tangent_offset) * tangent
                    uu = int(round(float(uv[0])))
                    vv = int(round(float(uv[1])))
                    if 0 <= uu < width and 0 <= vv < height:
                        uv_samples.append((uu, vv))
            if uv_samples:
                uv_array = np.asarray(uv_samples, dtype=np.int32)
                values = metric[uv_array[:, 1], uv_array[:, 0]]
                good = np.isfinite(values) & (values > 0.02)
                uv_array = uv_array[good]
                values = values[good]
            else:
                uv_array = np.zeros((0, 2), dtype=np.int32)
                values = np.zeros(0, dtype=np.float64)
            if len(values):
                points = _backproject_pixels(
                    uv_array[:, 0].astype(np.float64),
                    uv_array[:, 1].astype(np.float64),
                    values.astype(np.float64),
                    np.asarray(intrinsic, dtype=np.float64),
                    np.asarray(extrinsic, dtype=np.float64),
                )
            else:
                points = np.zeros((0, 3), dtype=np.float64)
            side_points.append(points)
            side_depths.append(values)
            side_uv.append([tuple(int(x) for x in pair) for pair in uv_array])
        if (
            len(side_points[0]) < EDGE_MIN_SIDE_SUPPORT
            or len(side_points[1]) < EDGE_MIN_SIDE_SUPPORT
        ):
            continue
        z_minus = side_points[0][:, 2]
        z_plus = side_points[1][:, 2]
        median_minus = float(np.median(z_minus))
        median_plus = float(np.median(z_plus))
        contrast = abs(median_plus - median_minus)
        spread = float(
            np.median(np.abs(z_minus - median_minus))
            + np.median(np.abs(z_plus - median_plus))
        )
        low_side = -1 if median_minus <= median_plus else 1
        high_side = 1 if low_side == -1 else -1
        low_points = side_points[0 if low_side == -1 else 1]
        high_points = side_points[0 if high_side == -1 else 1]
        low_median = min(median_minus, median_plus)
        high_median = max(median_minus, median_plus)
        low_near_support = support_z is None or low_median <= float(support_z) + 0.020
        support_step = (
            support_z is None
            or low_median <= float(support_z) + EDGE_MIN_SUPPORT_PLANE_DELTA_M
            or high_median - low_median >= EDGE_MIN_CONTRAST_M
        )
        score = contrast / max(0.001, spread + 0.001)
        record = {
            "ray_index": ray_index,
            "angle_deg": math.degrees(angle),
            "support_points_minus": int(len(side_points[0])),
            "support_points_plus": int(len(side_points[1])),
            "median_z_minus_m": median_minus,
            "median_z_plus_m": median_plus,
            "contrast_m": contrast,
            "spread_m": spread,
            "score": float(score),
            "low_side": "minus" if low_side == -1 else "plus",
            "low_near_support": bool(low_near_support),
            "support_step": bool(support_step),
            "side_uv_minus": side_uv[0][:: max(1, len(side_uv[0]) // 8)],
            "side_uv_plus": side_uv[1][:: max(1, len(side_uv[1]) // 8)],
        }
        records.append(record)
    if not records:
        audit["reason"] = "too_few_side_samples"
        return None, audit
    records.sort(
        key=lambda item: (float(item["score"]), float(item["contrast_m"])), reverse=True
    )
    audit["rays"] = records[: min(12, len(records))]
    best = records[0]
    valid_best = bool(
        float(best["contrast_m"]) >= EDGE_MIN_CONTRAST_M
        and bool(best["low_near_support"])
        and bool(best["support_step"])
    )
    if not valid_best:
        audit["reason"] = "no_support_object_step"
        audit["best"] = best
        return None, audit
    angle = math.radians(float(best["angle_deg"]))
    image_direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
    # Re-sample the winning ray to form a metric world displacement.  The
    # direction is intentionally projected to the support plane before use as
    # an EEF approach; pitch is explored by the candidate generator.
    low_side = -1.0 if best["low_side"] == "minus" else 1.0
    high_side = -low_side

    def sample_side(sign: float) -> np.ndarray:
        points: list[np.ndarray] = []
        tangent = np.array([-image_direction[1], image_direction[0]])
        for radius in EDGE_SAMPLE_RADII_PX:
            for tangent_offset in range(
                -EDGE_TANGENTIAL_HALF_PX, EDGE_TANGENTIAL_HALF_PX + 1
            ):
                uv = np.array([u0, v0], dtype=np.float64)
                uv += sign * float(radius) * image_direction
                uv += float(tangent_offset) * tangent
                uu, vv = int(round(float(uv[0]))), int(round(float(uv[1])))
                if 0 <= uu < width and 0 <= vv < height:
                    value = float(metric[vv, uu])
                    if math.isfinite(value) and value > 0.02:
                        points.append(
                            _backproject_pixels(
                                np.asarray([uu], dtype=np.float64),
                                np.asarray([vv], dtype=np.float64),
                                np.asarray([value], dtype=np.float64),
                                np.asarray(intrinsic, dtype=np.float64),
                                np.asarray(extrinsic, dtype=np.float64),
                            )[0]
                        )
        return np.asarray(points, dtype=np.float64)

    low_points = sample_side(low_side)
    high_points = sample_side(high_side)
    displacement = np.median(high_points, axis=0) - np.median(low_points, axis=0)
    displacement[2] = 0.0
    norm = float(np.linalg.norm(displacement))
    if not math.isfinite(norm) or norm < 1e-5:
        audit["reason"] = "world_edge_direction_degenerate"
        audit["best"] = best
        return None, audit
    approach = displacement / norm
    audit.update(
        {
            "ok": True,
            "usable": True,
            "reason": "support_to_high_surface",
            "confidence": "strong" if float(best["score"]) >= 2.0 else "weak",
            "best": best,
            "image_direction_low_to_high": (high_side * image_direction).tolist(),
            "approach_world": approach.tolist(),
            "edge_tangent_world": np.cross(
                np.array([0.0, 0.0, 1.0]), approach
            ).tolist(),
            "low_surface_world": np.median(low_points, axis=0).tolist(),
            "high_surface_world": np.median(high_points, axis=0).tolist(),
        }
    )
    return approach, audit


def select_nearest_surface_anchor(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    center_world: np.ndarray,
    pixel_u: int,
    pixel_v: int,
    *,
    radius_m: float = SURFACE_ANCHOR_RADIUS_M,
    preserve_click: bool = False,
) -> tuple[np.ndarray, int, int, dict[str, Any]]:
    """Audit the local surface, optionally preserving a vertical-grasp click.

    The camera-nearest convention remains available to legacy callers. A
    downward pinch must not move a valid top/rim click toward the camera: that
    systematically moves the narrow finger contact patch toward an edge.
    """
    depth_array = _finite_depth(depth)
    height, width = depth_array.shape
    u0, v0 = int(pixel_u), int(pixel_v)
    if not 0 <= u0 < width or not 0 <= v0 < height:
        raise ValueError("surface anchor center pixel is outside the depth image")
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("surface anchor radius must be positive and finite")

    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center)):
        raise ValueError("surface anchor center must be finite")
    center_depth = float(depth_array[v0, u0])
    if not math.isfinite(center_depth) or center_depth <= 0.02:
        raise ValueError("surface anchor center depth is invalid")

    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    transform = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(transform)):
        raise ValueError("surface anchor camera matrices must be finite")
    fx, fy = float(k[0, 0]), float(k[1, 1])
    if abs(fx) <= 1e-10 or abs(fy) <= 1e-10:
        raise ValueError("surface anchor camera focal length is zero")

    center_optical_x = (u0 - float(k[0, 2])) * center_depth / fx
    center_optical_y = (v0 - float(k[1, 2])) * center_depth / fy
    minimum_depth = max(center_depth - radius, 1e-6)
    # This conservative projection bound contains every point in the 3-D ball,
    # including off-axis clicks whose depth changes across the sphere.
    half_u = (
        int(
            math.ceil(
                abs(fx)
                * radius
                * (1.0 + abs(center_optical_x) / center_depth)
                / minimum_depth
            )
        )
        + 1
    )
    half_v = (
        int(
            math.ceil(
                abs(fy)
                * radius
                * (1.0 + abs(center_optical_y) / center_depth)
                / minimum_depth
            )
        )
        + 1
    )
    u_low, u_high = max(0, u0 - half_u), min(width - 1, u0 + half_u)
    v_low, v_high = max(0, v0 - half_v), min(height - 1, v0 + half_v)

    grid_u, grid_v = np.meshgrid(
        np.arange(u_low, u_high + 1, dtype=np.int32),
        np.arange(v_low, v_high + 1, dtype=np.int32),
    )
    distances = depth_array[grid_v, grid_u]
    valid = np.isfinite(distances) & (distances > 0.02)
    sample_u = grid_u[valid]
    sample_v = grid_v[valid]
    sample_depth = distances[valid]
    points = _backproject_pixels(
        sample_u.astype(np.float64),
        sample_v.astype(np.float64),
        sample_depth.astype(np.float64),
        k,
        transform,
    )
    center_distance = np.linalg.norm(points - center.reshape(1, 3), axis=1)
    in_sphere = center_distance <= radius + 1e-9
    points = points[in_sphere]
    sample_u = sample_u[in_sphere]
    sample_v = sample_v[in_sphere]
    sample_depth = sample_depth[in_sphere]
    center_distance = center_distance[in_sphere]
    if len(points) == 0:
        raise RuntimeError("surface anchor sphere contains no visible RGB-D points")

    camera_position = transform[:3, 3]
    camera_distance = np.linalg.norm(points - camera_position.reshape(1, 3), axis=1)
    # Camera distance is primary. Remaining keys make equal-depth surfaces
    # deterministic and favor the original click when ranges are identical.
    order = np.lexsort(
        (
            sample_u,
            sample_v,
            center_distance,
            camera_distance,
        )
    )
    selected_index = (
        int(np.flatnonzero((sample_u == u0) & (sample_v == v0))[0])
        if preserve_click
        else int(order[0])
    )
    selected = points[selected_index]
    selected_u = int(sample_u[selected_index])
    selected_v = int(sample_v[selected_index])
    selected_depth = float(sample_depth[selected_index])
    selected_center_distance = float(center_distance[selected_index])
    selected_camera_distance = float(camera_distance[selected_index])
    center_camera_distance = float(np.linalg.norm(center - camera_position))
    relative_u = 1000.0 * selected_u / max(1, width - 1)
    relative_v = 1000.0 * selected_v / max(1, height - 1)
    audit = {
        "source": "frozen_evaluator_depth_visible_surface",
        "selection": (
            "original_visible_depth_click_for_vertical_pinch"
            if preserve_click
            else "nearest_to_capture_camera_in_click_centered_sphere"
        ),
        "diameter_m": float(radius * 2.0),
        "radius_m": radius,
        "sample_stride_px": 1,
        "center_pixel_uv": [u0, v0],
        "center_world": center.tolist(),
        "center_depth_m": center_depth,
        "search_window_px": [u_low, v_low, u_high, v_high],
        "valid_window_point_count": int(np.count_nonzero(valid)),
        "sphere_surface_point_count": int(len(points)),
        "selected_pixel_uv": [selected_u, selected_v],
        "selected_relative_uv": [relative_u, relative_v],
        "selected_world": selected.tolist(),
        "selected_depth_m": selected_depth,
        "selected_camera_distance_m": selected_camera_distance,
        "center_to_selected_m": selected_center_distance,
        "camera_distance_improvement_m": (
            center_camera_distance - selected_camera_distance
        ),
        "selected_is_original_click": bool(selected_u == u0 and selected_v == v0),
        "uses_segmentation": False,
        "uses_simulator_geometry": False,
    }
    return selected, selected_u, selected_v, audit


def estimate_outward_normal(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    pixel_u: int,
    pixel_v: int,
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    """Fit the BEHAVIOR-style multiscale organized-depth PCA normal."""
    return _estimate_outward_normal_scales(
        depth,
        intrinsic,
        extrinsic,
        pixel_u,
        pixel_v,
        (4, 6, 8),
    )


def _estimate_outward_normal_scales(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    pixel_u: int,
    pixel_v: int,
    half_windows: tuple[int, ...],
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    """The same robust fit and quality limits for an explicit image scale set."""
    height, width = depth.shape
    center_depth = float(depth[pixel_v, pixel_u])
    audit: dict[str, Any] = {
        "source": "organized_depth_pointcloud_multiscale_pca",
        "pixel": [int(pixel_u), int(pixel_v)],
        "half_windows_px": list(half_windows),
        "depth_jump_m": 0.02,
        "radius_m": 0.04,
        "min_points": 20,
        "uses_segmentation": False,
        "uses_simulator_mesh": False,
        "scales": [],
    }
    if not math.isfinite(center_depth) or center_depth <= 0.02:
        audit["reason"] = "center_depth_invalid"
        return None, "none", audit
    center = _backproject_pixels(
        np.asarray([pixel_u], dtype=np.float64),
        np.asarray([pixel_v], dtype=np.float64),
        np.asarray([center_depth], dtype=np.float64),
        intrinsic,
        extrinsic,
    )[0]
    camera_position = extrinsic[:3, 3]
    passing: list[dict[str, Any]] = []
    for half_window in half_windows:
        u_values = np.arange(
            max(0, pixel_u - half_window),
            min(width, pixel_u + half_window + 1),
            dtype=np.int32,
        )
        v_values = np.arange(
            max(0, pixel_v - half_window),
            min(height, pixel_v + half_window + 1),
            dtype=np.int32,
        )
        grid_u, grid_v = np.meshgrid(u_values, v_values)
        distances = depth[grid_v, grid_u]
        valid = (
            np.isfinite(distances)
            & (distances > 0.02)
            & (np.abs(distances - center_depth) <= 0.02)
        )
        if not np.any(valid):
            scale = {
                "half_window_px": half_window,
                "support_points": 0,
                "ok": False,
                "reason": "too_few_points",
            }
            audit["scales"].append(scale)
            continue
        sample_u = grid_u[valid].astype(np.float64)
        sample_v = grid_v[valid].astype(np.float64)
        points = _backproject_pixels(
            sample_u,
            sample_v,
            distances[valid].astype(np.float64),
            intrinsic,
            extrinsic,
        )
        radius_keep = np.linalg.norm(points - center.reshape(1, 3), axis=1) <= 0.04
        points = points[radius_keep]
        sample_u = sample_u[radius_keep]
        sample_v = sample_v[radius_keep]
        scale = {
            "half_window_px": half_window,
            "support_points": int(len(points)),
            "ok": False,
        }
        if len(points) < 20:
            scale["reason"] = "too_few_points"
            audit["scales"].append(scale)
            continue
        delta = np.column_stack((sample_u - pixel_u, sample_v - pixel_v))
        sigma = max(1.0, half_window * 0.65)
        weights = np.exp(-0.5 * np.sum(delta * delta, axis=1) / (sigma * sigma))

        def fit(values: np.ndarray, fit_weights: np.ndarray):
            total = max(float(fit_weights.sum()), 1e-12)
            centroid = np.sum(values * fit_weights[:, None], axis=0) / total
            centered = values - centroid
            covariance = centered.T @ (centered * fit_weights[:, None]) / total
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            return centroid, eigenvalues, eigenvectors

        centroid, eigenvalues, eigenvectors = fit(points, weights)
        normal = eigenvectors[:, 0]
        residuals = np.abs((points - centroid) @ normal)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        residual_limit = max(0.0015, median + 3.0 * 1.4826 * mad)
        keep = residuals <= residual_limit
        if 20 <= int(np.count_nonzero(keep)) < len(points):
            points = points[keep]
            weights = weights[keep]
            centroid, eigenvalues, eigenvectors = fit(points, weights)
            normal = eigenvectors[:, 0]
        normal /= np.linalg.norm(normal) + 1e-12
        if float(normal @ (camera_position - center)) < 0.0:
            normal = -normal
        residual = (points - centroid) @ normal
        rms = float(np.sqrt(np.average(residual * residual, weights=weights)))
        eigen_sum = float(eigenvalues.sum())
        planarity = (
            float(eigenvalues[0]) / eigen_sum if eigen_sum > 1e-15 else float("inf")
        )
        scale.update(
            {
                "ok": bool(planarity <= 0.08 and rms <= 0.003),
                "reason": "fit",
                "support_points": int(len(points)),
                "normal_world": normal.tolist(),
                "planarity": planarity,
                "rms_m": rms,
                "residual_limit_m": residual_limit,
            }
        )
        audit["scales"].append(scale)
        if scale["ok"]:
            passing.append(scale)
    if not passing:
        audit["reason"] = "insufficient_quality_scales"
        audit["passing_scales"] = 0
        return None, "none", audit
    maximum_angle = 0.0
    for left_index in range(len(passing)):
        for right_index in range(left_index + 1, len(passing)):
            left = np.asarray(passing[left_index]["normal_world"])
            right = np.asarray(passing[right_index]["normal_world"])
            maximum_angle = max(
                maximum_angle,
                math.degrees(math.acos(float(np.clip(left @ right, -1.0, 1.0)))),
            )
    selected = min(
        passing,
        key=lambda value: (
            float(value["planarity"]),
            float(value["rms_m"]),
            -int(value["support_points"]),
        ),
    )
    confidence = "strong" if len(passing) >= 2 and maximum_angle <= 12.0 else "weak"
    normal = np.asarray(selected["normal_world"], dtype=np.float64)
    audit.update(
        {
            "ok": True,
            "reason": "ok" if confidence == "strong" else "single_quality_scale",
            "confidence": confidence,
            "passing_scales": int(len(passing)),
            "max_scale_angle_deg": float(maximum_angle),
            "normal_world": normal.tolist(),
            "selected_half_window_px": int(selected["half_window_px"]),
            "selected_support_points": int(selected["support_points"]),
            "selected_planarity": float(selected["planarity"]),
            "selected_rms_m": float(selected["rms_m"]),
        }
    )
    return normal, confidence, audit


class DepthRayOccupancy:
    """Query the 3 mm organized-shell completion without a simulator mesh."""

    def __init__(
        self,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        *,
        support_z: float | None = None,
        target_anchor: np.ndarray | None = None,
        target_radius_m: float = OPENING_TARGET_RADIUS_M,
    ) -> None:
        self.depth = _finite_depth(depth)
        self.intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
        self.extrinsic = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
        self.height, self.width = self.depth.shape

        self.support_z = (
            None
            if support_z is None or not math.isfinite(float(support_z))
            else float(support_z)
        )
        self.target_anchor = (
            None
            if target_anchor is None
            else np.asarray(target_anchor, dtype=np.float64).reshape(3)
        )
        self.target_radius_m = float(target_radius_m)

    def occupied_mask(
        self,
        points_eef: np.ndarray,
        rotation: np.ndarray,
        position: np.ndarray,
        *,
        surface_only: bool = False,
        target_anchor: np.ndarray | None = None,
        target_radius_m: float | None = None,
    ) -> np.ndarray:
        """Return observed-shell membership for model points.

        ``surface_only`` is used exclusively for the opening volume.  It
        removes pixels whose reconstructed world height is the dominant
        support plane (the table), and optionally limits the opening query to
        the click neighbourhood.  Solid overlap queries intentionally retain
        support points so table collisions remain visible to the prefilter.
        """
        points = np.asarray(points_eef, dtype=np.float64).reshape(-1, 3)
        rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        points_world = points @ rot.T + pos.reshape(1, 3)
        points_usd = (points_world - self.extrinsic[:3, 3]) @ self.extrinsic[:3, :3]
        query_depth = -points_usd[:, 2]
        valid = np.isfinite(query_depth) & (query_depth > 0.02)
        columns = np.zeros(len(points_world), dtype=np.int32)
        rows = np.zeros(len(points_world), dtype=np.int32)
        columns[valid] = np.rint(
            float(self.intrinsic[0, 2])
            + float(self.intrinsic[0, 0]) * points_usd[valid, 0] / query_depth[valid]
        ).astype(np.int32)
        rows[valid] = np.rint(
            float(self.intrinsic[1, 2])
            - float(self.intrinsic[1, 1]) * points_usd[valid, 1] / query_depth[valid]
        ).astype(np.int32)
        valid &= (
            (columns >= 0) & (columns < self.width) & (rows >= 0) & (rows < self.height)
        )
        result = np.zeros(len(points_world), dtype=bool)
        indices = np.nonzero(valid)[0]
        if len(indices) == 0:
            return result
        observed = self.depth[rows[indices], columns[indices]]
        delta = query_depth[indices] - observed
        occupied = (
            np.isfinite(observed)
            & (observed > 0.02)
            & (delta >= -0.5 * VOXEL_M)
            & (delta <= BACK_EXTRUSION_M + 0.5 * VOXEL_M)
        )
        if surface_only and np.any(occupied):
            support_z = self.support_z
            if support_z is not None:
                observed_optical = np.column_stack(
                    (
                        (columns[indices] - float(self.intrinsic[0, 2]))
                        * observed
                        / float(self.intrinsic[0, 0]),
                        (rows[indices] - float(self.intrinsic[1, 2]))
                        * observed
                        / float(self.intrinsic[1, 1]),
                        observed,
                    )
                )
                observed_world = (
                    observed_optical * np.array([1.0, -1.0, -1.0])
                ) @ self.extrinsic[:3, :3].T + self.extrinsic[:3, 3]
                occupied &= observed_world[:, 2] >= support_z + OPENING_SUPPORT_MARGIN_M
            anchor = (
                self.target_anchor
                if target_anchor is None
                else np.asarray(target_anchor, dtype=np.float64).reshape(3)
            )
            radius = (
                self.target_radius_m
                if target_radius_m is None
                else float(target_radius_m)
            )
            if anchor is not None and math.isfinite(radius) and radius > 0.0:
                # Use the observed point rather than the query point: this
                # makes the neighbourhood gate stable under shell extrusion.
                if "observed_world" not in locals():
                    observed_optical = np.column_stack(
                        (
                            (columns[indices] - float(self.intrinsic[0, 2]))
                            * observed
                            / float(self.intrinsic[0, 0]),
                            (rows[indices] - float(self.intrinsic[1, 2]))
                            * observed
                            / float(self.intrinsic[1, 1]),
                            observed,
                        )
                    )
                    observed_world = (
                        observed_optical * np.array([1.0, -1.0, -1.0])
                    ) @ self.extrinsic[:3, :3].T + self.extrinsic[:3, 3]
                occupied &= (
                    np.linalg.norm(observed_world - anchor.reshape(1, 3), axis=1)
                    <= radius
                )
        result[indices] = occupied
        return result

    def count(
        self,
        points_eef: np.ndarray,
        rotation: np.ndarray,
        position: np.ndarray,
        *,
        surface_only: bool = False,
        target_anchor: np.ndarray | None = None,
        target_radius_m: float | None = None,
    ) -> int:
        return int(
            np.count_nonzero(
                self.occupied_mask(
                    points_eef,
                    rotation,
                    position,
                    surface_only=surface_only,
                    target_anchor=target_anchor,
                    target_radius_m=target_radius_m,
                )
            )
        )


def adaptive_grasp_volume_floor(
    safe_max_cm3: float,
    *,
    quality_ratio: float,
    absolute_min_cm3: float,
) -> float:
    safe_max = max(0.0, float(safe_max_cm3))
    if safe_max < float(absolute_min_cm3):
        return float(absolute_min_cm3)
    return float(
        min(
            MIN_GRASP_VOL_CM3,
            safe_max,
            max(float(absolute_min_cm3), float(quality_ratio) * safe_max),
        )
    )


def _rank_ik_input(
    strict: list[dict[str, Any]],
    normal_confidence: str,
    *,
    vertical: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quality = sorted(
        strict,
        key=lambda pose: (
            -int(pose["grasp_vox"]),
            int(pose["inflated_overlap_vox"]),
            int(pose["overlap_vox"]),
            float(pose["tie_break"]),
        ),
    )
    selected = quality[: min(IK_QUALITY_QUOTA, len(quality))]
    selected_ids = {id(pose) for pose in selected}
    alignment_added = 0
    if (vertical or normal_confidence in ("strong", "weak")) and quality:
        best_grasp = int(quality[0]["grasp_vox"])
        ratio = 0.75 if vertical else 0.85 if normal_confidence == "strong" else 0.50
        alignment_source = (
            quality
            if vertical
            else [
                pose for pose in quality if int(pose["grasp_vox"]) >= ratio * best_grasp
            ]
        )
        alignment = sorted(
            alignment_source,
            key=lambda pose: (
                float(pose.get("approach_tilt_deg", float("inf"))),
                float(pose["normal_alignment_deg"]),
                float(pose.get("closure_alignment_deg", 0.0)),
                -int(pose["grasp_vox"]),
                int(pose["inflated_overlap_vox"]),
            ),
        )
        quota = IK_ALIGNMENT_QUOTA if vertical or normal_confidence == "strong" else 12
        for pose in alignment[:quota]:
            if len(selected) >= IK_INPUT_MAX:
                break
            if id(pose) not in selected_ids:
                selected.append(pose)
                selected_ids.add(id(pose))
                alignment_added += 1
    for pose in quality:
        if len(selected) >= IK_INPUT_MAX:
            break
        if id(pose) not in selected_ids:
            selected.append(pose)
            selected_ids.add(id(pose))
    return selected, {
        "input_pose_count": int(len(strict)),
        "limit": IK_INPUT_MAX,
        "quality_quota": IK_QUALITY_QUOTA,
        "alignment_quota": IK_ALIGNMENT_QUOTA,
        "vertical_alignment_volume_prefiltered": False if vertical else None,
        "alignment_added_count": int(alignment_added),
        "selected_count": int(len(selected)),
    }


def _vertical_contact_patch(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    hit: np.ndarray,
    pixel_u: int,
    pixel_v: int,
    support_normal: np.ndarray,
    support_z: float,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Center a compact, observed support-parallel contact patch when justified.

    This is target-pose fitting, not scene reconstruction or instance labeling.
    A connected, same-height RGB-D patch must have a bounded footprint and an
    observed center. A cup rim therefore cannot be recentered into its hole;
    a large garment/laptop patch cannot be silently treated as a complete box.
    """
    audit: dict[str, Any] = {
        "source": "click_connected_public_depth_support_parallel_patch",
        "applied": False,
        "reason": None,
        "uses_semantic_segmentation": False,
        "uses_simulator_state": False,
        "surface_band_m": 0.0025,
        "search_radius_m": 0.055,
    }
    anchor = np.asarray(hit, dtype=np.float64)
    normal = np.asarray(support_normal, dtype=np.float64)
    if float(anchor[2]) - float(support_z) < 2 * OPENING_SUPPORT_MARGIN_M:
        audit["reason"] = "surface_too_close_to_support_for_compact_patch_fit"
        return anchor.copy(), None, audit
    h, w = depth.shape
    vv, uu = np.indices((h, w))
    points = _backproject_pixels(
        uu.ravel(), vv.ravel(), depth.ravel(), intrinsic, extrinsic
    )
    delta = points - anchor
    heights = delta @ normal
    tangent = delta - heights[:, None] * normal
    radius = np.linalg.norm(tangent, axis=1)
    valid = np.isfinite(depth.ravel()) & (depth.ravel() > 0.02)
    valid &= (np.abs(heights) <= audit["surface_band_m"]) & (
        radius <= audit["search_radius_m"]
    )
    start = int(pixel_v) * w + int(pixel_u)
    if not valid[start]:
        audit["reason"] = "click_not_in_observed_patch"
        return anchor.copy(), None, audit
    # Four-connected image adjacency is used only to separate surface patches;
    # no points or faces are removed from the whole-view scene geometry.
    stack = [start]
    selected: list[int] = []
    valid[start] = False
    while stack:
        index = stack.pop()
        selected.append(index)
        row, col = divmod(index, w)
        for neighbor in (
            index - w if row else -1,
            index + w if row + 1 < h else -1,
            index - 1 if col else -1,
            index + 1 if col + 1 < w else -1,
        ):
            if neighbor >= 0 and valid[neighbor]:
                valid[neighbor] = False
                stack.append(neighbor)
    ids = np.asarray(selected, dtype=np.int64)
    patch = points[ids]
    audit["point_count"] = len(ids)
    if len(ids) < 16:
        audit["reason"] = "insufficient_patch_samples"
        return anchor.copy(), None, audit
    if float(np.max(radius[ids])) >= audit["search_radius_m"] - 2 * VOXEL_M:
        audit["reason"] = "patch_reaches_search_boundary_shape_not_observed"
        return anchor.copy(), None, audit
    if np.any(
        (uu.ravel()[ids] == 0)
        | (uu.ravel()[ids] == w - 1)
        | (vv.ravel()[ids] == 0)
        | (vv.ravel()[ids] == h - 1)
    ):
        audit["reason"] = "patch_reaches_image_boundary"
        return anchor.copy(), None, audit
    ex = np.array([1.0, 0.0, 0.0])
    ex = ex - (ex @ normal) * normal
    ex /= np.linalg.norm(ex)
    ey = np.cross(normal, ex)
    coordinates = np.column_stack((tangent[ids] @ ex, tangent[ids] @ ey))
    ordered = sorted(set(map(tuple, coordinates.tolist())))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        audit["reason"] = "degenerate_patch_footprint"
        return anchor.copy(), None, audit
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    axes = edges[lengths > 1e-8] / lengths[lengths > 1e-8, None]
    perpendicular = np.column_stack((-axes[:, 1], axes[:, 0]))
    along, across = coordinates @ axes.T, coordinates @ perpendicular.T
    low, high = along.min(axis=0), along.max(axis=0)
    low2, high2 = across.min(axis=0), across.max(axis=0)
    areas = (high - low) * (high2 - low2)
    chosen = int(np.argmin(areas))
    dimensions = np.array([high[chosen] - low[chosen], high2[chosen] - low2[chosen]])
    center = (
        axes[chosen] * (high[chosen] + low[chosen]) / 2
        + perpendicular[chosen] * (high2[chosen] + low2[chosen]) / 2
    )
    center_distances = np.linalg.norm(coordinates - center, axis=1)
    nearest = int(np.argmin(center_distances))
    hull_area = (
        abs(
            float(
                np.sum(
                    hull[:, 0] * np.roll(hull[:, 1], -1)
                    - hull[:, 1] * np.roll(hull[:, 0], -1)
                )
            )
        )
        / 2
    )
    rectangle_fill = hull_area / max(float(areas[chosen]), 1e-12)
    audit.update(
        footprint_dimensions_m=dimensions.tolist(),
        rectangularity=rectangle_fill,
        center_offset_m=float(np.linalg.norm(center)),
        center_observed_distance_m=float(center_distances[nearest]),
    )
    if float(np.max(dimensions)) > 0.075 or float(np.min(dimensions)) < 2 * VOXEL_M:
        audit["reason"] = "patch_not_compact_within_opening_scale"
        return anchor.copy(), None, audit
    if np.linalg.norm(center) > VERTICAL_ANCHOR_TRUST_M:
        audit["reason"] = "center_outside_local_anchor_trust_region"
        return anchor.copy(), None, audit
    if center_distances[nearest] > 1.5 * VOXEL_M:
        audit["reason"] = "center_has_no_same_height_observed_surface_possible_rim"
        return anchor.copy(), None, audit
    centered = patch[nearest]
    # An elongated observed footprint has a narrow closing direction even
    # when its curved ends do not fill a rectangle (e.g. a ridge or handle).
    # Closing along the long axis is not an equivalent antipodal grasp. For
    # approximately square rectangular patches either face normal is valid.
    # This prior changes candidate yaw/ranking only, never scene occupancy.
    footprint_axes = np.stack(
        (
            axes[chosen][0] * ex + axes[chosen][1] * ey,
            perpendicular[chosen][0] * ex + perpendicular[chosen][1] * ey,
        )
    )
    aspect_ratio = float(np.max(dimensions) / np.min(dimensions))
    # Rounded, compact footprints can carry a resolved yaw without filling
    # a rectangle. Use only measured caliper widths, never a semantic shape
    # class. A variation below two geometry cells is unresolved and must not
    # manufacture an axis for a circular cap. This is a sampling/ranking cue,
    # not evidence of antipodal contact or a new acceptance gate.
    caliper_widths = np.concatenate((high - low, high2 - low2))
    width_variation = float(np.ptp(caliper_widths))
    width_resolution = 2 * VOXEL_M
    if aspect_ratio >= 2.0:
        direction = footprint_axes[int(np.argmin(dimensions))]
        axis_mode = "short_axis"
    elif rectangle_fill >= 0.90:
        direction = footprint_axes[0]
        axis_mode = "either_rectangle_axis"
    elif width_variation >= width_resolution:
        direction = footprint_axes[int(np.argmin(dimensions))]
        axis_mode = "either_observed_footprint_axis"
    else:
        direction = None
        axis_mode = "unconstrained"
    audit.update(
        applied=True,
        reason="bounded_patch_with_observed_center",
        anchor_world=centered.tolist(),
        anchor_pixel_uv=[int(uu.ravel()[ids[nearest]]), int(vv.ravel()[ids[nearest]])],
        closing_axis_world=None if direction is None else direction.tolist(),
        closing_axis_mode=axis_mode,
        footprint_aspect_ratio=aspect_ratio,
        footprint_axes_world=footprint_axes.tolist(),
        caliper_width_range_m=[
            float(caliper_widths.min()),
            float(caliper_widths.max()),
        ],
        caliper_width_variation_m=width_variation,
        caliper_resolution_m=width_resolution,
        footprint_yaw_is_contact_evidence=False,
    )
    return centered.copy(), direction, audit


def _projected_surface_closing_axis(
    normal_audit: dict[str, Any],
    support_normal: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Infer pinch yaw, never approach tilt, from observed wall normals.

    A rim's multiscale normals can disagree in elevation yet agree about the
    wall's horizontal normal. Pinching along the wall tangent can score more
    visible surface while pushing the wall out of the fingers on closure.
    Require at least two reliable, non-horizontal surface fits with agreeing
    support-plane projections. Missing evidence leaves yaw unconstrained; no
    hole center, backside, material thickness, or object identity is invented.
    """
    up = np.asarray(support_normal, dtype=np.float64).reshape(3)
    up = up / np.linalg.norm(up)
    audit = {
        "source": "multiscale_public_depth_normals_projected_onto_support_plane",
        "applied": False,
        "reason": None,
        "minimum_scales": 2,
        "minimum_tangent_normal_magnitude": 0.5,
        "maximum_projected_axis_disagreement_deg": 15.0,
        "uses_object_identity": False,
        "uses_simulator_state": False,
        "changes_approach_axis": False,
        "changes_anchor": False,
    }
    axes = []
    for scale in normal_audit.get("scales", []):
        if (
            not scale.get("ok")
            or int(scale.get("support_points", 0)) < 20
            or float(scale.get("planarity", float("inf"))) > 0.08
            or float(scale.get("rms_m", float("inf"))) > 0.003
        ):
            continue
        normal = np.asarray(scale.get("normal_world", []), dtype=np.float64)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            continue
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        tangent = normal - (normal @ up) * up
        length = np.linalg.norm(tangent)
        if length >= audit["minimum_tangent_normal_magnitude"]:
            axes.append(tangent / length)
    audit["usable_scale_count"] = len(axes)
    if len(axes) < audit["minimum_scales"]:
        audit["reason"] = "insufficient_observed_wall_normal_scales"
        return None, audit
    disagreement = max(
        math.degrees(math.acos(float(np.clip(abs(a @ b), 0.0, 1.0))))
        for i, a in enumerate(axes)
        for b in axes[i + 1 :]
    )
    audit["projected_axis_disagreement_deg"] = disagreement
    if disagreement > audit["maximum_projected_axis_disagreement_deg"]:
        audit["reason"] = "inconsistent_projected_surface_normals"
        return None, audit
    aligned = [axis if axis @ axes[0] >= 0.0 else -axis for axis in axes]
    direction = np.sum(aligned, axis=0)
    direction /= np.linalg.norm(direction)
    audit.update(
        applied=True,
        reason="consistent_observed_wall_normal_projection",
        closing_axis_world=direction.tolist(),
        closing_axis_mode="projected_surface_normal",
    )
    return direction, audit


def _adaptive_wall_normal_evidence(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    pixel_u: int,
    pixel_v: int,
    coarse_audit: dict[str, Any],
    support_normal: np.ndarray,
    *,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Supplement insufficient wall evidence using the same captured depth.

    The existing fit needs 20 points, so a 5x5 window is the smallest possible
    square and 7x7 is the other scale below the existing 9x9 window. Both use
    the original quality gates. Keep every coarse scale: conflicting evidence
    cannot be discarded in order to manufacture an agreeing pinch direction.
    This only informs closure yaw; it changes neither mesh nor approach tilt.
    """
    audit = {
        "source": "same_frozen_public_depth_finer_organized_pca",
        "applied": False,
        "reason": "explicit_legacy_or_mesh_fallback_route",
        "half_windows_px": [2, 3],
        "scale_selection": "smallest_squares_supporting_original_20_point_gate_below_9x9",
        "quality_thresholds_relaxed": False,
        "coarse_evidence_discarded": False,
        "extra_observations_requested": 0,
        "changes_anchor": False,
        "changes_approach_axis": False,
        "uses_object_identity": False,
        "uses_simulator_state": False,
    }
    if not enabled:
        return coarse_audit, audit
    _, coarse_axis = _projected_surface_closing_axis(coarse_audit, support_normal)
    audit["coarse_axis_evidence"] = coarse_axis
    if coarse_axis["reason"] != "insufficient_observed_wall_normal_scales":
        audit["reason"] = (
            "existing_wall_evidence_sufficient"
            if coarse_axis["applied"]
            else "conflicting_coarse_evidence_preserved"
        )
        return coarse_audit, audit
    _, _, fine = _estimate_outward_normal_scales(
        depth,
        intrinsic,
        extrinsic,
        pixel_u,
        pixel_v,
        (2, 3),
    )
    scales = list(coarse_audit.get("scales", [])) + list(fine["scales"])
    combined = {"scales": scales}
    _, combined_axis = _projected_surface_closing_axis(combined, support_normal)
    audit.update(
        applied=bool(combined_axis["applied"]),
        reason=(
            "consistent_fine_and_coarse_wall_evidence"
            if combined_axis["applied"]
            else combined_axis["reason"]
        ),
        fine_fit=fine,
        combined_axis_evidence=combined_axis,
    )
    return combined, audit


def _closing_axis_alignment_deg(
    rotation: np.ndarray,
    closing_axis: np.ndarray | None,
    support_normal: np.ndarray,
    mode: str,
) -> float:
    if closing_axis is None:
        return 0.0
    alignment = abs(float(rotation[:, 1] @ closing_axis))
    if mode in ("either_rectangle_axis", "either_observed_footprint_axis"):
        alignment = max(
            alignment,
            abs(float(rotation[:, 1] @ np.cross(support_normal, closing_axis))),
        )
    return math.degrees(math.acos(float(np.clip(alignment, 0.0, 1.0))))


def _resolve_closing_axis(
    footprint_axis: np.ndarray | None,
    contact_patch_audit: dict[str, Any],
    normal_audit: dict[str, Any],
    support_normal: np.ndarray,
    *,
    use_observed_wall: bool,
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    """Prefer observed wall normals to any support-height footprint yaw.

    A narrow, bounded same-height slice can come from a sloping wall, not
    just a horizontal handle. Its short/rectangle axis is a sampling cue,
    not evidence that the opposing finger has clearance. Do not let that
    cue suppress the existing wall-direction gate and transverse repair.
    This arbitration changes neither the observed anchor nor occupancy.
    The explicit legacy route retains its footprint-only behavior.
    """
    axis = None if footprint_axis is None else np.asarray(footprint_axis).copy()
    mode = contact_patch_audit.get("closing_axis_mode", "unconstrained")
    audit = {
        "source": "bounded_observed_contact_patch" if axis is not None else "none",
        "applied": axis is not None,
        "reason": "contact_patch_axis_available"
        if axis is not None
        else "unconstrained",
        "closing_axis_mode": mode,
        "closing_axis_world": None if axis is None else axis.tolist(),
        "changes_approach_axis": False,
        "changes_anchor": False,
    }
    if use_observed_wall:
        wall_axis, wall_audit = _projected_surface_closing_axis(
            normal_audit,
            support_normal,
        )
        if wall_axis is not None:
            prior = dict(audit)
            axis, mode = wall_axis, "projected_surface_normal"
            audit = dict(wall_audit)
            audit.update(
                evidence_priority="reliable_wall_normal_before_height_slice_footprint",
                footprint_axis_overridden=footprint_axis is not None,
                footprint_prior=prior,
            )
        elif axis is None:
            audit = dict(wall_audit)
        else:
            audit["wall_normal_fallback"] = wall_audit
    required = mode == "projected_surface_normal"
    audit.update(
        required_for_candidate_acceptance=required,
        maximum_closure_alignment_deg=CLOSURE_AXIS_MAX_DEG if required else None,
        threshold_source="half_of_existing_roll_sample_spacing",
    )
    return axis, mode, audit


def _observed_closing_axis_pass(
    angle_deg: float, *, required: bool, fk_tolerance_deg: float = 0.0
) -> bool:
    """Do not let a larger tangent-wall score defeat observed pinch yaw.

    The bound is half of the existing roll sample spacing, not an object-
    specific parameter. It is only required when agreeing observed wall
    normals supplied the axis. Top surfaces keep the existing patch policy.
    """
    return (
        not required
        or float(angle_deg) <= CLOSURE_AXIS_MAX_DEG + fk_tolerance_deg + 1e-9
    )


def _approach_specs(
    edge_approach: np.ndarray | None,
    support_normal: np.ndarray | None = None,
    *,
    max_tilt_deg: float = MAX_DOWNWARD_TILT_DEG,
) -> list[tuple[str, np.ndarray, float | None]]:
    """Twenty approaches in one explicitly bounded downward cone.

    X5 local +X points from the palm toward the fingertips. The reported
    gripper vector is -X, so it aligns with the *upward* support normal.
    The clicked surface normal must not turn a rim grasp into a side grasp.
    """
    if (
        not math.isfinite(max_tilt_deg)
        or max_tilt_deg <= 0.0
        or max_tilt_deg > OFFLINE_MAX_DOWNWARD_TILT_DEG
    ):
        raise ValueError(
            "downward cone must be positive and at most "
            f"{OFFLINE_MAX_DOWNWARD_TILT_DEG:g} degrees"
        )
    up = (
        np.array([0.0, 0.0, 1.0])
        if support_normal is None
        else np.asarray(support_normal, dtype=np.float64).reshape(3).copy()
    )
    if not np.all(np.isfinite(up)) or np.linalg.norm(up) < 1e-8:
        raise ValueError("invalid support normal")
    up /= np.linalg.norm(up)
    if up[2] < 0:
        up = -up
    tangent = (
        np.array([1.0, 0.0, 0.0])
        if edge_approach is None
        else np.asarray(edge_approach, dtype=np.float64).copy()
    )
    tangent -= up * float(tangent @ up)
    if not np.all(np.isfinite(tangent)) or np.linalg.norm(tangent) < 1e-8:
        tangent = np.cross(up, np.array([0.0, 1.0, 0.0]))
    tangent /= np.linalg.norm(tangent)
    other = np.cross(up, tangent)
    result = [("table_normal_downward", -up.copy(), 0.0)]
    # Scale the inner ring with the public cap so a strict production plan
    # never samples a wider angle through a fixed inner-ring shortcut.
    inner_tilt_deg = min(7.5, 0.5 * float(max_tilt_deg))
    for tilt_deg, count in ((inner_tilt_deg, 7), (max_tilt_deg, 12)):
        tilt = math.radians(tilt_deg)
        for index in range(count):
            angle = 2 * math.pi * index / count
            radial = math.cos(angle) * tangent + math.sin(angle) * other
            vector = -math.cos(tilt) * up + math.sin(tilt) * radial
            result.append(("table_normal_downward", vector, tilt))
    return result


def _joint_limit_margin(joints: Any) -> float:
    """Return the smallest policy-local distance to an X5 joint limit."""
    values = np.asarray(joints, dtype=np.float64).reshape(-1)
    if values.size != 6 or not np.all(np.isfinite(values)):
        return 0.0
    lower = np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -3.14])
    upper = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 3.14])
    return float(np.min(np.minimum(values - lower, upper - values)))


def _capture_arm_state(
    robot: dict[str, Any], arm: str
) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(robot[f"arm_{arm}_qpos"], dtype=np.float64).reshape(-1)
    eef_value = robot[f"eef_{arm}"]
    if isinstance(eef_value, dict):
        pose = np.concatenate(
            (
                np.asarray(eef_value["pos"], dtype=np.float64).reshape(3),
                np.asarray(eef_value["quat"], dtype=np.float64).reshape(4),
            )
        )
    else:
        pose = np.asarray(eef_value, dtype=np.float64).reshape(-1)
    if joints.size != 6 or pose.size != 7:
        raise ValueError(f"capture-time {arm} X5 proprioception is incomplete")
    if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(pose)):
        raise ValueError(f"capture-time {arm} X5 proprioception is not finite")
    return joints, pose


def _prepare_raw_grasp_points(occupancy: MeshSceneOccupancy) -> None:
    """Retain original observed material points independently of the band."""
    points = np.asarray(
        occupancy.result.observed_points_world, dtype=np.float64
    ).reshape(-1, 3)
    valid = np.all(np.isfinite(points), axis=1)
    anchor = occupancy.target_anchor
    if anchor is not None:
        valid &= np.linalg.norm(points - anchor, axis=1) <= occupancy.target_radius_m
    if occupancy.support_z is not None:
        side = 1.0 if anchor is None or anchor[2] >= occupancy.support_z else -1.0
        valid &= (
            side * (points[:, 2] - occupancy.support_z) >= occupancy.support_margin_m
        )
    separation = occupancy.result.metadata.get("support_separation", {})
    if separation.get("applied"):
        plane = np.asarray(separation["plane_coefficients_z_from_xy"])
        normal = np.asarray(separation["normal_world"])
        valid &= (points - [0.0, 0.0, plane[2]]) @ normal > occupancy.support_margin_m
    occupancy._raw_grasp_points_world = points[valid]


def _raw_surface_evidence(
    points_world: np.ndarray,
    rotation: np.ndarray,
    position: np.ndarray,
    anchor_world: np.ndarray,
) -> dict[str, Any]:
    """Require raw depth evidence, and report its axial tracking margin.

    A band-only score can be nonzero while no observation is in the opening.
    Moving the material +X relative to the EEF tests a small undershoot of a
    downward grasp. This is evidence robustness, not physical-contact truth.
    """
    local = (points_world - position) @ rotation
    inside = points_inside_exact_x5_opening(local, [1.0])
    occupied = local[inside]
    perturbed = occupied.copy()
    perturbed[:, 0] += RAW_SURFACE_TRACKING_MARGIN_M
    robust = points_inside_exact_x5_opening(perturbed, [1.0])
    cells = np.unique(np.floor(occupied / VOXEL_M).astype(np.int64), axis=0)
    count = len(occupied)
    return {
        "required": True,
        "source": "unexpanded_public_depth_points_excluding_support",
        "opening_query_version": OPENING_QUERY_VERSION,
        "raw_point_count": count,
        "raw_voxel_count": len(cells),
        "raw_voxel_proxy_cm3": len(cells) * VOXEL_M**3 * 1e6,
        "raw_voxel_proxy_is_material_volume": False,
        "raw_transverse_span_m": float(np.max(np.ptp(occupied[:, 1:], axis=0)))
        if count
        else 0.0,
        "axial_undershoot_test_m": RAW_SURFACE_TRACKING_MARGIN_M,
        "undershoot_surviving_point_count": int(robust.sum()),
        "anchor_inside_exact_opening": bool(
            points_inside_exact_x5_opening(
                [(anchor_world - position) @ rotation], [1.0]
            )[0]
        ),
        "ok": count >= RAW_SURFACE_MIN_POINTS and len(cells) >= 2,
        "reason": "ok"
        if count >= RAW_SURFACE_MIN_POINTS and len(cells) >= 2
        else "insufficient_raw_surface_in_exact_opening",
        "physical_contact_truth_used": False,
    }


def _score_gripper_pose(
    *,
    occupancy: Any,
    geometry: dict[str, Any],
    rotation: Any,
    position: Any,
    surface_only_opening: bool,
    target_anchor: Any,
    voxel_cm3: float,
) -> dict[str, Any]:
    """Score one concrete EEF pose with the planner's exact occupancy rules."""
    rotation_value = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    position_value = np.asarray(position, dtype=np.float64).reshape(3)
    anchor_value = np.asarray(target_anchor, dtype=np.float64).reshape(3)
    overlap_mask = occupancy.occupied_mask(
        geometry["gripper_voxels"], rotation_value, position_value
    )
    grasp_mask = occupancy.occupied_mask(
        geometry["opening_voxels"],
        rotation_value,
        position_value,
        surface_only=surface_only_opening,
        target_anchor=anchor_value,
    )
    inflated_mask = occupancy.occupied_mask(
        geometry["inflated_gripper_voxels"], rotation_value, position_value
    )
    overlap = int(np.count_nonzero(overlap_mask))
    grasp = int(np.count_nonzero(grasp_mask))
    inflated = int(np.count_nonzero(inflated_mask))
    occupied_opening = np.asarray(geometry["opening_voxels"])[grasp_mask]
    if len(occupied_opening):
        axial_min = float(np.min(occupied_opening[:, 0]))
        axial_max = float(np.max(occupied_opening[:, 0]))
        axial_span = axial_max - axial_min
        tip_fraction = float(
            np.count_nonzero(occupied_opening[:, 0] >= 0.14) / len(occupied_opening)
        )
        surface_span = float(np.max(np.ptp(occupied_opening[:, 1:], axis=0)))
        contact_columns = len(
            np.unique(
                np.rint(occupied_opening[:, [0, 2]] / VOXEL_M).astype(np.int64), axis=0
            )
        )
    else:
        axial_min = None
        axial_max = None
        axial_span = 0.0
        tip_fraction = 0.0
        surface_span = 0.0
        contact_columns = 0
    tip_only = bool(
        axial_span < AXIAL_SPAN_MIN_M or (tip_fraction >= 0.85 and axial_span < 0.030)
    )
    support_clearance = None
    if isinstance(occupancy, MeshSceneOccupancy):
        support = occupancy.result.metadata.get("support_separation", {})
        if support.get("applied"):
            coefficients = np.asarray(support["plane_coefficients_z_from_xy"])
            normal = np.asarray(support["normal_world"])
            vertices_world = (
                np.asarray(geometry["visual_vertices"]) @ rotation_value.T
                + position_value
            )
            support_clearance = float(
                np.min(
                    (vertices_world - np.array([0.0, 0.0, coefficients[2]])) @ normal
                )
            )
    raw_surface = {"required": False, "reason": "explicit_legacy_geometry_route"}
    if geometry["metadata"].get("opening_query_version") == OPENING_QUERY_VERSION:
        if not hasattr(occupancy, "_raw_grasp_points_world"):
            _prepare_raw_grasp_points(occupancy)
        raw_surface = _raw_surface_evidence(
            occupancy._raw_grasp_points_world,
            rotation_value,
            position_value,
            anchor_value,
        )
    return {
        "overlap_vox": overlap,
        "grasp_vox": grasp,
        "inflated_overlap_vox": inflated,
        "overlap_vol_cm3": overlap * float(voxel_cm3),
        "grasp_vol_cm3": grasp * float(voxel_cm3),
        "inflated_overlap_vol_cm3": inflated * float(voxel_cm3),
        "overlap_frac": overlap / max(1, len(geometry["gripper_voxels"])),
        "opening_axial_min_m": axial_min,
        "opening_axial_max_m": axial_max,
        "opening_axial_span_m": float(axial_span),
        "opening_tip_fraction": float(tip_fraction),
        "tip_only": tip_only,
        "opening_transverse_span_m": surface_span,
        "opening_contact_column_count": int(contact_columns),
        "support_clearance_m": support_clearance,
        "raw_surface_evidence": raw_surface,
    }


def _volume_validation(
    ideal: dict[str, Any],
    final_fk: dict[str, Any],
    *,
    edge_usable: bool,
    vertical: bool = False,
) -> dict[str, Any]:
    """Require the ideal and executable FK poses to satisfy the same gates."""

    def pose_passes(metrics: dict[str, Any]) -> tuple[bool, str]:
        raw = metrics.get("raw_surface_evidence", {})
        if raw.get("required") and not raw.get("ok"):
            return False, "raw_surface_missing_in_exact_opening"
        if (
            metrics.get("support_clearance_m") is not None
            and float(metrics["support_clearance_m"]) < 0.0005
        ):
            return False, "analytic_support_plane_collision"
        if float(metrics["overlap_vol_cm3"]) > ORIGINAL_OVERLAP_MAX_CM3 + 1e-12:
            return False, "overlap"
        if float(metrics["inflated_overlap_vol_cm3"]) >= INFLATED_OVERLAP_MAX_CM3:
            return False, "inflated_overlap"
        if vertical and (
            int(metrics["grasp_vox"]) < VERTICAL_MIN_OPENING_VOXELS
            or float(metrics.get("opening_transverse_span_m", 0.0))
            < VERTICAL_MIN_SURFACE_SPAN_M
            or int(metrics.get("opening_contact_column_count", 0)) < 2
        ):
            return False, "vertical_opening_geometry"
        if (
            not vertical
            and edge_usable
            and (
                int(metrics["grasp_vox"]) < EDGE_MIN_OPENING_VOXELS
                or float(metrics["opening_axial_span_m"]) < AXIAL_SPAN_MIN_M
                or bool(metrics["tip_only"])
            )
        ):
            return False, "opening_geometry"
        return True, "ok"

    ideal_ok, ideal_reason = pose_passes(ideal)
    final_fk_ok, final_fk_reason = pose_passes(final_fk)
    robust = {
        "grasp_vox": min(int(ideal["grasp_vox"]), int(final_fk["grasp_vox"])),
        "grasp_vol_cm3": min(
            float(ideal["grasp_vol_cm3"]),
            float(final_fk["grasp_vol_cm3"]),
        ),
        "overlap_vox": max(int(ideal["overlap_vox"]), int(final_fk["overlap_vox"])),
        "overlap_vol_cm3": max(
            float(ideal["overlap_vol_cm3"]),
            float(final_fk["overlap_vol_cm3"]),
        ),
        "inflated_overlap_vox": max(
            int(ideal["inflated_overlap_vox"]),
            int(final_fk["inflated_overlap_vox"]),
        ),
        "inflated_overlap_vol_cm3": max(
            float(ideal["inflated_overlap_vol_cm3"]),
            float(final_fk["inflated_overlap_vol_cm3"]),
        ),
    }
    return {
        "ok": bool(ideal_ok and final_fk_ok),
        "ideal_ok": bool(ideal_ok),
        "ideal_reason": ideal_reason,
        "final_q_fk_ok": bool(final_fk_ok),
        "final_q_fk_reason": final_fk_reason,
        "ideal": dict(ideal),
        "final_q_fk": dict(final_fk),
        "robust": robust,
        "selection_grasp_semantics": "min_ideal_and_final_q_fk",
        "selection_overlap_semantics": "both_poses_must_pass",
    }


def _collision_metrics_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["overlap_vol_cm3"] <= ORIGINAL_OVERLAP_MAX_CM3 + 1e-12
        and metrics["inflated_overlap_vol_cm3"] < INFLATED_OVERLAP_MAX_CM3
        and (
            metrics.get("support_clearance_m") is None
            or metrics["support_clearance_m"] >= 0.0005
        )
    )


def _vertical_geometry_rejection_reasons(pose, *, closing_axis_required):
    """Actual vertical gates, distinct from collision and from diagnostic spans."""
    reasons = []
    if pose["grasp_vox"] < VERTICAL_MIN_OPENING_VOXELS:
        reasons.append("insufficient_surface_opening_voxels")
    if pose["opening_transverse_span_m"] < VERTICAL_MIN_SURFACE_SPAN_M:
        reasons.append("insufficient_transverse_surface_span")
    if pose["opening_contact_column_count"] < 2:
        reasons.append("insufficient_contact_columns")
    if pose["table_lift_m"] > EDGE_MAX_TABLE_LIFT_M:
        reasons.append("excessive_table_lift")
    raw = pose["raw_surface_evidence"]
    if raw["required"] and not raw.get("ok"):
        reasons.append("raw_surface_missing_in_exact_opening")
    if not _observed_closing_axis_pass(
        pose["closure_alignment_deg"],
        required=closing_axis_required,
    ):
        reasons.append("observed_closing_axis_misaligned")
    return reasons


def _refine_transverse_candidates(candidates, score_pose, *, closing_axis_required):
    """Repair colliding wall poses when centered evidence is absent or tip-only.

    Centering the opening on a visible wall is not a geometric requirement:
    the opposite open finger can collide with a second wall. Translate along
    the unchanged closing direction within the existing local anchor trust
    radius, keeping the observed anchor inside the exact opening. This does
    not reconstruct a backside or relax collision/IK gates. At most one
    initially colliding slot per orientation is replaced, preserving 1120
    public slots; every extra score and displaced center is audited. A valid
    but tip-only centered pose must not suppress a deeper search. Such poses
    stay untouched, and only an initially colliding, non-tip-only slot can be
    improved in that case. The existing tip diagnostic is a search trigger,
    not a new rejection threshold or a claim of contact/material thickness.
    """
    audit = {
        "mode": "observed_wall_transverse_collision_repair",
        "applied": False,
        "reason": None,
        "resolution_m": VOXEL_M,
        "max_displacement_m": VERTICAL_ANCHOR_TRUST_M,
        "base_candidate_count": len(candidates),
        "orientation_count": 0,
        "extra_geometry_evaluations": 0,
        "replacement_count": 0,
        "replacements": [],
        "collision_thresholds_relaxed": False,
        "anchor_changed": False,
        "approach_changed": False,
        "object_identity_used": False,
        "centered_passing_count": 0,
        "centered_non_tip_passing_count": 0,
        "tip_only_search_triggered": False,
        "existing_passing_candidates_preserved": True,
    }
    if not closing_axis_required:
        audit["reason"] = "no_consistent_observed_wall_axis"
        return list(candidates), audit
    centered = [
        pose
        for pose in candidates
        if _collision_metrics_pass(pose)
        and not _vertical_geometry_rejection_reasons(
            pose,
            closing_axis_required=True,
        )
    ]
    non_tip_centered = [pose for pose in centered if not pose.get("tip_only", False)]
    audit["centered_passing_count"] = len(centered)
    audit["centered_non_tip_passing_count"] = len(non_tip_centered)
    if non_tip_centered:
        audit["reason"] = "centered_candidate_already_passes"
        return list(candidates), audit
    deeper_search = bool(centered)
    audit["tip_only_search_triggered"] = deeper_search

    groups = {}
    for index, pose in enumerate(candidates):
        if deeper_search and (
            _collision_metrics_pass(pose) or pose.get("tip_only", False)
        ):
            # Preserve every previously passing slot; only supplement it with
            # deeper observed geometry by repairing a colliding slot.
            continue
        if not _vertical_geometry_rejection_reasons(pose, closing_axis_required=True):
            groups.setdefault((pose["ni"], pose["ri"]), []).append(index)
    result = list(candidates)
    steps = int(math.floor(VERTICAL_ANCHOR_TRUST_M / VOXEL_M))
    for indices in groups.values():
        index = min(
            indices,
            key=lambda i: (
                candidates[i]["inflated_overlap_vox"],
                candidates[i]["overlap_vox"],
                -candidates[i]["grasp_vox"],
                candidates[i]["ai"],
            ),
        )
        base = candidates[index]
        audit["orientation_count"] += 1
        best, best_quality = None, None
        rotation = np.asarray(base["R"])
        position = np.asarray(base["eef_pos"])
        anchor = np.asarray(base["anchor"])
        for step in range(1, steps + 1):
            for sign in (-1.0, 1.0):
                offset = sign * step * VOXEL_M
                proposed = position + offset * rotation[:, 1]
                metrics = score_pose(rotation, proposed)
                audit["extra_geometry_evaluations"] += 1
                raw = metrics["raw_surface_evidence"]
                if not raw.get("anchor_inside_exact_opening", False):
                    continue
                if not _volume_validation(
                    metrics, metrics, edge_usable=False, vertical=True
                )["ok"]:
                    continue
                if deeper_search and metrics.get("tip_only", True):
                    continue
                quality = (
                    int(
                        raw.get("undershoot_surviving_point_count", 0)
                        >= RAW_SURFACE_MIN_POINTS
                    ),
                    metrics["grasp_vox"],
                    -metrics["inflated_overlap_vox"],
                    -abs(offset),
                )
                if best_quality is None or quality > best_quality:
                    best_quality = quality
                    best = {
                        **base,
                        **metrics,
                        "eef_pos": proposed,
                        "anchor_local": (anchor - proposed) @ rotation,
                        "transverse_refinement": {
                            "offset_along_closing_axis_m": offset,
                            "center_translation_world_m": (
                                proposed - position
                            ).tolist(),
                            "original_eef_pos": position.tolist(),
                            "anchor_changed": False,
                            "rotation_changed": False,
                        },
                    }
        if best is not None:
            result[index] = best
            audit["replacement_count"] += 1
            audit["replacements"].append(
                {
                    "ni": base["ni"],
                    "ri": base["ri"],
                    "ai": base["ai"],
                    **best["transverse_refinement"],
                    "original_overlap_cm3_before": base["overlap_vol_cm3"],
                    "original_overlap_cm3_after": best["overlap_vol_cm3"],
                    "inflated_overlap_cm3_before": base["inflated_overlap_vol_cm3"],
                    "inflated_overlap_cm3_after": best["inflated_overlap_vol_cm3"],
                    "grasp_vol_cm3_after": best["grasp_vol_cm3"],
                    "raw_surface_evidence": best["raw_surface_evidence"],
                }
            )
    audit["applied"] = bool(audit["replacement_count"])
    audit["reason"] = (
        "observed_wall_offset_recovered"
        if audit["applied"]
        else "no_safe_anchor_preserving_offset"
    )
    return result, audit


def _drop_robot_collision_cells(occupancy, snapshot) -> int:
    """Remove depth-matched robot cells from the final-pose collision prior.

    The path guard already withholds these points: they are the arms at the
    capture, not a static obstacle the moving arm must thread. Cells that do
    not project onto that mask stay occupied. Occluded space is not filled in.
    """
    from tools.util.x5_kinematics import (
        invert_transform,
        pose_to_transform,
        x5_forward_kinematics,
    )
    from tools.util.rgbd_scene_mesh_robodojo_v1 import VoxelOccupancy, _project_world
    from tools.util.x5_grasp_geometry import x5_gripper_joint_position_m
    from tools.util.x5_rgbd_path_guard import _rigid_transform
    from tools.util.x5_robot_observation import observed_robot_mask

    collision = occupancy.collision
    if not len(collision.occupied_indices):
        return 0
    current_q, bases, fingers = {}, {}, {}
    for arm in ("left", "right"):
        joints = np.asarray(snapshot.robot[f"arm_{arm}_qpos"], dtype=np.float64)
        pose = snapshot.robot[f"eef_{arm}"]
        current_q[arm] = joints
        bases[arm] = _rigid_transform(
            pose_to_transform(np.r_[pose["pos"], pose["quat"]])
            @ invert_transform(x5_forward_kinematics(joints))
        )
        fingers[arm] = x5_gripper_joint_position_m(
            np.asarray(snapshot.robot[f"gripper_{arm}_qpos"]).reshape(-1)
        )
    mask, _audit = observed_robot_mask(snapshot, bases, current_q, fingers)
    centers = (
        collision.origin_world
        + (collision.occupied_indices.astype(np.float64) + 0.5) * collision.voxel_size_m
    )
    uv, distance = _project_world(
        centers,
        np.asarray(snapshot.intrinsic_matrix),
        np.asarray(snapshot.extrinsic_matrix),
    )
    height, width = mask.shape
    columns = np.rint(uv[:, 0]).astype(np.int64)
    rows = np.rint(uv[:, 1]).astype(np.int64)
    valid = (
        np.isfinite(distance)
        & (distance > 0)
        & (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    on_robot = np.zeros(len(centers), dtype=bool)
    selected = np.flatnonzero(valid)
    on_robot[selected] = mask[rows[selected], columns[selected]]
    removed = int(on_robot.sum())
    if not removed:
        return 0
    occupancy.collision = VoxelOccupancy(
        voxel_size_m=collision.voxel_size_m,
        origin_world=collision.origin_world,
        occupied_indices=collision.occupied_indices[~on_robot],
        source=str(collision.source) + "_robot_withheld",
        query_radius_cells=0,
    )
    occupancy.collision_query_radius_cells = 0
    anchor = occupancy.target_anchor
    if anchor is not None:
        occupancy.collision.prepare_query_region(anchor, 0.40, query_radius_cells=0)
    return removed


def _roll_about_approach(rotation, delta_deg):
    """Rotate the opening about the existing approach without changing tilt."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if not delta_deg:
        return rotation
    delta = math.radians(float(delta_deg))
    cosine, sine = math.cos(delta), math.sin(delta)
    x_axis, y_axis, z_axis = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    return np.column_stack(
        (
            x_axis,
            cosine * y_axis + sine * z_axis,
            -sine * y_axis + cosine * z_axis,
        )
    )


def _refine_in_opening_center(
    candidates, score_pose, support=None, occupancy=None, snapshot=None
):
    """Sample an in-opening center when every centered pose collides.

    The public roll and tilt grid keeps the click on the opening centerline.
    A foreground shell can fill that line while a shift of a few centimeters
    still leaves the click inside the fingers and the tilt inside the cone.
    Search that shift at the existing voxel size, first along opening Y and
    then along opening Z, and replace at most one slot per orientation.
    """
    audit = {
        "mode": "in_opening_center_shift_when_centered_collision",
        "applied": False,
        "reason": None,
        "resolution_m": VOXEL_M,
        "y_limit_m": 14 * VOXEL_M,
        "z_limit_m": 8 * VOXEL_M,
        "roll_step_deg": 15,
        "base_candidate_count": len(candidates),
        "orientation_count": 0,
        "extra_geometry_evaluations": 0,
        "replacement_count": 0,
        "replacements": [],
        "collision_thresholds_relaxed": False,
        "anchor_changed": False,
        "approach_changed": False,
        "object_identity_used": False,
        "collision_query_radius_cells": 0,
        "closest_overlap_cm3": None,
        "closest_inflated_cm3": None,
        "closest_reject_reason": None,
    }
    if any(_collision_metrics_pass(pose) for pose in candidates):
        audit["reason"] = "centered_collision_candidate_exists"
        return list(candidates), audit
    result = list(candidates)
    groups = {}
    for index, pose in enumerate(candidates):
        if pose["grasp_vox"] < VERTICAL_MIN_OPENING_VOXELS:
            continue
        if pose["table_lift_m"] > EDGE_MAX_TABLE_LIFT_M:
            continue
        if not pose["raw_surface_evidence"].get("ok"):
            continue
        # One seed per approach. Roll is searched below, so the centered yaw
        # that happens to collide least must not hide another approach.
        groups.setdefault(pose["ni"], []).append(index)
    ranked = [
        min(
            indices,
            key=lambda i: (
                candidates[i]["inflated_overlap_vox"],
                candidates[i]["overlap_vox"],
                -candidates[i]["grasp_vox"],
                candidates[i]["ai"],
            ),
        )
        for indices in groups.values()
    ]

    def _install(index, base, rotation, position, metrics, **shift):
        anchor = np.asarray(base["anchor"])
        result[index] = {
            **base,
            **metrics,
            "R": rotation,
            "quat": matrix_to_quaternion(rotation),
            "eef_pos": position,
            "gripper_vector_world": -np.asarray(rotation)[:, 0],
            "anchor_local": (anchor - position) @ rotation,
            "in_opening_center_shift": {
                "anchor_changed": False,
                "approach_changed": False,
                **shift,
            },
        }
        audit["replacement_count"] = 1
        audit["applied"] = True
        audit["replacements"].append(
            {
                "ni": base["ni"],
                "ri": base["ri"],
                "ai": base["ai"],
                "overlap_cm3_after": metrics["overlap_vol_cm3"],
                "inflated_overlap_cm3_after": metrics["inflated_overlap_vol_cm3"],
                "grasp_vol_cm3_after": metrics["grasp_vol_cm3"],
                **shift,
            }
        )

    if snapshot is not None and occupancy is not None:
        audit["robot_collision_cells_withheld"] = _drop_robot_collision_cells(
            occupancy,
            snapshot,
        )
        rescored = []
        for index in ranked:
            base = candidates[index]
            rotation = np.asarray(base["R"])
            position = np.asarray(base["eef_pos"])
            metrics = score_pose(rotation, position)
            audit["extra_geometry_evaluations"] += 1
            rescored.append(
                (metrics["overlap_vol_cm3"], metrics["inflated_overlap_vol_cm3"], index)
            )
            raw = metrics["raw_surface_evidence"]
            if (
                raw.get("anchor_inside_exact_opening")
                and _volume_validation(
                    metrics,
                    metrics,
                    edge_usable=False,
                    vertical=True,
                )["ok"]
            ):
                _install(
                    index,
                    base,
                    rotation,
                    position,
                    metrics,
                    robot_body_withheld=True,
                    rotation_changed=False,
                )
                audit["reason"] = "robot_body_withheld_from_final_pose_collision"
                return result, audit
        rescored.sort()
        ranked = [index for _overlap, _inflated, index in rescored[:3]]

    def measure(rotation, position, offset_y, offset_z):
        proposed = position + offset_y * rotation[:, 1] + offset_z * rotation[:, 2]
        if support is not None:
            normal, plane_z, visual = support
            clearance = float(
                np.min(
                    (visual @ rotation.T + proposed - np.array([0.0, 0.0, plane_z]))
                    @ normal
                )
            )
            extra = max(0.0, VERTICAL_TABLE_CLEARANCE_M - clearance)
            if extra > EDGE_MAX_TABLE_LIFT_M:
                audit["extra_geometry_evaluations"] += 1
                return proposed, None, False, False
            proposed = proposed + extra * normal
        metrics = score_pose(rotation, proposed)
        audit["extra_geometry_evaluations"] += 1
        raw = metrics["raw_surface_evidence"]
        inside = bool(raw.get("anchor_inside_exact_opening", False))
        validation = _volume_validation(
            metrics,
            metrics,
            edge_usable=False,
            vertical=True,
        )
        ok = inside and validation["ok"]
        if inside and (
            audit["closest_inflated_cm3"] is None
            or metrics["inflated_overlap_vol_cm3"] < audit["closest_inflated_cm3"]
        ):
            audit["closest_inflated_cm3"] = metrics["inflated_overlap_vol_cm3"]
            audit["closest_overlap_cm3"] = metrics["overlap_vol_cm3"]
            audit["closest_reject_reason"] = None if ok else validation["ideal_reason"]
        return proposed, metrics, ok, inside

    found = None
    watched = None
    for index in ranked:
        base = candidates[index]
        audit["orientation_count"] += 1
        rotation = np.asarray(base["R"])
        position = np.asarray(base["eef_pos"])
        anchor = np.asarray(base["anchor"])
        axial = float(base["axial_x_m"])
        lift_vec = position - (anchor - rotation @ np.array([axial, 0.0, 0.0]))
        for roll_deg in range(0, 180, 15):
            rolled = _roll_about_approach(rotation, roll_deg)
            center = anchor - rolled @ np.array([axial, 0.0, 0.0]) + lift_vec
            for step in range(-12, 13, 2):
                if step == 0 and roll_deg == 0:
                    continue
                proposed, metrics, ok, inside = measure(
                    rolled,
                    center,
                    step * VOXEL_M,
                    0.0,
                )
                if metrics is not None and (
                    watched is None or metrics["overlap_vol_cm3"] < watched[0]
                ):
                    watched = (
                        metrics["overlap_vol_cm3"],
                        index,
                        base,
                        rolled,
                        center,
                        step * VOXEL_M,
                    )
                if ok:
                    found = (
                        index,
                        base,
                        proposed,
                        metrics,
                        step * VOXEL_M,
                        0.0,
                        rolled,
                        roll_deg,
                    )
                    break
            if found is not None:
                break
        if found is not None:
            break
    if found is None and watched is not None:
        # One local 3 mm patch around the nearest coarse sample. This is the
        # whole remaining budget, not another full cone.
        _overlap, index, base, seed_rotation, seed_center, seed_y = watched
        for roll_deg in (0, 15, -15):
            if found is not None:
                break
            rolled = _roll_about_approach(seed_rotation, roll_deg)
            center = seed_center
            for step_y in range(-2, 3):
                if found is not None:
                    break
                for step_z in range(-6, 7):
                    proposed, metrics, ok, inside = measure(
                        rolled,
                        center,
                        seed_y + step_y * VOXEL_M,
                        step_z * VOXEL_M,
                    )
                    if ok:
                        found = (
                            index,
                            base,
                            proposed,
                            metrics,
                            step_y * VOXEL_M,
                            step_z * VOXEL_M,
                            rolled,
                            roll_deg,
                        )
                        break
    if found is not None:
        index, base, proposed, metrics, offset_y, offset_z, used_rotation, roll_deg = (
            found
        )
        anchor = np.asarray(base["anchor"])
        result[index] = {
            **base,
            **metrics,
            "R": used_rotation,
            "quat": matrix_to_quaternion(used_rotation),
            "eef_pos": proposed,
            "gripper_vector_world": -used_rotation[:, 0],
            "anchor_local": (anchor - proposed) @ used_rotation,
            "in_opening_center_shift": {
                "offset_y_m": float(offset_y),
                "offset_z_m": float(offset_z),
                "roll_delta_deg": float(roll_deg),
                "original_eef_pos": np.asarray(base["eef_pos"]).tolist(),
                "anchor_changed": False,
                "approach_changed": False,
                "rotation_changed": bool(roll_deg),
            },
        }
        audit["replacement_count"] += 1
        audit["replacements"].append(
            {
                "ni": base["ni"],
                "ri": base["ri"],
                "ai": base["ai"],
                "offset_y_m": float(offset_y),
                "offset_z_m": float(offset_z),
                "overlap_cm3_after": metrics["overlap_vol_cm3"],
                "inflated_overlap_cm3_after": metrics["inflated_overlap_vol_cm3"],
                "grasp_vol_cm3_after": metrics["grasp_vol_cm3"],
            }
        )
    audit["applied"] = bool(audit["replacement_count"])
    audit["reason"] = (
        "in_opening_shift_recovered"
        if audit["applied"]
        else "no_in_opening_collision_free_center"
    )
    return result, audit


def _refine_axial_candidates(candidates, score_pose):
    """Resolve collision transitions missed by the seven coarse depths.

    A 12 mm axial interval can contain useful insertion depths even when its
    endpoints are a grazing tip pose and a colliding pose. Refine each such
    interval at the *existing voxel resolution*, for every orientation. Keep
    one representative per original slot so the 20x8x7 public candidate
    contract is unchanged; separately report every extra geometry evaluation.
    No target identity, live feedback, relaxed collision gate or new motion
    is involved. Both IK and final-FK gates still run after this operation.
    """
    result = list(candidates)
    groups = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault((candidate["ni"], candidate["ri"]), []).append(index)
    audit = {
        "mode": "whole_candidate_set_collision_transition_voxel_refinement",
        "resolution_m": VOXEL_M,
        "base_candidate_count": len(candidates),
        "extra_geometry_evaluations": 0,
        "interval_count": 0,
        "replacement_count": 0,
        "replacements": [],
        "collision_thresholds_relaxed": False,
        "object_identity_used": False,
    }

    def quality(pose):
        raw = pose.get("raw_surface_evidence", {})
        return (
            int(
                raw.get("undershoot_surviving_point_count", 0) >= RAW_SURFACE_MIN_POINTS
            ),
            pose["grasp_vox"],
            raw.get("undershoot_surviving_point_count", 0),
            raw.get("raw_point_count", 0),
            -pose["inflated_overlap_vox"],
        )

    for indices in groups.values():
        indices.sort(key=lambda i: candidates[i]["axial_x_m"])
        for deep_index, shallow_index in zip(indices, indices[1:]):
            deep, shallow = candidates[deep_index], candidates[shallow_index]
            span = float(shallow["axial_x_m"] - deep["axial_x_m"])
            if (
                not _collision_metrics_pass(shallow)
                or _collision_metrics_pass(deep)
                or span <= VOXEL_M + 1e-10
            ):
                continue
            audit["interval_count"] += 1
            intervals = max(1, int(math.ceil(span / VOXEL_M - 1e-9)))
            best = shallow
            for step in range(1, intervals):
                fraction = step / intervals
                position = (1.0 - fraction) * np.asarray(
                    deep["eef_pos"]
                ) + fraction * np.asarray(shallow["eef_pos"])
                # Equal rotation makes signed plane clearance affine in the
                # interpolated translation. It is independently rescored too.
                metrics = score_pose(shallow["R"], position)
                audit["extra_geometry_evaluations"] += 1
                axial = (1.0 - fraction) * deep["axial_x_m"] + fraction * shallow[
                    "axial_x_m"
                ]
                trial = {
                    **shallow,
                    **metrics,
                    "eef_pos": position,
                    "axial_x_m": axial,
                    "anchor_local": np.array([axial, 0.0, 0.0]),
                    "table_lift_m": (1.0 - fraction) * deep["table_lift_m"]
                    + fraction * shallow["table_lift_m"],
                    "axial_refinement": {
                        "coarse_axial_x_m": shallow["axial_x_m"],
                        "interval_m": [deep["axial_x_m"], shallow["axial_x_m"]],
                        "fraction": fraction,
                        "resolution_m": VOXEL_M,
                    },
                }
                if (
                    trial["table_lift_m"] <= EDGE_MAX_TABLE_LIFT_M
                    and _volume_validation(
                        metrics, metrics, edge_usable=False, vertical=True
                    )["ok"]
                    and quality(trial) > quality(best)
                ):
                    best = trial
            if best is not shallow:
                result[shallow_index] = best
                audit["replacement_count"] += 1
                audit["replacements"].append(
                    {
                        "ni": shallow["ni"],
                        "ri": shallow["ri"],
                        "ai": shallow["ai"],
                        **best["axial_refinement"],
                        "selected_axial_x_m": best["axial_x_m"],
                        "before_grasp_vol_cm3": shallow["grasp_vol_cm3"],
                        "after_grasp_vol_cm3": best["grasp_vol_cm3"],
                        "after_inflated_overlap_vol_cm3": best[
                            "inflated_overlap_vol_cm3"
                        ],
                        "raw_surface_evidence": best["raw_surface_evidence"],
                    }
                )
    return result, audit


def _paired_ik_seed_candidates(current_q: np.ndarray):
    """Bounded numerical restarts, independent of object/task identities.

    The capture remains the calibration reference for every trial. Only the
    solver initial iterate changes; no robot state or signed target is changed.
    """
    seeds = [
        ("capture_joints", np.asarray(current_q, dtype=np.float64).copy()),
        ("neutral", np.zeros(6, dtype=np.float64)),
    ]
    for sign, name in ((-1.0, "elbow_negative"), (1.0, "elbow_positive")):
        seed = np.asarray(current_q, dtype=np.float64).copy()
        seed[1:4] += sign * np.array([0.8, 1.6, 0.8])
        seeds.append((name, seed))
    unique = []
    for name, seed in seeds:
        if not any(
            np.allclose(seed, previous, rtol=0.0, atol=1e-6) for _, previous in unique
        ):
            unique.append((name, seed))
    return unique


def _refine_paired_ik_solution(current_q, current_pose, target, solution):
    """Finish sub-voxel convergence before strict FK occupancy is evaluated.

    This never relaxes an acceptance gate or changes the intended target. A
    failed refinement retains the original solution for the ordinary FK gate.
    """
    needed = bool(
        solution.converged
        and (
            solution.position_error_m > IK_REFINEMENT_POS_TOL_M
            or solution.orientation_error_deg > IK_REFINEMENT_ORI_TOL_DEG
        )
    )
    audit = {
        "attempted": needed,
        "applied": False,
        "target_position_tolerance_m": IK_REFINEMENT_POS_TOL_M,
        "target_orientation_tolerance_deg": IK_REFINEMENT_ORI_TOL_DEG,
        "initial_position_error_m": solution.position_error_m,
        "initial_orientation_error_deg": solution.orientation_error_deg,
    }
    if not needed:
        return solution, audit
    refined = solve_calibrated_x5_ik(
        current_q,
        current_pose,
        target,
        initial_joints=solution.joint_target,
        max_iterations=60,
        position_tolerance_m=IK_REFINEMENT_POS_TOL_M,
        orientation_tolerance_deg=IK_REFINEMENT_ORI_TOL_DEG,
        damping=0.018,
    )
    joint_change = float(np.max(np.abs(refined.joint_target - solution.joint_target)))
    accepted = bool(refined.converged and joint_change <= 0.25)
    audit.update(
        applied=accepted,
        converged=bool(refined.converged),
        joint_change_inf_rad=joint_change,
        refined_position_error_m=refined.position_error_m,
        refined_orientation_error_deg=refined.orientation_error_deg,
    )
    return (refined if accepted else solution), audit


def _solve_paired_arm_ik(
    current_q, current_pose, final_pose, safe_pose, *, finger_position_m=0.044
):
    """Keep the first valid same-branch final/overhead pair and audit failures."""
    trials = []
    baseline = None
    for name, seed in _paired_ik_seed_candidates(current_q):
        final = solve_calibrated_x5_ik(
            current_q,
            current_pose,
            final_pose,
            initial_joints=seed,
            max_iterations=120,
            position_tolerance_m=FINAL_IK_POS_TOL_M,
            orientation_tolerance_deg=FINAL_IK_ORI_TOL_DEG,
            damping=0.018,
        )
        final, final_refinement = _refine_paired_ik_solution(
            current_q,
            current_pose,
            final_pose,
            final,
        )
        info = {
            "ok": False,
            "reason": "final_ik",
            "final_position_error_m": final.position_error_m,
            "final_orientation_error_deg": final.orientation_error_deg,
            "final_refinement": final_refinement,
        }
        safe = None
        if final.converged:
            safe = solve_calibrated_x5_ik(
                current_q,
                current_pose,
                safe_pose,
                initial_joints=final.joint_target,
                max_iterations=120,
                position_tolerance_m=SAFE_IK_POS_TOL_M,
                orientation_tolerance_deg=SAFE_IK_ORI_TOL_DEG,
                damping=0.018,
            )
            safe, safe_refinement = _refine_paired_ik_solution(
                current_q,
                current_pose,
                safe_pose,
                safe,
            )
            gap = float(np.max(np.abs(safe.joint_target - final.joint_target)))
            info.update(
                ok=bool(safe.converged and gap <= SAFE_FINAL_BRANCH_GAP_RAD),
                reason=(
                    "ok"
                    if safe.converged and gap <= SAFE_FINAL_BRANCH_GAP_RAD
                    else "safe_final_branch_gap"
                    if safe.converged
                    else "safe_ik"
                ),
                safe_position_error_m=safe.position_error_m,
                safe_orientation_error_deg=safe.orientation_error_deg,
                safe_final_branch_gap_rad=gap,
                safe_refinement=safe_refinement,
            )
            if info["ok"]:
                self_collision = check_paired_path(
                    current_q,
                    safe.joint_target,
                    final.joint_target,
                    finger_position_m=finger_position_m,
                )
                info["self_collision"] = self_collision
                if not self_collision["ok"]:
                    info.update(ok=False, reason="robot_self_collision_envelope")
        trials.append({"seed": name, **info})
        if baseline is None or (not baseline[0].converged and final.converged):
            # Preserve a geometric rejection over an earlier numerical miss;
            # otherwise all retries could hide the actual collision reason.
            baseline = (final, safe, dict(info))
        if info["ok"]:
            return (
                final,
                safe,
                info,
                {
                    "mode": "bounded_deterministic_paired_restart",
                    "selected_seed": name,
                    "trial_count": len(trials),
                    "trials": trials,
                    "capture_calibration_unchanged": True,
                    "tolerances_relaxed": False,
                },
            )
    final, safe, info = baseline
    return (
        final,
        safe,
        info,
        {
            "mode": "bounded_deterministic_paired_restart",
            "selected_seed": None,
            "trial_count": len(trials),
            "trials": trials,
            "capture_calibration_unchanged": True,
            "tolerances_relaxed": False,
        },
    )


def _solve_candidate_ik(
    pose: dict[str, Any],
    *,
    robot: dict[str, Any],
    active_arms: tuple[str, ...],
) -> dict[str, Any]:
    final_position = np.asarray(pose["eef_pos"], dtype=np.float64)
    final_pose = np.concatenate((final_position, pose["quat"]))
    # The same overhead pose is the requested post-close world-Z lift goal.
    # Paired IK therefore checks both approach and the complete +0.10 m lift.
    safe_position = final_position + np.array([0.0, 0.0, SAFE_PLAN_BACK_M])
    safe_pose = np.concatenate((safe_position, pose["quat"]))
    by_arm: dict[str, Any] = {}
    for arm in active_arms:
        if (
            float(np.linalg.norm(final_position - X5_SHOULDERS[arm])) > 1.05
            or float(np.linalg.norm(safe_position - X5_SHOULDERS[arm])) > 1.05
        ):
            by_arm[arm] = {"ok": False, "reason": "conservative_reach_bound"}
            continue
        current_q, current_pose = _capture_arm_state(robot, arm)
        final, safe, pair_info, seed_audit = _solve_paired_arm_ik(
            current_q,
            current_pose,
            final_pose,
            safe_pose,
            finger_position_m=x5_gripper_joint_position_m(
                robot.get(f"gripper_{arm}_qpos", [1.0]),
            ),
        )
        if not final.converged:
            by_arm[arm] = {**pair_info, "seed_audit": seed_audit}
            continue
        branch_gap = float(np.max(np.abs(safe.joint_target - final.joint_target)))
        final_fk_pose = calibrated_pose_for_joints(
            current_q,
            current_pose,
            final.joint_target,
        )
        safe_fk_pose = calibrated_pose_for_joints(
            current_q,
            current_pose,
            safe.joint_target,
        )
        by_arm[arm] = {
            "ok": pair_info["ok"],
            "reason": pair_info["reason"],
            "seed_audit": seed_audit,
            "self_collision": pair_info.get("self_collision"),
            "final_q": final.joint_target.tolist(),
            "safe_q": safe.joint_target.tolist(),
            "final_position_error_m": final.position_error_m,
            "final_orientation_error_deg": final.orientation_error_deg,
            "safe_position_error_m": safe.position_error_m,
            "safe_orientation_error_deg": safe.orientation_error_deg,
            "final_joint_limit_margin_rad": _joint_limit_margin(final.joint_target),
            "safe_joint_limit_margin_rad": _joint_limit_margin(safe.joint_target),
            "final_iterations": final.iterations,
            "safe_iterations": safe.iterations,
            "safe_final_branch_gap_rad": branch_gap,
            "final_q_fk_pose": final_fk_pose.tolist(),
            "safe_q_fk_pose": safe_fk_pose.tolist(),
            "ik_allerr": (
                1000.0 * (final.position_error_m + safe.position_error_m)
                + final.orientation_error_deg
                + safe.orientation_error_deg
            ),
            "calibrated_base_translation_m": final.calibration_translation_m.tolist(),
        }
    return {
        "safe_position": safe_position,
        "safe_pose": safe_pose,
        "by_arm": by_arm,
    }


def _plan_x5_grasp_in_downward_cone(
    *,
    snapshot: Snapshot,
    u: float,
    v: float,
    plan_arm: str = "any",
    seed: int = 42,
    return_scene_mesh_result: bool = False,
    max_downward_tilt_deg: float = MAX_DOWNWARD_TILT_DEG,
) -> dict[str, Any]:
    started = time.perf_counter()
    if snapshot.depth is None:
        raise ValueError("frozen capture has no depth")
    if snapshot.intrinsic_matrix is None or snapshot.extrinsic_matrix is None:
        raise ValueError("frozen capture has no camera matrices")
    depth = _finite_depth(snapshot.depth)
    intrinsic = np.asarray(snapshot.intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(snapshot.extrinsic_matrix, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(extrinsic)):
        raise ValueError("frozen camera matrices must be finite")
    requested_arm = str(plan_arm or "any").strip().lower()
    if requested_arm not in ("left", "right", "any"):
        raise ValueError("plan_arm must be left, right, or any")
    input_pixel_u, input_pixel_v = relative_uv_to_pixel(
        u, v, depth.shape[1], depth.shape[0]
    )
    clicked_hit, click_observation = unproject_relative_uv(
        depth, intrinsic, extrinsic, u, v
    )
    hit, pixel_u, pixel_v, surface_anchor_audit = select_nearest_surface_anchor(
        depth,
        intrinsic,
        extrinsic,
        clicked_hit,
        input_pixel_u,
        input_pixel_v,
        preserve_click=True,
    )
    point_observation = {
        **click_observation,
        "pixel_uv": [int(pixel_u), int(pixel_v)],
        "relative_uv": list(surface_anchor_audit["selected_relative_uv"]),
        "depth_m": float(surface_anchor_audit["selected_depth_m"]),
        "selection": surface_anchor_audit["selection"],
        "original_click": click_observation,
        "surface_anchor": surface_anchor_audit,
    }
    outward_normal, normal_confidence, normal_audit = estimate_outward_normal(
        depth, intrinsic, extrinsic, pixel_u, pixel_v
    )

    geometry = x5_gripper_geometry()
    support_z, support_plane_audit = estimate_support_plane_z(
        depth, intrinsic, extrinsic
    )
    edge_approach, edge_audit = estimate_depth_edge_frame(
        depth,
        intrinsic,
        extrinsic,
        pixel_u,
        pixel_v,
        support_z=support_z,
    )
    edge_usable = edge_approach is not None
    if support_z is None:
        raise X5GraspPlanningError(
            "geometry=uncertain_support_plane; cannot establish downward approach"
        )
    support_normal = np.asarray(support_plane_audit["normal_world"], dtype=np.float64)
    approach_specs = _approach_specs(
        edge_approach,
        support_normal,
        max_tilt_deg=max_downward_tilt_deg,
    )
    target_anchor, closing_axis, contact_patch_audit = _vertical_contact_patch(
        depth,
        intrinsic,
        extrinsic,
        hit,
        pixel_u,
        pixel_v,
        support_normal,
        support_z,
    )
    occupancy, mesh_result, effective_mesh_version, mesh_fallback_reason = (
        _build_scene_occupancy(
            snapshot,
            support_z=support_z,
            target_anchor=target_anchor,
        )
    )
    mesh_provenance = {
        "effective_version": effective_mesh_version,
        "fallback_reason": mesh_fallback_reason,
        **(
            {
                key: mesh_result.metadata.get(key)
                for key in (
                    "build",
                    "mesh_source_sha256",
                    "mesh_revision_binding",
                    "allowed_inputs",
                    "forbidden_inputs_used",
                    "normal_estimation",
                )
            }
            if mesh_result is not None
            else {}
        ),
    }
    robot_material_audit = {
        "applied": False,
        "reason": "explicit_legacy_or_mesh_fallback_route",
    }
    if mesh_result is not None:
        # Whole-view RGB-D includes the observer's own fingers. Their observed
        # band must not vote as graspable material. Keep all collision evidence
        # (including both arms) unchanged: removing a robot ray does not reveal
        # free space behind it or authorize a path through the other arm.
        mesh_result, robot_mask = exclude_observed_gripper_material(
            snapshot, mesh_result
        )
        robot_material_audit = mesh_result.metadata[
            "planner_gripper_material_partition"
        ]
        if (
            robot_mask[input_pixel_v, input_pixel_u]
            or points_on_withheld_rays(
                np.asarray([target_anchor]),
                robot_mask,
                intrinsic,
                extrinsic,
            )[0]
        ):
            raise X5GraspPlanningError(
                "geometry=anchor_on_observed_robot_or_contact_ambiguity; "
                "capture an unoccluded public surface before planning",
                audit={
                    "build": BUILD,
                    "surface_anchor": surface_anchor_audit,
                    "robot_material_partition": robot_material_audit,
                    "mesh_provenance": mesh_provenance,
                },
            )
        occupancy = MeshSceneOccupancy(
            mesh_result,
            support_z=support_z,
            target_anchor=target_anchor,
            target_radius_m=OPENING_TARGET_RADIUS_M,
            support_margin_m=OPENING_SUPPORT_MARGIN_M,
            surface_query_radius_cells=0,
        )
        geometry = exact_x5_planning_geometry()
        _prepare_raw_grasp_points(occupancy)
        mesh_result.metadata["planner_opening_contract"] = {
            "opening_query_version": OPENING_QUERY_VERSION,
            "surface_query_radius_cells": 0,
            "raw_surface_gate": True,
            "grasp_score_is_observed_band_proxy_not_true_material_volume": True,
        }
    wall_normal_evidence, wall_refinement_audit = _adaptive_wall_normal_evidence(
        depth,
        intrinsic,
        extrinsic,
        pixel_u,
        pixel_v,
        normal_audit,
        support_normal,
        enabled=mesh_result is not None,
    )
    closing_axis, closing_axis_mode, closing_axis_audit = _resolve_closing_axis(
        closing_axis,
        contact_patch_audit,
        wall_normal_evidence,
        support_normal,
        use_observed_wall=mesh_result is not None,
    )
    closing_axis_audit["adaptive_normal_evidence"] = wall_refinement_audit
    observed_closing_axis_required = closing_axis_mode == "projected_surface_normal"
    # v1 exposes two occupancy layers.  Grasp material scoring must always
    # query the observed surface layer; the conservative completion shell is
    # reserved for gripper collision overlap.  The v4 rollback retains its
    # historical edge-gated opening behavior below.
    surface_only_opening = (
        bool(mesh_result is not None)
        if mesh_result is not None
        else bool(edge_usable and support_z is not None)
    )
    # The historical table-lift heuristic assumes graspable material is on
    # the positive world-z side of the support. The v1 surface query infers
    # the observed side from the anchor, so only lift that positive-side case.
    support_lift_enabled = bool(
        surface_only_opening
        and support_z is not None
        and float(target_anchor[2]) >= float(support_z) + OPENING_SUPPORT_MARGIN_M
    )
    if isinstance(occupancy, MeshSceneOccupancy):
        # All fixed X5 candidates lie within this anchor-centered bound.  A
        # local neighborhood index keeps the exact one-cell occupancy contract
        # while avoiding 27 global searches for each of the 1,120 poses.
        max_gripper_radius = float(
            np.max(
                np.linalg.norm(
                    np.asarray(geometry["inflated_gripper_voxels"], dtype=np.float64),
                    axis=1,
                )
            )
        )
        max_axial_offset = float(
            np.max(np.abs(np.asarray(geometry["axial_x_m"], dtype=np.float64)))
        )
        # Exact occupied cells only. The inflated finger shell is already
        # dilated by 15 mm; a further 3 mm query halo was rejecting in-cone
        # poses whose exact overlap is under the published limits.
        occupancy.collision_query_radius_cells = 0
        occupancy.prepare_query_region(
            target_anchor,
            max_gripper_radius
            + max_axial_offset
            + float(EDGE_MAX_TABLE_LIFT_M)
            + 0.012,
        )
    mesh_anchor_audit = (
        None
        if mesh_result is None
        else mesh_result.anchor_audit(hit, max_distance_m=max(0.018, 6.0 * VOXEL_M))
    )
    if mesh_anchor_audit is not None and mesh_anchor_audit.get("status") != "ok":
        raise X5GraspPlanningError(
            "RGB-D surface anchor is outside the reconstructed observed surface "
            f"({mesh_anchor_audit.get('nearest_surface_distance_mm')}mm); "
            "geometry=surface_missing",
            audit={
                "build": BUILD,
                "mesh_provenance": mesh_provenance,
                "mesh_anchor": mesh_anchor_audit,
            },
        )
    voxel_cm3 = VOXEL_M**3 * 1e6
    random = np.random.default_rng(int(seed))
    candidates: list[dict[str, Any]] = []
    metric_started = time.perf_counter()
    for approach_index, (approach_source, approach, edge_pitch) in enumerate(
        approach_specs
    ):
        zero_frame = x5_eef_frame_from_approach_roll(approach, 0.0)
        roll_phase = (
            0.0
            if closing_axis is None
            else math.atan2(
                float(closing_axis @ zero_frame[:, 2]),
                float(closing_axis @ zero_frame[:, 1]),
            )
        )
        for roll_index in range(N_ROLL):
            roll = roll_phase + 2.0 * math.pi * roll_index / N_ROLL
            rotation = x5_eef_frame_from_approach_roll(approach, roll)
            quaternion = matrix_to_quaternion(rotation)
            gripper_vector = -rotation[:, 0]
            normal_angle = (
                math.degrees(
                    math.acos(
                        float(np.clip(support_normal @ gripper_vector, -1.0, 1.0))
                    )
                )
                if support_normal is not None
                else float("inf")
            )
            for axial_index, axial_x in enumerate(geometry["axial_x_m"]):
                anchor_local = np.array([float(axial_x), 0.0, 0.0])
                eef_position = target_anchor - rotation @ anchor_local
                table_lift = 0.0
                raw_visual = geometry["visual_vertices"] @ rotation.T + eef_position
                if support_lift_enabled:
                    plane_point = np.array(
                        [
                            0.0,
                            0.0,
                            support_plane_audit["plane_coefficients_z_from_xy"][2],
                        ]
                    )
                    minimum_clearance = float(
                        np.min((raw_visual - plane_point) @ support_normal)
                    )
                    table_lift = max(
                        0.0,
                        VERTICAL_TABLE_CLEARANCE_M - minimum_clearance,
                    )
                    # A large lift means the click/approach is not physically
                    # compatible with the support plane.  Keep the candidate
                    # for deterministic accounting, but it cannot win the
                    # edge robustness filter below.
                if table_lift > 0.0:
                    eef_position = eef_position + table_lift * support_normal
                ideal_metrics = _score_gripper_pose(
                    occupancy=occupancy,
                    geometry=geometry,
                    rotation=rotation,
                    position=eef_position,
                    surface_only_opening=surface_only_opening,
                    target_anchor=target_anchor,
                    voxel_cm3=voxel_cm3,
                )
                candidates.append(
                    {
                        "pi": 0,
                        "ni": approach_index,
                        "ri": roll_index,
                        "ai": axial_index,
                        "approach_source": approach_source,
                        "edge_pitch_rad": edge_pitch,
                        "approach_tilt_deg": float(
                            math.degrees(
                                math.acos(
                                    np.clip(
                                        -float(support_normal @ approach),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        ),
                        "axial_point": f"p{axial_index + 1}",
                        "axial_x_m": float(axial_x),
                        "anchor": target_anchor.copy(),
                        "hit_anchor": hit.copy(),
                        "anchor_local": anchor_local,
                        "eef_pos": eef_position,
                        "R": rotation,
                        "quat": quaternion,
                        "gripper_vector_world": gripper_vector,
                        "normal_alignment_deg": normal_angle,
                        "closure_alignment_deg": _closing_axis_alignment_deg(
                            rotation,
                            closing_axis,
                            support_normal,
                            closing_axis_mode,
                        ),
                        **ideal_metrics,
                        "table_lift_m": float(table_lift),
                        "table_clearance_m": (
                            float(VERTICAL_TABLE_CLEARANCE_M)
                            if support_lift_enabled
                            else None
                        ),
                        "tie_break": float(random.random()),
                    }
                )
    expected_count = 20 * N_ROLL * len(geometry["axial_x_m"])
    if len(candidates) != expected_count:
        raise RuntimeError(
            f"X5 RGB-D lite generated {len(candidates)} poses, expected {expected_count}"
        )
    refinement_audit = {
        "mode": "disabled_for_explicit_legacy_geometry",
        "extra_geometry_evaluations": 0,
    }
    transverse_refinement_audit = {
        "mode": "disabled_for_explicit_legacy_geometry",
        "extra_geometry_evaluations": 0,
    }
    opening_offset_audit = {
        "mode": "disabled_for_explicit_legacy_geometry",
        "extra_geometry_evaluations": 0,
    }
    if mesh_result is not None:
        score_pose = lambda rotation, position: _score_gripper_pose(
            occupancy=occupancy,
            geometry=geometry,
            rotation=rotation,
            position=position,
            surface_only_opening=surface_only_opening,
            target_anchor=target_anchor,
            voxel_cm3=voxel_cm3,
        )
        candidates, refinement_audit = _refine_axial_candidates(candidates, score_pose)
        candidates, transverse_refinement_audit = _refine_transverse_candidates(
            candidates,
            score_pose,
            closing_axis_required=observed_closing_axis_required,
        )
        candidates, opening_offset_audit = _refine_in_opening_center(
            candidates,
            score_pose,
            support=(
                support_normal,
                float(support_plane_audit["plane_coefficients_z_from_xy"][2]),
                np.asarray(geometry["visual_vertices"]),
            ),
            occupancy=occupancy,
            snapshot=snapshot,
        )
    strict_overlap = [pose for pose in candidates if _collision_metrics_pass(pose)]
    edge_robust = [
        pose
        for pose in strict_overlap
        if not _vertical_geometry_rejection_reasons(
            pose,
            closing_axis_required=observed_closing_axis_required,
        )
    ]
    candidate_audit = {
        "build": BUILD,
        "support_plane": support_plane_audit,
        "maximum_candidate_tilt_deg": float(max_downward_tilt_deg),
        "effective_mesh_version": effective_mesh_version,
        "mesh_provenance": mesh_provenance,
        "robot_material_partition": robot_material_audit,
        "surface_anchor": surface_anchor_audit,
        "vertical_contact_patch": contact_patch_audit,
        "closing_axis": closing_axis_audit,
        "axial_refinement": refinement_audit,
        "transverse_refinement": transverse_refinement_audit,
        "in_opening_center_shift": opening_offset_audit,
        "collision_query_radius_cells": (
            getattr(occupancy, "collision_query_radius_cells", None)
            if isinstance(occupancy, MeshSceneOccupancy)
            else None
        ),
        "strict_overlap_count": len(strict_overlap),
        "closing_axis_rejected_count": sum(
            not _observed_closing_axis_pass(
                p["closure_alignment_deg"], required=observed_closing_axis_required
            )
            for p in strict_overlap
        ),
        "vertical_geometry_pass_count": len(edge_robust),
        "opening_query_version": geometry["metadata"].get(
            "opening_query_version", "legacy_voxel_columns"
        ),
        "raw_surface_pass_count": sum(
            p["raw_surface_evidence"].get("ok", False) for p in strict_overlap
        ),
        "vertical_rejection_counts": {
            reason: sum(
                reason
                in _vertical_geometry_rejection_reasons(
                    p,
                    closing_axis_required=observed_closing_axis_required,
                )
                for p in strict_overlap
            )
            for reason in sorted(
                {
                    reason
                    for p in strict_overlap
                    for reason in _vertical_geometry_rejection_reasons(
                        p, closing_axis_required=observed_closing_axis_required
                    )
                }
            )
        },
        "best_surface_candidates": [
            {
                key: pose[key]
                for key in (
                    "ni",
                    "ri",
                    "ai",
                    "grasp_vol_cm3",
                    "overlap_vol_cm3",
                    "inflated_overlap_vol_cm3",
                    "table_lift_m",
                    "eef_pos",
                    "approach_tilt_deg",
                    "opening_transverse_span_m",
                    "opening_contact_column_count",
                    "raw_surface_evidence",
                )
            }
            for pose in sorted(candidates, key=lambda item: -item["grasp_vox"])[:12]
        ],
    }
    if mesh_result is not None and surface_only_opening:
        observed_surface_max = max(
            (int(pose["grasp_vox"]) for pose in strict_overlap),
            default=0,
        )
        if observed_surface_max <= 0:
            if not strict_overlap:
                raise X5GraspPlanningError(
                    "no candidate passed collision gates before surface-volume acceptance; "
                    "geometry=no_collision_free_candidate (not evidence of a missing surface)",
                    audit=candidate_audit,
                )
            raise X5GraspPlanningError(
                "no observed RGB-D surface occupancy lies inside the X5 opening; "
                "geometry=surface_missing (collision shell is not grasp material)",
                audit=candidate_audit,
            )
    # Observed pinch yaw is kept when it still leaves a collision-free pose.
    # If every collision-free pose fails only that yaw test, the wall normal
    # does not get to discard an otherwise valid vertical grasp.
    if not edge_robust and strict_overlap and observed_closing_axis_required:
        edge_robust = [
            pose
            for pose in strict_overlap
            if not _vertical_geometry_rejection_reasons(
                pose,
                closing_axis_required=False,
            )
        ]
        candidate_audit["closing_axis_kept_only_when_a_collision_free_pose_remains"] = (
            True
        )
    # A failed opening/direction gate is not rescued by collision clearance.
    strict = edge_robust
    global_safe_max = max(
        (float(pose["grasp_vol_cm3"]) for pose in strict), default=0.0
    )
    if not strict:
        best = min(
            candidates,
            key=lambda pose: (
                pose["inflated_overlap_vox"],
                pose["overlap_vox"],
                -pose["grasp_vox"],
            ),
        )
        raise X5GraspPlanningError(
            "no RGB-D grasp candidate passed the vertical opening/direction gates "
            f"(collision_safe={len(strict_overlap)}, "
            f"rejections={candidate_audit['vertical_rejection_counts']}); "
            f"(best original={best['overlap_vol_cm3']:.3f}cm3, "
            f"inflated={best['inflated_overlap_vol_cm3']:.3f}cm3, "
            f"grasp={best['grasp_vol_cm3']:.3f}cm3)",
            audit=candidate_audit,
        )
    ik_input, rank_audit = _rank_ik_input(strict, normal_confidence, vertical=True)
    active_arms = (
        (requested_arm,) if requested_arm in ("left", "right") else ("left", "right")
    )
    ik_started = time.perf_counter()
    reachable: list[dict[str, Any]] = []
    ik_failure_counts: dict[str, int] = {}
    final_q_fk_volume_failure_counts: dict[str, int] = {}
    final_q_fk_volume_validation_count = 0
    path_gate = _CandidatePathGate(snapshot, enabled=mesh_result is not None)
    candidate_audit["non_gripper_path_gate"] = path_gate.audit
    for pose in ik_input:
        solved = _solve_candidate_ik(
            pose, robot=dict(snapshot.robot), active_arms=active_arms
        )
        pose["safe_position"] = solved["safe_position"]
        pose["safe_pose"] = solved["safe_pose"]
        pose["ik_filter"] = solved["by_arm"]
        ideal_metrics = {
            key: pose[key]
            for key in (
                "overlap_vox",
                "grasp_vox",
                "inflated_overlap_vox",
                "overlap_vol_cm3",
                "grasp_vol_cm3",
                "inflated_overlap_vol_cm3",
                "overlap_frac",
                "opening_axial_min_m",
                "opening_axial_max_m",
                "opening_axial_span_m",
                "opening_tip_fraction",
                "tip_only",
                "opening_transverse_span_m",
                "opening_contact_column_count",
                "support_clearance_m",
                "raw_surface_evidence",
            )
        }
        for arm, info in solved["by_arm"].items():
            if not bool(info.get("ok")):
                continue
            final_fk_pose = np.asarray(
                info["final_q_fk_pose"], dtype=np.float64
            ).reshape(7)
            fk_rotation = quaternion_to_matrix(final_fk_pose[3:])
            tilt = math.degrees(
                math.acos(
                    float(np.clip(-fk_rotation[:, 0] @ support_normal, -1.0, 1.0))
                )
            )
            info["downward_tilt_deg"] = tilt
            if tilt > max_downward_tilt_deg + DOWNWARD_FK_TILT_TOL_DEG:
                info.update(ok=False, reason="final_q_not_downward")
                continue
            closure_angle = _closing_axis_alignment_deg(
                fk_rotation,
                closing_axis,
                support_normal,
                closing_axis_mode,
            )
            info["closure_alignment_deg"] = closure_angle
            ideal_yaw_ok = _observed_closing_axis_pass(
                pose["closure_alignment_deg"],
                required=observed_closing_axis_required,
            )
            if ideal_yaw_ok and not _observed_closing_axis_pass(
                closure_angle, required=True, fk_tolerance_deg=FINAL_IK_ORI_TOL_DEG
            ):
                info.update(ok=False, reason="final_q_observed_closing_axis_mismatch")
                continue
            final_fk_metrics = _score_gripper_pose(
                occupancy=occupancy,
                geometry=geometry,
                rotation=quaternion_to_matrix(final_fk_pose[3:]),
                position=final_fk_pose[:3],
                surface_only_opening=surface_only_opening,
                target_anchor=target_anchor,
                voxel_cm3=voxel_cm3,
            )
            validation = _volume_validation(
                ideal_metrics,
                final_fk_metrics,
                edge_usable=edge_usable,
                vertical=True,
            )
            final_q_fk_volume_validation_count += 1
            info["volume_validation"] = validation
            if not bool(validation["ok"]):
                info["ok"] = False
                failed_pose = "ideal" if not validation["ideal_ok"] else "final_q_fk"
                info["reason"] = (
                    f"{failed_pose}_volume_{validation[f'{failed_pose}_reason']}"
                )
                final_q_fk_volume_failure_counts[info["reason"]] = (
                    final_q_fk_volume_failure_counts.get(info["reason"], 0) + 1
                )
        # A correct endpoint opening does not certify the camera/arm path.
        # Use the same full-view frozen observation for every arm/candidate;
        # no new capture, object crop, identity or simulator query is allowed.
        try:
            for arm, info in solved["by_arm"].items():
                path_gate.validate(arm, info)
        except X5GraspPlanningError as error:
            raise X5GraspPlanningError(
                str(error),
                audit={**candidate_audit, **error.audit},
            ) from error
        valid_arms = [
            arm for arm, info in solved["by_arm"].items() if bool(info.get("ok"))
        ]
        if valid_arms:
            preferred_arm = min(
                valid_arms,
                key=lambda arm: (
                    -float(
                        solved["by_arm"][arm]["volume_validation"]["robust"][
                            "grasp_vol_cm3"
                        ]
                    ),
                    float(
                        solved["by_arm"][arm]["volume_validation"]["robust"][
                            "inflated_overlap_vol_cm3"
                        ]
                    ),
                    float(
                        solved["by_arm"][arm]["volume_validation"]["robust"][
                            "overlap_vol_cm3"
                        ]
                    ),
                    float(solved["by_arm"][arm]["ik_allerr"]),
                ),
            )
            preferred = solved["by_arm"][preferred_arm]
            robust_metrics = preferred["volume_validation"]["robust"]
            pose["valid_arms"] = valid_arms
            pose["preferred_arm"] = preferred_arm
            pose["ik_allerr"] = float(preferred["ik_allerr"])
            pose["joint_limit_margin_rad"] = float(
                preferred.get("final_joint_limit_margin_rad", 0.0)
            )
            pose["robust_grasp_vox"] = int(robust_metrics["grasp_vox"])
            pose["robust_grasp_vol_cm3"] = float(robust_metrics["grasp_vol_cm3"])
            pose["robust_overlap_vol_cm3"] = float(robust_metrics["overlap_vol_cm3"])
            pose["robust_inflated_overlap_vol_cm3"] = float(
                robust_metrics["inflated_overlap_vol_cm3"]
            )
            reachable.append(pose)
        else:
            for info in solved["by_arm"].values():
                reason = str(info.get("reason", "unknown"))
                ik_failure_counts[reason] = ik_failure_counts.get(reason, 0) + 1
    if not reachable:
        raise X5GraspPlanningError(
            "no RGB-D grasp candidate passed paired IK, final_q FK volume and observed path validation "
            f"for {','.join(active_arms)}; failures={ik_failure_counts}",
            audit={
                **candidate_audit,
                "ik_failure_counts": ik_failure_counts,
                "final_q_fk_volume_failure_counts": final_q_fk_volume_failure_counts,
                "ik_input_count": len(ik_input),
            },
        )
    joint_safe_reachable = [
        pose
        for pose in reachable
        if float(pose.get("joint_limit_margin_rad", 0.0)) >= JOINT_LIMIT_MARGIN_RAD
    ]
    if edge_usable and joint_safe_reachable:
        reachable = joint_safe_reachable
    reachable_safe_max = max(float(pose["robust_grasp_vol_cm3"]) for pose in reachable)
    # Orientation comes from the audited support plane, not the clicked
    # object's (often ill-conditioned) side/top normal.
    quality_ratio = NORMAL_ADAPTIVE_GRASP_QUALITY_RATIO
    absolute_min = VERTICAL_MIN_OPENING_VOXELS * voxel_cm3
    grasp_threshold = adaptive_grasp_volume_floor(
        reachable_safe_max,
        quality_ratio=quality_ratio,
        absolute_min_cm3=absolute_min,
    )
    passing = [
        pose
        for pose in reachable
        if float(pose["robust_grasp_vol_cm3"]) >= grasp_threshold - 1e-12
    ]
    if not passing:
        raise X5GraspPlanningError(
            "paired-IK candidates did not meet the adaptive grasp-volume floor "
            f"({reachable_safe_max:.3f}cm3 available, {grasp_threshold:.3f}cm3 required)"
        )
    if support_normal is not None:
        ratio = 0.75
        slack = 0.25
        best_grasp = max(float(pose["robust_grasp_vol_cm3"]) for pose in passing)
        alignment_pool = [
            pose
            for pose in passing
            if float(pose["robust_grasp_vol_cm3"])
            >= max(grasp_threshold, ratio * best_grasp) - 1e-12
        ]
        minimum_approach_tilt = min(
            float(pose.get("approach_tilt_deg", float("inf")))
            for pose in alignment_pool
        )
        vertical_pool = [
            pose
            for pose in alignment_pool
            if float(pose.get("approach_tilt_deg", float("inf")))
            <= minimum_approach_tilt + 1e-12
        ]
        minimum_angle = min(
            float(pose["normal_alignment_deg"]) for pose in vertical_pool
        )
        selection_pool = [
            pose
            for pose in vertical_pool
            if float(pose["normal_alignment_deg"]) <= minimum_angle + slack + 1e-12
        ]
        selection_pool.sort(
            key=lambda pose: (
                float(pose.get("approach_tilt_deg", float("inf"))),
                float(pose.get("closure_alignment_deg", 0.0)),
                -float(pose["robust_grasp_vol_cm3"]),
                float(pose["normal_alignment_deg"]),
                float(pose["robust_inflated_overlap_vol_cm3"]),
                float(pose["robust_overlap_vol_cm3"]),
                float(pose["ik_allerr"]),
            )
        )
    else:
        selection_pool = sorted(
            passing,
            key=lambda pose: (
                float(pose.get("approach_tilt_deg", float("inf"))),
                -float(pose["robust_grasp_vol_cm3"]),
                -float(pose.get("opening_axial_span_m", 0.0)),
                -float(pose.get("joint_limit_margin_rad", 0.0)),
                float(pose["robust_inflated_overlap_vol_cm3"]),
                float(pose["robust_overlap_vol_cm3"]),
                float(pose["ik_allerr"]),
            ),
        )
    selected = selection_pool[0]
    if requested_arm in ("left", "right"):
        selected_arm = requested_arm
    else:
        selected_arm = str(selected["preferred_arm"])
    selected_ik = selected["ik_filter"][selected_arm]
    volume_validation = dict(selected_ik["volume_validation"])
    ideal_metrics = dict(volume_validation["ideal"])
    final_fk_metrics = dict(volume_validation["final_q_fk"])
    robust_metrics = dict(volume_validation["robust"])
    ideal_eef_position = np.asarray(selected["eef_pos"], dtype=np.float64)
    ideal_eef_quaternion = np.asarray(selected["quat"], dtype=np.float64)
    ideal_safe_position = np.asarray(selected["safe_position"], dtype=np.float64)
    final_fk_pose = np.asarray(
        selected_ik["final_q_fk_pose"], dtype=np.float64
    ).reshape(7)
    safe_fk_pose = np.asarray(selected_ik["safe_q_fk_pose"], dtype=np.float64).reshape(
        7
    )
    eef_position = final_fk_pose[:3]
    eef_quaternion = final_fk_pose[3:]
    safe_position = safe_fk_pose[:3]
    safe_quaternion = safe_fk_pose[3:]
    next_position = eef_position + np.array([0.0, 0.0, 0.10], dtype=np.float64)
    selected_pose_ik = {
        "mode": "policy_local_observation_calibrated_x5_dls_fk",
        "constraint": (
            f"final pos<={FINAL_IK_POS_TOL_M:g}m ori<={FINAL_IK_ORI_TOL_DEG:g}deg; "
            f"safe pos<={SAFE_IK_POS_TOL_M:g}m ori<={SAFE_IK_ORI_TOL_DEG:g}deg; "
            f"branch_gap<={SAFE_FINAL_BRANCH_GAP_RAD:g}rad"
        ),
        "validated_by_local_fk": True,
        "final_q_fk_volume_validated": True,
        "execution_pose_source": "capture_calibrated_fk_of_signed_final_q",
        "validated_by_simulator_fk": False,
        "left_ok": bool(selected["ik_filter"].get("left", {}).get("ok")),
        "right_ok": bool(selected["ik_filter"].get("right", {}).get("ok")),
        "left_q_arm": selected["ik_filter"].get("left", {}).get("final_q"),
        "right_q_arm": selected["ik_filter"].get("right", {}).get("final_q"),
        "ik_allerr": float(selected_ik["ik_allerr"]),
        "ideal_final_pose": np.concatenate(
            (ideal_eef_position, ideal_eef_quaternion)
        ).tolist(),
        "final_q_fk_pose": final_fk_pose.tolist(),
        "ideal_safe_pose": np.concatenate(
            (ideal_safe_position, ideal_eef_quaternion)
        ).tolist(),
        "safe_q_fk_pose": safe_fk_pose.tolist(),
        "volume_validation": volume_validation,
        "rgbd_non_gripper_path_validation": selected_ik.get(
            "rgbd_non_gripper_path_validation"
        ),
    }
    selected_pose_ik_q = {
        arm: info["final_q"]
        for arm, info in selected["ik_filter"].items()
        if bool(info.get("ok"))
    }
    realized_rotation = quaternion_to_matrix(eef_quaternion)
    realized_gripper_vector = -realized_rotation[:, 0]
    realized_normal_angle = (
        math.degrees(
            math.acos(
                float(
                    np.clip(
                        support_normal @ realized_gripper_vector,
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        if support_normal is not None
        else float("inf")
    )
    grip_fit = {
        "build": BUILD,
        "robot_material_partition": robot_material_audit,
        "opening_query_version": geometry["metadata"].get(
            "opening_query_version", "legacy_voxel_columns"
        ),
        "surface_query_radius_cells": 0 if mesh_result is not None else None,
        "raw_surface_evidence": final_fk_metrics["raw_surface_evidence"],
        "grasp_volume_semantics": "observed_uncertainty_band_quadrature_not_physical_material_volume",
        "vertical_contact_patch": contact_patch_audit,
        "closing_axis": closing_axis_audit,
        "closure_alignment_deg": float(selected.get("closure_alignment_deg", 0.0)),
        "volume_source": (
            (
                "frozen_rgbd_observed_surface_band_opening_surface_only"
                if surface_only_opening
                else "frozen_rgbd_conservative_collision_shell"
            )
            if mesh_result is not None
            else (
                "frozen_depth_organized_shell_3mm_opening_surface_only"
                if surface_only_opening
                else "frozen_depth_organized_shell_3mm"
            )
        ),
        "mesh_requested_version": str(
            os.environ.get(MESH_ROUTE_ENV, DEFAULT_MESH_ROUTE)
        ),
        "mesh_effective_version": effective_mesh_version,
        "mesh_fallback_reason": mesh_fallback_reason,
        "support_plane_z_m": support_z,
        "opening_support_margin_m": (
            float(OPENING_SUPPORT_MARGIN_M) if surface_only_opening else None
        ),
        "opening_target_radius_m": (
            float(OPENING_TARGET_RADIUS_M) if surface_only_opening else None
        ),
        "edge_approach_world": (
            None if edge_approach is None else edge_approach.tolist()
        ),
        "edge_approach_usable": bool(edge_usable),
        "edge_anchor_inset_m": (0.0),
        "approach_source": selected["approach_source"],
        "edge_pitch_rad": selected["edge_pitch_rad"],
        "table_lift_m": float(selected["table_lift_m"]),
        "table_clearance_m": selected["table_clearance_m"],
        "opening_axial_min_m": final_fk_metrics["opening_axial_min_m"],
        "opening_axial_max_m": final_fk_metrics["opening_axial_max_m"],
        "opening_axial_span_m": float(final_fk_metrics["opening_axial_span_m"]),
        "opening_tip_fraction": float(final_fk_metrics["opening_tip_fraction"]),
        "tip_only": bool(final_fk_metrics["tip_only"]),
        "joint_limit_margin_rad": float(selected.get("joint_limit_margin_rad", 0.0)),
        "opening_geometry": geometry["metadata"].get("opening_geometry"),
        "opening_projection": geometry["metadata"].get("opening_projection"),
        "opening_generation": geometry["metadata"].get("opening_generation"),
        "opening_column_count": int(len(geometry["opening_columns"])),
        # 3 mm quadrature samples the exact opening, not the legacy eroded
        # columns. Its observed-band intersection remains a score, never truth.
        "exact_opening_mesh_volume_cm3": float(
            geometry["metadata"].get("exact_opening_mesh_volume_cm3", 0.0)
        ),
        "exact_opening_mesh_source": geometry["metadata"].get(
            "exact_opening_generation"
        ),
        "exact_opening_geometry": geometry["metadata"].get("exact_opening_geometry"),
        "exact_opening_integrated_volume_cm3": float(
            geometry["metadata"].get("exact_opening_integrated_volume_cm3", 0.0)
        ),
        "exact_opening_y_boundary": geometry["metadata"].get(
            "exact_opening_y_boundary"
        ),
        "exact_opening_inner_y_bounds_m": geometry["metadata"].get(
            "exact_opening_inner_y_bounds_reference_m"
        ),
        "exact_opening_mesh_watertight": bool(
            geometry["metadata"].get("exact_opening_mesh_watertight", False)
        ),
        "exact_opening_metric_independent_of_voxel": True,
        "reported_pose_source": "capture_calibrated_fk_of_signed_final_q",
        "candidate_volume_validation": volume_validation,
        "grasp_vol_cm3": float(final_fk_metrics["grasp_vol_cm3"]),
        "overlap_vol_cm3": float(final_fk_metrics["overlap_vol_cm3"]),
        "inflated_overlap_vol_cm3": float(final_fk_metrics["inflated_overlap_vol_cm3"]),
        "overlap_frac": float(final_fk_metrics["overlap_frac"]),
        "ideal_grasp_vol_cm3": float(ideal_metrics["grasp_vol_cm3"]),
        "ideal_overlap_vol_cm3": float(ideal_metrics["overlap_vol_cm3"]),
        "ideal_inflated_overlap_vol_cm3": float(
            ideal_metrics["inflated_overlap_vol_cm3"]
        ),
        "final_q_fk_grasp_vol_cm3": float(final_fk_metrics["grasp_vol_cm3"]),
        "final_q_fk_overlap_vol_cm3": float(final_fk_metrics["overlap_vol_cm3"]),
        "final_q_fk_inflated_overlap_vol_cm3": float(
            final_fk_metrics["inflated_overlap_vol_cm3"]
        ),
        "robust_grasp_vol_cm3": float(robust_metrics["grasp_vol_cm3"]),
        "robust_overlap_vol_cm3": float(robust_metrics["overlap_vol_cm3"]),
        "robust_inflated_overlap_vol_cm3": float(
            robust_metrics["inflated_overlap_vol_cm3"]
        ),
        "gripper_vol_cm3": float(len(geometry["gripper_voxels"]) * voxel_cm3),
        "gap_voxel_n": int(final_fk_metrics["grasp_vox"]),
        "grasp_vox": int(final_fk_metrics["grasp_vox"]),
        "overlap_vox": int(final_fk_metrics["overlap_vox"]),
        "inflated_overlap_vox": int(final_fk_metrics["inflated_overlap_vox"]),
        "original_overlap_threshold_cm3": ORIGINAL_OVERLAP_MAX_CM3,
        "inflated_overlap_threshold_cm3": INFLATED_OVERLAP_MAX_CM3,
        "grasp_vol_threshold_cm3": float(grasp_threshold),
        "grasp_vol_threshold_cap_cm3": MIN_GRASP_VOL_CM3,
        "reachable_safe_grasp_max_cm3": float(reachable_safe_max),
        "global_safe_grasp_max_cm3": float(global_safe_max),
        "adaptive_grasp_quality_ratio": float(quality_ratio),
        "adaptive_grasp_absolute_min_cm3": float(absolute_min),
        "opening_voxel_total": int(len(geometry["opening_voxels"])),
        "anchor_marker_pt": target_anchor.tolist(),
        "surface_hit_pt": hit.tolist(),
        "anchor_dist_mm": float(surface_anchor_audit["center_to_selected_m"] * 1000.0),
        "anchor_pixel_uv": [int(pixel_u), int(pixel_v)],
        "input_pixel_uv": [int(input_pixel_u), int(input_pixel_v)],
        "axial_point": selected["axial_point"],
        "axial_x_mm": float(selected["axial_x_m"] * 1000.0),
        "approach_tilt_deg": float(selected.get("approach_tilt_deg", float("nan"))),
        "axial_refinement": selected.get("axial_refinement"),
        "transverse_refinement": selected.get("transverse_refinement"),
        "outward_normal_world": (
            None if outward_normal is None else outward_normal.tolist()
        ),
        "gripper_vector_world": realized_gripper_vector.tolist(),
        "ideal_gripper_vector_world": selected["gripper_vector_world"].tolist(),
        "normal_alignment_deg": (
            float(realized_normal_angle)
            if math.isfinite(float(realized_normal_angle))
            else None
        ),
        "ideal_normal_alignment_deg": (
            float(selected["normal_alignment_deg"])
            if math.isfinite(float(selected["normal_alignment_deg"]))
            else None
        ),
        "normal_confidence": normal_confidence,
        "inflation_geometry": {
            "source": "official_x5_component_voxels",
            "finger_inward_inflate_m": float(
                geometry["metadata"]["finger_inflation_m"]
            ),
            "camera_inflate_m": 0.0,
            "camera_geometry_included": False,
        },
        "mesh_anchor": mesh_anchor_audit,
    }
    plan_audit = {
        "build": BUILD,
        "non_gripper_path_gate": path_gate.audit,
        "orientation_policy": {
            "mode": "table_normal_downward",
            "support_normal_up_world": support_normal.tolist(),
            "finger_approach_axis": "+X",
            "reported_gripper_vector_axis": "-X",
            "maximum_candidate_tilt_deg": max_downward_tilt_deg,
            "maximum_fk_tilt_deg": max_downward_tilt_deg + DOWNWARD_FK_TILT_TOL_DEG,
            "selected_approach_tilt_deg": float(
                selected.get("approach_tilt_deg", float("nan"))
            ),
            "selected_fk_tilt_deg": realized_normal_angle,
            "surface_normal_used_for_orientation": (
                closing_axis_mode == "projected_surface_normal"
            ),
            "surface_normal_used_for_approach": False,
            "projected_surface_normal_used_for_yaw": (
                closing_axis_mode == "projected_surface_normal"
            ),
        },
        "allowed_observations": [
            "frozen_evaluator_rgb",
            "frozen_evaluator_depth_linear",
            "frozen_evaluator_camera_intrinsics",
            "frozen_evaluator_camera_extrinsics",
            "capture_time_evaluator_proprioception",
            "policy_local_static_official_x5_model",
        ],
        "forbidden_observations_used": [],
        "used_segmentation": False,
        "used_object_identity": False,
        "used_simulator_scene_mesh": False,
        "used_simulator_fk_ik": False,
        "hit": point_observation,
        "surface_anchor": surface_anchor_audit,
        "mesh_anchor": mesh_anchor_audit,
        "surface_normal": normal_audit,
        "support_plane": support_plane_audit,
        "vertical_contact_patch": contact_patch_audit,
        "closing_axis": closing_axis_audit,
        "depth_edge": edge_audit,
        "opening_filter": {
            "surface_only": bool(surface_only_opening),
            "support_lift_enabled": bool(support_lift_enabled),
            "support_plane_z_m": support_z,
            "support_margin_m": float(OPENING_SUPPORT_MARGIN_M),
            "target_radius_m": float(OPENING_TARGET_RADIUS_M),
            "edge_robust_count": int(len(edge_robust)),
            "strict_overlap_count_before_edge_filter": int(len(strict_overlap)),
            "strict_overlap_count_after_edge_filter": int(len(strict)),
            "min_opening_voxels": int(VERTICAL_MIN_OPENING_VOXELS),
            "min_transverse_span_m": float(VERTICAL_MIN_SURFACE_SPAN_M),
            "min_contact_column_count": 2,
            "min_axial_span_m": None,
            "tip_only_is_diagnostic_not_rejection": True,
            "max_table_lift_m": float(EDGE_MAX_TABLE_LIFT_M),
        },
        "reconstruction": {
            "method": (
                "robodojo_v1_whole_view_surface_repair_dual_occupancy"
                if mesh_result is not None
                else "organized_depth_ray_shell"
            ),
            "voxel_m": VOXEL_M,
            "back_extrusion_mm": (
                12.0 if mesh_result is not None else BACK_EXTRUSION_M * 1000.0
            ),
            "depth_shape": list(depth.shape),
            "requested_version": str(
                os.environ.get(MESH_ROUTE_ENV, DEFAULT_MESH_ROUTE)
            ),
            "effective_version": effective_mesh_version,
            "fallback_reason": mesh_fallback_reason,
            "opening_geometry": geometry["metadata"].get("opening_geometry"),
            "opening_projection": geometry["metadata"].get("opening_projection"),
            "opening_generation": geometry["metadata"].get("opening_generation"),
            "opening_column_count": int(len(geometry["opening_columns"])),
            "opening_voxel_total": int(len(geometry["opening_voxels"])),
            "exact_opening_mesh_volume_cm3": float(
                geometry["metadata"].get("exact_opening_mesh_volume_cm3", 0.0)
            ),
            "exact_opening_mesh_source": geometry["metadata"].get(
                "exact_opening_generation"
            ),
            "exact_opening_geometry": geometry["metadata"].get(
                "exact_opening_geometry"
            ),
            "exact_opening_integrated_volume_cm3": float(
                geometry["metadata"].get("exact_opening_integrated_volume_cm3", 0.0)
            ),
            "exact_opening_y_boundary": geometry["metadata"].get(
                "exact_opening_y_boundary"
            ),
            "exact_opening_inner_y_bounds_m": geometry["metadata"].get(
                "exact_opening_inner_y_bounds_reference_m"
            ),
            "exact_opening_metric_independent_of_voxel": True,
            "surface_occupancy": (
                None
                if mesh_result is None
                else mesh_result.surface_occupancy.metadata()
            ),
            "collision_occupancy": (
                None
                if mesh_result is None
                else mesh_result.collision_occupancy.metadata()
            ),
            "mesh_audit": None if mesh_result is None else mesh_result.metadata,
        },
        "sampling": {
            "anchor_count": 1,
            "anchor_source": surface_anchor_audit["selection"],
            "anchor_sphere_diameter_mm": (
                float(surface_anchor_audit["diameter_m"]) * 1000.0
            ),
            "anchor_surface_candidate_count": int(
                surface_anchor_audit["sphere_surface_point_count"]
            ),
            "approach_count": 20,
            "roll_count": N_ROLL,
            "axial_point_count": int(len(geometry["axial_x_m"])),
            "poses_per_anchor": expected_count,
            "raw_pose_count": int(len(candidates)),
            "axial_x_mm": (geometry["axial_x_m"] * 1000.0).round(3).tolist(),
            "axial_refinement": refinement_audit,
            "transverse_refinement": transverse_refinement_audit,
            "in_opening_center_shift": opening_offset_audit,
            "evaluated_pose_count": (
                expected_count
                + refinement_audit["extra_geometry_evaluations"]
                + transverse_refinement_audit["extra_geometry_evaluations"]
                + opening_offset_audit.get("extra_geometry_evaluations", 0)
            ),
        },
        "prefilter": {
            **rank_audit,
            "strict_overlap_safe_count": int(len(strict)),
            "global_safe_grasp_max_cm3": float(global_safe_max),
            "global_ideal_safe_grasp_max_cm3": float(global_safe_max),
            "joint_safe_reachable_count": int(len(joint_safe_reachable)),
            "joint_limit_margin_rad": float(JOINT_LIMIT_MARGIN_RAD),
        },
        "ik": {
            "mode": "paired_final_safe_bounded_multistart_x5_dls_fk_volume_rescore",
            "active_arms": list(active_arms),
            "input_count": int(len(ik_input)),
            "reachable_count": int(len(reachable)),
            "failure_counts": ik_failure_counts,
            "final_q_fk_volume_validation_count": int(
                final_q_fk_volume_validation_count
            ),
            "final_q_fk_volume_failure_counts": (final_q_fk_volume_failure_counts),
            "selected_arm": selected_arm,
            "selected": selected_ik,
            "selected_volume_validation": volume_validation,
            "execution_pose_source": "capture_calibrated_fk_of_signed_final_q",
        },
        "thresholds": {
            "original_overlap_max_cm3": ORIGINAL_OVERLAP_MAX_CM3,
            "inflated_overlap_max_cm3_exclusive": INFLATED_OVERLAP_MAX_CM3,
            "adaptive_grasp_min_cm3": float(grasp_threshold),
            "final_ik_position_tolerance_m": FINAL_IK_POS_TOL_M,
            "final_ik_orientation_tolerance_deg": FINAL_IK_ORI_TOL_DEG,
        },
        "timing_s": {
            "volume_metrics": float(ik_started - metric_started),
            "paired_ik": float(time.perf_counter() - ik_started),
            "total": float(time.perf_counter() - started),
        },
        "static_gripper_model": geometry["metadata"],
    }
    result = {
        "ok": True,
        "mode": "grasp_point_filter_rgbd_lite",
        "planner_variant": "rgbd_lite_1120_volume_paired_x5_ik",
        "exec_sequence": "move_only",
        "target_robodojo_env_m": hit.tolist(),
        "point_observation": point_observation,
        "pixel_uv": [int(pixel_u), int(pixel_v)],
        "input_pixel_uv": [int(input_pixel_u), int(input_pixel_v)],
        "vlm_pixel": [float(u), float(v)],
        "hit_world": hit.tolist(),
        "grasp_anchor_world": target_anchor.tolist(),
        "arm": selected_arm,
        "recommended_arm": selected_arm,
        "execution_pose_source": "capture_calibrated_fk_of_signed_final_q",
        "eef_pose": {"pos": eef_position.tolist(), "quat": eef_quaternion.tolist()},
        "safe_eef_pose": {
            "pos": safe_position.tolist(),
            "quat": safe_quaternion.tolist(),
        },
        "ideal_eef_pose": {
            "pos": ideal_eef_position.tolist(),
            "quat": ideal_eef_quaternion.tolist(),
        },
        "ideal_safe_eef_pose": {
            "pos": ideal_safe_position.tolist(),
            "quat": ideal_eef_quaternion.tolist(),
        },
        "next_eef_move": {
            "pos": next_position.tolist(),
            "quat": eef_quaternion.tolist(),
        },
        "candidate_count": int(len(candidates)),
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": selected_pose_ik_q,
        "ik_solution": {
            "arm": selected_arm,
            "safe_q": selected_ik["safe_q"],
            "final_q": selected_ik["final_q"],
            "safe_final_branch_gap_rad": selected_ik["safe_final_branch_gap_rad"],
            "final_position_error_m": selected_ik["final_position_error_m"],
            "final_orientation_error_deg": selected_ik["final_orientation_error_deg"],
            "safe_position_error_m": selected_ik["safe_position_error_m"],
            "safe_orientation_error_deg": selected_ik["safe_orientation_error_deg"],
            "final_q_fk_pose": final_fk_pose.tolist(),
            "safe_q_fk_pose": safe_fk_pose.tolist(),
            "volume_validation": volume_validation,
            "self_collision": selected_ik.get("self_collision"),
            "rgbd_non_gripper_path_validation": selected_ik.get(
                "rgbd_non_gripper_path_validation"
            ),
        },
        "grasp_vol_cm3": float(final_fk_metrics["grasp_vol_cm3"]),
        "overlap_vol_cm3": float(final_fk_metrics["overlap_vol_cm3"]),
        "inflated_overlap_vol_cm3": float(final_fk_metrics["inflated_overlap_vol_cm3"]),
        "overlap_frac": float(final_fk_metrics["overlap_frac"]),
        "ideal_grasp_vol_cm3": float(ideal_metrics["grasp_vol_cm3"]),
        "ideal_overlap_vol_cm3": float(ideal_metrics["overlap_vol_cm3"]),
        "ideal_inflated_overlap_vol_cm3": float(
            ideal_metrics["inflated_overlap_vol_cm3"]
        ),
        "robust_grasp_vol_cm3": float(robust_metrics["grasp_vol_cm3"]),
        "robust_overlap_vol_cm3": float(robust_metrics["overlap_vol_cm3"]),
        "robust_inflated_overlap_vol_cm3": float(
            robust_metrics["inflated_overlap_vol_cm3"]
        ),
        "exact_opening_mesh_volume_cm3": float(
            geometry["metadata"].get("exact_opening_mesh_volume_cm3", 0.0)
        ),
        "exact_opening_mesh_source": geometry["metadata"].get(
            "exact_opening_generation"
        ),
        "exact_opening_geometry": geometry["metadata"].get("exact_opening_geometry"),
        "exact_opening_integrated_volume_cm3": float(
            geometry["metadata"].get("exact_opening_integrated_volume_cm3", 0.0)
        ),
        "exact_opening_y_boundary": geometry["metadata"].get(
            "exact_opening_y_boundary"
        ),
        "exact_opening_inner_y_bounds_m": geometry["metadata"].get(
            "exact_opening_inner_y_bounds_reference_m"
        ),
        "exact_opening_metric_independent_of_voxel": True,
        "grip_fit": grip_fit,
        "plan_audit": plan_audit,
    }
    if return_scene_mesh_result:
        # Private in-process handoff used by the official tool wrapper so its
        # three-view renderer does not reconstruct the same frozen RGB-D frame
        # a second time. The wrapper removes this object before signing or
        # returning the public JSON result.
        result["_scene_mesh_result"] = mesh_result
    return result


def plan_x5_grasp_point_filter_rgbd_lite(
    *,
    snapshot: Snapshot,
    u: float,
    v: float,
    plan_arm: str = "any",
    seed: int = 42,
    return_scene_mesh_result: bool = False,
) -> dict[str, Any]:
    """Return a plan from the strict near-vertical production cone.

    The public tool keeps the same immutable RGB-D/proprioception inputs and
    the same collision, surface, and paired-IK/FK gates, but it never widens
    the approach cone after a failed search. A returned endpoint is therefore
    bounded by the configured five-degree candidate cone plus the numerical
    final-FK allowance. Wider cones may be evaluated only by the separate
    offline audit script, never by this public planner.
    """
    started = time.perf_counter()
    requested_route = (
        str(os.environ.get(MESH_ROUTE_ENV, DEFAULT_MESH_ROUTE)).strip().lower()
    )
    legacy = requested_route in {"v4", "legacy", "organized_shell", "robodojo_v4"}
    tiers = DOWNWARD_CONE_TIERS_DEG[:1] if legacy else DOWNWARD_CONE_TIERS_DEG
    attempts = []

    def search_audit():
        return {
            "policy": f"strict_single_downward_cone_max_{int(MAX_DOWNWARD_TILT_DEG)}_degree",
            "configured_tiers_deg": list(tiers),
            "near_vertical_preferred": True,
            "hard_hemisphere_boundary_deg": 90.0,
            "maximum_final_fk_tilt_deg": max(tiers) + DOWNWARD_FK_TILT_TOL_DEG,
            "hard_maximum_candidate_tilt_deg": max(tiers),
            "large_tilt_fallback_enabled": False,
            "large_tilt_fallback_is_not_near_vertical": False,
            "attempts": list(attempts),
            "total_elapsed_s": time.perf_counter() - started,
            "same_frozen_capture": True,
            "collision_or_ik_tolerances_relaxed": False,
            "legacy_single_tier": legacy,
        }

    for angle in tiers:
        tier_started = time.perf_counter()
        try:
            result = _plan_x5_grasp_in_downward_cone(
                snapshot=snapshot,
                u=u,
                v=v,
                plan_arm=plan_arm,
                seed=seed,
                return_scene_mesh_result=return_scene_mesh_result,
                max_downward_tilt_deg=angle,
            )
        except X5GraspPlanningError as error:
            attempts.append(
                {
                    "maximum_candidate_tilt_deg": angle,
                    "ok": False,
                    "elapsed_s": time.perf_counter() - tier_started,
                    "error": str(error),
                    "candidate_audit": error.audit,
                }
            )
            # The public route has a hard single-tier policy. Preserve the
            # detailed candidate audit, but never retry with a wider cone.
            raise X5GraspPlanningError(
                str(error),
                audit={**error.audit, "downward_cone_search": search_audit()},
            ) from error
        attempts.append(
            {
                "maximum_candidate_tilt_deg": angle,
                "ok": True,
                "elapsed_s": time.perf_counter() - tier_started,
                "selected_fk_tilt_deg": result["plan_audit"]["orientation_policy"][
                    "selected_fk_tilt_deg"
                ],
            }
        )
        # Keep the public contract explicit even if a future private kernel or
        # a legacy mesh route accidentally returns a wider pose.  The private
        # kernel already applies this FK gate; this postcondition prevents a
        # wider result from crossing the public API boundary silently.
        selected_fk_tilt = float(
            result["plan_audit"]["orientation_policy"]["selected_fk_tilt_deg"]
        )
        if (
            not math.isfinite(selected_fk_tilt)
            or selected_fk_tilt > angle + DOWNWARD_FK_TILT_TOL_DEG + 1e-9
        ):
            failure_audit = search_audit()
            failure_audit["attempts"] = list(attempts)
            failure_audit["postcondition"] = {
                "ok": False,
                "reason": "selected_fk_tilt_exceeds_public_vertical_cap",
                "selected_fk_tilt_deg": selected_fk_tilt,
                "maximum_candidate_tilt_deg": float(angle),
                "maximum_final_fk_tilt_deg": float(angle + DOWNWARD_FK_TILT_TOL_DEG),
            }
            raise X5GraspPlanningError(
                "selected FK pose exceeded the public strict-vertical cap",
                audit={
                    **result.get("plan_audit", {}),
                    "downward_cone_search": failure_audit,
                },
            )
        audit = search_audit()
        result["plan_audit"]["downward_cone_search"] = audit
        result["grip_fit"]["downward_cone_search"] = audit
        return result
    raise AssertionError(
        "nonempty cone schedule must return a plan or raise a planning error"
    )


__all__ = [
    "BUILD",
    "INFLATED_OVERLAP_MAX_CM3",
    "MIN_GRASP_VOL_CM3",
    "ORIGINAL_OVERLAP_MAX_CM3",
    "SAFE_PLAN_BACK_M",
    "SURFACE_ANCHOR_DIAMETER_M",
    "SURFACE_ANCHOR_RADIUS_M",
    "X5GraspPlanningError",
    "adaptive_grasp_volume_floor",
    "estimate_outward_normal",
    "plan_x5_grasp_point_filter_rgbd_lite",
    "select_nearest_surface_anchor",
]
