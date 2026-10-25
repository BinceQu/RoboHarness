from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import numpy as np
from tools.util.capabilities import TOOL_VERSION
from tools.util.contract import (
    ARM_DOF,
    ROBOT_PROFILE,
    validate_move_tracked_point_args,
)
from tools.util.geometry import (
    json_ready,
)
from tools.util.storage import Snapshot, file_data_url
from tools.util.x5_kinematics import (
    X5_JOINT_LOWER,
    X5_JOINT_UPPER,
    quaternion_to_matrix,
)
from tools.util.x5_gripper_visualization import (
    render_x5_gripper_overlay,
)
from tools.util.x5_grasp_geometry import ROBODOJO_X5_GRIPPER_SCALE_M
from tools.util.tracked_point_motion_x5_support import LocalRobotState

from tools.util.motion_common import _move_tracked_exports, _set_result


def move_tracked_point(
    ctx,
    points: list[dict[str, Any]],
    relations: Sequence[Mapping[str, Any]] | None = None,
    pos_tol: float = 0.03,
    ori_tol_deg: float = 20.0,
    max_steps: int = 360,
    timeout_s: float = 90.0,
    quick_constraint: Mapping[str, Any] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
    mode: str | None = None,
    execution_mode: str = "exec",
    inequalities: Sequence[Mapping[str, Any]] | None = None,
):
    """Move or plan tracked/predefined points with spatial collinear order.

    The reserved names ``left_finger_tip``, ``right_finger_tip``, and
    ``gripper_slide_center`` bind to the selected physical X5 gripper using
    synchronized evaluator proprioception and submission-local FK. They do
    not require ``track_object_distance`` registration.

    Explicit collinear quick/typed rows are the required axial sequence: 1,3,2
    places 3 between 1 and 2. Coordinate rows alone do not activate that quick
    constraint. Robot-base reading direction is left-to-right (-Y), top-to-bottom
    (-Z), or near-to-far (+X), using the largest axis component (ties Y,Z,X).
    Camera viewpoint does not change the order. Reversed or shuffled endpoints
    cannot be returned as successful approximate/overlap-fallback plans.

    The ``faceto`` and ``reverse_faceto`` camera pose is frozen from the same
    evaluator observation as the tracked points and is reused for planning,
    signed-plan reload, and live validation.

    ``inequalities`` contains strict affine variable constraints. Each item is
    ``{"lhs": <variable-or-affine-expression>, "op": ">"|"<", "rhs":
    <finite-number>}``; for example ``a-b > 0`` selects the signed vertical
    ordering when two point rows use Z variables ``a`` and ``b``.
    """

    try:
        move_exports = _move_tracked_exports()
    except ImportError as exc:
        _set_result(
            ctx,
            "move_tracked_point",
            {
                "ok": False,
                "reason": "hot_reload_failed",
                "error": str(exc),
                "failure_stage": "hot_reload",
                "implementation": "workspace source",
            },
        )
        return

    lifecycle: dict[str, Any] = {}
    motion_retention_lease_state: dict[str, Any] = {}
    implementation = move_exports["run"](
        ctx,
        points=points,
        relations=relations,
        pos_tol=pos_tol,
        ori_tol_deg=ori_tol_deg,
        max_steps=max_steps,
        timeout_s=timeout_s,
        quick_constraint=quick_constraint,
        quick_constraints=quick_constraints,
        mode=mode,
        execution_mode=execution_mode,
        inequalities=inequalities,
        result_writer=_set_result,
        lifecycle_state=lifecycle,
        motion_retention_lease_state=motion_retention_lease_state,
    )
    try:
        yield from implementation
    finally:
        implementation.close()
        manager = lifecycle.get("source_manager") or ctx.tracker
        move_exports["release_motion_retention"](manager, motion_retention_lease_state)
        prior_lease = str(lifecycle.get("eef_prior_lease_id") or "")
        if prior_lease:
            try:
                manager.deactivate_on_hand_eef_observation_prior(prior_lease)
            except Exception:
                pass


# Full tracking/constraint execution algorithm follows.
"""BEHAVIOR v2 move_tracked_point execution adapted to dual RoboDojo X5."""


import concurrent.futures
import hashlib
import math
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tools.util.context import ToolCancelled
from tools.util.geometry import json_ready
from tools.util.storage import Snapshot, file_data_url
from tools.util.x5_kinematics import (
    X5_JOINT_LOWER,
    X5_JOINT_UPPER,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from tools.util.capabilities import TOOL_VERSION
from tools.util.contract import (
    ACTION_DIM,
    ARM_DOF,
    MOVE_TRACKED_POINT_MAX_POINTS,
    MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS,
    ROBOT_PROFILE,
    coordinate_collinear_relations,
    validate_move_tracked_point_args as _validate_move_tracked_point_args_contract,
)
from tools.util.tracked_point_motion_local import (
    ENDPOINT_PATH_FALLBACK_ORIENTATION_REGULARIZATION_WEIGHT,
    PlanningDeadlineExceeded,
    _affine_pose_family_targets,
    evaluate_rigid_geometry_consistency,
    evaluate_target_constraints,
    _affine_form,
    plan_cartesian_trajectory,
    plan_endpoint,
    plan_joint_space_trajectory,
    rigid_pair_distance_tolerance,
    shared_line_target_pose_family,
)
from tools.util.tracked_point_constraints_local import (
    evaluate_quick_constraint,
    geometry_rank,
    geometry_report_difference,
    quick_constraint_relation,
    rigid_geometry_report,
)
from tools.util.tracked_point_execution_local import schedule_execution_waypoints
from tools.util.predefined_eef_points_local import (
    is_predefined_eef_point_name,
    partition_predefined_eef_point_names,
    predefined_eef_point_local_m,
    predefined_eef_points_robot_base_m,
)
from tools.util.tracked_point_motion_x5_support import (
    LocalRobotState,
    eef_pose,
    mat_to_quat_xyzw,
    quat_to_mat_xyzw,
)
from tools.util.x5_grasp_geometry import (
    ROBODOJO_X5_GRIPPER_SCALE_M,
    x5_gripper_joint_position_m,
)
from tools.util.x5_move_tracked_overlap import evaluate_x5_move_endpoint_overlap
from tools.util.x5_gripper_visualization import render_x5_gripper_overlay


def _resident_contract_implicitly_infers_coordinate_order() -> bool:
    """Detect the pre-v79 contract retained by an already-running process."""

    probe = {
        "points": [
            {
                "name": "_probe_fixed",
                "role": "off_hand",
                "target_xyz_m": ["x", "y", "z"],
            },
            {
                "name": "_probe_first",
                "role": "on_hand",
                "target_xyz_m": ["x", "y", "a"],
            },
            {
                "name": "_probe_second",
                "role": "on_hand",
                "target_xyz_m": ["x", "y", "b"],
            },
        ],
        "relations": [],
    }
    try:
        normalized = _validate_move_tracked_point_args_contract(probe)
    except (TypeError, ValueError):
        return False
    inferred = coordinate_collinear_relations(normalized["points"])
    return any(relation in inferred for relation in normalized.get("relations", []))


_RESIDENT_CONTRACT_IMPLICITLY_INFERS_COORDINATE_ORDER = (
    _resident_contract_implicitly_infers_coordinate_order()
)


def validate_move_tracked_point_args(args: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one request without reviving legacy coordinate-order inference.

    The hot-reloadable module can run inside an interface process whose resident
    contract predates the coordinate/quick-mode separation.  Such a contract
    appends ``ordered_collinear`` whenever affine coordinate rows happen to lie
    on one line.  Remove exactly those derivable records here; a selected quick
    collinear preset is lowered to its ordered relation later by
    ``_relations_with_quick_constraints``.
    """

    normalized = _validate_move_tracked_point_args_contract(args)
    if not _RESIDENT_CONTRACT_IMPLICITLY_INFERS_COORDINATE_ORDER:
        return normalized
    inferred = coordinate_collinear_relations(normalized["points"])
    if inferred:
        normalized["relations"] = [
            relation
            for relation in normalized.get("relations", [])
            if relation not in inferred
        ]
    return normalized


MOVE_TRACKED_POINT_LOCAL_BUILD = (
    "official_v2_live_tracked_symbolic_rigid_trajectory_"
    "v87_coordinate_constraints_offhand_anchor_"
    "raw_rgbd_visual_gate_sub2mm_bounded_execution_x5_"
    "v91_no_step_ceiling"
)
TRAJECTORY_SCHEMA = "official_v2_move_tracked_point_trajectory"
TRAJECTORY_SCHEMA_VERSION = 4
MAX_ANCHOR_ERROR_M = 0.05
START_EEF_ANCHOR_ERROR_M = 0.03
STREAM_TRACKING_ERROR_RAD = 0.12
# A signed 0.018 rad knot is small enough for the fast X5 joints to finish
# in one evaluator action.  A slower neighbor can still be dragged while
# those joints keep moving.  The old stream treated anything under 0.12 rad
# as "waypoint done" and immediately commanded the next knot, so that lag
# grew until the following sample was labeled
# ``waypoint_tracking_not_progressing``.  Advance only inside this settle
# band.  Above it, reissue the same target: the fast joints stop, the
# disturbed joint can catch up, and the command itself is not enlarged.
STREAM_SETTLE_RAD = 0.030
# Last knot only.  0.005 rad is about 1.5 mm at a 0.30 m wrist lever,
# inside the 2 mm Cartesian gate before the visual correction reserve.
FINAL_EXECUTION_JOINT_SETTLE_RAD = 0.005
STREAM_MAX_CATCHUP_STEPS = 24
STREAM_CATCHUP_STALL_LIMIT = 4
STREAM_CATCHUP_GRACE_STEPS = 2
STREAM_CATCHUP_MIN_PROGRESS_RAD = 1.0e-4
START_STABLE_STEPS = 3
START_MAX_SETTLE_STEPS = 90
START_CAPTURE_TOLERANCE_RAD = 0.12
START_POSITION_STEP_RAD = 0.02
START_ARM_QVEL_RAD_S = 0.15
START_SOURCE_DRIFT_FLOOR_M = 0.02
MAX_TRANSLATION_M = 0.75
MAX_JOINT_STEP_RAD = 0.018
WAYPOINT_TOLERANCE_RAD = 0.018
MAX_TRACKING_ERROR_RAD = 0.35
STREAM_RECOVERY_ERROR_RAD = 0.22
# Allow more recovery attempts for compressed (省步) trajectories where each
# waypoint step is small (0.018 rad). The X5 controller needs additional
# observations to settle after a bounded execution bridge step.
STREAM_MAX_RECOVERY_STEPS = 6
# BEHAVIOR v37 pauses the immutable arm trajectory only when a near-open R1
# finger drifts inward. RoboDojo exposes one normalized X5 position target per
# side instead of independent finger efforts, so the equivalent controller
# uses the official X5 URDF scale and consecutive evaluator qpos observations.
GRIPPER_START_HANDOFF_LIMIT_M = 0.020
GRIPPER_RECOVERY_SOFT_LIMIT_M = 0.006
GRIPPER_RECOVERY_RELEASE_M = 0.003
GRIPPER_RECOVERY_HARD_LIMIT_M = 0.010
GRIPPER_RECOVERY_MAX_STEPS = 8
GRIPPER_RECOVERY_STABLE_STEPS = 2
GRIPPER_RECOVERY_POSITION_STEP_M = 0.001
GRIPPER_RECOVERY_UPSTREAM_QVEL_MPS = 0.02
GRIPPER_RECOVERY_UPSTREAM_EFFORT_N = 3.0
GRIPPER_OPEN_HOLD_MARGIN_M = 0.005
GRIPPER_OPEN_HOLD_THRESHOLD_M = max(
    0.0,
    float(ROBODOJO_X5_GRIPPER_SCALE_M[1]) - GRIPPER_OPEN_HOLD_MARGIN_M,
)
GRIPPER_OPEN_RECOVERY_DEADBAND_M = 0.001
FINAL_MAX_STABILITY_STEPS = 90
FINAL_COHERENCE_POSITION_STEP_RAD = 0.002
FINAL_LOAD_STABLE_STEPS = 3
FINAL_LOAD_MIN_RESIDUAL_RAD = 0.004
FINAL_LOAD_MAX_ROUNDS = 3
FINAL_LOAD_MAX_ROUND_RAD = 0.06
FINAL_LOAD_MAX_TOTAL_RAD = 0.12
FINAL_VISUAL_MIN_POINTS = 2
# A single RGB-D point is sufficient to correct a translational endpoint when
# an off-hand point supplies the fixed world reference.  Keep the historical
# two-point threshold for rigid-object/rebase diagnostics, but do not disable
# the terminal correction merely because the request has one moving point.
FINAL_VISUAL_CORRECTION_MIN_POINTS = 1
FINAL_VISUAL_MAX_ROUNDS = 2
FINAL_VISUAL_MAX_ANCHOR_REBASE_M = 0.030
FINAL_VISUAL_MAX_CONSTRAINT_ERROR_M = 0.050
FINAL_VISUAL_MAX_TRANSLATION_M = 0.080
FINAL_VISUAL_MAX_PLANNING_S = 12.0
FINAL_VISUAL_MIN_EXECUTION_RESERVE_S = 4.0
FINAL_OCCLUSION_MIN_DEPTH_MARGIN_M = 0.002
FINAL_EEF_DEPTH_REOBSERVE_RADIUS_PX = 2
FINAL_EEF_DEPTH_REOBSERVE_MAX_DEPTH_ERROR_M = 0.008
FINAL_EEF_DEPTH_REOBSERVE_MAX_POINT_ERROR_M = 0.006
FINAL_VISUAL_REACQUIRE_MAX_STEPS = 12
# A caller may retain the historical 30 mm public default for compatibility,
# but a successful execution must be certified inside this tighter envelope.
# Keep 0.5 mm of measurement/serialization margin below the user-visible 2 mm
# requirement; requests with a smaller explicit tolerance remain unchanged.
FINAL_EXECUTION_POSITION_TOLERANCE_M = 0.0015
FINAL_EXECUTION_POSITION_LIMIT_M = 0.002
# Leave a small action reserve after the compressed signed path.  It is only
# consumed when the RGB-D terminal gate requests a correction; ordinary paths
# still use the same number of evaluator actions as before.
FINAL_PRECISION_RESERVE_STEPS = 8
POST_PLAN_REBASE_MAX_ROUNDS = 2
POST_PLAN_REBASE_MAX_SOURCE_DRIFT_M = 0.030
POST_PLAN_REBASE_MIN_EXECUTION_RESERVE_S = 4.0
# The planner still emits the complete signed path.  Only the execution view
# may use a certified monotone subsequence.  The execution view can skip
# signed knots for scheduling, but the physical bridge must put the skipped
# motion back into controller-sized increments.  RoboDojo's evaluator reports
# proprioception after each action and its X5 position controller does not
# reliably follow a 0.18 rad command jump; such a jump used to leave the
# feedback corridor and produce a false ``waypoint_tracking_not_progressing``
# failure.  Keep the latency metadata for auditability, while clamping every
# emitted bridge command to the signed 0.018 rad X5 step.  A plan explicitly
# marked as a dense legacy replay remains dense for backwards compatibility.
# Conservative max for tight gripper-hold safety.  The 12x skill kept failing
# ``gripper_drift`` because per-adjacent-skip jumps still crossed the
# physical inertia envelope once any finger was holding a captured object.
# A 5x cap matches the existing visual-point bound and keeps every selected
# step within controller reach so the gripper cannot be shaken past the
# 10 mm hard limit on a single observation-to-observation correction.
# New plans subsample signed knots, then the bridge re-emits them at
# ``EXECUTION_MAX_COMMAND_DELTA_RAD``.  That saves actions relative to one
# command per 0.018 rad knot.  There is no separate step ceiling: a longer
# path simply uses more actions.  Legacy plans without a policy field stay
# dense.  Short paths stay dense because coalescing them saves nothing.
EXECUTION_DENSE_WAYPOINTS = False
EXECUTION_MAX_SPEEDUP_FACTOR = 2.8
EXECUTION_PREFERRED_MAX_STEPS = 60
EXECUTION_DENSE_SHORT_PATH_THRESHOLD = 48
# ``EXECUTION_MAX_COMMAND_DELTA_RAD`` is a transport/bridge limit, not a
# planner speedup hint.  It must not exceed the signed knot size: the evaluator
# needs one observation-sized X5 command per physical increment.  The scheduler
# may still select a monotone subsequence, but ``_execution_bridge`` expands it
# back to this bound before any action is emitted.
EXECUTION_SPEEDUP_FALLBACK_FACTOR = 2.8
# 0.05 rad is inside the X5 velocity limit for one 25 Hz evaluator step
# (limit is about 0.16 rad) and keeps a 2 rad joint path near 40 actions.
# 0.18 rad commands were the ones the controller could not finish.
EXECUTION_MAX_COMMAND_DELTA_RAD = 0.05
EXECUTION_MAX_JOINT_CORRIDOR_DEVIATION_RAD = 0.03
EXECUTION_SAFETY_TIME_STEPS = 2
NPOINT_VISUAL_MIN_POINTS = 3
NPOINT_VISUAL_MAX_SPEEDUP_FACTOR = 5.0
NPOINT_REACQUISITION_MAX_HOLD_STEPS = 12
REGISTRATION_EEF_INLIER_M = 0.008
REGISTRATION_EEF_MAX_RECOVERY_M = 0.030
REGISTRATION_REACQUIRE_MAX_STEPS = 16
REGISTRATION_REACQUIRE_MAX_S = 8.0
RIGID_PAIR_ACTIVATION_STABLE_STEPS = 2
RIGID_PAIR_ACTIVATION_MAX_WAIT_STEPS = 6
RIGID_PAIR_ACTIVATION_MAX_WAIT_S = 20.0
RIGID_PAIR_SOURCE_MAX_POINT_CORRECTION_M = 0.003
RIGID_PAIR_ACTIVATION_INITIAL_RETRY_REASONS = frozenset(
    {
        "rigid_pair_rgbd_correction_exceeded",
        "measured_pair_collapsed",
        "rigid_pair_point_unobserved",
        "rigid_pair_projection_requires_fresh_confirmation",
    }
)
RIGID_PAIR_ACTIVATION_FRESH_RETRY_REASONS = frozenset(
    {
        "rigid_pair_rgbd_correction_exceeded",
        "measured_pair_collapsed",
        "rigid_pair_point_unobserved",
        "joint_rgbd_observation_unavailable",
        "pair_candidate_unobserved",
        "rigid_pair_distance_mismatch",
        "rigid_pair_identity_ambiguous",
        "individual_candidate_gate_rejected",
        "required_observation_prior_unavailable",
        "rigid_pair_projection_requires_fresh_confirmation",
    }
)
# Keep scheduler and bridge comparisons stable across JSON serialization and
# repeated floating-point subtraction. This is far below a physical command.
EXECUTION_STEP_NUMERIC_TOLERANCE_RAD = 1.0e-8
# Budget against the slowest evaluator period observed during this request,
# with the same modest jitter reserve as BEHAVIOR official_v2.
EXECUTION_CADENCE_SAFETY_FACTOR = 1.25
SUCCESSOR_RETENTION_EXTRA_S = 180.0
PLANNING_FRACTION = 0.60
MIN_EXECUTION_RESERVE_S = 0.05
MAX_TRAJECTORY_WAYPOINTS = 4096
STABLE_STEPS = 3
# The RoboDojo evaluator port used by this interface has no 50-step tool
# ceiling.  Keep the public ``max_steps`` field for compatibility/validation,
# but give dense execution a generous port-local ceiling so recovery,
# reacquisition, and final stability checks never trigger a compression
# fallback.  ``MAX_TRAJECTORY_WAYPOINTS`` remains the independent finite input
# guard; this value is the transport allowance, not a waypoint selector.
EXECUTION_DENSE_ACTION_BUDGET = 100_000


def _stream_tracking_decision(
    *,
    tracking_error_rad: float,
    previous_error_rad: float,
    recovery_step: int,
    stall_steps: int,
    settle_rad: float = STREAM_SETTLE_RAD,
) -> dict[str, Any]:
    """Decide whether the current X5 waypoint may advance.

    ``advance`` commands the next signed knot.  ``hold`` reissues this knot
    so a disturbed joint can settle without a larger jerk.  The two failure
    actions are the existing motion reasons.
    """

    error = float(tracking_error_rad)
    previous = float(previous_error_rad)
    recovery = int(recovery_step)
    stall = int(stall_steps)
    if not math.isfinite(error) or error < 0.0:
        return {
            "action": "waypoint_tracking_error",
            "recovery_step": recovery,
            "stall_steps": stall,
            "progress_delta_rad": None,
        }
    if error <= float(settle_rad):
        return {
            "action": "advance",
            "recovery_step": 0,
            "stall_steps": 0,
            "progress_delta_rad": (
                None if not math.isfinite(previous) else float(previous - error)
            ),
        }
    progress = None if not math.isfinite(previous) else float(previous - error)
    improved = bool(progress is None or progress >= STREAM_CATCHUP_MIN_PROGRESS_RAD)
    if error > STREAM_RECOVERY_ERROR_RAD:
        if recovery >= STREAM_CATCHUP_GRACE_STEPS and not improved:
            return {
                "action": "waypoint_tracking_error",
                "recovery_step": recovery,
                "stall_steps": stall,
                "progress_delta_rad": progress,
            }
        if recovery >= STREAM_MAX_CATCHUP_STEPS:
            return {
                "action": "waypoint_tracking_error",
                "recovery_step": recovery,
                "stall_steps": stall,
                "progress_delta_rad": progress,
            }
        return {
            "action": "hold",
            "recovery_step": recovery + 1,
            "stall_steps": 0 if improved else stall,
            "progress_delta_rad": progress,
        }
    if recovery >= STREAM_MAX_CATCHUP_STEPS and not improved:
        return {
            "action": "waypoint_tracking_not_progressing",
            "recovery_step": recovery,
            "stall_steps": stall,
            "progress_delta_rad": progress,
        }
    if recovery >= STREAM_CATCHUP_GRACE_STEPS and not improved:
        stall += 1
    elif improved:
        stall = 0
    if stall >= STREAM_CATCHUP_STALL_LIMIT:
        return {
            "action": "waypoint_tracking_not_progressing",
            "recovery_step": recovery,
            "stall_steps": stall,
            "progress_delta_rad": progress,
        }
    return {
        "action": "hold",
        "recovery_step": recovery + 1,
        "stall_steps": stall,
        "progress_delta_rad": progress,
    }


def _promote_unique_frame_origin(
    points: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use the unique zero-offset full-rank point as the world reference.

    Calls such as digit ``z+0.1`` plus ``place`` at ``(x, y, z)`` send both
    points as on-hand.  Averaging those rows pulls the target off ``place``
    by centimetres, so the 2 mm gate can never pass.  The point whose three
    axes already span every shared variable and whose constants are strictly
    the smallest is the frame origin; it is reclassified as off-hand.
    Explicit off-hand points and ambiguous frames are left unchanged.
    """

    copied = [dict(point) for point in points]
    if len(copied) < 2:
        return copied, {"applied": False, "reason": "single_point"}
    if any(str(point.get("role") or "on_hand") == "off_hand" for point in copied):
        return copied, {"applied": False, "reason": "explicit_off_hand_present"}
    variable_names: set[str] = set()
    parsed: list[tuple[dict[str, Any], np.ndarray, float]] = []
    for point in copied:
        target = point.get("target_xyz_m")
        if not isinstance(target, Mapping):
            return copied, {"applied": False, "reason": "non_affine_target"}
        forms = []
        try:
            for axis in ("x", "y", "z"):
                forms.append(
                    _affine_form(
                        target.get(axis),
                        label=f"{point.get('name')}.{axis}",
                    )
                )
        except (TypeError, ValueError):
            return copied, {"applied": False, "reason": "affine_parse_failed"}
        if any(bool(form.free) for form in forms):
            return copied, {"applied": False, "reason": "free_coordinate"}
        for form in forms:
            variable_names.update(name for name, _coef in form.coefficients)
        parsed.append((point, forms, 0.0))
    if not variable_names:
        return copied, {"applied": False, "reason": "no_shared_variables"}
    ordered = sorted(variable_names)
    candidates: list[tuple[float, str]] = []
    for index, (point, forms, _unused) in enumerate(parsed):
        matrix = np.zeros((len(forms), len(ordered)), dtype=np.float64)
        constant_l1 = 0.0
        for row_index, form in enumerate(forms):
            constant_l1 += abs(float(form.constant))
            coefficients = dict(form.coefficients)
            for column, name in enumerate(ordered):
                matrix[row_index, column] = float(coefficients.get(name, 0.0))
        if int(np.linalg.matrix_rank(matrix)) != len(ordered):
            continue
        candidates.append((constant_l1, str(point.get("name"))))
        parsed[index] = (point, forms, constant_l1)
    if not candidates:
        return copied, {"applied": False, "reason": "no_full_rank_origin"}
    minimum = min(cost for cost, _name in candidates)
    origins = [name for cost, name in candidates if cost == minimum]
    if len(origins) != 1:
        return copied, {"applied": False, "reason": "ambiguous_frame_origin"}
    origin = origins[0]
    for point in copied:
        if str(point.get("name")) == origin:
            point["role"] = "off_hand"
    return copied, {
        "applied": True,
        "reason": "unique_minimal_constant_full_rank_frame",
        "reference_point": origin,
        "constant_l1_m": float(minimum),
    }


def _gentle_joint_command(
    measured_q: np.ndarray,
    target_q: np.ndarray,
    *,
    max_step_rad: float,
) -> np.ndarray:
    """Move at most ``max_step_rad`` from the measured joints toward ``target_q``."""

    measured = np.asarray(measured_q, dtype=np.float64).reshape(-1)
    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    delta = target - measured
    span = float(np.linalg.norm(delta, ord=np.inf))
    limit = float(max_step_rad)
    if not math.isfinite(span) or span <= limit:
        return target.copy()
    return measured + delta * (limit / span)


def _trajectory_uses_dense_execution(trajectory: Mapping[str, Any]) -> bool:
    """Return the execution policy recorded by a signed trajectory.

    Plans written before the bounded execution policy did not carry an
    explicit policy field.  Treating those artifacts as dense preserves their
    original replay semantics; newly generated plans always write the field.
    """

    tracking = trajectory.get("tracking")
    if not isinstance(tracking, Mapping):
        return bool(EXECUTION_DENSE_WAYPOINTS)
    policy = str(tracking.get("execution_waypoint_policy") or "").strip()
    if policy == "all_signed_waypoints":
        return True
    if policy == "bounded_monotone_subsample":
        return False
    compression_disabled = tracking.get("execution_compression_disabled")
    if isinstance(compression_disabled, bool):
        return compression_disabled
    # Legacy manually-authored/snapshotted plans have no policy metadata.
    return True


def _x5_gripper_capture_error(
    current: Mapping[str, Any],
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    per_side: dict[str, dict[str, float]] = {}
    for side in ("left", "right"):
        current_values = np.asarray(current[side], dtype=np.float64).reshape(-1)
        capture_values = np.asarray(capture[side], dtype=np.float64).reshape(-1)
        if (
            current_values.size != 1
            or capture_values.size != 1
            or not np.all(np.isfinite(current_values))
            or not np.all(np.isfinite(capture_values))
        ):
            raise ValueError(f"{side} X5 gripper state must contain one finite value")
        current_normalized = float(current_values[0])
        capture_normalized = float(capture_values[0])
        current_joint_m = x5_gripper_joint_position_m(current_values)
        capture_joint_m = x5_gripper_joint_position_m(capture_values)
        signed_inward_error_m = capture_joint_m - current_joint_m
        per_side[side] = {
            "current_normalized": current_normalized,
            "capture_normalized": capture_normalized,
            "normalized_error": abs(current_normalized - capture_normalized),
            "current_joint_position_m": current_joint_m,
            "capture_joint_position_m": capture_joint_m,
            "error_m": abs(current_joint_m - capture_joint_m),
            "signed_inward_error_m": signed_inward_error_m,
            "near_open_capture": bool(capture_joint_m >= GRIPPER_OPEN_HOLD_THRESHOLD_M),
        }
    return {
        "per_side": per_side,
        "max_normalized_error": max(
            item["normalized_error"] for item in per_side.values()
        ),
        "max_error_m": max(item["error_m"] for item in per_side.values()),
    }


def _x5_gripper_hold_targets(
    ctx,
    frozen: Mapping[str, Any],
) -> tuple[dict[str, list[float]], dict[str, dict[str, Any]]]:
    """Translate the v37 finger hold policy to X5 position commands."""

    commands: dict[str, list[float]] = {}
    report: dict[str, dict[str, Any]] = {}
    frozen_latches = frozen.get("gripper_command_latch") or {}
    for side in ("left", "right"):
        capture = float(
            np.asarray(frozen["gripper_q"][side], dtype=np.float64).reshape(1)[0]
        )
        observed = float(
            np.asarray(ctx.world.gripper_qpos_list(side), dtype=np.float64).reshape(1)[
                0
            ]
        )
        raw_latch = frozen_latches.get(side)
        close_latched = bool(raw_latch is not None and float(raw_latch) <= 1.0e-9)
        capture_joint_m = x5_gripper_joint_position_m([capture])
        near_open = bool(capture_joint_m >= GRIPPER_OPEN_HOLD_THRESHOLD_M)
        if close_latched:
            command = float(raw_latch)
            mode = "verified_close_keepalive"
        elif near_open:
            # X5 has no zero-effort action. Holding the farther-open of the
            # frozen capture and current observation is its no-close analogue.
            command = max(capture, observed)
            mode = "near_open_outward_only_position_hold"
        else:
            command = capture
            mode = "capture_qpos_static_position_hold"
        command = float(np.clip(command, 0.0, 1.0))
        commands[side] = [command]
        report[side] = {
            "mode": mode,
            "capture_normalized": capture,
            "observed_normalized": observed,
            "command_normalized": command,
            "capture_joint_position_m": capture_joint_m,
            "observed_joint_position_m": x5_gripper_joint_position_m([observed]),
            "near_open_capture": near_open,
            "close_keepalive_active": close_latched,
            "inward_position_command": bool(command < observed - 1.0e-12),
        }
    return commands, report


def _planning_budget(timeout_s: float) -> float:
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("move-tracked timeout_s must be finite and positive")
    reserve = min(max(MIN_EXECUTION_RESERVE_S, 0.01), 0.25 * timeout)
    return max(
        0.01,
        min(timeout, PLANNING_FRACTION * timeout, timeout - reserve),
    )


def _camera_pose_payload(
    position: Sequence[float] | None,
    quaternion: Sequence[float] | None,
    *,
    label: str = "move-tracked head camera pose",
) -> dict[str, Any] | None:
    """Canonicalize the frozen evaluator camera pose used by faceto."""

    if position is None and quaternion is None:
        return None
    if position is None or quaternion is None:
        raise ValueError(f"{label} requires both position and quaternion")
    try:
        position_array = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion_array = np.asarray(quaternion, dtype=np.float64).reshape(4)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain a 3-vector and a 4-vector") from exc
    if not np.all(np.isfinite(position_array)) or not np.all(
        np.isfinite(quaternion_array)
    ):
        raise ValueError(f"{label} contains a non-finite value")
    norm = float(np.linalg.norm(quaternion_array))
    if norm <= 1.0e-12:
        raise ValueError(f"{label} quaternion is degenerate")
    return {
        "position_robot_base_m": position_array.astype(float).tolist(),
        "quaternion_xyzw": (quaternion_array / norm).astype(float).tolist(),
        "frame": "robot_base",
        "source": "frozen_evaluator_head_extrinsic",
    }


def _head_camera_pose_x5(
    ctx,
    *,
    expected_sequence: int,
    base_rotation: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Read one synchronized head camera pose in the X5 robot-base frame."""

    camera = ctx.world._official_adapter.camera("head", require_depth=True)
    if int(camera.get("sequence", -1)) != int(expected_sequence):
        raise RuntimeError(
            "head camera pose is not synchronized with the tracked-point observation"
        )
    extrinsic = np.asarray(camera.get("extrinsic_matrix"), dtype=np.float64).reshape(
        4, 4
    )
    base_rotation_array = np.asarray(base_rotation, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(extrinsic)) or not np.all(
        np.isfinite(base_rotation_array)
    ):
        raise RuntimeError("head camera pose contains a non-finite value")
    camera_rotation = base_rotation_array.T @ extrinsic[:3, :3]
    camera_position = base_rotation_array.T @ extrinsic[:3, 3]
    camera_quaternion = mat_to_quat_xyzw(camera_rotation)
    _camera_pose_payload(camera_position, camera_quaternion)
    return camera_position, camera_quaternion


def _move_tracked_release_motion_retention(
    manager,
    lease_state: dict[str, Any],
) -> dict[str, Any]:
    """Release a request-scoped tracker lease exactly once."""

    cached = lease_state.get("release")
    if isinstance(cached, dict):
        release = dict(cached)
    else:
        lease_id = str(lease_state.get("lease_id") or "").strip()
        if not lease_id:
            release = {
                "ok": True,
                "released": False,
                "lease_id": None,
                "reason": "no_motion_retention_lease",
            }
        else:
            end_retention = getattr(manager, "end_motion_retention", None)
            if not callable(end_retention):
                release = {
                    "ok": False,
                    "released": False,
                    "lease_id": lease_id,
                    "reason": "motion_retention_release_unavailable",
                }
            else:
                try:
                    raw_release = end_retention(lease_id)
                    if not isinstance(raw_release, dict):
                        raise TypeError(
                            "motion retention release must return an object"
                        )
                    release = dict(raw_release)
                except Exception as exc:
                    # The tracker lease remains bounded even if explicit cleanup fails.
                    release = {
                        "ok": False,
                        "released": False,
                        "lease_id": lease_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        lease_state["release"] = dict(release)

    # Pair mode is committed by the tracker before its first observation can
    # be validated. Keep its cleanup in the request-finally path so planning,
    # cancellation, and validation failures cannot leak pair state.
    pair_lease = lease_state.get("_rigid_pair_lease_state")
    if (
        isinstance(pair_lease, dict)
        and bool(pair_lease.get("active"))
        and not bool(pair_lease.get("preserve_after_success"))
    ):
        if manager is None:
            cleanup = {
                "ok": False,
                "active": None,
                "reason": "rigid_pair_manager_unavailable_during_cleanup",
            }
        else:
            cleanup = _deactivate_rigid_pair_x5(
                manager,
                reason="move_tracked_point_request_did_not_complete_successfully",
                required=False,
            )
        pair_lease["active"] = False
        pair_lease["cleanup"] = json_ready(cleanup)
    return dict(release)


def _move_tracked_preserve_motion_retention_for_successor(
    lease_state: dict[str, Any],
) -> dict[str, Any]:
    """Keep only tracker identity state across a bounded successful handoff."""

    cached = lease_state.get("release")
    if isinstance(cached, dict):
        return dict(cached)
    lease_id = str(lease_state.get("lease_id") or "").strip()
    lease = lease_state.get("lease")
    if not lease_id or not isinstance(lease, Mapping):
        return {
            "ok": False,
            "released": False,
            "lease_id": lease_id or None,
            "reason": "successor_motion_retention_lease_missing",
        }
    release = {
        "ok": True,
        "released": False,
        "lease_id": lease_id,
        "reason": "bounded_successor_identity_retention",
        "duration_s": float(lease.get("duration_s", 0.0)),
        "coordinates_published_while_unobserved": False,
        "expires_without_successor": True,
        "superseded_by_next_matching_request": True,
    }
    lease_state["release"] = dict(release)
    return release


def _trim_visual_correction_to_first_feasible(
    *,
    state: LocalRobotState,
    arm: str,
    q_start: Sequence[float],
    endpoint: Mapping[str, Any],
    path: Mapping[str, Any],
    target_points: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]] | None,
    fixed_points_robot_base_m: Sequence[Sequence[float]] | None,
    fixed_target_points: Sequence[Mapping[str, Any]] | None,
    pos_tol_m: float,
    ori_tol_deg: float,
    inequalities: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Keep the first locally verified waypoint satisfying the public goal."""

    dense_raw = path.get("waypoints")
    if not isinstance(dense_raw, list) or not dense_raw:
        raise ValueError("final visual correction path is empty")
    controlled_targets = [dict(item) for item in target_points]
    fixed_targets = (
        None
        if fixed_points_robot_base_m is None
        else [dict(item) for item in (fixed_target_points or [])]
    )
    anchors = _strict_point_matrix(
        endpoint.get("anchors_eef_m"),
        expected_count=len(controlled_targets),
        label="final visual correction EEF anchors",
    )
    start = np.asarray(q_start, dtype=np.float64).reshape(ARM_DOF)
    if not np.all(np.isfinite(start)):
        raise ValueError("final visual correction trim start q is non-finite")
    if np.any(start < X5_JOINT_LOWER) or np.any(start > X5_JOINT_UPPER):
        raise ValueError("final visual correction trim start q exceeds X5 limits")
    start_position, start_quaternion = eef_pose(state, arm, start)
    start_position = np.asarray(start_position, dtype=np.float64).reshape(3)
    start_quaternion = np.asarray(start_quaternion, dtype=np.float64).reshape(4)

    selected: dict[str, Any] | None = None
    validated_waypoints: list[np.ndarray] = []
    for waypoint_index, raw_q in enumerate(dense_raw):
        q = np.asarray(raw_q, dtype=np.float64).reshape(ARM_DOF)
        if not np.all(np.isfinite(q)):
            raise ValueError("final visual correction waypoint is non-finite")
        if np.any(q < X5_JOINT_LOWER) or np.any(q > X5_JOINT_UPPER):
            raise ValueError("final visual correction waypoint exceeds X5 limits")
        validated_waypoints.append(q.copy())
        position, quaternion = eef_pose(state, arm, q)
        position = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
        resolved_points = (
            position[None, :] + (quat_to_mat_xyzw(quaternion) @ anchors.T).T
        )
        constraints = evaluate_target_constraints(
            resolved_points,
            controlled_targets,
            tolerance_m=float(pos_tol_m),
            relations=[dict(item) for item in (relations or [])],
            inequalities=[dict(item) for item in (inequalities or [])],
            orientation_tolerance_deg=float(ori_tol_deg),
            fixed_points_robot_base_m=fixed_points_robot_base_m,
            fixed_target_points=fixed_targets,
        )
        if bool(constraints.get("ok")):
            selected = {
                "index": int(waypoint_index),
                "q": q,
                "position": position,
                "quaternion": quaternion,
                "points": resolved_points,
                "constraints": constraints,
            }
            break
    if selected is None:
        raise ValueError(
            "final visual correction path contains no waypoint satisfying "
            "the immutable goal"
        )

    selected_index = int(selected["index"])
    retained_waypoints = validated_waypoints[: selected_index + 1]
    selected_q = np.asarray(selected["q"], dtype=np.float64).reshape(ARM_DOF)
    selected_position = np.asarray(selected["position"], dtype=np.float64).reshape(3)
    selected_quaternion = np.asarray(selected["quaternion"], dtype=np.float64).reshape(
        4
    )
    selected_points = np.asarray(selected["points"], dtype=np.float64).reshape(
        len(controlled_targets), 3
    )
    joint_path = np.vstack([start, *retained_waypoints])
    joint_deltas = np.diff(joint_path, axis=0)
    actual_max_joint_step = float(
        np.max(np.abs(joint_deltas)) if joint_deltas.size else 0.0
    )
    actual_joint_path_length = float(
        np.sum(np.max(np.abs(joint_deltas), axis=1)) if joint_deltas.size else 0.0
    )
    original_endpoint_position = np.asarray(
        endpoint.get("final_eef_position_m"), dtype=np.float64
    ).reshape(3)
    original_translation = float(
        np.linalg.norm(original_endpoint_position - start_position)
    )
    selected_translation = float(np.linalg.norm(selected_position - start_position))
    orientation_dot = abs(float(np.dot(start_quaternion, selected_quaternion)))
    selected_orientation = math.degrees(
        2.0 * math.acos(float(np.clip(orientation_dot, -1.0, 1.0)))
    )

    trimmed_endpoint = dict(endpoint)
    trimmed_endpoint.update(
        {
            "q_final": selected_q,
            "resolved_points_robot_base_m": selected_points,
            "final_eef_position_m": selected_position,
            "final_eef_quaternion_xyzw": selected_quaternion,
            "constraints": selected["constraints"],
            "visual_correction_first_feasible_waypoint_index": selected_index,
            "visual_correction_original_endpoint_q": json_ready(
                endpoint.get("q_final")
            ),
        }
    )
    trimmed_path = dict(path)
    trimmed_path.update(
        {
            "waypoints": [q.astype(float).tolist() for q in retained_waypoints],
            "joint_waypoint_count": int(len(retained_waypoints)),
            "translation_m": selected_translation,
            "orientation_deg": selected_orientation,
            "actual_max_joint_step_rad": actual_max_joint_step,
            "joint_path_length_inf_rad": actual_joint_path_length,
            "visual_correction_prefix_trimmed": bool(
                selected_index + 1 < len(dense_raw)
            ),
            "visual_correction_original_dense_waypoint_count": int(len(dense_raw)),
            "visual_correction_first_feasible_waypoint_index": selected_index,
        }
    )
    return {
        "endpoint": trimmed_endpoint,
        "path": trimmed_path,
        "report": {
            "applied": bool(selected_index + 1 < len(dense_raw)),
            "original_dense_waypoint_count": int(len(dense_raw)),
            "first_feasible_waypoint_index": selected_index,
            "retained_dense_waypoint_count": int(len(retained_waypoints)),
            "discarded_dense_waypoint_count": int(
                len(dense_raw) - len(retained_waypoints)
            ),
            "original_endpoint_translation_m": original_translation,
            "selected_endpoint_translation_m": selected_translation,
            "selected_endpoint_orientation_deg": selected_orientation,
            "selected_constraints": json_ready(selected["constraints"]),
            "selected_points_robot_base_m": selected_points.astype(float).tolist(),
            "selection_rule": "first_complete_constraint_feasible_fk_waypoint",
        },
    }


def _strict_point_matrix(
    raw: Any,
    *,
    expected_count: int,
    label: str,
) -> np.ndarray:
    """Validate point boundaries instead of flattening an arbitrary buffer."""

    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a nested N x 3 matrix") from exc
    if matrix.ndim != 2 or matrix.shape != (int(expected_count), 3):
        raise ValueError(
            f"{label} must be a nested N x 3 matrix with shape "
            f"({int(expected_count)}, 3)"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must contain finite coordinates")
    return matrix.copy()


def _common_result() -> dict[str, Any]:
    return {
        "execution": "official_action_only",
        "observation_contract": [
            "current_evaluator_head_rgb",
            "current_evaluator_depth_linear",
            "current_evaluator_camera_relative_pose",
            "same_observation_evaluator_proprioception",
            "subsequent_evaluator_proprioception",
            "submission_local_static_official_x5_model",
            "policy_owned_tracked_point_state",
        ],
        "direct_simulator_mutation": False,
        "global_pose_available": False,
        "scene_geometry_truth_available": False,
        "collision_truth_available": False,
        "contact_truth_used": False,
        "grasp_truth_used": False,
        "segmentation_used": False,
        "object_identity_verified": False,
    }


def _set_failure(
    ctx,
    result_writer: Callable[..., None],
    *,
    stage: str,
    error: str,
    args: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> None:
    result_writer(
        ctx,
        "move_tracked_point",
        {
            **_common_result(),
            "ok": False,
            "error": str(error),
            "failure_stage": str(stage),
            "request": json_ready(args),
            **json_ready(dict(details or {})),
        },
    )


def _sequence(ctx) -> int:
    return int(ctx.world._official_adapter.status()["sequence"])


def _live(
    ctx,
    names: list[str],
    episode_id: str,
    *,
    manager=None,
) -> dict[str, Any]:
    source = manager if manager is not None else ctx.tracker
    try:
        snapshot = source.observed_active_points_snapshot(
            names,
            episode_id=episode_id,
        )
    except Exception as exc:
        return {
            "ok": False,
            "reason": "tracked_point_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "entries": {},
            "unavailable": {str(name): "observation_provider_error" for name in names},
            "observation_sequence": None,
            "episode_id": str(episode_id),
        }
    if not isinstance(snapshot, dict):
        return {
            "ok": False,
            "reason": "tracked_point_invalid",
            "error": "active atomic snapshot must be an object",
            "entries": {},
            "unavailable": {str(name): "invalid_snapshot" for name in names},
            "observation_sequence": None,
            "episode_id": str(episode_id),
        }
    return snapshot


def _x5_kinematic_observation(ctx, arm: str) -> dict[str, Any]:
    """Read one X5 EEF and gripper from the current evaluator sequence."""

    position, quaternion = _eef_base_xyzw(ctx, arm)
    gripper = np.asarray(ctx.world.gripper_qpos_list(arm), dtype=np.float64).reshape(1)
    if not (
        np.all(np.isfinite(position))
        and np.all(np.isfinite(quaternion))
        and np.all(np.isfinite(gripper))
    ):
        raise ValueError(f"current {arm} X5 kinematics contain non-finite values")
    return {
        "position": position.copy(),
        "quaternion": quaternion.copy(),
        "gripper": gripper.copy(),
        "sequence": _sequence(ctx),
    }


class _MoveTrackedPointSourceManager:
    """Merge visual RGB-D tracks with live physical X5 EEF landmarks."""

    def __init__(
        self,
        ctx,
        tracker,
        *,
        predefined_names: Sequence[str],
        episode_id: str,
        arm: str = "right",
    ) -> None:
        self._ctx = ctx
        self._tracker = tracker
        self._predefined_names = frozenset(str(name) for name in predefined_names)
        self._episode_id = str(episode_id)
        self._arm = "right"
        self.set_arm(arm)
        token = hashlib.sha256(self._episode_id.encode("utf-8")).hexdigest()[:16]
        self._session_id = f"eef_points_{token}"
        self._image_id = "current_proprio_fk"
        self._synthetic_leases: set[str] = set()

    @property
    def predefined_names(self) -> frozenset[str]:
        return self._predefined_names

    def set_arm(self, arm: str) -> None:
        side = str(arm).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("predefined EEF point arm must be left or right")
        self._arm = side

    def visual_names(self, names: Sequence[str]) -> list[str]:
        return [str(name) for name in names if str(name) not in self._predefined_names]

    def visual_point_count(self, names: Sequence[str]) -> int:
        return len(self.visual_names(names))

    def observed_active_points_snapshot(
        self,
        names: Sequence[str],
        *,
        episode_id: str,
    ) -> dict[str, Any]:
        requested = [str(name) for name in names]
        if str(episode_id) != self._episode_id:
            return {
                "ok": False,
                "reason": "episode_changed",
                "entries": {},
                "unavailable": {name: "episode_changed" for name in requested},
                "observation_sequence": None,
                "episode_id": str(episode_id),
            }
        sequence = _sequence(self._ctx)
        visual_names = self.visual_names(requested)
        if visual_names:
            if self._tracker is None:
                return {
                    "ok": False,
                    "reason": "tracked_point_provider_unavailable",
                    "entries": {},
                    "unavailable": {
                        name: "tracked_point_provider_unavailable"
                        for name in visual_names
                    },
                    "observation_sequence": sequence,
                    "episode_id": self._episode_id,
                }
            base = self._tracker.observed_active_points_snapshot(
                visual_names,
                episode_id=self._episode_id,
            )
            if not isinstance(base, Mapping):
                return {
                    "ok": False,
                    "reason": "tracked_point_invalid",
                    "entries": {},
                    "unavailable": {name: "invalid_snapshot" for name in visual_names},
                    "observation_sequence": sequence,
                    "episode_id": self._episode_id,
                }
            base = dict(base)
            session_id = str(base.get("session_id") or "").strip()
            image_id = str(base.get("image_id") or "").strip()
            if session_id:
                self._session_id = session_id
            if image_id:
                self._image_id = image_id
        else:
            base = {
                "ok": True,
                "reason": None,
                "entries": {},
                "unavailable": {},
                "observation_sequence": sequence,
                "episode_id": self._episode_id,
            }

        entries = dict(base.get("entries") or {})
        predefined_requested = [
            name for name in requested if name in self._predefined_names
        ]
        if predefined_requested:
            observation = _x5_kinematic_observation(self._ctx, self._arm)
            points = predefined_eef_points_robot_base_m(
                predefined_requested,
                eef_position_robot_base_m=observation["position"],
                eef_quaternion_xyzw=observation["quaternion"],
                gripper_q_m=observation["gripper"],
            )
            for name in predefined_requested:
                entries[name] = {
                    "name": name,
                    "status": "observed",
                    "observed": True,
                    "retired": False,
                    "xyz_in_robot_base_coord_m": points[name].astype(float).tolist(),
                    "xyz_source": (
                        "submission_local_static_x5_eef_geometry_and_current_"
                        "evaluator_proprio_fk"
                    ),
                    "measurement_source": (
                        "submission_local_static_x5_eef_geometry_and_current_"
                        "evaluator_proprio_fk"
                    ),
                    "point_source_kind": "predefined_eef_point",
                    "predefined_eef_point": True,
                    "predefined_eef_local_m": predefined_eef_point_local_m(
                        name, observation["gripper"]
                    )
                    .astype(float)
                    .tolist(),
                    "arm": self._arm,
                    "observation_sequence": sequence,
                    "episode_id": self._episode_id,
                }
        sequence_after = _sequence(self._ctx)
        if sequence_after != sequence:
            return {
                "ok": False,
                "reason": "tracker_observation_mismatch",
                "entries": {},
                "unavailable": {
                    name: "observation_changed_while_resolving_predefined_eef_point"
                    for name in requested
                },
                "observation_sequence": sequence_after,
                "episode_id": self._episode_id,
            }

        unavailable = dict(base.get("unavailable") or {})
        missing = [name for name in requested if name not in entries]
        for name in missing:
            unavailable.setdefault(name, "tracked_point_unavailable")
        try:
            base_sequence = int(base.get("observation_sequence", sequence))
        except (TypeError, ValueError):
            base_sequence = -1
        synchronized = base_sequence == sequence
        ok = bool(base.get("ok") and synchronized and not missing)
        return {
            **base,
            "ok": ok,
            "reason": (
                None
                if ok
                else str(
                    base.get("reason")
                    or (
                        "tracker_observation_mismatch"
                        if not synchronized
                        else "tracked_point_unavailable"
                    )
                )
            ),
            "entries": {name: entries[name] for name in requested if name in entries},
            "unavailable": unavailable,
            "requested_names": requested,
            "observation_sequence": sequence,
            "episode_id": self._episode_id,
            "session_id": str(base.get("session_id") or self._session_id),
            "image_id": str(base.get("image_id") or self._image_id),
            "measurement_source": (
                "hybrid_visual_rgbd_and_submission_local_x5_eef_fk"
                if visual_names and predefined_requested
                else "submission_local_static_x5_eef_geometry_and_current_evaluator_proprio_fk"
                if predefined_requested
                else base.get("measurement_source")
            ),
            "predefined_eef_point_names": predefined_requested,
            "visual_tracked_point_names": visual_names,
            "predefined_eef_arm": self._arm if predefined_requested else None,
            "predictions_published": False,
        }

    def begin_motion_retention(
        self,
        names: Sequence[str],
        *,
        episode_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        visual_names = self.visual_names(names)
        if visual_names:
            if self._tracker is None:
                return {"ok": False, "reason": "tracker_unavailable", "lease_id": None}
            return self._tracker.begin_motion_retention(
                visual_names,
                episode_id=episode_id,
                timeout_s=timeout_s,
            )
        lease_id = f"predefined_eef_{secrets.token_hex(8)}"
        self._synthetic_leases.add(lease_id)
        return {
            "ok": True,
            "lease_id": lease_id,
            "names": [],
            "predefined_eef_point_names": [str(name) for name in names],
            "duration_s": float(timeout_s),
            "reason": "no_visual_tracking_required",
        }

    def end_motion_retention(self, lease_id: str) -> dict[str, Any]:
        token = str(lease_id)
        if token in self._synthetic_leases:
            self._synthetic_leases.remove(token)
            return {
                "ok": True,
                "released": True,
                "lease_id": token,
                "reason": "predefined_eef_lease_released",
            }
        if self._tracker is None:
            return {
                "ok": False,
                "released": False,
                "lease_id": token,
                "reason": "tracker_unavailable",
            }
        return self._tracker.end_motion_retention(token)

    def activate_rigid_points(
        self, names: Sequence[str], *, episode_id: str
    ) -> dict[str, Any]:
        visual_names = self.visual_names(names)
        if len(visual_names) < 2:
            return {
                "ok": True,
                "active": False,
                "reason": "predefined_eef_points_are_exact_local_fk_anchors",
                "requested_names": [str(name) for name in names],
                "activated_names": [],
                "visual_tracked_point_names": visual_names,
            }
        activate_all = getattr(self._tracker, "activate_rigid_points", None)
        if callable(activate_all):
            return activate_all(visual_names, episode_id=episode_id)
        if len(visual_names) == 2:
            activate_pair = getattr(self._tracker, "activate_rigid_pair", None)
            if callable(activate_pair):
                return activate_pair(visual_names, episode_id=episode_id)
        return {
            "ok": True,
            "active": False,
            "reason": "tracker_has_no_n_point_fusion; local_geometry_validation_only",
            "requested_names": [str(name) for name in names],
            "activated_names": [],
            "visual_tracked_point_names": visual_names,
        }

    def activate_rigid_pair(
        self, names: Sequence[str], *, episode_id: str
    ) -> dict[str, Any]:
        return self.activate_rigid_points(names, episode_id=episode_id)

    def activate_on_hand_eef_observation_prior(
        self,
        names: Sequence[str],
        *,
        episode_id: str,
        session_id: str,
        image_id: str,
        arm: str,
        anchors_eef_m: Any,
    ) -> dict[str, Any]:
        requested = [str(name) for name in names]
        visual_names = self.visual_names(requested)
        if not visual_names:
            return {
                "ok": True,
                "active": False,
                "reason": "predefined_eef_points_use_direct_local_fk",
                "names": requested,
                "activation_observation_sequence": _sequence(self._ctx),
            }
        activate = getattr(
            self._tracker, "activate_on_hand_eef_observation_prior", None
        )
        if not callable(activate):
            return {
                "ok": True,
                "active": False,
                "reason": "tracker_provider_has_no_eef_observation_prior",
                "names": visual_names,
            }
        anchors = np.asarray(anchors_eef_m, dtype=np.float64)
        indices = [requested.index(name) for name in visual_names]
        return activate(
            visual_names,
            episode_id=episode_id,
            session_id=session_id,
            image_id=image_id,
            arm=arm,
            anchors_eef_m=anchors[indices],
        )

    def status(self) -> dict[str, Any]:
        status = getattr(self._tracker, "status", None)
        return status() if callable(status) else {"active_rigid_pair": None}

    def deactivate_rigid_pair(self, *args, **kwargs):
        deactivate = getattr(self._tracker, "deactivate_rigid_pair", None)
        if callable(deactivate):
            return deactivate(*args, **kwargs)
        return {"ok": True, "active": False, "reason": "no_visual_rigid_pair"}

    def deactivate_on_hand_eef_observation_prior(self, *args, **kwargs):
        deactivate = getattr(
            self._tracker, "deactivate_on_hand_eef_observation_prior", None
        )
        if callable(deactivate):
            return deactivate(*args, **kwargs)
        return {"ok": True, "active": False, "reason": "no_visual_eef_prior"}

    def __getattr__(self, name: str):
        tracker = object.__getattribute__(self, "_tracker")
        if tracker is None:
            raise AttributeError(name)
        return getattr(tracker, name)


def _select_predefined_eef_arm(
    *,
    points: Sequence[Mapping[str, Any]],
    roles_by_name: Mapping[str, str],
    observations: Mapping[str, Mapping[str, Any]],
    visual_entries: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Select which X5 arm owns generic predefined EEF point names."""

    predefined_names = [
        str(point["name"])
        for point in points
        if is_predefined_eef_point_name(point.get("name"))
    ]
    dynamic_on_hand = [
        str(point["name"])
        for point in points
        if roles_by_name.get(str(point["name"])) == "on_hand"
        and not is_predefined_eef_point_name(point.get("name"))
        and str(point["name"]) in visual_entries
    ]
    fixed_coordinates: list[tuple[str, int, float]] = []
    for point in points:
        name = str(point["name"])
        if name not in predefined_names:
            continue
        target = point.get("target_xyz_m")
        if not isinstance(target, Mapping):
            continue
        for axis_index, axis in enumerate(("x", "y", "z")):
            value = target.get(axis)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fixed_coordinates.append((name, axis_index, float(value)))
    off_hand_points: list[np.ndarray] = []
    for point in points:
        name = str(point["name"])
        if roles_by_name.get(name) != "off_hand" or name not in visual_entries:
            continue
        try:
            xyz = np.asarray(
                visual_entries[name]["xyz_in_robot_base_coord_m"],
                dtype=np.float64,
            ).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(xyz)):
            off_hand_points.append(xyz)

    score_by_arm: dict[str, float] = {}
    basis = "deterministic_right_default"
    for side in ("left", "right"):
        observation = observations[side]
        positions = predefined_eef_points_robot_base_m(
            predefined_names,
            eef_position_robot_base_m=observation["position"],
            eef_quaternion_xyzw=observation["quaternion"],
            gripper_q_m=observation["gripper"],
        )
        if dynamic_on_hand:
            basis = "nearest_to_visual_on_hand_points"
            score = float(
                np.mean(
                    [
                        np.linalg.norm(
                            np.asarray(
                                visual_entries[name]["xyz_in_robot_base_coord_m"],
                                dtype=np.float64,
                            )
                            - np.asarray(observation["position"], dtype=np.float64)
                        )
                        for name in dynamic_on_hand
                    ]
                )
            )
        elif fixed_coordinates:
            basis = "minimum_fixed_coordinate_displacement"
            score = float(
                np.mean(
                    [
                        abs(positions[name][axis] - target)
                        for name, axis, target in fixed_coordinates
                    ]
                )
            )
        elif off_hand_points:
            basis = "nearest_to_off_hand_reference_geometry"
            centroid = np.mean(np.vstack(list(positions.values())), axis=0)
            score = float(
                np.min(np.linalg.norm(np.vstack(off_hand_points) - centroid, axis=1))
            )
        else:
            score = 0.0 if side == "right" else 1.0
        score_by_arm[side] = score
    arm = min(
        ("left", "right"),
        key=lambda side: (score_by_arm[side], side != "right"),
    )
    return arm, {
        "selected_arm": arm,
        "selection_basis": basis,
        "score_by_arm": score_by_arm,
        "predefined_eef_point_names": predefined_names,
        "visual_on_hand_point_names": dynamic_on_hand,
        "off_hand_reference_count": len(off_hand_points),
    }


def _quick_constraint_specs(
    *,
    quick_constraint: Mapping[str, Any] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if quick_constraint is not None and quick_constraints is not None:
        raise ValueError(
            "quick_constraint and quick_constraints are mutually exclusive"
        )
    if quick_constraint is not None:
        if not isinstance(quick_constraint, Mapping):
            raise ValueError("quick_constraint must be an object")
        return [dict(quick_constraint)]
    if quick_constraints is None:
        return []
    if isinstance(quick_constraints, (str, bytes)) or not isinstance(
        quick_constraints, Sequence
    ):
        raise ValueError("quick_constraints must be an array")
    if not 1 <= len(quick_constraints) <= MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS:
        raise ValueError(
            "quick_constraints must contain one to "
            f"{MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS} groups"
        )
    result: list[dict[str, Any]] = []
    for index, constraint in enumerate(quick_constraints):
        if not isinstance(constraint, Mapping):
            raise ValueError(f"quick_constraints[{index}] must be an object")
        result.append(dict(constraint))
    return result


def _quick_group_points(
    quick_constraint: Mapping[str, Any],
    *,
    controlled_names: Sequence[str],
    controlled_points: np.ndarray,
    reference_names: Sequence[str],
    reference_points: np.ndarray | None,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray | None]:
    controlled_order = [str(name) for name in controlled_names]
    reference_order = [str(name) for name in reference_names]
    controlled = _strict_point_matrix(
        controlled_points,
        expected_count=len(controlled_order),
        label="quick-constraint controlled points",
    )
    if reference_order:
        references = _strict_point_matrix(
            reference_points,
            expected_count=len(reference_order),
            label="quick-constraint off-hand points",
        )
    else:
        references = None
        if reference_points is not None and np.asarray(reference_points).size:
            raise ValueError("quick-constraint reference points have no bound names")
    controlled_index = {name: index for index, name in enumerate(controlled_order)}
    reference_index = {name: index for index, name in enumerate(reference_order)}
    raw_on_names = quick_constraint.get("on_hand_points")
    raw_off_names = quick_constraint.get("off_hand_points")
    if not isinstance(raw_on_names, (list, tuple)) or not isinstance(
        raw_off_names, (list, tuple)
    ):
        raise ValueError("quick constraint group role bindings are missing")
    on_names = [str(name) for name in raw_on_names]
    off_names = [str(name) for name in raw_off_names]
    if any(name not in controlled_index for name in on_names):
        raise ValueError("quick constraint group references a non-on-hand point")
    if any(name not in reference_index for name in off_names):
        raise ValueError("quick constraint group references a non-off-hand point")
    on_points = np.asarray(
        [controlled[controlled_index[name]] for name in on_names],
        dtype=np.float64,
    )
    off_points = (
        np.asarray(
            [references[reference_index[name]] for name in off_names],
            dtype=np.float64,
        )
        if off_names
        else None
    )
    return on_names, on_points, off_names, off_points


def _evaluate_quick_constraints(
    controlled_points: np.ndarray,
    reference_points: np.ndarray | None,
    quick_constraints: Sequence[Mapping[str, Any]],
    *,
    controlled_names: Sequence[str],
    reference_names: Sequence[str],
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
    camera_position_robot_base_m: Sequence[float] | None = None,
    camera_quaternion_xyzw: Sequence[float] | None = None,
) -> dict[str, Any]:
    specs = _quick_constraint_specs(quick_constraints=quick_constraints)
    reports: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        on_names, on_points, off_names, off_points = _quick_group_points(
            spec,
            controlled_names=controlled_names,
            controlled_points=controlled_points,
            reference_names=reference_names,
            reference_points=reference_points,
        )
        report = evaluate_quick_constraint(
            on_points,
            off_points,
            spec,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_deg=orientation_tolerance_deg,
            camera_position_robot_base_m=camera_position_robot_base_m,
            camera_quaternion_xyzw=camera_quaternion_xyzw,
        )
        reports.append(
            {
                **json_ready(report),
                "group_index": index,
                "quick_constraint": json_ready(spec),
                "on_hand_point_names": on_names,
                "off_hand_point_names": off_names,
            }
        )
    if len(reports) == 1:
        return {**reports[0], "group_count": 1, "groups": reports}
    errors_m = [float(report.get("max_error_m", 0.0)) for report in reports]
    errors_deg = [float(report.get("max_error_deg", 0.0)) for report in reports]
    first_failure = next((report for report in reports if not report.get("ok")), None)
    return {
        "ok": bool(reports and all(report.get("ok") for report in reports)),
        "applicable": bool(reports),
        "type": "quick_constraint_group_set",
        "group_count": len(reports),
        "groups": reports,
        "max_error_m": max(errors_m, default=0.0),
        "max_error_deg": max(errors_deg, default=0.0),
        "error": (
            None
            if first_failure is None
            else f"quick_constraint_group_{first_failure['group_index'] + 1}: "
            f"{first_failure.get('error') or 'constraint_not_satisfied'}"
        ),
        "position_tolerance_m": float(position_tolerance_m),
        "orientation_tolerance_deg": float(orientation_tolerance_deg),
    }


def _relations_with_quick_constraints(
    relations: Sequence[Mapping[str, Any]],
    quick_constraints: Sequence[Mapping[str, Any]],
    *,
    controlled_names: Sequence[str],
    controlled_points: np.ndarray,
    reference_names: Sequence[str],
    reference_points: np.ndarray | None,
    camera_position_robot_base_m: Sequence[float] | None = None,
    camera_quaternion_xyzw: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Lower every quick group against its exact named frozen subset."""

    lowered = [dict(relation) for relation in relations]
    for spec in quick_constraints:
        on_names, _on_points, _off_names, off_points = _quick_group_points(
            spec,
            controlled_names=controlled_names,
            controlled_points=controlled_points,
            reference_names=reference_names,
            reference_points=reference_points,
        )
        lowered.append(
            quick_constraint_relation(
                spec,
                on_hand_names=on_names,
                on_hand_points_robot_base_m=_on_points,
                off_hand_points_robot_base_m=off_points,
                camera_position_robot_base_m=camera_position_robot_base_m,
                camera_quaternion_xyzw=camera_quaternion_xyzw,
            )
        )
    return lowered


def _occluded_on_hand_evidence(live: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Require a nearer RGB-D layer at the missing marker's projected pixel."""

    point_name = str(name)
    fusion = live.get("rigid_pair_fusion")
    if not isinstance(fusion, Mapping):
        return {
            "ok": False,
            "name": point_name,
            "reason": "rigid_pair_occlusion_report_unavailable",
        }
    visual = fusion.get("visual_tracker_pair")
    if not isinstance(visual, Mapping):
        return {
            "ok": False,
            "name": point_name,
            "reason": "visual_pair_occlusion_report_unavailable",
        }
    names = [str(value) for value in fusion.get("names", [])]
    track_ids = [str(value) for value in visual.get("track_ids", [])]
    if point_name not in names or len(names) != len(track_ids):
        return {
            "ok": False,
            "name": point_name,
            "reason": "occluded_point_not_bound_to_rigid_pair",
        }
    track_id = track_ids[names.index(point_name)]
    per_point = visual.get("per_point")
    detail = per_point.get(track_id) if isinstance(per_point, Mapping) else None
    if not isinstance(detail, Mapping):
        return {
            "ok": False,
            "name": point_name,
            "track_id": track_id,
            "reason": "occluded_point_diagnostics_unavailable",
        }

    candidates: list[dict[str, Any]] = []
    for raw_candidate in detail.get("rejected_candidates_before_acceptance", []):
        if not isinstance(raw_candidate, Mapping):
            continue
        if str(raw_candidate.get("reason") or "") != "eef_prior_depth_layer_mismatch":
            continue
        try:
            candidate_depth = float(raw_candidate["candidate_depth_m"])
            predicted_depth = float(raw_candidate["predicted_depth_m"])
            pixel_error = float(raw_candidate["pixel_error_px"])
            pixel_limit = float(raw_candidate["pixel_error_limit_px"])
        except (KeyError, TypeError, ValueError):
            continue
        values = (candidate_depth, predicted_depth, pixel_error, pixel_limit)
        if not all(math.isfinite(value) for value in values):
            continue
        depth_margin = predicted_depth - candidate_depth
        candidate = {
            "candidate_method": str(raw_candidate.get("candidate_method") or ""),
            "candidate_status": str(raw_candidate.get("candidate_status") or ""),
            "candidate_depth_m": candidate_depth,
            "predicted_depth_m": predicted_depth,
            "foreground_depth_margin_m": depth_margin,
            "pixel_error_px": pixel_error,
            "pixel_error_limit_px": pixel_limit,
            "ok": bool(
                depth_margin >= FINAL_OCCLUSION_MIN_DEPTH_MARGIN_M
                and pixel_error <= pixel_limit
            ),
        }
        candidates.append(candidate)
    accepted = next((item for item in candidates if item["ok"]), None)
    return {
        "ok": accepted is not None,
        "name": point_name,
        "track_id": track_id,
        "status": str(detail.get("status") or ""),
        "reason": (
            None
            if accepted is not None
            else "nearer_rgbd_occluder_not_observed_at_predicted_pixel"
        ),
        "accepted_candidate": accepted,
        "candidate_count": len(candidates),
        "prediction_values_published_as_observation": False,
    }


def _eef_anchor_depth_reobservation_x5(
    adapter,
    predicted_points_robot_base_m: Any,
    *,
    current_sequence: int,
    base_rotation: Any | None = None,
) -> dict[str, Any]:
    """Measure X5 EEF anchors from one synchronized evaluator depth image.

    Local FK supplies only a projected pixel and expected depth layer. Every
    published XYZ is independently back-projected from RoboDojo depth_linear.
    """

    report: dict[str, Any] = {
        "ok": False,
        "reason": None,
        "observation_sequence": int(current_sequence),
        "camera_role": "head",
        "source": "current_evaluator_depth_linear_at_local_fk_projected_anchor",
        "prediction_values_published_as_observation": False,
        "per_point": [],
    }
    if adapter is None:
        report["reason"] = "official_evaluator_adapter_unavailable"
        return report
    try:
        expected_count = len(predicted_points_robot_base_m)
        expected = _strict_point_matrix(
            predicted_points_robot_base_m,
            expected_count=expected_count,
            label="EEF anchor depth reobservation predictions",
        )
        sequence_before = int(adapter.status().get("sequence", -1))
        camera = adapter.camera("head", require_depth=True)
        camera_sequence = int(camera.get("sequence", -1))
        sequence_after = int(adapter.status().get("sequence", -1))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        report.update(
            {
                "reason": "evaluator_rgbd_snapshot_unavailable",
                "error": str(exc),
            }
        )
        return report
    report.update(
        {
            "sequence_before": sequence_before,
            "camera_sequence": camera_sequence,
            "sequence_after": sequence_after,
        }
    )
    if not (
        sequence_before == camera_sequence == sequence_after == int(current_sequence)
    ):
        report["reason"] = "evaluator_rgbd_sequence_mismatch"
        return report

    rgb_array = np.asarray(camera.get("rgb"))
    depth_array = np.asarray(camera.get("depth"), dtype=np.float64).squeeze()
    if (
        rgb_array.ndim < 2
        or depth_array.ndim != 2
        or tuple(rgb_array.shape[:2]) != tuple(depth_array.shape)
    ):
        report["reason"] = "evaluator_head_rgbd_shape_mismatch"
        return report
    try:
        intrinsics = np.asarray(camera["intrinsic_matrix"], dtype=np.float64).reshape(
            3, 3
        )
        camera_to_env = np.asarray(
            camera["extrinsic_matrix"], dtype=np.float64
        ).reshape(4, 4)
        rotation = (
            np.eye(3, dtype=np.float64)
            if base_rotation is None
            else np.asarray(base_rotation, dtype=np.float64).reshape(3, 3)
        )
        if not (
            np.all(np.isfinite(intrinsics))
            and np.all(np.isfinite(camera_to_env))
            and np.all(np.isfinite(rotation))
        ):
            raise ValueError("camera matrices and base rotation must be finite")
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
        if fx <= 1.0e-10 or fy <= 1.0e-10:
            raise ValueError("camera focal lengths must be positive")
        base_from_env = np.eye(4, dtype=np.float64)
        base_from_env[:3, :3] = rotation.T
        camera_to_robot = base_from_env @ camera_to_env
        robot_to_camera = np.linalg.inv(camera_to_robot)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        report.update(
            {
                "reason": "evaluator_head_camera_matrices_invalid",
                "error": str(exc),
            }
        )
        return report

    height, width = depth_array.shape
    measured_points: list[np.ndarray] = []
    radius = int(FINAL_EEF_DEPTH_REOBSERVE_RADIUS_PX)
    for point_index, predicted in enumerate(expected):
        point_camera = (robot_to_camera @ np.append(predicted, 1.0))[:3]
        predicted_depth = -float(point_camera[2])
        point_report: dict[str, Any] = {
            "index": int(point_index),
            "predicted_point_robot_base_m": predicted.astype(float).tolist(),
            "predicted_depth_m": predicted_depth,
            "ok": False,
        }
        if not math.isfinite(predicted_depth) or predicted_depth <= 1.0e-6:
            point_report["reason"] = "predicted_anchor_behind_camera"
            report["per_point"].append(point_report)
            continue
        predicted_pixel = np.asarray(
            [
                cx + fx * float(point_camera[0]) / predicted_depth,
                cy - fy * float(point_camera[1]) / predicted_depth,
            ],
            dtype=np.float64,
        )
        point_report["predicted_pixel"] = predicted_pixel.astype(float).tolist()
        center_x = int(round(float(predicted_pixel[0])))
        center_y = int(round(float(predicted_pixel[1])))
        if not (0 <= center_x < width and 0 <= center_y < height):
            point_report["reason"] = "predicted_anchor_out_of_view"
            report["per_point"].append(point_report)
            continue

        candidates: list[tuple[float, float, np.ndarray, int, int, float]] = []
        for pixel_y in range(
            max(0, center_y - radius), min(height, center_y + radius + 1)
        ):
            for pixel_x in range(
                max(0, center_x - radius), min(width, center_x + radius + 1)
            ):
                depth_m = float(depth_array[pixel_y, pixel_x])
                if not math.isfinite(depth_m) or depth_m <= 1.0e-6:
                    continue
                camera_point = np.asarray(
                    [
                        (float(pixel_x) - cx) / fx * depth_m,
                        -(float(pixel_y) - cy) / fy * depth_m,
                        -depth_m,
                    ],
                    dtype=np.float64,
                )
                measured = (camera_to_robot @ np.append(camera_point, 1.0))[:3]
                if not np.all(np.isfinite(measured)):
                    continue
                depth_error = abs(depth_m - predicted_depth)
                point_error = float(np.linalg.norm(measured - predicted))
                pixel_error = float(
                    np.linalg.norm(
                        np.asarray([pixel_x, pixel_y], dtype=np.float64)
                        - predicted_pixel
                    )
                )
                candidates.append(
                    (
                        point_error,
                        depth_error,
                        measured,
                        pixel_x,
                        pixel_y,
                        pixel_error,
                    )
                )
        if not candidates:
            point_report["reason"] = "no_valid_depth_near_predicted_anchor"
            report["per_point"].append(point_report)
            continue
        candidate = min(candidates, key=lambda item: (item[0], item[1], item[5]))
        point_error, depth_error, measured, pixel_x, pixel_y, pixel_error = candidate
        point_report.update(
            {
                "measured_pixel": [int(pixel_x), int(pixel_y)],
                "measured_depth_m": float(depth_array[pixel_y, pixel_x]),
                "measured_point_robot_base_m": measured.astype(float).tolist(),
                "pixel_error_px": pixel_error,
                "depth_error_m": depth_error,
                "point_error_m": point_error,
                "max_depth_error_m": float(FINAL_EEF_DEPTH_REOBSERVE_MAX_DEPTH_ERROR_M),
                "max_point_error_m": float(FINAL_EEF_DEPTH_REOBSERVE_MAX_POINT_ERROR_M),
                "ok": bool(
                    depth_error <= FINAL_EEF_DEPTH_REOBSERVE_MAX_DEPTH_ERROR_M
                    and point_error <= FINAL_EEF_DEPTH_REOBSERVE_MAX_POINT_ERROR_M
                ),
            }
        )
        if not point_report["ok"]:
            point_report["reason"] = "predicted_anchor_depth_surface_mismatch"
        else:
            point_report["reason"] = None
            measured_points.append(measured)
        report["per_point"].append(point_report)

    if len(measured_points) != len(expected):
        report["reason"] = "complete_eef_anchor_depth_bundle_not_observed"
        return report
    report.update(
        {
            "ok": True,
            "reason": None,
            "points_robot_base_m": np.vstack(measured_points),
            "point_count": int(len(measured_points)),
            "patch_radius_px": radius,
            "camera_matrix_source": "current_robodojo_evaluator_observation",
            "camera_coordinate_convention": "usd_x_right_y_up_z_backward",
            "robot_base_axes": "x_forward_y_left_z_up",
        }
    )
    return report


def _final_observation_bundle_x5(
    live: Mapping[str, Any],
    *,
    controlled_names: Sequence[str],
    fixed_names: Sequence[str],
    current_position: np.ndarray,
    current_quaternion: np.ndarray,
    anchors_eef: np.ndarray,
    current_sequence: int,
    anchor_tolerance_m: float,
    pos_tol_m: float,
    adapter=None,
    base_rotation: Any | None = None,
) -> dict[str, Any]:
    """Build the final synchronized RGB-D bundle without publishing FK truth."""

    controlled = [str(name) for name in controlled_names]
    fixed = [str(name) for name in fixed_names]
    report: dict[str, Any] = {
        "ok": False,
        "reason": None,
        "mode": "unavailable",
        "controlled_names": controlled,
        "fixed_names": fixed,
        "measured_controlled_names": [],
        "fk_completed_controlled_names": [],
        "depth_reobserved_controlled_names": [],
        "eef_anchor_depth_reobservation": {
            "ok": False,
            "applicable": False,
            "reason": "not_requested",
            "prediction_values_published_as_observation": False,
        },
        "point_sources": {},
        "prediction_values_published_as_observation": False,
    }
    try:
        tracker_sequence = int(live.get("observation_sequence", -1))
    except (TypeError, ValueError):
        tracker_sequence = -1
    report["tracker_observation_sequence"] = tracker_sequence
    if tracker_sequence != int(current_sequence):
        report["reason"] = "tracker_observation_mismatch"
        return report

    entries = live.get("entries")
    if not isinstance(entries, Mapping):
        report["reason"] = "tracked_point_invalid"
        return report
    try:
        position = np.asarray(current_position, dtype=np.float64).reshape(3)
        quaternion = np.asarray(current_quaternion, dtype=np.float64).reshape(4)
        anchors = _strict_point_matrix(
            anchors_eef,
            expected_count=len(controlled),
            label="final EEF anchor bundle",
        )
        predicted = position[None, :] + (quat_to_mat_xyzw(quaternion) @ anchors.T).T
    except (TypeError, ValueError) as exc:
        report["reason"] = f"final_fk_bundle_invalid: {exc}"
        return report
    if not np.all(np.isfinite(predicted)):
        report["reason"] = "final_fk_bundle_invalid"
        return report

    measured: dict[str, np.ndarray] = {}
    invalid_names: list[str] = []
    for name in controlled + fixed:
        entry = entries.get(name)
        if entry is None:
            continue
        try:
            xyz = np.asarray(
                entry["xyz_in_robot_base_coord_m"], dtype=np.float64
            ).reshape(3)
        except (KeyError, TypeError, ValueError):
            invalid_names.append(name)
            continue
        if not np.all(np.isfinite(xyz)):
            invalid_names.append(name)
            continue
        measured[name] = xyz
    if invalid_names:
        report.update(
            {"reason": "tracked_point_invalid", "invalid_names": invalid_names}
        )
        return report

    missing_fixed = [name for name in fixed if name not in measured]
    if missing_fixed:
        report.update(
            {
                "reason": str(live.get("reason") or "tracked_point_unavailable"),
                "missing_fixed_names": missing_fixed,
            }
        )
        return report
    missing_controlled = [name for name in controlled if name not in measured]
    measured_controlled = [name for name in controlled if name in measured]
    report["measured_controlled_names"] = measured_controlled
    depth_reobserved_names: list[str] = []
    preemptive_depth_reobservation: dict[str, Any] | None = None
    visible_anchor_limit = min(float(anchor_tolerance_m), max(0.012, float(pos_tol_m)))

    # The tracker can reject a complete rigid bundle even when all projected
    # surfaces remain visible. FK only nominates pixels; accepted XYZ values
    # must still come atomically from one current evaluator depth frame.
    if (
        missing_controlled
        and len(missing_controlled) == len(controlled)
        and not measured_controlled
        and str(live.get("reason") or "") == "tracked_point_unavailable"
        and adapter is not None
    ):
        preemptive_depth_reobservation = _eef_anchor_depth_reobservation_x5(
            adapter,
            predicted,
            current_sequence=int(current_sequence),
            base_rotation=base_rotation,
        )
        preemptive_depth_reobservation["applicable"] = True
        preemptive_depth_reobservation["trigger"] = (
            "complete_controlled_tracker_bundle_unavailable"
        )
        report["eef_anchor_depth_reobservation"] = preemptive_depth_reobservation
        if bool(preemptive_depth_reobservation.get("ok")):
            reobserved = _strict_point_matrix(
                preemptive_depth_reobservation["points_robot_base_m"],
                expected_count=len(controlled),
                label="EEF anchor depth-reobserved missing controlled points",
            )
            for index, name in enumerate(controlled):
                measured[name] = reobserved[index]
            missing_controlled = []
            measured_controlled = list(controlled)
            depth_reobserved_names = list(controlled)
            report["measured_controlled_names"] = measured_controlled
            report["depth_reobserved_controlled_names"] = depth_reobserved_names

    if not missing_controlled:
        if not bool(live.get("ok")) and not depth_reobserved_names:
            report["reason"] = str(live.get("reason") or "tracked_point_unavailable")
            return report
        tracked_points = np.vstack([measured[name] for name in controlled])
        tracked_anchor_errors = np.linalg.norm(tracked_points - predicted, axis=1)
        anchor_error_key = (
            "depth_reobserved_anchor_error_m"
            if depth_reobserved_names
            else "raw_tracker_anchor_error_m"
        )
        report[anchor_error_key] = {
            name: float(tracked_anchor_errors[index])
            for index, name in enumerate(controlled)
        }
        depth_reobservation = (
            preemptive_depth_reobservation
            if preemptive_depth_reobservation is not None
            else {
                "ok": False,
                "applicable": False,
                "reason": "tracker_bundle_within_strict_anchor_envelope",
                "prediction_values_published_as_observation": False,
            }
        )
        if (
            not depth_reobserved_names
            and tracked_anchor_errors.size
            and float(np.max(tracked_anchor_errors)) > visible_anchor_limit
        ):
            depth_reobservation = _eef_anchor_depth_reobservation_x5(
                adapter,
                predicted,
                current_sequence=int(current_sequence),
                base_rotation=base_rotation,
            )
            depth_reobservation["applicable"] = True
            if bool(depth_reobservation.get("ok")):
                reobserved = _strict_point_matrix(
                    depth_reobservation["points_robot_base_m"],
                    expected_count=len(controlled),
                    label="EEF anchor depth-reobserved controlled points",
                )
                for index, name in enumerate(controlled):
                    measured[name] = reobserved[index]
                depth_reobserved_names = list(controlled)
        report["eef_anchor_depth_reobservation"] = depth_reobservation
        report["depth_reobserved_controlled_names"] = depth_reobserved_names
        mode = (
            "complete_evaluator_depth_reobservation_at_eef_anchors"
            if depth_reobserved_names
            else "all_controlled_points_measured_rgbd"
        )
        occlusion_reports: dict[str, Any] = {}
    else:
        unavailable = live.get("unavailable")
        unavailable_names = (
            {str(name) for name in unavailable}
            if isinstance(unavailable, Mapping)
            else set()
        )
        if (
            len(controlled) != 2
            or len(missing_controlled) != 1
            or len(measured_controlled) != 1
            or unavailable_names != set(missing_controlled)
            or str(live.get("reason") or "") != "tracked_point_unavailable"
        ):
            report.update(
                {
                    "reason": str(live.get("reason") or "tracked_point_unavailable"),
                    "missing_controlled_names": missing_controlled,
                }
            )
            return report
        occlusion_reports = {
            name: _occluded_on_hand_evidence(live, name) for name in missing_controlled
        }
        if not all(item.get("ok") for item in occlusion_reports.values()):
            report.update(
                {
                    "reason": "on_hand_point_unavailable_without_occlusion_evidence",
                    "missing_controlled_names": missing_controlled,
                    "occlusion_evidence": occlusion_reports,
                }
            )
            return report
        mode = "partial_rgbd_with_occluded_on_hand_fk_completion"

    points = predicted.copy()
    for index, name in enumerate(controlled):
        if name in measured:
            points[index] = measured[name]
            report["point_sources"][name] = (
                "current_evaluator_depth_linear_at_local_fk_projected_anchor"
                if name in depth_reobserved_names
                else "current_evaluator_rgbd_and_camera_relative_pose"
            )
        else:
            report["point_sources"][name] = (
                "frozen_rgbd_eef_anchor_and_current_evaluator_proprio_local_fk"
            )
    fixed_points = (
        np.vstack([measured[name] for name in fixed])
        if fixed
        else np.empty((0, 3), dtype=np.float64)
    )
    for name in fixed:
        report["point_sources"][name] = (
            "current_evaluator_rgbd_and_camera_relative_pose"
        )
    visible_indices = [controlled.index(name) for name in measured_controlled]
    visible_anchor_errors = np.linalg.norm(
        points[visible_indices] - predicted[visible_indices], axis=1
    )
    if missing_controlled and (
        not visible_anchor_errors.size
        or float(np.max(visible_anchor_errors)) > visible_anchor_limit
    ):
        report.update(
            {
                "reason": "visible_on_hand_point_not_rigid_with_selected_eef",
                "visible_anchor_tolerance_m": visible_anchor_limit,
                "visible_anchor_error_m": {
                    name: float(visible_anchor_errors[index])
                    for index, name in enumerate(measured_controlled)
                },
                "occlusion_evidence": occlusion_reports,
            }
        )
        return report

    report.update(
        {
            "ok": True,
            "reason": None,
            "mode": mode,
            "points": points,
            "fixed_points": fixed_points,
            "predicted_points": predicted,
            "fk_completed_controlled_names": missing_controlled,
            "occlusion_evidence": occlusion_reports,
            "visible_anchor_tolerance_m": visible_anchor_limit,
            "visible_anchor_error_m": {
                name: float(visible_anchor_errors[index])
                for index, name in enumerate(measured_controlled)
            },
        }
    )
    return report


def _activate_rigid_points(
    ctx,
    names: list[str],
    episode_id: str,
    *,
    manager=None,
    allow_observation_failure: bool = False,
) -> dict[str, Any]:
    if len(names) < 2:
        return {
            "ok": True,
            "active": False,
            "reason": "single_point_request",
            "requested_names": list(names),
            "all_points_same_rigid_object_policy_assertion": False,
        }
    source = manager if manager is not None else ctx.tracker
    activate_all = getattr(source, "activate_rigid_points", None)
    if callable(activate_all):
        report = activate_all(list(names), episode_id=episode_id)
        if not isinstance(report, dict) or not bool(report.get("ok")):
            reason = (
                report.get("reason") if isinstance(report, dict) else "invalid_result"
            )
            if allow_observation_failure:
                return {
                    **(report if isinstance(report, dict) else {}),
                    "ok": False,
                    "reason": reason,
                    "requested_names": list(names),
                    "activated_names": list(names),
                    "all_points_same_rigid_object_policy_assertion": True,
                }
            raise ValueError(f"tracked rigid point bundle is inconsistent: {reason}")
        return {
            **report,
            "requested_names": list(names),
            "activated_names": list(names),
            "all_points_same_rigid_object_policy_assertion": True,
        }
    if len(names) > 2:
        return {
            "ok": True,
            "active": False,
            "requested_names": list(names),
            "activated_names": [],
            "activation_scope": "n_point_local_geometry_only",
            "all_points_same_rigid_object_policy_assertion": True,
            "reason": (
                "tracker_has_no_n_point_fusion; all requested points are checked "
                "atomically"
            ),
        }
    report = source.activate_rigid_pair(list(names), episode_id=episode_id)
    if not isinstance(report, dict) or not bool(report.get("ok")):
        reason = report.get("reason") if isinstance(report, dict) else "invalid_result"
        if allow_observation_failure:
            return {
                **(report if isinstance(report, dict) else {}),
                "ok": False,
                "reason": reason,
                "requested_names": list(names),
                "activated_names": list(names[:2]),
                "activation_scope": "two_point_rigid_pair",
                "all_points_same_rigid_object_policy_assertion": True,
            }
        raise ValueError(f"tracked rigid pair is inconsistent: {reason}")
    return {
        **report,
        "requested_names": list(names),
        "activated_names": list(names),
        "activation_scope": "two_point_rigid_pair",
        "all_points_same_rigid_object_policy_assertion": True,
    }


def _off_hand_reference_consensus(
    names: Sequence[str],
    measured_points: Any,
    reference_points: Any,
    *,
    tolerance_m: float,
    allow_visual_outlier_suppression: bool,
) -> dict[str, Any]:
    ordered_names = [str(name) for name in names]
    if not ordered_names:
        return {
            "ok": True,
            "applicable": False,
            "all_points_verified": True,
            "tracking_outlier_suppressed": False,
            "constraint_reference_source": "not_applicable",
            "max_drift_m": 0.0,
        }
    measured = _strict_point_matrix(
        measured_points,
        expected_count=len(ordered_names),
        label="off-hand measured points",
    )
    reference = _strict_point_matrix(
        reference_points,
        expected_count=len(ordered_names),
        label="frozen off-hand reference points",
    )
    tolerance = max(1.0e-6, float(tolerance_m))
    if not math.isfinite(tolerance):
        raise ValueError("off-hand reference tolerance must be finite")
    errors = np.linalg.norm(measured - reference, axis=1)
    stable_mask = errors <= tolerance
    stable_names = [
        name for index, name in enumerate(ordered_names) if stable_mask[index]
    ]
    outlier_names = [
        name for index, name in enumerate(ordered_names) if not stable_mask[index]
    ]
    pair_errors: dict[str, float] = {}
    for first in range(len(ordered_names)):
        for second in range(first + 1, len(ordered_names)):
            source_distance = float(
                np.linalg.norm(reference[second] - reference[first])
            )
            measured_distance = float(
                np.linalg.norm(measured[second] - measured[first])
            )
            pair_errors[f"{ordered_names[first]}::{ordered_names[second]}"] = abs(
                measured_distance - source_distance
            )
    max_pair_error = max(pair_errors.values(), default=0.0)
    rigid_geometry_consistent = bool(max_pair_error <= tolerance)
    required_stable_count = (
        1 if len(ordered_names) == 2 else len(ordered_names) // 2 + 1
    )
    suppress_outliers = bool(
        allow_visual_outlier_suppression
        and len(ordered_names) >= 2
        and outlier_names
        and len(stable_names) >= required_stable_count
        and not rigid_geometry_consistent
    )
    all_verified = not outlier_names
    ok = bool(all_verified or suppress_outliers)
    max_drift = float(np.max(errors)) if errors.size else 0.0
    return {
        "ok": ok,
        "applicable": True,
        "reason": (
            None
            if all_verified
            else (
                "off_hand_visual_tracking_outlier_suppressed"
                if suppress_outliers
                else "off_hand_points_moved"
            )
        ),
        "error": (
            None
            if ok
            else (
                f"off-hand reference drift {max_drift:.6f}m exceeds "
                f"{tolerance:.6f}m without a suppressible visual-outlier consensus"
            )
        ),
        "tolerance_m": tolerance,
        "max_drift_m": max_drift,
        "point_drift_m": {
            name: float(errors[index]) for index, name in enumerate(ordered_names)
        },
        "stable_names": stable_names,
        "outlier_names": outlier_names,
        "stable_count": len(stable_names),
        "required_stable_count": required_stable_count,
        "all_points_verified": all_verified,
        "tracking_outlier_suppressed": suppress_outliers,
        "constraint_reference_source": (
            "frozen_off_hand_reference_due_visual_outlier"
            if suppress_outliers
            else "frozen_off_hand_reference_verified_by_live_rgbd"
        ),
        "rigid_geometry_consistent": rigid_geometry_consistent,
        "max_pair_distance_error_m": max_pair_error,
        "pair_distance_error_m": pair_errors,
        "coherent_rigid_reference_motion_accepted": False,
    }


def _strict_position_precision_gate(
    constraints: Mapping[str, Any],
    *,
    limit_m: float = FINAL_EXECUTION_POSITION_LIMIT_M,
) -> dict[str, Any]:
    """Certify the terminal coordinate error independently of public tolerance.

    ``pos_tol`` is intentionally kept as a backwards-compatible request field,
    but it used to be possible for a 30 mm tolerance to turn a visibly wrong
    endpoint into a successful tool result.  This gate is the final safety
    contract: a finite coordinate residual must be *strictly* below the
    user-facing precision limit.
    """

    try:
        limit = float(limit_m)
    except (TypeError, ValueError):
        limit = float("nan")
    try:
        error = float(constraints.get("max_constraint_error_m", float("inf")))
    except (TypeError, ValueError):
        error = float("inf")
    ok = bool(
        math.isfinite(limit) and limit > 0.0 and math.isfinite(error) and error < limit
    )
    return {
        "ok": ok,
        "max_constraint_error_m": error,
        "limit_m": limit,
        "comparison": "strict_less_than",
        "margin_m": (
            float(limit - error)
            if math.isfinite(limit) and math.isfinite(error)
            else float("-inf")
        ),
        "source": "live_rgbd_coordinate_constraint_report",
    }


def _pose_env_to_base(ctx, pose: Any) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(value)):
        raise ValueError("EEF pose contains non-finite values")
    env_from_base = np.asarray(ctx.world.base_rotation, dtype=np.float64).reshape(3, 3)
    base_position = env_from_base.T @ value[:3]
    base_rotation = env_from_base.T @ quaternion_to_matrix(value[3:])
    return np.concatenate([base_position, matrix_to_quaternion(base_rotation)])


def _eef_base_xyzw(ctx, arm: str) -> tuple[np.ndarray, np.ndarray]:
    pose = _pose_env_to_base(ctx, ctx.world.eef_pose_list(arm))
    return pose[:3], np.asarray([pose[4], pose[5], pose[6], pose[3]])


def _freeze_state(ctx) -> dict[str, Any]:
    q = {
        side: np.asarray(ctx.world.arm_qpos_list(side), dtype=np.float64).reshape(
            ARM_DOF
        )
        for side in ("left", "right")
    }
    qvel = {
        side: np.asarray(ctx.world.arm_qvel_list(side), dtype=np.float64).reshape(
            ARM_DOF
        )
        for side in ("left", "right")
    }
    gripper = {
        side: np.asarray(ctx.world.gripper_qpos_list(side), dtype=np.float64).reshape(1)
        for side in ("left", "right")
    }
    latch_reader = getattr(ctx.world, "gripper_command_latch", None)
    gripper_command_latch: dict[str, float | None] = {}
    for side in ("left", "right"):
        raw_latch = latch_reader(side) if callable(latch_reader) else None
        gripper_command_latch[side] = None if raw_latch is None else float(raw_latch)
    eef_base_wxyz = {
        side: _pose_env_to_base(ctx, ctx.world.eef_pose_list(side))
        for side in ("left", "right")
    }
    if not all(
        np.all(np.isfinite(value))
        for mapping in (q, qvel, gripper, eef_base_wxyz)
        for value in mapping.values()
    ):
        raise ValueError("current evaluator proprioception contains non-finite values")
    local_state = LocalRobotState(
        arm_dof=ARM_DOF,
        arm_left_q=q["left"].copy(),
        arm_right_q=q["right"].copy(),
        eef_left_pose_base_wxyz=eef_base_wxyz["left"].copy(),
        eef_right_pose_base_wxyz=eef_base_wxyz["right"].copy(),
    )
    return {
        "state": local_state,
        "arm_q": q,
        "arm_qvel": qvel,
        "gripper_q": gripper,
        "gripper_command_latch": gripper_command_latch,
        "eef_base_wxyz": eef_base_wxyz,
        "base_rotation": np.asarray(ctx.world.base_rotation, dtype=np.float64).reshape(
            3, 3
        ),
        "motion_epoch": int(ctx.world.motion_epoch()),
        "episode_id": str(ctx.world.episode_id()),
    }


def _base_pose_wxyz_to_env_wxyz(
    pose_robot_base_wxyz: Any,
    base_rotation: Any,
) -> np.ndarray:
    pose = np.asarray(pose_robot_base_wxyz, dtype=np.float64).reshape(7)
    env_from_base = np.asarray(base_rotation, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(env_from_base)):
        raise ValueError("X5 base-to-environment pose conversion is non-finite")
    rotation_env = env_from_base @ quaternion_to_matrix(pose[3:])
    return np.concatenate(
        [env_from_base @ pose[:3], matrix_to_quaternion(rotation_env)]
    )


def _base_pose_xyzw_to_env_wxyz(
    position_robot_base_m: Any,
    quaternion_xyzw: Any,
    base_rotation: Any,
) -> np.ndarray:
    position = np.asarray(position_robot_base_m, dtype=np.float64).reshape(3)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    env_from_base = np.asarray(base_rotation, dtype=np.float64).reshape(3, 3)
    if not all(
        np.all(np.isfinite(value)) for value in (position, quaternion, env_from_base)
    ):
        raise ValueError("X5 planned EEF pose conversion is non-finite")
    rotation_env = env_from_base @ quat_to_mat_xyzw(quaternion)
    return np.concatenate(
        [env_from_base @ position, matrix_to_quaternion(rotation_env)]
    )


def _freeze_plan_capture_x5(
    ctx,
    *,
    session_id: str,
    registration_image_id: str,
    observation_sequence: int,
    frozen: Mapping[str, Any],
) -> tuple[Snapshot, dict[str, Any]]:
    """Persist one evaluator RGB-D/proprio frame without rebinding markers."""

    adapter = ctx.world._official_adapter
    before = adapter.status()
    if int(before.get("sequence", -1)) != int(observation_sequence) or str(
        before.get("episode_id") or ""
    ) != str(frozen["episode_id"]):
        raise ValueError("plan RGB-D observation changed before freezing")
    camera = adapter.camera("head", require_depth=True)
    if int(camera.get("sequence", -1)) != int(observation_sequence) or str(
        camera.get("episode_id") or ""
    ) != str(frozen["episode_id"]):
        raise ValueError("plan RGB-D is not synchronized to frozen proprioception")
    rgb = np.asarray(camera["rgb"])
    depth = np.asarray(camera["depth"], dtype=np.float32)
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(camera["extrinsic_matrix"], dtype=np.float64).reshape(4, 4)
    if (
        rgb.ndim != 3
        or rgb.shape[2] < 3
        or depth.shape != rgb.shape[:2]
        or not np.any(np.isfinite(depth) & (depth > 0.0))
        or not np.all(np.isfinite(intrinsic))
        or not np.all(np.isfinite(extrinsic))
    ):
        raise ValueError("plan requires valid synchronized head RGB-D calibration")

    robot: dict[str, Any] = {
        "base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "base_qvel": [0.0, 0.0, 0.0],
        "base_available": False,
        "trunk_qpos": [0.0, 0.0, 0.0, 0.0],
        "trunk_qvel": [0.0, 0.0, 0.0, 0.0],
        "trunk_available": False,
        "motion_epoch": int(frozen["motion_epoch"]),
        "episode_id": str(frozen["episode_id"]),
        "frame": "robodojo_env_fixed_base",
        "robot_profile": ROBOT_PROFILE,
    }
    for side in ("left", "right"):
        eef_env = _base_pose_wxyz_to_env_wxyz(
            frozen["eef_base_wxyz"][side], frozen["base_rotation"]
        )
        robot.update(
            {
                f"arm_{side}_qpos": np.asarray(frozen["arm_q"][side], dtype=np.float64)
                .astype(float)
                .tolist(),
                f"arm_{side}_qvel": np.asarray(
                    frozen["arm_qvel"][side], dtype=np.float64
                )
                .astype(float)
                .tolist(),
                f"eef_{side}": {
                    "pos": eef_env[:3].astype(float).tolist(),
                    "quat": eef_env[3:].astype(float).tolist(),
                    "frame": "robodojo_env",
                },
                f"gripper_{side}_qpos": np.asarray(
                    frozen["gripper_q"][side], dtype=np.float64
                )
                .astype(float)
                .tolist(),
                f"gripper_{side}_qvel": [0.0],
            }
        )
    snapshot = ctx.snapshots.save(
        session_id=session_id,
        role="head",
        camera={
            **camera,
            "rgb": rgb[:, :, :3].copy(),
            "depth": depth.copy(),
            "intrinsic_matrix": intrinsic.copy(),
            "extrinsic_matrix": extrinsic.copy(),
        },
        robot=robot,
    )
    after = adapter.status()
    if int(after.get("sequence", -1)) != int(observation_sequence) or str(
        after.get("episode_id") or ""
    ) != str(frozen["episode_id"]):
        raise ValueError("plan RGB-D and proprio changed while freezing")
    binding = {
        "ok": True,
        "trackable": False,
        "reason": "plan_preview_capture_does_not_rebind_active_tracker",
    }
    metadata = {
        "image_id": snapshot.image_id,
        "registration_image_id": str(registration_image_id),
        "observation_sequence": int(observation_sequence),
        "episode_id": snapshot.episode_id,
        "integrity_digest": snapshot.integrity_digest,
        "paths": dict(snapshot.paths),
        "all_inputs_same_observation": True,
        "track_object_distance_binding": binding,
        "source": "move_tracked_point_current_plan_freeze",
    }
    return snapshot, metadata


def _plan_overlap_context_x5(
    snapshot: Snapshot,
    *,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    if snapshot.depth is None:
        raise ValueError("move-tracked plan overlap filter requires frozen depth")
    required_paths = (snapshot.paths.get("rgb_array"), snapshot.paths.get("depth"))
    if any(not value or not Path(str(value)).is_file() for value in required_paths):
        raise FileNotFoundError("move-tracked frozen RGB-D artifacts are unavailable")
    return {
        "rgb_array_path": str(required_paths[0]),
        "depth_path": str(required_paths[1]),
        "intrinsic_matrix": np.asarray(snapshot.intrinsic_matrix, dtype=np.float64)
        .reshape(3, 3)
        .astype(float)
        .tolist(),
        "extrinsic_matrix": np.asarray(snapshot.extrinsic_matrix, dtype=np.float64)
        .reshape(4, 4)
        .astype(float)
        .tolist(),
        "base_rotation": np.asarray(frozen["base_rotation"], dtype=np.float64)
        .reshape(3, 3)
        .astype(float)
        .tolist(),
        "session_id": snapshot.session_id,
        "image_id": snapshot.image_id,
        "observation_sequence": int(snapshot.sequence),
        "episode_id": snapshot.episode_id,
        "artifact_root": str(Path(str(required_paths[1])).parent),
        "observation_source": "robodojo_evaluator_frozen_rgbd_proprio",
    }


def _eef_waypoints_x5(
    frozen: Mapping[str, Any],
    arm: str,
    waypoints: Sequence[Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw_q in enumerate(waypoints):
        q = np.asarray(raw_q, dtype=np.float64).reshape(ARM_DOF)
        position_base, quaternion_xyzw = eef_pose(frozen["state"], arm, q)
        pose_env = _base_pose_xyzw_to_env_wxyz(
            position_base,
            quaternion_xyzw,
            frozen["base_rotation"],
        )
        result.append(
            {
                "label": f"move_tracked_waypoint_{index + 1}",
                "pose": pose_env.astype(float).tolist(),
                "planned_joint_target": q.astype(float).tolist(),
                "pose_source": (
                    "submission_local_static_x5_fk_from_signed_joint_waypoint"
                ),
                "pose_frame": "robodojo_env",
            }
        )
    return result


def _rigid_pair_activation_summary_x5(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw = report if isinstance(report, Mapping) else {}
    keys = (
        "ok",
        "reason",
        "names",
        "requested_names",
        "activated_names",
        "observation_sequence",
        "registration_observation_sequence",
        "reference_distance_m",
        "measured_distance_m",
        "fused_distance_m",
        "distance_error_m",
        "point_correction_m",
        "max_point_correction_m",
        "correction_limit_m",
        "fusion_applied",
        "tracker_estimator_ok",
        "direct_rgbd_projection_accepted_for_move_source",
        "move_source_max_point_correction_m",
        "fresh_observation_confirmation_required",
    )
    summary = {key: raw.get(key) for key in keys if key in raw}
    visual = raw.get("visual_tracker_pair")
    if isinstance(visual, Mapping):
        summary["visual_tracker_pair"] = {
            key: visual.get(key)
            for key in (
                "ok",
                "reason",
                "observation_sequence",
                "track_ids",
                "identity_margin_px",
                "pair_distance_error_m",
                "per_point",
            )
            if key in visual
        }
    return json_ready(summary)


def _gate_rigid_pair_source_projection_x5(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require small RGB-D projection corrections before freezing a source."""

    gated = dict(report) if isinstance(report, Mapping) else {}
    estimator_ok = bool(gated.get("ok"))
    gated["tracker_estimator_ok"] = estimator_ok
    gated["move_source_max_point_correction_m"] = float(
        RIGID_PAIR_SOURCE_MAX_POINT_CORRECTION_M
    )
    if not estimator_ok:
        gated["direct_rgbd_projection_accepted_for_move_source"] = False
        return gated
    raw_correction = gated.get("max_point_correction_m")
    if raw_correction is None:
        gated["direct_rgbd_projection_accepted_for_move_source"] = True
        return gated
    try:
        correction = float(raw_correction)
    except (TypeError, ValueError):
        correction = float("inf")
    accepted = bool(
        math.isfinite(correction)
        and correction <= RIGID_PAIR_SOURCE_MAX_POINT_CORRECTION_M + 1.0e-12
    )
    gated["direct_rgbd_projection_accepted_for_move_source"] = accepted
    if not accepted:
        gated.update(
            {
                "ok": False,
                "reason": "rigid_pair_projection_requires_fresh_confirmation",
                "fresh_observation_confirmation_required": True,
            }
        )
    return gated


def _rigid_pair_activation_error_x5(report: Mapping[str, Any]) -> str:
    reason = str(report.get("reason") or "rigid_pair_activation_recovery_failed")
    diagnostic_sources = [
        source
        for source in (report.get("last_pair_report"), report.get("first_failure"))
        if isinstance(source, Mapping)
    ]
    if not diagnostic_sources:
        return f"tracked rigid pair activation failed: {reason}"

    diagnostics: list[str] = []
    for key, label in (
        ("reference_distance_m", "reference"),
        ("measured_distance_m", "measured"),
        ("distance_error_m", "distance_error"),
        ("max_point_correction_m", "required_point_correction"),
        ("correction_limit_m", "correction_limit"),
    ):
        for source in diagnostic_sources:
            try:
                number = float(source.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                diagnostics.append(f"{label}={number:.6f}m")
                break
    suffix = f" ({', '.join(diagnostics)})" if diagnostics else ""
    return f"tracked rigid pair activation failed: {reason}{suffix}"


def _deactivate_rigid_pair_x5(
    manager,
    *,
    reason: str,
    required: bool,
) -> dict[str, Any]:
    deactivate = getattr(manager, "deactivate_rigid_pair", None)
    if not callable(deactivate):
        if required:
            return {
                "ok": False,
                "active": None,
                "reason": "tracker_provider_has_no_rigid_pair_deactivation",
                "requested_reason": str(reason),
            }
        return {
            "ok": True,
            "active": None,
            "reason": "tracker_provider_has_no_rigid_pair_deactivation",
            "requested_reason": str(reason),
        }
    try:
        try:
            report = deactivate(reason=str(reason))
        except TypeError:
            report = deactivate()
    except Exception as exc:
        return {
            "ok": False,
            "active": None,
            "reason": "rigid_pair_deactivation_failed",
            "requested_reason": str(reason),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(report, Mapping):
        return {
            "ok": False,
            "active": None,
            "reason": "invalid_rigid_pair_deactivation_result",
            "requested_reason": str(reason),
        }
    return json_ready(dict(report))


def _clear_incompatible_rigid_pair_x5(
    manager,
    names: Sequence[str],
) -> dict[str, Any]:
    requested = [str(name) for name in names]
    status_fn = getattr(manager, "status", None)
    if not callable(status_fn):
        return {"ok": True, "active": None, "reason": "tracker_provider_has_no_status"}
    status = status_fn()
    active = status.get("active_rigid_pair") if isinstance(status, Mapping) else None
    if not isinstance(active, Mapping):
        return {"ok": True, "active": False, "reason": "no_prior_rigid_pair"}
    active_names = [str(name) for name in active.get("names", ())]
    if len(requested) == 2 and active_names == requested:
        return {
            "ok": True,
            "active": True,
            "reason": "current_request_will_reactivate_exact_pair",
            "names": active_names,
        }
    report = _deactivate_rigid_pair_x5(
        manager,
        reason="rigid_pair_not_part_of_current_controlled_point_set",
        required=True,
    )
    if not bool(report.get("ok")):
        raise RuntimeError(
            "could not retire a rigid pair left by a previous move request: "
            f"{report.get('reason') or 'unknown'}"
        )
    return report


def _registered_points_snapshot_x5(
    manager,
    names: Sequence[str],
    *,
    episode_id: str,
) -> dict[str, Any]:
    requested = [str(name) for name in names]
    snapshot_fn = getattr(manager, "registered_points_snapshot", None)
    if not callable(snapshot_fn):
        return {
            "ok": True,
            "available": False,
            "reason": "tracker_provider_has_no_registration_snapshot",
            "requested_names": requested,
        }
    snapshot = snapshot_fn(requested, episode_id=str(episode_id))
    if not isinstance(snapshot, Mapping):
        raise ValueError("registered point snapshot must be an object")
    if snapshot.get("available") is False:
        return {
            **dict(snapshot),
            "ok": True,
            "available": False,
            "requested_names": requested,
        }
    if not bool(snapshot.get("ok")):
        raise ValueError(
            "registered point snapshot is unavailable: "
            f"{snapshot.get('reason') or 'unknown'}"
        )
    entries = snapshot.get("entries")
    if not isinstance(entries, Mapping):
        raise ValueError("registered point snapshot has no entries")
    points = _strict_point_matrix(
        [entries[name]["xyz_in_robot_base_coord_m"] for name in requested],
        expected_count=len(requested),
        label="registered tracked points",
    )
    session_id = str(snapshot.get("session_id") or "").strip()
    image_id = str(snapshot.get("image_id") or "").strip()
    snapshot_episode = str(snapshot.get("episode_id") or "").strip()
    if not session_id or not image_id:
        raise ValueError("registered point snapshot has no capture binding")
    if snapshot_episode != str(episode_id):
        raise ValueError("registered point snapshot episode changed")
    try:
        registration_sequence = int(snapshot["registration_observation_sequence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "registered point snapshot has no observation sequence"
        ) from exc
    return {
        "ok": True,
        "available": True,
        "reason": None,
        "requested_names": requested,
        "points_robot_base_m": points,
        "session_id": session_id,
        "image_id": image_id,
        "episode_id": snapshot_episode,
        "registration_observation_sequence": registration_sequence,
        "measurement_source": str(
            snapshot.get("measurement_source")
            or "policy_owned_frozen_head_rgbd_registration"
        ),
        "predictions_published": False,
    }


def _snapshot_x5_kinematics_robot_base(
    snapshot: Snapshot,
    *,
    base_rotation: Any,
) -> dict[str, Any]:
    env_from_base = np.asarray(base_rotation, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(env_from_base)):
        raise ValueError("registration capture base rotation is invalid")
    eef: dict[str, dict[str, np.ndarray]] = {}
    gripper: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        raw_pose = snapshot.robot.get(f"eef_{side}")
        if not isinstance(raw_pose, Mapping):
            raise ValueError(f"registration capture has no {side} EEF pose")
        position_env = np.asarray(raw_pose.get("pos"), dtype=np.float64).reshape(3)
        quaternion_env_wxyz = np.asarray(
            raw_pose.get("quat"), dtype=np.float64
        ).reshape(4)
        position_base = env_from_base.T @ position_env
        rotation_base = env_from_base.T @ quaternion_to_matrix(quaternion_env_wxyz)
        quaternion_base_wxyz = matrix_to_quaternion(rotation_base)
        gripper_q = np.asarray(
            snapshot.robot.get(f"gripper_{side}_qpos"), dtype=np.float64
        ).reshape(1)
        if not all(
            np.all(np.isfinite(value))
            for value in (position_base, quaternion_base_wxyz, gripper_q)
        ):
            raise ValueError("registration capture X5 kinematics are non-finite")
        eef[side] = {
            "position": position_base,
            "quaternion_xyzw": np.asarray(
                [
                    quaternion_base_wxyz[1],
                    quaternion_base_wxyz[2],
                    quaternion_base_wxyz[3],
                    quaternion_base_wxyz[0],
                ],
                dtype=np.float64,
            ),
        }
        gripper[side] = gripper_q
    return {
        "episode_id": snapshot.episode_id,
        "observation_sequence": int(snapshot.sequence),
        "eef": eef,
        "gripper_q": gripper,
    }


def _reconcile_registered_on_hand_source_x5(
    ctx,
    manager,
    names: Sequence[str],
    *,
    episode_id: str,
    current_frozen: Mapping[str, Any],
    live_points_robot_base_m: Any,
) -> dict[str, Any]:
    """Apply BEHAVIOR's bounded majority recovery in X5 robot-base axes."""

    requested = [str(name) for name in names]
    live = _strict_point_matrix(
        live_points_robot_base_m,
        expected_count=len(requested),
        label="live on-hand tracked points",
    )
    if len(requested) < NPOINT_VISUAL_MIN_POINTS:
        return {
            "ok": True,
            "applicable": False,
            "applied": False,
            "reason": "registration_eef_source_recovery_requires_three_points",
            "names": requested,
            "source_points_robot_base_m": live.astype(float).tolist(),
            "predictions_published_as_observation": False,
        }
    registration = _registered_points_snapshot_x5(
        manager, requested, episode_id=episode_id
    )
    if not bool(registration.get("available")):
        return {
            "ok": True,
            "applicable": False,
            "applied": False,
            "reason": registration.get("reason"),
            "names": requested,
            "source_points_robot_base_m": live.astype(float).tolist(),
            "registration_snapshot": json_ready(registration),
            "predictions_published_as_observation": False,
        }
    capture = ctx.snapshots.load(registration["session_id"], registration["image_id"])
    registration_state = _snapshot_x5_kinematics_robot_base(
        capture, base_rotation=current_frozen["base_rotation"]
    )
    if registration_state["episode_id"] != str(episode_id):
        raise ValueError("registration capture episode changed")
    if str(current_frozen.get("episode_id") or "") != str(episode_id):
        raise ValueError("current proprioception episode changed")
    registration_sequence = int(registration["registration_observation_sequence"])
    if int(capture.sequence) != registration_sequence:
        raise ValueError(
            "registration RGB-D and capture-time proprioception are not synchronized"
        )
    registered = _strict_point_matrix(
        registration["points_robot_base_m"],
        expected_count=len(requested),
        label="registered on-hand tracked points",
    )
    registration_distances = {
        side: float(
            np.mean(
                np.linalg.norm(
                    registered
                    - registration_state["eef"][side]["position"].reshape(1, 3),
                    axis=1,
                )
            )
        )
        for side in ("left", "right")
    }
    arm = min(("left", "right"), key=lambda side: registration_distances[side])
    registration_position = registration_state["eef"][arm]["position"]
    registration_rotation = quat_to_mat_xyzw(
        registration_state["eef"][arm]["quaternion_xyzw"]
    )
    anchors = (
        registration_rotation.T @ (registered - registration_position.reshape(1, 3)).T
    ).T
    current_pose_wxyz = np.asarray(
        current_frozen["eef_base_wxyz"][arm], dtype=np.float64
    ).reshape(7)
    current_quaternion_xyzw = np.asarray(
        [
            current_pose_wxyz[4],
            current_pose_wxyz[5],
            current_pose_wxyz[6],
            current_pose_wxyz[3],
        ],
        dtype=np.float64,
    )
    expected = (
        current_pose_wxyz[:3].reshape(1, 3)
        + (quat_to_mat_xyzw(current_quaternion_xyzw) @ anchors.T).T
    )
    errors = np.linalg.norm(live - expected, axis=1)
    inlier_mask = errors <= REGISTRATION_EEF_INLIER_M + 1.0e-12
    inlier_count = int(np.count_nonzero(inlier_mask))
    required_inliers = max(2, len(requested) // 2 + 1)
    maximum_error = float(np.max(errors)) if errors.size else 0.0
    inlier_names = [
        requested[index] for index in range(len(requested)) if bool(inlier_mask[index])
    ]
    outlier_names = [
        requested[index]
        for index in range(len(requested))
        if not bool(inlier_mask[index])
    ]
    current_gripper = np.asarray(
        current_frozen["gripper_q"][arm], dtype=np.float64
    ).reshape(1)
    registration_gripper = np.asarray(
        registration_state["gripper_q"][arm], dtype=np.float64
    ).reshape(1)
    common = {
        "applicable": True,
        "names": requested,
        "selected_arm": arm,
        "registration_eef_mean_distance_m": registration_distances,
        "registration_observation_sequence": registration_sequence,
        "registration_measurement_source": registration["measurement_source"],
        "anchors_eef_m": anchors.astype(float).tolist(),
        "expected_points_robot_base_m": expected.astype(float).tolist(),
        "live_points_robot_base_m": live.astype(float).tolist(),
        "per_point_live_to_registered_eef_error_m": {
            requested[index]: float(errors[index]) for index in range(len(requested))
        },
        "max_live_to_registered_eef_error_m": maximum_error,
        "inlier_tolerance_m": REGISTRATION_EEF_INLIER_M,
        "max_recovery_error_m": REGISTRATION_EEF_MAX_RECOVERY_M,
        "inlier_names": inlier_names,
        "outlier_names": outlier_names,
        "inlier_count": inlier_count,
        "required_inlier_count": required_inliers,
        "registration_to_current_gripper_delta_inf_normalized": float(
            np.max(np.abs(current_gripper - registration_gripper))
        ),
        "registration_to_current_gripper_delta_inf_m": abs(
            x5_gripper_joint_position_m(current_gripper)
            - x5_gripper_joint_position_m(registration_gripper)
        ),
        "planning_source_is_model_derived": bool(outlier_names),
        "predictions_published_as_observation": False,
    }
    if not outlier_names:
        return {
            **common,
            "ok": True,
            "applied": False,
            "reason": "all_live_points_agree_with_registered_eef_anchors",
            "source_points_robot_base_m": live.astype(float).tolist(),
            "planning_source_is_model_derived": False,
        }
    if inlier_count < required_inliers:
        return {
            **common,
            "ok": False,
            "applied": False,
            "reason": "registered_eef_anchor_consensus_insufficient",
            "source_points_robot_base_m": live.astype(float).tolist(),
        }
    if maximum_error > REGISTRATION_EEF_MAX_RECOVERY_M + 1.0e-12:
        return {
            **common,
            "ok": False,
            "applied": False,
            "reason": "registered_eef_anchor_recovery_exceeded",
            "source_points_robot_base_m": live.astype(float).tolist(),
        }
    return {
        **common,
        "ok": True,
        "applied": True,
        "reason": "bounded_tracker_outlier_recovered_from_registered_eef_anchors",
        "source_points_robot_base_m": expected.astype(float).tolist(),
        "recovered_point_names": outlier_names,
        "planning_source_is_model_derived": True,
    }


def _bootstrap_registered_on_hand_eef_prior_x5(
    ctx,
    manager,
    names: Sequence[str],
    *,
    episode_id: str,
    current_frozen: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the tracker candidate gate from registration RGB-D and X5 FK."""

    requested = [str(name) for name in names]
    if (
        not 1 <= len(requested) <= MOVE_TRACKED_POINT_MAX_POINTS
        or any(not name for name in requested)
        or len(set(requested)) != len(requested)
    ):
        return {
            "ok": False,
            "active": False,
            "reason": "registration_eef_bootstrap_point_names_invalid",
            "names": requested,
        }
    registration = _registered_points_snapshot_x5(
        manager, requested, episode_id=episode_id
    )
    if not bool(registration.get("available")):
        return {
            "ok": True,
            "active": False,
            "reason": registration.get("reason"),
            "names": requested,
            "registration_snapshot": json_ready(registration),
        }
    capture = ctx.snapshots.load(registration["session_id"], registration["image_id"])
    capture_state = _snapshot_x5_kinematics_robot_base(
        capture,
        base_rotation=current_frozen["base_rotation"],
    )
    if str(capture_state["episode_id"]) != str(episode_id):
        raise ValueError("registration capture episode changed")
    registration_sequence = int(registration["registration_observation_sequence"])
    if int(capture.sequence) != registration_sequence:
        raise ValueError(
            "registration RGB-D and capture-time proprioception are not synchronized"
        )
    points = _strict_point_matrix(
        registration["points_robot_base_m"],
        expected_count=len(requested),
        label="registered on-hand tracked points",
    )
    distances = {
        side: float(
            np.mean(
                np.linalg.norm(
                    points - capture_state["eef"][side]["position"].reshape(1, 3),
                    axis=1,
                )
            )
        )
        for side in ("left", "right")
    }
    arm = min(("left", "right"), key=lambda side: distances[side])
    registration_position = capture_state["eef"][arm]["position"]
    registration_rotation = quat_to_mat_xyzw(
        capture_state["eef"][arm]["quaternion_xyzw"]
    )
    anchors = (
        registration_rotation.T @ (points - registration_position.reshape(1, 3)).T
    ).T
    current_gripper = np.asarray(
        current_frozen["gripper_q"][arm], dtype=np.float64
    ).reshape(1)
    registration_gripper = np.asarray(
        capture_state["gripper_q"][arm], dtype=np.float64
    ).reshape(1)
    prior = manager.activate_on_hand_eef_observation_prior(
        requested,
        episode_id=episode_id,
        session_id=registration["session_id"],
        image_id=registration["image_id"],
        arm=arm,
        anchors_eef_m=anchors.tolist(),
    )
    if not isinstance(prior, Mapping):
        raise ValueError("on-hand EEF observation prior result must be an object")
    return {
        **dict(prior),
        "bootstrap": "registration_rgbd_capture_proprio_local_x5_fk",
        "selected_arm": arm,
        "eef_mean_distance_m": distances,
        "anchors_eef_m": anchors.astype(float).tolist(),
        "registration_observation_sequence": registration_sequence,
        "registration_measurement_source": registration["measurement_source"],
        "registration_to_current_gripper_delta_inf_normalized": float(
            np.max(np.abs(current_gripper - registration_gripper))
        ),
        "registration_to_current_gripper_delta_inf_m": abs(
            x5_gripper_joint_position_m(current_gripper)
            - x5_gripper_joint_position_m(registration_gripper)
        ),
        "prediction_values_published_as_observation": False,
    }


def _confirm_fresh_rigid_pair_activation_x5(
    ctx,
    manager,
    names: Sequence[str],
    *,
    episode_id: str,
    observation_sequence: int,
    initial_report: Mapping[str, Any],
    hold_state: Mapping[str, Any],
    deadline_monotonic: float,
):
    """Confirm a provisional pair switch on two strictly newer RGB-D frames."""

    requested = [str(name) for name in names]
    initial = dict(initial_report) if isinstance(initial_report, Mapping) else {}
    initial_reason = str(initial.get("reason") or "")
    base = {
        "required": not bool(initial.get("ok")),
        "recovered": False,
        "attempt_count": 1,
        "recovery_hold_steps": 0,
        "required_stable_steps": int(RIGID_PAIR_ACTIVATION_STABLE_STEPS),
        "max_wait_steps": int(RIGID_PAIR_ACTIVATION_MAX_WAIT_STEPS),
        "observation_sequence_start": int(observation_sequence),
        "observation_sequence_end": int(observation_sequence),
        "first_failure": (
            None
            if bool(initial.get("ok"))
            else _rigid_pair_activation_summary_x5(initial)
        ),
        "last_pair_report": _rigid_pair_activation_summary_x5(initial),
        "history": [],
        "correction_limit_unchanged": True,
    }
    if bool(initial.get("ok")):
        return {
            **base,
            "ok": True,
            "reason": None,
            "confirmation_source": "initial_activation_fast_path",
        }
    if initial_reason not in RIGID_PAIR_ACTIVATION_INITIAL_RETRY_REASONS:
        return {
            **base,
            "ok": False,
            "reason": initial_reason or "invalid_rigid_pair_fusion_result",
            "error": (
                "tracked rigid pair activation is not recoverable from a new "
                "observation"
            ),
        }
    local_deadline = min(
        float(deadline_monotonic),
        time.monotonic() + RIGID_PAIR_ACTIVATION_MAX_WAIT_S,
    )
    stable_steps = 0
    previous_sequence = int(observation_sequence)
    last_pair_report = dict(initial)
    history: list[dict[str, Any]] = []
    for wait_step in range(1, RIGID_PAIR_ACTIVATION_MAX_WAIT_STEPS + 1):
        ctx.raise_if_cancelled("move_tracked_point rigid-pair activation recovery")
        if time.monotonic() >= local_deadline:
            return {
                **base,
                "ok": False,
                "reason": "rigid_pair_activation_recovery_timeout",
                "attempt_count": int(1 + len(history)),
                "recovery_hold_steps": int(wait_step - 1),
                "observation_sequence_end": int(previous_sequence),
                "last_pair_report": _rigid_pair_activation_summary_x5(last_pair_report),
                "history": history,
            }
        yield _observation_advance_action(ctx, dict(hold_state))
        ctx.raise_if_cancelled("move_tracked_point rigid-pair activation recovery")
        if time.monotonic() >= local_deadline:
            return {
                **base,
                "ok": False,
                "reason": "rigid_pair_activation_recovery_timeout",
                "attempt_count": int(1 + len(history)),
                "recovery_hold_steps": int(wait_step),
                "observation_sequence_end": int(previous_sequence),
                "last_pair_report": _rigid_pair_activation_summary_x5(last_pair_report),
                "history": history,
            }
        current_sequence = _sequence(ctx)
        if current_sequence <= previous_sequence:
            return {
                **base,
                "ok": False,
                "reason": "rigid_pair_activation_observation_not_advanced",
                "attempt_count": int(1 + len(history)),
                "recovery_hold_steps": int(wait_step),
                "observation_sequence_end": int(current_sequence),
                "last_pair_report": _rigid_pair_activation_summary_x5(last_pair_report),
                "history": history,
            }
        if str(ctx.world.episode_id()) != str(episode_id):
            return {
                **base,
                "ok": False,
                "reason": "episode_changed",
                "recovery_hold_steps": int(wait_step),
                "observation_sequence_end": int(current_sequence),
                "history": history,
            }
        live = _live(ctx, requested, episode_id, manager=manager)
        raw_pair_report = live.get("rigid_pair_fusion")
        if not isinstance(raw_pair_report, Mapping):
            raw_pair_report = {
                "ok": False,
                "reason": str(
                    live.get("reason") or "rigid_pair_measurement_unavailable"
                ),
                "observation_sequence": live.get("observation_sequence"),
            }
        pair_report = _gate_rigid_pair_source_projection_x5(raw_pair_report)
        last_pair_report = dict(pair_report)
        live_sequence = live.get("observation_sequence")
        synchronized = bool(
            live_sequence is not None and int(live_sequence) == int(current_sequence)
        )
        frame_ok = bool(
            live.get("ok")
            and pair_report.get("ok")
            and synchronized
            and str(live.get("episode_id") or "") == str(episode_id)
        )
        frame_reason = (
            None
            if frame_ok
            else str(
                pair_report.get("reason")
                or live.get("reason")
                or (
                    "rigid_pair_activation_observation_not_synchronized"
                    if not synchronized
                    else "rigid_pair_measurement_invalid"
                )
            )
        )
        stable_steps = stable_steps + 1 if frame_ok else 0
        history.append(
            {
                "wait_step": int(wait_step),
                "observation_sequence": int(current_sequence),
                "live_observation_sequence": live_sequence,
                "synchronized": synchronized,
                "frame_ok": frame_ok,
                "reason": frame_reason,
                "stable_steps": int(stable_steps),
                "pair_report": _rigid_pair_activation_summary_x5(pair_report),
            }
        )
        previous_sequence = int(current_sequence)
        if frame_ok and stable_steps >= RIGID_PAIR_ACTIVATION_STABLE_STEPS:
            return {
                **base,
                "ok": True,
                "reason": None,
                "recovered": True,
                "attempt_count": int(1 + len(history)),
                "recovery_hold_steps": int(wait_step),
                "observation_sequence_end": int(current_sequence),
                "stable_steps": int(stable_steps),
                "last_pair_report": _rigid_pair_activation_summary_x5(pair_report),
                "history": history,
                "confirmation_source": (
                    "strictly_new_joint_rgbd_rigid_pair_observations"
                ),
            }
        if (
            not frame_ok
            and frame_reason not in RIGID_PAIR_ACTIVATION_FRESH_RETRY_REASONS
        ):
            return {
                **base,
                "ok": False,
                "reason": frame_reason,
                "attempt_count": int(1 + len(history)),
                "recovery_hold_steps": int(wait_step),
                "observation_sequence_end": int(current_sequence),
                "last_pair_report": _rigid_pair_activation_summary_x5(pair_report),
                "history": history,
            }
    return {
        **base,
        "ok": False,
        "reason": "rigid_pair_activation_recovery_exhausted",
        "attempt_count": int(1 + len(history)),
        "recovery_hold_steps": int(len(history)),
        "observation_sequence_end": int(previous_sequence),
        "last_failure_reason": str(
            last_pair_report.get("reason") or "rigid_pair_measurement_invalid"
        ),
        "last_pair_report": _rigid_pair_activation_summary_x5(last_pair_report),
        "history": history,
    }


def _reacquire_registered_bundle_x5(
    ctx,
    manager,
    *,
    names: Sequence[str],
    controlled_names: Sequence[str],
    episode_id: str,
    session_id: str,
    image_id: str,
    observation_sequence: int,
    hold_state: Mapping[str, Any],
    initial_report: Mapping[str, Any],
    lease_state: dict[str, Any],
    deadline_monotonic: float,
):
    """Require two fresh measured frames after registration-anchor failure."""

    report: dict[str, Any] = {
        "ok": False,
        "recovered": False,
        "initial_failure": json_ready(initial_report),
        "history": [],
        "hold_steps": 0,
        "predictions_published_as_observation": False,
    }
    deadline = min(
        float(deadline_monotonic),
        time.monotonic() + REGISTRATION_REACQUIRE_MAX_S,
    )
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        return {**report, "reason": "registered_bundle_reacquisition_timeout"}
    prior = _bootstrap_registered_on_hand_eef_prior_x5(
        ctx,
        manager,
        controlled_names,
        episode_id=episode_id,
        current_frozen=hold_state,
    )
    report["observation_prior"] = json_ready(prior)
    if not bool(prior.get("ok")) or not bool(prior.get("active")):
        return {
            **report,
            "reason": "registered_bundle_reacquisition_prior_unavailable",
        }
    lease_id = str(prior.get("lease_id") or "").strip()
    if not lease_id:
        return {
            **report,
            "reason": "registered_bundle_reacquisition_prior_has_no_lease",
        }
    lease_state.update(prior)
    lease_state["eef_prior_lease_id"] = lease_id
    previous_sequence = max(
        int(observation_sequence),
        int(prior.get("activation_observation_sequence", observation_sequence)),
    )
    stable_frames = 0
    for step in range(REGISTRATION_REACQUIRE_MAX_STEPS):
        ctx.raise_if_cancelled("move_tracked_point registered bundle reacquisition")
        if time.monotonic() >= deadline:
            break
        yield _observation_advance_action(ctx, dict(hold_state))
        report["hold_steps"] = int(step + 1)
        if str(ctx.world.episode_id()) != str(episode_id):
            return {**report, "reason": "episode_changed"}
        sequence = _sequence(ctx)
        if sequence <= previous_sequence:
            return {
                **report,
                "reason": "registered_bundle_observation_not_advanced",
            }
        previous_sequence = int(sequence)
        live = _live(ctx, list(names), episode_id, manager=manager)
        if not bool(live.get("ok")):
            report["history"].append(
                {
                    "sequence": int(sequence),
                    "reason": live.get("reason"),
                    "unavailable": json_ready(live.get("unavailable")),
                }
            )
            stable_frames = 0
            continue
        if (
            str(live.get("episode_id") or "") != str(episode_id)
            or str(live.get("session_id") or "") != str(session_id)
            or str(live.get("image_id") or "") != str(image_id)
        ):
            return {**report, "reason": "registered_bundle_binding_changed"}
        if int(live.get("observation_sequence", -1)) != int(sequence):
            return {
                **report,
                "reason": "registered_bundle_observation_mismatch",
            }
        current_frozen = _freeze_state(ctx)
        if _sequence(ctx) != sequence:
            return {
                **report,
                "reason": "registered_bundle_observation_mismatch",
            }
        matrix = _strict_point_matrix(
            [live["entries"][name]["xyz_in_robot_base_coord_m"] for name in names],
            expected_count=len(names),
            label="reacquired tracked source points",
        )
        source_by_name = {
            str(name): matrix[index].copy() for index, name in enumerate(names)
        }
        controlled = np.asarray(
            [source_by_name[str(name)] for name in controlled_names],
            dtype=np.float64,
        )
        reconciliation = _reconcile_registered_on_hand_source_x5(
            ctx,
            manager,
            controlled_names,
            episode_id=episode_id,
            current_frozen=current_frozen,
            live_points_robot_base_m=controlled,
        )
        visually_confirmed = bool(
            reconciliation.get("ok")
            and reconciliation.get("applicable")
            and not reconciliation.get("applied")
            and not reconciliation.get("outlier_names")
        )
        stable_frames = stable_frames + 1 if visually_confirmed else 0
        report["history"].append(
            {
                "sequence": int(sequence),
                "reason": reconciliation.get("reason"),
                "max_error_m": reconciliation.get("max_live_to_registered_eef_error_m"),
                "stable_frames": int(stable_frames),
            }
        )
        report["final_reconciliation"] = json_ready(reconciliation)
        if stable_frames >= 2:
            report.update(ok=True, recovered=True, reason=None)
            reconciliation["fresh_rgbd_reacquisition"] = json_ready(report)
            return {
                "ok": True,
                "report": report,
                "observation_sequence": int(sequence),
                "live": live,
                "frozen": current_frozen,
                "source_by_name": source_by_name,
                "reconciliation": reconciliation,
                "observation_prior": prior,
            }
    return {**report, "reason": "registered_bundle_reacquisition_timeout"}


def _observation_advance_action(ctx, frozen: dict[str, Any]) -> dict[str, list[float]]:
    """Hold the captured robot state only to obtain a fresh evaluator frame."""

    gripper_targets, _report = _x5_gripper_hold_targets(ctx, frozen)
    return ctx.world.make_joint_action(
        left_arm=frozen["arm_q"]["left"].tolist(),
        left_gripper=gripper_targets["left"],
        right_arm=frozen["arm_q"]["right"].tolist(),
        right_gripper=gripper_targets["right"],
    )


def _planning_targets(
    points: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    """Strip role metadata and preserve optional coordinate constraints."""

    by_name = {str(point.get("name") or ""): point for point in points}
    result: list[dict[str, Any]] = []
    for raw_name in names:
        name = str(raw_name)
        point = by_name.get(name)
        if point is None:
            raise ValueError(f"move-tracked target point {name!r} is missing")
        target = point.get("target_xyz_m")
        if target is None:
            target = {axis: {"free": True} for axis in ("x", "y", "z")}
        result.append({"name": name, "target_xyz_m": json_ready(target)})
    return result


MOVE_TRACKED_OVERLAP_FALLBACK_POLICY = "best_accuracy_then_minimum_overlap_v1"


def _best_accuracy_records(
    records: Sequence[tuple[dict[str, Any], str]],
    *,
    reference_records: Sequence[tuple[dict[str, Any], str]] | None = None,
) -> list[tuple[dict[str, Any], str]]:
    """Keep endpoint accuracy authoritative over collision or motion ranking."""

    def accuracy(candidate: Mapping[str, Any]) -> tuple[bool, float]:
        selected = candidate.get("selected_candidate") or {}
        precise = bool(
            selected.get("precise", candidate.get("selected_is_precise", False))
        )
        try:
            cost = float(selected.get("accuracy_cost_mm_plus_deg", math.inf))
        except (TypeError, ValueError):
            cost = math.inf
        return precise, cost if math.isfinite(cost) and cost >= 0.0 else math.inf

    reference = list(records if reference_records is None else reference_records)
    has_precise = any(accuracy(candidate)[0] for candidate, _ in reference)
    best_cost = min(
        (accuracy(candidate)[1] for candidate, _ in reference), default=math.inf
    )
    return [
        (candidate, preference)
        for candidate, preference in records
        if (
            accuracy(candidate)[0]
            if has_precise
            else math.isfinite(best_cost)
            and math.isclose(
                accuracy(candidate)[1], best_cost, rel_tol=0.0, abs_tol=1.0e-9
            )
        )
    ]


def _overlap_fallback_records(
    records: Sequence[tuple[dict[str, Any], str]],
) -> list[tuple[dict[str, Any], str]]:
    """Rank colliding X5 endpoints without weakening their accuracy class."""

    eligible: list[tuple[dict[str, Any], str, bool, float, float, float | None]] = []
    for candidate, preference in records:
        report = candidate.get("overlap_volume_filter")
        selected = candidate.get("selected_candidate")
        if (
            "q_final" not in candidate
            or not isinstance(report, Mapping)
            or not isinstance(selected, Mapping)
            or bool(report.get("ok"))
            or report.get("evaluation_available") is False
        ):
            continue
        try:
            original = float(report["overlap_vol_cm3"])
            raw_inflated = report.get("inflated_overlap_vol_cm3")
            inflated = None if raw_inflated is None else float(raw_inflated)
            cost = float(selected["accuracy_cost_mm_plus_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not math.isfinite(original)
            or original < 0.0
            or not math.isfinite(cost)
            or cost < 0.0
            or (
                inflated is not None and (not math.isfinite(inflated) or inflated < 0.0)
            )
            or (inflated is None and not bool(report.get("inflated_query_skipped")))
        ):
            continue
        precise = bool(
            selected.get("precise", candidate.get("selected_is_precise", False))
        )
        eligible.append((candidate, preference, precise, cost, original, inflated))
    if not eligible:
        return []
    has_precise = any(item[2] for item in eligible)
    minimum_cost = min(item[3] for item in eligible)
    frontier = [
        item
        for item in eligible
        if (
            item[2]
            if has_precise
            else math.isclose(item[3], minimum_cost, rel_tol=0.0, abs_tol=1.0e-9)
        )
    ]
    frontier.sort(
        key=lambda item: (
            item[4],
            math.inf if item[5] is None else item[5],
            item[3],
            float(item[0]["selected_candidate"].get("motion_score", math.inf)),
        )
    )
    result: list[tuple[dict[str, Any], str]] = []
    for candidate, preference, _precise, cost, _original, _inflated in frontier:
        candidate = {
            **candidate,
            "overlap_volume_filter": {
                **candidate["overlap_volume_filter"],
                "fallback_selection": {
                    "policy": MOVE_TRACKED_OVERLAP_FALLBACK_POLICY,
                    "accuracy_class": "precise" if has_precise else "minimum_error",
                    "accuracy_cost_mm_plus_deg": cost,
                    "accuracy_frontier_count": len(frontier),
                    "overlap_order": "original_then_inflated_cm3",
                },
            },
        }
        result.append((candidate, preference))
    return result


def _affine_constraint_pose_family(
    *,
    candidate: Mapping[str, Any],
    source_points: Sequence[Sequence[float]],
    target_points: Sequence[Mapping[str, Any]],
    fixed_points: Sequence[Sequence[float]] | None,
    fixed_target_points: Sequence[Mapping[str, Any]] | None,
    point_names: Sequence[str],
    relations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build affine and relation-defined rigid null-space members for X5."""

    try:
        position = np.asarray(
            candidate["final_eef_position_m"], dtype=np.float64
        ).reshape(3)
        rotation = quat_to_mat_xyzw(candidate["final_eef_quaternion_xyzw"])
        anchors = np.asarray(candidate["anchors_eef_m"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("affine pose-family candidate is incomplete") from exc
    poses = list(
        _affine_pose_family_targets(
            source_points_robot_base_m=source_points,
            anchors_eef_m=anchors,
            base_rotation=rotation,
            base_position=position,
            target_points=target_points,
            fixed_points_robot_base_m=fixed_points,
            fixed_target_points=fixed_target_points,
        )
    )

    # ``_affine_pose_family_targets`` exactly solves coordinate equations. A
    # relation such as ordered_collinear can provide the free axial rotation
    # even when frozen RGB-D noise makes those equations only tolerance-close.
    # Preserve that legitimate relation null space with a proper half-turn;
    # normal IK and the complete constraint checker still decide acceptance.
    names = [str(name) for name in point_names]
    resolved = np.asarray(
        candidate.get("resolved_points_robot_base_m"), dtype=np.float64
    )
    if (
        len(set(names)) == len(names)
        and all(names)
        and resolved.shape == (len(names), 3)
        and np.all(np.isfinite(resolved))
    ):
        name_to_index = {name: index for index, name in enumerate(names)}
        for relation_index, relation in enumerate(relations):
            if str(relation.get("type") or "").strip().lower() != ("ordered_collinear"):
                continue
            relation_names = [str(name) for name in relation.get("point_names") or []]
            controlled = [name for name in relation_names if name in name_to_index]
            best_pair: tuple[str, str] | None = None
            best_distance = 0.0
            for first_index, first_name in enumerate(controlled[:-1]):
                first = resolved[name_to_index[first_name]]
                for second_name in controlled[first_index + 1 :]:
                    second = resolved[name_to_index[second_name]]
                    distance = float(np.linalg.norm(second - first))
                    if distance > best_distance:
                        best_pair = (first_name, second_name)
                        best_distance = distance
            if best_pair is None or best_distance <= 1.0e-6:
                continue
            first_name, second_name = best_pair
            axis_origin = resolved[name_to_index[first_name]]
            axis = (resolved[name_to_index[second_name]] - axis_origin) / best_distance
            axial_flip = 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
            target_position = axis_origin + axial_flip @ (position - axis_origin)
            target_rotation = axial_flip @ rotation
            duplicate = False
            for pose in poses:
                try:
                    pose_position = np.asarray(
                        pose["final_eef_position_m"], dtype=np.float64
                    ).reshape(3)
                    pose_rotation = quat_to_mat_xyzw(pose["final_eef_quaternion_xyzw"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.allclose(
                    target_position, pose_position, atol=1.0e-9, rtol=0.0
                ) and np.allclose(
                    target_rotation, pose_rotation, atol=1.0e-9, rtol=0.0
                ):
                    duplicate = True
                    break
            if duplicate:
                continue
            invariant_errors = [
                float(
                    np.linalg.norm(
                        axis_origin
                        + axial_flip @ (resolved[name_to_index[name]] - axis_origin)
                        - resolved[name_to_index[name]]
                    )
                )
                for name in controlled
            ]
            poses.append(
                {
                    "final_eef_position_m": target_position.astype(float).tolist(),
                    "final_eef_quaternion_xyzw": mat_to_quat_xyzw(target_rotation)
                    .astype(float)
                    .tolist(),
                    "pose_family": "ordered_collinear_axial_normal_flip",
                    "relation_index": int(relation_index),
                    "axis_point_names": [first_name, second_name],
                    "axial_rotation_deg": 180.0,
                    "axis_origin_robot_base_m": axis_origin.astype(float).tolist(),
                    "axis_direction_robot_base": axis.astype(float).tolist(),
                    "invariant_max_axis_point_error_m": float(max(invariant_errors)),
                }
            )
    return {
        "ok": bool(poses),
        "reason": None if poses else "constraint_pose_family_empty",
        "pose_family": "affine_and_relation_rigid_null_space",
        "pose_count": int(len(poses)),
        "poses": poses,
    }


def _plan_frozen(
    *,
    state: LocalRobotState,
    arm: str,
    q_start: np.ndarray,
    source_points: np.ndarray,
    target_points: list[dict[str, Any]],
    relations: Sequence[Mapping[str, Any]] | None = None,
    inequalities: Sequence[Mapping[str, Any]] | None = None,
    fixed_points: np.ndarray | None = None,
    fixed_target_points: list[dict[str, Any]] | None = None,
    quick_constraint: Mapping[str, Any] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
    off_hand_points: np.ndarray | None = None,
    off_hand_target_points: Sequence[Mapping[str, Any]] | None = None,
    on_hand_names: Sequence[str] | None = None,
    off_hand_names: Sequence[str] | None = None,
    head_camera_position_robot_base_m: Sequence[float] | None = None,
    head_camera_quaternion_xyzw: Sequence[float] | None = None,
    pos_tol_m: float,
    ori_tol_deg: float,
    max_steps: int,
    overlap_filter_context: Mapping[str, Any] | None = None,
    cancel_event: threading.Event | None = None,
    planning_deadline_monotonic: float | None = None,
    hard_deadline_monotonic: float | None = None,
    isolate_process: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Plan exclusively from frozen policy-owned state and RGB-D artifacts."""

    # Overlap evaluation may visit several endpoint frontiers.  Give the X5
    # adapter a request-private mutable context so it can reuse the immutable
    # RGB-D scene reconstruction without mutating the caller's wire payload.
    if overlap_filter_context is not None:
        overlap_filter_context = dict(overlap_filter_context)

    if isolate_process:
        from tools.util.tracked_point_planning_worker import run_plan_worker

        worker_kwargs = {
            "state": state,
            "arm": arm,
            "q_start": q_start,
            "source_points": source_points,
            "target_points": target_points,
            "relations": relations,
            "inequalities": inequalities,
            "fixed_points": fixed_points,
            "fixed_target_points": fixed_target_points,
            "quick_constraint": quick_constraint,
            "quick_constraints": quick_constraints,
            "off_hand_points": off_hand_points,
            "off_hand_target_points": off_hand_target_points,
            "on_hand_names": on_hand_names,
            "off_hand_names": off_hand_names,
            "head_camera_position_robot_base_m": (head_camera_position_robot_base_m),
            "head_camera_quaternion_xyzw": head_camera_quaternion_xyzw,
            "pos_tol_m": pos_tol_m,
            "ori_tol_deg": ori_tol_deg,
            "max_steps": max_steps,
            "overlap_filter_context": overlap_filter_context,
            "planning_deadline_monotonic": planning_deadline_monotonic,
        }
        deadline = (
            float(hard_deadline_monotonic)
            if hard_deadline_monotonic is not None
            else time.monotonic() + 90.0
        )
        return run_plan_worker(
            worker_kwargs,
            cancel_event=cancel_event,
            hard_deadline_monotonic=deadline,
            progress_callback=progress_callback,
        )

    phase_started = time.monotonic()
    phase_history: list[dict[str, Any]] = []

    def planning_stage(name: str, **details: Any) -> None:
        now = time.monotonic()
        if phase_history:
            phase_history[-1]["elapsed_s"] = (
                now - phase_history[-1]["started_monotonic"]
            )
        phase_history.append(
            {"stage": name, "started_monotonic": now, "elapsed_s": 0.0}
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": name,
                    "elapsed_s": now - phase_started,
                    "stages": [dict(item) for item in phase_history],
                    **json_ready(details),
                }
            )

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("frozen tracked-point planning was cancelled")
        if hard_deadline_monotonic is not None and time.monotonic() >= float(
            hard_deadline_monotonic
        ):
            raise PlanningDeadlineExceeded("frozen planning hard deadline exhausted")

    check_cancelled()
    planning_stage("input_validation")
    if fixed_points is not None and off_hand_points is not None:
        if not np.allclose(fixed_points, off_hand_points, atol=0.0, rtol=0.0):
            raise ValueError("fixed_points and off_hand_points disagree")
    reference_points = fixed_points if off_hand_points is None else off_hand_points
    if fixed_target_points is not None and off_hand_target_points is not None:
        if json_ready(fixed_target_points) != json_ready(off_hand_target_points):
            raise ValueError("fixed_target_points and off_hand_target_points disagree")
    reference_targets = (
        fixed_target_points
        if off_hand_target_points is None
        else list(off_hand_target_points)
    )

    planning_relations = [dict(item) for item in (relations or [])]
    quick_specs = _quick_constraint_specs(
        quick_constraint=quick_constraint,
        quick_constraints=quick_constraints,
    )
    faceto_requested = any(
        str(spec.get("type") or "").strip().lower().replace("-", "_").replace(" ", "_")
        in {"faceto", "reverse_faceto"}
        for spec in quick_specs
    )
    faceto_camera_pose = _camera_pose_payload(
        head_camera_position_robot_base_m,
        head_camera_quaternion_xyzw,
    )
    if faceto_requested and faceto_camera_pose is None:
        raise ValueError(
            "faceto/reverse_faceto planning requires the frozen head-camera pose"
        )
    controlled_names = (
        [str(name) for name in on_hand_names]
        if on_hand_names is not None
        else [str(item.get("name") or "") for item in target_points]
    )
    reference_names = [str(name) for name in (off_hand_names or [])]
    if quick_specs:
        planning_relations = _relations_with_quick_constraints(
            planning_relations,
            quick_specs,
            controlled_names=controlled_names,
            controlled_points=np.asarray(source_points, dtype=np.float64),
            reference_names=reference_names,
            reference_points=(
                None
                if reference_points is None
                else np.asarray(reference_points, dtype=np.float64)
            ),
            camera_position_robot_base_m=head_camera_position_robot_base_m,
            camera_quaternion_xyzw=head_camera_quaternion_xyzw,
        )

    endpoint_kwargs = dict(
        state=state,
        arm=arm,
        q_start=q_start,
        source_points_robot_base_m=source_points,
        target_points=target_points,
        relations=planning_relations,
        inequalities=list(inequalities or []),
        fixed_points_robot_base_m=reference_points,
        fixed_target_points=reference_targets,
        pos_tol_m=pos_tol_m,
        ori_tol_deg=ori_tol_deg,
        cancel_requested=(cancel_event.is_set if cancel_event is not None else None),
        planning_deadline_monotonic=planning_deadline_monotonic,
    )
    planning_stage("endpoint_ik")
    endpoint_solution = plan_endpoint(**endpoint_kwargs)
    check_cancelled()
    candidate_queue = [
        (candidate, "nearest_tracked_point_transform")
        for candidate in list(
            endpoint_solution.pop("_ranked_endpoint_candidates", [endpoint_solution])
        )
    ]
    endpoint_accuracy_pool = list(candidate_queue)
    overlap_rejected: list[tuple[dict[str, Any], str]] = []
    overlap_batches: list[dict[str, Any]] = []
    overlap_candidate_count = 0
    overlap_passing_count = 0

    def filter_overlap(
        records: Sequence[tuple[dict[str, Any], str]], *, stage: str
    ) -> list[tuple[dict[str, Any], str]]:
        nonlocal overlap_candidate_count, overlap_passing_count
        local = list(records)
        if overlap_filter_context is None or not local:
            return local
        check_cancelled()
        planning_stage("overlap_frontier", frontier=stage, candidate_count=len(local))
        poses = [
            {
                "eef_pos": candidate["final_eef_position_m"],
                "quat": candidate["final_eef_quaternion_xyzw"],
            }
            for candidate, _ in local
        ]
        evaluation = evaluate_x5_move_endpoint_overlap(overlap_filter_context, poses)
        reports = list(evaluation.get("poses") or [])
        if len(reports) != len(local):
            raise RuntimeError("X5 overlap filter returned a mismatched pose count")
        accepted: list[tuple[dict[str, Any], str]] = []
        for index, ((candidate, preference), report) in enumerate(zip(local, reports)):
            report = {
                **dict(report),
                "stage": stage,
                "candidate_index_in_batch": index,
                "endpoint_preference": preference,
                "endpoint_score_rank": int(
                    (candidate.get("selected_candidate") or {}).get(
                        "endpoint_score_rank", index
                    )
                ),
            }
            candidate["overlap_volume_filter"] = report
            if bool(report.get("ok")):
                accepted.append((candidate, preference))
            elif "q_final" in candidate:
                overlap_rejected.append((candidate, preference))
        overlap_candidate_count += len(local)
        overlap_passing_count += len(accepted)
        overlap_batches.append(
            {
                **{key: value for key, value in evaluation.items() if key != "poses"},
                "stage": stage,
                "poses": reports,
            }
        )
        planning_stage(
            "overlap_frontier_complete",
            frontier=stage,
            passing_count=len(accepted),
            rejected_count=len(local) - len(accepted),
        )
        # Raw shared-line pose-family records have not gone through IK yet and
        # therefore do not carry endpoint accuracy fields.  They must only be
        # collision-prefiltered here; accuracy-class filtering belongs to the
        # post-IK frontier, matching BEHAVIOR official_v2.
        if str(stage).startswith("overlap_aware_pose_family_prefilter"):
            return accepted
        return _best_accuracy_records(
            accepted, reference_records=endpoint_accuracy_pool
        )

    candidate_queue = filter_overlap(candidate_queue, stage="endpoint_frontier")
    fallback_generated = geometry_rank(source_points) == 0
    failures: list[dict[str, Any]] = []
    tested: list[tuple[dict[str, Any], str, int]] = []
    collision_fallback_generated = overlap_filter_context is None
    affine_pose_family_generated = overlap_filter_context is None
    shared_line_generated = overlap_filter_context is None

    def overlap_summary(candidate: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if overlap_filter_context is None:
            return None
        selected = (
            candidate.get("overlap_volume_filter")
            if isinstance(candidate, Mapping)
            else None
        )
        fallback = (
            selected.get("fallback_selection")
            if isinstance(selected, Mapping)
            else None
        )
        return {
            "ok": bool(isinstance(selected, Mapping) and selected.get("ok")),
            "enabled": True,
            "required_for_execution_mode": "plan",
            "candidate_count_evaluated": overlap_candidate_count,
            "candidate_count_passing": overlap_passing_count,
            "candidate_count_rejected": len(overlap_rejected),
            "selected_pose": json_ready(selected),
            "fallback_applied": bool(fallback),
            "fallback_selection": json_ready(fallback),
            "warning": (
                "overlap_thresholds_exceeded_minimum_overlap_fallback"
                if fallback
                else None
            ),
            "batches": json_ready(overlap_batches),
            "grasp_opening_volume_queried": False,
            "filter_order": (
                "original_overlap_all_then_inflated_overlap_original_passers"
            ),
            "segmentation_used": False,
            "simulator_mesh_used": False,
            "direct_simulator_mutation": False,
        }

    def try_path(
        candidate: dict[str, Any], preference: str, path_rank: int
    ) -> dict[str, Any] | None:
        rank = int(
            candidate.get("selected_candidate", {}).get(
                "endpoint_score_rank", path_rank
            )
        )
        try:
            planning_stage("cartesian_path", endpoint_score_rank=rank)
            translation = float(
                np.linalg.norm(
                    np.asarray(candidate["final_eef_position_m"])
                    - np.asarray(candidate["start_eef_position_m"])
                )
            )
            if translation > MAX_TRANSLATION_M:
                raise ValueError(
                    f"planned EEF translation {translation:.3f}m exceeds "
                    f"{MAX_TRANSLATION_M:.3f}m"
                )
            path = plan_cartesian_trajectory(
                state=state,
                arm=arm,
                q_start=candidate["q_start"],
                q_final=candidate["q_final"],
                pos_tol_m=pos_tol_m,
                ori_tol_deg=ori_tol_deg,
                max_joint_step_rad=MAX_JOINT_STEP_RAD,
                max_waypoints=MAX_TRAJECTORY_WAYPOINTS,
                tracked_anchors_eef=candidate.get("anchors_eef_m"),
                cancel_requested=(
                    cancel_event.is_set if cancel_event is not None else None
                ),
                planning_deadline_monotonic=planning_deadline_monotonic,
            )
        except (PlanningDeadlineExceeded, RuntimeError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "endpoint_score_rank": rank,
                    "path_candidate_rank": path_rank,
                    "endpoint_preference": preference,
                    "path_mode": "cartesian",
                    "error": str(exc),
                }
            )
            try:
                planning_stage("joint_path_fallback", endpoint_score_rank=rank)
                path = plan_joint_space_trajectory(
                    state=state,
                    arm=arm,
                    q_start=candidate["q_start"],
                    q_final=candidate["q_final"],
                    max_joint_step_rad=MAX_JOINT_STEP_RAD,
                    max_waypoints=MAX_TRAJECTORY_WAYPOINTS,
                    tracked_anchors_eef=candidate.get("anchors_eef_m"),
                    cancel_requested=(
                        cancel_event.is_set if cancel_event is not None else None
                    ),
                )
                path["path_mode"] = "joint_space_fallback"
            except (RuntimeError, TypeError, ValueError) as fallback_exc:
                failures.append(
                    {
                        "endpoint_score_rank": rank,
                        "path_candidate_rank": path_rank,
                        "endpoint_preference": preference,
                        "path_mode": "joint_space_fallback",
                        "error": str(fallback_exc),
                    }
                )
                return None
        candidate["path_candidate_rank"] = path_rank
        candidate["path_endpoint_preference"] = preference
        candidate["path_candidates_tested"] = len(tested)
        candidate["path_candidate_failures"] = list(failures)
        report = overlap_summary(candidate)
        if report is not None:
            path["frozen_rgbd_overlap_filter"] = report
        planning_stage("complete", selected_constraints=candidate.get("constraints"))
        return {
            "ok": True,
            "endpoint": candidate,
            "path": path,
            "path_plan": path,
            "failures": failures,
            "path_failures": failures,
            "planning_target_points": json_ready(target_points),
            "planning_relations": json_ready(planning_relations),
            "planning_inequalities": json_ready(list(inequalities or [])),
            "off_hand_target_points": json_ready(reference_targets or []),
            "faceto_camera_pose": json_ready(faceto_camera_pose),
            "frozen_rgbd_overlap_filter": report,
        }

    while (
        candidate_queue
        or not affine_pose_family_generated
        or not shared_line_generated
        or not fallback_generated
        or not collision_fallback_generated
    ):
        check_cancelled()
        if not candidate_queue:
            if not affine_pose_family_generated:
                affine_pose_family_generated = True
                raw_family_records: list[tuple[dict[str, Any], str]] = []
                base_records = _best_accuracy_records(
                    overlap_rejected,
                    reference_records=endpoint_accuracy_pool,
                )[:2]
                for base_candidate, _base_preference in base_records:
                    family = _affine_constraint_pose_family(
                        candidate=base_candidate,
                        source_points=source_points,
                        target_points=target_points,
                        fixed_points=reference_points,
                        fixed_target_points=reference_targets,
                        point_names=controlled_names,
                        relations=planning_relations,
                    )
                    raw_family_records.extend(
                        (dict(item), "overlap_aware_affine_pose_family")
                        for item in list(family.get("poses") or [])
                    )
                passing_family = filter_overlap(
                    raw_family_records,
                    stage="overlap_aware_pose_family_prefilter_affine",
                )
                if passing_family:
                    pose_targets = [item[0] for item in passing_family[:24]]
                    solved = plan_endpoint(
                        **endpoint_kwargs,
                        eef_pose_targets=pose_targets,
                        pose_targets_only=True,
                        max_endpoint_candidates=24,
                    )
                    solved_records = [
                        (item, "overlap_aware_affine_pose_family")
                        for item in solved.pop("_ranked_endpoint_candidates", [solved])
                    ]
                    endpoint_accuracy_pool.extend(solved_records)
                    candidate_queue.extend(
                        filter_overlap(
                            solved_records,
                            stage="overlap_aware_pose_family_post_ik_affine",
                        )
                    )
                if candidate_queue:
                    continue
            if not shared_line_generated:
                shared_line_generated = True
                line_count = sum(
                    str(relation.get("type") or "") == "line_through_point"
                    for relation in planning_relations
                )
                if line_count >= 2 and overlap_rejected:
                    base_candidates = [item[0] for item in overlap_rejected[:2]]
                    start_position, start_quaternion = eef_pose(state, arm, q_start)
                    family = shared_line_target_pose_family(
                        source_points_robot_base_m=source_points,
                        point_names=controlled_names,
                        relations=[
                            relation
                            for relation in planning_relations
                            if relation.get("type") != "ordered_collinear"
                        ],
                        start_eef_position_m=start_position,
                        start_eef_quaternion_xyzw=start_quaternion,
                        base_target_eef_quaternions_xyzw=[
                            item["final_eef_quaternion_xyzw"]
                            for item in base_candidates
                        ],
                        angular_offsets_deg=(-20.0, 0.0, 20.0),
                    )
                    if bool(family.get("ok")):
                        raw_family = [
                            (dict(item), "overlap_aware_shared_line_pose_family")
                            for item in list(family.get("poses") or [])
                        ]
                        passing_family = filter_overlap(
                            raw_family,
                            stage="overlap_aware_pose_family_prefilter",
                        )
                        if passing_family:
                            pose_targets = [item[0] for item in passing_family[:24]]
                            solved = plan_endpoint(
                                **endpoint_kwargs,
                                eef_pose_targets=pose_targets,
                                pose_targets_only=True,
                                max_endpoint_candidates=24,
                            )
                            solved_records = [
                                (item, "overlap_aware_shared_line_pose_family")
                                for item in solved.pop(
                                    "_ranked_endpoint_candidates", [solved]
                                )
                            ]
                            endpoint_accuracy_pool.extend(solved_records)
                            candidate_queue.extend(
                                filter_overlap(
                                    solved_records,
                                    stage="overlap_aware_pose_family_post_ik",
                                )
                            )
                if candidate_queue:
                    continue
            if not fallback_generated:
                fallback_generated = True
                try:
                    fallback_solution = plan_endpoint(
                        **endpoint_kwargs,
                        eef_orientation_regularization_weight=(
                            ENDPOINT_PATH_FALLBACK_ORIENTATION_REGULARIZATION_WEIGHT
                        ),
                    )
                    fallback_records = [
                        (candidate, "minimum_orientation_path_fallback")
                        for candidate in fallback_solution.pop(
                            "_ranked_endpoint_candidates", [fallback_solution]
                        )
                    ]
                    endpoint_accuracy_pool.extend(fallback_records)
                    candidate_queue.extend(
                        filter_overlap(
                            fallback_records,
                            stage="minimum_orientation_path_fallback_frontier",
                        )
                    )
                except (
                    PlanningDeadlineExceeded,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    failures.append(
                        {
                            "endpoint_score_rank": len(tested),
                            "endpoint_preference": "minimum_orientation_path_fallback",
                            "error": f"fallback endpoint solve failed: {exc}",
                        }
                    )
                if candidate_queue:
                    continue
            if not collision_fallback_generated:
                collision_fallback_generated = True
                candidate_queue.extend(
                    _overlap_fallback_records(
                        _best_accuracy_records(
                            overlap_rejected,
                            reference_records=endpoint_accuracy_pool,
                        )
                    )
                )
                if candidate_queue:
                    continue
            break
        candidate, preference = candidate_queue.pop(0)
        path_rank = len(tested)
        tested.append((candidate, preference, path_rank))
        result = try_path(candidate, preference, path_rank)
        if result is not None:
            return result

    report = overlap_summary(None)
    detail = "; ".join(
        f"rank {item.get('path_candidate_rank', -1)}: {item['error']}"
        for item in failures
    )
    return {
        "ok": False,
        "error": "no endpoint candidate has a valid same-branch Cartesian path"
        + (f": {detail}" if detail else ""),
        "failures": failures,
        "path_failures": failures,
        "frozen_rgbd_overlap_filter": report,
    }


def _trajectory_record(
    *,
    args: dict[str, Any],
    frozen: dict[str, Any],
    episode_id: str,
    observation_sequence: int,
    session_id: str,
    image_id: str,
    names: list[str],
    nearest_name: str,
    source_points: np.ndarray,
    arm: str,
    endpoint: dict[str, Any],
    path: dict[str, Any],
    tracking_names: list[str] | None = None,
    controlled_names: list[str] | None = None,
    reference_names: list[str] | None = None,
    reference_points: np.ndarray | None = None,
    quick_constraint: Mapping[str, Any] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
    planning_target_points: list[dict[str, Any]] | None = None,
    fixed_target_points: list[dict[str, Any]] | None = None,
    planning_relations: list[dict[str, Any]] | None = None,
    head_camera_position_robot_base_m: Sequence[float] | None = None,
    head_camera_quaternion_xyzw: Sequence[float] | None = None,
    plan_motion_epoch: int | None = None,
    registered_source_reconciliation: Mapping[str, Any] | None = None,
    predefined_arm_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    waypoints = list(path["waypoints"])
    if len(waypoints) > MAX_TRAJECTORY_WAYPOINTS:
        raise ValueError("planned trajectory exceeds the signed-path waypoint limit")
    dense_policy_for_record = bool(
        EXECUTION_DENSE_WAYPOINTS
        or len(waypoints) <= int(EXECUTION_DENSE_SHORT_PATH_THRESHOLD)
    )
    # ``names`` remains the controlled/IK point order for legacy callers.
    # Explicit roles retain a separate complete tracker order and freeze every
    # off-hand reference in the signed record in both coordinate and quick mode.
    controlled_names = list(controlled_names or names)
    tracking_names = list(tracking_names or names)
    reference_names = list(reference_names or [])
    point_count = len(controlled_names)
    quick_specs = _quick_constraint_specs(
        quick_constraint=quick_constraint,
        quick_constraints=quick_constraints,
    )
    faceto_requested = any(
        str(spec.get("type") or "").strip().lower().replace("-", "_").replace(" ", "_")
        in {"faceto", "reverse_faceto"}
        for spec in quick_specs
    )
    faceto_camera_pose = _camera_pose_payload(
        head_camera_position_robot_base_m,
        head_camera_quaternion_xyzw,
    )
    if faceto_requested and faceto_camera_pose is None:
        raise ValueError(
            "faceto/reverse_faceto trajectory requires the frozen head-camera pose"
        )
    source_matrix = _strict_point_matrix(
        source_points,
        expected_count=point_count,
        label="move-tracked source points",
    )
    resolved_matrix = _strict_point_matrix(
        endpoint["resolved_points_robot_base_m"],
        expected_count=point_count,
        label="move-tracked endpoint resolved points",
    )
    anchor_matrix = _strict_point_matrix(
        endpoint["anchors_eef_m"],
        expected_count=point_count,
        label="move-tracked endpoint anchors",
    )
    relation_payload = [
        dict(item)
        for item in (
            planning_relations
            if planning_relations is not None
            else args.get("relations", [])
        )
    ]
    if reference_names:
        if reference_points is None:
            raise ValueError("move-tracked trajectory is missing off-hand references")
        reference_matrix = _strict_point_matrix(
            reference_points,
            expected_count=len(reference_names),
            label="move-tracked off-hand reference points",
        )
    else:
        reference_matrix = None
    geometry_report = rigid_geometry_report(
        source_matrix,
        resolved_matrix,
        tolerance_m=float(args["pos_tol"]),
    )
    predefined_names, visual_names = partition_predefined_eef_point_names(
        tracking_names
    )
    eef_waypoints = _eef_waypoints_x5(frozen, arm, waypoints)
    return {
        "tool": "move_tracked_point",
        "source_tool": "move_tracked_point",
        "tool_version": TOOL_VERSION,
        "schema": TRAJECTORY_SCHEMA,
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "robot_profile": ROBOT_PROFILE,
        "arm_dof": ARM_DOF,
        "action_dim": ACTION_DIM,
        "active_arm": arm,
        "episode_id": str(episode_id),
        "motion_epoch": int(
            frozen["motion_epoch"] if plan_motion_epoch is None else plan_motion_epoch
        ),
        "source_image_id": image_id,
        "tracking_binding": {
            "session_id": session_id,
            "image_id": image_id,
            "point_names": controlled_names,
            "point_count": point_count,
            "point_a_name": nearest_name,
            "observation_sequence": int(observation_sequence),
            "frame": "current_robot_base",
            "all_points_same_rigid_object_policy_assertion": (
                not quick_specs and point_count >= 2
            ),
            "two_points_same_rigid_object_policy_assertion": (
                not quick_specs and point_count == 2
            ),
            "controlled_point_names": controlled_names,
            "reference_point_names": reference_names,
            "tracking_point_names": tracking_names,
            "tracking_point_count": len(tracking_names),
            "quick_constraint_mode": bool(quick_specs),
            "predefined_eef_point_names": predefined_names,
            "visual_tracked_point_names": visual_names,
            "predefined_eef_arm": arm if predefined_names else None,
        },
        "start_state": {
            "episode_id": episode_id,
            "motion_epoch": int(frozen["motion_epoch"]),
            "base_rotation": np.asarray(frozen["base_rotation"], dtype=np.float64)
            .astype(float)
            .tolist(),
            "eef_base_wxyz": {
                side: np.asarray(frozen["eef_base_wxyz"][side], dtype=np.float64)
                .astype(float)
                .tolist()
                for side in ("left", "right")
            },
            "arm_qpos": {
                side: frozen["arm_q"][side].tolist() for side in ("left", "right")
            },
            "arm_qvel": {
                side: frozen["arm_qvel"][side].tolist() for side in ("left", "right")
            },
            "gripper_qpos": {
                side: frozen["gripper_q"][side].tolist() for side in ("left", "right")
            },
        },
        "parameters": json_ready(args),
        "target": {
            "requested_points": json_ready(args["points"]),
            "planning_target_points": json_ready(
                planning_target_points
                if planning_target_points is not None
                else args["points"]
            ),
            "fixed_target_points": json_ready(list(fixed_target_points or [])),
            "point_count": point_count,
            "point_names": controlled_names,
            "source_points_robot_base_m": source_matrix.tolist(),
            "resolved_points_robot_base_m": resolved_matrix.tolist(),
            "anchors_eef_m": anchor_matrix.tolist(),
            "eef_position_robot_base_m": np.asarray(
                endpoint["final_eef_position_m"]
            ).tolist(),
            "eef_quaternion_xyzw": np.asarray(
                endpoint["final_eef_quaternion_xyzw"]
            ).tolist(),
            "endpoint_q": np.asarray(endpoint["q_final"]).tolist(),
            "rigid_transform": json_ready(endpoint["rigid_transform"]),
            "constraints": json_ready(endpoint["constraints"]),
            "relations": json_ready(relation_payload),
            "requested_relations": json_ready(args.get("relations", [])),
            "inequalities": json_ready(args.get("inequalities", [])),
            "requested_inequalities": json_ready(args.get("inequalities", [])),
            "quick_constraint": (
                json_ready(dict(quick_constraint))
                if quick_constraint is not None
                else None
            ),
            "quick_constraints": (
                json_ready([dict(item) for item in quick_constraints])
                if quick_constraints is not None
                else None
            ),
            "faceto_camera_pose": json_ready(faceto_camera_pose),
            "on_hand_point_names": controlled_names,
            "off_hand_point_names": reference_names,
            "off_hand_source_points_robot_base_m": (
                None if reference_matrix is None else reference_matrix.tolist()
            ),
            "geometry_rank": int(endpoint["geometry_rank"]),
            "geometry_preflight": json_ready(endpoint["geometry_preflight"]),
            "geometry_report": json_ready(geometry_report),
            "kabsch_seed": json_ready(endpoint.get("kabsch_seed")),
            "predefined_eef_point_names": predefined_names,
            "visual_tracked_point_names": visual_names,
            "predefined_eef_arm": arm if predefined_names else None,
            "predefined_eef_arm_selection": json_ready(
                dict(predefined_arm_selection or {})
            ),
            "registered_on_hand_source_reconciliation": json_ready(
                dict(registered_source_reconciliation or {})
            ),
            "overlap_volume_filter": json_ready(endpoint.get("overlap_volume_filter")),
        },
        "tracking": {
            "point_count": point_count,
            "point_names": controlled_names,
            "tracking_point_names": tracking_names,
            "tracking_point_count": len(tracking_names),
            "controlled_point_names": controlled_names,
            "reference_point_names": reference_names,
            "quick_constraint_mode": bool(quick_specs),
            "all_points_same_rigid_object_policy_assertion": (
                not quick_specs and point_count >= 2
            ),
            "waypoint_tolerance_rad": WAYPOINT_TOLERANCE_RAD,
            "max_tracking_error_rad": STREAM_TRACKING_ERROR_RAD,
            "stable_steps": STABLE_STEPS,
            "execution_mode": "continuous_waypoint_stream",
            "max_joint_step_rad": MAX_JOINT_STEP_RAD,
            "execution_max_speedup_factor": EXECUTION_MAX_SPEEDUP_FACTOR,
            "execution_max_command_delta_rad": EXECUTION_MAX_COMMAND_DELTA_RAD,
            "execution_compression_disabled": dense_policy_for_record,
            "execution_waypoint_policy": (
                "all_signed_waypoints"
                if dense_policy_for_record
                else "bounded_monotone_subsample"
            ),
            "stream_recovery_error_rad": STREAM_RECOVERY_ERROR_RAD,
            "stream_max_recovery_steps": STREAM_MAX_RECOVERY_STEPS,
            "rigid_anchor_tolerance_m": max(
                MAX_ANCHOR_ERROR_M, 4.0 * float(args["pos_tol"])
            ),
            "rigid_pair_distance_tolerance_m": rigid_pair_distance_tolerance(
                float(args["pos_tol"])
            ),
        },
        "waypoints": waypoints,
        "eef_waypoints": eef_waypoints,
        "planner": {
            **{
                key: json_ready(value)
                for key, value in path.items()
                if key != "waypoints"
            },
            "endpoint_solver": str(endpoint["solver"]),
            "endpoint_selected_candidate": json_ready(endpoint["selected_candidate"]),
            "endpoint_feasible_candidate_count": int(
                endpoint["feasible_candidate_count"]
            ),
            "endpoint_path_candidate_rank": int(endpoint.get("path_candidate_rank", 0)),
            "endpoint_path_candidates_tested": int(
                endpoint.get("path_candidates_tested", 1)
            ),
            "endpoint_path_candidate_failures": json_ready(
                endpoint.get("path_candidate_failures", [])
            ),
            "object_truth_used": False,
        },
    }


def _validate_loaded_plan(plan: dict[str, Any], *, session_id: str) -> None:
    expected = {
        "tool": "move_tracked_point",
        "source_tool": "move_tracked_point",
        "tool_version": TOOL_VERSION,
        "schema": TRAJECTORY_SCHEMA,
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "robot_profile": ROBOT_PROFILE,
        "arm_dof": ARM_DOF,
        "action_dim": ACTION_DIM,
        "session_id": session_id,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"move-tracked trajectory {key} does not match contract")
    if str(plan.get("episode_id") or "") != str(
        (plan.get("start_state") or {}).get("episode_id") or ""
    ):
        raise ValueError("move-tracked trajectory episode binding is invalid")
    if isinstance(plan.get("motion_epoch"), bool) or not isinstance(
        plan.get("motion_epoch"), int
    ):
        raise ValueError("move-tracked trajectory motion epoch is invalid")
    arm = str(plan.get("active_arm") or "")
    if arm not in ("left", "right"):
        raise ValueError("move-tracked trajectory active arm is invalid")
    binding = plan.get("tracking_binding")
    target = plan.get("target")
    tracking = plan.get("tracking")
    planner = plan.get("planner")
    if not all(
        isinstance(value, dict) for value in (binding, target, tracking, planner)
    ):
        raise ValueError("move-tracked trajectory metadata is incomplete")

    quick_constraint = target.get("quick_constraint")
    quick_constraints = target.get("quick_constraints")
    quick_specs = _quick_constraint_specs(
        quick_constraint=quick_constraint,
        quick_constraints=quick_constraints,
    )
    quick_mode = bool(quick_specs)
    faceto_requested = any(
        str(spec.get("type") or "").strip().lower().replace("-", "_").replace(" ", "_")
        in {"faceto", "reverse_faceto"}
        for spec in quick_specs
    )
    faceto_camera_position: np.ndarray | None = None
    faceto_camera_quaternion: np.ndarray | None = None
    if faceto_requested:
        raw_camera = target.get("faceto_camera_pose")
        if not isinstance(raw_camera, Mapping):
            raise ValueError(
                "move-tracked faceto trajectory is missing its frozen head-camera pose"
            )
        try:
            faceto_camera_position = np.asarray(
                raw_camera["position_robot_base_m"], dtype=np.float64
            ).reshape(3)
            faceto_camera_quaternion = np.asarray(
                raw_camera["quaternion_xyzw"], dtype=np.float64
            ).reshape(4)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "move-tracked faceto trajectory camera pose is invalid"
            ) from exc
        canonical_camera = _camera_pose_payload(
            faceto_camera_position,
            faceto_camera_quaternion,
        )
        if json_ready(canonical_camera) != json_ready(dict(raw_camera)):
            raise ValueError(
                "move-tracked faceto trajectory camera pose is not canonical"
            )
    raw_names = binding.get("point_names")
    point_count = binding.get("point_count")
    if (
        isinstance(point_count, bool)
        or not isinstance(point_count, int)
        or not 1 <= point_count <= MOVE_TRACKED_POINT_MAX_POINTS
    ):
        raise ValueError(
            "move-tracked trajectory point_count must be in 1.."
            f"{MOVE_TRACKED_POINT_MAX_POINTS}"
        )
    if (
        not isinstance(raw_names, list)
        or len(raw_names) != point_count
        or len(set(raw_names)) != point_count
        or any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in raw_names
        )
    ):
        raise ValueError("move-tracked trajectory point names are invalid")
    names = list(raw_names)
    if binding.get("all_points_same_rigid_object_policy_assertion") != (
        not quick_mode and point_count >= 2
    ):
        raise ValueError("move-tracked trajectory rigid-object assertion is invalid")
    if (
        target.get("point_count") != point_count
        or target.get("point_names") != names
        or tracking.get("point_count") != point_count
        or tracking.get("point_names") != names
    ):
        raise ValueError("move-tracked trajectory point binding does not match")
    if tracking.get("all_points_same_rigid_object_policy_assertion") != (
        not quick_mode and point_count >= 2
    ):
        raise ValueError("move-tracked tracking rigid-object assertion is invalid")

    tracking_names = binding.get("tracking_point_names", names)
    reference_names = binding.get("reference_point_names", [])
    controlled_names = binding.get("controlled_point_names", names)
    if (
        not isinstance(tracking_names, list)
        or not isinstance(reference_names, list)
        or not isinstance(controlled_names, list)
        or controlled_names != names
        or len(set(tracking_names)) != len(tracking_names)
        or any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in tracking_names
        )
        or any(name not in tracking_names for name in names + reference_names)
        or set(names) & set(reference_names)
        or set(names) | set(reference_names) != set(tracking_names)
    ):
        raise ValueError("move-tracked trajectory role binding is invalid")
    if (
        binding.get("tracking_point_count") != len(tracking_names)
        or binding.get("quick_constraint_mode") != quick_mode
        or tracking.get("tracking_point_names") != tracking_names
        or tracking.get("tracking_point_count") != len(tracking_names)
        or tracking.get("controlled_point_names") != names
        or tracking.get("reference_point_names") != reference_names
        or tracking.get("quick_constraint_mode") != quick_mode
    ):
        raise ValueError("move-tracked trajectory tracking roles do not match")

    source = _strict_point_matrix(
        target.get("source_points_robot_base_m"),
        expected_count=point_count,
        label="move-tracked trajectory source points",
    )
    resolved = _strict_point_matrix(
        target.get("resolved_points_robot_base_m"),
        expected_count=point_count,
        label="move-tracked trajectory resolved points",
    )
    anchors = _strict_point_matrix(
        target.get("anchors_eef_m"),
        expected_count=point_count,
        label="move-tracked trajectory anchors",
    )
    endpoint_position = np.asarray(
        target.get("eef_position_robot_base_m"), dtype=np.float64
    ).reshape(3)
    endpoint_quaternion = np.asarray(
        target.get("eef_quaternion_xyzw"), dtype=np.float64
    ).reshape(4)
    if not np.all(np.isfinite(endpoint_position)) or not np.all(
        np.isfinite(endpoint_quaternion)
    ):
        raise ValueError("move-tracked trajectory endpoint pose is invalid")
    resolved_from_pose = (
        endpoint_position[None, :]
        + (quat_to_mat_xyzw(endpoint_quaternion) @ anchors.T).T
    )
    if not np.allclose(resolved, resolved_from_pose, atol=1e-7, rtol=0.0):
        raise ValueError(
            "move-tracked trajectory resolved points do not match endpoint anchors"
        )

    requested_points = target.get("requested_points")
    planning_target_points = target.get("planning_target_points")
    fixed_target_points = target.get("fixed_target_points", [])
    relations = target.get("relations")
    requested_relations = target.get("requested_relations", [])
    requested_inequalities = target.get("requested_inequalities", [])
    if (
        not isinstance(requested_points, list)
        or not isinstance(planning_target_points, list)
        or not isinstance(fixed_target_points, list)
        or not isinstance(relations, list)
        or not isinstance(requested_relations, list)
        or not isinstance(requested_inequalities, list)
    ):
        raise ValueError("move-tracked trajectory target request is incomplete")

    request_payload: dict[str, Any] = {
        "points": requested_points,
        "relations": requested_relations,
        "inequalities": requested_inequalities,
        "execution_mode": str(
            (plan.get("parameters") or {}).get("execution_mode", "exec")
        ),
    }
    if quick_constraint is not None:
        request_payload.update(
            {
                "mode": "quick_constraint",
                "quick_constraint": quick_constraint,
            }
        )
    elif quick_constraints is not None:
        request_payload.update(
            {
                "mode": "quick_constraint",
                "quick_constraints": quick_constraints,
            }
        )
    normalized_request = validate_move_tracked_point_args(request_payload)
    if json_ready(normalized_request["points"]) != json_ready(requested_points):
        raise ValueError("move-tracked trajectory point expressions are not canonical")
    if json_ready(normalized_request["relations"]) != json_ready(requested_relations):
        raise ValueError("move-tracked requested relations are not canonical")
    if json_ready(normalized_request["inequalities"]) != json_ready(
        requested_inequalities
    ):
        raise ValueError("move-tracked requested inequalities are not canonical")
    planning_inequalities = target.get("inequalities", [])
    if json_ready(planning_inequalities) != json_ready(
        normalized_request["inequalities"]
    ):
        raise ValueError("move-tracked planning inequalities are inconsistent")
    if quick_constraint is not None:
        if json_ready(normalized_request["quick_constraint"]) != json_ready(
            quick_constraint
        ):
            raise ValueError(
                "move-tracked trajectory quick constraint is not canonical"
            )
    elif quick_constraints is not None and json_ready(
        normalized_request["quick_constraints"]
    ) != json_ready(quick_constraints):
        raise ValueError("move-tracked trajectory quick constraints are not canonical")

    requested_names = [str(point["name"]) for point in requested_points]
    if requested_names != tracking_names:
        raise ValueError(
            "move-tracked requested point names do not match complete tracking binding"
        )
    roles_by_name = {
        str(point["name"]): str(point.get("role") or "on_hand")
        for point in normalized_request["points"]
    }
    expected_controlled = [
        name for name in requested_names if roles_by_name[name] == "on_hand"
    ]
    expected_reference = [
        name for name in requested_names if roles_by_name[name] == "off_hand"
    ]
    if names != expected_controlled or reference_names != expected_reference:
        raise ValueError("move-tracked requested roles do not match trajectory binding")
    if (
        target.get("on_hand_point_names") != names
        or target.get("off_hand_point_names") != reference_names
    ):
        raise ValueError("move-tracked trajectory role metadata does not match")
    if quick_constraint is not None and (
        names != list(quick_constraint.get("on_hand_points", []))
        or reference_names != list(quick_constraint.get("off_hand_points", []))
    ):
        raise ValueError("move-tracked trajectory quick role order does not match")

    off_hand_raw = target.get("off_hand_source_points_robot_base_m")
    if reference_names:
        reference_points = _strict_point_matrix(
            off_hand_raw,
            expected_count=len(reference_names),
            label="move-tracked trajectory off-hand reference points",
        )
    else:
        if off_hand_raw is not None:
            raise ValueError(
                "move-tracked off-hand source points exist without references"
            )
        reference_points = None

    expected_planning_targets = _planning_targets(normalized_request["points"], names)
    if json_ready(planning_target_points) != json_ready(expected_planning_targets):
        raise ValueError("move-tracked planning target coordinates are inconsistent")
    expected_fixed_targets = _planning_targets(
        normalized_request["points"], reference_names
    )
    if json_ready(fixed_target_points) != json_ready(expected_fixed_targets):
        raise ValueError("move-tracked off-hand target coordinates are inconsistent")

    reference_name_set = set(reference_names)
    for relation in normalized_request["relations"]:
        if relation.get(
            "type"
        ) != "ordered_collinear" and reference_name_set.intersection(
            relation.get("point_names", [])
        ):
            raise ValueError(
                "typed relations may reference only on-hand points; use a "
                "role-based quick preset for on-hand/off-hand geometry"
            )
    expected_relations = _relations_with_quick_constraints(
        normalized_request["relations"],
        quick_specs,
        controlled_names=names,
        controlled_points=source,
        reference_names=reference_names,
        reference_points=reference_points,
        camera_position_robot_base_m=faceto_camera_position,
        camera_quaternion_xyzw=faceto_camera_quaternion,
    )
    if json_ready(relations) != json_ready(expected_relations):
        raise ValueError("move-tracked planning relations are inconsistent")
    endpoint_constraint_report = evaluate_target_constraints(
        resolved,
        expected_planning_targets,
        tolerance_m=float(plan["parameters"]["pos_tol"]),
        relations=expected_relations,
        inequalities=normalized_request["inequalities"],
        orientation_tolerance_deg=float(plan["parameters"]["ori_tol_deg"]),
        fixed_points_robot_base_m=reference_points,
        fixed_target_points=(expected_fixed_targets if reference_names else None),
    )
    if not endpoint_constraint_report.get("inequality_constraints_ok", True):
        raise ValueError(
            "move-tracked signed endpoint violates its strict inequality constraints"
        )
    reported_geometry = target.get("geometry_report")
    recomputed_geometry = rigid_geometry_report(
        source,
        resolved,
        tolerance_m=float(plan["parameters"]["pos_tol"]),
    )
    mismatch = geometry_report_difference(
        reported_geometry,
        recomputed_geometry,
        atol=1e-9,
    )
    if mismatch:
        raise ValueError(f"move-tracked trajectory {mismatch}")
    if target.get("geometry_rank") != recomputed_geometry.get("source_rank"):
        raise ValueError("move-tracked trajectory geometry rank is inconsistent")
    for key in ("joint_limits_checked", "local_fk_finite_checked"):
        if planner.get(key) is not True:
            raise ValueError(f"move-tracked trajectory {key} is missing")
    if planner.get("simulator_collision_checked") is not False:
        raise ValueError("move-tracked trajectory contains forbidden collision truth")
    waypoints = plan.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("move-tracked trajectory has no waypoints")
    if len(waypoints) > MAX_TRAJECTORY_WAYPOINTS:
        raise ValueError("move-tracked trajectory has too many waypoints")
    if tracking.get("execution_mode") != "continuous_waypoint_stream":
        raise ValueError("move-tracked trajectory execution mode is invalid")
    if float(tracking.get("max_joint_step_rad", math.inf)) != MAX_JOINT_STEP_RAD:
        raise ValueError("move-tracked trajectory joint-step contract is invalid")
    previous = np.asarray(plan["start_state"]["arm_qpos"][arm], dtype=np.float64)
    for raw in waypoints:
        waypoint = np.asarray(raw, dtype=np.float64).reshape(-1)
        if waypoint.size != ARM_DOF or not np.all(np.isfinite(waypoint)):
            raise ValueError("move-tracked trajectory waypoint is invalid")
        if np.any(waypoint < X5_JOINT_LOWER) or np.any(waypoint > X5_JOINT_UPPER):
            raise ValueError("move-tracked trajectory waypoint exceeds joint limits")
        if (
            float(np.linalg.norm(waypoint - previous, ord=np.inf))
            > MAX_JOINT_STEP_RAD + 1e-8
        ):
            raise ValueError("move-tracked trajectory waypoint step is too large")
        previous = waypoint

    eef_waypoints = plan.get("eef_waypoints")
    if not isinstance(eef_waypoints, list) or len(eef_waypoints) != len(waypoints):
        raise ValueError("move-tracked trajectory EEF waypoint view is incomplete")
    start_state = plan["start_state"]
    local_state = LocalRobotState(
        arm_dof=ARM_DOF,
        arm_left_q=np.asarray(start_state["arm_qpos"]["left"], dtype=np.float64),
        arm_right_q=np.asarray(start_state["arm_qpos"]["right"], dtype=np.float64),
        eef_left_pose_base_wxyz=np.asarray(
            start_state["eef_base_wxyz"]["left"], dtype=np.float64
        ),
        eef_right_pose_base_wxyz=np.asarray(
            start_state["eef_base_wxyz"]["right"], dtype=np.float64
        ),
    )
    base_rotation = np.asarray(start_state["base_rotation"], dtype=np.float64).reshape(
        3, 3
    )
    for index, (raw_q, raw_eef) in enumerate(zip(waypoints, eef_waypoints)):
        if not isinstance(raw_eef, Mapping):
            raise ValueError("move-tracked trajectory EEF waypoint is invalid")
        q = np.asarray(raw_q, dtype=np.float64).reshape(ARM_DOF)
        planned_q = np.asarray(
            raw_eef.get("planned_joint_target"), dtype=np.float64
        ).reshape(ARM_DOF)
        pose = np.asarray(raw_eef.get("pose"), dtype=np.float64).reshape(7)
        position_base, quaternion_xyzw = eef_pose(local_state, arm, q)
        expected_pose = _base_pose_xyzw_to_env_wxyz(
            position_base, quaternion_xyzw, base_rotation
        )
        if (
            not np.all(np.isfinite(pose))
            or not np.allclose(planned_q, q, atol=1.0e-10, rtol=0.0)
            or not np.allclose(pose[:3], expected_pose[:3], atol=1.0e-8, rtol=0.0)
            or abs(
                float(
                    np.dot(
                        pose[3:] / np.linalg.norm(pose[3:]),
                        expected_pose[3:] / np.linalg.norm(expected_pose[3:]),
                    )
                )
            )
            < 1.0 - 1.0e-10
        ):
            raise ValueError(
                f"move-tracked trajectory EEF waypoint {index + 1} does not "
                "match its signed X5 joint waypoint"
            )


def _plan_preview_result_x5(
    ctx,
    *,
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Render the signed X5 endpoint into its exact frozen head RGB frame."""

    session_id = str(trajectory["session_id"])
    plan_id = str(trajectory["plan_id"])
    arm = str(trajectory["active_arm"])
    target = dict(trajectory["target"])
    frozen_scene = dict(
        (trajectory.get("planner") or {}).get("frozen_plan_capture") or {}
    )
    image_id = str(frozen_scene.get("image_id") or "").strip()
    if not image_id:
        raise ValueError("move-tracked plan has no frozen preview image")
    snapshot = ctx.snapshots.load(session_id, image_id)
    binding_sequence = int(trajectory["tracking_binding"]["observation_sequence"])
    if (
        snapshot.role != "head"
        or snapshot.depth is None
        or int(snapshot.sequence) != int(frozen_scene["observation_sequence"])
        or int(snapshot.sequence) != binding_sequence
        or snapshot.episode_id != str(trajectory["episode_id"])
    ):
        raise ValueError(
            "plan preview RGB-D does not match the frozen planning observation"
        )
    if (
        frozen_scene.get("integrity_digest")
        and str(frozen_scene["integrity_digest"]) != snapshot.integrity_digest
    ):
        raise ValueError("move-tracked frozen preview capture digest changed")

    endpoint_base = {
        "pos": json_ready(target["eef_position_robot_base_m"]),
        "quat": json_ready(target["eef_quaternion_xyzw"]),
    }
    endpoint_env = _base_pose_xyzw_to_env_wxyz(
        target["eef_position_robot_base_m"],
        target["eef_quaternion_xyzw"],
        trajectory["start_state"]["base_rotation"],
    )
    rgb_path = Path(snapshot.paths.get("rgb", snapshot.paths["rgb_array"]))
    overlay_path = rgb_path.with_name(f"{snapshot.image_id}.{plan_id}.gripper.png")
    visualization = render_x5_gripper_overlay(
        rgb=snapshot.rgb,
        depth_linear=snapshot.depth,
        intrinsic_matrix=snapshot.intrinsic_matrix,
        extrinsic_matrix=snapshot.extrinsic_matrix,
        eef_pose={
            "pos": endpoint_env[:3].astype(float).tolist(),
            "quat": endpoint_env[3:].astype(float).tolist(),
        },
        output_path=overlay_path,
    )
    visualization = {
        **dict(visualization),
        "camera_frame": "robodojo_env",
        "eef_pose_frame": "robodojo_env",
        "planned_pose_robot_base": endpoint_base,
    }
    visualization_ok = bool(visualization.get("ok"))
    overlap = target.get("overlap_volume_filter")
    overlap = dict(overlap) if isinstance(overlap, Mapping) else {}
    fallback_applied = bool(overlap.get("fallback_selection"))
    warning = (
        "overlap thresholds exceeded; the minimum-overlap endpoint from the "
        "best constraint-accuracy class was selected; collision-free motion "
        "is not guaranteed"
        if fallback_applied
        else None
    )
    render_path = str(overlay_path) if visualization_ok else None
    result = {
        **_common_result(),
        "ok": True,
        "error": None,
        "failure_stage": None,
        "build": MOVE_TRACKED_POINT_LOCAL_BUILD,
        "execution_mode": "plan",
        "execution": "plan_only_no_robot_motion",
        "session_id": session_id,
        "image_id": image_id,
        "source_image_id": trajectory.get("source_image_id"),
        "plan_id": plan_id,
        "plan_record_path": trajectory.get("plan_record_path"),
        "trajectory_schema": trajectory["schema"],
        "trajectory_schema_version": trajectory["schema_version"],
        "trajectory_digest": trajectory["integrity"]["digest"],
        "trajectory_waypoint_count": len(trajectory["waypoints"]),
        "move_trajectory_schema": trajectory["schema"],
        "move_trajectory_schema_version": trajectory["schema_version"],
        "move_trajectory_digest": trajectory["integrity"]["digest"],
        "move_trajectory_waypoint_count": len(trajectory["waypoints"]),
        "robot_profile": ROBOT_PROFILE,
        "arm": arm,
        "recommended_arm": arm,
        "eef_pose": endpoint_base,
        "eef_pose_frame": "current_robot_base",
        "eef_pose_robodojo_env": {
            "pos": endpoint_env[:3].astype(float).tolist(),
            "quat": endpoint_env[3:].astype(float).tolist(),
        },
        "selected_pose_ik_q": {arm: target["endpoint_q"]},
        "on_hand_point_names": target.get("on_hand_point_names"),
        "off_hand_point_names": target.get("off_hand_point_names"),
        "predefined_eef_point_names": target.get("predefined_eef_point_names", []),
        "quick_constraint": target.get("quick_constraint"),
        "quick_constraints": target.get("quick_constraints"),
        "inequalities": target.get("inequalities", []),
        "overlap_volume_filter": overlap or None,
        "overlap_vol_cm3": overlap.get("overlap_vol_cm3"),
        "inflated_overlap_vol_cm3": overlap.get("inflated_overlap_vol_cm3"),
        "overlap_fallback_applied": fallback_applied,
        "overlap_thresholds_satisfied": bool(overlap.get("ok"))
        and not fallback_applied,
        "warnings": [warning] if warning else [],
        "resolved_variables": (target.get("constraints") or {}).get(
            "resolved_variables", {}
        ),
        "target_constraints": target.get("constraints"),
        "resolved_target_points_robot_base_m": target.get(
            "resolved_points_robot_base_m"
        ),
        "endpoint_solver": trajectory["planner"].get("endpoint_solver"),
        "planner": trajectory["planner"],
        "frozen_plan_capture": frozen_scene,
        "track_object_distance_binding": frozen_scene.get(
            "track_object_distance_binding"
        ),
        "gripper_visualization": visualization,
        "visualization": visualization,
        "render_image": file_data_url(overlay_path) if visualization_ok else None,
        "render_image_path": render_path,
        "rgb_overlay_path": render_path,
        "marked_image_url": render_path,
        "plan_preview_overlay": (
            "red_gripper_on_frozen_head" if visualization_ok else None
        ),
        "debug_images": [render_path] if render_path else [],
        "segmentation_used": False,
        "object_identity_used": False,
        "simulator_mesh_used": False,
        "simulator_fk_ik_used": False,
        "direct_simulator_mutation": False,
    }
    return json_ready(result)


def _monitor(
    ctx,
    *,
    manager=None,
    names: list[str],
    episode_id: str,
    session_id: str,
    image_id: str,
    anchors: np.ndarray,
    arm: str,
    tolerance_m: float,
    geometry_tolerance_m: float,
    tracking_names: list[str] | None = None,
    reference_names: list[str] | None = None,
    reference_points: np.ndarray | None = None,
    quick_constraint: Mapping[str, Any] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
    position_tolerance_m: float | None = None,
    orientation_tolerance_deg: float = 5.0,
    camera_position_robot_base_m: Sequence[float] | None = None,
    camera_quaternion_xyzw: Sequence[float] | None = None,
    allow_visual_outlier_suppression: bool = True,
) -> tuple[str | None, dict[str, Any]]:
    sequence = _sequence(ctx)
    controlled_names = list(names)
    all_names = list(tracking_names or controlled_names)
    reference_names = list(reference_names or [])
    live = _live(ctx, all_names, episode_id, manager=manager)
    report: dict[str, Any] = {
        "observation_sequence": sequence,
        "tracker_observation_sequence": live.get("observation_sequence"),
        "anchor_tolerance_m": tolerance_m,
    }
    if not bool(live.get("ok")):
        report["tracker_reason"] = live.get("reason")
        return str(live.get("reason") or "tracked_point_unavailable"), report
    if int(live.get("observation_sequence", -1)) != sequence:
        return "tracker_observation_mismatch", report
    if live.get("session_id") != session_id or live.get("image_id") != image_id:
        return "tracker_binding_replaced", report
    measured = np.asarray(
        [
            live["entries"][name]["xyz_in_robot_base_coord_m"]
            for name in controlled_names
        ],
        dtype=np.float64,
    ).reshape(len(controlled_names), 3)
    position, quaternion = _eef_base_xyzw(ctx, arm)
    predicted = position[None, :] + (quat_to_mat_xyzw(quaternion) @ anchors.T).T
    errors = np.linalg.norm(measured - predicted, axis=1)
    geometry = evaluate_rigid_geometry_consistency(
        measured,
        predicted,
        tolerance_m=geometry_tolerance_m,
    )
    report.update(
        {
            "max_anchor_error_m": float(np.max(errors)),
            "point_anchor_error_m": {
                name: float(errors[index])
                for index, name in enumerate(controlled_names)
            },
            "rigid_geometry": json_ready(geometry),
        }
    )
    reference_measured = None
    reference_report: dict[str, Any] = {"ok": True, "applicable": False}
    if reference_names and reference_points is None:
        return "off_hand_points_invalid", {
            **report,
            "off_hand_reference": {
                "ok": False,
                "applicable": True,
                "reason": "missing_frozen_reference_points",
            },
        }
    try:
        if reference_names:
            reference_measured = np.asarray(
                [
                    live["entries"][name]["xyz_in_robot_base_coord_m"]
                    for name in reference_names
                ],
                dtype=np.float64,
            ).reshape(len(reference_names), 3)
            reference_matrix = np.asarray(reference_points, dtype=np.float64).reshape(
                len(reference_names), 3
            )
            reference_limit = max(
                0.020,
                2.0
                * float(
                    position_tolerance_m
                    if position_tolerance_m is not None
                    else geometry_tolerance_m
                ),
            )
            reference_report = _off_hand_reference_consensus(
                reference_names,
                reference_measured,
                reference_matrix,
                tolerance_m=reference_limit,
                allow_visual_outlier_suppression=allow_visual_outlier_suppression,
            )
            if bool(reference_report.get("tracking_outlier_suppressed")):
                reference_measured = reference_matrix.copy()
        report["off_hand_reference"] = reference_report
        if not bool(reference_report["ok"]):
            return "off_hand_points_moved", report
        quick_specs = _quick_constraint_specs(
            quick_constraint=quick_constraint,
            quick_constraints=quick_constraints,
        )
        if quick_specs:
            report["quick_constraint"] = json_ready(
                _evaluate_quick_constraints(
                    measured,
                    reference_measured,
                    quick_specs,
                    controlled_names=controlled_names,
                    reference_names=reference_names,
                    position_tolerance_m=float(
                        position_tolerance_m
                        if position_tolerance_m is not None
                        else geometry_tolerance_m
                    ),
                    orientation_tolerance_deg=orientation_tolerance_deg,
                    camera_position_robot_base_m=(camera_position_robot_base_m),
                    camera_quaternion_xyzw=camera_quaternion_xyzw,
                )
            )
            # Quick presets describe terminal geometry. Intermediate SLERP or
            # joint-space samples may violate it while rotating toward the
            # target, so this report is diagnostic until final validation.
    except (KeyError, TypeError, ValueError) as exc:
        report["off_hand_reference"] = {
            "ok": False,
            "applicable": bool(reference_names),
            "reason": "off_hand_points_invalid",
            "error": str(exc),
        }
        return "off_hand_points_invalid", report
    if not bool(geometry.get("ok")):
        return "tracked_rigid_geometry_changed", report
    if float(np.max(errors)) > tolerance_m:
        return "tracked_points_not_rigid_with_selected_eef", report
    return None, report


def _execution_bridge(
    start_q: Any,
    signed_waypoints: Sequence[Sequence[float]],
    *,
    max_step_rad: float = MAX_JOINT_STEP_RAD,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Expand a scheduled X5 path exactly as it will be sent to the evaluator."""

    if not math.isfinite(float(max_step_rad)) or float(max_step_rad) <= 0.0:
        raise ValueError("execution bridge max_step_rad must be positive and finite")
    current = np.asarray(start_q, dtype=np.float64).reshape(ARM_DOF)
    if not np.all(np.isfinite(current)):
        raise ValueError("execution handoff joints must be finite")
    lower_overshoot = np.maximum(X5_JOINT_LOWER - current, 0.0)
    upper_overshoot = np.maximum(current - X5_JOINT_UPPER, 0.0)
    overshoot = float(max(np.max(lower_overshoot), np.max(upper_overshoot)))
    if overshoot > float(max_step_rad):
        raise ValueError("execution handoff exceeds X5 joint limits")

    execution: list[list[float]] = []
    bridge_count = 0
    if overshoot > 0.0:
        current = np.clip(current, X5_JOINT_LOWER, X5_JOINT_UPPER)
        execution.append(current.astype(float).tolist())
        bridge_count += 1

    for waypoint_index, raw_target in enumerate(signed_waypoints):
        target = np.asarray(raw_target, dtype=np.float64).reshape(ARM_DOF)
        if not np.all(np.isfinite(target)):
            raise ValueError(
                f"execution waypoint {waypoint_index + 1} contains non-finite joints"
            )
        if np.any(target < X5_JOINT_LOWER) or np.any(target > X5_JOINT_UPPER):
            raise ValueError(
                f"execution waypoint {waypoint_index + 1} exceeds X5 joint limits"
            )
        delta = target - current
        max_delta = float(np.linalg.norm(delta, ord=np.inf))
        effective_delta = max(
            0.0,
            max_delta - EXECUTION_STEP_NUMERIC_TOLERANCE_RAD,
        )
        segment_count = max(
            1,
            int(math.ceil(effective_delta / float(max_step_rad))),
        )
        for segment_index in range(1, segment_count + 1):
            point = current + (float(segment_index) / float(segment_count)) * delta
            if np.any(point < X5_JOINT_LOWER) or np.any(point > X5_JOINT_UPPER):
                raise ValueError("execution bridge waypoint exceeds X5 joint limits")
            execution.append(point.astype(float).tolist())
        if segment_count > 1:
            bridge_count += segment_count - 1
        current = target.copy()

    if not execution:
        raise ValueError("scheduled trajectory has no execution waypoints")
    return execution, {
        "ok": True,
        "source": "post_handoff_measured_qpos",
        "joint_limit_overshoot_normalized_rad": overshoot,
        "bridge_waypoint_count": int(bridge_count),
        "execution_waypoint_count": len(execution),
        "execution_max_step_rad": float(max_step_rad),
        "execution_step_numeric_tolerance_rad": float(
            EXECUTION_STEP_NUMERIC_TOLERANCE_RAD
        ),
        "signed_plan_rewritten": False,
    }


def _validate_execution_transitions(
    state: LocalRobotState,
    arm: str,
    start_q: Any,
    waypoints: Sequence[Sequence[float]],
) -> dict[str, Any]:
    previous = np.asarray(start_q, dtype=np.float64).reshape(ARM_DOF)
    checked_samples = 0
    maximum_step = 0.0
    for raw in waypoints:
        target = np.asarray(raw, dtype=np.float64).reshape(ARM_DOF)
        if not np.all(np.isfinite(target)):
            raise ValueError("scheduled execution waypoint is non-finite")
        if np.any(target < X5_JOINT_LOWER) or np.any(target > X5_JOINT_UPPER):
            raise ValueError("scheduled execution waypoint exceeds X5 joint limits")
        delta = target - previous
        step = float(np.linalg.norm(delta, ord=np.inf))
        maximum_step = max(maximum_step, step)
        sample_count = max(1, int(math.ceil(step / MAX_JOINT_STEP_RAD)))
        for index in range(1, sample_count + 1):
            sample = previous + (index / sample_count) * delta
            position, quaternion = eef_pose(state, arm, sample)
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
                raise ValueError("scheduled transition has invalid local X5 FK")
            checked_samples += 1
        previous = target
    return {
        "ok": True,
        "local_fk_transition_checked": True,
        "transition_sample_count": checked_samples,
        "actual_max_selected_step_rad": maximum_step,
    }


def _normalize_execution_start(
    start_q: Any,
    *,
    max_step_rad: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    start = np.asarray(start_q, dtype=np.float64).reshape(ARM_DOF)
    if not np.all(np.isfinite(start)):
        raise ValueError("execution scheduling start qpos contains non-finite joints")
    normalized = np.clip(start, X5_JOINT_LOWER, X5_JOINT_UPPER)
    corrections = []
    for joint_index, (observed, selected) in enumerate(zip(start, normalized)):
        correction = float(selected - observed)
        if correction == 0.0:
            continue
        corrections.append(
            {
                "joint_index": int(joint_index),
                "observed_rad": float(observed),
                "normalized_rad": float(selected),
                "correction_rad": correction,
            }
        )
    maximum_correction = max(
        (abs(float(item["correction_rad"])) for item in corrections),
        default=0.0,
    )
    if maximum_correction > float(max_step_rad):
        raise ValueError(
            "execution scheduling start qpos exceeds the recoverable X5 "
            "joint-limit residual"
        )
    return normalized, corrections


def _prepare_execution_schedule(
    *,
    trajectory: dict[str, Any],
    state: LocalRobotState,
    arm: str,
    start_q: Sequence[float],
    max_steps: int,
    available_time_s: float,
    observation_dt_s: float | None,
    visual_tracked_point_count: int | None = None,
    preferred_max_steps: int | None = EXECUTION_PREFERRED_MAX_STEPS,
    requested_public_max_steps: int | None = None,
) -> dict[str, Any]:
    """Build and bridge-preflight the execution view of a signed X5 path.

    Planning remains immutable.  The bounded policy selects only a monotone
    subsequence of already signed waypoints, retaining the endpoint and
    checking every skipped span with the joint corridor and local FK gates.
    Older trajectories which explicitly request dense replay are still sent
    one signed waypoint at a time.
    """

    tracking = trajectory.get("tracking")
    if not isinstance(tracking, dict):
        raise ValueError("move-tracked trajectory tracking metadata is missing")
    dense_policy = _trajectory_uses_dense_execution(trajectory)
    base_step = float(tracking.get("max_joint_step_rad", MAX_JOINT_STEP_RAD))
    if not math.isfinite(base_step) or base_step <= 0.0:
        raise ValueError("move-tracked execution max joint step is invalid")
    scheduled_start_q, start_limit_recovery = _normalize_execution_start(
        start_q,
        max_step_rad=base_step,
    )
    start_bridge_reserve = 1 if start_limit_recovery else 0
    # Compression is defined over the signed plan's own start state.  The
    # measured handoff may be stale or may follow an action on the other arm;
    # reserve its bounded entry actions separately instead of letting a far
    # measured qpos force the scheduler to choose a large in-path chord.
    signed_path_start_q = scheduled_start_q.copy()
    handoff_bridge_reserve = 0
    if not dense_policy:
        try:
            start_state = trajectory.get("start_state")
            start_arm_qpos = (
                start_state.get("arm_qpos")
                if isinstance(start_state, Mapping)
                else None
            )
            signed_path_start_q = np.asarray(
                start_arm_qpos.get(arm)
                if isinstance(start_arm_qpos, Mapping)
                else None,
                dtype=np.float64,
            ).reshape(ARM_DOF)
        except (AttributeError, TypeError, ValueError):
            return {
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "bounded execution is missing its signed arm start state",
            }
        if not np.all(np.isfinite(signed_path_start_q)):
            return {
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "bounded execution signed arm start state is non-finite",
            }
        if np.any(signed_path_start_q < X5_JOINT_LOWER) or np.any(
            signed_path_start_q > X5_JOINT_UPPER
        ):
            return {
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "bounded execution signed arm start state exceeds limits",
            }
        handoff_delta = float(
            np.linalg.norm(scheduled_start_q - signed_path_start_q, ord=np.inf)
        )
        handoff_bridge_reserve = max(
            0,
            int(
                math.ceil(
                    max(
                        0.0,
                        handoff_delta - EXECUTION_STEP_NUMERIC_TOLERANCE_RAD,
                    )
                    / EXECUTION_MAX_COMMAND_DELTA_RAD
                )
            ),
        )
    scheduled_max_steps = int(max_steps) - (
        start_bridge_reserve + handoff_bridge_reserve
    )
    scheduled_available_time_s = float(available_time_s)
    if observation_dt_s is not None:
        scheduled_available_time_s -= float(
            (start_bridge_reserve + handoff_bridge_reserve) * float(observation_dt_s)
        )
    if scheduled_max_steps < 1 or scheduled_available_time_s <= 0.0:
        return {
            "ok": False,
            "reason": "execution_budget_infeasible",
            "error": (
                "remaining evaluator budget cannot accommodate the observed "
                "joint-limit recovery and one trajectory action"
            ),
            "execution_start_joint_limit_recovery": json_ready(start_limit_recovery),
            "execution_bridge_reserved_waypoints": start_bridge_reserve,
            "execution_handoff_reserved_waypoints": handoff_bridge_reserve,
            "budget_steps": max(0, int(max_steps)),
        }

    # ``preferred_max_steps`` is retained in the private signature because old
    # exec-plan callers still pass it.  It must not reactivate compression in
    # an explicitly dense legacy trajectory.
    if dense_policy:
        preferred_max_steps = None

    # Replay is governed by the caller's ``max_steps`` and by the wall-clock
    # timeout in the action loop.  There is no separate step ceiling.  A stale
    # handoff gap can
    # look like a 5 s observation period and collapse that into a handful of
    # steps; the 0.05 rad bridge then no longer fits and the tool aborts
    # before moving.  Do not derive a second step budget from that cadence.
    # The handoff itself is still bridged and counted below.
    scheduler_start_q = None if dense_policy else signed_path_start_q.tolist()
    scheduler_available_time_s = None
    scheduler_observation_dt_s = None
    scheduler_safety_time_steps = 0
    schedule_kwargs = {
        "start_q": scheduler_start_q,
        "max_steps": scheduled_max_steps,
        "available_time_s": scheduler_available_time_s,
        "observation_dt_s": scheduler_observation_dt_s,
        "max_joint_step_rad": base_step,
        "preferred_max_steps": preferred_max_steps,
        "max_joint_corridor_deviation_rad": (
            EXECUTION_MAX_JOINT_CORRIDOR_DEVIATION_RAD
        ),
        "safety_time_steps": scheduler_safety_time_steps,
    }
    relaxed_execution = False
    visual_speedup_cap = 1.0
    schedule_attempts: list[dict[str, Any]] = []
    if dense_policy:
        # ``EXECUTION_DENSE_WAYPOINTS`` is currently the safe default until
        # the bounded scheduler can match dense-replay step counts on
        # trajectories whose knot spacing equals ``MAX_JOINT_STEP_RAD``.
        speedup_candidates: list[float | None] = [1.0]
    else:
        visual_speedup_cap = (
            NPOINT_VISUAL_MAX_SPEEDUP_FACTOR
            if visual_tracked_point_count is not None
            and int(visual_tracked_point_count) >= NPOINT_VISUAL_MIN_POINTS
            else EXECUTION_MAX_SPEEDUP_FACTOR
        )
        # Start with the visual-observation cap, then try the full physical
        # cap only when that cannot fit the hard action budget.  A successful
        # schedule is kept at the first cap that fits; this preserves the
        # lower-impulse N-point behavior while still handling long paths.
        speedup_candidates = [float(visual_speedup_cap)]
        for candidate in (
            float(EXECUTION_MAX_SPEEDUP_FACTOR),
            float(EXECUTION_SPEEDUP_FALLBACK_FACTOR),
        ):
            if candidate not in speedup_candidates:
                speedup_candidates.append(candidate)
        # Match the official bounded scheduler's last-resort pass.  A
        # path-derived cap is still only a candidate: the selected states are
        # rechecked below, the physical bridge bounds each command to the
        # signed X5 step, and bridge expansion must fit the same hard budget
        # before success.
        speedup_candidates.append(None)

    report: dict[str, Any] = {}
    for candidate in speedup_candidates:
        candidate_report = schedule_execution_waypoints(
            trajectory.get("waypoints", []),
            max_speedup_factor=candidate,
            preserve_all_waypoints=bool(dense_policy),
            **schedule_kwargs,
        )
        candidate_report["speedup_attempt"] = (
            None if candidate is None else float(candidate)
        )
        schedule_attempts.append(candidate_report)
        if not bool(candidate_report.get("ok")):
            # A later cap can recover an otherwise safe path which did not fit
            # the remaining action/time budget.
            continue
        report = candidate_report
        # The preferred count is soft.  Once a candidate is within the hard
        # action/time budget, keep that safer cap even if the path is too
        # curved to hit the preferred count.  Escalating merely to shave a few
        # more actions would turn a bounded command into a high-impulse jump.
        break
    if not report:
        report = (
            schedule_attempts[-1]
            if schedule_attempts
            else {
                "ok": False,
                "reason": "execution_budget_infeasible",
                "error": "execution scheduler produced no attempt",
            }
        )
    relaxed_execution = bool(report.get("ok") and report.get("speedup_attempt") is None)
    if len(schedule_attempts) > 1:
        report["speedup_attempts"] = [
            {
                "speedup_factor": item.get("speedup_attempt"),
                "ok": bool(item.get("ok")),
                "reason": item.get("reason"),
                "selected_waypoint_count": item.get("selected_waypoint_count"),
                "required_waypoints_at_max_speedup": item.get(
                    "required_waypoints_at_max_speedup"
                ),
            }
            for item in schedule_attempts
        ]
    if not bool(report.get("ok")):
        report["visual_tracked_point_count"] = visual_tracked_point_count
        report["visual_speedup_cap"] = float(visual_speedup_cap)
        return report

    # A dense revision is not allowed to degrade into the old sparse replay
    # merely because the scheduler found a shorter subsequence.  Returning a
    # pre-action failure is safer and makes the policy observable to callers.
    if dense_policy and int(report.get("dropped_waypoint_count", 0)) > 0:
        return {
            **report,
            "ok": False,
            "reason": "execution_dense_waypoint_policy_violation",
            "error": (
                "dense execution would drop signed planner waypoints; "
                "increase the explicit max_steps budget instead"
            ),
            "execution_compression_disabled": True,
            "execution_waypoint_policy": "all_signed_waypoints",
            "visual_tracked_point_count": visual_tracked_point_count,
            "visual_speedup_cap": float(visual_speedup_cap),
        }

    signed_waypoints = trajectory.get("waypoints")
    selected_indices = report.get("selected_indices")
    selected_waypoints = report.get("waypoints")
    schedule_drops_signed_samples = int(report.get("dropped_waypoint_count", 0)) > 0
    if relaxed_execution or schedule_drops_signed_samples:
        planner = trajectory.get("planner")
        if not isinstance(planner, dict):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "compressed execution requires signed planner metadata",
            }
        if (
            planner.get("same_branch_checked") is not True
            or planner.get("monotonic_progress_checked") is not True
            or planner.get("internal_stop_count") != 0
        ):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": (
                    "compressed execution requires a signed same-branch, "
                    "monotone trajectory"
                ),
            }
        endpoint_index_valid = False
        if isinstance(selected_indices, list) and selected_indices:
            try:
                endpoint_index_valid = bool(
                    isinstance(signed_waypoints, list)
                    and int(selected_indices[-1]) == len(signed_waypoints) - 1
                )
            except (TypeError, ValueError):
                endpoint_index_valid = False
        if (
            not isinstance(signed_waypoints, list)
            or not signed_waypoints
            or not isinstance(selected_indices, list)
            or not isinstance(selected_waypoints, list)
            or len(selected_indices) != len(selected_waypoints)
            or not selected_indices
            or not endpoint_index_valid
        ):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "compressed execution must retain the signed endpoint",
            }
        previous_index = -1
        # Validate skipped signed spans from the signed plan start.  A measured
        # handoff that differs from that capture is a separate bridge and must
        # not make an otherwise valid immutable span look discontinuous.
        previous_q = (
            scheduled_start_q.copy() if dense_policy else signed_path_start_q.copy()
        )
        transition_samples = 0
        transition_max_step = 0.0
        for selected_index, raw_q in zip(selected_indices, selected_waypoints):
            try:
                source_index = int(selected_index)
            except (TypeError, ValueError):
                return {
                    **report,
                    "ok": False,
                    "reason": "execution_schedule_invalid",
                    "error": "relaxed execution contains a non-integer source index",
                }
            if (
                source_index < 0
                or source_index >= len(signed_waypoints)
                or source_index <= previous_index
            ):
                return {
                    **report,
                    "ok": False,
                    "reason": "execution_schedule_invalid",
                    "error": "relaxed execution source indices are not increasing",
                }
            target_q = np.asarray(raw_q, dtype=np.float64).reshape(ARM_DOF)
            if not np.all(np.isfinite(target_q)):
                return {
                    **report,
                    "ok": False,
                    "reason": "execution_schedule_invalid",
                    "error": "relaxed execution transition is non-finite",
                }
            transition_step = float(np.linalg.norm(target_q - previous_q, ord=np.inf))
            transition_max_step = max(transition_max_step, transition_step)
            sample_count = max(
                1,
                min(32, int(math.ceil(transition_step / max(base_step, 1.0e-6)))),
            )
            for sample_index in range(1, sample_count + 1):
                fraction = float(sample_index) / float(sample_count)
                sample = previous_q + fraction * (target_q - previous_q)
                if np.any(sample < X5_JOINT_LOWER) or np.any(sample > X5_JOINT_UPPER):
                    return {
                        **report,
                        "ok": False,
                        "reason": "execution_schedule_invalid",
                        "error": "relaxed execution transition exceeds X5 limits",
                    }
                position, quaternion = eef_pose(state, arm, sample)
                if not np.all(np.isfinite(position)) or not np.all(
                    np.isfinite(quaternion)
                ):
                    return {
                        **report,
                        "ok": False,
                        "reason": "execution_schedule_invalid",
                        "error": "relaxed execution local FK transition is non-finite",
                    }
                transition_samples += 1
            previous_q = target_q.copy()
            previous_index = source_index
        report.update(
            {
                "same_branch_inherited_from_signed_path": True,
                "transition_local_fk_finite_checked": True,
                "transition_local_fk_sample_count": transition_samples,
                "transition_max_joint_step_rad": transition_max_step,
                "execution_path_validation": (
                    "signed_same_branch_with_local_fk_transition_sampling"
                ),
            }
        )

    checked: list[list[float]] = []
    # ``maximum_step`` describes the signed execution path, not the measured
    # handoff.  The latter is separately bridged and budgeted above.
    previous = (
        np.asarray(start_q, dtype=np.float64).reshape(ARM_DOF)
        if dense_policy
        else signed_path_start_q.copy()
    )
    maximum_step = 0.0
    for index, raw_q in enumerate(report.get("waypoints", [])):
        q = np.asarray(raw_q, dtype=np.float64).reshape(ARM_DOF)
        if not np.all(np.isfinite(q)):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "execution schedule contains a non-finite waypoint",
            }
        if np.any(q < X5_JOINT_LOWER) or np.any(q > X5_JOINT_UPPER):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "execution schedule waypoint exceeds X5 joint limits",
            }
        step = float(np.linalg.norm(q - previous, ord=np.inf))
        if not math.isfinite(step):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "execution schedule contains a non-finite joint step",
            }
        maximum_step = max(maximum_step, step)
        position, quaternion = eef_pose(state, arm, q)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            return {
                **report,
                "ok": False,
                "reason": "execution_schedule_invalid",
                "error": "execution schedule local X5 FK returned a non-finite pose",
            }
        checked.append(q.astype(float).tolist())
        previous = q
    if not checked:
        return {
            **report,
            "ok": False,
            "reason": "execution_schedule_invalid",
            "error": "execution schedule contains no waypoints",
        }

    execution_max_step = max(
        float(report.get("max_joint_step_rad", base_step)),
        float(report.get("actual_max_joint_step_rad", 0.0)),
        maximum_step,
    )
    if not dense_policy:
        execution_max_step = min(execution_max_step, EXECUTION_MAX_COMMAND_DELTA_RAD)
    try:
        predicted_execution, predicted_bridge = _execution_bridge(
            start_q,
            checked,
            max_step_rad=execution_max_step,
        )
    except (TypeError, ValueError) as exc:
        return {
            **report,
            "ok": False,
            "reason": "execution_schedule_invalid",
            "error": f"execution bridge preflight failed: {exc}",
        }
    selected_budget = int(report.get("budget_steps", scheduled_max_steps))
    total_budget = selected_budget + start_bridge_reserve + handoff_bridge_reserve
    predicted_count = len(predicted_execution)
    bridge_count = int(predicted_bridge["bridge_waypoint_count"])
    shared_fields = {
        "execution_start_joint_limit_recovery": json_ready(start_limit_recovery),
        "execution_bridge_reserved_waypoints": start_bridge_reserve,
        "execution_handoff_reserved_waypoints": handoff_bridge_reserve,
        "execution_handoff_delta_rad": float(
            np.linalg.norm(scheduled_start_q - signed_path_start_q, ord=np.inf)
        ),
        "predicted_execution_bridge_waypoint_count": bridge_count,
        "predicted_execution_waypoint_count": predicted_count,
        "execution_budget_steps_total": total_budget,
        "execution_budget_includes_bridge": True,
        "execution_max_joint_step_rad": execution_max_step,
    }
    if predicted_count > total_budget or predicted_count > int(max_steps):
        return {
            **report,
            "ok": False,
            "reason": "execution_budget_infeasible",
            "error": (
                "bridge-expanded execution trajectory exceeds the remaining "
                "action/time budget"
            ),
            **shared_fields,
        }
    return {
        **report,
        "waypoints": checked,
        "actual_max_joint_step_rad": maximum_step,
        "execution_step_numeric_tolerance_rad": (EXECUTION_STEP_NUMERIC_TOLERANCE_RAD),
        "local_fk_finite_checked": True,
        "visual_tracked_point_count": visual_tracked_point_count,
        "visual_speedup_cap": float(visual_speedup_cap),
        "execution_compression_disabled": bool(dense_policy),
        "execution_waypoint_policy": (
            "all_signed_waypoints" if dense_policy else "bounded_monotone_subsample"
        ),
        **shared_fields,
    }


def _execution_cadence_estimate(
    *,
    start_validation: Mapping[str, Any] | None,
    preplanning_source_sync: Mapping[str, Any] | None,
    planner: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Estimate a conservative evaluator period from this request only."""

    samples: list[dict[str, Any]] = []

    def add_sample(raw_value: Any, source: str) -> None:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value) or value < 1.0e-4 or value > 5.0:
            return
        samples.append({"source": str(source), "observation_dt_s": value})

    def add_validation(
        raw_report: Mapping[str, Any] | None,
        source_prefix: str,
    ) -> None:
        if not isinstance(raw_report, Mapping):
            return
        raw_samples = raw_report.get("handoff_observation_dt_samples_s")
        if isinstance(raw_samples, Sequence) and not isinstance(
            raw_samples, (str, bytes, bytearray)
        ):
            for index, value in enumerate(raw_samples):
                add_sample(value, f"{source_prefix}.handoff_window[{index}]")
        add_sample(
            raw_report.get("last_handoff_observation_dt_s"),
            f"{source_prefix}.last_handoff",
        )
        assessment = raw_report.get("handoff_assessment")
        if isinstance(assessment, Mapping):
            add_sample(
                assessment.get("observation_dt_s"),
                f"{source_prefix}.handoff_assessment",
            )

    add_validation(preplanning_source_sync, "preplanning_source_sync")
    add_validation(start_validation, "execution_start_validation")
    # CPU planning time is not an evaluator observation period. Background
    # planner polls deliberately emit no action and therefore cannot be used as
    # a cadence sample. Only measured, consecutive observation timestamps above
    # contribute to the execution budget.

    if not samples:
        return {
            "ok": False,
            "reason": "no_valid_observation_period_samples",
            "observation_dt_s": None,
            "max_observed_observation_dt_s": None,
            "safety_factor": float(EXECUTION_CADENCE_SAFETY_FACTOR),
            "samples": [],
        }
    max_observed = max(float(item["observation_dt_s"]) for item in samples)
    conservative = min(
        5.0,
        max_observed * float(EXECUTION_CADENCE_SAFETY_FACTOR),
    )
    return {
        "ok": True,
        "reason": "slowest_request_local_period_with_jitter_reserve",
        "observation_dt_s": float(conservative),
        "max_observed_observation_dt_s": float(max_observed),
        "safety_factor": float(EXECUTION_CADENCE_SAFETY_FACTOR),
        "samples": json_ready(samples),
    }


def _temporary_tracking_failure(reason: str | None) -> bool:
    normalized = str(reason or "").strip()
    return normalized in {
        "tracked_point_unavailable",
        "tracked_point_invalid",
        "tracker_observation_mismatch",
        "no_synchronized_frame",
        "point_unobserved",
        "stale_tracking_measurement",
    }


def _wait_for_stable_start_x5(
    ctx,
    *,
    manager=None,
    frozen: dict[str, Any],
    names: list[str],
    controlled_names: list[str],
    reference_names: list[str],
    source_points: np.ndarray,
    reference_points: np.ndarray | None,
    episode_id: str,
    session_id: str,
    image_id: str,
    arm: str,
    pos_tol_m: float,
    planned_anchors: np.ndarray,
    anchor_tolerance_m: float,
    geometry_tolerance_m: float,
    eef_prior_activation: Mapping[str, Any],
    deadline_monotonic: float,
):
    """Wait for a synchronized, stationary X5 handoff observation."""

    stable_trace: list[dict[str, Any]] = []
    stable_count = 0
    previous_positions: np.ndarray | None = None
    previous_observation_timestamp: float | None = None
    last_handoff_observation_dt_s: float | None = None
    handoff_observation_dt_samples_s: list[float] = []
    reference_limit = max(0.020, 2.0 * float(pos_tol_m))
    prior_active = bool(
        eef_prior_activation.get("active") or eef_prior_activation.get("applicable")
    )
    minimum_tracker_sequence = (
        int(eef_prior_activation.get("activation_observation_sequence", -1))
        if prior_active
        else None
    )
    source_drift_limit = (
        max(0.012, float(pos_tol_m))
        if prior_active
        else max(START_SOURCE_DRIFT_FLOOR_M, 2.0 * float(pos_tol_m))
    )
    off_hand_validation: dict[str, Any] = {
        "ok": True,
        "applicable": False,
    }
    reason = "start_state_not_stable"
    last_source_drift = float("inf")
    last_source_point_drift: dict[str, float] = {}
    last_live_points: np.ndarray | None = None
    last_anchor_error = float("inf")
    last_anchor_point_error: dict[str, float] = {}
    last_geometry: dict[str, Any] = {
        "ok": not prior_active,
        "applicable": prior_active,
    }

    for settle_step in range(START_MAX_SETTLE_STEPS + 1):
        ctx.raise_if_cancelled("move_tracked_point start-state validation")
        if time.monotonic() >= float(deadline_monotonic):
            reason = "timeout"
            break
        if str(ctx.world.episode_id()) != str(episode_id):
            reason = "episode_changed_during_planning"
            break

        current_q = {
            side: np.asarray(ctx.world.arm_qpos_list(side), dtype=np.float64)
            for side in ("left", "right")
        }
        current_gripper = {
            side: np.asarray(ctx.world.gripper_qpos_list(side), dtype=np.float64)
            for side in ("left", "right")
        }
        packed = np.concatenate([current_q["left"], current_q["right"]])
        status = ctx.world._official_adapter.status()
        observation_sequence = int(status["sequence"])
        timestamp_raw = status.get("updated_at_monotonic")
        observation_timestamp = (
            float(timestamp_raw)
            if timestamp_raw is not None and math.isfinite(float(timestamp_raw))
            else None
        )
        capture_error = max(
            float(np.linalg.norm(current_q[side] - frozen["arm_q"][side], ord=np.inf))
            for side in ("left", "right")
        )
        gripper_report = _x5_gripper_capture_error(current_gripper, frozen["gripper_q"])
        gripper_error_m = float(gripper_report["max_error_m"])
        has_previous_position = previous_positions is not None
        position_step = (
            float("inf")
            if previous_positions is None
            else float(np.linalg.norm(packed - previous_positions, ord=np.inf))
        )
        observation_dt = (
            None
            if previous_observation_timestamp is None or observation_timestamp is None
            else observation_timestamp - previous_observation_timestamp
        )
        if (
            observation_dt is not None
            and math.isfinite(observation_dt)
            and 1.0e-4 <= observation_dt <= 5.0
        ):
            last_handoff_observation_dt_s = float(observation_dt)
            handoff_observation_dt_samples_s.append(float(observation_dt))
        derived_velocity = (
            position_step / observation_dt
            if observation_dt is not None
            and math.isfinite(observation_dt)
            and 1.0e-4 <= observation_dt <= 5.0
            else None
        )
        previous_positions = packed
        previous_observation_timestamp = observation_timestamp

        current_live = _live(ctx, names, episode_id, manager=manager)
        live_ok = bool(current_live.get("ok"))
        sequence_ok = bool(
            live_ok
            and int(current_live.get("observation_sequence", -1))
            == observation_sequence
        )
        binding_replaced = bool(
            live_ok
            and (
                current_live.get("session_id") != session_id
                or current_live.get("image_id") != image_id
            )
        )
        awaiting_post_prior = bool(
            sequence_ok
            and minimum_tracker_sequence is not None
            and int(current_live.get("observation_sequence", -1))
            <= minimum_tracker_sequence
        )
        binding_ok = bool(
            sequence_ok and not binding_replaced and not awaiting_post_prior
        )
        reference_drift = 0.0 if not reference_names else float("inf")
        reference_errors = np.empty(0, dtype=np.float64)
        last_source_drift = float("inf")
        last_source_point_drift = {}
        last_anchor_error = float("inf")
        last_anchor_point_error = {}
        last_geometry = {
            "ok": not prior_active,
            "applicable": prior_active,
        }
        if binding_ok:
            try:
                current_points = _strict_point_matrix(
                    [
                        current_live["entries"][name]["xyz_in_robot_base_coord_m"]
                        for name in controlled_names
                    ],
                    expected_count=len(controlled_names),
                    label="start-state controlled point data",
                )
                last_live_points = current_points.copy()
                source_errors = np.linalg.norm(current_points - source_points, axis=1)
                last_source_drift = float(np.max(source_errors))
                last_source_point_drift = {
                    name: float(source_errors[index])
                    for index, name in enumerate(controlled_names)
                }
                if prior_active:
                    eef_position, eef_quaternion = _eef_base_xyzw(ctx, arm)
                    predicted = (
                        eef_position[None, :]
                        + (
                            quat_to_mat_xyzw(eef_quaternion)
                            @ np.asarray(planned_anchors, dtype=np.float64).T
                        ).T
                    )
                    anchor_errors = np.linalg.norm(current_points - predicted, axis=1)
                    last_anchor_error = float(np.max(anchor_errors))
                    last_anchor_point_error = {
                        name: float(anchor_errors[index])
                        for index, name in enumerate(controlled_names)
                    }
                    last_geometry = evaluate_rigid_geometry_consistency(
                        current_points,
                        source_points,
                        tolerance_m=float(geometry_tolerance_m),
                    )
                if reference_names:
                    current_references = _strict_point_matrix(
                        [
                            current_live["entries"][name]["xyz_in_robot_base_coord_m"]
                            for name in reference_names
                        ],
                        expected_count=len(reference_names),
                        label="start-state off-hand point data",
                    )
                    reference_consensus = _off_hand_reference_consensus(
                        reference_names,
                        current_references,
                        reference_points,
                        tolerance_m=reference_limit,
                        allow_visual_outlier_suppression=False,
                    )
                    reference_errors = np.asarray(
                        [
                            reference_consensus["point_drift_m"][name]
                            for name in reference_names
                        ],
                        dtype=np.float64,
                    )
                    reference_drift = float(reference_consensus["max_drift_m"])
            except (KeyError, TypeError, ValueError):
                binding_ok = False

        off_hand_validation = (
            {
                "ok": bool(reference_drift <= reference_limit),
                "applicable": True,
                "tolerance_m": reference_limit,
                "max_drift_m": reference_drift,
                "point_drift_m": {
                    name: float(reference_errors[index])
                    for index, name in enumerate(reference_names)
                },
            }
            if reference_names and reference_errors.size == len(reference_names)
            else {
                "ok": not bool(reference_names),
                "applicable": bool(reference_names),
                "reason": ("off_hand_points_invalid" if reference_names else None),
            }
        )
        stable = bool(
            binding_ok
            and has_previous_position
            and capture_error <= START_CAPTURE_TOLERANCE_RAD
            and position_step <= START_POSITION_STEP_RAD
            and (derived_velocity is None or derived_velocity <= START_ARM_QVEL_RAD_S)
            and gripper_error_m <= GRIPPER_START_HANDOFF_LIMIT_M
            and last_source_drift <= source_drift_limit
            and reference_drift <= reference_limit
            and (not prior_active or last_anchor_error <= anchor_tolerance_m)
            and (not prior_active or bool(last_geometry.get("ok")))
        )
        stable_count = stable_count + 1 if stable else 0
        stable_trace.append(
            {
                "settle_step": settle_step,
                "observation_sequence": observation_sequence,
                "observation_updated_at_monotonic": observation_timestamp,
                "capture_arm_error_inf_rad": capture_error,
                "position_step_inf_rad": (
                    None if not math.isfinite(position_step) else position_step
                ),
                "raw_qvel_available": False,
                "observation_dt_s": observation_dt,
                "position_derived_velocity_rad_s": derived_velocity,
                "stationarity_source": (
                    "consecutive_position_delta_over_observation_dt"
                    if derived_velocity is not None
                    else "consecutive_evaluator_qpos_position_step"
                ),
                "gripper_capture_error_normalized": float(
                    gripper_report["max_normalized_error"]
                ),
                "gripper_capture_error_m": gripper_error_m,
                "gripper_capture_error": json_ready(gripper_report),
                "source_drift_m": (
                    None if not math.isfinite(last_source_drift) else last_source_drift
                ),
                "source_drift_tolerance_m": source_drift_limit,
                "eef_anchor_error_m": (
                    None if not math.isfinite(last_anchor_error) else last_anchor_error
                ),
                "rigid_geometry": json_ready(last_geometry),
                "off_hand_reference": json_ready(off_hand_validation),
                "tracker_binding_ok": binding_ok,
                "awaiting_post_prior_rgbd_observation": awaiting_post_prior,
                "stable": stable,
            }
        )

        if binding_replaced:
            reason = "tracker_binding_replaced"
            break
        if not binding_ok:
            reason = (
                "awaiting_post_prior_rgbd_observation"
                if awaiting_post_prior
                else str(current_live.get("reason") or "no_synchronized_frame")
            )
            if settle_step >= START_MAX_SETTLE_STEPS:
                reason = "start_tracker_reacquisition_failed"
                break
            yield _observation_advance_action(ctx, frozen)
            continue
        if prior_active and last_anchor_error > anchor_tolerance_m:
            reason = "tracked_points_not_rigid_with_selected_eef"
            break
        if prior_active and not bool(last_geometry.get("ok")):
            reason = "tracked_rigid_geometry_changed_during_planning"
            break
        if last_source_drift > source_drift_limit:
            reason = (
                "tracked_points_moved_after_source_sync"
                if prior_active
                else "tracked_points_moved_during_planning"
            )
            break
        if reference_drift > reference_limit:
            reason = "off_hand_points_moved_before_execution"
            break
        if capture_error > START_CAPTURE_TOLERANCE_RAD:
            reason = "start_position_jump"
            break
        if gripper_error_m > GRIPPER_START_HANDOFF_LIMIT_M:
            reason = "start_gripper_position_drift"
            break
        if stable_count >= START_STABLE_STEPS:
            reason = "stable_start_verified"
            break
        if settle_step >= START_MAX_SETTLE_STEPS:
            reason = "start_state_not_stable"
            break
        yield _observation_advance_action(ctx, frozen)

    return {
        "ok": reason == "stable_start_verified",
        "reason": reason,
        "error": None if reason == "stable_start_verified" else reason,
        "required_stable_steps": START_STABLE_STEPS,
        "stable_steps": stable_count,
        "max_settle_steps": START_MAX_SETTLE_STEPS,
        "last_handoff_observation_dt_s": last_handoff_observation_dt_s,
        "handoff_observation_dt_samples_s": [
            float(value) for value in handoff_observation_dt_samples_s
        ],
        "max_handoff_observation_dt_s": (
            None
            if not handoff_observation_dt_samples_s
            else float(max(handoff_observation_dt_samples_s))
        ),
        "trace": json_ready(stable_trace),
        "off_hand_start_validation": json_ready(off_hand_validation),
        "observation_sequence_end": _sequence(ctx),
        "last_source_drift_m": (
            None if not math.isfinite(last_source_drift) else last_source_drift
        ),
        "last_source_point_drift_m": last_source_point_drift,
        "source_drift_tolerance_m": source_drift_limit,
        "last_live_points_robot_base_m": (
            None
            if last_live_points is None
            else last_live_points.astype(float).tolist()
        ),
        "last_eef_anchor_error_m": (
            None if not math.isfinite(last_anchor_error) else last_anchor_error
        ),
        "last_eef_anchor_point_error_m": last_anchor_point_error,
        "last_rigid_geometry_consistency": json_ready(last_geometry),
        "eef_observation_prior_active": prior_active,
        "stale_tracker_coordinates_used_while_waiting": False,
    }


def _final_goal_report_x5(
    ctx,
    *,
    manager=None,
    arm: str,
    names: list[str],
    controlled_names: list[str],
    reference_names: list[str],
    episode_id: str,
    session_id: str,
    image_id: str,
    anchors: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray | None,
    target_points: list[dict[str, Any]],
    fixed_target_points: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    quick_constraint: Mapping[str, Any] | None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
    anchor_tolerance_m: float,
    geometry_tolerance_m: float,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    pos_tol_m: float,
    ori_tol_deg: float,
    camera_position_robot_base_m: Sequence[float] | None = None,
    camera_quaternion_xyzw: Sequence[float] | None = None,
    inequalities: Sequence[Mapping[str, Any]] | None = None,
    use_live_reference_for_constraints: bool = False,
) -> dict[str, Any]:
    sequence = _sequence(ctx)
    live = _live(ctx, names, episode_id, manager=manager)
    report: dict[str, Any] = {
        "ok": False,
        "reason": None,
        "observation_sequence": sequence,
        "tracker_observation_sequence": live.get("observation_sequence"),
        "goal_satisfied": False,
    }
    if live.get("session_id") != session_id or live.get("image_id") != image_id:
        report["reason"] = "tracker_binding_replaced"
        return report
    position, quaternion = _eef_base_xyzw(ctx, arm)
    bundle = _final_observation_bundle_x5(
        live,
        controlled_names=controlled_names,
        fixed_names=reference_names,
        current_position=position,
        current_quaternion=quaternion,
        anchors_eef=anchors,
        current_sequence=sequence,
        anchor_tolerance_m=anchor_tolerance_m,
        pos_tol_m=pos_tol_m,
        adapter=ctx.world._official_adapter,
        base_rotation=ctx.world.base_rotation,
    )
    report["on_hand_observation_completion"] = json_ready(bundle)
    if not bool(bundle.get("ok")):
        report["reason"] = str(
            bundle.get("reason") or live.get("reason") or "tracked_point_unavailable"
        )
        return report
    measured = np.asarray(bundle["points"], dtype=np.float64)
    measured_references = (
        np.asarray(bundle["fixed_points"], dtype=np.float64)
        if reference_names
        else None
    )
    predicted = np.asarray(bundle["predicted_points"], dtype=np.float64)
    anchor_errors = np.linalg.norm(measured - predicted, axis=1)
    geometry = evaluate_rigid_geometry_consistency(
        measured,
        source_points,
        tolerance_m=geometry_tolerance_m,
    )
    constraint_references = measured_references
    reference_report: dict[str, Any] = {"ok": True, "applicable": False}
    if reference_names:
        reference_report = _off_hand_reference_consensus(
            reference_names,
            measured_references,
            reference_points,
            tolerance_m=max(0.020, 2.0 * float(pos_tol_m)),
            allow_visual_outlier_suppression=True,
        )
        if not bool(reference_report.get("ok")):
            report.update(
                {
                    "reason": str(
                        reference_report.get("reason") or "off_hand_points_moved"
                    ),
                    "off_hand_reference": json_ready(reference_report),
                }
            )
            return report
        # Live RGB-D remains mandatory evidence that each off-hand scene point
        # stayed put.  Precision-mode executions additionally use the current
        # synchronized reference for the terminal correction, so the reported
        # moving/reference pair is what the operator actually sees.  Legacy
        # callers which explicitly requested a tight historical tolerance keep
        # the frozen-goal behavior for compatibility.
        constraint_references = (
            measured_references.copy()
            if use_live_reference_for_constraints
            else np.asarray(reference_points, dtype=np.float64).copy()
        )

    base_constraints = evaluate_target_constraints(
        measured,
        target_points,
        tolerance_m=float(pos_tol_m),
        relations=relations,
        inequalities=list(inequalities or []),
        orientation_tolerance_deg=float(ori_tol_deg),
        fixed_points_robot_base_m=constraint_references,
        fixed_target_points=(fixed_target_points if reference_names else None),
    )
    quick_specs = _quick_constraint_specs(
        quick_constraint=quick_constraint,
        quick_constraints=quick_constraints,
    )
    faceto_specs = [
        spec
        for spec in quick_specs
        if str(spec.get("type") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        in {"faceto", "reverse_faceto"}
    ]
    quick_report = (
        _evaluate_quick_constraints(
            measured,
            constraint_references,
            faceto_specs,
            controlled_names=controlled_names,
            reference_names=reference_names,
            position_tolerance_m=float(pos_tol_m),
            orientation_tolerance_deg=float(ori_tol_deg),
            camera_position_robot_base_m=camera_position_robot_base_m,
            camera_quaternion_xyzw=camera_quaternion_xyzw,
        )
        if faceto_specs
        else {
            "ok": True,
            "applicable": False,
            "group_count": 0,
            "groups": [],
        }
    )
    constraints = {
        **base_constraints,
        "ok": bool(
            base_constraints.get("ok")
            and quick_report.get("ok")
            and reference_report.get("ok")
        ),
        "quick_constraint": json_ready(quick_report),
        "off_hand_reference": json_ready(reference_report),
    }
    precision_gate = _strict_position_precision_gate(base_constraints)
    constraints["strict_position_precision_gate"] = precision_gate
    constraints["ok"] = bool(constraints.get("ok") and precision_gate["ok"])
    if not precision_gate["ok"] and not constraints.get("error"):
        constraints["error"] = (
            "strict terminal position gate failed: "
            f"max_constraint_error={precision_gate['max_constraint_error_m']:.6f}m "
            f"must be < {precision_gate['limit_m']:.6f}m"
        )
    position_error = float(np.linalg.norm(position - target_position))
    dot = abs(float(np.dot(quaternion, target_quaternion)))
    orientation_error = math.degrees(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))
    anchor_ok = bool(float(np.max(anchor_errors)) <= anchor_tolerance_m)
    pose_ok = bool(
        position_error <= max(0.001, float(pos_tol_m))
        and orientation_error <= max(0.1, float(ori_tol_deg))
    )
    report.update(
        {
            "ok": True,
            "reason": None,
            "goal_satisfied": bool(
                constraints.get("ok") and geometry.get("ok") and anchor_ok and pose_ok
            ),
            "pose_satisfied": pose_ok,
            "anchor_satisfied": anchor_ok,
            "position_error_m": position_error,
            "orientation_error_deg": orientation_error,
            "max_anchor_error_m": float(np.max(anchor_errors)),
            "point_anchor_error_m": {
                name: float(anchor_errors[index])
                for index, name in enumerate(controlled_names)
            },
            "rigid_pair_consistency": json_ready(geometry),
            "points_robot_base_m": {
                name: measured[index].astype(float).tolist()
                for index, name in enumerate(controlled_names)
            },
            "fixed_points_robot_base_m": {
                name: measured_references[index].astype(float).tolist()
                for index, name in enumerate(reference_names)
            }
            if reference_names
            else {},
            "constraint_fixed_points_robot_base_m": {
                name: constraint_references[index].astype(float).tolist()
                for index, name in enumerate(reference_names)
            }
            if reference_names
            else {},
            "constraint_reference_source": (
                "live_off_hand_reference_for_strict_precision"
                if use_live_reference_for_constraints and reference_names
                else (
                    "frozen_off_hand_reference" if reference_names else "not_applicable"
                )
            ),
            "use_live_reference_for_constraints": bool(
                use_live_reference_for_constraints
            ),
            "strict_position_precision_gate": precision_gate,
            "off_hand_reference": json_ready(reference_report),
            "constraints": json_ready(constraints),
        }
    )
    return report


def run_move_tracked_point(
    ctx,
    *,
    points: list[dict[str, Any]],
    relations: Sequence[Mapping[str, Any]] | None = None,
    pos_tol: float,
    ori_tol_deg: float,
    max_steps: int,
    timeout_s: float,
    quick_constraint: Mapping[str, Any] | None = None,
    quick_constraints: Sequence[Mapping[str, Any]] | None = None,
    mode: str | None = None,
    execution_mode: str = "exec",
    inequalities: Sequence[Mapping[str, Any]] | None = None,
    result_writer: Callable[..., None],
    lifecycle_state: dict[str, Any] | None = None,
    motion_retention_lease_state: dict[str, Any] | None = None,
):
    """Execute the current BEHAVIOR tracked-point policy on dual X5 observations.

    Public target, relation, preset, plan-only, and result semantics stay
    unchanged. Only R1Pro embodiment operations are replaced by fixed-base
    six-DoF X5 FK/IK, X5 gripper control, and RoboDojo observation/action IO.
    """

    lifecycle = lifecycle_state if lifecycle_state is not None else {}
    retention_state = (
        motion_retention_lease_state if motion_retention_lease_state is not None else {}
    )
    rigid_pair_lease_state = retention_state.setdefault("_rigid_pair_lease_state", {})
    raw_args = {
        "points": points,
        "relations": list(relations or []),
        "inequalities": list(inequalities or []),
        "pos_tol": pos_tol,
        "ori_tol_deg": ori_tol_deg,
        "max_steps": max_steps,
        "timeout_s": timeout_s,
        "execution_mode": execution_mode,
    }
    if mode is not None:
        raw_args["mode"] = mode
    if quick_constraint is not None:
        raw_args["quick_constraint"] = quick_constraint
        if mode is None:
            raw_args["mode"] = "quick_constraint"
    if quick_constraints is not None:
        raw_args["quick_constraints"] = list(quick_constraints)
        if mode is None:
            raw_args["mode"] = "quick_constraint"
    try:
        args = validate_move_tracked_point_args(raw_args)
    except (TypeError, ValueError) as exc:
        _set_failure(
            ctx, result_writer, stage="input validation", error=str(exc), args=raw_args
        )
        yield ctx.world.hold_action()
        return

    requested_position_tolerance_m = float(args["pos_tol"])
    effective_position_tolerance_m = min(
        requested_position_tolerance_m,
        FINAL_EXECUTION_POSITION_TOLERANCE_M,
    )
    # The UI/default request historically omitted ``pos_tol`` (30 mm).  In
    # that compatibility path, a terminal RGB-D correction must use the live
    # off-hand reference so the displayed moving/reference pair is certified,
    # not merely the frozen planning snapshot.  Callers that explicitly opt
    # into the old <=4 mm contract retain its frozen-reference semantics.
    strict_live_reference_goal = bool(requested_position_tolerance_m >= 0.010)
    # Keep the public argument shape and call sites unchanged.  All planning,
    # endpoint ranking, live correction, and terminal validation below consume
    # this execution tolerance; the original caller value is reported beside
    # it for auditability.
    args["pos_tol"] = float(effective_position_tolerance_m)

    # Keep the public request value for reporting/contract compatibility, but
    # let the dense RoboDojo execution branch give the planner the same large
    # room that execution receives.  Otherwise a caller's historical
    # ``max_steps=80`` would cap trajectory *generation* before the no-
    # compression scheduler ever saw it.
    requested_public_max_steps = int(args["max_steps"])
    dense_planning_max_steps = (
        max(int(EXECUTION_DENSE_ACTION_BUDGET), requested_public_max_steps)
        if EXECUTION_DENSE_WAYPOINTS and args["execution_mode"] == "exec"
        else requested_public_max_steps
    )

    names = [str(point["name"]) for point in args["points"]]
    quick_spec = args.get("quick_constraint")
    quick_specs = _quick_constraint_specs(
        quick_constraint=(quick_spec if isinstance(quick_spec, Mapping) else None),
        quick_constraints=args.get("quick_constraints"),
    )
    quick_mode = bool(quick_specs)
    faceto_requested = any(
        str(spec.get("type") or "").strip().lower().replace("-", "_").replace(" ", "_")
        in {"faceto", "reverse_faceto"}
        for spec in quick_specs
    )
    head_camera_position_robot_base_m: np.ndarray | None = None
    head_camera_quaternion_xyzw: np.ndarray | None = None
    if not quick_specs and not list(args.get("relations") or []):
        promoted_points, frame_origin_promotion = _promote_unique_frame_origin(
            args["points"]
        )
        args["points"] = promoted_points
    else:
        frame_origin_promotion = {
            "applied": False,
            "reason": "relations_or_quick_preset",
        }
    args["frame_origin_promotion"] = frame_origin_promotion
    roles_by_name = {
        str(point["name"]): str(point.get("role") or "on_hand")
        for point in args["points"]
    }
    controlled_names = [name for name in names if roles_by_name[name] == "on_hand"]
    reference_names = [name for name in names if roles_by_name[name] == "off_hand"]
    reference_name_set = set(reference_names)
    for relation in args["relations"]:
        if relation.get(
            "type"
        ) != "ordered_collinear" and reference_name_set.intersection(
            relation.get("point_names", [])
        ):
            _set_failure(
                ctx,
                result_writer,
                stage="input validation",
                error=(
                    "typed relations may reference only on-hand points; use a "
                    "role-based quick preset for on-hand/off-hand geometry"
                ),
                args=args,
            )
            yield ctx.world.hold_action()
            return
    planning_target_points = _planning_targets(args["points"], controlled_names)
    fixed_target_points = _planning_targets(args["points"], reference_names)
    planning_relations = list(args["relations"])
    planning_started = time.monotonic()
    planning_budget_s = _planning_budget(float(args["timeout_s"]))
    planning_deadline = planning_started + planning_budget_s
    episode_id = str(ctx.world.episode_id())
    predefined_names, visual_names = partition_predefined_eef_point_names(names)
    predefined_name_set = set(predefined_names)
    visual_controlled_names = [
        name for name in controlled_names if name not in predefined_name_set
    ]
    tracker_status = ctx.tracker.status()
    registered_names = set(tracker_status.get("registered_names") or [])
    manager_backed_tracking = set(visual_names).issubset(registered_names)
    predefined_arm_selection: dict[str, Any] = {
        "applicable": bool(predefined_names),
        "selected_arm": None,
        "predefined_eef_point_names": list(predefined_names),
    }
    provisional_arm = "right"
    manager = _MoveTrackedPointSourceManager(
        ctx,
        ctx.tracker,
        predefined_names=predefined_names,
        episode_id=episode_id,
        arm=provisional_arm,
    )
    lifecycle["source_manager"] = manager
    rigid_pair_activation: dict[str, Any] = {"applicable": False}
    rigid_pair_activation_confirmation: dict[str, Any] = {
        "ok": True,
        "required": False,
        "reason": "not_attempted",
        "recovery_hold_steps": 0,
    }
    rigid_pair_bootstrap_eef_prior: dict[str, Any] = {
        "ok": True,
        "active": False,
        "reason": "not_required",
    }
    rigid_pair_activation_hold_steps = 0
    activation_hold_state: dict[str, Any] = {}
    prior_rigid_pair_cleanup: dict[str, Any] = {}
    failed_rigid_pair_cleanup: dict[str, Any] = {}
    rigid_pair_activation_provisional = False
    motion_retention: dict[str, Any] = {"applicable": False}
    registered_source_reconciliation: dict[str, Any] = {
        "ok": True,
        "applicable": False,
        "applied": False,
        "reason": "not_attempted",
    }
    if manager_backed_tracking:
        try:
            motion_retention = manager.begin_motion_retention(
                names,
                episode_id=episode_id,
                timeout_s=min(
                    660.0,
                    float(args["timeout_s"]) + SUCCESSOR_RETENTION_EXTRA_S,
                ),
            )
            lease_id = str(motion_retention.get("lease_id") or "").strip()
            if not bool(motion_retention.get("ok", True)) or not lease_id:
                raise ValueError("tracker request retention lease was not created")
            motion_retention["applicable"] = True
            retention_state.update(
                {
                    "lease_id": lease_id,
                    "lease": dict(motion_retention),
                }
            )
            activation_sequence = _sequence(ctx)
            activation_hold_state = _freeze_state(ctx)
            if _sequence(ctx) != activation_sequence:
                raise ValueError(
                    "evaluator observation changed while latching rigid-pair "
                    "activation hold"
                )
            prior_rigid_pair_cleanup = _clear_incompatible_rigid_pair_x5(
                manager, visual_controlled_names
            )
            if len(visual_controlled_names) == 2:
                rigid_pair_lease_state.update(
                    {
                        "active": True,
                        "preserve_after_success": False,
                        "names": list(visual_controlled_names),
                        "episode_id": episode_id,
                    }
                )
            rigid_pair_activation = _gate_rigid_pair_source_projection_x5(
                _activate_rigid_points(
                    ctx,
                    visual_controlled_names,
                    episode_id,
                    manager=manager,
                    allow_observation_failure=True,
                )
            )
            rigid_pair_activation["applicable"] = len(visual_controlled_names) >= 2
            activation_reason = str(rigid_pair_activation.get("reason") or "")
            if (
                len(visual_controlled_names) == 2
                and not bool(rigid_pair_activation.get("ok"))
                and activation_reason in RIGID_PAIR_ACTIVATION_INITIAL_RETRY_REASONS
            ):
                try:
                    rigid_pair_bootstrap_eef_prior = (
                        _bootstrap_registered_on_hand_eef_prior_x5(
                            ctx,
                            manager,
                            visual_controlled_names,
                            episode_id=episode_id,
                            current_frozen=activation_hold_state,
                        )
                    )
                except Exception as exc:
                    rigid_pair_bootstrap_eef_prior = {
                        "ok": False,
                        "active": False,
                        "reason": "registration_eef_bootstrap_unavailable",
                        "error": f"{type(exc).__name__}: {exc}",
                        "prediction_values_published_as_observation": False,
                    }
                if bool(rigid_pair_bootstrap_eef_prior.get("active")):
                    bootstrap_lease_id = str(
                        rigid_pair_bootstrap_eef_prior.get("lease_id") or ""
                    ).strip()
                    if bootstrap_lease_id:
                        lifecycle["eef_prior_lease_id"] = bootstrap_lease_id

            rigid_pair_activation_provisional = bool(
                len(visual_controlled_names) == 2
                and not rigid_pair_activation.get("ok")
            )
            try:
                if args["execution_mode"] == "plan":
                    activation_ok = bool(rigid_pair_activation.get("ok"))
                    rigid_pair_activation_confirmation = {
                        "ok": activation_ok,
                        "required": not activation_ok,
                        "recovered": False,
                        "reason": (
                            None
                            if activation_ok
                            else activation_reason or "invalid_rigid_pair_fusion_result"
                        ),
                        "attempt_count": 1,
                        "recovery_hold_steps": 0,
                        "required_stable_steps": int(
                            RIGID_PAIR_ACTIVATION_STABLE_STEPS
                        ),
                        "max_wait_steps": int(RIGID_PAIR_ACTIVATION_MAX_WAIT_STEPS),
                        "observation_sequence_start": int(activation_sequence),
                        "observation_sequence_end": int(activation_sequence),
                        "first_failure": (
                            None
                            if activation_ok
                            else _rigid_pair_activation_summary_x5(
                                rigid_pair_activation
                            )
                        ),
                        "last_pair_report": _rigid_pair_activation_summary_x5(
                            rigid_pair_activation
                        ),
                        "history": [],
                        "correction_limit_unchanged": True,
                        "confirmation_source": (
                            "current_synchronized_observation_only"
                        ),
                        "action_free_plan_mode": True,
                        "evaluator_action_emitted": False,
                        "fresh_observation_recovery_skipped": not activation_ok,
                    }
                else:
                    rigid_pair_activation_confirmation = yield from (
                        _confirm_fresh_rigid_pair_activation_x5(
                            ctx,
                            manager,
                            visual_controlled_names,
                            episode_id=episode_id,
                            observation_sequence=activation_sequence,
                            initial_report=rigid_pair_activation,
                            hold_state=activation_hold_state,
                            deadline_monotonic=(
                                planning_started + float(args["timeout_s"])
                            ),
                        )
                    )
            finally:
                if (
                    rigid_pair_activation_provisional
                    and not rigid_pair_activation_confirmation.get("ok")
                    and not failed_rigid_pair_cleanup
                ):
                    cleanup_reason = (
                        "rigid_pair_activation_recovery_failed"
                        if rigid_pair_activation_confirmation
                        else "rigid_pair_activation_recovery_interrupted"
                    )
                    failed_rigid_pair_cleanup = _deactivate_rigid_pair_x5(
                        manager,
                        reason=cleanup_reason,
                        required=False,
                    )
                    rigid_pair_lease_state["active"] = False
                    rigid_pair_lease_state["cleanup"] = json_ready(
                        failed_rigid_pair_cleanup
                    )
            rigid_pair_activation_hold_steps = int(
                rigid_pair_activation_confirmation.get("recovery_hold_steps", 0)
            )
            if not bool(rigid_pair_activation_confirmation.get("ok")):
                raise ValueError(
                    _rigid_pair_activation_error_x5(rigid_pair_activation_confirmation)
                )
            rigid_pair_activation_provisional = False
            if bool(rigid_pair_activation_confirmation.get("recovered")):
                initial_activation_report = _rigid_pair_activation_summary_x5(
                    rigid_pair_activation
                )
                confirmed_pair_report = rigid_pair_activation_confirmation.get(
                    "last_pair_report"
                )
                if not isinstance(confirmed_pair_report, Mapping):
                    confirmed_pair_report = {}
                rigid_pair_activation = {
                    **rigid_pair_activation,
                    **dict(confirmed_pair_report),
                    "ok": True,
                    "reason": None,
                    "initial_activation_report": initial_activation_report,
                    "fresh_observation_confirmation": json_ready(
                        rigid_pair_activation_confirmation
                    ),
                }
                accepted_sequence = int(
                    rigid_pair_activation_confirmation.get(
                        "observation_sequence_end", -1
                    )
                )
                if _sequence(ctx) != accepted_sequence:
                    raise ValueError(
                        "evaluator observation changed after rigid-pair "
                        "activation confirmation"
                    )
            else:
                rigid_pair_activation = {
                    **rigid_pair_activation,
                    "fresh_observation_confirmation": json_ready(
                        rigid_pair_activation_confirmation
                    ),
                }
        except (KeyError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            _set_failure(
                ctx,
                result_writer,
                stage="tracking setup",
                error=str(exc),
                args=args,
                details={
                    "registered_names": sorted(registered_names),
                    "controlled_point_names": controlled_names,
                    "visual_controlled_point_names": visual_controlled_names,
                    "rigid_pair_activation": json_ready(rigid_pair_activation),
                    "rigid_pair_activation_confirmation": json_ready(
                        rigid_pair_activation_confirmation
                    ),
                    "rigid_pair_activation_hold_steps": int(
                        rigid_pair_activation_hold_steps
                    ),
                    "prior_rigid_pair_cleanup": json_ready(prior_rigid_pair_cleanup),
                    "failed_rigid_pair_cleanup": json_ready(failed_rigid_pair_cleanup),
                    "rigid_pair_bootstrap_eef_observation_prior": json_ready(
                        rigid_pair_bootstrap_eef_prior
                    ),
                },
            )
            yield ctx.world.hold_action()
            return
    try:
        observation_sequence = _sequence(ctx)
        initial_observation_sequence = observation_sequence
        live = _live(ctx, names, episode_id, manager=manager)
        if not bool(live.get("ok")):
            raise ValueError(
                f"tracked points are unavailable: {live.get('reason') or 'unknown'}"
            )
        if int(live.get("observation_sequence", -1)) != observation_sequence:
            raise ValueError(
                "tracked points are not synchronized to current proprioception"
            )
        session_id = str(live.get("session_id") or "").strip()
        image_id = str(live.get("image_id") or "").strip()
        if not session_id or not image_id:
            raise ValueError("active tracker binding has no session_id or image_id")
        if predefined_names:
            observations = {
                side: _x5_kinematic_observation(ctx, side) for side in ("left", "right")
            }
            provisional_arm, predefined_arm_selection = _select_predefined_eef_arm(
                points=args["points"],
                roles_by_name=roles_by_name,
                observations=observations,
                visual_entries=live.get("entries") or {},
            )
            predefined_arm_selection["applicable"] = True
            manager.set_arm(provisional_arm)
            rebound_live = _live(ctx, names, episode_id, manager=manager)
            if not bool(rebound_live.get("ok")):
                raise ValueError(
                    "predefined EEF points could not be rebound to the selected "
                    f"arm: {rebound_live.get('reason') or 'unknown'}"
                )
            if int(rebound_live.get("observation_sequence", -1)) != (
                observation_sequence
            ):
                raise ValueError(
                    "predefined EEF point arm selection changed the frozen observation"
                )
            if (
                str(rebound_live.get("session_id") or "") != session_id
                or str(rebound_live.get("image_id") or "") != image_id
            ):
                raise ValueError(
                    "predefined EEF point arm selection changed the tracker binding"
                )
            live = rebound_live
        frozen = _freeze_state(ctx)
        if _sequence(ctx) != observation_sequence:
            raise ValueError(
                "evaluator observation changed while freezing tracked points"
            )
        source_by_name = {
            name: np.asarray(
                live["entries"][name]["xyz_in_robot_base_coord_m"], dtype=np.float64
            ).reshape(3)
            for name in names
        }
        if not all(np.all(np.isfinite(value)) for value in source_by_name.values()):
            raise ValueError("tracked point XYZ contains non-finite values")
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        _set_failure(
            ctx,
            result_writer,
            stage="observation validation",
            error=str(exc),
            args=args,
        )
        yield ctx.world.hold_action()
        return

    source_points = np.asarray(
        [source_by_name[name] for name in controlled_names],
        dtype=np.float64,
    )
    try:
        if visual_controlled_names:
            visual_controlled_points = np.asarray(
                [source_by_name[name] for name in visual_controlled_names],
                dtype=np.float64,
            )
            registered_source_reconciliation = _reconcile_registered_on_hand_source_x5(
                ctx,
                manager,
                visual_controlled_names,
                episode_id=episode_id,
                current_frozen=frozen,
                live_points_robot_base_m=visual_controlled_points,
            )
        else:
            registered_source_reconciliation = {
                "ok": True,
                "applicable": False,
                "applied": False,
                "reason": "all_on_hand_points_are_predefined_eef_geometry",
                "names": [],
                "source_points_robot_base_m": [],
                "predictions_published_as_observation": False,
            }
        if (
            args["execution_mode"] == "plan"
            and not registered_source_reconciliation.get("ok")
            and registered_source_reconciliation.get("reason")
            in {
                "registered_eef_anchor_consensus_insufficient",
                "registered_eef_anchor_recovery_exceeded",
            }
        ):
            reacquisition = {
                "ok": False,
                "recovered": False,
                "reason": "plan_mode_requires_fresh_rgbd_reacquisition",
                "initial_failure": json_ready(registered_source_reconciliation),
                "history": [],
                "hold_steps": 0,
                "action_free_plan_mode": True,
                "evaluator_action_emitted": False,
                "predictions_published_as_observation": False,
            }
            registered_source_reconciliation = {
                **registered_source_reconciliation,
                "reason": reacquisition["reason"],
                "fresh_rgbd_reacquisition": json_ready(reacquisition),
            }
        if not bool(registered_source_reconciliation.get("ok")):
            raise ValueError(
                "on-hand tracked points are inconsistent with their "
                "registration-time EEF anchors: "
                f"{registered_source_reconciliation.get('reason') or 'unknown'}"
            )
        if bool(registered_source_reconciliation.get("applied")):
            reconciled_visual = _strict_point_matrix(
                registered_source_reconciliation["source_points_robot_base_m"],
                expected_count=len(visual_controlled_names),
                label="registered-EEF-reconciled on-hand planning source",
            )
            for index, name in enumerate(visual_controlled_names):
                source_by_name[name] = reconciled_visual[index].copy()
            source_points = np.asarray(
                [source_by_name[name] for name in controlled_names],
                dtype=np.float64,
            )
    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
    ) as exc:
        reconciliation_reason = str(
            registered_source_reconciliation.get("reason") or ""
        )
        _set_failure(
            ctx,
            result_writer,
            stage=(
                "timeout"
                if reconciliation_reason == "registered_bundle_reacquisition_timeout"
                else "observation validation"
            ),
            error=str(exc),
            args=args,
            details={
                "rigid_pair_activation": json_ready(rigid_pair_activation),
                "rigid_pair_activation_confirmation": json_ready(
                    rigid_pair_activation_confirmation
                ),
                "rigid_pair_activation_hold_steps": int(
                    rigid_pair_activation_hold_steps
                ),
                "rigid_pair_bootstrap_eef_observation_prior": json_ready(
                    rigid_pair_bootstrap_eef_prior
                ),
                "registered_on_hand_source_reconciliation": json_ready(
                    registered_source_reconciliation
                ),
            },
        )
        yield ctx.world.hold_action()
        return
    reference_points = (
        np.asarray([source_by_name[name] for name in reference_names], dtype=np.float64)
        if reference_names
        else None
    )
    if faceto_requested:
        (
            head_camera_position_robot_base_m,
            head_camera_quaternion_xyzw,
        ) = _head_camera_pose_x5(
            ctx,
            expected_sequence=int(observation_sequence),
            base_rotation=frozen["base_rotation"],
        )
    if quick_mode:
        # The preset is lowered into the same relation vocabulary as coordinate
        # mode.  Submitted XYZ/affine constraints remain active, including
        # coordinates attached to frozen off-hand references.
        planning_relations = _relations_with_quick_constraints(
            args["relations"],
            quick_specs,
            controlled_names=controlled_names,
            controlled_points=source_points,
            reference_names=reference_names,
            reference_points=reference_points,
            camera_position_robot_base_m=head_camera_position_robot_base_m,
            camera_quaternion_xyzw=head_camera_quaternion_xyzw,
        )

    distances = {
        side: float(
            np.mean(
                [
                    np.linalg.norm(
                        source_by_name[name] - frozen["eef_base_wxyz"][side][:3]
                    )
                    for name in controlled_names
                ]
            )
        )
        for side in ("left", "right")
    }
    reconciled_arm = str(registered_source_reconciliation.get("selected_arm") or "")
    arm = (
        provisional_arm
        if predefined_names
        else reconciled_arm
        if bool(registered_source_reconciliation.get("applied"))
        and reconciled_arm in {"left", "right"}
        else min(("left", "right"), key=lambda side: distances[side])
    )
    manager.set_arm(arm)
    nearest_name = min(
        controlled_names,
        key=lambda name: float(
            np.linalg.norm(source_by_name[name] - frozen["eef_base_wxyz"][arm][:3])
        ),
    )
    eef_prior_activation: dict[str, Any] = {"applicable": False}
    preplanning_source_sync: dict[str, Any] = {
        "ok": True,
        "applicable": False,
        "reason": "eef_observation_prior_not_applicable",
    }
    preplanning_hold_steps = 0
    if (
        manager_backed_tracking
        and manager.visual_point_count(controlled_names) > 0
        and args["execution_mode"] == "exec"
    ):
        try:
            frozen_eef = np.asarray(
                frozen["eef_base_wxyz"][arm], dtype=np.float64
            ).reshape(7)
            frozen_eef_xyzw = np.asarray(
                [frozen_eef[4], frozen_eef[5], frozen_eef[6], frozen_eef[3]],
                dtype=np.float64,
            )
            preplanning_anchors = (
                quat_to_mat_xyzw(frozen_eef_xyzw).T @ (source_points - frozen_eef[:3]).T
            ).T
            eef_prior_activation = manager.activate_on_hand_eef_observation_prior(
                controlled_names,
                episode_id=episode_id,
                session_id=session_id,
                image_id=image_id,
                arm=arm,
                anchors_eef_m=preplanning_anchors.tolist(),
            )
            if not isinstance(eef_prior_activation, dict) or not bool(
                eef_prior_activation.get("ok", True)
            ):
                raise ValueError("on-hand EEF observation prior was not activated")
            prior_lease_id = str(eef_prior_activation.get("lease_id") or "").strip()
            if not prior_lease_id:
                raise ValueError("on-hand EEF observation prior has no lease_id")
            eef_prior_activation["applicable"] = True
            lifecycle["eef_prior_lease_id"] = prior_lease_id

            prior_sequence = observation_sequence
            previous_sync_positions: np.ndarray | None = None
            sync_stable_steps = 0
            sync_trace: list[dict[str, Any]] = []
            last_sync_sequence = prior_sequence
            prior_status = ctx.world._official_adapter.status()
            prior_timestamp_raw = prior_status.get("updated_at_monotonic")
            previous_sync_timestamp = (
                float(prior_timestamp_raw)
                if int(prior_status.get("sequence", -1)) == prior_sequence
                and prior_timestamp_raw is not None
                and math.isfinite(float(prior_timestamp_raw))
                else None
            )
            last_sync_observation_dt_s: float | None = None
            sync_observation_dt_samples_s: list[float] = []
            post_prior_live: dict[str, Any] | None = None
            post_prior_sequence: int | None = None
            for sync_step in range(START_MAX_SETTLE_STEPS + 1):
                if time.monotonic() - planning_started >= float(args["timeout_s"]):
                    raise TimeoutError(
                        "timed out waiting for a stable post-prior planning source"
                    )
                yield _observation_advance_action(ctx, frozen)
                preplanning_hold_steps += 1
                sync_status = ctx.world._official_adapter.status()
                current_sync_sequence = int(sync_status["sequence"])
                sync_timestamp_raw = sync_status.get("updated_at_monotonic")
                current_sync_timestamp = (
                    float(sync_timestamp_raw)
                    if sync_timestamp_raw is not None
                    and math.isfinite(float(sync_timestamp_raw))
                    else None
                )
                sync_observation_dt = (
                    None
                    if previous_sync_timestamp is None or current_sync_timestamp is None
                    else current_sync_timestamp - previous_sync_timestamp
                )
                if (
                    sync_observation_dt is not None
                    and math.isfinite(sync_observation_dt)
                    and 1.0e-4 <= sync_observation_dt <= 5.0
                ):
                    last_sync_observation_dt_s = float(sync_observation_dt)
                    sync_observation_dt_samples_s.append(float(sync_observation_dt))
                previous_sync_timestamp = current_sync_timestamp
                if current_sync_sequence <= last_sync_sequence:
                    raise ValueError("post-prior evaluator observation did not advance")
                last_sync_sequence = current_sync_sequence
                current_sync_live = _live(ctx, names, episode_id, manager=manager)
                if not bool(current_sync_live.get("ok")):
                    unavailable = current_sync_live.get("unavailable")
                    unavailable_reasons = (
                        [str(value) for value in unavailable.values()]
                        if isinstance(unavailable, dict)
                        else []
                    )
                    retryable = bool(
                        current_sync_live.get("reason") == "no_synchronized_frame"
                        or (
                            current_sync_live.get("reason")
                            == "tracked_point_unavailable"
                            and unavailable_reasons
                            and all(
                                value
                                in {"point_unobserved", "stale_tracking_measurement"}
                                for value in unavailable_reasons
                            )
                        )
                    )
                    sync_trace.append(
                        {
                            "sync_step": sync_step,
                            "observation_sequence": current_sync_sequence,
                            "observation_updated_at_monotonic": (
                                current_sync_timestamp
                            ),
                            "observation_dt_s": sync_observation_dt,
                            "tracker_ok": False,
                            "reason": current_sync_live.get("reason"),
                            "stale_coordinates_consumed": False,
                        }
                    )
                    if retryable:
                        sync_stable_steps = 0
                        continue
                    raise ValueError(
                        "post-prior tracked points are unavailable: "
                        f"{current_sync_live.get('reason') or 'unknown'}"
                    )
                if (
                    int(current_sync_live.get("observation_sequence", -1))
                    != current_sync_sequence
                ):
                    sync_trace.append(
                        {
                            "sync_step": sync_step,
                            "observation_sequence": current_sync_sequence,
                            "observation_updated_at_monotonic": (
                                current_sync_timestamp
                            ),
                            "observation_dt_s": sync_observation_dt,
                            "tracker_ok": False,
                            "reason": "tracker_observation_mismatch",
                            "stale_coordinates_consumed": False,
                        }
                    )
                    sync_stable_steps = 0
                    continue
                if (
                    current_sync_live.get("session_id") != session_id
                    or current_sync_live.get("image_id") != image_id
                ):
                    raise ValueError(
                        "tracker binding was replaced while activating EEF prior"
                    )
                sync_positions = np.concatenate(
                    [
                        np.asarray(ctx.world.arm_qpos_list(side), dtype=np.float64)
                        for side in ("left", "right")
                    ]
                )
                sync_position_step = (
                    math.inf
                    if previous_sync_positions is None
                    else float(
                        np.linalg.norm(
                            sync_positions - previous_sync_positions, ord=np.inf
                        )
                    )
                )
                previous_sync_positions = sync_positions
                stationary = bool(sync_position_step <= START_POSITION_STEP_RAD)
                sync_stable_steps = sync_stable_steps + 1 if stationary else 0
                sync_trace.append(
                    {
                        "sync_step": sync_step,
                        "observation_sequence": current_sync_sequence,
                        "observation_updated_at_monotonic": current_sync_timestamp,
                        "observation_dt_s": sync_observation_dt,
                        "tracker_ok": True,
                        "position_step_inf_rad": (
                            None
                            if not math.isfinite(sync_position_step)
                            else sync_position_step
                        ),
                        "stationary": stationary,
                        "stable_steps": sync_stable_steps,
                        "stale_coordinates_consumed": False,
                    }
                )
                if sync_stable_steps >= START_STABLE_STEPS:
                    post_prior_live = current_sync_live
                    post_prior_sequence = current_sync_sequence
                    break
            if post_prior_live is None or post_prior_sequence is None:
                raise ValueError("post-prior planning source did not become stable")
            refined_by_name = {
                name: np.asarray(
                    post_prior_live["entries"][name]["xyz_in_robot_base_coord_m"],
                    dtype=np.float64,
                ).reshape(3)
                for name in names
            }
            refined_controlled = _strict_point_matrix(
                [refined_by_name[name] for name in controlled_names],
                expected_count=len(controlled_names),
                label="post-prior controlled points",
            )
            current_eef_position, current_eef_quaternion = _eef_base_xyzw(ctx, arm)
            predicted_from_prior = (
                current_eef_position[None, :]
                + (quat_to_mat_xyzw(current_eef_quaternion) @ preplanning_anchors.T).T
            )
            prior_anchor_errors = np.linalg.norm(
                refined_controlled - predicted_from_prior, axis=1
            )
            if float(np.max(prior_anchor_errors)) > START_EEF_ANCHOR_ERROR_M:
                raise ValueError(
                    "post-prior tracked points violate the live EEF anchors"
                )
            prior_geometry = evaluate_rigid_geometry_consistency(
                refined_controlled,
                source_points,
                tolerance_m=rigid_pair_distance_tolerance(float(args["pos_tol"])),
            )
            if not bool(prior_geometry.get("ok")):
                raise ValueError("post-prior tracked rigid geometry changed")
            if reference_names:
                refined_references = _strict_point_matrix(
                    [refined_by_name[name] for name in reference_names],
                    expected_count=len(reference_names),
                    label="post-prior off-hand points",
                )
                reference_sync = _off_hand_reference_consensus(
                    reference_names,
                    refined_references,
                    reference_points,
                    tolerance_m=max(0.020, 2.0 * float(args["pos_tol"])),
                    allow_visual_outlier_suppression=False,
                )
                if not bool(reference_sync.get("ok")):
                    raise ValueError(
                        "off-hand points moved before post-prior source sync"
                    )
            else:
                reference_sync = {"ok": True, "applicable": False}

            source_by_name = refined_by_name
            source_points = refined_controlled
            reference_points = (
                np.asarray(
                    [refined_by_name[name] for name in reference_names],
                    dtype=np.float64,
                )
                if reference_names
                else None
            )
            frozen = _freeze_state(ctx)
            if _sequence(ctx) != post_prior_sequence:
                raise ValueError("observation changed while freezing post-prior source")
            observation_sequence = post_prior_sequence
            if faceto_requested:
                (
                    head_camera_position_robot_base_m,
                    head_camera_quaternion_xyzw,
                ) = _head_camera_pose_x5(
                    ctx,
                    expected_sequence=int(observation_sequence),
                    base_rotation=frozen["base_rotation"],
                )
            if quick_mode:
                planning_relations = _relations_with_quick_constraints(
                    args["relations"],
                    quick_specs,
                    controlled_names=controlled_names,
                    controlled_points=source_points,
                    reference_names=reference_names,
                    reference_points=reference_points,
                    camera_position_robot_base_m=(head_camera_position_robot_base_m),
                    camera_quaternion_xyzw=head_camera_quaternion_xyzw,
                )
            preplanning_source_sync = {
                "ok": True,
                "applicable": True,
                "source_observation_sequence": prior_sequence,
                "accepted_observation_sequence": post_prior_sequence,
                "initial_observation_sequence": initial_observation_sequence,
                "planning_observation_sequence": post_prior_sequence,
                "max_eef_anchor_error_m": float(np.max(prior_anchor_errors)),
                "eef_anchor_tolerance_m": START_EEF_ANCHOR_ERROR_M,
                "rigid_geometry": json_ready(prior_geometry),
                "off_hand_reference": json_ready(reference_sync),
                "source_refrozen_before_planner": True,
                "planning_source_rebased": True,
                "planning_source_points_robot_base_m": (
                    refined_controlled.astype(float).tolist()
                ),
                "all_points_frozen_from_one_observation": True,
                "stable_steps": sync_stable_steps,
                "required_stable_steps": START_STABLE_STEPS,
                "last_handoff_observation_dt_s": last_sync_observation_dt_s,
                "handoff_observation_dt_samples_s": [
                    float(value) for value in sync_observation_dt_samples_s
                ],
                "max_handoff_observation_dt_s": (
                    None
                    if not sync_observation_dt_samples_s
                    else float(max(sync_observation_dt_samples_s))
                ),
                "trace": json_ready(sync_trace),
                "stale_tracker_coordinates_used_while_waiting": False,
            }
        except (KeyError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            _set_failure(
                ctx,
                result_writer,
                stage="preplanning source synchronization",
                error=str(exc),
                args=args,
                details={
                    "selected_arm": arm,
                    "on_hand_eef_observation_prior": json_ready(eef_prior_activation),
                },
            )
            yield ctx.world.hold_action()
            return
    elif args["execution_mode"] == "plan":
        preplanning_source_sync = {
            "ok": True,
            "applicable": True,
            "reason": "plan_mode_current_synchronized_tracker_frame",
            "planning_source_rebased": False,
            "initial_observation_sequence": int(initial_observation_sequence),
            "planning_observation_sequence": int(observation_sequence),
            "planning_source_points_robot_base_m": (
                source_points.astype(float).tolist()
            ),
            "all_points_frozen_from_one_observation": True,
            "post_prior_refreeze_required_for_direct_exec_mode": True,
        }
    elif bool(registered_source_reconciliation.get("applied")):
        preplanning_source_sync = {
            "ok": True,
            "applicable": True,
            "reason": "registered_eef_anchor_outlier_recovery",
            "planning_source_rebased": True,
            "initial_observation_sequence": int(initial_observation_sequence),
            "planning_observation_sequence": int(observation_sequence),
            "planning_source_points_robot_base_m": (
                source_points.astype(float).tolist()
            ),
            "live_validation_points_from_one_observation": True,
            "planning_source_uses_registration_rgbd_and_current_proprio": True,
            "predictions_published_as_observation": False,
        }
    preplanning_eef_prior_activation = json_ready(eef_prior_activation)
    frozen_plan_capture: dict[str, Any] | None = None
    plan_overlap_filter_context: dict[str, Any] | None = None
    if args["execution_mode"] == "plan":
        try:
            snapshot, frozen_plan_capture = _freeze_plan_capture_x5(
                ctx,
                session_id=session_id,
                registration_image_id=image_id,
                observation_sequence=observation_sequence,
                frozen=frozen,
            )
            plan_overlap_filter_context = _plan_overlap_context_x5(
                snapshot, frozen=frozen
            )
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            _set_failure(
                ctx,
                result_writer,
                stage="plan capture",
                error=str(exc),
                args=args,
                details={
                    "selected_arm": arm,
                    "registered_on_hand_source_reconciliation": json_ready(
                        registered_source_reconciliation
                    ),
                },
            )
            yield ctx.world.hold_action()
            return
    cancel_event = threading.Event()
    planning_hold_steps = preplanning_hold_steps
    planning_defer_polls = 0
    planning_progress: dict[str, Any] = {}
    planner = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="official-v2-move-tracked-x5"
    )
    future = planner.submit(
        _plan_frozen,
        state=frozen["state"],
        arm=arm,
        q_start=frozen["arm_q"][arm].copy(),
        source_points=source_points.copy(),
        target_points=(
            list(planning_target_points)
            if planning_target_points is not None
            else list(args["points"])
        ),
        relations=list(args["relations"]),
        inequalities=list(args["inequalities"]),
        quick_constraint=(quick_spec if quick_mode else None),
        quick_constraints=(args.get("quick_constraints") if quick_mode else None),
        off_hand_points=(
            reference_points.copy() if reference_points is not None else None
        ),
        off_hand_target_points=(
            list(fixed_target_points)
            if reference_points is not None and fixed_target_points is not None
            else None
        ),
        on_hand_names=controlled_names,
        off_hand_names=reference_names,
        head_camera_position_robot_base_m=(head_camera_position_robot_base_m),
        head_camera_quaternion_xyzw=head_camera_quaternion_xyzw,
        pos_tol_m=float(args["pos_tol"]),
        ori_tol_deg=float(args["ori_tol_deg"]),
        max_steps=dense_planning_max_steps,
        overlap_filter_context=plan_overlap_filter_context,
        cancel_event=cancel_event,
        planning_deadline_monotonic=planning_deadline,
        hard_deadline_monotonic=planning_deadline,
        isolate_process=args["execution_mode"] == "plan",
        progress_callback=(
            lambda report: (
                planning_progress.update({"report": report})
                if args["execution_mode"] == "plan"
                else None
            )
        ),
    )
    try:
        while True:
            ctx.raise_if_cancelled("move_tracked_point frozen local planning")
            if str(ctx.world.episode_id()) != episode_id:
                raise RuntimeError("episode_changed_during_planning")
            if time.monotonic() >= planning_deadline:
                raise TimeoutError("frozen local planning timed out")
            try:
                planned = future.result(timeout=0.005)
                break
            except concurrent.futures.TimeoutError:
                planning_defer_polls += 1
                yield ctx.world.defer_action("move_tracked_point frozen local planning")
        if not bool(planned.get("ok")):
            raise ValueError(
                str(planned.get("error") or "tracked-point planning failed")
            )
        endpoint = planned["endpoint"]
        path = {
            **planned["path"],
            "planning_observation_action_loop": bool(planning_hold_steps),
            "planning_hold_steps": planning_hold_steps,
            "planning_defer_polls": planning_defer_polls,
            "planning_deferred_without_evaluator_action": bool(planning_defer_polls),
            "planning_elapsed_s": time.monotonic() - planning_started,
            "planning_budget_s": planning_budget_s,
            "planning_deadline_monotonic": planning_deadline,
            "planning_budget_exhausted": False,
            "planning_observation_sequence_start": observation_sequence,
            "planning_observation_sequence_end": _sequence(ctx),
            "planning_diagnostics": json_ready(planning_progress.get("report", {})),
            "frozen_plan_capture": json_ready(frozen_plan_capture),
            "registered_on_hand_source_reconciliation": json_ready(
                registered_source_reconciliation
            ),
        }
        record = _trajectory_record(
            args=args,
            frozen=frozen,
            episode_id=episode_id,
            observation_sequence=observation_sequence,
            session_id=session_id,
            image_id=image_id,
            names=controlled_names,
            nearest_name=nearest_name,
            source_points=source_points,
            arm=arm,
            endpoint=endpoint,
            path=path,
            tracking_names=names,
            controlled_names=controlled_names,
            reference_names=reference_names,
            reference_points=reference_points,
            quick_constraint=quick_spec if quick_mode else None,
            quick_constraints=(args.get("quick_constraints") if quick_mode else None),
            planning_target_points=list(
                planned.get("planning_target_points") or planning_target_points
            ),
            fixed_target_points=list(
                planned.get("off_hand_target_points") or fixed_target_points
            ),
            planning_relations=list(
                planned.get("planning_relations") or planning_relations
            ),
            head_camera_position_robot_base_m=(head_camera_position_robot_base_m),
            head_camera_quaternion_xyzw=head_camera_quaternion_xyzw,
            plan_motion_epoch=int(ctx.world.motion_epoch()),
            registered_source_reconciliation=registered_source_reconciliation,
            predefined_arm_selection=predefined_arm_selection,
        )
        saved = ctx.plans.save(session_id, record)
        plan_id = str(saved["plan_id"])
        trajectory = ctx.plans.load(session_id, plan_id)
        _validate_loaded_plan(trajectory, session_id=session_id)
        plan_path = str(trajectory["plan_record_path"])
        digest = str(trajectory["integrity"]["digest"])
        if args["execution_mode"] == "plan":
            plan_result = _plan_preview_result_x5(ctx, trajectory=trajectory)
            plan_result.update(
                {
                    "preplanning_on_hand_eef_observation_prior": json_ready(
                        eef_prior_activation
                    ),
                    "preplanning_source_sync": json_ready(preplanning_source_sync),
                    "rigid_pair_activation": json_ready(rigid_pair_activation),
                    "rigid_pair_activation_confirmation": json_ready(
                        rigid_pair_activation_confirmation
                    ),
                    "rigid_pair_activation_hold_steps": int(
                        rigid_pair_activation_hold_steps
                    ),
                    "prior_rigid_pair_cleanup": json_ready(prior_rigid_pair_cleanup),
                    "failed_rigid_pair_cleanup": json_ready(failed_rigid_pair_cleanup),
                    "rigid_pair_bootstrap_eef_observation_prior": json_ready(
                        rigid_pair_bootstrap_eef_prior
                    ),
                    "registered_on_hand_source_reconciliation": json_ready(
                        registered_source_reconciliation
                    ),
                }
            )
            result_writer(ctx, "move_tracked_point", plan_result)
            yield ctx.world.hold_action()
            return
        planned_anchors = _strict_point_matrix(
            trajectory["target"]["anchors_eef_m"],
            expected_count=len(controlled_names),
            label="move-tracked trajectory anchors",
        )
        if (
            manager_backed_tracking
            and manager.visual_point_count(controlled_names) > 0
            and args["execution_mode"] == "exec"
        ):
            eef_prior_activation = manager.activate_on_hand_eef_observation_prior(
                controlled_names,
                episode_id=episode_id,
                session_id=session_id,
                image_id=image_id,
                arm=arm,
                anchors_eef_m=planned_anchors.tolist(),
            )
            if not isinstance(eef_prior_activation, dict) or not bool(
                eef_prior_activation.get("ok", True)
            ):
                raise ValueError(
                    "planned on-hand EEF observation prior was not activated"
                )
            prior_lease_id = str(eef_prior_activation.get("lease_id") or "").strip()
            if not prior_lease_id:
                raise ValueError(
                    "planned on-hand EEF observation prior has no lease_id"
                )
            eef_prior_activation["applicable"] = True
            lifecycle["eef_prior_lease_id"] = prior_lease_id
    except Exception as exc:
        cancel_event.set()
        future.cancel()
        stage = (
            "timeout"
            if isinstance(exc, TimeoutError)
            else "cancellation"
            if "cancel" in str(exc).lower()
            else "planning"
        )
        _set_failure(
            ctx,
            result_writer,
            stage=stage,
            error=str(exc),
            args=args,
            details={
                "selected_arm": arm,
                "eef_mean_distance_m": distances,
                "planning_observation_action_loop": bool(planning_hold_steps),
                "planning_hold_steps": planning_hold_steps,
                "planning_defer_polls": planning_defer_polls,
                "planning_deferred_without_evaluator_action": bool(
                    planning_defer_polls
                ),
                "endpoint_path_candidate_failures": json_ready(
                    locals().get("planned", {}).get("failures", [])
                ),
            },
        )
        yield ctx.world.hold_action()
        return
    finally:
        planner.shutdown(wait=False, cancel_futures=True)

    post_planning_source_replans: list[dict[str, Any]] = []
    total_deadline = planning_started + float(args["timeout_s"])
    while True:
        start_generator = _wait_for_stable_start_x5(
            ctx,
            manager=manager,
            frozen=frozen,
            names=names,
            controlled_names=controlled_names,
            reference_names=reference_names,
            source_points=source_points,
            reference_points=reference_points,
            episode_id=episode_id,
            session_id=session_id,
            image_id=image_id,
            arm=arm,
            pos_tol_m=float(args["pos_tol"]),
            planned_anchors=planned_anchors,
            anchor_tolerance_m=min(
                MAX_ANCHOR_ERROR_M,
                float(
                    eef_prior_activation.get(
                        "max_point_error_m", START_EEF_ANCHOR_ERROR_M
                    )
                ),
            ),
            geometry_tolerance_m=rigid_pair_distance_tolerance(float(args["pos_tol"])),
            eef_prior_activation=eef_prior_activation,
            deadline_monotonic=total_deadline,
        )
        try:
            while True:
                yield next(start_generator)
        except StopIteration as stopped:
            start_validation = stopped.value
        except ToolCancelled as exc:
            _set_failure(
                ctx,
                result_writer,
                stage="cancellation",
                error=str(exc),
                args=args,
                details={
                    "arm": arm,
                    "plan_id": plan_id,
                    "trajectory_digest": digest,
                    "cancellation_evaluator_action_emitted": False,
                },
            )
            yield ctx.world.hold_action()
            return

        off_hand_start_validation = dict(
            start_validation.get("off_hand_start_validation") or {}
        )
        if bool(start_validation.get("ok")):
            break
        reason = str(start_validation.get("reason") or "start_state_not_stable")
        can_rebase = bool(
            reason == "tracked_points_moved_after_source_sync"
            and (
                eef_prior_activation.get("active")
                or eef_prior_activation.get("applicable")
            )
            and len(controlled_names) >= FINAL_VISUAL_MIN_POINTS
            and len(post_planning_source_replans) < POST_PLAN_REBASE_MAX_ROUNDS
        )
        if not can_rebase:
            break

        old_plan_id = str(plan_id)
        old_digest = str(digest)
        replan_round = len(post_planning_source_replans) + 1
        diagnostic: dict[str, Any] = {
            "round": replan_round,
            "trigger": reason,
            "trigger_source_drift_m": start_validation.get("last_source_drift_m"),
            "trigger_eef_anchor_error_m": start_validation.get(
                "last_eef_anchor_error_m"
            ),
            "discarded_plan_id": old_plan_id,
            "discarded_trajectory_digest": old_digest,
            "discarded_plan_executed": False,
        }
        replan_executor = None
        replan_future = None
        replan_cancel = threading.Event()
        try:
            rebase_drift = float(
                start_validation.get("last_source_drift_m", float("inf"))
            )
            rebase_anchor_error = float(
                start_validation.get("last_eef_anchor_error_m", float("inf"))
            )
            rebase_geometry = start_validation.get("last_rigid_geometry_consistency")
            if (
                not math.isfinite(rebase_drift)
                or rebase_drift > POST_PLAN_REBASE_MAX_SOURCE_DRIFT_M
            ):
                raise RuntimeError(
                    "post-plan RGB-D refinement exceeds the bounded rebase "
                    f"envelope ({rebase_drift:.4f}m > "
                    f"{POST_PLAN_REBASE_MAX_SOURCE_DRIFT_M:.4f}m)"
                )
            if (
                not math.isfinite(rebase_anchor_error)
                or rebase_anchor_error > POST_PLAN_REBASE_MAX_SOURCE_DRIFT_M
            ):
                raise RuntimeError(
                    "post-plan RGB-D bundle is not close enough to the live "
                    "EEF for a synchronized rebase"
                )
            if not isinstance(rebase_geometry, Mapping) or not bool(
                rebase_geometry.get("ok")
            ):
                raise RuntimeError("post-plan RGB-D bundle changed rigid geometry")
            if (
                total_deadline - time.monotonic()
                <= POST_PLAN_REBASE_MIN_EXECUTION_RESERVE_S
            ):
                raise RuntimeError(
                    "not enough tool timeout remains for a synchronized "
                    "post-plan rebase"
                )

            rebase_sequence = int(start_validation.get("observation_sequence_end", -1))
            rebase_live = _live(ctx, names, episode_id, manager=manager)
            if not bool(rebase_live.get("ok")):
                raise RuntimeError(
                    "tracked points became unavailable during post-plan "
                    f"rebase: {rebase_live.get('reason') or 'unknown'}"
                )
            if int(rebase_live.get("observation_sequence", -1)) != rebase_sequence:
                raise RuntimeError(
                    "post-plan rebase points are not synchronized to proprioception"
                )
            if (
                rebase_live.get("session_id") != session_id
                or rebase_live.get("image_id") != image_id
            ):
                raise RuntimeError(
                    "tracked-point registration changed during post-plan rebase"
                )
            rebase_matrix = _strict_point_matrix(
                [
                    rebase_live["entries"][name]["xyz_in_robot_base_coord_m"]
                    for name in names
                ],
                expected_count=len(names),
                label="post-plan synchronized tracked source point data",
            )
            rebase_by_name = {
                name: rebase_matrix[index].copy() for index, name in enumerate(names)
            }
            rebase_source_points = np.asarray(
                [rebase_by_name[name] for name in controlled_names],
                dtype=np.float64,
            )
            gate_points = _strict_point_matrix(
                start_validation.get("last_live_points_robot_base_m"),
                expected_count=len(controlled_names),
                label="post-plan handoff point data",
            )
            if (
                float(
                    np.max(np.linalg.norm(rebase_source_points - gate_points, axis=1))
                )
                > 1.0e-6
            ):
                raise RuntimeError(
                    "post-plan rebase point bundle changed after the stable "
                    "handoff sample"
                )

            rebase_references = (
                np.asarray(
                    [rebase_by_name[name] for name in reference_names],
                    dtype=np.float64,
                )
                if reference_names
                else None
            )
            reference_validation = {"ok": True, "applicable": False}
            if reference_names:
                reference_validation = _off_hand_reference_consensus(
                    reference_names,
                    rebase_references,
                    reference_points,
                    tolerance_m=max(0.020, 2.0 * float(args["pos_tol"])),
                    allow_visual_outlier_suppression=False,
                )
                if not bool(reference_validation.get("ok")):
                    raise RuntimeError(
                        str(
                            reference_validation.get("error")
                            or "off-hand reference points moved before post-plan rebase"
                        )
                    )

            frozen = _freeze_state(ctx)
            if _sequence(ctx) != rebase_sequence:
                raise RuntimeError(
                    "evaluator observation changed while freezing the post-plan rebase"
                )
            source_by_name = rebase_by_name
            source_points = rebase_source_points
            reference_points = rebase_references
            observation_sequence = rebase_sequence
            if faceto_requested:
                (
                    head_camera_position_robot_base_m,
                    head_camera_quaternion_xyzw,
                ) = _head_camera_pose_x5(
                    ctx,
                    expected_sequence=int(observation_sequence),
                    base_rotation=frozen["base_rotation"],
                )
            distances = {
                side: float(
                    np.mean(
                        [
                            np.linalg.norm(
                                source_by_name[name] - frozen["eef_base_wxyz"][side][:3]
                            )
                            for name in controlled_names
                        ]
                    )
                )
                for side in ("left", "right")
            }
            nearest_name = min(
                controlled_names,
                key=lambda name: float(
                    np.linalg.norm(
                        source_by_name[name] - frozen["eef_base_wxyz"][arm][:3]
                    )
                ),
            )
            if quick_mode:
                planning_relations = _relations_with_quick_constraints(
                    args["relations"],
                    quick_specs,
                    controlled_names=controlled_names,
                    controlled_points=source_points,
                    reference_names=reference_names,
                    reference_points=reference_points,
                    camera_position_robot_base_m=(head_camera_position_robot_base_m),
                    camera_quaternion_xyzw=head_camera_quaternion_xyzw,
                )
            current_position, current_quaternion = _eef_base_xyzw(ctx, arm)
            rebase_anchors = (
                quat_to_mat_xyzw(current_quaternion).T
                @ (source_points - current_position[None, :]).T
            ).T
            eef_prior_activation = manager.activate_on_hand_eef_observation_prior(
                controlled_names,
                episode_id=episode_id,
                session_id=session_id,
                image_id=image_id,
                arm=arm,
                anchors_eef_m=rebase_anchors.tolist(),
            )
            if not isinstance(eef_prior_activation, dict) or not bool(
                eef_prior_activation.get("ok", True)
            ):
                raise RuntimeError(
                    "rebased on-hand EEF observation prior was not activated"
                )
            rebase_lease = str(eef_prior_activation.get("lease_id") or "").strip()
            if not rebase_lease:
                raise RuntimeError(
                    "rebased on-hand EEF observation prior has no lease_id"
                )
            eef_prior_activation["applicable"] = True
            lifecycle["eef_prior_lease_id"] = rebase_lease

            round_deadline = min(
                planning_deadline,
                total_deadline - POST_PLAN_REBASE_MIN_EXECUTION_RESERVE_S,
            )
            if time.monotonic() >= round_deadline:
                raise TimeoutError(
                    "post-plan synchronized replan has no remaining planning budget"
                )
            replan_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="official-v2-move-tracked-x5-rebase",
            )
            replan_future = replan_executor.submit(
                _plan_frozen,
                state=frozen["state"],
                arm=arm,
                q_start=frozen["arm_q"][arm].copy(),
                source_points=source_points.copy(),
                target_points=list(planning_target_points),
                relations=list(planning_relations),
                inequalities=list(args["inequalities"]),
                fixed_points=(
                    reference_points.copy() if reference_points is not None else None
                ),
                fixed_target_points=(
                    list(fixed_target_points) if reference_points is not None else None
                ),
                pos_tol_m=float(args["pos_tol"]),
                ori_tol_deg=float(args["ori_tol_deg"]),
                max_steps=dense_planning_max_steps,
                cancel_event=replan_cancel,
                planning_deadline_monotonic=round_deadline,
            )
            replan_started = time.monotonic()
            replanned = None
            while replanned is None:
                ctx.raise_if_cancelled(
                    "move_tracked_point post-plan synchronized replanning"
                )
                if str(ctx.world.episode_id()) != episode_id:
                    raise RuntimeError("episode_changed_during_post_plan_replanning")
                if time.monotonic() >= round_deadline:
                    raise TimeoutError(
                        "post-plan synchronized local replanning timed out"
                    )
                try:
                    replanned = replan_future.result(timeout=0.005)
                except concurrent.futures.TimeoutError:
                    planning_defer_polls += 1
                    yield ctx.world.defer_action(
                        "move_tracked_point synchronized local replanning"
                    )
            if not bool(replanned.get("ok")):
                raise RuntimeError(
                    str(
                        replanned.get("error")
                        or "post-plan synchronized replanning failed"
                    )
                )
            endpoint = replanned["endpoint"]
            path = {
                **replanned["path"],
                "planning_observation_action_loop": bool(planning_hold_steps),
                "planning_hold_steps": planning_hold_steps,
                "planning_defer_polls": planning_defer_polls,
                "planning_deferred_without_evaluator_action": bool(
                    planning_defer_polls
                ),
                "planning_elapsed_s": time.monotonic() - planning_started,
                "planning_budget_s": planning_budget_s,
                "planning_deadline_monotonic": planning_deadline,
                "planning_budget_exhausted": False,
                "planning_observation_sequence_start": rebase_sequence,
                "planning_observation_sequence_end": _sequence(ctx),
                "post_plan_source_rebase_round": replan_round,
            }
            record = _trajectory_record(
                args=args,
                frozen=frozen,
                episode_id=episode_id,
                observation_sequence=observation_sequence,
                session_id=session_id,
                image_id=image_id,
                names=controlled_names,
                nearest_name=nearest_name,
                source_points=source_points,
                arm=arm,
                endpoint=endpoint,
                path=path,
                tracking_names=names,
                controlled_names=controlled_names,
                reference_names=reference_names,
                reference_points=reference_points,
                quick_constraint=quick_spec if quick_mode else None,
                quick_constraints=(
                    args.get("quick_constraints") if quick_mode else None
                ),
                planning_target_points=planning_target_points,
                fixed_target_points=fixed_target_points,
                planning_relations=planning_relations,
                head_camera_position_robot_base_m=(head_camera_position_robot_base_m),
                head_camera_quaternion_xyzw=head_camera_quaternion_xyzw,
            )
            saved = ctx.plans.save(session_id, record)
            plan_id = str(saved["plan_id"])
            trajectory = ctx.plans.load(session_id, plan_id)
            _validate_loaded_plan(trajectory, session_id=session_id)
            plan_path = str(trajectory["plan_record_path"])
            digest = str(trajectory["integrity"]["digest"])
            planned_anchors = _strict_point_matrix(
                trajectory["target"]["anchors_eef_m"],
                expected_count=len(controlled_names),
                label="post-plan rebased trajectory anchors",
            )
            eef_prior_activation = manager.activate_on_hand_eef_observation_prior(
                controlled_names,
                episode_id=episode_id,
                session_id=session_id,
                image_id=image_id,
                arm=arm,
                anchors_eef_m=planned_anchors.tolist(),
            )
            final_rebase_lease = str(eef_prior_activation.get("lease_id") or "").strip()
            if not bool(eef_prior_activation.get("ok", True)) or not final_rebase_lease:
                raise RuntimeError(
                    "planned rebased on-hand EEF observation prior was not activated"
                )
            eef_prior_activation["applicable"] = True
            lifecycle["eef_prior_lease_id"] = final_rebase_lease
            diagnostic.update(
                {
                    "ok": True,
                    "source_observation_sequence": rebase_sequence,
                    "source_points_robot_base_m": source_points.tolist(),
                    "off_hand_reference_validation": json_ready(reference_validation),
                    "planning_elapsed_s": time.monotonic() - replan_started,
                    "new_plan_id": plan_id,
                    "new_trajectory_digest": digest,
                    "new_plan_schema_validated": True,
                    "new_plan_x5_joint_limits_validated": True,
                    "new_plan_j8_locked_zero": True,
                }
            )
            post_planning_source_replans.append(diagnostic)
            continue
        except Exception as exc:
            replan_cancel.set()
            if replan_future is not None:
                replan_future.cancel()
            diagnostic.update(
                {
                    "ok": False,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
            )
            post_planning_source_replans.append(diagnostic)
            start_validation = {
                **start_validation,
                "ok": False,
                "reason": "post_planning_source_replan_failed",
                "error": (
                    "tracked points changed coherently after planning, but "
                    f"the synchronized replacement plan failed: {exc}"
                ),
            }
            break
        finally:
            if replan_executor is not None:
                replan_executor.shutdown(wait=False, cancel_futures=True)

    if not bool(start_validation.get("ok")):
        reason = str(start_validation.get("reason") or "start_state_not_stable")
        _set_failure(
            ctx,
            result_writer,
            stage=("timeout" if reason == "timeout" else "start-state validation"),
            error=str(start_validation.get("error") or reason),
            args=args,
            details={
                "plan_id": plan_id,
                "trajectory_digest": digest,
                "start_state_validation": start_validation,
                "off_hand_start_validation": off_hand_start_validation,
                "post_planning_source_replans": json_ready(
                    post_planning_source_replans
                ),
                "post_planning_source_replan_count": len(post_planning_source_replans),
            },
        )
        yield ctx.world.hold_action()
        return

    anchors = np.asarray(trajectory["target"]["anchors_eef_m"], dtype=np.float64)
    target_faceto_camera = trajectory["target"].get("faceto_camera_pose")
    target_camera_position = (
        target_faceto_camera.get("position_robot_base_m")
        if isinstance(target_faceto_camera, Mapping)
        else None
    )
    target_camera_quaternion = (
        target_faceto_camera.get("quaternion_xyzw")
        if isinstance(target_faceto_camera, Mapping)
        else None
    )
    anchor_tolerance = float(trajectory["tracking"]["rigid_anchor_tolerance_m"])
    geometry_tolerance = float(
        trajectory["tracking"]["rigid_pair_distance_tolerance_m"]
    )
    dense_execution_policy = _trajectory_uses_dense_execution(trajectory)
    # ``max_steps`` remains part of the public contract and is still validated
    # at the boundary.  The active RoboDojo port itself has no small evaluator
    # action ceiling, however, so dense replay must not stop at (for example)
    # a caller's historical 40/80-step hint when the signed path contains more
    # waypoints.  Expand only the internal execution allowance; the wall-clock
    # ``timeout_s`` and all per-action tracking/cancellation checks remain hard
    # limits.  Keep the requested value in the report for auditability.
    requested_max_steps = requested_public_max_steps
    dense_step_budget_expanded = False
    if dense_execution_policy and args["execution_mode"] == "exec":
        dense_step_budget = max(
            dense_planning_max_steps,
            len(trajectory.get("waypoints") or [])
            + FINAL_MAX_STABILITY_STEPS
            + STABLE_STEPS
            + FINAL_PRECISION_RESERVE_STEPS,
        )
        if dense_step_budget > requested_max_steps:
            args["max_steps"] = int(dense_step_budget)
            dense_step_budget_expanded = True
    handoff_q = np.asarray(ctx.world.arm_qpos_list(arm), dtype=np.float64)
    terminal_precision_reserve_steps = min(
        FINAL_PRECISION_RESERVE_STEPS,
        max(0, int(args["max_steps"]) - STABLE_STEPS - 1),
    )
    try:
        execution_cadence = _execution_cadence_estimate(
            start_validation=start_validation,
            preplanning_source_sync=preplanning_source_sync,
            planner=trajectory.get("planner"),
        )
        raw_observation_dt = execution_cadence.get("observation_dt_s")
        observation_dt = (
            float(raw_observation_dt) if raw_observation_dt is not None else None
        )
        schedule_available_time_s = float(args["timeout_s"]) - (
            time.monotonic() - planning_started
        )
        if observation_dt is not None:
            schedule_available_time_s -= float(
                (STABLE_STEPS + terminal_precision_reserve_steps) * observation_dt
            )
        if schedule_available_time_s <= 0.0:
            schedule = {
                "ok": False,
                "reason": "execution_budget_infeasible",
                "error": "no evaluator time remains after the final-stability reserve",
            }
        else:
            schedule = _prepare_execution_schedule(
                trajectory=trajectory,
                state=frozen["state"],
                arm=arm,
                start_q=handoff_q,
                max_steps=max(
                    0,
                    int(args["max_steps"])
                    - STABLE_STEPS
                    - terminal_precision_reserve_steps,
                ),
                available_time_s=schedule_available_time_s,
                observation_dt_s=observation_dt,
                visual_tracked_point_count=manager.visual_point_count(controlled_names),
                requested_public_max_steps=requested_public_max_steps,
            )
            schedule["cadence_estimate"] = json_ready(execution_cadence)
        if not bool(schedule.get("ok")):
            raise ValueError(str(schedule.get("error") or "execution schedule failed"))
        execution_waypoints, execution_bridge = _execution_bridge(
            handoff_q,
            schedule["waypoints"],
            max_step_rad=float(schedule["execution_max_joint_step_rad"]),
        )
        if len(execution_waypoints) != int(
            schedule["predicted_execution_waypoint_count"]
        ):
            raise ValueError(
                "execution bridge expansion changed after budget preflight"
            )
        transition_validation = _validate_execution_transitions(
            frozen["state"],
            arm,
            handoff_q,
            execution_waypoints,
        )
        schedule["transition_validation"] = transition_validation
        schedule["bridge"] = execution_bridge
        schedule["signed_waypoint_count"] = len(trajectory["waypoints"])
        schedule["execution_waypoint_count"] = len(execution_waypoints)
        schedule["signed_plan_rewritten"] = False
        schedule["requested_max_steps"] = requested_max_steps
        schedule["effective_max_steps"] = int(args["max_steps"])
        schedule["terminal_precision_reserve_steps"] = int(
            terminal_precision_reserve_steps
        )
        schedule["dense_step_budget_expanded"] = bool(dense_step_budget_expanded)
        schedule["step_budget_policy"] = (
            "port_unbounded_dense_replay"
            if dense_execution_policy
            else "caller_bounded"
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        _set_failure(
            ctx,
            result_writer,
            stage="execution scheduling",
            error=str(exc),
            args=args,
            details={
                "plan_id": plan_id,
                "trajectory_digest": digest,
                "execution_schedule": json_ready(locals().get("schedule", {})),
            },
        )
        yield ctx.world.hold_action()
        return

    motion_trace: list[dict[str, Any]] = []
    tracking_continuity: dict[str, Any] = {
        "observed_steps": 0,
        "temporarily_unobserved_steps": 0,
        "current_unobserved_streak": 0,
        "max_unobserved_streak": 0,
        "reacquisition_count": 0,
        "last_unobserved_reason": None,
        "stale_coordinates_used": False,
        "trajectory_changed_for_reacquisition": False,
        "off_hand_tracking_outlier_steps": 0,
        "off_hand_tracking_outlier_names": [],
        "off_hand_reference_substitution_used": False,
        "controlled_anchor_disturbance_steps": 0,
        "controlled_rigid_geometry_disturbance_steps": 0,
        "max_controlled_anchor_error_m": 0.0,
        "max_controlled_pair_distance_error_m": 0.0,
        "execution_pauses_for_visual_reacquisition": bool(
            len(visual_controlled_names) >= NPOINT_VISUAL_MIN_POINTS
        ),
        "visual_reacquisition_max_hold_steps": int(NPOINT_REACQUISITION_MAX_HOLD_STEPS),
    }
    pause_for_visual_reacquisition = bool(
        len(visual_controlled_names) >= NPOINT_VISUAL_MIN_POINTS
    )
    action_steps = 0
    completed_waypoints = 0
    motion_ok = True
    motion_reason = "trajectory_complete"
    max_observed_tracking_error = 0.0
    max_inactive_arm_drift = 0.0
    tracking_recovery_events: list[dict[str, Any]] = []
    tracker_reacquisition_hold_q: np.ndarray | None = None
    tracker_reacquisition_hold_steps = 0
    tracker_reacquisition_events: list[dict[str, Any]] = []
    gripper_recovery_hold_q: np.ndarray | None = None
    gripper_recovery_fingers: dict[str, list[int]] = {}
    gripper_recovery_steps = 0
    gripper_recovery_streak = 0
    gripper_recovery_stable_steps = 0
    gripper_recovery_events: list[dict[str, Any]] = []
    gripper_recovery_samples: list[dict[str, Any]] = []
    previous_gripper_joint_positions_m: dict[str, float] | None = None
    max_gripper_drift_m = 0.0
    max_gripper_drift_normalized = 0.0
    final_gripper_drift_m = 0.0
    last_gripper_command_report: dict[str, dict[str, Any]] = {}
    last_step_monitor: dict[str, Any] = {}
    for waypoint_index, raw_waypoint in enumerate(execution_waypoints):
        waypoint = np.asarray(raw_waypoint, dtype=np.float64)
        recovery_step = 0
        stall_steps = 0
        pre_action_q = np.asarray(ctx.world.arm_qpos_list(arm), dtype=np.float64)
        previous_error = float(np.linalg.norm(pre_action_q - waypoint, ord=np.inf))
        while True:
            try:
                ctx.raise_if_cancelled("move_tracked_point trajectory execution")
            except ToolCancelled as exc:
                _set_failure(
                    ctx,
                    result_writer,
                    stage="cancellation",
                    error=str(exc),
                    args=args,
                    details={
                        "arm": arm,
                        "plan_id": plan_id,
                        "trajectory_digest": digest,
                        "cancellation_evaluator_action_emitted": False,
                    },
                )
                yield ctx.world.hold_action()
                return
            if str(ctx.world.episode_id()) != episode_id:
                motion_ok = False
                motion_reason = "episode_changed"
                break
            if action_steps >= int(
                args["max_steps"]
            ) - STABLE_STEPS or time.monotonic() - planning_started >= float(
                args["timeout_s"]
            ):
                motion_ok = False
                motion_reason = "timeout"
                break
            command_is_tracker_reacquisition_hold = bool(
                tracker_reacquisition_hold_q is not None
            )
            if (
                command_is_tracker_reacquisition_hold
                and tracker_reacquisition_hold_steps
                >= NPOINT_REACQUISITION_MAX_HOLD_STEPS
            ):
                tracker_reacquisition_events.append(
                    {
                        "event": "reacquisition_exhausted",
                        "waypoint_index": int(waypoint_index + 1),
                        "observation_sequence": int(_sequence(ctx)),
                        "hold_steps": int(tracker_reacquisition_hold_steps),
                        "max_hold_steps": int(NPOINT_REACQUISITION_MAX_HOLD_STEPS),
                    }
                )
                motion_ok = False
                motion_reason = "tracked_point_reacquisition_failed"
                break
            command_is_gripper_recovery_hold = bool(
                not command_is_tracker_reacquisition_hold
                and gripper_recovery_hold_q is not None
            )
            command_q = (
                tracker_reacquisition_hold_q.copy()
                if command_is_tracker_reacquisition_hold
                else (
                    gripper_recovery_hold_q.copy()
                    if command_is_gripper_recovery_hold
                    else _gentle_joint_command(
                        np.asarray(
                            ctx.world.arm_qpos_list(arm),
                            dtype=np.float64,
                        ),
                        waypoint,
                        max_step_rad=EXECUTION_MAX_COMMAND_DELTA_RAD,
                    )
                )
            )
            gripper_targets, last_gripper_command_report = _x5_gripper_hold_targets(
                ctx, frozen
            )
            sequence_before = _sequence(ctx)
            yield ctx.world.make_joint_action(
                left_arm=(
                    command_q.tolist()
                    if arm == "left"
                    else frozen["arm_q"]["left"].tolist()
                ),
                left_gripper=gripper_targets["left"],
                right_arm=(
                    command_q.tolist()
                    if arm == "right"
                    else frozen["arm_q"]["right"].tolist()
                ),
                right_gripper=gripper_targets["right"],
            )
            if command_is_tracker_reacquisition_hold:
                tracker_reacquisition_hold_steps += 1
            action_steps += 1
            sequence_after = _sequence(ctx)
            observed_q = np.asarray(ctx.world.arm_qpos_list(arm), dtype=np.float64)
            tracking_error = float(np.linalg.norm(observed_q - command_q, ord=np.inf))
            waypoint_tracking_error = float(
                np.linalg.norm(observed_q - waypoint, ord=np.inf)
            )
            inactive_arm = "right" if arm == "left" else "left"
            inactive_error = float(
                np.linalg.norm(
                    np.asarray(ctx.world.arm_qpos_list(inactive_arm), dtype=np.float64)
                    - frozen["arm_q"][inactive_arm],
                    ord=np.inf,
                )
            )
            max_observed_tracking_error = max(
                max_observed_tracking_error,
                tracking_error,
            )
            max_inactive_arm_drift = max(
                max_inactive_arm_drift,
                inactive_error,
            )
            current_gripper = {
                side: np.asarray(
                    ctx.world.gripper_qpos_list(side), dtype=np.float64
                ).reshape(1)
                for side in ("left", "right")
            }
            gripper_error_report = _x5_gripper_capture_error(
                current_gripper,
                frozen["gripper_q"],
            )
            gripper_error = float(gripper_error_report["max_normalized_error"])
            gripper_error_m = float(gripper_error_report["max_error_m"])
            max_gripper_drift_normalized = max(
                max_gripper_drift_normalized,
                gripper_error,
            )
            max_gripper_drift_m = max(max_gripper_drift_m, gripper_error_m)
            final_gripper_drift_m = gripper_error_m
            current_gripper_joint_positions_m = {
                side: float(
                    gripper_error_report["per_side"][side]["current_joint_position_m"]
                )
                for side in ("left", "right")
            }
            gripper_position_steps_m = {
                side: (
                    float("inf")
                    if previous_gripper_joint_positions_m is None
                    else abs(
                        current_gripper_joint_positions_m[side]
                        - previous_gripper_joint_positions_m[side]
                    )
                )
                for side in ("left", "right")
            }
            gripper_position_step_m = max(gripper_position_steps_m.values())
            previous_gripper_joint_positions_m = dict(current_gripper_joint_positions_m)
            monitor_reason, monitor = _monitor(
                ctx,
                manager=manager,
                names=controlled_names,
                episode_id=episode_id,
                session_id=session_id,
                image_id=image_id,
                anchors=anchors,
                arm=arm,
                tolerance_m=anchor_tolerance,
                geometry_tolerance_m=geometry_tolerance,
                tracking_names=names,
                reference_names=reference_names,
                reference_points=reference_points,
                quick_constraint=quick_spec if quick_mode else None,
                quick_constraints=(
                    args.get("quick_constraints") if quick_mode else None
                ),
                position_tolerance_m=float(args["pos_tol"]),
                orientation_tolerance_deg=float(args["ori_tol_deg"]),
                camera_position_robot_base_m=target_camera_position,
                camera_quaternion_xyzw=target_camera_quaternion,
                allow_visual_outlier_suppression=True,
            )
            temporarily_unobserved = _temporary_tracking_failure(monitor_reason)
            rigid_geometry_disturbed = bool(
                monitor_reason == "tracked_rigid_geometry_changed"
            )
            anchor_disturbed = bool(
                monitor_reason == "tracked_points_not_rigid_with_selected_eef"
            )
            try:
                maximum_anchor_error = float(monitor.get("max_anchor_error_m", 0.0))
            except (TypeError, ValueError):
                maximum_anchor_error = 0.0
            if math.isfinite(maximum_anchor_error):
                tracking_continuity["max_controlled_anchor_error_m"] = max(
                    float(tracking_continuity["max_controlled_anchor_error_m"]),
                    maximum_anchor_error,
                )
            rigid_report = monitor.get("rigid_geometry") or {}
            try:
                maximum_pair_error = float(
                    rigid_report.get("max_pair_distance_error_m", 0.0)
                )
            except (TypeError, ValueError):
                maximum_pair_error = 0.0
            if math.isfinite(maximum_pair_error):
                tracking_continuity["max_controlled_pair_distance_error_m"] = max(
                    float(tracking_continuity["max_controlled_pair_distance_error_m"]),
                    maximum_pair_error,
                )
            if anchor_disturbed:
                tracking_continuity["controlled_anchor_disturbance_steps"] += 1
            if rigid_geometry_disturbed:
                tracking_continuity["controlled_rigid_geometry_disturbance_steps"] += 1
            if temporarily_unobserved:
                tracker_reason = str(monitor_reason or "tracked_point_unavailable")
                tracking_continuity["temporarily_unobserved_steps"] += 1
                tracking_continuity["current_unobserved_streak"] += 1
                tracking_continuity["max_unobserved_streak"] = max(
                    int(tracking_continuity["max_unobserved_streak"]),
                    int(tracking_continuity["current_unobserved_streak"]),
                )
                tracking_continuity["last_unobserved_reason"] = tracker_reason
                monitor.update(
                    {
                        "tracker_reason": tracker_reason,
                        "temporarily_unobserved": bool(pause_for_visual_reacquisition),
                        "intermediate_visual_measurement_unavailable": True,
                        "trajectory_continues_from_immutable_plan": bool(
                            not pause_for_visual_reacquisition
                        ),
                        "trajectory_paused_for_reacquisition": bool(
                            pause_for_visual_reacquisition
                        ),
                        "final_reacquisition_required": True,
                    }
                )
                monitor_reason = None
            elif rigid_geometry_disturbed:
                tracking_continuity["temporarily_unobserved_steps"] += 1
                tracking_continuity["current_unobserved_streak"] += 1
                tracking_continuity["max_unobserved_streak"] = max(
                    int(tracking_continuity["max_unobserved_streak"]),
                    int(tracking_continuity["current_unobserved_streak"]),
                )
                tracking_continuity["last_unobserved_reason"] = (
                    "tracked_point_rigid_geometry_inconsistent"
                )
                monitor.update(
                    {
                        "tracker_reason": ("tracked_point_rigid_geometry_inconsistent"),
                        "temporarily_unobserved": bool(pause_for_visual_reacquisition),
                        "intermediate_visual_measurement_unavailable": False,
                        "intermediate_visual_geometry_inconsistent": True,
                        "trajectory_continues_from_immutable_plan": bool(
                            not pause_for_visual_reacquisition
                        ),
                        "trajectory_paused_for_reacquisition": bool(
                            pause_for_visual_reacquisition
                        ),
                        "final_reacquisition_required": True,
                    }
                )
                monitor_reason = None
            elif anchor_disturbed:
                # Anchor disturbances do not rewrite or stop the signed path.
                # The synchronized terminal verifier remains authoritative.
                monitor.update(
                    {
                        "intermediate_anchor_disturbance": True,
                        "trajectory_continues_from_immutable_plan": True,
                        "terminal_revalidation_required": True,
                    }
                )
                monitor_reason = None
            else:
                if int(tracking_continuity["current_unobserved_streak"]) > 0:
                    tracking_continuity["reacquisition_count"] += 1
                tracking_continuity["current_unobserved_streak"] = 0
                tracking_continuity["observed_steps"] += 1
                reference_report = monitor.get("off_hand_reference") or {}
                if bool(reference_report.get("tracking_outlier_suppressed")):
                    tracking_continuity["off_hand_tracking_outlier_steps"] += 1
                    tracking_continuity["off_hand_reference_substitution_used"] = True
                    known_outliers = set(
                        tracking_continuity["off_hand_tracking_outlier_names"]
                    )
                    known_outliers.update(
                        str(name) for name in reference_report.get("outlier_names", [])
                    )
                    tracking_continuity["off_hand_tracking_outlier_names"] = sorted(
                        known_outliers
                    )
            last_step_monitor = json_ready(monitor)
            trace_item = {
                "action_step": action_steps,
                "waypoint_index": waypoint_index,
                "observation_sequence_before": sequence_before,
                "observation_sequence_after": sequence_after,
                "command_role": (
                    "tracker_reacquisition_hold"
                    if command_is_tracker_reacquisition_hold
                    else (
                        "x5_gripper_recovery_hold"
                        if command_is_gripper_recovery_hold
                        else "trajectory_waypoint"
                    )
                ),
                "tracking_error_inf_rad": tracking_error,
                "waypoint_tracking_error_inf_rad": waypoint_tracking_error,
                "inactive_arm_error_inf_rad": inactive_error,
                "gripper_capture_error_normalized": gripper_error,
                "gripper_capture_error_m": gripper_error_m,
                "gripper_position_step_m": (
                    None
                    if not math.isfinite(gripper_position_step_m)
                    else gripper_position_step_m
                ),
                "gripper_capture_error": json_ready(gripper_error_report),
                "gripper_command": json_ready(last_gripper_command_report),
                "recovery_step": recovery_step,
                "gripper_recovery_streak": gripper_recovery_streak,
                "temporarily_unobserved": temporarily_unobserved,
                "monitor": monitor,
            }
            motion_trace.append(trace_item)
            if sequence_after <= sequence_before:
                motion_ok = False
                motion_reason = "observation_not_advanced"
                break
            if monitor_reason is not None:
                motion_ok = False
                motion_reason = monitor_reason
                break
            if inactive_error > START_CAPTURE_TOLERANCE_RAD:
                motion_ok = False
                motion_reason = "inactive_arm_position_drift"
                break
            if gripper_error_m > GRIPPER_RECOVERY_HARD_LIMIT_M:
                gripper_recovery_events.append(
                    {
                        "event": "hard_limit_exceeded",
                        "waypoint_index": int(waypoint_index + 1),
                        "observation_sequence": int(sequence_after),
                        "drift_inf_m": gripper_error_m,
                        "hard_limit_m": GRIPPER_RECOVERY_HARD_LIMIT_M,
                        "gripper_observation": json_ready(
                            gripper_error_report["per_side"]
                        ),
                    }
                )
                motion_ok = False
                motion_reason = "gripper_drift"
                break
            monitor_requests_reacquisition_hold = bool(
                monitor.get("temporarily_unobserved", False)
            )
            if monitor_requests_reacquisition_hold:
                if tracker_reacquisition_hold_q is None:
                    tracker_reacquisition_hold_q = observed_q.copy()
                    tracker_reacquisition_events.append(
                        {
                            "event": "hold_started",
                            "waypoint_index": int(waypoint_index + 1),
                            "observation_sequence": int(sequence_after),
                            "hold_q_rad": tracker_reacquisition_hold_q.astype(
                                float
                            ).tolist(),
                            "reason": monitor.get("tracker_reason"),
                        }
                    )
                previous_error = float("inf")
                recovery_step = 0
                stall_steps = 0
                continue
            if command_is_tracker_reacquisition_hold:
                tracker_reacquisition_events.append(
                    {
                        "event": "observation_reacquired",
                        "waypoint_index": int(waypoint_index + 1),
                        "observation_sequence": int(sequence_after),
                        "hold_steps": int(tracker_reacquisition_hold_steps),
                    }
                )
                tracker_reacquisition_hold_q = None
                previous_error = float("inf")
                recovery_step = 0
                stall_steps = 0
                # Reissue the same immutable waypoint after the visual bundle
                # has recovered; the signed path is never skipped or changed.
                continue
            if gripper_recovery_hold_q is not None:
                recovery_supported = bool(
                    gripper_recovery_fingers
                    and all(
                        bool(
                            gripper_error_report["per_side"][side]["near_open_capture"]
                        )
                        and not bool(
                            last_gripper_command_report[side]["close_keepalive_active"]
                        )
                        for side in gripper_recovery_fingers
                    )
                )
                if not recovery_supported:
                    gripper_recovery_events.append(
                        {
                            "event": "hard_limit_or_mode_failure",
                            "waypoint_index": int(waypoint_index + 1),
                            "observation_sequence": int(sequence_after),
                            "drift_inf_m": gripper_error_m,
                            "gripper_observation": json_ready(
                                gripper_error_report["per_side"]
                            ),
                        }
                    )
                    motion_ok = False
                    motion_reason = "gripper_drift"
                    break
                gripper_recovery_steps += 1
                gripper_recovery_streak += 1
                recovery_drift_m = max(
                    abs(
                        float(
                            gripper_error_report["per_side"][side][
                                "signed_inward_error_m"
                            ]
                        )
                    )
                    for side in gripper_recovery_fingers
                )
                recovery_position_step_m = max(
                    float(gripper_position_steps_m[side])
                    for side in gripper_recovery_fingers
                )
                recovery_sample_stable = bool(
                    recovery_drift_m <= GRIPPER_RECOVERY_RELEASE_M
                    and recovery_position_step_m <= GRIPPER_RECOVERY_POSITION_STEP_M
                )
                gripper_recovery_stable_steps = (
                    gripper_recovery_stable_steps + 1 if recovery_sample_stable else 0
                )
                sample = {
                    "waypoint_index": int(waypoint_index + 1),
                    "observation_sequence": int(sequence_after),
                    "recovery_streak": int(gripper_recovery_streak),
                    "drift_inf_m": recovery_drift_m,
                    "overall_drift_inf_m": gripper_error_m,
                    "drift_inf_normalized": gripper_error,
                    "position_step_inf_m": (
                        None
                        if not math.isfinite(recovery_position_step_m)
                        else recovery_position_step_m
                    ),
                    "overall_position_step_inf_m": (
                        None
                        if not math.isfinite(gripper_position_step_m)
                        else gripper_position_step_m
                    ),
                    "max_qvel_mps": None,
                    "recovery_fingers": {
                        side: list(indices)
                        for side, indices in gripper_recovery_fingers.items()
                    },
                    "stable": recovery_sample_stable,
                    "stable_steps": int(gripper_recovery_stable_steps),
                    "qpos_m": {
                        side: [float(current_gripper_joint_positions_m[side])]
                        for side in ("left", "right")
                    },
                    "qpos_normalized": {
                        side: float(current_gripper[side][0])
                        for side in ("left", "right")
                    },
                    "command_normalized": {
                        side: float(gripper_targets[side][0])
                        for side in ("left", "right")
                    },
                    "stability_observation": (
                        "consecutive_evaluator_qpos_position_step"
                    ),
                }
                gripper_recovery_samples.append(sample)
                trace_item["gripper_recovery_sample"] = sample
                trace_item["gripper_recovery_streak"] = gripper_recovery_streak
                if gripper_recovery_stable_steps >= GRIPPER_RECOVERY_STABLE_STEPS:
                    gripper_recovery_events.append(
                        {
                            "event": "recovered",
                            "waypoint_index": int(waypoint_index + 1),
                            "observation_sequence": int(sequence_after),
                            "drift_inf_m": recovery_drift_m,
                            "overall_drift_inf_m": gripper_error_m,
                            "position_step_inf_m": recovery_position_step_m,
                            "max_qvel_mps": None,
                            "recovery_streak": int(gripper_recovery_streak),
                            "stable_steps": int(gripper_recovery_stable_steps),
                        }
                    )
                    gripper_recovery_hold_q = None
                    gripper_recovery_fingers = {}
                    gripper_recovery_streak = 0
                    gripper_recovery_stable_steps = 0
                    previous_error = float("inf")
                    recovery_step = 0
                    stall_steps = 0
                    # Recovery never rewrites or skips the signed waypoint.
                    continue
                if gripper_recovery_streak >= GRIPPER_RECOVERY_MAX_STEPS:
                    gripper_recovery_events.append(
                        {
                            "event": "recovery_exhausted",
                            "waypoint_index": int(waypoint_index + 1),
                            "observation_sequence": int(sequence_after),
                            "drift_inf_m": recovery_drift_m,
                            "overall_drift_inf_m": gripper_error_m,
                            "position_step_inf_m": (
                                None
                                if not math.isfinite(recovery_position_step_m)
                                else recovery_position_step_m
                            ),
                            "max_qvel_mps": None,
                            "recovery_streak": int(gripper_recovery_streak),
                        }
                    )
                    motion_ok = False
                    motion_reason = "gripper_drift"
                    break
                previous_error = float("inf")
                recovery_step = 0
                stall_steps = 0
                continue
            recoverable_fingers: dict[str, list[int]] = {}
            for side, item in gripper_error_report["per_side"].items():
                if bool(last_gripper_command_report[side]["close_keepalive_active"]):
                    continue
                if (
                    bool(item["near_open_capture"])
                    and float(item["signed_inward_error_m"])
                    > GRIPPER_RECOVERY_SOFT_LIMIT_M
                ):
                    recoverable_fingers[side] = [0]
            if recoverable_fingers:
                gripper_recovery_hold_q = observed_q.copy()
                gripper_recovery_fingers = {
                    side: list(indices) for side, indices in recoverable_fingers.items()
                }
                gripper_recovery_streak = 0
                gripper_recovery_stable_steps = 0
                gripper_recovery_events.append(
                    {
                        "event": "hold_started",
                        "waypoint_index": int(waypoint_index + 1),
                        "observation_sequence": int(sequence_after),
                        "drift_inf_m": gripper_error_m,
                        "drift_inf_normalized": gripper_error,
                        "recovery_fingers": {
                            side: list(indices)
                            for side, indices in gripper_recovery_fingers.items()
                        },
                        "soft_limit_m": GRIPPER_RECOVERY_SOFT_LIMIT_M,
                        "release_limit_m": GRIPPER_RECOVERY_RELEASE_M,
                        "hard_limit_m": GRIPPER_RECOVERY_HARD_LIMIT_M,
                        "gripper_observation": json_ready(
                            gripper_error_report["per_side"]
                        ),
                        "hold_q_rad": gripper_recovery_hold_q.astype(float).tolist(),
                    }
                )
                previous_error = float("inf")
                recovery_step = 0
                stall_steps = 0
                continue
            settle_rad = (
                FINAL_EXECUTION_JOINT_SETTLE_RAD
                if waypoint_index + 1 == len(execution_waypoints)
                else STREAM_SETTLE_RAD
            )
            decision = _stream_tracking_decision(
                tracking_error_rad=waypoint_tracking_error,
                previous_error_rad=previous_error,
                recovery_step=recovery_step,
                stall_steps=stall_steps,
                settle_rad=settle_rad,
            )
            recovery_step = int(decision["recovery_step"])
            stall_steps = int(decision["stall_steps"])
            if decision["action"] == "advance":
                completed_waypoints += 1
                break
            if decision["action"] == "hold":
                tracking_recovery_events.append(
                    {
                        "waypoint_index": int(waypoint_index + 1),
                        "attempt": int(recovery_step),
                        "tracking_error_inf_rad": tracking_error,
                        "progress_delta_rad": decision["progress_delta_rad"],
                        "stall_steps": int(stall_steps),
                        "reason": "joint_settle_catchup_hold",
                    }
                )
                previous_error = waypoint_tracking_error
                continue
            motion_ok = False
            motion_reason = str(decision["action"])
            break
        if not motion_ok:
            break

    motion = {
        "ok": motion_ok,
        "reason": motion_reason,
        "controller": "immutable_signed_joint_trajectory_stream",
        "waypoint_completion_semantics": (
            "one_official_action_per_waypoint_with_proprio_tracking_corridor"
        ),
        "observation_action_loop": True,
        "steps_executed": action_steps,
        "action_steps": action_steps,
        "last_observation_sequence": _sequence(ctx),
        "trace": motion_trace,
        "official_action_only": True,
        "execution_mode": "continuous_waypoint_stream",
        "waypoint_count": len(execution_waypoints),
        "signed_waypoint_count": len(trajectory["waypoints"]),
        "execution_schedule_applied": True,
        "execution_compression_disabled": bool(dense_execution_policy),
        "execution_waypoint_policy": (
            "all_signed_waypoints"
            if dense_execution_policy
            else "bounded_monotone_subsample"
        ),
        "requested_max_steps": requested_max_steps,
        "effective_max_steps": int(args["max_steps"]),
        "dense_step_budget_expanded": bool(dense_step_budget_expanded),
        "step_budget_policy": (
            "port_unbounded_dense_replay"
            if dense_execution_policy
            else "caller_bounded"
        ),
        "execution_max_step_rad": float(schedule["execution_max_joint_step_rad"]),
        "execution_bridge_waypoint_count": int(
            schedule["bridge"]["bridge_waypoint_count"]
        ),
        "execution_start_rebased": True,
        "waypoints_completed": completed_waypoints,
        "waypoint_tolerance_rad": WAYPOINT_TOLERANCE_RAD,
        "max_tracking_error_limit_rad": STREAM_TRACKING_ERROR_RAD,
        "max_observed_tracking_error_rad": float(max_observed_tracking_error),
        "tracking_recovery_steps": int(
            sum(int(item.get("attempt", 0)) for item in tracking_recovery_events)
        ),
        "tracking_recovery_events": json_ready(tracking_recovery_events),
        "execution_schedule": json_ready(schedule),
        "tracking_continuity": tracking_continuity,
        "tracker_reacquisition_hold_steps": int(tracker_reacquisition_hold_steps),
        "tracker_reacquisition_max_steps": (
            int(NPOINT_REACQUISITION_MAX_HOLD_STEPS)
            if pause_for_visual_reacquisition
            else None
        ),
        "tracker_reacquisition_events": json_ready(tracker_reacquisition_events),
        "gripper_recovery_enabled": True,
        "gripper_recovery_steps": int(gripper_recovery_steps),
        "gripper_recovery_events": json_ready(gripper_recovery_events),
        "gripper_recovery_samples": json_ready(gripper_recovery_samples),
        "gripper_recovery_active_at_exit": bool(gripper_recovery_hold_q is not None),
        "gripper_recovery_fingers_active_at_exit": {
            side: list(indices) for side, indices in gripper_recovery_fingers.items()
        },
        "gripper_recovery_soft_limit_m": GRIPPER_RECOVERY_SOFT_LIMIT_M,
        "gripper_recovery_release_m": GRIPPER_RECOVERY_RELEASE_M,
        "gripper_recovery_hard_limit_m": GRIPPER_RECOVERY_HARD_LIMIT_M,
        "gripper_recovery_max_steps": GRIPPER_RECOVERY_MAX_STEPS,
        "gripper_recovery_required_stable_steps": (GRIPPER_RECOVERY_STABLE_STEPS),
        "gripper_recovery_position_step_threshold_m": (
            GRIPPER_RECOVERY_POSITION_STEP_M
        ),
        "gripper_recovery_qvel_threshold_mps": (GRIPPER_RECOVERY_UPSTREAM_QVEL_MPS),
        "gripper_recovery_effort_n": GRIPPER_RECOVERY_UPSTREAM_EFFORT_N,
        "gripper_recovery_target_policy": (
            "near_open_inward_drift_only_intermediate_and_close_latch_unchanged"
        ),
        "gripper_recovery_stability_observation": (
            "consecutive_evaluator_qpos_position_step"
        ),
        "gripper_open_hold_threshold_m": GRIPPER_OPEN_HOLD_THRESHOLD_M,
        "gripper_open_recovery_deadband_m": GRIPPER_OPEN_RECOVERY_DEADBAND_M,
        "max_gripper_drift_inf_m": float(max_gripper_drift_m),
        "final_gripper_drift_inf_m": float(final_gripper_drift_m),
        "max_gripper_drift_inf_normalized": float(max_gripper_drift_normalized),
        "max_inactive_arm_drift_inf_rad": float(max_inactive_arm_drift),
        # X5 has six arm joints, so the R1Pro-only locked J8 check is
        # structurally present but physically inapplicable.
        "max_observed_j8_abs_rad": 0.0,
        "gripper_command": json_ready(last_gripper_command_report),
        "safety": {
            "ok": True,
            "source": "robodojo_evaluator_observation_and_action_contract",
            "direct_simulator_access": False,
            "collision_truth_available": False,
            "j8_applicable": False,
        },
        "last_waypoint": (json_ready(motion_trace[-1]) if motion_trace else {}),
        "last_step_monitor": last_step_monitor,
        "x5_gripper_control": {
            "mode": "response_bounded_position_hold",
            "capture_hold_policy_applied_every_action": True,
            "capture_target_reissued_during_inward_recovery": True,
            "near_open_close_intent_suppressed": True,
            "verified_close_keepalive_unchanged": True,
            "effort_recovery_applicable": False,
            "position_recovery_applicable": True,
            "velocity_observation_available": False,
            "stability_observation": ("consecutive_evaluator_qpos_position_step"),
            "official_joint_scale_m": list(ROBODOJO_X5_GRIPPER_SCALE_M),
            "reason": (
                "RoboDojo X5 evaluator exposes normalized position control; "
                "recovery holds measured arm qpos while the near-open X5 "
                "position target drives outward; intermediate targets and "
                "verified close latches retain their prior semantics"
            ),
        },
    }

    stability_trace: list[dict[str, Any]] = []
    stable_pose_steps = 0
    target_position = np.asarray(
        trajectory["target"]["eef_position_robot_base_m"], dtype=np.float64
    ).reshape(3)
    target_quaternion = np.asarray(
        trajectory["target"]["eef_quaternion_xyzw"], dtype=np.float64
    ).reshape(4)
    target_q = np.asarray(trajectory["waypoints"][-1], dtype=np.float64).reshape(
        ARM_DOF
    )
    initial_target_q = target_q.copy()
    final_command_q = target_q.copy()
    final_previous_q = np.asarray(
        ctx.world.arm_qpos_list(arm), dtype=np.float64
    ).reshape(ARM_DOF)
    load_compensation_rounds = 0
    load_compensation_total = np.zeros(ARM_DOF, dtype=np.float64)
    load_compensation_trace: list[dict[str, Any]] = []
    pending_waypoints: list[np.ndarray] = []
    pending_command_kind: str | None = None
    load_stationary_steps = 0
    visual_correction_rounds: list[dict[str, Any]] = []
    visual_correction_failure: dict[str, Any] | None = None
    visual_tracking_pause_active = False
    visual_tracking_loss_steps = 0
    visual_tracking_inflight_waypoint: np.ndarray | None = None
    visual_tracking_recovery_events: list[dict[str, Any]] = []
    goal_latched = False
    goal_latch_sequence: int | None = None
    last_goal_report = _final_goal_report_x5(
        ctx,
        manager=manager,
        arm=arm,
        names=names,
        controlled_names=controlled_names,
        reference_names=reference_names,
        episode_id=episode_id,
        session_id=session_id,
        image_id=image_id,
        anchors=anchors,
        source_points=source_points,
        reference_points=reference_points,
        target_points=list(trajectory["target"].get("planning_target_points") or []),
        fixed_target_points=list(trajectory["target"].get("fixed_target_points") or []),
        relations=list(trajectory["target"].get("relations") or []),
        inequalities=list(trajectory["target"].get("inequalities") or []),
        quick_constraint=quick_spec if quick_mode else None,
        quick_constraints=(args.get("quick_constraints") if quick_mode else None),
        anchor_tolerance_m=anchor_tolerance,
        geometry_tolerance_m=geometry_tolerance,
        target_position=target_position,
        target_quaternion=target_quaternion,
        pos_tol_m=float(args["pos_tol"]),
        ori_tol_deg=float(args["ori_tol_deg"]),
        camera_position_robot_base_m=target_camera_position,
        camera_quaternion_xyzw=target_camera_quaternion,
        use_live_reference_for_constraints=strict_live_reference_goal,
    )
    if bool(last_goal_report.get("goal_satisfied")):
        observed_latch = np.asarray(
            ctx.world.arm_qpos_list(arm), dtype=np.float64
        ).reshape(ARM_DOF)
        overshoot = np.maximum(X5_JOINT_LOWER - observed_latch, 0.0) + np.maximum(
            observed_latch - X5_JOINT_UPPER, 0.0
        )
        if float(np.max(overshoot)) <= MAX_JOINT_STEP_RAD + 1.0e-8:
            final_command_q = np.clip(observed_latch, X5_JOINT_LOWER, X5_JOINT_UPPER)
            goal_latched = True
            goal_latch_sequence = _sequence(ctx)

    final_step_budget = max(
        0,
        min(FINAL_MAX_STABILITY_STEPS, int(args["max_steps"]) - action_steps),
    )
    final_steps = 0
    if motion_ok:
        for final_step in range(final_step_budget):
            try:
                ctx.raise_if_cancelled("move_tracked_point final stability")
            except ToolCancelled as exc:
                _set_failure(
                    ctx,
                    result_writer,
                    stage="cancellation",
                    error=str(exc),
                    args=args,
                    details={
                        "arm": arm,
                        "plan_id": plan_id,
                        "trajectory_digest": digest,
                        "cancellation_evaluator_action_emitted": False,
                    },
                )
                yield ctx.world.hold_action()
                return
            if time.monotonic() - planning_started >= float(args["timeout_s"]):
                motion_ok = False
                motion_reason = "timeout"
                break
            if pending_waypoints and not visual_tracking_pause_active:
                final_command_q = pending_waypoints.pop(0).copy()
                if pending_command_kind == "visual_endpoint_correction":
                    visual_tracking_inflight_waypoint = final_command_q.copy()
                stable_pose_steps = 0

            sequence_before = _sequence(ctx)
            final_gripper_targets, _final_gripper_report = _x5_gripper_hold_targets(
                ctx, frozen
            )
            yield ctx.world.make_joint_action(
                left_arm=(
                    final_command_q.tolist()
                    if arm == "left"
                    else frozen["arm_q"]["left"].tolist()
                ),
                left_gripper=final_gripper_targets["left"],
                right_arm=(
                    final_command_q.tolist()
                    if arm == "right"
                    else frozen["arm_q"]["right"].tolist()
                ),
                right_gripper=final_gripper_targets["right"],
            )
            action_steps += 1
            final_steps += 1
            try:
                ctx.raise_if_cancelled("move_tracked_point final stability")
            except ToolCancelled as exc:
                _set_failure(
                    ctx,
                    result_writer,
                    stage="cancellation",
                    error=str(exc),
                    args=args,
                    details={
                        "arm": arm,
                        "plan_id": plan_id,
                        "trajectory_digest": digest,
                        "cancellation_evaluator_action_emitted": False,
                    },
                )
                yield ctx.world.hold_action()
                return
            sequence_after = _sequence(ctx)
            if sequence_after <= sequence_before:
                motion_ok = False
                motion_reason = "observation_not_advanced"
                break

            position, quaternion = _eef_base_xyzw(ctx, arm)
            observed_final_q = np.asarray(
                ctx.world.arm_qpos_list(arm), dtype=np.float64
            ).reshape(ARM_DOF)
            position_step = float(
                np.linalg.norm(observed_final_q - final_previous_q, ord=np.inf)
            )
            final_previous_q = observed_final_q.copy()
            stationary = bool(position_step <= FINAL_COHERENCE_POSITION_STEP_RAD)
            load_stationary_steps = load_stationary_steps + 1 if stationary else 0
            last_goal_report = _final_goal_report_x5(
                ctx,
                manager=manager,
                arm=arm,
                names=names,
                controlled_names=controlled_names,
                reference_names=reference_names,
                episode_id=episode_id,
                session_id=session_id,
                image_id=image_id,
                anchors=anchors,
                source_points=source_points,
                reference_points=reference_points,
                target_points=list(
                    trajectory["target"].get("planning_target_points") or []
                ),
                fixed_target_points=list(
                    trajectory["target"].get("fixed_target_points") or []
                ),
                relations=list(trajectory["target"].get("relations") or []),
                inequalities=list(trajectory["target"].get("inequalities") or []),
                quick_constraint=quick_spec if quick_mode else None,
                quick_constraints=(
                    args.get("quick_constraints") if quick_mode else None
                ),
                anchor_tolerance_m=anchor_tolerance,
                geometry_tolerance_m=geometry_tolerance,
                target_position=target_position,
                target_quaternion=target_quaternion,
                pos_tol_m=float(args["pos_tol"]),
                ori_tol_deg=float(args["ori_tol_deg"]),
                camera_position_robot_base_m=target_camera_position,
                camera_quaternion_xyzw=target_camera_quaternion,
                use_live_reference_for_constraints=strict_live_reference_goal,
            )
            visual_tracker_blocked = bool(
                pending_command_kind == "visual_endpoint_correction"
                and not bool(last_goal_report.get("ok"))
            )
            if visual_tracker_blocked:
                if not visual_tracking_pause_active:
                    if visual_tracking_inflight_waypoint is not None:
                        pending_waypoints.insert(
                            0, visual_tracking_inflight_waypoint.copy()
                        )
                        visual_tracking_inflight_waypoint = None
                    measured_hold = observed_final_q.copy()
                    lower_error = np.maximum(X5_JOINT_LOWER - measured_hold, 0.0)
                    upper_error = np.maximum(measured_hold - X5_JOINT_UPPER, 0.0)
                    maximum_overshoot = float(
                        max(np.max(lower_error), np.max(upper_error))
                    )
                    if maximum_overshoot > MAX_JOINT_STEP_RAD + 1.0e-8:
                        visual_correction_failure = {
                            "reason": "tracking_loss_hold_invalid",
                            "error": (
                                "cannot form a bounded measured X5 hold while "
                                "final visual tracking is unavailable"
                            ),
                            "observation_sequence": sequence_after,
                            "joint_limit_overshoot_rad": maximum_overshoot,
                        }
                        motion_ok = False
                        motion_reason = "final_visual_correction_tracking_lost"
                        break
                    final_command_q = np.clip(
                        measured_hold, X5_JOINT_LOWER, X5_JOINT_UPPER
                    )
                    visual_tracking_pause_active = True
                    visual_tracking_recovery_events.append(
                        {
                            "event": len(visual_tracking_recovery_events) + 1,
                            "start_observation_sequence": sequence_after,
                            "start_reason": last_goal_report.get("reason"),
                            "measured_hold_q": final_command_q.astype(float).tolist(),
                            "joint_limit_normalization_max_rad": maximum_overshoot,
                            "requeued_inflight_waypoint": bool(pending_waypoints),
                            "recovered": False,
                        }
                    )
                visual_tracking_loss_steps += 1
                if visual_tracking_recovery_events:
                    visual_tracking_recovery_events[-1].update(
                        {
                            "wait_steps": visual_tracking_loss_steps,
                            "last_observation_sequence": sequence_after,
                            "last_reason": last_goal_report.get("reason"),
                        }
                    )
                if visual_tracking_loss_steps >= FINAL_VISUAL_REACQUIRE_MAX_STEPS:
                    visual_correction_failure = {
                        "reason": "tracked_point_reacquisition_failed",
                        "error": (
                            "final visual correction paused at measured joints, "
                            "but a synchronized tracked-point bundle did not "
                            f"reacquire within {FINAL_VISUAL_REACQUIRE_MAX_STEPS} "
                            "official observation steps"
                        ),
                        "observation_sequence": sequence_after,
                        "last_tracker_reason": last_goal_report.get("reason"),
                        "recovery_steps": visual_tracking_loss_steps,
                    }
                    motion_ok = False
                    motion_reason = "final_visual_correction_tracking_lost"
                    break
            elif visual_tracking_pause_active:
                visual_tracking_pause_active = False
                visual_tracking_loss_steps = 0
                visual_tracking_inflight_waypoint = None
                if visual_tracking_recovery_events:
                    visual_tracking_recovery_events[-1].update(
                        {
                            "recovered": True,
                            "recovered_observation_sequence": sequence_after,
                        }
                    )

            if not goal_latched and bool(last_goal_report.get("goal_satisfied")):
                lower_error = np.maximum(X5_JOINT_LOWER - observed_final_q, 0.0)
                upper_error = np.maximum(observed_final_q - X5_JOINT_UPPER, 0.0)
                if float(max(np.max(lower_error), np.max(upper_error))) <= (
                    MAX_JOINT_STEP_RAD + 1.0e-8
                ):
                    final_command_q = np.clip(
                        observed_final_q, X5_JOINT_LOWER, X5_JOINT_UPPER
                    )
                    goal_latched = True
                    goal_latch_sequence = sequence_after
                    stable_pose_steps = 0

            if bool(last_goal_report.get("goal_satisfied")):
                pending_waypoints = []
                pending_command_kind = None
                visual_tracking_inflight_waypoint = None
                visual_tracking_pause_active = False
                visual_tracking_loss_steps = 0
                load_stationary_steps = 0
            elif not visual_tracker_blocked:
                scheduled_recovery = False
                residual = target_q - observed_final_q
                residual_inf = float(np.linalg.norm(residual, ord=np.inf))
                if (
                    not pending_waypoints
                    and load_stationary_steps >= FINAL_LOAD_STABLE_STEPS
                    and residual_inf >= FINAL_LOAD_MIN_RESIDUAL_RAD
                    and load_compensation_rounds < FINAL_LOAD_MAX_ROUNDS
                ):
                    increment = np.clip(
                        residual,
                        -FINAL_LOAD_MAX_ROUND_RAD,
                        FINAL_LOAD_MAX_ROUND_RAD,
                    )
                    total_lower = np.maximum(
                        X5_JOINT_LOWER, target_q - FINAL_LOAD_MAX_TOTAL_RAD
                    )
                    total_upper = np.minimum(
                        X5_JOINT_UPPER, target_q + FINAL_LOAD_MAX_TOTAL_RAD
                    )
                    candidate_command = np.clip(
                        final_command_q + increment, total_lower, total_upper
                    )
                    applied = candidate_command - final_command_q
                    applied_inf = float(np.linalg.norm(applied, ord=np.inf))
                    if applied_inf > 0.0:
                        segment_count = max(
                            1, int(math.ceil(applied_inf / MAX_JOINT_STEP_RAD))
                        )
                        command_start = final_command_q.copy()
                        pending_waypoints = [
                            command_start
                            + (candidate_command - command_start)
                            * (index / segment_count)
                            for index in range(1, segment_count + 1)
                        ]
                        final_command_q = command_start
                        load_compensation_total = candidate_command - target_q
                        load_compensation_rounds += 1
                        load_compensation_trace.append(
                            {
                                "final_step": final_step,
                                "round": load_compensation_rounds,
                                "observed_q": observed_final_q.astype(float).tolist(),
                                "planned_q_final": target_q.astype(float).tolist(),
                                "command_start_q": command_start.astype(float).tolist(),
                                "command_target_q": candidate_command.astype(
                                    float
                                ).tolist(),
                                "joint_residual_rad": residual.astype(float).tolist(),
                                "observed_residual_inf_rad": residual_inf,
                                "increment_rad": applied.astype(float).tolist(),
                                "total_increment_rad": load_compensation_total.astype(
                                    float
                                ).tolist(),
                                "command_waypoint_count": segment_count,
                                "immutable_signed_target_changed": False,
                                "cartesian_replan_used": False,
                            }
                        )
                        pending_command_kind = "joint_load_compensation"
                        load_stationary_steps = 0
                        stable_pose_steps = 0
                        scheduled_recovery = True

                if not scheduled_recovery:
                    constraints = last_goal_report.get("constraints")
                    geometry = last_goal_report.get("rigid_pair_consistency")
                    reference_report = last_goal_report.get("off_hand_reference")
                    visual_geometry_ok = bool(
                        isinstance(geometry, dict)
                        and geometry.get("ok")
                        and (
                            geometry.get("applicable")
                            or len(controlled_names)
                            == FINAL_VISUAL_CORRECTION_MIN_POINTS
                        )
                    )
                    visual_applicable = bool(
                        not pending_waypoints
                        and len(controlled_names) >= FINAL_VISUAL_CORRECTION_MIN_POINTS
                        and len(visual_correction_rounds) < FINAL_VISUAL_MAX_ROUNDS
                        and load_stationary_steps >= FINAL_LOAD_STABLE_STEPS
                        and last_goal_report.get("ok")
                        and last_goal_report.get("pose_satisfied")
                        and isinstance(constraints, dict)
                        and not constraints.get("ok")
                        and visual_geometry_ok
                        and (
                            not reference_names
                            or (
                                isinstance(reference_report, dict)
                                and reference_report.get("ok")
                            )
                        )
                    )
                    if visual_applicable:
                        maximum_anchor_error = float(
                            last_goal_report.get("max_anchor_error_m", math.inf)
                        )
                        maximum_constraint_error = float(
                            constraints.get("max_constraint_error_m", math.inf)
                        )
                        constraint_rebase_limit = min(
                            FINAL_VISUAL_MAX_CONSTRAINT_ERROR_M,
                            max(0.030, 3.0 * float(args["pos_tol"])),
                        )
                        visual_applicable = bool(
                            math.isfinite(maximum_anchor_error)
                            and maximum_anchor_error
                            <= min(
                                anchor_tolerance,
                                FINAL_VISUAL_MAX_ANCHOR_REBASE_M,
                            )
                            and math.isfinite(maximum_constraint_error)
                            and maximum_constraint_error <= constraint_rebase_limit
                        )
                    if visual_applicable:
                        remaining_before_plan = float(args["timeout_s"]) - (
                            time.monotonic() - planning_started
                        )
                        remaining_steps = (
                            int(args["max_steps"]) - action_steps - STABLE_STEPS
                        )
                        if (
                            remaining_before_plan
                            <= FINAL_VISUAL_MIN_EXECUTION_RESERVE_S
                            or remaining_steps < 1
                        ):
                            visual_correction_failure = {
                                "reason": "insufficient_correction_budget",
                                "error": (
                                    "no bounded evaluator time/action budget remains "
                                    "for final visual correction"
                                ),
                                "remaining_time_s": max(0.0, remaining_before_plan),
                                "remaining_action_steps": max(0, remaining_steps),
                            }
                            motion_ok = False
                            motion_reason = "final_visual_correction_failed"
                            break
                        try:
                            correction_source = _strict_point_matrix(
                                [
                                    last_goal_report["points_robot_base_m"][name]
                                    for name in controlled_names
                                ],
                                expected_count=len(controlled_names),
                                label="final visual correction source points",
                            )
                            correction_references = (
                                _strict_point_matrix(
                                    [
                                        last_goal_report[
                                            "constraint_fixed_points_robot_base_m"
                                        ][name]
                                        for name in reference_names
                                    ],
                                    expected_count=len(reference_names),
                                    label="final visual correction fixed points",
                                )
                                if reference_names
                                else None
                            )
                            correction_frozen = _freeze_state(ctx)
                            if _sequence(ctx) != sequence_after:
                                raise ValueError(
                                    "evaluator observation changed while freezing "
                                    "final visual correction"
                                )
                            correction_q_start = observed_final_q.copy()
                            planning_budget = min(
                                FINAL_VISUAL_MAX_PLANNING_S,
                                remaining_before_plan
                                - FINAL_VISUAL_MIN_EXECUTION_RESERVE_S,
                            )
                            correction_plan_kwargs = dict(
                                state=correction_frozen["state"],
                                arm=arm,
                                q_start=correction_q_start,
                                source_points=correction_source,
                                target_points=list(
                                    trajectory["target"].get("planning_target_points")
                                    or []
                                ),
                                relations=list(
                                    trajectory["target"].get("relations") or []
                                ),
                                fixed_points=correction_references,
                                fixed_target_points=(
                                    list(
                                        trajectory["target"].get("fixed_target_points")
                                        or []
                                    )
                                    if reference_names
                                    else None
                                ),
                                pos_tol_m=float(args["pos_tol"]),
                                ori_tol_deg=float(args["ori_tol_deg"]),
                                max_steps=max(1, remaining_steps),
                                cancel_event=threading.Event(),
                                planning_deadline_monotonic=(
                                    time.monotonic() + max(0.01, planning_budget)
                                ),
                            )
                            if trajectory["target"].get("inequalities"):
                                correction_plan_kwargs["inequalities"] = list(
                                    trajectory["target"]["inequalities"]
                                )
                            if faceto_requested:
                                correction_plan_kwargs.update(
                                    {
                                        "quick_constraints": list(quick_specs),
                                        "off_hand_names": list(reference_names),
                                        "head_camera_position_robot_base_m": (
                                            target_camera_position
                                        ),
                                        "head_camera_quaternion_xyzw": (
                                            target_camera_quaternion
                                        ),
                                    }
                                )
                            correction_plan = _plan_frozen(**correction_plan_kwargs)
                            if not bool(correction_plan.get("ok")):
                                raise ValueError(
                                    str(
                                        correction_plan.get("error")
                                        or "no valid final visual correction path"
                                    )
                                )
                            trimmed = _trim_visual_correction_to_first_feasible(
                                state=correction_frozen["state"],
                                arm=arm,
                                q_start=correction_q_start,
                                endpoint=correction_plan["endpoint"],
                                path=correction_plan["path"],
                                target_points=list(
                                    trajectory["target"].get("planning_target_points")
                                    or []
                                ),
                                relations=list(
                                    trajectory["target"].get("relations") or []
                                ),
                                fixed_points_robot_base_m=correction_references,
                                fixed_target_points=(
                                    list(
                                        trajectory["target"].get("fixed_target_points")
                                        or []
                                    )
                                    if reference_names
                                    else None
                                ),
                                pos_tol_m=float(args["pos_tol"]),
                                ori_tol_deg=float(args["ori_tol_deg"]),
                                inequalities=list(
                                    trajectory["target"].get("inequalities") or []
                                ),
                            )
                            correction_endpoint = trimmed["endpoint"]
                            correction_path = trimmed["path"]
                            prefix_report = trimmed["report"]
                            endpoint_q = np.asarray(
                                correction_endpoint["q_final"], dtype=np.float64
                            ).reshape(ARM_DOF)
                            endpoint_position = np.asarray(
                                correction_endpoint["final_eef_position_m"],
                                dtype=np.float64,
                            ).reshape(3)
                            endpoint_quaternion = np.asarray(
                                correction_endpoint["final_eef_quaternion_xyzw"],
                                dtype=np.float64,
                            ).reshape(4)
                            endpoint_anchors = _strict_point_matrix(
                                correction_endpoint["anchors_eef_m"],
                                expected_count=len(controlled_names),
                                label="final visual correction EEF anchors",
                            )
                            correction_translation = float(
                                np.linalg.norm(endpoint_position - position)
                            )
                            if (
                                not math.isfinite(correction_translation)
                                or correction_translation
                                > FINAL_VISUAL_MAX_TRANSLATION_M
                            ):
                                raise ValueError(
                                    "final visual correction EEF translation exceeds "
                                    f"the bounded {FINAL_VISUAL_MAX_TRANSLATION_M:.3f}m "
                                    "envelope"
                                )
                            if not bool(
                                correction_endpoint.get("constraints", {}).get("ok")
                            ):
                                raise ValueError(
                                    "final visual correction endpoint does not satisfy "
                                    "the immutable goal"
                                )
                            remaining_after_plan = float(args["timeout_s"]) - (
                                time.monotonic() - planning_started
                            )
                            remaining_steps = (
                                int(args["max_steps"]) - action_steps - STABLE_STEPS
                            )
                            if remaining_after_plan <= 0.0 or remaining_steps < 1:
                                raise ValueError(
                                    "final visual correction planning consumed its "
                                    "execution budget"
                                )
                            dense_waypoints = list(
                                correction_path.get("waypoints") or []
                            )
                            correction_path_certified = bool(
                                correction_path.get("same_branch_checked") is True
                                and correction_path.get("monotonic_progress_checked")
                                is True
                                and int(correction_path.get("internal_stop_count", 0))
                                == 0
                            )
                            correction_dense_policy = bool(
                                EXECUTION_DENSE_WAYPOINTS
                                or len(dense_waypoints)
                                <= int(EXECUTION_DENSE_SHORT_PATH_THRESHOLD)
                                or not correction_path_certified
                            )
                            correction_trajectory = {
                                "tracking": {
                                    "max_joint_step_rad": float(
                                        correction_path.get(
                                            "max_joint_step_rad",
                                            MAX_JOINT_STEP_RAD,
                                        )
                                    ),
                                    "execution_max_speedup_factor": (
                                        EXECUTION_MAX_SPEEDUP_FACTOR
                                    ),
                                    "execution_max_command_delta_rad": (
                                        EXECUTION_MAX_COMMAND_DELTA_RAD
                                    ),
                                    "execution_compression_disabled": (
                                        correction_dense_policy
                                    ),
                                    "execution_waypoint_policy": (
                                        "all_signed_waypoints"
                                        if correction_dense_policy
                                        else "bounded_monotone_subsample"
                                    ),
                                },
                                "start_state": {
                                    "arm_qpos": {
                                        side: np.asarray(
                                            correction_frozen["arm_q"][side],
                                            dtype=np.float64,
                                        ).tolist()
                                        for side in ("left", "right")
                                    },
                                    "eef_base_wxyz": {
                                        side: np.asarray(
                                            correction_frozen["eef_base_wxyz"][side],
                                            dtype=np.float64,
                                        ).tolist()
                                        for side in ("left", "right")
                                    },
                                },
                                "waypoints": dense_waypoints,
                                "planner": {
                                    "same_branch_checked": bool(
                                        correction_path.get("same_branch_checked")
                                    ),
                                    "monotonic_progress_checked": bool(
                                        correction_path.get(
                                            "monotonic_progress_checked"
                                        )
                                    ),
                                    "internal_stop_count": int(
                                        correction_path.get("internal_stop_count", 0)
                                    ),
                                },
                            }
                            correction_schedule = _prepare_execution_schedule(
                                trajectory=correction_trajectory,
                                state=correction_frozen["state"],
                                arm=arm,
                                start_q=correction_q_start,
                                max_steps=max(1, remaining_steps),
                                available_time_s=max(0.01, remaining_after_plan),
                                observation_dt_s=None,
                                visual_tracked_point_count=(
                                    manager.visual_point_count(controlled_names)
                                ),
                                requested_public_max_steps=requested_public_max_steps,
                            )
                            if not bool(correction_schedule.get("ok")):
                                raise ValueError(
                                    str(
                                        correction_schedule.get("error")
                                        or "final visual correction cannot fit "
                                        "execution budget"
                                    )
                                )
                            scheduled_waypoints = [
                                np.asarray(raw_q, dtype=np.float64).reshape(ARM_DOF)
                                for raw_q in correction_schedule.get("waypoints") or []
                            ]
                            if not scheduled_waypoints:
                                raise ValueError(
                                    "final visual correction execution path is empty"
                                )
                            correction_execution_waypoints, correction_bridge = (
                                _execution_bridge(
                                    correction_q_start,
                                    scheduled_waypoints,
                                    max_step_rad=float(
                                        correction_schedule[
                                            "execution_max_joint_step_rad"
                                        ]
                                    ),
                                )
                            )
                            if len(correction_execution_waypoints) != int(
                                correction_schedule[
                                    "predicted_execution_waypoint_count"
                                ]
                            ):
                                raise ValueError(
                                    "final visual correction bridge expansion changed "
                                    "after budget preflight"
                                )
                            transition_report = _validate_execution_transitions(
                                correction_frozen["state"],
                                arm,
                                correction_q_start,
                                correction_execution_waypoints,
                            )
                            correction_schedule["bridge"] = correction_bridge
                            correction_schedule["execution_waypoint_count"] = len(
                                correction_execution_waypoints
                            )
                            correction_record = {
                                "tool": "move_tracked_point",
                                "source_tool": "move_tracked_point",
                                "tool_version": TOOL_VERSION,
                                "schema": (
                                    "official_v2_move_tracked_point_"
                                    "final_visual_correction"
                                ),
                                "schema_version": 1,
                                "round": len(visual_correction_rounds) + 1,
                                "robot_profile": ROBOT_PROFILE,
                                "arm": arm,
                                "episode_id": episode_id,
                                "source_observation_sequence": sequence_after,
                                "source_points_robot_base_m": correction_source,
                                "fixed_points_robot_base_m": correction_references,
                                "target_points": trajectory["target"].get(
                                    "planning_target_points"
                                ),
                                "relations": trajectory["target"].get("relations"),
                                "inequalities": trajectory["target"].get(
                                    "inequalities"
                                ),
                                "q_start": correction_q_start,
                                "q_final": endpoint_q,
                                "anchors_eef_m": endpoint_anchors,
                                "eef_position_robot_base_m": endpoint_position,
                                "eef_quaternion_xyzw": endpoint_quaternion,
                                "dense_waypoint_count": len(dense_waypoints),
                                "waypoints": scheduled_waypoints,
                                "execution_schedule": correction_schedule,
                            }
                            saved_correction = ctx.plans.save(
                                session_id, correction_record
                            )
                            loaded_correction = ctx.plans.load(
                                session_id, str(saved_correction["plan_id"])
                            )
                            reloaded_execution, reloaded_bridge = _execution_bridge(
                                correction_q_start,
                                loaded_correction["waypoints"],
                                max_step_rad=float(
                                    loaded_correction["execution_schedule"][
                                        "execution_max_joint_step_rad"
                                    ]
                                ),
                            )
                            if len(reloaded_execution) != int(
                                loaded_correction["execution_schedule"][
                                    "predicted_execution_waypoint_count"
                                ]
                            ) or not np.allclose(
                                np.asarray(reloaded_execution, dtype=np.float64),
                                np.asarray(
                                    correction_execution_waypoints,
                                    dtype=np.float64,
                                ),
                                rtol=0.0,
                                atol=1.0e-12,
                            ):
                                raise ValueError(
                                    "reloaded final visual correction bridge does not "
                                    "match its preflight"
                                )
                            if int(reloaded_bridge["bridge_waypoint_count"]) != int(
                                correction_bridge["bridge_waypoint_count"]
                            ):
                                raise ValueError(
                                    "reloaded final visual correction bridge count changed"
                                )
                            pending_waypoints = [
                                np.asarray(q, dtype=np.float64).reshape(ARM_DOF)
                                for q in reloaded_execution
                            ]
                            pending_command_kind = "visual_endpoint_correction"
                            visual_tracking_inflight_waypoint = None
                            visual_tracking_pause_active = False
                            visual_tracking_loss_steps = 0
                            target_q = endpoint_q.copy()
                            target_position = endpoint_position.copy()
                            target_quaternion = endpoint_quaternion.copy()
                            anchors = endpoint_anchors.copy()
                            final_command_q = correction_q_start.copy()
                            load_compensation_total = np.zeros(
                                ARM_DOF, dtype=np.float64
                            )
                            goal_latched = False
                            goal_latch_sequence = None
                            load_stationary_steps = 0
                            stable_pose_steps = 0
                            visual_correction_rounds.append(
                                {
                                    "round": len(visual_correction_rounds) + 1,
                                    "source_observation_sequence": sequence_after,
                                    "source_max_anchor_error_m": maximum_anchor_error,
                                    "source_max_constraint_error_m": (
                                        maximum_constraint_error
                                    ),
                                    "constraint_rebase_limit_m": (
                                        constraint_rebase_limit
                                    ),
                                    "eef_translation_m": correction_translation,
                                    "original_solver_eef_translation_m": float(
                                        prefix_report["original_endpoint_translation_m"]
                                    ),
                                    "dense_waypoint_count": len(dense_waypoints),
                                    "execution_waypoint_count": len(pending_waypoints),
                                    "trajectory_digest": loaded_correction["integrity"][
                                        "digest"
                                    ],
                                    "trajectory_plan_id": loaded_correction["plan_id"],
                                    "trajectory_record_path": loaded_correction[
                                        "plan_record_path"
                                    ],
                                    "trajectory_schema": loaded_correction["schema"],
                                    "trajectory_schema_version": loaded_correction[
                                        "schema_version"
                                    ],
                                    "source": (
                                        "synchronized_evaluator_rgbd_and_proprio_"
                                        "local_replan"
                                    ),
                                    "cartesian_replan_used": True,
                                    "execution_schedule": json_ready(
                                        correction_schedule
                                    ),
                                    "execution_transition_validation": json_ready(
                                        transition_report
                                    ),
                                    "resolved_endpoint_constraints": json_ready(
                                        correction_endpoint.get("constraints")
                                    ),
                                    "first_feasible_prefix": json_ready(prefix_report),
                                }
                            )
                        except (
                            RuntimeError,
                            TimeoutError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            visual_correction_failure = {
                                "reason": (
                                    "planning_timeout"
                                    if isinstance(exc, TimeoutError)
                                    else "planning_failed"
                                ),
                                "error": f"{type(exc).__name__}: {exc}",
                                "source_observation_sequence": sequence_after,
                                "source_max_anchor_error_m": maximum_anchor_error,
                                "source_max_constraint_error_m": (
                                    maximum_constraint_error
                                ),
                            }
                            motion_ok = False
                            motion_reason = "final_visual_correction_failed"
                            break

            q_error = float(np.linalg.norm(observed_final_q - target_q, ord=np.inf))
            stable = bool(
                goal_latched
                and last_goal_report.get("goal_satisfied")
                and not pending_waypoints
                and stationary
            )
            stable_pose_steps = stable_pose_steps + 1 if stable else 0
            stability_trace.append(
                {
                    "observation_sequence": sequence_after,
                    "goal_satisfied": bool(last_goal_report.get("goal_satisfied")),
                    "goal_reason": last_goal_report.get("reason"),
                    "q_error_inf_rad": q_error,
                    "position_step_inf_rad": position_step,
                    "stationary": stationary,
                    "stationarity_source": "consecutive_evaluator_qpos",
                    "raw_qvel_available": False,
                    "position_error_m": last_goal_report.get("position_error_m"),
                    "orientation_error_deg": last_goal_report.get(
                        "orientation_error_deg"
                    ),
                    "pending_command_kind": pending_command_kind,
                    "pending_waypoint_count": len(pending_waypoints),
                    "visual_tracking_pause_active": visual_tracking_pause_active,
                    "stable": stable,
                    "stable_steps": stable_pose_steps,
                    "command_q": final_command_q.astype(float).tolist(),
                    "goal_report": json_ready(last_goal_report),
                }
            )
            if len(stability_trace) > 16:
                del stability_trace[:-16]
            if stable_pose_steps >= STABLE_STEPS:
                motion_reason = "stable_pose_verified"
                break
        if motion_ok and stable_pose_steps < STABLE_STEPS:
            motion_ok = False
            motion_reason = (
                str(last_goal_report.get("reason"))
                if last_goal_report.get("reason")
                else "final_state_not_stable"
            )
    stability = {
        "ok": bool(motion_ok and stable_pose_steps >= STABLE_STEPS),
        "reason": (
            "stable_pose_verified"
            if motion_ok and stable_pose_steps >= STABLE_STEPS
            else motion_reason
        ),
        "stable_steps": stable_pose_steps,
        "required_stable_steps": STABLE_STEPS,
        "verification_steps": final_steps,
        "verification_step_budget": final_step_budget,
        "trace": stability_trace,
        "recent_stability_trace": stability_trace,
        "max_stability_steps": FINAL_MAX_STABILITY_STEPS,
        "goal_latched": goal_latched,
        "goal_latch_observation_sequence": goal_latch_sequence,
        "initial_planned_target_q": initial_target_q.astype(float).tolist(),
        "planned_target_q": target_q.astype(float).tolist(),
        "hold_target_q": final_command_q.astype(float).tolist(),
        "velocity_evidence": {
            "raw_qvel_available": False,
            "source": "consecutive_evaluator_qpos",
            "position_step_tolerance_rad": FINAL_COHERENCE_POSITION_STEP_RAD,
        },
        "load_compensation": {
            "applicable": True,
            "mode": "bounded_joint_command_feedforward_x5_position_control",
            "rounds": load_compensation_rounds,
            "max_rounds": FINAL_LOAD_MAX_ROUNDS,
            "total_increment_rad": load_compensation_total.astype(float).tolist(),
            "immutable_signed_target_changed": False,
            "trace": load_compensation_trace,
        },
        "final_load_compensation_rounds": json_ready(load_compensation_trace),
        "final_load_compensation_round_count": load_compensation_rounds,
        "final_load_compensation_pending_waypoints": len(pending_waypoints),
        "pending_command_kind": pending_command_kind,
        "final_visual_tracking_pause_active": visual_tracking_pause_active,
        "final_visual_tracking_loss_steps": visual_tracking_loss_steps,
        "final_visual_tracking_recovery_events": json_ready(
            visual_tracking_recovery_events
        ),
        "final_visual_correction_rounds": json_ready(visual_correction_rounds),
        "final_visual_correction_round_count": len(visual_correction_rounds),
        "final_visual_correction_failure": json_ready(visual_correction_failure),
        "final_visual_correction": {
            "minimum_controlled_points": FINAL_VISUAL_MIN_POINTS,
            "maximum_rounds": FINAL_VISUAL_MAX_ROUNDS,
            "maximum_anchor_rebase_m": FINAL_VISUAL_MAX_ANCHOR_REBASE_M,
            "maximum_constraint_error_m": FINAL_VISUAL_MAX_CONSTRAINT_ERROR_M,
            "maximum_eef_translation_m": FINAL_VISUAL_MAX_TRANSLATION_M,
            "maximum_planning_s": FINAL_VISUAL_MAX_PLANNING_S,
            "tracking_reacquire_max_steps": FINAL_VISUAL_REACQUIRE_MAX_STEPS,
            "trajectory_integrity": "hmac-sha256",
            "simulator_state_used": False,
        },
        "last_observation_sequence": _sequence(ctx),
        "last_live_goal_report": json_ready(last_goal_report),
    }

    final_live = _live(ctx, names, episode_id, manager=manager)
    final_sequence = _sequence(ctx)
    final_position, final_quaternion = _eef_base_xyzw(ctx, arm)
    final_observation_bundle = _final_observation_bundle_x5(
        final_live,
        controlled_names=controlled_names,
        fixed_names=reference_names,
        current_position=final_position,
        current_quaternion=final_quaternion,
        anchors_eef=anchors,
        current_sequence=final_sequence,
        anchor_tolerance_m=anchor_tolerance,
        pos_tol_m=float(args["pos_tol"]),
        adapter=ctx.world._official_adapter,
        base_rotation=ctx.world.base_rotation,
    )
    final_binding_ok = bool(
        final_live.get("session_id") == session_id
        and final_live.get("image_id") == image_id
    )
    if not final_binding_ok:
        final_observation_bundle = {
            **final_observation_bundle,
            "ok": False,
            "reason": "tracker_binding_replaced",
            "expected_session_id": session_id,
            "tracker_session_id": final_live.get("session_id"),
            "expected_image_id": image_id,
            "tracker_image_id": final_live.get("image_id"),
        }
    final_tracking_ok = bool(final_observation_bundle.get("ok"))
    final_reference_validation: dict[str, Any] = {
        "ok": True,
        "applicable": False,
    }
    final_reference_points = None
    final_reference_points_for_constraints = None
    if final_tracking_ok:
        try:
            final_points = np.asarray(
                final_observation_bundle["points"], dtype=np.float64
            )
            final_reference_points = (
                np.asarray(final_observation_bundle["fixed_points"], dtype=np.float64)
                if reference_names
                else None
            )
            final_all_points = np.vstack(
                [
                    final_points,
                    final_reference_points
                    if final_reference_points is not None
                    else np.empty((0, 3), dtype=np.float64),
                ]
            )
            predicted = np.asarray(
                final_observation_bundle["predicted_points"], dtype=np.float64
            )
            final_anchor_errors = np.linalg.norm(final_points - predicted, axis=1)
            final_anchor_ok = bool(
                float(np.max(final_anchor_errors)) <= anchor_tolerance
            )
            final_geometry = evaluate_rigid_geometry_consistency(
                final_points,
                source_points,
                tolerance_m=geometry_tolerance,
            )
            if reference_names:
                reference_matrix = np.asarray(
                    reference_points, dtype=np.float64
                ).reshape(len(reference_names), 3)
                reference_limit = max(0.020, 2.0 * float(args["pos_tol"]))
                final_reference_validation = _off_hand_reference_consensus(
                    reference_names,
                    final_reference_points,
                    reference_matrix,
                    tolerance_m=reference_limit,
                    allow_visual_outlier_suppression=True,
                )
                # Validate the live scene references above.  The default/UI
                # precision policy evaluates the terminal pair in that same
                # synchronized frame; explicit legacy tight-tolerance calls
                # retain the frozen planning coordinates for compatibility.
                final_reference_points_for_constraints = (
                    final_reference_points.copy()
                    if strict_live_reference_goal
                    else reference_matrix
                )
            else:
                final_reference_points = None
                final_reference_points_for_constraints = None
                final_reference_validation = {"ok": True, "applicable": False}
            base_constraints = evaluate_target_constraints(
                final_points,
                list(trajectory["target"].get("planning_target_points") or []),
                tolerance_m=float(args["pos_tol"]),
                relations=list(trajectory["target"].get("relations") or []),
                inequalities=list(trajectory["target"].get("inequalities") or []),
                orientation_tolerance_deg=float(args["ori_tol_deg"]),
                fixed_points_robot_base_m=final_reference_points_for_constraints,
                fixed_target_points=(
                    list(trajectory["target"].get("fixed_target_points") or [])
                    if reference_names
                    else None
                ),
            )
            quick_report = (
                _evaluate_quick_constraints(
                    final_points,
                    final_reference_points_for_constraints,
                    quick_specs,
                    controlled_names=controlled_names,
                    reference_names=reference_names,
                    position_tolerance_m=float(args["pos_tol"]),
                    orientation_tolerance_deg=float(args["ori_tol_deg"]),
                    camera_position_robot_base_m=target_camera_position,
                    camera_quaternion_xyzw=target_camera_quaternion,
                )
                if quick_mode
                else {"ok": True, "applicable": False}
            )
            final_constraints = {
                **base_constraints,
                "ok": bool(
                    base_constraints.get("ok")
                    and quick_report.get("ok")
                    and final_reference_validation.get("ok")
                ),
                "quick_constraint": quick_report,
                "quick_constraints": quick_report.get("groups", []),
                "off_hand_reference": final_reference_validation,
                "error": (
                    None
                    if base_constraints.get("ok")
                    and quick_report.get("ok")
                    and final_reference_validation.get("ok")
                    else str(
                        quick_report.get("error")
                        or (
                            "off-hand reference points moved"
                            if not final_reference_validation.get("ok")
                            else "combined coordinate and geometry constraints failed"
                        )
                    )
                ),
            }
            precision_gate = _strict_position_precision_gate(base_constraints)
            final_constraints["strict_position_precision_gate"] = precision_gate
            final_constraints["ok"] = bool(
                final_constraints.get("ok") and precision_gate["ok"]
            )
            if not precision_gate["ok"] and not final_constraints.get("error"):
                final_constraints["error"] = (
                    "strict terminal position gate failed: "
                    f"max_constraint_error={precision_gate['max_constraint_error_m']:.6f}m "
                    f"must be < {precision_gate['limit_m']:.6f}m"
                )
        except (KeyError, TypeError, ValueError) as exc:
            final_tracking_ok = False
            final_all_points = np.empty((0, 3), dtype=np.float64)
            final_points = np.empty((0, 3), dtype=np.float64)
            final_anchor_errors = np.full(len(controlled_names), float("inf"))
            final_anchor_ok = False
            final_geometry = {
                "applicable": len(controlled_names) >= 2,
                "ok": False,
                "tolerance_m": geometry_tolerance,
                "error": f"tracked_constraint_validation_failed: {exc}",
            }
            final_reference_validation = {
                "ok": False,
                "applicable": bool(reference_names),
                "reason": f"tracked_constraint_validation_failed: {exc}",
            }
            final_constraints = {
                "ok": False,
                "error": f"tracked_constraint_validation_failed: {exc}",
                "relations": [],
                "quick_constraint": {
                    "ok": False,
                    "error": f"tracked_constraint_validation_failed: {exc}",
                },
                "quick_constraints": [],
                "off_hand_reference": final_reference_validation,
            }
    else:
        final_all_points = np.empty((0, 3), dtype=np.float64)
        final_points = np.empty((0, 3), dtype=np.float64)
        final_anchor_errors = np.full(len(controlled_names), float("inf"))
        final_anchor_ok = False
        final_geometry = {
            "applicable": len(controlled_names) >= 2,
            "ok": False,
            "tolerance_m": geometry_tolerance,
            "error": str(
                final_observation_bundle.get("reason")
                or final_live.get("reason")
                or "tracker_observation_mismatch"
            ),
        }
        final_constraints = {
            "ok": False,
            "error": str(
                final_observation_bundle.get("reason")
                or final_live.get("reason")
                or "tracker_observation_mismatch"
            ),
        }
        final_reference_validation = {
            "ok": not bool(reference_names),
            "applicable": bool(reference_names),
            "reason": (
                "final_tracker_observation_unavailable" if reference_names else None
            ),
        }
    final_live_points = (
        {
            name: final_all_points[index].astype(float).tolist()
            for index, name in enumerate(names)
        }
        if final_tracking_ok
        else {}
    )
    per_point_constraint_errors = {name: 0.0 for name in names}
    for item in list(final_constraints.get("fixed_coordinates", [])) + list(
        final_constraints.get("affine_constraints", [])
    ):
        point_name = str(item.get("point") or "")
        raw_error = item.get("error_m", item.get("residual_m", 0.0))
        try:
            error_value = abs(float(raw_error))
        except (TypeError, ValueError):
            error_value = float("inf")
        if point_name in per_point_constraint_errors:
            per_point_constraint_errors[point_name] = max(
                per_point_constraint_errors[point_name], error_value
            )
    per_relation_errors = list(final_constraints.get("relations", []))
    per_inequality_errors = list(final_constraints.get("inequalities", []))
    max_anchor_error = (
        float(np.max(final_anchor_errors)) if final_anchor_errors.size else float("inf")
    )
    pair_distance_drift = float(
        final_geometry.get(
            "max_pair_distance_error_m",
            final_geometry.get("distance_error_m", float("inf")),
        )
    )
    final_precision_gate = final_constraints.get("strict_position_precision_gate")
    if not isinstance(final_precision_gate, Mapping):
        final_precision_gate = _strict_position_precision_gate(final_constraints)
    post_validation = {
        "ok": bool(
            final_tracking_ok
            and final_constraints.get("ok")
            and final_geometry.get("ok")
            and final_reference_validation.get("ok")
            and final_precision_gate.get("ok")
        ),
        "required_for_success": True,
        "source": (
            "subsequent_evaluator_rgbd_and_proprio_local_fk_for_explicit_occlusion"
            if final_observation_bundle.get("fk_completed_controlled_names")
            else "subsequent_evaluator_observation"
        ),
        "episode_id": episode_id,
        "observation_sequence": final_sequence,
        "tracker_observation_sequence": final_live.get("observation_sequence"),
        "synchronized_with_final_proprioception": final_tracking_ok,
        "input_constraints": args["points"],
        "input_relations": args["relations"],
        "input_inequalities": args["inequalities"],
        "quick_constraint": json_ready(quick_spec) if quick_mode else None,
        "quick_constraints": json_ready(quick_specs) if quick_mode else [],
        "on_hand_point_names": list(controlled_names),
        "off_hand_point_names": list(reference_names),
        "off_hand_reference": json_ready(final_reference_validation),
        "points_robot_base_m": final_live_points,
        "point_sources": json_ready(final_observation_bundle.get("point_sources", {})),
        "measured_on_hand_point_names": list(
            final_observation_bundle.get("measured_controlled_names", [])
        ),
        "fk_completed_occluded_on_hand_point_names": list(
            final_observation_bundle.get("fk_completed_controlled_names", [])
        ),
        "on_hand_observation_completion": json_ready(final_observation_bundle),
        "prediction_values_published_as_observation": False,
        "off_hand_constraint_reference_points_robot_base_m": (
            {
                name: final_reference_points_for_constraints[index]
                .astype(float)
                .tolist()
                for index, name in enumerate(reference_names)
            }
            if final_tracking_ok
            and reference_names
            and final_reference_points_for_constraints is not None
            else {}
        ),
        "tolerance_m": float(args["pos_tol"]),
        "requested_tolerance_m": float(requested_position_tolerance_m),
        "effective_tolerance_m": float(effective_position_tolerance_m),
        "strict_position_precision_gate": json_ready(final_precision_gate),
        "max_constraint_error_m": final_constraints.get("max_constraint_error_m"),
        "constraint_report": json_ready(final_constraints),
        "rigid_pair_consistency": json_ready(final_geometry),
        "per_point_constraint_errors": per_point_constraint_errors,
        "per_relation_errors": json_ready(per_relation_errors),
        "per_inequality_errors": json_ready(per_inequality_errors),
        "max_inequality_violation_m": final_constraints.get(
            "max_inequality_violation_m", float("inf")
        ),
        "minimum_inequality_clearance_m": final_constraints.get(
            "minimum_inequality_clearance_m"
        ),
        "max_anchor_error_m": max_anchor_error,
        "pair_distance_drift_m": pair_distance_drift,
        "max_kabsch_bundle_residual_m": float(
            final_geometry.get("max_kabsch_bundle_residual_m", float("inf"))
        ),
    }
    # A base-frame constraint report is not sufficient to claim that the
    # rendered scene is aligned: points from different cameras or evaluator
    # sequences can still fit one 3-D line.  For a three-point request, use
    # the tracker's strict same-camera RGB-D gate when the live implementation
    # provides it.  Synthetic/legacy test providers without camera geometry
    # remain explicitly ``not_applicable`` rather than being rejected here.
    visual_alignment: dict[str, Any] = {
        "ok": True,
        "applicable": False,
        "reason": "visual_alignment_verifier_unavailable",
        "point_names": list(visual_names),
    }
    visual_alignment_gate = getattr(ctx.tracker, "visual_alignment_gate", None)
    # EEF anchors are kinematic bookkeeping points, not image tracks.  The
    # strict visual check must inspect exactly the user-selected RGB-D points;
    # including a predefined anchor would either make the check meaningless
    # or let an old compatibility entry select the wrong camera group.
    if callable(visual_alignment_gate) and len(visual_names) >= 3:
        try:
            visual_alignment = visual_alignment_gate(
                ctx.world._official_adapter,
                visual_names,
                expected_sequence=int(final_sequence),
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            visual_alignment = {
                "ok": False,
                "applicable": True,
                "reason": f"visual_alignment_verifier_error: {exc}",
                "point_names": list(visual_names),
                "expected_sequence": int(final_sequence),
            }
    visual_alignment_required = bool(visual_alignment.get("applicable"))
    post_validation["visual_alignment"] = json_ready(visual_alignment)
    post_validation["visual_alignment_point_names"] = list(visual_names)
    post_validation["visual_alignment_required"] = visual_alignment_required
    if visual_alignment_required:
        post_validation["ok"] = bool(
            post_validation.get("ok") and visual_alignment.get("ok")
        )
    overall_ok = bool(
        motion["ok"] and stability["ok"] and post_validation["ok"] and final_anchor_ok
    )
    motion_reason = str(motion.get("reason") or "tracking_failed")
    stability_reason = str(stability.get("reason") or "tracking_failed")
    failure_stage = None
    result_error = None
    if not overall_ok:
        failure_stage = (
            "timeout" if "timeout" in {motion_reason, stability_reason} else "tracking"
        )
        if not bool(motion.get("ok")):
            result_error = f"trajectory execution failed: {motion_reason}"
        elif not bool(stability.get("ok")):
            result_error = str(
                stability.get("error") or f"final stability failed: {stability_reason}"
            )
        elif not bool(post_validation.get("ok")):
            if reference_names and not bool(final_reference_validation.get("ok")):
                result_error = str(
                    final_reference_validation.get("error")
                    or "off-hand reference points moved during execution"
                )
            elif not bool(final_precision_gate.get("ok")):
                result_error = (
                    "strict terminal position gate failed: "
                    f"max_constraint_error={float(final_precision_gate.get('max_constraint_error_m', float('inf'))):.6f}m "
                    f"must be < {float(final_precision_gate.get('limit_m', FINAL_EXECUTION_POSITION_LIMIT_M)):.6f}m"
                )
            elif visual_alignment_required and not bool(visual_alignment.get("ok")):
                result_error = "final same-frame visual alignment failed: " + str(
                    visual_alignment.get("reason")
                    or "visual alignment gate rejected the result"
                )
            elif not bool(final_geometry.get("ok")):
                result_error = (
                    "final tracked rigid geometry changed: "
                    f"max_pair_error={float(final_geometry.get('max_pair_distance_error_m', final_geometry.get('distance_error_m', float('inf')))):.4f}m "
                    f"max_kabsch_bundle_error={float(final_geometry.get('max_kabsch_bundle_residual_m', float('inf'))):.4f}m"
                )
            else:
                result_error = str(
                    final_constraints.get("error")
                    or "final live tracked-point constraints were not satisfied"
                )
        else:
            result_error = "final tracked points were not rigid with the selected EEF"
    if overall_ok:
        if len(visual_controlled_names) == 2:
            rigid_pair_lease_state["preserve_after_success"] = True
        motion_retention_release = (
            _move_tracked_preserve_motion_retention_for_successor(retention_state)
        )
    else:
        motion_retention_release = _move_tracked_release_motion_retention(
            manager, retention_state
        )
    result_writer(
        ctx,
        "move_tracked_point",
        {
            **_common_result(),
            "ok": overall_ok,
            "error": result_error,
            "failure_stage": failure_stage,
            "build": MOVE_TRACKED_POINT_LOCAL_BUILD,
            "execution_mode": "exec",
            "session_id": session_id,
            "image_id": image_id,
            "plan_id": plan_id,
            "plan_record_path": plan_path,
            "trajectory_schema": trajectory["schema"],
            "trajectory_schema_version": trajectory["schema_version"],
            "trajectory_digest": digest,
            "trajectory_waypoint_count": len(trajectory["waypoints"]),
            "execution_waypoint_count": int(
                schedule.get(
                    "selected_waypoint_count",
                    len(trajectory["waypoints"]),
                )
            ),
            "robot_profile": ROBOT_PROFILE,
            "arm": arm,
            "eef_mean_distance_m": distances,
            "input_point_order": names,
            "internal_point_order": names,
            "point_count": len(names),
            "point_a_selection": (
                "nearest_to_selected_eef" if len(names) >= 2 else "single_point"
            ),
            "nearest_point_name": nearest_name,
            "all_points_same_rigid_object_required": (
                not quick_mode and len(controlled_names) >= 2
            ),
            "two_points_same_rigid_object_required": (
                not quick_mode and len(controlled_names) == 2
            ),
            "quick_constraint": json_ready(quick_spec) if quick_mode else None,
            "quick_constraints": json_ready(quick_specs) if quick_mode else [],
            "inequalities": json_ready(trajectory["target"].get("inequalities") or []),
            "on_hand_point_names": list(controlled_names),
            "off_hand_point_names": list(reference_names),
            "predefined_eef_point_names": list(predefined_names),
            "visual_tracked_point_names": list(visual_names),
            "predefined_eef_arm_selection": json_ready(predefined_arm_selection),
            "off_hand_start_validation": json_ready(off_hand_start_validation),
            "off_hand_final_validation": json_ready(final_reference_validation),
            "rigid_pair_activation": json_ready(rigid_pair_activation),
            "rigid_pair_activation_confirmation": json_ready(
                rigid_pair_activation_confirmation
            ),
            "rigid_pair_activation_hold_steps": int(rigid_pair_activation_hold_steps),
            "prior_rigid_pair_cleanup": json_ready(prior_rigid_pair_cleanup),
            "failed_rigid_pair_cleanup": json_ready(failed_rigid_pair_cleanup),
            "rigid_pair_bootstrap_eef_observation_prior": json_ready(
                rigid_pair_bootstrap_eef_prior
            ),
            "on_hand_eef_observation_prior": json_ready(
                locals().get("eef_prior_activation", {"applicable": False})
            ),
            "preplanning_on_hand_eef_observation_prior": json_ready(
                preplanning_eef_prior_activation
            ),
            "preplanning_source_synchronization": json_ready(preplanning_source_sync),
            "preplanning_source_sync": json_ready(preplanning_source_sync),
            "post_planning_source_replans": json_ready(post_planning_source_replans),
            "post_planning_source_replan_count": len(post_planning_source_replans),
            "motion_tracking_retention": {
                "lease": json_ready(motion_retention),
                "release": json_ready(motion_retention_release),
            },
            "frozen_observation": {
                "episode_id": episode_id,
                "motion_epoch": frozen["motion_epoch"],
                "observation_sequence": observation_sequence,
                "initial_observation_sequence": initial_observation_sequence,
                "tracker_and_proprio_same_observation": True,
                "post_eef_prior_source_synchronized": bool(
                    preplanning_source_sync.get("planning_source_rebased", False)
                ),
            },
            "source_points_robot_base_m": {
                name: source_by_name[name].tolist() for name in names
            },
            "controlled_source_points_robot_base_m": {
                name: source_by_name[name].tolist() for name in controlled_names
            },
            "resolved_target_points_robot_base_m": trajectory["target"][
                "resolved_points_robot_base_m"
            ],
            "resolved_variables": trajectory["target"]["constraints"].get(
                "resolved_variables", {}
            ),
            "target_constraints": trajectory["target"]["constraints"],
            "endpoint_solver": trajectory["planner"]["endpoint_solver"],
            "planner": trajectory["planner"],
            "motion": motion,
            "execution_schedule": json_ready(schedule),
            "requested_max_steps": requested_max_steps,
            "effective_max_steps": int(args["max_steps"]),
            "requested_position_tolerance_m": float(requested_position_tolerance_m),
            "effective_position_tolerance_m": float(effective_position_tolerance_m),
            "strict_position_limit_m": float(FINAL_EXECUTION_POSITION_LIMIT_M),
            "strict_live_reference_goal": bool(strict_live_reference_goal),
            "dense_step_budget_expanded": bool(dense_step_budget_expanded),
            "step_budget_policy": (
                "port_unbounded_dense_replay"
                if dense_execution_policy
                else "caller_bounded"
            ),
            "tracking_continuity": json_ready(tracking_continuity),
            "gripper_recovery": {
                "applicable": True,
                "mode": "x5_v37_near_open_bounded_position_recovery",
                "effort_recovery_applicable": False,
                "position_recovery_applicable": True,
                "velocity_observation_available": False,
                "reason": (
                    "RoboDojo X5 exposes position-controlled grippers; the "
                    "BEHAVIOR R1 effort command is not physically applicable, "
                    "so the X5-equivalent recovery holds measured arm qpos and "
                    "uses consecutive evaluator gripper qpos as its quiet gate"
                ),
                "capture_hold_policy_applied_every_action": True,
                "capture_target_reissued_during_inward_recovery": True,
                "near_open_close_intent_suppressed": True,
                "verified_close_keepalive_unchanged": True,
                "target_policy": (
                    "near_open_inward_drift_only_intermediate_and_close_latch_unchanged"
                ),
                "stability_observation": ("consecutive_evaluator_qpos_position_step"),
                "official_joint_scale_m": list(ROBODOJO_X5_GRIPPER_SCALE_M),
                "open_hold_threshold_m": GRIPPER_OPEN_HOLD_THRESHOLD_M,
                "soft_limit_m": GRIPPER_RECOVERY_SOFT_LIMIT_M,
                "release_limit_m": GRIPPER_RECOVERY_RELEASE_M,
                "hard_limit_m": GRIPPER_RECOVERY_HARD_LIMIT_M,
                "max_steps": GRIPPER_RECOVERY_MAX_STEPS,
                "required_stable_steps": GRIPPER_RECOVERY_STABLE_STEPS,
                "position_step_threshold_m": GRIPPER_RECOVERY_POSITION_STEP_M,
                "steps": int(gripper_recovery_steps),
                "events": json_ready(gripper_recovery_events),
            },
            "stability": stability,
            "final_live_tracking": json_ready(final_live),
            "final_constraints": json_ready(final_constraints),
            "final_visual_alignment": json_ready(visual_alignment),
            "final_rigid_pair_consistency": json_ready(final_geometry),
            "post_execution_track_validation": post_validation,
            "final_anchor_ok": final_anchor_ok,
            "final_anchor_error_m": {
                name: float(final_anchor_errors[index])
                for index, name in enumerate(controlled_names)
            },
            "final_anchor_tolerance_m": anchor_tolerance,
            "final_live_points_robot_base_m": final_live_points,
            "per_point_constraint_errors": per_point_constraint_errors,
            "per_relation_errors": json_ready(per_relation_errors),
            "max_constraint_error_m": final_constraints.get(
                "max_constraint_error_m", float("inf")
            ),
            "strict_position_precision_gate": json_ready(final_precision_gate),
            "max_orientation_error_deg": final_constraints.get(
                "max_relation_error_deg", 0.0
            ),
            "max_anchor_error_m": max_anchor_error,
            "pair_distance_drift_m": pair_distance_drift,
            "max_kabsch_bundle_residual_m": float(
                final_geometry.get("max_kabsch_bundle_residual_m", float("inf"))
            ),
            "geometry_rank": final_geometry.get("geometry_rank"),
            "start_state_validation": start_validation,
            "collision_checked": False,
        },
    )
    yield ctx.world.hold_action()


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("move_tracked_point", move_tracked_point)
