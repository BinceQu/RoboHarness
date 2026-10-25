from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence
import numpy as np
from tools.util.contract import (
    ARM_DOF,
)
from tools.util.geometry import (
    finite_vector,
    json_ready,
    quaternion_angle_deg,
)
from tools.util.x5_kinematics import (
    X5_JOINT_LOWER,
    X5_JOINT_UPPER,
    calibrated_pose_for_joints,
    quaternion_to_matrix,
    solve_calibrated_x5_ik,
)
from tools.util.x5_grasp_geometry import x5_gripper_geometry
from tools.util.tracked_point_motion_x5_support import LocalRobotState

from tools.util.motion_constants import (
    EEF_MAX_JOINT_DELTA_PER_ACTION,
    EEF_IK_ORIENTATION_WEIGHT_M_PER_RAD,
    EXEC_JOINT_SETTLE_TOLERANCE_RAD,
    EXEC_JOINT_TRACKING_BLOCKED_RAD,
    EXEC_MAX_ACTION_STEPS,
    EXEC_TOTAL_ACTION_BUDGET,
    EXEC_MOVE_TRACKED_DENSE_ACTION_BUDGET,
    EXEC_MOVE_TRACKED_SCHEDULE_BUDGET,
    EXEC_MOVE_TRACKED_PREFERRED_WAYPOINTS,
    EXEC_STALL_LIMIT,
    EXEC_MIN_PROGRESS_SCORE,
    EXEC_MIN_OBSERVED_JOINT_STEP_RAD,
    EXEC_MIN_OBSERVED_TRANSLATION_STEP_M,
    EXEC_MIN_OBSERVED_ORIENTATION_STEP_DEG,
    EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD,
    EXEC_FINAL_STABLE_STEPS,
    EXEC_FINAL_TRANSLATION_STEP_TOLERANCE_M,
    EXEC_FINAL_ORIENTATION_STEP_TOLERANCE_DEG,
    EXEC_LOAD_STABLE_STEPS,
    EXEC_LOAD_MIN_RESIDUAL_RAD,
    EXEC_LOAD_MAX_ROUNDS,
    EXEC_LOAD_MAX_ROUND_RAD,
    EXEC_LOAD_MAX_TOTAL_RAD,
    EXEC_CLEARANCE_BRIDGE_CARTESIAN_STEP_M,
    EXEC_CLEARANCE_BRIDGE_ORIENTATION_STEP_DEG,
    EXEC_CLEARANCE_BRIDGE_JOINT_STEP_RAD,
    EXEC_CLEARANCE_BRIDGE_LIFT_M,
    EXEC_CLEARANCE_BRIDGE_INITIAL_RISE_M,
    EXEC_CLEARANCE_BRIDGE_ROTATION_CLEARANCE_M,
    EXEC_CLEARANCE_BRIDGE_Z_TOLERANCE_M,
    EXEC_CLEARANCE_BRIDGE_MAX_WAYPOINTS,
    _COMPAT_PLAN_MODES,
)
from tools.util.motion_common import (
    _finite_number,
    _move_tracked_exports,
    _observation_sequence,
    _quaternion_slerp,
    _set_result,
    _stationary_capture_confirmation,
)


def _trajectory_uses_dense_execution(
    trajectory: Mapping[str, Any],
) -> bool:
    """Read the signed execution policy across resident module revisions.

    Perception hot reload stages ``tools.py`` in a private namespace while
    preserving the resident move module.  An older resident process may not
    expose this helper yet, so keep the small schema fallback here instead of
    making grasp/tracking deployment depend on the age of an unrelated module.
    """

    from tools import move_tracked_point as _resident_move_tracked_module

    resident_helper = getattr(
        _resident_move_tracked_module,
        "_trajectory_uses_dense_execution",
        None,
    )
    if callable(resident_helper):
        return bool(resident_helper(trajectory))
    tracking = trajectory.get("tracking")
    if not isinstance(tracking, Mapping):
        return False
    policy = str(tracking.get("execution_waypoint_policy") or "").strip()
    if policy == "all_signed_waypoints":
        return True
    if policy == "bounded_monotone_subsample":
        return False
    compression_disabled = tracking.get("execution_compression_disabled")
    if isinstance(compression_disabled, bool):
        return compression_disabled
    # Legacy plans have no policy metadata and retain dense replay semantics.
    return True


def _forward_signed_segment_admission(
    observed_pose: Any,
    waypoint_pose: Any,
    next_waypoint_pose: Any | None,
    *,
    pos_tolerance: float,
    ori_tolerance_deg: float,
) -> dict[str, Any]:
    """Accept a nonterminal waypoint already passed on its signed segment."""

    observed = finite_vector(observed_pose, 7, "observed EEF pose")
    start = finite_vector(waypoint_pose, 7, "signed waypoint pose")
    if next_waypoint_pose is None:
        return {
            "checked": False,
            "ok": False,
            "reason": "terminal_waypoint_has_no_forward_segment",
        }
    end = finite_vector(next_waypoint_pose, 7, "next signed waypoint pose")
    segment = end[:3] - start[:3]
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1.0e-12:
        return {
            "checked": True,
            "ok": False,
            "reason": "degenerate_forward_segment",
            "segment_length_m": math.sqrt(max(0.0, length_sq)),
        }

    segment_length = math.sqrt(length_sq)
    raw_fraction = float(np.dot(observed[:3] - start[:3], segment) / length_sq)
    fraction = float(np.clip(raw_fraction, 0.0, 1.0))
    expected_position = start[:3] + fraction * segment
    expected_quaternion = _quaternion_slerp(start[3:], end[3:], fraction)
    cross_track_error = float(np.linalg.norm(observed[:3] - expected_position))
    orientation_error = quaternion_angle_deg(observed[3:], expected_quaternion)
    endpoint_extension = float(pos_tolerance) / segment_length
    minimum_forward_fraction = endpoint_extension
    inside_longitudinal_bounds = bool(
        minimum_forward_fraction <= raw_fraction <= 1.0 + endpoint_extension
    )
    admitted = bool(
        inside_longitudinal_bounds
        and cross_track_error <= pos_tolerance
        and orientation_error <= ori_tolerance_deg
    )
    return {
        "checked": True,
        "ok": admitted,
        "reason": (
            "observed_pose_is_on_forward_signed_segment"
            if admitted
            else "observed_pose_is_outside_forward_signed_segment"
        ),
        "projection_fraction_unclamped": raw_fraction,
        "projection_fraction": fraction,
        "segment_length_m": segment_length,
        "along_track_progress_m": fraction * segment_length,
        "minimum_forward_progress_m": float(pos_tolerance),
        "minimum_forward_fraction": minimum_forward_fraction,
        "cross_track_error_m": cross_track_error,
        "position_tolerance_m": float(pos_tolerance),
        "orientation_error_deg": orientation_error,
        "orientation_tolerance_deg": float(ori_tolerance_deg),
        "expected_pose": [
            *expected_position.tolist(),
            *expected_quaternion.tolist(),
        ],
        "signed_waypoint_pose": start.tolist(),
        "next_signed_waypoint_pose": end.tolist(),
        "terminal_waypoint_skipping_allowed": False,
    }


def _next_x5_load_compensation_command(
    *,
    terminal_joint: np.ndarray,
    command_target: np.ndarray,
    observed_joint: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Offset the position command while preserving the signed observed target."""

    residual = terminal_joint - observed_joint
    increment = np.clip(
        residual,
        -EXEC_LOAD_MAX_ROUND_RAD,
        EXEC_LOAD_MAX_ROUND_RAD,
    )
    total_lower = np.maximum(
        X5_JOINT_LOWER,
        terminal_joint - EXEC_LOAD_MAX_TOTAL_RAD,
    )
    total_upper = np.minimum(
        X5_JOINT_UPPER,
        terminal_joint + EXEC_LOAD_MAX_TOTAL_RAD,
    )
    candidate = np.clip(command_target + increment, total_lower, total_upper)
    applied = candidate - command_target
    return candidate, {
        "observed_joint_position": observed_joint.tolist(),
        "signed_terminal_joint_target": terminal_joint.tolist(),
        "command_target_before": command_target.tolist(),
        "command_target_after": candidate.tolist(),
        "observed_residual_rad": residual.tolist(),
        "observed_residual_inf_rad": float(np.linalg.norm(residual, ord=np.inf)),
        "increment_rad": applied.tolist(),
        "increment_inf_rad": float(np.linalg.norm(applied, ord=np.inf)),
        "total_increment_rad": (candidate - terminal_joint).tolist(),
        "immutable_signed_target_changed": False,
        "cartesian_replan_used": False,
    }


def _drive_exec_waypoint(
    ctx,
    *,
    arm: str,
    target: Any,
    forward_target: Any | None,
    planned_joint_target: Any | None,
    pos_tolerance: float,
    ori_tolerance_deg: float,
    max_steps: int,
    timeout_s: float,
    require_stable_capture: bool = False,
):
    """Replay one immutable joint target and verify it against live EEF state."""
    target_pose = finite_vector(target, 7, f"{arm} EEF target")
    started = time.monotonic()
    deadline = started + timeout_s
    action_limit = max(1, min(int(max_steps), EXEC_MAX_ACTION_STEPS))
    observed_joint = np.asarray(ctx.world.arm_qpos_list(arm), dtype=np.float64)
    observed_pose = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
    joint_target_source = "signed_planned_joint_target"
    ik_metadata: dict[str, Any] | None = None

    try:
        if planned_joint_target is not None:
            terminal_joint = finite_vector(
                planned_joint_target,
                6,
                f"{arm} signed planned joint target",
            )
        else:
            joint_target_source = "waypoint_entry_policy_local_ik"
            solution = solve_calibrated_x5_ik(
                observed_joint,
                observed_pose,
                target_pose,
                orientation_weight_m_per_rad=(EEF_IK_ORIENTATION_WEIGHT_M_PER_RAD),
            )
            initial_score = solution.initial_position_error_m + (
                EEF_IK_ORIENTATION_WEIGHT_M_PER_RAD
                * math.radians(solution.initial_orientation_error_deg)
            )
            solved_score = solution.position_error_m + (
                EEF_IK_ORIENTATION_WEIGHT_M_PER_RAD
                * math.radians(solution.orientation_error_deg)
            )
            if not solution.converged and solved_score >= initial_score - 1e-5:
                raise ValueError(
                    "policy-local X5 IK could not find a Cartesian-improving "
                    "joint target"
                )
            terminal_joint = solution.joint_target.copy()
            ik_metadata = {
                "converged": solution.converged,
                "iterations": solution.iterations,
                "position_error_m": solution.position_error_m,
                "orientation_error_deg": solution.orientation_error_deg,
                "initial_position_error_m": (solution.initial_position_error_m),
                "initial_orientation_error_deg": (
                    solution.initial_orientation_error_deg
                ),
                "calibrated_base_translation_m": (
                    solution.calibration_translation_m.tolist()
                ),
            }
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        position_error = float(np.linalg.norm(observed_pose[:3] - target_pose[:3]))
        orientation_error = quaternion_angle_deg(observed_pose[3:], target_pose[3:])
        return {
            "converged": False,
            "observed_pose": observed_pose.tolist(),
            "observed_joint_position": observed_joint.tolist(),
            "position_error_m": position_error,
            "orientation_error_deg": orientation_error,
            "terminal_joint_error_rad": None,
            "action_steps": 0,
            "elapsed_s": time.monotonic() - started,
            "trace": [],
            "stop_reason": f"joint_target_resolution_failed: {exc}",
            "controller": "immutable_joint_target_replay",
            "joint_target_source": joint_target_source,
            "terminal_joint_target": None,
            "joint_settle_tolerance_rad": (EXEC_JOINT_SETTLE_TOLERANCE_RAD),
            "joint_tracking_blocked_threshold_rad": (EXEC_JOINT_TRACKING_BLOCKED_RAD),
            "max_joint_delta_per_action_rad": (EEF_MAX_JOINT_DELTA_PER_ACTION),
            "max_joint_tracking_error_rad": 0.0,
            "predicted_terminal_pose": None,
            "entry_ik": ik_metadata,
            "final_stable_steps": 0,
            "final_stable_steps_required": EXEC_FINAL_STABLE_STEPS,
            "final_position_step_tolerance_rad": EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD,
            "terminal_acceptance": None,
            "load_compensation": {
                "rounds": 0,
                "immutable_signed_target_changed": False,
            },
            "stationary_capture_confirmation": _stationary_capture_confirmation(
                require_stable_capture,
                0,
            ),
            "forward_segment_admission": {
                "checked": False,
                "ok": False,
                "reason": "joint_target_resolution_failed",
            },
        }

    predicted_terminal_pose = calibrated_pose_for_joints(
        observed_joint,
        observed_pose,
        terminal_joint,
    )
    position_error = float(np.linalg.norm(observed_pose[:3] - target_pose[:3]))
    orientation_error = quaternion_angle_deg(observed_pose[3:], target_pose[3:])
    terminal_joint_error = float(np.max(np.abs(terminal_joint - observed_joint)))
    trace: list[dict[str, Any]] = []
    max_joint_tracking_error = 0.0
    command_joint_target = terminal_joint.copy()
    load_stationary_steps = 0
    load_compensation_rounds = 0
    load_compensation_trace: list[dict[str, Any]] = []
    best_score = max(
        terminal_joint_error / EXEC_JOINT_SETTLE_TOLERANCE_RAD,
        position_error / pos_tolerance,
        orientation_error / ori_tolerance_deg,
    )
    stalled_steps = 0
    stable_cartesian_steps = 0
    forward_segment_admission = _forward_signed_segment_admission(
        observed_pose,
        target_pose,
        forward_target,
        pos_tolerance=pos_tolerance,
        ori_tolerance_deg=ori_tolerance_deg,
    )

    def result(stop_reason: str, converged: bool) -> dict[str, Any]:
        return {
            "converged": converged,
            "observed_pose": observed_pose.tolist(),
            "observed_joint_position": observed_joint.tolist(),
            "position_error_m": position_error,
            "orientation_error_deg": orientation_error,
            "terminal_joint_error_rad": terminal_joint_error,
            "action_steps": len(trace),
            "elapsed_s": time.monotonic() - started,
            "trace": trace,
            "stop_reason": stop_reason,
            "controller": "immutable_joint_target_replay",
            "joint_target_source": joint_target_source,
            "terminal_joint_target": terminal_joint.tolist(),
            "joint_settle_tolerance_rad": (EXEC_JOINT_SETTLE_TOLERANCE_RAD),
            "joint_tracking_blocked_threshold_rad": (EXEC_JOINT_TRACKING_BLOCKED_RAD),
            "max_joint_delta_per_action_rad": (EEF_MAX_JOINT_DELTA_PER_ACTION),
            "max_joint_tracking_error_rad": max_joint_tracking_error,
            "predicted_terminal_pose": predicted_terminal_pose.tolist(),
            "entry_ik": ik_metadata,
            "final_stable_steps": stable_cartesian_steps,
            "final_stable_steps_required": EXEC_FINAL_STABLE_STEPS,
            "stationary_capture_confirmation": _stationary_capture_confirmation(
                require_stable_capture,
                stable_cartesian_steps,
            ),
            "final_position_step_tolerance_rad": (
                EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD
            ),
            "terminal_acceptance": (
                "forward_signed_segment_progress"
                if converged and stop_reason == "forward_signed_segment_progress"
                else (
                    (
                        "strict_joint_and_cartesian_after_bounded_load_compensation"
                        if load_compensation_rounds
                        else "strict_joint_and_cartesian"
                    )
                    if converged
                    and stop_reason in ("already_at_target", "target_reached")
                    else (
                        "stable_cartesian_with_bounded_static_joint_bias"
                        if converged
                        else None
                    )
                )
            ),
            "forward_segment_admission": forward_segment_admission,
            "load_compensation": {
                "enabled": True,
                "rounds": load_compensation_rounds,
                "max_rounds": EXEC_LOAD_MAX_ROUNDS,
                "stable_steps_before_round": EXEC_LOAD_STABLE_STEPS,
                "minimum_residual_rad": EXEC_LOAD_MIN_RESIDUAL_RAD,
                "max_round_increment_rad": EXEC_LOAD_MAX_ROUND_RAD,
                "max_total_increment_rad": EXEC_LOAD_MAX_TOTAL_RAD,
                "command_target_joint_position": command_joint_target.tolist(),
                "total_increment_rad": (command_joint_target - terminal_joint).tolist(),
                "trace": load_compensation_trace,
                "immutable_signed_target_changed": False,
            },
        }

    if (
        not require_stable_capture
        and terminal_joint_error <= EXEC_JOINT_SETTLE_TOLERANCE_RAD
        and position_error <= pos_tolerance
        and orientation_error <= ori_tolerance_deg
    ):
        return result("already_at_target", True)
    if forward_segment_admission["ok"] and not require_stable_capture:
        return result("forward_signed_segment_progress", True)

    for action_step in range(1, action_limit + 1):
        ctx.raise_if_cancelled(f"{arm} immutable joint target replay")
        sequence_before = _observation_sequence(ctx)
        observed_joint_before = observed_joint.copy()
        observed_pose_before = observed_pose.copy()
        full_delta = command_joint_target - observed_joint
        largest_delta = float(np.max(np.abs(full_delta)))
        command_scale = 1.0
        if largest_delta > EEF_MAX_JOINT_DELTA_PER_ACTION:
            command_scale = EEF_MAX_JOINT_DELTA_PER_ACTION / largest_delta
        commanded_joint = observed_joint + command_scale * full_delta
        predicted_command_pose = calibrated_pose_for_joints(
            observed_joint,
            observed_pose,
            commanded_joint,
        )
        yield ctx.world.make_joint_action(**{f"{arm}_arm": commanded_joint.tolist()})
        sequence_after = _observation_sequence(ctx)
        observed_joint = np.asarray(ctx.world.arm_qpos_list(arm), dtype=np.float64)
        observed_pose = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
        joint_tracking_error = np.abs(commanded_joint - observed_joint)
        step_max_joint_tracking_error = float(np.max(joint_tracking_error))
        max_joint_tracking_error = max(
            max_joint_tracking_error,
            step_max_joint_tracking_error,
        )
        terminal_joint_error = float(np.max(np.abs(terminal_joint - observed_joint)))
        position_error = float(np.linalg.norm(observed_pose[:3] - target_pose[:3]))
        orientation_error = quaternion_angle_deg(observed_pose[3:], target_pose[3:])
        score = max(
            terminal_joint_error / EXEC_JOINT_SETTLE_TOLERANCE_RAD,
            position_error / pos_tolerance,
            orientation_error / ori_tolerance_deg,
        )
        observed_joint_step = float(
            np.linalg.norm(observed_joint - observed_joint_before, ord=np.inf)
        )
        cartesian_ok = bool(
            position_error <= pos_tolerance and orientation_error <= ori_tolerance_deg
        )
        stationary = bool(
            observed_joint_step <= EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD
            and np.linalg.norm(observed_pose[:3] - observed_pose_before[:3])
            <= EXEC_FINAL_TRANSLATION_STEP_TOLERANCE_M
            and quaternion_angle_deg(observed_pose_before[3:], observed_pose[3:])
            <= EXEC_FINAL_ORIENTATION_STEP_TOLERANCE_DEG
        )
        bounded_static_bias = bool(
            terminal_joint_error <= EXEC_JOINT_TRACKING_BLOCKED_RAD
        )
        stable_cartesian_steps = (
            stable_cartesian_steps + 1
            if cartesian_ok
            and stationary
            and bounded_static_bias
            and sequence_after > sequence_before
            else 0
        )
        trace.append(
            {
                "action_step": action_step,
                "observation_sequence_before": sequence_before,
                "observation_sequence_after": sequence_after,
                "controller": "immutable_joint_target_replay",
                "joint_target_source": joint_target_source,
                "terminal_joint_target": terminal_joint.tolist(),
                "command_target_joint_position": command_joint_target.tolist(),
                "observed_joint_position_before": (observed_joint_before.tolist()),
                "observed_joint_position": observed_joint.tolist(),
                "commanded_joint_position": commanded_joint.tolist(),
                "command_scale": command_scale,
                "largest_full_joint_delta_rad": largest_delta,
                "terminal_joint_error_rad": terminal_joint_error,
                "joint_tracking_error_rad": (joint_tracking_error.tolist()),
                "max_joint_tracking_error_rad": (step_max_joint_tracking_error),
                "position_error_m": position_error,
                "orientation_error_deg": orientation_error,
                "observed_translation_step_m": float(
                    np.linalg.norm(observed_pose[:3] - observed_pose_before[:3])
                ),
                "observed_joint_step_inf_rad": observed_joint_step,
                "observed_orientation_step_deg": quaternion_angle_deg(
                    observed_pose_before[3:], observed_pose[3:]
                ),
                "predicted_command_pose": predicted_command_pose.tolist(),
                "convergence_score": score,
                "final_stationary": stationary,
                "bounded_static_joint_bias": bounded_static_bias,
                "final_stable_steps": stable_cartesian_steps,
                "load_stationary_steps": load_stationary_steps,
                "load_compensation_round": load_compensation_rounds,
            }
        )
        if sequence_after <= sequence_before:
            return result("observation_not_advanced", False)
        strict_target_reached = bool(
            terminal_joint_error <= EXEC_JOINT_SETTLE_TOLERANCE_RAD
            and position_error <= pos_tolerance
            and orientation_error <= ori_tolerance_deg
        )
        if strict_target_reached and (
            not require_stable_capture
            or stable_cartesian_steps >= EXEC_FINAL_STABLE_STEPS
        ):
            return result("target_reached", True)
        if strict_target_reached and require_stable_capture:
            # Arrival alone does not establish that GPU RGB-D has caught up.
            # Keep the same command, preserve gripper state, and charge every
            # fresh stationary observation to the existing action budget.
            stalled_steps = 0
            if time.monotonic() >= deadline:
                return result("capture_settle_timeout", False)
            continue

        forward_segment_admission = _forward_signed_segment_admission(
            observed_pose,
            target_pose,
            forward_target,
            pos_tolerance=pos_tolerance,
            ori_tolerance_deg=ori_tolerance_deg,
        )
        trace[-1]["forward_segment_admission"] = forward_segment_admission
        if forward_segment_admission["ok"] and not require_stable_capture:
            return result("forward_signed_segment_progress", True)

        load_stationary_steps = load_stationary_steps + 1 if stationary else 0
        if (
            load_stationary_steps >= EXEC_LOAD_STABLE_STEPS
            and terminal_joint_error >= EXEC_LOAD_MIN_RESIDUAL_RAD
            and load_compensation_rounds < EXEC_LOAD_MAX_ROUNDS
        ):
            candidate, compensation = _next_x5_load_compensation_command(
                terminal_joint=terminal_joint,
                command_target=command_joint_target,
                observed_joint=observed_joint,
            )
            if compensation["increment_inf_rad"] > 0.0:
                command_joint_target = candidate
                load_compensation_rounds += 1
                compensation.update(
                    {
                        "round": load_compensation_rounds,
                        "action_step": action_step,
                    }
                )
                load_compensation_trace.append(compensation)
                trace[-1]["scheduled_load_compensation"] = compensation
                load_stationary_steps = 0
                stable_cartesian_steps = 0
                stalled_steps = 0
                continue

        if (
            stable_cartesian_steps >= EXEC_FINAL_STABLE_STEPS
            and load_compensation_rounds >= EXEC_LOAD_MAX_ROUNDS
        ):
            return result("stable_cartesian_target_reached", True)

        improvement = best_score - score
        if improvement > EXEC_MIN_PROGRESS_SCORE:
            best_score = score
            stalled_steps = 0
        else:
            best_score = min(best_score, score)
            stalled_steps += 1
        if sequence_after <= sequence_before:
            return result("observation_not_advanced", False)
        if stalled_steps >= EXEC_STALL_LIMIT:
            if terminal_joint_error <= EXEC_JOINT_SETTLE_TOLERANCE_RAD and (
                position_error > pos_tolerance or orientation_error > ori_tolerance_deg
            ):
                stop_reason = "eef_target_mismatch"
            elif step_max_joint_tracking_error > EXEC_JOINT_TRACKING_BLOCKED_RAD:
                stop_reason = "joint_target_blocked"
            else:
                stop_reason = "tracking_stalled"
            return result(stop_reason, False)
        if time.monotonic() >= deadline:
            return result("timeout", False)

    return result(
        "capture_settle_budget_exhausted"
        if require_stable_capture
        and terminal_joint_error <= EXEC_JOINT_SETTLE_TOLERANCE_RAD
        and position_error <= pos_tolerance
        and orientation_error <= ori_tolerance_deg
        else "max_action_steps",
        False,
    )


def _schedule_move_tracked_exec_waypoints(
    plan: Mapping[str, Any],
    waypoints: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    start_q: Sequence[float],
    move_exports: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a certified execution view without rewriting signed waypoints.

    New bounded plans use the same monotone/corridor/local-FK scheduler as the
    direct tool path and reserve a few actions for feedback recovery.  Plans
    explicitly marked ``all_signed_waypoints`` (including legacy artifacts)
    retain their original dense replay semantics.
    """

    start_state = plan.get("start_state")
    if not isinstance(start_state, Mapping):
        raise ValueError("move-tracked plan start state is missing")
    arm_qpos = start_state.get("arm_qpos")
    eef_base_wxyz = start_state.get("eef_base_wxyz")
    if not isinstance(arm_qpos, Mapping) or not isinstance(eef_base_wxyz, Mapping):
        raise ValueError("move-tracked plan start state is incomplete")
    local_state = LocalRobotState(
        arm_dof=ARM_DOF,
        arm_left_q=finite_vector(
            arm_qpos.get("left"), ARM_DOF, "signed left start joints"
        ),
        arm_right_q=finite_vector(
            arm_qpos.get("right"), ARM_DOF, "signed right start joints"
        ),
        eef_left_pose_base_wxyz=finite_vector(
            eef_base_wxyz.get("left"), 7, "signed left start EEF pose"
        ),
        eef_right_pose_base_wxyz=finite_vector(
            eef_base_wxyz.get("right"), 7, "signed right start EEF pose"
        ),
    )

    binding = plan.get("tracking_binding")
    visual_controlled_count = 0
    if isinstance(binding, Mapping):
        controlled = set(binding.get("controlled_point_names") or [])
        visual = set(binding.get("visual_tracked_point_names") or [])
        visual_controlled_count = len(controlled & visual)
    dense_replay_requested = _trajectory_uses_dense_execution(plan)
    if dense_replay_requested:
        # Keep the old transport allowance for plans created before bounded
        # execution was recorded in the trajectory schema.
        schedule_budget = max(
            int(EXEC_MOVE_TRACKED_DENSE_ACTION_BUDGET),
            int(len(waypoints)) + 64,
        )
        preferred_waypoints = None
    else:
        # Leave room below the public 50-action stream guard for transient
        # tracking/recovery actions.  The scheduler itself may use fewer than
        # this preferred target when the signed path requires it.
        schedule_budget = int(EXEC_MOVE_TRACKED_SCHEDULE_BUDGET)
        preferred_waypoints = int(EXEC_MOVE_TRACKED_PREFERRED_WAYPOINTS)
    schedule = move_exports["prepare_execution_schedule"](
        trajectory=dict(plan),
        state=local_state,
        arm=arm,
        start_q=start_q,
        max_steps=schedule_budget,
        available_time_s=60.0,
        observation_dt_s=None,
        visual_tracked_point_count=visual_controlled_count,
        preferred_max_steps=preferred_waypoints,
    )
    if not bool(schedule.get("ok")):
        raise ValueError(
            "move-tracked execution schedule cannot fit the "
            f"{schedule_budget}-step stream budget: "
            f"{schedule.get('error') or schedule.get('reason') or 'unknown error'}"
        )
    selected_indices = schedule.get("selected_indices")
    dense_schedule = bool(
        schedule.get("execution_compression_disabled")
        or str(schedule.get("execution_waypoint_policy") or "")
        == "all_signed_waypoints"
        or str(schedule.get("mode") or "") == "dense_identity"
    )
    if (
        not isinstance(selected_indices, list)
        or not selected_indices
        or (
            not dense_schedule
            and len(selected_indices) > EXEC_MOVE_TRACKED_SCHEDULE_BUDGET
        )
        or int(selected_indices[-1]) != len(waypoints) - 1
    ):
        raise ValueError("move-tracked execution schedule did not retain its endpoint")
    scheduled: list[dict[str, Any]] = []
    previous_index = -1
    for raw_index in selected_indices:
        index = int(raw_index)
        if index <= previous_index or index < 0 or index >= len(waypoints):
            raise ValueError(
                "move-tracked execution indices are not strictly increasing"
            )
        entry = dict(waypoints[index])
        planned_q = finite_vector(
            entry.get("planned_joint_target"),
            ARM_DOF,
            f"signed move-tracked waypoint {index + 1}",
        )
        scheduled_q = finite_vector(
            schedule["waypoints"][len(scheduled)],
            ARM_DOF,
            f"scheduled move-tracked waypoint {index + 1}",
        )
        if not np.allclose(planned_q, scheduled_q, atol=1.0e-10, rtol=0.0):
            raise ValueError("execution schedule rewrote a signed joint waypoint")
        entry["signed_source_index"] = index
        # A bounded schedule is allowed to skip signed knots, but each
        # retained chord must still pass through the controller's per-action
        # delta guard.  The stream loop uses this marker to split a retained
        # target over feedback actions when the measured handoff is large.
        if not dense_schedule:
            entry["bound_joint_command"] = True
        scheduled.append(entry)
        previous_index = index
    schedule["dense_signed_replay"] = bool(dense_schedule)
    schedule["execution_action_budget_policy"] = (
        "port_unbounded_dense_replay" if dense_schedule else "bounded_monotone_schedule"
    )
    return scheduled, schedule


def _uses_unverified_joint_space_fallback(plan: Mapping[str, Any]) -> bool:
    planner = plan.get("planner")
    return bool(
        isinstance(planner, Mapping)
        and str(planner.get("path_mode") or "") == "joint_space_fallback"
        and not bool(planner.get("cartesian_path_verified"))
    )


def _interpolate_clearance_pose_segment(
    start: np.ndarray,
    target: np.ndarray,
) -> list[np.ndarray]:
    translation = float(np.linalg.norm(target[:3] - start[:3]))
    orientation = quaternion_angle_deg(start[3:], target[3:])
    count = max(
        1,
        int(math.ceil(translation / EXEC_CLEARANCE_BRIDGE_CARTESIAN_STEP_M)),
        int(math.ceil(orientation / EXEC_CLEARANCE_BRIDGE_ORIENTATION_STEP_DEG)),
    )
    return [
        np.concatenate(
            [
                start[:3] + (target[:3] - start[:3]) * (index / count),
                _quaternion_slerp(start[3:], target[3:], index / count),
            ]
        )
        for index in range(1, count + 1)
    ]


def _gripper_mesh_min_z(
    pose: np.ndarray,
    vertices_eef: np.ndarray,
) -> float:
    rotation = quaternion_to_matrix(pose[3:])
    return float(np.min((vertices_eef @ rotation.T + pose[:3].reshape(1, 3))[:, 2]))


def _plan_x5_clearance_bridge(
    ctx,
    *,
    plan: Mapping[str, Any],
    arm: str,
    target_waypoint: Mapping[str, Any],
    pos_tolerance: float,
    ori_tolerance_deg: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a policy-local, collision-clear transport to one signed pose.

    A joint-space fallback only proves its endpoint; interpolating that joint
    branch can sweep the long X5 fingers through the table.  The bridge keeps
    the signed Cartesian endpoint immutable, solves a continuous alternate IK
    branch above the endpoint, and descends only after the final orientation
    is established.
    """

    current_q = finite_vector(
        ctx.world.arm_qpos_list(arm), ARM_DOF, f"current {arm} joints"
    )
    current_pose = finite_vector(
        ctx.world.eef_pose_list(arm), 7, f"current {arm} EEF pose"
    )
    target_pose = finite_vector(
        target_waypoint.get("pose"), 7, "signed fallback endpoint pose"
    )
    signed_terminal_q = finite_vector(
        target_waypoint.get("planned_joint_target"),
        ARM_DOF,
        "signed fallback endpoint joints",
    )

    clearance_z = max(
        float(current_pose[2]) + EXEC_CLEARANCE_BRIDGE_INITIAL_RISE_M,
        float(target_pose[2]) + EXEC_CLEARANCE_BRIDGE_LIFT_M,
    )
    rise_translate = current_pose.copy()
    rise_translate[:3] = [target_pose[0], target_pose[1], clearance_z]
    rotate_above_target = rise_translate.copy()
    rotate_above_target[3:] = target_pose[3:]
    phases = (
        ("rise_translate", rise_translate),
        ("rotate_above_target", rotate_above_target),
        ("terminal_descent", target_pose),
    )

    sampled: list[tuple[str, np.ndarray, np.ndarray]] = []
    previous_pose = current_pose
    previous_q = current_q
    for phase, phase_target in phases:
        for sampled_pose in _interpolate_clearance_pose_segment(
            previous_pose, phase_target
        ):
            solution = solve_calibrated_x5_ik(
                current_q,
                current_pose,
                sampled_pose,
                initial_joints=previous_q,
                max_iterations=240,
                position_tolerance_m=0.002,
                orientation_tolerance_deg=0.5,
                damping=0.012,
                orientation_weight_m_per_rad=(EEF_IK_ORIENTATION_WEIGHT_M_PER_RAD),
            )
            if not solution.converged:
                raise ValueError(
                    "policy-local X5 clearance bridge IK did not converge "
                    f"during {phase}: {solution.position_error_m:.4f}m, "
                    f"{solution.orientation_error_deg:.2f}deg"
                )
            sampled.append((phase, sampled_pose.copy(), solution.joint_target.copy()))
            previous_q = solution.joint_target.copy()
        previous_pose = phase_target

    bridge: list[dict[str, Any]] = []
    previous_q = current_q
    for phase, _sampled_pose, solved_q in sampled:
        subdivisions = max(
            1,
            int(
                math.ceil(
                    float(np.linalg.norm(solved_q - previous_q, ord=np.inf))
                    / EXEC_CLEARANCE_BRIDGE_JOINT_STEP_RAD
                )
            ),
        )
        for subdivision in range(1, subdivisions + 1):
            fraction = subdivision / subdivisions
            transport_q = previous_q + fraction * (solved_q - previous_q)
            predicted_pose = calibrated_pose_for_joints(
                current_q, current_pose, transport_q
            )
            bridge.append(
                {
                    "label": f"clearance_bridge_{len(bridge) + 1}",
                    "pose": predicted_pose.tolist(),
                    "planned_joint_target": transport_q.tolist(),
                    "pose_source": ("policy_local_calibrated_x5_clearance_bridge_fk"),
                    "signed_source_index": -1,
                    "transport_phase": phase,
                    "bound_joint_command": True,
                }
            )
        previous_q = solved_q

    if not bridge:
        raise ValueError("policy-local X5 clearance bridge is empty")
    if len(bridge) > EXEC_CLEARANCE_BRIDGE_MAX_WAYPOINTS:
        raise ValueError(
            "policy-local X5 clearance bridge needs "
            f"{len(bridge)} waypoints, above the "
            f"{EXEC_CLEARANCE_BRIDGE_MAX_WAYPOINTS}-waypoint limit"
        )

    terminal_q = finite_vector(
        bridge[-1]["planned_joint_target"],
        ARM_DOF,
        "clearance bridge terminal joints",
    )
    terminal_prediction = calibrated_pose_for_joints(
        current_q, current_pose, terminal_q
    )
    terminal_position_error = float(
        np.linalg.norm(terminal_prediction[:3] - target_pose[:3])
    )
    terminal_orientation_error = quaternion_angle_deg(
        terminal_prediction[3:], target_pose[3:]
    )
    if terminal_position_error > min(
        float(pos_tolerance), 0.003
    ) or terminal_orientation_error > min(float(ori_tolerance_deg), 1.0):
        raise ValueError(
            "policy-local X5 clearance bridge changed the signed Cartesian "
            f"endpoint: {terminal_position_error:.4f}m, "
            f"{terminal_orientation_error:.2f}deg"
        )
    bridge[-1]["pose"] = target_pose.tolist()
    bridge[-1]["pose_source"] = "immutable_signed_cartesian_endpoint"
    bridge[-1]["signed_source_index"] = int(
        target_waypoint.get("signed_source_index", -1)
    )

    vertices = np.asarray(x5_gripper_geometry()["visual_vertices"], dtype=np.float64)
    current_mesh_min_z = _gripper_mesh_min_z(current_pose, vertices)
    target_mesh_min_z = _gripper_mesh_min_z(target_pose, vertices)
    minimum_by_phase: dict[str, float] = {}
    for entry in bridge:
        predicted_pose = calibrated_pose_for_joints(
            current_q,
            current_pose,
            entry["planned_joint_target"],
        )
        phase = str(entry["transport_phase"])
        mesh_min_z = _gripper_mesh_min_z(predicted_pose, vertices)
        entry["predicted_gripper_mesh_min_z_m"] = mesh_min_z
        minimum_by_phase[phase] = min(minimum_by_phase.get(phase, math.inf), mesh_min_z)
        if phase == "rotate_above_target":
            minimum_allowed = (
                target_mesh_min_z + EXEC_CLEARANCE_BRIDGE_ROTATION_CLEARANCE_M
            )
        elif phase == "terminal_descent":
            minimum_allowed = target_mesh_min_z - EXEC_CLEARANCE_BRIDGE_Z_TOLERANCE_M
        else:
            minimum_allowed = min(current_mesh_min_z, target_mesh_min_z) - (
                EXEC_CLEARANCE_BRIDGE_Z_TOLERANCE_M
            )
        if mesh_min_z < minimum_allowed:
            raise ValueError(
                "policy-local X5 clearance bridge failed its official-mesh "
                f"vertical clearance audit during {phase}: "
                f"{mesh_min_z:.4f}m < {minimum_allowed:.4f}m"
            )

    alternate_branch_delta = float(
        np.linalg.norm(terminal_q - signed_terminal_q, ord=np.inf)
    )
    report = {
        "ok": True,
        "reason": "unverified_joint_space_fallback_requires_clearance_transport",
        "policy": ("rise_and_translate_then_rotate_above_target_then_descend"),
        "waypoint_count": len(bridge),
        "clearance_z_m": clearance_z,
        "cartesian_sample_step_m": EXEC_CLEARANCE_BRIDGE_CARTESIAN_STEP_M,
        "orientation_sample_step_deg": (EXEC_CLEARANCE_BRIDGE_ORIENTATION_STEP_DEG),
        "max_joint_step_rad": EXEC_CLEARANCE_BRIDGE_JOINT_STEP_RAD,
        "current_gripper_mesh_min_z_m": current_mesh_min_z,
        "signed_endpoint_gripper_mesh_min_z_m": target_mesh_min_z,
        "minimum_gripper_mesh_z_by_phase_m": minimum_by_phase,
        "official_gripper_mesh_audited": True,
        "signed_endpoint_pose": target_pose.tolist(),
        "signed_endpoint_joint_target": signed_terminal_q.tolist(),
        "transport_endpoint_joint_target": terminal_q.tolist(),
        "alternate_ik_branch": bool(
            alternate_branch_delta > EXEC_JOINT_SETTLE_TOLERANCE_RAD
        ),
        "alternate_ik_branch_delta_inf_rad": alternate_branch_delta,
        "terminal_position_equivalence_error_m": terminal_position_error,
        "terminal_orientation_equivalence_error_deg": (terminal_orientation_error),
        "plan_artifact_rewritten": False,
        "signed_cartesian_endpoint_rewritten": False,
    }
    return bridge, report


def _drive_move_tracked_exec_stream(
    ctx,
    *,
    plan: Mapping[str, Any],
    waypoints: Sequence[Mapping[str, Any]],
    arm: str,
    pos_tolerance: float,
    ori_tolerance_deg: float,
    max_steps: int,
    dense_replay: bool = False,
):
    """Execute a signed move-tracked path with continuous-stream semantics.

    ``dense_replay`` is set only after the schedule has positively reported
    ``all_signed_waypoints``.  In that mode this function honors the caller's
    large port budget instead of applying the legacy 50-action cap; all other
    callers remain bounded exactly as before.
    """

    tracking = plan.get("tracking")
    if not isinstance(tracking, Mapping):
        raise ValueError("move-tracked plan tracking metadata is missing")
    waypoint_tolerance = _finite_number(
        tracking.get("waypoint_tolerance_rad"),
        "move-tracked waypoint tolerance",
    )
    max_tracking_error = _finite_number(
        tracking.get("max_tracking_error_rad"),
        "move-tracked tracking-error limit",
    )
    recovery_error = _finite_number(
        tracking.get("stream_recovery_error_rad"),
        "move-tracked recovery-error limit",
    )
    raw_recovery_steps = tracking.get("stream_max_recovery_steps")
    if (
        isinstance(raw_recovery_steps, bool)
        or not isinstance(raw_recovery_steps, (int, float))
        or int(raw_recovery_steps) != raw_recovery_steps
        or int(raw_recovery_steps) < 0
    ):
        raise ValueError("move-tracked recovery-step limit is invalid")
    recovery_steps = int(raw_recovery_steps)
    raw_stable_steps = tracking.get("stable_steps", 3)
    if (
        isinstance(raw_stable_steps, bool)
        or not isinstance(raw_stable_steps, (int, float))
        or int(raw_stable_steps) != raw_stable_steps
        or not 1 <= int(raw_stable_steps) <= 5
    ):
        raise ValueError("move-tracked stable-step count is invalid")
    stable_steps_required = int(raw_stable_steps)
    if not (0.0 < waypoint_tolerance <= max_tracking_error <= recovery_error <= 1.0):
        raise ValueError("move-tracked streaming tolerances are inconsistent")

    expected_episode_id = str(plan.get("episode_id") or "")
    requested_action_limit = int(max_steps)
    if requested_action_limit < 1:
        raise ValueError("move-tracked execution action budget must be positive")
    action_limit = (
        requested_action_limit
        if dense_replay
        else min(requested_action_limit, EXEC_TOTAL_ACTION_BUDGET)
    )
    action_steps = 0
    max_observed_tracking_error = 0.0
    completed_waypoints = 0
    executed: list[dict[str, Any]] = []
    failure_reason: str | None = None
    entry_handoff_action_steps = 0
    entry_handoff_initial_error_rad: float | None = None
    entry_handoff_max_command_error_rad = 0.0
    load_compensation_rounds = 0
    load_compensation_total = np.zeros(ARM_DOF, dtype=np.float64)
    load_compensation_trace: list[dict[str, Any]] = []

    for execution_index, waypoint in enumerate(waypoints):
        target_pose = finite_vector(
            waypoint.get("pose"), 7, "move-tracked EEF waypoint"
        )
        target_joint = finite_vector(
            waypoint.get("planned_joint_target"),
            ARM_DOF,
            "signed move-tracked joint waypoint",
        )
        source_index = int(waypoint.get("signed_source_index", execution_index))
        is_final = execution_index == len(waypoints) - 1
        recovery_attempt = 0
        stable_steps = 0
        cartesian_stable_steps = 0
        load_stationary_steps = 0
        stalled_steps = 0
        trace: list[dict[str, Any]] = []
        command_joint_target = target_joint.copy()
        observed_joint = np.asarray(ctx.world.arm_qpos_list(arm), dtype=np.float64)
        observed_pose = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
        previous_tracking_error = float(
            np.linalg.norm(target_joint - observed_joint, ord=np.inf)
        )
        if execution_index == 0:
            entry_handoff_initial_error_rad = previous_tracking_error
        best_cartesian_score = max(
            float(np.linalg.norm(observed_pose[:3] - target_pose[:3])) / pos_tolerance,
            quaternion_angle_deg(observed_pose[3:], target_pose[3:])
            / ori_tolerance_deg,
        )
        converged = False
        stop_reason = "not_started"

        while action_steps < action_limit:
            ctx.raise_if_cancelled("move-tracked plan continuous execution")
            if str(ctx.world.episode_id()) != expected_episode_id:
                failure_reason = "episode_changed"
                stop_reason = failure_reason
                break
            sequence_before = _observation_sequence(ctx)
            observed_joint_before = observed_joint.copy()
            observed_pose_before = observed_pose.copy()
            full_delta = command_joint_target - observed_joint
            largest_delta = float(np.linalg.norm(full_delta, ord=np.inf))
            command_scale = 1.0
            if (
                execution_index == 0 or bool(waypoint.get("bound_joint_command"))
            ) and largest_delta > EEF_MAX_JOINT_DELTA_PER_ACTION:
                command_scale = EEF_MAX_JOINT_DELTA_PER_ACTION / largest_delta
            commanded_joint = observed_joint + command_scale * full_delta
            yield ctx.world.make_joint_action(
                **{f"{arm}_arm": commanded_joint.tolist()}
            )
            action_steps += 1
            sequence_after = _observation_sequence(ctx)
            observed_joint = np.asarray(ctx.world.arm_qpos_list(arm), dtype=np.float64)
            observed_pose = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
            tracking_error = float(
                np.linalg.norm(target_joint - observed_joint, ord=np.inf)
            )
            command_tracking_error = float(
                np.linalg.norm(commanded_joint - observed_joint, ord=np.inf)
            )
            entry_handoff_max_command_error_rad = max(
                entry_handoff_max_command_error_rad,
                command_tracking_error,
            )
            position_error = float(np.linalg.norm(observed_pose[:3] - target_pose[:3]))
            orientation_error = quaternion_angle_deg(observed_pose[3:], target_pose[3:])
            cartesian_score = max(
                position_error / pos_tolerance,
                orientation_error / ori_tolerance_deg,
            )
            observed_joint_step = float(
                np.linalg.norm(observed_joint - observed_joint_before, ord=np.inf)
            )
            observed_translation_step = float(
                np.linalg.norm(observed_pose[:3] - observed_pose_before[:3])
            )
            observed_orientation_step = quaternion_angle_deg(
                observed_pose_before[3:], observed_pose[3:]
            )
            max_observed_tracking_error = max(
                max_observed_tracking_error, tracking_error
            )
            trace.append(
                {
                    "action_step": action_steps,
                    "observation_sequence_before": sequence_before,
                    "observation_sequence_after": sequence_after,
                    "commanded_joint_position": commanded_joint.tolist(),
                    "signed_waypoint_joint_position": target_joint.tolist(),
                    "command_target_joint_position": (command_joint_target.tolist()),
                    "command_scale": command_scale,
                    "largest_full_joint_delta_rad": largest_delta,
                    "command_tracking_error_inf_rad": command_tracking_error,
                    "observed_joint_position_before": (observed_joint_before.tolist()),
                    "observed_joint_position": observed_joint.tolist(),
                    "observed_joint_step_inf_rad": observed_joint_step,
                    "tracking_error_inf_rad": tracking_error,
                    "position_error_m": position_error,
                    "orientation_error_deg": orientation_error,
                    "observed_translation_step_m": observed_translation_step,
                    "observed_orientation_step_deg": observed_orientation_step,
                    "recovery_attempt": recovery_attempt,
                    "final_stable_steps": stable_steps,
                    "load_stationary_steps": load_stationary_steps,
                    "load_compensation_round": load_compensation_rounds,
                }
            )
            if sequence_after <= sequence_before:
                failure_reason = "observation_not_advanced"
                stop_reason = failure_reason
                break

            # A same-episode plan remains valid after other arm or gripper
            # actions.  Re-enter its first signed waypoint from the measured
            # joints with bounded commands before applying the normal stream
            # corridor.  This changes only the handoff, never a signed target.
            if execution_index == 0 and tracking_error > max_tracking_error:
                entry_handoff_action_steps += 1
                trace[-1]["entry_handoff"] = True
                progressing = tracking_error < previous_tracking_error - 1.0e-5
                if command_tracking_error > recovery_error:
                    failure_reason = "entry_handoff_command_tracking_error"
                    stop_reason = failure_reason
                    break
                if progressing:
                    previous_tracking_error = tracking_error
                    stalled_steps = 0
                    continue
                stalled_steps += 1
                if stalled_steps >= EXEC_STALL_LIMIT:
                    failure_reason = "entry_handoff_stalled"
                    stop_reason = failure_reason
                    break
                continue

            if is_final:
                cartesian_ok = bool(
                    position_error <= pos_tolerance
                    and orientation_error <= ori_tolerance_deg
                )
                stationary = bool(
                    observed_joint_step <= EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD
                    and observed_translation_step
                    <= EXEC_FINAL_TRANSLATION_STEP_TOLERANCE_M
                    and observed_orientation_step
                    <= EXEC_FINAL_ORIENTATION_STEP_TOLERANCE_DEG
                )
                load_stationary_steps = load_stationary_steps + 1 if stationary else 0
                trace[-1]["final_stationary"] = stationary
                strict_joint_ok = bool(tracking_error <= waypoint_tolerance)
                stable_steps = (
                    stable_steps + 1
                    if cartesian_ok and strict_joint_ok and stationary
                    else 0
                )
                cartesian_stable_steps = (
                    cartesian_stable_steps + 1 if cartesian_ok and stationary else 0
                )
                trace[-1]["final_stable_steps"] = stable_steps
                trace[-1]["final_cartesian_stable_steps"] = cartesian_stable_steps
                trace[-1]["load_stationary_steps"] = load_stationary_steps
                if stable_steps >= stable_steps_required:
                    converged = True
                    stop_reason = "stable_signed_cartesian_target_reached"
                    break

                if (
                    load_stationary_steps >= EXEC_LOAD_STABLE_STEPS
                    and tracking_error >= EXEC_LOAD_MIN_RESIDUAL_RAD
                    and load_compensation_rounds < EXEC_LOAD_MAX_ROUNDS
                ):
                    candidate, compensation = _next_x5_load_compensation_command(
                        terminal_joint=target_joint,
                        command_target=command_joint_target,
                        observed_joint=observed_joint,
                    )
                    if compensation["increment_inf_rad"] > 0.0:
                        command_joint_target = candidate
                        load_compensation_rounds += 1
                        load_compensation_total = command_joint_target - target_joint
                        compensation.update(
                            {
                                "round": load_compensation_rounds,
                                "action_step": action_steps,
                                "signed_source_index": source_index,
                            }
                        )
                        load_compensation_trace.append(compensation)
                        trace[-1]["scheduled_load_compensation"] = compensation
                        load_stationary_steps = 0
                        stable_steps = 0
                        cartesian_stable_steps = 0
                        stalled_steps = 0
                        continue

                if (
                    cartesian_stable_steps >= stable_steps_required
                    and load_compensation_rounds >= EXEC_LOAD_MAX_ROUNDS
                ):
                    converged = True
                    stop_reason = "stable_cartesian_target_reached"
                    break
                improvement = best_cartesian_score - cartesian_score
                if improvement > EXEC_MIN_PROGRESS_SCORE:
                    best_cartesian_score = cartesian_score
                    stalled_steps = 0
                elif not cartesian_ok:
                    best_cartesian_score = min(best_cartesian_score, cartesian_score)
                    stalled_steps += 1
                if tracking_error > recovery_error:
                    failure_reason = "endpoint_tracking_error"
                    stop_reason = failure_reason
                    break
                if stalled_steps >= EXEC_STALL_LIMIT:
                    failure_reason = "endpoint_cartesian_stalled"
                    stop_reason = failure_reason
                    break
                continue

            if tracking_error <= max_tracking_error:
                converged = True
                stop_reason = "stream_tracking_corridor_validated"
                break
            tracking_progress = previous_tracking_error - tracking_error
            cartesian_progress = best_cartesian_score - cartesian_score
            physically_moving = bool(
                observed_joint_step >= EXEC_MIN_OBSERVED_JOINT_STEP_RAD
                or observed_translation_step >= EXEC_MIN_OBSERVED_TRANSLATION_STEP_M
                or observed_orientation_step >= EXEC_MIN_OBSERVED_ORIENTATION_STEP_DEG
            )
            trace[-1]["stream_recovery_tracking_progress_rad"] = tracking_progress
            trace[-1]["stream_recovery_cartesian_progress_score"] = cartesian_progress
            trace[-1]["stream_recovery_physically_moving"] = physically_moving
            if tracking_error <= recovery_error and recovery_attempt < recovery_steps:
                recovery_attempt += 1
                trace[-1]["stream_recovery_scheduled"] = True
                trace[-1]["stream_recovery_attempt_next"] = recovery_attempt
                if (
                    tracking_progress > 1.0e-5
                    or cartesian_progress > EXEC_MIN_PROGRESS_SCORE
                    or physically_moving
                ):
                    stalled_steps = 0
                else:
                    stalled_steps += 1
                previous_tracking_error = tracking_error
                best_cartesian_score = min(best_cartesian_score, cartesian_score)
                continue
            failure_reason = (
                "waypoint_tracking_error"
                if tracking_error > recovery_error
                else "waypoint_tracking_recovery_exhausted"
            )
            stop_reason = failure_reason
            break

        if not converged and failure_reason is None:
            failure_reason = "action_budget_exhausted"
            stop_reason = failure_reason
        position_error = float(np.linalg.norm(observed_pose[:3] - target_pose[:3]))
        orientation_error = quaternion_angle_deg(observed_pose[3:], target_pose[3:])
        terminal_joint_error = float(
            np.linalg.norm(target_joint - observed_joint, ord=np.inf)
        )
        executed.append(
            {
                "index": source_index,
                "execution_index": execution_index,
                "signed_source_index": source_index,
                "label": str(waypoint.get("label", f"waypoint_{source_index}")),
                "target_pose": target_pose.tolist(),
                "target_pose_source": str(
                    waypoint.get("pose_source", "signed_cartesian_waypoint")
                ),
                "ideal_target_pose": waypoint.get("ideal_pose"),
                "observed_pose": observed_pose.tolist(),
                "position_error_m": position_error,
                "control_tracking_position_error_m": position_error,
                "orientation_error_deg": orientation_error,
                "control_tracking_orientation_error_deg": orientation_error,
                "converged": converged,
                "action_steps": len(trace),
                "stop_reason": stop_reason,
                "controller": "immutable_signed_joint_trajectory_stream",
                "joint_target_source": "signed_planned_joint_target",
                "terminal_joint_target": target_joint.tolist(),
                "observed_joint_position": observed_joint.tolist(),
                "terminal_joint_error_rad": terminal_joint_error,
                "joint_settle_tolerance_rad": waypoint_tolerance,
                "joint_tracking_blocked_threshold_rad": recovery_error,
                "max_joint_tracking_error_rad": max(
                    (float(item["tracking_error_inf_rad"]) for item in trace),
                    default=terminal_joint_error,
                ),
                "predicted_terminal_pose": target_pose.tolist(),
                "entry_ik": None,
                "tracking_trace": trace,
                "final_stable_steps": stable_steps if is_final else None,
                "stationary_capture_confirmation": _stationary_capture_confirmation(
                    is_final,
                    cartesian_stable_steps,
                    stable_steps_required,
                ),
                "final_stable_steps_required": (
                    stable_steps_required if is_final else None
                ),
                "final_position_step_tolerance_rad": (
                    EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD if is_final else None
                ),
                "terminal_acceptance": (
                    "strict_joint_and_cartesian_after_bounded_load_compensation"
                    if is_final
                    and converged
                    and stop_reason == "stable_signed_cartesian_target_reached"
                    and load_compensation_rounds
                    else (
                        "strict_joint_and_cartesian"
                        if is_final
                        and converged
                        and stop_reason == "stable_signed_cartesian_target_reached"
                        else (
                            "stable_cartesian_after_bounded_load_compensation"
                            if is_final and converged
                            else None
                        )
                    )
                ),
                "load_compensation_rounds": (
                    load_compensation_rounds if is_final else 0
                ),
            }
        )
        if not converged:
            break
        completed_waypoints += 1

    return {
        "ok": bool(failure_reason is None and completed_waypoints == len(waypoints)),
        "reason": failure_reason or "stable_cartesian_target_reached",
        "segments": executed,
        "action_steps": action_steps,
        "action_limit": action_limit,
        "dense_replay": bool(dense_replay),
        "completed_waypoints": completed_waypoints,
        "max_observed_tracking_error_rad": max_observed_tracking_error,
        "entry_handoff_action_steps": entry_handoff_action_steps,
        "entry_handoff_initial_error_rad": entry_handoff_initial_error_rad,
        "entry_handoff_max_command_tracking_error_rad": (
            entry_handoff_max_command_error_rad
        ),
        "entry_handoff_max_command_delta_rad": (EEF_MAX_JOINT_DELTA_PER_ACTION),
        "stream_recovery_action_steps": sum(
            max(0, int(segment.get("action_steps", 0)) - 1)
            for segment in executed
            if segment.get("final_stable_steps") is None
        ),
        "load_compensation": {
            "enabled": True,
            "rounds": load_compensation_rounds,
            "max_rounds": EXEC_LOAD_MAX_ROUNDS,
            "stable_steps_before_round": EXEC_LOAD_STABLE_STEPS,
            "minimum_residual_rad": EXEC_LOAD_MIN_RESIDUAL_RAD,
            "max_round_increment_rad": EXEC_LOAD_MAX_ROUND_RAD,
            "max_total_increment_rad": EXEC_LOAD_MAX_TOTAL_RAD,
            "total_increment_rad": load_compensation_total.tolist(),
            "trace": load_compensation_trace,
            "immutable_signed_target_changed": False,
        },
        "waypoint_completion_semantics": (
            "one_action_or_bounded_feedback_recovery_per_intermediate_"
            "waypoint_with_tracking_corridor_and_bounded_load_compensated_"
            "stable_signed_cartesian_endpoint"
        ),
    }


def exec_plan_pose(ctx, **kwargs):
    dense_move_replay = False
    try:
        move_exports: dict[str, Any] | None = None
        clearance_bridge_report: dict[str, Any] | None = None
        clearance_bridge_preflight_error: str | None = None
        session_id = str(kwargs.get("session_id", "") or "").strip()
        plan_id = str(kwargs.get("plan_id", "") or "").strip()
        if not plan_id:
            raise ValueError("plan_id is required")
        plan = ctx.plans.load_any(plan_id, session_id or None)
        if str(plan.get("episode_id")) != ctx.world.episode_id():
            raise ValueError("plan belongs to a different evaluator episode")
        planned_arm = str(plan["active_arm"])
        requested_arm = str(kwargs.get("arm", "") or "").strip().lower()
        if requested_arm and requested_arm != planned_arm:
            raise ValueError("arm override does not match the immutable plan")
        move_tracked_plan = str(plan.get("source_tool") or "") == "move_tracked_point"
        if move_tracked_plan:
            move_exports = _move_tracked_exports()
            move_exports["validate_loaded_plan"](
                plan, session_id=str(plan["session_id"])
            )
        waypoints = list(
            plan.get("eef_waypoints" if move_tracked_plan else "waypoints", [])
        )
        if not waypoints:
            raise ValueError("plan has no EEF waypoints")
        stop_after_safe = bool(kwargs.get("stop_after_safe", False))
        if stop_after_safe:
            waypoints = waypoints[:1]
        parameters = dict(plan.get("parameters", {}))
        position_tolerance = _finite_number(
            parameters.get("pos_tol", 0.03), "signed plan position tolerance"
        )
        orientation_tolerance = _finite_number(
            parameters.get("ori_tol_deg", 20.0),
            "signed plan orientation tolerance",
        )
        if not 0.001 <= position_tolerance <= 0.10:
            raise ValueError("signed plan position tolerance must be in 0.001..0.10m")
        if not 0.1 <= orientation_tolerance <= 45.0:
            raise ValueError(
                "signed plan orientation tolerance must be in 0.1..45 degrees"
            )
        compat_mode = str(parameters.get("compat_plan_mode", "") or "")
        requested_gripper_transition = str(
            parameters.get("compat_gripper_transition", "preserve") or "preserve"
        )
        if compat_mode and compat_mode not in _COMPAT_PLAN_MODES:
            raise ValueError("plan contains an invalid compatibility mode")
        if requested_gripper_transition not in (
            "preserve",
            "close_before_motion",
            "close_after_waypoint",
        ):
            raise ValueError("plan contains an invalid gripper transition")
        start_motion_epoch = int(plan.get("motion_epoch", -1))
        current_motion_epoch = int(ctx.world.motion_epoch())
        if start_motion_epoch > current_motion_epoch:
            raise ValueError("plan motion epoch is newer than the evaluator state")
        stale_steps = current_motion_epoch - start_motion_epoch
        current_start_q = finite_vector(
            ctx.world.arm_qpos_list(planned_arm),
            ARM_DOF,
            f"current {planned_arm} arm joints",
        )
        planned_start_q: np.ndarray | None = None
        start_state = plan.get("start_state")
        if isinstance(start_state, Mapping):
            start_arm_qpos = start_state.get("arm_qpos")
            if isinstance(start_arm_qpos, Mapping):
                planned_start_q = finite_vector(
                    start_arm_qpos.get(planned_arm),
                    ARM_DOF,
                    f"signed {planned_arm} start joints",
                )
        if planned_start_q is None:
            start_arm_qpos = plan.get("start_arm_qpos")
            if isinstance(start_arm_qpos, Mapping):
                planned_start_q = finite_vector(
                    start_arm_qpos.get(planned_arm),
                    ARM_DOF,
                    f"signed {planned_arm} start joints",
                )
        start_joint_offset_rad = (
            None
            if planned_start_q is None
            else float(np.linalg.norm(current_start_q - planned_start_q, ord=np.inf))
        )
        execution_rebased = bool(
            stale_steps > 0
            or (
                start_joint_offset_rad is not None
                and start_joint_offset_rad > EXEC_JOINT_SETTLE_TOLERANCE_RAD
            )
        )
        signed_waypoint_count = len(waypoints)
        execution_schedule: dict[str, Any] | None = None
        if move_tracked_plan and not stop_after_safe:
            if move_exports is None:
                raise RuntimeError("move-tracked implementation was not bound")
            if _uses_unverified_joint_space_fallback(plan):
                target_waypoint = dict(waypoints[-1])
                target_waypoint["signed_source_index"] = len(waypoints) - 1
                try:
                    waypoints, clearance_bridge_report = _plan_x5_clearance_bridge(
                        ctx,
                        plan=plan,
                        arm=planned_arm,
                        target_waypoint=target_waypoint,
                        pos_tolerance=position_tolerance,
                        ori_tolerance_deg=orientation_tolerance,
                    )
                    execution_schedule = dict(clearance_bridge_report)
                    execution_schedule["mode"] = "x5_clearance_bridge"
                    execution_schedule["selected_indices"] = [
                        int(entry["signed_source_index"]) for entry in waypoints
                    ]
                    execution_schedule["dropped_waypoint_count"] = (
                        signed_waypoint_count - 1
                    )
                    execution_schedule["mandatory_indices_retained"] = True
                except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                    clearance_bridge_preflight_error = str(exc)
            if clearance_bridge_report is None:
                waypoints, execution_schedule = _schedule_move_tracked_exec_waypoints(
                    plan,
                    waypoints,
                    arm=planned_arm,
                    start_q=ctx.world.arm_qpos_list(planned_arm),
                    move_exports=move_exports,
                )
                dense_move_replay = bool(
                    execution_schedule.get("dense_signed_replay")
                    or execution_schedule.get("execution_compression_disabled")
                    or str(execution_schedule.get("execution_waypoint_policy") or "")
                    == "all_signed_waypoints"
                )
    except (
        FileNotFoundError,
        KeyError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        _set_result(ctx, "exec_plan_pose", {"ok": False, "error": str(exc)})
        yield ctx.world.hold_action()
        return

    executed = []
    execution_ok = True
    total_action_steps = 0
    action_budget_exhausted = False
    execution_failure_reason: str | None = None
    gripper_before = ctx.world.gripper_qpos_list(planned_arm)
    effective_action_budget_steps = (
        EXEC_MOVE_TRACKED_DENSE_ACTION_BUDGET
        if move_tracked_plan and not stop_after_safe and dense_move_replay
        else EXEC_TOTAL_ACTION_BUDGET
    )
    if move_tracked_plan and not stop_after_safe:
        streaming = yield from _drive_move_tracked_exec_stream(
            ctx,
            plan=plan,
            waypoints=waypoints,
            arm=planned_arm,
            pos_tolerance=position_tolerance,
            ori_tolerance_deg=orientation_tolerance,
            max_steps=effective_action_budget_steps,
            dense_replay=dense_move_replay,
        )
        executed = list(streaming["segments"])
        if clearance_bridge_report is not None:
            for segment, waypoint in zip(executed, waypoints):
                segment["controller"] = "policy_local_x5_clearance_bridge_joint_stream"
                segment["transport_phase"] = waypoint.get("transport_phase")
                segment["predicted_gripper_mesh_min_z_m"] = waypoint.get(
                    "predicted_gripper_mesh_min_z_m"
                )
                segment["signed_endpoint_joint_target"] = clearance_bridge_report[
                    "signed_endpoint_joint_target"
                ]
                segment["transport_endpoint_joint_target"] = clearance_bridge_report[
                    "transport_endpoint_joint_target"
                ]
            streaming["waypoint_completion_semantics"] = (
                "bounded_feedback_tracking_of_policy_local_clearance_"
                "transport_with_strict_signed_cartesian_endpoint"
            )
        total_action_steps = int(streaming["action_steps"])
        execution_ok = bool(streaming["ok"])
        execution_failure_reason = None if execution_ok else str(streaming["reason"])
        action_budget_exhausted = bool(
            total_action_steps >= effective_action_budget_steps
        )
    else:
        for execution_index, waypoint in enumerate(waypoints if execution_ok else []):
            ctx.raise_if_cancelled("plan execution")
            remaining_action_steps = EXEC_TOTAL_ACTION_BUDGET - total_action_steps
            if remaining_action_steps <= 0:
                execution_ok = False
                action_budget_exhausted = True
                break
            target = finite_vector(waypoint["pose"], 7, "plan EEF waypoint")
            tracking = yield from _drive_exec_waypoint(
                ctx,
                arm=planned_arm,
                target=target,
                forward_target=(
                    None
                    if (not execution_rebased or execution_index + 1 >= len(waypoints))
                    else waypoints[execution_index + 1]["pose"]
                ),
                planned_joint_target=waypoint.get("planned_joint_target"),
                pos_tolerance=position_tolerance,
                ori_tolerance_deg=orientation_tolerance,
                max_steps=min(EXEC_MAX_ACTION_STEPS, remaining_action_steps),
                timeout_s=60.0,
                require_stable_capture=(execution_index + 1 == len(waypoints)),
            )
            observed = np.asarray(tracking["observed_pose"], dtype=np.float64)
            total_action_steps += int(tracking["action_steps"])
            executed.append(
                {
                    "index": int(waypoint.get("signed_source_index", execution_index)),
                    "execution_index": execution_index,
                    "signed_source_index": waypoint.get("signed_source_index"),
                    "label": str(waypoint.get("label", f"waypoint_{execution_index}")),
                    "target_pose": target.tolist(),
                    "target_pose_source": str(
                        waypoint.get("pose_source", "signed_cartesian_waypoint")
                    ),
                    "ideal_target_pose": waypoint.get("ideal_pose"),
                    "planning_ik_position_residual_m": waypoint.get(
                        "planning_ik_position_residual_m"
                    ),
                    "planning_ik_orientation_residual_deg": waypoint.get(
                        "planning_ik_orientation_residual_deg"
                    ),
                    "observed_pose": observed.tolist(),
                    "position_error_m": tracking["position_error_m"],
                    "control_tracking_position_error_m": tracking["position_error_m"],
                    "orientation_error_deg": tracking["orientation_error_deg"],
                    "control_tracking_orientation_error_deg": tracking[
                        "orientation_error_deg"
                    ],
                    "converged": tracking["converged"],
                    "action_steps": tracking["action_steps"],
                    "stop_reason": tracking["stop_reason"],
                    "controller": tracking["controller"],
                    "joint_target_source": tracking["joint_target_source"],
                    "terminal_joint_target": tracking["terminal_joint_target"],
                    "observed_joint_position": tracking["observed_joint_position"],
                    "terminal_joint_error_rad": tracking["terminal_joint_error_rad"],
                    "joint_settle_tolerance_rad": tracking[
                        "joint_settle_tolerance_rad"
                    ],
                    "max_joint_tracking_error_rad": tracking[
                        "max_joint_tracking_error_rad"
                    ],
                    "joint_tracking_blocked_threshold_rad": tracking[
                        "joint_tracking_blocked_threshold_rad"
                    ],
                    "predicted_terminal_pose": tracking["predicted_terminal_pose"],
                    "entry_ik": tracking["entry_ik"],
                    "final_stable_steps": tracking["final_stable_steps"],
                    "final_stable_steps_required": tracking[
                        "final_stable_steps_required"
                    ],
                    "final_position_step_tolerance_rad": tracking[
                        "final_position_step_tolerance_rad"
                    ],
                    "terminal_acceptance": tracking["terminal_acceptance"],
                    "forward_segment_admission": tracking["forward_segment_admission"],
                    "load_compensation": tracking["load_compensation"],
                    "stationary_capture_confirmation": tracking[
                        "stationary_capture_confirmation"
                    ],
                    "tracking_trace": tracking["trace"],
                }
            )
            if not tracking["converged"]:
                execution_ok = False
                execution_failure_reason = str(tracking["stop_reason"])
                action_budget_exhausted = bool(
                    total_action_steps >= effective_action_budget_steps
                )
                break
    final = executed[-1] if executed else None
    gripper_after = ctx.world.gripper_qpos_list(planned_arm)
    ok = bool(
        execution_ok
        and final is not None
        and len(executed) == len(waypoints)
        and final["converged"]
    )
    _set_result(
        ctx,
        "exec_plan_pose",
        {
            "ok": ok,
            "error": (
                None
                if ok
                else (
                    "plan execution reached the hard "
                    f"{effective_action_budget_steps}-action evaluator budget"
                    if action_budget_exhausted
                    else (
                        "plan execution did not converge to every target"
                        + (
                            f": {execution_failure_reason}"
                            if execution_failure_reason
                            else ""
                        )
                    )
                )
            ),
            "session_id": str(plan["session_id"]),
            "plan_id": plan_id,
            "trajectory_digest": plan["integrity"]["digest"],
            "arm": planned_arm,
            "segments_executed": executed,
            "stationary_capture_confirmation": (
                final.get("stationary_capture_confirmation") if final else None
            ),
            "action_steps": total_action_steps,
            "action_budget_steps": effective_action_budget_steps,
            "action_budget_remaining_steps": max(
                0, effective_action_budget_steps - total_action_steps
            ),
            "action_budget_exhausted": action_budget_exhausted,
            "action_budget_policy": (
                "port_unbounded_dense_replay"
                if dense_move_replay
                else "legacy_bounded_50_step_plan_budget"
            ),
            "dense_waypoint_replay": bool(dense_move_replay),
            "signed_waypoint_count": signed_waypoint_count,
            "execution_waypoint_count": len(waypoints),
            "execution_schedule": (
                None
                if execution_schedule is None
                else {
                    key: json_ready(value)
                    for key, value in execution_schedule.items()
                    if key != "waypoints"
                }
            ),
            "execution_controller": (
                "policy_local_x5_clearance_bridge_joint_stream"
                if clearance_bridge_report is not None
                else "immutable_signed_joint_trajectory_stream"
                if move_tracked_plan and not stop_after_safe
                else "immutable_joint_target_replay"
            ),
            "waypoint_completion_semantics": (
                None
                if not (move_tracked_plan and not stop_after_safe)
                else streaming["waypoint_completion_semantics"]
            ),
            "stream_recovery_action_steps": (
                int(streaming["stream_recovery_action_steps"])
                if move_tracked_plan and not stop_after_safe
                else 0
            ),
            "execution_start_rebase": {
                "applied": execution_rebased,
                "policy": (
                    "same_episode_measured_state_to_policy_local_x5_"
                    "clearance_transport_to_immutable_signed_cartesian_endpoint"
                    if clearance_bridge_report is not None
                    else "same_episode_measured_arm_qpos_to_immutable_signed_plan"
                ),
                "plan_motion_epoch": start_motion_epoch,
                "execution_motion_epoch": current_motion_epoch,
                "intervening_evaluator_action_steps": stale_steps,
                "planned_start_joint_position": (
                    None if planned_start_q is None else planned_start_q.tolist()
                ),
                "measured_start_joint_position": current_start_q.tolist(),
                "start_joint_offset_inf_rad": start_joint_offset_rad,
                "signed_targets_rewritten": False,
                "signed_joint_trajectory_replayed": bool(
                    clearance_bridge_report is None
                ),
                "entry_handoff_action_steps": (
                    int(streaming["entry_handoff_action_steps"])
                    if move_tracked_plan and not stop_after_safe
                    else int(
                        sum(int(segment["action_steps"]) for segment in executed[:1])
                    )
                ),
            },
            "execution_failure_reason": execution_failure_reason,
            "execution_target_semantics": str(
                "immutable_signed_cartesian_endpoint_via_equivalent_x5_ik_branch"
                if clearance_bridge_report is not None
                else parameters.get(
                    "execution_pose_source", "signed_cartesian_waypoint"
                )
            ),
            "clearance_bridge": clearance_bridge_report,
            "clearance_bridge_preflight_error": (clearance_bridge_preflight_error),
            "error_decomposition": {
                "planning_ik_residual": (
                    "ideal Cartesian pose to signed joint target FK"
                ),
                "control_tracking_error": (
                    "signed joint target FK pose to evaluator observation"
                ),
            },
            "compat_plan_mode": compat_mode or None,
            "exec_sequence": "move_only",
            "compat_gripper_transition": "preserve",
            "plan_requested_gripper_transition": requested_gripper_transition,
            "gripper_transitions": [],
            "gripper_transition_handled": False,
            "gripper_control": "separate_tool_only",
            "gripper_qpos_before": gripper_before,
            "gripper_qpos_after": gripper_after,
            "plan_stale_action_steps_at_start": stale_steps,
            "stop_after_safe": stop_after_safe,
            "requested_back_m": kwargs.get("back_m"),
            "position_tolerance_m": position_tolerance,
            "orientation_tolerance_deg": orientation_tolerance,
            "collision_checked": False,
        },
    )
    yield ctx.world.hold_action()


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("exec_plan_pose", exec_plan_pose)
