"""reset_body

腰部三关节复位直立；keep_ori_arm 保持入口 J1-J4 命令不变，每帧仅用 J5-J7 追踪入口世界系 EEF 姿态；EEF 平移只监控；其余手臂全关节锁定；完成后拍主视图。

How this tool works:
Restore torso/head posture through a bounded joint trajectory while preserving carrying grippers. This is physical joint motion, not teleportation or an evaluator episode reset.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from dataclasses import dataclass
from tools.util.action_constants import CONTROL_HZ
from tools.util.action_constants import TRUNK_LIMITS
from tools.util.geometry.contract import ARM_DOF
from tools.util.geometry.eef_adjustment_local import local_robot_state
from tools.util.geometry.eef_adjustment_local import locked_arm_vector
from tools.util.geometry.eef_adjustment_local import numerical_eef_jacobian
from tools.util.geometry.eef_adjustment_local import orientation_error_deg
from tools.util.geometry.eef_adjustment_local import solve_j567_orientation_target
from tools.util.geometry.grasp_geometry_local import mat_to_quat_xyzw as _mat_to_quat_xyzw
from tools.util.geometry.grasp_kinematics_local import LocalRobotState
from tools.util.geometry.grasp_kinematics_local import arm_joint_limits as _local_arm_joint_limits
from tools.util.geometry.grasp_kinematics_local import eef_pose as _local_eef_pose
from tools.util.motion_common import _adjust_gripper_vector
from tools.util.motion_common import _adjust_kinematic_observation
from tools.util.motion_common import _axis_rotation
from tools.util.motion_common import _check_cancelled
from tools.util.motion_common import _quat_to_mat_xyzw
from tools.util.motion_common import _rotation_vector
from tools.util.motion_common import _set_result
from tools.util.motion_common import _smoothstep
from typing import Any
import math
import numpy as np
import time
from tools.util.tool import main

NAME = 'reset_body'
SCHEMA = {'type': 'object',
 'properties': {'keep_ori_arm': {'type': 'string',
                                 'enum': ['none', 'left', 'right', 'both'],
                                 'default': 'none'},
                'timeout_s': {'type': 'number', 'default': 45.0, 'description': 's'},
                'trunk_max_step': {'type': 'number', 'default': 0.06, 'description': 'rad/步'},
                'shoulder_iters': {'type': 'number', 'default': 4, 'description': '兼容保留/已忽略'}},
 'additionalProperties': False}


def actions(
    robot,
    keep_ori_arm: str = "none",
    timeout_s: float = 45.0,
    trunk_max_step: float | None = None,
    shoulder_iters: int = 4,
    pitch_deg: float = 0.0,
    **kwargs,
):
    # Retained only for compatibility with the old simulator-backed signature.
    del shoulder_iters
    try:
        keep_arms = _reset_body_keep_arms(keep_ori_arm)
    except ValueError as exc:
        _set_result(robot, "reset_body", {"ok": False, "error": str(exc)})
        yield robot.hold_action()
        return
    tolerance = max(1e-4, float(kwargs.get("tolerance_rad", 0.035)))
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        _set_result(
            robot,
            "reset_body",
            {"ok": False, "error": "timeout_s must be a positive finite value"},
        )
        yield robot.hold_action()
        return
    start = np.asarray(robot.trunk_qpos(), dtype=np.float64).reshape(4)
    target = R1PRO_UPRIGHT_TRUNK_Q.copy()
    target[2] = -math.radians(float(pitch_deg))
    target = np.clip(target, TRUNK_LIMITS[:, 0], TRUNK_LIMITS[:, 1])
    defaulted_max_step = trunk_max_step is None
    requested_max_step = (
        RESET_BODY_KEEP_DEFAULT_TRUNK_MAX_STEP_RAD
        if keep_arms and defaulted_max_step
        else RESET_BODY_DEFAULT_TRUNK_MAX_STEP_RAD
        if defaulted_max_step
        else float(trunk_max_step)
    )
    if not math.isfinite(requested_max_step) or requested_max_step <= 0.0:
        _set_result(
            robot,
            "reset_body",
            {
                "ok": False,
                "error": "trunk_max_step must be a positive finite value",
            },
        )
        yield robot.hold_action()
        return
    keep_initialization_error: str | None = None
    try:
        keep_states = _reset_body_init_keep_states(robot, keep_arms)
    except Exception as exc:  # keep-arm initialization is advisory
        # keep_ori_arm is advisory. Missing arm kinematics must never prevent
        # the independently controlled trunk from returning upright.
        keep_states = {}
        keep_initialization_error = str(exc)
    keep_requested = bool(keep_arms)
    max_step = (
        min(requested_max_step, RESET_BODY_KEEP_TRUNK_MAX_STEP_RAD)
        if keep_requested
        else requested_max_step
    )
    steps = max(
        2,
        int(
            math.ceil(
                (
                    float(np.linalg.norm(target - start, ord=np.inf))
                    / max_step
                )
                if keep_requested
                else (
                    SMOOTHSTEP_MAX_SLOPE
                    * float(np.linalg.norm(target - start, ord=np.inf))
                    / max_step
                )
            )
        ),
    )
    deadline = time.monotonic() + timeout
    action_steps = 0
    timed_out = False
    body_failure: str | None = None
    # Compatibility metric: these are now concurrent catch-up actions.  The
    # body is never pinned for an orientation-only recovery during motion.
    keep_recovery_steps = 0
    keep_tracking_guard_steps = 0
    keep_relaxed_plan_steps = 0
    keep_fallback_steps = 0
    keep_action_fallback_steps = 0
    keep_observation_failure_steps = 0
    keep_hard_limit_observation_steps = 0
    keep_settle_steps_after_trunk = 0
    keep_plan_backoff_steps = 0
    last_keep_plan: dict[str, Any] = {"ok": True, "arms": {}}
    body_progress_steps = 0
    stalled = False
    stall_steps = 0
    if keep_requested:
        body_step_limit = steps + RESET_BODY_TRACKING_STALL_STEPS
        body_best_error = float("inf")
        body_stall_steps = 0
        while body_progress_steps < body_step_limit:
            _check_cancelled(robot, "reset body interpolation")
            current_trunk = np.asarray(
                robot.trunk_qpos(),
                dtype=np.float64,
            ).reshape(4)
            observed_error = float(np.max(np.abs(target - current_trunk)))
            if observed_error <= tolerance:
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            observed_keep = _reset_body_safe_observed_keep_report(
                robot,
                keep_states,
            )
            if observed_keep.get("observation_error"):
                keep_observation_failure_steps += 1
            orientation_error = max(
                (
                    float(item["effective_orientation_error_deg"])
                    for item in observed_keep["arms"].values()
                ),
                default=0.0,
            )
            if not observed_keep.get("within_orientation_hard_limits", True):
                keep_hard_limit_observation_steps += 1
            position_error = max(
                (
                    float(item["effective_position_error_m"])
                    for item in observed_keep["arms"].values()
                ),
                default=0.0,
            )
            tracking_guard_exceeded = bool(
                not observed_keep.get(
                    "within_orientation_tracking_guard",
                    orientation_error
                    <= RESET_BODY_KEEP_ORIENTATION_TRACKING_GUARD_DEG,
                )
            )
            if tracking_guard_exceeded:
                keep_tracking_guard_steps += 1
                keep_recovery_steps += 1
            if observed_error + 1e-4 < body_best_error:
                body_best_error = observed_error
                body_stall_steps = 0
            else:
                body_stall_steps += 1
            stall_steps = body_stall_steps
            if body_stall_steps >= RESET_BODY_TRACKING_STALL_STEPS:
                stalled = True
                break
            command, _step_cap, _slowdown_scale = (
                _reset_body_keep_trunk_command(
                    current_trunk,
                    target,
                    requested_max_step,
                    body_progress_steps,
                    position_error,
                )
            )
            wrist_step = (
                RESET_BODY_KEEP_J567_CATCHUP_MAX_STEP_RAD
                if tracking_guard_exceeded
                else RESET_BODY_KEEP_J567_MAX_STEP_RAD
            )
            arm_overrides: dict[str, list[float]] = {}
            if keep_states:
                (
                    arm_overrides,
                    last_keep_plan,
                    used_best_effort,
                    used_fallback,
                ) = _reset_body_best_effort_keep_overrides(
                    robot,
                    command,
                    keep_states,
                    j567_max_step_rad=wrist_step,
                )
                if used_best_effort:
                    keep_relaxed_plan_steps += 1
                if used_fallback:
                    keep_fallback_steps += 1
            action, arm_action_error = _reset_body_body_priority_action(
                robot,
                command,
                arm_overrides,
            )
            if arm_action_error is not None:
                keep_action_fallback_steps += 1
                last_keep_plan = {
                    **last_keep_plan,
                    "arm_action_error": arm_action_error,
                    "arm_action_fallback": "trunk_with_existing_arm_pins",
                }
            yield action
            action_steps += 1
            body_progress_steps += 1
        if (
            body_progress_steps >= body_step_limit
            and not timed_out
            and body_failure is None
            and not stalled
        ):
            achieved_now = np.asarray(
                robot.trunk_qpos(),
                dtype=np.float64,
            ).reshape(4)
            if float(np.max(np.abs(target - achieved_now))) > tolerance:
                body_failure = "reset_body body-step budget exhausted"
    else:
        for index in range(1, steps + 1):
            if time.monotonic() >= deadline:
                timed_out = True
                break
            _check_cancelled(robot, "reset body interpolation")
            alpha = _smoothstep(index / steps)
            command = start + alpha * (target - start)
            yield robot.make_action(trunk=command.tolist())
            action_steps += 1
        body_progress_steps = steps

    required_stable = (
        RESET_BODY_KEEP_SETTLE_STABLE_STEPS if keep_requested else 3
    )
    settle_stall_limit = (
        RESET_BODY_KEEP_SETTLE_STALL_STEPS
        if keep_requested
        else RESET_BODY_TRACKING_STALL_STEPS
    )
    max_settle_steps = max(
        required_stable,
        min(
            int(math.ceil(timeout * CONTROL_HZ)),
            RESET_BODY_KEEP_SETTLE_MAX_STEPS if keep_requested else 90,
        ),
    )
    stable = 0
    settle_steps = 0
    if not stalled:
        stall_steps = 0
    best_error = float("inf")
    if not timed_out and body_failure is None and not stalled:
        for _ in range(max_settle_steps):
            if time.monotonic() >= deadline:
                current_error = float(
                    np.max(
                        np.abs(
                            target
                            - np.asarray(
                                robot.trunk_qpos(),
                                dtype=np.float64,
                            ).reshape(4)
                        )
                    )
                )
                timed_out = current_error > tolerance
                break
            _check_cancelled(robot, "reset body settle")
            arm_overrides: dict[str, list[float]] = {}
            if keep_states:
                pre_action_keep = _reset_body_safe_observed_keep_report(
                    robot,
                    keep_states,
                )
                if pre_action_keep.get("observation_error"):
                    keep_observation_failure_steps += 1
                pre_action_orientation_error = max(
                    (
                        float(item["effective_orientation_error_deg"])
                        for item in pre_action_keep["arms"].values()
                    ),
                    default=0.0,
                )
                if not pre_action_keep.get(
                    "within_orientation_hard_limits",
                    True,
                ):
                    keep_hard_limit_observation_steps += 1
                catchup = bool(
                    not pre_action_keep.get(
                        "within_orientation_tracking_guard",
                        pre_action_orientation_error
                        <= RESET_BODY_KEEP_ORIENTATION_TRACKING_GUARD_DEG,
                    )
                )
                if catchup:
                    keep_tracking_guard_steps += 1
                    keep_recovery_steps += 1
                (
                    arm_overrides,
                    last_keep_plan,
                    used_best_effort,
                    used_fallback,
                ) = _reset_body_best_effort_keep_overrides(
                    robot,
                    target,
                    keep_states,
                    j567_max_step_rad=(
                        RESET_BODY_KEEP_J567_CATCHUP_MAX_STEP_RAD
                        if catchup
                        else RESET_BODY_KEEP_J567_MAX_STEP_RAD
                    ),
                )
                if used_best_effort:
                    keep_relaxed_plan_steps += 1
                if used_fallback:
                    keep_fallback_steps += 1
            action, arm_action_error = _reset_body_body_priority_action(
                robot,
                target,
                arm_overrides,
            )
            if arm_action_error is not None:
                keep_action_fallback_steps += 1
                last_keep_plan = {
                    **last_keep_plan,
                    "arm_action_error": arm_action_error,
                    "arm_action_fallback": "trunk_with_existing_arm_pins",
                }
            yield action
            action_steps += 1
            settle_steps += 1
            observed = np.asarray(
                robot.trunk_qpos(),
                dtype=np.float64,
            ).reshape(4)
            observed_error = float(np.max(np.abs(target - observed)))
            observed_keep = _reset_body_safe_observed_keep_report(
                robot,
                keep_states,
            )
            if observed_keep.get("observation_error"):
                keep_observation_failure_steps += 1
            if not observed_keep.get("within_orientation_hard_limits", True):
                keep_hard_limit_observation_steps += 1
            body_metric = observed_error / tolerance
            if observed_error <= tolerance:
                stable += 1
            else:
                stable = 0
            trunk_is_stable = stable >= required_stable
            if trunk_is_stable:
                if (
                    not keep_states
                    or observed_keep["within_orientation_final_tolerances"]
                ):
                    break
                keep_settle_steps_after_trunk += 1
                # The trunk is already stable. Any remaining actions are a
                # bounded wrist-only quality pass and cannot mark body stall.
                continue
            if body_metric + 1e-4 < best_error:
                best_error = body_metric
                stall_steps = 0
            else:
                stall_steps += 1
            if stall_steps >= settle_stall_limit:
                stalled = True
                break
    achieved = np.asarray(
        robot.trunk_qpos(),
        dtype=np.float64,
    ).reshape(4)
    error = target - achieved
    max_error = float(np.max(np.abs(error)))
    final_keep = _reset_body_safe_observed_keep_report(robot, keep_states)
    if final_keep.get("observation_error"):
        keep_observation_failure_steps += 1
    if not final_keep.get("within_orientation_hard_limits", True):
        keep_hard_limit_observation_steps += 1
    keep_ok = bool(
        keep_initialization_error is None
        and final_keep["within_orientation_final_tolerances"]
    )
    position_drift_warning = any(
        float(item.get("effective_position_error_m") or 0.0)
        > RESET_BODY_KEEP_POSITION_WARNING_M
        for item in (final_keep.get("arms") or {}).values()
    )
    ok = bool(
        max_error <= tolerance
        and not timed_out
        and not stalled
        and body_failure is None
    )
    keep_value = (
        "both"
        if len(keep_arms) == 2
        else (keep_arms[0] if keep_arms else "none")
    )
    keep_report = _reset_body_keep_result(keep_states, final_keep)
    keep_report.update(
        {
            "requested": keep_requested,
            "requested_arms": list(keep_arms),
            "initialized_arms": list(keep_states),
            "initialization_error": keep_initialization_error,
            "orientation_satisfied": bool(keep_ok),
            "affects_trunk_motion": False,
            "affects_reset_success": False,
            "trunk_priority": "absolute",
        }
    )
    keep_degraded = bool(
        keep_requested
        and (
            not keep_ok
            or keep_initialization_error is not None
            or keep_fallback_steps > 0
            or keep_action_fallback_steps > 0
            or keep_observation_failure_steps > 0
            or keep_hard_limit_observation_steps > 0
        )
    )
    warnings: list[str] = []
    if keep_initialization_error is not None:
        warnings.append(
            "keep-orientation initialization failed; trunk reset continued"
        )
    if keep_observation_failure_steps:
        warnings.append(
            "keep-orientation observation failed; trunk reset continued"
        )
    if keep_hard_limit_observation_steps:
        warnings.append(
            "kept EEF exceeded the 25 deg diagnostic threshold; trunk reset continued"
        )
    if keep_fallback_steps:
        warnings.append(
            "keep-orientation IK degraded to measured-follow; trunk reset continued"
        )
    if keep_action_fallback_steps:
        warnings.append(
            "kept-arm action override was rejected; trunk-only command was used"
        )
    if keep_requested and not keep_ok:
        warnings.append(
            "kept EEF orientation did not reach the advisory final tolerance"
        )
    result = {
        "ok": bool(ok),
        "implementation_version": RESET_BODY_IMPLEMENTATION_VERSION,
        "target_trunk_q": target.astype(float).tolist(),
        "achieved_qpos": achieved.astype(float).tolist(),
        "error_rad": error.astype(float).tolist(),
        "max_abs_error_rad": max_error,
        "tolerance_rad": tolerance,
        "settle_steps": int(settle_steps),
        "action_steps": int(action_steps),
        "interpolation_steps": int(steps),
        "body_progress_steps": int(body_progress_steps),
        "planned_interpolation_steps": int(steps),
        "requested_max_step_rad": requested_max_step,
        "defaulted_max_step": bool(defaulted_max_step),
        "effective_max_step_rad": max_step,
        "keep_ori_arm": keep_value,
        "trunk_priority": "absolute",
        "keep_orientation_affects_trunk": False,
        "keep_orientation_affects_success": False,
        "keep_orientation_satisfied": bool(keep_ok),
        "keep_orientation_degraded": bool(keep_degraded),
        "keep_orientation": keep_report,
        "position_drift_warning": bool(position_drift_warning),
        "keep_recovery_steps": int(keep_recovery_steps),
        "keep_tracking_guard_steps": int(keep_tracking_guard_steps),
        "keep_relaxed_plan_steps": int(keep_relaxed_plan_steps),
        "keep_fallback_steps": int(keep_fallback_steps),
        "keep_action_fallback_steps": int(keep_action_fallback_steps),
        "keep_observation_failure_steps": int(keep_observation_failure_steps),
        "keep_hard_limit_observation_steps": int(
            keep_hard_limit_observation_steps
        ),
        "keep_settle_steps_after_trunk": int(keep_settle_steps_after_trunk),
        "stationary_recovery_steps_during_body_motion": 0,
        "keep_plan_backoff_steps": int(keep_plan_backoff_steps),
        "last_keep_plan": last_keep_plan,
        "warnings": warnings,
        "stall_steps": int(stall_steps),
        "tracking_stalled": bool(stalled),
        "timed_out": bool(timed_out),
        "timeout_s": timeout,
        "arms_grippers": (
            "selected arms software-compliant; other limbs action-pinned"
            if keep_states
            else (
                "keep-orientation unavailable; limbs action-pinned"
                if keep_requested
                else "pinned through action targets"
            )
        ),
        "direct_simulator_mutation": False,
    }
    if not ok:
        result["error"] = (
            body_failure
            if body_failure is not None
            else (
                "reset_body timed out before trunk reached target"
                if timed_out
                else (
                    "reset_body trunk tracking stalled before reaching target"
                    if stalled
                    else "trunk did not converge using evaluator-proprio action control"
                )
            )
        )
    _set_result(robot, "reset_body", result)
    if ok:
        arm_overrides: dict[str, list[float]] = {}
        if keep_states:
            arm_overrides, _final_plan, _best_effort, _fallback = (
                _reset_body_best_effort_keep_overrides(
                    robot,
                    target,
                    keep_states,
                )
            )
        final_action, _arm_action_error = _reset_body_body_priority_action(
            robot,
            target,
            arm_overrides,
        )
        yield final_action
    else:
        yield robot.hold_action()


R1PRO_UPRIGHT_TRUNK_Q = np.array(
    [0.45, -0.4, 0.0, 0.0],
    dtype=np.float64,
)



RESET_BODY_TRACKING_STALL_STEPS = 20



RESET_BODY_IMPLEMENTATION_VERSION = "body_first_nonblocking_keep_ori_v4"



RESET_BODY_DEFAULT_TRUNK_MAX_STEP_RAD = 0.060



RESET_BODY_KEEP_DEFAULT_TRUNK_MAX_STEP_RAD = 0.080



RESET_BODY_KEEP_TRUNK_MAX_STEP_RAD = 0.080



RESET_BODY_KEEP_TRUNK_RAMP_STEPS = 3



RESET_BODY_KEEP_POSITION_DEADBAND_M = 0.020



RESET_BODY_KEEP_POSITION_WARNING_M = 0.060



RESET_BODY_KEEP_POSITION_FINAL_TOL_M = 0.100



RESET_BODY_KEEP_POSITION_HARD_LIMIT_M = 0.120



RESET_BODY_KEEP_ORIENTATION_SOLVE_TOL_DEG = 2.0



RESET_BODY_KEEP_ORIENTATION_FINAL_TOL_DEG = 3.0



RESET_BODY_KEEP_ORIENTATION_PLAN_LIMIT_DEG = 4.0



RESET_BODY_KEEP_ORIENTATION_TRACKING_GUARD_DEG = 5.0



RESET_BODY_KEEP_ORIENTATION_HARD_LIMIT_DEG = 25.0



RESET_BODY_KEEP_ORIENTATION_BEST_EFFORT_LIMIT_DEG = 180.0



RESET_BODY_KEEP_RAW_FEEDBACK_MAX_GAIN = 1.50



RESET_BODY_KEEP_RAW_FEEDBACK_FULL_GAIN_DEG = 10.0



RESET_BODY_KEEP_Q1234_MAX_STEP_RAD = 0.040



RESET_BODY_KEEP_J567_MAX_STEP_RAD = 0.080



RESET_BODY_KEEP_J567_CATCHUP_MAX_STEP_RAD = 0.240



RESET_BODY_KEEP_POSITION_DAMPING = 0.040



RESET_BODY_KEEP_ALTERNATING_PASSES = 2



RESET_BODY_KEEP_POSITION_LINE_SEARCH_SCALES = (1.0, 0.5, 0.25)



RESET_BODY_KEEP_SETTLE_MAX_STEPS = 12



RESET_BODY_KEEP_SETTLE_STALL_STEPS = 12



RESET_BODY_KEEP_SETTLE_STABLE_STEPS = 2



SMOOTHSTEP_MAX_SLOPE = 1.5



@dataclass
class _ResetBodyKeepOriState:
    arm: str
    reference_position: np.ndarray
    reference_quaternion: np.ndarray
    initial_q: np.ndarray
    last_command: np.ndarray
    raw_reference_position: np.ndarray | None
    raw_reference_quaternion: np.ndarray | None
    solve_calls: int = 0
    max_observed_position_error_m: float = 0.0
    max_observed_orientation_error_deg: float = 0.0
    max_command_q1234_delta_rad: float = 0.0
    max_command_j567_delta_rad: float = 0.0
    max_command_tracking_error_rad: float = 0.0
    last_solver_report: dict[str, Any] | None = None



def _reset_body_keep_arms(value: Any) -> tuple[str, ...]:
    normalized = str(value if value is not None else "none").strip().lower()
    if normalized in ("", "false", "0", "no"):
        normalized = "none"
    if normalized == "none":
        return ()
    if normalized == "both":
        return ("left", "right")
    if normalized in ("left", "right"):
        return (normalized,)
    raise ValueError("keep_ori_arm must be none, left, right, or both")



def _reset_body_raw_eef_pose(ctx, arm: str) -> dict[str, np.ndarray] | None:
    adapter = getattr(ctx.world, "_official_adapter", None)
    if adapter is None:
        return None
    try:
        raw = dict((adapter.eef_pose() or {}).get(arm) or {})
        position = np.asarray(raw.get("pos"), dtype=np.float64).reshape(3)
        quaternion = np.asarray(raw.get("quat"), dtype=np.float64).reshape(4)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(quaternion))
    if (
        not np.all(np.isfinite(position))
        or not np.all(np.isfinite(quaternion))
        or norm <= 1e-9
    ):
        return None
    return {
        "position": position.copy(),
        "quaternion": quaternion / norm,
    }



def _reset_body_init_keep_states(
    ctx,
    arms: tuple[str, ...],
) -> dict[str, _ResetBodyKeepOriState]:
    states: dict[str, _ResetBodyKeepOriState] = {}
    for arm in arms:
        observed = _adjust_kinematic_observation(ctx, arm)
        raw_pose = _reset_body_raw_eef_pose(ctx, arm)
        states[arm] = _ResetBodyKeepOriState(
            arm=arm,
            reference_position=observed["position"].copy(),
            reference_quaternion=observed["quaternion"].copy(),
            initial_q=observed["q"].copy(),
            last_command=observed["q"].copy(),
            raw_reference_position=(
                None if raw_pose is None else raw_pose["position"].copy()
            ),
            raw_reference_quaternion=(
                None if raw_pose is None else raw_pose["quaternion"].copy()
            ),
        )
    return states



def _reset_body_local_state(ctx, trunk_q: Any) -> LocalRobotState:
    trunk = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    left = locked_arm_vector(
        ctx.world.arm_qpos_list("left"),
        label="left arm proprioception",
    )
    right = locked_arm_vector(
        ctx.world.arm_qpos_list("right"),
        label="right arm proprioception",
    )
    return local_robot_state(
        trunk_q=trunk,
        arm_left_q=left,
        arm_right_q=right,
        gripper_left_q=_adjust_gripper_vector(
            ctx.world.gripper_qpos_list("left")
        ),
        gripper_right_q=_adjust_gripper_vector(
            ctx.world.gripper_qpos_list("right")
        ),
    )



def _reset_body_orientation_residual_deg(solve_report: dict[str, Any]) -> float:
    for key in ("ori_err_deg", "residual_deg"):
        raw = solve_report.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return float("inf")



def _reset_body_solve_bounded_j567(
    state: LocalRobotState,
    arm: str,
    q_candidate: np.ndarray,
    measured_q: np.ndarray,
    target_quaternion: np.ndarray,
    *,
    max_step_rad: float = RESET_BODY_KEEP_J567_MAX_STEP_RAD,
) -> tuple[np.ndarray, dict[str, Any], int]:
    """Solve J5-J7 and validate the command after per-action clipping."""
    candidate = np.asarray(q_candidate, dtype=np.float64).reshape(ARM_DOF).copy()
    measured = np.asarray(measured_q, dtype=np.float64).reshape(ARM_DOF)
    solved, solve_report = solve_j567_orientation_target(
        state,
        arm,
        candidate,
        target_quaternion,
        ori_tol_deg=RESET_BODY_KEEP_ORIENTATION_SOLVE_TOL_DEG,
        allow_global_search=False,
    )
    solve_calls = 1
    residual_deg = _reset_body_orientation_residual_deg(solve_report)
    if residual_deg > RESET_BODY_KEEP_ORIENTATION_FINAL_TOL_DEG:
        solved, solve_report = solve_j567_orientation_target(
            state,
            arm,
            candidate,
            target_quaternion,
            ori_tol_deg=RESET_BODY_KEEP_ORIENTATION_SOLVE_TOL_DEG,
            allow_global_search=True,
        )
        solve_calls += 1
        residual_deg = _reset_body_orientation_residual_deg(solve_report)

    lower, upper = _local_arm_joint_limits(state, arm)
    bounded_step = max(
        1e-6,
        min(
            float(max_step_rad),
            RESET_BODY_KEEP_J567_CATCHUP_MAX_STEP_RAD,
        ),
    )
    candidate[4:7] = np.clip(
        solved[4:7],
        measured[4:7] - bounded_step,
        measured[4:7] + bounded_step,
    )
    candidate[4:7] = np.clip(candidate[4:7], lower[4:7], upper[4:7])
    if ARM_DOF == 8:
        candidate[7] = 0.0
    _position, commanded_quaternion = _local_eef_pose(state, arm, candidate)
    fk_commanded_error_deg = orientation_error_deg(
        commanded_quaternion,
        target_quaternion,
    )
    solver_report_valid = math.isfinite(residual_deg)
    commanded_error_deg = (
        fk_commanded_error_deg if solver_report_valid else float("inf")
    )
    report = {
        **dict(solve_report),
        "solver_report_valid": bool(solver_report_valid),
        "unclipped_orientation_error_deg": float(residual_deg),
        "commanded_orientation_error_deg": float(commanded_error_deg),
        "fk_commanded_orientation_error_deg": float(fk_commanded_error_deg),
        "commanded_within_final_tolerance": bool(
            commanded_error_deg <= RESET_BODY_KEEP_ORIENTATION_FINAL_TOL_DEG
        ),
        "j567_clipped_to_action_step": bool(
            np.linalg.norm(candidate[4:7] - solved[4:7], ord=np.inf) > 1e-9
        ),
        "j567_action_step_limit_rad": float(bounded_step),
    }
    return candidate, report, solve_calls



def _reset_body_feedback_orientation_target(
    ctx,
    arm: str,
    keeper: _ResetBodyKeepOriState,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map evaluator EEF error onto the local-FK orientation target.

    The local URDF is used for J5-J7 IK, but evaluator EEF quaternion is the
    authoritative measurement.  Applying the measured raw-frame correction
    to the current local pose closes the model-offset loop instead of repeatedly
    issuing an already-satisfied local target.
    """
    raw_pose = _reset_body_raw_eef_pose(ctx, arm)
    if (
        raw_pose is None
        or keeper.raw_reference_quaternion is None
    ):
        return keeper.reference_quaternion.copy(), {
            "source": "fixed_local_entry_orientation",
            "raw_feedback_available": False,
        }

    actual_state = _reset_body_local_state(ctx, ctx.world.trunk_qpos())
    measured = (
        actual_state.arm_left_q
        if arm == "left"
        else actual_state.arm_right_q
    )
    _position, local_quaternion = _local_eef_pose(
        actual_state,
        arm,
        measured,
    )
    raw_current_rotation = _quat_to_mat_xyzw(raw_pose["quaternion"])
    raw_reference_rotation = _quat_to_mat_xyzw(
        keeper.raw_reference_quaternion
    )
    local_current_rotation = _quat_to_mat_xyzw(local_quaternion)
    correction = raw_reference_rotation @ raw_current_rotation.T
    correction_vector = _rotation_vector(correction)
    correction_angle = float(np.linalg.norm(correction_vector))
    raw_error_deg = math.degrees(correction_angle)
    gain_fraction = np.clip(
        (
            raw_error_deg - RESET_BODY_KEEP_ORIENTATION_FINAL_TOL_DEG
        )
        / (
            RESET_BODY_KEEP_RAW_FEEDBACK_FULL_GAIN_DEG
            - RESET_BODY_KEEP_ORIENTATION_FINAL_TOL_DEG
        ),
        0.0,
        1.0,
    )
    feedback_gain = 1.0 + float(gain_fraction) * (
        RESET_BODY_KEEP_RAW_FEEDBACK_MAX_GAIN - 1.0
    )
    if correction_angle > 1e-9:
        correction = _axis_rotation(
            correction_vector / correction_angle,
            correction_angle * feedback_gain,
        )
    target_rotation = correction @ local_current_rotation
    target_quaternion = _mat_to_quat_xyzw(target_rotation)
    return np.asarray(target_quaternion, dtype=np.float64).reshape(4), {
        "source": "evaluator_eef_delta_mapped_to_local_fk",
        "raw_feedback_available": True,
        "raw_orientation_error_deg": float(raw_error_deg),
        "raw_orientation_feedback_gain": float(feedback_gain),
        "local_target_offset_from_entry_deg": orientation_error_deg(
            target_quaternion,
            keeper.reference_quaternion,
        ),
    }



def _reset_body_soft_position_step(
    state: LocalRobotState,
    arm: str,
    q_current: np.ndarray,
    measured_q: np.ndarray,
    reference_position: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    command = np.asarray(q_current, dtype=np.float64).reshape(ARM_DOF).copy()
    measured = np.asarray(measured_q, dtype=np.float64).reshape(ARM_DOF)
    jacobian, position, _quaternion = numerical_eef_jacobian(
        state,
        arm,
        command,
        joint_indices=(0, 1, 2, 3),
    )
    error = np.asarray(reference_position, dtype=np.float64).reshape(3) - position
    error_norm = float(np.linalg.norm(error))
    applied_error = np.zeros(3, dtype=np.float64)
    delta = np.zeros(4, dtype=np.float64)
    if error_norm > RESET_BODY_KEEP_POSITION_DEADBAND_M:
        applied_error = error * (
            (error_norm - RESET_BODY_KEEP_POSITION_DEADBAND_M)
            / max(error_norm, 1e-12)
        )
        translation_jacobian = jacobian[:3, :]
        lhs = (
            translation_jacobian @ translation_jacobian.T
            + RESET_BODY_KEEP_POSITION_DAMPING**2 * np.eye(3)
        )
        try:
            delta = translation_jacobian.T @ np.linalg.solve(lhs, applied_error)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(
                translation_jacobian,
                applied_error,
                rcond=None,
            )[0]
        delta_norm = float(np.linalg.norm(delta, ord=np.inf))
        if delta_norm > RESET_BODY_KEEP_Q1234_MAX_STEP_RAD:
            delta *= RESET_BODY_KEEP_Q1234_MAX_STEP_RAD / (
                delta_norm + 1e-12
            )

    lower, upper = _local_arm_joint_limits(state, arm)
    best_command = command.copy()
    best_position = np.asarray(position, dtype=np.float64).copy()
    best_error_norm = error_norm
    accepted_scale = 0.0
    for scale in RESET_BODY_KEEP_POSITION_LINE_SEARCH_SCALES:
        candidate = command.copy()
        proposed_q1234 = command[:4] + float(scale) * delta
        proposed_q1234 = np.clip(
            proposed_q1234,
            measured[:4] - RESET_BODY_KEEP_Q1234_MAX_STEP_RAD,
            measured[:4] + RESET_BODY_KEEP_Q1234_MAX_STEP_RAD,
        )
        candidate[:4] = np.clip(proposed_q1234, lower[:4], upper[:4])
        candidate_position, _candidate_quaternion = _local_eef_pose(
            state,
            arm,
            candidate,
        )
        candidate_error_norm = float(
            np.linalg.norm(candidate_position - reference_position)
        )
        if candidate_error_norm < best_error_norm - 1e-6:
            best_command = candidate
            best_position = np.asarray(
                candidate_position,
                dtype=np.float64,
            ).copy()
            best_error_norm = candidate_error_norm
            accepted_scale = float(scale)
    command = best_command
    if ARM_DOF == 8:
        command[7] = 0.0
    return command, {
        "position_before_m": np.asarray(position, dtype=np.float64).tolist(),
        "position_error_before_m": error_norm,
        "deadband_m": RESET_BODY_KEEP_POSITION_DEADBAND_M,
        "applied_translation_error_m": applied_error.tolist(),
        "accepted_line_search_scale": accepted_scale,
        "position_after_m": best_position.tolist(),
        "position_error_after_m": best_error_norm,
        "q1234_delta_from_measured_rad": (
            command[:4] - measured[:4]
        ).tolist(),
    }



def _reset_body_plan_keep_overrides(
    ctx,
    trunk_command: Any,
    keep_states: dict[str, _ResetBodyKeepOriState],
    *,
    orientation_limit_deg: float = RESET_BODY_KEEP_ORIENTATION_PLAN_LIMIT_DEG,
    j567_max_step_rad: float = RESET_BODY_KEEP_J567_MAX_STEP_RAD,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    state = _reset_body_local_state(ctx, trunk_command)
    measured_by_arm = {
        "left": state.arm_left_q.copy(),
        "right": state.arm_right_q.copy(),
    }
    overrides: dict[str, list[float]] = {}
    reports: dict[str, Any] = {}
    for arm, keeper in keep_states.items():
        measured = measured_by_arm[arm]
        target_quaternion, target_report = (
            _reset_body_feedback_orientation_target(ctx, arm, keeper)
        )
        tracking_error = float(
            np.linalg.norm(keeper.last_command - measured, ord=np.inf)
        )
        candidate = measured.copy()
        position_reports: list[dict[str, Any]] = []
        solve_report: dict[str, Any] | None = None
        for _ in range(RESET_BODY_KEEP_ALTERNATING_PASSES):
            position_candidate, position_report = _reset_body_soft_position_step(
                state,
                arm,
                candidate,
                measured,
                keeper.reference_position,
            )
            position_reports.append(position_report)
            candidate, solve_report, solve_calls = _reset_body_solve_bounded_j567(
                state,
                arm,
                position_candidate,
                measured,
                target_quaternion,
                max_step_rad=j567_max_step_rad,
            )
            keeper.solve_calls += solve_calls
            commanded_error_deg = float(
                solve_report["commanded_orientation_error_deg"]
            )
            if commanded_error_deg > RESET_BODY_KEEP_ORIENTATION_FINAL_TOL_DEG:
                orientation_candidate, orientation_report, fallback_calls = (
                    _reset_body_solve_bounded_j567(
                        state,
                        arm,
                        measured,
                        measured,
                        target_quaternion,
                        max_step_rad=j567_max_step_rad,
                    )
                )
                keeper.solve_calls += fallback_calls
                fallback_error_deg = float(
                    orientation_report["commanded_orientation_error_deg"]
                )
                if fallback_error_deg < commanded_error_deg:
                    candidate = orientation_candidate
                    solve_report = orientation_report
                    commanded_error_deg = fallback_error_deg
                    position_report["discarded_for_orientation"] = True
                    position_report["orientation_only_fallback"] = True
            if commanded_error_deg > float(orientation_limit_deg):
                reports[arm] = {
                    "ok": False,
                    "error": "J5-J7 cannot preserve the entry orientation",
                    "orientation_acceptance_limit_deg": float(
                        orientation_limit_deg
                    ),
                    "orientation_target": target_report,
                    "orientation_solve": dict(solve_report),
                    "position_passes": position_reports,
                }
                return {}, {"ok": False, "arms": reports}
            if bool(position_report.get("orientation_only_fallback", False)):
                break

        final_position, final_quaternion = _local_eef_pose(
            state,
            arm,
            candidate,
        )
        position_error = float(
            np.linalg.norm(final_position - keeper.reference_position)
        )
        orientation_target_error = orientation_error_deg(
            final_quaternion,
            target_quaternion,
        )
        local_entry_orientation_error = orientation_error_deg(
            final_quaternion,
            keeper.reference_quaternion,
        )
        q1234_delta = float(
            np.linalg.norm(candidate[:4] - measured[:4], ord=np.inf)
        )
        j567_delta = float(
            np.linalg.norm(candidate[4:7] - measured[4:7], ord=np.inf)
        )
        keeper.last_command = candidate.copy()
        keeper.last_solver_report = None if solve_report is None else dict(solve_report)
        keeper.max_command_tracking_error_rad = max(
            keeper.max_command_tracking_error_rad,
            tracking_error,
        )
        keeper.max_command_q1234_delta_rad = max(
            keeper.max_command_q1234_delta_rad,
            q1234_delta,
        )
        keeper.max_command_j567_delta_rad = max(
            keeper.max_command_j567_delta_rad,
            j567_delta,
        )
        overrides[f"arm_{arm}"] = candidate.astype(float).tolist()
        reports[arm] = {
            "ok": True,
            "measured_q_rad": measured.astype(float).tolist(),
            "command_q_rad": candidate.astype(float).tolist(),
            "command_tracking_error_rad": tracking_error,
            "q1234_delta_from_measured_rad": q1234_delta,
            "j567_delta_from_measured_rad": j567_delta,
            "predicted_position_error_m": position_error,
            "predicted_orientation_error_deg": orientation_target_error,
            "predicted_local_entry_orientation_error_deg": (
                local_entry_orientation_error
            ),
            "orientation_acceptance_limit_deg": float(orientation_limit_deg),
            "orientation_target": target_report,
            "accepted_best_effort_orientation": bool(
                orientation_target_error
                > RESET_BODY_KEEP_ORIENTATION_SOLVE_TOL_DEG
            ),
            "position_passes": position_reports,
            "orientation_solve": dict(solve_report or {}),
        }
    return overrides, {"ok": True, "arms": reports}



def _reset_body_keep_trunk_command(
    current_trunk: np.ndarray,
    target_trunk: np.ndarray,
    requested_max_step: float,
    progress_steps: int,
    max_position_error_m: float,
) -> tuple[np.ndarray, float, float]:
    """Choose one bounded body step while treating EEF position as secondary.

    The command is relative to the latest observed trunk state.  This avoids
    accumulating an open-loop smoothstep trajectory while the evaluator is
    rendering and stepping the robot.  Position drift only reduces the speed
    near the emergency limit; it does not freeze the body at the soft leash.
    """
    max_step = min(
        float(requested_max_step),
        RESET_BODY_KEEP_TRUNK_MAX_STEP_RAD,
    )
    ramp_scale = min(
        1.0,
        float(progress_steps + 1) / RESET_BODY_KEEP_TRUNK_RAMP_STEPS,
    )
    # Position is best-effort.  Slowing or reversing the trunk cannot recover
    # an EEF target that becomes unreachable at the requested upright pose.
    # Keep the body step independent while J1-J4 use bounded local correction.
    _ = max_position_error_m
    slowdown_scale = 1.0
    step_cap = max_step * ramp_scale * slowdown_scale
    current = np.asarray(current_trunk, dtype=np.float64).reshape(4)
    target = np.asarray(target_trunk, dtype=np.float64).reshape(4)
    command = current + np.clip(target - current, -step_cap, step_cap)
    return command, float(step_cap), float(slowdown_scale)



def _reset_body_measured_follow_overrides(
    ctx,
    keep_states: dict[str, _ResetBodyKeepOriState],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Fall back to zero-delta arm targets without affecting the trunk command."""
    overrides: dict[str, list[float]] = {}
    reports: dict[str, Any] = {}
    for arm, keeper in keep_states.items():
        source = "measured_qpos"
        try:
            command = locked_arm_vector(
                ctx.world.arm_qpos_list(arm),
                label=f"{arm} arm proprioception",
            )
        except Exception:  # keep-arm faults are isolated from body control
            command = keeper.last_command.copy()
            source = "last_valid_command"
        tracking_error = float(
            np.linalg.norm(keeper.last_command - command, ord=np.inf)
        )
        keeper.max_command_tracking_error_rad = max(
            keeper.max_command_tracking_error_rad,
            tracking_error,
        )
        keeper.last_command = command.copy()
        overrides[f"arm_{arm}"] = command.astype(float).tolist()
        reports[arm] = {
            "ok": False,
            "fallback": "software_compliant_measured_follow",
            "fallback_source": source,
            "command_q_rad": command.astype(float).tolist(),
            "command_tracking_error_rad": tracking_error,
            "q1234_delta_from_measured_rad": 0.0,
            "j567_delta_from_measured_rad": 0.0,
        }
    return overrides, reports



def _reset_body_best_effort_keep_overrides(
    ctx,
    trunk_command: Any,
    keep_states: dict[str, _ResetBodyKeepOriState],
    *,
    j567_max_step_rad: float = RESET_BODY_KEEP_J567_MAX_STEP_RAD,
) -> tuple[dict[str, list[float]], dict[str, Any], bool, bool]:
    """Plan the arm at the full trunk step and degrade without changing that step."""
    try:
        overrides, report = _reset_body_plan_keep_overrides(
            ctx,
            trunk_command,
            keep_states,
            orientation_limit_deg=(
                RESET_BODY_KEEP_ORIENTATION_BEST_EFFORT_LIMIT_DEG
            ),
            j567_max_step_rad=j567_max_step_rad,
        )
    except Exception as exc:  # keep-arm faults are isolated from body control
        overrides = {}
        report = {"ok": False, "error": str(exc), "arms": {}}

    if bool(report.get("ok", False)):
        nominal = all(
            float(item.get("predicted_orientation_error_deg", float("inf")))
            <= RESET_BODY_KEEP_ORIENTATION_PLAN_LIMIT_DEG
            for item in (report.get("arms") or {}).values()
        )
        report = dict(report)
        report["mode"] = "nominal" if nominal else "best_effort"
        report["trunk_command_reduced_for_arm"] = False
        return overrides, report, not nominal, False

    fallback_overrides, fallback_reports = _reset_body_measured_follow_overrides(
        ctx,
        keep_states,
    )
    return (
        fallback_overrides,
        {
            "ok": False,
            "mode": "measured_follow_fallback",
            "error": "keep-orientation planning unavailable; trunk continued",
            "planning_failure": report,
            "arms": fallback_reports,
            "trunk_command_reduced_for_arm": False,
        },
        True,
        True,
    )



def _reset_body_safe_observed_keep_report(
    ctx,
    keep_states: dict[str, _ResetBodyKeepOriState],
) -> dict[str, Any]:
    """Make arm observation failures diagnostic-only for body reset."""
    try:
        return _reset_body_observed_keep_report(ctx, keep_states)
    except Exception as exc:  # keep-arm faults are isolated from body control
        return {
            "ok": False,
            "within_hard_limits": False,
            "within_position_hard_limits": False,
            "within_orientation_hard_limits": False,
            "within_orientation_tracking_guard": False,
            "within_position_final_tolerances": False,
            "within_orientation_final_tolerances": False,
            "within_final_tolerances": False,
            "arms": {},
            "observation_error": str(exc),
        }



def _reset_body_body_priority_action(
    ctx,
    trunk_command: Any,
    arm_overrides: dict[str, list[float]],
) -> tuple[np.ndarray, str | None]:
    """Drop invalid best-effort arm overrides before allowing them to block body."""
    trunk = np.asarray(trunk_command, dtype=np.float64).reshape(4).tolist()
    try:
        return ctx.world.make_action(trunk=trunk, **arm_overrides), None
    except Exception as exc:  # retry without optional arm fields
        if not arm_overrides:
            raise
        return ctx.world.make_action(trunk=trunk), str(exc)



def _reset_body_observed_keep_report(
    ctx,
    keep_states: dict[str, _ResetBodyKeepOriState],
) -> dict[str, Any]:
    if not keep_states:
        return {
            "ok": True,
            "within_hard_limits": True,
            "within_position_hard_limits": True,
            "within_orientation_hard_limits": True,
            "within_orientation_tracking_guard": True,
            "within_position_final_tolerances": True,
            "within_orientation_final_tolerances": True,
            "within_final_tolerances": True,
            "arms": {},
        }
    state = _reset_body_local_state(ctx, ctx.world.trunk_qpos())
    measured_by_arm = {
        "left": state.arm_left_q,
        "right": state.arm_right_q,
    }
    reports: dict[str, Any] = {}
    within_hard_limits = True
    within_orientation_hard_limits = True
    within_orientation_tracking_guard = True
    within_position_hard_limits = True
    within_position_final_tolerances = True
    within_orientation_final_tolerances = True
    within_final_tolerances = True
    for arm, keeper in keep_states.items():
        measured = measured_by_arm[arm]
        position, quaternion = _local_eef_pose(state, arm, measured)
        local_position_error = float(
            np.linalg.norm(position - keeper.reference_position)
        )
        local_orientation_error = orientation_error_deg(
            quaternion,
            keeper.reference_quaternion,
        )
        raw_pose = _reset_body_raw_eef_pose(ctx, arm)
        raw_position_error: float | None = None
        raw_orientation_error: float | None = None
        if (
            raw_pose is not None
            and keeper.raw_reference_position is not None
            and keeper.raw_reference_quaternion is not None
        ):
            raw_position_error = float(
                np.linalg.norm(
                    raw_pose["position"] - keeper.raw_reference_position
                )
            )
            raw_orientation_error = orientation_error_deg(
                raw_pose["quaternion"],
                keeper.raw_reference_quaternion,
            )
        # Evaluator EEF pose is the physical observation and therefore wins
        # when present.  Local FK remains the deterministic IK model and audit
        # fallback; taking max(raw, local) can create an error that no single
        # target can satisfy when the two models have a small fixed offset.
        position_error = (
            local_position_error
            if raw_position_error is None
            else raw_position_error
        )
        orientation_error = (
            local_orientation_error
            if raw_orientation_error is None
            else raw_orientation_error
        )
        keeper.max_observed_position_error_m = max(
            keeper.max_observed_position_error_m,
            position_error,
        )
        keeper.max_observed_orientation_error_deg = max(
            keeper.max_observed_orientation_error_deg,
            orientation_error,
        )
        arm_within_position_hard = bool(
            position_error <= RESET_BODY_KEEP_POSITION_HARD_LIMIT_M
        )
        arm_within_orientation_hard = bool(
            orientation_error <= RESET_BODY_KEEP_ORIENTATION_HARD_LIMIT_DEG
        )
        arm_within_orientation_tracking_guard = bool(
            orientation_error
            <= RESET_BODY_KEEP_ORIENTATION_TRACKING_GUARD_DEG
        )
        arm_within_hard = bool(
            arm_within_position_hard and arm_within_orientation_hard
        )
        arm_within_position_final = bool(
            position_error <= RESET_BODY_KEEP_POSITION_FINAL_TOL_M
        )
        arm_within_orientation_final = bool(
            orientation_error <= RESET_BODY_KEEP_ORIENTATION_FINAL_TOL_DEG
        )
        arm_within_final = bool(
            arm_within_position_final and arm_within_orientation_final
        )
        within_hard_limits = within_hard_limits and arm_within_hard
        within_position_hard_limits = (
            within_position_hard_limits and arm_within_position_hard
        )
        within_orientation_hard_limits = (
            within_orientation_hard_limits and arm_within_orientation_hard
        )
        within_orientation_tracking_guard = (
            within_orientation_tracking_guard
            and arm_within_orientation_tracking_guard
        )
        within_position_final_tolerances = (
            within_position_final_tolerances and arm_within_position_final
        )
        within_orientation_final_tolerances = (
            within_orientation_final_tolerances and arm_within_orientation_final
        )
        within_final_tolerances = within_final_tolerances and arm_within_final
        raw_q = np.asarray(
            ctx.world.arm_qpos_list(arm),
            dtype=np.float64,
        ).reshape(-1)
        reports[arm] = {
            "within_hard_limits": arm_within_hard,
            "within_position_hard_limits": arm_within_position_hard,
            "within_orientation_hard_limits": arm_within_orientation_hard,
            "within_orientation_tracking_guard": (
                arm_within_orientation_tracking_guard
            ),
            "within_position_final_tolerance": arm_within_position_final,
            "within_orientation_final_tolerance": arm_within_orientation_final,
            "within_final_tolerances": arm_within_final,
            "measured_q_rad": raw_q.astype(float).tolist(),
            "local_position_robot_m": np.asarray(position, dtype=np.float64).tolist(),
            "local_quaternion_xyzw": np.asarray(quaternion, dtype=np.float64).tolist(),
            "local_position_error_m": local_position_error,
            "local_orientation_error_deg": local_orientation_error,
            "raw_evaluator_position_error_m": raw_position_error,
            "raw_evaluator_orientation_error_deg": raw_orientation_error,
            "effective_position_error_m": position_error,
            "effective_orientation_error_deg": orientation_error,
            "effective_pose_source": (
                "local_fk"
                if raw_orientation_error is None
                else "evaluator_eef_pose"
            ),
            "observed_j8_rad": (
                float(raw_q[7]) if ARM_DOF == 8 and raw_q.size >= 8 else None
            ),
        }
    return {
        "ok": bool(within_final_tolerances),
        "within_hard_limits": bool(within_hard_limits),
        "within_position_hard_limits": bool(within_position_hard_limits),
        "within_orientation_hard_limits": bool(within_orientation_hard_limits),
        "within_orientation_tracking_guard": bool(
            within_orientation_tracking_guard
        ),
        "within_position_final_tolerances": bool(
            within_position_final_tolerances
        ),
        "within_orientation_final_tolerances": bool(
            within_orientation_final_tolerances
        ),
        "within_final_tolerances": bool(within_final_tolerances),
        "arms": reports,
    }



def _reset_body_keep_result(
    keep_states: dict[str, _ResetBodyKeepOriState],
    observed_report: dict[str, Any],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    observed_arms = dict(observed_report.get("arms") or {})
    for arm, keeper in keep_states.items():
        raw_reference = None
        if (
            keeper.raw_reference_position is not None
            and keeper.raw_reference_quaternion is not None
        ):
            raw_reference = {
                "position": keeper.raw_reference_position.astype(float).tolist(),
                "quaternion_xyzw": (
                    keeper.raw_reference_quaternion.astype(float).tolist()
                ),
            }
        arms[arm] = {
            "reference_local_position_robot_m": (
                keeper.reference_position.astype(float).tolist()
            ),
            "reference_local_quaternion_xyzw": (
                keeper.reference_quaternion.astype(float).tolist()
            ),
            "reference_raw_evaluator_pose": raw_reference,
            "initial_q_rad": keeper.initial_q.astype(float).tolist(),
            "last_command_q_rad": keeper.last_command.astype(float).tolist(),
            "solve_calls": int(keeper.solve_calls),
            "max_observed_position_error_m": float(
                keeper.max_observed_position_error_m
            ),
            "max_observed_orientation_error_deg": float(
                keeper.max_observed_orientation_error_deg
            ),
            "max_command_tracking_error_rad": float(
                keeper.max_command_tracking_error_rad
            ),
            "max_command_q1234_delta_rad": float(
                keeper.max_command_q1234_delta_rad
            ),
            "max_command_j567_delta_rad": float(
                keeper.max_command_j567_delta_rad
            ),
            "last_orientation_solve": keeper.last_solver_report,
            "final_observation": observed_arms.get(arm),
            "j1234_policy": "measured_follow_with_best_effort_position_correction",
            "j567_policy": "fixed_candidate_j1234_local_urdf_orientation_tracking",
            "j8_policy": "locked_zero" if ARM_DOF == 8 else "not_present",
        }
    return {
        "enabled": bool(keep_states),
        "arms": arms,
        "controller": "body_first_nonblocking_keep_ori",
        "position_priority": "best_effort_advisory",
        "position_affects_body_speed": False,
        "position_deadband_m": RESET_BODY_KEEP_POSITION_DEADBAND_M,
        "position_warning_m": RESET_BODY_KEEP_POSITION_WARNING_M,
        "position_final_tolerance_m": RESET_BODY_KEEP_POSITION_FINAL_TOL_M,
        "position_hard_limit_m": RESET_BODY_KEEP_POSITION_HARD_LIMIT_M,
        "orientation_final_tolerance_deg": (
            RESET_BODY_KEEP_ORIENTATION_FINAL_TOL_DEG
        ),
        "orientation_planning_limit_deg": (
            RESET_BODY_KEEP_ORIENTATION_PLAN_LIMIT_DEG
        ),
        "orientation_tracking_guard_deg": (
            RESET_BODY_KEEP_ORIENTATION_TRACKING_GUARD_DEG
        ),
        "orientation_hard_limit_deg": RESET_BODY_KEEP_ORIENTATION_HARD_LIMIT_DEG,
        "orientation_hard_limit_behavior": "warning_only",
        "q1234_max_delta_from_measured_per_action_rad": (
            RESET_BODY_KEEP_Q1234_MAX_STEP_RAD
        ),
        "j567_max_delta_from_measured_per_action_rad": (
            RESET_BODY_KEEP_J567_CATCHUP_MAX_STEP_RAD
        ),
        "j567_nominal_max_delta_from_measured_per_action_rad": (
            RESET_BODY_KEEP_J567_MAX_STEP_RAD
        ),
        "j567_catchup_max_delta_from_measured_per_action_rad": (
            RESET_BODY_KEEP_J567_CATCHUP_MAX_STEP_RAD
        ),
        "stationary_recovery_actions_during_body_motion": 0,
        "software_compliance": bool(keep_states),
        "physical_impedance_available": False,
        "arm_quality_affects_trunk_motion": False,
        "arm_quality_affects_success": False,
        "trunk_step_backoff_for_arm": False,
        "reference_frame": "robot_base_from_evaluator_proprio_and_local_fk",
    }


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
