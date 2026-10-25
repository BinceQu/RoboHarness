from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any
import numpy as np
from tools.util.capabilities import TOOL_VERSION
from tools.util.contract import (
    ROBOT_PROFILE,
)
from tools.util.geometry import (
    finite_vector,
    json_ready,
    normalize_quaternion,
    optical_vector_to_world,
    pose_with_delta,
    quaternion_angle_deg,
    unproject_relative_uv,
)
from tools.util.storage import Snapshot, file_data_url
from tools.util.x5_kinematics import (
    X5_JOINT_LOWER,
    X5_JOINT_UPPER,
    calibrated_pose_for_joints,
    solve_calibrated_x5_ik,
)
from tools.util.x5_gripper_visualization import (
    _estimate_x5_gripper_joint_position_from_depth,
    render_x5_grasp_volume_three_views,
    render_x5_wrist_grasp_volume_overlay,
)
from tools.util.x5_grasp_geometry import ROBODOJO_X5_GRIPPER_SCALE_M

from tools.util.motion_constants import (
    GRIPPER_POSITION_TOLERANCE,
    GRIPPER_OPEN_POSITION_TOLERANCE,
    GRIPPER_OPEN_CONTROL_HZ,
    GRIPPER_OPEN_CONFIRMATION_STEPS,
    GRIPPER_OPEN_SETTLE_STEP,
    GRIPPER_OPEN_STALL_STEPS,
    GRIPPER_OPEN_MAX_ACTION_STEPS,
    GRIPPER_OPEN_DEPTH_TOLERANCE_M,
    GRIPPER_OPEN_DEPTH_CONFIRMATION_SAMPLES,
    GRIPPER_OPEN_DEPTH_RECHECK_STEPS,
    GRIPPER_CLOSE_CONTROL_HZ,
    GRIPPER_CLOSE_CONFIRMATION_STEPS,
    GRIPPER_CLOSE_SETTLE_WINDOW_STEPS,
    GRIPPER_CLOSE_SETTLE_SPAN,
    GRIPPER_CLOSE_BLOCKAGE_MAX_POSITION,
    GRIPPER_CLOSE_REBOUND_TOLERANCE,
    GRIPPER_CLOSE_COLLAPSE_TOLERANCE,
    GRIPPER_CLOSE_MAX_ACTION_STEPS,
    GRIPPER_CLOSE_EVIDENCE_BUILD,
    EEF_MAX_ACTION_STEPS,
    EEF_MAX_JOINT_DELTA_PER_ACTION,
    EEF_ADJUST_MAX_ACTION_STEPS,
    EEF_ADJUST_MAX_CARTESIAN_STEP_M,
    EEF_ADJUST_MAX_ORIENTATION_STEP_DEG,
    EEF_ADJUST_MAX_JOINT_DELTA_PER_ACTION,
    EEF_IK_ORIENTATION_WEIGHT_M_PER_RAD,
    EEF_JOINT_TRACKING_BLOCKED_RAD,
    CONTROL_STALL_LIMIT,
    EXEC_JOINT_SETTLE_TOLERANCE_RAD,
    EXEC_JOINT_TRACKING_BLOCKED_RAD,
    EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD,
    EXEC_FINAL_STABLE_STEPS,
    EXEC_FINAL_TRANSLATION_STEP_TOLERANCE_M,
    EXEC_FINAL_ORIENTATION_STEP_TOLERANCE_DEG,
    EXEC_LOAD_STABLE_STEPS,
    EXEC_LOAD_MIN_RESIDUAL_RAD,
    EXEC_LOAD_MAX_ROUNDS,
    TRACKING_CAMERA_ROLES,
    _EXIT_CAPTURE_HEAD,
    _EXIT_CAPTURE_LEFT_WRIST,
    _EXIT_CAPTURE_RIGHT_WRIST,
    _EXIT_CAPTURE_RESULT_ARM_WRIST,
    EXIT_CAPTURE_ROLE_POLICIES,
    _EXIT_CAPTURE_PROMOTED_FIELDS,
)


def _move_tracked_exports():
    # A session runs an immutable copy of these readable workspace files.
    from tools import move_tracked_point as implementation

    return {
        "run": implementation.run_move_tracked_point,
        "release_motion_retention": implementation._move_tracked_release_motion_retention,
        "prepare_execution_schedule": implementation._prepare_execution_schedule,
        "validate_loaded_plan": implementation._validate_loaded_plan,
    }


def _set_result(ctx, tool_name: str, payload: dict[str, Any]) -> None:
    ctx.set_result(
        json_ready(
            {
                **dict(payload),
                "tool": str(tool_name),
                "tool_version": TOOL_VERSION,
                "backend": "robodojo",
                "robot_profile": ROBOT_PROFILE,
                "direct_simulator_access": False,
            }
        )
    )


def _observation_sequence(ctx) -> int:
    return int(ctx.world._official_adapter.status()["sequence"])


def _quaternion_slerp(start: Any, target: Any, fraction: float) -> np.ndarray:
    start_q = normalize_quaternion(start)
    target_q = normalize_quaternion(target)
    amount = float(np.clip(fraction, 0.0, 1.0))
    dot = float(np.dot(start_q, target_q))
    if dot < 0.0:
        target_q = -target_q
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(start_q + amount * (target_q - start_q))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return normalize_quaternion(
        math.sin((1.0 - amount) * angle) / sine * start_q
        + math.sin(amount * angle) / sine * target_q
    )


def _bounded_eef_servo_target(
    current_pose: np.ndarray,
    terminal_pose: np.ndarray,
    *,
    max_cartesian_step_m: float | None,
    max_orientation_step_deg: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = finite_vector(current_pose, 7, "current EEF pose")
    terminal = finite_vector(terminal_pose, 7, "terminal EEF pose")
    position_delta = terminal[:3] - current[:3]
    position_distance = float(np.linalg.norm(position_delta))
    position_fraction = 1.0
    if max_cartesian_step_m is not None and position_distance > float(
        max_cartesian_step_m
    ):
        position_fraction = float(max_cartesian_step_m) / position_distance

    orientation_distance = quaternion_angle_deg(current[3:], terminal[3:])
    orientation_fraction = 1.0
    if max_orientation_step_deg is not None and orientation_distance > float(
        max_orientation_step_deg
    ):
        orientation_fraction = float(max_orientation_step_deg) / orientation_distance

    servo_target = np.concatenate(
        [
            current[:3] + position_fraction * position_delta,
            _quaternion_slerp(current[3:], terminal[3:], orientation_fraction),
        ]
    )
    return servo_target, {
        "servo_target_pose": servo_target.tolist(),
        "terminal_target_pose": terminal.tolist(),
        "terminal_position_error_before_step_m": position_distance,
        "terminal_orientation_error_before_step_deg": orientation_distance,
        "servo_translation_step_m": float(
            np.linalg.norm(servo_target[:3] - current[:3])
        ),
        "servo_orientation_step_deg": quaternion_angle_deg(
            current[3:], servo_target[3:]
        ),
        "cartesian_path_step_limited": bool(position_fraction < 1.0),
        "orientation_path_step_limited": bool(orientation_fraction < 1.0),
    }


def _eef_joint_command(
    ctx,
    *,
    arm: str,
    target_pose: np.ndarray,
    max_joint_delta_per_action: float = EEF_MAX_JOINT_DELTA_PER_ACTION,
):
    """Convert one observed-frame Cartesian target to a legal joint action."""
    joint_step_limit = float(max_joint_delta_per_action)
    if not math.isfinite(joint_step_limit) or joint_step_limit <= 0.0:
        raise ValueError("max_joint_delta_per_action must be positive and finite")
    current_q = np.asarray(ctx.world.arm_qpos_list(arm), dtype=np.float64)
    current_pose = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
    solution = solve_calibrated_x5_ik(
        current_q,
        current_pose,
        target_pose,
        orientation_weight_m_per_rad=EEF_IK_ORIENTATION_WEIGHT_M_PER_RAD,
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
            "policy-local X5 IK could not find a Cartesian-improving joint target"
        )

    full_delta = solution.joint_target - current_q
    largest_delta = float(np.max(np.abs(full_delta)))
    command_scale = 1.0
    if largest_delta > joint_step_limit:
        command_scale = joint_step_limit / largest_delta
    commanded_q = current_q + command_scale * full_delta
    predicted_pose = calibrated_pose_for_joints(
        current_q,
        current_pose,
        commanded_q,
    )
    action = ctx.world.make_joint_action(**{f"{arm}_arm": commanded_q.tolist()})
    return action, {
        "controller": "policy_local_x5_joint_cartesian_servo",
        "current_joint_position": current_q.tolist(),
        "solved_joint_position": solution.joint_target.tolist(),
        "commanded_joint_position": commanded_q.tolist(),
        "command_scale": command_scale,
        "max_joint_delta_per_action_rad": joint_step_limit,
        "largest_full_joint_delta_rad": largest_delta,
        "predicted_command_pose": predicted_pose.tolist(),
        "ik_converged": solution.converged,
        "ik_iterations": solution.iterations,
        "ik_position_error_m": solution.position_error_m,
        "ik_orientation_error_deg": solution.orientation_error_deg,
        "ik_initial_position_error_m": solution.initial_position_error_m,
        "ik_initial_orientation_error_deg": (solution.initial_orientation_error_deg),
        "calibrated_base_translation_m": (solution.calibration_translation_m.tolist()),
    }


class _GripperArmHold:
    """Hold immutable arm targets with bounded public-feedback load correction.

    ``make_joint_action`` defaults unspecified arms to the newest measured
    joints. Repeating that default during finger contact ratchets any external
    deflection into the next arm target. Capture public proprioception once;
    do not replan, use contact truth, or alter another tool's default behavior.
    A small stationary load residual may require a different motor command to
    maintain that same measured target. The correction does not follow the
    displaced arm or turn a blocked/unstable gripper into a detected grasp.
    """

    build = "gripper_fixed_entry_bounded_load_compensation_v2"

    def __init__(self, world):
        self.world = world
        self.targets = {
            side: np.asarray(world.arm_qpos_list(side), dtype=np.float64).copy()
            for side in ("left", "right")
        }
        self.entry_poses = {
            side: np.asarray(world.eef_pose_list(side), dtype=np.float64).copy()
            for side in self.targets
        }
        self.command_targets = {side: q.copy() for side, q in self.targets.items()}
        self.previous_joints = {side: q.copy() for side, q in self.targets.items()}
        self.previous_poses = {side: p.copy() for side, p in self.entry_poses.items()}
        self.stable_steps = dict.fromkeys(self.targets, 0)
        self.compensation_rounds = dict.fromkeys(self.targets, 0)
        self.compensation_trace: list[dict[str, Any]] = []
        self.max_joint_drift = dict.fromkeys(self.targets, 0.0)
        self.max_translation_drift = dict.fromkeys(self.targets, 0.0)
        self.observation_count = 0

    def action(self, arm, target):
        return self.world.make_joint_action(
            left_arm=self.command_targets["left"].tolist(),
            right_arm=self.command_targets["right"].tolist(),
            **{f"{arm}_gripper": [target]},
        )

    def observe(self):
        drift = {}
        for side, target in self.targets.items():
            joints = np.asarray(self.world.arm_qpos_list(side), dtype=np.float64)
            pose = np.asarray(self.world.eef_pose_list(side), dtype=np.float64)
            if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(pose)):
                raise ValueError(
                    "gripper arm hold requires finite public proprioception"
                )
            joint_error = float(np.max(np.abs(joints - target)))
            delta = pose[:3] - self.entry_poses[side][:3]
            translation = float(np.linalg.norm(delta))
            self.max_joint_drift[side] = max(self.max_joint_drift[side], joint_error)
            self.max_translation_drift[side] = max(
                self.max_translation_drift[side], translation
            )
            drift[side] = {
                "joint_error_inf_rad": joint_error,
                "eef_delta_from_entry_m": delta.tolist(),
            }
            stationary = bool(
                np.max(np.abs(joints - self.previous_joints[side]))
                <= EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD
                and np.linalg.norm(pose[:3] - self.previous_poses[side][:3])
                <= EXEC_FINAL_TRANSLATION_STEP_TOLERANCE_M
                and quaternion_angle_deg(pose[3:], self.previous_poses[side][3:])
                <= EXEC_FINAL_ORIENTATION_STEP_TOLERANCE_DEG
            )
            self.stable_steps[side] = self.stable_steps[side] + 1 if stationary else 0
            # Reuse existing tracking/stationarity bounds. Strong blockage is
            # not an invitation to increase force. Corrections are smaller
            # than exec compensation, limited in number and at joint limits.
            if (
                self.stable_steps[side] >= EXEC_LOAD_STABLE_STEPS
                and EXEC_LOAD_MIN_RESIDUAL_RAD
                <= joint_error
                < EXEC_JOINT_TRACKING_BLOCKED_RAD
                and self.compensation_rounds[side] < EXEC_LOAD_MAX_ROUNDS
            ):
                before = self.command_targets[side]
                increment = np.clip(
                    target - joints,
                    -EXEC_JOINT_SETTLE_TOLERANCE_RAD / 2.0,
                    EXEC_JOINT_SETTLE_TOLERANCE_RAD / 2.0,
                )
                command = np.clip(
                    before + increment,
                    np.maximum(
                        X5_JOINT_LOWER, target - EXEC_JOINT_TRACKING_BLOCKED_RAD
                    ),
                    np.minimum(
                        X5_JOINT_UPPER, target + EXEC_JOINT_TRACKING_BLOCKED_RAD
                    ),
                )
                if np.max(np.abs(command - before)) > 1e-12:
                    self.command_targets[side] = command
                    self.compensation_rounds[side] += 1
                    self.stable_steps[side] = 0
                    event = {
                        "arm": side,
                        "feedback_sample": self.observation_count + 1,
                        "round": self.compensation_rounds[side],
                        "reference_joint_target": target.tolist(),
                        "observed_joint_position": joints.tolist(),
                        "command_before": before.tolist(),
                        "command_after": command.tolist(),
                        "reference_target_changed": False,
                        "uses_public_proprioception_only": True,
                    }
                    self.compensation_trace.append(event)
                    drift[side]["scheduled_load_compensation"] = event
            self.previous_joints[side] = joints.copy()
            self.previous_poses[side] = pose.copy()
        self.observation_count += 1
        return drift

    def audit(self):
        return {
            "build": self.build,
            "target_source": "tool_entry_public_arm_joint_observation",
            "joint_targets": {side: q.tolist() for side, q in self.targets.items()},
            "motor_command_targets": {
                side: q.tolist() for side, q in self.command_targets.items()
            },
            "entry_eef_poses": {
                side: pose.tolist() for side, pose in self.entry_poses.items()
            },
            "max_observed_joint_drift_rad": dict(self.max_joint_drift),
            "max_observed_eef_translation_m": dict(self.max_translation_drift),
            "feedback_samples": self.observation_count,
            "targets_follow_measurement": False,
            "reference_targets_immutable": True,
            "bounded_load_compensation": {
                "source": "stationary_public_joint_and_eef_tracking_residual",
                "rounds_by_arm": dict(self.compensation_rounds),
                "max_rounds_per_arm": EXEC_LOAD_MAX_ROUNDS,
                "max_round_increment_rad": EXEC_JOINT_SETTLE_TOLERANCE_RAD / 2.0,
                "max_total_increment_rad": EXEC_JOINT_TRACKING_BLOCKED_RAD,
                "minimum_residual_rad": EXEC_LOAD_MIN_RESIDUAL_RAD,
                "blocked_residual_rad": EXEC_JOINT_TRACKING_BLOCKED_RAD,
                "required_stationary_observations": EXEC_LOAD_STABLE_STEPS,
                "trace": list(self.compensation_trace),
                "adds_gripper_actions_or_budget": False,
            },
            "cartesian_replan_used": False,
            "physical_contact_truth_used": False,
        }


def _gripper_open_depth_evidence(ctx, arm: str) -> dict[str, Any]:
    """Estimate finger opening from public wrist depth, never simulator state.

    ``*_ee_joint_state`` is RoboDojo's rate-limited command proxy, not a
    measured finger joint. A reliable visible partial opening vetoes success.
    Missing/occluded depth is reported explicitly, not fabricated as full open.
    """
    evidence: dict[str, Any] = {
        "available": False,
        "fully_open": False,
        "camera_role": f"{arm}_wrist",
        "observation_sequence": _observation_sequence(ctx),
        "full_open_joint_position_m": ROBODOJO_X5_GRIPPER_SCALE_M[1],
        "joint_position_tolerance_m": GRIPPER_OPEN_DEPTH_TOLERANCE_M,
        "simulator_state_used": False,
    }
    try:
        camera = ctx.world._official_adapter.camera(f"{arm}_wrist", require_depth=True)
        _, _, _, registration = _estimate_x5_gripper_joint_position_from_depth(
            scene_depth=camera["depth"],
            intrinsic_matrix=camera["intrinsic_matrix"],
            extrinsic_matrix=camera["extrinsic_matrix"],
            eef_pose=ctx.world.eef_pose(arm),
            gripper_qpos=ctx.world.gripper_qpos_list(arm),
        )
        evidence["registration"] = registration
        if int(camera["sequence"]) != evidence["observation_sequence"]:
            evidence["reason"] = "camera_observation_changed"
            return evidence
        position_m = float(registration["selected_joint_position_m"])
        evidence["available"] = bool(
            registration["applied"] and math.isfinite(position_m)
        )
        if evidence["available"]:
            evidence["estimated_joint_position_m"] = position_m
            evidence["fully_open"] = bool(
                abs(position_m - ROBODOJO_X5_GRIPPER_SCALE_M[1])
                <= GRIPPER_OPEN_DEPTH_TOLERANCE_M + 1e-9
            )
        else:
            evidence["reason"] = "wrist_depth_registration_not_reliable"
    except Exception as exc:
        evidence["reason"] = "wrist_depth_unavailable"
        evidence["detail"] = f"{type(exc).__name__}: {exc}"
    return evidence


def _drive_open_gripper_target(
    ctx,
    *,
    arm: str,
    timeout_s: float,
    arm_hold: _GripperArmHold,
):
    """Open and settle, without mistaking a command proxy for finger position."""
    started = time.monotonic()
    target = 1.0
    tolerance = GRIPPER_OPEN_POSITION_TOLERANCE
    best_error = math.inf
    stalled_steps = 0
    confirmation_steps = 0
    depth_confirmation_samples = 0
    reliable_depth_seen = False
    next_depth_step = 0
    depth_evidence: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = []
    observed = float(ctx.world.gripper_qpos_list(arm)[0])
    previous_observed = observed
    converged = False
    # Count evaluator control time, not HTTP/rendering latency. An idle policy
    # advances no physics: even an initial command proxy of 1 needs settling.
    action_budget = min(
        GRIPPER_OPEN_MAX_ACTION_STEPS,
        max(1, int(math.ceil(timeout_s * GRIPPER_OPEN_CONTROL_HZ))),
    )
    ctx.raise_if_cancelled(f"{arm} gripper open-and-settle")
    ctx.world.latch_gripper_command(arm, target)

    for action_step in range(1, action_budget + 1):
        ctx.raise_if_cancelled(f"{arm} gripper open-and-settle")
        sequence_before = _observation_sequence(ctx)
        yield arm_hold.action(arm, target)
        sequence_after = _observation_sequence(ctx)
        observed = float(ctx.world.gripper_qpos_list(arm)[0])
        error = abs(observed - target)
        entry: dict[str, Any] = {
            "action_step": action_step,
            "observation_sequence_before": sequence_before,
            "observation_sequence_after": sequence_after,
            "observed": observed,
            "error": error,
            "observed_proxy_step": abs(observed - previous_observed),
            "arm_hold_drift": arm_hold.observe(),
        }
        trace.append(entry)
        if sequence_after <= sequence_before:
            stop_reason = "observation_not_advanced"
            break
        if not math.isfinite(observed):
            stop_reason = "invalid_gripper_feedback"
            break

        if error <= tolerance:
            stalled_steps = 0
            if abs(previous_observed - target) <= tolerance and (
                abs(observed - previous_observed) <= GRIPPER_OPEN_SETTLE_STEP
            ):
                confirmation_steps += 1
            else:
                confirmation_steps = 0
                depth_confirmation_samples = 0
            if confirmation_steps >= GRIPPER_OPEN_CONFIRMATION_STEPS and (
                action_step >= next_depth_step
            ):
                depth_evidence = _gripper_open_depth_evidence(ctx, arm)
                entry["open_depth_evidence"] = depth_evidence
                if depth_evidence["available"]:
                    reliable_depth_seen = True
                    if depth_evidence["fully_open"]:
                        depth_confirmation_samples += 1
                        next_depth_step = action_step + 1
                    else:
                        depth_confirmation_samples = 0
                        next_depth_step = action_step + GRIPPER_OPEN_DEPTH_RECHECK_STEPS
                    converged = (
                        depth_confirmation_samples
                        >= GRIPPER_OPEN_DEPTH_CONFIRMATION_SAMPLES
                    )
                    stop_reason = "open_depth_confirmed"
                else:
                    depth_confirmation_samples = 0
                    next_depth_step = action_step + GRIPPER_OPEN_DEPTH_RECHECK_STEPS
                    # Do not let lost visibility erase a previous physical veto.
                    converged = not reliable_depth_seen
                    stop_reason = "open_command_settled_depth_unavailable"
        else:
            confirmation_steps = 0
            depth_confirmation_samples = 0
            if best_error - error > 1e-4:
                best_error = error
                stalled_steps = 0
            else:
                stalled_steps += 1
        entry["confirmation_step"] = confirmation_steps
        entry["depth_confirmation_samples"] = depth_confirmation_samples
        previous_observed = observed
        if converged:
            break
        if stalled_steps >= GRIPPER_OPEN_STALL_STEPS:
            stop_reason = "tracking_stalled"
            break
    else:
        stop_reason = (
            "open_depth_not_confirmed"
            if reliable_depth_seen
            else "open_control_budget_exhausted"
        )

    return {
        "converged": converged,
        "observed": observed,
        "error": abs(observed - target),
        "action_steps": len(trace),
        "elapsed_s": time.monotonic() - started,
        "control_elapsed_s": len(trace) / GRIPPER_OPEN_CONTROL_HZ,
        "action_budget": action_budget,
        "confirmation_steps": confirmation_steps,
        "required_confirmation_steps": GRIPPER_OPEN_CONFIRMATION_STEPS,
        "confirmation_control_s": confirmation_steps / GRIPPER_OPEN_CONTROL_HZ,
        "depth_confirmation_samples": depth_confirmation_samples,
        "physical_open_verified": bool(
            converged
            and depth_confirmation_samples >= GRIPPER_OPEN_DEPTH_CONFIRMATION_SAMPLES
        ),
        "depth_evidence": depth_evidence,
        "trace": trace,
        "stop_reason": stop_reason,
    }


def _close_stall_eligibility(
    initial: float, observed: float, tolerance: float
) -> dict[str, Any]:
    """Admit a wide stable stall only after substantial closing feedback.

    A fixed aperture ceiling rejects wide objects even when the command proxy
    has moved materially toward closed. Retain the earlier partial-aperture
    condition, and additionally accept progress at least as large as the
    existing target tolerance (and larger than the rebound noise allowance).
    This is control evidence, not physical finger motion or grasp/contact truth.
    """
    finite = math.isfinite(initial) and math.isfinite(observed)
    in_range = finite and 0.0 <= initial <= 1.0 and 0.0 <= observed <= 1.0
    minimum_progress = max(float(tolerance), GRIPPER_CLOSE_REBOUND_TOLERANCE)
    progress = initial - observed if finite else None
    legacy = bool(in_range and observed <= GRIPPER_CLOSE_BLOCKAGE_MAX_POSITION)
    progressed = bool(in_range and progress >= minimum_progress - 1e-9)
    return {
        "eligible": legacy or progressed,
        "criterion": (
            "legacy_partial_aperture"
            if legacy
            else "closing_progress_from_tool_entry"
            if progressed
            else "insufficient_closing_progress"
        ),
        "initial_proxy": initial,
        "observed_proxy": observed,
        "closing_proxy_progress": progress,
        "minimum_progress_for_wide_stall": minimum_progress,
        "progress_threshold_source": "existing_target_tolerance_and_rebound_allowance",
        "physical_contact_proven": False,
    }


def _drive_close_gripper_target(
    ctx,
    *,
    arm: str,
    timeout_s: float,
    arm_hold: _GripperArmHold,
    tolerance: float = GRIPPER_POSITION_TOLERANCE,
):
    """Close, settle, and preload an X5 gripper using legal proprioception only."""
    started = time.monotonic()
    target = 0.0
    observed = float(ctx.world.gripper_qpos_list(arm)[0])
    initial_observed = observed
    trace: list[dict[str, Any]] = []
    recent_observations: list[float] = []
    closure_candidate_reason: str | None = None
    closure_candidate_position: float | None = None
    confirmation_steps = 0
    rebound_resets = 0
    blockage_loss_resets = 0
    witnessed_nonlimit_blockage = False

    # One evaluator action advances one 25 Hz observation. RPC and rendering
    # latency must not shorten the physical contact window.
    requested_budget = int(math.ceil(timeout_s * GRIPPER_CLOSE_CONTROL_HZ))
    minimum_budget = (
        GRIPPER_CLOSE_SETTLE_WINDOW_STEPS + GRIPPER_CLOSE_CONFIRMATION_STEPS
    )
    action_budget = min(
        GRIPPER_CLOSE_MAX_ACTION_STEPS,
        max(minimum_budget, requested_budget),
    )

    # Latch before the first yielded action so every later arm command keeps
    # applying closing preload. Only open_gripper or reset clears this target.
    ctx.world.latch_gripper_command(arm, target)

    for action_step in range(1, action_budget + 1):
        ctx.raise_if_cancelled(f"{arm} gripper close-and-hold")
        sequence_before = _observation_sequence(ctx)
        yield arm_hold.action(arm, target)
        sequence_after = _observation_sequence(ctx)
        observed = float(ctx.world.gripper_qpos_list(arm)[0])
        error = abs(observed - target)
        entry: dict[str, Any] = {
            "action_step": action_step,
            "observation_sequence_before": sequence_before,
            "observation_sequence_after": sequence_after,
            "observed": observed,
            "error": error,
            "arm_hold_drift": arm_hold.observe(),
            "phase": (
                "confirm_preload"
                if closure_candidate_reason is not None
                else "seek_closure"
            ),
        }
        trace.append(entry)

        if sequence_after <= sequence_before:
            stop_reason = "observation_not_advanced"
            break

        if closure_candidate_reason is None:
            recent_observations.append(observed)
            if len(recent_observations) > GRIPPER_CLOSE_SETTLE_WINDOW_STEPS:
                del recent_observations[0]

            candidate_reason: str | None = None
            stall_evidence = _close_stall_eligibility(
                initial_observed, observed, tolerance
            )
            entry["stall_eligibility"] = stall_evidence
            if error <= tolerance:
                candidate_reason = "target_band"
            elif (
                len(recent_observations) == GRIPPER_CLOSE_SETTLE_WINDOW_STEPS
                and max(recent_observations) - min(recent_observations)
                <= GRIPPER_CLOSE_SETTLE_SPAN
                and stall_evidence["eligible"]
            ):
                # RoboDojo exposes a bounded X5 gripper command proxy rather
                # than per-finger contact truth. A repeatable non-limit stall
                # is closure evidence, but is never reported as a detected grasp.
                candidate_reason = "stable_nonlimit_blockage"

            if candidate_reason is not None:
                closure_candidate_reason = candidate_reason
                closure_candidate_position = observed
                confirmation_steps = 0
                if candidate_reason == "stable_nonlimit_blockage":
                    witnessed_nonlimit_blockage = True
                entry["closure_candidate_reason"] = candidate_reason
                entry["closure_candidate_position"] = observed
            continue

        assert closure_candidate_position is not None
        rebound = observed - closure_candidate_position
        entry["rebound_from_candidate"] = rebound
        if rebound > GRIPPER_CLOSE_REBOUND_TOLERANCE:
            entry["confirmation_reset"] = "opening_rebound"
            rebound_resets += 1
            closure_candidate_reason = None
            closure_candidate_position = None
            confirmation_steps = 0
            recent_observations = [observed]
            continue

        collapse = closure_candidate_position - observed
        entry["collapse_from_candidate"] = collapse
        if (
            closure_candidate_reason == "stable_nonlimit_blockage"
            and collapse > GRIPPER_CLOSE_COLLAPSE_TOLERANCE
        ):
            # A grasped object blocks the X5 fingers away from the lower
            # limit. If that blockage disappears while maximum closing preload
            # is still commanded, the object moved or escaped; treating the
            # resulting fully closed hand as a confirmed grasp is unsafe.
            entry["confirmation_reset"] = "nonlimit_blockage_collapse"
            blockage_loss_resets += 1
            closure_candidate_reason = None
            closure_candidate_position = None
            confirmation_steps = 0
            recent_observations = [observed]
            continue

        confirmation_steps += 1
        entry["confirmation_step"] = confirmation_steps
        if confirmation_steps >= GRIPPER_CLOSE_CONFIRMATION_STEPS:
            if (
                closure_candidate_reason == "target_band"
                and witnessed_nonlimit_blockage
                and blockage_loss_resets > 0
            ):
                return {
                    "converged": False,
                    "observed": observed,
                    "error": error,
                    "action_steps": action_step,
                    "elapsed_s": time.monotonic() - started,
                    "control_elapsed_s": (action_step / GRIPPER_CLOSE_CONTROL_HZ),
                    "trace": trace,
                    "stop_reason": "nonlimit_blockage_lost_to_target",
                    "closure_candidate_reason": closure_candidate_reason,
                    "confirmation_steps": confirmation_steps,
                    "required_confirmation_steps": (GRIPPER_CLOSE_CONFIRMATION_STEPS),
                    "confirmation_control_s": (
                        confirmation_steps / GRIPPER_CLOSE_CONTROL_HZ
                    ),
                    "rebound_resets": rebound_resets,
                    "blockage_loss_resets": blockage_loss_resets,
                    "initial_observed": initial_observed,
                    "action_budget": action_budget,
                }
            return {
                "converged": True,
                "observed": observed,
                "error": error,
                "action_steps": action_step,
                "elapsed_s": time.monotonic() - started,
                "control_elapsed_s": action_step / GRIPPER_CLOSE_CONTROL_HZ,
                "trace": trace,
                "stop_reason": "close_preload_confirmed",
                "closure_candidate_reason": closure_candidate_reason,
                "confirmation_steps": confirmation_steps,
                "required_confirmation_steps": (GRIPPER_CLOSE_CONFIRMATION_STEPS),
                "confirmation_control_s": (
                    confirmation_steps / GRIPPER_CLOSE_CONTROL_HZ
                ),
                "rebound_resets": rebound_resets,
                "blockage_loss_resets": blockage_loss_resets,
                "initial_observed": initial_observed,
                "action_budget": action_budget,
            }
    else:
        stop_reason = "close_control_budget_exhausted"

    return {
        "converged": False,
        "observed": observed,
        "error": abs(observed - target),
        "action_steps": len(trace),
        "elapsed_s": time.monotonic() - started,
        "control_elapsed_s": len(trace) / GRIPPER_CLOSE_CONTROL_HZ,
        "trace": trace,
        "stop_reason": stop_reason,
        "closure_candidate_reason": closure_candidate_reason,
        "confirmation_steps": confirmation_steps,
        "required_confirmation_steps": GRIPPER_CLOSE_CONFIRMATION_STEPS,
        "confirmation_control_s": confirmation_steps / GRIPPER_CLOSE_CONTROL_HZ,
        "rebound_resets": rebound_resets,
        "blockage_loss_resets": blockage_loss_resets,
        "initial_observed": initial_observed,
        "action_budget": action_budget,
    }


def _stationary_capture_confirmation(
    required: bool,
    stable_steps: int,
    required_steps: int = EXEC_FINAL_STABLE_STEPS,
) -> dict[str, Any]:
    """Evidence for a settled exit frame, not a renderer timestamp guarantee."""
    return {
        "build": "public_feedback_stationary_exit_v1",
        "required": bool(required),
        "satisfied": bool(required and stable_steps >= required_steps),
        "consecutive_fresh_observations": int(stable_steps),
        "required_observations": int(required_steps) if required else 0,
        "max_joint_step_rad": EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD,
        "max_translation_step_m": EXEC_FINAL_TRANSLATION_STEP_TOLERANCE_M,
        "max_orientation_step_deg": EXEC_FINAL_ORIENTATION_STEP_TOLERANCE_DEG,
        "counts_against_action_budget": True,
        "renderer_timestamp_verified": False,
        "camera_pose_rewritten": False,
        "gripper_transition_added": False,
    }


def _drive_eef_target(
    ctx,
    *,
    arm: str,
    target: Any,
    pos_tolerance: float,
    ori_tolerance_deg: float,
    max_steps: int,
    timeout_s: float,
    action_step_limit: int = EEF_MAX_ACTION_STEPS,
    max_cartesian_step_m: float | None = None,
    max_orientation_step_deg: float | None = None,
    max_joint_delta_per_action: float = EEF_MAX_JOINT_DELTA_PER_ACTION,
    require_stable_capture: bool = False,
):
    """Drive one EEF target using observation-calibrated X5 joint actions."""
    target_pose = finite_vector(target, 7, f"{arm} EEF target")
    started = time.monotonic()
    deadline = started + timeout_s
    action_limit = max(1, min(int(max_steps), int(action_step_limit)))
    joint_tracking_blocked_threshold = min(
        EEF_JOINT_TRACKING_BLOCKED_RAD,
        0.5 * float(max_joint_delta_per_action),
    )
    motion_limits = {
        "cartesian_path_limited": max_cartesian_step_m is not None,
        "max_cartesian_step_m": max_cartesian_step_m,
        "max_orientation_step_deg": max_orientation_step_deg,
        "max_joint_delta_per_action_rad": max_joint_delta_per_action,
        "joint_tracking_blocked_threshold_rad": (joint_tracking_blocked_threshold),
        "action_step_limit": action_limit,
    }
    best_score = math.inf
    stalled_steps = 0
    max_joint_tracking_error = 0.0
    trace: list[dict[str, Any]] = []
    stable_steps = 0
    settle_joint_target = None

    def confirmation():
        return _stationary_capture_confirmation(require_stable_capture, stable_steps)

    observed = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
    position_error = float(np.linalg.norm(observed[:3] - target_pose[:3]))
    orientation_error = quaternion_angle_deg(observed[3:], target_pose[3:])
    initially_at_target = bool(
        position_error <= pos_tolerance and orientation_error <= ori_tolerance_deg
    )
    if initially_at_target and require_stable_capture:
        settle_joint_target = np.asarray(ctx.world.arm_qpos_list(arm), dtype=np.float64)
    if initially_at_target and not require_stable_capture:
        return {
            "converged": True,
            "observed_pose": observed.tolist(),
            "position_error_m": position_error,
            "orientation_error_deg": orientation_error,
            "action_steps": 0,
            "elapsed_s": time.monotonic() - started,
            "trace": trace,
            "stop_reason": "already_at_target",
            "max_joint_tracking_error_rad": 0.0,
            "stationary_capture_confirmation": confirmation(),
            **motion_limits,
        }

    for action_step in range(1, action_limit + 1):
        ctx.raise_if_cancelled(f"{arm} EEF motion")
        sequence_before = _observation_sequence(ctx)
        joint_before_action = np.asarray(ctx.world.arm_qpos_list(arm), dtype=np.float64)
        settling = settle_joint_target is not None
        try:
            servo_target, servo_step = _bounded_eef_servo_target(
                observed,
                observed if settling else target_pose,
                max_cartesian_step_m=max_cartesian_step_m,
                max_orientation_step_deg=max_orientation_step_deg,
            )
            if settling:
                # Reuse the command that reached tolerance. Re-solving IK here
                # would keep moving the camera while supposedly settling it.
                delta = settle_joint_target - joint_before_action
                largest_delta = float(np.max(np.abs(delta)))
                scale = min(1.0, max_joint_delta_per_action / max(largest_delta, 1e-12))
                commanded = joint_before_action + scale * delta
                action = ctx.world.make_joint_action(
                    **{f"{arm}_arm": commanded.tolist()}
                )
                command = {
                    "controller": "policy_local_x5_joint_target_settle",
                    "current_joint_position": joint_before_action.tolist(),
                    "solved_joint_position": settle_joint_target.tolist(),
                    "commanded_joint_position": commanded.tolist(),
                    "command_scale": scale,
                    "largest_full_joint_delta_rad": largest_delta,
                    "predicted_command_pose": calibrated_pose_for_joints(
                        joint_before_action,
                        observed,
                        commanded,
                    ).tolist(),
                }
                servo_step["terminal_target_pose"] = target_pose.tolist()
                servo_step["terminal_position_error_before_step_m"] = position_error
                servo_step["terminal_orientation_error_before_step_deg"] = (
                    orientation_error
                )
            else:
                action, command = _eef_joint_command(
                    ctx,
                    arm=arm,
                    target_pose=servo_target,
                    max_joint_delta_per_action=max_joint_delta_per_action,
                )
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            stop_reason = f"ik_failed: {exc}"
            break
        observed_before_action = observed.copy()
        yield action
        sequence_after = _observation_sequence(ctx)
        observed = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
        observed_joint = np.asarray(ctx.world.arm_qpos_list(arm), dtype=np.float64)
        commanded_joint = np.asarray(
            command["commanded_joint_position"], dtype=np.float64
        )
        joint_tracking_error = np.abs(observed_joint - commanded_joint)
        step_max_joint_tracking_error = float(np.max(joint_tracking_error))
        max_joint_tracking_error = max(
            max_joint_tracking_error, step_max_joint_tracking_error
        )
        position_error = float(np.linalg.norm(observed[:3] - target_pose[:3]))
        orientation_error = quaternion_angle_deg(observed[3:], target_pose[3:])
        translation_step = float(
            np.linalg.norm(observed[:3] - observed_before_action[:3])
        )
        orientation_step = quaternion_angle_deg(
            observed_before_action[3:], observed[3:]
        )
        joint_step = float(np.max(np.abs(observed_joint - joint_before_action)))
        cartesian_ok = bool(
            position_error <= pos_tolerance and orientation_error <= ori_tolerance_deg
        )
        stationary = bool(
            joint_step <= EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD
            and translation_step <= EXEC_FINAL_TRANSLATION_STEP_TOLERANCE_M
            and orientation_step <= EXEC_FINAL_ORIENTATION_STEP_TOLERANCE_DEG
        )
        stable_steps = (
            stable_steps + 1
            if cartesian_ok and stationary and sequence_after > sequence_before
            else 0
        )
        trace.append(
            {
                "action_step": action_step,
                "observation_sequence_before": sequence_before,
                "observation_sequence_after": sequence_after,
                "position_error_m": position_error,
                "orientation_error_deg": orientation_error,
                "observed_translation_step_m": float(
                    np.linalg.norm(observed[:3] - observed_before_action[:3])
                ),
                "observed_orientation_step_deg": quaternion_angle_deg(
                    observed_before_action[3:], observed[3:]
                ),
                "observed_joint_position": observed_joint.tolist(),
                "joint_tracking_error_rad": joint_tracking_error.tolist(),
                "max_joint_tracking_error_rad": step_max_joint_tracking_error,
                "observed_joint_position_before": joint_before_action.tolist(),
                "observed_joint_step_inf_rad": joint_step,
                "final_stationary": stationary,
                "final_stable_steps": stable_steps,
                "capture_settling": settling,
                **servo_step,
                **command,
            }
        )
        if sequence_after <= sequence_before:
            stop_reason = "observation_not_advanced"
            break
        if cartesian_ok and (
            not require_stable_capture or stable_steps >= EXEC_FINAL_STABLE_STEPS
        ):
            return {
                "converged": True,
                "observed_pose": observed.tolist(),
                "position_error_m": position_error,
                "orientation_error_deg": orientation_error,
                "action_steps": action_step,
                "elapsed_s": time.monotonic() - started,
                "trace": trace,
                "stop_reason": "target_reached",
                "max_joint_tracking_error_rad": max_joint_tracking_error,
                "stationary_capture_confirmation": confirmation(),
                **motion_limits,
            }

        if require_stable_capture and cartesian_ok:
            if settle_joint_target is None:
                settle_joint_target = commanded_joint.copy()
            stalled_steps = 0
            if time.monotonic() >= deadline:
                stop_reason = "capture_settle_timeout"
                break
            continue
        settle_joint_target = None

        score = position_error / pos_tolerance + orientation_error / ori_tolerance_deg
        improvement = best_score - score
        if improvement > 1e-3:
            best_score = score
            stalled_steps = 0
        else:
            stalled_steps += 1
        if sequence_after <= sequence_before:
            stop_reason = "observation_not_advanced"
            break
        if stalled_steps >= CONTROL_STALL_LIMIT:
            stop_reason = (
                "joint_tracking_blocked"
                if step_max_joint_tracking_error > joint_tracking_blocked_threshold
                else "tracking_stalled"
            )
            break
        if time.monotonic() >= deadline:
            stop_reason = "timeout"
            break
    else:
        stop_reason = (
            "capture_settle_budget_exhausted"
            if require_stable_capture and settle_joint_target is not None
            else "max_action_steps"
        )

    observed = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
    return {
        "converged": False,
        "observed_pose": observed.tolist(),
        "position_error_m": float(np.linalg.norm(observed[:3] - target_pose[:3])),
        "orientation_error_deg": quaternion_angle_deg(observed[3:], target_pose[3:]),
        "action_steps": len(trace),
        "elapsed_s": time.monotonic() - started,
        "trace": trace,
        "stop_reason": stop_reason,
        "max_joint_tracking_error_rad": max_joint_tracking_error,
        "stationary_capture_confirmation": confirmation(),
        **motion_limits,
    }


def _robot_state(ctx) -> dict[str, Any]:
    return {
        "base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "base_qvel": [0.0, 0.0, 0.0],
        "base_available": False,
        "trunk_qpos": [0.0, 0.0, 0.0, 0.0],
        "trunk_qvel": [0.0, 0.0, 0.0, 0.0],
        "trunk_available": False,
        "arm_left_qpos": ctx.world.arm_qpos_list("left"),
        "arm_left_qvel": ctx.world.arm_qvel_list("left"),
        "eef_left": ctx.world.eef_pose("left"),
        "gripper_left_qpos": ctx.world.gripper_qpos_list("left"),
        "gripper_left_qvel": ctx.world.gripper_qvel_list("left"),
        "arm_right_qpos": ctx.world.arm_qpos_list("right"),
        "arm_right_qvel": ctx.world.arm_qvel_list("right"),
        "eef_right": ctx.world.eef_pose("right"),
        "gripper_right_qpos": ctx.world.gripper_qpos_list("right"),
        "gripper_right_qvel": ctx.world.gripper_qvel_list("right"),
        "motion_epoch": ctx.world.motion_epoch(),
        "episode_id": ctx.world.episode_id(),
        "frame": "robodojo_env_fixed_base",
    }


def _capture_payload(
    ctx,
    *,
    role: str,
    session_id: str,
    interaction_progress_steps: int = 0,
) -> dict[str, Any]:
    camera = ctx.world._official_adapter.camera(role)
    robot = _robot_state(ctx)
    snapshot = ctx.snapshots.save(
        session_id=str(session_id),
        role=role,
        camera=camera,
        robot=robot,
    )
    tracking_binding: dict[str, Any] | None = None
    tracking_binding_error: str | None = None
    if role in TRACKING_CAMERA_ROLES:
        try:
            tracking_binding = ctx.tracker.register_capture(snapshot)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            tracking_binding_error = f"{type(exc).__name__}: {exc}"
    rgb_path = snapshot.paths.get("rgb", snapshot.paths["rgb_array"])
    display_rgb_path = rgb_path
    instruction = ctx.world._official_adapter.instruction()
    omitted = [
        "segmentation",
        "object_pose",
        "scene_graph",
        "reward_state",
        "task_success_state",
        "simulator_mesh",
    ]
    overlay_key = "base_path_overlay" if role == "head" else "grasp_zone_overlay"
    overlay = {
        "ok": False,
        "code": "fixed_base",
        "error": "RoboDojo dual-X5 has no mobile-base path",
    }
    wrist_artifacts: dict[str, Any] = {}
    if role in ("left_wrist", "right_wrist"):
        side = role.removesuffix("_wrist")
        prefix = Path(rgb_path).with_suffix("")
        overlay_path = prefix.with_name(f"{prefix.name}.{role}.grasp_volume.png")
        red_mask_path = prefix.with_name(f"{prefix.name}.{role}.red_mask.npy")
        three_view_path = prefix.with_name(
            f"{prefix.name}.{role}.grasp_volume_three_views.png"
        )
        if (
            snapshot.depth is not None
            and snapshot.intrinsic_matrix is not None
            and snapshot.extrinsic_matrix is not None
        ):
            overlay = render_x5_wrist_grasp_volume_overlay(
                rgb=snapshot.rgb,
                depth_linear=snapshot.depth,
                intrinsic_matrix=snapshot.intrinsic_matrix,
                extrinsic_matrix=snapshot.extrinsic_matrix,
                eef_pose=robot[f"eef_{side}"],
                gripper_qpos=robot[f"gripper_{side}_qpos"],
                gripper_qvel=robot[f"gripper_{side}_qvel"],
                output_path=overlay_path,
                mask_path=red_mask_path,
            )
        else:
            overlay = {
                "ok": False,
                "code": "missing_wrist_rgbd_geometry",
                "error": (
                    "X5 grasp-volume overlay requires evaluator wrist depth and "
                    "camera matrices"
                ),
                "path": str(overlay_path),
                "mask_path": str(red_mask_path),
            }
        observed_gripper_qpos = robot[f"gripper_{side}_qpos"]
        geometry_gripper_qpos = overlay.get(
            "gripper_geometry_qpos_normalized",
            observed_gripper_qpos,
        )
        three_views = render_x5_grasp_volume_three_views(
            gripper_qpos=geometry_gripper_qpos,
            command_qpos=observed_gripper_qpos,
            output_path=three_view_path,
        )
        if bool(overlay.get("ok")):
            display_rgb_path = str(overlay_path)
        wrist_artifacts = {
            "red_mask_path": (str(red_mask_path) if bool(overlay.get("ok")) else None),
            "grasp_volume_three_views": (
                file_data_url(three_view_path) if bool(three_views.get("ok")) else None
            ),
            "grasp_volume_three_views_path": (
                str(three_view_path) if bool(three_views.get("ok")) else None
            ),
            "grasp_volume_three_views_metadata": three_views,
            "debug_images": {
                "wrist_grasp_volume_overlay": (
                    str(overlay_path) if bool(overlay.get("ok")) else None
                ),
                "x5_grasp_volume_three_views": (
                    str(three_view_path) if bool(three_views.get("ok")) else None
                ),
            },
        }
    result = {
        "ok": True,
        "session_id": snapshot.session_id,
        "feed": role,
        "image_id": snapshot.image_id,
        "image_width": int(snapshot.rgb.shape[1]),
        "image_height": int(snapshot.rgb.shape[0]),
        "rgb_main": file_data_url(display_rgb_path),
        "rgb_main_path": str(display_rgb_path),
        "rgb_path": rgb_path,
        "rgb_overlay_path": str(display_rgb_path),
        overlay_key: overlay,
        **wrist_artifacts,
        "camera": snapshot.camera_metadata(),
        "robot": robot,
        "modalities": [
            "rgb",
            *([] if snapshot.depth is None else ["depth"]),
            *([] if snapshot.intrinsic_matrix is None else ["intrinsic_matrix"]),
            *([] if snapshot.extrinsic_matrix is None else ["extrinsic_matrix"]),
            *(
                ["red_grasp_volume_mask"]
                if role != "head" and bool(overlay.get("ok"))
                else []
            ),
        ],
        "forbidden_modalities_omitted": omitted,
        "interaction_progress_action_steps": int(interaction_progress_steps),
        "memory": {
            "instruction": instruction,
            "image_id": snapshot.image_id,
            "observation_sequence": snapshot.sequence,
            "episode_id": snapshot.episode_id,
            **ctx.tracker.memory_fields(),
            "omitted": omitted,
        },
    }
    if role in TRACKING_CAMERA_ROLES:
        trackable = bool(
            snapshot.depth is not None
            and snapshot.intrinsic_matrix is not None
            and snapshot.extrinsic_matrix is not None
            and tracking_binding is not None
        )
        result["track_object_distance_binding"] = (
            {
                **dict(tracking_binding),
                "reason": "bound",
                "integrity_digest": snapshot.integrity_digest,
            }
            if tracking_binding is not None
            else {
                "ok": False,
                "trackable": False,
                "reason": (
                    "binding_failed"
                    if tracking_binding_error is not None
                    else "missing_rgbd_geometry"
                ),
                "session_id": snapshot.session_id,
                "image_id": snapshot.image_id,
                "observation_sequence": snapshot.sequence,
                "episode_id": snapshot.episode_id,
                "camera_role": role,
                "integrity_digest": snapshot.integrity_digest,
            }
        )
        if not trackable:
            result["track_object_distance_binding"]["error"] = (
                tracking_binding_error
                or f"{role} capture lacks evaluator RGB-D camera geometry"
            )
    if snapshot.depth is not None:
        result.update(
            {
                "depth_path": snapshot.paths["depth"],
                "depth_visual_path": snapshot.paths.get("depth_visual"),
                "depth": file_data_url(snapshot.paths.get("depth_visual", "")),
            }
        )
    return result


def _exit_capture_role(
    ctx,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> str:
    policy = EXIT_CAPTURE_ROLE_POLICIES[str(tool_name)]
    if policy in (
        _EXIT_CAPTURE_HEAD,
        _EXIT_CAPTURE_LEFT_WRIST,
        _EXIT_CAPTURE_RIGHT_WRIST,
    ):
        return policy

    candidates = [result.get("arm"), arguments.get("arm")]
    if policy == _EXIT_CAPTURE_RESULT_ARM_WRIST:
        plan_id = str(arguments.get("plan_id", "") or "").strip()
        session_id = str(arguments.get("session_id", "") or "").strip()
        if plan_id:
            try:
                plan = ctx.plans.load_any(plan_id, session_id or None)
            except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
                pass
            else:
                candidates.append(plan.get("active_arm"))
    for candidate in candidates:
        arm = str(candidate or "").strip().lower()
        if arm in ("left", "right"):
            return f"{arm}_wrist"
    return _EXIT_CAPTURE_RIGHT_WRIST


def attach_exit_capture(
    ctx,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    """Attach one evaluator-backed exit frame without emitting another action."""

    adapted = dict(result)
    adapted["exit_capture_attempted"] = True
    requested_role = _exit_capture_role(
        ctx,
        tool_name=tool_name,
        arguments=arguments,
        result=adapted,
    )
    capture_tool = (
        "capture_head_camera"
        if requested_role == "head"
        else f"capture_{requested_role}_camera"
    )
    has_existing_image = bool(
        adapted.get("feed") == requested_role
        and adapted.get("image_id")
        and (
            adapted.get("rgb_main")
            or adapted.get("rgb_main_path")
            or adapted.get("rgb_path")
        )
    )

    observation = dict(adapted) if has_existing_image else None
    capture_errors: list[str] = []
    selected_role = requested_role
    if observation is None:
        roles = [requested_role]
        if requested_role != "head":
            roles.append("head")
        for role in roles:
            try:
                payload = _capture_payload(
                    ctx,
                    role=role,
                    session_id=session_id,
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                capture_errors.append(f"{role}: {exc}")
                continue
            selected_role = role
            observation = json_ready(
                {
                    **payload,
                    "tool": (
                        "capture_head_camera"
                        if role == "head"
                        else f"capture_{role}_camera"
                    ),
                    "tool_version": TOOL_VERSION,
                    "backend": "robodojo",
                    "robot_profile": ROBOT_PROFILE,
                    "direct_simulator_access": False,
                }
            )
            break

    if observation is None:
        adapted["observation"] = None
        adapted["capture_error"] = "; ".join(capture_errors) or "exit capture failed"
        return json_ready(adapted)

    adapted["observation"] = observation
    adapted["exit_capture_skill"] = (
        "capture_head_camera"
        if selected_role == "head"
        else f"capture_{selected_role}_camera"
    )
    if selected_role != requested_role:
        adapted["exit_capture_fallback"] = {
            "requested_skill": capture_tool,
            "actual_skill": adapted["exit_capture_skill"],
            "errors": capture_errors,
        }
    adapted.pop("capture_error", None)
    input_image_id = str(adapted.get("image_id") or "").strip()
    output_image_id = str(observation.get("image_id") or "").strip()
    if input_image_id and output_image_id and input_image_id != output_image_id:
        adapted.setdefault("input_image_id", input_image_id)
    for key in _EXIT_CAPTURE_PROMOTED_FIELDS:
        if key in observation:
            adapted[key] = observation[key]
    return json_ready(adapted)


def _snapshot_geometry(snapshot: Snapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if snapshot.depth is None:
        raise ValueError("frozen capture has no depth")
    if snapshot.intrinsic_matrix is None or snapshot.extrinsic_matrix is None:
        raise ValueError("frozen capture has no camera matrices")
    return snapshot.depth, snapshot.intrinsic_matrix, snapshot.extrinsic_matrix


def _current_snapshot(ctx, snapshot: Snapshot) -> Snapshot:
    if str(snapshot.episode_id) != ctx.world.episode_id():
        raise ValueError("capture belongs to a different evaluator episode")
    return snapshot


def _point(snapshot: Snapshot, u: float, v: float, depth: float | None = None):
    depth_image, intrinsic, extrinsic = _snapshot_geometry(snapshot)
    return unproject_relative_uv(
        depth_image,
        intrinsic,
        extrinsic,
        u,
        v,
        depth_override=depth,
    )


def _capture(ctx, *, tool_name: str, role: str, session_id: str):
    session = str(session_id or "").strip()
    if not session:
        _set_result(ctx, tool_name, {"ok": False, "error": "session_id is required"})
        yield ctx.world.hold_action()
        return

    try:
        result = _capture_payload(
            ctx,
            role=role,
            session_id=session,
            interaction_progress_steps=0,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _set_result(ctx, tool_name, {"ok": False, "feed": role, "error": str(exc)})
        yield ctx.world.hold_action()
        return
    _set_result(ctx, tool_name, result)
    yield ctx.world.hold_action()


def _unsupported(ctx, tool_name: str, requested: dict[str, Any]):
    _set_result(
        ctx,
        tool_name,
        {
            "ok": False,
            "code": "unsupported_embodiment",
            "error": "RoboDojo dual-X5 has a fixed base and no articulated torso",
            "semantic_status": "unsupported",
            "requested": requested,
            "action_emitted": False,
        },
    )
    yield ctx.world.hold_action()


def _gripper(
    ctx,
    tool_name: str,
    arm: str,
    target: float,
    timeout_s: float,
    *,
    confirm_close_preload: bool = False,
):
    side = str(arm).strip().lower()
    if side not in ("left", "right"):
        _set_result(ctx, tool_name, {"ok": False, "error": "arm must be left or right"})
        yield ctx.world.hold_action()
        return
    if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0:
        _set_result(
            ctx, tool_name, {"ok": False, "error": "timeout_s must be positive"}
        )
        yield ctx.world.hold_action()
        return
    before = ctx.world.gripper_qpos_list(side)
    arm_hold = _GripperArmHold(ctx.world)
    if confirm_close_preload:
        tracking = yield from _drive_close_gripper_target(
            ctx,
            arm=side,
            timeout_s=float(timeout_s),
            arm_hold=arm_hold,
        )
    else:
        tracking = yield from _drive_open_gripper_target(
            ctx,
            arm=side,
            timeout_s=float(timeout_s),
            arm_hold=arm_hold,
        )
    after = ctx.world.gripper_qpos_list(side)
    error = abs(float(after[0]) - float(target))
    ok = bool(tracking["converged"])
    latched_target = ctx.world.gripper_command_latch(side)
    result = {
        "ok": ok,
        "error": (
            None
            if ok
            else (
                "observed X5 gripper did not establish a stable close-and-hold state"
                if confirm_close_preload
                else "X5 gripper opening did not pass settling/depth confirmation"
            )
        ),
        "arm": side,
        "gripper_qpos_before": before,
        "gripper_qpos_after": after,
        "commanded_gripper_action": [float(target)],
        "gripper_command_latched": bool(
            latched_target is not None and abs(latched_target - float(target)) <= 1e-9
        ),
        "persistent_gripper_target": latched_target,
        "control_mode": "normalized_position",
        "tracking_error": error,
        "tracking_tolerance": (
            GRIPPER_POSITION_TOLERANCE
            if confirm_close_preload
            else GRIPPER_OPEN_POSITION_TOLERANCE
        ),
        "action_steps": tracking["action_steps"],
        "elapsed_s": tracking["elapsed_s"],
        "stop_reason": tracking["stop_reason"],
        "tracking_trace": tracking["trace"],
        "arm_position_hold": arm_hold.audit(),
    }
    if confirm_close_preload:
        result.update(
            {
                "closure_candidate_reason": tracking["closure_candidate_reason"],
                "preload_confirmation_steps": tracking["confirmation_steps"],
                "required_preload_confirmation_steps": tracking[
                    "required_confirmation_steps"
                ],
                "preload_confirmation_control_s": tracking["confirmation_control_s"],
                "control_elapsed_s": tracking["control_elapsed_s"],
                "close_action_budget": tracking["action_budget"],
                "opening_rebound_resets": tracking["rebound_resets"],
                "blockage_loss_resets": tracking["blockage_loss_resets"],
                "closure_evidence_source": "legal_gripper_proprioception",
                "closure_evidence_build": GRIPPER_CLOSE_EVIDENCE_BUILD,
                "closure_evidence_scope": "command_proxy_progress_and_preload_stability_not_grasp_truth",
                "closure_stall_eligibility": _close_stall_eligibility(
                    float(before[0]),
                    float(after[0]),
                    GRIPPER_POSITION_TOLERANCE,
                ),
            }
        )
    else:
        result.update(
            {
                "opening_controller": "x5_open_settle_depth_v1",
                "gripper_feedback_source": "evaluator_rate_limited_command_proxy",
                "open_confirmation_steps": tracking["confirmation_steps"],
                "required_open_confirmation_steps": tracking[
                    "required_confirmation_steps"
                ],
                "open_confirmation_control_s": tracking["confirmation_control_s"],
                "control_elapsed_s": tracking["control_elapsed_s"],
                "open_action_budget": tracking["action_budget"],
                "physical_open_verified": tracking["physical_open_verified"],
                "open_depth_confirmation_samples": tracking[
                    "depth_confirmation_samples"
                ],
                "open_depth_evidence": tracking["depth_evidence"],
            }
        )
    _set_result(ctx, tool_name, result)
    yield ctx.world.hold_action()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _head_snapshot(ctx, arguments: dict[str, Any]) -> Snapshot:
    session_id = str(arguments.get("session_id", "") or "").strip()
    image_id = str(arguments.get("image_id", "") or "").strip()
    if image_id:
        if not session_id:
            session_id = ctx.snapshots.latest(role="head").session_id
        snapshot = _current_snapshot(ctx, ctx.snapshots.load(session_id, image_id))
        if snapshot.role != "head":
            raise ValueError("selected image_id is not a head-camera capture")
        return snapshot
    return _current_snapshot(
        ctx,
        ctx.snapshots.latest(role="head", session_id=session_id or None),
    )


def _select_arm(ctx, target: np.ndarray, requested: str = "any") -> str:
    arm = str(requested or "any").strip().lower()
    if arm in ("left", "right"):
        return arm
    if arm not in ("", "any"):
        raise ValueError("arm must be left, right, or any")
    distances = {
        side: float(
            np.linalg.norm(
                target - np.asarray(ctx.world.eef_pose_list(side)[:3], dtype=np.float64)
            )
        )
        for side in ("left", "right")
    }
    return min(distances, key=distances.get)


def _target_from_click(
    snapshot: Snapshot,
    u: Any,
    v: Any,
    depth: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth_override = None
    if depth not in (None, ""):
        depth_override = _finite_number(depth, "depth")
        if depth_override <= 0.0:
            raise ValueError("depth must be positive")
    return _point(
        snapshot,
        _finite_number(u, "u"),
        _finite_number(v, "v"),
        depth_override,
    )


def _eef_adjust(
    ctx,
    *,
    tool_name: str,
    arm: str,
    frame: str,
    forward: float | None,
    upward: float | None,
    leftward: float | None,
    x: float | None,
    y: float | None,
    z: float | None,
    roll: float,
    pitch: float,
    yaw: float,
    pos_tol: float,
    ori_tol_deg: float,
    max_steps: int,
    timeout_s: float,
):
    try:
        pos_tolerance = _finite_number(pos_tol, "pos_tol")
        ori_tolerance = _finite_number(ori_tol_deg, "ori_tol_deg")
        timeout = _finite_number(timeout_s, "timeout_s")
        if pos_tolerance <= 0.0 or ori_tolerance <= 0.0 or timeout <= 0.0:
            raise ValueError("tolerances and timeout_s must be positive")
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")

        base_family = any(value is not None for value in (x, y, z))
        camera_family = any(value is not None for value in (forward, upward, leftward))
        if frame == "head" and base_family and camera_family:
            raise ValueError("use either x/y/z or forward/leftward/upward, not both")
        if base_family:
            base_delta = np.array(
                [
                    _finite_number(0.0 if x is None else x, "x"),
                    _finite_number(0.0 if y is None else y, "y"),
                    _finite_number(0.0 if z is None else z, "z"),
                ],
                dtype=np.float64,
            )
            translation = ctx.world.base_rotation @ base_delta
            translation_frame = "robot_base"
        else:
            camera_role = "head" if frame == "head" else f"{arm}_wrist"
            camera = ctx.world._official_adapter.camera(camera_role)
            extrinsic = camera.get("extrinsic_matrix")
            if extrinsic is None:
                raise ValueError(
                    f"{camera_role} extrinsic_matrix is required for frame motion"
                )
            local_delta = np.array(
                [
                    -_finite_number(0.0 if leftward is None else leftward, "leftward"),
                    -_finite_number(0.0 if upward is None else upward, "upward"),
                    _finite_number(0.0 if forward is None else forward, "forward"),
                ],
                dtype=np.float64,
            )
            translation = optical_vector_to_world(extrinsic, local_delta)
            translation_frame = f"starting_{camera_role}_camera"
        if float(np.linalg.norm(translation)) > 0.75:
            raise ValueError("requested EEF translation exceeds 0.75m")

        start = np.asarray(ctx.world.eef_pose_list(arm), dtype=np.float64)
        target = np.asarray(
            pose_with_delta(start, translation, roll, pitch, yaw),
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        _set_result(ctx, tool_name, {"ok": False, "error": str(exc), "arm": arm})
        yield ctx.world.hold_action()
        return

    tracking = yield from _drive_eef_target(
        ctx,
        arm=arm,
        target=target,
        pos_tolerance=pos_tolerance,
        ori_tolerance_deg=ori_tolerance,
        max_steps=int(max_steps),
        timeout_s=timeout,
        action_step_limit=EEF_ADJUST_MAX_ACTION_STEPS,
        max_cartesian_step_m=EEF_ADJUST_MAX_CARTESIAN_STEP_M,
        max_orientation_step_deg=EEF_ADJUST_MAX_ORIENTATION_STEP_DEG,
        max_joint_delta_per_action=EEF_ADJUST_MAX_JOINT_DELTA_PER_ACTION,
        require_stable_capture=True,
    )
    observed = np.asarray(tracking["observed_pose"], dtype=np.float64)
    position_error = float(tracking["position_error_m"])
    orientation_error = float(tracking["orientation_error_deg"])
    ok = bool(tracking["converged"])
    _set_result(
        ctx,
        tool_name,
        {
            "ok": ok,
            "error": None if ok else "observed EEF pose did not reach tolerance",
            "arm": arm,
            "translation_frame": translation_frame,
            "translation_robodojo_env_m": translation.tolist(),
            "start_eef_pose": start.tolist(),
            "target_eef_pose": target.tolist(),
            "observed_eef_pose": observed.tolist(),
            "position_error_m": position_error,
            "orientation_error_deg": orientation_error,
            "position_tolerance_m": pos_tolerance,
            "orientation_tolerance_deg": ori_tolerance,
            "max_steps": int(max_steps),
            "timeout_s": timeout,
            "action_steps": tracking["action_steps"],
            "elapsed_s": tracking["elapsed_s"],
            "stop_reason": tracking["stop_reason"],
            "tracking_trace": tracking["trace"],
            "control_policy": "observation_feedback_bounded_cartesian_servo",
            "stationary_capture_confirmation": tracking[
                "stationary_capture_confirmation"
            ],
            "cartesian_path_limited": tracking["cartesian_path_limited"],
            "max_cartesian_step_m": tracking["max_cartesian_step_m"],
            "max_orientation_step_deg": tracking["max_orientation_step_deg"],
            "max_joint_delta_per_action_rad": tracking[
                "max_joint_delta_per_action_rad"
            ],
            "gripper_command_latch": ctx.world.gripper_command_latch(arm),
            "max_joint_tracking_error_rad": tracking["max_joint_tracking_error_rad"],
            "joint_tracking_blocked_threshold_rad": tracking[
                "joint_tracking_blocked_threshold_rad"
            ],
            "collision_checked": False,
        },
    )
    yield ctx.world.hold_action()


def _save_pose_plan(
    ctx,
    *,
    tool_name: str,
    snapshot: Snapshot,
    arm: str,
    waypoints: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "schema": "robodojo_official_v2_eef_plan",
        "schema_version": 1,
        "episode_id": ctx.world.episode_id(),
        "motion_epoch": ctx.world.motion_epoch(),
        "active_arm": arm,
        "source_image_id": snapshot.image_id,
        "source_episode_id": snapshot.episode_id,
        "source_capture_digest": snapshot.integrity_digest,
        "source_observation_sequence": snapshot.sequence,
        "start_eef_pose": {
            side: ctx.world.eef_pose_list(side) for side in ("left", "right")
        },
        "start_arm_qpos": {
            side: ctx.world.arm_qpos_list(side) for side in ("left", "right")
        },
        "waypoints": waypoints,
        "parameters": json_ready(parameters),
        "planner_basis": "frozen_evaluator_rgbd_and_observed_eef_state",
        "collision_checked": False,
        "simulator_state_used": False,
    }
    saved = ctx.plans.save(snapshot.session_id, record)
    return {
        "ok": True,
        "plan_id": saved["plan_id"],
        "plan_record_path": saved["plan_record_path"],
        "trajectory_schema": saved["schema"],
        "trajectory_schema_version": saved["schema_version"],
        "trajectory_digest": saved["integrity"]["digest"],
        "trajectory_waypoint_count": len(waypoints),
        "session_id": snapshot.session_id,
        "image_id": snapshot.image_id,
        "arm": arm,
        "tool": tool_name,
        "exec_sequence": "move_only",
        "collision_checked": False,
    }
