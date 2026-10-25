from __future__ import annotations

import time
from typing import Any
import numpy as np
from tools.util.contract import (
    ARM_DOF,
)
from tools.util.geometry import (
    finite_vector,
    quaternion_angle_deg,
)

from tools.util.motion_constants import (
    GRASP_PREP_DEFAULT_MAX_DQ_PER_STEP,
    GRASP_PREP_DEFAULT_TIMEOUT_S,
    GRASP_PREP_MAX_DQ_PER_STEP,
    GRASP_PREP_MAX_ACTION_STEPS,
    GRASP_PREP_STALL_LIMIT,
    GRASP_PREP_PROGRESS_EPS_RAD,
    GRASP_PREP_MIN_OBSERVED_MOTION_RAD,
    GRASP_PREP_RETRACE_ANCHOR_TOLERANCE_RAD,
    GRASP_PREP_RETRACE_WAYPOINT_TOLERANCE_RAD,
    GRASP_PREP_RETRACE_MAX_WAYPOINTS,
    GRASP_PREP_CAPTURE_BUILD,
    GRIPPER_POSITION_TOLERANCE,
    EEF_JOINT_TRACKING_BLOCKED_RAD,
    EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD,
    EXEC_FINAL_STABLE_STEPS,
    EXEC_FINAL_TRANSLATION_STEP_TOLERANCE_M,
    EXEC_FINAL_ORIENTATION_STEP_TOLERANCE_DEG,
    X5_GRASP_READY,
)
from tools.util.motion_common import (
    _finite_number,
    _observation_sequence,
    _set_result,
    _stationary_capture_confirmation,
)


def set_arm_to_grasp_position(
    ctx,
    arm: str = "right",
    keep_ori_arm: str = "none",
    gripper: str = "keep",
    max_dq_per_step: float = GRASP_PREP_DEFAULT_MAX_DQ_PER_STEP,
    tol: float = 0.08,
    timeout_s: float = GRASP_PREP_DEFAULT_TIMEOUT_S,
    **kwargs,
):
    try:
        requested = str(arm or "right").strip().lower()
        arms = ["left", "right"] if requested == "both" else [requested]
        if any(side not in ("left", "right") for side in arms):
            raise ValueError("arm must be left, right, or both")
        keep_ori = str(keep_ori_arm or "none").strip().lower()
        if keep_ori in ("", "false", "0", "no"):
            keep_ori = "none"
        if keep_ori not in ("none", "left", "right", "both"):
            raise ValueError("keep_ori_arm must be none, left, right, or both")
        requested_keep_arms = (
            {"left", "right"} if keep_ori == "both" else {keep_ori}
        ) - {"none"}
        invalid_keep_arms = requested_keep_arms - set(arms)
        if invalid_keep_arms:
            raise ValueError(
                f"keep_ori_arm={keep_ori} includes an arm outside arm={requested}"
            )
        if requested_keep_arms:
            raise ValueError(
                "keep_ori_arm is not available in the fixed joint-space "
                "official_v2 replacement; use none"
            )
        mode = "open" if kwargs.get("open_gripper") is True else str(gripper).lower()
        if mode not in ("keep", "open"):
            raise ValueError("gripper must be keep or open")
        max_step = _finite_number(max_dq_per_step, "max_dq_per_step")
        tolerance = _finite_number(tol, "tol")
        timeout = _finite_number(timeout_s, "timeout_s")
        if max_step <= 0.0 or tolerance <= 0.0 or timeout <= 0.0:
            raise ValueError("max_dq_per_step, tol, and timeout_s must be positive")
        effective_max_step = min(max_step, GRASP_PREP_MAX_DQ_PER_STEP)
        starts = {
            side: np.asarray(ctx.world.arm_qpos_list(side), dtype=np.float64)
            for side in arms
        }
        gripper_starts = {
            side: float(ctx.world.gripper_qpos_list(side)[0]) for side in arms
        }
    except (KeyError, TypeError, ValueError) as exc:
        _set_result(ctx, "set_arm_to_grasp_position", {"ok": False, "error": str(exc)})
        yield ctx.world.hold_action()
        return

    started = time.monotonic()
    observation_sequence_start = _observation_sequence(ctx)
    action_steps = 0
    achieved = {side: starts[side].copy() for side in arms}
    gripper_achieved = dict(gripper_starts)
    tracking_trace: list[dict[str, Any]] = []
    stage_reports: list[dict[str, Any]] = []
    retrace_by_arm: dict[str, list[dict[str, Any]]] = {}
    stage_queue: list[dict[str, Any]] = []

    retrace_provider = getattr(ctx.world, "arm_retrace_waypoints", None)
    for side in arms:
        history = (
            retrace_provider(
                side,
                X5_GRASP_READY[side],
                anchor_tolerance_rad=max(
                    tolerance,
                    GRASP_PREP_RETRACE_ANCHOR_TOLERANCE_RAD,
                ),
                max_waypoints=GRASP_PREP_RETRACE_MAX_WAYPOINTS,
            )
            if callable(retrace_provider)
            else []
        )
        retrace_by_arm[side] = history
        if (
            history
            and float(np.max(np.abs(starts[side] - X5_GRASP_READY[side]))) > tolerance
        ):
            for waypoint_index, entry in enumerate(history, start=1):
                stage_queue.append(
                    {
                        "kind": "observed_path_retrace",
                        "arm": side,
                        "waypoint_index": waypoint_index,
                        "waypoint_count": len(history),
                        "source_tool": entry.get("tool"),
                        "source_motion_epoch": entry.get("motion_epoch"),
                        "targets": {
                            side: finite_vector(
                                entry.get("qpos"),
                                ARM_DOF,
                                f"{side} retrace waypoint {waypoint_index}",
                            )
                        },
                        "tolerance": min(
                            tolerance,
                            GRASP_PREP_RETRACE_WAYPOINT_TOLERANCE_RAD,
                        ),
                    }
                )
    stage_queue.append(
        {
            "kind": "final_neutral_servo",
            "arm": "both" if len(arms) == 2 else arms[0],
            "waypoint_index": 1,
            "waypoint_count": 1,
            "source_tool": None,
            "source_motion_epoch": None,
            "targets": {side: X5_GRASP_READY[side].copy() for side in arms},
            "tolerance": tolerance,
        }
    )

    stop_reason = "already_at_target"
    failed_stage: dict[str, Any] | None = None
    terminal_stall_counts = {side: np.zeros(ARM_DOF, dtype=np.int64) for side in arms}
    max_command_tracking_error = 0.0
    final_stable_steps = 0

    for stage_index, stage in enumerate(stage_queue, start=1):
        stage_targets: dict[str, np.ndarray] = stage["targets"]
        stage_tolerance = float(stage["tolerance"])
        stall_counts = {
            side: np.zeros(ARM_DOF, dtype=np.int64) for side in stage_targets
        }
        previous_abs_errors = {
            side: np.abs(target - achieved[side])
            for side, target in stage_targets.items()
        }
        gripper_stall_steps = 0
        previous_gripper_error = (
            max(abs(1.0 - gripper_achieved[side]) for side in arms)
            if mode == "open"
            else 0.0
        )
        stage_action_start = action_steps

        while True:
            abs_errors = {
                side: np.abs(target - achieved[side])
                for side, target in stage_targets.items()
            }
            stage_arm_error = max(float(np.max(error)) for error in abs_errors.values())
            max_gripper_error = (
                max(abs(1.0 - gripper_achieved[side]) for side in arms)
                if mode == "open"
                else 0.0
            )
            if (
                stage_arm_error <= stage_tolerance
                and max_gripper_error <= GRIPPER_POSITION_TOLERANCE
                and (
                    stage["kind"] != "final_neutral_servo"
                    or final_stable_steps >= EXEC_FINAL_STABLE_STEPS
                )
            ):
                stage_reports.append(
                    {
                        "stage_index": stage_index,
                        "kind": stage["kind"],
                        "arm": stage["arm"],
                        "waypoint_index": stage["waypoint_index"],
                        "waypoint_count": stage["waypoint_count"],
                        "source_tool": stage["source_tool"],
                        "source_motion_epoch": stage["source_motion_epoch"],
                        "action_steps": action_steps - stage_action_start,
                        "max_abs_error_rad": stage_arm_error,
                        "tolerance_rad": stage_tolerance,
                        "stop_reason": (
                            "already_at_waypoint"
                            if action_steps == stage_action_start
                            else "waypoint_reached"
                        ),
                    }
                )
                if stage["kind"] == "final_neutral_servo":
                    stop_reason = (
                        "already_at_target" if action_steps == 0 else "target_reached"
                    )
                break
            if time.monotonic() - started >= timeout:
                stop_reason = (
                    "capture_settle_timeout"
                    if stage["kind"] == "final_neutral_servo"
                    and stage_arm_error <= stage_tolerance
                    else "timeout"
                )
                failed_stage = stage
                break
            if action_steps >= GRASP_PREP_MAX_ACTION_STEPS:
                stop_reason = (
                    "capture_settle_budget_exhausted"
                    if stage["kind"] == "final_neutral_servo"
                    and stage_arm_error <= stage_tolerance
                    else "max_action_steps"
                )
                failed_stage = stage
                break

            ctx.raise_if_cancelled("grasp preparation")
            sequence_before = _observation_sequence(ctx)
            action_args: dict[str, Any] = {}
            commanded: dict[str, np.ndarray] = {}
            observed_before = {side: achieved[side].copy() for side in arms}
            poses_before = {
                side: np.asarray(ctx.world.eef_pose_list(side), dtype=np.float64)
                for side in stage_targets
            }
            for side, target in stage_targets.items():
                command = achieved[side] + np.clip(
                    target - achieved[side],
                    -effective_max_step,
                    effective_max_step,
                )
                commanded[side] = command
                action_args[f"{side}_arm"] = command.tolist()
            if mode == "open":
                for side in arms:
                    action_args[f"{side}_gripper"] = [1.0]

            yield ctx.world.make_joint_action(**action_args)
            action_steps += 1
            sequence_after = _observation_sequence(ctx)
            achieved = {
                side: np.asarray(ctx.world.arm_qpos_list(side), dtype=np.float64)
                for side in arms
            }
            gripper_achieved = {
                side: float(ctx.world.gripper_qpos_list(side)[0]) for side in arms
            }
            current_abs_errors = {
                side: np.abs(target - achieved[side])
                for side, target in stage_targets.items()
            }
            command_tracking_error = {
                side: np.abs(commanded[side] - achieved[side]) for side in stage_targets
            }
            step_tracking_error = max(
                float(np.max(error)) for error in command_tracking_error.values()
            )
            max_command_tracking_error = max(
                max_command_tracking_error, step_tracking_error
            )
            observed_motion = {
                side: np.abs(achieved[side] - observed_before[side])
                for side in stage_targets
            }
            poses_after = {
                side: np.asarray(ctx.world.eef_pose_list(side), dtype=np.float64)
                for side in stage_targets
            }
            translation_steps = {
                side: float(
                    np.linalg.norm(poses_after[side][:3] - poses_before[side][:3])
                )
                for side in stage_targets
            }
            orientation_steps = {
                side: quaternion_angle_deg(
                    poses_after[side][3:], poses_before[side][3:]
                )
                for side in stage_targets
            }
            for side in stage_targets:
                for joint_index in range(ARM_DOF):
                    if current_abs_errors[side][joint_index] <= stage_tolerance:
                        stall_counts[side][joint_index] = 0
                        continue
                    improvement = (
                        previous_abs_errors[side][joint_index]
                        - current_abs_errors[side][joint_index]
                    )
                    moved = observed_motion[side][joint_index]
                    if improvement > GRASP_PREP_PROGRESS_EPS_RAD or (
                        improvement > 0.0 and moved > GRASP_PREP_MIN_OBSERVED_MOTION_RAD
                    ):
                        stall_counts[side][joint_index] = 0
                    else:
                        stall_counts[side][joint_index] += 1
            current_gripper_error = (
                max(abs(1.0 - gripper_achieved[side]) for side in arms)
                if mode == "open"
                else 0.0
            )
            if current_gripper_error <= GRIPPER_POSITION_TOLERANCE:
                gripper_stall_steps = 0
            elif (
                previous_gripper_error - current_gripper_error
                > GRASP_PREP_PROGRESS_EPS_RAD
            ):
                gripper_stall_steps = 0
            else:
                gripper_stall_steps += 1

            target_error_after = max(
                float(np.max(error)) for error in current_abs_errors.values()
            )
            # The public depth can reflect the preceding action while the
            # same observation already contains current joints. Do not rewrite
            # either input or read historical state in the planner. As in exec,
            # finish the normal servo only after fresh stationary feedback.
            stationary = all(
                float(np.max(observed_motion[side]))
                <= EXEC_FINAL_POSITION_STEP_TOLERANCE_RAD
                and translation_steps[side] <= EXEC_FINAL_TRANSLATION_STEP_TOLERANCE_M
                and orientation_steps[side] <= EXEC_FINAL_ORIENTATION_STEP_TOLERANCE_DEG
                for side in stage_targets
            )
            final_stable_steps = (
                final_stable_steps + 1
                if stage["kind"] == "final_neutral_servo"
                and target_error_after <= stage_tolerance
                and current_gripper_error <= GRIPPER_POSITION_TOLERANCE
                and stationary
                and sequence_after > sequence_before
                else 0
            )
            tracking_trace.append(
                {
                    "action_step": action_steps,
                    "control_stage": stage["kind"],
                    "stage_index": stage_index,
                    "retrace_arm": (
                        stage["arm"]
                        if stage["kind"] == "observed_path_retrace"
                        else None
                    ),
                    "retrace_waypoint_index": stage["waypoint_index"],
                    "retrace_waypoint_count": stage["waypoint_count"],
                    "source_tool": stage["source_tool"],
                    "source_motion_epoch": stage["source_motion_epoch"],
                    "observation_sequence_before": sequence_before,
                    "observation_sequence_after": sequence_after,
                    "commanded_qpos": {
                        side: commanded[side].tolist() for side in stage_targets
                    },
                    "observed_qpos": {side: achieved[side].tolist() for side in arms},
                    "observed_joint_motion_rad": {
                        side: observed_motion[side].tolist() for side in stage_targets
                    },
                    "joint_tracking_error_rad": {
                        side: command_tracking_error[side].tolist()
                        for side in stage_targets
                    },
                    "joint_stall_steps": {
                        side: stall_counts[side].astype(int).tolist()
                        for side in stage_targets
                    },
                    "max_joint_tracking_error_rad": step_tracking_error,
                    "max_abs_target_error_rad": target_error_after,
                    "target_tolerance_rad": stage_tolerance,
                    "max_gripper_target_error": current_gripper_error,
                    "observed_translation_step_m": translation_steps,
                    "observed_orientation_step_deg": orientation_steps,
                    "final_stationary": stationary,
                    "final_stable_steps": final_stable_steps,
                }
            )
            previous_abs_errors = current_abs_errors
            previous_gripper_error = current_gripper_error

            if sequence_after <= sequence_before:
                stop_reason = "observation_not_advanced"
                failed_stage = stage
                break
            unsatisfied = [
                (side, joint_index)
                for side in stage_targets
                for joint_index in range(ARM_DOF)
                if current_abs_errors[side][joint_index] > stage_tolerance
            ]
            arm_tracking_stalled = bool(
                unsatisfied
                and all(
                    stall_counts[side][joint_index] >= GRASP_PREP_STALL_LIMIT
                    for side, joint_index in unsatisfied
                )
            )
            if arm_tracking_stalled or (
                not unsatisfied and gripper_stall_steps >= GRASP_PREP_STALL_LIMIT
            ):
                stop_reason = (
                    "joint_tracking_blocked"
                    if step_tracking_error > EEF_JOINT_TRACKING_BLOCKED_RAD
                    else "tracking_stalled"
                )
                failed_stage = stage
                terminal_stall_counts.update(stall_counts)
                break

        if failed_stage is not None:
            break

    errors = {side: (X5_GRASP_READY[side] - achieved[side]).tolist() for side in arms}
    max_error = max(
        float(np.max(np.abs(X5_GRASP_READY[side] - achieved[side]))) for side in arms
    )
    max_gripper_error = (
        max(abs(1.0 - gripper_achieved[side]) for side in arms)
        if mode == "open"
        else 0.0
    )
    converged = (
        max_error <= tolerance
        and max_gripper_error <= GRIPPER_POSITION_TOLERANCE
        and failed_stage is None
        and final_stable_steps >= EXEC_FINAL_STABLE_STEPS
    )
    if converged:
        if mode == "open":
            for side in arms:
                ctx.world.latch_gripper_command(side, 1.0)
        clear_history = getattr(ctx.world, "clear_arm_motion_history", None)
        if callable(clear_history):
            clear_history(arms)

    stalled_joints = {
        side: [
            {
                "index": joint_index,
                "name": f"J{joint_index + 1}",
                "qpos_rad": float(achieved[side][joint_index]),
                "target_rad": float(X5_GRASP_READY[side][joint_index]),
                "error_rad": float(errors[side][joint_index]),
                "stall_steps": int(
                    terminal_stall_counts.get(side, np.zeros(ARM_DOF, dtype=np.int64))[
                        joint_index
                    ]
                ),
            }
            for joint_index in range(ARM_DOF)
            if abs(float(errors[side][joint_index])) > tolerance
        ]
        for side in arms
    }
    failure_stage_name = None if failed_stage is None else failed_stage["kind"]
    failure_detail = "; ".join(
        f"{side}:" + ",".join(item["name"] for item in entries)
        for side, entries in stalled_joints.items()
        if entries
    )
    error_message = None
    if not converged:
        error_message = (
            "arm reached tolerance but fresh stationary exit was not confirmed"
            if max_error <= tolerance
            and max_gripper_error <= GRIPPER_POSITION_TOLERANCE
            else "observed arm state did not reach tolerance"
        )
        if failure_stage_name:
            error_message += f" during {failure_stage_name}"
        if failure_detail:
            error_message += f"; residual joints {failure_detail}"
    _set_result(
        ctx,
        "set_arm_to_grasp_position",
        {
            "ok": converged,
            "error": error_message,
            "arms": arms,
            "target_qpos": {side: X5_GRASP_READY[side].tolist() for side in arms},
            "start_qpos": {side: starts[side].tolist() for side in arms},
            "achieved_qpos": {side: achieved[side].tolist() for side in arms},
            "error_rad": errors,
            "max_abs_error_rad": max_error,
            "tolerance_rad": tolerance,
            "gripper_qpos_before": gripper_starts,
            "gripper_qpos_after": gripper_achieved,
            "max_gripper_error": max_gripper_error,
            "gripper_tolerance": GRIPPER_POSITION_TOLERANCE,
            "requested_max_dq_per_step": max_step,
            "effective_max_dq_per_step": effective_max_step,
            "action_steps": action_steps,
            "capture_exit_build": GRASP_PREP_CAPTURE_BUILD,
            "stationary_capture_confirmation": _stationary_capture_confirmation(
                True,
                final_stable_steps,
            ),
            "elapsed_s": time.monotonic() - started,
            "stop_reason": stop_reason,
            "failure_stage": failure_stage_name,
            "stalled_joints_by_arm": stalled_joints,
            "stall_limit": GRASP_PREP_STALL_LIMIT,
            "max_joint_tracking_error_rad": max_command_tracking_error,
            "observation_sequence_start": observation_sequence_start,
            "observation_sequence_end": _observation_sequence(ctx),
            "retrace_used": any(
                report["kind"] == "observed_path_retrace" for report in stage_reports
            ),
            "retrace_waypoints_available_by_arm": {
                side: len(retrace_by_arm[side]) for side in arms
            },
            "retrace_waypoints_completed": sum(
                report["kind"] == "observed_path_retrace" for report in stage_reports
            ),
            "retrace_source": (
                "policy_recorded_evaluator_observed_qpos"
                if any(retrace_by_arm.values())
                else None
            ),
            "control_stages": stage_reports,
            "tracking_trace": tracking_trace,
            "final_tracking_trace": tracking_trace,
            "keep_ori_arm": keep_ori,
            "gripper": mode,
            "command_policy": (
                "observed_path_retrace_then_per_joint_bounded_neutral_servo"
            ),
            "target_semantics": "official_scene_initial_joint_qpos",
            "retrace_uses_only_policy_action_observation_history": True,
            "collision_checked": False,
            "direct_simulator_mutation": False,
        },
    )
    yield ctx.world.hold_action()


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("set_arm_to_grasp_position", set_arm_to_grasp_position)
