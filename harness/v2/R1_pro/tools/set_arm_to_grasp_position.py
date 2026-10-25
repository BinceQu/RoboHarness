"""set_arm_to_grasp_position

将选定手臂闭环移动到固定 grasp-prep。每条 action 都由最新 proprio 的 qpos/qvel 节流；成功需位置和速度连续稳定；仍有实际进展时 timeout_s 最多续期两次，失败则锁住最后实测姿态。J8 始终锁定 0rad；当前 official_v2 固定关节替代仅支持 keep_ori_arm=none。完成或失败后都拍 head camera。

How this tool works:
Read the chosen arm, fingers and torso. Restore a supported pre-grasp posture through joint interpolation, coordinate the requested finger state, and check measured convergence. Do not request open for an arm carrying an object.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.action_constants import CONTROL_HZ
from tools.util.action_constants import LOCKED_J8_VALUE_RAD
from tools.util.action_constants import _ARM_LIMITS
from tools.util.geometry.contract import ARM_DOF
from tools.util.geometry.contract import PROPRIO_DIM
from tools.util.geometry.contract import PROPRIO_SLICES
from tools.util.motion_common import _check_cancelled
from tools.util.motion_common import _reach_observation_sequence
from tools.util.motion_common import _set_result
from typing import Any
import math
import numpy as np
import time
from tools.util.tool import main

NAME = 'set_arm_to_grasp_position'
GRASP_PREP_DEFAULT_MAX_DQ_PER_STEP = 0.30
GRASP_PREP_DEFAULT_TIMEOUT_S = 15.0
SCHEMA = {'type': 'object',
 'properties': {'arm': {'type': 'string', 'enum': ['right', 'left', 'both'], 'default': 'right'},
                'keep_ori_arm': {'type': 'string',
                                 'enum': ['none', 'left', 'right', 'both'],
                                 'default': 'none'},
                'gripper': {'type': 'string', 'enum': ['open', 'keep'], 'default': 'keep'},
                'max_dq_per_step': {'type': 'number', 'default': 0.3, 'description': 'rad/frame'},
                'tol': {'type': 'number', 'default': 0.08, 'description': 'rad'},
                'timeout_s': {'type': 'number', 'default': 15.0, 'description': 's'}},
 'additionalProperties': False}


def actions(
    robot,
    arm: str = "right",
    keep_ori_arm: str = "none",
    gripper: str = "keep",
    max_dq_per_step: float = GRASP_PREP_DEFAULT_MAX_DQ_PER_STEP,
    tol: float = 0.08,
    timeout_s: float = GRASP_PREP_DEFAULT_TIMEOUT_S,
    **kwargs,
):
    function_started_monotonic = time.monotonic()
    observation_sequence_start = _reach_observation_sequence(robot)
    arm_eff = str(arm).strip().lower()
    arms = ["left", "right"] if arm_eff == "both" else [arm_eff]
    if any(side not in ("left", "right") for side in arms):
        _set_result(
            robot,
            "set_arm_to_grasp_position",
            {"ok": False, "error": "arm must be left, right, or both"},
        )
        yield robot.hold_action()
        return
    keep_ori = str(keep_ori_arm or "none").strip().lower()
    if keep_ori in ("", "false", "0", "no"):
        keep_ori = "none"
    if keep_ori not in ("none", "left", "right", "both"):
        _set_result(
            robot,
            "set_arm_to_grasp_position",
            {
                "ok": False,
                "error": "keep_ori_arm must be none, left, right, or both",
            },
        )
        yield robot.hold_action()
        return
    requested_keep_arms = (
        {"left", "right"} if keep_ori == "both" else {keep_ori}
    ) - {"none"}
    invalid_keep_arms = requested_keep_arms - set(arms)
    if invalid_keep_arms:
        _set_result(
            robot,
            "set_arm_to_grasp_position",
            {
                "ok": False,
                "error": (
                    f"keep_ori_arm={keep_ori} includes an arm outside arm={arm_eff}"
                ),
            },
        )
        yield robot.hold_action()
        return
    if requested_keep_arms:
        _set_result(
            robot,
            "set_arm_to_grasp_position",
            {
                "ok": False,
                "error": (
                    "keep_ori_arm is not available in the fixed joint-space "
                    "official_v2 replacement; use none"
                ),
                "keep_ori_arm": keep_ori,
            },
        )
        yield robot.hold_action()
        return
    tolerance = float(tol)
    timeout = float(timeout_s)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        _set_result(
            robot,
            "set_arm_to_grasp_position",
            {"ok": False, "error": "tol must be a positive finite value"},
        )
        yield robot.hold_action()
        return
    if not math.isfinite(timeout) or timeout <= 0.0:
        _set_result(
            robot,
            "set_arm_to_grasp_position",
            {"ok": False, "error": "timeout_s must be a positive finite value"},
        )
        yield robot.hold_action()
        return
    mode = str(gripper or "keep").strip().lower()
    if kwargs.get("open_gripper") is True:
        mode = "open"
    if mode not in ("keep", "open"):
        _set_result(
            robot,
            "set_arm_to_grasp_position",
            {"ok": False, "error": "gripper must be keep or open"},
        )
        yield robot.hold_action()
        return
    requested_max_step = float(max_dq_per_step)
    if not math.isfinite(requested_max_step) or requested_max_step <= 0.0:
        _set_result(
            robot,
            "set_arm_to_grasp_position",
            {
                "ok": False,
                "error": "max_dq_per_step must be a positive finite value",
            },
        )
        yield robot.hold_action()
        return
    if ARM_DOF > GRASP_PREP_Q.size:
        raise ValueError(
            f"grasp-prep target has {GRASP_PREP_Q.size} joints, "
            f"but the active profile requires {ARM_DOF}"
        )

    # Respect the caller's speed request exactly (the old lower clamp silently
    # turned a 0.01-rad safety request into 0.03 rad per frame).
    max_step = min(requested_max_step, GRASP_PREP_MAX_DQ_PER_STEP)
    if ARM_DOF == 8:
        for side in ("left", "right"):
            robot.set_tool_roll_pin_qpos(side, LOCKED_J8_VALUE_RAD)
    target_template = GRASP_PREP_Q[:ARM_DOF].copy()
    targets = {side: target_template.copy() for side in arms}
    if ARM_DOF == 8:
        for side in arms:
            targets[side][7] = LOCKED_J8_VALUE_RAD

    close_keepalive = {
        side: bool(
            getattr(
                robot,
                "gripper_close_keepalive_active",
                lambda _arm: False,
            )(side)
        )
        for side in arms
    }
    step_limits = {
        side: min(
            max_step,
            GRASP_PREP_CARRY_MAX_DQ_PER_STEP
            if close_keepalive[side]
            else GRASP_PREP_MAX_DQ_PER_STEP,
        )
        for side in arms
    }

    terminal_pins_latched = False
    entry_hold = {
        "steps": 0,
        "stable_steps": 0,
        "settled": True,
        "initial_max_qvel_rad_s": 0.0,
        "terminal_max_qvel_rad_s": 0.0,
        "terminal_derived_max_qvel_rad_s": None,
        "stationarity_evidence": "raw_qvel",
        "raw_qvel_incoherence_detected": False,
    }
    failure_hold = {
        "steps": 0,
        "stable_steps": 0,
        "settled": True,
        "initial_max_qvel_rad_s": 0.0,
        "terminal_max_qvel_rad_s": 0.0,
        "terminal_derived_max_qvel_rad_s": None,
        "stationarity_evidence": "raw_qvel",
        "raw_qvel_incoherence_detected": False,
    }
    failure_reason: str | None = None
    failure_qpos: dict[str, np.ndarray] | None = None
    failure_qvel: dict[str, np.ndarray] | None = None
    converged = False
    timed_out = False
    stalled = False
    timeout_reason: str | None = None
    action_steps = 0
    command_advance_steps = 0
    stable_steps = 0
    stall_steps = 0
    max_observed_qvel = 0.0
    max_tracking_lag = 0.0
    max_command_step = 0.0
    terminal_derived_max_qvel: float | None = None
    stationarity_evidence = "raw_qvel"
    raw_qvel_incoherence_detected = False
    extensions = 0
    extension_reasons: list[str] = []
    open_steps = 0

    try:
        observed_q, observed_qvel = _grasp_prep_snapshot(robot, arms)
        max_observed_qvel = _grasp_prep_max_velocity(observed_qvel)
        if (
            max_observed_qvel
            > GRASP_PREP_TRACKING_QVEL_TOLERANCE_RAD_S
        ):
            entry_hold_generator = _grasp_prep_latch_measured_hold(
                robot,
                arms=arms,
                max_steps=GRASP_PREP_ENTRY_HANDOFF_MAX_STEPS,
                stage="grasp-prep entry stabilization",
                check_cancelled=True,
                targets=targets,
                step_limits=step_limits,
            )
            while True:
                try:
                    yield next(entry_hold_generator)
                except StopIteration as stopped:
                    entry_hold = stopped.value
                    break
            observed_q = entry_hold["qpos"]
            observed_qvel = entry_hold["qvel"]
            max_observed_qvel = max(
                max_observed_qvel,
                float(entry_hold["initial_max_qvel_rad_s"]),
                float(entry_hold["terminal_max_qvel_rad_s"]),
            )
            # Do not reject a request merely because the entry arm is still
            # moving.  The main measured-relative loop below continues with a
            # fresh start and brakes it without replaying an old waypoint.

        starts = {side: observed_q[side].copy() for side in arms}
        start_qvel = {
            side: observed_qvel[side].copy() for side in arms
        }
        commands = {side: starts[side].copy() for side in arms}
        active_indices = _grasp_prep_active_indices()
        best_joint_error = {
            side: np.abs(targets[side] - starts[side]).astype(np.float64)
            for side in arms
        }
        joint_stall_counts = {
            side: np.zeros(ARM_DOF, dtype=np.int64) for side in arms
        }
        last_observed_q = {
            side: starts[side].copy() for side in arms
        }
        best_progress_metric = float(
            sum(float(np.sum(best_joint_error[side][active_indices])) for side in arms)
        )
        extension_reference_metric = best_progress_metric
        last_observation_sequence = _reach_observation_sequence(robot)
        first_feedback = True
        fresh_observation_count = 0
        stale_observation_steps = 0
        motion_since_extension = False
        command_reversal_count = 0
        handoff_forced = bool(
            max_observed_qvel
            > GRASP_PREP_TRACKING_QVEL_TOLERANCE_RAD_S
            and not bool(entry_hold["settled"])
        )
        relaxed_limit_by_arm = {
            side: np.zeros(ARM_DOF, dtype=bool) for side in arms
        }
        max_gap = max(
            float(
                np.max(
                    np.abs(
                        targets[side][active_indices]
                        - starts[side][active_indices]
                    )
                )
            )
            for side in arms
        )
        planned_interpolation_steps = max(
            1,
            max(
                int(
                    math.ceil(
                        float(
                            np.max(
                                np.abs(
                                    targets[side][active_indices]
                                    - starts[side][active_indices]
                                )
                            )
                        )
                        / step_limits[side]
                    )
                )
                for side in arms
            ),
        )
        initial_error = max_gap
        best_error = initial_error
        progress_threshold = max(
            GRASP_PREP_EXTENSION_MIN_PROGRESS_RAD,
            0.25 * tolerance,
        )
        soft_step_limit = max(1, int(math.ceil(timeout * CONTROL_HZ)))
        grace_s = min(GRASP_PREP_PROGRESS_GRACE_MAX_S, timeout)
        grace_steps = max(1, int(math.ceil(grace_s * CONTROL_HZ)))
        hard_timeout = grasp_prep_hard_timeout_s(timeout)
        hard_step_limit = (
            soft_step_limit
            + GRASP_PREP_PROGRESS_GRACE_EXTENSIONS * grace_steps
        )
        started_monotonic = time.monotonic()
        soft_deadline = started_monotonic + timeout
        hard_deadline = started_monotonic + hard_timeout
        stall_limit = GRASP_PREP_TRACKING_STALL_STEPS

        while failure_reason is None:
            observed_q, observed_qvel = _grasp_prep_snapshot(robot, arms)
            errors = {
                side: targets[side] - observed_q[side] for side in arms
            }
            observed_error = max(
                float(np.max(np.abs(error[active_indices])))
                for error in errors.values()
            )
            observed_max_qvel = _grasp_prep_max_velocity(observed_qvel)
            max_observed_qvel = max(max_observed_qvel, observed_max_qvel)

            current_sequence = _reach_observation_sequence(robot)
            sequence_delta: int | None = None
            fresh_observation = bool(
                first_feedback
                or current_sequence is None
                or last_observation_sequence is None
                or current_sequence > last_observation_sequence
            )
            if fresh_observation and action_steps > 0:
                if current_sequence is None or last_observation_sequence is None:
                    sequence_delta = 1
                elif current_sequence > last_observation_sequence:
                    sequence_delta = max(
                        1,
                        int(current_sequence) - int(last_observation_sequence),
                    )
            if current_sequence is not None:
                last_observation_sequence = max(
                    int(last_observation_sequence or current_sequence),
                    int(current_sequence),
                )
            first_feedback = False
            if fresh_observation:
                fresh_observation_count += 1
                stale_observation_steps = 0
            else:
                stale_observation_steps += 1

            derived_qvel = (
                _grasp_prep_derived_qvel(
                    last_observed_q,
                    observed_q,
                    observation_steps=sequence_delta,
                )
                if fresh_observation and sequence_delta is not None
                else None
            )
            stationarity = _grasp_prep_stationarity(
                observed_qvel,
                derived_qvel,
            )
            terminal_derived_max_qvel = stationarity[
                "derived_max_qvel_rad_s"
            ]
            stationarity_evidence = stationarity["evidence"]
            raw_qvel_incoherence_detected = bool(
                raw_qvel_incoherence_detected
                or stationarity["raw_qvel_incoherent"]
                or entry_hold["raw_qvel_incoherence_detected"]
            )

            accepted_position = True
            accepted_masks_by_arm: dict[str, np.ndarray] = {}
            for side in arms:
                strict_mask, relaxed_mask = _grasp_prep_position_status(
                    side,
                    targets[side],
                    observed_q[side],
                    tolerance,
                )
                accepted_mask = strict_mask | relaxed_mask
                accepted_masks_by_arm[side] = accepted_mask
                relaxed_limit_by_arm[side] = relaxed_mask
                accepted_position = bool(
                    accepted_position and np.all(accepted_mask[active_indices])
                )
                if fresh_observation:
                    absolute_error = np.abs(errors[side])
                    observed_motion = np.abs(
                        observed_q[side] - last_observed_q[side]
                    )
                    for index in active_indices:
                        if accepted_mask[index]:
                            joint_stall_counts[side][index] = 0
                        elif (
                            absolute_error[index]
                            <= best_joint_error[side][index]
                            - GRASP_PREP_PROGRESS_EPS_RAD
                        ):
                            best_joint_error[side][index] = absolute_error[
                                index
                            ]
                            joint_stall_counts[side][index] = 0
                        elif (
                            observed_motion[index]
                            >= GRASP_PREP_MIN_OBSERVED_MOTION_RAD
                        ):
                            # The joint is responding; do not call it stalled
                            # merely because this frame's error improvement is
                            # smaller than the progress epsilon.
                            joint_stall_counts[side][index] = 0
                        elif action_steps > 0:
                            joint_stall_counts[side][index] += 1
                    if bool(
                        np.any(
                            observed_motion[active_indices]
                            >= GRASP_PREP_MIN_OBSERVED_MOTION_RAD
                        )
                    ):
                        motion_since_extension = True
                    last_observed_q[side] = observed_q[side].copy()
            if fresh_observation:
                current_progress_metric = float(
                    sum(
                        float(np.sum(np.abs(errors[side][active_indices])))
                        for side in arms
                    )
                )
                best_progress_metric = min(
                    best_progress_metric,
                    current_progress_metric,
                )

            velocity_ok = bool(stationarity["stationary"])

            if (
                fresh_observation
                and accepted_position
                and velocity_ok
            ):
                stable_steps += 1
                if stable_steps >= GRASP_PREP_REQUIRED_STABLE_STEPS:
                    converged = True
                    break
            elif fresh_observation:
                stable_steps = 0

            if fresh_observation and observed_error <= best_error - GRASP_PREP_PROGRESS_EPS_RAD:
                best_error = observed_error

            unsatisfied = [
                (side, int(index))
                for side in arms
                for index in active_indices
                if not accepted_masks_by_arm[side][index]
            ]
            if unsatisfied:
                stall_steps = min(
                    int(joint_stall_counts[side][index])
                    for side, index in unsatisfied
                )
                all_joints_stalled = all(
                    int(joint_stall_counts[side][index]) >= stall_limit
                    for side, index in unsatisfied
                )
            else:
                stall_steps = 0
                all_joints_stalled = False
            if all_joints_stalled and action_steps > 0:
                stalled = True
                failure_reason = (
                    "arm tracking stalled before reaching grasp prep"
                )
                break
            if stale_observation_steps >= stall_limit:
                stalled = True
                stall_steps = stall_limit
                failure_reason = (
                    "arm tracking stalled before reaching grasp prep "
                    "(evaluator proprioception did not advance)"
                )
                break

            now = time.monotonic()
            hard_budget_hit = bool(
                now >= hard_deadline or action_steps >= hard_step_limit
            )
            soft_budget_hit = bool(
                now >= soft_deadline or action_steps >= soft_step_limit
            )
            if hard_budget_hit:
                timed_out = True
                timeout_reason = "hard_progress_deadline"
                failure_reason = (
                    "arms did not reach a stationary grasp-prep pose before "
                    "the progress-aware hard deadline"
                )
                break
            if soft_budget_hit:
                progress = extension_reference_metric - best_progress_metric
                near_target = observed_error <= max(2.0 * tolerance, 0.15)
                if (
                    extensions < GRASP_PREP_PROGRESS_GRACE_EXTENSIONS
                    and (
                        progress >= progress_threshold
                        or near_target
                        or motion_since_extension
                    )
                ):
                    reason = (
                        "near_target_settling"
                        if near_target and progress < progress_threshold
                        else "proprioceptive_error_progress"
                    )
                    extension_reasons.append(reason)
                    extensions += 1
                    extension_reference_metric = best_progress_metric
                    motion_since_extension = False
                    soft_deadline = min(
                        hard_deadline,
                        max(soft_deadline, now) + grace_s,
                    )
                    soft_step_limit = min(
                        hard_step_limit,
                        soft_step_limit + grace_steps,
                    )
                else:
                    timed_out = True
                    timeout_reason = "soft_deadline_without_progress"
                    failure_reason = (
                        "arms did not reach a stationary grasp-prep pose "
                        "before timeout and no continuing proprioceptive "
                        "progress justified an extension"
                    )
                    break

            _check_cancelled(robot, "feedback-governed grasp preparation")
            next_commands: dict[str, np.ndarray] = {}
            advanced = False
            for side in arms:
                previous = commands[side]
                measured = observed_q[side]
                target = targets[side]
                tracking_lag = float(
                    np.max(
                        np.abs(
                            previous[active_indices]
                            - measured[active_indices]
                        )
                    )
                )
                max_tracking_lag = max(max_tracking_lag, tracking_lag)
                command_qvel = observed_qvel[side]
                if derived_qvel is not None:
                    raw_side_max = float(
                        np.max(np.abs(command_qvel[active_indices]))
                    )
                    derived_side_max = float(
                        np.max(np.abs(derived_qvel[side][active_indices]))
                    )
                    if (
                        raw_side_max
                        > GRASP_PREP_TRACKING_QVEL_TOLERANCE_RAD_S
                        and derived_side_max
                        <= GRASP_PREP_TRACKING_QVEL_TOLERANCE_RAD_S
                    ):
                        command_qvel = derived_qvel[side]
                command, command_metrics = _grasp_prep_measured_command(
                    measured=measured,
                    qvel=command_qvel,
                    target=target,
                    start=starts[side],
                    previous=previous,
                    step_limit=step_limits[side],
                )
                command_reversal_count += int(
                    command_metrics[
                        "command_reversals_before_target_crossing"
                    ]
                )
                if float(
                    np.max(np.abs(command[active_indices] - measured[active_indices]))
                ) > 1.0e-9:
                    advanced = True
                max_command_step = max(
                    max_command_step,
                    float(
                        np.max(
                            np.abs(
                                command[active_indices]
                                - previous[active_indices]
                            )
                        )
                    ),
                )
                next_commands[side] = command

            commands = next_commands
            if advanced:
                command_advance_steps += 1
            yield robot.make_action(
                **{
                    f"arm_{side}": commands[side].astype(float).tolist()
                    for side in arms
                }
            )
            action_steps += 1

        if converged:
            if mode == "open":
                for _ in range(4):
                    _check_cancelled(robot, "grasp-prep gripper opening")
                    yield robot.make_action(
                        **{f"gripper_{side}": [1.0] for side in arms}
                    )
                    open_steps += 1
        else:
            failure_qpos, failure_qvel = _grasp_prep_snapshot(robot, arms)
            failure_hold_generator = _grasp_prep_latch_measured_hold(
                robot,
                arms=arms,
                max_steps=GRASP_PREP_FAILURE_HOLD_MAX_STEPS,
                stage="grasp-prep failure measured hold",
                check_cancelled=False,
            )
            while True:
                try:
                    yield next(failure_hold_generator)
                except StopIteration as stopped:
                    failure_hold = stopped.value
                    break
            terminal_pins_latched = True

        achieved, achieved_qvel = _grasp_prep_snapshot(robot, arms)
        errors = {
            side: targets[side] - achieved[side] for side in arms
        }
        max_error = max(
            float(np.max(np.abs(error[active_indices])))
            for error in errors.values()
        )
        terminal_max_qvel = _grasp_prep_max_velocity(achieved_qvel)
        max_observed_qvel = max(max_observed_qvel, terminal_max_qvel)
        final_strict_position_by_arm: dict[str, bool] = {}
        final_accepted_position_by_arm: dict[str, bool] = {}
        final_relaxed_limit_by_arm: dict[str, list[int]] = {}
        if converged:
            for side in arms:
                strict_mask, relaxed_mask = _grasp_prep_position_status(
                    side,
                    targets[side],
                    achieved[side],
                    tolerance,
                )
                final_strict_position_by_arm[side] = bool(
                    np.all(strict_mask[active_indices])
                )
                final_accepted_position_by_arm[side] = bool(
                    np.all((strict_mask | relaxed_mask)[active_indices])
                )
                final_relaxed_limit_by_arm[side] = [
                    int(index)
                    for index in active_indices
                    if bool(relaxed_mask[index])
                ]
                # A hard-limit plateau is accepted only as a stationary
                # measured pose.  Re-pinning that joint to the unreachable
                # mathematical endpoint would reintroduce post-success
                # chatter; strict convergence still pins the exact target.
                terminal_q = (
                    targets[side]
                    if final_strict_position_by_arm[side]
                    else achieved[side]
                )
                robot.set_arm_pin_qpos(side, terminal_q)
            terminal_pins_latched = True
        observation_sequence_end = _reach_observation_sequence(robot)
        observation_delta = (
            None
            if observation_sequence_start is None
            or observation_sequence_end is None
            else max(
                0,
                int(observation_sequence_end)
                - int(observation_sequence_start),
            )
        )
        wall_s = max(
            0.0,
            time.monotonic() - function_started_monotonic,
        )
        observed_action_hz = (
            None
            if observation_delta is None or wall_s <= 1e-9
            else float(observation_delta) / wall_s
        )
        limiting_joint_by_arm = {}
        for side in arms:
            joint_index = int(
                active_indices[
                    int(np.argmax(np.abs(errors[side][active_indices])))
                ]
            )
            limiting_joint_by_arm[side] = {
                "index": joint_index,
                "name": f"J{joint_index + 1}",
                "error_rad": float(errors[side][joint_index]),
                "qpos_rad": float(achieved[side][joint_index]),
                "qvel_rad_s": float(achieved_qvel[side][joint_index]),
                "target_rad": float(targets[side][joint_index]),
            }
        total_action_steps = (
            int(entry_hold["steps"])
            + action_steps
            + int(failure_hold["steps"])
            + open_steps
        )
        _set_result(
            robot,
            "set_arm_to_grasp_position",
            {
                "ok": bool(converged),
                "arms": arms,
                "keep_ori_arm": keep_ori,
                "target_qpos": {
                    side: targets[side].astype(float).tolist() for side in arms
                },
                "start_qpos": {
                    side: starts[side].astype(float).tolist() for side in arms
                },
                "start_qvel": {
                    side: start_qvel[side].astype(float).tolist()
                    for side in arms
                },
                "achieved_qpos": {
                    side: achieved[side].astype(float).tolist() for side in arms
                },
                "achieved_qvel": {
                    side: achieved_qvel[side].astype(float).tolist()
                    for side in arms
                },
                "error_rad": {
                    side: errors[side].astype(float).tolist() for side in arms
                },
                "converged": bool(converged),
                "tolerance_rad": tolerance,
                "velocity_tolerance_rad_s": (
                    GRASP_PREP_TRACKING_QVEL_TOLERANCE_RAD_S
                ),
                "convergence_requires_position_and_velocity": True,
                "stationarity_velocity_sources": [
                    "raw_qvel",
                    "consecutive_qpos_delta",
                ],
                "timeout_s": timeout,
                "soft_timeout_s": timeout,
                "hard_timeout_s": float(hard_timeout),
                "timeout_reason": timeout_reason,
                "progress_deadline_extensions": int(extensions),
                "progress_extension_reasons": extension_reasons,
                "requested_max_dq_per_step": requested_max_step,
                "effective_max_dq_per_step": float(max(step_limits.values())),
                "effective_max_dq_per_step_by_arm": {
                    side: float(step_limits[side]) for side in arms
                },
                "carry_aware_step_limit_active": bool(
                    any(close_keepalive.values())
                ),
                "close_keepalive_by_arm": close_keepalive,
                "planned_interpolation_steps": int(
                    planned_interpolation_steps
                ),
                "interpolation_steps": int(command_advance_steps),
                "action_steps": int(action_steps),
                "total_action_steps": int(total_action_steps),
                "entry_settle_steps": int(entry_hold["steps"]),
                "entry_settled": bool(entry_hold["settled"]),
                "entry_stationarity_evidence": entry_hold[
                    "stationarity_evidence"
                ],
                "entry_handoff_forced": bool(handoff_forced),
                "failure_hold_steps": int(failure_hold["steps"]),
                "failure_hold_settled": bool(failure_hold["settled"]),
                "failure_hold_stationarity_evidence": failure_hold[
                    "stationarity_evidence"
                ],
                "terminal_hold_source": (
                    "fixed_grasp_prep_target"
                    if converged
                    else "latest_evaluator_proprioception"
                ),
                "stable_steps": int(stable_steps),
                "required_stable_steps": GRASP_PREP_REQUIRED_STABLE_STEPS,
                "stall_steps": int(stall_steps),
                "stall_limit": int(stall_limit),
                "tracking_stalled": bool(stalled),
                "timed_out": bool(timed_out),
                "strict_position_converged": bool(
                    all(final_strict_position_by_arm.values())
                    if final_strict_position_by_arm
                    else False
                ),
                "accepted_position_converged": bool(
                    all(final_accepted_position_by_arm.values())
                    if final_accepted_position_by_arm
                    else False
                ),
                "limit_residual_joints_by_arm": final_relaxed_limit_by_arm,
                "max_abs_error_rad": float(max_error),
                "best_abs_error_rad": float(best_error),
                "max_observed_qvel_rad_s": float(max_observed_qvel),
                "terminal_max_qvel_rad_s": float(terminal_max_qvel),
                "terminal_derived_max_qvel_rad_s": (
                    None
                    if terminal_derived_max_qvel is None
                    else float(terminal_derived_max_qvel)
                ),
                "terminal_stationarity_evidence": stationarity_evidence,
                "raw_qvel_incoherence_detected": bool(
                    raw_qvel_incoherence_detected
                    or failure_hold["raw_qvel_incoherence_detected"]
                ),
                "max_tracking_lag_rad": float(max_tracking_lag),
                "max_command_step_rad": float(max_command_step),
                "command_reversals_before_target_crossing": int(
                    command_reversal_count
                ),
                "handoff_max_steps": int(GRASP_PREP_ENTRY_HANDOFF_MAX_STEPS),
                "stale_observation_steps": int(stale_observation_steps),
                "fresh_observation_count": int(fresh_observation_count),
                "joint_stall_steps_by_arm": {
                    side: joint_stall_counts[side].astype(int).tolist()
                    for side in arms
                },
                "limiting_joint_by_arm": limiting_joint_by_arm,
                "wall_s": float(wall_s),
                "observation_sequence_start": observation_sequence_start,
                "observation_sequence_end": observation_sequence_end,
                "fresh_observation_delta": observation_delta,
                "observed_fresh_action_hz": observed_action_hz,
                "gripper": mode,
                "execution": (
                    "feedback_governed_observed_qpos_to_absolute_targets"
                ),
                "failure_observed_qpos": (
                    None
                    if failure_qpos is None
                    else {
                        side: failure_qpos[side].astype(float).tolist()
                        for side in arms
                    }
                ),
                "failure_observed_qvel": (
                    None
                    if failure_qvel is None
                    else {
                        side: failure_qvel[side].astype(float).tolist()
                        for side in arms
                    }
                ),
                "collision_checked": False,
                "held_object_collision_truth_used": False,
                "direct_simulator_mutation": False,
                **({} if converged else {"error": failure_reason}),
            },
        )
        yield robot.hold_action()
    finally:
        if not terminal_pins_latched:
            try:
                latest_q, _latest_qvel = _grasp_prep_snapshot(robot, arms)
                for side in arms:
                    robot.set_arm_pin_qpos(side, latest_q[side])
            except (RuntimeError, TypeError, ValueError):
                pass


GRASP_PREP_Q = np.array(
    [
        0.0,
        0.0,
        0.0,
        -2.0943951024,
        0.0,
        -1.0471975512,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)






GRASP_PREP_MAX_DQ_PER_STEP = 0.30






GRASP_PREP_TRACKING_STALL_STEPS = 20



GRASP_PREP_CARRY_MAX_DQ_PER_STEP = 0.12



GRASP_PREP_TRACKING_QVEL_TOLERANCE_RAD_S = 0.35



GRASP_PREP_ADVANCE_QVEL_LIMIT_RAD_S = 1.50



GRASP_PREP_REQUIRED_STABLE_STEPS = 3



GRASP_PREP_ENTRY_HANDOFF_MAX_STEPS = 12



GRASP_PREP_FAILURE_HOLD_MAX_STEPS = 6



GRASP_PREP_HOLD_REQUIRED_STABLE_STEPS = 2



GRASP_PREP_DERIVED_HOLD_REQUIRED_STABLE_STEPS = 3



GRASP_PREP_PROGRESS_EPS_RAD = 0.003



GRASP_PREP_MIN_OBSERVED_MOTION_RAD = 5.0e-4



GRASP_PREP_EXTENSION_MIN_PROGRESS_RAD = 0.02



GRASP_PREP_PROGRESS_GRACE_MAX_S = 15.0



GRASP_PREP_PROGRESS_GRACE_EXTENSIONS = 2



GRASP_PREP_LIMIT_TARGET_EPS_RAD = 1.0e-4



GRASP_PREP_LIMIT_ACCEPTANCE_CAP_RAD = 0.12



GRASP_PREP_LIMIT_ACCEPTANCE_SCALE = 1.5



GRASP_PREP_LIMIT_ACCEPTANCE_JOINTS = (5,)



GRASP_PREP_BRAKE_STEP_SCALE = 0.35



GRASP_PREP_CRUISE_STEP_SCALE = 0.60



def grasp_prep_hard_timeout_s(timeout_s: float) -> float:
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout_s must be a positive finite value")
    grace = min(GRASP_PREP_PROGRESS_GRACE_MAX_S, timeout)
    return timeout + GRASP_PREP_PROGRESS_GRACE_EXTENSIONS * grace



def _grasp_prep_snapshot(
    ctx,
    arms: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Read one adapter-owned proprio vector when available.  Reading qpos and
    # qvel through separate provider calls can straddle a WebSocket update and
    # mix two evaluator frames in one control decision.
    adapter = getattr(ctx.world, "_official_adapter", None)
    proprio_provider = getattr(adapter, "proprio_vector", None)
    shared_proprio: np.ndarray | None = None
    if callable(proprio_provider):
        raw_proprio = proprio_provider()
        if raw_proprio is not None:
            shared_proprio = np.asarray(
                raw_proprio,
                dtype=np.float64,
            ).reshape(-1)
            if shared_proprio.size < PROPRIO_DIM:
                raise ValueError(
                    "evaluator proprio has "
                    f"{shared_proprio.size} values; active robot contract "
                    f"requires at least {PROPRIO_DIM}"
                )
            if not np.all(np.isfinite(shared_proprio)):
                raise ValueError("evaluator proprio contains non-finite values")
    qpos: dict[str, np.ndarray] = {}
    qvel: dict[str, np.ndarray] = {}
    for side in arms:
        if shared_proprio is None:
            position = np.asarray(
                ctx.world.arm_qpos_list(side),
                dtype=np.float64,
            ).reshape(-1)
            velocity = np.asarray(
                ctx.world.arm_qvel_list(side),
                dtype=np.float64,
            ).reshape(-1)
        else:
            position = shared_proprio[
                PROPRIO_SLICES[f"arm_{side}_qpos"]
            ].copy()
            velocity = shared_proprio[
                PROPRIO_SLICES[f"arm_{side}_qvel"]
            ].copy()
        if position.size != ARM_DOF or velocity.size != ARM_DOF:
            raise ValueError(
                f"{side} arm proprio has qpos/qvel dimensions "
                f"{position.size}/{velocity.size}; active robot contract "
                f"requires {ARM_DOF}/{ARM_DOF}"
            )
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError(f"{side} arm proprio contains non-finite values")
        qpos[side] = position
        qvel[side] = velocity
    return qpos, qvel



def _grasp_prep_active_indices() -> np.ndarray:
    """Return joints that are actually controlled by grasp preparation.

    The custom profile exposes J8 in the action/proprio contract, but that
    joint is a fixed tool-roll lock.  It must not make a moving arm look
    unstable and it must never participate in progress or convergence.
    """
    return np.arange(ARM_DOF - 1 if ARM_DOF == 8 else ARM_DOF, dtype=int)



def _grasp_prep_max_velocity(qvel: dict[str, np.ndarray]) -> float:
    active = _grasp_prep_active_indices()
    if active.size == 0:
        return 0.0
    return max(float(np.max(np.abs(value[active]))) for value in qvel.values())



def _grasp_prep_derived_qvel(
    previous: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    *,
    observation_steps: int,
) -> dict[str, np.ndarray]:
    """Estimate arm velocity from consecutive official proprio qpos frames."""
    steps = max(1, int(observation_steps))
    scale = float(CONTROL_HZ) / float(steps)
    derived: dict[str, np.ndarray] = {}
    for side, current_q in current.items():
        velocity = (
            np.asarray(current_q, dtype=np.float64).reshape(ARM_DOF)
            - np.asarray(previous[side], dtype=np.float64).reshape(ARM_DOF)
        ) * scale
        if ARM_DOF == 8:
            velocity[7] = 0.0
        derived[side] = velocity
    return derived



def _grasp_prep_stationarity(
    raw_qvel: dict[str, np.ndarray],
    derived_qvel: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    """Resolve stationarity without trusting an incoherent raw qvel alone."""
    raw_max = _grasp_prep_max_velocity(raw_qvel)
    derived_max = (
        None
        if derived_qvel is None
        else _grasp_prep_max_velocity(derived_qvel)
    )
    raw_stationary = bool(
        raw_max <= GRASP_PREP_TRACKING_QVEL_TOLERANCE_RAD_S
    )
    derived_stationary = bool(
        derived_max is not None
        and derived_max <= GRASP_PREP_TRACKING_QVEL_TOLERANCE_RAD_S
    )
    return {
        "stationary": bool(raw_stationary or derived_stationary),
        "evidence": (
            "raw_qvel"
            if raw_stationary
            else (
                "consecutive_qpos_delta"
                if derived_stationary
                else "none"
            )
        ),
        "raw_max_qvel_rad_s": float(raw_max),
        "derived_max_qvel_rad_s": (
            None if derived_max is None else float(derived_max)
        ),
        "raw_qvel_incoherent": bool(
            not raw_stationary and derived_stationary
        ),
    }



def _grasp_prep_position_status(
    side: str,
    target: np.ndarray,
    measured: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return strict/limit-relaxed per-joint position masks.

    A relaxed mask is deliberately limited to a target that is coincident
    with a static hard joint limit.  This handles the observed R1Pro J6
    controller plateau without weakening the caller's tolerance for normal
    joints or for an arbitrary unreachable pose.
    """
    target = np.asarray(target, dtype=np.float64).reshape(ARM_DOF)
    measured = np.asarray(measured, dtype=np.float64).reshape(ARM_DOF)
    errors = np.abs(target - measured)
    active = _grasp_prep_active_indices()
    strict = np.ones(ARM_DOF, dtype=bool)
    strict[active] = errors[active] <= float(tolerance)
    allowance = max(
        float(tolerance),
        min(
            GRASP_PREP_LIMIT_ACCEPTANCE_CAP_RAD,
            float(tolerance) * GRASP_PREP_LIMIT_ACCEPTANCE_SCALE,
        ),
    )
    limits = _ARM_LIMITS.get(str(side).strip().lower())
    if limits is None:
        return strict, np.zeros(ARM_DOF, dtype=bool)
    relaxed = np.zeros(ARM_DOF, dtype=bool)
    for index in active:
        if (
            strict[index]
            or index >= len(limits)
            or index not in GRASP_PREP_LIMIT_ACCEPTANCE_JOINTS
        ):
            continue
        lower, upper = limits[index]
        target_at_lower = abs(float(target[index]) - float(lower)) <= (
            GRASP_PREP_LIMIT_TARGET_EPS_RAD
        )
        target_at_upper = abs(float(target[index]) - float(upper)) <= (
            GRASP_PREP_LIMIT_TARGET_EPS_RAD
        )
        at_lower_side = target_at_lower and float(measured[index]) >= (
            float(target[index]) - GRASP_PREP_LIMIT_TARGET_EPS_RAD
        )
        at_upper_side = target_at_upper and float(measured[index]) <= (
            float(target[index]) + GRASP_PREP_LIMIT_TARGET_EPS_RAD
        )
        if (
            (at_lower_side or at_upper_side)
            and errors[index] <= allowance
        ):
            relaxed[index] = True
    return strict, relaxed



def _grasp_prep_measured_command(
    *,
    measured: np.ndarray,
    qvel: np.ndarray,
    target: np.ndarray,
    start: np.ndarray,
    previous: np.ndarray,
    step_limit: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one target-directed command from the latest evaluator frame.

    ``previous`` is used only for telemetry.  It is intentionally never used
    as a fallback target: resending an old waypoint after the measured arm has
    moved is what caused the previous back-and-forth behavior.
    """
    measured = np.asarray(measured, dtype=np.float64).reshape(ARM_DOF)
    qvel = np.asarray(qvel, dtype=np.float64).reshape(ARM_DOF)
    target = np.asarray(target, dtype=np.float64).reshape(ARM_DOF)
    start = np.asarray(start, dtype=np.float64).reshape(ARM_DOF)
    previous = np.asarray(previous, dtype=np.float64).reshape(ARM_DOF)
    command = measured.copy()
    active = _grasp_prep_active_indices()
    reversal_count = 0
    for index in active:
        error = float(target[index] - measured[index])
        if abs(error) <= 1.0e-9:
            # At the target, mirror the measured position while the joint is
            # still moving; this is a small damping handoff, not a retreat.
            command[index] = (
                measured[index]
                if abs(float(qvel[index]))
                > GRASP_PREP_TRACKING_QVEL_TOLERANCE_RAD_S
                else target[index]
            )
            continue
        direction = 1.0 if error > 0.0 else -1.0
        speed_toward = float(qvel[index]) * direction
        if speed_toward > GRASP_PREP_ADVANCE_QVEL_LIMIT_RAD_S:
            scale = GRASP_PREP_BRAKE_STEP_SCALE
        elif speed_toward > GRASP_PREP_TRACKING_QVEL_TOLERANCE_RAD_S:
            scale = GRASP_PREP_CRUISE_STEP_SCALE
        else:
            scale = 1.0
        step = min(abs(error), max(1.0e-9, float(step_limit)) * scale)
        command[index] = measured[index] + direction * step

        # A reversal is only expected after the measured position has crossed
        # the final target.  Compare command direction to the latest measured
        # state, not to the prior command; a lagging measured arm can make two
        # perfectly target-directed absolute commands appear numerically out
        # of order.
        initial_direction = float(np.sign(target[index] - start[index]))
        command_delta = float(command[index] - measured[index])
        crossed = (
            initial_direction != 0.0
            and (measured[index] - target[index]) * initial_direction > 0.0
        )
        if (
            initial_direction != 0.0
            and command_delta * initial_direction < -1.0e-6
            and not crossed
        ):
            reversal_count += 1
    if ARM_DOF == 8:
        command[7] = LOCKED_J8_VALUE_RAD
    return command, {
        "command_reversals_before_target_crossing": int(reversal_count),
        "max_measured_to_command_step_rad": float(
            np.max(np.abs(command - measured))
        ),
    }



def _grasp_prep_latch_measured_hold(
    ctx,
    *,
    arms: list[str],
    max_steps: int,
    stage: str,
    check_cancelled: bool,
    targets: dict[str, np.ndarray] | None = None,
    step_limits: dict[str, float] | None = None,
):
    steps = 0
    raw_stable_steps = 0
    derived_stable_steps = 0
    qpos, qvel = _grasp_prep_snapshot(ctx, arms)
    initial_max_qvel = _grasp_prep_max_velocity(qvel)
    previous_qpos = {side: qpos[side].copy() for side in arms}
    previous_sequence = _reach_observation_sequence(ctx)
    derived_qvel: dict[str, np.ndarray] | None = None
    stationarity = _grasp_prep_stationarity(qvel, derived_qvel)
    raw_qvel_incoherent = False
    settled = False
    while True:
        raw_stable_steps = (
            raw_stable_steps + 1
            if stationarity["evidence"] == "raw_qvel"
            else 0
        )
        derived_stable_steps = (
            derived_stable_steps + 1
            if stationarity["evidence"] == "consecutive_qpos_delta"
            else 0
        )
        raw_qvel_incoherent = bool(
            raw_qvel_incoherent or stationarity["raw_qvel_incoherent"]
        )
        settled = bool(
            raw_stable_steps >= GRASP_PREP_HOLD_REQUIRED_STABLE_STEPS
            or derived_stable_steps
            >= GRASP_PREP_DERIVED_HOLD_REQUIRED_STABLE_STEPS
        )
        if settled or steps >= max(0, int(max_steps)):
            break
        if check_cancelled:
            _check_cancelled(ctx, stage)
        commands: dict[str, list[float]] = {}
        for side in arms:
            measured = qpos[side]
            target = None if targets is None else targets.get(side)
            if target is None:
                command = measured.copy()
            else:
                command, _metrics = _grasp_prep_measured_command(
                    measured=measured,
                    qvel=qvel[side],
                    target=target,
                    start=measured,
                    previous=measured,
                    step_limit=(
                        float(step_limits.get(side, GRASP_PREP_MAX_DQ_PER_STEP))
                        if step_limits is not None
                        else GRASP_PREP_MAX_DQ_PER_STEP
                    ),
                )
            if ARM_DOF == 8:
                command[7] = LOCKED_J8_VALUE_RAD
            commands[f"arm_{side}"] = command.astype(float).tolist()
        yield ctx.world.make_action(**commands)
        steps += 1
        next_qpos, next_qvel = _grasp_prep_snapshot(ctx, arms)
        current_sequence = _reach_observation_sequence(ctx)
        sequence_advanced = bool(
            current_sequence is None
            or previous_sequence is None
            or current_sequence > previous_sequence
        )
        if sequence_advanced:
            observation_steps = (
                1
                if current_sequence is None or previous_sequence is None
                else max(1, int(current_sequence) - int(previous_sequence))
            )
            derived_qvel = _grasp_prep_derived_qvel(
                previous_qpos,
                next_qpos,
                observation_steps=observation_steps,
            )
            previous_qpos = {
                side: next_qpos[side].copy() for side in arms
            }
            previous_sequence = current_sequence
        else:
            derived_qvel = None
        qpos, qvel = next_qpos, next_qvel
        stationarity = _grasp_prep_stationarity(qvel, derived_qvel)

    for side in arms:
        ctx.world.set_arm_pin_qpos(side, qpos[side])
    return {
        "steps": int(steps),
        "stable_steps": int(max(raw_stable_steps, derived_stable_steps)),
        "raw_stable_steps": int(raw_stable_steps),
        "derived_stable_steps": int(derived_stable_steps),
        "settled": bool(settled),
        "stationarity_evidence": stationarity["evidence"],
        "raw_qvel_incoherence_detected": bool(raw_qvel_incoherent),
        "initial_max_qvel_rad_s": float(initial_max_qvel),
        "terminal_max_qvel_rad_s": float(_grasp_prep_max_velocity(qvel)),
        "terminal_derived_max_qvel_rad_s": stationarity[
            "derived_max_qvel_rad_s"
        ],
        "qpos": qpos,
        "qvel": qvel,
    }


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
