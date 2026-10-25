"""exec_plan_pose

执行 plan_id 对应的 EEF pose：当前→safe→目标 pose，不合爪；press_point 直接回放规划阶段已验证的 GPU IK anchors；完成后拍实际执行臂的 wrist camera。

How this tool works:
Load and validate the plan's session, episode and posture binding. Interpolate the chosen arm's joint targets with measured feedback, preserving the other arm and finger holds. Report pose errors and show the corresponding wrist view.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.action_constants import CONTROL_HZ
from tools.util.action_constants import DEFAULT_TRAJECTORY_STEP_RAD
from tools.util.action_constants import EXEC_SAFE_BACK_MAX_M
from tools.util.action_constants import EXEC_SAFE_FINAL_BRANCH_GAP_RAD
from tools.util.action_constants import EXEC_SAFE_ORI_TOL_DEG
from tools.util.action_constants import EXEC_SAFE_POS_TOL_M
from tools.util.action_constants import GRIPPER_CLOSE_FULL_OPEN_QPOS_TOLERANCE_M
from tools.util.action_constants import GRIPPER_CLOSE_UPPER_LIMIT_M
from tools.util.action_constants import GRIPPER_OPEN_EFFORT_N
from tools.util.action_constants import GRIPPER_OPEN_MAX_STEPS
from tools.util.action_constants import LOCKED_J8_CAPTURE_TOLERANCE_RAD
from tools.util.action_constants import LOCKED_J8_INDEX
from tools.util.action_constants import LOCKED_J8_SERIALIZED_TOLERANCE_RAD
from tools.util.action_constants import LOCKED_J8_VALUE_RAD
from tools.util.action_constants import MAX_TRAJECTORY_WAYPOINTS
from tools.util.action_constants import TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
from tools.util.action_constants import TRAJECTORY_STEP_VALIDATION_EPS_RAD
from tools.util.action_constants import _ARM_LIMITS
from tools.util.geometry.contract import ARM_DOF
from tools.util.geometry.eef_adjustment_local import local_robot_state
from tools.util.geometry.eef_adjustment_local import orientation_error_deg
from tools.util.geometry.grasp_kinematics_local import eef_pose as _local_eef_pose
from tools.util.geometry.grasp_kinematics_local import run_external_ik as _run_external_ik
from tools.util.motion_common import _arm_vector
from tools.util.motion_common import _check_cancelled
from tools.util.motion_common import _exec_safe_back_m_candidates
from tools.util.motion_common import _interpolate_joint_segment
from tools.util.motion_common import _j8_locked_copy
from tools.util.motion_common import _json_ready
from tools.util.motion_common import _load_plan_trajectory
from tools.util.motion_common import _normalize_arm_q_to_joint_limits
from tools.util.motion_common import _quat_to_mat_xyzw
from tools.util.motion_common import _r1pro_arm_fk_robot
from tools.util.motion_common import _require_episode_initialized
from tools.util.motion_common import _require_serialized_j8_zero
from tools.util.motion_common import _set_result
from tools.util.motion_common import _validate_exec_final_q_for_arm
from typing import Any
import math
import numpy as np
from tools.util.tool import main

NAME = 'exec_plan_pose'
SCHEMA = {'type': 'object',
 'properties': {'plan_id': {'type': 'string'},
                'arm': {'type': 'string', 'enum': ['', 'left', 'right']},
                'back_m': {'type': 'string', 'default': '0.10', 'description': '后退安全距离(m)，默认0.10'}},
 'additionalProperties': False,
 'required': ['plan_id']}


def actions(robot, **kwargs):
    session_id = str(kwargs.get("session_id", "")).strip()
    plan_id = str(kwargs.get("plan_id", "")).strip()
    requested_arm = str(kwargs.get("arm", "")).strip().lower()
    stop_after_safe = bool(kwargs.get("stop_after_safe", False))
    reset_tool_roll = bool(
        kwargs.get("reset_tool_roll_at_start", True)
    )
    requested_back_m = kwargs.get("back_m")
    active_arm = None
    selected_j8_before_action = None
    pre_exec_gripper_open = None
    j8_reset = None
    start_arm_velocity_observation: dict[str, Any] = {}
    try:
        _require_episode_initialized(robot, "exec_plan_pose")
        trajectory, plan_path, plan_record = _load_plan_trajectory(
            session_id,
            plan_id,
            include_record=True,
        )
        planned_arm = str(trajectory["active_arm"])
        if requested_arm in ("", "any"):
            active_arm = planned_arm
        elif requested_arm in ("left", "right"):
            active_arm = requested_arm
        else:
            raise ValueError("arm must be left, right, or omitted")
        parameters = dict(trajectory["parameters"])
        if reset_tool_roll != bool(
            parameters["reset_tool_roll_at_start"]
        ):
            raise ValueError(
                "reset_tool_roll_at_start does not match the immutable plan"
            )
        if requested_back_m not in (None, ""):
            back_m = float(requested_back_m)
            if not math.isfinite(back_m):
                raise ValueError("back_m must be finite")
            if (
                back_m != 0.0
                and not 0.02 <= back_m <= EXEC_SAFE_BACK_MAX_M
            ):
                raise ValueError(
                    "back_m must be 0 or in "
                    f"0.02..{EXEC_SAFE_BACK_MAX_M:.2f}m"
                )
        else:
            back_m = float(parameters["back_m"])

        tracking = dict(trajectory["tracking"])
        start_state = dict(trajectory["start_state"])
        if str(robot.episode_id()) != str(start_state["episode_id"]):
            raise ValueError(
                "plan is stale because the evaluator episode changed"
            )
        current_epoch = int(robot.motion_epoch())
        if current_epoch != int(start_state["motion_epoch"]):
            raise ValueError(
                "plan is stale because base or trunk motion occurred after capture"
            )
        final_q, final_q_resolution = _resolve_exec_final_q(
            trajectory=trajectory,
            record=plan_record,
            execution_arm=active_arm,
        )
        arm_switched = active_arm != planned_arm
        current_arms: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            current_q = _arm_vector(
                robot.arm_qpos_list(side),
                label=f"current {side} arm_qpos",
            )
            if side == active_arm and LOCKED_J8_INDEX is not None:
                selected_j8_before_action = float(
                    current_q[LOCKED_J8_INDEX]
                )
            current_arms[side] = (
                _j8_locked_copy(
                    current_q,
                    label=f"current {side} normalized arm_qpos",
                )
                if side == active_arm
                else current_q.copy()
            )
            current_qvel = _arm_vector(
                robot.arm_qvel_list(side),
                label=f"current {side} arm_qvel",
            )
            qvel_limit = float(
                tracking["stationary_qvel_tolerance_rad_s"]
            )
            start_arm_velocity_observation[side] = {
                "values_rad_s": current_qvel.astype(float).tolist(),
                "max_abs_rad_s": float(np.max(np.abs(current_qvel))),
                "legacy_stationary_tolerance_rad_s": qvel_limit,
                "exceeds_legacy_stationary_tolerance": bool(
                    float(np.max(np.abs(current_qvel))) > qvel_limit
                ),
                "policy": "diagnostic_only",
            }
        current_trunk = np.asarray(
            robot.trunk_qpos(),
            dtype=np.float64,
        ).reshape(-1)
        planned_trunk = np.asarray(
            start_state["trunk_qpos"],
            dtype=np.float64,
        ).reshape(4)
        trunk_error = float(
            np.linalg.norm(current_trunk - planned_trunk, ord=np.inf)
        )
        trunk_tolerance = float(
            tracking["start_trunk_tolerance_rad"]
        )
        if trunk_error > trunk_tolerance:
            raise ValueError(
                f"plan start trunk mismatch {trunk_error:.4f}rad "
                f"exceeds {trunk_tolerance:.4f}rad"
            )
        current_grippers: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            current_gripper = np.asarray(
                robot.gripper_qpos_list(side),
                dtype=np.float64,
            ).reshape(-1)
            if current_gripper.size == 0 or not np.all(np.isfinite(current_gripper)):
                raise ValueError(
                    f"current {side} gripper proprioception is invalid"
                )
            current_grippers[side] = current_gripper
        stored_back_m = float(parameters["back_m"])
        runtime_safe_q = None
        if arm_switched or abs(back_m - stored_back_m) > EXEC_SAFE_BACK_MATCH_TOL_M:
            final_pose_source = (
                "signed_final_q_submission_local_fk"
                if not arm_switched
                else (
                    "signed_requested_arm_final_q_submission_local_fk"
                    if str(final_q_resolution["source"])
                    == "signed_final_q_by_arm"
                    else "locally_verified_requested_arm_final_q_submission_local_fk"
                )
            )
            runtime_safe_q, back_m_resolution = _recompute_exec_safe_q(
                ctx=robot,
                execution_arm=active_arm,
                final_q=final_q,
                current_arms=current_arms,
                current_trunk=current_trunk,
                current_grippers=current_grippers,
                requested_back_m=back_m,
                fallback_back_m=(None if arm_switched else stored_back_m),
                final_q_source=final_pose_source,
            )
        else:
            back_m_resolution = {
                "mode": "signed_planned_safe",
                "requested_back_m": float(back_m),
                "active_back_m": float(stored_back_m),
                "back_m_adapted": False,
                "solver": "planner_signed_safe_q",
                "execution_arm": active_arm,
                "j8_locked": LOCKED_J8_INDEX is not None,
            }
        execution_segments, start_resolution = _prepare_exec_segments(
            trajectory=trajectory,
            current_q=current_arms[active_arm],
            current_trunk=current_trunk,
            stop_after_safe=stop_after_safe,
            execution_arm=active_arm,
            runtime_safe_q=runtime_safe_q,
            runtime_final_q=(final_q if arm_switched else None),
            back_m_resolution=back_m_resolution,
        )

        # All read-only path checks have passed. Pin every non-gripper
        # controller to the current evaluator observation before emitting the
        # first action, then establish the selected gripper's open state before
        # J8 reset or trajectory motion.
        for side in ("left", "right"):
            robot.set_arm_pin_qpos(side, current_arms[side])
        robot.set_trunk_pin_qpos(current_trunk.astype(float).tolist())

        pre_exec_gripper_open = {}
        for open_action in _ensure_exec_selected_gripper_open(
            robot,
            active_arm,
            result_out=pre_exec_gripper_open,
        ):
            yield open_action

        j8_reset = {}
        for reset_action in _reset_and_lock_selected_j8(
            robot,
            active_arm,
            stage="exec_plan_pose.start",
            result_out=j8_reset,
        ):
            yield reset_action
        if not bool(j8_reset["ok"]):
            raise RuntimeError(
                f"exec_plan_pose failed to reset and lock {active_arm} J8: "
                f"{j8_reset.get('error', 'unknown J8 reset failure')}"
            )
        if selected_j8_before_action is not None:
            j8_reset["before_deg"] = math.degrees(
                selected_j8_before_action
            )
            j8_reset["already_zero"] = bool(
                abs(selected_j8_before_action) <= 1e-9
            )
            j8_reset["lock_applied_from_first_exec_action"] = True
        observed_active_arm = _arm_vector(
            robot.arm_qpos_list(active_arm),
            label=f"post-reset {active_arm} arm_qpos",
        )
        if (
            LOCKED_J8_INDEX is not None
            and abs(float(observed_active_arm[LOCKED_J8_INDEX]))
            > LOCKED_J8_CAPTURE_TOLERANCE_RAD
        ):
            raise ValueError(
                f"exec start failed to lock {active_arm} J8 at zero; observed "
                f"{observed_active_arm[LOCKED_J8_INDEX]:.4f}rad after reset"
            )
        current_arms[active_arm] = _j8_locked_copy(
            observed_active_arm,
            label=f"post-reset {active_arm} normalized arm_qpos",
        )
        observed_gripper = np.asarray(
            robot.gripper_qpos_list(active_arm),
            dtype=np.float64,
        ).reshape(-1)
        if observed_gripper.size and np.all(np.isfinite(observed_gripper)):
            current_grippers[active_arm] = observed_gripper
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _set_result(
            robot,
            "exec_plan_pose",
            {
                "ok": False,
                "error": str(exc),
                "session_id": session_id,
                "plan_id": plan_id,
                "arm": active_arm,
                "start_arm_velocity_observation": _json_ready(
                    start_arm_velocity_observation
                ),
                "pre_exec_gripper_open": _json_ready(
                    pre_exec_gripper_open
                ),
                "j8_reset": _json_ready(j8_reset),
                "execution": (
                    "rejected_after_j8_reset"
                    if j8_reset is not None
                    and int(j8_reset.get("action_steps", 0)) > 0
                    else "rejected_after_pre_exec_gripper_open"
                    if pre_exec_gripper_open is not None
                    and int(
                        pre_exec_gripper_open.get("open_action_steps", 0)
                    )
                    > 0
                    else "rejected_before_action"
                ),
            },
        )
        yield robot.hold_action()
        return

    settle_tolerance = float(tracking["settle_tolerance_rad"])
    max_tracking_error = float(tracking["max_tracking_error_rad"])
    waypoint_timeout_steps = max(
        1,
        int(
            math.ceil(
                float(tracking["waypoint_timeout_s"]) * CONTROL_HZ
            )
        ),
    )
    stall_limit = max(1, int(tracking["stall_steps"]))
    executed_segments: list[str] = []
    action_steps = 0
    max_observed_error = 0.0

    for side in ("left", "right"):
        robot.set_arm_pin_qpos(side, current_arms[side])
    robot.set_trunk_pin_qpos(current_trunk.astype(float).tolist())

    gripper_holds: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        close_active = bool(
            getattr(
                robot,
                "gripper_close_keepalive_active",
                lambda _arm: False,
            )(side)
        )
        uses_effort = bool(
            getattr(
                robot,
                "gripper_uses_effort",
                lambda _arm: False,
            )(side)
        )
        hold_target = None
        if close_active:
            hold_mode = "close_keepalive"
        elif uses_effort:
            capture_hold = getattr(
                robot,
                "capture_gripper_hold_qpos",
                None,
            )
            hold_target = (
                capture_hold(side) if callable(capture_hold) else None
            )
            hold_mode = (
                "qpos_effort_servo"
                if hold_target is not None
                else "effort_hold_unavailable"
            )
        else:
            hold_mode = "position_controller_pin"
        gripper_holds[side] = {
            "mode": hold_mode,
            "entry_qpos": current_grippers[side].astype(float).tolist(),
            "hold_qpos": hold_target,
            "hold_allows_close_effort": getattr(
                robot,
                "gripper_hold_allows_close_effort",
                lambda _arm: None,
            )(side),
        }

    def pin_observed_body_for_hold() -> tuple[np.ndarray, np.ndarray]:
        observed_q = _arm_vector(
            robot.arm_qpos_list(active_arm),
            label=f"observed {active_arm} arm_qpos",
        )
        robot.set_arm_pin_qpos(active_arm, observed_q)
        observed_trunk = np.asarray(
            robot.trunk_qpos(), dtype=np.float64
        ).reshape(4)
        if not np.all(np.isfinite(observed_trunk)):
            raise ValueError("observed trunk qpos is non-finite")
        observed_trunk[3] = 0.0
        robot.set_trunk_pin_qpos(observed_trunk.astype(float).tolist())
        return observed_q, observed_trunk

    for segment in execution_segments:
        segment_name = str(segment["name"])
        for waypoint_index, raw_target in enumerate(segment["waypoints"]):
            target = _require_serialized_j8_zero(
                raw_target,
                label=f"{segment_name} waypoint",
            )
            previous_error = float("inf")
            stall_steps = 0
            for settle_step in range(waypoint_timeout_steps):
                is_cancelled = getattr(robot, "is_cancelled", None)
                if callable(is_cancelled) and bool(is_cancelled()):
                    pin_observed_body_for_hold()
                _check_cancelled(
                    robot,
                    f"exec_plan_pose {segment_name} waypoint {waypoint_index}",
                )
                action_overrides: dict[str, Any] = {
                    f"arm_{active_arm}": target.tolist()
                }
                yield robot.make_action(**action_overrides)
                action_steps += 1
                observed = _arm_vector(
                    robot.arm_qpos_list(active_arm),
                    label=f"observed {active_arm} arm_qpos",
                )
                arm_error = float(
                    np.linalg.norm(target - observed, ord=np.inf)
                )
                error = arm_error
                max_observed_error = max(max_observed_error, error)
                if error <= settle_tolerance:
                    break
                if error > max_tracking_error:
                    failure = (
                        f"tracking error {error:.4f}rad exceeds "
                        f"{max_tracking_error:.4f}rad"
                    )
                    pin_observed_body_for_hold()
                    _set_result(
                        robot,
                        "exec_plan_pose",
                        {
                            "ok": False,
                            "error": failure,
                            "session_id": session_id,
                            "plan_id": plan_id,
                            "arm": active_arm,
                            "start_arm_velocity_observation": _json_ready(
                                start_arm_velocity_observation
                            ),
                            "pre_exec_gripper_open": _json_ready(
                                pre_exec_gripper_open
                            ),
                            "segment": segment_name,
                            "waypoint_index": int(waypoint_index),
                            "action_steps": int(action_steps),
                            "execution": "stopped_on_tracking_error",
                        },
                    )
                    yield robot.hold_action()
                    return
                if previous_error - error <= 1e-4:
                    stall_steps += 1
                else:
                    stall_steps = 0
                previous_error = error
                if stall_steps >= stall_limit:
                    pin_observed_body_for_hold()
                    _set_result(
                        robot,
                        "exec_plan_pose",
                        {
                            "ok": False,
                            "error": "proprioceptive tracking stalled",
                            "session_id": session_id,
                            "plan_id": plan_id,
                            "arm": active_arm,
                            "start_arm_velocity_observation": _json_ready(
                                start_arm_velocity_observation
                            ),
                            "pre_exec_gripper_open": _json_ready(
                                pre_exec_gripper_open
                            ),
                            "segment": segment_name,
                            "waypoint_index": int(waypoint_index),
                            "action_steps": int(action_steps),
                            "execution": "stopped_on_tracking_stall",
                        },
                    )
                    yield robot.hold_action()
                    return
            else:
                pin_observed_body_for_hold()
                _set_result(
                    robot,
                    "exec_plan_pose",
                    {
                        "ok": False,
                        "error": "waypoint tracking timeout",
                        "session_id": session_id,
                        "plan_id": plan_id,
                        "arm": active_arm,
                        "start_arm_velocity_observation": _json_ready(
                            start_arm_velocity_observation
                        ),
                        "pre_exec_gripper_open": _json_ready(
                            pre_exec_gripper_open
                        ),
                        "segment": segment_name,
                        "waypoint_index": int(waypoint_index),
                        "action_steps": int(action_steps),
                        "execution": "stopped_on_tracking_timeout",
                    },
                )
                yield robot.hold_action()
                return
        executed_segments.append(segment_name)

    final_q = _require_serialized_j8_zero(
        execution_segments[-1]["endpoint_q"],
        label="executed endpoint_q",
    )
    robot.set_arm_pin_qpos(active_arm, final_q)
    _set_result(
        robot,
        "exec_plan_pose",
        {
            "ok": True,
            "session_id": session_id,
            "plan_id": plan_id,
            "plan_record_path": plan_path,
            "trajectory_digest": trajectory["integrity"]["digest"],
            "source_image_id": trajectory["source_image_id"],
            "arm": active_arm,
            "planned_arm": planned_arm,
            "requested_arm": requested_arm or None,
            "arm_switched": bool(arm_switched),
            "start_arm_velocity_observation": _json_ready(
                start_arm_velocity_observation
            ),
            "pre_exec_gripper_open": _json_ready(
                pre_exec_gripper_open
            ),
            "j8_reset": _json_ready(j8_reset),
            "final_q_resolution": _json_ready(final_q_resolution),
            "segments_executed": executed_segments,
            "start_mode": start_resolution["mode"],
            "start_resolution": start_resolution,
            "requested_back_m": float(
                start_resolution["back_m_resolution"]["requested_back_m"]
            ),
            "active_back_m": float(
                start_resolution["back_m_resolution"]["active_back_m"]
            ),
            "back_m_adapted": bool(
                start_resolution["back_m_resolution"]["back_m_adapted"]
            ),
            "gripper_policy": (
                "open_selected_before_motion_then_hold_observed_qpos"
            ),
            "gripper_holds": gripper_holds,
            "stop_after_safe": bool(stop_after_safe),
            "action_steps": int(action_steps),
            "max_observed_tracking_error_rad": float(
                max_observed_error
            ),
            "endpoint_q": final_q.astype(float).tolist(),
            "trunk_motion": False,
            "execution": "official_action_joint_trajectory_replay",
            "direct_simulator_mutation": False,
        },
    )
    yield robot.hold_action()


WRIST_ROLL_POSITION_TOLERANCE_RAD = math.radians(0.5)



WRIST_ROLL_VELOCITY_TOLERANCE_RAD_S = math.radians(3.0)



WRIST_ROLL_REQUIRED_STABLE_STEPS = 3



FORCED_WRIST_ROLL_RESET_TIMEOUT_S = 8.0



EXEC_SAFE_BACK_MATCH_TOL_M = 1e-4



def _stored_record_final_q_for_arm(
    record: dict[str, Any],
    arm: str,
) -> tuple[Any, str] | tuple[None, None]:
    """Find the plan-filter final IK branch using the same v2 record fields."""
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None, None
    locations: list[tuple[str, dict[str, Any]]] = [("payload", payload)]
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        candidate = candidates[0]
        locations.append(("payload.candidates[0]", candidate))
        candidate_meta = candidate.get("meta")
        if isinstance(candidate_meta, dict):
            locations.append(("payload.candidates[0].meta", candidate_meta))
    plan_audit = payload.get("plan_audit")
    if isinstance(plan_audit, dict):
        locations.append(("payload.plan_audit", plan_audit))
        pick = plan_audit.get("pick")
        if isinstance(pick, dict):
            locations.append(("payload.plan_audit.pick", pick))

    for location_name, location in locations:
        q_map = location.get("selected_pose_ik_q")
        if isinstance(q_map, dict) and q_map.get(arm) is not None:
            return q_map[arm], f"{location_name}.selected_pose_ik_q"
        ik_summary = location.get("selected_pose_ik")
        if isinstance(ik_summary, dict):
            raw_q = ik_summary.get(f"{arm}_q_arm")
            if raw_q is None and str(location.get("arm", "")) == arm:
                raw_q = ik_summary.get("q_arm")
            if raw_q is not None:
                return raw_q, f"{location_name}.selected_pose_ik"
        ik_filter = location.get("ik_filter")
        if isinstance(ik_filter, dict):
            arm_info = ik_filter.get(arm)
            if isinstance(arm_info, dict) and arm_info.get("q_arm") is not None:
                return arm_info["q_arm"], f"{location_name}.ik_filter"
    return None, None



def _resolve_exec_final_q(
    *,
    trajectory: dict[str, Any],
    record: dict[str, Any],
    execution_arm: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    planned_arm = str(trajectory["active_arm"])
    if execution_arm == planned_arm:
        final_q = _require_serialized_j8_zero(
            trajectory["segments"][2]["endpoint_q"],
            label=f"signed {planned_arm} final endpoint_q",
        )
        return final_q, {
            "source": "signed_active_trajectory_endpoint",
            "planned_arm": planned_arm,
            "execution_arm": execution_arm,
            "arm_switched": False,
            "submission_local_fk_checked": True,
            "j8_locked": LOCKED_J8_INDEX is not None,
        }

    signed_by_arm = trajectory.get("final_q_by_arm")
    if isinstance(signed_by_arm, dict) and signed_by_arm.get(execution_arm) is not None:
        raw_q = signed_by_arm[execution_arm]
        source = "signed_final_q_by_arm"
    else:
        raw_q, record_location = _stored_record_final_q_for_arm(
            record,
            execution_arm,
        )
        if raw_q is None:
            raise ValueError(
                f"plan has no stored final joint solution for requested arm "
                f"{execution_arm!r}"
            )
        source = f"legacy_plan_record_locally_verified:{record_location}"

    return _validate_exec_final_q_for_arm(
        trajectory=trajectory,
        arm=execution_arm,
        raw_q=raw_q,
        source=source,
    )



def _prepare_exec_segments(
    *,
    trajectory: dict[str, Any],
    current_q: np.ndarray,
    current_trunk: np.ndarray,
    stop_after_safe: bool,
    execution_arm: str | None = None,
    runtime_safe_q: np.ndarray | None = None,
    runtime_final_q: np.ndarray | None = None,
    back_m_resolution: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a fresh observed-start bridge to the signed safe pose."""
    active_arm = str(execution_arm or trajectory["active_arm"])
    start_state = dict(trajectory["start_state"])
    observed_q = _j8_locked_copy(
        current_q,
        label=f"current {active_arm} arm_qpos",
    )
    planned_start = _require_serialized_j8_zero(
        start_state["arm_qpos"][active_arm],
        label=f"planned {active_arm} arm_qpos",
    )
    limits = _ARM_LIMITS[active_arm][:ARM_DOF]
    if runtime_safe_q is None and active_arm != str(trajectory["active_arm"]):
        raise ValueError("arm-switched execution requires a locally resolved safe_q")
    start_error = float(np.linalg.norm(observed_q - planned_start, ord=np.inf))
    safe_q = (
        _require_serialized_j8_zero(
            trajectory["segments"][1]["endpoint_q"],
            label="signed safe endpoint_q",
        )
        if runtime_safe_q is None
        else _j8_locked_copy(
            runtime_safe_q,
            label="runtime replanned safe endpoint_q",
        )
    )
    bridge_waypoints = _interpolate_joint_segment(observed_q, safe_q)
    if runtime_safe_q is None and runtime_final_q is None:
        final_segment = trajectory["segments"][2]
    else:
        final_q = _require_serialized_j8_zero(
            (
                trajectory["segments"][2]["endpoint_q"]
                if runtime_final_q is None
                else runtime_final_q
            ),
            label=f"resolved {active_arm} final endpoint_q",
        )
        if np.any(final_q < limits[:, 0]) or np.any(final_q > limits[:, 1]):
            raise ValueError("resolved final_q exceeds local arm joint limits")
        final_segment = {
            "name": "safe_to_final",
            "waypoints": _interpolate_joint_segment(safe_q, final_q),
            "endpoint_q": final_q.astype(float).tolist(),
        }
    signed_final_count = (
        0 if stop_after_safe else len(final_segment["waypoints"])
    )
    if len(bridge_waypoints) + signed_final_count > MAX_TRAJECTORY_WAYPOINTS:
        raise ValueError("runtime bridge produces too many waypoints")

    def local_limit_violation(q: np.ndarray) -> np.ndarray:
        return np.maximum(
            np.maximum(limits[:, 0] - q, q - limits[:, 1]),
            0.0,
        )

    observed_limit_violation = local_limit_violation(observed_q)
    previous_limit_violation = observed_limit_violation
    previous_q = observed_q
    previous_remaining = np.abs(safe_q - observed_q)
    envelope_lower = np.minimum(observed_q, safe_q)
    envelope_upper = np.maximum(observed_q, safe_q)
    fk_samples = [observed_q]
    for raw_q in bridge_waypoints:
        q = _require_serialized_j8_zero(
            raw_q,
            label="runtime current_to_safe waypoint",
        )
        step = float(np.linalg.norm(q - previous_q, ord=np.inf))
        if step > DEFAULT_TRAJECTORY_STEP_RAD + TRAJECTORY_STEP_VALIDATION_EPS_RAD:
            raise ValueError("runtime current-to-safe bridge exceeds the step limit")
        if np.any(q < envelope_lower - TRAJECTORY_STEP_VALIDATION_EPS_RAD) or np.any(
            q > envelope_upper + TRAJECTORY_STEP_VALIDATION_EPS_RAD
        ):
            raise ValueError("runtime current-to-safe bridge leaves its endpoint envelope")
        remaining = np.abs(safe_q - q)
        if np.any(
            remaining
            > previous_remaining + TRAJECTORY_STEP_VALIDATION_EPS_RAD
        ):
            raise ValueError("runtime current-to-safe bridge does not move toward safe_q")
        limit_violation = local_limit_violation(q)
        if np.any(
            limit_violation
            > previous_limit_violation + TRAJECTORY_STEP_VALIDATION_EPS_RAD
        ):
            raise ValueError(
                "runtime current-to-safe bridge moves farther outside local joint limits"
            )
        fk_samples.append(q)
        previous_q = q
        previous_remaining = remaining
        previous_limit_violation = limit_violation
    if not np.allclose(
        previous_q,
        safe_q,
        rtol=0.0,
        atol=LOCKED_J8_SERIALIZED_TOLERANCE_RAD,
    ):
        raise ValueError("runtime current-to-safe bridge does not end at safe_q")
    for q in fk_samples:
        position, rotation = _r1pro_arm_fk_robot(
            active_arm,
            current_trunk,
            q,
        )
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            raise ValueError("runtime bridge local FK produced a non-finite pose")

    prepared_segments = [
        {
            "name": "runtime_current_to_safe",
            "waypoints": bridge_waypoints,
            "endpoint_q": safe_q.astype(float).tolist(),
        }
    ]
    if not stop_after_safe:
        prepared_segments.append(final_segment)
        if runtime_safe_q is not None:
            fk_samples.extend(
                _require_serialized_j8_zero(
                    raw_q,
                    label="runtime safe_to_final waypoint",
                )
                for raw_q in final_segment["waypoints"]
            )
            for q in fk_samples[len(bridge_waypoints) + 1 :]:
                position, rotation = _r1pro_arm_fk_robot(
                    active_arm,
                    current_trunk,
                    q,
                )
                if not np.all(np.isfinite(position)) or not np.all(
                    np.isfinite(rotation)
                ):
                    raise ValueError(
                        "runtime safe-to-final local FK produced a non-finite pose"
                    )
    outside_joint_indices = np.flatnonzero(
        observed_limit_violation
        > TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
    )
    return prepared_segments, {
        "mode": "fresh_current_to_safe",
        "planned_start_error_rad": start_error,
        "start_source": "evaluator_proprioception",
        "signed_prefix_replay": False,
        "signed_waypoint_resume": False,
        "runtime_bridge": {
            "waypoint_count": int(len(bridge_waypoints)),
            "current_start_joint_limits_enforced": False,
            "current_start_within_local_limits": not bool(
                outside_joint_indices.size
            ),
            "current_start_max_local_limit_violation_rad": float(
                np.max(observed_limit_violation)
            ),
            "current_start_outside_local_limit_joints": [
                f"J{int(index) + 1}" for index in outside_joint_indices
            ],
            "signed_safe_joint_limits_checked": True,
            "endpoint_envelope_checked": True,
            "monotonic_to_safe_checked": True,
            "monotonic_local_limit_recovery_checked": True,
            "max_step_checked": True,
            "local_fk_finite_checked": True,
            "j8_locked": LOCKED_J8_INDEX is not None,
            "frozen_rgbd_collision_checked": False,
            "self_collision_checked": False,
            "safe_to_final_recomputed": (
                runtime_safe_q is not None or runtime_final_q is not None
            ),
            "final_arm_trajectory_recomputed": runtime_final_q is not None,
        },
        "back_m_resolution": _json_ready(back_m_resolution or {}),
    }



def _recompute_exec_safe_q(
    *,
    ctx,
    execution_arm: str,
    final_q: np.ndarray,
    current_arms: dict[str, np.ndarray],
    current_trunk: np.ndarray,
    current_grippers: dict[str, np.ndarray],
    requested_back_m: float,
    fallback_back_m: float | None,
    final_q_source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Re-solve a requested safe retreat from signed final-q and observations."""
    active_arm = str(execution_arm).strip().lower()
    if active_arm not in ("left", "right"):
        raise ValueError("execution arm must be left or right")
    final_q = _require_serialized_j8_zero(
        final_q,
        label=f"resolved {active_arm} final endpoint_q",
    )
    ik_arms = {
        side: np.asarray(current_arms[side], dtype=np.float64).copy()
        for side in ("left", "right")
    }
    # The observed arm remains the source of the action bridge.  Use the
    # signed, in-limit final branch as the local solver retract so a measured
    # start just outside static limits cannot block back_m recomputation.
    ik_arms[active_arm] = final_q.copy()
    state = local_robot_state(
        trunk_q=current_trunk,
        arm_left_q=ik_arms["left"],
        arm_right_q=ik_arms["right"],
        gripper_left_q=current_grippers["left"],
        gripper_right_q=current_grippers["right"],
    )
    final_position, final_quaternion = _local_eef_pose(
        state,
        active_arm,
        final_q,
    )
    pointing = _quat_to_mat_xyzw(final_quaternion)[:, 2]
    pointing /= max(float(np.linalg.norm(pointing)), 1e-12)
    candidates = _exec_safe_back_m_candidates(
        requested_back_m,
        fallback_back_m=fallback_back_m,
    )

    if candidates and abs(float(candidates[0])) <= 1e-6:
        return final_q.copy(), {
            "mode": "runtime_local_safe_replan",
            "requested_back_m": float(requested_back_m),
            "active_back_m": 0.0,
            "back_m_adapted": abs(float(requested_back_m)) > 1e-4,
            "candidate_index": 0,
            "candidate_count": int(len(candidates)),
            "solver": "signed_final_q_as_zero_back_safe",
            "final_pose_source": str(final_q_source),
            "ik_retract_source": "signed_final_q",
            "safe_endpoint_local_fk_checked": True,
            "safe_to_final_joint_gap_rad": 0.0,
            "safe_final_branch_gap_soft": False,
            "j8_locked": LOCKED_J8_INDEX is not None,
        }

    poses = [
        {
            "eef_pos": (
                final_position - float(back_m) * pointing
            ).astype(float).tolist(),
            "quat": np.asarray(
                final_quaternion,
                dtype=np.float64,
            ).astype(float).tolist(),
            "ik_warm_start_q_by_arm": {
                active_arm: final_q.astype(float).tolist()
            },
        }
        for back_m in candidates
    ]
    _check_cancelled(ctx, "exec_plan_pose local back_m IK")
    try:
        left_results, right_results, worker_meta = _run_external_ik(
            state,
            poses,
            pos_tol_m=EXEC_SAFE_POS_TOL_M,
            ori_tol_deg=EXEC_SAFE_ORI_TOL_DEG,
            active_arms=(active_arm,),
            ctx=ctx,
        )
    except Exception as exc:
        raise ValueError(
            "submission-local safe IK worker failed before action: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    arm_results = left_results if active_arm == "left" else right_results
    if len(arm_results) != len(candidates):
        raise ValueError(
            "local safe IK worker returned an incomplete candidate batch"
        )

    attempts: list[dict[str, Any]] = []
    for candidate_index, (back_m, info) in enumerate(
        zip(candidates, arm_results)
    ):
        attempt = {
            "candidate_index": int(candidate_index),
            "back_m": float(back_m),
            "solver_ok": bool(info.get("ok")),
            "worker_pos_err_m": _json_ready(info.get("pos_err_m")),
            "worker_ori_err_deg": _json_ready(info.get("ori_err_deg")),
            "error": info.get("error"),
        }
        raw_q = info.get("q_arm")
        if not bool(info.get("ok")) or raw_q is None:
            attempts.append(attempt)
            continue
        try:
            safe_q = _j8_locked_copy(
                raw_q,
                label=f"runtime safe IK candidate {candidate_index}",
            )
            safe_q, normalizations = _normalize_arm_q_to_joint_limits(
                safe_q,
                arm=active_arm,
                label=f"runtime safe IK candidate {candidate_index}",
                point="runtime_safe_q",
            )
            actual_position, actual_quaternion = _local_eef_pose(
                state,
                active_arm,
                safe_q,
            )
            target_position = np.asarray(
                poses[candidate_index]["eef_pos"],
                dtype=np.float64,
            )
            position_error = float(
                np.linalg.norm(actual_position - target_position)
            )
            orientation_error = float(
                orientation_error_deg(
                    actual_quaternion,
                    final_quaternion,
                )
            )
            if (
                position_error > EXEC_SAFE_POS_TOL_M
                or orientation_error > EXEC_SAFE_ORI_TOL_DEG
            ):
                attempt.update(
                    {
                        "error": "safe_endpoint_local_fk_failed",
                        "local_fk_pos_err_m": position_error,
                        "local_fk_ori_err_deg": orientation_error,
                    }
                )
                attempts.append(attempt)
                continue
        except (TypeError, ValueError) as exc:
            attempt["error"] = str(exc)
            attempts.append(attempt)
            continue

        branch_gap = float(np.linalg.norm(safe_q - final_q, ord=np.inf))
        attempt.update(
            {
                "accepted": True,
                "local_fk_pos_err_m": position_error,
                "local_fk_ori_err_deg": orientation_error,
                "safe_to_final_joint_gap_rad": branch_gap,
            }
        )
        attempts.append(attempt)
        return safe_q, {
            "mode": "runtime_local_safe_replan",
            "requested_back_m": float(requested_back_m),
            "active_back_m": float(back_m),
            "back_m_adapted": abs(
                float(back_m) - float(requested_back_m)
            )
            > 1e-4,
            "candidate_index": int(candidate_index),
            "candidate_count": int(len(candidates)),
            "attempts": attempts,
            "solver": "submission_local_curobo_gpu_worker",
            "worker": _json_ready(worker_meta),
            "final_pose_source": str(final_q_source),
            "ik_retract_source": "signed_final_q",
            "safe_endpoint_local_fk_checked": True,
            "safe_endpoint_pos_err_m": position_error,
            "safe_endpoint_ori_err_deg": orientation_error,
            "safe_to_final_joint_gap_rad": branch_gap,
            "safe_final_branch_gap_soft": bool(
                branch_gap > EXEC_SAFE_FINAL_BRANCH_GAP_RAD
            ),
            "joint_limit_normalizations": normalizations,
            "j8_locked": LOCKED_J8_INDEX is not None,
        }

    attempted = ", ".join(f"{value:.2f}" for value in candidates)
    raise ValueError(
        "no locally feasible safe pose for requested "
        f"back_m={float(requested_back_m):.3f}m; tried [{attempted}]"
    )



def _reset_and_lock_selected_j8(
    ctx,
    arm: str,
    *,
    stage: str,
    timeout_s: float = FORCED_WRIST_ROLL_RESET_TIMEOUT_S,
    result_out: dict[str, Any] | None = None,
):
    """Drive one J8 to zero, then revoke motion and retain a zero pin."""
    def finish(result: dict[str, Any]) -> dict[str, Any]:
        if result_out is not None:
            result_out.clear()
            result_out.update(result)
        return result

    side = str(arm).strip().lower()
    if side not in ("left", "right"):
        return finish({
            "ok": False,
            "arm": side,
            "error": "arm must be left or right",
            "action_steps": 0,
        })
    if ARM_DOF != 8:
        return finish({
            "ok": True,
            "arm": side,
            "supported": False,
            "action_steps": 0,
            "locked_after_motion": False,
        })
    if not bool(ctx.world.has_tool_roll(side)):
        return finish({
            "ok": False,
            "arm": side,
            "error": "8DOF profile has no independent J8 controller",
            "action_steps": 0,
        })

    entry_arm = np.asarray(
        ctx.world.arm_qpos_list(side),
        dtype=np.float64,
    ).reshape(-1)
    if entry_arm.size != ARM_DOF or not np.all(np.isfinite(entry_arm)):
        ctx.world.set_tool_roll_pin_qpos(side, 0.0)
        ctx.world.end_tool_roll_motion(side)
        return finish({
            "ok": False,
            "arm": side,
            "error": f"{side} arm proprioception must contain {ARM_DOF} finite values",
            "action_steps": 0,
            "locked_after_motion": True,
            "lock_target_deg": 0.0,
        })

    before = float(entry_arm[7])
    fixed_arm = entry_arm.copy()
    entry_vel = np.asarray(
        ctx.world.arm_qvel_list(side),
        dtype=np.float64,
    ).reshape(-1)
    if entry_vel.size != ARM_DOF or not np.all(np.isfinite(entry_vel)):
        zero_arm = fixed_arm.copy()
        zero_arm[7] = 0.0
        ctx.world.set_tool_roll_pin_qpos(side, 0.0)
        ctx.world.set_arm_pin_qpos(side, zero_arm)
        ctx.world.end_tool_roll_motion(side)
        return finish({
            "ok": False,
            "arm": side,
            "error": f"{side} arm velocity must contain {ARM_DOF} finite values",
            "action_steps": 0,
            "locked_after_motion": True,
            "lock_target_deg": 0.0,
        })
    initial_raw_velocity = float(entry_vel[7])

    if (
        abs(before) <= 1e-9
        and abs(float(entry_vel[7]))
        <= WRIST_ROLL_VELOCITY_TOLERANCE_RAD_S
    ):
        fixed_arm[7] = 0.0
        ctx.world.set_tool_roll_pin_qpos(side, 0.0)
        ctx.world.set_arm_pin_qpos(side, fixed_arm)
        ctx.world.end_tool_roll_motion(side)
        return finish({
            "ok": True,
            "arm": side,
            "supported": True,
            "joint": f"{side}_arm_joint8",
            "before_deg": math.degrees(before),
            "after_deg": math.degrees(before),
            "final_error_deg": -math.degrees(before),
            "target_deg": 0.0,
            "lock_target_deg": 0.0,
            "locked_after_motion": True,
            "selected_j1_j7_fixed": True,
            "selected_j1_j7_max_drift_deg": 0.0,
            "action_steps": 0,
            "stable_steps": 1,
            "timed_out": False,
            "already_zero": True,
            "initial_raw_velocity_rad_s": initial_raw_velocity,
            "final_raw_velocity_rad_s": initial_raw_velocity,
            "final_derived_velocity_rad_s": None,
            "stationarity_evidence": "raw_qvel",
            "raw_qvel_incoherence_detected": False,
        })

    ctx.world.set_tool_roll_pin_qpos(side, before)
    ctx.world.set_arm_pin_qpos(side, fixed_arm)
    ctx.world.begin_tool_roll_motion(side)
    action_steps = 0
    stable_steps = 0
    converged = False
    failure = None
    previous_observed_j8 = before
    last_raw_velocity = initial_raw_velocity
    last_derived_velocity: float | None = None
    stationarity_evidence = "raw_qvel"
    raw_qvel_incoherence_detected = False
    max_steps = max(
        1,
        int(math.ceil(float(timeout_s) * CONTROL_HZ)),
    )
    try:
        for step in range(max_steps):
            _check_cancelled(ctx, f"{stage}.j8_reset_{side}_{step}")
            observed_arm = np.asarray(
                ctx.world.arm_qpos_list(side),
                dtype=np.float64,
            ).reshape(-1)
            observed_vel = np.asarray(
                ctx.world.arm_qvel_list(side),
                dtype=np.float64,
            ).reshape(-1)
            if (
                observed_arm.size != ARM_DOF
                or observed_vel.size != ARM_DOF
                or not np.all(np.isfinite(observed_arm))
                or not np.all(np.isfinite(observed_vel))
            ):
                failure = "J8 reset lost finite arm proprioception"
                break

            error = -float(observed_arm[7])
            last_raw_velocity = float(observed_vel[7])
            last_derived_velocity = (
                abs(float(observed_arm[7]) - previous_observed_j8) * CONTROL_HZ
                if action_steps > 0
                else None
            )
            raw_velocity_settled = bool(
                abs(last_raw_velocity)
                <= WRIST_ROLL_VELOCITY_TOLERANCE_RAD_S
            )
            derived_velocity_settled = bool(
                last_derived_velocity is not None
                and last_derived_velocity
                <= WRIST_ROLL_VELOCITY_TOLERANCE_RAD_S
            )
            if derived_velocity_settled and not raw_velocity_settled:
                # The 250 kg velocity-driven base can enter a physics-substep
                # limit cycle whose instantaneous qvel aliases into every 30 Hz
                # observation even though sampled joint qpos is stationary.
                # Consecutive official qpos samples are the only independent,
                # observation-backed stationarity evidence available here.
                raw_qvel_incoherence_detected = True
                stationarity_evidence = "consecutive_qpos_delta"
            elif raw_velocity_settled:
                stationarity_evidence = "raw_qvel"
            settled = bool(
                abs(error) <= WRIST_ROLL_POSITION_TOLERANCE_RAD
                and (raw_velocity_settled or derived_velocity_settled)
            )
            stable_steps = stable_steps + 1 if settled else 0
            previous_observed_j8 = float(observed_arm[7])
            arm_command = fixed_arm.copy()
            arm_command[7] = 0.0
            yield ctx.world.make_action(
                **{f"arm_{side}": arm_command.tolist()}
            )
            action_steps += 1
            if stable_steps >= WRIST_ROLL_REQUIRED_STABLE_STEPS:
                converged = True
                break
    finally:
        zero_arm = fixed_arm.copy()
        zero_arm[7] = 0.0
        ctx.world.set_tool_roll_pin_qpos(side, 0.0)
        ctx.world.set_arm_pin_qpos(side, zero_arm)
        ctx.world.end_tool_roll_motion(side)

    observed_after = np.asarray(
        ctx.world.arm_qpos_list(side),
        dtype=np.float64,
    ).reshape(-1)
    after = (
        float(observed_after[7])
        if observed_after.size == ARM_DOF and np.all(np.isfinite(observed_after))
        else float("nan")
    )
    result = {
        "ok": bool(converged),
        "arm": side,
        "supported": True,
        "joint": f"{side}_arm_joint8",
        "before_deg": math.degrees(before),
        "after_deg": math.degrees(after) if math.isfinite(after) else None,
        "final_error_deg": -math.degrees(after) if math.isfinite(after) else None,
        "target_deg": 0.0,
        "lock_target_deg": 0.0,
        "locked_after_motion": True,
        "selected_j1_j7_fixed": True,
        "selected_j1_j7_max_drift_deg": (
            math.degrees(
                float(np.max(np.abs(observed_after[:7] - fixed_arm[:7])))
            )
            if observed_after.size == ARM_DOF
            and np.all(np.isfinite(observed_after))
            else None
        ),
        "action_steps": int(action_steps),
        "stable_steps": int(stable_steps),
        "timed_out": bool(not converged and action_steps >= max_steps),
        "initial_raw_velocity_rad_s": initial_raw_velocity,
        "final_raw_velocity_rad_s": last_raw_velocity,
        "final_derived_velocity_rad_s": last_derived_velocity,
        "stationarity_evidence": stationarity_evidence,
        "raw_qvel_incoherence_detected": bool(
            raw_qvel_incoherence_detected
        ),
    }
    if not converged:
        result["error"] = failure or "J8 did not converge to zero before timeout"
    return finish(result)



def _exec_gripper_qpos_is_open(qpos: Any) -> bool:
    try:
        observed = np.asarray(
            qpos if qpos is not None else [],
            dtype=np.float64,
        ).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(
        observed.size == 2
        and np.all(np.isfinite(observed))
        and np.all(
            observed
            >= GRIPPER_CLOSE_UPPER_LIMIT_M
            - GRIPPER_CLOSE_FULL_OPEN_QPOS_TOLERANCE_M
        )
    )



def _ensure_exec_selected_gripper_open(
    ctx,
    arm: str,
    *,
    result_out: dict[str, Any],
):
    """Open the execution gripper when needed without blocking arm execution."""
    side = str(arm).strip().lower()
    before = ctx.world.gripper_qpos_list(side)
    already_open = _exec_gripper_qpos_is_open(before)
    close_keepalive_before = bool(
        getattr(
            ctx.world,
            "gripper_close_keepalive_active",
            lambda _arm: False,
        )(side)
    )
    release_keepalive = getattr(
        ctx.world,
        "release_gripper_close_keepalive",
        None,
    )
    if callable(release_keepalive):
        release_keepalive(side)
    if LOCKED_J8_INDEX is not None:
        # Gripper-only frames still cross the same official action boundary;
        # the selected arm's locked J8 must therefore be zero from frame one.
        ctx.world.set_tool_roll_pin_qpos(side, LOCKED_J8_VALUE_RAD)

    uses_effort = bool(
        getattr(ctx.world, "gripper_uses_effort", lambda _arm: False)(side)
    )
    command = [GRIPPER_OPEN_EFFORT_N, GRIPPER_OPEN_EFFORT_N]
    action_steps = 0
    target_reached = bool(already_open)
    if not already_open:
        for _ in range(GRIPPER_OPEN_MAX_STEPS):
            _check_cancelled(ctx, f"exec_plan_pose open {side} gripper")
            if uses_effort:
                yield ctx.world.make_action(
                    **{f"gripper_effort_{side}": command}
                )
            else:
                yield ctx.world.make_action(**{f"gripper_{side}": [1.0]})
            action_steps += 1
            target_reached = _exec_gripper_qpos_is_open(
                ctx.world.gripper_qpos_list(side)
            )
            if target_reached:
                break

    hold_target = None
    if uses_effort:
        capture_hold = getattr(
            ctx.world,
            "capture_gripper_hold_qpos",
            None,
        )
        if callable(capture_hold):
            hold_target = capture_hold(
                side,
                allow_close_effort=False,
            )
    after = ctx.world.gripper_qpos_list(side)
    result_out.clear()
    result_out.update(
        {
            "arm": side,
            "checked_before_motion": True,
            "already_open": bool(already_open),
            "open_invoked": bool(not already_open),
            "open_action_steps": int(action_steps),
            "open_target_reached": bool(target_reached),
            "execution_continues": True,
            "gripper_qpos_before": before,
            "gripper_qpos_after": after,
            "commanded_gripper_action": (
                command if uses_effort else [1.0]
            ),
            "control_mode": (
                "independent_effort" if uses_effort else "position"
            ),
            "post_open_hold_qpos": hold_target,
            "close_keepalive_before": bool(close_keepalive_before),
            "close_keepalive_after": bool(
                getattr(
                    ctx.world,
                    "gripper_close_keepalive_active",
                    lambda _arm: False,
                )(side)
            ),
        }
    )


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
