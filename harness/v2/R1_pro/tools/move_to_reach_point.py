"""move_to_reach_point

点选 (u,v) 反解 3D reach point，并移动到底盘/肩部可达位置；keep_ori_arm 仅在底盘旋转完成后的最终俯仰阶段生效，保持原 J1-J4 轨迹不变并逐帧仅用 J5-J7 追踪入口世界系 EEF 姿态；EEF 平移只监控。

How this tool works:
Project the selected object point into 3D. Compare it with the shoulders and arm reach. Plan and execute a guarded base displacement, including the existing recovery trajectory when necessary; report the achieved reach geometry.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.action_constants import ADJUST_CHASSIS_BASE_MAX_ANG_RADPS
from tools.util.action_constants import ADJUST_CHASSIS_BASE_MAX_LIN_MPS
from tools.util.action_constants import ADJUST_HEIGHT_DEFAULT_TOLERANCE_RAD
from tools.util.action_constants import CONTROL_HZ
from tools.util.action_constants import REACH_MAX_SHOULDER_DISTANCE_M
from tools.util.action_constants import TRUNK_LIMITS
from tools.util.action_constants import _ARM_LIMITS
from tools.util.geometry.contract import ARM_DOF
from tools.util.motion_common import _check_cancelled
from tools.util.motion_common import _load_frozen_capture
from tools.util.motion_common import _normalize_angle
from tools.util.motion_common import _point_from_relative_uv
from tools.util.motion_common import _point_in_robot_frame
from tools.util.motion_common import _quat_to_mat_xyzw
from tools.util.motion_common import _r1pro_arm_fk_robot
from tools.util.motion_common import _r1pro_chest_pose_robot
from tools.util.motion_common import _r1pro_shoulder_positions_robot
from tools.util.motion_common import _reach_observation_sequence
from tools.util.motion_common import _rotation_vector
from tools.util.motion_common import _set_result
from tools.util.motion_common import _smoothstep
from tools.util.motion_common import _yield_adjust_height_motion
from typing import Any
import math
import numpy as np
import time
from tools.util.tool import main

NAME = 'move_to_reach_point'
SCHEMA = {'type': 'object',
 'properties': {'image_id': {'type': 'string',
                             'description': 'The latest clickable head-camera image_id, actually '
                                            '720 x 720; re-ground the point when image_id changes. '
                                            'Wrist images are not clickable.'},
                'u': {'type': 'integer',
                      'minimum': 0,
                      'maximum': 719,
                      'description': 'Integer original-image pixel column in the latest 720 x 720 '
                                     'head image: 0 is the left edge and 719 is the right edge. Do '
                                     'not normalize or rescale.'},
                'v': {'type': 'integer',
                      'minimum': 0,
                      'maximum': 719,
                      'description': 'Integer original-image pixel row in the latest 720 x 720 '
                                     'head image: 0 is the top edge and 719 is the bottom edge. Do '
                                     'not normalize or rescale.'},
                'reach': {'type': 'number', 'default': 0.0, 'description': 'm(弦球半径，0=默认0.60m)'},
                'nav_timeout_s': {'type': 'number',
                                  'default': 120.0,
                                  'description': 's(底盘闭环导航超时，默认120)'},
                'keep_ori_arm': {'type': 'string',
                                 'enum': ['none', 'left', 'right', 'both'],
                                 'default': 'none'}},
 'additionalProperties': False,
 'required': ['image_id', 'u', 'v']}


def actions(
    robot,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    reach: float = 0.0,
    nav_timeout_s: float = 120.0,
    keep_ori_arm: str = "none",
    **kwargs,
):
    del kwargs
    tool_started_s = time.monotonic()
    tool_start_sequence = _reach_observation_sequence(robot)
    stage_metrics: dict[str, dict[str, Any]] = {}
    phase_started_s = time.monotonic()
    phase_start_sequence = _reach_observation_sequence(robot)
    try:
        capture = _load_frozen_capture(session_id, image_id)
        target_policy, point_meta = _point_from_relative_uv(capture, u, v)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        _set_result(
            robot,
            "move_to_reach_point",
            {
                "ok": False,
                "error": f"failed to load clicked evaluator RGB-D point: {exc}",
            },
        )
        yield robot.hold_action()
        return
    stage_metrics["capture_and_point"] = _reach_phase_metrics(
        robot,
        started_s=phase_started_s,
        start_sequence=phase_start_sequence,
        action_steps=0,
    )

    reach_m = float(reach)
    if reach_m <= 0.01:
        reach_m = REACH_DEFAULT_M
    if not REACH_MIN_M <= reach_m <= REACH_MAX_M:
        _set_result(
            robot,
            "move_to_reach_point",
            {
                "ok": False,
                "error": (
                    f"reach must be in {REACH_MIN_M:.2f}..{REACH_MAX_M:.2f}m"
                ),
            },
        )
        yield robot.hold_action()
        return

    keep_value = str(keep_ori_arm or "none").strip().lower()
    if keep_value in ("", "false", "0", "no"):
        keep_value = "none"
    if keep_value not in ("none", "left", "right", "both"):
        _set_result(
            robot,
            "move_to_reach_point",
            {
                "ok": False,
                "error": "keep_ori_arm must be none, left, right, or both",
            },
        )
        yield robot.hold_action()
        return

    phase_started_s = time.monotonic()
    phase_start_sequence = _reach_observation_sequence(robot)
    start_pose = robot.robot_pose()
    trunk_start = np.asarray(
        robot.trunk_qpos(),
        dtype=np.float64,
    ).reshape(4)
    pre_lift_upward_m, pre_lift_report = _reach_pre_lift_request(
        float(target_policy[2]),
        trunk_start,
    )
    pre_lift_target = trunk_start.copy()
    pre_lift_steps = 0
    if pre_lift_report["applied"]:
        lift_actions = _yield_adjust_height_motion(
            robot,
            upward=pre_lift_upward_m,
            timeout_s=max(1.0, float(nav_timeout_s)),
            tolerance=ADJUST_HEIGHT_DEFAULT_TOLERANCE_RAD,
            stage_prefix="move-to-reach pre-lift",
        )
        while True:
            try:
                yield next(lift_actions)
            except StopIteration as stop:
                lift_report = stop.value
                break
        pre_lift_steps = int(lift_report.get("action_steps") or 0)
        pre_lift_target = np.asarray(
            robot.trunk_qpos(),
            dtype=np.float64,
        ).reshape(4)
        shoulders_after_lift = _r1pro_shoulder_positions_robot(
            pre_lift_target
        )
        shoulder_mid_after_lift = 0.5 * (
            shoulders_after_lift["left"] + shoulders_after_lift["right"]
        )
        pre_lift_report.update(
            {
                "ok": bool(lift_report.get("ok")),
                "shoulder_z_after_m": float(shoulder_mid_after_lift[2]),
                "dz_after_m": float(
                    shoulder_mid_after_lift[2] - float(target_policy[2])
                ),
                "actual_upward_m": lift_report.get("actual_upward_m"),
                "target_trunk_q": lift_report.get("target_trunk_q"),
                "achieved_trunk_q": pre_lift_target.astype(float).tolist(),
                "adjust_height": lift_report,
                "execution": "shared_adjust_height_internal_executor",
            }
        )
        if not lift_report.get("ok"):
            stage_metrics["pre_lift"] = _reach_phase_metrics(
                robot,
                started_s=phase_started_s,
                start_sequence=phase_start_sequence,
                action_steps=pre_lift_steps,
            )
            _set_result(
                robot,
                "move_to_reach_point",
                {
                    "ok": False,
                    "error": (
                        "move-to-reach pre-lift failed: "
                        + str(
                            lift_report.get(
                                "error",
                                "adjust-height trajectory did not converge",
                            )
                        )
                    ),
                    "target_point_policy_local_m": (
                        target_policy.astype(float).tolist()
                    ),
                    "point_observation": point_meta,
                    "pre_lift": pre_lift_report,
                    "direct_simulator_mutation": False,
                },
            )
            yield robot.hold_action()
            return
    else:
        pre_lift_report.update(
            {
                "ok": True,
                "achieved_trunk_q": pre_lift_target.astype(float).tolist(),
                "execution": "skipped_within_height_band",
            }
        )
    stage_metrics["pre_lift"] = _reach_phase_metrics(
        robot,
        started_s=phase_started_s,
        start_sequence=phase_start_sequence,
        action_steps=pre_lift_steps,
    )

    phase_started_s = time.monotonic()
    phase_start_sequence = _reach_observation_sequence(robot)
    try:
        chord_plan = _reach_chord_plan(
            target_policy,
            pre_lift_target,
            reach_m,
        )
        base_goal = _pick_reach_base_goal(
            target_policy_local=target_policy,
            start_pose=robot.robot_pose(),
            chord_plan=chord_plan,
        )
    except (KeyError, TypeError, ValueError) as exc:
        _set_result(
            robot,
            "move_to_reach_point",
            {
                "ok": False,
                "error": f"observation-only reach planning failed: {exc}",
                "target_point_policy_local_m": target_policy.astype(float).tolist(),
                "point_observation": point_meta,
                "pre_lift": pre_lift_report,
            },
        )
        yield robot.hold_action()
        return
    stage_metrics["reach_planning"] = _reach_phase_metrics(
        robot,
        started_s=phase_started_s,
        start_sequence=phase_start_sequence,
        action_steps=0,
    )

    target_base_xy = np.asarray(base_goal["base_xy"], dtype=np.float64).reshape(2)
    target_base_yaw = float(base_goal["yaw_rad"])
    adapter = getattr(robot, "_official_adapter", None)
    phase_started_s = time.monotonic()
    phase_start_sequence = _reach_observation_sequence(robot)
    base_actions = _yield_reach_base_xy_yaw_controller(
        robot,
        bx=float(target_base_xy[0]),
        by=float(target_base_xy[1]),
        target_yaw_rad=target_base_yaw,
        timeout_s=max(1.0, float(nav_timeout_s)),
        pos_tol_m=REACH_XY_TOL_M,
        yaw_tol_deg=REACH_YAW_TOL_DEG,
    )
    while True:
        try:
            yield next(base_actions)
        except StopIteration as stop:
            base_report = stop.value
            break
    consumed_steps = int(base_report.get("action_steps") or 0)
    stage_metrics["base_motion"] = _reach_phase_metrics(
        robot,
        started_s=phase_started_s,
        start_sequence=phase_start_sequence,
        action_steps=consumed_steps,
    )
    xy_steps = consumed_steps
    path_align_steps = 0
    forward_steps = 0
    yaw_steps = 0
    xy_ok = bool(base_report.get("xy_ok"))
    yaw_ok = bool(base_report.get("yaw_ok"))
    safety_abort = (
        base_report.get("reason") if not base_report.get("ok") else None
    )
    live_guard_last = None
    collision_reach_recovery = False

    keep_arms = (
        []
        if keep_value == "none"
        else ["left", "right"]
        if keep_value == "both"
        else [keep_value]
    )
    keep_states: dict[str, dict[str, Any]] = {}
    raw_eef = {} if adapter is None else adapter.eef_pose()
    for side in keep_arms:
        arm_q = np.asarray(
            robot.arm_qpos_list(side),
            dtype=np.float64,
        ).reshape(-1)
        if arm_q.size != ARM_DOF:
            continue
        entry_position, model_rotation = _r1pro_arm_fk_robot(
            side,
            pre_lift_target,
            arm_q,
        )
        eef_quat = np.asarray(
            (raw_eef.get(side) or {}).get("quat", []),
            dtype=np.float64,
        ).reshape(-1)
        target_rotation = (
            _quat_to_mat_xyzw(eef_quat)
            if eef_quat.size == 4 and float(np.linalg.norm(eef_quat)) > 1e-8
            else model_rotation
        )
        keep_states[side] = {
            "fixed_q": arm_q.copy(),
            "seed_q": arm_q.copy(),
            "target_rotation": target_rotation,
            "entry_position": entry_position,
            "max_position_drift_m": 0.0,
            "max_orientation_residual_deg": 0.0,
            "all_solutions_ok": True,
        }

    pitch_start = pre_lift_target.copy()
    pitch_target = np.asarray(
        chord_plan["target_trunk_q"],
        dtype=np.float64,
    ).reshape(4)
    pitch_gap = float(np.linalg.norm(pitch_target - pitch_start, ord=np.inf))
    pitch_steps = _reach_pitch_interpolation_steps(pitch_gap)
    pitch_executed = bool(xy_ok and yaw_ok) or collision_reach_recovery
    pitch_action_steps = 0
    phase_started_s = time.monotonic()
    phase_start_sequence = _reach_observation_sequence(robot)
    if pitch_executed:
        for index in range(1, pitch_steps + 1):
            _check_cancelled(robot, "move-to-reach final pitch")
            alpha = _smoothstep(index / pitch_steps)
            trunk_command = pitch_start + alpha * (pitch_target - pitch_start)
            yield robot.make_action(
                **_reach_trunk_action_overrides(trunk_command, keep_states)
            )
            pitch_action_steps += 1
    stage_metrics["final_pitch"] = _reach_phase_metrics(
        robot,
        started_s=phase_started_s,
        start_sequence=phase_start_sequence,
        action_steps=pitch_action_steps,
    )

    phase_started_s = time.monotonic()
    phase_start_sequence = _reach_observation_sequence(robot)
    compensation_actions = _yield_reach_compensation(
        robot,
        target_policy_local=target_policy,
        keep_states=keep_states,
        max_dist_m=REACH_MAX_SHOULDER_DISTANCE_M,
    )
    compensation_action_steps = 0
    while True:
        try:
            yield next(compensation_actions)
            compensation_action_steps += 1
        except StopIteration as stop:
            reach_compensation = stop.value
            break
    stage_metrics["reach_compensation"] = _reach_phase_metrics(
        robot,
        started_s=phase_started_s,
        start_sequence=phase_start_sequence,
        action_steps=compensation_action_steps,
    )
    phase_started_s = time.monotonic()
    phase_start_sequence = _reach_observation_sequence(robot)
    final_measurement = _reach_same_target_measurement(robot, target_policy)
    trunk_for_measurement = np.asarray(
        final_measurement["trunk_q"],
        dtype=np.float64,
    ).reshape(4)
    target_robot_final = np.asarray(
        final_measurement["target_robot_m"],
        dtype=np.float64,
    ).reshape(3)
    final_target_meta = {
        "frame": "robot_base",
        "source": str(final_measurement["source"]),
        "tracking": None,
        "target_identity": "immutable_frozen_RGB-D_click",
    }
    final_shoulders = final_measurement["shoulders"]
    shoulder_distances = {
        "left": float(final_measurement["left_m"]),
        "right": float(final_measurement["right_m"]),
    }
    minimum_distance = float(final_measurement["minimum_m"])
    final_pose = robot.robot_pose()
    keep_report = {
        side: {
            "ok": bool(state["all_solutions_ok"]),
            "j1234_policy": "unchanged_from_pitch_entry",
            "j567_policy": "local_orientation_tracking",
            "j8_policy": "unchanged_from_pitch_entry",
            "max_orientation_residual_deg": float(
                state["max_orientation_residual_deg"]
            ),
            "max_predicted_position_drift_m": float(
                state["max_position_drift_m"]
            ),
            "translation_policy": "monitored_only",
        }
        for side, state in keep_states.items()
    }
    keep_ok = bool(
        len(keep_states) == len(keep_arms)
        and all(item["ok"] for item in keep_report.values())
    )
    if not keep_arms:
        keep_ok = True
    reach_ok_threshold_m = REACH_MAX_SHOULDER_DISTANCE_M
    reach_ok = bool(minimum_distance <= reach_ok_threshold_m)
    result_ok = bool(reach_ok and pitch_executed and keep_ok)
    stage_metrics["final_measurement"] = _reach_phase_metrics(
        robot,
        started_s=phase_started_s,
        start_sequence=phase_start_sequence,
        action_steps=0,
    )
    tool_end_sequence = _reach_observation_sequence(robot)
    stage_timing = {
        "clock": "monotonic",
        "control_hz": float(CONTROL_HZ),
        "phases": stage_metrics,
        "total_wall_s_to_result": round(
            max(0.0, time.monotonic() - tool_started_s),
            6,
        ),
        "total_action_steps_to_result": int(
            pre_lift_steps
            + consumed_steps
            + pitch_action_steps
            + compensation_action_steps
        ),
        "observation_sequence_start": tool_start_sequence,
        "observation_sequence_end": tool_end_sequence,
        "observation_delta": (
            None
            if tool_start_sequence is None or tool_end_sequence is None
            else max(0, tool_end_sequence - tool_start_sequence)
        ),
        "final_result_hold_action_excluded": True,
    }
    _set_result(
        robot,
        "move_to_reach_point",
        {
            "ok": result_ok,
            "error": (
                None
                if result_ok
                else "observation-derived reach estimate did not satisfy all constraints"
            ),
            "image_id": str(image_id),
            "relative_uv": [float(u), float(v)],
            "target_point_policy_local_m": target_policy.astype(float).tolist(),
            "target_envelope_policy_local_m": {
                "min": (
                    target_policy - REACH_TARGET_ENVELOPE_M
                ).astype(float).tolist(),
                "max": (
                    target_policy + REACH_TARGET_ENVELOPE_M
                ).astype(float).tolist(),
                "source": "synthetic_envelope_around_clicked_point",
            },
            "point_observation": point_meta,
            "target_current_estimate": final_target_meta,
            "reach_m": float(reach_m),
            "stage_timing": stage_timing,
            "pre_lift": pre_lift_report,
            "chord_plan": {
                "target_trunk_q": pitch_target.astype(float).tolist(),
                "target_offset_robot_m": np.asarray(
                    chord_plan["target_offset_robot_m"],
                    dtype=np.float64,
                ).astype(float).tolist(),
                "chest_forward_robot_m": np.asarray(
                    chord_plan["chest_forward_robot_m"],
                    dtype=np.float64,
                ).astype(float).tolist(),
                "pitch_branch": str(chord_plan["pitch_branch"]),
                "forward_candidate_count": int(
                    chord_plan["forward_candidate_count"]
                ),
                "rejected_back_facing_candidate_count": int(
                    chord_plan["rejected_back_facing_candidate_count"]
                ),
                "shoulder_span_m": float(chord_plan["shoulder_span_m"]),
                "longitudinal_chord_m": float(
                    chord_plan["longitudinal_chord_m"]
                ),
                "left_distance_m": float(chord_plan["left_distance_m"]),
                "right_distance_m": float(chord_plan["right_distance_m"]),
                "target_z_error_m": float(chord_plan["target_z_error_m"]),
            },
            "base_plan": {
                "start_odometry": start_pose.as_dict(),
                "target_xy_policy_local_m": target_base_xy.astype(float).tolist(),
                "target_yaw_deg": math.degrees(target_base_yaw),
                "path_clearance_m": None,
                "required_clearance_m": None,
                "clearance_source": str(base_goal["clearance_source"]),
                "pick_policy": str(base_goal["pick_policy"]),
                "selected_yaw_offset_deg": int(base_goal["yaw_offset_deg"]),
                "candidate_count": int(base_goal["candidate_count"]),
                "clear_candidate_count": int(
                    base_goal["clear_candidate_count"]
                ),
            },
            "base_execution": {
                "motion_model": "v2_signed_holonomic_xy_yaw",
                "lateral_action_locked": False,
                "requested_speed_scale": float(
                    REACH_BASE_COMMAND_SPEED_SCALE
                ),
                "requested_max_linear_mps": float(
                    REACH_BASE_MAX_SPEED_MPS
                ),
                "requested_max_yaw_rate_radps": float(
                    REACH_BASE_MAX_YAW_RATE_RADPS
                ),
                "official_max_linear_mps": float(
                    REACH_BASE_OFFICIAL_MAX_SPEED_MPS
                ),
                "official_max_yaw_rate_radps": float(
                    REACH_BASE_OFFICIAL_MAX_YAW_RATE_RADPS
                ),
                "official_action_limit_respected": True,
                "xy_ok": bool(xy_ok),
                "yaw_ok": bool(yaw_ok),
                "collision_reach_recovery": collision_reach_recovery,
                "xy_action_steps": int(xy_steps),
                "path_align_action_steps": int(path_align_steps),
                "forward_action_steps": int(forward_steps),
                "yaw_action_steps": int(yaw_steps),
                "safety_abort": safety_abort,
                "last_rgbd_guard": live_guard_last,
                "rgbd_motion_guard_used": False,
                "controller_report": base_report,
                "final_odometry": final_pose.as_dict(),
                "verification": (
                    "v2 signed holonomic control over policy-local odometry and base_qvel"
                ),
                "unavailable_v2_truth_guards": [
                    "simulator_contacts",
                    "global_base_z_vz_tilt",
                    "simulator_collision_truth",
                ],
            },
            "pitch_execution": {
                "executed": bool(pitch_executed),
                "action_steps": int(pitch_action_steps),
                "pitch_gap_rad": float(pitch_gap),
                "noop_skipped": bool(
                    pitch_executed and pitch_action_steps == 0
                ),
                "target_trunk_q": pitch_target.astype(float).tolist(),
                "final_commanded_trunk_q": np.asarray(
                    trunk_for_measurement,
                    dtype=np.float64,
                ).astype(float).tolist(),
                "post_reach_compensation_applied": bool(
                    reach_compensation.get("attempted")
                    and reach_compensation["phase2_pitch"]["steps"]
                ),
            },
            "reach_compensation": reach_compensation,
            "shoulder_distance_estimate": {
                "left_m": float(shoulder_distances["left"]),
                "right_m": float(shoulder_distances["right"]),
                "minimum_m": float(minimum_distance),
                "target_m": float(reach_m),
                "error_to_target_m": float(minimum_distance - float(reach_m)),
                "threshold_m": reach_ok_threshold_m,
                "ok": reach_ok,
                "source": str(final_measurement["source"]),
            },
            "keep_ori_requested_arm": keep_value,
            "keep_ori_scope": "final_trunk_pitch_after_base_yaw",
            "keep_ori_ok": keep_ok,
            "keep_ori": keep_report,
            "observation_contract": [
                "head_rgb",
                "head_depth_linear",
                "camera_relative_pose",
                "proprioception",
                "policy_owned_odometry",
                "submission_local_static_robot_model",
            ],
            "global_pose_available": False,
            "scene_geometry_truth_available": False,
            "collision_truth_available": False,
            "segmentation_used": False,
            "object_aabb_used": False,
            "direct_simulator_mutation": False,
        },
    )
    yield robot.hold_action()


MOVE_TO_REACH_PRE_LIFT_IMPLEMENTATION_VERSION = (
    "shared_adjust_height_absolute_lut_sampled_18_v2"
)



REACH_DEFAULT_M = 0.60



REACH_MIN_M = 0.35



REACH_MAX_M = 0.90



REACH_PRE_LIFT_MAX_DZ_M = 0.40



REACH_XY_TOL_M = 0.08



REACH_YAW_TOL_DEG = 3.0



REACH_TRUNK_TOL_RAD = math.radians(3.0)



REACH_BASE_COMMAND_SPEED_SCALE = 1.30



REACH_BASE_OFFICIAL_MAX_SPEED_MPS = 0.75



REACH_BASE_OFFICIAL_MAX_YAW_RATE_RADPS = 1.0



REACH_BASE_MAX_SPEED_MPS = (
    REACH_BASE_OFFICIAL_MAX_SPEED_MPS * REACH_BASE_COMMAND_SPEED_SCALE
)



REACH_BASE_MAX_YAW_RATE_RADPS = (
    REACH_BASE_OFFICIAL_MAX_YAW_RATE_RADPS * REACH_BASE_COMMAND_SPEED_SCALE
)



REACH_PITCH_NOOP_EPS_RAD = 1e-6



REACH_BASE_MAX_LIN_ACCEL_MPS2 = 2.40



REACH_BASE_MAX_ANG_ACCEL_RADPS2 = 3.00



REACH_BASE_LIN_SLOW_RADIUS_M = 0.55



REACH_BASE_ANG_SLOW_RADIUS_RAD = math.radians(30.0)



REACH_BASE_BRAKE_DECEL_SAFETY = 0.75



REACH_BASE_GOAL_STOP_MARGIN_M = 0.015



REACH_BASE_GOAL_PLANE_TOL_M = 0.010



REACH_BASE_GOAL_HARD_STOP_STEPS = 6



REACH_BASE_STUCK_MIN_LIN_CMD_MPS = 0.05



REACH_BASE_STUCK_MIN_ANG_CMD_RADPS = 0.07



REACH_BASE_STUCK_WINDOW_S = 0.80



REACH_BASE_STUCK_XY_PROGRESS_M = 0.015



REACH_BASE_STUCK_YAW_PROGRESS_DEG = 1.5



REACH_TARGET_ENVELOPE_M = 0.03



REACH_FORWARD_AXIS_EPS = 1e-9



REACH_TARGET_FORWARD_EPS_M = 1e-6



REACH_COMP_FORWARD_STEP_M = 0.05



REACH_COMP_FORWARD_MAX_STEPS = 15



REACH_COMP_FORWARD_TIMEOUT_S = 20.0



REACH_COMP_FORWARD_TOL_M = 0.01



REACH_COMP_PITCH_STEP_DEG = 5.0



REACH_COMP_PITCH_MAX_DEG = 30.0



REACH_COMP_Q3_MIN_RAD = -1.8326



REACH_COMP_MAX_CHEST_PITCH_DEG = 165.0



REACH_COMP_IMPROVEMENT_EPS_M = 0.001



def _yaw_matrix(yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )



def _solve_j567_keep_orientation(
    *,
    arm: str,
    trunk_q,
    fixed_arm_q,
    seed_arm_q,
    target_rotation,
    tolerance_deg: float = 2.0,
    max_iterations: int = 24,
) -> tuple[np.ndarray, dict[str, Any]]:
    side = str(arm).strip().lower()
    fixed = np.asarray(fixed_arm_q, dtype=np.float64).reshape(-1)
    seed = np.asarray(seed_arm_q, dtype=np.float64).reshape(-1)
    if fixed.size != ARM_DOF or seed.size != ARM_DOF:
        raise ValueError(
            f"{side} arm model requires {ARM_DOF} values, "
            f"got fixed={fixed.size} seed={seed.size}"
        )
    command = fixed.copy()
    command[4:7] = seed[4:7]
    if ARM_DOF == 8:
        command[7] = fixed[7]
    limits = _ARM_LIMITS[side][:ARM_DOF]
    target = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
    residual = float("inf")
    iterations = 0
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        _, current = _r1pro_arm_fk_robot(side, trunk_q, command)
        error = _rotation_vector(target @ current.T)
        residual = math.degrees(float(np.linalg.norm(error)))
        if residual <= float(tolerance_deg):
            break
        jacobian = np.zeros((3, 3), dtype=np.float64)
        epsilon = 1e-5
        for column, joint_index in enumerate(range(4, 7)):
            probe = command.copy()
            probe[joint_index] += epsilon
            _, rotation_plus = _r1pro_arm_fk_robot(side, trunk_q, probe)
            jacobian[:, column] = (
                _rotation_vector(rotation_plus @ current.T) / epsilon
            )
        damping = 2e-3
        lhs = jacobian @ jacobian.T + damping * np.eye(3)
        delta = jacobian.T @ np.linalg.solve(lhs, error)
        delta = np.clip(delta, -0.12, 0.12)
        command[4:7] = np.clip(
            command[4:7] + delta,
            limits[4:7, 0],
            limits[4:7, 1],
        )
        command[:4] = fixed[:4]
        if ARM_DOF == 8:
            command[7] = fixed[7]
    _, final_rotation = _r1pro_arm_fk_robot(side, trunk_q, command)
    residual = math.degrees(
        float(np.linalg.norm(_rotation_vector(target @ final_rotation.T)))
    )
    return command, {
        "ok": bool(residual <= float(tolerance_deg)),
        "residual_deg": float(residual),
        "iterations": int(iterations),
        "j1234_unchanged": bool(
            np.allclose(command[:4], fixed[:4], atol=1e-10)
        ),
        "j8_unchanged": bool(
            ARM_DOF != 8 or abs(float(command[7] - fixed[7])) <= 1e-10
        ),
    }



def _reach_trunk_action_overrides(
    trunk_command,
    keep_states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    command = np.asarray(trunk_command, dtype=np.float64).reshape(4)
    overrides: dict[str, Any] = {"trunk": command.tolist()}
    for side, state in keep_states.items():
        arm_command, solve = _solve_j567_keep_orientation(
            arm=side,
            trunk_q=command,
            fixed_arm_q=state["fixed_q"],
            seed_arm_q=state["seed_q"],
            target_rotation=state["target_rotation"],
        )
        state["seed_q"] = arm_command
        state["all_solutions_ok"] = bool(
            state["all_solutions_ok"] and solve["ok"]
        )
        state["max_orientation_residual_deg"] = max(
            float(state["max_orientation_residual_deg"]),
            float(solve["residual_deg"]),
        )
        predicted_position, _ = _r1pro_arm_fk_robot(
            side,
            command,
            arm_command,
        )
        state["max_position_drift_m"] = max(
            float(state["max_position_drift_m"]),
            float(
                np.linalg.norm(
                    predicted_position - state["entry_position"]
                )
            ),
        )
        overrides[f"arm_{side}"] = arm_command.tolist()
    return overrides



def _reach_pre_lift_request(target_z: float, trunk_q) -> tuple[float, dict[str, Any]]:
    """Convert the v2 shoulder-height band into an adjust-height request."""
    start = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    shoulders = _r1pro_shoulder_positions_robot(start)
    start_mid = 0.5 * (shoulders["left"] + shoulders["right"])
    start_dz = float(start_mid[2] - float(target_z))
    if abs(start_dz) <= REACH_PRE_LIFT_MAX_DZ_M:
        return 0.0, {
            "applied": False,
            "shoulder_z_before_m": float(start_mid[2]),
            "shoulder_z_after_m": float(start_mid[2]),
            "target_z_m": float(target_z),
            "dz_before_m": start_dz,
            "dz_after_m": start_dz,
            "requested_upward_m": 0.0,
            "planner": MOVE_TO_REACH_PRE_LIFT_IMPLEMENTATION_VERSION,
        }
    desired_z = float(target_z) + math.copysign(
        REACH_PRE_LIFT_MAX_DZ_M,
        start_dz,
    )
    requested_upward_m = float(desired_z - float(start_mid[2]))
    return requested_upward_m, {
        "applied": abs(requested_upward_m) > 1e-5,
        "shoulder_z_before_m": float(start_mid[2]),
        "desired_shoulder_z_m": float(desired_z),
        "target_z_m": float(target_z),
        "dz_before_m": start_dz,
        "requested_upward_m": requested_upward_m,
        "planner": MOVE_TO_REACH_PRE_LIFT_IMPLEMENTATION_VERSION,
    }



def _reach_chord_plan(
    target_policy_local,
    trunk_q,
    reach_m: float,
) -> dict[str, Any]:
    target = np.asarray(target_policy_local, dtype=np.float64).reshape(3)
    q_start = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    reach = float(reach_m)
    shoulder_start = _r1pro_shoulder_positions_robot(q_start)
    shoulder_span = float(
        np.linalg.norm(shoulder_start["left"] - shoulder_start["right"])
    )
    half_span = 0.5 * shoulder_span
    if reach <= half_span + 1e-6:
        raise ValueError(
            f"reach={reach:.3f}m must exceed shoulder half-span={half_span:.3f}m"
        )
    longitudinal = math.sqrt(max(0.0, reach * reach - half_span * half_span))
    candidates: list[dict[str, Any]] = []
    rejected_back_facing = 0
    for q3 in np.linspace(TRUNK_LIMITS[2, 0], TRUNK_LIMITS[2, 1], 1081):
        q = q_start.copy()
        q[2] = float(q3)
        shoulders = _r1pro_shoulder_positions_robot(q)
        midpoint = 0.5 * (shoulders["left"] + shoulders["right"])
        _, chest_rotation = _r1pro_chest_pose_robot(q)
        chest_forward = np.asarray(chest_rotation[:, 0], dtype=np.float64)
        desired_target_robot = midpoint + longitudinal * chest_forward
        # Unsigned pitch angles alias a back-facing q3 mirror; v2 reaches forward.
        if (
            float(chest_forward[0]) < -REACH_FORWARD_AXIS_EPS
            or float(desired_target_robot[0]) <= REACH_TARGET_FORWARD_EPS_M
        ):
            rejected_back_facing += 1
            continue
        z_error = float(desired_target_robot[2] - target[2])
        candidates.append(
            {
                "score": abs(z_error) + 1e-4 * abs(float(q3 - q_start[2])),
                "q": q,
                "midpoint": midpoint,
                "desired_target_robot": desired_target_robot,
                "chest_forward_robot": chest_forward,
                "z_error": z_error,
                "left": shoulders["left"],
                "right": shoulders["right"],
            }
        )
    if not candidates:
        raise ValueError(
            "submission-local chord geometry has no forward-facing q3 solution"
        )
    best = min(candidates, key=lambda item: item["score"])
    desired = np.asarray(best["desired_target_robot"], dtype=np.float64)
    horizontal_offset = math.atan2(float(desired[1]), float(desired[0]))
    return {
        "target_trunk_q": np.asarray(best["q"], dtype=np.float64),
        "target_offset_robot_m": desired,
        "target_offset_bearing_rad": float(horizontal_offset),
        "chest_forward_robot_m": np.asarray(
            best["chest_forward_robot"], dtype=np.float64
        ),
        "pitch_branch": "forward_half_plane",
        "forward_candidate_count": len(candidates),
        "rejected_back_facing_candidate_count": int(rejected_back_facing),
        "shoulder_mid_robot_m": np.asarray(
            best["midpoint"],
            dtype=np.float64,
        ),
        "shoulder_span_m": shoulder_span,
        "longitudinal_chord_m": longitudinal,
        "reach_m": reach,
        "target_z_error_m": float(best["z_error"]),
        "left_distance_m": float(
            np.linalg.norm(desired - np.asarray(best["left"], dtype=np.float64))
        ),
        "right_distance_m": float(
            np.linalg.norm(desired - np.asarray(best["right"], dtype=np.float64))
        ),
        "model": "submission-local R1Pro chord-sphere geometry",
    }



def _pick_reach_base_goal(
    *,
    target_policy_local,
    start_pose,
    chord_plan: dict[str, Any],
) -> dict[str, Any]:
    """Select the nearest of the same 13 chord slices used by Interface v2.

    The official observation contract exposes no scene-graph or collision
    clearance callback. Interface v2 treats that callback as optional and
    assigns every candidate free clearance when it is absent; this test-local
    implementation follows that branch instead of substituting RGB-D.
    """
    target = np.asarray(target_policy_local, dtype=np.float64).reshape(3)
    start_xy = np.asarray(start_pose.pos[:2], dtype=np.float64).reshape(2)
    offset_robot = np.asarray(
        chord_plan["target_offset_robot_m"],
        dtype=np.float64,
    ).reshape(3)
    target_delta = target[:2] - start_xy
    target_bearing = math.atan2(
        float(target_delta[1]),
        float(target_delta[0]),
    )
    nominal_yaw = _normalize_angle(
        target_bearing - float(chord_plan["target_offset_bearing_rad"])
    )
    candidate_yaws = [
        _normalize_angle(nominal_yaw + math.radians(offset))
        for offset in (
            0,
            15,
            -15,
            30,
            -30,
            45,
            -45,
            60,
            -60,
            90,
            -90,
            120,
            -120,
        )
    ]
    options: list[dict[str, Any]] = []
    for offset_deg, yaw in zip(
        (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120),
        candidate_yaws,
    ):
        base_xy = target[:2] - (_yaw_matrix(yaw) @ offset_robot)[:2]
        travel = float(np.linalg.norm(base_xy - start_xy))
        yaw_delta = abs(_normalize_angle(float(yaw) - float(start_pose.yaw)))
        options.append(
            {
                "base_xy": base_xy,
                "yaw_rad": float(yaw),
                "travel_m": travel,
                "yaw_delta_rad": yaw_delta,
                "yaw_offset_deg": int(offset_deg),
            }
        )
    selected = min(
        options,
        key=lambda item: (
            round(float(item["travel_m"]), 6),
            abs(float(item["yaw_offset_deg"])) * 0.001,
            float(item["yaw_delta_rad"]),
        ),
    )
    return {
        **selected,
        "clearance_m": None,
        "clear": True,
        "required_clearance_m": None,
        "clearance_source": "v2_optional_nav_clearance_unavailable",
        "pick_policy": "nearest_chord_travel_first",
        "candidate_count": len(options),
        "clear_candidate_count": len(options),
        "all_candidates": [
            {
                "base_xy": item["base_xy"].astype(float).tolist(),
                "yaw_deg": math.degrees(item["yaw_rad"]),
                "yaw_offset_deg": int(item["yaw_offset_deg"]),
                "clearance_m": None,
                "clear": True,
                "travel_m": float(item["travel_m"]),
            }
            for item in options
        ],
    }



def _base_action_from_physical_velocity(ctx, physical_cmd):
    """Map robot-frame physical velocity to the submitted controller action."""
    command = np.asarray(physical_cmd, dtype=np.float64).reshape(3)
    scales = np.array(
        [
            ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
            ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
            ADJUST_CHASSIS_BASE_MAX_ANG_RADPS,
        ],
        dtype=np.float64,
    )
    action = np.clip(command / scales, -1.0, 1.0)
    return ctx.world.set_base_velocity(*action.astype(float).tolist())



def _reach_pitch_interpolation_steps(pitch_gap_rad: float) -> int:
    gap = abs(float(pitch_gap_rad))
    if gap <= REACH_PITCH_NOOP_EPS_RAD:
        return 0
    return max(2, int(math.ceil(gap / 0.10)))



def _reach_phase_metrics(
    ctx,
    *,
    started_s: float,
    start_sequence: int | None,
    action_steps: int,
) -> dict[str, Any]:
    end_sequence = _reach_observation_sequence(ctx)
    observation_delta = (
        None
        if start_sequence is None or end_sequence is None
        else max(0, int(end_sequence) - int(start_sequence))
    )
    return {
        "wall_s": round(max(0.0, time.monotonic() - float(started_s)), 6),
        "action_steps": int(action_steps),
        "observation_sequence_start": start_sequence,
        "observation_sequence_end": end_sequence,
        "observation_delta": observation_delta,
    }



def _v2_local_base_command(
    pose,
    *,
    bx: float,
    by: float,
    target_yaw_rad: float,
    pos_tol_m: float,
    yaw_tol_deg: float,
) -> tuple[list[float], float, float]:
    """Interface-v2 signed holonomic command in physical units."""
    dx = float(bx) - float(pose.pos[0])
    dy = float(by) - float(pose.pos[1])
    yaw = float(pose.yaw)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    error_forward = cos_y * dx + sin_y * dy
    error_left = -sin_y * dx + cos_y * dy
    distance = float(math.hypot(dx, dy))
    yaw_error = _normalize_angle(float(target_yaw_rad) - yaw)

    if distance <= float(pos_tol_m):
        vx = 0.0
        vy = 0.0
    else:
        speed = REACH_BASE_MAX_SPEED_MPS
        if distance < REACH_BASE_LIN_SLOW_RADIUS_M:
            speed *= distance / REACH_BASE_LIN_SLOW_RADIUS_M
        inverse_distance = 1.0 / max(distance, 1e-9)
        vx = speed * error_forward * inverse_distance
        vy = speed * error_left * inverse_distance

    if abs(math.degrees(yaw_error)) <= float(yaw_tol_deg):
        wz = 0.0
    else:
        yaw_rate = REACH_BASE_MAX_YAW_RATE_RADPS
        if abs(yaw_error) < REACH_BASE_ANG_SLOW_RADIUS_RAD:
            yaw_rate *= abs(yaw_error) / REACH_BASE_ANG_SLOW_RADIUS_RAD
        wz = math.copysign(yaw_rate, yaw_error)

    linear_ratio = math.hypot(vx, vy) / max(REACH_BASE_MAX_SPEED_MPS, 1e-9)
    angular_ratio = abs(wz) / max(REACH_BASE_MAX_YAW_RATE_RADPS, 1e-9)
    combined_ratio = math.hypot(linear_ratio, angular_ratio)
    if combined_ratio > 1.0:
        vx /= combined_ratio
        vy /= combined_ratio
        wz /= combined_ratio
    return [float(vx), float(vy), float(wz)], distance, abs(
        math.degrees(yaw_error)
    )



def _ramp_v2_base_command(last, desired, *, dt_s: float) -> list[float]:
    previous = np.asarray(last, dtype=np.float64).reshape(3)
    target = np.asarray(desired, dtype=np.float64).reshape(3)
    output = previous.copy()
    linear_delta = target[:2] - previous[:2]
    linear_delta_norm = float(np.linalg.norm(linear_delta))
    max_linear_delta = REACH_BASE_MAX_LIN_ACCEL_MPS2 * max(1e-5, float(dt_s))
    if linear_delta_norm > max_linear_delta > 0.0:
        linear_delta *= max_linear_delta / linear_delta_norm
    output[:2] += linear_delta
    angular_delta = float(target[2] - previous[2])
    max_angular_delta = REACH_BASE_MAX_ANG_ACCEL_RADPS2 * max(
        1e-5, float(dt_s)
    )
    output[2] = previous[2] + float(
        np.clip(angular_delta, -max_angular_delta, max_angular_delta)
    )
    return output.astype(float).tolist()



def _cap_base_linear_speed(command, speed_limit_mps: float) -> list[float]:
    output = np.asarray(command, dtype=np.float64).reshape(3).copy()
    speed = float(np.linalg.norm(output[:2]))
    limit = max(0.0, float(speed_limit_mps))
    if speed > limit and speed > 1e-12:
        output[:2] *= limit / speed
    return output.astype(float).tolist()



def _cap_reach_command_to_official_action(command) -> list[float]:
    output = np.asarray(command, dtype=np.float64).reshape(3).copy()
    linear_speed = float(np.linalg.norm(output[:2]))
    if linear_speed > REACH_BASE_OFFICIAL_MAX_SPEED_MPS:
        output[:2] *= REACH_BASE_OFFICIAL_MAX_SPEED_MPS / linear_speed
    output[2] = float(
        np.clip(
            output[2],
            -REACH_BASE_OFFICIAL_MAX_YAW_RATE_RADPS,
            REACH_BASE_OFFICIAL_MAX_YAW_RATE_RADPS,
        )
    )
    return output.astype(float).tolist()



def _base_goal_brake_limit(
    *,
    distance_m: float,
    along_remaining_m: float | None,
    pos_tol_m: float,
) -> float:
    remaining = max(0.0, float(distance_m) - max(0.0, float(pos_tol_m)))
    if along_remaining_m is not None:
        remaining = min(
            remaining,
            max(
                0.0,
                float(along_remaining_m) - REACH_BASE_GOAL_STOP_MARGIN_M,
            ),
        )
    deceleration = max(
        1e-3,
        REACH_BASE_MAX_LIN_ACCEL_MPS2 * REACH_BASE_BRAKE_DECEL_SAFETY,
    )
    return math.sqrt(2.0 * deceleration * remaining)



def _yield_reach_base_xy_yaw_controller(
    ctx,
    *,
    bx: float,
    by: float,
    target_yaw_rad: float,
    timeout_s: float,
    pos_tol_m: float = REACH_XY_TOL_M,
    yaw_tol_deg: float = REACH_YAW_TOL_DEG,
):
    """Observation-only port of v2's signed holonomic base controller."""
    dt_s = 1.0 / float(CONTROL_HZ)
    max_steps = max(1, int(math.ceil(max(0.1, float(timeout_s)) / dt_s)))
    initial_pose = ctx.world.robot_pose()
    initial_xy = np.asarray(initial_pose.pos[:2], dtype=np.float64).reshape(2)
    goal_xy = np.asarray([float(bx), float(by)], dtype=np.float64)
    goal_delta = goal_xy - initial_xy
    goal_length = float(np.linalg.norm(goal_delta))
    goal_direction = (
        goal_delta / goal_length
        if goal_length > max(1e-6, float(pos_tol_m))
        else None
    )
    adapter = getattr(ctx.world, "_official_adapter", None)
    previous_sequence = None
    if adapter is not None:
        try:
            previous_sequence = int(adapter.status().get("sequence"))
        except (AttributeError, TypeError, ValueError):
            previous_sequence = None

    last_command = [0.0, 0.0, 0.0]
    last_xy_error = float("inf")
    last_yaw_error = float("inf")
    progress_reference_xy: float | None = None
    progress_reference_yaw: float | None = None
    no_progress_steps = 0
    action_steps = 0
    max_goal_overshoot = 0.0
    stop_reason = None

    for _ in range(max_steps):
        _check_cancelled(ctx, "move-to-reach signed holonomic base")
        pose = ctx.world.robot_pose()
        current_xy = np.asarray(pose.pos[:2], dtype=np.float64).reshape(2)
        last_xy_error = float(np.linalg.norm(goal_xy - current_xy))
        last_yaw_error = abs(
            math.degrees(_normalize_angle(float(target_yaw_rad) - float(pose.yaw)))
        )
        along_remaining = None
        if goal_direction is not None:
            along_remaining = float(np.dot(goal_xy - current_xy, goal_direction))
            max_goal_overshoot = max(max_goal_overshoot, max(0.0, -along_remaining))
            if along_remaining < -REACH_BASE_GOAL_PLANE_TOL_M:
                stop_reason = "goal_overshoot"
                for _ in range(REACH_BASE_GOAL_HARD_STOP_STEPS):
                    yield _base_action_from_physical_velocity(ctx, [0.0, 0.0, 0.0])
                    action_steps += 1
                break

        current_sequence = previous_sequence
        if adapter is not None:
            try:
                current_sequence = int(adapter.status().get("sequence"))
            except (AttributeError, TypeError, ValueError):
                current_sequence = previous_sequence
        fresh_observation = bool(
            adapter is None
            or (
                current_sequence is not None
                and previous_sequence is not None
                and current_sequence > previous_sequence
            )
        )
        if current_sequence is not None:
            previous_sequence = current_sequence
        command_active = bool(
            math.hypot(float(last_command[0]), float(last_command[1]))
            >= REACH_BASE_STUCK_MIN_LIN_CMD_MPS
            or abs(float(last_command[2])) >= REACH_BASE_STUCK_MIN_ANG_CMD_RADPS
        )
        if fresh_observation and command_active:
            if progress_reference_xy is None:
                progress_reference_xy = last_xy_error
                progress_reference_yaw = last_yaw_error
                no_progress_steps = 0
            else:
                xy_needed = float(progress_reference_xy) > float(pos_tol_m)
                yaw_needed = float(progress_reference_yaw or 0.0) > float(
                    yaw_tol_deg
                )
                progressed = bool(
                    (
                        xy_needed
                        and float(progress_reference_xy) - last_xy_error
                        >= REACH_BASE_STUCK_XY_PROGRESS_M
                    )
                    or (
                        yaw_needed
                        and float(progress_reference_yaw or 0.0) - last_yaw_error
                        >= REACH_BASE_STUCK_YAW_PROGRESS_DEG
                    )
                )
                if not xy_needed and not yaw_needed:
                    progress_reference_xy = None
                    progress_reference_yaw = None
                    no_progress_steps = 0
                elif progressed:
                    progress_reference_xy = last_xy_error
                    progress_reference_yaw = last_yaw_error
                    no_progress_steps = 0
                else:
                    no_progress_steps += 1
            if no_progress_steps >= int(
                math.ceil(REACH_BASE_STUCK_WINDOW_S * CONTROL_HZ)
            ):
                stop_reason = "no_progress"
                yield _base_action_from_physical_velocity(ctx, [0.0, 0.0, 0.0])
                action_steps += 1
                break
        elif not command_active:
            progress_reference_xy = None
            progress_reference_yaw = None
            no_progress_steps = 0

        desired, last_xy_error, last_yaw_error = _v2_local_base_command(
            pose,
            bx=float(bx),
            by=float(by),
            target_yaw_rad=float(target_yaw_rad),
            pos_tol_m=float(pos_tol_m),
            yaw_tol_deg=float(yaw_tol_deg),
        )
        brake_limit = _base_goal_brake_limit(
            distance_m=last_xy_error,
            along_remaining_m=along_remaining,
            pos_tol_m=float(pos_tol_m),
        )
        desired = _cap_base_linear_speed(desired, brake_limit)
        if last_xy_error <= float(pos_tol_m) and last_yaw_error <= float(yaw_tol_deg):
            desired = [0.0, 0.0, 0.0]
        if (
            last_xy_error <= float(pos_tol_m)
            and last_yaw_error <= float(yaw_tol_deg)
            and max(abs(float(value)) for value in last_command) <= 1e-3
        ):
            for _ in range(4):
                yield _base_action_from_physical_velocity(ctx, [0.0, 0.0, 0.0])
                action_steps += 1
            return {
                "ok": True,
                "xy_ok": True,
                "yaw_ok": True,
                "path_status": "v2_signed_holonomic_arrived",
                "reason": None,
                "safety_abort": False,
                "action_steps": int(action_steps),
                "xy_err_m": float(last_xy_error),
                "yaw_err_deg": float(last_yaw_error),
                "max_goal_overshoot_m": float(max_goal_overshoot),
                "requested_speed_scale": float(
                    REACH_BASE_COMMAND_SPEED_SCALE
                ),
                "requested_max_linear_mps": float(
                    REACH_BASE_MAX_SPEED_MPS
                ),
                "requested_max_yaw_rate_radps": float(
                    REACH_BASE_MAX_YAW_RATE_RADPS
                ),
                "official_max_linear_mps": float(
                    REACH_BASE_OFFICIAL_MAX_SPEED_MPS
                ),
                "official_max_yaw_rate_radps": float(
                    REACH_BASE_OFFICIAL_MAX_YAW_RATE_RADPS
                ),
                "official_action_limit_respected": True,
            }

        physical_command = _ramp_v2_base_command(
            last_command,
            desired,
            dt_s=dt_s,
        )
        physical_command = _cap_base_linear_speed(physical_command, brake_limit)
        physical_command = _cap_reach_command_to_official_action(
            physical_command
        )
        last_command = physical_command
        yield _base_action_from_physical_velocity(ctx, physical_command)
        action_steps += 1
    else:
        stop_reason = "timeout"

    for _ in range(4):
        yield _base_action_from_physical_velocity(ctx, [0.0, 0.0, 0.0])
        action_steps += 1
    final_pose = ctx.world.robot_pose()
    final_xy_error = float(
        np.linalg.norm(goal_xy - np.asarray(final_pose.pos[:2], dtype=np.float64))
    )
    final_yaw_error = abs(
        math.degrees(
            _normalize_angle(float(target_yaw_rad) - float(final_pose.yaw))
        )
    )
    xy_ok = bool(final_xy_error <= float(pos_tol_m))
    yaw_ok = bool(final_yaw_error <= float(yaw_tol_deg))
    return {
        "ok": bool(xy_ok and yaw_ok),
        "xy_ok": xy_ok,
        "yaw_ok": yaw_ok,
        "path_status": f"v2_signed_holonomic_{stop_reason}",
        "reason": stop_reason,
        "safety_abort": bool(stop_reason in {"goal_overshoot", "no_progress"}),
        "action_steps": int(action_steps),
        "xy_err_m": float(final_xy_error),
        "yaw_err_deg": float(final_yaw_error),
        "max_goal_overshoot_m": float(max_goal_overshoot),
        "requested_speed_scale": float(REACH_BASE_COMMAND_SPEED_SCALE),
        "requested_max_linear_mps": float(REACH_BASE_MAX_SPEED_MPS),
        "requested_max_yaw_rate_radps": float(
            REACH_BASE_MAX_YAW_RATE_RADPS
        ),
        "official_max_linear_mps": float(
            REACH_BASE_OFFICIAL_MAX_SPEED_MPS
        ),
        "official_max_yaw_rate_radps": float(
            REACH_BASE_OFFICIAL_MAX_YAW_RATE_RADPS
        ),
        "official_action_limit_respected": True,
    }



def _reach_same_target_measurement(
    ctx,
    target_policy_local,
) -> dict[str, Any]:
    """Measure both shoulders against the immutable clicked policy-local point."""
    target_robot = _point_in_robot_frame(ctx.world, target_policy_local)
    trunk_q = np.asarray(
        ctx.world.trunk_qpos(),
        dtype=np.float64,
    ).reshape(4)
    shoulders = _r1pro_shoulder_positions_robot(trunk_q)
    distances = {
        side: float(np.linalg.norm(target_robot - shoulder))
        for side, shoulder in shoulders.items()
    }
    return {
        "target_robot_m": target_robot,
        "trunk_q": trunk_q,
        "shoulders": shoulders,
        "left_m": float(distances["left"]),
        "right_m": float(distances["right"]),
        "minimum_m": float(min(distances.values())),
        "source": "frozen_RGB-D_click_plus_policy_local_odometry_and_proprio",
    }



def _yield_reach_signed_base_step(
    ctx,
    *,
    signed_distance_m: float,
    target_policy_local,
):
    requested = float(signed_distance_m)
    distance = abs(requested)
    direction_sign = 1.0 if requested >= 0.0 else -1.0
    start_pose = ctx.world.robot_pose()
    heading = np.array(
        [math.cos(float(start_pose.yaw)), math.sin(float(start_pose.yaw))],
        dtype=np.float64,
    )
    if distance <= REACH_COMP_FORWARD_TOL_M:
        return {
            "ok": True,
            "requested_signed_m": requested,
            "actual_signed_m": 0.0,
            "action_steps": 0,
            "safety_abort": None,
            "last_rgbd_guard": None,
        }
    del target_policy_local
    target_xy = np.asarray(start_pose.pos[:2], dtype=np.float64) + (
        direction_sign * distance * heading
    )
    move_report = yield from _yield_reach_base_xy_yaw_controller(
        ctx,
        bx=float(target_xy[0]),
        by=float(target_xy[1]),
        target_yaw_rad=float(start_pose.yaw),
        timeout_s=REACH_COMP_FORWARD_TIMEOUT_S,
        pos_tol_m=0.02,
        yaw_tol_deg=REACH_YAW_TOL_DEG,
    )
    final_pose = ctx.world.robot_pose()
    displacement = np.asarray(final_pose.pos[:2], dtype=np.float64) - np.asarray(
        start_pose.pos[:2],
        dtype=np.float64,
    )
    progress = max(0.0, direction_sign * float(np.dot(displacement, heading)))
    return {
        "ok": bool(move_report.get("ok")),
        "requested_signed_m": requested,
        "actual_signed_m": float(direction_sign * progress),
        "action_steps": int(move_report.get("action_steps") or 0),
        "safety_abort": (
            move_report.get("reason")
            if not move_report.get("ok")
            else None
        ),
        "last_rgbd_guard": None,
        "start_odometry": start_pose.as_dict(),
        "final_odometry": final_pose.as_dict(),
        "verification": "v2 signed holonomic controller over policy-local odometry",
        "base_controller": move_report,
    }



def _yield_reach_compensation(
    ctx,
    *,
    target_policy_local,
    keep_states: dict[str, dict[str, Any]],
    max_dist_m: float = REACH_MAX_SHOULDER_DISTANCE_M,
):
    report: dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "trigger_max_dist_m": float(max_dist_m),
        "min_distance_before_m": None,
        "min_distance_after_m": None,
        "target_identity": "immutable_frozen_RGB-D_click",
        "phase1_forward": {
            "steps": [],
            "stopped_reason": None,
            "best_min_shoulder_m": None,
            "rolled_back_to_best": False,
        },
        "phase2_pitch": {
            "steps": [],
            "stopped_reason": None,
            "total_pitch_deg": 0.0,
        },
        "observation_action_only": True,
    }
    measurement = _reach_same_target_measurement(ctx, target_policy_local)
    distance_now = float(measurement["minimum_m"])
    report["min_distance_before_m"] = distance_now
    if distance_now <= float(max_dist_m):
        report["ok"] = True
        report["min_distance_after_m"] = distance_now
        return report

    report["attempted"] = True
    best_distance = distance_now
    best_progress = 0.0
    cumulative_progress = 0.0
    report["phase1_forward"]["best_min_shoulder_m"] = best_distance
    for step_index in range(1, REACH_COMP_FORWARD_MAX_STEPS + 1):
        move_report = yield from _yield_reach_signed_base_step(
            ctx,
            signed_distance_m=REACH_COMP_FORWARD_STEP_M,
            target_policy_local=target_policy_local,
        )
        cumulative_progress += max(
            0.0,
            float(move_report.get("actual_signed_m") or 0.0),
        )
        measurement = _reach_same_target_measurement(ctx, target_policy_local)
        distance_new = float(measurement["minimum_m"])
        stuck = bool(
            not move_report.get("ok")
            or move_report.get("safety_abort")
        )
        step_report = {
            "step": int(step_index),
            "requested_travel_m": float(REACH_COMP_FORWARD_STEP_M),
            "actual_travel_m": float(
                move_report.get("actual_signed_m") or 0.0
            ),
            "move_ok": bool(move_report.get("ok")),
            "safety_abort": move_report.get("safety_abort"),
            "min_shoulder_m": distance_new,
        }
        report["phase1_forward"]["steps"].append(step_report)
        if distance_new <= float(max_dist_m):
            report["phase1_forward"]["stopped_reason"] = "distance_ok"
            report["ok"] = True
            report["min_distance_after_m"] = distance_new
            return report

        improved = bool(
            distance_new
            < best_distance - REACH_COMP_IMPROVEMENT_EPS_M
        )
        if improved:
            best_distance = distance_new
            best_progress = cumulative_progress
            report["phase1_forward"]["best_min_shoulder_m"] = best_distance
        distance_now = distance_new
        if not improved and not stuck:
            report["phase1_forward"]["stopped_reason"] = (
                "distance_not_decreasing"
            )
            rollback_distance = best_progress - cumulative_progress
            rollback_report = yield from _yield_reach_signed_base_step(
                ctx,
                signed_distance_m=rollback_distance,
                target_policy_local=target_policy_local,
            )
            report["phase1_forward"]["rolled_back_to_best"] = True
            measurement = _reach_same_target_measurement(
                ctx,
                target_policy_local,
            )
            distance_now = float(measurement["minimum_m"])
            report["phase1_forward"]["rollback"] = {
                "requested_signed_m": float(rollback_distance),
                "move_ok": bool(rollback_report.get("ok")),
                "safety_abort": rollback_report.get("safety_abort"),
                "min_shoulder_after_m": distance_now,
            }
            if distance_now <= float(max_dist_m):
                report["ok"] = True
                report["min_distance_after_m"] = distance_now
                return report
            break
        if stuck:
            report["phase1_forward"]["stopped_reason"] = str(
                move_report.get("safety_abort") or "stuck_or_not_reached"
            )
            break
    else:
        report["phase1_forward"]["stopped_reason"] = "max_steps"

    max_pitch_steps = int(
        round(REACH_COMP_PITCH_MAX_DEG / REACH_COMP_PITCH_STEP_DEG)
    )
    total_pitch_deg = 0.0
    for step_index in range(1, max_pitch_steps + 1):
        q_now = np.asarray(
            ctx.world.trunk_qpos(),
            dtype=np.float64,
        ).reshape(4)
        q3_now = float(q_now[2])
        q3_floor_pitch = (
            float(q_now[0])
            + float(q_now[1])
            - math.radians(REACH_COMP_MAX_CHEST_PITCH_DEG)
        )
        q3_floor = max(REACH_COMP_Q3_MIN_RAD, q3_floor_pitch)
        q3_target = max(
            q3_now - math.radians(REACH_COMP_PITCH_STEP_DEG),
            q3_floor,
        )
        if q3_target >= q3_now - 1e-6:
            report["phase2_pitch"]["stopped_reason"] = (
                "q3_at_chest_pitch_limit"
                if q3_floor_pitch > REACH_COMP_Q3_MIN_RAD
                else "q3_at_urdf_limit"
            )
            break
        q_target = q_now.copy()
        q_target[2] = q3_target
        interpolation_steps = max(
            2,
            int(math.ceil(abs(q3_target - q3_now) / 0.08)),
        )
        for interpolation_index in range(1, interpolation_steps + 1):
            _check_cancelled(ctx, "move-to-reach compensation pitch")
            alpha = _smoothstep(interpolation_index / interpolation_steps)
            command = q_now + alpha * (q_target - q_now)
            yield ctx.world.make_action(
                **_reach_trunk_action_overrides(command, keep_states)
            )

        stable_steps = 0
        settle_steps = 0
        for _ in range(max(3, int(3.0 * CONTROL_HZ))):
            _check_cancelled(ctx, "move-to-reach compensation pitch settle")
            yield ctx.world.make_action(
                **_reach_trunk_action_overrides(q_target, keep_states)
            )
            settle_steps += 1
            observed = np.asarray(
                ctx.world.trunk_qpos(),
                dtype=np.float64,
            ).reshape(4)
            if float(np.max(np.abs(observed - q_target))) <= REACH_TRUNK_TOL_RAD:
                stable_steps += 1
                if stable_steps >= 3:
                    break
            else:
                stable_steps = 0

        observed = np.asarray(
            ctx.world.trunk_qpos(),
            dtype=np.float64,
        ).reshape(4)
        trunk_error = float(np.max(np.abs(observed - q_target)))
        trunk_ok = bool(stable_steps >= 3)
        measurement = _reach_same_target_measurement(ctx, target_policy_local)
        distance_new = float(measurement["minimum_m"])
        pitch_delta_deg = math.degrees(q3_now - q3_target)
        total_pitch_deg += pitch_delta_deg
        decreased = bool(
            distance_new < distance_now - REACH_COMP_IMPROVEMENT_EPS_M
        )
        report["phase2_pitch"]["steps"].append(
            {
                "step": int(step_index),
                "q3_before_rad": q3_now,
                "q3_target_rad": q3_target,
                "q3_observed_rad": float(observed[2]),
                "q3_floor_rad": q3_floor,
                "q3_floor_by": (
                    "chest_pitch_165deg"
                    if q3_floor_pitch > REACH_COMP_Q3_MIN_RAD
                    else "urdf_limit"
                ),
                "chest_pitch_target_deg": math.degrees(
                    float(q_target[0])
                    + float(q_target[1])
                    - q3_target
                ),
                "trunk_ok": trunk_ok,
                "trunk_error_rad": trunk_error,
                "settle_action_steps": int(settle_steps),
                "min_shoulder_m": distance_new,
                "decreased": decreased,
            }
        )
        report["phase2_pitch"]["total_pitch_deg"] = total_pitch_deg
        if distance_new <= float(max_dist_m):
            report["phase2_pitch"]["stopped_reason"] = "distance_ok"
            report["ok"] = True
            report["min_distance_after_m"] = distance_new
            return report
        if not trunk_ok:
            report["phase2_pitch"]["stopped_reason"] = "trunk_failed"
            distance_now = distance_new
            break
        if not decreased:
            report["phase2_pitch"]["stopped_reason"] = (
                "no_distance_decrease"
            )
            distance_now = distance_new
            break
        distance_now = distance_new
    else:
        report["phase2_pitch"]["stopped_reason"] = "max_pitch"

    final_measurement = _reach_same_target_measurement(
        ctx,
        target_policy_local,
    )
    report["min_distance_after_m"] = float(final_measurement["minimum_m"])
    report["ok"] = bool(
        float(final_measurement["minimum_m"]) <= float(max_dist_m)
    )
    return report


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
