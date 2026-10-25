"""plan_grasp_point_filter_rgbd_lite

轻量 RGBD 抓取：候选、过滤和输出保持完整版本逻辑；冷启动 IK 使用 split8，完整 warm-start 阶段使用等状态量 warm32x6；不启用候选压缩。

How this tool works:
Back-project the object-body pixel into the frozen RGB-D. Generate grasp poses, test observed occupancy and finger clearance, solve arm IK, and retain a signed local plan with a visual preview. This selects a plan; exec_plan_pose performs its arm motion.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.action_constants import DEFAULT_START_ARM_TOLERANCE_RAD
from tools.util.action_constants import DEFAULT_START_GRIPPER_TOLERANCE_RAD
from tools.util.action_constants import DEFAULT_START_TRUNK_TOLERANCE_RAD
from tools.util.action_constants import DEFAULT_TRACKING_STALL_STEPS
from tools.util.action_constants import DEFAULT_WAYPOINT_MAX_TRACKING_ERROR_RAD
from tools.util.action_constants import DEFAULT_WAYPOINT_SETTLE_TOLERANCE_RAD
from tools.util.action_constants import DEFAULT_WAYPOINT_TIMEOUT_S
from tools.util.action_constants import JOINT_TRAJECTORY_SCHEMA
from tools.util.action_constants import JOINT_TRAJECTORY_SCHEMA_VERSION
from tools.util.action_constants import LOCKED_J8_CAPTURE_TOLERANCE_RAD
from tools.util.action_constants import LOCKED_J8_INDEX
from tools.util.action_constants import LOCKED_J8_VALUE_RAD
from tools.util.action_constants import MAX_TRAJECTORY_WAYPOINTS
from tools.util.action_constants import REACH_MAX_SHOULDER_DISTANCE_M
from tools.util.action_constants import TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
from tools.util.action_constants import _ARM_LIMITS
from tools.util.geometry.capabilities import TOOL_VERSION
from tools.util.geometry.contract import ACTION_DIM
from tools.util.geometry.contract import ARM_DOF
from tools.util.geometry.contract import ROBOT_PROFILE
from tools.util.geometry.grasp_kinematics_local import LocalRobotState
from tools.util.geometry.rgbd_grasp_lite import plan_grasp_point_filter_rgbd_lite as _plan_rgbd_lite_local
from tools.util.geometry.rgbd_grasp_planner import prepare_rgbd_grasp_target as _prepare_rgbd_grasp_target
from tools.util.geometry.visualization_local import render_plan_gripper_overlay
from tools.util.motion_common import _FrozenCapture
from tools.util.motion_common import _arm_vector
from tools.util.motion_common import _check_cancelled
from tools.util.motion_common import _compile_exec_arm_segments
from tools.util.motion_common import _file_to_data_url
from tools.util.motion_common import _image_path
from tools.util.motion_common import _images_dir
from tools.util.motion_common import _j8_locked_copy
from tools.util.motion_common import _json_ready
from tools.util.motion_common import _load_frozen_capture
from tools.util.motion_common import _next_plan_id
from tools.util.motion_common import _normalize_arm_q_to_joint_limits
from tools.util.motion_common import _normalize_trunk_q_to_joint_limits
from tools.util.motion_common import _plan_record_path
from tools.util.motion_common import _point_in_robot_frame
from tools.util.motion_common import _r1pro_arm_fk_robot
from tools.util.motion_common import _relative_uv_to_pixel
from tools.util.motion_common import _require_episode_initialized
from tools.util.motion_common import _require_serialized_j8_zero
from tools.util.motion_common import _set_result
from tools.util.motion_common import _shoulder_distances_robot
from tools.util.motion_common import _trajectory_digest
from tools.util.motion_common import _unproject_pixel
from tools.util.motion_common import _validate_exec_final_q_for_arm
from tools.util.motion_common import _write_plan_record
from typing import Any
import math
import numpy as np
from tools.util.tool import main

NAME = 'plan_grasp_point_filter_rgbd_lite'
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
                'plan_arm': {'type': 'string', 'enum': ['any', 'left', 'right'], 'default': 'any'},
                'seed': {'type': 'number', 'default': 42, 'description': '姿态采样种子'}},
 'additionalProperties': False,
 'required': ['image_id', 'u', 'v']}


def actions(
    robot,
    session_id: str,
    image_id: str,
    u: float,
    v: float,
    arm: str = "right",
    seed: int = 42,
    plan_arm: str = "any",
    **kwargs,
):
    del arm
    session = str(session_id or "").strip()
    image = str(image_id or "").strip()
    requested_arm = str(plan_arm or "any").strip().lower()
    if requested_arm not in ("left", "right", "any"):
        requested_arm = "any"
    try:
        _require_episode_initialized(
            robot,
            "plan_grasp_point_filter_rgbd_lite",
        )
        capture = _load_frozen_capture(session, image)
        height, width = capture.depth.shape[:2]
        pixel_u, pixel_v = _relative_uv_to_pixel(u, v, width, height)
        camera = dict(capture.camera)
        if not bool(camera.get("pose_available", True)):
            raise ValueError(
                "capture has no evaluator camera-relative pose"
            )
        camera_pos = np.asarray(
            camera["pos"],
            dtype=np.float64,
        ).reshape(3)
        camera_quat = np.asarray(
            camera["quat"],
            dtype=np.float64,
        ).reshape(4)
        focal_length = float(camera["focal_length"])
        horizontal_aperture = float(camera["horizontal_aperture"])
        robot_state = LocalRobotState.from_capture(
            dict(capture.robot or {}),
            arm_dof=ARM_DOF,
        )
        prepared_target = _prepare_rgbd_grasp_target(
            capture.depth,
            u=int(pixel_u),
            v=int(pixel_v),
            cam_pos=camera_pos,
            cam_quat=camera_quat,
            w=int(width),
            h=int(height),
            fl=focal_length,
            ha=horizontal_aperture,
        )
        target_robot, target_robot_meta = (
            _prepared_rgbd_target_robot_point(
                prepared_target=prepared_target,
                capture=capture,
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
                world=robot,
            )
        )
        shoulder_reach = _plan_grasp_shoulder_reach_report(
            target_robot=target_robot,
            trunk_qpos=robot_state.trunk_q,
            plan_arm=requested_arm,
        )
        rgb_path = _image_path(session, image, ".png")
        depth_path = _image_path(session, image, ".depth.npy")
        planner_session = {
            "session_id": session,
            "image_id": image,
            "init_dir": _images_dir(session),
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "tool_label": "plan_grasp_point_filter_rgbd_lite",
            "observation_source": "official_evaluator_frozen_rgbd_proprio",
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        _set_result(
            robot,
            "plan_grasp_point_filter_rgbd_lite",
            {"ok": False, "error": str(exc)},
        )
        yield robot.hold_action()
        return

    if hasattr(robot, "log"):
        robot.log(
            "[official_v2/plan_grasp_point_filter_rgbd_lite] "
            f"image={image} relative=({float(u):.1f},{float(v):.1f}) "
            f"pixel=({pixel_u},{pixel_v}) plan_arm={requested_arm} "
            f"arm_dof={ARM_DOF} hit_pixel="
            f"{prepared_target['hit_audit'].get('depth_pixel')} "
            f"shoulder_l/r={shoulder_reach['left_m']:.3f}/"
            f"{shoulder_reach['right_m']:.3f}m "
            f"reach_ok={shoulder_reach['ok']}"
        )
    if not bool(shoulder_reach["ok"]):
        _set_result(
            robot,
            "plan_grasp_point_filter_rgbd_lite",
            {
                "ok": False,
                "error": PLAN_GRASP_POINT_TOO_FAR_ERROR,
                "failure_stage": "projected_point_shoulder_distance_gate",
                "image_id": image,
                "relative_uv": [float(u), float(v)],
                "pixel_uv": [int(pixel_u), int(pixel_v)],
                "plan_arm": requested_arm,
                "hit_world": np.asarray(
                    prepared_target["hit"],
                    dtype=np.float64,
                ).astype(float).tolist(),
                "hit_observation": dict(prepared_target["hit_audit"]),
                "target_robot_observation": target_robot_meta,
                "shoulder_reach": shoulder_reach,
            },
        )
        yield robot.hold_action()
        return
    yield robot.hold_action()
    _check_cancelled(robot, "plan_grasp_point_filter_rgbd_lite")
    try:
        payload = _plan_rgbd_lite_local(
            world=robot_state,
            session=planner_session,
            u=int(pixel_u),
            v=int(pixel_v),
            cam_pos=camera_pos,
            cam_quat=camera_quat,
            w=int(width),
            h=int(height),
            fl=focal_length,
            ha=horizontal_aperture,
            plan_arm=requested_arm,
            seed=int(seed),
            ctx=robot,
            prepared_target=prepared_target,
        )
        plan_id = _next_plan_id(session)
        gripper_overlay_path = _image_path(
            session,
            image,
            f".{plan_id}.gripper.png",
        )
        gripper_visualization = render_plan_gripper_overlay(
            rgb_path=rgb_path,
            depth_linear=capture.depth,
            camera=camera,
            eef_pose=dict(payload.get("eef_pose") or {}),
            output_path=gripper_overlay_path,
        )
        payload_debug_images = dict(payload.get("debug_images") or {})
        if bool(gripper_visualization.get("ok")):
            payload_debug_images["frozen_rgb_overlay"] = (
                gripper_overlay_path
            )
        payload["debug_images"] = payload_debug_images
        payload["gripper_visualization"] = gripper_visualization
        trajectory = _compile_joint_trajectory(
            ctx=robot,
            plan_id=plan_id,
            session_id=session,
            image_id=image,
            capture=capture,
            payload=payload,
        )
        plan_path = _plan_record_path(session, plan_id)
        record = _json_ready(
            {
                "plan_id": plan_id,
                "tool": "plan_grasp_point_filter_rgbd_lite",
                "tool_version": TOOL_VERSION,
                "session_id": session,
                "image_id": image,
                "relative_uv": [float(u), float(v)],
                "pixel_uv": [int(pixel_u), int(pixel_v)],
                "plan_arm": requested_arm,
                "seed": int(seed),
                "target_robot_observation": target_robot_meta,
                "shoulder_reach": shoulder_reach,
                "robot_model": {
                    "profile": ROBOT_PROFILE,
                    "arm_dof": int(ARM_DOF),
                    "kinematics": "submission_local_urdf_curobo",
                },
                "visualization": gripper_visualization,
                "trajectory": trajectory,
                "payload": payload,
            }
        )
        _write_plan_record(plan_path, record)
    except Exception as exc:
        _set_result(
            robot,
            "plan_grasp_point_filter_rgbd_lite",
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "image_id": image,
                "relative_uv": [float(u), float(v)],
                "plan_arm": requested_arm,
            },
        )
        yield robot.hold_action()
        return

    result = {
        "ok": True,
        "plan_id": plan_id,
        "plan_record_path": plan_path,
        "trajectory_digest": trajectory["integrity"]["digest"],
        "trajectory_waypoint_count": sum(
            len(segment["waypoints"])
            for segment in trajectory["segments"]
        ),
        "session_id": session,
        "image_id": image,
        "mode": "grasp_point_filter_rgbd_lite",
        "plan_arm": requested_arm,
        "target_robot_observation": target_robot_meta,
        "shoulder_reach": shoulder_reach,
        "arm": payload.get("recommended_arm"),
        "recommended_arm": payload.get("recommended_arm"),
        "eef_pose": payload.get("eef_pose"),
        "next_move_world": payload.get("next_eef_move"),
        "hit_world": payload.get("hit_world"),
        "candidate_count": payload.get("candidate_count"),
        "selected_pose_ik": payload.get("selected_pose_ik"),
        "selected_pose_ik_q": payload.get("selected_pose_ik_q"),
        "ik_solution": payload.get("ik_solution"),
        "grasp_vol_cm3": payload.get("grasp_vol_cm3"),
        "overlap_vol_cm3": payload.get("overlap_vol_cm3"),
        "inflated_overlap_vol_cm3": payload.get(
            "inflated_overlap_vol_cm3"
        ),
        "grip_fit": payload.get("grip_fit"),
        "plan_audit": payload.get("plan_audit"),
        "debug_images": payload.get("debug_images"),
        "gripper_visualization": payload.get(
            "gripper_visualization"
        ),
        "render_image": (
            _file_to_data_url(gripper_overlay_path)
            if bool(gripper_visualization.get("ok"))
            else None
        ),
        "render_image_path": (
            gripper_overlay_path
            if bool(gripper_visualization.get("ok"))
            else None
        ),
        "rgb_overlay_path": (
            gripper_overlay_path
            if bool(gripper_visualization.get("ok"))
            else None
        ),
        "observation_contract": [
            "frozen_evaluator_rgb",
            "frozen_evaluator_depth_linear",
            "frozen_evaluator_camera_relative_pose",
            "capture_time_evaluator_proprioception",
            "submission_local_static_robot_model",
        ],
        "segmentation_used": False,
        "object_identity_used": False,
        "simulator_mesh_used": False,
        "simulator_fk_ik_used": False,
        "direct_simulator_mutation": False,
    }
    _set_result(
        robot,
        "plan_grasp_point_filter_rgbd_lite",
        _json_ready(result),
    )
    yield robot.hold_action()


PLAN_GRASP_POINT_TOO_FAR_ERROR = (
    "The projected point is too far away; please select another point or "
    "move closer."
)



def _selected_safe_q(payload: dict[str, Any], arm: str) -> tuple[np.ndarray, float]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("planner payload has no selected candidate list")
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise ValueError("planner selected candidate is invalid")
    candidate_arm = str(candidate.get("arm", "")).strip().lower()
    if candidate_arm != arm:
        raise ValueError(
            f"planner selected candidate arm {candidate_arm!r} "
            f"does not match recommended arm {arm!r}"
        )
    planned_safe = dict((candidate.get("meta") or {}).get("planned_safe") or {})
    if not bool(planned_safe.get("ok")):
        raise ValueError(f"planner selected candidate has no validated safe_q")
    safe_q = _j8_locked_copy(
        planned_safe.get("safe_q"),
        label=f"{arm} planned safe_q",
    )
    planned_safe["safe_q"] = safe_q.astype(float).tolist()
    candidate_meta = dict(candidate.get("meta") or {})
    candidate_meta["planned_safe"] = planned_safe
    candidate["meta"] = candidate_meta
    back_m = float(
        planned_safe.get(
            "active_back_m",
            planned_safe.get("requested_back_m", 0.10),
        )
    )
    if not math.isfinite(back_m) or back_m < 0.0:
        raise ValueError("planned safe back_m is invalid")
    return safe_q, back_m



def _compile_joint_trajectory(
    *,
    ctx,
    plan_id: str,
    session_id: str,
    image_id: str,
    capture: _FrozenCapture,
    payload: dict[str, Any],
) -> dict[str, Any]:
    _require_episode_initialized(ctx, "trajectory compilation")
    active_arm = str(payload.get("recommended_arm", "")).strip().lower()
    if active_arm not in ("left", "right"):
        raise ValueError("planner did not select a valid execution arm")
    final_q_map = payload.get("selected_pose_ik_q")
    if not isinstance(final_q_map, dict):
        raise ValueError("planner payload has no selected_pose_ik_q")
    final_q = _j8_locked_copy(
        final_q_map.get(active_arm),
        label=f"{active_arm} selected final_q",
    )
    safe_q, back_m = _selected_safe_q(payload, active_arm)

    robot = dict(capture.robot or {})
    start_arms = {}
    for side in ("left", "right"):
        observed_start = _arm_vector(
            robot.get(f"arm_{side}_qpos"),
            label=f"capture {side} arm_qpos",
        )
        if (
            LOCKED_J8_INDEX is not None
            and abs(float(observed_start[LOCKED_J8_INDEX]))
            > LOCKED_J8_CAPTURE_TOLERANCE_RAD
        ):
            raise ValueError(
                f"capture {side} arm J8={observed_start[LOCKED_J8_INDEX]:.6f}rad "
                "is outside the locked-joint initialization tolerance; recapture"
            )
        start_arms[side] = _j8_locked_copy(
            observed_start,
            label=f"capture {side} normalized arm_qpos",
        )
    # 对齐 v2：冻结起点只记录、必要时回收到限位，不因物理穿透判规划失败。
    joint_limit_normalizations: list[dict[str, Any]] = []
    for side in ("left", "right"):
        recovered, normalized = _normalize_arm_q_to_joint_limits(
            start_arms[side],
            arm=side,
            label=f"capture {side} start_q",
            point="start_q" if side == active_arm else f"{side}_start_q",
            recover_observed_overshoot=True,
        )
        start_arms[side] = recovered
        joint_limit_normalizations.extend(normalized)
    start_q = start_arms[active_arm]
    captured_start_recovered = bool(
        any(
            item.get("recovered_observed_overshoot")
            for item in joint_limit_normalizations
        )
    )
    safe_q, normalized = _normalize_arm_q_to_joint_limits(
        safe_q,
        arm=active_arm,
        label=f"{active_arm} planned safe_q",
        point="safe_q",
    )
    joint_limit_normalizations.extend(normalized)
    final_q, normalized = _normalize_arm_q_to_joint_limits(
        final_q,
        arm=active_arm,
        label=f"{active_arm} selected final_q",
        point="final_q",
    )
    joint_limit_normalizations.extend(normalized)
    start_arms[active_arm] = start_q
    final_q_map[active_arm] = final_q.astype(float).tolist()
    payload["candidates"][0]["meta"]["planned_safe"]["safe_q"] = (
        safe_q.astype(float).tolist()
    )
    trunk_q = np.asarray(
        robot.get("trunk_qpos"),
        dtype=np.float64,
    ).reshape(-1)
    if trunk_q.size != 4 or not np.all(np.isfinite(trunk_q)):
        raise ValueError("capture trunk_qpos must contain four finite values")
    # 冻结观测与 start arm 对齐：ULP/float32 残差回收到限位后再签名。
    trunk_q, trunk_normalized = _normalize_trunk_q_to_joint_limits(
        trunk_q,
        label="capture trunk_qpos",
        point="start_trunk",
        recover_observed_overshoot=True,
    )
    joint_limit_normalizations.extend(trunk_normalized)
    capture_motion_epoch = robot.get("motion_epoch")
    if not isinstance(capture_motion_epoch, int) or capture_motion_epoch < 0:
        raise ValueError("capture has no valid policy motion_epoch")
    capture_episode_id = str(robot.get("episode_id", "")).strip()
    if not capture_episode_id:
        raise ValueError("capture has no valid evaluator episode_id")
    if str(ctx.world.episode_id()) != capture_episode_id:
        raise ValueError("capture belongs to a different evaluator episode")

    segments = _compile_exec_arm_segments(start_q, safe_q, final_q)
    waypoint_count = sum(len(segment["waypoints"]) for segment in segments)
    if waypoint_count > MAX_TRAJECTORY_WAYPOINTS:
        raise ValueError(
            f"compiled trajectory has {waypoint_count} waypoints; "
            f"limit is {MAX_TRAJECTORY_WAYPOINTS}"
        )

    limits = _ARM_LIMITS[active_arm][:ARM_DOF]
    all_waypoints = []
    for segment in segments:
        segment_name = str(segment["name"])
        for waypoint_index, raw_q in enumerate(segment["waypoints"]):
            q = _arm_vector(raw_q, label=f"{segment_name} waypoint")
            below = limits[:, 0] - q
            above = q - limits[:, 1]
            violation = np.maximum(below, above)
            if bool(np.any(violation > 0.0)):
                joint_index = int(np.argmax(violation))
                raise ValueError(
                    "compiled trajectory exceeds local arm joint limits: "
                    f"segment={segment_name} waypoint={waypoint_index + 1} "
                    f"arm={active_arm} J{joint_index + 1}="
                    f"{q[joint_index]:.12g}rad outside "
                    f"[{limits[joint_index, 0]:.12g},"
                    f"{limits[joint_index, 1]:.12g}] by "
                    f"{violation[joint_index]:.12g}rad"
                )
            all_waypoints.append(q)
    fk_positions = [
        _r1pro_arm_fk_robot(active_arm, trunk_q, q)[0]
        for q in all_waypoints
    ]
    if not all(np.all(np.isfinite(pos)) for pos in fk_positions):
        raise ValueError("local FK validation produced a non-finite pose")

    start_state = {
        "motion_epoch": int(capture_motion_epoch),
        "episode_id": capture_episode_id,
        "arm_qpos": {
            side: start_arms[side].astype(float).tolist()
            for side in ("left", "right")
        },
        "arm_qvel": {
            side: _arm_vector(
                robot.get(f"arm_{side}_qvel", [0.0] * ARM_DOF),
                label=f"capture {side} arm_qvel",
            ).astype(float).tolist()
            for side in ("left", "right")
        },
        "trunk_qpos": trunk_q.astype(float).tolist(),
        "gripper_qpos": {
            side: _json_ready(robot.get(f"gripper_{side}_qpos"))
            for side in ("left", "right")
        },
    }
    signed_final_q_by_arm = {
        active_arm: final_q.astype(float).tolist(),
    }
    for side in ("left", "right"):
        if side == active_arm or final_q_map.get(side) is None:
            continue
        alternate_q = _require_serialized_j8_zero(
            final_q_map[side],
            label=f"planner {side} selected final_q",
        )
        alternate_q, normalized = _normalize_arm_q_to_joint_limits(
            alternate_q,
            arm=side,
            label=f"planner {side} selected final_q",
            point=f"{side}_final_q",
        )
        joint_limit_normalizations.extend(normalized)
        signed_final_q_by_arm[side] = alternate_q.astype(float).tolist()
    trajectory: dict[str, Any] = {
        "schema": JOINT_TRAJECTORY_SCHEMA,
        "schema_version": JOINT_TRAJECTORY_SCHEMA_VERSION,
        "plan_id": str(plan_id),
        "session_id": str(session_id),
        "tool_version": TOOL_VERSION,
        "source_tool": "plan_grasp_point_filter_rgbd_lite",
        "source_image_id": str(image_id),
        "robot_profile": ROBOT_PROFILE,
        "arm_dof": int(ARM_DOF),
        "action_dim": int(ACTION_DIM),
        "active_arm": active_arm,
        "final_q_by_arm": signed_final_q_by_arm,
        "locked_arm_joints": (
            {"J8": LOCKED_J8_VALUE_RAD}
            if LOCKED_J8_INDEX is not None
            else {}
        ),
        "start_state": start_state,
        "parameters": {
            "back_m": float(back_m),
            "reset_tool_roll_at_start": True,
        },
        "tracking": {
            "start_arm_tolerance_rad": DEFAULT_START_ARM_TOLERANCE_RAD,
            "start_trunk_tolerance_rad": DEFAULT_START_TRUNK_TOLERANCE_RAD,
            "start_gripper_tolerance_rad": (
                DEFAULT_START_GRIPPER_TOLERANCE_RAD
            ),
            "stationary_qvel_tolerance_rad_s": 0.35,
            "settle_tolerance_rad": DEFAULT_WAYPOINT_SETTLE_TOLERANCE_RAD,
            "max_tracking_error_rad": DEFAULT_WAYPOINT_MAX_TRACKING_ERROR_RAD,
            "waypoint_timeout_s": DEFAULT_WAYPOINT_TIMEOUT_S,
            "stall_steps": DEFAULT_TRACKING_STALL_STEPS,
        },
        "segments": segments,
        "safety_validation": {
            "status": "conditional",
            "joint_limits_checked": True,
            "joint_limit_numeric_tolerance_rad": (
                TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
            ),
            "joint_limit_normalizations": joint_limit_normalizations,
            "captured_start_joint_limits_enforced": False,
            "captured_start_recovered_to_local_limits": (
                captured_start_recovered
            ),
            "local_fk_finite_checked": True,
            "safe_final_pair_validated_by_planner": True,
            "frozen_rgbd_collision_checked": False,
            "self_collision_checked": False,
            "runtime_may_recompute_kinematics": False,
        },
    }
    final_q_validation_by_arm = {
        active_arm: {
            "source": "signed_active_trajectory_endpoint",
            "planned_arm": active_arm,
            "execution_arm": active_arm,
            "arm_switched": False,
            "pair_pos_err_m": 0.0,
            "pair_ori_err_deg": 0.0,
            "submission_local_fk_checked": True,
            "j8_locked": LOCKED_J8_INDEX is not None,
        }
    }
    for side, raw_q in signed_final_q_by_arm.items():
        if side == active_arm:
            continue
        validated_q, validation = _validate_exec_final_q_for_arm(
            trajectory=trajectory,
            arm=side,
            raw_q=raw_q,
            source="planner_selected_pose_ik_q",
        )
        signed_final_q_by_arm[side] = validated_q.astype(float).tolist()
        final_q_validation_by_arm[side] = validation
    trajectory["final_q_validation_by_arm"] = final_q_validation_by_arm
    trajectory["integrity"] = {
        "algorithm": "hmac-sha256",
        "digest": _trajectory_digest(trajectory),
    }
    return trajectory



def _prepared_rgbd_target_robot_point(
    *,
    prepared_target: dict[str, Any],
    capture: _FrozenCapture,
    focal_length: float,
    horizontal_aperture: float,
    world,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Express the planner's selected depth sample in robot-base frame."""
    hit_audit = dict(prepared_target.get("hit_audit") or {})
    depth_pixel = np.asarray(
        hit_audit.get("depth_pixel", []),
        dtype=np.int64,
    ).reshape(-1)
    depth_m = float(hit_audit.get("depth_m", float("nan")))
    camera_relative_pose = dict(
        capture.camera.get("robot_relative_pose") or {}
    )
    relative_pos = camera_relative_pose.get("pos")
    relative_quat = camera_relative_pose.get("quat")
    if (
        depth_pixel.size == 2
        and math.isfinite(depth_m)
        and depth_m > 0.0
        and relative_pos is not None
        and relative_quat is not None
    ):
        height, width = capture.depth.shape[:2]
        focal_px = (
            float(focal_length)
            / float(horizontal_aperture)
            * float(width)
        )
        relative_camera = {
            "fx": focal_px,
            "fy": focal_px,
            "cx": float(width) * 0.5,
            "cy": float(height) * 0.5,
            "pos": relative_pos,
            "quat": relative_quat,
        }
        point = _unproject_pixel(
            px=int(depth_pixel[0]),
            py=int(depth_pixel[1]),
            depth_m=depth_m,
            camera=relative_camera,
        )
        return point, {
            "frame": "robot_base",
            "source": "planner_depth_column_hit_and_cam_rel_pose",
            "depth_pixel": depth_pixel.astype(int).tolist(),
            "depth_m": depth_m,
        }

    point = _point_in_robot_frame(world, prepared_target["hit"])
    return point, {
        "frame": "robot_base",
        "source": "planner_hit_and_policy_owned_odometry_fallback",
        "depth_pixel": depth_pixel.astype(int).tolist(),
        "depth_m": depth_m if math.isfinite(depth_m) else None,
    }



def _plan_grasp_shoulder_reach_report(
    *,
    target_robot: np.ndarray,
    trunk_qpos: np.ndarray,
    plan_arm: str,
    threshold_m: float = REACH_MAX_SHOULDER_DISTANCE_M,
) -> dict[str, Any]:
    requested_arm = str(plan_arm or "any").strip().lower()
    if requested_arm not in ("left", "right", "any"):
        requested_arm = "any"
    target = np.asarray(target_robot, dtype=np.float64).reshape(3)
    shoulders, distances = _shoulder_distances_robot(target, trunk_qpos)
    checked_distance = (
        min(distances.values())
        if requested_arm == "any"
        else distances[requested_arm]
    )
    return {
        "ok": bool(checked_distance < float(threshold_m)),
        "plan_arm": requested_arm,
        "comparison": "strict_less_than",
        "threshold_m": float(threshold_m),
        "checked_distance_m": float(checked_distance),
        "left_m": distances["left"],
        "right_m": distances["right"],
        "target_robot_m": target.astype(float).tolist(),
        "left_shoulder_robot_m": shoulders["left"].astype(float).tolist(),
        "right_shoulder_robot_m": shoulders["right"].astype(float).tolist(),
        "trunk_source": "frozen_capture",
        "static_model": "exact submission-local R1Pro URDF FK",
    }


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
