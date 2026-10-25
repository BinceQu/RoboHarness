"""move_chassis_to_directly_facing_surface

在同一张冻结 head RGB-D 图上选择恰好三个不共线的表面点。工具反投影三点并取三角形中心；法向符号自动选择为与冻结 head 相机视线成钝角的一侧。底盘中心先平移到中心沿该法向 0.8m 后投影到地面的 XY 点，再原地旋转，使机器人前向与该法向的水平投影平行且方向相反。三点顺序不会翻转最终法向；执行只使用 official action，并由后续 base_qvel proprioception 验证。

How this tool works:
Use three visible points on one surface to estimate its plane and heading. Solve a stand-off pose, execute the guarded base path, and verify the measured final position and facing direction.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.action_constants import CONTROL_HZ
from tools.util.geometry.contract import validate_move_chassis_to_directly_facing_surface_args
from tools.util.geometry.surface_facing_local import SURFACE_FACING_HEADING_TOLERANCE_DEG
from tools.util.geometry.surface_facing_local import SURFACE_FACING_STANDOFF_M
from tools.util.geometry.surface_facing_local import rebase_robot_points as _rebase_surface_points
from tools.util.geometry.surface_facing_local import rebase_robot_vector as _rebase_surface_vector
from tools.util.geometry.surface_facing_local import solve_surface_facing_target as _solve_surface_facing_target
from tools.util.motion_common import _FrozenCapture
from tools.util.motion_common import _depth_value
from tools.util.motion_common import _load_frozen_capture
from tools.util.motion_common import _normalize_angle
from tools.util.motion_common import _quat_to_mat_xyzw
from tools.util.motion_common import _relative_uv_to_pixel
from tools.util.motion_common import _set_result
from tools.util.motion_common import _unproject_pixel
from tools.util.motion_common import _yield_adjust_chassis_controller
from typing import Any
from typing import Mapping
from typing import Sequence
import math
import numpy as np
from tools.util.tool import main

NAME = 'move_chassis_to_directly_facing_surface'
SCHEMA = {'type': 'object',
 'properties': {'image_id': {'type': 'string',
                             'description': 'The latest clickable head-camera image_id, actually '
                                            '720 x 720; re-ground the point when image_id changes. '
                                            'Wrist images are not clickable.'},
                'points': {'type': 'array',
                           'minItems': 3,
                           'maxItems': 3,
                           'items': {'additionalProperties': False,
                                     'properties': {'u': {'type': 'integer',
                                                          'minimum': 0,
                                                          'maximum': 719,
                                                          'description': 'Integer original-image '
                                                                         'pixel column in the '
                                                                         'latest 720 x 720 head '
                                                                         'image: 0 is the left '
                                                                         'edge and 719 is the '
                                                                         'right edge. Do not '
                                                                         'normalize or rescale.'},
                                                    'v': {'type': 'integer',
                                                          'minimum': 0,
                                                          'maximum': 719,
                                                          'description': 'Integer original-image '
                                                                         'pixel row in the latest '
                                                                         '720 x 720 head image: 0 '
                                                                         'is the top edge and 719 '
                                                                         'is the bottom edge. Do '
                                                                         'not normalize or '
                                                                         'rescale.'}},
                                     'required': ['u', 'v'],
                                     'type': 'object'}},
                'nav_timeout_s': {'type': 'number', 'default': 120.0, 'description': 's'},
                'pos_tol_m': {'type': 'number', 'default': 0.04, 'description': 'm'}},
 'additionalProperties': False,
 'required': ['image_id', 'points']}


def actions(
    robot,
    session_id: str,
    image_id: str,
    points: list[dict[str, Any]],
    nav_timeout_s: float = 120.0,
    pos_tol_m: float = 0.04,
    **kwargs,
):
    """Stand 0.8m along a surface normal, then face its opposite XY projection."""

    del kwargs
    common = {
        "ok": False,
        "error": None,
        "image_id": str(image_id or "").strip(),
        "execution": "official_action_only",
        "execution_order": "translate_xy_then_rotate_in_place",
        "target_reference": "chassis_base_center",
        "standoff_m": float(SURFACE_FACING_STANDOFF_M),
        "observation_contract": [
            "frozen head RGB",
            "frozen head depth_linear",
            "frozen head cam_rel_poses",
            "policy-local action odometry",
            "subsequent base_qvel proprioception",
        ],
        "direct_simulator_mutation": False,
        "global_base_pose_truth": False,
        "simulator_collision_truth": False,
    }
    try:
        request = validate_move_chassis_to_directly_facing_surface_args(
            {
                "session_id": session_id,
                "image_id": image_id,
                "points": points,
                "nav_timeout_s": nav_timeout_s,
                "pos_tol_m": pos_tol_m,
            }
        )
    except ValueError as exc:
        _set_result(
            robot,
            "move_chassis_to_directly_facing_surface",
            {
                **common,
                "error": str(exc),
                "failure_stage": "input validation",
            },
        )
        yield robot.hold_action()
        return

    try:
        capture = _load_frozen_capture(
            request["session_id"], request["image_id"], role="head"
        )
        solution, point_metadata = _surface_facing_capture_geometry(
            capture, request["points"]
        )
        capture_robot = capture.robot
        if not isinstance(capture_robot, Mapping):
            raise ValueError("frozen head capture has no robot state metadata")
        capture_episode = str(capture_robot.get("episode_id") or "").strip()
        if not capture_episode:
            raise ValueError("frozen head capture has no evaluator episode identifier")
        if capture_episode != str(robot.episode_id()):
            raise ValueError("frozen head capture belongs to a different evaluator episode")
        capture_base = capture_robot.get("base_pose")
        if not isinstance(capture_base, Mapping):
            raise ValueError("frozen head capture has no policy-local base pose")
        capture_position, capture_yaw = _surface_facing_base_pose(
            capture_base, label="capture"
        )
        current_position, current_yaw = _surface_facing_current_policy_base_pose(
            robot
        )

        current_points = _rebase_surface_points(
            solution.points_robot_base_m,
            source_base_position_policy_m=capture_position,
            source_base_yaw_rad=capture_yaw,
            target_base_position_policy_m=current_position,
            target_base_yaw_rad=current_yaw,
        )
        stand_current = _rebase_surface_points(
            solution.stand_point_robot_base_m.reshape(1, 3),
            source_base_position_policy_m=capture_position,
            source_base_yaw_rad=capture_yaw,
            target_base_position_policy_m=current_position,
            target_base_yaw_rad=current_yaw,
        )[0]
        facing_current = _rebase_surface_vector(
            solution.desired_chassis_forward_robot_base,
            source_base_yaw_rad=capture_yaw,
            target_base_yaw_rad=current_yaw,
        )
        desired_spin_deg = math.degrees(
            math.atan2(float(facing_current[1]), float(facing_current[0]))
        )
        desired_spin_deg = math.degrees(
            _normalize_angle(math.radians(desired_spin_deg))
        )
        current_cosine = math.cos(current_yaw)
        current_sine = math.sin(current_yaw)
        target_policy_position = current_position.copy()
        target_policy_position[:2] += np.asarray(
            [
                current_cosine * stand_current[0]
                - current_sine * stand_current[1],
                current_sine * stand_current[0]
                + current_cosine * stand_current[1],
            ],
            dtype=np.float64,
        )
        target_policy_position[2] = 0.0
        target_policy_yaw = _normalize_angle(
            current_yaw + math.radians(desired_spin_deg)
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        _set_result(
            robot,
            "move_chassis_to_directly_facing_surface",
            {
                **common,
                "error": str(exc),
                "failure_stage": "observation validation",
            },
        )
        yield robot.hold_action()
        return

    geometry = {
        **solution.as_dict(),
        "point_observations": point_metadata,
        "points_current_robot_base_m": current_points.astype(float).tolist(),
        "stand_point_current_robot_base_m": stand_current.astype(float).tolist(),
        "desired_chassis_forward_current_robot_base": (
            facing_current.astype(float).tolist()
        ),
        "desired_spin_from_current_deg": float(desired_spin_deg),
        "normal_sign_rule": "dot(normal_toward_robot, camera_forward) < 0",
        "position_rebase": "capture_robot_base_to_policy_local_to_current_robot_base",
    }
    translation = _yield_adjust_chassis_controller(
        robot,
        forward=float(stand_current[0]),
        translation=float(stand_current[1]),
        spin=0.0,
        timeout_s=float(request["nav_timeout_s"]),
        _linear_tolerance_m=float(request["pos_tol_m"]),
        _require_linear_target=True,
        _commit_result=False,
        _emit_terminal_hold=False,
        _result_tool_name="move_chassis_to_directly_facing_surface",
        _linear_body_frame_feedback=True,
        _componentwise_linear_limit=True,
    )
    translation_steps = 0
    while True:
        try:
            action = next(translation)
        except StopIteration as stopped:
            translation_report = stopped.value
            break
        translation_steps += 1
        yield action
    if not isinstance(translation_report, Mapping):
        translation_report = {
            "ok": False,
            "error": "chassis translation returned no structured outcome",
            "failure_stage": "tracking",
        }
    if translation_report.get("ok") is not True:
        _set_result(
            robot,
            "move_chassis_to_directly_facing_surface",
            {
                **common,
                "error": str(
                    translation_report.get("error")
                    or "chassis translation did not reach the surface stand point"
                ),
                "failure_stage": str(
                    translation_report.get("failure_stage") or "tracking"
                ),
                "geometry": geometry,
                "translation_phase": dict(translation_report),
                "rotation_phase": None,
                "action_steps": int(translation_steps),
            },
        )
        yield robot.hold_action()
        return

    linear_yaw_drift_deg = float(
        translation_report.get("linear_yaw_drift_deg", 0.0)
    )
    remaining_spin_deg = math.degrees(
        _normalize_angle(
            math.radians(desired_spin_deg - linear_yaw_drift_deg)
        )
    )
    remaining_timeout_s = float(request["nav_timeout_s"]) - (
        float(translation_steps) / float(CONTROL_HZ)
    )
    if remaining_timeout_s <= 0.0:
        _set_result(
            robot,
            "move_chassis_to_directly_facing_surface",
            {
                **common,
                "error": "surface-facing motion exhausted its timeout before rotation",
                "failure_stage": "timeout",
                "geometry": geometry,
                "translation_phase": dict(translation_report),
                "rotation_phase": None,
                "remaining_spin_deg": float(remaining_spin_deg),
                "action_steps": int(translation_steps),
            },
        )
        yield robot.hold_action()
        return

    rotation = _yield_adjust_chassis_controller(
        robot,
        forward=0.0,
        translation=0.0,
        spin=float(remaining_spin_deg),
        timeout_s=float(remaining_timeout_s),
        _commit_result=False,
        _emit_terminal_hold=False,
        _result_tool_name="move_chassis_to_directly_facing_surface",
        _skip_linear_settle_for_spin=True,
    )
    rotation_steps = 0
    while True:
        try:
            action = next(rotation)
        except StopIteration as stopped:
            rotation_report = stopped.value
            break
        rotation_steps += 1
        yield action
    if not isinstance(rotation_report, Mapping):
        rotation_report = {
            "ok": False,
            "error": "chassis rotation returned no structured outcome",
            "failure_stage": "tracking",
        }

    try:
        final_policy_position, final_policy_yaw = (
            _surface_facing_current_policy_base_pose(robot)
        )
        final_position_error_m = float(
            np.linalg.norm(
                final_policy_position[:2] - target_policy_position[:2]
            )
        )
        final_heading_error_deg = abs(
            math.degrees(
                _normalize_angle(target_policy_yaw - final_policy_yaw)
            )
        )
        final_pose_available = True
    except (TypeError, ValueError):
        final_policy_position = np.full(3, np.nan, dtype=np.float64)
        final_policy_yaw = math.nan
        final_position_error_m = math.inf
        final_heading_error_deg = math.inf
        final_pose_available = False

    position_ok = bool(
        final_pose_available
        and final_position_error_m <= float(request["pos_tol_m"])
    )
    heading_ok = bool(
        final_pose_available
        and final_heading_error_deg <= SURFACE_FACING_HEADING_TOLERANCE_DEG
    )
    ok = bool(rotation_report.get("ok") is True and position_ok and heading_ok)
    if rotation_report.get("ok") is not True:
        final_error = str(
            rotation_report.get("error")
            or "chassis did not finish the surface-facing rotation"
        )
        failure_stage = str(rotation_report.get("failure_stage") or "tracking")
    elif not final_pose_available:
        final_error = "final policy-local chassis odometry is unavailable"
        failure_stage = "observation validation"
    elif not position_ok:
        final_error = (
            "final chassis base centre is outside the requested position "
            f"tolerance ({final_position_error_m:.6f}m > "
            f"{float(request['pos_tol_m']):.6f}m)"
        )
        failure_stage = "tracking"
    elif not heading_ok:
        final_error = (
            "final chassis heading is not parallel and opposite to the "
            "surface normal's horizontal projection "
            f"({final_heading_error_deg:.6f}deg > "
            f"{SURFACE_FACING_HEADING_TOLERANCE_DEG:.6f}deg)"
        )
        failure_stage = "tracking"
    else:
        final_error = None
        failure_stage = None
    _set_result(
        robot,
        "move_chassis_to_directly_facing_surface",
        {
            **common,
            "ok": ok,
            "error": final_error,
            "failure_stage": failure_stage,
            "geometry": geometry,
            "requested_translation_current_robot_base_m": [
                float(stand_current[0]),
                float(stand_current[1]),
            ],
            "requested_total_spin_deg": float(desired_spin_deg),
            "translation_yaw_drift_deg": float(linear_yaw_drift_deg),
            "requested_post_translation_spin_deg": float(remaining_spin_deg),
            "target_policy_local_base_position_m": (
                target_policy_position.astype(float).tolist()
            ),
            "target_policy_local_base_yaw_deg": float(
                math.degrees(target_policy_yaw)
            ),
            "final_policy_local_base_position_m": (
                final_policy_position.astype(float).tolist()
                if final_pose_available
                else None
            ),
            "final_policy_local_base_yaw_deg": (
                float(math.degrees(final_policy_yaw))
                if final_pose_available
                else None
            ),
            "final_position_error_m": (
                float(final_position_error_m) if final_pose_available else None
            ),
            "final_heading_error_deg": (
                float(final_heading_error_deg) if final_pose_available else None
            ),
            "final_position_ok": position_ok,
            "final_heading_ok": heading_ok,
            "final_heading_tolerance_deg": float(
                SURFACE_FACING_HEADING_TOLERANCE_DEG
            ),
            "translation_phase": dict(translation_report),
            "rotation_phase": dict(rotation_report),
            "action_steps": int(translation_steps + rotation_steps),
            "arrival_verification": (
                "evaluator_base_qvel_plus_policy_local_odometry"
            ),
        },
    )
    yield robot.hold_action()


def _surface_facing_capture_geometry(
    capture: _FrozenCapture,
    points: Sequence[Mapping[str, Any]],
):
    """Backproject one three-click surface request in the capture robot frame."""

    if capture.role != "head":
        raise ValueError("surface-facing motion requires a frozen head capture")
    if capture.rgb is None:
        raise ValueError("frozen head capture has no RGB artifact")
    if capture.rgb.shape[:2] != capture.depth.shape:
        raise ValueError("frozen head RGB and depth_linear resolutions do not match")
    relative_pose = capture.camera.get("robot_relative_pose")
    if not isinstance(relative_pose, Mapping):
        raise ValueError("frozen head capture has no camera-relative pose")
    try:
        camera_position = np.asarray(
            relative_pose.get("pos"), dtype=np.float64
        ).reshape(3)
        camera_quaternion = np.asarray(
            relative_pose.get("quat"), dtype=np.float64
        ).reshape(4)
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen head camera-relative pose is invalid") from exc
    if not (
        np.all(np.isfinite(camera_position))
        and np.all(np.isfinite(camera_quaternion))
        and float(np.linalg.norm(camera_quaternion)) > 1.0e-12
    ):
        raise ValueError("frozen head camera-relative pose is invalid")

    height, width = capture.depth.shape
    try:
        capture_width = int(capture.camera["image_width"])
        capture_height = int(capture.camera["image_height"])
        fx = float(capture.camera["fx"])
        fy = float(capture.camera["fy"])
        cx = float(capture.camera["cx"])
        cy = float(capture.camera["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen head camera calibration is incomplete") from exc
    if (capture_height, capture_width) != (height, width):
        raise ValueError(
            "frozen head camera calibration and RGB-D resolution do not match"
        )
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or min(
        fx, fy
    ) <= 0.0:
        raise ValueError("frozen head camera calibration is invalid")

    camera_robot = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "pos": camera_position,
        "quat": camera_quaternion,
    }
    points_robot = []
    point_metadata = []
    for point in points:
        u = float(point["u"])
        v = float(point["v"])
        px, py = _relative_uv_to_pixel(u, v, width, height)
        depth_m = _depth_value(capture.depth, px, py)
        points_robot.append(
            _unproject_pixel(
                px=px,
                py=py,
                depth_m=depth_m,
                camera=camera_robot,
            )
        )
        point_metadata.append(
            {
                "relative_uv": [u, v],
                "pixel_uv": [int(px), int(py)],
                "depth_m": float(depth_m),
            }
        )
    camera_forward = _quat_to_mat_xyzw(camera_quaternion) @ np.array(
        [0.0, 0.0, -1.0], dtype=np.float64
    )
    solution = _solve_surface_facing_target(
        np.asarray(points_robot, dtype=np.float64),
        camera_forward,
        standoff_m=SURFACE_FACING_STANDOFF_M,
    )
    return solution, point_metadata



def _surface_facing_base_pose(
    value: Mapping[str, Any],
    *,
    label: str,
) -> tuple[np.ndarray, float]:
    try:
        position = np.asarray(value.get("pos"), dtype=np.float64).reshape(3)
        quaternion = np.asarray(value.get("quat"), dtype=np.float64).reshape(4)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} policy-local base pose is invalid") from exc
    if not (
        np.all(np.isfinite(position))
        and np.all(np.isfinite(quaternion))
        and float(np.linalg.norm(quaternion)) > 1.0e-12
    ):
        raise ValueError(f"{label} policy-local base pose is invalid")
    rotation = _quat_to_mat_xyzw(quaternion)
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return position, float(yaw)



def _surface_facing_current_policy_base_pose(world) -> tuple[np.ndarray, float]:
    """Read the action-integrated base pose without any simulator pose handle."""

    provider = getattr(world, "policy_local_base_pose", None)
    if not callable(provider):
        raise ValueError("policy-local chassis odometry is unavailable")
    try:
        planar_pose = np.asarray(provider(), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("policy-local chassis odometry is invalid") from exc
    if planar_pose.size != 3 or not np.all(np.isfinite(planar_pose)):
        raise ValueError("policy-local chassis odometry must be finite x/y/yaw")
    return (
        np.asarray([planar_pose[0], planar_pose[1], 0.0], dtype=np.float64),
        float(planar_pose[2]),
    )


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
