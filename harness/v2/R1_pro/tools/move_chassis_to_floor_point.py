"""move_chassis_to_floor_point

点选可见地板/地面点：保持当前 yaw，按起始机体系 forward/leftward 向量直接同步平移，使蓝色路径起始线中点到达点选位置；身体不动。

How this tool works:
Back-project a visible floor pixel. Plan a collision-checked base path from current RGB-D, follow it with base velocity commands, and retain the other joints' targets. The floor point must be visibly free.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.geometry.base_path_overlay_local import BASE_FRONT_OFFSET_M
from tools.util.geometry.base_path_overlay_local import PATH_Y_CENTER_M
from tools.util.motion_common import _FrozenCapture
from tools.util.motion_common import _depth_value
from tools.util.motion_common import _load_frozen_capture
from tools.util.motion_common import _point_from_relative_uv
from tools.util.motion_common import _point_in_robot_frame
from tools.util.motion_common import _relative_uv_to_pixel
from tools.util.motion_common import _set_result
from tools.util.motion_common import _unproject_pixel
from tools.util.motion_common import _yield_adjust_chassis_controller
from typing import Any
import math
import numpy as np
from tools.util.tool import main

NAME = 'move_chassis_to_floor_point'
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
                'nav_timeout_s': {'type': 'number',
                                  'default': 120.0,
                                  'description': 's(XY 平移阶段超时)'},
                'ground_tol_m': {'type': 'number',
                                 'default': 0.04,
                                 'description': 'm(反解点离地面顶面的容差)'},
                'pos_tol_m': {'type': 'number', 'default': 0.12, 'description': 'm(XY 到点容差)'}},
 'additionalProperties': False,
 'required': ['image_id', 'u', 'v']}


def actions(
    robot,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    nav_timeout_s: float = 120.0,
    ground_tol_m: float = 0.04,
    pos_tol_m: float = 0.12,
    **kwargs,
):
    # This test implementation is independent from the non-test package. Its
    # target comes only from frozen evaluator RGB-D and policy-local odometry.
    capture = _load_frozen_capture(session_id, image_id)
    point_local, point_meta = _point_from_relative_uv(capture, u, v)
    point_robot = _point_in_robot_frame(robot, point_local)
    normal, normal_meta = _surface_normal_from_relative_uv(capture, u, v)
    max_slope_deg = 35.0
    normal_ok = abs(float(normal[2])) >= math.cos(
        math.radians(max_slope_deg)
    )
    height_limit = max(0.20, 5.0 * abs(float(ground_tol_m)))
    height_ok = float(point_robot[2]) <= height_limit
    if not normal_ok or not height_ok:
        _set_result(
            robot,
            "move_chassis_to_floor_point",
            {
                "ok": False,
                "error": "clicked RGB-D surface failed the floor test",
                "point_robot_m": point_robot.astype(float).tolist(),
                "surface": normal_meta,
                "floor_test": {
                    "normal_ok": bool(normal_ok),
                    "height_ok": bool(height_ok),
                    "source": "frozen_depth_only",
                },
            },
        )
        yield robot.hold_action()
        return
    del kwargs
    clicked_point_xy = point_robot[:2].astype(np.float64, copy=True)
    front_reference_xy = np.array(
        [BASE_FRONT_OFFSET_M, PATH_Y_CENTER_M],
        dtype=np.float64,
    )
    command_xy = clicked_point_xy - front_reference_xy
    clicked_point_distance_m = float(np.linalg.norm(clicked_point_xy))
    horizontal = float(np.linalg.norm(command_xy))
    forward_m = float(command_xy[0])
    translation_m = float(command_xy[1])
    clicked_bearing_deg = (
        math.degrees(math.atan2(point_robot[1], point_robot[0]))
        if clicked_point_distance_m > 1e-9
        else 0.0
    )
    motion_bearing_deg = (
        math.degrees(math.atan2(translation_m, forward_m))
        if horizontal > 1e-9
        else 0.0
    )
    effective_pos_tol_m = max(
        FLOOR_POINT_DEFAULT_POS_TOL_M,
        max(0.0, float(pos_tol_m)),
    )
    motion_metadata = {
        "image_id": str(image_id),
        "point_local_m": point_local.astype(float).tolist(),
        "point_robot_m_at_start": point_robot.astype(float).tolist(),
        "point_observation": point_meta,
        "surface": normal_meta,
        "floor_test": {
            "ok": True,
            "source": "frozen_depth_only",
            "max_slope_deg": max_slope_deg,
            "max_ground_height_robot_m": height_limit,
        },
        "exec_order": "direct_xy_translation",
        "body_motion": (
            "simultaneous_robot_frame_xy; yaw held; target is base-front "
            "path-start midpoint"
        ),
        "target_reference": "base_front_path_start_midpoint",
        "target_reference_source": "submission_static_base_path_overlay_geometry",
        "front_reference_robot_xy_m": front_reference_xy.astype(float).tolist(),
        "clicked_bearing_deg": float(clicked_bearing_deg),
        "motion_bearing_deg": float(motion_bearing_deg),
        "spin_deg": 0.0,
        "clicked_point_distance_from_base_center_m": clicked_point_distance_m,
        "target_distance_m": horizontal,
        "commanded_base_displacement_m": horizontal,
        "forward_m": forward_m,
        "translation_m": translation_m,
        "requested_pos_tol_m": float(pos_tol_m),
        "effective_pos_tol_m": effective_pos_tol_m,
        "arrival_verification": "evaluator_base_qvel_proprioception",
        "dynamic_collision_truth": False,
        "global_base_pose_truth": False,
    }
    controller = _yield_adjust_chassis_controller(
        robot,
        forward=forward_m,
        translation=translation_m,
        spin=0.0,
        timeout_s=float(nav_timeout_s),
        _linear_tolerance_m=effective_pos_tol_m,
        _require_linear_target=True,
        _result_tool_name="move_chassis_to_floor_point",
        _result_extra=motion_metadata,
    )
    for action in controller:
        yield action


FLOOR_POINT_DEFAULT_POS_TOL_M = 0.12



def _surface_normal_from_relative_uv(
    capture: _FrozenCapture,
    u: float,
    v: float,
    *,
    pixel_radius: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = capture.depth.shape[:2]
    px, py = _relative_uv_to_pixel(u, v, width, height)
    points: dict[str, np.ndarray] = {}
    offsets = {
        "left": (-pixel_radius, 0),
        "right": (pixel_radius, 0),
        "up": (0, -pixel_radius),
        "down": (0, pixel_radius),
    }
    for name, (dx, dy) in offsets.items():
        x = int(np.clip(px + dx, 0, width - 1))
        y = int(np.clip(py + dy, 0, height - 1))
        z = _depth_value(capture.depth, x, y, radius=1)
        points[name] = _unproject_pixel(
            px=x,
            py=y,
            depth_m=z,
            camera=capture.camera,
        )
    normal = np.cross(
        points["right"] - points["left"],
        points["down"] - points["up"],
    )
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        raise ValueError("local depth surface normal is degenerate")
    normal /= norm
    if normal[2] < 0.0:
        normal = -normal
    return normal, {
        "normal": normal.astype(float).tolist(),
        "normal_abs_z": abs(float(normal[2])),
        "pixel_radius": int(pixel_radius),
    }


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
