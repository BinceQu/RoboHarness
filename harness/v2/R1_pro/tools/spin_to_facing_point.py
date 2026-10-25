"""spin_to_facing_point

只水平旋转底盘，让选中的 head 图点落到画面中线；不俯仰、不平移。

How this tool works:
Back-project the selected head-image pixel through its frozen RGB-D and camera pose. Compute its horizontal bearing in the base frame, then rotate the base using measured angular velocity until facing it.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.action_constants import CONTROL_HZ
from tools.util.motion_common import _camera_intrinsics
from tools.util.motion_common import _check_cancelled
from tools.util.motion_common import _load_frozen_capture
from tools.util.motion_common import _set_result
from typing import Any
import math
import numpy as np
from tools.util.tool import main

NAME = 'spin_to_facing_point'
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
                'max_abs_spin': {'type': 'number', 'default': 60.0, 'description': 'deg(安全限幅)'}},
 'additionalProperties': False,
 'required': ['image_id', 'u', 'v']}


def actions(
    robot,
    u: float,
    v: float,
    image_id: str = "",
    max_abs_spin: float = 60.0,
    timeout_s: float = 150.0,
    session_id: str = "",
    **kwargs,
):
    del v, kwargs
    intrinsics = _camera_intrinsics("head", 720, 720)
    source = "official_head_nominal"
    if str(session_id or "").strip() and str(image_id or "").strip():
        try:
            intrinsics = _load_frozen_capture(
                session_id,
                image_id,
            ).camera
            source = "capture_meta"
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            pass
    width = int(intrinsics["image_width"])
    px = float(u) / 1000.0 * max(0, width - 1)
    fx = float(intrinsics["fx"])
    cx = float(intrinsics["cx"])
    ray_angle_deg = math.degrees(math.atan((px - cx) / fx))
    spin_deg = -ray_angle_deg
    limit = abs(float(max_abs_spin))
    if limit > 0.0:
        spin_deg = float(np.clip(spin_deg, -limit, limit))
    if abs(spin_deg) < 0.25:
        spin_deg = 0.0
    spin_rad = math.radians(spin_deg)
    speed = 1.0
    shape = _motion_step_shape(spin_rad, speed)
    max_steps = int(max(0.0, float(timeout_s)) * CONTROL_HZ)
    command = (
        math.copysign(speed, spin_rad) if abs(spin_rad) > 1e-9 else 0.0
    )
    full_steps = min(shape["full_steps"], max_steps)
    for _ in range(full_steps):
        _check_cancelled(robot, "spin to facing point")
        yield robot.set_base_velocity(0.0, 0.0, command)
    remainder = 0.0
    if full_steps == shape["full_steps"] and (
        shape["remainder_scale"] > 1e-9 and full_steps < max_steps
    ):
        remainder = shape["remainder_scale"]
        yield robot.set_base_velocity(
            0.0,
            0.0,
            command * remainder,
        )
    for _ in range(3):
        yield robot.set_base_velocity(0.0, 0.0, 0.0)
    _set_result(
        robot,
        "spin_to_facing_point",
        {
            "ok": True,
            "input_u_relative": float(u),
            "input_u_px": float(px),
            "ray_angle_deg": float(ray_angle_deg),
            "spin_deg": float(spin_deg),
            "intrinsics_source": source,
            "execution_steps": int(full_steps + int(remainder > 1e-9)),
        },
    )
    yield robot.hold_action()


def _motion_step_shape(distance: float, speed: float) -> dict[str, Any]:
    duration = abs(float(distance)) / float(speed) if speed > 0.0 else 0.0
    exact_steps = duration * CONTROL_HZ
    full_steps = int(math.floor(exact_steps + 1e-9))
    remainder = float(np.clip(exact_steps - full_steps, 0.0, 1.0))
    return {
        "duration_s": float(duration),
        "full_steps": int(full_steps),
        "remainder_scale": remainder,
        "steps": int(full_steps + int(remainder > 1e-9)),
    }


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
