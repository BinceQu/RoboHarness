"""adjust_height

只调整机器人高度 upward，单位米；超出 submission-local LUT 可达范围时自动饱和到最高或最低缓存行，完成后拍 head camera。

How this tool works:
Read torso angles and solve a trunk posture for an upward displacement in metres. Interpolate the three unlocked torso targets, retaining the camera pitch as specified by the controller and protecting the arm posture.
Check actual_upward_m and ok: if the current posture maps to a LUT hold but
the unsaturated requested height remains unreached, the tool returns failure.
It does not create an unvalidated trajectory to connect that posture to the LUT.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.action_constants import ADJUST_HEIGHT_DEFAULT_TOLERANCE_RAD
from tools.util.motion_common import _set_result
from tools.util.motion_common import _yield_adjust_height_motion
from tools.util.tool import main

NAME = 'adjust_height'
SCHEMA = {'type': 'object',
 'properties': {'upward': {'type': 'number', 'default': 0.0, 'description': 'm(+升/-降)'}},
 'additionalProperties': False}


def actions(
    robot,
    upward: float = 0.0,
    timeout_s: float = 120.0,
    **kwargs,
):
    tolerance = float(
        kwargs.get(
            "tol",
            kwargs.get("tolerance_rad", ADJUST_HEIGHT_DEFAULT_TOLERANCE_RAD),
        )
    )
    motion = _yield_adjust_height_motion(
        robot,
        upward=float(upward),
        timeout_s=float(timeout_s),
        tolerance=tolerance,
        stage_prefix="adjust height",
    )
    while True:
        try:
            yield next(motion)
        except StopIteration as stop:
            report = stop.value
            break
    _set_result(robot, "adjust_height", report)
    yield robot.hold_action()




if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
