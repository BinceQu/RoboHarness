"""adjust_chassis

只调整底盘：机体系 forward/translation(+左/-右) 同帧合成为二维直移，随后执行水平 spin；不改 pitch/upward，完成后拍 head camera。

How this tool works:
Combine forward/left offsets into one base-frame displacement, follow it using measured base velocity, then rotate by spin degrees. Preserve arm, torso and finger targets. Detect stalls using measured base velocity; this tool does not use an RGB-D obstacle guard. The base may stop and settle before reaching the requested distance: inspect actual, linear_target_reached and obstacle_limited before treating the move as complete. Absence of an error does not mean the requested motion was completed.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.motion_common import _yield_adjust_chassis_controller
from tools.util.tool import main

NAME = 'adjust_chassis'
SCHEMA = {'type': 'object',
 'properties': {'forward': {'type': 'number', 'default': 0.0, 'description': 'm(+前进/-倒车)'},
                'translation': {'type': 'number', 'default': 0.0, 'description': 'm(+左移/-右移)'},
                'spin': {'type': 'number', 'default': 0.0, 'description': 'deg(+左转CCW/-右转)'}},
 'additionalProperties': False}


def actions(
    robot,
    forward: float = 0.0,
    translation: float = 0.0,
    spin: float = 0.0,
    timeout_s: float = 120.0,
    **kwargs,
):
    """Track one robot-frame XY vector, then spin, using official actions."""
    controller = _yield_adjust_chassis_controller(
        robot,
        forward=forward,
        translation=translation,
        spin=spin,
        timeout_s=timeout_s,
        vmax=kwargs.get("vmax", 0.5),
        wmax=kwargs.get("wmax", 1.0),
    )
    for action in controller:
        yield action




if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
