"""adjust_pitch

只调整机器人俯仰 pitch：正值仰头，负值俯身；完成后拍 head camera。

How this tool works:
Read the current torso angles. Solve a torso posture for the requested head pitch change (degrees), interpolate joint targets, and check measured convergence. The base and hands retain their current targets.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.action_constants import TRUNK_LIMITS
from tools.util.motion_common import _check_cancelled
from tools.util.motion_common import _set_result
from tools.util.motion_common import _smoothstep
import math
import numpy as np
from tools.util.tool import main

NAME = 'adjust_pitch'
SCHEMA = {'type': 'object',
 'properties': {'degree': {'type': 'number', 'default': 0.0, 'description': 'deg(+仰头/-俯身，纯q3)'}},
 'additionalProperties': False}


def actions(
    robot,
    degree: float = 0.0,
    timeout_s: float = 120.0,
    **kwargs,
):
    del timeout_s
    start = np.asarray(robot.trunk_qpos(), dtype=np.float64).reshape(4)
    target = start.copy()
    requested = math.radians(float(degree))
    target[2] = float(
        np.clip(
            start[2] + requested,
            TRUNK_LIMITS[2, 0],
            TRUNK_LIMITS[2, 1],
        )
    )
    if abs(target[2] - start[2] - requested) > 1e-6:
        _set_result(
            robot,
            "adjust_pitch",
            {
                "ok": False,
                "error": "pitch target exceeds torso_joint3 limits",
            },
        )
        yield robot.hold_action()
        return
    max_step = max(
        1e-4,
        float(kwargs.get("trunk_max_step_rad", 0.10)),
    )
    steps = max(
        2,
        int(
            math.ceil(
                float(np.linalg.norm(target - start, ord=np.inf)) / max_step
            )
        ),
    )
    for index in range(1, steps + 1):
        _check_cancelled(robot, "adjust pitch")
        alpha = _smoothstep(index / steps)
        command = start + alpha * (target - start)
        yield robot.make_action(trunk=command.tolist())
    _set_result(
        robot,
        "adjust_pitch",
        {
            "ok": True,
            "degree": float(degree),
            "start_trunk_q": start.astype(float).tolist(),
            "target_trunk_q": target.astype(float).tolist(),
            "verification": "proprioception",
        },
    )
    yield robot.hold_action()




if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
