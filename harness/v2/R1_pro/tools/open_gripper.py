"""open_gripper

打开指定 left/right 夹爪；不移动 EEF。

How this tool works:
Release that arm's close-force latch. Send positive finger effort repeatedly while reading finger positions, then retain the open posture. This intentionally releases any object held by that gripper.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.action_constants import GRIPPER_CLOSE_FULL_OPEN_QPOS_TOLERANCE_M
from tools.util.action_constants import GRIPPER_CLOSE_UPPER_LIMIT_M
from tools.util.action_constants import GRIPPER_OPEN_EFFORT_N
from tools.util.action_constants import GRIPPER_OPEN_MAX_STEPS
from tools.util.geometry.contract import ACTION_DIM
from tools.util.motion_common import _check_cancelled
from tools.util.motion_common import _set_result
import numpy as np
from tools.util.tool import main

NAME = 'open_gripper'
SCHEMA = {'type': 'object',
 'properties': {'arm': {'type': 'string', 'enum': ['right', 'left'], 'default': 'right'}},
 'additionalProperties': False,
 'required': ['arm']}


def actions(robot, arm: str = "right", **kwargs):
    del kwargs
    arm_eff = str(arm).strip().lower()
    if arm_eff not in ("left", "right"):
        _set_result(
            robot,
            "open_gripper",
            {"ok": False, "error": "arm must be left or right"},
        )
        yield robot.hold_action()
        return
    before = robot.gripper_qpos_list(arm_eff)
    release_keepalive = getattr(
        robot,
        "release_gripper_close_keepalive",
        None,
    )
    if callable(release_keepalive):
        release_keepalive(arm_eff)
    uses_effort = bool(
        getattr(robot, "gripper_uses_effort", lambda _arm: False)(arm_eff)
    )
    command = [GRIPPER_OPEN_EFFORT_N, GRIPPER_OPEN_EFFORT_N]
    action_steps = 0
    target_reached = False
    for _ in range(GRIPPER_OPEN_MAX_STEPS):
        _check_cancelled(robot, "open gripper")
        if uses_effort:
            yield robot.make_action(
                **{f"gripper_effort_{arm_eff}": command}
            )
        else:
            yield robot.make_action(**{f"gripper_{arm_eff}": [1.0]})
        action_steps += 1
        observed_qpos = np.asarray(
            robot.gripper_qpos_list(arm_eff),
            dtype=np.float64,
        ).reshape(-1)
        target_reached = bool(
            observed_qpos.size == 2
            and np.all(np.isfinite(observed_qpos))
            and np.all(
                observed_qpos
                >= GRIPPER_CLOSE_UPPER_LIMIT_M
                - GRIPPER_CLOSE_FULL_OPEN_QPOS_TOLERANCE_M
            )
        )
        if target_reached:
            break
    hold_target = None
    if uses_effort:
        capture_hold = getattr(
            robot,
            "capture_gripper_hold_qpos",
            None,
        )
        if callable(capture_hold):
            hold_target = capture_hold(
                arm_eff,
                allow_close_effort=False,
            )
    _set_result(
        robot,
        "open_gripper",
        {
            "ok": True,
            "arm": arm_eff,
            "gripper_qpos_before": before,
            "open_action_steps": action_steps,
            "open_target_reached": target_reached,
            "gripper_qpos_after": robot.gripper_qpos_list(arm_eff),
            "commanded_gripper_action": command if uses_effort else [1.0],
            "control_mode": "independent_effort" if uses_effort else "position",
            "post_open_hold_mode": (
                "qpos_effort_servo" if hold_target is not None else "position"
            ),
            "post_open_hold_qpos": hold_target,
            "close_keepalive_active": bool(
                getattr(
                    robot,
                    "gripper_close_keepalive_active",
                    lambda _arm: False,
                )(arm_eff)
            ),
            "implementation": f"official_action_{ACTION_DIM}d_gripper",
        },
    )
    yield robot.hold_action()




if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
