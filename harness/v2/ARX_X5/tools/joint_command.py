"""Send exactly one complete 14-value absolute joint action to the evaluator.

Six radians plus one normalized gripper value per arm. No IK, hidden motion,
reset, simulator access, automatic repeat or collision checking occurs.
"""

import math
from tools.util.x5_kinematics import X5_JOINT_LOWER, X5_JOINT_UPPER

KEYS = {
    "left_arm_joint_state": 6,
    "left_ee_joint_state": 1,
    "right_arm_joint_state": 6,
    "right_ee_joint_state": 1,
}


def validate_command(action):
    if not isinstance(action, dict) or set(action) != set(KEYS):
        raise ValueError("Supply all four joint-state fields; exactly 14 scalars")
    result = {}
    for key, size in KEYS.items():
        values = action[key]
        if not isinstance(values, (list, tuple)) or len(values) != size:
            raise ValueError(f"{key} must contain {size} scalars")
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in values
        ):
            raise ValueError("Joint commands must be finite numbers")
        bounds = [(0.0, 1.0)] if size == 1 else zip(X5_JOINT_LOWER, X5_JOINT_UPPER)
        for value, (low, high) in zip(values, bounds):
            if not low <= value <= high:
                raise ValueError(f"{key}: {value} outside [{low}, {high}]")
        result[key] = [float(v) for v in values]
    return result


def joint_command(ctx, action, session_id=""):
    command = validate_command(action)
    # Latch both grippers exactly as the normal arm-motion tools do.
    yield ctx.world.make_joint_action(
        left_arm=command["left_arm_joint_state"],
        left_gripper=command["left_ee_joint_state"][0],
        right_arm=command["right_arm_joint_state"],
        right_gripper=command["right_ee_joint_state"][0],
    )
    ctx.set_result(
        {
            "ok": True,
            "tool": "joint_command",
            "action": command,
            "observation_sequence": ctx.world._official_adapter.status()["sequence"],
            "collision_checked": False,
        }
    )


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("joint_command", joint_command)
