"""Send the evaluator's exact 27-value action for a bounded number of control steps.

Layout: base[0:3], torso[3:7], left arm[7:15], left fingers[15:17],
right arm[17:25], right fingers[25:27]. Revolute targets are radians;
fingers are forces in N. Slot 6 (torso yaw) must be zero. Slots 14 and 24
control the two J8 tool-roll joints, including nonzero targets.
See robot_config/joints.md before using this primitive.
"""
from tools.util.tool import main
from tools.util.model import ACTION_SLICES, validate_action, proprio_positions

NAME = "robot_step"
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "array", "items": {"type": "number"}, "minItems": 27, "maxItems": 27},
        "steps": {"type": "integer", "minimum": 1, "maximum": 900, "default": 1},
    },
    "required": ["action"],
    "additionalProperties": False,
}


def actions(robot, action, steps=1, **kwargs):
    """Every yield is one actual evaluator action; no simulator state is teleported."""
    action = validate_action(action).tolist()
    for _ in range(steps):
        # An explicit primitive replaces previous targets, including a latched
        # finger force. Later tools must hold these new targets, not old ones.
        for arm, slot in (("left", 14), ("right", 24)):
            robot.set_direct_gripper_effort(arm, action[ACTION_SLICES[f"gripper_{arm}"]])
            robot.set_tool_roll_pin_qpos(arm, action[slot])
        controls = {(name.replace("gripper_", "gripper_effort_")
                     if name.startswith("gripper_") else name): action[indices]
                    for name, indices in ACTION_SLICES.items()}
        yield robot.make_action(**controls)
    robot.set_result({"ok": True, "steps": steps, "action": action,
                      "joint_positions": proprio_positions(robot._official_adapter.proprio_vector())})


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
