"""Direct joint control with named inputs, implemented as an explicit 27-value action.

Example (small right wrist bend; all other body joints retain their held targets):
  python robot_config/control.py '{"joints":{"right_arm_joint6":-0.9},"steps":15}'

Base: m/s forward, m/s left, rad/s counterclockwise. Joints: absolute radians.
Fingers: N per finger, negative closes, positive opens. Read joints.md first.
This primitive does not plan collision avoidance or guess a safe grasp.
"""
from tools.util.model import ACTION_NAMES, ACTION_SLICES, LOCKED_INDICES, validate_action, proprio_positions
from tools.util.tool import main

SCHEMA = {
    "type": "object", "properties": {
        "joints": {"type": "object", "additionalProperties": {"type": "number"}},
        "base_velocity": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "left_finger_effort": {"type": "number", "minimum": -20, "maximum": 20},
        "right_finger_effort": {"type": "number", "minimum": -20, "maximum": 20},
        "steps": {"type": "integer", "minimum": 1, "maximum": 300},
    }, "additionalProperties": False,
}


def actions(robot, joints=None, base_velocity=None, left_finger_effort=None, right_finger_effort=None, steps=1, **kwargs):
    """Keep established joint targets/holding efforts; replace only named controls."""
    # Read explicit force targets before the baseline hold helper installs
    # its optional finger-position servo. Omitting a force must retain it.
    effort = {arm: robot.gripper_pin_effort_list(arm) for arm in ("left", "right")}
    command = robot.hold_action().tolist()
    for arm, held in effort.items():
        if held is not None:
            command[ACTION_SLICES[f"gripper_{arm}"]] = held
    # hold_action preserves a previously closed gripper's carrying effort.
    if base_velocity is not None:
        command[0:3] = [base_velocity[0] / .75, base_velocity[1] / .75, base_velocity[2]]
    for name, angle in (joints or {}).items():
        if name not in ACTION_NAMES[3:] or "finger" in name:
            raise ValueError(f"{name} is not a rotary joint; see joints.md")
        index = ACTION_NAMES.index(name)
        if index in LOCKED_INDICES and angle != 0:
            raise ValueError(f"{name} is locked at zero")
        command[index] = angle
    if left_finger_effort is not None:
        command[15:17] = [left_finger_effort, left_finger_effort]
    if right_finger_effort is not None:
        command[25:27] = [right_finger_effort, right_finger_effort]
    command = validate_action(command).tolist()
    for _ in range(steps):
        for arm, slot in (("left", 14), ("right", 24)):
            supplied = left_finger_effort if arm == "left" else right_finger_effort
            if supplied is not None:
                robot.set_direct_gripper_effort(arm, command[ACTION_SLICES[f"gripper_{arm}"]])
            robot.set_tool_roll_pin_qpos(arm, command[slot])
        controls = {(name.replace("gripper_", "gripper_effort_")
                     if name.startswith("gripper_") else name): command[indices]
                    for name, indices in ACTION_SLICES.items()}
        yield robot.make_action(**controls)
    robot.set_result({"ok": True, "steps": steps, "action": command,
                      "joint_positions": proprio_positions(robot._official_adapter.proprio_vector())})


if __name__ == "__main__":
    main("joint_control", SCHEMA, actions)
