# Joint and controller index

Axes are local to each joint origin, not the image. Positive rotation uses the right-hand rule.

| Slot | Joint | Meaning | Command |
|---|---|---|---|
| 0 | base forward | +X forward | normalized −1…+1; ×0.75 m/s |
| 1 | base left | +Y left | normalized −1…+1; ×0.75 m/s |
| 2 | base yaw | +Z counterclockwise | normalized −1…+1; ×1 rad/s |
| 3 | [torso_joint1](joints/torso_joint1.md) | Lower torso pitch; changes the trunk's lean and camera position. | absolute angle in rad |
| 4 | [torso_joint2](joints/torso_joint2.md) | Middle torso pitch; combines with T1/T3 to change height. | absolute angle in rad |
| 5 | [torso_joint3](joints/torso_joint3.md) | Upper torso pitch; combines with T1/T2 to orient the chest/head. | absolute angle in rad |
| 6 | [torso_joint4](joints/torso_joint4.md) | Chest yaw. Locked at zero by the submitted evaluator profile. | 0 only (locked) |
| 7 | [left_arm_joint1](joints/left_arm_joint1.md) | First shoulder rotation: moves the entire arm about the marked axis. | absolute angle in rad |
| 8 | [left_arm_joint2](joints/left_arm_joint2.md) | Second shoulder rotation: swings the entire upper arm about its local axis. | absolute angle in rad |
| 9 | [left_arm_joint3](joints/left_arm_joint3.md) | Upper-arm axial rotation: changes the elbow's bending plane. | absolute angle in rad |
| 10 | [left_arm_joint4](joints/left_arm_joint4.md) | Elbow bend: rotates the forearm relative to the upper arm. | absolute angle in rad |
| 11 | [left_arm_joint5](joints/left_arm_joint5.md) | Forearm axial rotation: rotates the wrist and gripper about the forearm. | absolute angle in rad |
| 12 | [left_arm_joint6](joints/left_arm_joint6.md) | Wrist bend: changes fingertip direction relative to the forearm. | absolute angle in rad |
| 13 | [left_arm_joint7](joints/left_arm_joint7.md) | Final wrist rotation before J8; see the marked local axis. | absolute angle in rad |
| 14 | [left_arm_joint8](joints/left_arm_joint8.md) | Tool-axis roll. Direct joint commands can rotate it; baseline motion tools normally pin it at zero. | absolute angle in rad |
| 15 | [left_gripper_finger_joint1](joints/left_gripper_finger_joint1.md) | First sliding finger. Position 0 m = closed, 0.05 m = open; action is effort in N. | force in N, −20…+20 |
| 16 | [left_gripper_finger_joint2](joints/left_gripper_finger_joint2.md) | Opposing sliding finger. Position 0 m = closed, 0.05 m = open; action is effort in N. | force in N, −20…+20 |
| 17 | [right_arm_joint1](joints/right_arm_joint1.md) | First shoulder rotation: moves the entire arm about the marked axis. | absolute angle in rad |
| 18 | [right_arm_joint2](joints/right_arm_joint2.md) | Second shoulder rotation: swings the entire upper arm about its local axis. | absolute angle in rad |
| 19 | [right_arm_joint3](joints/right_arm_joint3.md) | Upper-arm axial rotation: changes the elbow's bending plane. | absolute angle in rad |
| 20 | [right_arm_joint4](joints/right_arm_joint4.md) | Elbow bend: rotates the forearm relative to the upper arm. | absolute angle in rad |
| 21 | [right_arm_joint5](joints/right_arm_joint5.md) | Forearm axial rotation: rotates the wrist and gripper about the forearm. | absolute angle in rad |
| 22 | [right_arm_joint6](joints/right_arm_joint6.md) | Wrist bend: changes fingertip direction relative to the forearm. | absolute angle in rad |
| 23 | [right_arm_joint7](joints/right_arm_joint7.md) | Final wrist rotation before J8; see the marked local axis. | absolute angle in rad |
| 24 | [right_arm_joint8](joints/right_arm_joint8.md) | Tool-axis roll. Direct joint commands can rotate it; baseline motion tools normally pin it at zero. | absolute angle in rad |
| 25 | [right_gripper_finger_joint1](joints/right_gripper_finger_joint1.md) | First sliding finger. Position 0 m = closed, 0.05 m = open; action is effort in N. | force in N, −20…+20 |
| 26 | [right_gripper_finger_joint2](joints/right_gripper_finger_joint2.md) | Opposing sliding finger. Position 0 m = closed, 0.05 m = open; action is effort in N. | force in N, −20…+20 |

The steering and wheel joints in model.urdf are driven by the evaluator's base controller. They have no independent action slot. Use slots 0–2; do not invent wheel-angle commands.

Finger position is sensed in metres, while finger commands are forces. Negative effort closes; positive effort opens. Keep a negative effort while carrying an object. Never use a finger position as its force command.
