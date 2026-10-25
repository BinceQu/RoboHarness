# [14] left_arm_joint8

Tool-axis roll. Direct joint commands can rotate it; baseline motion tools normally pin it at zero.

Controller slot: `action[14]`. Command: **absolute angle in rad**.

Joint: `left_arm_link7` → `left_gripper_link`. Local positive axis: `[0.0, 0.0, 1.0]`.

URDF position limits: -3.14159 to 3.14159 rad. 

The three panels use identical camera and scale. All other joints are zero. Red labels identify illustrative poses that cannot be commanded. This is a motion explanation, not a collision-safe trajectory.

![Three reference positions](left_arm_joint8.png)
