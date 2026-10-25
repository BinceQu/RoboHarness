# Robot body and atomic control

This is the RoboDojo dual-X5 fixed rig: left and right arms, each with six
revolute joints plus a parallel gripper. There is no mobile base or trunk.
The model and meshes here are used only for local kinematics and offline
body visualization. They are not the evaluator's scene or object assets.

View `visuals/three_views.png`. Each `visuals/SIDE_jointN.png` shows one
joint at −90°, 0°, +90° with all other arm joints at zero, using the same
camera and scale. Finger joints instead show 0, 22 and 44 mm. Orange marks
the selected link; cyan is its positive axis. These are explanatory model
poses, not guaranteed collision-free commands. Positive angles follow the
right-hand rule about the indicated local axis.

The manual diagrams are reference poses. To inspect the current public joint
state as front/side/top views, run `.venv/bin/python tools/read_proprioception.py '{}'`
from the workspace root. Inspect its returned image, or open that path with
native Read. This tool sends no action and renders only when requested;
ordinary tools update numeric joints without rendering. Request it when you
need to understand posture, not automatically after every action. Its episode
and sequence bind the image to that observation; old renders are historical.

| Per-arm index | URDF joint | Meaning | Local axis | Absolute command |
|---|---|---|---|---|
| 0 | joint1 | Base/shoulder yaw | +Z | radians, [−10,10] |
| 1 | joint2 | Shoulder pitch | +Y | radians, [−10,10] |
| 2 | joint3 | Elbow pitch | +Y in its rotated joint frame | radians, [−10,10] |
| 3 | joint4 | Wrist bend | +Y | radians, [−10,10] |
| 4 | joint5 | Wrist swivel | +Z | radians, [−10,10] |
| 5 | joint6 | Tool roll | +X | radians, [−3.14,3.14] |
| gripper | joint7 and joint8 together | Parallel fingers | opposing sliding axes | normalized opening, [0,1] |

The large arm limits above are the source URDF/controller limits; they do
not certify collision clearance. Read each triptych and current posture.
The left and right arms use identical joint ordering. An arm command is an
absolute target, not a delta. A complete evaluator action contains exactly
14 scalars and includes both arms and both grippers:

```python
from robot_config.control import Robot

robot = Robot()
observation = robot.observe()  # No simulator step; includes image paths.
# Read current state first. Replace every value with the intended target.
action = {
    "left_arm_joint_state": [0, 0, 0, 0, 0, 0],
    "left_ee_joint_state": [1],
    "right_arm_joint_state": [0, 0, 0, 0, 0, 0],
    "right_ee_joint_state": [1],
}
# robot.step(action)  # One evaluator step, then the resulting observation.
```

The same API is callable from Bash:

```bash
.venv/bin/python robot_config/control.py '{"action":{"left_arm_joint_state":[0,0,0,0,0,0],"left_ee_joint_state":[1],"right_arm_joint_state":[0,0,0,0,0,0],"right_ee_joint_state":[1]}}'
```

Gripper 0 means close, 1 means open. The evaluator maps opening `g` to
finger translation `−0.01 + 0.054*g` metres; the negative closing target
creates contact pressure. The URDF geometric interval is 0 to 0.044 m.
The public gripper state can be a normalized control proxy, not measured
finger contact. Body rendering uses this proxy and cannot prove a grasp.
Motion macros latch requested gripper openings across subsequent arm moves.

Evaluator world: +X right, +Y forward, +Z up. Arm bases are at
left `(−0.30,−0.45,0.765)` and right `(0.30,−0.45,0.765)` metres, both
rotated +90° about world Z. EEF poses use `[x,y,z,qw,qx,qy,qz]` in world
coordinates. Some original tool geometry uses robot-base axes (forward,
left, up); returned fields explicitly identify their frame. Do not mix them.

RGB-D clicks use normalized coordinates. At image width W and height H:
`px=round(u*(W−1)/1000)`, `py=round(v*(H−1)/1000)`.
Depth is in metres. Camera matrices are preserved from the same observation;
USD camera axes are converted to optical axes (+X right,+Y down,+Z forward)
inside the geometry code. Neither rendering nor the harness resizes RGB-D.

The policy sees only instruction, RGB, depth, camera matrices and public
joint/EEF state. Reward, success flags, segmentation and scene object poses
are excluded by `tools/util/observation.py`. Only the evaluator owns physics,
reset, task termination and scoring. See `transport.md` for the exact boundary.
