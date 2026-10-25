# Physical robot API

Start with `joints.md`. Open `joints/JOINT_NAME.png` for the selected joint's
−90°, 0°, +90° poses and `joints/JOINT_NAME.md` for its local axis, limits and
controller index. Angles outside the physical limit are illustrated and
labelled, never sent as valid commands. The sliding fingers use metres in
their diagrams because they translate rather than rotate.

`three_views.png` shows the model's zero posture. For a current measured posture,
run `python tools/read_proprioception.py` from the workspace root and inspect its
returned image. Both use URDF visual meshes and forward kinematics, rendered
on the CPU without the simulator. The tool consumes no physics ticks and shows
no scene or held objects. `../perception/proprioception/robot_three_views.png`
is the last requested render, not an automatically refreshed view; its sequence
is recorded in `../perception/proprioception/render.json`. Request another
render after motion only when needed. Highlight colours in joint illustrations
show the selected link, not a camera material.

## Direct commands

```bash
python robot_config/control.py --schema
python robot_config/control.py '{"joints":{"right_arm_joint6":-0.9},"steps":15}'
```

The example requests an absolute wrist angle in radians; it is not a relative
rotation and not universally collision-free. Inspect current posture first.
`base_velocity` is `[forward_m_per_s, left_m_per_s, counterclockwise_rad_per_s]`.
`left_finger_effort` and `right_finger_effort` apply newtons to each finger of
that hand. Omitted fields retain the current targets/holding efforts.
An explicit finger force also remains the target for subsequent tool holds,
until another explicit finger command, open/close operation or episode reset.
Zero means zero force; it does not mean “keep gripping”.

For the exact evaluator vector, use `tools/robot_step.py`: 27 finite values,
one action per 1/30-second physics step. The complete ordering is in `joints.md`.
Base slots are normalized velocities; arm and torso slots are absolute radians;
finger slots are newtons. Only slot 6 (torso yaw) is locked by this evaluator
profile. J8 is controllable through slots 14 and 24. The task01 motion tools
normally pin J8 to zero for planning; restore both J8 targets to zero before
returning from custom joint motions to the baseline navigation/pick/place flow.

No action sets a simulator pose directly. The evaluator applies the command
through its own joint controllers and physics. A tool receives the next RGB-D
and proprioception only after the action has been consumed. While the agent
reads/thinks, the evaluator waits and consumes no additional action steps.

## Observation interface

Every tool result gives `perception`, `latest_clickable_image_id`, `images`,
`rollout_budget`, and `joint_commands_sent`. `perception/latest.json` references
one coherent snapshot: native RGB, linear depth in metres, camera poses relative
to the robot base, 65 measured proprioception values and joint-name positions.
There is no simulator object list, segmentation truth, hidden goal state or
object-pose lookup. Tracks are inferred from these allowed observations.

The readable Python client is `tools/util/robot.py` (`tools.util.robot.Robot`). Its
`observe()` method reads allowed arrays without moving. `make_action(...)`
builds target vectors; `step(action)` sends exactly one vector and waits for
its resulting observation. Public tool files show how feedback loops use it.
Use the two recorded command CLIs above when issuing direct control from Bash.

For a custom Python script, `robot.capture("head")` and
`robot.command(action, steps=15)` execute those same readable files and return
the same recorded receipts and refreshed perception. Use these public methods
for a complete operation. The lower-level `step` loop is intended for a tool's
already-recorded `actions()` generator.

`model.urdf`, `meshes/`, `assets/` and the local Python environment contain
everything the client, planner and offline renderer need. Scene data and the
physics simulator belong to the separate evaluator, not this robot harness.
See `transport.md` for the exact process and wire boundary.
