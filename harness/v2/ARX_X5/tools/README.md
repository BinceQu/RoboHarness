# Tool catalogue

Run `.venv/bin/python tools/NAME.py 'JSON'`. Session identity is inserted automatically.

Click coordinates retain the RoboDojo normalized 0..1000 contract; images are not resized.

Each file contains its algorithm. Shared code is in util/. Use --schema for the callable signature.

`ctx` and `session_id` are supplied by the local worker; omit them in Bash
arguments. Other parameter names and defaults are shown below. Image IDs
always identify a frozen capture, not a filename chosen by the model.

| Purpose | Tools | Evaluator actions |
|---|---|---|
| See head / wrists | `capture_head_camera`, `capture_left_wrist_camera`, `capture_right_wrist_camera` | None |
| Inspect current body posture on demand | `read_proprioception` | None; public joints → offline front/side/top image, no camera capture |
| Read a frozen depth / track named surface points | `read_depth`, `track_object_distance` | None |
| Check arm reach | `measure_shoulder_distance`, `move_to_reach_point` | None; fixed bases never move |
| Plan a grasp / press / translation | `plan_grasp_point_filter_rgbd_lite`, `plan_press_point`, `plan_eef_translation_to_uvd_point` | None; inspect then execute |
| Execute the signed plan | `exec_plan_pose` | Bounded joint commands, with feedback |
| Change gripper opening | `open_gripper`, `close_gripper` | Joint commands with latched opening |
| Adjust an arm in camera axes | `adjust_left_eef_pose_in_head_frame`, `adjust_right_eef_pose_in_head_frame`, `adjust_left_eef_pose_in_wrist_frame`, `adjust_right_eef_pose_in_wrist_frame` | Local IK → joint commands |
| Move held points using RGB-D / constraints | `move_point_to_point`, `move_tracked_point` | Plans or joint commands, according to mode |
| Retract and prepare an arm | `set_arm_to_grasp_position` | Joint commands, preserving gripper targets |
| Exact atomic action | `joint_command`, `robot_config.control.Robot.step` | Exactly one 14-value command |
| Select / exit instructions | `activate_skill`, `deactivate_skill` | None |
| Unsupported mobile-body operations | `adjust_chassis`, `adjust_pitch`, `adjust_height`, `spin_to_facing_point`, `move_chassis_to_floor_point`, `reset_body` | None; explicit unsupported result |

Tool returns contain measured outcomes and errors, without a generic success
flag. Check the tool's specific completion/progress fields and current images;
absence of an error does not prove arrival, contact or grasp.

## read_proprioception

```bash
.venv/bin/python tools/read_proprioception.py '{}'
```

No arguments. Read current public joints and render the local robot model;
inspect the returned image or use native Read on its path. This is the only
tool that refreshes the body three-view. Ordinary tools still update numeric
joint state, RGB-D and tracks. Do not request a render after every action:
use it when posture inspection is useful. The image shows no scene or held
objects and cannot be used for clicks or grasp confirmation. Its episode and
sequence are recorded with the image in this call's immutable history.

## capture_head_camera

```python
capture_head_camera(ctx, session_id: 'str', **kwargs)
```

See the complete implementation in capture_head_camera.py


## capture_left_wrist_camera

```python
capture_left_wrist_camera(ctx, session_id: 'str', **kwargs)
```

See the complete implementation in capture_left_wrist_camera.py


## capture_right_wrist_camera

```python
capture_right_wrist_camera(ctx, session_id: 'str', **kwargs)
```

See the complete implementation in capture_right_wrist_camera.py


## read_depth

```python
read_depth(ctx, session_id: 'str', image_id: 'str', u: 'float', v: 'float')
```

See the complete implementation in read_depth.py


## track_object_distance

```python
track_object_distance(ctx, session_id: 'str', image_id: 'str', points: 'list[dict[str, Any]]')
```

Ground a complete named set on one frozen frame, then track it live.


## move_chassis_to_floor_point

```python
move_chassis_to_floor_point(ctx, session_id: 'str', image_id: 'str', u: 'int', v: 'int', nav_timeout_s: 'float' = 120.0, ground_tol_m: 'float' = 0.04, pos_tol_m: 'float' = 0.12, **kwargs)
```

See the complete implementation in move_chassis_to_floor_point.py


## adjust_chassis

```python
adjust_chassis(ctx, forward: 'float' = 0.0, translation: 'float' = 0.0, spin: 'float' = 0.0, timeout_s: 'float' = 120.0, **kwargs)
```

See the complete implementation in adjust_chassis.py


## adjust_pitch

```python
adjust_pitch(ctx, degree: 'float' = 0.0, timeout_s: 'float' = 120.0, **kwargs)
```

See the complete implementation in adjust_pitch.py


## adjust_height

```python
adjust_height(ctx, upward: 'float' = 0.0, timeout_s: 'float' = 120.0, **kwargs)
```

See the complete implementation in adjust_height.py


## spin_to_facing_point

```python
spin_to_facing_point(ctx, u: 'float', v: 'float', image_id: 'str' = '', max_abs_spin: 'float' = 60.0, timeout_s: 'float' = 150.0, session_id: 'str' = '', **kwargs)
```

See the complete implementation in spin_to_facing_point.py


## open_gripper

```python
open_gripper(ctx, arm: 'str' = 'right', **kwargs)
```

See the complete implementation in open_gripper.py


## close_gripper

```python
close_gripper(ctx, arm: 'str' = 'right', **kwargs)
```

See the complete implementation in close_gripper.py


## adjust_left_eef_pose_in_head_frame

```python
adjust_left_eef_pose_in_head_frame(ctx, forward: 'float | None' = None, upward: 'float | None' = None, leftward: 'float | None' = None, x: 'float | None' = None, y: 'float | None' = None, z: 'float | None' = None, roll: 'float' = 0.0, pitch: 'float' = 0.0, yaw: 'float' = 0.0, pos_tol: 'float' = 0.012, ori_tol_deg: 'float' = 3.0, max_steps: 'int' = 240, timeout_s: 'float' = 60.0)
```

See the complete implementation in adjust_left_eef_pose_in_head_frame.py


## adjust_right_eef_pose_in_head_frame

```python
adjust_right_eef_pose_in_head_frame(ctx, forward: 'float | None' = None, upward: 'float | None' = None, leftward: 'float | None' = None, x: 'float | None' = None, y: 'float | None' = None, z: 'float | None' = None, roll: 'float' = 0.0, pitch: 'float' = 0.0, yaw: 'float' = 0.0, pos_tol: 'float' = 0.012, ori_tol_deg: 'float' = 3.0, max_steps: 'int' = 240, timeout_s: 'float' = 60.0)
```

See the complete implementation in adjust_right_eef_pose_in_head_frame.py


## adjust_left_eef_pose_in_wrist_frame

```python
adjust_left_eef_pose_in_wrist_frame(ctx, forward: 'float' = 0.0, leftward: 'float' = 0.0, upward: 'float' = 0.0, roll: 'float' = 0.0, pitch: 'float' = 0.0, yaw: 'float' = 0.0, pos_tol: 'float' = 0.012, ori_tol_deg: 'float' = 3.0, max_steps: 'int' = 240, timeout_s: 'float' = 60.0)
```

See the complete implementation in adjust_left_eef_pose_in_wrist_frame.py


## adjust_right_eef_pose_in_wrist_frame

```python
adjust_right_eef_pose_in_wrist_frame(ctx, forward: 'float' = 0.0, leftward: 'float' = 0.0, upward: 'float' = 0.0, roll: 'float' = 0.0, pitch: 'float' = 0.0, yaw: 'float' = 0.0, pos_tol: 'float' = 0.012, ori_tol_deg: 'float' = 3.0, max_steps: 'int' = 240, timeout_s: 'float' = 60.0)
```

See the complete implementation in adjust_right_eef_pose_in_wrist_frame.py


## move_point_to_point

```python
move_point_to_point(ctx, session_id: 'str', image_id: 'str', points: 'list[Any]', above_target_point_m: 'float' = 0.0, pos_tol: 'float' = 0.012, ori_tol_deg: 'float' = 5.0, max_steps: 'int' = 360, timeout_s: 'float' = 90.0)
```

See the complete implementation in move_point_to_point.py


## move_tracked_point

```python
move_tracked_point(ctx, points: 'list[dict[str, Any]]', relations: 'Sequence[Mapping[str, Any]] | None' = None, pos_tol: 'float' = 0.03, ori_tol_deg: 'float' = 20.0, max_steps: 'int' = 360, timeout_s: 'float' = 90.0, quick_constraint: 'Mapping[str, Any] | None' = None, quick_constraints: 'Sequence[Mapping[str, Any]] | None' = None, mode: 'str | None' = None, execution_mode: 'str' = 'exec', inequalities: 'Sequence[Mapping[str, Any]] | None' = None)
```

Move or plan tracked/predefined points with spatial collinear order.

The reserved names ``left_finger_tip``, ``right_finger_tip``, and
``gripper_slide_center`` bind to the selected physical X5 gripper using
synchronized evaluator proprioception and submission-local FK. They do
not require ``track_object_distance`` registration.

Explicit collinear quick/typed rows are the required axial sequence: 1,3,2
places 3 between 1 and 2. Coordinate rows alone do not activate that quick
constraint. Robot-base reading direction is left-to-right (-Y), top-to-bottom
(-Z), or near-to-far (+X), using the largest axis component (ties Y,Z,X).
Camera viewpoint does not change the order. Reversed or shuffled endpoints
cannot be returned as successful approximate/overlap-fallback plans.

The ``faceto`` and ``reverse_faceto`` camera pose is frozen from the same
evaluator observation as the tracked points and is reused for planning,
signed-plan reload, and live validation.

``inequalities`` contains strict affine variable constraints. Each item is
``{"lhs": <variable-or-affine-expression>, "op": ">"|"<", "rhs":
<finite-number>}``; for example ``a-b > 0`` selects the signed vertical
ordering when two point rows use Z variables ``a`` and ``b``.


## plan_eef_translation_to_uvd_point

```python
plan_eef_translation_to_uvd_point(ctx, **kwargs)
```

See the complete implementation in plan_eef_translation_to_uvd_point.py


## move_to_reach_point

```python
move_to_reach_point(ctx, session_id: 'str', image_id: 'str', u: 'int', v: 'int', reach: 'float' = 0.0, nav_timeout_s: 'float' = 120.0, keep_ori_arm: 'str' = 'none', **kwargs)
```

See the complete implementation in move_to_reach_point.py


## measure_shoulder_distance

```python
measure_shoulder_distance(ctx, session_id: 'str', image_id: 'str', u: 'int', v: 'int', object_name: 'str' = '', **kwargs)
```

See the complete implementation in measure_shoulder_distance.py


## plan_grasp_point_filter_rgbd_lite

```python
plan_grasp_point_filter_rgbd_lite(ctx, session_id: 'str', image_id: 'str', u: 'float', v: 'float', arm: 'str' = 'right', seed: 'int' = 42, plan_arm: 'str' = 'any', **kwargs)
```

See the complete implementation in plan_grasp_point_filter_rgbd_lite.py


## plan_press_point

```python
plan_press_point(ctx, **kwargs)
```

See the complete implementation in plan_press_point.py


## exec_plan_pose

```python
exec_plan_pose(ctx, **kwargs)
```

See the complete implementation in exec_plan_pose.py


## set_arm_to_grasp_position

```python
set_arm_to_grasp_position(ctx, arm: 'str' = 'right', keep_ori_arm: 'str' = 'none', gripper: 'str' = 'keep', max_dq_per_step: 'float' = 0.12, tol: 'float' = 0.08, timeout_s: 'float' = 45.0, **kwargs)
```

See the complete implementation in set_arm_to_grasp_position.py


## reset_body

```python
reset_body(ctx, keep_ori_arm: 'str' = 'none', timeout_s: 'float' = 45.0, trunk_max_step: 'float | None' = None, shoulder_iters: 'int' = 4, pitch_deg: 'float' = 0.0, **kwargs)
```

See the complete implementation in reset_body.py


## joint_command

```python
joint_command(ctx, action, session_id="")
```

See the complete implementation in joint_command.py


## Skill state

`activate_skill.py '{"name":"pick-up-object"}'` loads one skill. Empty name lists available skills.

`deactivate_skill.py '{}'` clears the task skill and preserves robodojo-baseline.
