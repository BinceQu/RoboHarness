# Robot tools

Each file is directly executable: `python tools/NAME.py '{"key":"value"}'`.
Read `SCHEMA` for physical-tool inputs, or run with `--schema` without moving
the robot. The two skill tools document their `name`/`reason` fields in their
files and expose `--help`.
`actions(robot, ...)` is the actual sequence: each `yield` sends one 27-value
evaluator command, waits for new proprioception and RGB-D, then continues.
The tool's private helpers are below its main sequence in the same file.
Imports from `tools.util` refer only to shared code in this workspace; NumPy,
PyTorch and the other third-party imports are installed in its own `.venv`.
Nothing here calls an HTTP macro-tool service or imports another project.

Results contain measured outcomes and errors, without a generic success flag.
For `adjust_chassis`, check `actual`, `linear_target_reached`, and
`obstacle_limited`: stopping safely does not mean reaching the requested
displacement. Absence of an error is not proof of arrival, contact or grasp.
Use the tool's specific completion fields and current images.

Results keep measurements, errors, tracks and ticks. Paths within this workspace
are relative to its root. Inspect attached pixels; if absent, open `images[].path`
with your image reader. Only the newest head view is clickable. Each image has
one short reminder; implementation details remain in `record`. Its sibling
`input.json` contains the exact call arguments. Output is already compact;
no need to filter it with `grep` or `head`.

| Tools | Purpose |
| --- | --- |
| `capture_head_camera`, `capture_left_wrist_camera`, `capture_right_wrist_camera` | Read and freeze a current camera view. No physics step. |
| `read_proprioception` | On demand: read measured joints and render front/left/top views of the robot alone. No motion, ticks or camera capture. |
| `adjust_chassis`, `adjust_pitch`, `adjust_height`, `reset_body` | Move base or torso; reset_body restores posture, not the episode. |
| `spin_to_facing_point`, `move_chassis_to_floor_point`, `move_chassis_to_directly_facing_surface`, `move_to_reach_point` | Use a current head-image point to plan/execute base motion. |
| `measure_shoulder_distance` | Project a current head point into measured RGB-D and report reach distance. |
| `set_arm_to_grasp_position` | Move one arm to the pre-grasp posture and requested gripper state. |
| `plan_grasp_point_filter_rgbd_lite` | Compute and filter reachable grasp poses from frozen RGB-D. |
| `plan_press_point` | Preserved baseline entry; this profile reports this mode unavailable. |
| `exec_plan_pose` | Execute an existing locally stored plan with joint targets. |
| `open_gripper`, `close_gripper` | Control two finger efforts and inspect measured closure. |
| `adjust_left_eef_pose_in_head_frame`, `adjust_right_eef_pose_in_head_frame` | Move a hand by offsets in `robot_base` or `head`, selected by `translation_frame`; rotations use local gripper axes. |
| `adjust_left_eef_pose_in_wrist_frame`, `adjust_right_eef_pose_in_wrist_frame` | Move a hand by offsets expressed in that wrist frame. |
| `track_object_distance`, `move_tracked_point` | Track visible points and execute constrained motion. |
| `robot_step` | Send an explicit validated 27-value evaluator command. See joint manual. |
| `activate_skill`, `deactivate_skill` | Read/switch the current task procedure; no motion. |

Use `python tools/read_proprioception.py` when you need to inspect the current
arm/torso configuration or understand a joint movement. Inspect its attached
image; if your host only shows a path, open it with native Read. This renderer
does not depict scene objects, held objects or collision clearance. It cannot
be used for image clicks. Ordinary tools still update measured joint JSON and
RGB-D, but do not render meshes. The saved `perception/proprioception/robot_three_views.png`
is the last requested render; `render.json` records its observation sequence.
After movement, generate another view only if needed, not after every action.

Each call leaves a small record in `tool_call_history/NNNNNN_NAME/`:
`input.json` (exactly what you requested), `result.json` (the concise receipt),
the returned image(s) `returned_image_*` (for example the red gripper
visualization), and, for a point call, `click_IMAGE_crosshair.png` with a
crosshair drawn at every selected pixel. Full outputs, joint commands, metric
depth and the complete trajectory are stored by the harness outside this
workspace; they are not needed for decisions. Rejected JSON and connection
failures are also recorded; they carry no image because no fresh observation
was obtained.
