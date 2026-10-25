# Operating this robot

Read `skills/behavior-v2-baseline/SKILL.md`, `tools/README.md`, and
`robot_config/joints.md` before the first action. Before first using a tool,
read its Python file's physical overview and input schema; inspect the detailed
action sequence when needed. Open a joint's
illustration before commanding an unfamiliar joint. This workspace contains
ordinary files; no plugin installation is needed to read and follow a skill.

Use this workspace's `.venv/bin/python` (the launcher activates it in PATH).
Operate using `python tools/NAME.py '{"argument":"value"}'` in Bash.
Print physical-tool inputs with `python tools/NAME.py --schema`; skill tools
document their `name`/`reason` arguments in their files. Calls are sequential:
wait for the result, inspect it and the new image, then decide the next action.
Each tool's full algorithm and private helpers are in that file. Shared code
is in `tools/util/`; it is also available for inspection. The readable Python
tool executes here; its yielded 27-value commands go
directly to the evaluator. There is no intermediate macro-tool HTTP server.

The latest observation is the current physical state. Older images, tracks,
coordinates and descriptions are history. After movement, locate the object
again in the actual newest image. Do not translate an old mental picture.
Every point uses the current head image's native 720 × 720 pixels: integer
column `u`, row `v`; origin top-left. Bind it to the exact current `image_id`.
Wrist images are for inspection only. The runner verifies dimensions and
frame identity on every point call. Read `perception/latest.json` if unsure.

Returned images are actual observations only when your host displays their
pixels. Claude Code's provided hook attaches new image content to Bash tool
results; other hosts must open the returned path with their image reader.
If an image cannot be displayed, do not infer its contents from its filename.
Saving a crop with Python does not display it: use native Read to inspect that
new file. For background tool calls, wait for completion and the attached image.
When you need to understand the current arm/torso posture, run
`python tools/read_proprioception.py` and inspect its returned three-view image
(or open the returned path with native Read). It reads measured joints and
renders the robot alone, without moving or consuming physics ticks. Scene and
held objects are not shown; never use this render for scene clicks.
Ordinary tools update `perception/proprioception/joint_state.json` but do not
render posture images. `robot_three_views.png` is only the last requested
render: its sequence is in `perception/proprioception/render.json`. After
motion, request another render only if posture inspection is needed; do not
assume the saved PNG automatically refreshed or render after every action.

The baseline skill is always active. Activate one task skill using
`python tools/activate_skill.py '{"name":"SKILL_NAME"}'`, read its complete
returned instructions, and follow its exit conditions. Deactivate with
`python tools/deactivate_skill.py '{"name":"SKILL_NAME"}'`. This changes
instructions, not physics. `skills/active.json` is authoritative after resume
or context compaction. The parent task determines task order and completion.
Only four skills are supplied: `behavior-v2-baseline`, `pick-up-object`,
`place-object-in-container`, and `open-doors-and-drawers`. Read the baseline
first and load a task skill only when needed; do not import other skills.

For direct control, read `robot_config/control.py` and the joint manual, then
invoke it in Bash. Revolute joint inputs are absolute radians; finger inputs
are forces in newtons; base inputs are velocities. Direct control has no
collision planner. Slots marked locked must remain zero. Preserve a carrying
gripper's force unless intentionally releasing it. Direct calls are recorded
and refresh perception just like named tools.

Tool returns report concrete evidence, without a generic success flag. For
base motion, compare requested and actual displacement, `linear_target_reached`
and `obstacle_limited`; a settled stop or absence of an error does not prove
arrival. For other tools, use their measured progress, completion/contact
fields, error reasons and returned images to determine what actually happened.
Tool errors do not imply that no motion occurred. Inspect the result's sent
command count and current perception before recovery. Do not alter the task,
reset an episode, inspect evaluator answers, or manage simulator processes
unless the parent instruction explicitly requests that operation.
