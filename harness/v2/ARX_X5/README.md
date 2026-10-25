# ARX X5 robot workspace

You control two fixed six-joint X5 arms through ordinary Python in Bash.
Start by reading `robot_config/README.md` and viewing
`robot_config/visuals/three_views.png`. Read a joint's individual triptych
before reasoning about its rotation. All robot implementation is here.

```bash
.venv/bin/python tools/capture_head_camera.py '{}'
.venv/bin/python tools/read_proprioception.py '{}'
.venv/bin/python tools/activate_skill.py '{"name":"pick-up-object"}'
.venv/bin/python tools/TOOL_NAME.py --schema
```

Use this workspace's `.venv/bin/python`. Every public tool has a complete
algorithm in `tools/TOOL_NAME.py`; reused geometry/control functions live in
`tools/util`. The catalogue is `tools/README.md`. You may read the code and
write your own Python using `robot_config.control.Robot` and third-party
libraries in this environment. `Robot.step(action)` accepts the evaluator's
14-value joint command. There is no simulator API available through Robot.

Only the newest observation represents the robot's CURRENT state. Inspect
its actual pixels before every visual decision. After any movement, locate
the target again in the new image; do not shift, extrapolate or recall an
old bounding box, point, imagined object structure or track location.
Historical observations show past states. Track estimates are evidence to
check against the latest RGB, not a replacement for looking.

Clicks use normalized image coordinates: top left `(0,0)`, bottom right
`(1000,1000)`. This is the original RoboDojo tool contract. Images remain at
their original resolution, usually 640×480. Never send a raw pixel value
as a normalized coordinate. Bind every click to its returned `image_id`.
All conversions use that frozen image's width and height exactly once.

Current information:

- `perception/latest.json`: current image IDs, sequence and camera paths.
- `perception/exteroception/head/rgb.png`: latest unmodified head RGB.
- `perception/exteroception/{left_wrist,right_wrist}/`: latest wrist RGB-D.
- Each camera's `depth.npy`: metric depth, and `depth.png`: display only.
- `perception/exteroception/tracks.json`: current track estimates and validity.
- `perception/proprioception/state.json`: measured public robot state.
- `perception/proprioception/three_views.png`: available after an explicit `read_proprioception` call; ordinary observation updates expire this link.
- `tool_call_history/`: each call's input, result, joint commands and click crosshairs.

Claude Code receives saved tool images as actual image content through its
native Bash hooks, without a model bridge or conversation rewrite. When an
image is attached, inspect it directly. If your agent does not support these
hooks, open the returned PNG with its native image reader. Printing a path
or base64 string alone does not show an image to a model.

To inspect the current robot posture, run
`.venv/bin/python tools/read_proprioception.py '{}'` and inspect its returned
front/side/top image (use native Read if only a path is displayed). It reads
public joints and renders the local model without moving, consuming physics
steps or capturing cameras. Ordinary robot tools still update numeric joints,
RGB-D and tracks, but do not render body views. Request a body view only when
posture inspection is useful, such as before unfamiliar joint control; do not
add one after every action. A body diagram shows no scene or held objects and
cannot be clicked. Its recorded episode/sequence identifies the observed pose.
Older requested images remain in `tool_call_history/` as historical evidence.

`skills/robodojo-baseline` always applies. Task skills are loaded on demand;
activate at the start of the relevant manipulation and deactivate when it
ends. Native agents may read the same `SKILL.md` files directly. Only one
task skill is active at a time. Deactivation changes current instructions,
not historical context.

This robot cannot drive, change torso height, or tilt its fixed head camera.
Those compatibility tool names return `unsupported`; never repeat them.
Do not infer grasp success from the gripper command proxy. Inspect wrist
RGB-D, object motion and current tracks. Stop and reassess a failed plan on
fresh pixels before retrying. Never claim the task passed without evidence.

Each session gets a private copy of code, skills, interpreter, libraries,
perception and history. `.runtime` holds private worker state, signed plans,
caches and native agent transcripts; it is not the robot manual.

For operators: rebuild with `bash setup.sh`; create a new workspace with
`python3 session.py --destination /new/session/ARX_X5 --id short-id --task TASK`.
The evaluator operator connects that session before the agent starts. The
agent itself needs no original RoboDojo or R1 Pro implementation imports.
