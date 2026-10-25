# R1 Pro robot workspace

This directory is your robot's manual, controls, current senses, and memory.
Start with `AGENTS.md`. The evaluator is the source of physical observations;
Python tools compute actions here and send joint commands directly to it.

| Directory | What to read |
| --- | --- |
| `robot_config/` | Controller units, every joint's axis and limits, one illustrated pose triptych per joint, and direct joint control. |
| `tools/` | One Python file per task01 tool. Read a file to understand its inputs and action sequence; run it with Bash. |
| `skills/` | Reusable task procedures. `active.json` identifies the current task skill. |
| `perception/proprioception/` | Latest measured joints; posture three-views are rendered only when you call `tools/read_proprioception.py`. No scene is used. |
| `perception/exteroception/` | Latest head/wrist RGB, head overlay, 0–5 m depth previews and tracks (real files, current observation only). |
| `tool_call_history/` | One small folder per call: `input.json`, `result.json`, returned images (e.g. red gripper visualization) and, for point calls, a crosshair image. Bulk evidence (metric depth, joint commands, full output) is kept by the harness outside the workspace. |

`perception/latest.json` binds camera, depth, track and joint files to one observation sequence.
To inspect the current robot posture, run `python tools/read_proprioception.py`
and view its returned image. This read-only tool consumes no physics ticks.
`perception/proprioception/render.json` identifies the last requested render's
sequence and measured joints. Its PNG is not refreshed by ordinary tools;
after movement, request a new render only when posture inspection is needed.
Older views remain in `tool_call_history/`; they are historical, not current.
Latest-file links point to history inside this workspace. No writable file is
shared with another session. A fresh session starts from a clean template.

```bash
python tools/capture_head_camera.py
python tools/read_proprioception.py
python tools/adjust_chassis.py --schema
python tools/adjust_chassis.py '{"forward":0.10}'
python tools/activate_skill.py '{"name":"pick-up-object"}'
```

A JSON path alone does not show you an image. Use your host's native image
reader. In Claude Code, the provided native hooks also attach new tool images
directly to Bash results; context management remains Claude Code's own.
The native Stop hook gives one completion self-check after a new workspace
tool call. It asks you to execute a pending step or explain the observed stop
condition. It does not judge the task, command the robot, or prevent a second
stop when no new workspace call has occurred. Its source is
`tools/util/stop_check.py`.

The complete implementation is here. Each tool's algorithm and private helpers
live in `tools/NAME.py`; shared transport, geometry, rendering and persistence
are readable in `tools/util/`. Robot meshes and model resources are in
`robot_config/`. No external RoboHarness package or source checkout is imported.
Image compression uses Pillow from `.venv/`; no host image-converter executable
is required. Native 720 × 720 head images retain their dimensions and orientation.

Use this directory's Python: `source .venv/bin/activate` (or invoke
`.venv/bin/python` directly). `.venv/` contains third-party libraries and
`.python/` contains the interpreter. To recreate the environment, run
`bash setup.sh`; this also refreshes environment paths after a manual copy or
rename. `uv`, a CUDA-capable Linux host and Mesa/EGL are prerequisites.
The included cuRobo wheel targets Linux x86_64 / Python 3.10 / PyTorch 2.6 CUDA
12.4. `requirements.lock` fixes the remaining package versions.

Every live session has its own `.runtime/` for controller records, plans,
compilation files and native Claude transcripts. The current observation is in
`perception/`; human-readable call records remain in `tool_call_history/`.
You normally do not need to read the hidden environment/runtime directories.

For an already running evaluator connected to `serve_robot.py`, create a
fresh complete copy with:

```bash
.venv/bin/python session.py --workspace /absolute/path/to/new-session \
  --socket /absolute/path/to/control.sock --session-id r1-001 \
  --task-name picking_up_trash
```

The copy owns its code, assets, interpreter, libraries, skill state and history;
it does not import the template's implementation. New sessions cannot reuse an
existing directory. The socket connects to the evaluator boundary; its protocol
is documented in `robot_config/transport.md`.
