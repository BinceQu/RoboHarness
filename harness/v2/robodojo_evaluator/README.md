# RoboDojo evaluator for ARX X5 workspaces

This component connects a private copy of `../ARX_X5` to the official RoboDojo
joint-action evaluation loop. The original RoboDojo checkout supplies physics,
scenes, scoring and the WebSocket policy protocol. It does not supply any robot
tool implementation to the agent workspace. The R1 workspace is unchanged.

For batches, `fleet.py` supervises a set of sessions, each with its own GPU,
policy port and task. It checks workspace heartbeats, preserves every attempt
and retries finished or failed slots using new workspaces. Restarting the fleet
supervisor retains live evaluators. Do not manually duplicate an assigned port.

For a separate development check, start one session on an assigned free GPU
and policy port (choose a free port and capacity first):

```bash
cd harness/v2   # from the repository root
ARX_X5/.venv/bin/python robodojo_evaluator/start.py \
  --gpu 1 --port 19082 --task general_pickup --seed 0
```

The command prints the timestamped run directory, private workspace, PIDs and
exact commands. `--robodojo` and `--evaluator-env` select the installed evaluator
and its environment. These operator dependencies are separate from the self-
contained robot workspace. Do not launch a duplicate on an occupied port.

In the printed workspace, check readiness and call a tool:

```bash
.venv/bin/python -c 'from robot_config.control import Robot; print(Robot().status())'
.venv/bin/python tools/capture_head_camera.py '{}'
```

Wait for `evaluator.connected=true`; policy-port availability alone does not
mean the scene has loaded. An idle policy returns no action and consumes no
evaluator steps. All capture/planning calls retain the original zero-step
semantics. `robot_config/control.py` sends one exact joint action and waits
for its corresponding new observation.

Launch native Claude Code in that workspace:

```bash
export EMBODIED_ANTHROPIC_BASE_URL=http://YOUR_MODEL_HOST:31000
.venv/bin/python -m tools.util.launch -- \
  -p --output-format stream-json --verbose < /path/to/task-prompt.txt
```

The default model is `Qwen3.8-Flash-Next-MLX-oQ6-MTP`; `--model` or `QWEN_MODEL`
can select an available model on the same direct Messages endpoint. There is
no model bridge. A new agent session requires a new workspace; native
`--resume` can continue the exact same session. Claude's native transcript is
inside `ARX_X5/.runtime/claude-home/projects/`. Images are original PNG bytes
attached through native Bash hooks, also available to any agent via Read.

Each robot tool keeps its full implementation in `ARX_X5/tools/NAME.py`, with
shared functions in `tools/util`. There are 26 public robot tool names,
one additional atomic `joint_command`, and skill activation/deactivation.
Fixed-base compatibility names retain explicit unsupported results rather
than inventing mobile-base motion. The four skills load on demand; only the
robot baseline is included at session start.

`assets.py` creates a per-run asset-path overlay because the upstream X5
CuRobo YAML contains an installation-specific absolute URDF path. Only those
paths change. No friction, geometry, task conditions, budgets or scores are
modified. `path_shim` runs exclusively in the evaluator process.

Run regression tests with the workspace environment:

```bash
cd robodojo_evaluator
../ARX_X5/.venv/bin/python -m pytest tests -q --tb=short
```

The upstream algorithm and public-tool tests are retained under `tests/upstream`;
new tests exercise atomic commands, image-byte delivery, perception, history,
session ownership and skill state. Synthetic tests are not task success
scores. Actual run evidence belongs in `runs/`; only the official completed
evaluator output establishes an episode's score.

The [trajectory viewer](trajectory_viewer/README.md) displays the native model
conversation, including Thinking, every exact Bash/Read call, and image
returns in four aligned columns. Run it with:

```bash
python3 robodojo_evaluator/trajectory_viewer/server.py --port 18794
```

It is a read-only localhost monitor. It does not sit between the robot worker
and evaluator and cannot send commands.
