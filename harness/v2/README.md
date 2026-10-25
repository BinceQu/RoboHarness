# RoboHarness V2: workspace harness

V2 gives a coding agent a complete robot workspace instead of a tool server.
The agent starts inside a directory that contains the robot model, joint
manuals, readable Python tools, skills, current perception and its own call
history. Every tool ultimately sends the evaluator's atomic joint/controller
commands, which the agent can also send directly.

The simulator, task definitions, prompts, budgets and scoring are the ones the
V1 runner uses (`../../run.sh`); only the agent side differs. Running the
repository README's commands still uses V1 and reproduces the paper results.

```text
harness/v2/
├── R1_pro/               # R1 Pro workspace for BEHAVIOR (27-value controller command)
├── ARX_X5/               # dual ARX X5 workspace for RoboDojo (14 joint scalars)
├── behavior_evaluator/   # runs archived BEHAVIOR cases with fresh R1_pro copies
└── robodojo_evaluator/   # connects ARX_X5 copies to a RoboDojo evaluator
```

Each workspace is self-contained: `tools/NAME.py` holds a tool's algorithm,
`tools/util/` the shared code, `robot_config/` the URDF, meshes and joint
manuals, `skills/` the task skills, and `AGENTS.md` / `CLAUDE.md` the operating
instructions. A workspace never imports code from outside its directory.
Agents operate with ordinary shell and file access, for example
`python tools/capture_head_camera.py '{}'`, so any coding agent can use it;
`tools/util/launch.py` starts Claude Code with the workspace's image hooks.

## Setup

Install [uv](https://docs.astral.sh/uv/), then build each workspace's own
Python 3.10 environment. `setup.sh` first fetches the meshes, geometry data
and wheels that are hosted outside git (see `../../assets.lock.json`):

```bash
bash harness/v2/R1_pro/setup.sh     # BEHAVIOR; PyTorch 2.6 with CUDA 12.4
bash harness/v2/ARX_X5/setup.sh     # RoboDojo; CPU only
```

The BEHAVIOR simulator, evaluator environment and datasets are the V1 ones;
follow the repository README and [setup guide](../../docs/setup.md) first.

## BEHAVIOR

Copy [`behavior_evaluator/config.example.json`](behavior_evaluator/config.example.json),
fill in the evaluator paths from your V1 configuration and set `agent_python`
to `harness/v2/R1_pro/.venv/bin/python`. Then, from the repository root:

```bash
harness/v2/R1_pro/.venv/bin/python harness/v2/behavior_evaluator/run.py \
  --task task01 --gpu 0 --config my-v2-config.json --dry-run
harness/v2/R1_pro/.venv/bin/python harness/v2/behavior_evaluator/run.py \
  --task task01 --gpu 0 --config my-v2-config.json
```

Each case gets a fresh copy of `R1_pro` under the run directory. The evaluator
connects to a local atomic action driver; a separate read-only monitor serves
status. `--smoke` runs an unscored physical check on the first instance, and
`--instances` selects a diagnostic subset. Sequential instances use the
evaluator's opt-in restoration of task objects that a previous instance's
transition rules deleted. Browse finished runs with the read-only
[trajectory viewer](behavior_evaluator/trajectory_viewer/README.md).

## RoboDojo

`robodojo_evaluator` drives a private copy of `ARX_X5` with an existing
RoboDojo checkout; set `ROBODOJO_ROOT` and `ROBODOJO_EVALUATOR_ENV` or pass
`--robodojo` / `--evaluator-env`. See its [README](robodojo_evaluator/README.md).

## Tests

```bash
cd harness/v2
python3 ../../scripts/fetch_assets.py --group v2-tests   # RoboDojo fixtures
R1_pro/.venv/bin/python -m pytest R1_pro/.dev/tests behavior_evaluator -q
ARX_X5/.venv/bin/python -m pytest robodojo_evaluator/tests robodojo_evaluator/trajectory_viewer -q
```

`behavior_evaluator/test_evaluator_lifecycle.py` also checks that the default
(V1) evaluator path never resets or rebinds task objects.
