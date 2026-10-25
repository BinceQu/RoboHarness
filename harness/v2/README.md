# RoboHarness V2: workspace harness

V2 gives a coding agent a complete robot workspace instead of a tool server.
The agent starts inside a directory that contains the robot model, joint
manuals, readable Python tools, skills, current perception and its own call
history. Every tool ultimately sends the evaluator's atomic joint/controller
commands, which the agent can also send directly.

```text
harness/v2/
├── R1_pro/   # R1 Pro workspace for BEHAVIOR (27-value controller command)
└── ARX_X5/   # dual ARX X5 workspace for RoboDojo (14 joint scalars)
```

Each workspace is self-contained: `tools/NAME.py` holds a tool's algorithm,
`tools/util/` the shared code, `robot_config/` the URDF, meshes and joint
manuals, `skills/` the task skills, and `AGENTS.md` / `CLAUDE.md` the operating
instructions. A workspace never imports code from outside its directory.
Agents operate with ordinary shell and file access, for example
`python tools/capture_head_camera.py '{}'`, so any coding agent can use it;
`tools/util/launch.py` starts Claude Code with the workspace's image hooks.

The evaluation side is shared with V1 and lives in [`eval/`](../../eval):
the runner ([`eval/runner`](../../eval/runner)), the BEHAVIOR evaluator wrapper
([`eval/behavior`](../../eval/behavior)), the RoboDojo bridge
([`eval/robodojo`](../../eval/robodojo/README.md)) and read-only trajectory
viewers ([`eval/viewer`](../../eval/viewer)).

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

V2 uses the same runner, configuration, tasks, prompts, budgets, evaluator and
scoring as V1; only `--harness v2` selects the workspace agent side:

```bash
./run.sh --harness v2 --task task01 --gpu 0 --dry-run
./run.sh --harness v2 --task task01 --gpu 0
```

The runner re-executes itself under `harness/v2/R1_pro/.venv/bin/python`
(override with `workspace_python` in the configuration; see
[`configs/v2.example.json`](../../configs/v2.example.json)). Each case gets a
fresh copy of `R1_pro` under the run directory. The evaluator connects to a
local atomic action driver; a separate read-only monitor serves status.
`--smoke` runs an unscored physical check on the first instance, and
`--instances` selects a diagnostic subset. Sequential instances use the
evaluator's opt-in restoration of task objects that a previous instance's
transition rules deleted; V1 runs never enable it. Browse finished runs with
the read-only [trajectory viewer](../../eval/viewer/behavior/README.md).

## RoboDojo

[`eval/robodojo`](../../eval/robodojo/README.md) drives a private copy of
`ARX_X5` with an existing RoboDojo checkout; set `ROBODOJO_ROOT` and
`ROBODOJO_EVALUATOR_ENV` or pass `--robodojo` / `--evaluator-env`.

## Tests

```bash
python3 scripts/fetch_assets.py --group v2-tests   # RoboDojo fixtures
cd harness/v2
R1_pro/.venv/bin/python -m pytest R1_pro/.dev/tests -q
cd ../../eval/robodojo
../../harness/v2/ARX_X5/.venv/bin/python -m pytest tests ../viewer/robodojo -q
```

The evaluator change V2 relies on is covered by
`eval/behavior/behavior_interface_eval_test/test_evaluator_lifecycle.py`, which
also checks that the default (V1) path never resets or rebinds task objects.
