<div align="center">

<img src="docs/assets/roboharness-wordmark.png" alt="RoboHarness" width="500">

## A Simple Harness Can Outperform VLA and World Action Models

**[Project Page](https://bincequ.github.io/RoboHarness/) · [Quick Start](#run-a-task) · [Documentation](docs/setup.md) · [中文](README.zh-CN.md)**

[![License: MIT](https://img.shields.io/badge/License-MIT-263c78?style=flat-square)](LICENSE) [![Benchmark](https://img.shields.io/badge/BEHAVIOR-Challenge_2025-d56045?style=flat-square)](#reproduction-contract)

[![RoboHarness overview](docs/assets/overview.png)](docs/assets/overview.pdf)

<sub>RoboHarness overview: visual keypoints are tracked persistently, converted into geometric constraints, executed, and verified in a closed loop. [View original PDF](docs/assets/overview.pdf)</sub>

</div>

## Overview

Without training models, RoboHarness combines visual clicks, keypoint tracking,
and geometric constraints to provide an embodied control panel for an LLM agent.

The LLM marks points of interest on a 2D image and continuously receives their
positions via optical-flow tracking and depth back-projection; with this
information, it composes accurate primitive actions to perform complex
manipulation.

The reported BEHAVIOR results use Claude Code 2.1.259 with
`Qwen3.8-Flash-Next-FP8` and the R1 Pro robot. A Codex harness is also available.

## What is included

| Component | Purpose |
| --- | --- |
| [`BEHAVIOR/`](BEHAVIOR) | BEHAVIOR simulator and evaluator, pinned to v3.9.1 |
| [`interface/`](interface) | RGB-D observations, robot control and interactive interface |
| [`harness/claude_code/`](harness/claude_code) | Claude Code harness |
| [`harness/codex/`](harness/codex) | Codex harness |
| [`prompt/`](prompt) | Task-specific instructions for household manipulation |
| [`tasks/`](tasks) | Task definitions, evaluation instances and step budgets |
| [`reference_results/`](reference_results) | Reference evaluator scores |
| [`roboharness/`](roboharness) | Task runner and evaluation pipeline |
| [`validation_results/`](validation_results) | Evaluation reports and scores |

## Installation

Use Linux x86-64, an NVIDIA RTX GPU, Python 3.11 and the CUDA 12.4 toolkit.
The validation host has 48 GB VRAM per GPU; this is the tested capacity, not a
measured minimum. You also need BEHAVIOR scene data, local cache space and a
separately configured model endpoint.

```bash
git clone --recurse-submodules https://github.com/BinceQu/RoboHarness.git
cd RoboHarness
./scripts/setup.sh
cp configs/example.json configs/local.json
```

Follow the [installation guide](docs/setup.md) to obtain the licensed datasets,
install Claude Code 2.1.259, and configure the model endpoint and authentication.
Edit `configs/local.json` for this machine's data and Python paths.

## Run a task

After installation and model configuration, one command runs all five evaluation
instances of a task:

```bash
./run.sh --list
./scripts/reproduce_task.sh task01 --gpu 0
```

The wrapper creates a session configuration at `.local/session-config.json`
from `configs/local.json` on first use, then reuses it. Edit that session file
for later launches. It does not change global Claude or Codex configuration.

```bash
# Inspect the resolved plan without launching the simulator.
./scripts/reproduce_task.sh task03 --gpu 0 --dry-run

# Run selected instances.
./scripts/reproduce_task.sh task08 --gpu 0 --instances 301,304

# Use the alternative harness with a Responses-compatible model endpoint.
./scripts/reproduce_task.sh task01 --gpu 0 --harness codex \
  --model YOUR_MODEL --model-url YOUR_RESPONSES_URL
```

Codex requires its CLI and `OPENAI_API_KEY`; see [model setup](docs/setup.md).
Use `./run.sh --list` to see the supported tasks; task04 is not included.

The runner starts the interface, idle gate, official evaluator and agent, then
loads the task configuration. It saves the run plan, prompt,
native agent transcript, trajectory, official scoring JSON and summary under a
fresh `runs/<run-id>/` directory. Ctrl-C cleans up only that run's owned processes.

The interface is available at `http://127.0.0.1:<port>/`. HTTP defaults to
`15070 + task index`, with policy and idle-gate ports at HTTP +1000 and +2000.
Use the session file's `task_ports` mapping to choose all three listeners. The
[1507* example](docs/setup.md#session-configuration-and-ports) assigns distinct
ports to task01, task03 and task08. Use SSH forwarding for a remote host.

## Reproduction contract

The paper's BEHAVIOR evaluation protocol is:

- **Task-specific prompts, shared across instances.** The prompt is specific
  to each task but shared across instances, and is written by a human. It
  specifies the execution steps and the behavioral boundaries the model must
  respect. See [task prompts](docs/case-prompts.md).
- **Five instances per task.** The sample is fixed by seed `20260911`. Every
  task uses evaluation slots **0, 3, 5, 7, 9**, corresponding to instance IDs
  **301, 304, 306, 308, 310**. Each instance is one official rollout.
- **Challenge 2025, multiplier 2.** The maximum tick count is set separately
  for each task to twice the mean length of its human demonstrations, counted
  in simulator control steps. An episode ends when the goal is satisfied or
  this limit is reached. See the [step budgets](docs/provenance.md#evaluation-budgets).
- **Mean Q-score.** The official BEHAVIOR Challenge 2025 evaluator provides
  `q_score.final` for each rollout. The task mean is the unweighted average of
  all five scores, reported to four decimal places.

The runner defaults to `session_timeout_s: 0`, so there is no additional
wall-clock limit. Execution time varies with hardware, model serving and
concurrent load; evaluation budgets are measured in simulation control steps.

Collect and check runs with:

```bash
python3 scripts/report_validation.py runs/YOUR_RUN_A runs/YOUR_RUN_B \
  --output validation_results/latest --check-live --require-match
```

The report compares each complete task mean with its reference value using a
tolerance of `1e-6`. Individual instance scores may differ. `--require-match`
exits nonzero if a task is incomplete, its mean differs, or the run fails the
evaluation checks. Use `--check-live` on the evaluation host; omit it when
analyzing copied runs. Add `--watch` for continuous reporting.

## Generalization across objects

[![The 100 objects used to evaluate generalization across objects](docs/assets/object-generalization.png)](docs/assets/object-generalization.pdf)

The 100 objects used to evaluate generalization across objects. The catalog is
drawn from the BEHAVIOR object library.

[![Successes out of 100 objects versus estimated hours of robot data](docs/assets/robot-data-comparison.png)](docs/assets/robot-data-comparison.pdf)

Successes out of 100 objects versus estimated hours of robot data. Colors: VLA
(blue), world action model (green), RoboHarness (red), and the ASPIRE-pick
baseline (gray).

## Development and license

**Contributors:** [BinceQu](https://github.com/BinceQu) and **Codex** (AI coding agent).

Run the installed environments' CPU checks with `./scripts/check.sh`.
See [contribution guidelines](CONTRIBUTING.md) for focused checks and the
information needed in a reproducibility issue.

Repository code is released under the [MIT license](LICENSE), subject to the
component-specific [third-party notices](THIRD_PARTY_NOTICES.md). BEHAVIOR data,
NVIDIA Isaac Sim and model weights have their own access and license terms.
