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

The model server and the evaluation runner are separate processes. Keep your
model server running throughout evaluation. `setup.sh` installs the simulation
and agent environments; it does not download LLM weights or start a model server.

**1. Connect to your model service.**

The default Claude Code harness uses an **Anthropic-compatible `/v1/messages`
endpoint with image input, tool calls and streaming**. The paper configuration
uses `Qwen3.8-Flash-Next-FP8`. If your server only provides OpenAI Chat
Completions, follow the [bridge setup](docs/setup.md#chat-completions-services)
first. Codex uses a different [Responses configuration](docs/setup.md#codex-model-service).

On the **evaluation machine**, open a terminal in this repository and replace
the host, served model ID and key below with your service's values:

```bash
export ROBOHARNESS_MODEL_URL='http://YOUR_MODEL_HOST:31000'
export ROBOHARNESS_MODEL='Qwen3.8-Flash-Next-FP8'
export ANTHROPIC_API_KEY='YOUR_MODEL_SERVER_KEY'
export ANTHROPIC_AUTH_TOKEN="$ANTHROPIC_API_KEY"
```

Use the origin **without `/v1`** for `ROBOHARNESS_MODEL_URL`. The model ID must
match the name exposed by your server. For a server with authentication disabled,
use `local-no-auth` as the key. `127.0.0.1` works only when the model service is
on this evaluation machine, or an SSH tunnel forwards it here.

Send a small request before starting the simulator:

```bash
python3 - <<'PY' | curl --fail-with-body --silent --show-error \
  "${ROBOHARNESS_MODEL_URL%/}/v1/messages" \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" \
  --data-binary @-
import json, os
print(json.dumps({
    "model": os.environ["ROBOHARNESS_MODEL"], "max_tokens": 128,
    "messages": [{"role": "user", "content": "Reply with OK."}]
}))
PY
```

Expect a Messages response with `"type": "message"`. This checks reachability,
authentication and the model ID; the service must also support the image and
tool features above. Connection errors, HTTP 401/403 and HTTP 404 are covered in
[model connection troubleshooting](docs/setup.md#model-connection-troubleshooting).

**2. Save the session configuration.**

In the same terminal, copy your installation settings on first use and save
the model address and name. Existing session settings are retained:

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ.get("ROBOHARNESS_SESSION_CONFIG", ".local/session-config.json"))
source = path if path.exists() else Path("configs/local.json")
config = json.loads(source.read_text())
config.update(model_url=os.environ["ROBOHARNESS_MODEL_URL"],
              model=os.environ["ROBOHARNESS_MODEL"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, indent=2) + "\n")
print("Saved", path)
PY
```

The default session file is `.local/session-config.json`; set
`ROBOHARNESS_SESSION_CONFIG` to select another file. Review it: `data_path` must point to the
[BEHAVIOR data root](docs/setup.md), and `interface_python`, `evaluator_python`
and `agent_python` must point to the installed environments. The example paths
work with `setup.sh`. Set `cache_dir` to writable scratch space outside any
Git checkout when needed. Relative paths are resolved from the repository root.
Credentials stay in the shell environment; export them again in each new
terminal. Later configuration edits belong in this session file.

**3. Inspect the plan, then launch.**

```bash
claude --version  # must be 2.1.259 for the paper configuration
./run.sh --list
./scripts/reproduce_task.sh task01 --gpu 0 --dry-run
./scripts/reproduce_task.sh task01 --gpu 0 --run-dir runs/task01-001
```

In the dry-run output, check `model_url`, `model`, `gpu`, the three ports and
the five instance IDs. Dry-run prints configuration; it does not contact the
model or validate the simulator installation. The final command starts the
interface, evaluator and Claude Code agent and runs all five instances in order.
`--gpu 0` selects the **simulation GPU**; model-serving GPUs are selected when
you start your own model server. Choose resources with enough capacity for both.

Change `task01` to another listed task; task04 is not included. Use a new output
directory for every run, or omit `--run-dir` to generate one automatically.
For a single-instance trial, add `--instances 301` and use a separate directory;
the full task mean requires all five instances.

**4. Watch the rollout and read the results.**

With default ports, task01's interface is `http://127.0.0.1:15071` on the
evaluation machine. For a remote machine, run this on your **own computer**, then
open that address in your browser:

```bash
ssh -N -L 15071:127.0.0.1:15071 YOUR_USER@YOUR_EVALUATION_HOST
```

The launcher prints the HTTP, policy and gate ports. Configure them with
[`task_ports`](docs/setup.md#session-configuration-and-ports) if needed.
For the command above, outputs are in:

| Path | Contents |
| --- | --- |
| `runs/task01-001/summary.json` | Completed instance count, scores and current mean Q-score |
| `runs/task01-001/output/json/` | Official per-instance evaluator results |
| `runs/task01-001/instance_301/` | Prompt, agent output, error log and trajectory for instance 301 |
| `runs/task01-001/run.log` and `runs/task01-001/logs/` | Controller, interface and evaluator logs |

After all five instances finish, generate the comparison report:

```bash
python3 scripts/report_validation.py runs/task01-001 \
  --output .local/reports/task01-001 --check-live --require-match
```

Open `.local/reports/task01-001/README.md` for the result. `--require-match`
returns a nonzero exit code for an incomplete or mismatched evaluation.
Both `runs/` and `.local/` are ignored by Git.

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
