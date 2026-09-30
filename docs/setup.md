# Installation

Use Linux x86-64 with an NVIDIA RTX GPU, a compatible driver, Python 3.11 and
the CUDA 12.4 toolkit for the interface's cuRobo build. The validation host has
48 GB VRAM per GPU. A run also needs sufficient RAM and local scratch space
for Isaac Sim and decrypted scene assets. Leave at least 30 GB free scratch.
Each simulator run creates a private cache (about 7–8 GB on the validation
host); completed and failed attempts retain theirs. The launcher checks for
at least 10 GiB free at `cache_dir` before starting. Concurrent runs require
additional headroom. Remove only caches belonging to stopped runs, or select
a larger scratch filesystem in `configs/local.json`.

Clone this repository with its submodule, then install:

```bash
git submodule update --init --recursive
./scripts/setup.sh
```

The installer builds the pinned cuRobo source with the interface environment's
build tools. Its example meshes and videos are skipped; RoboHarness supplies
the custom robot's URDF and collision configuration.

The pinned upstream installer asks for its license agreements. Its flags can
be supplied after `all`, for example `./scripts/setup.sh all --accept-nvidia-eula`.
Dataset installation is separate: follow the pinned
[BEHAVIOR installation guide](../BEHAVIOR/docs/getting_started/installation.md).
The downloaded data root must contain:

```
data/
  behavior-1k-assets/                 # version 3.9.0
  omnigibson-robot-assets/            # version 3.8.2
  2026-challenge-task-instances/
  omnigibson.key
```

Existing directories can be linked under `data/`. The runner builds an
immutable sibling overlay containing the custom robot; it does not replace
the robot in the shared dataset. Dataset files, the decryption key and local
configuration are ignored by Git. Even v3.9.1 uses the `2026-challenge-task-instances`
directory; the archived rollout budgets remain the explicit 2025 ×2 values.

The original interface uses SAM 2.1 small for held-object masks when available.
The installer fetches the pinned [SAM 2 source](https://github.com/facebookresearch/sam2)
under `data/sam2` and `sam2.1_hiera_small.pt` under `data/`, then verifies the
source and checkpoint hashes in `configs/perception.json`. These external
files are ignored by Git. Existing copies can be verified without a download:
`python3 scripts/setup_perception.py --check`. The interface requirements
include its Python dependencies. You can override
`EEF_NEAR_SAM2_REPO`, `EEF_NEAR_SAM2_DEPS`, `EEF_NEAR_SAM2_CKPT` and
`EEF_NEAR_SAM2_DEVICE` in the environment. Without the checkpoint the interface
uses its existing 3D growth fallback; this is a different perception configuration.

Install Claude Code **2.1.259** separately and make `claude` available on PATH.
The reproduction launcher checks this version before starting the simulator. The
archived runs use the Claude Code harness with `Qwen3.8-Flash-Next-FP8` through
an Anthropic-compatible `/v1/messages` server. The `model_url` is the origin,
without `/v1`. Authentication is supplied through `ANTHROPIC_API_KEY` or
`ANTHROPIC_AUTH_TOKEN` in the environment; keys are never committed.

```bash
cp configs/example.json configs/local.json
# Edit data_path, Python executables and model_url for this machine.
./run.sh --list
./run.sh --task task01 --gpu 0
```

Relative configuration paths are resolved from the repository root. Reusing
existing environments is supported by `interface_python`, `evaluator_python`,
and `agent_python`; the runner always imports the interface and evaluator
source from this checkout. The reference runtime is Python 3.10 / Torch
2.6.0+cu124 / Warp 1.12.1 for the interface, Python 3.11 / Torch 2.7.0+cu128 /
Isaac Sim 5.1.0 / Warp 1.12.0 for the evaluator, and MCP 2.1.1 for the agent.
The evaluator also verifies the exact OmniClient binary before applying a
process-local asset-watch workaround. An unknown binary fails startup. See
[native asset reload failure](native-asset-reload.md) for the required hash,
the internal-API limitation and native regression evidence.

`unmask_evaluator_cuda: true` is a host compatibility option for multi-GPU
Vulkan installations that cannot initialize Kit with a CUDA mask. Rendering
and simulation still select `--gpu`; the interface remains masked. In this
mode Kit can create small CUDA contexts on other visible cards. The default
is a single visible GPU. Each run records the selected mode.

For Codex, install a compatible Codex CLI with plugin, hook and direct MCP
support. Use `--harness codex --model YOUR_MODEL --model-url YOUR_RESPONSES_URL`
and provide `OPENAI_API_KEY`. The launcher builds a private `CODEX_HOME` and
retains the original embodied tool restrictions. It does not read another
user's relay configuration. This is an alternate harness: the supplied
reference scores are Claude Code / Qwen runs. The launcher installs the local
plugin through the CLI into that run's private home and verifies its source
and version. See the
[Codex plugin documentation](https://developers.openai.com/plugins/build/plugins).

Run CPU checks with `./scripts/check.sh`. Existing interpreters can be selected
using `ROBOHARNESS_INTERFACE_PYTHON` and `ROBOHARNESS_AGENT_PYTHON`.

The archived budget is a simulation-step budget. The default
`session_timeout_s: 0` therefore disables any additional per-case wall-clock
cap. An operator may set a positive number of seconds in the local configuration
to opt into a safety timeout. If that timeout forces submission, the official
result is retained but the strict reporter excludes the run from reproduction
verification, even when its score matches. The selected timeout is recorded
in each new run's runtime configuration.
