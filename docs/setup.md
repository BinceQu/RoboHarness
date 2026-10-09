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

For archived Claude runs, `cache_dir` must also be outside Git checkouts and
worktrees (including their ancestors). Each case launches from a fresh private
workspace there. Otherwise the native CLI adds the release branch, dirty files
and recent commits to the model's system context. The launcher rejects such a
cache path before starting the simulator; no global Git or CLI settings change.

Clone this repository with its submodule, then install:

```bash
git clone --recurse-submodules https://github.com/BinceQu/RoboHarness.git
cd RoboHarness
./scripts/setup.sh
```

For an existing checkout, initialize the pinned source with
`git submodule update --init --recursive` before setup.

The installer builds the pinned cuRobo source with the interface environment's
build tools. Its example meshes and videos are skipped; RoboHarness supplies
the custom robot's URDF and collision configuration.

The interface pins `opencv-python-headless==4.10.0.84`. Its imported OpenCV
binary matches the reference interface environment. Multiple OpenCV wheel
variants share the `cv2` namespace, so installed package metadata alone does
not identify the binary Python will load. Use a separate interface environment
with only this variant; see the [runtime dependency audit](../validation_results/opencv-runtime-20261001/README.md).

Evaluator setup fixes LeRobot to commit
`436812bd8ee39b768c645c248c91f1330834e687`, recorded in the reference environment
and [source manifest](../configs/evaluator-sources.json). Upstream requests the
moving `release/b1k` branch. The installer serves that request from a private
Git snapshot under `.local/setup-sources` and verifies the installed commit.
The Git URL mapping exists only in the installer child process; it writes no
global Git configuration and does not modify the BEHAVIOR submodule. Existing
evaluator environments can be checked with their Python interpreter:
`python scripts/with_evaluator_sources.py --check`.

After upstream setup, the installer applies
[evaluator runtime pins](../requirements/evaluator-runtime.txt) for Warp,
Pillow, PyArrow, websockets and three support libraries. These pins correct
version drift observed in an independent installation and match the effective
reference evaluator. They are applied with `--no-deps` after upstream has
installed the dependency set; they are not a complete environment lockfile.
The SDK declares exact versions of several shared libraries that conflict
with the evaluation dependencies, including Pillow and packaging. `pip check`
therefore does not pass for this combined environment; the validation record
distinguishes these declarations from the imports and protocol behavior
actually checked. The final versions follow the working reference evaluator.
See the [runtime pin audit](../validation_results/evaluator-runtime-pins-20261001/README.md)
for the exact remaining declarations and the checks performed.

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

The `Qwen3-VL` label in v2 tool descriptions names the relative image-coordinate
convention: `u` and `v` range from 0 to 1000. The configured main agent supplies
those coordinates to the local RGB-D grasp planner. The
[tool-routing observation](../validation_results/model-service-provenance-20261003/README.md)
records the active model configuration and the v2 catalog used for validation.

A recovered deployment report names vLLM `0.28.1rc1.dev202+gffc445f`; the current
validation endpoint reports that same version. The
[model service evidence](../validation_results/model-service-provenance-20261003/README.md)
preserves the original host's launch script and distinguishes its defaults
from verified live settings. The custom serving installation, checkpoint and
historical weight hashes are not bundled. An identical model name alone does
not establish equivalent server behavior.

```bash
cp configs/example.json configs/local.json
# Edit data_path, Python executables and model_url for this machine.
./run.sh --list
./run.sh --task task01 --gpu 0
```

## Session configuration and ports

For a session-scoped launch, use `scripts/reproduce_task.sh task01 --gpu 0`.
It creates or reuses only
`.local/session-config.json`; it does not modify `configs/local.json`, the
global Claude home, or the global Codex home.

All three listeners can be selected in that session file with a `task_ports`
mapping, for example:

```json
{
  "task_ports": {
    "task01": {"http": 15071, "policy": 15070, "gate": 15072},
    "task03": {"http": 15073, "policy": 15074, "gate": 15075},
    "task08": {"http": 15078, "policy": 15076, "gate": 15077}
  }
}
```

This example fits three concurrent tasks in 15070–15078. Unlisted tasks keep
the default HTTP/policy/gate offsets. `--port` overrides only HTTP when a
task has an explicit mapping. Each task needs three distinct unprivileged
ports; overlapping active runs fail before launch. The wrapper leaves the
mapping intact, and `--dry-run` shows all three resolved ports.

The launcher raises its own file-descriptor soft limit to `nofile_soft_limit`
(default 65536), inherited only by its new child processes. The hard limit
must already allow this value; no system-wide settings are edited. The actual
limits are saved in each run's `runtime_config.json`. Replay history closes
completed chunk files and opens them only during reads, so retained frames do
not each consume a permanently open descriptor. Files live in private local
scratch and are removed after the last reader releases them; an abrupt process
kill can leave files in the stopped run's scratch directory.

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

Execution time varies with hardware, model serving and concurrent load.
The evaluation budget is measured in simulation control steps. The default
`session_timeout_s: 0` therefore disables any additional per-case wall-clock
cap. An operator may set a positive number of seconds in the local configuration
to opt into a safety timeout. If that timeout forces submission, the official
result is retained but the strict reporter excludes the run from reproduction
verification, even when its score matches. The selected timeout is recorded
in each new run's runtime configuration.

## Additional public test instances

The default remains the five archived instances. Additional instances need an
explicit archived prompt/context template, for example:

```bash
./run.sh --task task01 --supplemental-instances 302,303,305,307,309 \
  --context-instance 301 --gpu 0 --config .local/session-config.json
```

These cases use the same task budget and the selected template's prompt, MCP
name and native context. They are marked `supplemental`; their historical
reference scores and score differences are `null`. Their mean is reported
separately from reproduction on the five archived instances. The template
instance is recorded in every case, and is not itself rerun by this command.
