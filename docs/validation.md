# Release validation

Validation date: September 29, 2026. The source checkout is RoboHarness,
with BEHAVIOR v3.9.1 at `26f2c7ef7b9cf96bd0414f81e1e751e493762779`.

## Automated checks

The independent Git clone passed all 360 tests in `scripts/check.sh`:
344 passed and 16 were skipped. Three additional result-reporting tests cover
incomplete runs, incomplete completion claims, and modified scoring files.
This covers the task/result manifests, owned-process cleanup, cross-session
request rejection, the official observation protocol, RGBD wrapper, custom
robot, both MCP adapters, image coordinates, skill lifecycle and launchers.
The skips concern optional simulator/native CLI tests and the old harness
parity suite whose sibling source layout is not present in this release.
The GPU evaluations below exercise the actual simulator and Claude CLI.

All 45 archived case prompt/result hashes pass validation. The 16 retained
prompt texts and original scoring JSON are distinct from new run output.
Python syntax checks passed for the curated source files. Shell syntax checks
passed for all 13 release launch/build scripts. The external SAM 2 source
(34 model/configuration files) and checkpoint hashes also match their pins.
The interface requirements were also installed in a new Python 3.11.16
virtual environment in the independent clone. `pip check` passes. The pinned
cuRobo commit built successfully with CUDA 12.4 for SM 8.9; all five CUDA
extensions load, and CUDA forward kinematics succeeds for both custom R1Pro
arm configurations. All 172 cuRobo Python, C++/CUDA and YAML source/configuration
files match those in the original runtime environment byte for byte.
The 11 runner/report checks and 62 selected interface checks were repeated
in this fresh environment: 71 passed and two optional checks were skipped.
The fresh environment also starts the interface HTTP service successfully:
session isolation is enabled and the UI returns HTTP 200.
The installer explicitly provides cuRobo's build tools and skips its unused
Git LFS example assets. Evaluator setup also isolates the caller's Conda
prefix so upstream cleanup cannot affect a separate active environment.

The Codex plugin manifest passes the plugin validator. Codex CLI 0.153.4
successfully installed it in a private home and loaded its `embodied` profile
and direct MCP configuration. Installation verifies the source directory and
manifest version to avoid a collision with another local plugin. This is a
configuration check; the historical scores were not obtained with Codex.

Historical tests that compared old skill prose verbatim were removed from the
release test suite without changing those skill documents. Tests for disabled
skills now verify that they cannot activate. A separate image-grounding
benchmark and its tests were excluded because they are not these nine tasks.
An existing lifecycle regression exposed two errors: an empty skill name could
deactivate an already inactive state, and corrupt state could raise outside
the error handler. Both are fixed and covered by the existing regression tests.
Tests for unshipped development benchmarks, live benchmark launchers and
submission utilities were also excluded. Remaining runtime tests are retained;
the optional interface suite collects 1,280 tests without missing-module errors.
Collection alone does not assert that this larger suite passes.
The two retained test modules edited during this cleanup passed 13 tests in
the default stock profile. Two Node.js-dependent UI checks and an 8-DOF-only
fixture were skipped in that configuration.
With the released 8-DOF profile selected, the same modules passed 14 tests;
only the two Node.js checks were skipped.

## GPU evaluation

GPU 5 was cleared of the selected previous evaluation processes. Unrelated
GPU jobs remain running. Three new independent runs use the released source,
the matching archived prompts, and all five instances per task:

| Task | Run directory | Archived mean Q | Status |
| --- | --- | ---: | --- |
| task01 | `runs/validation-task01-r2` | 0.866667 | Running |
| task06 | `runs/validation-task06` | 0.422222 | Running |
| task08 | `runs/validation-task08` | 0.400000 | Running |

Each simulator has connected and initialized the custom robot. The agents
have executed real camera and chassis calls through the new interface.
**Final scores are not available yet; this is not a claim that the archived
means have been reproduced.** Each run writes its official results and
per-case differences to its own `summary.json` when cases finish.

A local watcher updates the [combined result report](../validation_results/gpu5-20260929/README.md)
as cases finish and copies their original official scoring JSON. It does not
substitute archived scores for missing new results.

The host uses the existing interface and evaluator environments described in
[setup](setup.md), a new agent virtual environment, Claude Code 2.1.259 and
the existing Qwen3.8-Flash-Next-FP8 model server. The evaluator uses the host
compatibility option `unmask_evaluator_cuda=true`. At the initial live check,
GPU 5 used approximately 38 GB for the three runs. The shared model server
has a request queue, so wall time includes model waiting time.

The full evaluator and dataset installation has not been executed on a clean
host. The GPU evaluations reuse existing numerical/simulator environments but import all
RoboHarness interface, evaluator wrapper and harness code from this checkout.
Datasets and the SAM 2 checkpoint remain external dependencies.
