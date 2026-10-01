# Release validation

**The three requested task means have not yet been verified.** This status
snapshot was checked on October 1, 2026 at 20:23 Asia/Shanghai. The
[live r5 report](../validation_results/gpu5-20260930-r5/README.md) records
completed official scores; a running process or a passing software test is
not evidence that a task mean matches.

## Mean-Q acceptance clarification

Each selected task must complete all five archived instance IDs: 301, 304,
306, 308 and 310. The arithmetic mean of its newly generated official
Q-scores must match `archive_reported_mean_q` in the trusted packaged task
manifest, using absolute tolerance `1e-6`. Directory-reported means take
precedence over conflicting raw archive files. Individual case scores may
differ and compensate within the same task; they are retained as diagnostics.
Different tasks are checked separately.

Prompt, budget, model/context and source-state fidelity checks remain required.
Missing cases, failed or interrupted runs, an unavailable archived starting
state, and wall-clock-forced submissions cannot pass verification. A saved
run plan cannot redefine the target mean. The
[acceptance audit](../validation_results/mean-q-acceptance-20261001/README.md)
records the regression checks, canonical targets and activation.

```bash
python3 scripts/report_validation.py runs/YOUR_RUN_A runs/YOUR_RUN_B --watch --check-live --require-match
```

Use `--check-live` on the Linux evaluation host. Omit it for copied runs, whose
recorded process IDs cannot establish liveness on another host. The report
retains official JSON files and their hashes, incomplete-run status, per-case
differences and the task-level comparison.

## GPU evaluation

The selected tasks and immutable archive budgets are:

| Task | Instances | Challenge 2025 ×2 step limit | Archived mean Q |
| --- | --- | ---: | ---: |
| task01 | 301, 304, 306, 308, 310 | 10535 | 0.8666666666666666 |
| task03 | 301, 304, 306, 308, 310 | 27392 | 0.2571428571428571 |
| task08 | 301, 304, 306, 308, 310 | 17886 | 0.4 |

At the snapshot time, all three r5 controllers and evaluators were alive.
Each had one completed official case out of five. Task03/301 scored Q=0
after the old 24-hour wall-clock cap truncated it at 10,024 of 27,392 steps;
case 304 started automatically. Its [cutoff audit](../validation_results/task03-301-r5-timeout-20261001/README.md)
records the official JSON, exclusion and successful handoff. Task01/301
scored 2/3 and task08/301 scored 0.5. These are partial diagnostic results;
none establishes a final task mean. The existing
runs retain their launch-time runtime and 24-hour per-case safety cap. They
continue under the instruction to leave running tests in progress. Known
native-context differences make r5 diagnostic evidence rather than a verified
reproduction attempt, even if a resulting mean happens to match.

The corrected r6 evaluation is queued behind those runs and has not launched
as of this snapshot. It uses an independent checkout pinned to
`c4763eb0947adf7f0186834b476a0ec375ee1d1e` and the BEHAVIOR v3.9.1 submodule
at `26f2c7ef7b9cf96bd0414f81e1e751e493762779`. Its reporter uses the task-mean
acceptance rule. This pin includes the OpenCV dependency correction and the
fixed LeRobot source during installation. Simulation, harness and reporting
code are unchanged from the preceding mean-Q revision `46e51e4`.
The queue checks source revisions, a clean tracked worktree, GPU capacity
and listener availability before launching.

Future r6 listeners are supplied by session-local configuration:

| Task | HTTP | Policy | Idle gate |
| --- | ---: | ---: | ---: |
| task01 | 15071 | 15070 | 15072 |
| task03 | 15073 | 15074 | 15075 |
| task08 | 15078 | 15076 | 15077 |

All nine listeners are distinct and in `1507*`; global configuration was not
changed. The runner validates and locks configured ports. See the
[session routing audit](../validation_results/session-port-routing-20261001/README.md).
New runs have no additional wall-clock cap (`session_timeout_s=0`), while the
archived simulator step limits remain enforced.

## Automated checks

The OpenCV-corrected release passed `scripts/check.sh` using the independent Python 3.11
interface environment after the OpenCV correction: **450 tests, with 432
passed and 18 skipped** across its five check groups. The root-suite portion
is 71 tests: 70 passed and one optional native-CLI test skipped. GPU access was
disabled for these checks. The
[dependency audit and check summaries](../validation_results/opencv-runtime-20261001/README.md)
record the actual loaded binary, matching wheel hash and results. The
independent interface environment also passes `pip check`.

The subsequent evaluator source pin passed the updated root suite: **71
passed and one skipped** (72 total). A real LeRobot installation into a fresh
private venv produces 402 package files identical to the reference environment,
and an added Git regression verifies the fixed source despite branch updates
and an unavailable upstream after caching. See the
[source pin audit](../validation_results/evaluator-source-pin-20261001/README.md).
This source-only install does not establish a full evaluator installation.
The root-suite runs overlap and their counts must not be added.

The earlier [mean-Q regression and activation evidence](../validation_results/mean-q-acceptance-20261001/README.md)
documents the acceptance rule at `46e51e4`. Those runtime files are unchanged
in the current pin. Skipped checks are not counted as passing checks.

The following evidence supports specific runtime changes. Earlier check counts
belong to separate runs and must not be added to the current release total.

| Area | Observed result and scope | Evidence |
| --- | --- | --- |
| Native startup | Actual Claude Code/MCP starts passed both archived Skill-list variants, case namespaces and non-Git working-directory checks against non-actuating local stubs. These are not simulator rollouts. | [Pinned startup](../validation_results/pinned-runner-startup-20261001/README.md), [updated session routing](../validation_results/session-port-routing-20261001/README.md) |
| Archived tools | 57 harness checks and 13 opt-in real-CLI checks passed, including both namespaces and compaction. Historical calls establish tool availability, not successful robot actions. | [Tool-profile audit](../validation_results/archive-tool-profile-20261001/README.md) |
| Asset subscriptions | GPU 5 regression survived 20 content-preserving texture mtime events, a 90-second observation window and an explicit scene reset. | [Native regression](native-asset-reload.md) |
| HTTP transport | Real local HTTP-server checks verify connection retries do not replay transmitted POSTs; a captured loopback connection conflict was recovered by a separate read-only client. | [Transport evidence](http-transport.md) |
| OpenCV dependency | A fresh installation loads the same OpenCV headless 4.10.0.84 binary as the active reference interface. The earlier 4.11 requirement came from overlapping package metadata. | [Runtime dependency audit](../validation_results/opencv-runtime-20261001/README.md) |
| Evaluator Git dependency | LeRobot is installed from the recorded commit despite the upstream moving-branch requirement. All 402 fresh package files match the reference environment. | [Source pin audit](../validation_results/evaluator-source-pin-20261001/README.md) |
| Installation and packaging | Earlier independent-clone, fresh interface-environment, cuRobo build and private Codex-install checks cover their recorded environments. They do not establish a complete clean-host evaluator installation. | [Historical checks](validation-history.md#automated-checks) |

## Runtime issues and fixes

The [runtime postmortem](runtime_postmortem.md) and linked audits retain the
evidence and limits of each diagnosis. The principal changes include:

- Process ownership and detached user services preserve unrelated jobs and
  expose external termination instead of leaving a stale successful-looking run.
- A version/hash-gated, process-local native asset-subscription workaround
  addresses the independently reproduced Carbonite mutex abort. It changes
  neither installed SDK files nor global settings. The external event that
  triggered the earlier production crash remains unconfirmed.
- Private non-Git agent workspaces, archived Skill lists, per-case MCP
  namespaces and recovered tool profiles restore recorded startup behavior.
  The runner checks actual native startup and MCP manifests.
- Loopback connection-establishment retries avoid replaying transmitted
  actions. Scratch-space preflight and bounded report-startup waiting address
  observed initialization failures.
- The reporter checks official outputs, source/plan fidelity, liveness and
  complete task means; matching an isolated case cannot certify a task.

The completed r5 task01/301 episode scored 2/3. Its
[placement diagnosis](../validation_results/task01-301-r5-outcome-20261001/README.md)
preserves image and transcript evidence without altering the archived prompt
or robot timing. A single-case difference does not determine the task mean,
and that diagnostic does not isolate the physical cause from context changes.

## Remaining limits

- All fifteen corrected r6 cases and their three complete task-mean comparisons
  remain outstanding. No archived result is substituted for a new result.
- The full evaluator and dataset installation has not been executed on a clean
  host. GPU tests use existing simulator/numerical environments, the packaged
  runtime code, Claude Code 2.1.259 and the shared Qwen3.8-Flash-Next-FP8 server.
  Model queueing contributes to wall time. Datasets and checkpoints are external
  dependencies; see [setup](setup.md).
- Historical transcripts do not retain every original model-request component.
  Startup checks establish the recorded content they cover, not byte identity
  of an unrecorded complete system request or remote model weights.
- Task06/301's archive starts after earlier interface operations whose complete
  simulator state was not recovered. It cannot be certified from a fresh reset
  and is excluded from the selected three tasks. Its archived scores remain
  unchanged; see [provenance](provenance.md).

## Earlier attempts

The [historical validation record](validation-history.md) preserves the earlier
interruptions, failed and superseded runs, environment checks, test counts and
evidence links. Its older statements describe their own checkpoints rather
than the current acceptance rule. Those attempts are not counted as successful
reproductions.
