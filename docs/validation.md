# Release validation

**The three requested task means have not yet been verified.** This status
snapshot was checked on October 4, 2026 at 12:51 Asia/Shanghai. The
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

At the snapshot time, task01 had completed all 5/5 r5 cases and its service
had exited successfully. Its recorded GPU processes and listeners were
released. Task03 had 3/5 official scores and was running instance 308;
task08 had 3/5 scores and had advanced to instance 308. Their evaluators and
active agents were alive with their recorded process identities. Completed
official scores were:

| Task | Instance 301 | Instance 304 | Instance 306 | Instance 308 | Instance 310 |
| --- | ---: | ---: | ---: | ---: | ---: |
| task01 | 2/3 | 2/3 | 1 | 2/3 | 1 |
| task03 | 0 (wall-clock cutoff) | 2/7 | 1/7 | pending | pending |
| task08 | 1/2 | 0 (wall-clock cutoff) | 1/2 | pending | pending |

Task01/308 ended by model decision at 5,484 steps. Its
[outcome audit](../validation_results/task01-308-r5-outcome-20261003/README.md)
traces the lost blue can to a transport plan executed through the tool that
opens its selected gripper before motion, followed by the archived 0.5 m
hand-in rule. Instance 310 subsequently completed at 6,148 steps with Q=1.0
and `evaluator_end`. The full task01 mean is **0.80**, below its
0.8666666666666666 target by 0.06666666666666654. The differences for 301 and
304 cancel, leaving the net deficit from 308. The
[completion audit](../validation_results/task01-r5-complete-20261003/README.md)
verifies all five official score hashes, the arithmetic, normal cleanup and
the acceptance command's expected exit code 2. This is a measured final
diagnostic mean, not a successful reproduction.

Task03/301 was truncated at 10,024 of 27,392 steps, and task08/304 at
12,089 of 17,886 steps, by the old 24-hour cap. Their recorded Q=0 scores
remain diagnostic evidence; neither is a valid full-budget reproduction.
The [task03 cutoff audit](../validation_results/task03-301-r5-timeout-20261001/README.md)
and [r5 report](../validation_results/gpu5-20260930-r5/README.md) retain the
official scores and hashes. These two partial task results establish no final
mean for task03 or task08.
Known native-context differences also prevent r5 from verifying reproduction,
even if a resulting mean happens to match.

On October 2 the requested wall-clock policy was changed to no additional
cap, permitting episodes longer than 72 hours when needed. New launches use
`session_timeout_s=0`. The already loaded r5 controllers still contain their
old deadline, so a temporary supervisor confined to this validation session
was armed without restarting their agents or evaluators. It holds only the
coordinator at its polling sleep and resumes it when the evaluator supplies
the score, or handles normal model completion through the existing finish
request. Three isolated legacy-controller integration checks passed. The
supervisor intervened for task03/306 at 09:05:30 on October 3. Its original
agent and native CLI subsequently passed 24 hours of execution; between
11:07:21 and 11:12:24 the actual monitor advanced from 16,556 to 16,565 steps,
with a new native tool call and matching result. The
[live audit](../validation_results/task03-306-r5-past-24h-20261003/README.md)
retains process, transcript and score-hash evidence. Final scoring and handoff
were pending at that observation and subsequently completed as recorded below.
This supervisor is a host-specific measure for diagnostic r5, not a dependency
of the packaged runner or a claim that those runs reproduce the archive.

Task08/306 subsequently completed at **Q=0.5 and 12,651 steps**, compared with
the archived case Q=0.25. Its native agent returned normally after about
35 hours and 13 minutes. On October 4 the supervisor requested model completion
at 02:59:39 and validated the official score before resuming the original
coordinator at 02:59:49. Instance 308 then started on port 15078 with a new
agent and native session; the existing interface and evaluator remained alive.
The [completion and handoff audit](../validation_results/task08-306-r5-completion-20261004/README.md)
verifies this real continuation through final scoring. It retains the legacy
controller's evaluator_end label and the supervisor's preceding model_done
request. This case was not wall-clock truncated. The current task08 partial
mean of 1/3 is not a complete five-case mean or a verified reproduction.

Task03/306 then completed at **Q=1/7 and 25,852 steps**, compared with the
archived case Q=2/7. Its native agent returned normally after **49 hours and
29 minutes**. The supervisor requested model completion at 12:35:02 on
October 4 and validated the official score before resuming the same
coordinator at 12:35:12. Instance 308 started with a new agent and native
session on port 15073; the existing interface and evaluator remained alive.
The [task03 completion audit](../validation_results/task03-306-r5-completion-20261004/README.md)
preserves both the model_done trigger and the resumed controller's
evaluator_end label. This case was not wall-clock truncated. The current
task03 partial mean is 1/7; instances 308 and 310 remain unfinished, and the
earlier truncated instance 301 still requires a fresh full rollout.

The corrected r6 evaluation is queued behind the remaining r5 runs and has not launched
as of this snapshot. It uses an independent checkout pinned to
`c4763eb0947adf7f0186834b476a0ec375ee1d1e` and the BEHAVIOR v3.9.1 submodule
at `26f2c7ef7b9cf96bd0414f81e1e751e493762779`. Its reporter uses the task-mean
acceptance rule. This pin includes the OpenCV dependency correction and the
fixed LeRobot source during installation. Simulation, harness and reporting
code are unchanged from the preceding mean-Q revision `46e51e4`.
The queue checks source revisions, a clean tracked worktree, GPU capacity
and listener availability before launching.

The two truncated cases are explicitly bound to fresh r6 evaluations:
task03/301 and task08/304. Each task will run all five archived instances in
order, with new output directories, initial scenes and native agent sessions.
The sequence remains task01, task03, task08. The queue requires a fresh session
and an untruncated result for each mandatory retest; the launcher refuses to
reuse an existing run directory. Sixteen session-queue checks passed after
adding these requirements. The old scores are retained separately and do not
contribute to the new task means. Retests remain queued until r5 releases GPU
5 and the reserved listeners; none had started at the snapshot time.

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

A subsequent [independent evaluator installation](../validation_results/evaluator-clean-install-20261001/README.md)
completed the evaluator stages in an initially empty Python 3.11 venv after
recovering two dependency downloads. All 16 dependency imports and the
process-local OmniClient ABI check passed, with no simulator or CUDA context
started. The loaded OpenCV binary matches the reference evaluator. That
initial installation had three compared version differences and seven
dependency declaration conflicts, retained in its historical audit.

The fresh evaluator also passed an [actual TCP protocol check](../validation_results/evaluator-wire-20261001/README.md)
against the independent interface environment: six full-resolution RGB-D
observations (44,065,560 array bytes), six 27-element actions, two resets and
two connections preserved the expected data. This used the release's client
and server functions with a non-actuating runtime stub on port 15079; the
listener was released afterwards. It establishes transport compatibility for
those versions and payloads, not simulation or task-mean correctness.

The [subsequent runtime pins](../validation_results/evaluator-runtime-pins-20261001/README.md)
correct the observed drift to the reference versions. All 976 payload files
across the six adjusted packages match the reference. Twenty-two imports and
the native ABI check passed, and all 22 compared package versions now match.
The custom robot's TCP exchange also passed with the corrected websockets
17.0.1, including the six observations, actions, two resets and reconnect.
Six unsatisfied SDK dependency declarations remain recorded; `pip check`
still exits 1. The fresh environment has not been selected by r6 or exercised
in a GPU rollout. These installer changes leave the queued runtime code,
source pin and session configuration unchanged.

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
| Fresh evaluator environment | Evaluator installation stages completed with download recovery. Subsequent runtime pins match six package payloads and all 22 compared versions; 22 imports, native ABI and the custom-robot protocol passed. Six SDK declaration conflicts remain. No simulator was started. | [Installation audit](../validation_results/evaluator-clean-install-20261001/README.md), [runtime pin audit](../validation_results/evaluator-runtime-pins-20261001/README.md) |
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

The later task01/308 episode also scored 2/3. Its
[action and feedback audit](../validation_results/task01-308-r5-outcome-20261003/README.md)
identifies opening the selected gripper before transport as the immediate
failure mechanism. The successful archive records the same executor policy
but uses direct transport execution. The audit does not establish why the
model chose a different sequence or validate the queued context corrections.

## Remaining limits

- All fifteen corrected r6 cases and their three complete task-mean comparisons
  remain outstanding. No archived result is substituted for a new result.
- The full evaluator and dataset installation has not been executed on a clean
  host. The fresh venv check covers evaluator installation and CPU imports,
  with its recorded download recoveries and dependency declaration conflicts. GPU tests
  use existing simulator/numerical environments, the packaged
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
