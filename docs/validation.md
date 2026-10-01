# Release validation

Validation dates: September 29–October 1, 2026 (Asia/Shanghai). The source checkout is RoboHarness,
with BEHAVIOR v3.9.1 at `26f2c7ef7b9cf96bd0414f81e1e751e493762779`.

The native crash has now been reproduced independently of the model: an
unchanged texture's mtime event triggers the same carb.assets mutex abort.
The process-local native subscription workaround passed 20 texture mtime
events, a 90-second observation window and an explicit scene reset on GPU 5.
At runtime commit 23fefd0, the full CPU suite ran 392 checks: 375 passed and
17 optional checks skipped. Passing checks include native ABI rejection. Subsequent
targeted reporter regressions require strict per-case score comparison;
matching task means alone no longer satisfy reproduction verification.
An additional regression rejects a matching score when the wall-clock safety
timeout forced episode submission. All 19 current runner/report/budget tests
pass with this guard; it changes result validation, not the running agents.
The launcher now defaults to no additional wall-clock cap (session_timeout_s=0);
operators can opt into a finite safety timeout. Three more runner regressions
cover configuration validation, large clock advances with the cap disabled,
and explicit timeout labeling. All 22 runner/report/budget tests pass.
The already running r5 controllers retain their original 24-hour per-case cap;
they are not restarted for this change. A cap-triggered result cannot pass
strict verification, and subsequent launches use the uncapped default.
The reporter's local liveness check additionally detects externally terminated
controllers/services, PID reuse and unreaped exited processes. It re-reads
terminal summaries to distinguish normal cleanup from interruption, and does
not require an agent to remain alive after normal completion. Six added
regressions include killing a real child controller and verifying that the
watcher exits nonzero while retaining completed scores and the original summary.
All 28 current runner/report/budget tests pass. This report-only check is
enabled with `--check-live` on the evaluation host; copied runs can be inspected
without consulting unrelated local PIDs.
See [native asset reload analysis](native-asset-reload.md). This does not yet
establish the three requested task scores.

A subsequent read-only diagnostic captured a loopback TCP establishment
timeout with a client in SYN-SENT and the reverse server tuple in LAST-ACK.
The controller now has bounded connection-establishment retries, matching
the policy already present in the Claude agent, without replaying transmitted
HTTP requests. All 33 root regressions pass, including five new transport
checks and real HTTP-server checks that count received POSTs. Read-only GETs
through the new handler succeeded against all three running interfaces.
Existing evaluation processes were not restarted. See the
[transport analysis and evidence](http-transport.md); this is not score validation.

## Automated checks

The October 1 report audit found that equal scores could previously certify a
saved plan with a different budget, model or harness. The report now checks the
plan against the packaged task archive: Challenge 2025 ×2 and its exact step
limit, evaluator revision/seed, scene/robot configuration, model/harness, case
slots, prompt/reference hashes, MCP namespace and directory-authoritative
target scores. Missing archive evidence or a mismatch prevents verification;
redefining a target in the run plan cannot make a score pass. Model and harness
overrides are also marked diagnostic at launch. The root suite contains 49
tests: 48 passed and one optional real-CLI test skipped in this invocation.
This validates report acceptance and saved plans, not remote model weights or
the unrecorded portions of historical model requests. The three existing r5
plans and queued r6 plans pass the new configuration check; r5's native listing
mismatches still prevent reproduction verification.

An independent startup check reproduced a race in the queued report service:
the controllers are launched asynchronously, so the reporter can start before
their initial plan.json and summary.json exist. The optional
--wait-for-start-s argument now waits for those files with a finite deadline;
expiry returns a failure and never fabricates results. The session-local r6
queue uses 2400 seconds. Tests cover delayed atomic publication, expiry, and
invalid timeouts. Only the queue watcher was reloaded to pick up this option;
all three r5 controller identities remained unchanged. This wait concerns
report startup and does not impose a rollout wall-clock limit.

The independent Git clone passed all 360 tests in `scripts/check.sh`:
344 passed and 16 were skipped. Five additional result-reporting tests cover
incomplete runs, incomplete completion claims, modified scoring files, and
superseded attempts or an unavailable archived starting state that must not
count as reproduction claims. All 15 current runner/report/budget checks pass.
The scratch-space preflight also passes with real free space and rejects a
simulated full filesystem before launching the simulator.
Two budget regression tests also pass: all nine tasks use the archived 2025
×2 limits in the launcher, catalog and monitor, and altered budget/protocol
manifests are rejected before launch.
Two additional archived-context regressions verify the SessionStart text
against all 45 selected archived transcripts, all four activated skill
bodies found there, the seven-skill catalog, and MCP replies without later
`rollout_budget` additions. Claude's saved MCP instruction preview is truncated;
the recorded 2,048-character prefix is verified separately.
The full updated Claude suite passes: 170 passed, 15 optional checks skipped
(185 total). All 14 runner/report/budget checks also pass with the new manifests.
The independent clone was updated to `680288e`; its 14 runner/report/budget
checks pass with the exact transcript prompts and per-case namespace manifests.
The real Claude CLI also passes both recorded MCP namespace variants, with
native Skill activation, deactivation and compaction against a local test model.
This covers the task/result manifests, owned-process cleanup, cross-session
request rejection, the official observation protocol, RGBD wrapper, custom
robot, both MCP adapters, image coordinates, skill lifecycle and launchers.
The skips concern optional simulator/native CLI tests and the old harness
parity suite whose sibling source layout is not present in this release.
The GPU evaluations below exercise the actual simulator and Claude CLI.

All 45 archived case prompt/result hashes pass validation. The 15 retained
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
The 13 runner/report/budget checks and 62 selected interface checks were repeated
in this fresh environment: 73 passed and two optional checks were skipped.
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
skills verify the standalone default profile; the archived reproduction profile
exposes the seven skills recorded in the original sessions. A separate image-grounding
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

The current three-task attempt started on September 30 at 19:21 CST from
runtime commit 23fefd0 after the native regression passed. All five cases
per task are selected, with HTTP ports task01=15071, task03=15073 and
task08=15078. Budgets remain 10535, 27392 and 17886 (2025 ×2). These ports
are explicit session launch arguments; subsequent launches also default to
15070 plus the task index, with configuration isolated to this session.
Each task runs in its own user service, with a separate strict result reporter.
The [current report](../validation_results/gpu5-20260930-r5/README.md) and
[launch audit](../validation_results/gpu5-20260930-r5/launch_audit.json) record
progress. Starting these processes is not evidence that scores match.
The [live budget/context audit](../validation_results/gpu5-20260930-r5/budget_audit.json)
checks the first sessions against their archived prompts, startup context,
skill listing and namespaces, and confirms each limit in the process arguments,
environment, monitor and evaluator log. The [native watch audit](../validation_results/gpu5-20260930-r5/native_watch_audit.json)
confirms the workaround in all three actual evaluator processes. The first
agent-visible head-camera frames agree visually with the archived layouts and
viewpoints, with small rendering differences; [frame hashes and limitations](../validation_results/gpu5-20260930-r5/initial_frame_audit.json)
are recorded. This is not a pixel-identical or complete state comparison.

The independent clone was updated to runtime commit 23fefd0. All 40 current
runner/report/budget/entrypoint/native-watch checks pass there, and the pinned
native ABI check succeeds in the actual evaluator environment.
It was then advanced to reporter commit fcaed41; all 28 current
runner/report/budget/session-ownership checks pass in that clone. The
[reporter regression audit](../validation_results/gpu5-20260930-r5/reporter_regression.json)
records the command, source commit and test-log hash. The three evaluation
controllers retain their original process identities and runtime source; only
the independent reporting service was reloaded with local process checking.

### Earlier attempts

On September 30 at 10:21 CST, a fresh process check found all three prior
controllers and their recorded children absent. Their final interface log entries
are around 05:29 CST, with no official scoring JSON and no terminal summary.
Execution records subsequently confirmed that all three launcher commands and
result watcher ended at 05:29:15.970–971 CST. The watcher exit code is 137,
consistent with SIGKILL; the launchers report -1 without a specific signal.
This simultaneous termination is consistent with shared command-session cleanup,
but the initiating event remains unconfirmed. System/kernel journals are not
readable by the current account, so an OOM or other external kill is not ruled out.
The observed step counts and elapsed times are below the configured limits.
The stale running statuses were corrected to failed/interrupted, while the
original summaries, trajectories and logs were retained. The
[interruption audit](../validation_results/gpu5-20260930-r2/interruption_audit.json)
records the process identities and log hashes. At approximately 10:24 CST, all
three tasks were relaunched from their initial scenes as independent one-shot
user services, using the same archived configurations and all five instances.
These are new attempts; the interrupted cases are not scored as zero.

GPU 5 was cleared of the selected previous evaluation processes. Unrelated
GPU jobs remain running. Three new independent runs use the released source,
the matching archived prompts, and all five instances per task:

| Task | Run directory | Archived mean Q | Status |
| --- | --- | ---: | --- |
| task01 | `runs/validation-20260930-task01-r4` | 0.866667 | Failed: native mutex assertion |
| task03 | `runs/validation-20260930-task03-r4` | 0.257143 | Failed: native mutex assertion |
| task08 | `runs/validation-20260930-task08-r4` | 0.400000 | Failed: native mutex assertion |

At 17:16:47 CST on September 30, all three replacement evaluators aborted
with the same Carbonite BaseMutex::unlock ownership assertion:
"unlock() called by non-owning thread". The evaluator exit codes are -6
(SIGABRT). The runners detected the exits, recorded failed summaries and
cleaned up their remaining owned processes; systemd recorded service failures
at 17:17:02–04 CST. None of the 15 selected cases produced an official score.
The common initiating trigger remains unresolved. The
[native crash audit](../validation_results/gpu5-20260930-r4/simulator_crash_audit.json)
preserves the assertion excerpts, log hashes, last step counts and service
exit records. These failures do not establish an archived score mismatch.

The [pre-interruption audit](../validation_results/gpu5-20260930-r2/budget_audit.json)
confirmed 10535, 27392 and 17886 steps for the preceding three attempts through evaluator
arguments, environment, monitor and evaluator logs. Their first Claude sessions
match the archived SessionStart text, recorded MCP instruction preview, native
plugin skill listing, tool namespace and source prompt bytes. Initial camera
frames were also visually compared against the archived first frames; scene
layout and starting viewpoint agree, with small rendering differences. This
visual check does not claim pixel-identical or complete simulator states.

The [first-attempt budget audit](../validation_results/gpu5-20260929/budget_audit.json)
matches all nine limits to their original archive JSON fields and hashes.
For those three evaluators, command-line arguments, environment, monitor
and evaluator log all agree on 10535, 15239 and 17886 steps respectively.
These runs already used explicit 2025 ×2 limits; the catalog and monitor's
unused 2026 fallback defaults have also been corrected for future launches.
The first attempt's task06 instance 301 scored **0.555556**, compared
with **1.0** in the archive. The model declared completion after 6354 steps;
it did not reach the 15239-step limit. This case has not reproduced its archived
score. Its original evaluator JSON and hash are preserved in the
[first-attempt report](../validation_results/gpu5-20260929/README.md).

Those three attempts were stopped after finding model-visible context drift:
the later source harness advertised three skills instead of the archived seven,
and added `rollout_budget` fields absent from the archived tool replies. They
are retained as superseded diagnostic attempts. Commit `a3d0e3f` restores the
archived context when the launcher selects `archived-v391-x2`. An initial
September 30 restart confirmed those text bodies, then was stopped after
finding another launcher difference: native Skill listing was suppressed,
and some cases used a different MCP namespace. All 45 source transcripts
were subsequently matched to restore each case's recorded namespace and exact
prompt bytes, including final newlines stripped by monitor records.
Those [intermediate attempts](../validation_results/gpu5-20260930/README.md)
are retained separately. The observed score difference
does not establish that this context drift caused the failure.

The next task01/task08 initialization attempts exhausted local scratch space
and exited with signal 11 after texture-cache write errors. No agent case or
official score was produced. Their
[failure summaries](../validation_results/gpu5-20260930-r2/initialization_failures.json)
are retained. Caches from stopped validation attempts were removed, and the
two runs were restarted. The launcher now checks scratch space before launch.

Task06's corrected harness did match the archived startup text, native skills,
tool namespace and 15239-step limit. Comparing first observations then exposed
a different archived starting state in instance 301: the source scene had
already received seven interface operations before the selected Claude session.
The [live context audit](../validation_results/gpu5-20260930-r2/task06_context_audit.json)
and [archived initial-state evidence](../reference_results/initial_states.json)
preserve this finding. That fresh-reset attempt was stopped as a diagnostic;
task03 replaces task06 in the three-task score validation. The source Q=1.0 and
task06 mean remain unchanged. See [provenance](provenance.md) for the limitation.

**Full task scores are not available yet; this is not a claim that the archived
means have been reproduced.** Each run writes its official results and
per-case differences to its own `summary.json` when cases finish.

The earlier local watcher updated the [r4 result report](../validation_results/gpu5-20260930-r4/README.md)
and exited after all three runs failed. No fresh scoring JSON was available. It does not
substitute archived scores for missing new results.

The host uses the existing interface and evaluator environments described in
[setup](setup.md), a new agent virtual environment, Claude Code 2.1.259 and
the existing Qwen3.8-Flash-Next-FP8 model server. The evaluator uses the host
compatibility option `unmask_evaluator_cuda=true`. The shared model server
has a request queue, so wall time includes model waiting time.

The full evaluator and dataset installation has not been executed on a clean
host. The GPU evaluations reuse existing numerical/simulator environments but import all
RoboHarness interface, evaluator wrapper and harness code from this checkout.
Datasets and the SAM 2 checkpoint remain external dependencies.
