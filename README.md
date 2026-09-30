# RoboHarness

RoboHarness packages the BEHAVIOR evaluation interface, Claude Code / Codex
embodied harnesses, and the per-case prompts used in the archived experiments.

整理自已有测试轨迹：9 个任务、45 个 case。保留各 case 实际使用的提示词，
提供按任务运行、独立记录和成绩对比。

```text
BEHAVIOR/          upstream submodule, pinned to v3.9.1
interface/         evaltest interface, robot profile, shared code, idle gate
harness/
  claude_code/     archived experiments' harness
  codex/           alternate Codex harness
prompt/            only the prompt texts referenced by archived cases
tasks/            per-case prompt / instance / budget / provenance manifests
reference_results/ original official scoring JSON
roboharness/       portable runner and process lifecycle
scripts/           setup, verification, provenance import
```

Follow [installation and model setup](docs/setup.md), then:

```bash
./run.sh --list
./run.sh --task task01 --gpu 0
./run.sh --task task08 --instances 301,304 --gpu 0
./run.sh --task task06 --gpu 0 --dry-run
# session-safe 1507* wrapper:
./scripts/reproduce_task.sh task01 --gpu 0
```

A task defaults to all five archived instance IDs: 301, 304, 306, 308, 310.
Rollout budgets use **Challenge 2025 ×2**, with the exact integer limits from
the archived plans. The launcher rejects a changed year, multiplier, step
limit or evaluator revision. See the [budget table](docs/provenance.md#evaluation-budgets).
Claude Code reproduction also uses the archived seven-skill catalog and tool
replies without the later rollout-budget telemetry. The recovered startup
context and activated skill bodies are checked against archived transcripts.
The [case prompt index](docs/case-prompts.md) links each instance to its archived prompt.
`--instances` takes actual IDs, not slot indices. `--port` selects the HTTP
port (default 15070 + task index); policy and idle gate use port+1000 and
port+2000. `--write-video` enables evaluator video output. Different tasks
can be launched independently when GPU and host memory permit.
The interface UI is served at `http://127.0.0.1:<port>/`; use SSH port
forwarding when running on a remote machine.

The runner starts the observation interface, fail-closed idle gate, official
evaluator and selected harness, waits for each instance to finish loading,
and uses that case's exact prompt. It records the rendered prompt, session,
agent trajectory, evaluator JSON and Q differences under `runs/<run-id>/`.
Both the directory-reported score and the raw reference JSON score are kept
when an archive contains conflicting records.
Ctrl-C stops only that run's owned processes. A failed startup or model error
is recorded as a failure; it is not fabricated into a scored episode.

The archive's stated scores are retained alongside the original scoring
files. Read [provenance and known archive discrepancies](docs/provenance.md)
for task00, task02, task05 and the continued scene in task06/301.
Use new evaluator JSON to assess reproduction;
reference JSON files are never used as outputs of a new run. GPU validation
results are documented in [validation](docs/validation.md).

To collect several completed or ongoing runs into one report:

```bash
python3 scripts/report_validation.py runs/YOUR_RUN_A runs/YOUR_RUN_B --watch --check-live --require-match
```

The report is written to `validation_results/latest/`. An incomplete run
never receives a final score comparison. Completed official JSON files are
copied into the report alongside their hashes and per-case differences.
Use `--check-live` only on the Linux host running these evaluations. It checks
the controller and recorded service PIDs together with their start times,
detects external termination or PID reuse, and reports `interrupted` even if
the saved summary still says `running`. It preserves the original summaries
and scores. Omit this option when inspecting runs copied from another host.
`--require-match` exits nonzero for a failed, incomplete or mismatched run.
Verification requires every selected case to match the directory-reported
score; equal means with different case scores do not pass.
An episode forced to submit by the wall-clock safety timeout is also excluded
from verification, even if its score happens to match. Its official JSON is
retained for diagnosis.

Code is MIT licensed; see [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md). External datasets, keys and
model credentials are not included.
