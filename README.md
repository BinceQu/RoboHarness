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
```

A task defaults to all five archived instance IDs: 301, 304, 306, 308, 310.
`--instances` takes actual IDs, not slot indices. `--port` selects the HTTP
port (default 16060 + task index); policy and idle gate use port+1000 and
port+2000. `--write-video` enables evaluator video output. Different tasks
can be launched independently when GPU and host memory permit.

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
for task00, task02 and task05. Use new evaluator JSON to assess reproduction;
reference JSON files are never used as outputs of a new run. GPU validation
results are documented in [validation](docs/validation.md).

Code is MIT licensed; see [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md). External datasets, keys and
model credentials are not included.
