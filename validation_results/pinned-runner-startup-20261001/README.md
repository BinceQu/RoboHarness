# Pinned-source runner startup audit — 2026-10-01

**This is a startup integration check, not a GPU or score reproduction result.**

The next r6 validation uses source commit
`83bb6f0461ee29c4736868299ab0ac06c7780571` in a separate checkout, with BEHAVIOR
at `26f2c7ef7b9cf96bd0414f81e1e751e493762779`. The session-local queue verifies
both revisions and a clean tracked worktree before it starts any tasks.
Development edits in the main repository therefore do not change later r6
cases. Its outputs remain under the main repository's `runs/` directories.

The real `Run.start_agent` implementation was executed for task01/301,
task03/301 and task08/304. The shell launcher, pinned Claude Code 2.1.259,
case-local config, native plugin and MCP recorder ran normally. A local stub
on port 15079 supplied the captured interface catalog and a fixed terminal
model response. The simulator and official evaluator were not started, and
the stub rejected robot operations.

All three starts produced exactly one model request and passed:

- Full archived native Skill-listing hash and non-Git workspace checks.
- Recorded MCP exclusions, restored tools and case-session checks.
- Exact archived source prompt hashes and session-begin identity.
- Both archived MCP namespaces and both native listing patterns.

The pinned checkout's root suite ran 62 tests: 61 passed and the opt-in native
test was skipped. The real CLI was exercised independently by these three
startup checks. [audit.json](audit.json) records the evidence, hashes and scope;
the per-case directories retain the actual case and MCP recorder manifests.
The pinned checkout also passed dependency and dataset preflight; its packaged
robot asset was extracted locally and matched the recorded SHA-256. Its tracked
worktree remained clean after these checks.

Only the r6 queue was reloaded. The three running r5 controller PIDs were
checked before and after and remained unchanged. Full score validation still
requires all five cases of each selected task and the trajectory-directory
scores. None of these startup checks count toward those fifteen evaluations.
