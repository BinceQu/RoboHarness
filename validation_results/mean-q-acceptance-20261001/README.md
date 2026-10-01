# Mean-Q acceptance clarification — 2026-10-01

The user clarified that task-level mean Q-score is the acceptance metric.
The earlier requirement for every individual case score to match was too strict
and is removed in commit 46e51e4e8ffaafdfa802d8a9580aa42f8a5737f2.

Each task must finish its complete archived instance set. Its new mean is then
compared directly with archive_reported_mean_q from the trusted task manifest,
using absolute tolerance 1e-6. Different individual case scores may compensate
within that task. Different tasks are checked separately. Case-level values
remain in the report for diagnosis and provenance, including conflicts between
directory reports and original raw result files. Saved run plans cannot redefine
the authoritative mean target.

Prompt, budget, model/context, source-state and official-result checks remain.
Incomplete runs, missing instances and invalid or missing mean targets cannot
be verified. The regressions explicitly accept equal means with different case
scores, reject a completed subset and reject a run redefining its target.

The independent pinned source passed 71 checks: 70 passed and one optional
native-CLI check skipped. Preflight passed. The evaluation runner, interface,
harnesses, prompts, task manifests, BEHAVIOR submodule and task launch wrapper
are byte-identical to the prior functional pin; this update changes reporting
and acceptance, not agent behavior.

The live r5 reporter and the queued r6 source now use this rule. Only the report
service and session queue were reloaded; all three running r5 controller PIDs
remained unchanged. Future listeners still use the session-only 1507* mapping.
This evidence is not a claim that the three task means have been reproduced.

Selected task targets are task01 = 0.8666666666666666,
task03 = 0.2571428571428571 and task08 = 0.4. Audit details and all nine canonical
targets are in audit.json; regression output is in root-tests.log.
