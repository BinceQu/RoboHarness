# task03/301: observed r5 wall-clock cutoff

The official evaluator recorded **Q=0 at 10,024 steps** after the old r5
controller requested `wall_timeout` on October 1, 2026 at 19:29:40
Asia/Shanghai. The archived Challenge 2025 ×2 limit is 27,392 steps.
The case had started at 19:29:39 on September 30; this was the old runner's
24-hour safety deadline, not exhaustion of the archived simulation budget.

[The original official JSON](official.json) is copied byte for byte, and
[the audit](audit.json) retains its hash, the controller log excerpt and
the report's timeout exclusion. The score after consuming the full archived
step budget is unknown. This result does not establish a complete task mean.

The previous launcher and native CLI processes exited. Case 304 started
automatically at 19:29:55 under the same live controller, interface and
evaluator. No service was restarted to perform this handoff.

The queued r6 validation sets `session_timeout_s=0` and keeps the archived
27,392-step limit. The release reporter excludes wall-clock-forced submissions
from strict reproduction verification, while preserving their official scores
as diagnostic evidence.
