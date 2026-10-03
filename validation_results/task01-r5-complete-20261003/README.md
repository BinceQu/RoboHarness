# task01 r5 completion: mean Q 0.80, target 0.866667

Checked on October 3, 2026 at 19:47 Asia/Shanghai. All five official scores
are present. This diagnostic run **does not reproduce the archived task
mean**: the new mean is 0.80 versus 0.8666666666666666, a difference of
-0.06666666666666654. All five initial native contexts also differ from the
archive, so this run cannot pass the reproduction gate.

| Instance | Official Q | Archived Q | Steps | Finish reason |
| --- | ---: | ---: | ---: | --- |
| [301](../gpu5-20260930-r5/task01/picking_up_trash_301_0.json) | 2/3 | 1 | 7,071 | `model_done` |
| [304](../gpu5-20260930-r5/task01/picking_up_trash_304_0.json) | 2/3 | 1/3 | 10,536 | `evaluator_end` |
| [306](../gpu5-20260930-r5/task01/picking_up_trash_306_0.json) | 1 | 1 | 6,017 | `evaluator_end` |
| [308](../gpu5-20260930-r5/task01/picking_up_trash_308_0.json) | 2/3 | 1 | 5,484 | `model_done` |
| [310](../gpu5-20260930-r5/task01/picking_up_trash_310_0.json) | 1 | 1 | 6,148 | `evaluator_end` |

The differences for 301 and 304 cancel. Instance 308 accounts for the net
deficit of 1/3 across the five scores; its
[outcome audit](../task01-308-r5-outcome-20261003/README.md) records the
observed transport and hand-in behavior. The final instance, 310, completed
with Q=1.0 before the step limit and without wall-clock-forced submission.

The recorded budget is Challenge 2025 times 2, with the archived integer
limit of 10,535 steps. The 10,536 value for instance 304 is retained exactly
as emitted by the official evaluator. Packaged task01 prompt hashes and the
saved archive contract pass verification; these checks do not remove the
initial native-context mismatch.

The task01 systemd service exited successfully with status 0. Its recorded
GPU processes were absent from GPU5, and listeners 15071, 16071 and 17071
were released at the cleanup check. Task03 and task08 remained running;
the full r6 queue was still waiting for the remaining r5 runs to release
resources. No r6 task mean had been measured.

[audit.json](audit.json) records all five original and published score hashes,
the trusted manifest hash, prompt hashes, arithmetic, native-context findings,
and the service/GPU/listener cleanup check. The existing acceptance command
was run on the completed task and returned **2**, as required for an
unverified reproduction:

```bash
python3 scripts/report_validation.py runs/validation-20260930-task01-r5 \
  --output .local/task01-completion-check-20261003 --check-live --require-match
```

All five task01 instances remain scheduled for fresh r6 evaluation. The
mandatory fresh retests task03/301 and task08/304 also remain in their
complete five-instance groups. Old results do not contribute to the new
means. See [release validation](../../docs/validation.md) for the source pin,
archived budgets, uncapped wall-clock policy and session port assignments.
