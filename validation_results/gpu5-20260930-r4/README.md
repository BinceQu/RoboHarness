# GPU validation results

Updated: 2026-09-30T09:17:03+00:00

Only fresh official evaluator JSON contributes to new scores. An incomplete
run has no final mean comparison and is not a successful reproduction claim.

Comparisons cover only the selected instances. Run all five archived instances
to compare a complete task mean.

| Task | Status | Completed | New Q (completed cases) | Archived Q (same completed cases) | Final difference |
| --- | --- | ---: | ---: | ---: | ---: |
| task01 | failed | 0/5 | — | — | — |
| task03 | failed | 0/5 | — | — | — |
| task08 | failed | 0/5 | — | — | — |

Case scores, hashes and failure details are in [report.json](report.json).

All three evaluators aborted at 2026-09-30 17:16:47 CST with the same native mutex assertion. See [simulator_crash_audit.json](simulator_crash_audit.json) for evidence; the common trigger is not yet established.
