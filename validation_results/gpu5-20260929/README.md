# GPU validation results

Updated: 2026-09-29T16:33:30+00:00

Only fresh official evaluator JSON contributes to new scores. An incomplete
run has no final mean comparison and is not a successful reproduction claim.

Comparisons cover only the selected instances. Run all five archived instances
to compare a complete task mean.

| Task | Status | Completed | New Q (completed cases) | Archived Q (same completed cases) | Final difference |
| --- | --- | ---: | ---: | ---: | ---: |
| task01 | failed | 0/5 | — | — | — |
| task06 | failed | 1/5 | 0.555556 | 1.000000 | — |
| task08 | failed | 0/5 | — | — | — |

**task01: superseded diagnostic attempt.** Model-visible harness context differed from archived transcripts: three rather than seven advertised skills and added rollout_budget telemetry. Restarting with archived context restoration in a3d0e3f. Preserve all official results; this attempt is diagnostic, not strict reproduction.
Replacement run: `validation-20260930-task01`.

**task06: superseded diagnostic attempt.** Model-visible harness context differed from archived transcripts: three rather than seven advertised skills and added rollout_budget telemetry. Restarting with archived context restoration in a3d0e3f. Preserve all official results; this attempt is diagnostic, not strict reproduction.
Replacement run: `validation-20260930-task06`.

**task08: superseded diagnostic attempt.** Model-visible harness context differed from archived transcripts: three rather than seven advertised skills and added rollout_budget telemetry. Restarting with archived context restoration in a3d0e3f. Preserve all official results; this attempt is diagnostic, not strict reproduction.
Replacement run: `validation-20260930-task08`.

Case scores, hashes and failure details are in [report.json](report.json).
