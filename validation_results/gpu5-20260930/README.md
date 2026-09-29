# GPU validation results

Updated: 2026-09-29T16:46:01+00:00

Only fresh official evaluator JSON contributes to new scores. An incomplete
run has no final mean comparison and is not a successful reproduction claim.

Comparisons cover only the selected instances. Run all five archived instances
to compare a complete task mean.

| Task | Status | Completed | New Q (completed cases) | Archived Q (same completed cases) | Final difference |
| --- | --- | ---: | ---: | ---: | ---: |
| task01 | failed | 0/5 | — | — | — |
| task06 | failed | 0/5 | — | — | — |
| task08 | failed | 0/5 | — | — | — |

**task01: superseded diagnostic attempt.** SessionStart and MCP instruction bodies match the archive, but the CLI advertises plugin-prefixed MCP names and suppresses the native skill listing found in archived sessions. Restoring those launcher settings before score validation.
Replacement run: `validation-20260930-task01-r2`.

**task06: superseded diagnostic attempt.** SessionStart and MCP instruction bodies match the archive, but the CLI advertises plugin-prefixed MCP names and suppresses the native skill listing found in archived sessions. Restoring those launcher settings before score validation.
Replacement run: `validation-20260930-task06-r2`.

**task08: superseded diagnostic attempt.** SessionStart and MCP instruction bodies match the archive, but the CLI advertises plugin-prefixed MCP names and suppresses the native skill listing found in archived sessions. Restoring those launcher settings before score validation.
Replacement run: `validation-20260930-task08-r2`.

Case scores, hashes and failure details are in [report.json](report.json).
