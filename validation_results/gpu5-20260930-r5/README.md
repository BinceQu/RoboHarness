# GPU validation results

Updated: 2026-10-02T18:07:21+00:00

Only fresh official evaluator JSON contributes to new scores. An incomplete
run has no final mean comparison and is not a successful reproduction claim.

Acceptance compares each complete task mean Q-score with its directory-reported
task mean (absolute tolerance 1e-6). All archived instances must be present.
Individual case differences are diagnostic and do not prevent a mean match.
Archived prompt, budget and runtime fidelity checks still apply.

| Task | Status | Local processes | Completed | New mean Q (completed cases) | Archived task mean Q | Final difference | Mean matches | Verified |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| task01 | running | alive | 4/5 | 0.750000 | 0.866667 | — | — | no |
| task03 | running | alive | 2/5 | 0.142857 | 0.257143 | — | — | no |
| task08 | running | alive | 2/5 | 0.250000 | 0.400000 | — | — | no |

**task01/301: reproduction limitation.** Initial native context is mismatch; the full listing and archived workspace metadata must match. Native workspace Git metadata differs from archive

**task01/304: reproduction limitation.** Initial native context is mismatch; the full listing and archived workspace metadata must match. Native workspace Git metadata differs from archive

**task01/306: reproduction limitation.** Initial native context is mismatch; the full listing and archived workspace metadata must match. Native workspace Git metadata differs from archive

**task01/308: reproduction limitation.** Initial native context is mismatch; the full listing and archived workspace metadata must match. Native workspace Git metadata differs from archive

**task01/310: reproduction limitation.** Initial native context is mismatch; the full listing and archived workspace metadata must match. Native workspace Git metadata differs from archive

**task03/301: reproduction limitation.** The wall-clock safety timeout forced episode submission; this is not termination under the archived step budget or model completion.

**task03/301: reproduction limitation.** Initial native context is mismatch; the full listing and archived workspace metadata must match. Native workspace Git metadata differs from archive

**task03/304: reproduction limitation.** Initial native context is mismatch; the full listing and archived workspace metadata must match. Native workspace Git metadata differs from archive

**task03/306: reproduction limitation.** Initial native context is mismatch; the full listing and archived workspace metadata must match. Native workspace Git metadata differs from archive

**task08/304: reproduction limitation.** The wall-clock safety timeout forced episode submission; this is not termination under the archived step budget or model completion.

**task08/301: reproduction limitation.** Initial native context is mismatch; the full listing and archived workspace metadata must match. Native workspace Git metadata differs from archive

**task08/304: reproduction limitation.** Initial native context is mismatch; the full listing and archived workspace metadata must match. Native workspace Git metadata differs from archive

**task08/306: reproduction limitation.** Initial native context is mismatch; the full listing and archived workspace metadata must match. Native workspace Git metadata differs from archive

Case scores, hashes and failure details are in [report.json](report.json).
