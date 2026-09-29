# Archived experiment provenance

The source is `BEHAVIOR-1K/test_results`, with tasks 00, 01, 02, 03, 05, 06,
07, 08 and 09. There is no task04 archive. Each task contains the five actual
instance IDs 301, 304, 306, 308, 310, corresponding to public_test slots
0, 3, 5, 7, 9. The selection seed 20260911 is not the simulation seed;
the pinned upstream evaluator seeds Python, NumPy and Torch with 0.

The directory's reported numbers are retained. The table also exposes means
computed from the selected original evaluator JSON; no historical result is
edited to reconcile conflicting records.

| Task | Name | Directory reported Q | Selected JSON Q |
| --- | --- | ---: | ---: |
| task00 | turning_on_radio | 0.4000 | 0.6000 |
| task01 | picking_up_trash | 0.8667 | 0.8667 |
| task02 | putting_away_Halloween_decorations | 0.6857 | 0.6857 |
| task03 | cleaning_up_plates_and_food | 0.2571 | 0.2571 |
| task05 | setting_mousetraps | 0.8333 | 0.7000 |
| task06 | hiding_Easter_eggs | 0.4222 | 0.4222 |
| task07 | picking_up_toys | 0.3667 | 0.3667 |
| task08 | rearranging_kitchen_furniture | 0.4000 | 0.4000 |
| task09 | putting_up_Christmas_decorations_inside | 0.2222 | 0.2222 |

- task00: INDEX.md and manifest.json report 0.4, selecting older quarantined
  zeros for 301/304. The separate `5test` JSON set averages 0.6. Both values
  are retained; the selected prompt/session set follows the 5test sessions.
  A unique pairing of the manifest's older zeros to those later prompts
  cannot be established from the saved records.
- task05: summary.json reports 0.8333 (301=1, 310=2/3). The JSON files at the
  paths it names contain 301=5/6 and 310=1/6, averaging 0.7. The original
  files are preserved, and the directory report is retained separately.
- task02: the top SUMMARY.md reports 0.4857 using an earlier v17 run for
  instance310. That instance's own later directory records the v18 rerun,
  Q=1.0, so the final per-instance set averages 0.6857.
- Prompt filename annotations sometimes differ from session contents:
  task00 later sessions match v24; task05 instance308 matches v8; task06
  instance301 matches v7; task09's sessions match v3. The actual user prompt
  bytes in each selected session are authoritative for the released prompts.
- task08's matching prompt text was recovered from the monitor session named
  by `session_end_<id>.json`; that monitor lived outside test_results. Its
  source path, session ID and hash are recorded in the task manifest.

Every `tasks/taskXX.json` records the source paths and SHA-256 of the prompt,
source session and official result. `reference_results` contains byte-for-byte
copies of evaluator scores. `prompt` contains only texts referenced by cases;
version names are matched against archived/source prompt files, with a digest
suffix so different texts are never conflated. A `recovered` name means no
versioned file exactly matched the session text. The runner preserves the
source text and changes only a historical connection-port hint in its runtime
copy. Both hashes are recorded.

The runtime pins BEHAVIOR v3.9.1 at
`26f2c7ef7b9cf96bd0414f81e1e751e493762779`, the custom `r1pro_8dof_hf250`
profile, explicit archived ×2 budgets, the fail-closed idle gate, and each
case's prompt. The interface and harness are curated from the supplied local
working trees (September 29, 2026), which include changes not committed in
the original source repository; this is not a claim that every runtime file
was snapshotted on the original experiment date.

Some historical JSON files exceed their declared tick budget. New runs keep
the declared budget and report their actual official scores. Model serving,
CLI changes, floating point simulation and sampling can change outcomes;
replaying the same setup is not a guarantee of identical stochastic scores.
New results are stored separately under runs and never replace reference data.
