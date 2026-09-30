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
- task06/301: the selected session starts in a scene that had already received
  seven navigation/arm API operations, before Claude started at 23:46 on
  September 16. Its first observation is at local odometry (-5.999, -2.411)
  m, yaw -149.374 degrees, looking down at the basket. A fresh reset starts
  near local odometry (0, 0), looking toward the lawn fence. The earlier
  request parameters and a full scene snapshot were not found in this run's
  saved files. The archived Q=1.0 is preserved, but the fresh-reset launcher
  cannot reproduce this case's starting state from the available records.
  This limitation is recorded in its manifest and prevents a final strict
  reproduction claim in the combined report, even if numerical means match.
  [Initial observation evidence](../reference_results/initial_states.json)
  records all 45 first robot observations and the earlier task06/301 API log
  entries. These observations are not complete simulator snapshots.

Every `tasks/taskXX.json` records the source paths and SHA-256 of the prompt,
source Claude transcript, original archive prompt record and official result.
All 45 selected cases have a matching original transcript, including records
recovered from the supplied harness's private session directories. These
retain the final newline that the monitor's `user_prompt` field had stripped.
`reference_results` contains byte-for-byte
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

## Harness context

The archive's initial Claude Code context exposes seven task skills:
close-box, cut-object, navigate-to-target, open-doors-and-drawers, pick-up-object,
place-object-in-container and traverse-narrow-passages. The supplied later
working tree hid four of these and added rollout-budget telemetry to MCP
instructions and replies. These changes affect the model-visible context.

The reproduction runner selects the recovered archive contract through
`ROBOHARNESS_PROTOCOL=archived-v391-x2`: it restores the seven-skill catalog
and omits that later telemetry. The evaluator and UI still enforce/display
the explicit 2025 ×2 budget. Standalone harness use retains its supplied
default behavior.

The SessionStart hook context matches all 45 selected archived Claude transcripts
byte for byte (SHA-256
`a1ef3ea478487ad0e8d5dc8a162b0042365e874908f22169b785df75b0f838ff`).
The activated pick-up-object, place-object-in-container,
traverse-narrow-passages and open-doors-and-drawers bodies also match every
saved activation in those transcripts.
[The regression fixture](../harness/claude_code/tests/fixtures/archive_context.json)
records source transcript hashes and recovered context/skill hashes.
Claude truncates the saved MCP instruction attachment after 2,048 characters;
the fixture verifies that recorded prefix, not the unrecorded suffix.
Each case also selects its recorded MCP namespace: task02/310, all task07
cases and task08/304 use `plugin:embodied-claude-code:behavior-v2`; the other
38 cases use `behavior-v2`. The launcher preserves Claude's native Skill tool
and listing, and requires the archived Claude Code version 2.1.259.
Claude Code ranks Skill descriptions using a case-local persisted `skillUsage`
store. The original counters were not part of the archived transcript, so the
release records each case's observed initial listing and seeds only the minimal
priority entries needed to reproduce that listing in a fresh `CLAUDE_CONFIG_DIR`.
The seed is written with exclusive creation under the case directory and is
never merged with a user's global Claude home. Every live transcript must still
pass the full initial-listing hash check; names alone are insufficient.
An isolated real-CLI test exercises both namespaces, Skill activation,
deactivation and compaction without contacting a simulator or external model.
This verifies the recovered context components. Complete historical model request
bodies were not recorded, so it does not prove whole-request byte identity or
that every simulator/interface file equals its historical version. The r5 initial
requests also report approximately 1,600 fewer input tokens than the corresponding
archives. The small Skill listing text difference does not explain that entire
difference; it remains a separate unresolved fidelity observation.

## Evaluation budgets

These experiments use **Challenge 2025, multiplier 2**. The exact final integer
limits are taken from the archived plans and original ×2 task table, including
their original rounding. They are not recalculated from the 2026 statistics.
The [archive budget audit](../validation_results/gpu5-20260929/budget_audit.json)
records each source JSON field and SHA-256, alongside the live evaluator checks.

| Task | Maximum simulation steps |
| --- | ---: |
| task00 | 4299 |
| task01 | 10535 |
| task02 | 27664 |
| task03 | 27392 |
| task05 | 20343 |
| task06 | 15239 |
| task07 | 37781 |
| task08 | 17886 |
| task09 | 27437 |

Each manifest records `challenge_year: 2025`, `budget_multiplier: 2` and its
exact `max_steps`. The runner validates these fields and the pinned evaluator
revision, then supplies the same limit explicitly to the evaluator and monitor.
The interface task catalog and monitor fallback also default to 2025 ×2.
The explicit 2026 lookup is retained only to interpret other source records;
it is not used by the reproduction launcher.

The upstream dataset directory is named `2026-challenge-task-instances` even
with the pinned v3.9.1 evaluator. This name does not select the budget year:
the evaluator receives the explicit archived `--max-steps` value above.
