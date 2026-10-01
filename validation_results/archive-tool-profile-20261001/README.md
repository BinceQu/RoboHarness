# Archived MCP tool profile audit — 2026-10-01

This is context and regression evidence, **not an official score result**.

The supplied later harness profile hides `plan_press_point`, `adjust_plan_pose`
and `cut_object`. The surviving paper-period profile contains six exclusions.
Archive-mode launches now use that recovered profile; standalone mode
keeps the supplied default. See [audit.json](audit.json) for source hashes.

All 45 source transcript hashes were reverified. They contain 25 press-planner
calls and six plan-adjustment calls, with recorded adapter responses for all
31: 27 remote HTTP 400 errors, three pixel-policy rejections and one transport
timeout. This proves adapter availability, not action success. No archived
`cut_object` call was observed; its visibility comes from the saved profile.

Controlled native CLI captures used Claude Code 2.1.259, a private local stub
on port 15079, the actual read-only interface catalog and the archived prompt.
No model generation or robot operation was requested. Adding only the three
recovered tool definitions increases the current endpoint's token count from
18,858 to 20,619. The new full non-Git capture counts 20,613, compared with
20,647 in archived task01/301. The remaining difference and the unrecorded
historical system prompt prevent a claim of complete request equivalence.

Runtime acceptance independently re-reads the actual MCP recorder manifest.
A different exclusion list, fixed arguments, missing recovered tools, foreign
session or absent/ambiguous final manifest prevents a reproduction claim, even
if the native Skill listing and official scores match.

Validation: root suite 62 checks (61 passed, one optional native-CLI check skipped);
harness profile/catalog/MCP/pixel suite 57 passed; opt-in native-CLI suite 13
passed, including both archived MCP namespaces and compaction. Saved logs and
tested file hashes are listed in the audit. These are regressions, not GPU scores.
