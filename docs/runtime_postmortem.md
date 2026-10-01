# Validation runtime postmortem

The first GPU5 validation batch (r5) is diagnostic only. It was launched
from revision 23fefd0736ad45c7e91631bce2a12abba3e57937, before the release
fidelity fixes landed, so its results must not be used as a strict reproduction
claim.

The runtime investigation identified these independent defects:

1. The old runner interpreted session_timeout_s: 0 as a deadline equal to
   the current monotonic time. A case could therefore submit a finish request
   immediately even though the archived Challenge 2025 step budget was still
   active. The release runner treats zero as no wall-clock cap; the official
   simulation-step budget remains authoritative.
2. The old local HTTP poller did not force loopback requests around the host
   proxy. Health and monitor polls could intermittently fall into the
   “waiting for official score” path even while the interface and evaluator
   were alive. The release runner uses an explicit loopback transport and
   records the polling exception when a local endpoint is unavailable.
3. The old Claude launcher did not seed the case-local native Skill usage
   state. Claude consequently emitted the default 5,959-character Skill
   listing instead of the archived 5,987-character listing for task01/301.
   The release runner seeds the minimal per-case state in a fresh
   CLAUDE_CONFIG_DIR, requires the full listing hash, and never touches the
   user's global Claude home.
4. Running the CLI from the new Git checkout exposed the release branch,
   worktree status and recent commits in its system prompt. All 45 archived
   transcripts record the original harness cwd and the pinned CLI's `HEAD`
   branch marker, also observed in the controlled non-Git launch. A controlled
   capture shows that `GIT_CEILING_DIRECTORIES` alone does not suppress the
   native metadata. New runs use a fresh case-local non-Git workspace in the
   run's private cache, clear inherited Git overrides only for the agent child,
   and verify the archived branch marker alongside the listing.
5. The supplied later MCP profile also hid `plan_press_point`,
   `adjust_plan_pose` and `cut_object`. The first two appear in 31 archived
   adapter calls, and a surviving September 16 profile contains the earlier
   six-tool exclusion list. Archive mode now selects that earlier list while
   standalone mode retains the supplied default. The saved
   [audit](../validation_results/archive-tool-profile-20261001/audit.json)
   distinguishes registered error-returning calls from successful actions.

New launches also verify that emitted listing while the agent is running. A
known mismatch, missing listing before the first assistant response, ambiguous
transcript, or unreadable evidence fails the run immediately and saves
`instance_<id>/native-context.check.json`. An unpublished or partially written
startup transcript remains pending. Completion re-reads the real transcript,
so neither a stale successful check nor a matching score bypasses the contract.
The same guard checks the actual MCP recorder manifest for the archived
exclusions, fixed arguments, required tools and case session. Missing final
evidence, ambiguous manifests and a changed tool policy cannot pass merely
because the native listing matches. These guards are not injected into the
already running r5 controllers.

The Skill listing mismatch does not establish the cause of the score difference.
Controlled token counting with the current model endpoint attributes 12 tokens
to the two listing variants and 206 tokens to the captured release Git context.
Each comparison changes only that component of one captured request, and
neither invokes model generation. These differences do not explain the larger
historical deficit, and the Git context actually adds tokens.
The first requests recorded for four observed r5 cases also report roughly
1,600 fewer input tokens than their archived counterparts. The complete
historical model request bodies were not saved. A subsequent controlled
capture identifies the filtered tool definitions as a major contributor:
replacing only the tool list adds 1,761 tokens (18,858 to 20,619). All previously
present tool definitions remain unchanged. The new non-Git capture counts
20,613 tokens versus 20,647 in archived task01/301. Remaining path/date-dependent
context is not proven byte-identical, and similar token counts do not establish
complete request equivalence or score reproduction.

The release path is therefore:

- use the current repository revision, not an in-flight diagnostic run;
- run Challenge 2025 with multiplier 2 and the archived integer max_steps;
- launch in the session-local .local/session-config.json and reserved 1507*
  port range;
- require every selected case to have an official score, the directory's
  archive_reported_q, a matching native Skill-context hash, and the verified
  archived MCP tool profile;
- reject a result if it is partial, timed out, superseded, or only mean-matched.

The queued r6 watcher preserves the existing r5 processes and starts the
strict release validation only after those processes exit, GPU5 has enough free
memory, and the reserved ports are available.

The queued r6 source is a separate checkout; its exact revision is recorded in
the session-local `.local/session-config-r6-source.json`, with BEHAVIOR at the
archived commit. The initial pinned-source audit used
`83bb6f0461ee29c4736868299ab0ac06c7780571`.
Both the evaluator launches and reporter run from that checkout. Before launch,
the session-local watcher verifies its revision, submodule revision and clean
tracked worktree. This prevents ongoing edits to the development checkout from
entering later cases of the same long-running validation. The original r5
controllers were left running.

The [pinned-source startup audit](../validation_results/pinned-runner-startup-20261001/README.md)
exercises the real `Run.start_agent` path, launcher, native CLI and recorder for
three representative archived cases. It verifies both MCP namespaces, both
native Skill-listing patterns, source prompt hashes and case session identity.
Its HTTP model/interface endpoints are controlled local stubs, so this is
startup integration evidence, not a simulator run or an official score result.

The original port selection covered only HTTP; the internal policy and idle
gate listeners still used offsets outside the requested session range. New
runs can set all three listeners with `task_ports` in the session config.
The queued tasks use disjoint triples within 15070–15078, and the queue checks
those actual ports before launch. The runner uses the same explicit values
for its plan, interface environment, policy server, idle gate, evaluator and
port locks. No global configuration or running r5 service is changed.
The [session port audit](../validation_results/session-port-routing-20261001/README.md)
records the selected listeners, updated pinned revision, regression evidence
and unchanged r5 process identities.

## Completed r5 task01/301 outcome

The completed r5 case scored 2/3 and ended through model_done. Recorded images
show a placement miss; the archived prompt then directs termination for a
remaining outside can less than 0.5 m away. A comparison of archived gripper
responses does not establish a control-timing regression. This is diagnostic
evidence only; it does not isolate the failure cause or validate the queued
fixes. See [the outcome audit](../validation_results/task01-301-r5-outcome-20261001/README.md).
