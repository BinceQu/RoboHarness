# Validation runtime postmortem

The first GPU5 validation batch (r5) is diagnostic only. It was launched
from revision 23fefd0736ad45c7e91631bce2a12abba3e57937, before the release
fidelity fixes landed, so its results must not be used as a strict reproduction
claim.

Three independent defects explain the observed behavior:

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

The release path is therefore:

- use the current repository revision, not an in-flight diagnostic run;
- run Challenge 2025 with multiplier 2 and the archived integer max_steps;
- launch in the session-local .local/session-config.json and reserved 1507*
  port range;
- require every selected case to have an official score, the directory's
  archive_reported_q, and a matching native Skill-context hash;
- reject a result if it is partial, timed out, superseded, or only mean-matched.

The queued r6 watcher preserves the existing r5 processes and starts the
strict release validation only after those processes exit, GPU5 has enough free
memory, and the reserved ports are available.
