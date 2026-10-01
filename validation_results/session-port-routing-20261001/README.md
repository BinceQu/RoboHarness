# Session listener routing audit — 2026-10-01

**This check does not establish score reproduction.**

The previous HTTP selection left policy and idle-gate listeners at HTTP+1000
and HTTP+2000. Commit `f547d5a0c23ba3bcf67ff038b11ff02c054fb8ee` adds explicit
per-task listener configuration. The runner applies the same values to the
saved plan, interface environment, policy server, gate, evaluator and locks.
The session wrapper no longer overwrites a configured HTTP port.

The queued GPU5 validation now uses:

| Task | HTTP | Policy | Idle gate |
| --- | ---: | ---: | ---: |
| task01 | 15071 | 15070 | 15072 |
| task03 | 15073 | 15074 | 15075 |
| task08 | 15078 | 15076 | 15077 |

All nine listeners are distinct and inside 15070–15078. The session config
also has explicit 1507* mappings for the other archived tasks; those mappings
share slots and are not intended to run all nine tasks simultaneously. A port
already owned by another run is rejected before process launch.

Only `.local/session-config.json`, `.local/session-config-r6.json` and the
session-local source pin/queue were activated. The global configuration file
hashes remained unchanged. The three r5 controller PIDs stayed unchanged and
active before and after reloading only the queue.

Validation from the independent pinned checkout included:

- 67 root tests: 66 passed, one opt-in native test skipped. Five new tests cover
  configuration validation, CLI precedence, the actual wrapper, prepared
  service commands/environment, and a conflicting policy-port lock.
- All nine task dry-run plans stayed within 1507*, selected five cases, and
  retained their archived Challenge 2025 ×2 integer budgets.
- Dependency and packaged robot-asset preflight passed with a clean tracked
  worktree; the robot asset matched its recorded SHA-256.
- Three actual Claude Code/MCP startup checks passed both archived native
  listing patterns and namespaces, prompt hashes and session identity. They
  used non-actuating HTTP stubs on 15079 and produced no official scores.
- Queue checks rejected overlapping ports, out-of-range ports and booleans.
  Its launch commands use the pinned source, and it waits for the running r5
  services, their child processes, GPU memory and the actual nine listeners.

[audit.json](audit.json) links the evidence and file hashes. The r6 official
fifteen-case evaluation remains queued; none of these checks count toward it.
