# Native asset reload failure

On September 30, 2026, a controlled GPU 5 diagnostic reproduced the native
`BaseMutex::unlock: unlock() called by non-owning thread` abort. Updating only
the mtime of a loaded R1Pro texture, with its SHA-256 unchanged, triggered
SIGABRT on `carb.tasking3`. The native stack contains `libcarb.assets.plugin.so`
followed by `libcarb.tasking.plugin.so`. No agent or model request was involved.
Touching only generated private USD files did not reproduce the abort.

The two existing settings, `/app/extensions/fsWatcherEnabled=false` and
`/app/material/disableMdlReload=true`, do not disable OmniClient asset
subscriptions. The control evaluator still had 235 inotify watches, including
robot textures and scene-object materials. This explains why the earlier
watcher settings were insufficient. It does not identify the external event
that caused the three production evaluators to abort at 17:16:47 CST.

## Process-local mitigation

The evaluator wrapper disables native subscriptions before starting Kit. It
calls the internal `testSetWatchesEnabled(bool)` entry point exported by the
pinned OmniClient binary. This is **not a supported public NVIDIA API**.
The wrapper requires both the exact library hash and version below, sets the
ctypes ABI explicitly, and fails startup on a different binary or missing
symbol. It never modifies the installed SDK binary or global configuration.

- Isaac Sim: 5.1.0, Linux x86-64.
- OmniClient: `2.67.0-release.6072+gl.6293b5e9`.
- Library: `kit/extscore/omni.client.lib/bin/libomniclient.so`.
- SHA-256: `96d901619a7ed20db00e96b8cf41c523865289d11d426cd520bdf9f7c6e60302`.

Evaluation uses immutable asset contents. Initial loading and explicit scene
changes still use the original native read implementation. Automatic reactions
to external file edits are disabled. Do not edit asset files during a run or
use this evaluator for interactive asset development. Camera configuration,
physics, task budgets, prompts and official scoring remain unchanged by this
workaround. A different SDK build requires a new native regression, rather
than adding its hash without testing.

## Evidence and verification

The [native diagnostic audit](../validation_results/native-asset-reload-20260930/audit.json)
and [faulting thread stack](../validation_results/native-asset-reload-20260930/control-fault-stack.txt)
preserve the control evidence. These diagnostics produce no reproduction
score. The repaired runtime passed 20 content-preserving texture mtime events, a
90-second observation window and an explicit official scene reset on GPU 5.
Camera frames were available before and after, and only the device watch
remained. See [fixed regression](../validation_results/native-asset-reload-20260930/fixed-regression.json).
Only new official scoring JSON can establish score reproduction.

CPU regressions exercise unknown binaries, version mismatch, missing symbols,
ABI configuration and idempotence:

```bash
PYTHONPATH=interface .venv-interface/bin/python -m unittest \
  behavior_interface_eval_test.test_native_asset_watches \
  behavior_interface_eval_test.test_official_evaluator_entrypoint
```

They are also included in `scripts/check.sh`. CPU tests alone do not establish
that a native simulator failure is fixed.
