# Independent evaluator installation

The evaluator installation stages completed in an initially empty Python
3.11.16 venv on the existing host. All **16 dependency imports passed**,
and the release's native asset-watch helper passed its binary hash, version
and API checks after the normal Isaac Sim bootstrap. CUDA remained
uninitialized, and no `SimulationApp` or OmniGibson simulator was constructed.
This check does not establish an official Q-score or a task mean.

The source checkout was pinned to
`c4763eb0947adf7f0186834b476a0ec375ee1d1e`, with BEHAVIOR v3.9.1 at
`26f2c7ef7b9cf96bd0414f81e1e751e493762779`. Its tracked files remained clean.
The [audit](audit.json) records source hashes, installed provenance, imports,
native-library identity, version comparisons and the checks' exact scope.
The [package inventory](packages.json) contains all 265 installed
distributions; [the probe output](check-summary.txt) retains the final result.

## Installation and recovery

The check executed the release's evaluator stages: create the venv, install
the build tools, run the fixed-source wrapper around upstream
`setup.sh --bddl --omnigibson --joylo --eval --confirm-no-conda --accept-nvidia-eula`,
verify LeRobot provenance, and install Warp 1.12.0. Dataset downloads and the
other release environments were outside this check.

Downloads required recovery, so this was not a successful unattended first
attempt. A GitHub fetch failed with `curl 52`; the exact recorded LeRobot
commit was then fetched from the already verified private full-history cache.
The PyG `torch-cluster` wheel transfer timed out in pip. Downloading the same
official URL with curl HTTP/2 completed, and its 3,345,861-byte length and
ETag matched. The wheel's SHA256 is
`d39f094625809297a9e35fb2ca1eb99e3dd4cd72f46b9bb9b75f2ebf65b869fe`.
Installing that wheel completed the remaining upstream evaluation step.
Download settings were local to the installation process.

An initial private native-library probe omitted Isaac Sim's bootstrap and
therefore could not resolve `libcarb.so`. The final probe follows the existing
official evaluator's order: import `isaacsim.SimulationApp`, then call
`disable_native_asset_watches()` without constructing an application. It
passes without any change to release runtime code or installed SDK files.

## Observed compatibility and limits

The installed LeRobot commit is
`436812bd8ee39b768c645c248c91f1330834e687`. PyTorch, TorchCodec, Isaac Sim,
Kit, Warp, NumPy, SciPy and the other compared numerical versions match the
effective reference evaluator. The actual OpenCV shared object matches the
reference evaluator byte for byte. This evaluator uses OpenCV 4.11; the
separate [interface environment uses headless OpenCV 4.10](../opencv-runtime-20261001/README.md).

Three compared package versions differ:

| Package | Reference evaluator | Fresh evaluator |
| --- | --- | --- |
| Pillow | 11.0.0 | 11.3.0 |
| pyarrow | 25.0.0 | 25.0.1 |
| websockets | 17.0.1 | 17.1 |

`pip check` reports seven dependency declaration conflicts involving psutil,
typing-extensions, websockets, Pillow, click and packaging. The audit retains
the full output. Successful imports do not prove these conflicts or version
differences harmless in every runtime path.

The newly installed environment has not run a simulator and is not selected
by the pending r6 queue. This is an independent venv check on an existing
host, not a clean-host or dataset installation check. At 21:12 Asia/Shanghai,
all three r5 controller identities and their five recorded process roles were
still alive, each task had one completed case, and r6 remained queued with
the same session configuration. All fifteen corrected cases and the three
complete mean Q-score comparisons remain outstanding.
