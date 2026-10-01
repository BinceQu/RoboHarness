# Evaluator runtime dependency pins

The installer now applies the reference versions of Pillow, PyArrow,
websockets, packaging, psutil and click after upstream setup, alongside the
existing Warp pin. This corrects the version drift observed in the
[independent evaluator installation](../evaluator-clean-install-20261001/README.md).
The change is in commit `f440767232f9f7e59fc58b37b6afcc37f6792de7` and
[requirements/evaluator-runtime.txt](../../requirements/evaluator-runtime.txt).

Applying that final installation stage to the private evaluator venv succeeded.
Only the following six distribution versions changed; Warp remained 1.12.0.
All **976 payload files (171,253,183 bytes)** across these six distributions
match the reference environment by relative path, length and SHA256, including
bundled shared libraries. Distribution metadata, bytecode caches and external
console scripts were excluded from that comparison.

| Package | Before correction | Reference and corrected version |
| --- | --- | --- |
| Pillow | 11.3.0 | 11.0.0 |
| PyArrow | 25.0.1 | 25.0.0 |
| websockets | 17.1 | 17.0.1 |
| packaging | 23.0 | 25.0 |
| psutil | 5.9.8 | 7.2.2 |
| click | 8.1.7 | 8.4.2 |

All **22 imports** and the process-local native asset-watch ABI check passed.
The 22 package versions compared in the preceding audit now match the
reference, with no remaining differences in that comparison. The imported
OpenCV binary still matches. CUDA remained uninitialized, and no simulator
was constructed. The [audit](audit.json) records wheel hashes, package-content
digests, imports, dependency declarations and process checks; the
[inventory](packages.json) contains the resulting 265 distributions.

The corrected evaluator's websockets 17.0.1 also passed a real TCP exchange
with the independent interface's 16.1.1, using the release client and server
functions. The custom `r1pro_8dof_hf250` profile exchanged six full-resolution
RGB-D observations with 65-value proprioception, six 27-value actions, two
resets and two connections. All 44,065,560 observation-array bytes and the
actions matched. The server used a non-actuating runtime stub on 15079; both
processes exited successfully and released the listener.

Two private probe configuration errors are retained in the audit. The first
omitted the data-directory variable and could not load task metadata. The
second omitted the robot-profile variable and passed only the stock robot's
contract. The final check supplies both variables as the release launcher
does and explicitly verifies the custom profile and dimensions. Neither
earlier probe is counted as verification of the archived robot configuration.

`pip check` still exits 1, reporting **six unsatisfied SDK declarations** for
packaging, click, Pillow, psutil, typing-extensions and websockets. Some upstream
requirements are mutually exclusive: for example, Isaac Sim requires
Pillow 11.3.0 while OmniGibson requires 11.0.x, and Isaac Sim requires packaging
23.0 while LeRobot requires at least 24.2. The final pins follow the reference
runtime; this is not a complete lockfile or a claim that all simulator paths
tolerate those metadata conflicts.

This check reused the independently installed venv after its documented
download recoveries. It did not repeat the complete installation on a clean
host, download datasets or run a GPU rollout. At 22:16 Asia/Shanghai, all three
r5 controllers and their recorded processes were still alive, each with one
completed case. The r6 source and session configuration remained unchanged,
and the private evaluator venv was still unselected. Runtime source files are
identical to the queued `c4763eb` revision; these changes affect installation
only. All fifteen corrected cases and their three task mean Q-score
comparisons remain outstanding.
