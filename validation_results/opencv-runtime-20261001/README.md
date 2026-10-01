# OpenCV runtime dependency audit

The interface requirement now selects `opencv-python-headless==4.10.0.84`.
The earlier requirement, `opencv-python==4.11.0.86`, came from package metadata
in a reused environment with three overlapping OpenCV distributions.

The imported `cv2` reports version `4.10.0`. Its binary SHA-256 matches the
wheel RECORD of `opencv-python-headless==4.10.0.84`; it matches neither the
`opencv-python` nor the `opencv-contrib-python` RECORD in that environment.
The three running r5 interface processes map that same binary path.

In the independent Python 3.11 interface environment, the previous OpenCV
wheel was removed and only the headless wheel was installed. `pip check`
passes, and its imported binary is byte-identical to the reference interface's
binary. [The audit](audit.json) records both environments and hashes.

The repository's `scripts/check.sh` passed with this independent interface
environment and GPU access disabled for the checks:

| Check group | Tests run | Passed | Skipped |
| --- | ---: | ---: | ---: |
| Runner, report and archive contracts | 71 | 70 | 1 |
| Interface protocol, RGBD wrapper and robot profile | 62 | 60 | 2 |
| Native asset guard and evaluator entrypoint | 22 | 22 | 0 |
| Claude harness | 187 | 172 | 15 |
| Codex harness | 108 | 108 | 0 |

This establishes dependency identity and software-check compatibility. It
does not identify the cause of any Q-score difference or prove a complete
task mean. The active r5 environments and processes were left intact.
