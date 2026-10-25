# Shared implementation

All paths below are inside this robot workspace. Tool-specific sequences and
private helpers are in `../TOOL_NAME.py`, including camera capture, grasp
planning, reach motion, tracked motion and skill activation/deactivation.

| Files | Responsibility |
| --- | --- |
| `robot.py`, `protocol.py`, `driver.py` | Atomic commands, allowed observation transport, and the evaluator connection. |
| `controller.py`, `controller_state.py`, `model.py`, `atomic_contract.py` | Joint targets/holding efforts, measured state, limits and exact 27-value layout. |
| `tool.py`, `coordinates.py` | JSON validation, latest-frame checks, one internal geometry-coordinate conversion, operation execution. |
| `history.py`, `perception.py`, `render.py` | Immutable call records, current RGB/depth/tracks, CPU rendering of measured joints. |
| `motion_common.py`, `action_constants.py`, `camera_frames.py`, `trajectory_execution.py` | Functions/constants actually shared by multiple tools; each function retains the baseline algorithm. |
| `geometry/` | Shared RGB-D reconstruction, occupancy, visual tracking, kinematics, IK and numerical worker code. |
| `skills.py`, `session_context.py` | Shared skill-state storage and native context restoration. The activate/deactivate decisions are in their tool files. |
| `session.py`, `settings.py`, `environment.py` | Fresh complete workspace copies, local paths, and isolated Python installation checks. |
| `launch.py`, `observation_hook.py` | Native Claude launch and saved image attachment to Bash results. |

There is no installed first-party implementation package. You can follow every
local import to its source. The evaluator receives joint vectors, never a
macro name such as `pick` or `move_to_reach_point`.
