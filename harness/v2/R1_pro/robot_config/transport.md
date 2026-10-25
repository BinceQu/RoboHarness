# Atomic control and observation boundary

The readable path is:

`tools/NAME.py → tools/util/robot.py → tools/util/driver.py → evaluator`

`driver.py` belongs to this harness. It holds the native evaluator connection,
allowed observations and controller targets across short Python tool processes.
It does not dispatch named robot tools. The evaluator only receives an `action`
array of 27 controller values; see `joints.md` for each slot and unit.

The official evaluator is a WebSocket client: it connects to the harness's
policy port, sends an observation and waits for an action. Start the listener
with this directory's Python before starting the evaluator:

```bash
.venv/bin/python serve_robot.py --policy-port 16070 \
  --socket /tmp/r1-control/control.sock --status /tmp/r1-control/status.json
```

Use a unique, previously unused socket directory and policy port for each
evaluator. The task01 evaluator runner manages these processes and paths.
Binding a session requires a new initialized episode and rejects an episode
already owned by another session.

The native wire format is MsgPack. `tools/util/protocol.py` implements ndarray
encoding. After the `/healthz` probe and WebSocket handshake, the evaluator
sends reset and observation messages. The harness replies with
`{"action": <float32[27]>, "server_timing": {...}}`. It waits while the agent
thinks rather than sending idle commands. After one action, the next observation
acknowledges that action. Reset/disconnection invalidates pending commands.

Local Python processes use the private Unix socket. Each request is a 4-byte
big-endian length followed by MsgPack. `Robot` adds `session_id` and `episode_id`;
the driver rejects a mismatched owner. The physical operation is `step` with
an action and current sequence. `observe` reads the current snapshot without
advancing physics. `joints` builds/retains targets and holding forces; `tracking`
maintains RGB-D tracks. Neither sends motion until a 27-value step is issued.

Explicit 27D or named joint inputs are validated against the joint limits.
Internally constructed hold/interpolation targets are bounded to those same
limits, matching the evaluator's position-control clipping before actuation.
This handles small physics-solver residuals in measured positions without
altering the observations or finger efforts. History records the actual
bounded command sent on the wire.

Only RGB, metric depth, measured proprioception, camera-relative poses and the
allowed budget/task metadata enter the harness. `observation.py` contains the
filter and the 65-value proprioception mapping. Planning, visual tracking and
rendering use this measured data and the included robot model. There is no
simulator scene handle in the agent workspace.

For custom recorded control:

```python
from tools.util.robot import Robot

robot = Robot()
receipt = robot.capture("head")
# Inspect receipt["images"] using the host's native image reader.
# Supply a complete validated 27-value command when ready:
# receipt = robot.command(action, steps=15)
```

`robot_config/control.py` provides the named-joint alternative. Both paths
record inputs, output, action acknowledgements and current perception. The
human monitor is in the evaluator component and only reads these records.
