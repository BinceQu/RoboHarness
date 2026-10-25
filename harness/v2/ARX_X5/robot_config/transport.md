# Workspace → evaluator boundary

The operator starts a persistent local worker from this workspace. Separate
Bash processes submit requests over its private Unix socket. This is local
process coordination, needed to preserve tracker state and frozen plans;
it is not an MCP server or a remote implementation of the tools. The worker
imports the full algorithm from this session's `tools/NAME.py`.

The evaluator's standard policy loop calls `reset`, `update_obs` and
`get_action`. A generator may yield a complete 14-scalar joint dictionary.
That dictionary is validated and returned as a one-element action list.
Only the evaluator invokes `take_action`. The next command waits for a newer
observation sequence. Idle, capture, depth, tracking and planning calls emit
no action and spend no evaluator ticks. Original tool action budgets remain.

`robot_config.control.Robot.step` submits exactly one full joint dictionary;
it does no IK, interpolation or collision checking. Higher-level tools perform
their own local geometry and control before yielding these same dictionaries.
All sent commands are recorded in their call's `joint_commands.jsonl`.

Frozen images and plans are signed by a session-local key. Episode IDs,
image IDs, observation sequences and motion epochs retain their original
validation. A new session has a new directory, worker, key and history.
Linux socket peer credentials restrict local requests to the owner or root.
Directory isolation is not an operating-system sandbox for arbitrary Bash.
