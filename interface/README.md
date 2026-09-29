# Observation and action interface

`behavior_interface_eval_test` is the evaltest interface used by the archived
experiments. `official_policy_interface.py` serves the robot tools and monitor;
`official_rgbd_wrapper.py` converts evaluator observations; the evaluator
entrypoint adds GPU mapping, host resource limits and operator episode control.

`behavior_interface` supplies shared HTTP/UI, recording and geometry code.
The evaluation entrypoint selects the strict `official_v2` tool registry.
`official_idle_step_gate` pauses evaluator ticks while the model is thinking.
`official_eval_harness` retains the shared session guard and metadata helpers;
the release orchestration is in `../roboharness`, with a single `../run.sh` entry.

The agent receives RGB, depth, proprioception, camera-relative pose and task
identity. BDDL scores are kept in the evaluator/display path. The custom
R1Pro profile is under `behavior_interface_eval_test/robot_profiles`.

Use the root runner to create a complete stack. Historical machine-specific
launch scripts are deliberately excluded from this release.
