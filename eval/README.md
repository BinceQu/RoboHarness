# Evaluation side

Everything that runs an evaluation and scores it, independent of how the agent
sees and controls the robot (that is [`../harness`](../harness)).

| Directory | Purpose |
| --- | --- |
| [`runner/`](runner) | The task runner behind `./run.sh` (Python package `runner`). It loads a task from `tasks/`, starts the BEHAVIOR evaluator stack, launches the selected harness's agent for each instance, waits for the official evaluator result and writes `runs/<id>/summary.json`. `core.py` is the main pipeline; `native_context.py` restores the archived Claude Code context; `workspace*.py` add the V2 workspace harness (`--harness v2`). |
| [`behavior/`](behavior) | BEHAVIOR evaluator interface: the official evaluator wrapper and custom R1 Pro profile (`behavior_interface_eval_test`), observation/tool/monitor services (`behavior_interface`), the idle-tick gate and task catalog helpers. See its [README](behavior/README.md). |
| [`robodojo/`](robodojo) | Connects V2 `ARX_X5` workspaces to a RoboDojo evaluator. |
| [`viewer/`](viewer) | Read-only trajectory viewers for V2 BEHAVIOR and RoboDojo runs. |

The simulator itself is the pinned [`../BEHAVIOR`](../BEHAVIOR) submodule.
Task definitions and archived prompts are in [`../tasks`](../tasks) and
[`../prompt`](../prompt).
