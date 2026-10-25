# Tool call history

Each call gets a numbered folder with `input.json`, `result.json`, the exact
atomic joint commands sent to the evaluator (if any), and its observations.
Click calls also save the original frame with crosshairs and `clicks.json`.
Returned image receipts include byte hashes. Earlier folders describe past
states and must not replace the current images when choosing an action.

The native agent transcript, including its Bash commands and reasoning when
provided by the model, is separately stored in `.runtime/claude-home/`.
