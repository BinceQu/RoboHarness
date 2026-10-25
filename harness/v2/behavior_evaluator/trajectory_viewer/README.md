# Workspace trajectory viewer

Read-only browser for `behavior_evaluator/runs/*/instance_*/workspace`.
The server uses only Python's standard library. It imports no robot controller,
does not contact the evaluator, and has no action or mutation endpoint.

```bash
cd harness/v2   # from the repository root
/usr/bin/python3 behavior_evaluator/trajectory_viewer/server.py --port 18792
```

Open `http://127.0.0.1:18792`. The `id` query selects `RUN/instance_ID`;
`tab=calls`, `tab=native`, and `tab=access` select the main views. The optional
`call` parameter selects a native tool row (`tool-LINE-BLOCK`) or a robot history
record (`000007_spin_to_facing_point`). Existing port 18790
is a visual reference and remains independent of this viewer.

The nine-task BEHAVIOR V2 viewer is served on `http://127.0.0.1:18793`
with `--run-glob 'test-v3-task*'`. The session picker groups entries by task.
Labels use `task44_instance301_20261012-130225`, with the original instance
start time in Asia/Shanghai. Resuming an instance does not change its name;
restarting from a new scene has a separate entry. Underlying workspace names
and native session IDs are preserved. The RoboDojo viewer is on port 18794.

- Conversation trajectory: all native tools, in order, with four columns:
  preceding Thinking / exact command and parameters / text result / returned images.
  This includes Read, shell reads, schema queries, image processing, and robot
  tools. `Action only` filters robot entry points and joint API calls; it does
  not count source reads or `--schema` / `--help` requests as actions.
  Thinking is joined by model response ID (Claude saves streamed blocks as
  separate events); results are joined by tool ID within each native session.
  Tools in the same response can share Thinking, explicitly linked in the UI.
  Filtering out the first tool still displays that Thinking at the remaining
  tool. Text is never truncated in the API; long blocks scroll or expand.
  Each call defaults to a 360px row, with the four cells aligned. Multiple
  calls from one response appear consecutively. Each cell scrolls independently;
  `展开本次调用` removes the height limit and `恢复统一行高` restores it.
  JSON tool output is displayed as fields, without event envelopes.
  Tool-return thumbnails appear only in the fourth column, preserving their
  original order; calls without images leave that column empty. Thumbnails are
  actual native image blocks and open at original resolution.
  Operator crosshairs or posture images not delivered in the native result are
  not inserted into this model-context view. Original robot records remain
  accessible through `/api/call` and the confined `/file` endpoint.
- User prompts, saved compaction summaries and injected context are retained.
  Repeated hook execution logs, timestamps and transport metadata are hidden
  from the clean trajectory. Actual post-tool reminders/context are available
  under the result's `补充上下文`; this is distinct from tool output.
  The display is the saved conversation history, not a capture of each HTTP
  request. System prompts absent from the log and request-time context trimming
  cannot be reconstructed. A compaction marker does not imply all previous
  messages remain in the next model request.
- Original conversation: every saved JSONL event, including thinking, prompt,
  tool calls, tool results, hooks and context compaction. Images are loaded from
  the exact base64 block in their source JSONL; no re-encoding is performed.
- Workspace access: explicit reads/searches, schema requests, execution,
  programmatic data/image access and local imports are counted separately.
  Running a Python file does not establish reading or understanding its source.
  Shell/Python inspection is conservative and cannot infer arbitrary dynamic
  paths. Raw native messages remain available to check each classification.
- Scores: only official saved evaluator scores. Incomplete cases have no Q;
  an interrupted partial-run mean is never described as a full five-case mean.

The current session refreshes every 20 seconds while its tab is visible.
Refreshing appends new rows and updates pending results, preserving unchanged
DOM rows, reading position, expanded text and historical messages.
Native logs are incrementally parsed; a partially written last line is retried
on the next refresh. At most three native sessions are retained in memory.
The server binds localhost and serves only explicit trajectory files, not
general environment files or arbitrary absolute paths.

```bash
/usr/bin/python3 -m unittest discover -s behavior_evaluator/trajectory_viewer -p 'test_*.py' -v
```

`check_browser.cjs` additionally exercises the running page in Chromium:
all native tool counts, every Bash command and preceding Thinking, result image
counts, exact image SHA256, four-column layout, Action-only filtering, clipboard
fidelity, refresh preservation, context/compaction display, evidence navigation,
prompt display and mobile layout. `PLAYWRIGHT_PACKAGE` and `CHROMIUM_BIN` select an existing local browser
test installation; the viewer itself needs neither. Evidence is saved under
`../validation/trajectory-viewer/`.
