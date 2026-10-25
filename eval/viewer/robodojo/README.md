# ARX X5 trajectory viewer

Adapted from the tested R1 Pro viewer. All implementation and static assets
are copied here; the BEHAVIOR viewer remains unchanged.

```bash
python3 eval/viewer/robodojo/server.py --port 18794
```

Open `http://127.0.0.1:18794`. Port 18793 serves the BEHAVIOR V2 viewer.
Select a timestamped session. The trajectory
shows saved native Thinking / exact tool call / text result / returned images
in four aligned columns. Multiple calls in one response remain separate consecutive
rows. `Action only` filters robot calls; the access view distinguishes source
reads from execution. Image thumbnails come from the native tool result's
actual image bytes, not reconstructed camera state.

Both Claude JSONL and native Codex `codex-home/sessions/**/rollout-*.jsonl`
are supported. For Codex, saved Responses items define the displayed conversation;
separate CommandExecution events supply workspace-access evidence, not extra
stdout that the model did not receive. Outputs are paired by `call_id` within
their source log. An `exec` row can contain several shell commands: the original
JavaScript is preserved, with literal Bash commands also shown for readability.
The action filter selects whole native calls containing robot commands.

Public assistant commentary is labeled separately from reasoning. Some
providers save only encrypted reasoning with no readable summary. The viewer
marks that limitation; it cannot decrypt or reconstruct the model's thinking.
Any readable summary actually present in another log is labeled as a summary.
Images are served from the original saved image blocks, not mutable perception
paths. The official score panel uses the selected RoboDojo result's layout IDs.

From another machine, forward the port first:

```bash
ssh -N -L 18794:127.0.0.1:18794 USER@EVALUATION_HOST
```

Then open `http://localhost:18794/` and choose a saved or live session in the
selector; `?id=RUN_ID&tab=calls` links to one session. The page refreshes every 20 seconds.

History records also retain click crosshairs, RGB-D, measured joints and
offline body views. These are available as evidence without inserting images
the model never received into its conversation. The server is read-only,
binds localhost, and confines file access to the selected run's trajectory.
It cannot issue robot commands or read arbitrary environment files.

The transcript is the saved conversation, not a packet capture of each model
request. It cannot recover system prompts or request-time pruning absent from
the log. Only a saved official evaluator result establishes task success;
passing transport/grounding smoke tests does not establish a task score.

```bash
python3 -m unittest discover -s eval/viewer/robodojo -p 'test_*.py' -v
```
