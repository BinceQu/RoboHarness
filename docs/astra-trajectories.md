# GPT-6 Astra × RoboHarness V2: 100 BEHAVIOR trajectories

GPT-6 Astra runs inside a fresh RoboHarness V2 workspace to control the R1 Pro
on all 100 BEHAVIOR Challenge 2026 tasks. This release contains the counted
attempt for every task, including successes, partial scores and failures.

[Project](https://roboharness-lab.github.io/) ·
[Code](https://github.com/BinceQu/RoboHarness) ·
[Download files](https://huggingface.co/datasets/BinceQu/RoboHarness-assets/tree/main/astra-v2-behavior)

## Scores

| Set | Tasks | Mean Q |
| --- | ---: | ---: |
| All tasks, including one video estimate | 100 | 0.457166 |
| Official evaluator results only | 99 | 0.456734 |

There are 26 full successes, 43 partial scores and 31 zero scores when the
estimate is included. `score_source` distinguishes `official` from `video_estimate`.

T08, *rearranging kitchen furniture*, stopped at tick 11904 of 13414 without an
official final score. Its **estimated Q is 0.50**: the final head-camera view
shows the toaster and French press inside the same cabinet, while the food
processor remains outside and the cabinet doors remain open. Each of these
four conditions contributes 0.25. The estimate concerns the recorded endpoint;
RGB video cannot establish the evaluator's exact geometric predicates.
`estimates.json` records the conditions, evidence and limitations. The estimate
is included in the 100-task mean, but is never presented as an official score.

## Files

| Path | Contents |
| --- | --- |
| `episodes.json` | Scores, exact task prompts, shared agent settings and media checksums |
| `index.csv` | One row per task, with its trajectory archive and checksum |
| `trajectories.json` | Archive sizes, SHA-256 checksums, source attempt and record counts |
| `trajectories/tNN.tar.gz` | Agent-visible trajectory and executed joint commands for task NN |
| `videos/tNN-head.mp4` | Continuous head-camera recording, 448 × 448, original frame rate |
| `previews/tNN-head.mp4` | Eight-second head-camera time-lapse |
| `posters/tNN-head.jpg` | Head-camera poster |
| `card/` | Homepage video wall and recording provenance |
| `estimates.json` | T08 video-based score estimate |
| `evidence/` | Contact sheets supporting the T08 estimate |

Each trajectory archive contains:

- `prompt.txt`: the exact task prompt delivered to the agent.
- `episode.json`: task, instance, rollout, settings, score and source attempt.
- `metrics.json`: the official result, when one exists.
- `agent/conversation-*.jsonl`: messages, tool calls and tool responses from the
  native Codex conversation. Embedded images are extracted into `images/` and
  referenced by archive-relative paths.
- `agent/stream-*.jsonl`: native Codex event streams.
- `tool_calls/`: tool inputs, results, returned/click-annotated images,
  observation metadata and executed `joint_commands.jsonl` sequences.
- `workspace/`: the available workspace instructions, Python tools and skills.
- `manifest.json`: per-file sizes and SHA-256 checksums.

The videos show only the head camera. The trajectory bundles retain the images
actually returned to the agent, which can include wrist-camera observations.
They are agent-visible interaction records, not simulator checkpoints: raw
RGB-D caches, provider wire-debug logs, runtime authentication files and
encrypted transport payloads are excluded. Credentials and machine-local
service addresses are redacted; the task prompts remain unchanged.

## Use a trajectory

Download one episode with `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download

archive = hf_hub_download(
    repo_id="BinceQu/RoboHarness-assets",
    repo_type="dataset",
    filename="astra-v2-behavior/trajectories/t08.tar.gz",
    local_dir="roboharness-astra",
)
print(archive)
```

Extract each archive into its own directory, then read `prompt.txt` and
`episode.json`. Follow the conversation in `agent/` and the action/observation
sequence in `tool_calls/`. `joint_commands.jsonl` records the executed actions;
an archived workspace snapshot alone is not a standalone simulator installation.

## Shared setup

GPT-6 Astra uses native Codex CLI 0.161.0 with medium reasoning effort and auto
reasoning summaries, inside the RoboHarness V2 R1 Pro workspace. The task prompt
and readable Python tools provide access to perception and the robot's joint
controller. The four skills are `behavior-v2-baseline`, `pick-up-object`,
`place-object-in-container` and `open-doors-and-drawers`.

The benchmark uses official evaluator v3.9.3, public instance 301, rollout 0
and seed 0. Per-task tick budgets are included in `episodes.json`. All tasks
use the same agent setup; each has its own task instruction. The release
contains one counted attempt per task, with its source-attempt identifier.
