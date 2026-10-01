# Archived and r5 native response timing

The three completed r5 case-301 transcripts show longer response intervals
without a corresponding increase in typical output length. The archived
transcript hashes match the packaged task manifests. [The audit](audit.json)
records both source hashes, response counts, output statistics and timing
method.

| Case | Archived / r5 responses | Archived / r5 median output tokens | Archived / r5 median response interval | Archived / r5 transcript span |
| --- | ---: | ---: | ---: | ---: |
| task01/301 | 186 / 189 | 447 / 451 | 29.8 / 199.9 s | 3.11 / 13.19 h |
| task03/301 | 469 / 290 | 592 / 544.5 | 57.6 / 216.4 s | 12.40 / 23.81 h |
| task08/301 | 430 / 286 | 619 / 520 | 28.0 / 190.0 s | 6.33 / 20.26 h |

Each response groups native assistant records by `message.id`; token usage
uses the maximum cumulative `output_tokens` value for that ID. Its interval
starts at the last preceding user `tool_result` timestamp and ends at the last
assistant-record timestamp for that ID. These intervals include queueing,
generation, transport, CLI overhead and possible retries or continuation
behavior. They do not isolate server-only latency. Transcript span is measured
between the first and last recorded user/assistant events, excluding simulator
startup and official score finalization.

The trajectories have different actions and response counts. Comparing
equal-length response prefixes also gives similar or lower median output
lengths for r5, but those prefixes are not controlled repetitions of identical
model requests. All six transcripts name `Qwen3.8-Flash-Next-FP8`; this does not
prove unchanged remote weights, serving configuration or complete prompts.

Task03/301 was cut off by the old 24-hour deadline before its step budget was
consumed. Its untruncated outcome is unknown. Together with the longer native
response intervals, that cutoff demonstrates why a wall-clock deadline can
change evaluation exposure even when the simulator step limit is correct.
The queued release run already disables this extra deadline and runs tasks
sequentially. These timing measurements neither establish the cause of a
Q-score difference nor certify any complete task mean.

A separate [task03/304 event audit](task03-304-compaction-20261002.json) records
an automatic native context compaction on October 2 that took 1,600.712 seconds
while simulation ticks remained at 4,939. Tool calls resumed afterward and
ticks reached 4,972 without restarting the episode. This ongoing case is not
included in the case-301 statistics above; the CLI-reported compaction duration
does not isolate model-server timing or establish a successful task outcome.
