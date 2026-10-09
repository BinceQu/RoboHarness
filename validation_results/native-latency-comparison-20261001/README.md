# Archived and r5 native response records

The archived transcript hashes match the packaged task manifests.
[The audit](audit.json) records source hashes, response counts and output
statistics for the three completed r5 case-301 transcripts.

| Case | Archived / r5 responses | Archived / r5 median output tokens |
| --- | ---: | ---: |
| task01/301 | 186 / 189 | 447 / 451 |
| task03/301 | 469 / 290 | 592 / 544.5 |
| task08/301 | 430 / 286 | 619 / 520 |

Each response groups native assistant records by `message.id`; token usage
uses the maximum cumulative `output_tokens` value for that ID.

The trajectories have different actions and response counts. Comparing
equal-length response prefixes also gives similar or lower median output
lengths for r5, but those prefixes are not controlled repetitions of identical
model requests. All six transcripts name `Qwen3.8-Flash-Next-FP8`; this does not
prove unchanged remote weights, serving configuration or complete prompts.

Execution time depends on hardware, model serving, queueing and concurrent
load. These records are not a controlled inference-speed benchmark.
Task03/301 was cut off by the old wall-clock deadline before its step budget
was consumed. Its untruncated outcome is unknown. The runner disables this
extra deadline by default and evaluates each complete task mean separately.

A separate [task03/304 event audit](task03-304-compaction-20261002.json) records
an automatic native context compaction on October 2 while simulation ticks
remained at 4,939. Tool calls resumed afterward and ticks reached 4,972 without
restarting the episode. This ongoing case is not included in the case-301
statistics above; resumed tool calls do not establish a successful task outcome.
