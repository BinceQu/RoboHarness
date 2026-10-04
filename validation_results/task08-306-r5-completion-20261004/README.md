# task08/306: Q 0.5 and handoff after more than 35 hours

Checked at 2026-10-04T03:10:59.307394+08:00. The original r5 instance completed at **Q=0.5** and
**12,651 steps**, compared with the directory-reported archived Q=0.25.
Its official task-success flag is false. The native CLI returned a normal
result after 126,785.429 seconds (35 hours, 13 minutes, 5.429 seconds), with
417 turns and no native error flag. The archived Challenge 2025 times 2
step limit remains 17,886.

The original coordinator had been held at its polling sleep to prevent its
already loaded 24-hour deadline from truncating this case. On October 4,
the session-local supervisor observed normal model completion and requested
the existing finish operation at 02:59:39 Asia/Shanghai. It read and validated
the official Q=0.5 result before resuming the same coordinator at 02:59:49.
The prior agent and native CLI exited. The coordinator, interface, gate and
evaluator remained alive, and instance 308 started with a different agent and
native session on port 15078. That new session issued a camera tool call and
received its result.

The legacy controller records the completed case as evaluator_end because
the official file already existed when it resumed. The supervisor events
retain the actual model_done finish request. Neither record indicates a
wall-clock-forced submission. The audit preserves both labels without
rewriting the official scoring file.

This verifies a real final-score and handoff path for the temporary r5
supervisor. The earlier [task03 continuation observation](../task03-306-r5-past-24h-20261003/README.md)
established continued operation past 24 hours; final scoring for that
separate held case was pending at this observation. Later on October 4,
task03/306 also completed and handed off normally, at **Q=1/7 and 25,852
steps** after more than 49 hours. Its separate
[completion audit](../task03-306-r5-completion-20261004/README.md)
preserves that subsequent event.

The three current task08 scores are 0.5, 0 and 0.5 for instances 301, 304
and 306. Their partial mean of 1/3 is **not a complete task mean**. Instances
308 and 310 are unfinished; instance 304 was truncated and r5 retains known
native-context differences. This run does not verify reproduction even if
its eventual numeric mean matches the archived 0.4. The strict three-task
report command returned the expected exit code 2. Fresh r6 validation,
including a new full rollout of task08/304, remains queued.

[audit.json](audit.json) records the original and published score hashes,
the unchanged archived prompt, native completion metadata, supervisor event
prefix hash, process birth identities and the next session's first tool call.
The [official result](../gpu5-20260930-r5/task08/rearranging_kitchen_furniture_306_0.json)
and [live report](../gpu5-20260930-r5/README.md) retain the new score.
