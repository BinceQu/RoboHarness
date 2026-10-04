# task03/306: Q 1/7 and handoff after more than 49 hours

Checked at 2026-10-04T12:51:51.731607+08:00. The original r5 instance completed at **Q=1/7
(0.14285714285714285)** and **25,852 steps**, compared with the directory-reported
archived case Q=2/7. Its official task-success flag is false. The native CLI
returned normally after 178,169.447 seconds (**49 hours, 29 minutes, 29.447
seconds**), with 544 turns and no native error flag. The archived Challenge
2025 times 2 step limit remains 27,392.

The session-local supervisor had held the original coordinator at its polling
sleep on October 3 to prevent its already loaded 24-hour deadline from
truncating this case. On October 4 it observed normal model completion and
requested the existing finish operation at **12:35:02 Asia/Shanghai**. It
validated the official Q=1/7 result before resuming the same coordinator at
**12:35:12**. The original agent and native CLI exited. The coordinator,
interface, gate and evaluator remained alive, and instance **308** started
with a different agent and native session on port **15073**. That session
issued its first camera call at 12:37:00 and received its result at 12:37:05.

The legacy controller records evaluator_end because the official score file
already existed when it resumed. Supervisor events retain the preceding
model_done finish request. The audit preserves both labels and verifies that
this case was not submitted because of a wall-clock cutoff.

The new native process uses Qwen3.8-Flash-Next-FP8 at
http://100.101.73.1:31000. This observation verifies two selected environment
keys; it does not prove historical model weights or server arguments. The
completed and next case prompts match their archived case manifests, including
the recorded render with this session's port.

This completes the runtime path whose earlier
[past-24-hour observation](../task03-306-r5-past-24h-20261003/README.md)
had established continued execution. The separate
[task08 completion](../task08-306-r5-completion-20261004/README.md)
had already verified another real final-score and handoff event.

The three task03 scores are 0, 2/7 and 1/7 for instances 301, 304 and 306.
Their partial mean is 1/7; **the complete five-case task mean remains
unverified**. Instances 308 and 310 are unfinished, instance 301 was truncated,
and r5 retains known native-context differences. The acceptance command returned
the expected exit code 2. These scores are diagnostic; individual case
differences do not replace the user's complete-task mean acceptance rule.
Fresh r6 validation, including a new full rollout of task03/301, remains queued.

[audit.json](audit.json) records the original and published score hashes,
prompt hashes, native completion metadata, supervisor event prefix, process
birth identities and the next session's first call/result pair. The
[official result](../gpu5-20260930-r5/task03/cleaning_up_plates_and_food_306_0.json)
and [live report](../gpu5-20260930-r5/README.md) retain the new score.
