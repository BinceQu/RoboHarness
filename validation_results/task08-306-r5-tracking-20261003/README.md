# task08/306 r5 distance-binding diagnosis — 2026-10-03

This is a snapshot of an ongoing diagnostic episode. It supplies no new
official Q-score and does not validate the queued r6 corrections.

Between 21:10 and 21:52 Asia/Shanghai, recorder turns 330–336 made seven
consecutive `track_object_distance` calls on `img_0304`. All seven operations
returned `ok=true` and `is_error=false`. The first-to-last completion interval
was 42 minutes 7.779 seconds; the seven recorded tool durations total 9.478
seconds. Inter-call time includes inference, transport and CLI behavior; these
measurements do not isolate model-server latency.

The [archived v25 prompt](../../prompt/task08/v25_62c12d99b1ce.txt) requires a
cabinet-lip height of 1.40–1.45 m and camera depth at most 0.7 m. It instructs
the agent to try nearby pixels when a candidate fails those conditions. This
case's saved source prompt is byte-identical to the packaged prompt, and its
rendered prompt changes only the HTTP port from 15068 to 15078.

| Recorder turn | Result of the numeric lip checks | Attached tracking snapshot |
| --- | --- | --- |
| 330–333 | No selected lip candidate passes both checks | Available |
| 334 | `lip_b` and `lip_c` pass | Unavailable |
| 335 | The same two candidates pass after repeating the same selection | Available |
| 336 | Selected `cabinet_lip` at (360, 300) passes | Available |

At turn 336 the attached snapshot reports depth 0.635028303 m and height
1.437908891 m for `cabinet_lip`. The points remain model annotations with
`identity_verified=false`; passing the numeric conditions does not establish
that a point is on the correct shelf. At 22:06:52, turn 337 completed a new
`adjust_chassis(translation=0.18)` operation. The agent therefore moved on
from this binding sequence. That action does not establish successful placement.

The turn-334 warning concerns the additional `/api/memory` snapshot, while
the underlying binding response succeeds and contains point measurements.
The warning does not preserve the original transport or parsing exception.
Turn 335 restores the snapshot. This evidence cannot identify a new connection
defect or justify changing the archived prompt or the running tool behavior.

The exact archived task08/306 native transcript was independently hashed
against its manifest. It contains 500 unique tool calls, including 47 distance
bindings. Two sequences contain six consecutive bindings on the same image.
Of its 428 explicit `persistent_tracking` text snapshots, 13 also report
`available=false`. These historical observations show that repeated binding
and unavailable telemetry predate this release; they do not establish an
identical failure cause or controlled latency comparison. The archived case's
Q=0.25 remains a reference, not the current episode's score.

[audit.json](audit.json) retains the live recorder's bounded prefix hash,
selected record hashes, measured coordinates, prompt hashes, archived
transcript hash, and source lines for the historical counts. It contains no
model reasoning or image payloads. No running process, prompt, source pin,
timeout, or queued evaluation was changed by this inspection.
