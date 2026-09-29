---
name: traverse-narrow-passages
description: Navigate a BEHAVIOR v2 robot through an already-open doorway, narrow threshold, or tight aisle by safely staging outside the entrance, optionally pitching down only for a low obstacle that hides the floor overlay, squaring the chassis to the passage with `adjust_chassis`, then driving `forward` through the opening. After a successful `forward`, keep commanding `forward` until the trailing footprint is through; pitch, lateral translation, and side-on are only for a `forward` whose actual displacement falls far short of the request. Use only when the robot is already close to a passage or door and the opening's visible width is about the same as the blue floor path. Do not use for an ordinary corridor that is clearly wider than that path, to open doors, or to move obstacles.
---

# Traverse narrow passages

<!-- claude-pixel-contract:start -->
The latest image in the conversation represents the current observed state.
Older images are historical context; their points, bounding boxes, and
object-part locations do not describe the current view. After any motion,
re-locate the target in the latest image; do not shift, rotate, or scale an
older location estimate to choose a new click.

Before EVERY point selection, inspect the latest attached clickable head image
and re-locate the physical target. Use its actual 720 x 720 original-image
pixels: integer `u` (column) and `v` (row) in 0..719, with top-left `(0,0)` and
bottom-right `(719,719)`. Do not normalize, rescale, or use crop coordinates.
Bind every point, including every row of a multi-point call, to that exact
current `image_id`. Never copy locations from older images, earlier reasoning,
reference photos, tracking text, or overlay labels. Wrist images are
observation-only and cannot be clicked. If the MCP image check rejects a
click, capture a fresh head image and ground again; do not guess an image_id.
<!-- claude-pixel-contract:end -->

<!-- claude-task-lifecycle:start -->
Follow this procedure only while `traverse-narrow-passages` is the active task
Skill in the latest lifecycle result or current `task_skill_state` notice.
Its presence in history, including after compaction, does not activate it.
After success and all required post-actions, call `deactivate_skill` with
`name="traverse-narrow-passages"`, then continue the parent task under the baseline.
You may also deactivate to cancel or hand control back without claiming success.
Instructions below to stay, retry, or exit describe pursuit of success; they do
not prohibit explicit cancellation or handoff. Carry unfinished work and any
pending cleanup back to the parent task. Physical constraints still apply;
deactivation itself does not move the robot or release a grasp.
<!-- claude-task-lifecycle:end -->

Follow the `behavior-v2-baseline` Skill and use only currently advertised
`behavior-v2` MCP tools. Never manipulate the door, door handle, frame, or a
nearby object. Preserve the arm and gripper states. This Skill completes only
when the complete trailing footprint has cleared the threshold. A stopped,
stalled, timed-out, or `ok=false` sub-step requires reobservation or recovery.
Do not end the session or exit this Skill.

Activate this Skill only after the robot is already close to a passage or
door, and a fresh head view shows that opening's width is about the same as
the blue floor path. A hallway or room entry that is clearly wider than the
blue path is ordinary driving, not this Skill.

## Required workflow

Follow these stages in order. Before considering side-on rotation, establish
the designated staging pose immediately outside the entrance with the complete
chassis still outside the threshold. Overlay conflicts beyond that endpoint are
passage-fit evidence, not staging-route evidence; they cannot justify skipping
the approach or choosing side-on from the initial distant observation.

### 1. Stage at the passage entrance

1. Begin with `capture_head_camera`. Identify the near threshold, both fixed
   passage boundaries, any door leaf intruding into the opening, and a visible
   unobstructed floor point immediately before the entrance.
2. Call `move_chassis_to_floor_point` only when the selected point and the
   complete direct translation to it stay on visible traversable floor and the
   complete chassis envelope has visible positive clearance from both passage
   boundaries and nearby obstacles along that finite route segment. The final
   chassis envelope must remain completely outside the threshold. Reconstruct
   the swept corridor only from the current pose through the preserved-yaw XY
   staging endpoint; do not extend the staging test through the passage. Do not
   substitute the blue base-forward overlay unless the direct displacement is
   forward-aligned. That overlay starts at the base front collision edge, is
   not a complete outline of the stationary chassis, and may continue far past
   the selected staging point. Ignore overlay conflicts strictly beyond the
   staging endpoint when deciding whether the finite approach is safe. Judge
   the whole finite reconstructed corridor, not only the clicked point or route
   centerline. This tool preserves yaw; the current yaw does not need to be
   suitable for crossing.
3. If the complete direct staging corridor is safe, the first chassis motion
   must be `move_chassis_to_floor_point`. Do not treat merely being outside the
   passage, seeing the entrance, or having room to rotate as completion of
   staging. Skip this move only when fresh geometry proves that the complete
   chassis already occupies the designated pose immediately before the
   threshold.
4. If either boundary of the finite reconstructed staging corridor touches,
   overlaps, or cannot be visibly separated from an obstacle before the chosen
   endpoint, do not use passage geometry beyond the endpoint to justify a
   side-on rotation at the current distant pose. Select another visible exterior
   staging point or approach only when its complete finite swept corridor is
   safe. If no safe route to an entrance staging pose can be established, do
   not enter the passage from this pose; choose another safe exterior staging
   route. Inability to reach staging is not evidence that side-on traversal is
   ready.
   Use a blue side rail as a staging boundary only for a forward-aligned finite
   displacement. If a staging attempt has already stalled, first obtain a fresh
   observation. Treat the exact inverse of the
   immediately preceding approach as a history-confirmed recovery path only
   when trusted tool history from the current physical episode identifies its
   last clear start pose, the fresh robot pose matches the recorded stalled end
   pose, the approach had no rotation and only one non-negligible translation
   axis, the scene has not changed, and the robot state is stable. First make one
   contact-relief `adjust_chassis` step of at most `0.05 m` on that exact reverse
   axis, and no farther than necessary. Then command the measured remaining
   inverse path toward the recorded clear pose. Choose any intermediate
   endpoints from the recorded path geometry, current visible clearance, the
   advertised tool range, and a positive stopping margin; split the recovery
   only when uncertainty or tool feedback requires another check, not because
   of a universal per-call distance cap. Continue only when each result moves
   in the expected reverse direction and reduces the measured displacement to
   the clear pose. In every other case, retreat only when the reverse path is
   visibly safe. Never rotate while contact or separation is uncertain or
   before the complete chassis has returned outside the passage. If neither a
   safe recovery route nor a safe exterior inspection/rotation area is
   established, do not rotate or re-enter; seek a safe exterior recovery or
   inspection pose.

### 2. Expose the floor overlay

Only after reaching and freshly verifying the entrance staging pose.
Staging pitch is optional and is only for seeing a hidden floor overlay.
Do not pitch merely because this stage exists. Do not use staging pitch
to decide stuck or side-on.

Pitch only when a low obstacle, such as a bed, sofa, or similar low
furniture, hides the chassis or the floor overlay and a downward look is
needed to see those. Then `-50` is the usual `adjust_pitch` increment
from the reset upright upper-body pose so the head camera can see the
chassis. If the current head image already shows the chassis, do not
pitch. `adjust_pitch.degree` is incremental: never stack another `-50`
after the chassis is already visible.

Do not pitch for a door, door leaf, door frame, or other tall vertical
obstacle. Pitching the upper body into that geometry can jam the trunk
or arms against the leaf or frame. Keep the upright pose and judge
alignment from the current head image.

Do not pitch at a more distant exterior pose in order to choose the
side-on branch.

### 3. Align the blue path outside the threshold

- Use `adjust_chassis` to rotate, translate left or right, and move forward or
  backward until the complete blue path is aligned with the intended route.
  Its robot-base axes are `forward` for +X forward, positive `translation` for
  +Y left, and positive `spin` for counter-clockwise yaw. It has no `leftward`
  argument.
- Change only one of `forward`, `translation`, or `spin` per call throughout
  this Skill. Choose the command magnitude from the measured center or yaw
  error, the complete verified swept path, the advertised tool range, and a
  positive stopping margin. Prefer a geometry-driven correction to arbitrary
  fixed-size increments; split a move only when visibility, clearance uncertainty, or
  tool feedback requires another observation. Do not impose universal
  `0.10 m` translation or `10` degree rotation caps, and do not use tiny
  motions without a specific measured residual to correct. Inspect the
  model-visible post-action head image returned by each call; call
  `capture_head_camera` only when that image is missing or unusable.
- Keep the entire chassis outside the threshold while aligning. Align both the
  near and far blue path edges through the narrowest part of the passage, not
  merely its centerline.
- Separate center/yaw misalignment from a width deficit. Rotation and lateral
  translation can place the blue path between the passage boundaries, but they
  cannot make the blue path narrower. Align by putting as much of the blue
  path as possible inside the visible passage floor. Unused passage floor on
  one side of the blue path or chassis is a center offset: translate toward
  that floor. Positive `translation` is left; negative `translation` is
  right. A blue rail or chassis edge on the bed, wall, or other boundary
  while the opposite side still shows unused passage floor is unfinished
  alignment, not a width deficit. Use lateral translation when the opening
  center is visibly offset from the blue-path center and the opposite blue
  edge still has clearance. Never translate repeatedly toward one boundary
  merely to clear the other blue edge.
- Establish yaw from directions, not from raw gap changes. For a forward route,
  compare the blue longitudinal rails with the fixed passage's longitudinal
  direction across at least two separated slices. A yaw error exists when those
  directions are not parallel. Only after directional alignment, derive center
  offset from the displacement between the blue route centerline and the local
  midpoint of the two passage boundaries at corresponding slices. A tapered,
  stepped, or locally narrower passage can change raw left-right gaps without a
  yaw error; classify that geometry separately as a local center shift, width
  deficit, or obstacle. If the longitudinal direction or local boundary pair is
  not observable, stop that yaw inference instead of inferring yaw from gap
  imbalance; obtain a clearer observation.
- After each alignment motion, identify the limiting blue edge and passage
  boundary in the returned image. Do not issue another translation in the same
  direction unless the previous move visibly reduced a measured center offset
  without creating or worsening a conflict at the opposite edge. Use no fixed
  call-count limit: continue only while each measured result has the expected
  sign, reduces the quantified residual center offset, and adds usable
  post-action evidence. Stop on the first result that fails any of those
  conditions. A tight or touching blue rail is alignment or overlay geometry,
  not permission to rotate side-on. If the passage direction or both fixed
  sides are still ambiguous, stop that heading guess and obtain a clearer
  observation instead of inventing a side-on heading.

### 4. Square on, then keep forwarding until through

From the verified entrance staging pose, use `adjust_chassis` until the
chassis faces the passage squarely. Square-on means the blue longitudinal
rails are visually parallel to the passage direction across at least two
slices, and the blue-path center sits on the local midpoint of the two
fixed sides. Use `spin` for yaw and `translation` only for a measured
center offset.

Once this square-on alignment is established, alignment is finished.
Do not issue another `translation`, extra retreat, or `spin` to refine
center, enlarge a rotation circle, or reach an open-area spin pose.
The next chassis command must be `forward` through the passage.

Then command `forward`. After crossing starts, a successful `forward`
means the chassis is not stuck. Compare `requested` with `actual`. Treat
the step as successful when `actual.forward_m` has the requested sign
and the commanded distance was completed (`linear_target_reached` or
`near_target_ok`, or `|actual|` is not far below `|requested|`). After
that success, command only more `forward` until the complete trailing
footprint has cleared the threshold. Do not `translation`, `spin`, or
`adjust_pitch` after a successful `forward`. Unused floor, a visible
door leaf, or a blue-rail gap in the image does not authorize
adjustment after a successful `forward`.

A `forward` is stuck enough to stop that attempt only when the actual
displacement falls far short of the request: a large
`|requested|-|actual|`, `|actual_axis| < 0.20 * |requested_axis|` when
`|requested_axis| >= 0.05 m`, `obstacle_limited` without reaching, or
a velocity stall with almost no progress. Do not keep forwarding in
that same attempt, and do not treat leftover floor beside the path as
proof that the last `forward` succeeded.

Only after that shortfall: do not translate, retreat farther for
rotation room, or keep aligning. If any part of the chassis is between
the passage boundaries, one reverse `forward` on the same axis is
allowed solely to put the complete chassis outside the threshold.
Then go to Stage 5 and spin in place. Do not start side-on from a
successful `forward` or from a predicted tight fit.

Treat the blue overlay as a base-forward swept-path corridor that begins at the
base front collision edge, not as a complete outline of the stationary chassis.
For a forward-aligned motion, use its rails, fill, and scale as heading and
scale evidence while squaring on and choosing `forward` size. A completed
forward step that made progress is valid only when a visible
positive gap separates each aligned blue rail from its passage boundary at the
same narrowest longitudinal slice and the unrendered current chassis envelope
also remains clear. A blue rail that coincides with, touches, or cannot be
visibly separated from a post, door leaf, wall, bed, or other obstacle is a
width deficit in the overlay, not a completed forward fit and not
by itself a side-on decision. Side-on waits until that real `forward`
has already fallen far short. Do not try to make the blue path narrower
with further lateral translation once alignment is finished.

### 5. Use an image-aligned side-on crossing only when needed

Use side-on traversal only after Stage 4: a real `forward` from the
aligned entrance fell far short of the request. Do not rotate side-on
from a predicted tight fit or after a successful `forward`. The
chassis must still be completely outside the threshold. Never rotate
while any part of the chassis is between the passage boundaries.

The chassis is wider left-right than front-back. Side-on means the long
left-right axis lies along the passage, so the short front-back axis
spans the gap. In the current head image, the blue overlay's
**near/lower transverse border** is the nearest full-width line at the
base front collision edge, marked by the `0.2 m` and `0.4 m` lateral
labels and the short forward ticks on that line. That line is the
chassis left-right axis now. Do not confuse it with a longitudinal blue
side rail, route centerline, direction arrow, or `0.5 m`
forward-distance tick.

1. Choose the turn direction from this image by which way is more
   convenient for the later crossing and the work after it. Positive
   `spin` is counter-clockwise (front swings toward chassis-left).
   Negative `spin` is clockwise (front swings toward chassis-right).
2. Spin in place at the aligned pose. If Stage 4 did square the
   chassis to the entrance, command about `90` in that chosen sign.
   Do not `translation` or extra `forward` before this spin to create
   rotation-sweep clearance. The post-exit rotation-sweep proof below
   does not apply to this Stage 5 spin.
3. That in-place `spin` is the last yaw change. Do not add another
   `spin` to chase residual angle or a prettier lower-border line.
   Requested `spin` and `actual.spin_deg` only confirm that rotation
   happened.
4. After this spin, the chassis center is still the Stage 4 aligned
   pose: that is the most accurate occupancy of the passage. Do not
   `forward`, extra `translation` to recenter, or retreat to "sit in
   the strip." The next chassis command must be the signed
   `translation` that crosses along the passage.
5. Apply the signed-translation rule below and cross. After a correct
   side-on yaw, `translation` moves the long left-right axis along the
   passage; the blue forward overlay now points across the gap and is
   not the crossing.

Before rotating, preserve a stable exterior image, base pose, yaw, the near and
far threshold references, both fixed boundaries, the intended signed lateral
motion axis, and the complete planned swept corridor from the current footprint
through the far-side exit. The later side-on head view cannot replace this
record. After rotation, the blue forward overlay remains chassis-forward and
points across the passage, not along the signed `translation`; it is not the
crossing and cannot prove lateral-sweep clearance. Neither can a stationary
footprint gap or an overlay ending at the camera-facing wall.

After the image shows the blue lower border parallel to the passage,
separately reconstruct the stationary chassis envelope and its signed lateral
sweep; require visible positive gaps on both opposing sides at the near
threshold, narrowest slice, and farther swept corridor. Do not infer those
gaps from the blue longitudinal rails or fill after rotation, because they
still describe base-forward motion across the passage. Do not accept a local
stationary gap when the reconstructed chassis sweep converges toward a
boundary farther through the passage. If the lower border, passage
direction, chassis envelope, or any limiting clearance is ambiguous, do not
rotate on that guessed geometry; obtain a clearer observation. Stop rotating as
soon as these image and corridor conditions hold.

If a nonzero `spin` returns `actual.spin_deg=0`, the command produced no
measured rotation even if top-level `ok=true`; treat it as a controller
dead-zone/no-motion result, not a transport failure and not progress. Never
repeat that command or issue another small corrective `spin` to chase the
residual angle. If the image already proves that the blue lower border is
parallel to the passage direction and the independently reconstructed lateral
chassis corridor has positive clearance, keep yaw fixed and start the
signed `translation` immediately. Otherwise abandon this side-on
alignment attempt; return to a verified exterior staging pose and choose
another route. Do not let a nominal yaw target override the missing image
evidence. Do not insert a post-spin `forward` to "occupy" the passage.

Transform the preserved passage direction through the measured actual pose to
select the signed base translation axis; do not infer the sign solely from a
nominal rotation amount. When the desired crossing direction maps to current
chassis-right, use negative `translation`; when it maps to current chassis-left,
use positive `translation`. A completed `spin` is not itself a crossing
command; the first chassis command after that spin is this signed
`translation`. The pre-spin square-on pose already occupies the intended
passage; do not require a post-spin `forward` occupancy step. The blue
forward overlay after rotation cannot supply lateral-sweep proof. If the
signed corridor cannot be reconstructed from the preserved image sequence,
scale, and actual pose transform, reconstruct it from a clearer observation
without leaving the aligned side-on pose.
Once the in-place spin and the signed-translation axis are established,
keep yaw fixed and choose each single-axis translation endpoint from that
verified corridor and the remaining distance needed to clear the trailing
chassis envelope.

## Cross and clear the doorway

For a valid forward route, choose each `forward` endpoint from the visible
clear corridor, scale, required remaining displacement, and a positive stopping
margin. For a side-on route, use the signed `translation` established above.
After every move, verify that the route remains unobstructed and the newest
robot state shows no unexpected chassis height, tilt, collision, yaw drift, or
reverse progress. Halt the current chassis or arm command on collision
evidence or unstable chassis state, then recover from the observed state.

Treat top-level `ok=true` as confirmation that the tool call completed, not that
the requested chassis displacement succeeded. After every `adjust_chassis`
call, compare `requested` with `actual`, inspect `linear_target_reached` when it
is reported, and inspect `obstacle_limited` and `obstacle_stop_reason`. Count
only measured `actual` displacement along the commanded axis as progress. Let
`requested_axis` and `actual_axis` be the signed values for that one commanded
linear axis. When `|requested_axis| >= 0.05 m`, require `actual_axis` to have the
same expected sign and `|actual_axis| >= 0.20 * |requested_axis|`. Treat zero,
opposite-sign motion, or `|actual_axis| < 0.20 * |requested_axis|` as a hard stall
even if top-level `ok=true`. Apply this symmetrically to positive and negative
`forward` or `translation` requests. Thus, a `+0.10 m` or `-0.10 m` request that
moves only `0.0037 m`, `0.0007 m`, or `0.0009 m` in magnitude is obviously
blocked; the first such result must end that motion attempt. If
`obstacle_limited=true`, `linear_target_reached=false`, a stall is reported, or
this below-20% rule applies, stop immediately and never repeat the same motion
direction. This ends that motion attempt, not this Skill or the session.
After a successful `forward`, those flags are not a reason to stop
and adjust; keep commanding `forward`. After a large requested-versus-actual
shortfall, do not keep forwarding and do not use leftover floor as
permission to realign. Do not inspect-pitch or `translation` to
recenter; go to Stage 5 and spin in place. Treat a reported
velocity stall, partial target failure, or uncommanded yaw drift as
decisive even when `obstacle_limited=false` and no dynamic collision is
reported. Do not add millimeter-scale results across
retries or claim progress from them. Do not claim cumulative progress from
requested distances.

The history-confirmed exact-reverse recovery in Stage 1 applies only after a
hard stall with no unused opposite-side passage floor left to absorb a
center correction. It reverses rather than retries the blocked
direction and applies to a true-contact stall during either staging or
crossing. For a
side-on crossing stall, first obtain a fresh observation and require its pose
to match the recorded stalled end in the same unchanged episode. Sum the
measured signed translations since the last trusted pose whose complete
footprint was still outside the passage. Use one necessary contact-relief
reverse move of at most `0.05 m`, then reduce the measured remaining
displacement along the same axis toward that fully exterior pose. A successful
intermediate slice inside the passage is not a trusted exterior recovery
endpoint. Stop that recovery on any reverse stall, wrong-way displacement,
unexpected yaw or lateral drift, unstable chassis state, or new obstacle
evidence. Do not rotate until a fresh observation confirms that the complete
chassis is back outside the passage and separated from both boundaries.

After recovery, never retry the stalled centerline. A different centerline is
eligible only when new observations identify the fixed boundary that limited
each previous attempt and the transformed swept-corridor proof shows positive
clearance at every previously stalled longitudinal slice. Reclassify the stall
from geometry, not raw near-to-far gap trend: compare the route direction with
the fixed passage direction, then compare the direction-aligned route centerline
with each slice's local passage midpoint and width. For a side-on route, use the
near/lower transverse border as the direction reference; for a forward route,
use the blue longitudinal rails. Treat taper, a step, or a local bottleneck as
passage geometry rather than automatic yaw evidence. Retaining the same
unverified yaw while translating to another centerline is not a different route.
If the route direction diverged from the passage direction, correct yaw while
fully outside and re-establish the applicable visual parallelism before
considering a new swept corridor. Merely shifting the base, seeing local gaps at
the new stationary pose, or guessing the midpoint between stalled lines is
insufficient. Repeated stalls do not authorize blind probing or establish a
useful centerline by interpolation. If the limiting boundaries or any stalled
slice remain unobservable, do not claim traversal success; seek a newly
observable route or a different proven corridor.

Use the observation sequence, scale overlay, and cumulative chassis displacement
to clear the trailing footprint, rather than stopping as soon as the camera or
leading edge crosses the threshold. Continue only while the far-side route is
visibly clear.

Before crossing, preserve the near and far threshold references and the gap from
the leading side of the footprint to the near threshold in the last verified
exterior observation. For a side-on route, use the complete near-edge lateral
scale, or reported `path_width_m`, only as a conservative upper bound on the
chassis-Y motion-axis span needed to clear the trailing side. It is not an exact
stationary-footprint measurement or cross-passage clearance proof. Do not mistake
a half-width marker for the full span. Accumulate only `actual.translation_m`
values with the expected sign. Its magnitude must cover
the leading-side-to-near-threshold gap, the threshold-band depth, the full
motion-axis footprint span, and a positive far-side margin. Do not treat one
footprint span from the starting base pose as enough. If one of those metric
terms is visually uncertain, use the sequence of the same fixed threshold
references to establish that they receded past the trailing side; otherwise
leave traversal unconfirmed and keep using the image sequence; do not invent
a distance.

Use the latest usable post-action image as the fresh final observation. Call
`capture_head_camera` only when that image is missing or ambiguous, or when a
stationary observation is needed to establish settling or clearance. For a
forward route, require open traversable floor in the forward direction and no
passage boundary in the forward exit corridor. For a side-on route, do not
require open traversable floor in the robot's current forward direction: that
axis faces across the passage, so a fixed side boundary may occlude the
projected forward blue path beyond its near edge. Instead require
the latest observation and image sequence to show positive separation between
the stationary footprint and the side boundaries, a clear signed lateral exit
corridor, and the fixed threshold references behind the trailing side according
to the metric bound above. Do not confuse an occluded forward path projection
with a blocked signed lateral exit corridor, but treat any contact or ambiguous
separation at the robot origin as unconfirmed.

After the complete trailing footprint has cleared, do not rotate merely to
inspect the room, face a downstream target, or begin another Skill until the
entire rotation sweep is proven clear for the chassis and the preserved arms
and grippers. This post-exit rule does not apply to the Stage 5 side-on
spin and must not delay or displace that in-place `spin`. The blue overlay
represents only the base-forward swept corridor;
it does not prove clearance for an arm, wrist, finger, or gripper housing that
sweeps toward a wall, frame, table, or other obstacle. A single depth point on a
panel in front of the base, or clearance on only the camera-facing side, is not
a swept-volume proof. Establish positive clearance on every side used by the
planned rotation, using fresh head and wrist evidence when the head view does
not expose the appendage-side gap. If that complete envelope is not observable,
keep the verified exit yaw and stop instead of rotating to search.

After every post-exit rotation, compare the preserved and current arm joint and
gripper states as well as the returned head or wrist images. Treat uncommanded
arm-joint displacement or velocity, a new left-right finger-opening asymmetry,
close obstacle geometry inside a wrist grasp volume, or loss of visible
separation from a fixed boundary as contact evidence even when the chassis tool
reports no dynamic collision or commanded translation. Stop all chassis and arm
motion immediately; do not retry the same rotation direction, extend an arm, or
start a grasp. Do not reverse the rotation unless a fresh observation proves
that the complete reverse sweep back to the last contact-free pose is clear;
otherwise hold the contact-risk pose and keep observing until a contact-free
recovery is supported. Hand off to a manipulation Skill only after a stable,
contact-free post-exit pose is verified and the intended target is freshly and
uniquely reacquired. This is a parent-task Skill switch, not session end.

Do not require the head camera to see the whole chassis; it cannot provide that
view. If trailing-footprint clearance cannot be established from the scale and
motion history, do not claim success. Report the final `image_id` and concrete
visual and state evidence only when claiming a completed crossing.
After successful crossing and any required post-exit checks, call
`deactivate_skill` with `name="traverse-narrow-passages"` before returning
to the parent task or activating a manipulation Skill.
