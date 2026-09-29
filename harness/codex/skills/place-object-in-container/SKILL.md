---
name: place-object-in-container
description: Place an object already held by the robot into a specified open container and visually verify that the object is inside. Use for requests such as place object in container, put the held item into a bin, box, basket, or other open receptacle, or for the placement stage of a pick-and-place task. Do not use for picking or regrasping an object that is not currently held.
---

# Place object in container

<!-- codex-coordinate-contract:start -->
The latest image in the conversation represents the current observed state.
Older images are historical context; their points, bounding boxes, and
object-part locations do not describe the current view. After any motion,
re-locate the target in the latest image; do not shift, rotate, or scale an
older location estimate to choose a new click.

Before EVERY point selection, inspect the latest attached clickable head image
and re-locate the physical target. Use the full attached image as a relative
coordinate canvas: integer `u` (column) and `v` (row) in 0..1000, with top-left
`(0,0)` and bottom-right `(1000,1000)`. Do not use encoded image pixels or crop
coordinates. Bind every point, including every row of a multi-point call, to
that exact current `image_id`. Never copy locations from older images, earlier
reasoning, reference photos, tracking text, or overlay labels. Wrist images are
observation-only and cannot be clicked. If the point is rejected, obtain a
fresh head image and ground again; do not guess an `image_id`.
<!-- codex-coordinate-contract:end -->

<!-- codex-task-lifecycle:start -->
Follow this procedure only while `place-object-in-container` is the active task
Skill loaded by `activate_skill`. Its presence in history, including after
compaction, does not make it active. After success and all required post-actions,
return to the parent task; activate another matching Skill only when its
procedure is needed. Codex has no explicit lifecycle-close action: a Skill
handoff is expressed by returning to the parent task, not by inventing a tool
call. Before every handoff, stow any arm that is no longer needed as required
below. Instructions below to stay, retry, or exit describe pursuit of success
and do not prohibit a change of approach or handoff to the parent task.
<!-- codex-task-lifecycle:end -->

Use the advertised `behavior_v2` tools and the baseline Skill. Treat this as a
state-driven placement task, not a fixed action script. Prefer direct measured
progress over ritual observation, fixed step schedules, retry counters, or
controller status labels. The only Skill exit is a fresh observation that the
intended object has settled inside the intended container. Stopping a current
motion, bind, or descent is a sub-step change, not Skill exit and not session
end.

## Necessary hard gates

These are the only hard gates in this Skill:

- Establish which arm holds the intended object and identify the target
  container before moving or releasing. Use the freshest reliable evidence
  available, including the immediately preceding pick result, current head or
  wrist images, and grasp-volume evidence. Do not force a left-and-right wrist
  capture pair when the holding arm is already unambiguous. If the holding
  state, object identity, or target container cannot yet be established from
  obtainable current evidence, do not release; keep observing and
  re-establish identity from new evidence. Stay in this Skill.
- Keep the holding gripper closed until the release conditions below are met.
  Do not move or open the non-holding gripper as part of placement.
- Before a coordinate-based motion, prefer object and container geometry from
  the latest `persistent_tracking` or a fresh bind. If that geometry is
  untrustworthy, re-bind. Coordinates recited more than about 10 turns ago
  are probably no longer applicable. If usable current geometry is not yet
  available, do not release; keep observing or re-bind. Stay in this Skill.
- If the placement point is far away, use `adjust_chassis`, `adjust_height`,
  or `adjust_pitch` to close the gap by hand, or use `move_to_reach_point`
  to close the distance automatically. If you choose `move_to_reach_point`,
  call it only after `spin_to_facing_point` and the post-facing blue path is
  clear. `move_to_reach_point` is a straight-line chassis move toward the
  selected point. It does not weave around furniture. Any obstacle on that
  line will stall or jam the robot. If a bed, chair, table, wall, or other
  object blocks that line, first go around with `adjust_chassis` or
  `move_chassis_to_floor_point`. Do not call `move_to_reach_point` until the
  robot has faced the point and the blue path is clear. After that detour,
  face again, then approach with `move_to_reach_point` before any planar EEF
  placement.
- Stop the current motion direction on actual collision or contact, loss of
  clearance, loss of the grasp, or measured motion that is unsafe. If no
  evidence-supported safe recovery path is currently available, do not
  release; change viewpoint or geometry and keep looking for a safe
  recovery. Stay in this Skill.
- After each interior bind, state `interior_surface=far_wall` or
  `interior_surface=floor` before any planar holding-arm XY or descent.
  The only two legal interior surfaces are the container floor and the far
  inner wall. For an ashcan or trash can, do not use `far_wall`.
  Empty-ashcan `interior` must be the visible bottom disk with Z below
  0.01 m, or an already-placed in-can soda (no Z gate). If empty-bottom
  `interior` Z is greater than 0.01 m, that click is the side wall:
  re-bind the bottom. Do not start planar XY or release on that wall
  click. If the quadrilateral XY branch below is active, also state
  `align_mode=constraint_rect` and the current `x_lo`, `x_hi`, `y_lo`, and
  `y_hi`. Release only when the classified XY band below is satisfied and
  current visual evidence shows that the object's complete downward
  footprint is inside the opening, the fall path and rim clearance are
  safe, and the release height is appropriate for the visible geometry.
  When `align_mode=constraint_rect` is stated, that rectangle inequality
  is the classified XY band.
  For an elongated object such as a board, puzzle sheet, or box, both
  visible lengthwise endpoints of that downward projection and the
  gravity-aligned sweep volume must fit inside the opening. A grip point,
  centroid, visible narrow edge, tracked interior point, or any single pixel
  on the object does not authorize release. A large-looking opening does
  not authorize release while the classified XY band is missed.
- Report success only after a fresh observation shows the intended object
  settled inside the intended container, and `set_arm_to_grasp_position`
  has returned both arms to the grasp-prep pose with `gripper="open"`.
  That report is this Skill's only exit. It does not end the Codex
  session. Return to the parent task; if more requested objects remain,
  activate the next matching Skill.
- Before every Skill exit (success, cancel, or handoff to parent Second
  logic or `pick-up-object`), if the object-holding arm is no longer needed,
  call `set_arm_to_grasp_position` with `gripper="open"` on that arm.
  `arm="both"` is required. Do not
  start `move_to_reach_point` or spin the chassis with an
  outstretched unused arm. A small withdrawal to unocclude is not a stow.
  Keep this habit: when an arm's job is done, set-arm it and leave the
  workspace clean.
- If the object falls from the gripper, this Skill cannot finish placement
  while empty-handed. First stow any unused placement arm with
  `set_arm_to_grasp_position` as required above, then return to the parent
  task. The parent chooses the next unused can. A
  can left outside after a miss is not that next can unless the parent
  last-step far-can rule says so. Do not immediately re-pick this same
  object inside this Skill. Do not report placement success. Do not end
  the session from this Skill; after returning, the parent may end the
  session if its last-step check says every remaining outside can is
  under 0.5 m from the ashcan. Do not attempt an improvised regrasp in
  this Skill.
- If the ashcan interior is not visible, this Skill cannot finish
  placement. First stow any unused placement arm with
  `set_arm_to_grasp_position` as required above, then return to the parent
  task so the parent Second logic can stand the
  ashcan, then resume the parent pick/place loop. Do not improvise a
  stand-up procedure in this Skill.

Do not add hard action counts, alignment cycles, retry budgets, fixed
orientation tolerances, or fixed IK-residual thresholds. Do not add
universal descent-distance schedules. The 30 cm `near_rim` clearance
below is this Skill's hard height gate while the classified XY band is
not yet satisfied: the held object and the holding gripper/wrist must
stay at least 30 cm above current `near_rim` Z and must never go below
that. After the classified XY band is satisfied, the lower bound
becomes 0.20 m: descend into the 0.20–0.25 m `held_z - near_rim_z`
band. After that descent, small XY corrections stay at the current
height; do not lift back to the 30 cm planar-travel gate. Visually
check, then release. The classified XY bands in Align are this
Skill's planar-target gates, not optional hints.
A tool's `ok`, timeout, non-convergence, final position error, or
`ori_err_deg` is diagnostic information, not by itself a reason to stop.
In particular, EEF orientation error must never override safe measured
XYZ progress, current containment, grasp stability, or collision evidence.

## Establish current state

Determine the holding arm without replaying preparation actions that have
already completed. Wrist images are useful when current holding evidence is
ambiguous: red overlay pixels are visible non-robot 3D points inside that
gripper's grasp volume, and they support a grasp only when they overlap the
intended object's visible surface. Near-field occupancy may identify which
gripper is occupied, but it does not prove object identity. Obtain additional
head or wrist evidence only when needed to resolve a material ambiguity.

Do not call `set_arm_to_grasp_position` merely because placement started. Use
it when the current arm pose itself prevents a safe observation or reach,
including the one-shot out-of-view recovery below and the stow-before-reach
fallback in this Skill, and preserve the grasp. After that arm's job is done,
the exit stow above is required, not optional.
Prefer a model-visible post-action head image when it is current and usable.
Call `capture_head_camera` only when the latest result has no usable current
image or the scene has changed enough to make it unreliable. Do not capture
after every action by default.

Use visible image points for runtime geometry. BDDL/task labels such as
`ashcan 1` describe the goal but are not guaranteed runtime object names.

## Use persistent tracking

This Skill does not schedule when to bind or re-bind. The two XY branches
below say what a placement bind contains, not when to call
`track_object_distance`. Bind or re-bind whenever current geometry is
missing or untrustworthy.
If you suspect a click missed the intended surface or that a bind is
stale, look at a fresh clickable head image and re-click the visible
target. Do not bury yourself reconciling historical coordinates,
blending old and new tracks, or computing a click from prior images;
look at the current image.

Inspect `persistent_tracking` on the latest tool result. `track_object_distance`
is an atomic full-set replacement, not an append or read operation: a
successful call replaces all previous bindings. Once a point is bound, the
tracker keeps following that named surface in the live head RGB-D for as
long as the point remains in view. `persistent_tracking` after each action
is the only current measurement. After initialization, consume the updated
coordinates from `persistent_tracking` after every action.
`observation_sequence`, current `u`/`v`, and
`xyz_in_robot_base_coord_m` describe the current update.
`source_image_id`, `source_session_id`, and `source_observation_sequence`
are seed provenance only.

Coordinates recited more than about 10 turns ago are probably no longer
applicable, especially after the robot has moved. Prefer the latest
`persistent_tracking` when reading object and container geometry. If that
snapshot is untrustworthy, re-bind.

Both XY branches include:

- a stable visible patch on the held object
- a `near_rim` point on the visible rim closest to the holding arm and the
  planned travel path

`near_rim` is the measured height of that approaching edge. It is required
for this geometry, not an optional extra. Additional rim or corner anchors
beyond that approaching-edge point remain optional. The remaining interior
points belong to the chosen XY branch below. Do not use an interior Z as
rim height. Select broad visible material patches. Do not select
background, an image boundary, a depth discontinuity, the rim as the
interior, or a guessed continuation of an occluded surface.
The best placement viewpoint shows the container floor, or a can
already sitting inside the opening. Get that view before trusting an
`interior` bind. If the empty ashcan bottom is not visible, lean the
trunk forward with one `adjust_pitch` of `-10` degrees, then look
again. Do not command a larger pitch. Do not keep pitching once the
bottom disk or an in-can soda is visible. After each lean, re-bind
and keep `clearance_z` at least 0.30 m. Raise, lower, or step the
chassis only if that one lean is not enough. Bind `interior` on
that floor patch, with margin from every opening edge, or on the
in-can object. Those surfaces are much more accurate than a side
wall. The dark disk in a steep top-down ashcan opening is usually
the inner funnel wall, not the floor. Tracking a side wall as
`interior` often drives the held object into the trash can or dumps
it outside. Do not start planar XY or release on a wall click when a
floor or in-can bind is obtainable. `interior_surface=floor` only
when the click is the container bottom or an object resting there,
with Z down on that bottom — not merely below `near_rim`. A mid-bin
Z is a wall. Empty-ashcan floor bind: `interior` Z must be below
0.01 m. If `interior` Z is greater than 0.01 m, the click is the
side wall, not the floor. Re-bind on the visible bottom disk. Do
not start planar XY or release on that wall click. An already-placed
in-can soda as `interior` has no Z gate. `near_rim` Z must be
greater than 0.2 m. That is rim height, not `clearance_z`. If
`near_rim` Z is not greater than 0.2 m, re-bind on the rim band.
If it is below 0.01 m, the click is the room floor. Do not use that
floor click as rim height. If a previously placed soda is already inside the
ashcan, bind `interior` on that visible in-can soda; do not click
the empty funnel wall or a point outside the opening. If the only
usable empty-ashcan floor click sits near a boundary, still require
Z below 0.01 m and re-bind a more centered bottom disk. Do not
switch that click to the `far_wall` Align branch.

A bind UV is a model annotation and can miss the intended surface. After a
bind, if the holding EEF moves but that object's track (`u`/`v` and
`xyz_in_robot_base_coord_m`) stays essentially unchanged, that mismatch is
an annotation failure: the click did not land on the held object. That
geometry is untrustworthy; re-bind the object point on a broad visible
patch of the actual held object, keeping every other still-needed point in
the same replacement call. Do not keep moving the EEF as if the unchanged
track were the object.

If a placement motion accidentally carries the held object out of the current
head view, or the held object is not visible, call
`set_arm_to_grasp_position` once for the holding arm with
`gripper="keep"` so the object returns to a visible stow pose. Then bring
the held object into the head image: left holding arm
`adjust_left_eef_pose_in_head_frame` with `leftward=-0.1`; right holding
arm `adjust_right_eef_pose_in_head_frame` with `leftward=0.1`. Then
re-bind `track_object_distance` on that new image. Do not repeat set-arm
as a search. This is a sub-step recovery, not Skill exit.

Because replacement is atomic, include every point that must remain active
in the replacement call. Treat tracker confidence and `status="observed"`
as measurement quality, not semantic identity proof. If coordinates and
current visual evidence materially contradict each other, the geometry is
untrustworthy.

## Approach the container

Approach and placement are different stages. Do not apply the placement-stage
arm-XY-first rule while the container is still out of reach. If the
placement point is far away, use `adjust_chassis`, `adjust_height`, or
`adjust_pitch` to close the gap by hand, or use `move_to_reach_point` to
close the distance automatically.

`move_to_reach_point` is a straight-line chassis move toward the selected
image point. It keeps the current yaw and drives the shortest line to a
reach pose. It does not plan around obstacles. If a bed, chair, table, wall,
or any other object sits on that line, the command will stall or jam. A
visible container does not authorize the call. The blue path overlay is the
forward corridor for the current yaw. If the selected point is still off to
one side, that overlay is not the route the reach will drive; only after
`spin_to_facing_point` is the blue path the actual travel route. Before every
`move_to_reach_point`, call `spin_to_facing_point` with the current head
`image_id` and target `u`/`v`, reidentify the same target on the post-facing
image, and call the reach only when that blue path is clear for the full
chassis footprint from here to that reach pose. Do not use
`spin_to_facing_point` as a substitute for the reach; facing only sets yaw.

If the target container is visible in the current head image and the holding
arm cannot yet place over the opening, but the blue path is blocked, do
not call `move_to_reach_point` yet. First go around the obstacle until a
clear straight corridor exists:

- Use `move_chassis_to_floor_point` to a visible traversable floor point on
  the clear side of the obstacle, requiring the complete path and final
  chassis footprint to stay clear.
- Use `adjust_chassis` for a measured forward, backward, lateral, or spin
  correction that increases clearance and opens that straight corridor.
- Use `traverse-narrow-passages` only when a narrow legal passage itself must
  be crossed.

Those chassis tools are the detour, not the final reach. After the new image
shows a clear straight corridor, call `spin_to_facing_point` on that current
head `image_id` and a visible point on the container, inspect the blue path,
and call `move_to_reach_point` only when that overlay is clear. That call is
the reach onto the opening. A search turn is allowed to bring the container
into view; seeing the container is not enough if the blue path is still
blocked.

Prefer a stable visible point on the opening interior, the rim, or a nearby
staging point on the same container body. Inspect the actual
`move_to_reach_point` completion evidence and the post-action image. Never
reuse the pre-motion UV for later placement.

If the selected point is on the floor, a depth discontinuity, the wrong
surface, or otherwise contradicts the visible container, that geometry is
untrustworthy; re-bind a nearby visible point on the same container. Do not
treat a bad click as a reason to skip the reach. Retry the reach after the
straight path is still clear. Stay in this Skill.

If `move_to_reach_point` stalls on contact, treat that as a blocked straight
path, not Skill exit and not session end. Stop the current direction, back
out to a clear footprint, go around with `adjust_chassis` or
`move_chassis_to_floor_point`, then `spin_to_facing_point` again, and call
`move_to_reach_point` only after a new image shows a clear blue path.

After a successful reach, or after chassis / height / pitch adjustment has
put the container at a clearly reachable place pose, consume the post-action
image. The two XY branches below describe the placement bind; when to bind
or re-bind remains the model's decision.

## Align from current coordinates

This section applies only after the container is at a clearly reachable pose
for the holding arm. Arm XY first, with optional whole-body reach, is a
placement-stage rule. It is not the approach rule.

The robot-base axes are X forward, Y left, and Z up. For
`adjust_*_eef_pose_in_head_frame`, the `x`, `y`, and `z` arguments are
robot-base deltas. The `forward`, `leftward`, and `upward` arguments are in the
starting head-camera frame. Do not mix these parameter families in one call.

### Choose the XY branch

Judge the container floor by what the container is, not by whether the
current view looks narrow or occluded:

- Small floor: an ash can, trash can, basin, or bowl. Use the
  single-point XY branch.
- Large floor: a toolbox, toolkit, or drawer. Use the quadrilateral XY
  branch. A divider or one visible bay does not make it small.

If the container is not in those lists, treat a trash-can-like or
bowl-like opening as small, and a toolbox-like or drawer-like opening as
large. State `floor_size=small` or `floor_size=large` and the chosen
branch. Both branches share the held-object bind, `near_rim`, rim
clearance, and travel rules in this section. They differ only in the XY
target.

After every interior bind or re-bind, and before any planar holding-arm XY
or descent, output one line that names the bound interior surface. Prefer
the current bind or the latest `persistent_tracking` for `x_lo`, `x_hi`,
`y_lo`, `y_hi`, interior Z, and `near_rim` Z:

`interior_surface=far_wall`

or

`interior_surface=floor`

Include the current interior Z and `near_rim` Z in that same message.
`far_wall` means the far inner wall, the wall farther from the robot along
+X, with Z near the rim and not on the floor. Do not use `far_wall` on an
ashcan. `floor` means the container floor inside the opening. For an empty
ashcan that requires `interior` Z below 0.01 m, not merely below
`near_rim`. A mid-bin Z above 0.01 m is the side wall. An in-can soda as
`interior` has no Z gate. Those are the
only two interior surfaces the model can bind. The quadrilateral branch is
an XY rule on `floor`; it does not add a third interior surface. If that
branch is active, include `align_mode=constraint_rect` and the current
`x_lo`, `x_hi`, `y_lo`, and `y_hi` in the same message. If a click is not
clearly floor or far_wall, that geometry is untrustworthy. Do not start
planar XY or descent until that line has been stated for the current bind.

### Clear the approaching rim

Current `near_rim` Z is the edge-height gate for the first planar
approach, not a reason to climb back after descent. Compute
`clearance_z = held_z - near_rim_z`. While the classified XY band is
not yet satisfied, never go below 0.30 m. A few centimeters between a
tracked color patch and the rim is not enough; the gripper housing
sits lower and farther out than the patch and will strike the edge.
Do not use the interior-point Z as this gate. If `clearance_z` is
below 0.30 m, lift first with a vertical EEF adjustment or
`adjust_height`. Do not sweep the arm across the rim while raising.
Only after current tracking shows at least 0.30 m, compute the planar
residual and move XY. Keep consuming `near_rim` after every planar
step; it remains the live edge height, not a one-shot pre-check.
After XY is in band, descend; the lower bound becomes 0.20 m. Run the
final Z stage below; do not skip it. After that descent, correct XY
at the current height; do not lift back to the 30 cm planar-travel
gate for that small correction.

### Quadrilateral XY

Use this branch when `floor_size=large`. In the same atomic call as the
held-object point and `near_rim`, bind:

- `x1` and `x2`: two points on the intended container floor. Only their
  robot-base X values are used. They mark the near and far feasible
  range along +X
- `y1` and `y2`: two points on the intended container floor. Only their
  robot-base Y values are used. They mark the left and right feasible
  range along +Y

The unused coordinate of each constraint point is ignored. Each
constraint point must sit on the intended container floor, with Z
clearly below `near_rim` and not on the room floor. After the bind,
sort the live coordinates:

`x_lo = min(x1_x, x2_x)`

`x_hi = max(x1_x, x2_x)`

`y_lo = min(y1_y, y2_y)`

`y_hi = max(y1_y, y2_y)`

The four lines `x = x_lo`, `x = x_hi`, `y = y_lo`, and `y = y_hi` form
an axis-aligned rectangle. Before using it, confirm from the current
image that this entire rectangle is a subset of the intended container
interior. Inset the four clicks from walls, partitions, and the
approaching rim so the rectangle stays inside the intended bay. If the
rectangle spans two drawers, covers a divider, or includes room floor
or cabinet top, re-bind inset on the correct interior. If
`x_lo >= x_hi` or `y_lo >= y_hi`, re-bind.

The XY target is `x_lo < object_x < x_hi` and `y_lo < object_y < y_hi`.
The held object's current X and Y must lie strictly inside that
rectangle. While this branch is active, do not also require the
single-interior 2 cm XY band. A far floor point is one extent of the
rectangle, not a reason to replace it with a click nearer the object.

### Single-point XY

Use this branch when `floor_size=small`. In the same atomic call as the
held-object point and `near_rim`, bind one stable point visibly inside
the target opening. Prefer the container floor or an already-placed
in-can object. For an ashcan, do not bind the far inner wall. Change
viewpoint until the bottom disk or in-can soda is visible. For other
small containers, bind the far inner wall only when a floor or in-can
click is not obtainable. Do not use that Z as rim height.

Use object and container coordinates from the same current observation
state. Compute the signed planar difference:

`delta_x = container_x - object_x`

`delta_y = container_y - object_y`

The classified surface picks the XY target band. Chassis or EEF motion
may be used to enter the band; arrival is judged from current tracking,
not from the commanded step size.

- `interior_surface=far_wall`: the target is `delta_x` of −3 to −5 cm
  (`-0.05 m` to `-0.03 m`) and `|delta_y| ≤ 2 cm`. Do not release while
  `delta_x` is still positive. The held object must go a few centimeters
  past that far-wall point so it clears the near rim.
- `interior_surface=floor`: the target is `|delta_x| ≤ 2 cm` and
  `|delta_y| ≤ 2 cm`. No far-wall bias.

### Shared travel

While the classified XY band is not yet satisfied, keep
`clearance_z` at least 0.30 m while moving XY. Do not close the
remaining height until the classified XY band is satisfied. A visually
centered footprint does not replace the classified XY band. After
descent, correct remaining XY at the current height; do not lift
back to the 30 cm planar-travel gate for that small correction.

Move only the holding arm using the largest collision-free portion of the
current difference supported by visible clearance. A direct robot-base X/Y
EEF call is appropriate when the arm can make the motion. A point-to-point or
whole-body reach tool may also be used when its current tool contract, selected
points, holding-arm choice, and clearance fit the state; do not ban a suitable
tool merely because another recovery was previously preferred.

After every action, use measured post-action coordinates and the current image.
Never update position by adding the requested displacement to an old value,
reuse a stale residual, or issue an unmeasured sequence of corrections. If the
object made safe measured XYZ progress, recompute from the new state and
continue even when the controller reports timeout, non-convergence,
`ok=false`, orientation error, or a residual. If the object did not make useful
position progress, do not blindly repeat the same request. Reassess
reachability, clearance, contact, and the current target geometry.

When the holding arm is at a workspace limit, use an evidence-supported
chassis, body, or whole-body reach adjustment to restore reachability while
preserving the grasp. If a vertical EEF lift cannot yet make the 0.30 m
`near_rim` clearance, raise with `adjust_height` or another body adjustment
that does not sweep the held object across the rim. After recovery, observe
as needed, consume updated tracking, and
recompute the complete remaining vector. Continue while distinct safe actions
produce measured progress. Halt the current correction when a hard gate
above applies; then keep seeking a different safe recovery while remaining
in this Skill.

If planar `adjust_*_eef_*` shows the opening is still out of reach and the
next step is to fall back to `move_to_reach_point`, first call
`set_arm_to_grasp_position` with `arm="both"` and `gripper="keep"` so both
arms return to the grasp-prep stow. The same stow is
required before returning to the parent or any parent chassis reach after release, using
`gripper="open"` on the unused placement arm. Do not start that chassis reach with an
extended placement pose: the outstretched holding arm can knock the
container or nearby objects. After both arms are stowed, re-observe, call `spin_to_facing_point` on the
current target, then call `move_to_reach_point` only if the blue path on
that post-facing image is still clear.

## Align and release

Once the classified XY band is satisfied, descend. Do not release
from a high hover.

Compute `clearance_z = held_z - near_rim_z` from current tracking.
After XY is in band, the lower bound becomes 0.20 m. Adjust only Z
until `clearance_z` is in 0.20–0.25 m. Never go below 0.20 m. If
`clearance_z` is above 0.25 m, descend. If it is below 0.20 m, lift.
Use a vertical EEF call or `adjust_height`. A pure-Z command can
still change X/Y. After each Z change, consume current tracking and
recompute the classified XY residual.

After descent, small XY corrections stay at the current height:

- If the classified XY band is no longer satisfied, or the held
  object is clearly outside the opening, or the image shows the
  can sitting on the opening boundary or rim, immediately call
  `track_object_distance` on the current clickable head `image_id`
  and replace the full point set this Skill requires, including
  bottom `interior` and `near_rim`. Then correct XY at the current
  height; do not lift back to the 30 cm planar-travel gate for
  that small correction. A horizontal correction does not force
  another descent when `clearance_z` is already in 0.20–0.25 m.
  A drop that strikes the rim can tip the container and fail the
  parent task.
- If the image shows the held object aligned over the bottom
  `interior`, not the rim, the classified XY band is still
  satisfied, and `clearance_z` is in 0.20–0.25 m, release.

Open only the holding gripper when the latest usable post-action evidence shows
all of the following:

- the classified XY band is still satisfied
- the current image does not show the held object clearly outside the opening
  or sitting on the opening boundary or rim
- the object's estimated downward footprint fits inside the visible opening
  and is aligned over the bottom `interior`
- the gravity-aligned fall path enters the interior without striking the rim,
  wall, handle, or existing contents
- the gripper has clearance to open without contacting the container
- current object/interior geometry does not contradict the visual judgment
- `clearance_z` is in 0.20–0.25 m and never below 0.20 m

Tracked point equality is not enough for release: points do not encode the
object's footprint or the opening boundary. The classified XY band is still
required and cannot be waived because the opening looks large. A grip point,
centroid, visible narrow edge, or tracked interior point must not substitute
for the complete downward footprint or sweep volume of an elongated object.

## Release and verify

After `open_gripper`, inspect its current post-action image when usable;
otherwise obtain a fresh head image. Allow settling evidence to guide timing.
A mid-fall frame, a thin edge crossing the rim, or a centroid inside the
opening is not enough to report this object complete. Wait for a settled
observation of the complete object inside the container.
If the arm occludes the result and current clearance supports it, make a small
safe withdrawal with the former holding gripper open, then verify. That
withdrawal is only for a clear view. It is not the required stow. Once the
object is visibly settled inside, withdraw the arm to a safe pose if needed,
then call `set_arm_to_grasp_position` with `arm="both"` and `gripper="open"`
so both arms return to the grasp-prep pose. Do not exit this Skill until that
set-arm has been called. Then report this object's placement complete. That
exits this Skill only. Do not end the Codex session. Do not require
`reset_body` as placement cleanup.

If the object is visibly outside the container after release, do not claim
success. If it is still held, continue alignment and release from current
evidence. If it is no longer held, first stow the unused placement arm with
`set_arm_to_grasp_position` as required above, then return to the parent
task. The parent chooses the next unused can. A can left outside after
a miss is not that next can unless the parent last-step far-can rule
says so. Do not immediately activate `pick-up-object` for this same
object. Do not end the session from this Skill; the parent may end
after its last-step 0.5 m check.

Treat a timeout or a response with missing completion details as an unknown
post-action state. Do not blindly replay the command or release. First inspect
the returned image or obtain a current observation, consume updated tracking,
and compare measured coordinates with the previous state. Continue from that
observed state when the grasp is safe and actual progress occurred. A retryable
HTTP 504 is not by itself a transport failure. Keep inspecting the returned
or newly captured state and continue this Skill. An unreachable interface or
crashed simulator is the global session-end condition, not a placement exit.
