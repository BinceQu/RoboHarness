---
name: open-doors-and-drawers
description: Open a specified hinged, sliding, or pull-out closure using robot perception and manipulation, including room doors, cabinet doors, drawers, microwave doors, refrigerator doors, and balcony or glass sliding doors. Use when the robot must open a closed or partially open door or drawer to gain access, expose its contents, or pass through. Do not use for closing a door or drawer, or when the target is already sufficiently open for the requested task.
---

# Open doors and drawers

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
fresh head image and ground again; do not guess an image_id.
<!-- codex-coordinate-contract:end -->

<!-- codex-task-lifecycle:start -->
Follow this procedure only while `open-doors-and-drawers` is the active task Skill loaded by
`activate_skill`. Its presence in history, including after compaction, does
not make it active. After success and all required post-actions, return to the
parent task; activate another matching Skill only when its procedure is needed.
Instructions below to stay, retry, or exit describe pursuit of success and do
not prohibit a change of approach or handoff to the parent task.
<!-- codex-task-lifecycle:end -->

Use the currently advertised `behavior_v2` MCP tools and follow the baseline
Skill. This Skill opens the requested closure; it must never command a closing
motion. Successful completion requires a hinged door opened fully, as close to
90 degrees as the swing allows; or a long low TV stand drawer opened by
**TV stand drawer special case**.
Stopping a current rotation, bind, plan, or pull is a sub-step change, not
Skill exit and not session end.

## Entry and observations

- Enter only when the requested door, cabinet door, drawer, or sliding closure
  is closed or not yet open far enough for the parent task. If an already-open
  door occupies the approach to the next target, first complete **Clear
  around an opened door before the next**, then continue with staging and
  reach.
- Use a current head-camera image to identify the requested closure, its handle,
  and its mechanism. Reuse a model-visible post-action head image when it is
  current and usable. Call `capture_head_camera` only when no such image is
  available or the relevant geometry is ambiguous.
- For a hinged closure, visually determine whether its hinge is on the left or
  right before grasp planning. Do not infer the hinge side from memory or from
  the handle side alone. If the hinge cannot be identified, obtain a better
  observation instead of guessing.
- A drawer or sliding door has no hinge side. Never invent one. A long low
  TV stand drawer uses only **TV stand drawer special case** after reach. It
  must not enter **Plan and grasp the handle**, **Retract before a new
  grasp plan**, **Probe the grasp and open the hinged door**, or **Clear
  around an opened door before the next**. Those four sections are the
  hinged-door path. The numeric pull loop is only for hinged closures.

## Establish a frontal staging pose

Establish a safe frontal staging pose before reaching for the handle.

From each current head image used for approach, state the facing judgment in
words: facing, or not facing, plus a short note about the panel plane.
Judge facing from the movable panel's plane. Treat the movable panel like a
poster on a wall. Facing means the top edge and the bottom edge of that
panel look level (horizontal). Do not judge facing from the left or right
side edges, their lengths, or their depths. A large visible top surface
after lowering the trunk does not mean not facing.

Not facing means the top edge or the bottom edge is clearly tilted. Visible
handles, or the cabinet sitting in the middle of the image, help find the
panel; facing is the panel-plane look.
If that evidence is mixed, say not facing.

`move_to_reach_point` keeps the current chassis yaw and approaches the
selected point in a straight line, so the panel's orientation stays
whatever it already is. It does not weave around furniture. Any obstacle
on that line will stall or jam the robot. Call it only from a pose whose
straight corridor to the selected point is clear.

When the stated judgment is facing and the robot is already safely in front
of the panel, continue to **Set trunk height after facing**, then
**Move into reach**.

When the stated judgment is not facing, finish this section first. When the
panel is still seen at a large inclination, first get farther from the panel
so there is room to turn, then rotate until the panel plane faces the robot,
then state the facing judgment again from the latest image. After that later
judgment is facing, continue to **Set trunk height after facing**, then
**Move into reach**.

1. From the current head image, select a visible traversable floor point in the
   frontal approach region of the target panel rather than far off to either
   side. Require the complete direct path and final chassis footprint to remain
   clear, with enough visible space for the later rotation and opening motion.
   If the current view is strongly oblique, prefer a farther supported floor
   point that increases standoff from the panel. If no such floor point is
   safely supported, do not force this approach.
2. Call `move_chassis_to_floor_point` with that current image and floor point.
   The move preserves yaw, so reaching the floor point alone does not prove
   that the robot faces the panel. Inspect its returned post-action head image;
   never continue from the pre-motion image.
3. If the new view is still strongly oblique, first increase standoff from the
   panel when the chassis is too close for a clear turn, then use a
   rotation-only `adjust_chassis` call: set `spin` and omit `forward` and
   `translation`. Choose its sign and magnitude from the current panel
   geometry, the clear rotation sweep, and measured tool feedback. Inspect each
   returned image. When the top and bottom edges of the panel look level,
   the base-forward direction is approximately normal to the target panel
   plane; continue to the next step. Do not use a fixed
   yaw target, spin amount, or retry count.
4. State the facing judgment from the latest post-rotation image. When that
   judgment is facing, reidentify the target panel, hinge when applicable, and
   handle in that image. Only then continue to **Move into reach** from that
   current observation; never reuse pre-staging or pre-rotation coordinates.
   Complete **Set trunk height after facing** before any `move_to_reach_point`.

## Set trunk height after facing

Complete this section after the latest stated judgment is facing and before
any `move_to_reach_point` on this panel. Classify the target from the current
image, not from task memory.

- Floor-standing closures: a cabinet, TV stand, washer, or similar body that
  sits on the floor, whose door or drawer is at lower-body height. Call
  `adjust_height` with a negative `upward` large enough to reach the lowest
  reachable body height. Out-of-range requests clamp to that lowest row. If
  the trunk is already at that lowest reachable height, skip the call.
- High closures: a wall cabinet, overhead cabinet, or other door whose
  handle sits well above standing torso height. Call `reset_body` with
  `keep_ori_arm="none"` (or omit that argument). `reset_body` is a
  trunk/posture tool, not an episode reset. A parent prompt such as
  "Do not reset" forbids resetting the episode; it does not forbid this
  posture. If the trunk is already upright, skip the call.

Inspect the returned head image. Re-state the facing judgment from that
image. When it is not facing, return to **Establish a frontal staging
pose**. When it is facing, continue to **Move into reach** from that
current observation; never reuse pre-height coordinates.

## Move into reach

This section is for a pose whose latest stated judgment is facing.
Do not call `move_to_reach_point` until **Set trunk height after facing**
has been completed for this panel.

1. State the facing judgment from the current head image, with a short
   panel-plane note. When that judgment is not facing, return to
   **Establish a frontal staging pose** and finish that section first.
2. In the current head image, select a visible point on the closure surface
   immediately above its handle. The point must lie on a depth-bearing physical
   surface, not in open air, on background beyond the closure, or on the handle
   itself.
3. Call `move_to_reach_point` with that current `image_id`, `u`, and `v` only
   when the current image shows a clear straight path to that point. Omit
   `reach`, `nav_timeout_s`, and `keep_ori_arm` unless current evidence requires
   a non-default value.
4. Inspect `shoulder_distance_estimate` in the completed call before grasp
   planning. Treat the pose as reach-ready only when at least one finite
   `left_m` or `right_m` value is strictly less than `0.7 m`. This is a
   distance gate, not an instruction to choose that arm. A passing distance
   gate still requires a facing judgment from the post-action image. After
   this move, state facing again from that image using the panel-plane look.
5. If neither shoulder passes the gate, or the current post-action image shows
   an awkward approach or pull pose, reposition before planning. When that
   image still shows the panel plane at a large inclination, return to
   **Establish a frontal staging pose**, get farther from the panel, rotate
   until the panel plane faces the robot, and then call `move_to_reach_point`.
   When the panel plane already faces the robot, from current geometry,
   clearance, and tool feedback, either rerun `move_to_reach_point` on a newly
   selected valid closure-surface point or use `adjust_chassis` for an
   appropriate forward, backward, or lateral correction. Choose the
   direction and magnitude from the scene; do not use a fixed correction,
   retry sequence, or arbitrary exact shoulder-distance target.
6. Inspect every repositioning result and use its post-action head image when
   usable. After an `adjust_chassis` correction, do not reuse the old image or
   shoulder estimate; use the returned current image to obtain a fresh
   `move_to_reach_point` result before planning. Once the current move result
   passes the distance gate and the latest facing judgment is facing,
   reidentify the closure, handle, and, for a hinged closure, the hinge side.
   Never plan from a pre-motion image. When the current image is a long low
   TV stand drawer and that current move result already passes the distance
   gate, go to **TV stand drawer special case** and stay there. Do not
   continue to **Plan and grasp the handle** or **Probe the grasp and open
   the hinged door**.

## Ground the physical handle

A long low TV stand drawer that already passed reach does not use this
section. Go to **TV stand drawer special case**. This eight-point bind is
for the hinged-door path.

Before selecting any grasp point:

1. Classify the target closure mechanism from current visual evidence before
   evaluating handle candidates:
   - A hinged door rotates about a visible or geometrically supported hinge
     axis. Its handle may be a vertical bar, horizontal pull, recessed pull, or
     knob on either side.
   - A drawer translates outward and has no hinge. Its pull or knob may be
     centered or offset on the drawer front.
   - A sliding or glass door translates along a track. It may have a recessed
     pull, projecting handle, or frame-integrated grip; a panel stile alone is
     not a handle.
   If the mechanism is ambiguous, obtain a clearer current observation instead
   of using handle orientation or position to guess the mechanism.
2. Identify the target movable panel and trace its visible boundary. A valid
   handle candidate must belong to that panel, not to a fixed frame, control
   panel, cabinet face, or neighboring closure.
3. Accept a candidate only with positive visual evidence of a bounded
   manipulable component attached to that panel: a projecting bar or knob, a
   bounded recessed cavity or lip with visible depth-bearing grasp structure,
   or another finite clampable body. Bounded extent, distinct surface geometry,
   local attachment, clearance, or depth separation may support identity; no
   single cue is mandatory. Color or expected layout alone is insufficient.
4. Use no universal orientation or side prior. Determine each candidate's
   orientation, visible extent, and attachment from the current image. A valid
   candidate may be vertical, horizontal, recessed, frame-integrated, a knob,
   or near any side.
5. Reject panel borders, seams, decorative trim, continuous frame stiles,
   glass, reflections, hinges, buttons, controls, and other closures' handles;
   do not grasp attachment endpoints or corners.
   Treat a boundary-following strip as a seam, lip, trim, or stile only when
   current evidence supports it as continuous structure without a bounded
   manipulable body. Boundary alignment or missing visible clearance alone is
   not enough to accept or reject it.
6. Treat the selected point as the planner's intended gripper-center target,
   not merely a semantic label for the handle. The camera ray through that
   point must terminate on solid, graspable handle material that the closed
   gripper can clamp. Never select a handle opening, through-hole, empty gap,
   or other negative space, including background visible through the handle.
   Do not select the floor or back wall of a recessed cavity unless that
   surface is itself the solid handle body to be clamped. For a recessed or
   loop handle, select a reachable solid lip, bar, or clamping surface rather
   than the empty center. Select near the center of that exposed solid surface,
   away from endpoints and the door-frame seam. If the handle body remains
   indistinguishable from these confusers or no solid grasp point is visible,
   obtain a clearer observation; never guess or plan a nearby point. Stay in
   this Skill.
7. When a recessed or loop handle borders negative space and a single RGB point
   cannot be separated confidently from its aperture, validate candidates with
   measured depth before grasp planning. This procedure applies only after the
   bounded component has positive visual evidence of being a handle. It is
   forbidden for a circular appliance-door rim, aperture, bezel, or panel; use
   the **Circular appliance door special case** instead. First inspect
   `persistent_tracking`:
   `track_object_distance` is an atomic full-set replacement. Do not destroy
   active bindings required by the parent task; if such bindings exist and RGB
   evidence is insufficient, do not replace those bindings. Obtain a clearer
   observation instead. Otherwise, in one
   `track_object_distance` call bind at most eight points on the same current
   image: up to six distinct solid-handle candidates distributed along the
   bounded lip, ledge, or bar; one deliberate aperture/recess-interior reference;
   and one broad movable-panel reference. The two reference points are
   measurement controls and will never be used as a grasp target. Do not cluster
   all candidates on one ambiguous patch or reuse the aperture center as a
   nominal handle candidate.

   Reject any nominal handle candidate whose measured depth clusters with the
   aperture/recess reference or lies at a depth discontinuity into that negative
   space. Depth never proves handle identity: a surviving point must still be
   visibly inside the same bounded, clampable handle component rather than the
   panel, ring, frame, seam, or trim. Prefer a central candidate with visible
   material margin. Re-locate that candidate in the latest clickable head image
   after the measurement, then pass its current `image_id`, `u`, and `v`
   relative-coordinate tuple to grasp planning. Do not reuse the
   pre-measurement image tuple when
   tracking has returned a newer image. If no candidate passes both visual identity and
   measured-depth checks, obtain a clearer observation and ground again.
   Do not rebind repeatedly or scatter new measured points after a failed set.
   Stay in this Skill.
8. Use integer relative coordinates in the latest clickable head image:
   `u` and `v` range from 0 through 1000. Use the full image margins, never
   coordinates relative to the appliance, target panel, handle, or a crop.
   Treat `image_id`, `u`,
   and `v` as one inseparable observation tuple. Pass the exact selected
   `image_id`; never combine coordinates from different images.
9. If the handle remains indistinguishable from the frame, obtain a clearer
   current observation and ground it again. Do not probe nearby planner points.
   After failure, do not scatter new points along the same presumed feature
   without first re-verifying it as the handle.

## Microwave special case

Apply this section only after the current image independently establishes this
microwave layout: a windowed movable door on the left and a fixed control panel
on the right. Other microwave layouts use only the general grounding rules
above.

1. Treat the narrow, bounded vertical bar attached to the door's right edge,
   immediately left of the fixed control panel, as the handle candidate. It
   must still satisfy movable-panel membership and positive grasp evidence.
2. Use finite top and bottom extent, its own visible body thickness or shading,
   and localized attachment to the movable door as positive evidence. One RGB
   view does not need to expose an open-air gap behind the bar. Do not reject
   that resolved bar solely because its rear gap is not visible.
3. Reject the pale control panel, circular controls, red buttons, straight
   door/control seam, window frame, reflections, and robot parts. The long dark
   horizontal strip along the window is trim, not the handle.
4. When unobstructed, select inside the middle third of the visible bar body,
   away from its endpoints and the adjacent seam. If an arm or gripper hides
   any of that central region, obtain a new observation instead of inferring
   a point.

Never transfer these asset-specific cues to other doors, drawers, or microwave
layouts. Without the verified entry layout, do not infer that a handle is
vertical or on the right.

## Circular appliance door special case

Apply this section when the current image shows a front-loading washing-machine
or dryer porthole door: one large circular door or window set into the appliance
front. The circular shape alone does not identify which ring surface moves with
the door.

1. Do not classify the circular rim, window aperture, bezel, or surrounding
   panel as a recessed or loop handle merely because they form concentric depth
   layers. Do not call `track_object_distance` on rim/aperture/panel candidates
   for this mechanism. Those measurements separate surfaces by depth but cannot
   prove movable-door membership, an undercut, or gripper clearance.
2. Require positive current-image evidence of a finite handle or latch attached
   to the movable door: for example a projecting tab or bar, or a bounded solid
   lip with a visible undercut and enough material for the closed gripper to
   clamp. Determine its location from the image; do not assume a side from the
   appliance category. The ring itself is not a handle by default.
3. After the handle is identified, follow **Plan and grasp the handle** and
   **Probe the grasp and open the hinged door**. An empty-looking close or
   missing bilateral contact is not a reason to skip the EEF probe. If the
   arm hides the mechanism before planning, move that arm to its grasp-ready
   observation pose once; if the viewpoint still cannot establish a finite
   handle, make one clearance-supported viewpoint correction and use its new
   head image. Re-ground a different solid component only from that changed
   observation.
4. If no positively identified clampable handle or latch is visible, continue
   changing the observation rather than planning on the circular ring or using
   depth tracking as a substitute for handle identity. Stay in this Skill.

## TV stand drawer special case

<!-- 电视柜抽屉：独立分支，不进铰链门主路径 -->
This is a separate branch, not a variant of the hinged-door path. After
**Set trunk height after facing** and **Move into reach**, a long low TV
stand drawer stays in this section until the drawer is open or this Skill
exits. Do not enter **Plan and grasp the handle**, **Retract before a new
grasp plan**, **Probe the grasp and open the hinged door**, or **Clear
around an opened door before the next**.

Apply this section only when the current image shows a long low TV stand
(media console) pull-out drawer, not a coffee table, wall cabinet, washer,
microwave, or hinged door. Open only the requested drawer. Do not open the
other drawer or the glass door. Do not close any drawer.

Do not call `plan_grasp_point_filter_rgbd_lite` on this mechanism. Do not
call wrist-frame `adjust_*_eef_pose_in_wrist_frame` with `forward=-0.02`.
After a passing `shoulder_distance_estimate` (`left_m` or `right_m`
strictly below `0.7 m`) from the current `move_to_reach_point`, do not
call `measure_shoulder_distance` and do not treat a chassis-forward
warning on the TV stand face as a reason to reverse or to skip the next
step. Do not use the eight-point recessed-handle bind in **Ground the
physical handle**.

<!-- 电视柜把手：柜面最上沿那条横向细条，不是柜面黑斑或旋钮 -->
The handle is the thin horizontal strip along the top edge of the
requested drawer front, just under the stand's top overhang. It runs
the width of that drawer as one long solid bar.

Then do these in order.

1. In the current head image, call `track_object_distance` once on the
   solid bar body near the center of that top strip. Name that point
   `handle`.
2. Call `move_tracked_point` with `execution_mode=plan` and this input:
   C1, vertical pinch through `handle` (same x,y):
     `left_finger_tip` on_hand `x,y,a`
     `handle` off_hand `x,y,z`
     `right_finger_tip` on_hand `x,y,b`
   C2, gripper slide at the same height as `handle` (same z):
     `handle` off_hand `x,y,z`
     `gripper_slide_center` on_hand `c,d,z`
   Use this input as written. If the plan fails or the red overlay misses
   `handle`, recapture, bind `handle` again, and call again with this
   same input.
3. Call `exec_plan_pose` with that `plan_id` and `back_m=0.1`. If this
   call fails, `open_gripper` if that gripper is closed, then
   `set_arm_to_grasp_position` for that arm with `gripper="open"` (or
   `arm="both"` if the other arm is also out), then return to step 2
   with the same input.
4. Call `close_gripper` on that same arm.
5. Call `adjust_chassis` with `forward=-0.3` and look at the returned head
   image. If the drawer is open: `open_gripper` on that arm, then
   `set_arm_to_grasp_position` with `arm="both"` and `gripper="open"`, then
   leave this Skill. If the drawer is still closed, `open_gripper` if
   needed, then `set_arm_to_grasp_position` with `arm="both"` and
   `gripper="open"`, then repeat from step 1 with the same input.

## Plan and grasp the handle

This section is the hinged-door path. A long low TV stand drawer must not
enter this section. Use **TV stand drawer special case** instead.

1. After the latest facing judgment is facing, `move_to_reach_point` has
   passed the shoulder gate, and the handle is grounded, bind one pull-witness
   point on solid handle material with `track_object_distance` on the current
   image. Name that point so it can be reread after each pull. This bind is
   the motion witness, not a second grasp target; planning still uses the
   grounded handle body, which may be the same point. `track_object_distance`
   is an atomic full-set replacement. Do not destroy active bindings required
   by the parent task; if such bindings exist, obtain a clearer observation
   instead of replacing them. On a circular appliance door, do not bind rim,
   aperture, bezel, or panel points; only a positively identified handle may
   be bound.
2. After completing the handle-grounding checks above, select a visible grasp
   point on the handle in the latest usable head image and call
   `plan_grasp_point_filter_rgbd_lite` with that exact image and point.
   Always omit the `plan_arm` argument and let the planner select whichever arm
   has a collision-safe reachable grasp. Do not constrain the grasp-planning
   arm from the hinge side, handle side, closure type, or a prior arm
   assumption. The hinge side controls the later pull direction only, never
   the grasp-planning arm. Do not pass the literal default value either.
3. Inspect the returned plan and grasp preview. Continue only when the plan is
   successful and the preview places the selected gripper on the intended
   handle without an evident collision.
4. Call `exec_plan_pose` with the returned `plan_id`. This tool moves the EEF
   but does not close the gripper.
5. After `exec_plan_pose`, inspect the returned wrist image. Do not call a
   wrist camera just to replace that image. Red overlay points are visible
   non-robot 3D points inside that gripper's grasp volume, between the two
   fingers. Judge whether the grasp pose is good enough to close from that
   overlay: both fingers should straddle only the movable handle, and
   judge whether handle material is visibly between the two fingers. The
   red must overlap the handle alone. The adjacent sidewall, fixed
   cabinet, frame, seam, or a neighboring closure must not have red; if
   they do, closing will clamp the handle and that fixed surface together
   and the door or drawer cannot open. Neither finger may rest on the
   fixed cabinet, frame, sidewall, seam, or a neighboring closure.
   Record that look; it does not decide whether to pull. Do not close yet.
6. After that inspect, first call `open_gripper` once on the same arm so
   the fingers are fully open. Then call that arm's
   `adjust_left_eef_pose_in_wrist_frame` or
   `adjust_right_eef_pose_in_wrist_frame` with only `forward=-0.02`.
   Inspect the new wrist overlay and judge whether the red left the
   sidewall or otherwise changed. Then call that same tool once more with
   only `forward=-0.02` and inspect the newest wrist image. Do not use
   `forward=-0.01`. Do not use a positive `forward` to seat deeper. Inspect
   again and close only when the red is on the handle alone. If the red still covers the
   handle and the sidewall after those two 2 cm retracts, do not close;
   Translation may use `forward`, `leftward`, and `upward` (positive or
   negative). Prefer translation-only calls first. If rotation is needed,
   change only one of `roll`, `pitch`, or `yaw` by `10` or `-10` degrees per call;
   do not use a larger rotation step. Rotation moves the
   wrist-frame axes, so it also changes later translations: inspect the
   new wrist image after each rotation before translating again, and do
   not combine a rotation with a translation in the same call. If several
   later adjusts still cannot isolate the handle, do not close; follow
   **Retract before a new grasp plan**, then return to **Ground the
   physical handle** and replan. Do not skip the `open_gripper` or the
   two `forward=-0.02` retracts after `exec_plan_pose`.
7. Use the arm selected by the successful plan as the holding arm. Call
   `close_gripper` for exactly that arm. Missing bilateral contact, an
   empty-looking close, or an uncertain wrist view is not a reason to skip
   the pull probe below. Stay in this Skill.

## Retract before a new grasp plan

Use this section when an already-executed grasp is abandoned and a new
`plan_grasp_point_filter_rgbd_lite` is needed: the wrist pose cannot be
corrected, the close missed, the probe did not move the handle, or the
chosen arm or point is being replaced. Replaying the current `plan_id`
after `open_gripper` is not this section.

1. If that gripper is closed, call `open_gripper` on the unsuccessful arm.
2. Call `set_arm_to_grasp_position` for that unsuccessful arm with
   `gripper="open"` so it leaves the handle. If the other arm is also
   extended near the same handle, panel, or a neighboring point, call
   `set_arm_to_grasp_position` with `arm="both"` and `gripper="open"`
   instead. Do not leave an arm parked at the target.
3. Only after that returned head image, re-ground the handle and call
   `plan_grasp_point_filter_rgbd_lite`. Never plan or `exec_plan_pose`
   for a second arm onto a nearby point while the first arm is still at
   the handle or panel.
4. Only one arm may occupy the handle region at a time.

## Probe the grasp and open the hinged door

The grasp test is whether the tracked handle point moves with a
handle-opening EEF pull. Close telemetry does not decide this.

1. First probe without moving the chassis. Call the holding arm's
   `adjust_left_eef_pose_in_head_frame` or
   `adjust_right_eef_pose_in_head_frame` with a pull away from the
   panel toward the robot. When the robot is facing the panel, use about
   `forward=-0.2` in the head-camera frame. Omit chassis motion on this
   probe. Do not rotate the chassis.
2. Read the tracked handle point. If its measured position moved in the same
   sense as that EEF pull, by a comparable amount rather than noise, the
   handle is held. Then open the door by coordinating further same-arm EEF
   pulls with `adjust_chassis`. Call `adjust_chassis` with a bounded combined
   robot-frame translation and omit `spin`:
   - Left-side hinge: use `forward=-0.1` and `translation=0.1` meters.
   - Right-side hinge: use `forward=-0.1` and `translation=-0.1` meters.
   The tool combines these two components into one same-frame two-dimensional
   chassis translation. Do not exceed `0.1 m` in magnitude on either chassis
   axis and never rotate the chassis while the handle is grasped. After each
   coordinated step, confirm the handle point still moved with the motion.
   If the full chassis step would no longer follow the observed door arc, use
   a smaller `forward` and/or `translation` magnitude on the next
   `adjust_chassis` call while preserving the hinge-specific signs above.
   Never reverse either sign merely to force progress. A first successful
   probe or a small opening is not completion. Keep repeating coordinated
   same-arm EEF pulls and `adjust_chassis` steps along the door arc. Those
   two tools together can carry the door through the full swing; do not
   leave the handle after the panel has only started to move.
3. If the handle point did not move with the probe, the gripper did not take
   the handle. Call `open_gripper` on that arm, then `exec_plan_pose` again
   on the current plan, then repeat the same post-exec sequence: call
   `open_gripper` once, then `forward=-0.02`, inspect the wrist image,
   then `forward=-0.02` again and inspect, then call `close_gripper` twice
   in a row on that same arm. Then repeat the chassis-still EEF probe and
   again judge by handle-point motion. Stay in this Skill.
4. Leave this Skill only after a current observation shows that the
   hinged closure is fully open, as close to 90 degrees as the swing
   allows. Report success without commanding any closing motion. Do not
   open the gripper or otherwise release the handle just because the door
   started to move. When that full-open observation is current, end this
   opening: if a gripper still holds the handle, call `open_gripper` on
   that arm, then call `set_arm_to_grasp_position` with `arm="both"` and
   `gripper="open"`. Then leave this Skill. If another door or leaf still
   needs opening, stay in this Skill and continue with **Clear around an
   opened door before the next**.

## Clear around an opened door before the next

Use this section after the current hinged door is fully open and another
door or leaf still needs opening, or when this Skill is entered while an
already-open door occupies the approach to the next target.

1. If a gripper still holds the opened door, call `open_gripper` on that
   arm, then continue with the retreat.
2. Call `adjust_chassis` with `forward=-0.5` meters and omit `spin` and
   `translation`. This is a straight retreat away from the opened door.
3. Call `set_arm_to_grasp_position` with `arm="both"` and `gripper="open"`.
4. In the current head image, select a visible traversable floor point
   that routes around the opened door so the complete direct path and
   final chassis footprint remain clear of that open leaf and its swing.
5. Call `move_chassis_to_floor_point` with that current image and floor
   point. Inspect the returned post-action head image and continue from
   that current image.
6. If the opened door or any other obstacle still blocks the straight
   corridor between the robot and the next target panel, use
   `adjust_chassis` until that corridor is clear. Choose the direction
   and magnitude from the scene.
7. Then restart this Skill from **Establish a frontal staging pose** for
   the next door. Re-state facing on that next panel. Use only
   coordinates from the latest post-clearance image.

## Failures

- A long low TV stand drawer stays in **TV stand drawer special case**.
  If `exec_plan_pose` fails or a pull leaves the drawer closed, first
  `set_arm_to_grasp_position`, then call again with the same input.
- On an explicit unsafe result, halt that pull or plan, reobserve, and
  continue this Skill. Do not end the session.
- When a motion reports success but omits completion details, do not treat that
  omission as completion or failure. Inspect a usable returned image or call
  `capture_head_camera`, then continue from the observed state.
- If the EEF probe or a later coordinated step produces no matching
  handle-point motion, use the miss path above: open the gripper, re-exec the
  current plan, then `open_gripper` once and two `forward=-0.02` wrist
  retracts with an inspect after each, close twice, and probe again. Do not
  skip the probe because the close
  looked uncertain. If that same-plan retry is abandoned for a new point or
  a different arm, follow **Retract before a new grasp plan** before the
  next `plan_grasp_point_filter_rgbd_lite`.
- Never leave one arm at the handle while the other plans or executes a
  nearby grasp. Retract the unsuccessful arm with `set_arm_to_grasp_position`
  first. Only one arm may occupy the handle region at a time.
- If the door has opened only a little, that is not done. Keep coordinating
  EEF and chassis along the observed arc until the panel is fully open,
  as close to 90 degrees as the swing allows. Do not release the handle
  after a small opening.
- After one door is fully open and another still needs opening, continue
  with **Clear around an opened door before the next**: retreat, stow both
  arms, drive around the opened door, then restart staging.
- If grasp planning fails or reports the handle as unreachable before contact,
  first reassess current shoulder distances and approach geometry. When the
  panel is still seen at a large inclination, recover facing from the panel
  plane first, then resume reach. When the evidence indicates a reach or pose
  problem while the panel plane already faces the robot, return to **Move into
  reach** and reposition instead of scattering nearby grasp points or repeating
  the same plan from the same pose.
