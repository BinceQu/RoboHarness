---
name: pick-up-object
description: Pick up, grasp, collect, or retrieve a visible object with an empty robot gripper. Use for picking stages of pick-and-place tasks, including picking up toys, games, puzzles, balls, household objects, or multiple objects before placing them elsewhere. Do not use when the requested object is already held or for the placement-only stage.
---

# Pick up object

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
Follow this procedure only while `pick-up-object` is the active task
Skill in the latest lifecycle result or current `task_skill_state` notice.
Its presence in history, including after compaction, does not activate it.
After success and all required post-actions, call `deactivate_skill` with
`name="pick-up-object"`, then continue the parent task under the baseline.
You may also deactivate to cancel or hand control back without claiming success.
Instructions below to stay, retry, or exit describe pursuit of success; they do
not prohibit explicit cancellation or handoff. Carry unfinished work and any
pending cleanup back to the parent task. Physical constraints still apply;
deactivation itself does not move the robot or release a grasp.
<!-- claude-task-lifecycle:end -->

Use the currently advertised `behavior-v2` MCP tools and follow the baseline
Skill. Treat this Skill as a picking subtask. Successful completion requires a visually
confirmed grasp of the requested object, followed by the size-specific
post-grasp stow in step 7. That stow is `set_arm_to_grasp_position` then
`reset_body` for a small object, or `reset_body` only for a large object.
Stopping a current approach, plan, or close is a sub-step change, not Skill
exit and not session end.

## Observe and select an arm

1. Call `capture_head_camera` and identify a visible point on the requested
   object's graspable body. Do not select nearby support surfaces, packaging,
   background, or another instance. For a hollow box, tray, or crate, click a
   solid visible rim or wall edge, not the empty interior or the floor inside:
   an interior click puts the gripper in overlap with the box walls and
   `plan_grasp_point_filter_rgbd_lite` fails.
2. Call `capture_left_wrist_camera`, then
   `capture_right_wrist_camera`. Start a new pick only when the intended
   gripper is visibly empty. A current wrist image with a red grasp-volume
   overlay means that gripper still holds an object: do not start a new pick
   on it and do not call `set_arm_to_grasp_position` with `gripper="open"`
   for that arm. If a gripper already holds the requested object,
   the pick subtask is already complete: confirm the grasp on current
   evidence, perform the required stow if not yet done, then exit this Skill.
   If a gripper holds a different object, do not open it; pick the requested
   object with the remaining empty gripper, or keep observing until an empty
   gripper can pick it. Do not end the session.
3. When the target is not already at a clearly reachable pose, face it
   before the chassis reach. First call `spin_to_facing_point` with the
   current head `image_id` and the target `u`/`v`. That call only yaws the
   chassis so the point is centered; it does not approach. Then, on the
   post-facing head image, reidentify the same target and inspect the blue
   path overlay. `move_to_reach_point` is a straight-line chassis move; it
   does not weave. After facing, read `chassis_forward_2m` and look at the
   marked nearest point in the current image. If that object is the target,
   it is fine. If it is anything else (table, sofa, wall), do not call
   `move_to_reach_point`. Clear it with `adjust_chassis` (translation or
   spin) by hand, or `move_chassis_to_floor_point` to a safe floor point.
   Then face again and reach. Call `move_to_reach_point` only when the line
   is clear. Inspect its completion image. Never reuse pre-motion click
   coordinates for grasp planning.
4. From the latest usable head image, call `measure_shoulder_distance` with the
   visible target point when arm reachability or arm choice is uncertain. Pick
   the arm supported by the measured shoulder geometry and a visibly clear
   approach. Do not choose an arm from task history or image side alone.
5. If furniture or another obstacle blocks a direct approach, a change of
   viewpoint or side is not by itself successful approach. After any detour,
   reacquire a fresh head image and call `measure_shoulder_distance` on the
   same visible target. Count the detour as useful approach only when the
   fresh measured shoulder distance is materially smaller than the
   pre-detour measurement for the intended arm, and the remaining chassis
   footprint corridor to that target is visibly clear. If either check fails,
   do not claim the target is closer and do not call the grasp planner with
   the same unreachable geometry. Choose a different clear approach or a
   different reachable instance of the requested object. Do not reuse the
   same unreachable geometry.

## Prepare before grasp planning

1. Before every first grasp-planning call for the current target, call
   `set_arm_to_grasp_position` for the selected arm with `gripper="open"`.
   This is a required precondition, not an optional recovery after planning
   fails. If both wrists are confirmed empty and the planner must choose the
   arm, call it with `arm="both"`; otherwise prepare only the selected arm.
2. Inspect the set-arm result and its post-action image. If no usable current
   head image is returned, call `capture_head_camera`. Reidentify the target
   after the arm motion and use only this post-set-arm `image_id` and point for
   grasp planning. The adapter normally labels this precision view
   `role=raw_rgb`; locate the physical object itself and ignore navigation
   lines, distance text, and coordinates from earlier images.
3. Call `plan_grasp_point_filter_rgbd_lite` once for that current point.
   Pass the same selected arm as `plan_arm`; use `plan_arm="any"` only when both
   arms were explicitly prepared and remain visibly clear. Inspect the returned
   grasp preview and require that the planned fingers enclose the intended
   object rather than its support or a neighbor. Do not reuse the same `u`/`v`
   after a failed plan.    If the plan says the point is too far to plan, do not
   jog the EEF or `close_gripper`: call `adjust_pitch` or `adjust_chassis` to
   close range, or call `move_to_reach_point` again on a fresh faced head
   image and target, then recapture, reidentify, and plan again. A manual
   `adjust_pitch` with a negative degree can reach a lower torso pitch than
   `move_to_reach_point`; that extra lean is often useful for a floor object.
   A successful
   IK preview is not clearance: still do the EEF-line check in Execute step 1
   before any `exec_plan_pose`.

## Execute and verify

1. Before every `exec_plan_pose`, inspect the latest head image and judge
   the straight line from the current selected EEF to the intended grasp
   point on the requested object. This is an EEF-line check, not the chassis
   blue path and not a shoulder-distance number. A cabinet side rail,
   vertical divider, door leaf, or shelf wall between that EEF and the
   object blocks the line even when planning returned IK and
   `measure_shoulder_distance` looks close. If the line is blocked, do not
   exec. Call `adjust_*_eef_pose_in_head_frame` for the planned arm and
   move the EEF around the obstacle into free space from which that
   straight line is clear. Use exactly one translation family: robot-base
   `x`/`y`/`z` (+X chassis-forward, +Y chassis-left, +Z up) or head-camera
   `forward`/`leftward`/`upward`. Keep each increment bounded and in free
   space; do not drive the EEF through the rail or door. Wrist-frame
   adjust is not this clearance step. After any EEF motion the previous
   `plan_id` is stale: reidentify the object on the post-adjust head image,
   call `plan_grasp_point_filter_rgbd_lite` again, and repeat this line
   check. Do not replace this clearance with `measure_shoulder_distance`,
   chassis inching, or pitch. If several EEF adjusts cannot make a clear
   straight line, change the chassis approach or the arm; do not exec a
   blocked plan. If the line is clear:
   Execute only the returned plan ID with `exec_plan_pose` and the planner's
   selected arm. Do not substitute another arm or replay a stale plan after any
   intervening motion.
2. Inspect the wrist image returned by that call. Red overlay pixels are
   visible non-robot 3D points inside that gripper's grasp volume, between the
   two fingers. They count as an object in the gap only when they overlap the
   requested object's visible surface.
3. Decide the next action from that returned wrist image, not from `ok`,
   `error`, timeout, stall, or any other exec return field. Do not infer a
   timeout or a cabinet-door block from those fields. If the wrist overlay
   already shows the requested object in red between the fingers, call
   `close_gripper` for that same arm. Do not call `exec_plan_pose` again on
   that `plan_id`. If the finger gap is empty, skip close and use the
   empty-gap restart below. If the image shows the EEF still on the far
   side of a cabinet side rail, divider, or door from the object, the line
   is blocked: halt that plan and clear it as in step 1.
4. Inspect the wrist overlay returned by `close_gripper`. Do not call a wrist
   camera just to replace that image. Capture that arm's wrist only when the
   close result has no usable current wrist overlay.
5. Decide from that wrist overlay, not from `ok`, timeout, `grasp_confirmed`,
   or a latched-close message:
   - Object in the gap: red overlay overlaps the requested object's surface
     between the two fingers. If the grasp is not yet visually confirmed as
     held, call `close_gripper` again for the same arm at the current pose.
     Leave the EEF where it is except for the jam relief below: do not
     `open_gripper`, do not `set_arm_to_grasp_position`, and do not
     `adjust_*_eef_pose_in_head_frame` while that red remains. Repeat
     from the newest close wrist overlay. If several in-place
     `close_gripper` calls leave the fingers visibly unchanged, the
     fingertips may be jammed against a background plane (floor or
     table). Call `adjust_*_eef_pose_in_wrist_frame` for that same arm
     with only `forward=-0.02`, then `close_gripper` again. Do not use
     this relief on an empty gap.
   - Empty gap: no red overlay between the fingers, or the red does not overlap
     the requested object. Restart: call `set_arm_to_grasp_position` for the
     same arm with `gripper="open"`, reidentify the target on the post-set-arm
     head image, call `plan_grasp_point_filter_rgbd_lite` on a visibly
     different point, then repeat the EEF-line check in step 1,
     then `exec_plan_pose` and continue from step 2. If the other arm is
     also extended near the same object, retract both with
     `set_arm_to_grasp_position` and `arm="both"` before that new plan.
     Never plan or `exec_plan_pose` for a second arm onto a nearby point
     while the first arm is still at the object. Only one arm may occupy
     the grasp region at a time.
6. Report a confirmed grasp only when the current wrist overlay uniquely shows
   the requested object held between the selected fingers. A controller
   `grasp_confirmed` field supports but does not replace that evidence. If that
   evidence is missing, do not stow and do not enter placement.
7. After a confirmed grasp, stow before exiting. Classify the held object from
   current visual evidence as small or large. This classification is about
   whether folding the holding arm to the fixed grasp-prep pose would sweep
   the object into the chassis, floor, or furniture.
   - Small: a compact handheld object (can, bottle, cup, fruit, remote, small
     toy). Call `set_arm_to_grasp_position` for the holding arm with
     `gripper="keep"`. Never pass `gripper="open"` here or the object will
     drop. Then call `reset_body` with `keep_ori_arm="none"` (or omit that
     argument) so the trunk stands up while the just-tucked arm stays put.
   - Large: a bulky or elongated object (board, box, tray, bag, appliance,
     large toy). Do not call `set_arm_to_grasp_position`. Call `reset_body`
     with `keep_ori_arm` set to the holding arm so the grasp orientation is
     preserved while the trunk stands up. `reset_body` is a trunk/posture
     tool, not an episode reset. A parent prompt such as "Do not reset"
     forbids resetting the episode or calling an episode-reset API; it does
     not forbid this stow. Do not replace this stow with
     `adjust_*_eef_pose_in_head_frame`, `adjust_height`, or any other lift.
     Do not exit this Skill until this `reset_body` has been called.
8. Only after that stow returns, call `deactivate_skill` with
   `name="pick-up-object"` and return control to the parent task. Follow the
   parent's placement instructions, activating a placement Skill only if needed.
   That is Skill exit,
   not session end. Do not start chassis travel, look-around pitching, or
   placement approach until this stow is done.

## Recovery

- If grasp planning reports no IK before set-arm preparation occurred, do not
  repeat it. Perform the required set-arm preparation, reacquire a fresh target
  point, and plan again with new evidence.
- If `plan_grasp_point_filter_rgbd_lite` returns that the point is too far to
  plan, close range first: `adjust_pitch`, `adjust_chassis`, or a new
  `move_to_reach_point` on a fresh faced head image. Then recapture, reidentify,
  and plan again. A manual `adjust_pitch` with a negative degree can reach a
  lower torso pitch than `move_to_reach_point`; that extra lean is often useful
  for a floor object. Do not replace that approach with `adjust_*_eef` or
  `close_gripper`.
- If the selected arm's measured shoulder distance already supports a grasp
  but `plan_grasp_point_filter_rgbd_lite` fails (no IK-reachable pose, overlap
  filters, or too-small grasp volume), treat the click as inaccurate first.
  Do not switch arms, back the chassis, or change pitch just because planning
  failed. Re-inspect the current head image, pick a visibly different point on
  the object's graspable body (not the floor, not a neighbor, not a guessed
  continuation), and call `plan_grasp_point_filter_rgbd_lite` again with that
  new point. If that retry still reports no IK, prefer `plan_arm="any"`:
  prepare both empty arms with `set_arm_to_grasp_position` `arm="both"` if
  needed, then plan with `plan_arm="any"`. On a hollow box, the new point
  must be a different rim or wall edge, not the interior. Never repeat
  identical `image_id`/`u`/`v`/`plan_arm` arguments.
- Only after a corrected click is visibly on the intended body and planning
  still fails, change reach positioning, arm selection, or sampling seed.
  If an arm already executed to the object, call `set_arm_to_grasp_position`
  for that unsuccessful arm with `gripper="open"` before any new plan or
  any other arm's `exec_plan_pose`. Do not leave both arms grasping nearby
  points.
- After a detour, do not treat an observer-side change as progress. Fresh
  `measure_shoulder_distance` must show a material decrease, and the remaining
  approach corridor must be clear, before planning a grasp.
- After `exec_plan_pose`, ignore `ok`, `error`, timeout, and stall
  strings. Read the returned wrist image. Red on the requested object
  between the fingers: `close_gripper` now; do not replay that `plan_id`.
  Empty gap: use the empty-gap restart. The image shows the EEF still on
  the far side of a cabinet side rail, divider, or door from the object:
  halt that plan. Do not call `measure_shoulder_distance` and do not inch
  the chassis to chase a shoulder number. Clear the line with
  `adjust_*_eef_pose_in_head_frame`, replan, and continue from Execute
  step 1.
- On collision, instability, unsafe arm clearance, or an execution result
  that fresh evidence does not show as safe partial progress: halt that
  current approach or plan, reobserve, and continue this Skill with a
  different clear approach or a different point. Do not end the session.
- Ambiguous target identity is not Skill exit: obtain a clearer observation
  and reidentify the requested object.
- An occupied intended gripper is handled in Observe: do not start a new
  pick on that gripper; continue as specified there.
- A `close_gripper` timeout, `grasp_confirmed=false`, or latched-close message
  is not by itself a miss. If the returned wrist overlay still shows red on
  the requested object between the fingers, keep closing in place. If
  several in-place closes leave the fingers unchanged, relieve a likely
  floor or table jam with `adjust_*_eef_pose_in_wrist_frame` and
  `forward=-0.02`, then close again.
- An empty finger gap after `exec_plan_pose` or `close_gripper` is the miss
  that requires restart: `set_arm_to_grasp_position` with `gripper="open"`,
  then a new plan and a new `exec_plan_pose`. Do not claim a grasp, do not
  stow, and do not enter placement from an empty gap.
