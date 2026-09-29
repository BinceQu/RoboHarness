---
name: close-box
description: Close a box that has a lid. Do not use on objects without a lid.
---

# Close box

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
Follow this procedure only while `close-box` is the active task Skill loaded by
`activate_skill`. Its presence in history, including after compaction, does
not make it active. After success and all required post-actions, return to the
parent task; activate another matching Skill only when its procedure is needed.
Instructions below to stay, retry, or exit describe pursuit of success and do
not prohibit a change of approach or handoff to the parent task.
<!-- codex-task-lifecycle:end -->

Use the advertised `behavior_v2` tools and the baseline Skill. This Skill
closes a lidded box. It is not for objects without a lid and not for
picking up the box.

Successful completion requires a fresh head image that shows the lid seated shut
on the box body. Stopping a bind, plan, reach, or adjust is a sub-step
change, not Skill exit and not session end.

1. Capture the head camera. On that image pick exactly two points:
   - `lid_edge`: a visible point on the lid's free edge, not the hinge.
   - `body_edge`: the matching point on the box body's rim.
   Call `track_object_distance` once with both named points.

2. Call `plan_grasp_point_filter_rgbd_lite` on `lid_edge`. If that plan
   fails, call `spin_to_facing_point` then `move_to_reach_point` on
   `lid_edge`, recapture, rebind both points, and plan again.

3. Call `exec_plan_pose` with the returned plan, then `close_gripper` on
   that same arm. Keep that gripper closed until the lid and the body
   form an acute angle.

4. After each action, read both current `xyz_in_robot_base_coord_m`
   values from `persistent_tracking`. Use `body_edge - lid_edge` only to
   judge which way the lid still needs to move. Do not follow a fixed
   step size, interpolation fraction, or z schedule.

5. Move the holding arm with `adjust_left_eef_pose_in_head_frame` or
   `adjust_right_eef_pose_in_head_frame`. Choose each increment from the
   latest tracks and the current image: direction, size, and height may
   change after every move. If either point is lost, recapture and
   rebind both points before the next adjust.

6. Call `open_gripper` on the holding arm as soon as a fresh view shows
   the lid and the body forming an acute angle. Do not keep closing
   after that.

7. Recapture the head camera. If the lid sits shut on the body, return to the
   parent task; that is Skill exit, not session end. If the box is still open,
   go back to step 1.
