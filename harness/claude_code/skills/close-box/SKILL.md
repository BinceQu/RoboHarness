---
name: close-box
description: Close a box that has a lid. Do not use on objects without a lid.
---

# Close box

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
Follow this procedure only while `close-box` is the active task
Skill in the latest lifecycle result or current `task_skill_state` notice.
Its presence in history, including after compaction, does not activate it.
After success and all required post-actions, call `deactivate_skill` with
`name="close-box"`, then continue the parent task under the baseline.
You may also deactivate to cancel or hand control back without claiming success.
Instructions below to stay, retry, or exit describe pursuit of success; they do
not prohibit explicit cancellation or handoff. Carry unfinished work and any
pending cleanup back to the parent task. Physical constraints still apply;
deactivation itself does not move the robot or release a grasp.
<!-- claude-task-lifecycle:end -->

Use the advertised `behavior-v2` tools and the baseline Skill. This Skill
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

7. Recapture the head camera. When the lid sits shut on the body, call
   `deactivate_skill` with `name="close-box"` and return to the parent task.
   If the box is still open, go back to step 1. That is not
   session end.
