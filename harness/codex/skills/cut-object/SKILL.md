---
name: cut-object
description: Cut a target object using a knife or axe already held by the robot. Use when the next task step requires cutting, chopping, or slicing an object.
---

# Cut object

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
Follow this procedure only while `cut-object` is the active task Skill loaded by
`activate_skill`. Its presence in history, including after compaction, does
not make it active. After success and all required post-actions, return to the
parent task; activate another matching Skill only when its procedure is needed.
Instructions below to stay, retry, or exit describe pursuit of success and do
not prohibit a change of approach or handoff to the parent task.
<!-- codex-task-lifecycle:end -->

1. Use one current head capture where the held knife or axe and the target
   object are both visible. Identify which arm holds the cutting tool and keep
   that gripper closed.
2. If the cutting edge is not visible, or is not yet at a convenient cutting
   orientation, call `control_wrist_roll` on the holding arm until the blade
   faces the most convenient cut pose. Recapture the head image after each
   roll. Do not call `set_arm_to_grasp_position` before or during the cut:
   that call zeros the previous J8 wrist-roll amount and undoes this
   orientation.
3. Select exactly two points on the same current image:
   - `cutting_tool_point`: any visible point on the knife or axe.
   - `target_object_point`: the point to touch on the target object.
4. Call `track_object_distance` once with that capture's `image_id` and both
   named points.
5. Read both current `xyz_in_robot_base_coord_m` values from
   `persistent_tracking`. Compute `target_object_point - cutting_tool_point`.
6. Move the holding arm by that robot-base XYZ difference with
   `adjust_left_eef_pose_in_head_frame` or
   `adjust_right_eef_pose_in_head_frame`. Pass the difference as `x`, `y`, and
   `z`; keep roll, pitch, and yaw unchanged.
7. After each move, read the updated tracked points and repeat steps 5-6 until
   the cutting-tool point touches the target point. If either point is lost,
   capture a new head image and bind both points again.
8. After the whole cut is complete, call `set_arm_to_grasp_position` once for
   the holding arm with `gripper="keep"`. This is the required last step. It
   also zeros the wrist roll from step 2. Then return control to the parent
   task; this is Skill exit, not session end.
