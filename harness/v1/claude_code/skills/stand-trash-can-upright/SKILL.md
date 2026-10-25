---
name: stand-trash-can-upright
description: Use when the trash can has fallen over.执行这个流程可以扶起他
disable-model-invocation: true
---

# Stand trash can upright

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
Follow this procedure only while `stand-trash-can-upright` is the active task
Skill in the latest lifecycle result or current `task_skill_state` notice.
Its presence in history, including after compaction, does not activate it.
After success and all required post-actions, call `deactivate_skill` with
`name="stand-trash-can-upright"`, then continue the parent task under the baseline.
You may also deactivate to cancel or hand control back without claiming success.
Instructions below to stay, retry, or exit describe pursuit of success; they do
not prohibit explicit cancellation or handoff. Carry unfinished work and any
pending cleanup back to the parent task. Physical constraints still apply;
deactivation itself does not move the robot or release a grasp.
<!-- claude-task-lifecycle:end -->

Use the advertised `behavior-v2` tools and the baseline Skill. This Skill is
only for a fallen trash can. It is not placement and not picking up trash.

Success requires that a fresh `capture_head_camera` image shows the
ashcan standing on its base with the opening facing up. Stopping a current
chassis move, plan, exec, roll, descent, or release is a sub-step change,
not Skill exit and not session end.

`control_wrist_roll` modes: `relative` / `absolute` change J8; `reset` returns
that arm's J8 to 0. Do not call `set_arm_to_grasp_position` after a roll:
that also zeros J8 and undoes the orientation.

Repeat this loop until that exit:

1. Face the opening. Call `adjust_chassis` to translate and/or spin until
   the robot faces the ashcan opening. Recapture the head camera after each
   chassis move and judge facing only from that fresh image. Do not plan
   while the opening is off to the side or behind the robot.
2. Plan the upper rim. On the latest head image, select a point on the
   currently elevated upper rim. Call `plan_grasp_point_filter_rgbd_lite`
   for that point. Use an empty gripper; if one hand already holds a can,
   keep that gripper closed and plan the other arm.
3. Exec, then close. Call `exec_plan_pose` with the returned plan. Then
   call `close_gripper` on that same arm.
4. Lift. After the gripper is closed, raise that same arm 0.4 m with
   `adjust_left_eef_pose_in_head_frame` or
   `adjust_right_eef_pose_in_head_frame`. Use `z=0.4` only.
5. Roll the opening up. Call `control_wrist_roll` on the holding arm
   (`mode="relative"`) until the ashcan opening faces up. Recapture the
   head camera after each roll and judge only from that fresh image.
6. Lower to the floor. On the latest head image, bind one point on the
   visible ashcan bottom face with `track_object_distance`. Read that
   point's current `xyz_in_robot_base_coord_m` z. Lower the holding arm
   by `z - 0.05` m: pass robot-base `z=0.05 - bottom_z` to the same
   `adjust_*_eef_pose_in_head_frame` tool. Do not add a large XY move in
   that call.
7. Release. Call `open_gripper` on the holding arm.
8. Check. Recapture the head camera. A body lying on its side on the
   floor is not upright, even if the opening is visible or a gripper has
   moved. That is not Skill exit and not session end: call
   `control_wrist_roll` with `mode="reset"` on the holding arm, then go
   back to step 1. Exit this Skill only when the ashcan stands on its
   base with the opening facing up. Then call `control_wrist_roll` with
   `mode="reset"` on the holding arm, then call `deactivate_skill` with
   `name="stand-trash-can-upright"` and return to the parent task. Activate
   `place-object-in-container` only if its placement procedure is needed.
