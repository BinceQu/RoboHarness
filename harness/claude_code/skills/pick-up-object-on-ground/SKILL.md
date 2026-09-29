---
name: pick-up-object-on-ground
disable-model-invocation: true
description: 捡起地上物体。用户说捡起地上的物体、把地上的东西拿起来、或做地面抓取时使用。按锁定的头顶相机、像素点、RGB-D 规划和 z=0.25 抬起顺序执行。
---

# Pick up object on ground

This is an isolated floor-grasp unit test on the already-running BEHAVIOR
interface. The scene is the custom picking_up_trash layout with the front
floor slot replaced by one target object (the specific object varies per
case — always trust the live image, never assume which item). Do not attempt
the parent picking_up_trash task. Do not navigate, reset the episode, use a
map, spin, or activate another Skill. Stay strictly in the sequence below.

The target is whatever single object the live head image shows on the floor
in front of the robot. It is NOT a known object category that you can guess
from this prompt — read the live image every step to identify it. Ignore
everything that is not the target (other props, debris, items to the side).
If two objects look similar or one is partially behind another, click the
one closest to the robot on the floor in front of the gripper.

Point contract, required for the click to land on the target: the latest head
image is 720 x 720. Send integer original-image pixels only. u is the column
and v is the row, top-left (0,0), bottom-right (719,719). Do not normalize to
0..1000 and do not pre-scale. This plugin converts each pixel to the interface
wire coordinate with round(pixel * 1000 / 719). A 0..1000 number sent as if it
were a pixel will miss the target. Bind the point to the image_id of the head
image you just inspected. Wrist images cannot be clicked.

Use these steps exactly:

0. Call capture_head_camera. Inspect the returned 720 x 720 head image and
   decide whether the target is a container (pot, pan, bowl, cup, box) or a
   solid object (banana, calculator, cube, tool, prop). For containers,
   plan_grasp_point_filter_rgbd_lite usually picks the empty interior as the
   surface, which has no thickness to grip. To force a graspable surface, you
   MUST click the rim or lip (the top edge of the container wall), not the
   empty inside. For solid objects, click any pixel clearly on the body,
   inside its silhouette.

1. Call capture_head_camera again to lock in the latest image. Visually
   locate the target's real, visible, graspable surface following step 0.
   Choose a pixel clearly on the body (or rim for containers), inside its
   silhouette / on its top edge. Do not click the floor, shadow, robot,
   blue ranging line, overlay text, or a point just outside the contour.
   Record the selected pixel as (u_orig, v_orig) — the original click
   coords you chose.

1b. Call track_object_distance with session_id (omit session_id, the plugin
    injects its adapter-owned one), image_id set to the same image_id from
    step 1, and points=[{"name":"obj","u":u_orig,"v":v_orig}] (the pixel
    you just chose, in original integer pixel coordinates). Read the
    response's `tracked_object_distances.obj.xyz_in_robot_base_coord_m[2]`
    (or the equivalent z field) — that is the floor-up height of the point
    you clicked, in meters.
       If the returned z is < 0.01: the click landed on the floor, NOT on
       the target. You MUST keep trying — go back to step 1: call
       capture_head_camera again, re-pick a different pixel that is
       clearly on the target's body, and re-run step 1b with the new
       pixel. You may retry step 1+1b up to 20 times total. After 20
       floor-misses without ever getting a z >= 0.01 reading, report
       ("could not place a click on the target") and stop — only then is
       this test considered a clean failure on point selection.
       If the returned z is >= 0.01: the click landed on the target
       surface. Proceed to step 2.

2. Call plan_grasp_point_filter_rgbd_lite on that same image_id and pixel.
   Pass the original integer pixel coordinates as `u` and `v`, `plan_arm` set
   to `any`, and `seed` set to 42. The plugin injects its adapter-owned
   session_id; do not invent or pass a different session_id. This planner
   consumes the frozen RGB-D and robot state for the captured image, so do
   not call a separate tracking or grounding tool.

   If the planner returns `ok: true`: record `plan_id` and `recommended_arm`,
   preserve its returned render/overlay image and plan record as evidence.
   Proceed to step 3.

   If the planner returns `ok: false` with a "too far" or "shoulder distance"
   error:
     (a) Call adjust_pitch with degree=-5. Wait for it to finish.
     (b) Call capture_head_camera again.
     (c) On that new image_id and the same (u_orig, v_orig) pixel, call
         plan_grasp_point_filter_rgbd_lite again.
     (d) If `ok: true`: record plan_id and recommended_arm. Proceed to step 3.

   For any other planner error (not a distance/shoulder error): choose another point on object, and go back to step 1b and replan

3. Call exec_plan_pose with that plan_id and back_m=0.2. Use the returned
   recommended_arm. exec_plan_pose always attempts to move; it does not
   return a "failed" outcome in this test — whatever it returns, you must
   keep going. Read the returned grasp_volume_overlay and grasp_volume
   metrics if present, but treat execution as completed regardless.

4. Call close_gripper with arm=recommended_arm. close_gripper always
   attempts the close regardless of what it returns; ignore any "failed"
   or "blocked" return value, treat it as completed. If the gripper felt
   suspicious (e.g. the wrist image in step 4(c) suggested the fingers were
   still open), call close_gripper once more on the same arm. After at most
   two close calls you must move on to step 6 — do not iterate further.

5. Call adjust_left_eef_pose_in_head_frame or
   adjust_right_eef_pose_in_head_frame for recommended_arm with exactly
   z=0.25. Do not add x, y, camera-frame translation, or rotation.

6. Call capture_head_camera once more and inspect the global view directly
   — do NOT call track_object_distance here. The capture lets you see
   whether the simulator performed an episode reset. A reset looks obvious:
   the robot is back at its spawn pose (arms retracted, head looking
   forward at the same default angle as step 0), AND the target on the
   floor is now a fresh instance (different orientation, slightly different
   position, or a different sampled object from the same category). Both
   signs together — robot jump + object swap — are required. If you only
   see one of them (e.g. the robot moved a bit but the target is the same
   object in the same spot, or the target looks new but the robot is still
   reaching), that is NOT a reset, treat it as a failure.
   There are exactly two possible outcomes:
     (a) Real episode reset (robot at spawn AND target swapped to a new
         instance) → this is a success. Report "scene reset, success" and
         stop. Do not invent this — only report it when both signs above
         are clearly visible.
     (b) Anything else (target still on the floor in the same spot as
         step 1, fingers missed, slipped off, or only one reset sign
         visible) → this is a failure.
         → Call open_gripper and set_arm_to_grasp_position with arm=recommended_arm.
         → Go back to step 1: call capture_head_camera, re-pick a visible
           pixel on the target, run step 1b, then step 2, step 3, step 4,
           step 5, and step 6 again in order.
         Each return to step 1 counts as one attempt. At 20 entries of
         step 1, stop and report "could not lift target after 20 attempts".
         That report ends this Claude Code session.

A single tool error is not a reason to stop before the caps above.
`evaluator_not_connected`, a timeout, `retryable: false`, HTTP 409, or any
other tool failure counts as one failed attempt: call capture_head_camera
again and continue from step 1. Do not deactivate this skill. Do not return
to baseline. Do not activate pick-up-object or any other skill. The stops
that end this Claude Code session are only: a real episode reset, 20
floor-misses that never reach z >= 0.01, or 20 entries of step 1. When one
of those three happens, report it and end the session. Do not keep calling
tools after that report. The session-context rule that a skill stop is only
a sub-step halt does not apply to these three stops.

First tool after this skill is loaded: capture_head_camera.
