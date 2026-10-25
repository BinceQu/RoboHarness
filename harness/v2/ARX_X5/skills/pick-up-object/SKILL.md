---
name: pick-up-object
description: Pick a visible object with a fixed X5 arm using RGB-D planning and visual verification.
---

Inspect the newest head RGB and choose a graspable part of the actual object
body. Use normalized 0..1000 coordinates with this image_id. Avoid support
surfaces, empty handles, transparent background and imagined points. There
is no base approach: use measure_shoulder_distance / move_to_reach_point only
as fixed-arm reach checks and choose a reachable arm.

Read plan_grasp_point_filter_rgbd_lite.py's schema, plan the grasp, inspect
its result and execute only a valid returned plan with exec_plan_pose. Use
open_gripper / close_gripper explicitly as required by the plan. Their
gripper commands are latched during subsequent motion.

After contact, inspect the active wrist RGB-D and latest head image. A close
command or a low normalized gripper value alone is not evidence of holding.
Confirm object motion with the gripper, wrist volume and/or valid tracks.
Lift conservatively using the supported arm controls and observe again.

If planning or execution fails, reconsider the clicked body surface in the
latest image before changing seeds or arms. Do not retry a remembered box.
If motion fails, inspect current joints and body rendering before recovery.
End this skill when the object is visibly held in a stable useful pose, or
when abandoning this attempt. Do not invoke unsupported mobile-body resets.
