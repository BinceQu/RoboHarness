---
name: navigate-to-target
description: Drive toward a goal that is either already marked on the live minimap or not yet seen. The minimap is the overlay on the top-right of the head-camera image; look there when you need the map. If that overlay has no clear structure, look around with consecutive spin=90 first. Do not use this Skill to open doors or cross a tight doorway.
---

# Navigate to target

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
Follow this procedure only while `navigate-to-target` is the active task Skill loaded by
`activate_skill`. Its presence in history, including after compaction, does
not make it active. After success and all required post-actions, return to the
parent task; activate another matching Skill only when its procedure is needed.
Instructions below to stay, retry, or exit describe pursuit of success and do
not prohibit a change of approach or handoff to the parent task.
<!-- codex-task-lifecycle:end -->

Use the advertised `behavior_v2` tools and the baseline Skill. Start with
`capture_head_camera`. The live minimap is the overlay on the top-right of
that head image; look at that corner when you need the map, plus any
`mark_on_map` / `persistent_tracking` places. This Skill moves the chassis.
It does not replace `traverse-narrow-passages`.

Classify the goal as A or B, then stay on that case.

## A. Marked object or place

The name is already on the minimap or in marked places.

1. From the top-right minimap on the current head image, read where the
   mark sits relative to the robot.
2. Pick a short, visible floor path toward that mark. Prefer
   `move_chassis_to_floor_point` on free floor; use `adjust_chassis`
   `forward` only when the path is already aligned.
3. Recapture after each move. Repeat until the chassis is at the mark or
   the marked object is clearly in view.
4. Do not spin in place to search.

## B. Unseen object or named place

The goal has no mark and is not in the current view (a room name, or an
object that has not been seen this episode).

If the current minimap has no clear structure (for example only a single
line), look around first with consecutive `spin=90` turns to see what
the surroundings roughly are. If mapping has already started and you are
searching for an object, drive along the direction where the map is
missing. After you find the object, mark it with `mark_on_map`.

1. From the current head image and minimap, choose the heading that most
   increases unknown/unmapped space, not a direction that stays in already
   mapped floor.
2. Drive that way with a finite floor move. Recapture. Repeat.
3. Do not keep adjusting on the same spot once the map has a clear
   structure.
4. Success requires a fresh `capture_head_camera` that shows the
   requested object, or the requested place has been reached (for example
   the living room). If the object was found, mark it with `mark_on_map`
   before leaving this Skill. Then return to the parent task.

This is Skill exit, not session end. Do not end the Codex session.
