---
name: behavior-v2-baseline
description: Use when observing or operating a robot through the Embodied Codex BEHAVIOR v2 MCP tools, interpreting returned camera images, collecting a direct tool trajectory, or building and testing a task-specific embodied Skill on this baseline.
---

# BEHAVIOR v2 baseline

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

The `behavior_v2` MCP server derives its tools from the active interface catalog
after machine policy. It also restores the stable `mark_on_map` contract when an
interface profile omits that catalog entry. Call the resulting tools directly
using their advertised descriptions and input schemas.

The adapter starts its session and recorder when MCP starts. It injects
`session_id`, applies catalog-fixed arguments such as `mode`, serializes calls,
and closes recording with the MCP session. Do not invent or pass hidden adapter
arguments.

For each decision:

1. Select one currently advertised tool whose preconditions are supported by the
   latest result or visual observation.
2. Call it once with only schema-declared arguments.
3. Inspect success status, structured data, and native MCP image content directly
   before the next action. Do not route camera output through a path, base64
   decoding, OCR, `view_image`, or another image-loading tool.
4. If evidence is ambiguous or a precondition is unsafe, do not issue that
   command. Reobserve or choose a different advertised tool. This is a change
   of the current decision, not Skill exit and not session end.

Held-object evidence. Empirically, an object already in a gripper generally
does not drop unless that gripper has executed `open_gripper`. Do not infer a drop
from finger `qpos`, chassis motion, a lost track, or a head image that no
longer shows the object. If a current wrist image of that gripper shows a
red grasp-volume overlay, the object has not dropped. Missing red is not a
drop: first treat the wrist view as badly angled or occluded, and recapture
or change viewpoint. Do not `open_gripper`, and do not call
`set_arm_to_grasp_position` with `gripper="open"` on that arm, while the
hold still stands.

This Skill has no task exit. Session end is only the global rule in the
session context: the robot has fallen or the parent task is physically
impossible, or a required tool or the simulator has crashed.

Do not use shell, `curl`, direct HTTP, file editing, or service-management
commands. This Skill intentionally defines no task order, object identity,
object-specific point, grasp policy, or recovery sequence.
