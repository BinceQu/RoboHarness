---
name: behavior-v2-baseline
description: Observe and operate R1_pro through readable workspace Python tools, interpret camera observations, and run embodied task skills.
---

# BEHAVIOR v2 baseline

<!-- claude-pixel-contract:start -->
The latest camera image in the conversation represents the current observed scene.
Older images are historical context; their points, bounding boxes, and
object-part locations do not describe the current view. After any motion,
re-locate the target in the latest image; do not shift, rotate, or scale an
older location estimate to choose a new click.

Before EVERY point selection, read the latest returned clickable head image
and re-locate the physical target. Use its actual 720 x 720 original-image
pixels: integer `u` (column) and `v` (row) in 0..719, with top-left `(0,0)` and
bottom-right `(719,719)`. Do not normalize, rescale, or use crop coordinates.
Bind every point, including every row of a multi-point call, to that exact
current `image_id`. Never copy locations from older images, earlier reasoning,
reference photos, tracking text, or overlay labels. Wrist images are
observation-only and cannot be clicked. If the Python image check rejects a
click, capture a fresh head image and ground again; do not guess an image_id.
<!-- claude-pixel-contract:end -->

Each tool is an ordinary readable Python file. Read `tools/README.md` for the list; read a schema with
`python tools/TOOL_NAME.py --schema`. Invoke a tool using
`python tools/TOOL_NAME.py '{"argument":"value"}'`.
The local runner binds the session, serializes robot actions, and records every input and output.
Do not pass hidden arguments. The tool computes its action sequence locally and sends
27-value joint commands directly to the evaluator.
Source is available in `tools/`; atomic APIs and controller units are in
`robot_config/`. The same official control algorithms back these commands.

For robot posture inspection, run `python tools/read_proprioception.py` when
needed and inspect its returned front/left/top render (native Read if only a
path is shown). It reads current measured joints without moving or consuming
physics ticks. It depicts only the robot, not the scene or held objects, and
is never a clickable camera image. Ordinary tools update measured joint JSON
but do not render posture views. The saved `perception/proprioception/robot_three_views.png`
describes the sequence in `perception/proprioception/render.json`; it does not
auto-refresh. After motion, request a new render only if you need to inspect
posture. Do not add a render after every action.

For each decision:

1. Select one currently advertised tool whose preconditions are supported by the
   latest result or visual observation.
2. Call it once with only schema-declared arguments.
3. Inspect measured progress, tool-specific completion fields, errors and every returned image before
   the next visual decision. This workspace's Claude hooks attach newly saved
   images directly to Bash results. If your host returns only paths, open each
   with its native image reader (Claude Code: Read). A filename is not image
   evidence. Historical files remain available but do not show the current view.
4. If evidence is ambiguous or a precondition is unsafe, do not issue that
   command. Reobserve or choose a different advertised tool. This is a change
   of the current decision, not Skill exit and not session end.

Tool returns have no generic success flag. In particular, a base may stop and
settle after moving only part of the requested distance. Check `actual`,
`linear_target_reached`, and `obstacle_limited` before treating a chassis move
as complete. No reported error does not establish arrival, contact or grasp;
use the corresponding measurements and current images.

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

Use Bash to invoke the Python tools and Read to inspect source and images.
Do not manage simulator processes or reset episodes unless the parent task
permits it. `reset_body` is ordinary posture motion, not an episode reset.
Activate/deactivate task skills via the corresponding Python commands; the
latest result's task_skill_state is authoritative after compaction too.
This Skill defines no task order, object identity, object-specific point,
grasp policy or recovery sequence. Follow the parent's explicit hand-in rule.
