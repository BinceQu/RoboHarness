---
name: open-doors-and-drawers
description: Use a reachable tabletop handle with fixed X5 arms and visually constrained incremental motion.
---

This robot cannot approach using a mobile base. First verify a handle is
visible and reachable from the fixed rig. Inspect fresh RGB-D to identify
the solid graspable handle and the opening direction. Use image-bound
normalized clicks, or register current RGB-D points for tracked constraints.

Read the relevant grasp and move_tracked_point schemas. Grasp the handle,
verify physical engagement visually, then use small motions consistent with
the observed drawer slide or door hinge. Do not assume a scene-graph hinge
or simulator object pose. Inspect new images and track validity after each
motion. Stop on stalled tracking, invalid depth, failed planning or unexpected
object motion and reassess from current observations. Release safely and
deactivate once the opening is confirmed or the attempt is abandoned.
