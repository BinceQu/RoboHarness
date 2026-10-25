---
name: place-object-in-container
description: Place a held object into a visible receptacle using the fixed dual X5 arms.
---

Confirm which gripper holds the object from current RGB-D. Inspect the latest
head/wrist views to locate the real opening and avoid walls/rims. Read the
translation and tracked-point tool schemas before planning. Keep the holding
gripper closed while moving. Use image-bound normalized coordinates or
currently valid tracked points; do not reuse old object locations.

Plan a reachable pose with clearance for both the object and the gripper.
Inspect the plan and execute a valid signed plan. Descend cautiously and
verify the object is supported inside the receptacle before opening. Retract
without dragging the object, then verify placement in new RGB-D. Do not use
an empty gripper command proxy as proof of placement. Deactivate this skill
when the placement is visually confirmed or the attempt is abandoned.
