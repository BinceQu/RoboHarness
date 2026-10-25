---
name: robodojo-baseline
description: Always-active perception and control rules for the fixed dual ARX X5 tabletop robot.
---

Use the English workspace README as the robot operating instructions. Read
the robot manual and examine its offline body view before unfamiliar joint
control. All commands run as `.venv/bin/python tools/NAME.py 'JSON'`.

For the current body posture, run
`.venv/bin/python tools/read_proprioception.py '{}'` only when you need to
inspect it, and view the returned front/side/top image (native Read if only a
path is shown). This reads public joints without motion or camera capture.
Ordinary tools update numeric joint state but do not render body views. Do
not add a render after every action. A saved body image describes its recorded
episode/sequence, not later motion. Body diagrams show no scene or held objects
and cannot be clicked or used to prove a grasp.

Inspect the actual latest image before selecting every point. Latest images
represent current state; older images, remembered coordinates and old tracks
represent past states. Re-identify the object after every movement; never
translate an old imagined object diagram into the new frame. If uncertain,
capture again. Use the returned image_id and normalized 0..1000 coordinates.
Do not confuse depth previews, body renderings or click-history overlays with
the live camera used for control. Original RGB pixels are never resized.

Use head and both wrist cameras, read_depth, track_object_distance and the
current state in perception/. Tracks have validity and observation bindings;
check them against current pixels. Read tool schemas before unfamiliar calls.
Learn parameter names from tools/README.md and each readable Python file.

Results report concrete measurements and errors without a generic success
flag. Absence of an error does not prove arrival, contact or grasp. Inspect
tool-specific completion/progress fields and the current camera observations.

The robot has two fixed six-joint arms. Mobile-base, torso and head-pitch
tools are unsupported. Use arm planning and wrist views to manipulate objects.
Prefer plan → inspect returned visualization/result → execute → inspect new
RGB-D. A planner failure is not evidence that the object has bad depth: first
check whether the clicked pixel is actually on the intended object surface.
Do not repeat seeds/arms at an unverified point on a support surface.

Activate only the task skill needed for the current manipulation. Complete
or abandon it explicitly with deactivate_skill; do not load an unrelated
skill just to clear the previous one. You may also read any skill file using
the agent's native skill/read mechanism. Do not inject every skill up front.

Atomic joint commands are available for informed adjustments. They bypass
macro-level IK and collision checks; read the current joints and axis diagrams
first, use small deliberate changes, and observe the result. Preserve the
other arm and both gripper targets unless intentionally changing them.

Use only public camera/state evidence. Do not read evaluator score/scene
truth to choose actions. Context management and image reading stay native to
the host agent. If native image attachment fails, use Read on the saved PNG.
