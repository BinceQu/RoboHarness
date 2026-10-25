# Working in this robot workspace

Read README.md, robot_config/README.md and skills/robodojo-baseline/SKILL.md.
Use this workspace's Python environment and tools. Inspect the latest image
before each visual decision. The normalized click contract and joint units
are defined in robot_config/README.md. The complete implementation is local.
Use native image reading when a tool image is not attached to the context.
Keep current perception and historical evidence distinct.

Tool returns contain measured outcomes and explicit errors, without a generic
success flag. Absence of an error does not prove arrival, contact or grasp.
Check the specific completion/progress fields and current camera evidence.

For current body posture, run `.venv/bin/python tools/read_proprioception.py '{}'`
only when needed and inspect the returned front/side/top image. Use native
Read if it is not attached. This reads public joints and renders the local
model without robot motion or camera capture. Ordinary tools update numeric
joint state but never render body views. Do not render after every action or
use a past body image as the current pose. Body diagrams contain no scene or
held objects and are not clickable camera views.
