# Current senses

Read `latest.json` first: it identifies the current episode, observation
sequence and clickable head image. All referenced files come from that same
observation. Publishing changes pointers only after the new files are complete.

`proprioception/joint_state.json` contains measured angles (rad) and measured
finger separation coordinates (m), not commanded targets. It updates after
ordinary tools without loading or rendering meshes.

To see the current posture, run `python tools/read_proprioception.py` and
inspect its returned image (native Read if it is not displayed). This read-only
tool renders front, left and top views using measured joints and URDF meshes
on a plain background; it issues no actions and consumes no physics ticks.
It neither accesses nor depicts the scene or held objects. Never click it.

`proprioception/robot_three_views.png` is only the last requested render and may
not exist yet. `proprioception/render.json` binds it to an observation sequence
and immutable joint JSON. Ordinary actions do not update these two files.
Compare that sequence with `joint_state.json` before treating it as current;
after motion, call the tool again only when you need a new posture view.

`exteroception/` contains the current head and wrist RGB, depth previews
and tracks. Depth previews are PNGs with a fixed 0–5 m grayscale; invalid depth
is black. `head_overlay.png` adds geometric/tracking annotations;
`head_rgb.png` is the unannotated view. These are real files holding only the
current observation. Tracks can be occluded or stale: inspect their
status/sequence, and never substitute an old tracked coordinate for visually
locating a fresh click.

Earlier returned images remain in `tool_call_history/`; they are historical,
not current.
