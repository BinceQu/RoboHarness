# Current perception

A connected session publishes the latest public observation after every
camera/action tool. Start with `latest.json` for image IDs and observation sequence.
`exteroception/` contains original-resolution head/wrist RGB, metric depth
arrays, depth previews, camera matrices and current track validity.
`proprioception/state.json` contains public joint/EEF state; updating it does
not load or render meshes.

For a current front/side/top body view, run
`.venv/bin/python tools/read_proprioception.py '{}'` and inspect its returned
image, using native Read if necessary. This renders only the robot model at
the latest received public joints, with no motion, camera capture or physics
steps. The result records the episode and sequence. Gripper geometry uses the
public gripper proxy and is not contact evidence; no scene or held object is
rendered. Do not click body diagrams.

`proprioception/three_views.png` exists only after that explicit request. The
next ordinary observation update expires the link; the actual image remains
in tool call history. A null `body_image` in `latest.json` means no body view
was rendered for that publication. Request another view only when posture
inspection is useful, not after every action. The body-only tool leaves camera
files and image IDs unchanged; it can also run before the first camera capture.

These paths point to immutable copies in this session's tool call history.
They represent the last observation received from the evaluator; they do
not keep simulating while the agent thinks. A new session starts empty.
