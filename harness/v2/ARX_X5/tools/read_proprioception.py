"""Read public joints and render front/side/top body views only on request.

Run: .venv/bin/python tools/read_proprioception.py '{}'
This reads the latest received evaluator state without sending an action,
advancing physics, capturing cameras or running IK. The scene-free image is
attached through the usual native image hook; other agents should open its
returned path. It is a body diagram, not a clickable camera or grasp proof.
"""

import json

from tools.util.perception import latest_link, publish_state
from tools.util.settings import write_json


def read_proprioception(runtime, session_id=None):
    """No user arguments. Render the current public joint state on demand."""
    if session_id not in (None, runtime.session_id):
        raise ValueError("Tool cannot access another session")
    if not runtime.adapter.connected():
        raise RuntimeError("Evaluator is not connected; no body view was rendered")

    body, state = publish_state(runtime, runtime.history)
    manifest_path = runtime.root / "perception/latest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    same_observation = manifest is not None and all(
        manifest.get(key) == state[key] for key in ("sequence", "episode_id")
    )
    if manifest is not None:
        manifest["body_image"] = None
        write_json(manifest_path, manifest)

    # Import and allocate meshes only for this tool. Reuse the existing owned
    # rendering thread on later explicit requests; never allocate a GPU.
    from tools.util.render import BodyViewRenderer

    if runtime.body_renderer is None:
        runtime.body_renderer = BodyViewRenderer(runtime.root / "robot_config")
    image_path = body / "three_views.png"
    runtime.body_renderer.save(
        state["render_joint_positions"], image_path,
        caption="ARX X5 body from public joints (gripper = observed proxy)",
        sequence=state["sequence"],
    )
    latest_link(image_path, runtime.root / "perception/proprioception/three_views.png")
    if same_observation:
        manifest.update(body_image=str(image_path), proprioception=str(body / "state.json"))
        write_json(manifest_path, manifest)
    caption = (
        f"BODY VIEW episode={state['episode_id']} sequence={state['sequence']}. "
        "Front/side/top from public joints and local meshes; no scene. "
        "Not a camera; do not click it or infer contact from the gripper proxy. "
        "After movement, request a new body view only if you need one."
    )
    return {
        "ok": True, "tool": "read_proprioception",
        "sequence": state["sequence"], "episode_id": state["episode_id"],
        "state": state, "state_path": str(body / "state.json"),
        "images": [runtime.history.image(image_path, caption)],
        "perception": str(manifest_path) if same_observation else None,
        "observation_instruction": "Inspect the attached body view, or open its returned path with native Read. Camera images and click IDs are unchanged.",
    }


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("read_proprioception", read_proprioception)
