"""Inspect the measured robot posture: python tools/read_proprioception.py

Render front, left and top views from the evaluator's current measured joints
and this workspace's URDF meshes. Call when you need to inspect arm/torso posture
or understand a joint motion; ordinary tools do not render these views.
This is read-only: no joint command, physics step or camera capture is issued.
The image depicts the robot alone, not the scene or held objects. Never click it.
"""
import argparse
from hashlib import sha256
import json
import time

import jsonschema

from tools.util.files import atomic_json, copy_latest, load_json
from tools.util.history import CallHistory, record_failure
from tools.util.perception import write_joint_state
from tools.util.robot import Robot
from tools.util.receipt import model_receipt_text

NAME = "read_proprioception"
SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def run(arguments, *, robot=None):
    robot = robot or Robot()
    with robot.exclusive():
        history = CallHistory(robot, NAME, arguments)
        images, manifest, error = [], None, None
        sequence = robot._official_adapter.current.sequence
        try:
            jsonschema.Draft202012Validator(SCHEMA).validate(arguments)
            folder = history.calls / "observation"
            folder.mkdir()
            archived_joints, snapshot = write_joint_state(robot, folder)
            latest = robot.workspace / "perception/proprioception"
            joint_path = latest / "joint_state.json"
            copy_latest(joint_path, archived_joints)
            # Keep renderer imports and mesh loading exclusively on this path.
            from tools.util.render import RobotRenderer
            renderer = RobotRenderer(robot.config["mesh_root"])
            try:
                picture = renderer.three_views(snapshot["joint_positions"], sequence=sequence)
                path = history.path / "returned_image_0.png"
                picture.save(path)
                width, height = picture.size
            finally:
                renderer.close()
            caption = (
                f"Measured robot posture at observation sequence {sequence}: front, left and top views. "
                "Offline URDF rendering, not a camera; scene and held objects are not shown. "
                "Do not use this image for scene clicks. After motion, call read_proprioception again "
                "only if you need an updated posture view.")
            media = {"path": str(path), "mime_type": "image/png", "width": width, "height": height,
                     "sha256": sha256(path.read_bytes()).hexdigest(), "caption": caption,
                     "feed": "proprioception", "clickable": False}
            manifest = {"session_id": robot.session_id, "episode_id": robot.config["episode_id"],
                "sequence": sequence, "received_at": snapshot["received_at"], "rendered_at": time.time(),
                "joint_state": str(joint_path), "robot_three_views": media,
                "refresh": "On demand only: python tools/read_proprioception.py"}
            manifest_path = folder / "render.json"
            atomic_json(manifest_path, manifest)
            copy_latest(latest / "robot_three_views.png", path)
            copy_latest(latest / "render.json", manifest_path)
            images = [media]
            response = {"observation_sequence": sequence,
                        "joint_positions": snapshot["joint_positions"], "joint_state": str(joint_path),
                        "render_metadata": str(latest / "render.json")}
        except Exception as exc:
            # Never return an older posture image as the result of a failed read.
            error = f"{type(exc).__name__}: {exc}"
            response = {"error": error}
        state = load_json(robot.state_dir / "state.json", {})
        receipt = {"tool_name": NAME, "response": response,
            "images": images, "rollout_budget": robot.budget, "joint_commands_sent": 0,
            "latest_clickable_image_id": state.get("latest_image_id", "")
                if state.get("latest_sequence") == sequence else "",
            "task_skill_state": {"baseline": "behavior-v2-baseline", "active_skill": state.get("active_skill")},
            "observation_instruction": "Read the attached posture image (or its path with native Read); it is on-demand and not for scene clicks."}
        if error:
            receipt["error"] = error
        return history.finish(receipt, {"operation_result": response, "perception": manifest, "error": error})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arguments", nargs="?", default="{}", help="Empty JSON object; this tool has no inputs")
    parser.add_argument("--schema", action="store_true", help="Print schema without connecting or rendering")
    options = parser.parse_args()
    if options.schema:
        print(json.dumps(SCHEMA, indent=2))
        return
    arguments = options.arguments
    try:
        arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool input must be a JSON object")
        result = run(arguments)
    except Exception as exc:
        result = record_failure(NAME, arguments, {"tool_name": NAME,
            "error": f"{type(exc).__name__}: {exc}", "images": []})
    print(model_receipt_text(result), flush=True)


if __name__ == "__main__":
    main()
