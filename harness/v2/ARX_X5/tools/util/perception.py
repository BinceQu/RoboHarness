"""Publish full-resolution RGB-D, tracks and inexpensive numeric body state."""

import json
import os
from pathlib import Path
import shutil
import numpy as np
from tools.util.motion_common import _capture_payload
from tools.util.settings import write_json


def latest_link(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".new")
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(os.path.relpath(source, target.parent))
    temporary.replace(target)


def diagnostic_images(result, history):
    """Preserve planner/wrist evidence in addition to current camera pixels."""
    images, seen = [], set()
    keys = {
        "render_image_path",
        "render_image_3d_path",
        "grasp_volume_three_views_path",
    }

    def add(path, label):
        if isinstance(path, str) and path and path not in seen:
            seen.add(path)
            images.append(
                history.image(
                    path,
                    "DIAGNOSTIC "
                    + label
                    + ". This is a plan/frozen-geometry visualization, not a live camera to click.",
                )
            )

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in keys:
                    add(child, key)
                elif key == "debug_images":
                    for label, path in (
                        child.items()
                        if isinstance(child, dict)
                        else enumerate(child or [])
                    ):
                        add(path, str(label))
                elif isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(result)
    return images


def publish_state(runtime, history):
    """Save public joints without loading meshes or rendering any images."""
    body = history.path / "observation/proprioception"
    body.mkdir(parents=True, exist_ok=True)
    joints = {
        f"{side}_joint{i + 1}": value
        for side in ("left", "right")
        for i, value in enumerate(runtime.world.arm_qpos_list(side))
    }
    for side in ("left", "right"):
        proxy = runtime.world.gripper_qpos_list(side)[0]
        for number in (7, 8):
            joints[f"{side}_joint{number}"] = -0.01 + 0.054 * proxy
    state = {
        "sequence": runtime.adapter.status()["sequence"],
        "episode_id": runtime.world.episode_id(),
        "public_state": runtime.adapter.observation().get("state", {}),
        "render_joint_positions": joints,
        "gripper_note": "Finger position is inferred from the normalized public gripper proxy; not contact evidence.",
    }
    write_json(body / "state.json", state)
    latest_link(body / "state.json", runtime.root / "perception/proprioception/state.json")
    # Only an explicit read_proprioception call may publish a current body PNG.
    # Its immutable historical copy remains available after this link expires.
    (runtime.root / "perception/proprioception/three_views.png").unlink(missing_ok=True)
    return body, state


def publish(runtime, history, result):
    ctx = runtime.context()
    root = runtime.root
    captures = {}
    images = diagnostic_images(result, history)
    sequence = runtime.adapter.status()["sequence"]
    # Finish with the fresh head view; historical plan overlays precede it.
    for role in ("left_wrist", "right_wrist", "head"):
        try:
            existing = result.get("observation") or result
            payload = (
                existing
                if existing.get("feed") == role and existing.get("image_id")
                else _capture_payload(ctx, role=role, session_id=runtime.session_id)
            )
            snap = runtime.snapshots.load(runtime.session_id, payload["image_id"])
            if snap.sequence != sequence:
                raise ValueError("Capture and current observation sequence differ")
            folder = history.path / "observation" / role
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snap.paths["rgb"], folder / "rgb.png")
            if snap.depth is not None:
                np.save(folder / "depth.npy", snap.depth, allow_pickle=False)
                if snap.paths.get("depth_visual"):
                    shutil.copy2(snap.paths["depth_visual"], folder / "depth.png")
            metadata = {
                "image_id": snap.image_id,
                "episode_id": snap.episode_id,
                "sequence": snap.sequence,
                "role": role,
                **snap.camera_metadata(),
                "click_coordinates": "normalized 0..1000; top left origin",
                "rgb": str(folder / "rgb.png"),
                "depth_m": str(folder / "depth.npy")
                if snap.depth is not None
                else None,
            }
            write_json(folder / "camera.json", metadata)
            latest_link(folder, root / "perception/exteroception" / role)
            captures[role] = metadata
            # Deliver the original tool's visual, including useful wrist overlays.
            if role == "head" or role == result.get("feed"):
                path = (
                    payload.get("rgb_main_path")
                    or payload.get("rgb_path")
                    or snap.paths["rgb"]
                )
                images.append(
                    history.image(
                        path,
                        f"CURRENT {role} image_id={snap.image_id}; normalized coordinates 0..1000; sequence={sequence}. Inspect these pixels anew before clicking.",
                    )
                )
        except (KeyError, ValueError, OSError, RuntimeError) as exc:
            if role == "head":
                raise
            captures[role] = {"error": str(exc)}
    body, _ = publish_state(runtime, history)
    tracks = {
        "sequence": sequence,
        "episode_id": runtime.world.episode_id(),
        "status": runtime.tracker.status(),
        "memory": runtime.tracker.memory_fields(),
    }
    write_json(history.path / "observation/tracks.json", tracks)
    latest_link(
        history.path / "observation/tracks.json",
        root / "perception/exteroception/tracks.json",
    )
    write_json(
        root / "perception/latest.json",
        {
            "sequence": sequence,
            "episode_id": runtime.world.episode_id(),
            "cameras": captures,
            "proprioception": str(body / "state.json"),
            "body_image": None,
            "body_image_instruction": "Run .venv/bin/python tools/read_proprioception.py '{}' when a body view is needed; it is not rendered automatically.",
            "tracks": str(history.path / "observation/tracks.json"),
        },
    )
    state_file = root / ".runtime/state.json"
    session_state = json.loads(state_file.read_text())
    session_state["latest_image_id"] = captures["head"]["image_id"]
    write_json(state_file, session_state)
    result.update(
        images=images,
        latest_head_image_id=captures["head"]["image_id"],
        perception=str(root / "perception/latest.json"),
    )
    return result
