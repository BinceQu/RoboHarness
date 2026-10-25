"""Publish latest RGB/depth/tracks and measured joints, without rendering meshes."""
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from .coordinates import response_to_pixels
from .files import atomic_json, copy_latest, load_json, write_latest
from .model import proprio_positions


def freeze_camera(robot, role):
    from importlib import import_module
    if role not in {"head", "left_wrist", "right_wrist"}:
        raise ValueError("Unknown camera role")
    camera = import_module(f"tools.capture_{role}_camera")
    # Capture only reads and freezes the evaluator snapshot. Even building a
    # discarded hold action would mutate the controller's gripper targets.
    camera.actions(robot, robot.session_id)
    if not robot.result or robot.result.get("ok") is not True:
        raise RuntimeError(f"{role} capture failed: {robot.result}")
    return dict(robot.result)


def copy_file(source, destination):
    source, destination = Path(source), Path(destination)
    shutil.copyfile(source, destination)
    return {"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def write_joint_state(robot, folder):
    """Save measured joints from the current frozen observation; no mesh loading."""
    adapter = robot._official_adapter
    proprio = adapter.proprio_vector()
    snapshot = {"episode_id": robot.config["episode_id"], "sequence": adapter.current.sequence,
        "received_at": adapter.current.received_ts, "joint_positions": proprio_positions(proprio),
        "rotary_position_unit": "rad", "finger_position_unit": "m",
        "proprioception": proprio.tolist()}
    path = Path(folder) / "joint_state.json"
    atomic_json(path, snapshot)
    return path, snapshot


def depth_mm(depth):
    """Lossless to <1 mm for 0–65 m; invalid pixels become 0."""
    valid = np.isfinite(depth) & (depth > 0)
    return np.where(valid, np.clip(np.rint(np.where(valid, depth, 0) * 1000), 1, 65535), 0).astype(np.uint16)


def archive_once(robot, folder, name, write, payload):
    """Write `folder/name` unless identical bytes were archived by an earlier call."""
    index_path = Path(robot.config["trajectory_dir"]) / "archive-index.json"
    index = load_json(index_path, {})
    digest = hashlib.sha256(payload).hexdigest()
    path = Path(folder) / name
    if digest in index and Path(index[digest]).is_file():
        return {"path": index[digest], "sha256": digest, "duplicate_of_earlier_call": True}
    write(path)
    index[digest] = str(path)
    atomic_json(index_path, index)
    return {"path": str(path), "sha256": digest}


def publish(robot, history, head):
    """One manifest binds all files to the same evaluator observation.

    The workspace gets only the current readable files (RGB, overlay, 0–5 m depth
    preview, tracks, joints). The full per-call record (metric depth as uint16 mm
    npz, wrist RGB, manifest) goes to the trajectory dir, deduplicated by hash.
    """
    folder = history.calls / "observation"
    folder.mkdir()
    adapter = robot._official_adapter
    sequence = adapter.current.sequence
    perception = robot.workspace / "perception"
    exteroception = perception / "exteroception"
    paths, archive = {}, {}
    for role in ("head", "left_wrist", "right_wrist"):
        rgb = adapter.camera_rgb(role)
        depth = adapter.camera_depth_frame(role)
        if rgb is not None:
            path = exteroception / f"{role}_rgb.png"
            if role == "head":
                copy_latest(path, head["rgb_path"])
                archive["head_rgb"] = {"recordings_image_id": head["image_id"], "path": head["rgb_path"]}
            else:
                write_latest(path, lambda target, rgb=rgb: Image.fromarray(rgb).save(target))
                archive[f"{role}_rgb"] = archive_once(robot, folder, f"{role}_rgb.png",
                    lambda target, rgb=rgb: Image.fromarray(rgb).save(target), np.ascontiguousarray(rgb).tobytes())
            paths[f"{role}_rgb"] = {"path": str(path), "width": rgb.shape[1], "height": rgb.shape[0]}
        if depth is not None:
            valid = np.isfinite(depth) & (depth > 0)
            # Fixed 0–5 m preview scale; the metric measurement lives in the trajectory archive.
            gray = np.where(valid, np.clip(depth / 5., 0, 1) * 255, 0).astype(np.uint8)
            preview = exteroception / f"{role}_depth.png"
            write_latest(preview, lambda target, gray=gray: Image.fromarray(gray).save(target))
            paths[f"{role}_depth"] = {"path": str(preview), "unit": "m", "preview_range_m": [0, 5],
                "invalid_pixels": int((~valid).sum()),
                "note": "0-5 m grayscale preview only; metric depth is archived outside the workspace"}
            millimetres = depth_mm(depth)
            archive[f"{role}_depth"] = archive_once(robot, folder, f"{role}_depth_mm.npz",
                lambda target, millimetres=millimetres: np.savez_compressed(target, depth_mm=millimetres),
                millimetres.tobytes())
            archive[f"{role}_depth"].update(unit="mm", dtype="uint16", zero="invalid")
    if head.get("rgb_overlay_path"):
        copy_latest(exteroception / "head_overlay.png", head["rgb_overlay_path"])
        paths["head_overlay"] = {"path": str(exteroception / "head_overlay.png")}
        archive["head_overlay"] = {"recordings_image_id": head["image_id"], "path": head["rgb_overlay_path"]}
    joints_archive, snapshot = write_joint_state(robot, folder)
    joints_path = perception / "proprioception" / "joint_state.json"
    copy_latest(joints_path, joints_archive)
    tracks = response_to_pixels(robot.rpc("tracking", method="memory_fields"))
    track_payload = {"episode_id": robot.config["episode_id"], "sequence": sequence,
                     "current_head_image_id": head["image_id"], "tracking": tracks}
    track_path = exteroception / "tracks.json"
    atomic_json(track_path, track_payload)
    atomic_json(folder / "tracks.json", track_payload)
    manifest = {"session_id": robot.session_id, "episode_id": robot.config["episode_id"],
        "sequence": sequence, "received_at": adapter.current.received_ts, "published_at": time.time(),
        "head_image_id": head["image_id"], "images": paths,
        "joint_state": str(joints_path),
        "proprioception_render_tool": "python tools/read_proprioception.py",
        "tracks": str(track_path), "camera_relative_poses": adapter.camera_relative_poses()}
    atomic_json(folder / "manifest.json", {**manifest, "archive": archive})
    atomic_json(perception / "latest.json", manifest)
    state_path = robot.state_dir / "state.json"
    state = load_json(state_path, {})
    state.update(latest_image_id=head["image_id"], latest_sequence=sequence,
                 latest_head_path=str(exteroception / "head_rgb.png"))
    atomic_json(state_path, state)
    return manifest
