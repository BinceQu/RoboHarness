"""Capture tools backed only by evaluator-supplied observations."""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from .task_memory import task_memory_fields


def camera_intrinsics(role: str, width: int, height: int) -> dict[str, Any]:
    if role == "head":
        fx = fy = 306.0 * float(width) / 720.0
    else:
        fx = fy = 388.6639 * float(width) / 480.0
    focal_length = 17.0
    horizontal_aperture = focal_length * float(width) / fx
    return {
        "image_width": int(width),
        "image_height": int(height),
        "focal_length": float(focal_length),
        "horizontal_aperture": float(horizontal_aperture),
        "fx": float(fx),
        "fy": float(fy),
        "cx": (float(width) - 1.0) * 0.5,
        "cy": (float(height) - 1.0) * 0.5,
    }


def depth_visual(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.isfinite(arr) & (arr > 0.0)
    visual = np.zeros(arr.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(arr[valid], [2.0, 98.0])
        if float(hi) <= float(lo):
            hi = float(lo) + 1e-6
        scaled = 255.0 * (np.clip(arr, lo, hi) - lo) / (hi - lo)
        visual[valid] = np.asarray(255.0 - scaled[valid], dtype=np.uint8)
    return cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)


def capture_from_evaluator(
    ctx,
    adapter,
    *,
    session_id: str,
    role: str,
):
    from behavior_interface import agent_runs

    frames = adapter.camera_frames()
    depths = adapter.camera_depth_frames()
    frame = frames.get(role)
    tool_name = "capture" if role == "head" else f"capture_{role}_camera"
    if frame is None:
        ctx.set_result(
            {
                "ok": False,
                "tool": tool_name,
                "feed": role,
                "error": f"evaluator has not supplied {role} RGB yet",
            }
        )
        yield ctx.world.hold_action()
        return

    session = str(session_id or "").strip()
    if not session:
        ctx.set_result({"ok": False, "error": "session_id is required"})
        yield ctx.world.hold_action()
        return

    agent_runs.ensure_session(session)
    image_id = agent_runs.next_image_id(session)
    rgb_suffix = ".png" if role == "head" else f".{role}.png"
    rgb_path = agent_runs.image_path(session, image_id, rgb_suffix)
    if not cv2.imwrite(rgb_path, frame):
        ctx.set_result({"ok": False, "error": f"failed to write {rgb_path}"})
        yield ctx.world.hold_action()
        return

    modalities: dict[str, str] = {}
    depth = depths.get(role)
    depth_visual_path = None
    if depth is not None:
        depth_suffix = ".depth.npy" if role == "head" else f".{role}.depth.npy"
        visual_suffix = ".depth.png" if role == "head" else f".{role}.depth.png"
        depth_path = agent_runs.image_path(session, image_id, depth_suffix)
        depth_visual_path = agent_runs.image_path(session, image_id, visual_suffix)
        np.save(depth_path, np.asarray(depth, dtype=np.float32))
        cv2.imwrite(depth_visual_path, depth_visual(depth))
        modalities["depth_linear"] = depth_path

    relative_pose = adapter.camera_relative_poses().get(role)
    if relative_pose is None:
        camera_pose = {
            "pos": [0.0, 0.0, 0.0],
            "quat": [0.0, 0.0, 0.0, 1.0],
            "frame": "local_command_odometry",
            "pose_available": False,
        }
    else:
        camera_pose = ctx.world.local_pose_from_robot_relative(
            relative_pose["pos"],
            relative_pose["quat"],
        )
        camera_pose["pose_available"] = True
    height, width = frame.shape[:2]
    camera = {
        **camera_pose,
        **camera_intrinsics(role, width, height),
        "source": "official_evaluator_cam_rel_poses",
    }
    robot_state = {
        "base_pose": ctx.world.robot_pose().as_dict(),
        "eef_left": ctx.world.eef_pose("left"),
        "eef_right": ctx.world.eef_pose("right"),
        "trunk_qpos": ctx.world.trunk_qpos().astype(float).tolist(),
        "arm_left_qpos": ctx.world.arm_qpos_list("left"),
        "arm_right_qpos": ctx.world.arm_qpos_list("right"),
        "gripper_left_qpos": ctx.world.gripper_qpos_list("left"),
        "gripper_right_qpos": ctx.world.gripper_qpos_list("right"),
        "frame": "local_command_odometry",
    }
    omitted = [
        "segmentation",
        "normal",
        "object_pose",
        "scene_graph",
        "BDDL_live_state",
        "simulator_mesh",
    ]
    task_name = str(getattr(ctx, "task_name", "") or "")
    memory = {
        **task_memory_fields(task_name),
        "image_id": image_id,
        "observation_policy": "official_allowlist",
        "available": [
            "rgb",
            *([] if depth is None else ["depth_linear"]),
            "proprioception",
            "camera_relative_pose",
        ],
        "omitted": omitted,
    }
    meta = {
        "image_id": image_id,
        "session_id": session,
        "rgb": {role: os.path.basename(rgb_path)},
        "modalities": {
            key: os.path.basename(path) for key, path in modalities.items()
        },
        "camera": camera,
        "robot": robot_state,
        "observation_policy": "official_allowlist",
        "evaluator_sequence": adapter.status().get("sequence"),
    }
    agent_runs.save_image_meta(session, image_id, meta)

    result = {
        "ok": True,
        "tool": tool_name,
        "tool_version": "official_v1",
        "feed": role,
        "image_id": image_id,
        "image_width": int(width),
        "image_height": int(height),
        "rgb_main": agent_runs.file_to_data_url(rgb_path),
        "rgb_main_path": rgb_path,
        "rgb_path": rgb_path,
        "camera": camera,
        "robot": robot_state,
        "modalities": list(modalities),
        "forbidden_modalities_omitted": omitted,
        "memory": memory,
        "memory_text": (
            "Official evaluator observation: RGB"
            + (", depth_linear" if depth is not None else "")
            + ", proprioception, camera-relative pose. Simulator truth omitted."
        ),
    }
    if depth_visual_path is not None:
        result.update(
            {
                "depth_path": modalities["depth_linear"],
                "depth_visual_path": depth_visual_path,
                "depth": agent_runs.file_to_data_url(depth_visual_path),
            }
        )
    ctx.set_result(result)
    yield ctx.world.hold_action()
