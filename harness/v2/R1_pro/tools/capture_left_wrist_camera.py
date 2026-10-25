"""capture_left_wrist_camera

拍左腕 RGB + depth，不追加 head capture。图中红色图层对应左夹爪两爪中间区域；红色像素表示非机器人可见 3D 点严格落在该区域内；底边 x(m) 标尺以夹爪开口中心为 0，向右为正，范围 ±0.02m，每格 0.01m。

How this tool works:
Read the current evaluator left_wrist RGB-D, camera pose and proprioception without advancing physics. Freeze that coherent observation as files, register its frame identity, and return the image. Wrist images are for inspection, not pixel-click inputs.

This read-only tool freezes the latest evaluator observation. It sends no
joint commands and leaves all held control targets unchanged. RGB-D helpers
live in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.camera_frames import _capture_robot_state
from tools.util.camera_frames import _freeze_wrist_observation
from tools.util.camera_frames import _wrist_capture_alignment_report
from tools.util.geometry.task_memory import task_memory_fields
from tools.util.geometry.visualization_local import render_wrist_grasp_volume_overlay
from tools.util.motion_common import _camera_intrinsics
from tools.util.motion_common import _depth_visual
from tools.util.motion_common import _ensure_session
from tools.util.motion_common import _file_to_data_url
from tools.util.motion_common import _image_path
from tools.util.motion_common import _next_image_id
from tools.util.motion_common import _save_image_meta
from tools.util.motion_common import _set_result
import cv2
import numpy as np
import os
from tools.util.tool import main

NAME = 'capture_left_wrist_camera'
SCHEMA = {'type': 'object', 'properties': {}, 'additionalProperties': False}


def actions(robot, session_id: str, **kwargs):
    """Freeze one coherent RGB-D observation and its robot state."""
    return _capture_left_wrist_camera_sync(robot, session_id, **kwargs)


def _capture_left_wrist_camera_sync(ctx, session_id: str, **kwargs):
    del kwargs
    role = "left_wrist"
    adapter = getattr(ctx.world, "_official_adapter", None)
    if adapter is None:
        _set_result(
            ctx,
            "capture_left_wrist_camera",
            {"ok": False, "error": "official evaluator adapter is unavailable"},
        )
        return
    try:
        frozen_observation = _freeze_wrist_observation(adapter, role)
    except (RuntimeError, TypeError, ValueError) as exc:
        _set_result(
            ctx,
            "capture_left_wrist_camera",
            {"ok": False, "feed": role, "error": str(exc)},
        )
        return
    frame = frozen_observation.frame
    depth = frozen_observation.depth
    capture_settle = _wrist_capture_alignment_report(
        proprio=frozen_observation.proprio,
        side="left",
        sequence=frozen_observation.sequence,
    )
    session = str(session_id or "").strip()
    if frame is None:
        _set_result(
            ctx,
            "capture_left_wrist_camera",
            {"ok": False, "feed": role, "error": "left wrist RGB is unavailable"},
        )
        return
    if not session:
        _set_result(
            ctx,
            "capture_left_wrist_camera",
            {"ok": False, "error": "session_id is required"},
        )
        return
    if frozen_observation.proprio is None:
        _set_result(
            ctx,
            "capture_left_wrist_camera",
            {
                "ok": False,
                "feed": role,
                "error": (
                    "left wrist capture requires proprioception from the same "
                    "evaluator observation as RGB-D"
                ),
            },
        )
        return
    _ensure_session(session)
    image_id = _next_image_id(session)
    rgb_path = _image_path(session, image_id, ".left_wrist.png")
    if not cv2.imwrite(rgb_path, frame):
        _set_result(
            ctx,
            "capture_left_wrist_camera",
            {"ok": False, "error": f"failed to write {rgb_path}"},
        )
        return
    modalities: dict[str, str] = {}
    depth_visual_path = None
    if depth is not None:
        depth_path = _image_path(
            session,
            image_id,
            ".left_wrist.depth.npy",
        )
        depth_visual_path = _image_path(
            session,
            image_id,
            ".left_wrist.depth.png",
        )
        np.save(depth_path, np.asarray(depth, dtype=np.float32))
        cv2.imwrite(depth_visual_path, _depth_visual(depth))
        modalities["depth_linear"] = depth_path
    relative_pose = frozen_observation.camera_relative_pose
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
        **_camera_intrinsics(role, width, height),
        "source": "official_evaluator_cam_rel_poses",
        "robot_relative_pose": (
            None
            if relative_pose is None
            else {
                "pos": list(relative_pose["pos"]),
                "quat": list(relative_pose["quat"]),
                "frame": "robot_base",
            }
        ),
    }
    robot_state = _capture_robot_state(
        ctx,
        proprio=frozen_observation.proprio,
    )
    robot_state["observation_sequence"] = frozen_observation.sequence
    grasp_overlay_path = _image_path(
        session,
        image_id,
        ".left_wrist.grasp_volume.png",
    )
    red_mask_path = _image_path(
        session,
        image_id,
        ".left_wrist.red_mask.npy",
    )
    if (
        depth is not None
        and bool(camera.get("pose_available"))
    ):
        grasp_zone_overlay = render_wrist_grasp_volume_overlay(
            rgb_path=rgb_path,
            depth_linear=depth,
            camera=camera,
            eef_pose=robot_state["eef_left"],
            gripper_qpos=robot_state.get("gripper_left_qpos"),
            gripper_qvel=robot_state.get("gripper_left_qvel"),
            output_path=grasp_overlay_path,
            mask_path=red_mask_path,
        )
    else:
        grasp_zone_overlay = {
            "ok": False,
            "error": (
                "red grasp-volume overlay requires evaluator depth_linear "
                "and camera-relative pose"
            ),
            "path": grasp_overlay_path,
            "mask_path": red_mask_path,
        }
    display_rgb_path = (
        grasp_overlay_path
        if bool(grasp_zone_overlay.get("ok"))
        else rgb_path
    )
    if bool(grasp_zone_overlay.get("ok")):
        modalities["red_grasp_volume_mask"] = red_mask_path
    omitted = [
        "segmentation",
        "normal",
        "object_pose",
        "scene_graph",
        "BDDL_live_state",
        "simulator_mesh",
    ]
    meta = {
        "image_id": image_id,
        "session_id": session,
        "rgb": {
            role: os.path.basename(rgb_path),
            f"{role}_grasp_volume_overlay": os.path.basename(
                display_rgb_path
            ),
        },
        "modalities": {
            key: os.path.basename(path) for key, path in modalities.items()
        },
        "overlays": {
            "wrist_grasp_volume": os.path.basename(display_rgb_path),
        },
        "grasp_zone_overlay": grasp_zone_overlay,
        "capture_settle": capture_settle,
        "camera": camera,
        "robot": robot_state,
        "observation_policy": "official_allowlist",
        "evaluator_sequence": frozen_observation.sequence,
    }
    _save_image_meta(session, image_id, meta)
    result = {
        "ok": True,
        "feed": role,
        "image_id": image_id,
        "image_width": int(width),
        "image_height": int(height),
        "rgb_main": _file_to_data_url(display_rgb_path),
        "rgb_main_path": display_rgb_path,
        "rgb_path": rgb_path,
        "rgb_overlay_path": display_rgb_path,
        "grasp_zone_overlay": grasp_zone_overlay,
        "capture_settle": capture_settle,
        "red_mask_path": (
            red_mask_path
            if bool(grasp_zone_overlay.get("ok"))
            else None
        ),
        "camera": camera,
        "robot": robot_state,
        "modalities": list(modalities),
        "forbidden_modalities_omitted": omitted,
        "memory": {
            **task_memory_fields(
                str(getattr(ctx, "task_name", "") or "")
            ),
            "image_id": image_id,
            "observation_policy": "official_allowlist",
            "available": [
                "rgb",
                *([] if depth is None else ["depth_linear"]),
                "proprioception",
                "camera_relative_pose",
            ],
            "omitted": omitted,
        },
    }
    if depth_visual_path is not None:
        result.update(
            {
                "depth_path": modalities["depth_linear"],
                "depth_visual_path": depth_visual_path,
                "depth": _file_to_data_url(depth_visual_path),
            }
        )
    _set_result(ctx, "capture_left_wrist_camera", result)
    return result


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
