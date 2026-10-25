"""capture_head_camera

拍一张 head camera 主视图，返回 image_id（后续点选/规划都基于它）。

How this tool works:
Read the current evaluator head RGB-D, camera pose and proprioception without advancing physics. Freeze that coherent observation as files, register its frame identity, and return the image. Only this head image can be used for pixel selection.

This read-only tool freezes the latest evaluator observation. It sends no
joint commands and leaves all held control targets unchanged. RGB-D helpers
live in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.camera_frames import _capture_robot_state
from tools.util.geometry.base_path_overlay_local import compose_nearby_object_warning
from tools.util.geometry.task_memory import task_memory_fields
from tools.util.geometry.tracked_object_distance import TrackObjectDistanceBindingError
from tools.util.geometry.visualization_local import render_head_path_overlay
from tools.util.motion_common import _camera_intrinsics
from tools.util.motion_common import _depth_visual
from tools.util.motion_common import _ensure_session
from tools.util.motion_common import _file_to_data_url
from tools.util.motion_common import _image_path
from tools.util.motion_common import _next_image_id
from tools.util.motion_common import _save_image_meta
from tools.util.motion_common import _set_result
from typing import Any
import cv2
import numpy as np
import os
from tools.util.tool import main

NAME = 'capture_head_camera'
SCHEMA = {'type': 'object', 'properties': {}, 'additionalProperties': False}


def actions(robot, session_id: str, **kwargs):
    """Freeze one coherent RGB-D observation and its robot state."""
    return _capture_head_camera_sync(robot, session_id, **kwargs)


def _stamp_selected_minimap_on_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Keep tool-owned captures independent of process-global map renderers.

    The interface display layer may decorate its own copy with a minimap.  An
    official-v2 tool capture must remain inside this package and may only use
    its evaluator RGB-D/path overlay, so this hook deliberately preserves the
    local image unchanged.
    """

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("minimap display input must be an HxWx3 image")
    return image



def _capture_head_camera_sync(ctx, session_id: str, **kwargs):
    del kwargs
    role = "head"
    adapter = getattr(ctx.world, "_official_adapter", None)
    if adapter is None:
        _set_result(
            ctx,
            "capture_head_camera",
            {"ok": False, "error": "official evaluator adapter is unavailable"},
        )
        return
    frame = adapter.camera_frame(role)
    depth = adapter.camera_depth_frame(role)
    evaluator_sequence = adapter.status().get("sequence")
    session = str(session_id or "").strip()
    if frame is None:
        _set_result(
            ctx,
            "capture_head_camera",
            {"ok": False, "feed": role, "error": "head RGB is unavailable"},
        )
        return
    if not session:
        _set_result(
            ctx,
            "capture_head_camera",
            {"ok": False, "error": "session_id is required"},
        )
        return
    _ensure_session(session)
    image_id = _next_image_id(session)
    rgb_path = _image_path(session, image_id, ".png")
    if not cv2.imwrite(rgb_path, frame):
        _set_result(
            ctx,
            "capture_head_camera",
            {"ok": False, "error": f"failed to write {rgb_path}"},
        )
        return
    modalities: dict[str, str] = {}
    depth_visual_path = None
    if depth is not None:
        depth_path = _image_path(session, image_id, ".depth.npy")
        depth_visual_path = _image_path(session, image_id, ".depth.png")
        np.save(depth_path, np.asarray(depth, dtype=np.float32))
        cv2.imwrite(depth_visual_path, _depth_visual(depth))
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
    robot_state = _capture_robot_state(ctx)
    tracking_binding: dict[str, Any]
    distance_manager = getattr(
        ctx.world,
        "_official_tracked_object_distances",
        None,
    )
    if distance_manager is None:
        tracking_binding = {
            "ok": False,
            "trackable": False,
            "reason": "tracker_unavailable",
            "error": "official dynamic distance tracker is unavailable",
        }
    elif depth is None:
        tracking_binding = {
            "ok": False,
            "trackable": False,
            "reason": "missing_depth_linear",
            "error": "head depth_linear is unavailable for frame binding",
        }
    else:
        try:
            tracking_binding = distance_manager.register_capture(
                session_id=session,
                image_id=image_id,
                observation_sequence=evaluator_sequence,
                episode_id=robot_state["episode_id"],
                image_shape=(height, width),
            )
        except TrackObjectDistanceBindingError as exc:
            tracking_binding = {
                "ok": False,
                "trackable": False,
                "reason": exc.reason,
                "error": str(exc),
            }
        except (TypeError, ValueError) as exc:
            tracking_binding = {
                "ok": False,
                "trackable": False,
                "reason": "binding_registration_failed",
                "error": str(exc),
            }
    path_overlay_path = _image_path(session, image_id, ".path.png")
    if (
        depth is not None
        and bool(camera.get("pose_available"))
    ):
        base_path_overlay = render_head_path_overlay(
            rgb_path=rgb_path,
            depth_linear=depth,
            camera=camera,
            robot=robot_state,
            output_path=path_overlay_path,
        )
    else:
        base_path_overlay = {
            "ok": False,
            "error": (
                "v2 base-path HUD requires evaluator depth_linear "
                "and camera-relative pose"
            ),
            "path": path_overlay_path,
        }
    display_rgb_path = (
        path_overlay_path
        if bool(base_path_overlay.get("ok"))
        else rgb_path
    )
    spatial_map_overlay_applied = False
    try:
        display_bgr = cv2.imread(display_rgb_path, cv2.IMREAD_COLOR)
        if display_bgr is not None:
            stamped_bgr = _stamp_selected_minimap_on_bgr(display_bgr)
            spatial_map_overlay_applied = not np.array_equal(
                stamped_bgr, display_bgr
            )
            if spatial_map_overlay_applied:
                cv2.imwrite(display_rgb_path, stamped_bgr)
    except Exception:
        spatial_map_overlay_applied = False
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
            f"{role}_path_overlay": os.path.basename(display_rgb_path),
        },
        "modalities": {
            key: os.path.basename(path) for key, path in modalities.items()
        },
        "overlays": {
            "base_forward_path": os.path.basename(display_rgb_path),
        },
        "base_path_overlay": base_path_overlay,
        "spatial_map_overlay_applied": spatial_map_overlay_applied,
        "camera": camera,
        "robot": robot_state,
        "track_object_distance_binding": tracking_binding,
        "observation_policy": "official_allowlist",
        "evaluator_sequence": evaluator_sequence,
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
        "base_path_overlay": base_path_overlay,
        "spatial_map_overlay_applied": spatial_map_overlay_applied,
        "camera": camera,
        "robot": robot_state,
        "track_object_distance_binding": tracking_binding,
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
    hint = ""
    eef_near = ""
    nearby = ""
    if isinstance(base_path_overlay, dict):
        hint = str(base_path_overlay.get("chassis_forward_2m") or "").strip()
        eef_near = str(base_path_overlay.get("eef_near_0.1m") or "").strip()
        nearby = str(base_path_overlay.get("nearby_object_warning") or "").strip()
    if hint:
        result["chassis_forward_2m"] = hint
    if eef_near:
        result["eef_near_0.1m"] = eef_near
    if not nearby:
        # 模型预警只报底盘；eef_near 留给核对，先不并进去。
        nearby = compose_nearby_object_warning(hint)
    if nearby:
        result["nearby_object_warning"] = nearby
    if depth_visual_path is not None:
        result.update(
            {
                "depth_path": modalities["depth_linear"],
                "depth_visual_path": depth_visual_path,
                "depth": _file_to_data_url(depth_visual_path),
            }
        )
    _set_result(ctx, "capture_head_camera", result)
    return result


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
