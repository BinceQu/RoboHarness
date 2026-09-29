"""Direct wrist-camera RGB/depth capture with the two-finger grasp-zone overlay."""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from behavior_interface.camera_frames import rgb_array
from behavior_interface.skills import register_skill
from behavior_interface.skills.capture import (
    _camera_meta,
    _collect_robot_sensors,
    _ensure_modalities,
    _read_complete_frame,
    _refresh_camera_render_control,
    _robot_seg_ids_from_info,
    _save_depth,
    _save_rgb,
    _save_seg,
    _sensor_debug_names,
    _warm_sensor_obs,
    warm_obs_annotators,
)
from behavior_interface.skills.wrist_grasp_zone_overlay import (
    overlay_depth_inside_grasp_zone,
    project_opening_depth_bounds,
)
from behavior_interface.skills.plan_grasp_gripper_geom import normalize_gripper_qpos
from behavior_interface.skills.wrist_metric_ruler import draw_wrist_x_ruler
from behavior_interface.skills.vlm_lawn_dual import save_image_atomic, save_npy_atomic


def _to_np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _obs_rgb(obs: Dict[str, Any]) -> np.ndarray:
    arr = rgb_array(obs.get("rgb"))
    if arr is None:
        raise RuntimeError("wrist RGB buffer 缺失、为空或 shape 非法")
    return arr


def _save_rgb_array(rgb: np.ndarray, path: str) -> None:
    import cv2

    arr = rgb_array(rgb)
    if arr is None:
        raise RuntimeError("待保存 wrist RGB 为空或 shape 非法")
    if not save_image_atomic(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"wrist RGB 写入失败: {path}")


def _save_boundary_visual(
    depth_boundary: np.ndarray,
    *,
    boundary_vis_path: str,
) -> None:
    import cv2

    boundary = np.asarray(depth_boundary, dtype=np.float32)
    zone = np.isfinite(boundary) & (boundary > 0.0)
    vis = np.zeros(boundary.shape, dtype=np.uint8)
    if np.any(zone):
        lo = float(np.nanmin(boundary))
        hi = float(np.nanmax(boundary))
        denom = max(hi - lo, 1e-6)
        vis[zone] = np.rint((boundary[zone] - lo) / denom * 255.0).astype(np.uint8)
    save_image_atomic(boundary_vis_path, cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))


def _capture_wrist_camera(ctx, *, session_id: str, side: str, n_settle: int = 2) -> None:
    from behavior_interface import agent_runs

    world = ctx.world
    robot = getattr(world, "robot", None)
    feed = f"{side}_wrist"
    control = _refresh_camera_render_control()
    old_no_obs = bool(getattr(world, "_codex_fast_motion_no_obs", False))
    old_keep_cameras = bool(
        getattr(world, "_codex_keep_robot_camera_render_updates", False)
    )
    world._codex_keep_robot_camera_render_updates = True
    control.set_robot_camera_render_updates(world, True)
    world._codex_fast_motion_no_obs = True
    try:
        for _ in range(max(0, int(n_settle))):
            yield world.hold_action()
        try:
            import omnigibson as og

            for _ in range(2):
                og.sim.render()
        except Exception:
            og = None

        if robot is None:
            ctx.set_result({"ok": False, "tool": f"capture_{feed}_camera", "error": "robot 未加载"})
            yield world.hold_action()
            return

        sensors = _collect_robot_sensors(robot)
        sensor = sensors.get(feed)
        if sensor is None:
            ctx.set_result({
                "ok": False,
                "tool": f"capture_{feed}_camera",
                "feed": feed,
                "error": f"未找到 {feed} 相机",
                "sensors": _sensor_debug_names(robot),
            })
            yield world.hold_action()
            return

        if _ensure_modalities(sensor, ["rgb", "depth_linear", "seg_instance_id"]):
            yield world.hold_action()
            if og is not None:
                for _ in range(2):
                    og.sim.render()

    # —— 预热相机 annotator —— 与 head capture 同理：本 skill 可能紧接 no_obs
    #    运动而来，seg buffer 为空时直接 get_obs 会拿到空张量。这些渲染 tick 同时
    #    驱动纹理流式，并执行 vision_sensor 在 headless 下延后到「首次 capture /
    #    warmup」的相机参数传播；实测缺了这一步，加载/reset 后的首帧会偏糊。
        warm_obs_annotators(world, ctx, log_tag=f"capture.{feed}")
        _warm_sensor_obs(sensor, ctx, log_tag=f"capture.{feed}")

        obs, info, frame_errors = _read_complete_frame(
            sensor,
            ["rgb", "depth_linear", "seg_instance_id"],
            ctx,
            log_tag=f"capture.{feed}",
        )
        if frame_errors:
            ctx.set_result({
                "ok": False,
                "tool": f"capture_{feed}_camera",
                "feed": feed,
                "error": f"{feed} 相机帧未就绪: " + "; ".join(frame_errors),
            })
            yield world.hold_action()
            return

        agent_runs.ensure_session(session_id)
        image_id = agent_runs.next_image_id(session_id)
        img_dir = agent_runs.images_dir(session_id)
        os.makedirs(img_dir, exist_ok=True)
        rgb_path = agent_runs.image_path(session_id, image_id, f".{feed}.png")
        rgb_raw_path = agent_runs.image_path(session_id, image_id, f".{feed}.raw.png")
        depth_path = agent_runs.image_path(session_id, image_id, f".{feed}.depth.npy")
        depth_vis_path = agent_runs.image_path(session_id, image_id, f".{feed}.depth.png")
        seg_path = agent_runs.image_path(session_id, image_id, f".{feed}.seg.npy")
        seg_vis_path = agent_runs.image_path(session_id, image_id, f".{feed}.seg.png")
        near_path = agent_runs.image_path(session_id, image_id, f".{feed}.depth_near.npy")
        near_vis_path = agent_runs.image_path(
            session_id, image_id, f".{feed}.depth_near.png",
        )
        boundary_path = agent_runs.image_path(session_id, image_id, f".{feed}.depth_bianjie.npy")
        boundary_vis_path = agent_runs.image_path(
            session_id, image_id, f".{feed}.depth_bianjie.png",
        )
        zone_mask_path = agent_runs.image_path(
            session_id, image_id, f".{feed}.grasp_zone_mask.png",
        )
        red_mask_path = agent_runs.image_path(
            session_id, image_id, f".{feed}.red_mask.npy",
        )

        if not _save_rgb(obs, rgb_raw_path):
            ctx.set_result({
                "ok": False,
                "tool": f"capture_{feed}_camera",
                "feed": feed,
                "image_id": image_id,
                "error": f"{feed} RGB 渲染失败",
            })
            yield world.hold_action()
            return

        try:
            cam_meta: Dict[str, Any] = _camera_meta(sensor)
            cam_meta["feed"] = feed
        except Exception as e:
            ctx.log(f"capture {feed} WARN 取相机内外参失败: {e}")
            cam_meta = {"feed": feed}

        try:
            depth = _save_depth(obs, depth_path, depth_vis_path)
            if depth is None:
                raise RuntimeError("wrist depth_linear buffer 缺失")
            seg = _save_seg(obs, info, seg_path, seg_vis_path)
            if seg is None:
                raise RuntimeError("wrist seg_instance_id buffer 缺失")
            segmentation_meta = _robot_seg_ids_from_info(info)
            robot_ids = np.asarray(segmentation_meta["robot_ids"], dtype=np.int64)
            if robot_ids.size == 0:
                raise RuntimeError(
                    "wrist seg_instance_id 未识别到 robot IDs，"
                    f"映射 keys={list((info or {}).keys())}"
                )
            robot_mask = np.isin(seg, robot_ids)
            rgb = _obs_rgb(obs)
            h_native, w_native = depth.shape
            cam_meta["image_width"] = int(w_native)
            cam_meta["image_height"] = int(h_native)
            eef_pose = world.eef_pose(arm=side)
            raw_gripper_qpos = world.gripper_qpos_list(side)
            if not raw_gripper_qpos:
                raise RuntimeError(f"无法读取 {side} 夹爪当前 finger qpos")
            gripper_qpos = list(normalize_gripper_qpos(raw_gripper_qpos))
            depth_near, depth_bianjie, zone_meta = project_opening_depth_bounds(
                eef_pos_world=np.asarray(eef_pose["pos"], dtype=np.float64),
                eef_quat_world=np.asarray(eef_pose["quat"], dtype=np.float64),
                camera_pos_world=np.asarray(cam_meta["pos"], dtype=np.float64),
                camera_quat_world=np.asarray(cam_meta["quat"], dtype=np.float64),
                focal_length=float(cam_meta["focal_length"]),
                horizontal_aperture=float(cam_meta["horizontal_aperture"]),
                image_width=int(w_native),
                image_height=int(h_native),
                gripper_qpos=gripper_qpos,
            )
            overlay_rgb, red_mask = overlay_depth_inside_grasp_zone(
                rgb,
                depth,
                depth_near,
                depth_bianjie,
                visible_inside_mask=~robot_mask,
            )
            depth_inside_bounds = (
                np.isfinite(depth_near)
                & (depth_near > 0.0)
                & np.isfinite(depth_bianjie)
                & (depth_bianjie >= depth_near)
                & np.isfinite(depth)
                & (depth > 0.0)
                & (depth >= depth_near)
                & (depth <= depth_bianjie)
            )
            robot_red_candidate_n = int((depth_inside_bounds & robot_mask).sum())
            save_npy_atomic(near_path, depth_near.astype(np.float32))
            save_npy_atomic(boundary_path, depth_bianjie.astype(np.float32))
            save_npy_atomic(red_mask_path, red_mask)
            _save_boundary_visual(
                depth_near,
                boundary_vis_path=near_vis_path,
            )
            _save_boundary_visual(
                depth_bianjie,
                boundary_vis_path=boundary_vis_path,
            )
            zone = (
                np.isfinite(depth_near)
                & (depth_near > 0.0)
                & np.isfinite(depth_bianjie)
                & (depth_bianjie >= depth_near)
            )
            save_image_atomic(zone_mask_path, zone.astype(np.uint8) * 255)
            try:
                overlay_rgb, ruler_meta = draw_wrist_x_ruler(
                    overlay_rgb,
                    eef_pos_world=np.asarray(eef_pose["pos"], dtype=np.float64),
                    eef_quat_world=np.asarray(eef_pose["quat"], dtype=np.float64),
                    camera_pos_world=np.asarray(cam_meta["pos"], dtype=np.float64),
                    camera_quat_world=np.asarray(cam_meta["quat"], dtype=np.float64),
                    focal_length=float(cam_meta["focal_length"]),
                    horizontal_aperture=float(cam_meta["horizontal_aperture"]),
                )
            except Exception as ruler_exc:
                ruler_meta = {
                    "ok": False,
                    "error": f"{type(ruler_exc).__name__}: {ruler_exc}",
                }
                ctx.log(f"capture {feed} WARN wrist x ruler failed: {ruler_exc}")
            _save_rgb_array(overlay_rgb, rgb_path)
            zone_meta["red_pixel_n"] = int(red_mask.sum())
            zone_meta["robot_pixel_n"] = int(robot_mask.sum())
            zone_meta["robot_red_candidate_n"] = robot_red_candidate_n
            zone_meta["red_overlay_alpha"] = 0.58
            zone_meta["red_pixel_semantics"] = (
                "pixel ray crosses the projected two-finger plan_grasp opening, "
                "depth_near <= visible depth_linear <= far depth_bianjie, and "
                "segmentation is not robot"
            )
        except Exception as e:
            ctx.set_result({
                "ok": False,
                "tool": f"capture_{feed}_camera",
                "feed": feed,
                "image_id": image_id,
                "error": f"{feed} depth/两爪中间区域红层生成失败: {e}",
                "rgb_raw_path": rgb_raw_path,
            })
            yield world.hold_action()
            return

        meta = {
            "image_id": image_id,
            "session_id": session_id,
            "rgb": {
                feed: os.path.basename(rgb_path),
                f"{feed}_raw": os.path.basename(rgb_raw_path),
            },
            "modalities": {
                "depth_linear": os.path.basename(depth_path),
                "seg_instance_id": os.path.basename(seg_path),
                "depth_near": os.path.basename(near_path),
                "depth_bianjie": os.path.basename(boundary_path),
                "grasp_zone_mask": os.path.basename(zone_mask_path),
                "red_mask": os.path.basename(red_mask_path),
            },
            "camera": cam_meta,
            "eef": {"arm": side, **eef_pose},
            "gripper_qpos_m": [float(v) for v in gripper_qpos],
            "segmentation": segmentation_meta,
            "grasp_zone_overlay": zone_meta,
            "metric_x_ruler": ruler_meta,
        }
        agent_runs.save_image_meta(session_id, image_id, meta)

        w_native = int(cam_meta.get("image_width", 0)) or None
        h_native = int(cam_meta.get("image_height", 0)) or None
        ctx.set_result({
            "ok": True,
            "tool": f"capture_{feed}_camera",
            "feed": feed,
            "image_id": image_id,
            "image_width": w_native,
            "image_height": h_native,
            "rgb": agent_runs.file_to_data_url(rgb_path),
            "rgb_main": agent_runs.file_to_data_url(rgb_path),
            "rgb_path": rgb_path,
            "rgb_raw_path": rgb_raw_path,
            "depth": agent_runs.file_to_data_url(depth_vis_path),
            "depth_path": depth_path,
            "depth_visual_path": depth_vis_path,
            "seg_path": seg_path,
            "seg_visual_path": seg_vis_path,
            "depth_near": agent_runs.file_to_data_url(near_vis_path),
            "depth_near_path": near_path,
            "depth_near_visual_path": near_vis_path,
            "depth_bianjie": agent_runs.file_to_data_url(boundary_vis_path),
            "depth_bianjie_path": boundary_path,
            "depth_bianjie_visual_path": boundary_vis_path,
            "grasp_zone_mask_path": zone_mask_path,
            "red_mask_path": red_mask_path,
            "modalities": [
                "depth_linear",
                "seg_instance_id",
                "depth_near",
                "depth_bianjie",
                "grasp_zone_mask",
                "red_mask",
            ],
            "grasp_zone_overlay": zone_meta,
            "metric_x_ruler": ruler_meta,
            "gripper_qpos_m": [float(v) for v in gripper_qpos],
            "camera": cam_meta,
        })
        yield world.hold_action()
    finally:
        world._codex_fast_motion_no_obs = old_no_obs
        world._codex_keep_robot_camera_render_updates = old_keep_cameras


@register_skill(
    "capture_left_wrist_camera",
    description=(
        "拍左腕 RGB + depth；红色图层标出左夹爪两爪中间区域内、"
        "可见 depth 位于该 UV 的 depth_near 与 depth_bianjie 之间的像素；"
        "底边显示以夹爪开口中心为 x=0 的米制横向标尺。"
    ),
)
def capture_left_wrist_camera(ctx, session_id: str, n_settle: int = 2):
    yield from _capture_wrist_camera(ctx, session_id=session_id, side="left", n_settle=n_settle)


@register_skill(
    "capture_right_wrist_camera",
    description=(
        "拍右腕 RGB + depth；红色图层标出右夹爪两爪中间区域内、"
        "可见 depth 位于该 UV 的 depth_near 与 depth_bianjie 之间的像素；"
        "底边显示以夹爪开口中心为 x=0 的米制横向标尺。"
    ),
)
def capture_right_wrist_camera(ctx, session_id: str, n_settle: int = 2):
    yield from _capture_wrist_camera(ctx, session_id=session_id, side="right", n_settle=n_settle)
