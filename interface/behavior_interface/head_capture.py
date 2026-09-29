"""head 相机原图导出：不做 resize/crop，保持 sensor 原始宽高比。"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np

# BEHAVIOR Challenge 官方 R1Pro head（zed）相机，与 eval_utils / replay_obs 一致
HEAD_IMAGE_WIDTH = 720
HEAD_IMAGE_HEIGHT = 720
HEAD_FOCAL_LENGTH = 17.0
HEAD_HORIZONTAL_APERTURE = 40.0

# env.reset 后从真实 USD 快照的 parent 系位姿（勿手写 joylo 外参，与内置 Camera:0 不一致）
_HEAD_FACTORY_MOUNT: Optional[Tuple[Any, Any]] = None
_HEAD_SENSOR_MARKERS = ("zed_link", "zed", "head", "camera")
_WRIST_SENSOR_MARKERS = ("left_realsense", "right_realsense", "left_wrist", "right_wrist", "wrist")


def get_head_sensor(world) -> Optional[Any]:
    head_candidates = []
    rgb_candidates = []
    try:
        for sname, s in world.robot.sensors.items():
            modalities = list(getattr(s, "modalities", []) or [])
            has_rgb = "rgb" in modalities
            lname = str(sname).lower()
            if any(marker in lname for marker in _HEAD_SENSOR_MARKERS) and not any(
                marker in lname for marker in _WRIST_SENSOR_MARKERS
            ):
                if has_rgb:
                    return s
                head_candidates.append(s)
            if has_rgb:
                rgb_candidates.append(s)
    except Exception:
        pass
    if head_candidates:
        return head_candidates[0]
    if len(rgb_candidates) == 1:
        return rgb_candidates[0]
    return None


def configure_head_sensor(sensor, *, env=None) -> bool:
    """将 head VisionSensor 对齐 Challenge 数据采集配置（720×720，ha=40）。有变更才 reload obs。"""
    changed = False
    try:
        cur_ha = float(getattr(sensor, "horizontal_aperture", 0.0))
        if abs(cur_ha - HEAD_HORIZONTAL_APERTURE) > 1e-3:
            sensor.horizontal_aperture = HEAD_HORIZONTAL_APERTURE
            changed = True
    except Exception:
        sensor.horizontal_aperture = HEAD_HORIZONTAL_APERTURE
        changed = True
    try:
        w, h = head_camera_size(sensor)
    except Exception:
        w, h = 0, 0
    if w != HEAD_IMAGE_WIDTH:
        sensor.image_width = HEAD_IMAGE_WIDTH
        changed = True
    if h != HEAD_IMAGE_HEIGHT:
        sensor.image_height = HEAD_IMAGE_HEIGHT
        changed = True
    if changed and env is not None:
        env.load_observation_space()
    return changed


def configure_head_sensor_for_world(world, *, env=None) -> bool:
    head = get_head_sensor(world)
    if head is None:
        return False
    return configure_head_sensor(head, env=env)


def setup_head_after_env_reset(world, *, env=None, log_fn=None) -> None:
    """env.reset() 后统一配置 head：对齐内参 → 快照挂载 → 仅在脱节时复位。"""
    changed = configure_head_sensor_for_world(world, env=env)
    if log_fn and changed:
        log_fn(
            f"head 相机已对齐 Challenge：{HEAD_IMAGE_WIDTH}x{HEAD_IMAGE_HEIGHT} "
            f"horizontal_aperture={HEAD_HORIZONTAL_APERTURE}"
        )
    if snapshot_head_mount_pose(world):
        if log_fn:
            log_fn("head 相机出厂挂载位姿已快照")
    if reset_head_sensor_to_mount(world, force=False):
        if log_fn:
            log_fn("head 相机挂载已恢复（parent 系快照）")


def _find_zed_link(robot) -> Optional[Any]:
    try:
        for name, link in robot.links.items():
            if "zed_link" in name:
                return link
    except Exception:
        pass
    return None


def _pose_xyz(pose_pair) -> np.ndarray:
    p = pose_pair[0]
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
    return np.asarray(p, dtype=np.float64).reshape(-1)[:3]


def snapshot_head_mount_pose(world) -> bool:
    """在 env.reset 后记录 head 相对 zed_link 的 parent 系位姿，供后续复位。"""
    global _HEAD_FACTORY_MOUNT
    head = get_head_sensor(world)
    if head is None:
        return False
    try:
        _HEAD_FACTORY_MOUNT = head.get_position_orientation(frame="parent")
        return True
    except Exception:
        return False


def head_factory_mount_pose(world) -> Optional[Tuple[Any, Any]]:
    """Return the fixed camera pose in its parent frame.

    This is calibration data, not a camera or robot world pose. The reset-time
    snapshot is preferred so a later accidental sensor detach cannot pollute
    observation-only robot-base geometry.
    """
    if _HEAD_FACTORY_MOUNT is not None:
        return _HEAD_FACTORY_MOUNT
    head = get_head_sensor(world)
    if head is None:
        return None
    try:
        return head.get_position_orientation(frame="parent")
    except Exception:
        return None


def is_head_sensor_detached(world, *, max_delta_m: float = 0.15) -> bool:
    """head 世界位姿与 zed_link 偏差过大 → 曾被 set_position_orientation(world) 污染。"""
    head = get_head_sensor(world)
    link = _find_zed_link(world.robot)
    if head is None or link is None:
        return False
    try:
        hp = _pose_xyz(head.get_position_orientation())
        lp = _pose_xyz(link.get_position_orientation())
        return float(np.linalg.norm(hp - lp)) > max_delta_m
    except Exception:
        return False


def reset_head_sensor_to_mount(world, *, ctx=None, force: bool = False) -> bool:
    """仅在有出厂快照且传感器已脱节时，恢复 parent 系挂载（不猜四元数）。"""
    if _HEAD_FACTORY_MOUNT is None:
        if ctx is not None and force:
            ctx.log("  [head] 跳过复位：无出厂快照，需重启 interface 后 env.reset")
        return False
    if not force and not is_head_sensor_detached(world):
        return False
    head = get_head_sensor(world)
    if head is None:
        return False
    try:
        import torch as th
        import omnigibson as og

        p, q = _HEAD_FACTORY_MOUNT
        pos = p if hasattr(p, "detach") else th.tensor(p, dtype=th.float32)
        quat = q if hasattr(q, "detach") else th.tensor(q, dtype=th.float32)
        head.set_position_orientation(position=pos, orientation=quat, frame="parent")
        for _ in range(4):
            og.sim.render()
        if ctx is not None:
            ctx.log("  [head] 已恢复 zed_link 出厂挂载（parent 系快照）")
        return True
    except Exception as e:
        if ctx is not None:
            ctx.log(f"  [head] 复位挂载失败: {e}")
        return False


def head_camera_size(sensor) -> Tuple[int, int]:
    w = int(getattr(sensor, "image_width", HEAD_IMAGE_WIDTH))
    h = int(getattr(sensor, "image_height", HEAD_IMAGE_HEIGHT))
    return w, h


def head_intrinsics_dict(sensor) -> Dict[str, Any]:
    """从 head sensor 读取反投影用内参（读不到时用 Challenge 官方默认）。"""
    w, h = head_camera_size(sensor)
    return {
        "focal_length": float(getattr(sensor, "focal_length", HEAD_FOCAL_LENGTH)),
        "horizontal_aperture": float(getattr(sensor, "horizontal_aperture", HEAD_HORIZONTAL_APERTURE)),
        "image_width": w,
        "image_height": h,
    }


def head_intrinsics_tuple(sensor) -> Tuple[float, float, int, int]:
    d = head_intrinsics_dict(sensor)
    return d["focal_length"], d["horizontal_aperture"], d["image_width"], d["image_height"]


def head_intrinsics_fallback() -> Tuple[float, float, int, int]:
    return HEAD_FOCAL_LENGTH, HEAD_HORIZONTAL_APERTURE, HEAD_IMAGE_WIDTH, HEAD_IMAGE_HEIGHT


def capture_head_png(world, out_path: str, *, n_render: int = 6) -> Optional[Dict[str, Any]]:
    """从 head/zed 取一帧 RGB 原图写入 out_path（不缩放、不裁剪）。"""
    from behavior_interface.camera_render_control import set_robot_camera_render_updates
    from behavior_interface.skills.vlm_lawn_dual import _save_rgb

    head = get_head_sensor(world)
    if head is None:
        return None
    try:
        import omnigibson as og
        set_robot_camera_render_updates(world, True)
        if "rgb" not in list(getattr(head, "modalities", []) or []):
            head.add_modality("rgb")
        lock = getattr(world, "_codex_camera_io_lock", None)
        with lock if lock is not None else nullcontext():
            for _ in range(max(1, n_render)):
                og.sim.render()
            obs, _ = head.get_obs()
    except Exception:
        return None
    if not _save_rgb(obs, out_path):
        return None
    w, h = head_camera_size(head)
    return {"image_width": w, "image_height": h, "path": out_path}
