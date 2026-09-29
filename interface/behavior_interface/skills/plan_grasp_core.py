"""plan_grasp 共享后端：会话管理、2D→3D 反解、EEF pose 生成。"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np

from behavior_interface import agent_runs
from behavior_interface.head_capture import HEAD_HORIZONTAL_APERTURE
from behavior_interface.skills.vlm_grasp_verify import (
    _depth_to_world_point,
    _move_cam,
    _pixel_to_world_ray,
    _world_to_pixel,
)
from behavior_interface.skills.vlm_lawn_dual import (
    CAM_D,
    CAM_SIDE,
    CAM_UP,
    _build_pointcloud,
    _ray_pcd_hit,
    _save_depth,
    _save_rgb,
    _save_seg,
    save_npy_atomic,
)
from behavior_interface.skills.vlm_scene_grasp import (
    _point_to_pixel,
)
from behavior_interface.skills.plan_grasp_gripper_fit import (
    VIZ_BALL_RADIUS,
    build_object_pointcloud,
    compute_eef_from_pcd_gripper_fit,
    grip_fit_has_volume,
    resolve_hit_on_surface,
)

PLAN_GRASP_VIEWS = (
    "left_upper_front",
    "left_upper_back",
    "right_upper_front",
    "right_upper_back",
)
PLAN_GRASP_VIEW = "left_upper_front"
# 前/后沿门宽方向（tangent）偏移
CAM_FB = 0.45
VIZ_BALL_PATH = "/World/plan_grasp_hit_ball"
VIZ_RAY_HIT_BALL_PATH = "/World/plan_eef_ray_hit_ball"
VIZ_GRIPPER_ROOT = "/World/plan_grasp_gripper"
VIZ_EEF_GRIPPER_ROOT = "/World/plan_eef_gripper"
# plan 标注用 2D 叠加；下列 prim 仅作遗留清理，正常不应再创建
PLAN_VIZ_ROOTS = (
    VIZ_GRIPPER_ROOT,
    VIZ_BALL_PATH,
    VIZ_RAY_HIT_BALL_PATH,
    VIZ_EEF_GRIPPER_ROOT,
)
SESSION_ROOT = os.environ.get(
    "PLAN_GRASP_SESSION_ROOT", "/tmp/plan_grasp_sessions"
)


def session_dir(session_id: str) -> str:
    return agent_runs.managed_session_dir(SESSION_ROOT, session_id)


def session_json_path(session_id: str) -> str:
    return os.path.join(session_dir(session_id), "session.json")


def load_session(session_id: str) -> Dict[str, Any]:
    path = session_json_path(session_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"会话不存在: {session_id}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_session(session_id: str, data: Dict[str, Any]) -> None:
    d = session_dir(session_id)
    os.makedirs(d, exist_ok=True)
    with open(session_json_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def new_session_id(object_name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in object_name)
    return f"{safe}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


def normalize_click(u: float, v: float, w: int, h: int) -> Tuple[int, int]:
    """Round internal native-pixel coordinates after the public boundary."""
    del w, h
    return int(round(float(u))), int(round(float(v)))


def normalize_view(view: str) -> str:
    """规范化视角名；兼容旧 left_upper / right_upper。"""
    aliases = {
        "left_upper": "left_upper_front",
        "right_upper": "right_upper_front",
    }
    v = (view or PLAN_GRASP_VIEW).strip()
    v = aliases.get(v, v)
    if v not in PLAN_GRASP_VIEWS:
        raise ValueError(f"未知视角 {view!r}，可选: {list(PLAN_GRASP_VIEWS)}")
    return v


def _horizontal_tangent(
    outward: np.ndarray,
    up_w: np.ndarray,
    door_meta: Optional[Dict] = None,
) -> np.ndarray:
    """门宽方向水平切向，用于前/后相机偏移。"""
    if door_meta and door_meta.get("tangent_closed_world"):
        t = np.asarray(door_meta["tangent_closed_world"], dtype=np.float64)
    else:
        t = np.cross(outward, up_w)
    t = t - np.dot(t, up_w) * up_w
    tn = float(np.linalg.norm(t))
    if tn < 1e-6:
        t = np.cross(outward, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        tn = float(np.linalg.norm(t))
    return t / (tn + 1e-9)


def compute_cam_positions(
    handle_pos: np.ndarray,
    outward: np.ndarray,
    cam_dist: float,
    door_meta: Optional[Dict] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    up_w = np.array([0.0, 0.0, 1.0])
    perp = np.cross(up_w, outward)
    pn = np.linalg.norm(perp)
    if pn < 1e-6:
        perp = np.cross(np.array([1.0, 0.0, 0.0]), outward)
        pn = np.linalg.norm(perp)
    perp /= pn + 1e-9
    tangent_fb = _horizontal_tangent(outward, up_w, door_meta)
    focus = handle_pos.copy()
    base = handle_pos + outward * cam_dist + up_w * CAM_UP
    return {
        "left_upper_front":  base + perp * CAM_SIDE + tangent_fb * CAM_FB,
        "left_upper_back":   base + perp * CAM_SIDE - tangent_fb * CAM_FB,
        "right_upper_front": base - perp * CAM_SIDE + tangent_fb * CAM_FB,
        "right_upper_back":  base - perp * CAM_SIDE - tangent_fb * CAM_FB,
    }, focus


def _draw_hit_point(img_path: str, u: int, v: int, out_path: str) -> None:
    """在 RGB 图上用红色标记 3D 反解点（与仿真红球一致）。"""
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return
    size = 20
    red = (0, 0, 255)  # BGR 红
    cv2.line(img, (u - size, v), (u + size, v), red, 2)
    cv2.line(img, (u, v - size), (u, v + size), red, 2)
    cv2.circle(img, (u, v), 8, red, -1, lineType=cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def resolve_preview_sensor(
    world=None,
    gta=None,
    session: Optional[Dict[str, Any]] = None,
):
    """规划标注重渲染用相机。

    - v2 / grasp_obj（session.view=head）：用机器人 head/zed，恢复 capture 冻结外参；
    - 旧 plan_grasp 四视角（left_upper_*）：仍用 gta_view 固定门视角。
    """
    view = (session or {}).get("view", PLAN_GRASP_VIEW)
    if view == "head":
        if world is None:
            return None
        from behavior_interface.head_capture import get_head_sensor
        return get_head_sensor(world)
    if gta is not None:
        return gta
    if world is None:
        return None
    try:
        ext = getattr(world.env, "_external_sensors", None) or {}
        return ext.get("gta_view")
    except Exception:
        return None


def _load_session_cam_meta(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sid = session["session_id"]
    view = session.get("view", PLAN_GRASP_VIEW)
    init_dir = session.get("init_dir") or os.path.join(session_dir(sid), "init")
    meta_path = os.path.join(init_dir, f"camera_meta_{view}.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _restore_frozen_camera(sensor, session: Dict[str, Any], ctx=None) -> bool:
    """把传感器恢复到 capture 时冻结的外参（head=精确 pos/quat；门视角=gta look_at）。"""
    import omnigibson as og

    cam_meta = _load_session_cam_meta(session)
    if cam_meta is None:
        if ctx is not None:
            ctx.log(f"  restore_cam 缺 camera_meta_{session.get('view', PLAN_GRASP_VIEW)}.json")
        return False

    view = session.get("view", PLAN_GRASP_VIEW)
    cam_pos = np.asarray(cam_meta["cam_pos"], dtype=np.float64)

    if view == "head":
        q = cam_meta.get("cam_quat_xyzw")
        if q is None:
            if ctx is not None:
                ctx.log("  restore_cam head 缺 cam_quat_xyzw")
            return False
        cam_quat = np.asarray(q, dtype=np.float64).reshape(4)
        try:
            import torch as th
            sensor.set_position_orientation(
                position=th.tensor(cam_pos, dtype=th.float32),
                orientation=th.tensor(cam_quat, dtype=th.float32),
            )
        except Exception as e:
            if ctx is not None:
                ctx.log(f"  restore_cam head 失败: {e}")
            return False
    else:
        from behavior_interface.skills.vlm_grasp_verify import _quat_to_mat
        look_at = np.asarray(cam_meta.get("look_at", cam_pos), dtype=np.float64)
        if np.linalg.norm(look_at - cam_pos) < 0.08:
            q = cam_meta.get("cam_quat_xyzw")
            if q is not None:
                fwd = _quat_to_mat(np.asarray(q, dtype=np.float64)) @ np.array([0.0, 0.0, -1.0])
                look_at = cam_pos + fwd * 1.5
        _move_cam(sensor, cam_pos, look_at)

    for _ in range(12):
        og.sim.render()
    if ctx is not None:
        ctx.log(
            f"  restore_cam view={view} pos={cam_pos.round(3).tolist()} "
            f"(冻结 capture 外参，非 gta 跟拍)"
        )
    return True


def _obs_rgb_to_bgr_uint8(rgb) -> Optional["np.ndarray"]:
    """VisionSensor obs → BGR uint8（兼容 float01 / uint8 / RGBA）。"""
    import numpy as np

    if rgb is None:
        return None
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    if np.issubdtype(arr.dtype, np.floating):
        mx = float(np.nanmax(arr))
        if mx <= 1.0 + 1e-3:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    import cv2
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _capture_sensor_png(sensor, out_path: str, ctx=None, *, n_render: int = 24) -> bool:
    """从传感器截图；多帧 render + 亮度校验，避免 head 冻结外参后全黑图。"""
    import cv2
    import omnigibson as og

    best = None
    best_mean = -1.0
    for attempt in range(3):
        n = n_render + attempt * 12
        for _ in range(n):
            og.sim.render()
        try:
            obs, _ = sensor.get_obs()
        except Exception as e:
            if ctx is not None:
                ctx.log(f"  [cap] get_obs 失败: {e}")
            continue
        rgb = obs.get("rgb") if isinstance(obs, dict) else None
        if rgb is None:
            for k, v in obs.items():
                if "rgb" in str(k).lower():
                    rgb = v
                    break
        bgr = _obs_rgb_to_bgr_uint8(rgb)
        if bgr is None:
            continue
        mean = float(bgr.mean())
        if mean > best_mean:
            best_mean = mean
            best = bgr
        if mean >= 25.0:
            break
    if best is None:
        if ctx is not None:
            ctx.log("  [cap] 无有效 RGB")
        return False
    cv2.imwrite(out_path, best)
    if ctx is not None:
        ctx.log(f"  [cap] {os.path.basename(out_path)} mean={best_mean:.1f} ok={best_mean >= 25.0}")
    return best_mean >= 8.0


def capture_session_preview(
    sensor,
    session: Dict[str, Any],
    out_basename: str,
    ctx=None,
) -> Optional[str]:
    """恢复 capture 冻结外参 → 重渲染截图 → head 相机位姿还原（若有）。"""
    import omnigibson as og

    if sensor is None:
        return None
    sid = session["session_id"]
    view = session.get("view", PLAN_GRASP_VIEW)
    saved_head_pose = None
    if view == "head":
        try:
            saved_head_pose = sensor.get_position_orientation()
        except Exception:
            saved_head_pose = None

    try:
        if not _restore_frozen_camera(sensor, session, ctx):
            return None
        out_path = os.path.join(session_dir(sid), out_basename)
        ok = _capture_sensor_png(sensor, out_path, ctx=ctx, n_render=28)
        return out_path if ok else None
    finally:
        if view == "head" and saved_head_pose is not None:
            try:
                import torch as th
                p, q = saved_head_pose
                sensor.set_position_orientation(
                    position=p if hasattr(p, "detach") else th.tensor(p, dtype=th.float32),
                    orientation=q if hasattr(q, "detach") else th.tensor(q, dtype=th.float32),
                )
                for _ in range(4):
                    og.sim.render()
            except Exception as e:
                if ctx is not None:
                    ctx.log(f"  restore_cam head 还原跟拍位姿失败: {e}")


def _del_prim_path(path: str) -> None:
    try:
        import omnigibson as og
        p = og.sim.stage.GetPrimAtPath(path)
        if p.IsValid():
            og.sim.stage.RemovePrim(path)
    except Exception:
        pass


def clear_plan_viz_prims() -> None:
    """立即删除 plan 可视化 prim（红夹爪/黄球），不 env.step、不留场景图残留。"""
    for path in PLAN_VIZ_ROOTS:
        _del_prim_path(path)
    try:
        import omnigibson as og
        for _ in range(3):
            og.sim.render()
    except Exception:
        pass


def visualize_hit_sphere(hit_pos: np.ndarray, outward: Optional[np.ndarray] = None) -> str:
    """在仿真场景放置红色 3D 小球，球心严格在射线-表面交点。"""
    import omnigibson as og
    from behavior_interface.skills.viz_eef_v2 import _make_sphere

    _del_prim_path(VIZ_BALL_PATH)
    _make_sphere(
        og.sim.stage, VIZ_BALL_PATH, np.asarray(hit_pos, dtype=np.float64),
        radius=VIZ_BALL_RADIUS, color=(1.0, 0.0, 0.0), opacity=0.95,
    )
    for _ in range(8):
        og.sim.render()
    return VIZ_BALL_PATH


def visualize_ray_hit_ball_yellow(hit_pos: np.ndarray) -> str:
    """3D 黄球：grasp_obj=对称轴 1/3 表面锚点(gap_center)；grasp_point=射线命中点。"""
    import omnigibson as og
    from behavior_interface.skills.viz_eef_v2 import _make_sphere

    _del_prim_path(VIZ_RAY_HIT_BALL_PATH)
    _make_sphere(
        og.sim.stage, VIZ_RAY_HIT_BALL_PATH, np.asarray(hit_pos, dtype=np.float64),
        radius=VIZ_BALL_RADIUS, color=(1.0, 1.0, 0.0), opacity=0.92,
    )
    for _ in range(6):
        og.sim.render()
    return VIZ_RAY_HIT_BALL_PATH


def visualize_gripper(
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    outward: np.ndarray,
    *,
    gripper_q_m: float = 0.05,
) -> list:
    """在仿真场景放置红色夹爪 mesh（纯 RTX 可视化，无碰撞/无刚体，不 env.step）。"""
    import omnigibson as og
    from behavior_interface.skills.viz_eef_v2 import _create_gripper

    # 显示夹爪时去掉红球，避免遮挡
    _del_prim_path(VIZ_BALL_PATH)
    _del_prim_path(VIZ_GRIPPER_ROOT)
    paths = _create_gripper(
        og.sim.stage, VIZ_GRIPPER_ROOT,
        np.asarray(eef_pos, dtype=np.float64),
        np.asarray(eef_quat, dtype=np.float64),
        np.asarray(outward, dtype=np.float64),
        color=(1.0, 0.0, 0.0), opacity=0.92,
        gripper_q_m=gripper_q_m,
    )
    if not paths:
        from behavior_interface.skills.viz_eef_v2 import _OBJ_DIR
        raise RuntimeError(f"红夹爪 mesh 创建失败，请检查 OBJ 目录: {_OBJ_DIR}")
    # 仅 render 预热材质，不推进物理
    for _ in range(8):
        og.sim.render()
    return paths


def resolve_pixel_to_3d(
    *,
    gta,
    session: Dict[str, Any],
    u: int,
    v: int,
    ctx=None,
) -> Dict[str, Any]:
    """2D 像素 → 3D 击中点（不含 EEF）。"""
    import omnigibson as og

    sid = session["session_id"]
    sdir = session_dir(sid)
    init_dir = os.path.join(sdir, "init")
    view = session.get("view", PLAN_GRASP_VIEW)

    depth = np.load(os.path.join(init_dir, "depth.npy")).astype(np.float64)
    with open(os.path.join(init_dir, f"camera_meta_{view}.json"), encoding="utf-8") as f:
        cam_meta = json.load(f)

    w_img = int(session["image_width"])
    h_img = int(session["image_height"])
    fl_m = float(cam_meta.get("focal_length", session.get("focal_length", HEAD_FOCAL_LENGTH)))
    ha_m = float(cam_meta.get("horizontal_aperture", session.get("horizontal_aperture", HEAD_HORIZONTAL_APERTURE)))
    cam_pos = np.asarray(cam_meta["cam_pos"])
    cam_quat = np.asarray(cam_meta["cam_quat_xyzw"])
    focus = np.asarray(cam_meta["look_at"])
    handle_pos = np.asarray(session["handle_pos"])
    outward = np.asarray(session["outward"])
    outward /= np.linalg.norm(outward) + 1e-9

    _move_cam(gta, cam_pos, focus)
    for _ in range(6):
        og.sim.render()

    mode = session.get("grasp_mode", "hinge")
    seg_path = os.path.join(init_dir, "seg.npy")
    seg_arr = np.load(seg_path).astype(np.int32) if os.path.isfile(seg_path) else None
    obj_center = np.asarray(
        session.get("object_info", {}).get("center", handle_pos.tolist()), dtype=np.float64
    )
    pts_obj, obj_ids = build_object_pointcloud(
        depth, seg_arr, cam_pos, cam_quat, fl_m, ha_m, u, v, hit_ref=obj_center,
    )

    hit_pos = None
    hit_method = "none"

    if mode == "grasp":
        hit_pos, hit_method = resolve_hit_on_surface(
            u, v, depth, pts_obj, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m, gta=gta,
        )

    if hit_pos is None:
        p_depth = _depth_to_world_point(
            gta, u, v, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m
        )
        if p_depth is not None:
            hit_pos = p_depth
            hit_method = "depth_backproject"

    if hit_pos is None:
        pts_near = pts_obj if len(pts_obj) >= 5 else _build_pointcloud(
            depth, cam_pos, cam_quat, fl_m, ha_m
        )
        origin, ray_dir = _pixel_to_world_ray(
            cam_pos, cam_quat, u, v, w_img, h_img, fl_m, ha_m
        )
        if len(pts_near) >= 5:
            hit_pos = _ray_pcd_hit(pts_near, origin, ray_dir, max_perp=0.15)
            if hit_pos is not None:
                hit_method = "pcd_ray"
        if hit_pos is None and len(pts_near) >= 1:
            hit_pos = _ray_pcd_hit(pts_near, origin, ray_dir, max_perp=9999.0)
            if hit_pos is not None:
                hit_method = "pcd_nearest"

    if hit_pos is None:
        origin, ray_dir = _pixel_to_world_ray(
            cam_pos, cam_quat, u, v, w_img, h_img, fl_m, ha_m
        )
        ref = np.asarray(
            cam_meta.get(
                "handle_geom",
                session.get("object_info", {}).get("center", handle_pos.tolist()),
            )
        )
        t = float(np.dot(ref - origin, ray_dir))
        if t > 0.05:
            hit_pos = origin + ray_dir * t
            hit_method = "ray_ref_fallback"

    if hit_pos is None:
        return {"ok": False, "error": "无法从 depth/点云反解 3D 点"}
    pcd_path = os.path.join(sdir, "object_pcd.npy")
    if len(pts_obj):
        save_npy_atomic(pcd_path, pts_obj.astype(np.float64))

    rgb_path = os.path.join(init_dir, "rgb.png")
    marked_path = os.path.join(sdir, "point_on_image.png")
    _draw_hit_point(rgb_path, u, v, marked_path)

    reproj = _world_to_pixel(cam_pos, cam_quat, hit_pos, w_img, h_img, fl_m, ha_m)
    err_px = float(np.hypot(reproj[0] - u, reproj[1] - v)) if reproj else None

    preview_path = capture_session_preview(gta, session, "viz_3d.png", ctx=ctx)
    clear_plan_viz_prims()

    if ctx is not None:
        ctx.log(
            f"  [plan_grasp] 3D hit={hit_pos.round(3).tolist()} "
            f"method={hit_method} err={err_px}px "
            f"preview={preview_path}"
        )

    return {
        "ok": True,
        "step": "resolve_3d",
        "session_id": sid,
        "object_name": session.get("object_name"),
        "view": view,
        "pixel": {"u": u, "v": v},
        "hit_world": hit_pos.tolist(),
        "hit_method": hit_method,
        "pixel_error": err_px,
        "marked_image": marked_path,
        "object_pcd_path": pcd_path if len(pts_obj) else None,
        "object_pcd_n": int(len(pts_obj)),
        "seg_obj_ids": obj_ids,
        "preview_3d": preview_path,
        "preview_3d_url": f"/api/plan_grasp/session/{sid}/viz_3d" if preview_path else None,
    }


def build_grasp_from_hit(
    session: Dict[str, Any],
    hit_pos: np.ndarray,
    pixel: Optional[Dict[str, int]] = None,
    ctx=None,
) -> Dict[str, Any]:
    """由 3D 击中点生成 EEF pose 与 execute 兼容 candidate。"""
    from behavior_interface.skills.viz_eef_v2 import compute_eef_at_grasp

    mode = session.get("grasp_mode", "hinge")
    meta = session.get("door_meta", {})
    arm = session.get("arm", "right")
    object_name = session.get("object_name", "object")
    outward = np.asarray(session["outward"], dtype=np.float64)
    outward /= np.linalg.norm(outward) + 1e-9
    px = pixel or {}

    if mode == "grasp":
        pcd_path = session.get("object_pcd_path")
        if not pcd_path and session.get("last_3d", {}).get("object_pcd_path"):
            pcd_path = session["last_3d"]["object_pcd_path"]
        pcd = np.load(pcd_path).astype(np.float64) if pcd_path and os.path.isfile(pcd_path) else np.empty((0, 3))
        init_dir = session.get("init_dir", "")
        view = session.get("view", PLAN_GRASP_VIEW)
        cam_pos = cam_quat = None
        meta_path = os.path.join(init_dir, f"camera_meta_{view}.json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                cm = json.load(f)
            cam_pos = np.asarray(cm["cam_pos"], dtype=np.float64)
            cam_quat = np.asarray(cm["cam_quat_xyzw"], dtype=np.float64)
        eef_p = compute_eef_from_pcd_gripper_fit(
            hit_pos, pcd, cam_pos=cam_pos, cam_quat=cam_quat, ctx=ctx,
        )
        target = "grasp"
        gripper_cmd = 1.0
    else:
        geom_dict = {
            "outward_normal_world": outward.tolist(),
            "axis_world": meta.get("axis_world", [0.0, 0.0, 1.0]),
        }
        eef_p = compute_eef_at_grasp(hit_pos, geom_dict)
        target = "open"
        gripper_cmd = -1.0

    if eef_p is None:
        return {"ok": False, "error": "EEF pose 计算失败"}
    if mode == "grasp" and not grip_fit_has_volume(eef_p.get("grip_fit")):
        gf = eef_p.get("grip_fit") or {}
        return {
            "ok": False,
            "error": (
                "开口无有效占据体积，grasp 无效 "
                f"(vox={gf.get('gap_voxel_n', 0)} vol={gf.get('gap_vol_cm3', 0):.3f}cm³)"
            ),
            "grip_fit": gf,
        }

    refined_meta = dict(meta)
    refined_meta.update({
        "handle_closed_world": hit_pos.tolist(),
        "handle_mid_world": hit_pos.tolist(),
        "plan_grasp": True,
        "grasp_mode": mode,
        "pixel": px,
    })
    if mode == "grasp" and eef_p.get("grip_fit"):
        refined_meta["grip_fit"] = eef_p["grip_fit"]

    refined_cand = {
        "id": 0,
        "target": target,
        "arm": arm,
        "label": f"plan_grasp_{object_name}",
        "eef_target": {
            "pos": eef_p["pos"],
            "quat": eef_p["quat"],
            "approach": eef_p["approach"],
            "gripper_cmd": gripper_cmd,
        },
        "next_eef_move": [0.0, 0.0, 0.0],
        "reachable": True,
        "reach_reason": "plan_grasp",
        "score": float(eef_p.get("grip_fit", {}).get("score", 1.0)),
        "meta": refined_meta,
    }

    out = {
        "ok": True,
        "grasp": {
            "pos": eef_p["pos"],
            "quat": eef_p["quat"],
            "approach": eef_p["approach"],
        },
        "target": target,
        "grasp_mode": mode,
        "arm": arm,
        "object": session.get("object_info", {"input": object_name}),
        "candidates": [refined_cand],
    }
    if eef_p.get("grip_fit"):
        out["grip_fit"] = eef_p["grip_fit"]
    return out


def resolve_grasp_from_session(
    session: Dict[str, Any],
    gta=None,
    world=None,
    ctx=None,
) -> Dict[str, Any]:
    """从 session 中上次 3D 结果计算 grasp 并显示夹爪。"""
    last = session.get("last_3d")
    if not last or not last.get("hit_world"):
        return {"ok": False, "error": "请先执行 resolve_3d 计算 3D 点"}

    hit_pos = np.asarray(last["hit_world"], dtype=np.float64)
    payload = build_grasp_from_hit(session, hit_pos, last.get("pixel"), ctx=ctx)
    if not payload.get("ok"):
        return payload

    payload["step"] = "resolve_grasp"
    payload["session_id"] = session["session_id"]
    payload["object_name"] = session.get("object_name")
    payload["view"] = session.get("view")
    payload["hit_world"] = hit_pos.tolist()
    payload["pixel"] = last.get("pixel")

    outward = np.asarray(session.get("outward", [0.0, 0.0, 1.0]), dtype=np.float64)
    outward /= np.linalg.norm(outward) + 1e-9
    grip_paths: list = []
    preview_path = None
    world = world if world is not None else (getattr(ctx, "world", None) if ctx else None)
    sensor = resolve_preview_sensor(world=world, gta=gta, session=session)
    try:
        clear_plan_viz_prims()
        if sensor is not None and payload.get("grasp"):
            g = payload["grasp"]
            grip_paths = visualize_gripper(
                np.asarray(g["pos"], dtype=np.float64),
                np.asarray(g["quat"], dtype=np.float64),
                outward,
            )
            preview_path = capture_session_preview(sensor, session, "viz_grasp.png", ctx=ctx)
    finally:
        clear_plan_viz_prims()
    payload["viz_gripper_paths"] = grip_paths
    payload["preview_grasp"] = preview_path
    if preview_path:
        payload["preview_grasp_url"] = (
            f"/api/plan_grasp/session/{session['session_id']}/viz_grasp"
        )

    if ctx is not None:
        ctx.log(
            f"  [plan_grasp] grasp pos={np.asarray(payload['grasp']['pos']).round(3).tolist()} "
            f"preview={preview_path}"
        )
    return payload


def resolve_pixel_to_grasp(
    *,
    gta,
    world,
    session: Dict[str, Any],
    u: int,
    v: int,
    ctx=None,
) -> Dict[str, Any]:
    """共用后端：2D → 3D + grasp（兼容旧一步 resolve）。"""
    payload = resolve_pixel_to_3d(gta=gta, session=session, u=u, v=v, ctx=ctx)
    if not payload.get("ok"):
        return payload
    hit_pos = np.asarray(payload["hit_world"], dtype=np.float64)
    grasp_part = build_grasp_from_hit(session, hit_pos, payload.get("pixel"))
    if not grasp_part.get("ok"):
        return grasp_part
    payload.update(grasp_part)
    payload["step"] = "resolve"
    outward = np.asarray(session.get("outward", [0.0, 0.0, 1.0]), dtype=np.float64)
    outward /= np.linalg.norm(outward) + 1e-9
    grip_paths: list = []
    preview_path = None
    sensor = resolve_preview_sensor(world=world, gta=gta, session=session)
    try:
        clear_plan_viz_prims()
        if sensor is not None and grasp_part.get("grasp"):
            g = grasp_part["grasp"]
            grip_paths = visualize_gripper(
                np.asarray(g["pos"], dtype=np.float64),
                np.asarray(g["quat"], dtype=np.float64),
                outward,
            )
            preview_path = capture_session_preview(sensor, session, "viz_grasp.png", ctx=ctx)
    finally:
        clear_plan_viz_prims()
    payload["viz_gripper_paths"] = grip_paths
    if preview_path:
        payload["preview_grasp"] = preview_path
        payload["preview_grasp_url"] = (
            f"/api/plan_grasp/session/{session['session_id']}/viz_grasp"
        )
    return payload


def run_vlm_on_session(session: Dict[str, Any]) -> Tuple[int, int, Dict[str, Any]]:
    """VLM 模式：对 session 内 rgb 图调用 Nova，返回像素坐标与原始 item。"""
    import base64
    import os

    import requests

    sid = session["session_id"]
    view = session.get("view", PLAN_GRASP_VIEW)
    img_path = os.path.join(session_dir(sid), "init", "rgb.png")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"缺少 rgb: {img_path}")

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未设置 OPENROUTER_API_KEY")

    w = int(session["image_width"])
    h = int(session["image_height"])

    REAL_SCENE_PROMPT = """You are a vision module for a robot that opens hinged appliances.

Find the single best grasp point on the PHYSICAL DOOR HANDLE or DOOR EDGE to PULL the door open.

Output ONLY a JSON array:
[{"point_2d": [x, y], "label": "door handle grasp"}]

Rules:
- point_2d: TWO integers 0-1000, [x, y] center of grasp target.
- Target physical handle or door edge — NOT control knobs/buttons/displays."""

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    payload = {
        "model": "amazon/nova-premier-v1",
        "temperature": 0.1,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": REAL_SCENE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://behavior-1k.local",
    }
    proxies = None
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        v = os.environ.get(k, "").strip()
        if v:
            proxies = {"http": v, "https": v}
            break

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=180,
        proxies=proxies,
    )
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]

    import sys
    tools_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"
    )
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from vlm_nova_batch_cases import _parse_items

    items = _parse_items(raw)
    if not items:
        raise RuntimeError(f"VLM 无有效输出: {raw[:200]}")
    item = items[0]
    pix = _point_to_pixel(item, w, h)
    if pix is None:
        raise RuntimeError(f"VLM item 无有效点: {item}")
    u, v = pix
    return u, v, {"vlm_raw": raw, "items": items, "model": "amazon/nova-premier-v1", "view": view}
