"""plan_eef 共享后端：图像编号注册、多 mode EEF 规划、渲染叠加。"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from behavior_interface.skills.plan_grasp_core import (
    PLAN_GRASP_VIEW,
    PLAN_GRASP_VIEWS,
    build_grasp_from_hit,
    capture_session_preview,
    compute_cam_positions,
    load_session,
    normalize_click,
    normalize_view,
    resolve_pixel_to_3d,
    resolve_preview_sensor,
    save_session,
    session_dir,
    session_json_path,
    clear_plan_viz_prims,
    visualize_gripper,
    visualize_ray_hit_ball_yellow,
)
from behavior_interface.skills.plan_grasp_gripper_fit import (
    build_object_pointcloud,
    compute_eef_from_pcd_gripper_fit,
    grip_fit_has_volume,
    resolve_hit_on_surface,
)
from behavior_interface.skills.grasp import _is_reachable
from behavior_interface.skills.plan_grasp_object import compute_eef_from_pcd_grasp_object
from behavior_interface.skills.viz_eef_v2 import compute_eef_at_grasp
from behavior_interface.skills.vlm_grasp_verify import _world_to_pixel, _pixel_to_world_ray
from behavior_interface.head_capture import (
    HEAD_FOCAL_LENGTH,
    HEAD_HORIZONTAL_APERTURE,
    HEAD_IMAGE_HEIGHT,
    HEAD_IMAGE_WIDTH,
)
PLAN_EEF_ROOT = os.environ.get(
    "PLAN_EEF_ROOT", "/tmp/plan_eef_sessions"
)
REGISTRY_PATH = os.path.join(PLAN_EEF_ROOT, "image_registry.json")

PUSH_MODES = frozenset({
    "push_left", "push_right", "push_forward", "push_backward", "push_up", "push_down",
})
GRASP_OBJECT_MODES = frozenset({"grasp_obj", "grasp_obj_filter"})
GRASP_POINT_MODES = frozenset({"grasp_point", "grasp_point_filter"})
GRASP_MODES = GRASP_POINT_MODES | GRASP_OBJECT_MODES
HINGE_MODES = frozenset({"open", "close"})
ALL_MODES = PUSH_MODES | GRASP_MODES | HINGE_MODES

PUSH_STEP_M = 0.06
GRASP_LIFT_M = 0.12
GRASP_RETRACT_XY_M = 0.18
GRASP_RETRACT_Z_M = 0.06
# grasp_obj 验证抓取：next move 固定世界系 +Z 上提（不侧向收臂）
GRASP_OBJ_LIFT_M = 0.10

def _ensure_registry() -> Dict[str, Any]:
    os.makedirs(PLAN_EEF_ROOT, exist_ok=True)
    if os.path.isfile(REGISTRY_PATH):
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"counter": 0, "images": {}}


def _save_registry(reg: Dict[str, Any]) -> None:
    os.makedirs(PLAN_EEF_ROOT, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)


def allocate_image_id(session_id: str, meta: Dict[str, Any]) -> str:
    """为 capture 的图像分配递增编号 img_NNNN。"""
    reg = _ensure_registry()
    reg["counter"] = int(reg.get("counter", 0)) + 1
    image_id = f"img_{reg['counter']:04d}"
    reg["images"][image_id] = {
        "session_id": session_id,
        "created_at": time.time(),
        **meta,
    }
    _save_registry(reg)
    return image_id


def lookup_image(image_id: str) -> Dict[str, Any]:
    reg = _ensure_registry()
    entry = reg.get("images", {}).get(image_id.strip())
    if not entry:
        raise KeyError(f"未知 image_id={image_id!r}")
    return entry


def list_registered_images() -> List[Dict[str, Any]]:
    reg = _ensure_registry()
    out = []
    for iid, ent in sorted(reg.get("images", {}).items()):
        out.append({
            "image_id": iid,
            "session_id": ent.get("session_id"),
            "object_name": ent.get("object_name"),
            "view": ent.get("view"),
            "image_path": ent.get("image_path"),
        })
    return out


def _load_cam_meta(session: Dict[str, Any]) -> Dict[str, Any]:
    view = session.get("view", PLAN_GRASP_VIEW)
    init_dir = session.get("init_dir", os.path.join(session_dir(session["session_id"]), "init"))
    path = os.path.join(init_dir, f"camera_meta_{view}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_depth_seg(session: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    init_dir = session.get("init_dir", "")
    depth = np.load(os.path.join(init_dir, "depth.npy")).astype(np.float64)
    seg_path = os.path.join(init_dir, "seg.npy")
    seg = np.load(seg_path).astype(np.int32) if os.path.isfile(seg_path) else None
    if seg is not None and seg.ndim == 3:
        seg = seg[..., 0]
    return depth, seg


def _uv_for_object_seg(
    session: Dict[str, Any],
    u: int,
    v: int,
    *,
    prefer_handle: bool = False,
) -> Tuple[int, int]:
    """用物体中心/把手在图像上的投影作为 seg 种子，避免 VLM 点偏离物体。"""
    cam_meta = _load_cam_meta(session)
    cam_pos = np.asarray(cam_meta["cam_pos"], dtype=np.float64)
    cam_quat = np.asarray(cam_meta["cam_quat_xyzw"], dtype=np.float64)
    w = int(session["image_width"])
    h = int(session["image_height"])
    fl = float(cam_meta.get("focal_length", session.get("focal_length", HEAD_FOCAL_LENGTH)))
    ha = float(cam_meta.get("horizontal_aperture", session.get("horizontal_aperture", HEAD_HORIZONTAL_APERTURE)))

    ref = np.asarray((session.get("object_info") or {}).get("center", session["handle_pos"]), dtype=np.float64)
    if prefer_handle:
        ref = np.asarray(session.get("handle_pos", ref), dtype=np.float64)
    px = _world_to_pixel(cam_pos, cam_quat, ref, w, h, fl, ha)
    if px:
        return px[0], px[1]
    return u, v


def _build_session_object_pcd(
    session: Dict[str, Any],
    u: int,
    v: int,
    *,
    prefer_handle: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    """构建物体点云；点数不足时用物体中心/把手重试 seg。"""
    cam_meta = _load_cam_meta(session)
    depth, seg = _load_depth_seg(session)
    cam_pos = np.asarray(cam_meta["cam_pos"], dtype=np.float64)
    cam_quat = np.asarray(cam_meta["cam_quat_xyzw"], dtype=np.float64)
    fl = float(cam_meta.get("focal_length", session.get("focal_length", HEAD_FOCAL_LENGTH)))
    ha = float(cam_meta.get("horizontal_aperture", session.get("horizontal_aperture", HEAD_HORIZONTAL_APERTURE)))
    obj_center = np.asarray(
        (session.get("object_info") or {}).get("center", session["handle_pos"]), dtype=np.float64
    )

    pts = np.zeros((0, 3))
    ids: List[int] = []
    seeds = [(u, v)]
    seeds.append(_uv_for_object_seg(session, u, v, prefer_handle=prefer_handle))
    if prefer_handle:
        seeds.append(_uv_for_object_seg(session, u, v, prefer_handle=True))
    tight_seg = not prefer_handle
    crop_r = None
    oi = session.get("object_info") or {}
    ab = oi.get("aabb") or {}
    if ab.get("size"):
        crop_r = float(max(0.10, min(0.22, max(ab["size"]) * 0.75)))
    for pu, pv in seeds:
        pts, ids = build_object_pointcloud(
            depth, seg, cam_pos, cam_quat, fl, ha, pu, pv, hit_ref=obj_center,
            tight_seg=tight_seg,
            max_radius_from_ref=crop_r,
        )
        if len(pts) >= 8:
            return pts, ids
    pts_aabb = _build_object_pcd_from_aabb(session)
    if len(pts_aabb) >= 8:
        return pts_aabb, ids
    return pts, ids


def _obj_aabb_from_session_oi(oi: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """从 session.object_info 读取 AABB（兼容嵌套 aabb 与 aabb_min/max 两种格式）。"""
    ab = oi.get("aabb") or {}
    if ab.get("min") and ab.get("max"):
        return (
            np.asarray(ab["min"], dtype=np.float64),
            np.asarray(ab["max"], dtype=np.float64),
        )
    lo = np.asarray(oi.get("aabb_min"), dtype=np.float64).reshape(-1)
    hi = np.asarray(oi.get("aabb_max"), dtype=np.float64).reshape(-1)
    if lo.size == 3 and hi.size == 3:
        return lo, hi
    return None


def _build_object_pcd_from_aabb(session: Dict[str, Any]) -> np.ndarray:
    """seg 无效时：深度图点云按 look_at/把手 球形裁切（深度尺度可能与 AABB 不完全对齐）。"""
    from behavior_interface.skills.vlm_lawn_dual import _build_pointcloud

    cam_meta = _load_cam_meta(session)
    depth, _ = _load_depth_seg(session)
    cam_pos = np.asarray(cam_meta["cam_pos"], dtype=np.float64)
    cam_quat = np.asarray(cam_meta["cam_quat_xyzw"], dtype=np.float64)
    fl = float(cam_meta.get("focal_length", session.get("focal_length", HEAD_FOCAL_LENGTH)))
    ha = float(cam_meta.get("horizontal_aperture", session.get("horizontal_aperture", HEAD_HORIZONTAL_APERTURE)))
    pts = _build_pointcloud(depth, cam_pos, cam_quat, fl, ha)
    if len(pts) == 0:
        return pts

    oi = session.get("object_info", {})
    lo = np.asarray(oi.get("aabb_min"), dtype=np.float64).reshape(-1)
    hi = np.asarray(oi.get("aabb_max"), dtype=np.float64).reshape(-1)
    look_at = np.asarray(cam_meta.get("look_at", oi.get("center")), dtype=np.float64).reshape(-1)
    center = np.asarray(session.get("handle_pos", oi.get("center")), dtype=np.float64)
    if look_at.size != 3:
        look_at = center

    radius = 0.40
    if lo.size == 3 and hi.size == 3:
        radius = float(max(0.22, min(0.65, np.linalg.norm(hi - lo) * 0.85)))

    for anchor in (look_at, center):
        d = np.linalg.norm(pts - anchor, axis=1)
        sub = pts[d < radius]
        if len(sub) >= 8:
            return sub

    # 宽松 AABB（大 margin）兜底
    if lo.size == 3 and hi.size == 3:
        margin = 0.35
        m = np.all(pts >= lo - margin, axis=1) & np.all(pts <= hi + margin, axis=1)
        sub = pts[m]
        if len(sub) >= 8:
            return sub
    return pts


def _ensure_object_pcd(
    session: Dict[str, Any],
    u: int,
    v: int,
    mode: str,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """返回 (点云, seg_ids, depth)。"""
    depth, _ = _load_depth_seg(session)
    prefer_handle = mode in HINGE_MODES
    pts, ids = _build_session_object_pcd(session, u, v, prefer_handle=prefer_handle)
    if len(pts) < 8:
        pts = _build_object_pcd_from_aabb(session)
    return pts, ids, depth


def _project_pcd_to_image(
    pcd: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (u, v, depth_cam) 每个点。"""
    from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat

    if len(pcd) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    R = _quat_to_mat(cam_quat)
    pts_cam = (R.T @ (pcd - cam_pos).T).T
    z_c = -pts_cam[:, 2]
    valid = z_c > 0.02
    fx = fl / ha * w
    u = pts_cam[:, 0] / np.maximum(z_c, 1e-3) * fx + w / 2.0
    v = -pts_cam[:, 1] / np.maximum(z_c, 1e-3) * fx + h / 2.0
    return u[valid], v[valid], z_c[valid]


def _select_push_contact_uv(
    pcd: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    mode: str,
) -> Tuple[int, int, np.ndarray]:
    """按图像语义在物体点云上选 push 接触点。"""
    u, v, z_c = _project_pcd_to_image(pcd, cam_pos, cam_quat, w, h, fl, ha)
    if len(u) < 3:
        c = pcd.mean(axis=0)
        reproj = _world_to_pixel(cam_pos, cam_quat, c, w, h, fl, ha)
        return (reproj[0], reproj[1], c) if reproj else (w // 2, h // 2, c)

    idx = 0
    if mode == "push_left":
        idx = int(np.argmax(u))
    elif mode == "push_right":
        idx = int(np.argmin(u))
    elif mode == "push_forward":
        idx = int(np.argmin(z_c))
    elif mode == "push_backward":
        idx = int(np.argmax(z_c))
    elif mode == "push_up":
        idx = int(np.argmax(v))
    elif mode == "push_down":
        idx = int(np.argmin(v))
    else:
        idx = 0

    pu = int(np.clip(round(u[idx]), 0, w - 1))
    pv = int(np.clip(round(v[idx]), 0, h - 1))
    origin, ray = _pixel_to_world_ray(cam_pos, cam_quat, pu, pv, w, h, fl, ha)
    from behavior_interface.skills.vlm_lawn_dual import _ray_pcd_hit

    hit = _ray_pcd_hit(pcd, origin, ray, max_perp=0.12)
    if hit is None:
        hit = pcd[int(np.argmin(np.linalg.norm(pcd - origin, axis=1)))]
    return pu, pv, np.asarray(hit, dtype=np.float64)


def _world_push_direction(
    mode: str,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    contact: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
) -> np.ndarray:
    """图像语义 → 世界系单位 push 方向。"""
    from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat

    R = _quat_to_mat(cam_quat)
    cam_right = R @ np.array([1.0, 0.0, 0.0])
    cam_up = R @ np.array([0.0, 1.0, 0.0])
    to_scene = -R @ np.array([0.0, 0.0, 1.0])
    to_scene /= np.linalg.norm(to_scene) + 1e-9

    if mode == "push_left":
        d = cam_right
    elif mode == "push_right":
        d = -cam_right
    elif mode == "push_forward":
        d = to_scene
    elif mode == "push_backward":
        d = -to_scene
    elif mode == "push_up":
        d = -cam_up
    elif mode == "push_down":
        d = cam_up
    else:
        d = to_scene
    d = d - np.dot(d, np.array([0.0, 0.0, 1.0])) * np.array([0.0, 0.0, 1.0])
    n = float(np.linalg.norm(d))
    return d / (n + 1e-9) if n > 1e-6 else to_scene


def _push_eef_pose(contact: np.ndarray, push_dir: np.ndarray) -> Dict[str, Any]:
    """顶压 push：eef 在接触点上方，approach 向下。"""
    approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    eef_pos = contact + np.array([0.0, 0.0, 0.025])
    x = np.cross(np.array([0.0, 1.0, 0.0]), approach)
    if np.linalg.norm(x) < 1e-3:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(approach, x)
    R = np.column_stack([x, y, approach])
    from behavior_interface.skills.plan_grasp_gripper_fit import _mat_to_quat_xyzw

    return {
        "pos": eef_pos.tolist(),
        "quat": _mat_to_quat_xyzw(R).tolist(),
        "approach": approach.tolist(),
    }


def _grasp_next_world(
    eef_pos: np.ndarray,
    pcd: np.ndarray,
    cam_pos: np.ndarray,
    world: Any = None,
) -> np.ndarray:
    """
    提起后收向机器人/相机一侧：主视角可见、距机器人较近。
    策略：物体中心 → 相机方向退 GRASP_RETRACT_XY_M，抬高 GRASP_LIFT_M + GRASP_RETRACT_Z_M。
    """
    obj_c = pcd.mean(axis=0) if len(pcd) else eef_pos
    to_cam = cam_pos - obj_c
    to_cam[2] = 0.0
    tn = float(np.linalg.norm(to_cam))
    if tn < 1e-3:
        to_cam = np.array([0.0, -1.0, 0.0])
    else:
        to_cam /= tn

    if world is not None:
        try:
            bp = world.robot.get_position()
            robot_xy = np.array([float(bp[0]), float(bp[1])], dtype=np.float64)
            to_robot = robot_xy - obj_c[:2]
            if float(np.linalg.norm(to_robot)) > 0.15:
                to_cam = np.array([to_robot[0], to_robot[1], 0.0], dtype=np.float64)
                to_cam /= np.linalg.norm(to_cam) + 1e-9
        except Exception:
            pass

    retract = obj_c + to_cam * GRASP_RETRACT_XY_M
    retract[2] = max(float(eef_pos[2]) + GRASP_LIFT_M, float(obj_c[2]) + GRASP_RETRACT_Z_M)
    return retract.astype(np.float64)


def _hinge_next_from_meta(meta: Dict[str, Any], opening: bool) -> np.ndarray:
    hinge = np.asarray(meta["hinge_world"], dtype=np.float64)
    axis = np.asarray(meta["axis_world"], dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-9
    handle = np.asarray(meta.get("handle_closed_world", meta.get("handle_mid_world")), dtype=np.float64)
    hh = handle - hinge
    hh_perp = hh - np.dot(hh, axis) * axis
    if float(np.linalg.norm(hh_perp)) < 1e-6:
        return handle
    tangent = np.cross(axis, hh_perp)
    tangent /= np.linalg.norm(tangent) + 1e-9
    h_delta = np.asarray(meta.get("handle_open_world", handle)) - np.asarray(
        meta.get("handle_closed_world", handle)
    )
    if np.dot(tangent, h_delta) < 0:
        tangent = -tangent
    if not opening:
        tangent = -tangent
    arc = float(np.linalg.norm(hh_perp)) * abs(
        float(meta.get("open_q", 1.2)) - float(meta.get("closed_q", 0.0))
    )
    return handle + tangent * min(arc * 0.45, 0.22)


def _resolve_close_push(
    pcd: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    close_dir: np.ndarray,
) -> Tuple[int, int, np.ndarray]:
    """close：沿关门切向的反方向在点云上选 push 点。"""
    from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat

    if len(pcd) < 5:
        c = pcd.mean(axis=0) if len(pcd) else cam_pos
        r = _world_to_pixel(cam_pos, cam_quat, c, w, h, fl, ha)
        return (r[0], r[1], c) if r else (w // 2, h // 2, c)

    push_dir = -close_dir / (np.linalg.norm(close_dir) + 1e-9)
    scores = pcd @ push_dir
    idx = int(np.argmax(scores))
    contact = pcd[idx]
    reproj = _world_to_pixel(cam_pos, cam_quat, contact, w, h, fl, ha)
    if reproj:
        return reproj[0], reproj[1], contact
    return w // 2, h // 2, contact


def _draw_point_on_image(
    rgb_path: str,
    out_path: str,
    u: int,
    v: int,
    *,
    contact_px: Optional[Tuple[int, int]] = None,
    mode_label: str = "",
) -> None:
    """VLM/点击标注图：绿点=输入点，可选红点=接触点。"""
    import cv2

    img = cv2.imread(rgb_path)
    if img is None:
        return
    cv2.circle(img, (int(u), int(v)), 9, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (int(u), int(v)), 11, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    if contact_px and contact_px != (u, v):
        cu, cv = contact_px
        cv2.circle(img, (cu, cv), 7, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    if mode_label:
        cv2.putText(img, mode_label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(out_path, img)


def _draw_ray_hit_yellow_on_image(
    img_path: str,
    hit_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w_log: int,
    h_log: int,
    fl: float,
    ha: float,
    *,
    label: str = "ray3d",
) -> Optional[Tuple[int, int]]:
    """在返回图上用黄色圆标注 3D 锚点重投影（grasp_obj 为对称轴 1/3 表面点）。"""
    import cv2

    reproj = _world_to_pixel(
        cam_pos, cam_quat, np.asarray(hit_world, dtype=np.float64).reshape(3),
        w_log, h_log, fl, ha,
    )
    if reproj is None:
        return None
    img = cv2.imread(img_path)
    if img is None:
        return reproj
    h_act, w_act = img.shape[:2]
    if w_log == w_act and h_log == h_act:
        pu, pv = int(reproj[0]), int(reproj[1])
    else:
        pu = int(round(reproj[0] * w_act / max(w_log, 1)))
        pv = int(round(reproj[1] * h_act / max(h_log, 1)))
    yellow = (0, 255, 255)  # BGR
    cv2.circle(img, (pu, pv), 14, yellow, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (pu, pv), 16, (0, 180, 180), 2, lineType=cv2.LINE_AA)
    cv2.putText(
        img, label, (pu + 18, pv + 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, yellow, 1, lineType=cv2.LINE_AA,
    )
    cv2.imwrite(img_path, img)
    return (pu, pv)


def _resolve_open_hit_fast(
    session: Dict[str, Any],
    u: int,
    v: int,
    depth: np.ndarray,
    pts_obj: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w_img: int,
    h_img: int,
    fl_m: float,
    ha_m: float,
) -> Tuple[Optional[np.ndarray], str]:
    """open：仅用 session 深度/点云，不移动 gta（避免 plan 卡顿）。"""
    hit_pos, hit_method = resolve_hit_on_surface(
        u, v, depth, pts_obj, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m, gta=None,
    )
    if hit_pos is not None:
        return hit_pos, hit_method
    origin, ray = _pixel_to_world_ray(cam_pos, cam_quat, u, v, w_img, h_img, fl_m, ha_m)
    if len(pts_obj) >= 3:
        from behavior_interface.skills.vlm_lawn_dual import _ray_pcd_hit

        hit = _ray_pcd_hit(pts_obj, origin, ray, max_perp=0.12)
        if hit is not None:
            return np.asarray(hit, dtype=np.float64), "pcd_ray_open"
    return None, "none"


# 日志里出现此串即表示已走 mesh 路径（非旧版 2D 线框方块）
PLAN_VIZ_IMPL = "mesh_head_capture_v2"
# grasp_obj / grasp_point：冻结 head rgb + 2D 夹爪叠影 + 2D 锚点（不进场景，零物理接触）
PLAN_VIZ_GRASP_OBJ = "v7_frozen_2d_overlay"
PLAN_VIZ_GRASP_POINT = PLAN_VIZ_GRASP_OBJ


def _draw_anchor_marker_on_image(
    img_path: str,
    anchor_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w_log: int,
    h_log: int,
    fl: float,
    ha: float,
) -> Optional[Tuple[int, int]]:
    """2D 锚点：小黄球 + 红心（球心=gap_center 投影，v7 head 冻结底图叠影）。"""
    import cv2

    reproj = _world_to_pixel(
        cam_pos, cam_quat, np.asarray(anchor_world, dtype=np.float64).reshape(3),
        w_log, h_log, fl, ha,
    )
    if reproj is None:
        return None
    img = cv2.imread(img_path)
    if img is None:
        return reproj
    h_act, w_act = img.shape[:2]
    if w_log == w_act and h_log == h_act:
        pu, pv = int(reproj[0]), int(reproj[1])
    else:
        pu = int(round(reproj[0] * w_act / max(w_log, 1)))
        pv = int(round(reproj[1] * h_act / max(h_log, 1)))
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    cv2.circle(img, (pu, pv), 8, yellow, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (pu, pv), 10, (0, 200, 200), 1, lineType=cv2.LINE_AA)
    cv2.circle(img, (pu, pv), 3, red, -1, lineType=cv2.LINE_AA)
    cv2.imwrite(img_path, img)
    return (pu, pv)


def _annotate_grasp_obj_metrics_on_image(
    img_path: str,
    *,
    grasp_vol_cm3: float = 0.0,
    overlap_vol_cm3: float = 0.0,
    gripper_vol_cm3: float = 0.0,
    overlap_frac: float = 0.0,
    anchor_dist_mm: float = 0.0,
) -> None:
    """在 plan 标注图上叠加 grasp_vol / overlap_vol 读数。"""
    import cv2

    img = cv2.imread(img_path)
    if img is None:
        return
    ov_pct = 100.0 * float(overlap_frac)
    lines = [
        f"grasp_vol={grasp_vol_cm3:.2f}cm3 (obj between fingers)",
        f"overlap_vol={overlap_vol_cm3:.2f}cm3 ({ov_pct:.0f}% of gripper {gripper_vol_cm3:.1f}cm3)",
        f"center->surf={anchor_dist_mm:.2f}mm (0=center ON surface)",
    ]
    y = 22
    for i, text in enumerate(lines):
        color = (0, 255, 0) if i < 2 else (255, 220, 180)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += 20
    cv2.imwrite(img_path, img)


def _viz_grasp_obj_frozen_overlay(
    rgb_path: str,
    move_png: str,
    *,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w_img: int,
    h_img: int,
    fl_m: float,
    ha_m: float,
    hit_world: Optional[np.ndarray],
    grip_fit: Optional[Dict[str, Any]],
    u: int,
    v: int,
    contact_u: int,
    contact_v: int,
    next_world: np.ndarray,
    tool_label: str,
    plan_mode: str = "grasp_obj",
    gap_center_world: Optional[np.ndarray] = None,
    gripper_q_m: float = 0.05,
    gripper_only: bool = False,
) -> bool:
    """grasp_obj / grasp_point：冻结 capture 底图 + 2D 夹爪叠影 + 锚点 + 体积 label。"""
    import shutil

    from behavior_interface.skills.viz_gripper_overlay import render_gripper_overlay

    if not rgb_path or not os.path.isfile(rgb_path):
        return False
    shutil.copy2(rgb_path, move_png)
    try:
        import cv2
        im = cv2.imread(move_png)
        if im is not None:
            h_act, w_act = im.shape[:2]
            scene_depth = None
            depth_npy = os.path.join(os.path.dirname(rgb_path), "depth.npy")
            if plan_mode != "press_point" and os.path.isfile(depth_npy):
                scene_depth = np.load(depth_npy).astype(np.float32)
                if scene_depth.ndim == 3:
                    scene_depth = scene_depth[..., 0]
            render_gripper_overlay(
                move_png,
                eef_pos=eef_pos, eef_quat=eef_quat,
                cam_pos=cam_pos, cam_quat=cam_quat,
                w=int(w_act), h=int(h_act), fl=float(fl_m), ha=float(ha_m),
                alpha=0.78,
                base_color=(150, 18, 18),
                scene_depth=scene_depth,
                gripper_q_m=gripper_q_m,
            )
    except Exception:
        pass
    if gripper_only:
        return True
    gf = grip_fit or {}
    if plan_mode in GRASP_POINT_MODES:
        click = gf.get("click_hit_world") or hit_world
        if click is not None:
            _draw_ray_hit_yellow_on_image(
                move_png, np.asarray(click, dtype=np.float64).reshape(3),
                cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
                label="click",
            )
        anchor = gap_center_world
        if anchor is None:
            _am = gf.get("anchor_marker_pt")
            if _am is not None:
                anchor = np.asarray(_am, dtype=np.float64).reshape(3)
        if anchor is not None:
            _draw_anchor_marker_on_image(
                move_png, anchor, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
            )
    elif plan_mode != "press_point":
        anchor = hit_world
        _marker_pt = gf.get("marker_pt")
        if _marker_pt is not None:
            anchor = np.asarray(_marker_pt, dtype=np.float64).reshape(3)
        if anchor is not None:
            _draw_anchor_marker_on_image(
                move_png, anchor, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
            )
    if plan_mode in (GRASP_OBJECT_MODES | GRASP_POINT_MODES):
        _annotate_grasp_obj_metrics_on_image(
            move_png,
            grasp_vol_cm3=float(gf.get("grasp_vol_cm3", gf.get("open_intersect_vol_cm3", 0.0))),
            overlap_vol_cm3=float(gf.get("overlap_vol_cm3", 0.0)),
            gripper_vol_cm3=float(gf.get("gripper_vol_cm3", 0.0)),
            overlap_frac=float(gf.get("overlap_frac", 0.0)),
            anchor_dist_mm=float(gf.get("anchor_dist_mm", gf.get("gap_dist_to_obj_mm", 0.0))),
        )
    _annotate_plan_capture_on_image(
        move_png, move_png,
        cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
        eef_pos, next_world,
        contact_px=(
            None if plan_mode == "press_point" else (contact_u, contact_v)
        ),
        mode_label=tool_label,
        seg_click_px=(u, v) if plan_mode in (GRASP_OBJECT_MODES | GRASP_POINT_MODES) else None,
    )
    return True


def _paint_r1pro_gripper_mesh_on_image(
    img_path: str,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w_img: int,
    h_img: int,
    fl_m: float,
    ha_m: float,
    gripper_q_m: float = 0.05,
) -> int:
    """把 R1Pro 夹爪 OBJ 三角面投影到冻结 capture 底图上（实体红 mesh，非线框方块）。"""
    import cv2
    from behavior_interface.skills.viz_eef_v2 import (
        _load_obj,
        _quat_mul_xyzw,
        _quat_to_mat,
        gripper_links_eef,
    )
    from behavior_interface.skills.vlm_grasp_verify import _world_to_pixel

    img = cv2.imread(img_path)
    if img is None:
        return 0
    h_act, w_act = img.shape[:2]
    overlay = img.copy()
    R_eef = _quat_to_mat(np.asarray(eef_quat, dtype=np.float64))
    eef_pos = np.asarray(eef_pos, dtype=np.float64)
    n_tri = 0

    def _px(p_world: np.ndarray) -> Optional[Tuple[int, int]]:
        px = _world_to_pixel(
            cam_pos, cam_quat, p_world, w_img, h_img, fl_m, ha_m,
        )
        if px is None:
            return None
        u, v = px
        if w_img != w_act or h_img != h_act:
            u = int(round(u * w_act / max(w_img, 1)))
            v = int(round(v * h_act / max(h_img, 1)))
        if 0 <= u < w_act and 0 <= v < h_act:
            return u, v
        return None

    for link_name, local_t, local_q in gripper_links_eef(gripper_q_m):
        data = _load_obj(link_name)
        if data is None:
            continue
        verts, idx_list, count_list = data
        world_t = eef_pos + R_eef @ np.asarray(local_t, dtype=np.float64)
        world_q = _quat_mul_xyzw(
            np.asarray(eef_quat, dtype=np.float64),
            np.asarray(local_q, dtype=np.float64),
        )
        R_link = _quat_to_mat(world_q)
        ii = 0
        for cnt in count_list:
            if cnt < 3:
                ii += cnt
                continue
            tri = []
            ok = True
            for k in range(3):
                vi = idx_list[ii + k]
                pw = world_t + R_link @ np.asarray(verts[vi], dtype=np.float64)
                p2 = _px(pw)
                if p2 is None:
                    ok = False
                    break
                tri.append(p2)
            ii += cnt
            if not ok:
                continue
            pts = np.array(tri, dtype=np.int32)
            cv2.fillConvexPoly(overlay, pts, (0, 0, 255), lineType=cv2.LINE_AA)
            n_tri += 1

    if n_tri > 0:
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        cv2.imwrite(img_path, img)
    return n_tri


def _save_plan_debug_images(
    session: Dict[str, Any],
    sid: str,
    mode: str,
    u: int,
    v: int,
    contact_u: int,
    contact_v: int,
    eef_pos: np.ndarray,
    next_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w_img: int,
    h_img: int,
    fl_m: float,
    ha_m: float,
    outward: np.ndarray,
    eef_quat: np.ndarray,
    *,
    hit_world: Optional[np.ndarray] = None,
    grip_fit: Optional[Dict[str, Any]] = None,
    gap_center_world: Optional[np.ndarray] = None,
    gta=None,
    world=None,
    ctx=None,
    gripper_q_m: float = 0.05,
    force_frozen_gripper_overlay: bool = False,
    gripper_only_overlay: bool = False,
) -> Dict[str, str]:
    """生成调试图。grasp_obj/grasp_point 均 v7：冻结 head rgb + 2D 夹爪叠影（单张）。"""
    import shutil

    sdir = session_dir(sid)
    init_dir = session.get("init_dir", os.path.join(sdir, "init"))
    rgb_path = os.path.join(init_dir, "rgb.png")
    out: Dict[str, str] = {}

    image_png = os.path.join(sdir, f"plan_eef_{mode}_image.png")
    if os.path.isfile(rgb_path):
        shutil.copy2(rgb_path, image_png)
        out["image"] = image_png

    if not gripper_only_overlay:
        point_png = os.path.join(sdir, f"plan_eef_{mode}_point.png")
        _draw_point_on_image(
            rgb_path, point_png, u, v,
            contact_px=(contact_u, contact_v), mode_label=mode,
        )
        if hit_world is not None and mode in GRASP_POINT_MODES:
            _draw_ray_hit_yellow_on_image(
                point_png, hit_world, cam_pos, cam_quat,
                w_img, h_img, fl_m, ha_m,
                label="click",
            )
            _gc = gap_center_world
            if _gc is None:
                _am = (grip_fit or {}).get("anchor_marker_pt")
                if _am is not None:
                    _gc = np.asarray(_am, dtype=np.float64).reshape(3)
            if _gc is not None:
                _draw_anchor_marker_on_image(
                    point_png, _gc, cam_pos, cam_quat,
                    w_img, h_img, fl_m, ha_m,
                )
        elif hit_world is not None and mode in GRASP_OBJECT_MODES:
            _draw_ray_hit_yellow_on_image(
                point_png, hit_world, cam_pos, cam_quat,
                w_img, h_img, fl_m, ha_m,
                label="anchor",
            )
        elif hit_world is not None and mode != "press_point":
            _draw_ray_hit_yellow_on_image(
                point_png, hit_world, cam_pos, cam_quat,
                w_img, h_img, fl_m, ha_m,
            )
        out["point_on_image"] = point_png

    move_png = os.path.join(sdir, f"plan_eef_{mode}_move.png")
    tool_label = session.get("tool_label") or mode
    grip_paths: List[str] = []
    sensor = resolve_preview_sensor(world=world, gta=gta, session=session)
    cam_view = session.get("view", "head")

    use_grasp_plan_v7 = (
        mode in (GRASP_OBJECT_MODES | GRASP_POINT_MODES)
        or bool(force_frozen_gripper_overlay)
    )
    if ctx is not None:
        impl = (
            PLAN_VIZ_GRASP_POINT if mode in GRASP_POINT_MODES
            else PLAN_VIZ_GRASP_OBJ if mode in GRASP_OBJECT_MODES
            else PLAN_VIZ_IMPL
        )
        ctx.log(f"  plan_viz_impl={impl} sensor={'ok' if sensor is not None else 'NONE'}")

    def _cap_brightness_ok(path: Optional[str]) -> bool:
        if not path or not os.path.isfile(path):
            return False
        try:
            import cv2
            cap_im = cv2.imread(path)
            return cap_im is not None and float(cap_im.mean()) >= 25.0
        except Exception:
            return False

    def _fallback_frozen_2d_gripper() -> bool:
        """无 sensor 或 RTX 失败：回退冻结 rgb + 2D 红爪投影。"""
        if not rgb_path or not os.path.isfile(rgb_path):
            if ctx is not None:
                ctx.log("  plan标注: 回退失败，无 init/rgb.png")
            return False
        shutil.copy2(rgb_path, move_png)
        n_tri = _paint_r1pro_gripper_mesh_on_image(
            move_png,
            np.asarray(eef_pos, dtype=np.float64),
            np.asarray(eef_quat, dtype=np.float64),
            cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
            gripper_q_m=gripper_q_m,
        )
        if (
            not gripper_only_overlay
            and hit_world is not None
            and mode != "press_point"
        ):
            if mode in GRASP_POINT_MODES:
                _draw_ray_hit_yellow_on_image(
                    move_png, hit_world, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
                    label="click",
                )
                _gc = gap_center_world
                if _gc is None:
                    _am = (grip_fit or {}).get("anchor_marker_pt")
                    if _am is not None:
                        _gc = np.asarray(_am, dtype=np.float64).reshape(3)
                if _gc is not None:
                    _draw_anchor_marker_on_image(
                        move_png, _gc, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
                    )
            else:
                _draw_ray_hit_yellow_on_image(
                    move_png, hit_world, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
                    label="anchor" if mode in GRASP_OBJECT_MODES else "ray3d",
                )
        if not gripper_only_overlay:
            _annotate_plan_capture_on_image(
                move_png, move_png,
                cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
                eef_pos, next_world,
                contact_px=(
                    None if mode == "press_point" else (contact_u, contact_v)
                ),
                mode_label=tool_label,
                seg_click_px=(
                    (u, v)
                    if mode in (GRASP_OBJECT_MODES | GRASP_POINT_MODES)
                    else None
                ),
            )
        out["grasp_move_vis"] = move_png
        out["3d_point_render"] = move_png
        if ctx is not None:
            ctx.log(
                f"  plan标注=回退冻结底图+2D红夹爪({n_tri} tri) → {move_png}"
            )
        return True

    # grasp_obj / grasp_point：v7 方案——冻结 capture + 2D 夹爪叠影，不进 3D 场景
    if use_grasp_plan_v7:
        try:
            clear_plan_viz_prims()
            ok = _viz_grasp_obj_frozen_overlay(
                rgb_path, move_png,
                eef_pos=np.asarray(eef_pos, dtype=np.float64),
                eef_quat=np.asarray(eef_quat, dtype=np.float64),
                cam_pos=cam_pos, cam_quat=cam_quat,
                w_img=w_img, h_img=h_img, fl_m=fl_m, ha_m=ha_m,
                hit_world=hit_world, grip_fit=grip_fit,
                u=u, v=v, contact_u=contact_u, contact_v=contact_v,
                next_world=next_world, tool_label=tool_label,
                plan_mode=mode,
                gap_center_world=gap_center_world,
                gripper_q_m=gripper_q_m,
                gripper_only=gripper_only_overlay,
            )
            if ok:
                out["grasp_move_vis"] = move_png
                out["3d_point_render"] = move_png
                out["image"] = move_png
                out["point_on_image"] = move_png
                if ctx is not None:
                    ctx.log(f"  plan标注=v7 head冻结rgb+2D叠影 → {move_png}")
            else:
                _fallback_frozen_2d_gripper()
        finally:
            clear_plan_viz_prims()
        out["viz_gripper_paths"] = grip_paths
        return out

    try:
        clear_plan_viz_prims()
        if sensor is not None:
            # 红夹爪 + 黄球(对称轴1/3表面锚点 gap_center)；绿点仅在 2D 标注 seg 点击
            if hit_world is not None:
                visualize_ray_hit_ball_yellow(np.asarray(hit_world, dtype=np.float64))
            grip_paths = visualize_gripper(
                np.asarray(eef_pos, dtype=np.float64),
                np.asarray(eef_quat, dtype=np.float64),
                np.asarray(outward, dtype=np.float64),
                gripper_q_m=gripper_q_m,
            )
            cap_tmp = os.path.join(sdir, f"plan_eef_{mode}_{cam_view}_rgb.png")
            cap_path = capture_session_preview(
                sensor, session, os.path.basename(cap_tmp), ctx=ctx,
            )
            cap_ok = _cap_brightness_ok(cap_path)
            if cap_ok:
                shutil.copy2(cap_path, move_png)
                n_tri = _paint_r1pro_gripper_mesh_on_image(
                    move_png,
                    np.asarray(eef_pos, dtype=np.float64),
                    np.asarray(eef_quat, dtype=np.float64),
                    cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
                    gripper_q_m=gripper_q_m,
                )
                _annotate_plan_capture_on_image(
                    move_png, move_png,
                    cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
                    eef_pos, next_world,
                    contact_px=(contact_u, contact_v), mode_label=tool_label,
                )
                if hit_world is not None:
                    _draw_ray_hit_yellow_on_image(
                        move_png, hit_world, cam_pos, cam_quat,
                        w_img, h_img, fl_m, ha_m,
                    )
                out["grasp_move_vis"] = move_png
                out["3d_point_render"] = move_png
                if ctx is not None:
                    ctx.log(
                        f"  plan标注=RTX+2D叠涂红夹爪({len(grip_paths)} prim, "
                        f"{n_tri} tri) → {move_png}"
                    )
            elif os.path.isfile(rgb_path):
                if ctx is not None:
                    ctx.log("  plan标注: RTX 过暗，回退冻结 rgb + 2D 投影")
                _fallback_frozen_2d_gripper()
            elif ctx is not None:
                ctx.log("  plan标注: 无可用底图，无 grasp_move_vis")
        elif ctx is not None:
            ctx.log(
                f"  plan标注: 无预览相机 (view={cam_view})，"
                "v2 需 head/zed；旧 plan_grasp 门视角需 gta_view"
            )
    finally:
        clear_plan_viz_prims()

    out["viz_gripper_paths"] = grip_paths
    return out


def _annotate_plan_capture_on_image(
    src_path: str,
    out_path: str,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w_log: int,
    h_log: int,
    fl: float,
    ha: float,
    eef_pos: np.ndarray,
    next_world: np.ndarray,
    *,
    contact_px: Optional[Tuple[int, int]] = None,
    mode_label: str = "",
    seg_click_px: Optional[Tuple[int, int]] = None,
) -> None:
    """在冻结外参重渲染图（已含红夹爪 mesh）上叠加绿箭头；grasp_obj 用绿点标 seg 点击。"""
    import cv2
    import shutil

    if src_path != out_path:
        shutil.copy2(src_path, out_path)
    img = cv2.imread(out_path)
    if img is None:
        return
    h_act, w_act = img.shape[:2]

    def _px(u: int, v: int) -> Tuple[int, int]:
        if w_log == w_act and h_log == h_act:
            return int(u), int(v)
        return (
            int(round(u * w_act / max(w_log, 1))),
            int(round(v * h_act / max(h_log, 1))),
        )

    # grasp_obj/grasp_point：绿点表示用户点击（seg 或夹取邻域）
    click_px = seg_click_px if seg_click_px is not None else (
        contact_px if mode_label in (GRASP_OBJECT_MODES | GRASP_POINT_MODES) else None
    )
    if click_px:
        cu, cv = _px(click_px[0], click_px[1])
        cv2.circle(img, (cu, cv), 7, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cu, cv), 9, (0, 200, 0), 2, lineType=cv2.LINE_AA)
    elif contact_px:
        cu, cv = _px(contact_px[0], contact_px[1])
        cv2.circle(img, (cu, cv), 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    eef_px = _world_to_pixel(cam_pos, cam_quat, eef_pos, w_log, h_log, fl, ha)
    next_px = _world_to_pixel(cam_pos, cam_quat, next_world, w_log, h_log, fl, ha)
    if eef_px and next_px:
        p0 = _px(eef_px[0], eef_px[1])
        p1 = _px(next_px[0], next_px[1])
        cv2.arrowedLine(img, p0, p1, (0, 255, 0), 2, tipLength=0.25, line_type=cv2.LINE_AA)
    elif contact_px and next_px:
        cu, cv = _px(contact_px[0], contact_px[1])
        p1 = _px(next_px[0], next_px[1])
        cv2.arrowedLine(img, (cu, cv), p1, (0, 255, 0), 2, tipLength=0.25, line_type=cv2.LINE_AA)

    if mode_label:
        cv2.putText(img, mode_label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imwrite(out_path, img)


def _exec_sequence_for_mode(mode: str) -> str:
    if mode in GRASP_MODES or mode == "open":
        return "move_then_close"
    if mode in PUSH_MODES or mode == "close":
        return "close_then_move"
    return "move_then_close"


def _load_navigation_state(session_id: str) -> Dict[str, Any]:
    """读取 move_to_object 写入的臂展达标状态。"""
    import json
    from behavior_interface import agent_runs

    base = session_id.split("__")[0] if "__" in session_id else session_id
    path = os.path.join(agent_runs.run_dir(base), "navigation.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _target_for_mode(mode: str) -> str:
    if mode in GRASP_MODES:
        return "grasp"
    if mode in PUSH_MODES:
        return "push"
    if mode == "open":
        return "open"
    if mode == "close":
        return "close"
    return "grasp"


def plan_eef_from_session(
    session: Dict[str, Any],
    u: int,
    v: int,
    mode: str,
    *,
    gta=None,
    world=None,
    ctx=None,
    fast_plan: Optional[bool] = None,
    click_hit: Optional[np.ndarray] = None,
    grasp_obj_seed: Optional[int] = None,
    plan_arm: str = "any",
) -> Dict[str, Any]:
    """
    核心：image 已 capture 的 session + 像素点 + mode → eef_pose / next_eef_move / 候选。
    """
    mode = (mode or "grasp_point").lower().strip()
    if mode not in ALL_MODES:
        return {"ok": False, "error": f"未知 mode={mode!r}，可选: {sorted(ALL_MODES)}"}
    plan_arm = str(plan_arm or "any").lower().strip()
    if plan_arm not in ("left", "right", "any"):
        plan_arm = "any"

    sid = session["session_id"]
    nav = _load_navigation_state(sid)
    if nav and nav.get("arm_reachable") is False:
        err = (
            "move_to_object 臂展未达标，请先重新 move_to_object："
            f"{nav.get('arm_reach_reason', '')}"
        )
        if ctx:
            ctx.log(f"  [{mode}] 中止: {err}")
        return {"ok": False, "error": err}
    w_img = int(session["image_width"])
    h_img = int(session["image_height"])
    cam_meta = _load_cam_meta(session)
    cam_pos = np.asarray(cam_meta["cam_pos"], dtype=np.float64)
    cam_quat = np.asarray(cam_meta["cam_quat_xyzw"], dtype=np.float64)
    fl_m = float(cam_meta.get("focal_length", session.get("focal_length", HEAD_FOCAL_LENGTH)))
    ha_m = float(cam_meta.get("horizontal_aperture", session.get("horizontal_aperture", HEAD_HORIZONTAL_APERTURE)))
    outward = np.asarray(session["outward"], dtype=np.float64)
    outward /= np.linalg.norm(outward) + 1e-9
    arm = session.get("arm", "right")
    object_name = session.get("object_name", "")
    if fast_plan is None:
        fast_plan = session.get("staging") == "lawn"
    tool_label = session.get("tool_label") or mode
    if ctx and fast_plan:
        ctx.log(f"  [{tool_label}] fast_plan=1（草坪测试加速）")
    obj_center = np.asarray(
        (session.get("object_info") or {}).get("center", session["handle_pos"]), dtype=np.float64
    )

    depth = np.zeros((0,))
    pts_obj = np.zeros((0, 3))
    obj_ids: List[int] = []
    if mode == "open" and fast_plan:
        depth, _ = _load_depth_seg(session)
        pts_obj, obj_ids, _ = _ensure_object_pcd(session, u, v, mode)
    elif mode == "open":
        pass
    else:
        pts_obj, obj_ids, depth = _ensure_object_pcd(session, u, v, mode)
        if len(pts_obj) < 8:
            err = "物体点云过少"
            if ctx:
                ctx.log(f"  [{tool_label}] 中止: {err} (pts={len(pts_obj)} seg_px=({u},{v}))")
            return {"ok": False, "error": err}

    exec_seq = _exec_sequence_for_mode(mode)
    target = _target_for_mode(mode)
    # move_then_close + grasp：eef_target.gripper_cmd 表示到位后的夹爪状态（合爪=-1）
    if target == "grasp":
        gripper_cmd = -1.0
    elif exec_seq == "close_then_move":
        gripper_cmd = -1.0
    else:
        gripper_cmd = 1.0
    door_meta = dict(session.get("door_meta", {}))
    hit_pos = None
    hit_method = "mode"
    contact_u, contact_v = u, v
    reach_ok = True
    reach_reason = "plan_eef"

    # ── push 族 ──
    if mode in PUSH_MODES:
        contact_u, contact_v, hit_pos = _select_push_contact_uv(
            pts_obj, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m, mode,
        )
        push_dir = _world_push_direction(
            mode, cam_pos, cam_quat, hit_pos, w_img, h_img, fl_m, ha_m,
        )
        eef_p = _push_eef_pose(hit_pos, push_dir)
        next_world = hit_pos + push_dir * PUSH_STEP_M
        hit_method = f"push_{mode}"

    # ── grasp_point / grasp_point_filter（filter: hit 3cm + camera-face + dual IK）──
    elif mode in GRASP_POINT_MODES:
        point_mode_label = "grasp_point_filter" if mode == "grasp_point_filter" else "grasp_point"
        hit_pos, hit_method = resolve_hit_on_surface(
            u, v, depth, pts_obj, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m, gta=gta,
        )
        if hit_pos is None:
            return {"ok": False, "error": f"{point_mode_label} 无法反解 3D 点"}
        click_hit = np.asarray(hit_pos, dtype=np.float64).reshape(3)
        reproj_hit = _world_to_pixel(
            cam_pos, cam_quat, click_hit, w_img, h_img, fl_m, ha_m,
        )
        if ctx:
            err_px = (
                float(np.hypot(reproj_hit[0] - u, reproj_hit[1] - v))
                if reproj_hit
                else float("inf")
            )
            ctx.log(
                f"  [{point_mode_label}] ray_hit={click_hit.round(3).tolist()} "
                f"method={hit_method} reproj={reproj_hit} "
                f"click=({u},{v}) err={err_px:.2f}px"
            )
            if err_px > 2.0:
                ctx.log(
                    f"  [{point_mode_label}] WARN reproj_err={err_px:.1f}px>2，"
                    "黄球应与绿点重合，请检查 depth/外参"
                )
            ctx.set_status(f"{point_mode_label} 规划中…")
        contact_u, contact_v = u, v
        oi = session.get("object_info") or {}
        _cen = oi.get("center")
        if _cen is None:
            _cen = session.get("handle_pos")
        obj_ref = (
            np.asarray(_cen, dtype=np.float64)
            if _cen is not None
            else pts_obj.mean(axis=0)
        )
        vs = session.get("virtual_shoulder")
        if vs is not None:
            shoulder = np.asarray(vs, dtype=np.float64)
        else:
            shoulder = obj_ref + np.array([0.0, -0.65, 0.35], dtype=np.float64)
        g_seed = int(grasp_obj_seed) if grasp_obj_seed is not None else 42
        from behavior_interface.skills.plan_grasp_point import compute_eef_from_pcd_grasp_point

        try:
            if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
                ctx.raise_if_cancelled("grasp_point 规划")
            eef_p = compute_eef_from_pcd_grasp_point(
                pts_obj, shoulder, hit_world=click_hit,
                seed=g_seed, ctx=ctx,
                world=world, arm=(nav.get("arm") or arm),
                session=session,
                object_name=object_name or session.get("object_name"),
                dual_arm_ik_filter=(mode == "grasp_point_filter"),
                plan_arm=plan_arm if mode == "grasp_point_filter" else "any",
            )
        except Exception as e:
            from behavior_interface.errors import GraspObjPlanningError, SkillCancelled
            if isinstance(e, SkillCancelled):
                raise
            if isinstance(e, GraspObjPlanningError):
                if ctx:
                    ctx.log(f"  [{point_mode_label}] 规划失败: {e}")
                return {"ok": False, "error": str(e)}
            import traceback
            if ctx:
                ctx.log(f"  [{point_mode_label}] 规划异常: {e}\n{traceback.format_exc()}")
            return {"ok": False, "error": f"{point_mode_label} 规划异常: {e}"}
        if eef_p is None:
            return {"ok": False, "error": f"{point_mode_label} EEF 规划失败（无候选 pose）"}
        gf = eef_p.get("grip_fit") or {}
        if not grip_fit_has_volume(gf):
            return {
                "ok": False,
                "error": (
                    f"{point_mode_label} 无有效 grasp_vol "
                    f"(vox={gf.get('gap_voxel_n', 0)} "
                    f"grasp_vol={gf.get('grasp_vol_cm3', 0):.3f}cm³)"
                ),
                "grip_fit": gf,
            }
        hit_pos = click_hit
        if ctx:
            ctx.log(
                f"  [{point_mode_label}] click_hit={click_hit.round(3).tolist()} "
                f"gap_center={np.asarray(eef_p['gap_center']).round(3).tolist()} "
                f"grasp_vol={gf.get('grasp_vol_cm3', 0):.2f}cm3 "
                f"overlap_vol={gf.get('overlap_vol_cm3', 0):.2f}cm3 "
                f"({100.0 * float(gf.get('overlap_frac', 0)):.0f}% gripper)"
            )
        gripper_cmd = -1.0
        eef_arr = np.asarray(eef_p["pos"], dtype=np.float64)
        next_world = eef_arr + np.array([0.0, 0.0, GRASP_OBJ_LIFT_M], dtype=np.float64)
        reach_arm = eef_p.get("recommended_arm") or eef_p.get("arm") or nav.get("arm") or arm
        if world is not None:
            reach_ok, reach_reason = _is_reachable(
                world, {"pos": eef_arr.tolist()}, arm=reach_arm,
            )
        else:
            reach_ok, reach_reason = True, "no_world"
        if nav.get("arm_reachable") and not reach_ok and ctx:
            ctx.log(
                f"  WARN {point_mode_label} 规划 EEF 仍不可达 ({reach_reason})，"
                "请重新 move_to_object"
            )

    # ── grasp_obj / grasp_obj_filter（与 vlm_lawn_dual pcd_grasp_object 同规划器；仅 head 视角不同）──
    elif mode in GRASP_OBJECT_MODES:
        from behavior_interface.skills.vlm_lawn_dual import _build_pointcloud

        # (u,v) 仅用于 seg 裁剪「哪个物体」，与 plan_grasp_point 不同：不参与抓取锚点
        obj_mode_label = "grasp_obj_filter" if mode == "grasp_obj_filter" else "grasp_obj"
        if ctx:
            ctx.log(
                f"  [{obj_mode_label}] 进入规划 seg_px=({u},{v}) pts_obj={len(pts_obj)} "
                f"obj_ids={obj_ids}（点击=物体语义，非夹取位置）"
            )
            ctx.set_status(f"{obj_mode_label} 规划中…")
        pts_for_fit = pts_obj
        if len(pts_for_fit) < 8:
            pts_world = _build_pointcloud(depth, cam_pos, cam_quat, fl_m, ha_m)
            obj_ref = np.asarray(
                (session.get("object_info") or {}).get("center", session["handle_pos"]),
                dtype=np.float64,
            )
            near = pts_world[np.linalg.norm(pts_world - obj_ref, axis=1) < 0.35]
            if len(near) >= 8:
                pts_for_fit = near
        if len(pts_for_fit) < 8:
            return {"ok": False, "error": f"{obj_mode_label} 物体点云过少"}

        oi = session.get("object_info") or {}
        _cen = oi.get("center")
        if _cen is None:
            _cen = session.get("handle_pos")
        obj_ref = (
            np.asarray(_cen, dtype=np.float64)
            if _cen is not None
            else pts_for_fit.mean(axis=0)
        )
        vs = session.get("virtual_shoulder")
        if vs is not None:
            shoulder = np.asarray(vs, dtype=np.float64)
        else:
            shoulder = obj_ref + np.array([0.0, -0.65, 0.35], dtype=np.float64)

        from behavior_interface.skills.plan_grasp_object import build_object_mesh_grasp_cloud

        obj_aabb = _obj_aabb_from_session_oi(oi)
        g_seed = int(grasp_obj_seed) if grasp_obj_seed is not None else 42
        # mesh 表面/体内采样固定 seed，避免 pose seed 变化导致体积评估点云劣化
        mesh_pts = build_object_mesh_grasp_cloud(
            world, object_name or session.get("object_name"), ctx=ctx,
            obj_ref=obj_ref, obj_aabb=obj_aabb, seed=42,
        )
        try:
            if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
                ctx.raise_if_cancelled(f"{obj_mode_label} 规划")
            eef_p = compute_eef_from_pcd_grasp_object(
                pts_for_fit, shoulder, seed=g_seed, ctx=ctx, mesh_pcd=mesh_pts,
                obj_ref=obj_ref, obj_aabb=obj_aabb,
                world=world, arm=(nav.get("arm") or arm),
                session=session,
                object_name=object_name or session.get("object_name"),
                dual_arm_ik_filter=(mode == "grasp_obj_filter"),
                plan_arm=plan_arm if mode == "grasp_obj_filter" else "any",
            )
        except Exception as e:
            from behavior_interface.errors import GraspObjPlanningError, SkillCancelled
            if isinstance(e, SkillCancelled):
                raise
            if isinstance(e, GraspObjPlanningError):
                if ctx:
                    ctx.log(f"  [{obj_mode_label}] 规划失败: {e}")
                return {"ok": False, "error": str(e)}
            import traceback
            if ctx:
                ctx.log(f"  [{obj_mode_label}] 规划异常: {e}\n{traceback.format_exc()}")
            return {"ok": False, "error": f"{obj_mode_label} 规划异常: {e}"}
        if eef_p is None:
            return {"ok": False, "error": f"{obj_mode_label} EEF 规划失败（无候选 pose）"}
        gf = eef_p.get("grip_fit") or {}
        if not grip_fit_has_volume(gf) and ctx:
            ctx.log(
                f"  [{obj_mode_label}] WARN best_effort 开口体积偏低 "
                f"vox={gf.get('gap_voxel_n', 0)} vol={gf.get('gap_vol_cm3', 0):.3f} "
                f"ncol={gf.get('n_collision', 0)} 仍输出"
            )
        hit_pos = np.asarray(eef_p["gap_center"], dtype=np.float64)
        hit_method = "grasp_object_filter_auto" if mode == "grasp_obj_filter" else "grasp_object_auto"
        if ctx:
            gf = eef_p.get("grip_fit") or {}
            ctx.log(
                f"  [{obj_mode_label}] seg_px=({u},{v}) gap_center={hit_pos.round(3).tolist()} "
                f"pcd_n={len(pts_for_fit)} grasp_vol={gf.get('grasp_vol_cm3', 0):.2f}cm3 "
                f"overlap_vol={gf.get('overlap_vol_cm3', 0):.2f}cm3 "
                f"({100.0 * float(gf.get('overlap_frac', 0)):.0f}% gripper) "
                f"center->surf={gf.get('anchor_dist_mm', 0):.2f}mm "
                f"ncol={gf.get('n_collision')} mesh={gf.get('n_mesh_pts', 0)} "
                f"method={hit_method}"
            )
        contact_u, contact_v = u, v
        gripper_cmd = -1.0
        eef_arr = np.asarray(eef_p["pos"], dtype=np.float64)
        next_world = eef_arr + np.array([0.0, 0.0, GRASP_OBJ_LIFT_M], dtype=np.float64)
        reach_arm = eef_p.get("recommended_arm") or eef_p.get("arm") or nav.get("arm") or arm
        if world is not None:
            reach_ok, reach_reason = _is_reachable(
                world, {"pos": eef_arr.tolist()}, arm=reach_arm,
            )
        else:
            reach_ok, reach_reason = True, "no_world"
        if nav.get("arm_reachable") and not reach_ok and ctx:
            ctx.log(
                f"  WARN {obj_mode_label} 规划 EEF 仍不可达 ({reach_reason})，"
                "请重新 move_to_object"
            )

    # ── open ──
    elif mode == "open":
        if fast_plan:
            if len(pts_obj) < 8:
                depth, _ = _load_depth_seg(session)
                pts_obj, _, _ = _ensure_object_pcd(session, u, v, mode)
            hit_pos, hit_method = _resolve_open_hit_fast(
                session, u, v, depth, pts_obj, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
            )
            if hit_pos is None:
                return {"ok": False, "error": "open 无法反解 3D 点"}
        else:
            payload3d = resolve_pixel_to_3d(gta=gta, session=session, u=u, v=v, ctx=ctx)
            if not payload3d.get("ok"):
                return payload3d
            hit_pos = np.asarray(payload3d["hit_world"], dtype=np.float64)
            hit_method = payload3d.get("hit_method", "open")
        geom = {
            "outward_normal_world": outward.tolist(),
            "axis_world": door_meta.get("axis_world", [0.0, 0.0, 1.0]),
            "handle_closed_world": hit_pos.tolist(),
        }
        eef_p = compute_eef_at_grasp(hit_pos, geom)
        if eef_p is None:
            return {"ok": False, "error": "open EEF 计算失败"}
        gripper_cmd = -1.0
        next_world = _hinge_next_from_meta(door_meta, opening=True)
        exec_seq = "move_then_close"

    # ── close ──
    elif mode == "close":
        close_dir = np.zeros(3)
        if door_meta.get("hinge_world"):
            next_w = _hinge_next_from_meta(door_meta, opening=False)
            close_dir = next_w - np.asarray(
                door_meta.get("handle_closed_world", next_w), dtype=np.float64
            )
            cn = float(np.linalg.norm(close_dir))
            if cn > 1e-6:
                close_dir /= cn
        if float(np.linalg.norm(close_dir)) < 1e-6:
            close_dir = outward
        contact_u, contact_v, hit_pos = _resolve_close_push(
            pts_obj, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m, close_dir,
        )
        push_dir = close_dir
        eef_p = _push_eef_pose(hit_pos, push_dir)
        next_world = (
            np.asarray(door_meta.get("handle_closed_world", hit_pos), dtype=np.float64)
            + close_dir * PUSH_STEP_M * 2.0
        )
        if door_meta.get("hinge_world"):
            next_world = _hinge_next_from_meta(door_meta, opening=False)
        hit_method = "close_push"
        gripper_cmd = -1.0
        exec_seq = "close_then_move"

    else:
        return {"ok": False, "error": f"未实现 mode={mode}"}

    eef_pos = np.asarray(eef_p["pos"], dtype=np.float64)
    eef_quat = np.asarray(eef_p["quat"], dtype=np.float64)
    next_world = np.asarray(next_world, dtype=np.float64)
    next_delta = (next_world - eef_pos).tolist()
    plan_arm = str(eef_p.get("recommended_arm") or eef_p.get("arm") or arm).lower().strip()
    if plan_arm not in ("left", "right"):
        plan_arm = arm

    refined_meta = {
        "plan_eef": True,
        "mode": mode,
        "exec_sequence": exec_seq,
        "next_eef_move_world": next_world.tolist(),
        "hit_method": hit_method,
        "pixel": {"u": contact_u, "v": contact_v},
        "vlm_pixel": {"u": u, "v": v},
    }
    if eef_p.get("grip_fit"):
        refined_meta["grip_fit"] = eef_p["grip_fit"]
    if eef_p.get("plan_audit"):
        refined_meta["plan_audit"] = eef_p["plan_audit"]
    refined_meta["recommended_arm"] = eef_p.get("recommended_arm") or plan_arm
    if eef_p.get("selected_pose_ik"):
        refined_meta["selected_pose_ik"] = eef_p["selected_pose_ik"]
    if eef_p.get("selected_pose_ik_q"):
        refined_meta["selected_pose_ik_q"] = eef_p["selected_pose_ik_q"]
    if eef_p.get("ik_solution"):
        refined_meta["ik_solution"] = eef_p["ik_solution"]
    if object_name:
        refined_meta["object_name"] = object_name
    if mode in HINGE_MODES:
        refined_meta.update(door_meta)

    eef_target = {
        "pos": eef_pos.tolist(),
        "quat": eef_quat.tolist(),
        "gripper_cmd": gripper_cmd,
    }
    if mode in GRASP_MODES:
        # exec 落盘兼容：由 quat 推导夹爪指向，非规划语义字段
        from behavior_interface.skills.grasp import _quat_to_mat
        eef_target["approach"] = _quat_to_mat(eef_quat)[:, 2].tolist()
    elif eef_p.get("approach") is not None:
        eef_target["approach"] = eef_p["approach"]

    cand = {
        "id": 0,
        "target": target,
        "arm": plan_arm,
        "label": f"plan_eef_{mode}_{object_name}",
        "eef_target": eef_target,
        "next_eef_move": next_delta,
        "reachable": bool(reach_ok),
        "reach_reason": reach_reason,
        "score": 1.0,
        "meta": refined_meta,
    }
    if eef_p.get("selected_pose_ik"):
        cand["selected_pose_ik"] = eef_p["selected_pose_ik"]
    if eef_p.get("selected_pose_ik_q"):
        cand["selected_pose_ik_q"] = eef_p["selected_pose_ik_q"]
    if eef_p.get("ik_solution"):
        cand["ik_solution"] = eef_p["ik_solution"]

    _gap_center_viz = None
    if mode in GRASP_MODES and eef_p.get("gap_center") is not None:
        _gap_center_viz = np.asarray(eef_p["gap_center"], dtype=np.float64).reshape(3)
    debug_imgs = _save_plan_debug_images(
        session, sid, mode, u, v, contact_u, contact_v,
        eef_pos, next_world, cam_pos, cam_quat, w_img, h_img, fl_m, ha_m,
        outward, eef_quat,
        hit_world=hit_pos if mode in GRASP_MODES else None,
        grip_fit=eef_p.get("grip_fit") if mode in GRASP_MODES else None,
        gap_center_world=_gap_center_viz,
        gta=gta, world=world, ctx=ctx,
    )
    grip_paths = debug_imgs.get("viz_gripper_paths") or []
    overlay_path = debug_imgs.get("grasp_move_vis", "")
    preview_3d = debug_imgs.get("3d_point_render")

    if ctx:
        label = session.get("tool_label") or mode
        ctx.log(
            f"  [{label}] eef={eef_pos.round(3).tolist()} "
            f"next_world={next_world.round(3).tolist()} exec={exec_seq}"
        )

    selected_pose_ik = eef_p.get("selected_pose_ik")
    ik_constraint_summary = None
    if isinstance(selected_pose_ik, dict):
        tol = (
            selected_pose_ik.get("hard_constraint")
            or f"pos<={selected_pose_ik.get('pos_tol_mm', 10)}mm && "
               f"ori<={selected_pose_ik.get('ori_tol_deg', 3)}deg"
        )
        ik_constraint_summary = (
            f"pose_ik_hard_constraint ({tol}): "
            f"left={'OK' if selected_pose_ik.get('left_ok') else 'NO'} "
            f"{selected_pose_ik.get('left_pos_mm', '?')}mm/"
            f"{selected_pose_ik.get('left_ori_deg', '?')}deg; "
            f"right={'OK' if selected_pose_ik.get('right_ok') else 'NO'} "
            f"{selected_pose_ik.get('right_pos_mm', '?')}mm/"
            f"{selected_pose_ik.get('right_ori_deg', '?')}deg; "
            f"both={'OK' if selected_pose_ik.get('both_ok') else 'NO'}"
        )

    return {
        "ok": True,
        "step": "plan",
        "mode": mode,
        "exec_sequence": exec_seq,
        "target": target,
        "session_id": sid,
        "object_name": object_name,
        "view": session.get("view"),
        "pixel": {"u": contact_u, "v": contact_v},
        "vlm_pixel": {"u": u, "v": v},
        "hit_world": hit_pos.tolist() if hit_pos is not None else None,
        "hit_method": hit_method,
        "eef_pose": (
            {"pos": eef_pos.tolist(), "quat": eef_quat.tolist()}
            if mode in GRASP_MODES
            else {
                "pos": eef_pos.tolist(),
                "quat": eef_quat.tolist(),
                "approach": eef_p.get("approach"),
            }
        ),
        "next_eef_move": next_world.tolist(),
        "next_eef_move_delta": next_delta,
        "render_image": overlay_path,
        "render_image_3d": preview_3d,
        "debug_images": debug_imgs,
        "image": debug_imgs.get("image"),
        "point_on_image": debug_imgs.get("point_on_image"),
        "3d_point_render": preview_3d,
        "grasp_move_vis": debug_imgs.get("grasp_move_vis"),
        "viz_gripper_paths": grip_paths,
        "candidates": [cand],
        "object": session.get("object_info", {"input": object_name}),
        "arm": plan_arm,
        "recommended_arm": eef_p.get("recommended_arm") or plan_arm,
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": eef_p.get("selected_pose_ik_q"),
        "ik_constraint_summary": ik_constraint_summary,
        "ik_solution": eef_p.get("ik_solution"),
        "plan_audit": eef_p.get("plan_audit") if mode in GRASP_MODES else None,
        "grip_fit": eef_p.get("grip_fit") if mode in GRASP_MODES else None,
        "gap_center": eef_p.get("gap_center") if mode in GRASP_MODES else None,
        "grasp_vol_cm3": (
            float((eef_p.get("grip_fit") or {}).get("grasp_vol_cm3", 0.0))
            if mode in GRASP_MODES else None
        ),
        "overlap_vol_cm3": (
            float((eef_p.get("grip_fit") or {}).get("overlap_vol_cm3", 0.0))
            if mode in GRASP_MODES else None
        ),
        "overlap_frac": (
            float((eef_p.get("grip_fit") or {}).get("overlap_frac", 0.0))
            if mode in GRASP_MODES else None
        ),
    }
