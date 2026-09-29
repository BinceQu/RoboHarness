"""head 取景离线估算：由物体 AABB + 相机内参推算 base/trunk 目标位姿。

目标（默认）：
  - 物体 AABB 8 角点均在 head 画面内（留边）
  - 投影可见面积约 30%
  - 投影中心接近画面中心
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from behavior_interface.head_capture import HEAD_FOCAL_LENGTH, HEAD_HORIZONTAL_APERTURE

# 默认取景目标（move_to_object 可覆盖）
DEFAULT_TARGET_FILL = 0.30
DEFAULT_FILL_MIN = 0.24
DEFAULT_FILL_MAX = 0.36
DEFAULT_EDGE_MARGIN_FRAC = 0.06
DEFAULT_STANDOFF_MIN = 0.28
DEFAULT_STANDOFF_MAX = 1.02
PITCH_CENTER_DEG = 90.0
PITCH_MAX_DELTA = 28.0
Z_MIN = 0.72
Z_MAX = 1.15
# 与 move_to_object_v2 / grasp 臂展安全区一致（肩→物体中心）
REACH_MIN_SAFE_M = 0.18 * 1.3
REACH_MAX_SAFE_M = 1.05 * 0.85
_SHOULDER_SIDE_Y = 0.171
_SHOULDER_Z_ABOVE_CHEST = 0.303


def aabb_metrics(lo: np.ndarray, hi: np.ndarray) -> Dict[str, Any]:
    """物体 AABB 尺寸与中心。"""
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    ext = hi - lo
    C = (lo + hi) / 2.0
    return {
        "center": [float(C[0]), float(C[1]), float(C[2])],
        "extent_xyz": ext.tolist(),
        "extent_norm_m": float(np.linalg.norm(ext)),
        "extent_xy_m": float(np.linalg.norm(ext[:2])),
        "extent_z_m": float(ext[2]),
        "lo": [float(lo[0]), float(lo[1]), float(lo[2])],
        "hi": [float(hi[0]), float(hi[1]), float(hi[2])],
    }


def _aabb_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x0, y0, z0 = lo[0], lo[1], lo[2]
    x1, y1, z1 = hi[0], hi[1], hi[2]
    return np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)


def project_aabb_to_head(
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    *,
    edge_margin_frac: float = DEFAULT_EDGE_MARGIN_FRAC,
) -> Dict[str, Any]:
    """给定 head 外参，离线投影 AABB（与 move_to_object._head_view_stats 一致）。"""
    from behavior_interface.skills.vlm_grasp_verify import _world_to_pixel

    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    cam_quat = np.asarray(cam_quat, dtype=np.float64).reshape(4)
    margin_u = int(w * edge_margin_frac)
    margin_v = int(h * edge_margin_frac)
    corners = _aabb_corners(lo, hi)
    px_all: List[Tuple[int, int]] = []
    all_corners_in = True
    for p in corners:
        px = _world_to_pixel(cam_pos, cam_quat, p, w, h, fl, ha)
        if not px:
            all_corners_in = False
            continue
        pu, pv = int(px[0]), int(px[1])
        px_all.append((pu, pv))
        if not (margin_u <= pu < w - margin_u and margin_v <= pv < h - margin_v):
            all_corners_in = False

    if len(px_all) < 4:
        C = (lo + hi) / 2.0
        return {
            "ok": False,
            "reason": "too few corners in front of camera",
            "image_width": w, "image_height": h,
            "cam_to_obj_dist": float(np.linalg.norm(cam_pos - C)),
        }

    us = [p[0] for p in px_all]
    vs = [p[1] for p in px_all]
    u0, u1 = min(us), max(us)
    v0, v1 = min(vs), max(vs)
    u0c, u1c = max(0, u0), min(w - 1, u1)
    v0c, v1c = max(0, v0), min(h - 1, v1)
    vis_area = max(0, u1c - u0c) * max(0, v1c - v0c)
    fill_visible = vis_area / float(w * h)
    raw_area = max(1, (u1 - u0) * (v1 - v0))
    fill_raw = raw_area / float(w * h)
    overflow = (
        u0 < margin_u or u1 >= w - margin_u
        or v0 < margin_v or v1 >= h - margin_v
    )
    C = (lo + hi) / 2.0
    fy = fl / ha * w
    return {
        "ok": True,
        "image_width": w,
        "image_height": h,
        "fy": fy,
        "u0": u0, "u1": u1, "v0": v0, "v1": v1,
        "cu": (u0 + u1) / 2.0,
        "cv": (v0 + v1) / 2.0,
        "fill_visible": fill_visible,
        "fill_raw": fill_raw,
        "all_corners_in": all_corners_in,
        "overflow": overflow,
        "cam_to_obj_dist": float(np.linalg.norm(cam_pos - C)),
        "n_corners_visible": len(px_all),
    }


def framing_ok(
    st: Dict[str, Any],
    *,
    fill_min: float = DEFAULT_FILL_MIN,
    fill_max: float = DEFAULT_FILL_MAX,
    yaw_tol_frac: float = 0.05,
    v_tol_frac: float = 0.05,
) -> bool:
    if not st.get("ok"):
        return False
    w, h = int(st["image_width"]), int(st["image_height"])
    fill = float(st.get("fill_visible", st.get("fill", 0)))
    cu, cv = float(st["cu"]), float(st["cv"])
    return (
        bool(st.get("all_corners_in"))
        and not bool(st.get("overflow"))
        and fill_min <= fill <= fill_max
        and abs(cu - w / 2) <= w * yaw_tol_frac
        and abs(cv - h / 2) <= h * v_tol_frac
    )


def estimate_standoff_m(
    lo: np.ndarray,
    hi: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    *,
    target_fill: float = DEFAULT_TARGET_FILL,
    standoff_min: float = DEFAULT_STANDOFF_MIN,
    standoff_max: float = DEFAULT_STANDOFF_MAX,
) -> float:
    """由物体三维尺寸 + 目标 fill 反推肩距 standoff（相机到物体中心约等于该距离量级）。"""
    ext = np.asarray(hi - lo, dtype=np.float64)
    ex, ey, ez = float(ext[0]), float(ext[1]), float(ext[2])
    horiz = math.hypot(ex, ey)
    vert = max(ez, 0.02)
    fy = fl / ha * w
    fx = fy
    # 投影可见面积 ≈ (fx·horiz/d)·(fy·vert/d) / (w·h) → 反解 standoff（留边由 framing 修正）
    eff_fill = max(target_fill, 0.05) * 0.84
    denom = eff_fill * float(w * h)
    if horiz > 1e-3 and vert > 1e-3 and denom > 1e-6:
        d = math.sqrt(fx * fy * horiz * vert / denom)
    else:
        d = 0.55
    return float(np.clip(d, standoff_min, standoff_max))


def estimate_base_xy(
    C: np.ndarray,
    robot_xy: np.ndarray,
    standoff_m: float,
) -> np.ndarray:
    """在 C 周围沿 robot→C 反方向放置底盘，距离 standoff。"""
    C = np.asarray(C, dtype=np.float64)
    robot_xy = np.asarray(robot_xy, dtype=np.float64)
    base_dir = robot_xy - C[:2]
    n = float(np.linalg.norm(base_dir))
    if n < 1e-3:
        base_dir = np.array([1.0, 0.0], dtype=np.float64)
    else:
        base_dir = base_dir / n
    return C[:2] + base_dir * float(standoff_m)


def face_yaw_deg(bx: float, by: float, C: np.ndarray) -> float:
    return math.degrees(math.atan2(float(C[1]) - by, float(C[0]) - bx))


def _shoulder_dist_at_base(
    bx: float, by: float, chest_z: float, center: np.ndarray, arm: str,
) -> float:
    """离线估算：给定 base + trunk 时肩→物体中心的距离。"""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    side_y = -_SHOULDER_SIDE_Y if arm == "right" else _SHOULDER_SIDE_Y
    sh_z = float(chest_z) + _SHOULDER_Z_ABOVE_CHEST
    base_yaw = math.atan2(float(center[1]) - by, float(center[0]) - bx)
    sh_x = bx + side_y * (-math.sin(base_yaw))
    sh_y = by + side_y * math.cos(base_yaw)
    sh = np.array([sh_x, sh_y, sh_z], dtype=np.float64)
    return float(np.linalg.norm(center - sh))


def estimate_dual_arm_reach_at_base(
    bx: float,
    by: float,
    chest_z: float,
    center: np.ndarray,
    *,
    preferred_arm: str = "right",
) -> Dict[str, Any]:
    """离线双臂肩距：至少一侧需在安全区内才可用于站位筛选。"""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    arms: Dict[str, Any] = {}
    for arm in ("left", "right"):
        d = _shoulder_dist_at_base(bx, by, chest_z, center, arm)
        ok = REACH_MIN_SAFE_M <= d <= REACH_MAX_SAFE_M
        if d < REACH_MIN_SAFE_M:
            reason = f"too close ({d:.2f}m < {REACH_MIN_SAFE_M:.2f}m)"
        elif d > REACH_MAX_SAFE_M:
            reason = f"too far ({d:.2f}m > {REACH_MAX_SAFE_M:.2f}m)"
        else:
            reason = f"ok ({d:.2f}m)"
        arms[arm] = {
            "reachable": ok,
            "shoulder_to_object_m": round(d, 4),
            "reason": reason,
        }
    left_ok = bool(arms["left"]["reachable"])
    right_ok = bool(arms["right"]["reachable"])
    reachable_list = [a for a in ("left", "right") if arms[a]["reachable"]]
    pick = None
    pick_reason = "none"
    if preferred_arm in arms and arms[preferred_arm]["reachable"]:
        pick, pick_reason = preferred_arm, "preferred"
    elif left_ok and right_ok:
        pick, pick_reason = preferred_arm, "both_ok_prefer_hint"
    elif left_ok:
        pick, pick_reason = "left", "only_left"
    elif right_ok:
        pick, pick_reason = "right", "only_right"
    return {
        "left": arms["left"],
        "right": arms["right"],
        "reachable_left": left_ok,
        "reachable_right": right_ok,
        "any_reachable": bool(reachable_list),
        "both_reachable": left_ok and right_ok,
        "arms_reachable": reachable_list,
        "arm": pick,
        "arm_pick_reason": pick_reason,
    }


def estimate_trunk_pose(
    C: np.ndarray,
    base_xy: np.ndarray,
    standoff_m: float,
    *,
    pick_chest_fn,
) -> Tuple[float, float]:
    """估算 chest_z / theta_z：物体高度 + 水平距离 → 俯仰。"""
    C = np.asarray(C, dtype=np.float64)
    base_xy = np.asarray(base_xy, dtype=np.float64)
    chest_z, theta_z = pick_chest_fn(float(C[2]))
    chest_z = float(np.clip(chest_z, Z_MIN, Z_MAX))
    horiz = max(float(np.linalg.norm(C[:2] - base_xy)), 0.25)
    # 胸口约在 chest_z，head 光学中心再高约 0.30–0.35m
    cam_z_est = chest_z + 0.33
    # 与 grasp 一致：theta_z 越大越俯视（地面 120°，桌面 90°，高架 70°）
    dz_cam = cam_z_est - float(C[2])
    if dz_cam > 0.05:
        theta_z += min(PITCH_MAX_DELTA, dz_cam / horiz * 48.0)
    elif dz_cam < -0.08:
        theta_z -= min(18.0, (-dz_cam) / horiz * 32.0)
    theta_z = float(np.clip(theta_z, 70.0, 118.0))
    return chest_z, theta_z


def predict_head_pose_delta(
    world,
    base_xy: np.ndarray,
    yaw_deg: float,
    chest_z: float,
    theta_z_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """用当前位姿 + 目标 base/trunk 差分，一阶近似 head 外参（用于离线投影）。"""
    head = None
    try:
        from behavior_interface.head_capture import get_head_sensor
        head = get_head_sensor(world)
    except Exception:
        pass
    if head is None:
        raise RuntimeError("no head sensor")

    pos0, quat0 = head.get_position_orientation()
    cam0 = np.asarray(pos0, dtype=np.float64).reshape(3)
    quat0 = np.asarray(quat0, dtype=np.float64).reshape(4)

    pose0 = world.robot_pose()
    b0 = np.array([float(pose0.pos[0]), float(pose0.pos[1])], dtype=np.float64)
    yaw0 = float(pose0.yaw)
    try:
        chest0 = world.chest_pose()
        cz0 = float(chest0["z"])
    except Exception:
        cz0 = cam0[2] - 0.33

    db = np.asarray(base_xy, dtype=np.float64) - b0
    dyaw = math.radians(_norm_angle_deg(yaw_deg - math.degrees(yaw0)))
    dz = float(chest_z) - cz0
    dtz = float(theta_z_deg) - float(
        chest0.get("theta_z_deg", PITCH_CENTER_DEG) if isinstance(chest0, dict) else PITCH_CENTER_DEG
    )

    # 底盘平移 + 绕 z 旋转对 head 的影响（刚体近似）
    c, s = math.cos(dyaw), math.sin(dyaw)
    rot_db = np.array([c * db[0] - s * db[1], s * db[0] + c * db[1], 0.0])
    cam_pred = cam0 + rot_db
    cam_pred[2] += dz + dtz * 0.020  # 低头时光心略降，利于台面物体进画面

    # 俯仰改变视线方向：用 chest forward 近似，仅调整 quat 的 pitch 分量（小角度）
    # 大角度时离线投影误差大，后续靠实测修正
    from behavior_interface.skills.vlm_grasp_verify import _quat_to_mat, _mat_to_quat_xyzw
    R0 = _quat_to_mat(quat0)
    # 俯仰对视线影响放大，便于 move_to_object 离线选 theta_z（原 0.35 偏小，易选出「不低头」）
    pitch = math.radians(dtz * 0.72)
    Rp = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)],
    ], dtype=np.float64)
    R1 = R0 @ Rp
    quat_pred = _mat_to_quat_xyzw(R1)
    return cam_pred, quat_pred


def _norm_angle_deg(a_deg: float) -> float:
    while a_deg > 180.0:
        a_deg -= 360.0
    while a_deg <= -180.0:
        a_deg += 360.0
    return a_deg


def estimate_view_pose(
    world,
    lo: np.ndarray,
    hi: np.ndarray,
    robot_xy: np.ndarray,
    *,
    target_fill: float = DEFAULT_TARGET_FILL,
    standoff_override: float = 0.0,
    pick_chest_fn,
) -> Dict[str, Any]:
    """一次性离线估算 5D 目标位姿 + 预测投影统计。"""
    from behavior_interface.head_capture import get_head_sensor, head_intrinsics_tuple

    metrics = aabb_metrics(lo, hi)
    C = metrics["center"]
    head = get_head_sensor(world)
    if head is None:
        return {"ok": False, "error": "no head sensor"}

    fl, ha, w, h = head_intrinsics_tuple(head)

    so = float(standoff_override) if standoff_override is not None else 0.0
    d = so if so > 0 else estimate_standoff_m(
        lo, hi, w, h, fl, ha, target_fill=target_fill,
    )
    base_xy = estimate_base_xy(C, robot_xy, d)
    bx, by = float(base_xy[0]), float(base_xy[1])
    yaw = face_yaw_deg(bx, by, C)
    chest_z, theta_z = estimate_trunk_pose(C, base_xy, d, pick_chest_fn=pick_chest_fn)

    pred_st = None
    pred_err = None
    lateral_off_m = 0.0
    try:
        cam_pred, quat_pred = predict_head_pose_delta(
            world, np.array([bx, by], dtype=np.float64), yaw, chest_z, theta_z,
        )
        pred_st = project_aabb_to_head(cam_pred, quat_pred, lo, hi, w, h, fl, ha)
        if pred_st.get("ok"):
            fy = float(pred_st.get("fy", fl / ha * w))
            u_err = float(pred_st["cu"]) - w / 2.0
            if abs(u_err) > w * 0.06:
                lateral_off_m = float(np.clip(u_err / max(fy, 400.0) * d * 0.85, -0.14, 0.14))
                dir_cb = np.array([bx, by], dtype=np.float64) - C[:2]
                dn = float(np.linalg.norm(dir_cb))
                if dn > 1e-3:
                    dir_cb /= dn
                    perp = np.array([-dir_cb[1], dir_cb[0]], dtype=np.float64)
                    bx += float(perp[0] * lateral_off_m)
                    by += float(perp[1] * lateral_off_m)
                    yaw = face_yaw_deg(bx, by, C)
    except Exception as e:
        pred_err = str(e)

    reach_pred = estimate_dual_arm_reach_at_base(bx, by, chest_z, C)

    return {
        "ok": True,
        "object_metrics": metrics,
        "standoff_m": round(d, 4),
        "base_xy": [bx, by],
        "lateral_offset_m": round(lateral_off_m, 4),
        "theta_x_deg": round(yaw, 2),
        "chest_z": round(chest_z, 4),
        "theta_z_deg": round(theta_z, 2),
        "target_fill": target_fill,
        "camera": {"w": w, "h": h, "fl": fl, "ha": ha},
        "predicted_visibility": pred_st,
        "predicted_framing_ok": framing_ok(pred_st) if pred_st else False,
        "predict_error": pred_err,
        "predicted_arm_reach": reach_pred,
        "predicted_any_arm_reachable": reach_pred.get("any_reachable"),
    }


def correction_from_visibility(
    st: Dict[str, Any],
    *,
    standoff_m: float,
    chest_z: float,
    theta_z_deg: float,
    target_fill: float = DEFAULT_TARGET_FILL,
) -> Dict[str, float]:
    """根据实测 head 投影，计算下一拍位姿修正量（仍属离线几何，非逐步盲调）。"""
    if not st.get("ok"):
        # 角点不在相机前方：常见为俯仰不足，物体在画面外下方
        return {
            "delta_standoff": 0.06,
            "delta_chest_z": -0.03,
            "delta_theta_z": 10.0,
            "delta_yaw_deg": 0.0,
        }
    fill = float(st.get("fill_visible", 0))
    overflow = bool(st.get("overflow"))
    all_in = bool(st.get("all_corners_in"))
    w = int(st["image_width"])
    h = int(st["image_height"])
    u_err = float(st["cu"]) - w / 2.0
    v_err = float(st["cv"]) - h / 2.0

    # 距离：fill 缩放 ~ 1/d^2
    dd = 0.0
    if overflow or not all_in:
        dd += min(0.14, 0.06 + max(0.0, fill - DEFAULT_FILL_MAX) * 0.25)
    if fill > DEFAULT_FILL_MAX:
        dd += standoff_m * (math.sqrt(fill / max(target_fill, 0.05)) - 1.0) * 0.55
    elif fill < DEFAULT_FILL_MIN:
        dd -= min(0.12, standoff_m * (1.0 - math.sqrt(max(fill, 1e-3) / target_fill)) * 0.45)

    fy = float(st.get("fy", w * HEAD_FOCAL_LENGTH / HEAD_HORIZONTAL_APERTURE))
    dyaw = 0.0
    delta_lateral = 0.0
    if abs(u_err) > w * 0.08:
        delta_lateral = float(np.clip(u_err / max(fy, 400.0) * standoff_m * 0.75, -0.10, 0.10))
    elif abs(u_err) > w * 0.04:
        dyaw = float(np.clip(math.degrees(math.atan2(u_err, max(fy, 400.0))), -6.0, 6.0))

    dtz = 0.0
    if abs(v_err) > h * 0.04:
        dtz = float(np.clip(v_err / h * 22.0, -8.0, 8.0))

    dz = 0.0
    if v_err > h * 0.18:
        dz = -0.02
    elif v_err < -h * 0.18:
        dz = 0.02

    return {
        "delta_standoff": float(dd),
        "delta_chest_z": float(dz),
        "delta_theta_z": float(dtz),
        "delta_yaw_deg": float(dyaw),
        "delta_lateral_m": float(delta_lateral),
    }
