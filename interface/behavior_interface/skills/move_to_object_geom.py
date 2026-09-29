"""move_to_object / move_to_point 纯几何规划（reach=0.60 统一）。

当前管线：
  1. 真机先预降/升降，把 shoulder_z - target_z 尽量压到 0.4m 内；
  2. 离线模型按 q3 俯仰旋转胸廓 forward，取 S_mid + h3 * forward 的点，
     使其 z 与目标 z 对齐；
  3. 用 q3 俯仰后的肩高切面求弦球底盘 x,y,spin。

真机执行顺序：升降 → xy → spin → 俯仰。
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from behavior_interface.skills.base_chord_reach import (
    REACH_SPHERE_R_M,
    plan_base_chord_pose,
    reach_sphere_radius,
    shoulder_span_m,
    verify_chord_on_sphere,
    shoulder_positions_at_base,
)
from behavior_interface.trunk_vertical_lift import (
    R1PRO_Q_LIMITS,
    clamp_theta_z_pitch_q3_then_q1,
    estimate_chest_z_world,
    fk_torso_link4_forward,
    fk_torso_link4_theta_z_deg,
    solve_q3_for_theta_z_holding_q12,
    theta_z_from_trunk_q,
    theta_z_limits_pitch_q3_then_q1,
)

# 与弦球半径 R 相同；三角形高 h=√(R²−(肩宽/2)²)≈0.638，勿与 R 混淆
REACH_M = REACH_SPHERE_R_M

_SHOULDER_ABOVE_CHEST_M = 0.303
_CHEST_Z_MIN = 0.66
_CHEST_Z_MAX = 1.15
_THETA_Z_MIN = 74.0  # 规划搜索下限；上限由 URDF q1 限位 + 锁定 q2/q3 动态计算
# 俯身 θz 硬区间 [_THETA_Z_MIN_PITCH_DEG, _THETA_Z_MAX_PITCH_DEG] = [90, 165]：
# θz 为胸口 forward 与世界 +Z 的夹角。下限 90° 禁止 forward 反折朝上；
# 上限 165° 禁止躯干俯过头趴地导致自穿模/翻倒。
# q1/q2（升降）在调用规划前已确定，不受此区间影响；此处只钳俯身（q3）的目标 θz。
_THETA_Z_MIN_PITCH_DEG = float(
    os.environ.get("MOVE_TO_OBJECT_MIN_PITCH_THETA_Z_DEG", "90.0")
)
_THETA_Z_MAX_PITCH_DEG = float(
    os.environ.get("MOVE_TO_OBJECT_MAX_PITCH_THETA_Z_DEG", "165.0")
)
_LOW_Z_Q1_REPAIR_CHEST_Z_M = 0.73
_LOW_Z_Q1_REPAIR_MAX_FORWARD_LEAN_DEG = 90.0
_LOW_Z_Q1_REPAIR_MIN_THETA_Z_DEG = 90.0
_LOW_Z_Q1_REPAIR_MIN_CHEST_Z_M = 0.35
# 弦球半径放松：俯身解出来后若弦球容不下双肩，保持俯身（θz/肩高）不变，
# 只把弦球半径从默认 R 起每次 +_CHORD_RELAX_STEP_M 放松，直到弦球有解；
# 绝不因放松半径而重算俯身。上限到 _CHORD_RELAX_MAX_R_M（超出基本不可抓取）。
_CHORD_RELAX_STEP_M = float(os.environ.get("MOVE_TO_OBJECT_CHORD_RELAX_STEP_M", "0.01"))
_CHORD_RELAX_MAX_R_M = float(os.environ.get("MOVE_TO_OBJECT_CHORD_RELAX_MAX_R_M", "1.05"))


def isosceles_triangle_height(leg_m: float, base_m: float) -> float:
    """等腰三角形：底边 base，腰 leg → 顶点到底边中点的高。"""
    half = float(base_m) / 2.0
    return math.sqrt(max(leg_m * leg_m - half * half, 1e-9))


def _camera_view_dir_world(cam_quat_xyzw: np.ndarray) -> np.ndarray:
    """相机视线方向（世界系，沿光轴看出去）。"""
    from behavior_interface.skills.vlm_grasp_verify import _quat_to_mat

    R = _quat_to_mat(np.asarray(cam_quat_xyzw, dtype=np.float64).reshape(4))
    d = -R[:, 2]
    n = float(np.linalg.norm(d))
    return d / max(n, 1e-9)


def _pitch_axis_world(yaw_rad: float) -> np.ndarray:
    """绕水平轴俯仰（垂直于胸口 yaw）。"""
    return np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float64)


def _rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def _head_ray_at_theta_z(
    head_pos: np.ndarray,
    view_dir: np.ndarray,
    chest_pivot: np.ndarray,
    pitch_axis: np.ndarray,
    theta_z_now: float,
    theta_z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """胸口俯仰变化时，近似旋转 head 位置与视线。"""
    da = math.radians(float(theta_z) - float(theta_z_now))
    H = _rotate_about_axis(head_pos - chest_pivot, pitch_axis, da) + chest_pivot
    vd = _rotate_about_axis(view_dir, pitch_axis, da)
    vd = vd / max(float(np.linalg.norm(vd)), 1e-9)
    return H, vd


def _forward_from_yaw_theta(yaw_rad: float, theta_z_deg: float) -> np.ndarray:
    """胸廓 forward：theta_z 是与 +Z 的极角，yaw 是水平投影方向。"""
    tz = math.radians(float(theta_z_deg))
    return np.array([
        math.sin(tz) * math.cos(yaw_rad),
        math.sin(tz) * math.sin(yaw_rad),
        math.cos(tz),
    ], dtype=np.float64)


def _shoulder_forward_point_at_theta(
    pose: Dict[str, Any],
    theta_z: float,
    h3: float,
    base_link_z: float = 0.05,
) -> np.ndarray:
    """两肩中心出发，沿胸廓 forward 取 h3 距离的离线模型点。"""
    model = _q3_model_pose_at_theta(pose, float(theta_z), base_link_z=base_link_z)
    S_mid = np.asarray(model["shoulder_mid"], dtype=np.float64).reshape(3)
    yaw_rad = float(pose.get("theta_x_rad", 0.0))
    fwd = _forward_from_yaw_theta(yaw_rad, float(model["theta_z_deg"]))
    return S_mid + float(h3) * fwd


def _theta_z_limits_q3_only(q1: float, q2: float, n_scan: int = 128) -> Tuple[float, float]:
    """锁 q1/q2，只扫 q3 时的 θz 可达区间（仅统计 forward fx≥0 的未过折枝，
    否则 acos 折返会把真实俯角 >180° 的镜像读数混进区间）。"""
    lo3, hi3 = R1PRO_Q_LIMITS[2]
    vals = []
    for q3 in np.linspace(lo3, hi3, max(32, int(n_scan))):
        fwd = fk_torso_link4_forward(float(q1), float(q2), float(q3))
        if float(fwd[0]) < 0.0:
            continue
        fz = max(-1.0, min(1.0, float(fwd[2])))
        vals.append(math.degrees(math.acos(fz)))
    if not vals:
        vals = [
            fk_torso_link4_theta_z_deg(float(q1), float(q2), float(q3))
            for q3 in np.linspace(lo3, hi3, max(32, int(n_scan)))
        ]
    return float(min(vals)), float(max(vals))


def _q3_model_pose_at_theta(
    pose: Dict[str, Any],
    theta_z: float,
    base_link_z: float = 0.05,
) -> Dict[str, Any]:
    """锁 q1/q2，仅按 q3 反解 theta_z 后估算胸口/肩位置。"""
    q1 = float(pose.get("trunk_q1", 0.0))
    q2 = float(pose.get("trunk_q2", 0.0))
    q3_now = float(pose.get("trunk_q3", 0.0))
    q3 = solve_q3_for_theta_z_holding_q12(
        float(theta_z), q1, q2, prefer_q3=q3_now,
    )
    if q3 is None:
        q3 = q3_now
    try:
        from behavior_interface.trunk_vertical_lift import estimate_chest_z_world

        q4 = float(pose.get("trunk_q4", 0.0))
        q = np.array([q1, q2, float(q3), q4], dtype=np.float64)
        chest_z = float(estimate_chest_z_world(q, float(base_link_z)))
    except Exception:
        chest_z = float(pose.get("chest_z_now", 0.0))
    delta_z = chest_z - float(pose.get("chest_z_now", chest_z))
    theta_actual = float(theta_z_from_trunk_q(q1, q2, float(q3)))
    delta_theta = math.radians(theta_actual - float(pose.get("theta_z_now_deg", theta_actual)))
    axis = np.asarray(pose["pitch_axis"], dtype=np.float64).reshape(3)
    pivot0 = np.asarray(pose["chest_pivot"], dtype=np.float64).reshape(3)
    pivot = pivot0.copy()
    pivot[2] += delta_z

    out = dict(pose)
    for key in ("shoulder_mid", "left_shoulder", "right_shoulder", "head_pos"):
        if key not in pose:
            continue
        v0 = np.asarray(pose[key], dtype=np.float64).reshape(3)
        out[key] = _rotate_about_axis(v0 - pivot0, axis, delta_theta) + pivot
    out["chest_pivot"] = pivot
    out["theta_z_now_deg"] = theta_actual
    out["theta_z_deg"] = theta_actual
    out["trunk_q3"] = float(q3)
    out["chest_z_now"] = chest_z
    out["q3_model_delta_z_m"] = delta_z
    return out


def _pose_after_q3_theta_z(
    pose: Dict[str, Any],
    theta_z: float,
    base_link_z: float = 0.05,
) -> Dict[str, Any]:
    return _q3_model_pose_at_theta(pose, theta_z, base_link_z=base_link_z)


def _pose_after_trunk_q(
    pose: Dict[str, Any],
    trunk_q: np.ndarray,
    base_link_z: float = 0.05,
) -> Dict[str, Any]:
    """按目标 trunk q 估算胸口/肩/头位置；用于离线 q1+q3 俯仰修补。"""
    q = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    q1, q2, q3, q4 = [float(x) for x in q[:4]]
    try:
        chest_z = float(estimate_chest_z_world(q, float(base_link_z)))
    except Exception:
        chest_z = float(pose.get("chest_z_now", 0.0))
    theta_actual = float(theta_z_from_trunk_q(q1, q2, q3, q4))
    delta_z = chest_z - float(pose.get("chest_z_now", chest_z))
    delta_theta = math.radians(theta_actual - float(pose.get("theta_z_now_deg", theta_actual)))
    axis = np.asarray(pose["pitch_axis"], dtype=np.float64).reshape(3)
    pivot0 = np.asarray(pose["chest_pivot"], dtype=np.float64).reshape(3)
    pivot = pivot0.copy()
    pivot[2] += delta_z

    out = dict(pose)
    for key in ("shoulder_mid", "left_shoulder", "right_shoulder", "head_pos"):
        if key not in pose:
            continue
        v0 = np.asarray(pose[key], dtype=np.float64).reshape(3)
        out[key] = _rotate_about_axis(v0 - pivot0, axis, delta_theta) + pivot
    out["chest_pivot"] = pivot
    out["theta_z_now_deg"] = theta_actual
    out["theta_z_deg"] = theta_actual
    out["trunk_q1"] = q1
    out["trunk_q2"] = q2
    out["trunk_q3"] = q3
    out["trunk_q4"] = q4
    out["chest_z_now"] = chest_z
    out["q_model_delta_z_m"] = delta_z
    return out


def _shoulder_forward_point_at_trunk_q(
    pose: Dict[str, Any],
    trunk_q: np.ndarray,
    h3: float,
    base_link_z: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    model = _pose_after_trunk_q(pose, trunk_q, base_link_z=base_link_z)
    S_mid = np.asarray(model["shoulder_mid"], dtype=np.float64).reshape(3)
    yaw_rad = float(pose.get("theta_x_rad", 0.0))
    fwd = _forward_from_yaw_theta(yaw_rad, float(model["theta_z_deg"]))
    return S_mid + float(h3) * fwd, model


def _solve_q1_for_forward_point_z(
    pose: Dict[str, Any],
    *,
    object_z: float,
    h3: float,
    q2: float,
    q3: float,
    q4: float,
    base_link_z: float,
    prefer_q1: float,
    theta_z_min_deg: Optional[float] = None,
    theta_z_max_deg: Optional[float] = None,
    chest_z_min_m: Optional[float] = None,
) -> Optional[Tuple[float, np.ndarray, Dict[str, Any], float]]:
    """锁 q2/q3/q4，只调 q1，使 shoulder_mid+h3*forward 的 z 接近 object_z。"""
    lo1, hi1 = R1PRO_Q_LIMITS[0]
    z_obj = float(object_z)
    theta_z_min = None if theta_z_min_deg is None else float(theta_z_min_deg)
    theta_z_max = None if theta_z_max_deg is None else float(theta_z_max_deg)
    chest_z_min = None if chest_z_min_m is None else float(chest_z_min_m)

    def eval_q1(q1: float) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        q = np.array([float(q1), float(q2), float(q3), float(q4)], dtype=np.float64)
        P, model = _shoulder_forward_point_at_trunk_q(
            pose, q, h3, base_link_z=base_link_z,
        )
        return float(P[2] - z_obj), P, model

    def allowed(row: Tuple[float, float, np.ndarray, Dict[str, Any]]) -> bool:
        theta_z = float(row[3].get("theta_z_deg", 999.0))
        chest_z = float(row[3].get("chest_z_now", 999.0))
        if theta_z_min is not None and theta_z < theta_z_min - 0.25:
            return False
        if theta_z_max is None:
            theta_ok = True
        else:
            theta_ok = theta_z <= theta_z_max + 0.25
        if chest_z_min is not None and chest_z < chest_z_min - 1e-4:
            return False
        return theta_ok

    samples = []
    for q1 in np.linspace(lo1, hi1, 161):
        err, P, model = eval_q1(float(q1))
        samples.append((float(q1), err, P, model))

    allowed_samples = [row for row in samples if allowed(row)]
    search_samples = allowed_samples or samples
    best = min(
        search_samples,
        key=lambda row: (abs(row[1]), abs(row[0] - float(prefer_q1))),
    )
    bracket = None
    for a, b in zip(samples[:-1], samples[1:]):
        if not (allowed(a) and allowed(b)):
            continue
        if a[1] == 0.0 or a[1] * b[1] <= 0.0:
            mid_pref = 0.5 * (a[0] + b[0])
            score = abs(mid_pref - float(prefer_q1))
            if bracket is None or score < bracket[0]:
                bracket = (score, a, b)

    if bracket is not None:
        _, a, b = bracket
        lo, hi = float(a[0]), float(b[0])
        fa = float(a[1])
        best_local = best
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            fm, Pm, Mm = eval_q1(mid)
            mid_row = (mid, fm, Pm, Mm)
            if not allowed(mid_row):
                if hi > lo:
                    hi = mid
                continue
            if abs(fm) < abs(best_local[1]):
                best_local = (mid, fm, Pm, Mm)
            if abs(fm) < 0.0015:
                break
            if fa * fm <= 0.0:
                hi = mid
            else:
                lo = mid
                fa = fm
        best = best_local

    q1_best, err_best, P_best, model_best = best
    model_best["theta_z_min_constraint_deg"] = theta_z_min
    model_best["theta_z_max_constraint_deg"] = theta_z_max
    model_best["chest_z_min_constraint_m"] = chest_z_min
    model_best["theta_z_max_constraint_satisfied"] = allowed(best)
    q = np.array([q1_best, q2, q3, q4], dtype=np.float64)
    return float(q1_best), q, model_best, float(err_best)


def ray_sphere_forward_t(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> Optional[float]:
    """射线 O+t·d 与球 |P-C|=R 的第一个正向交点 t。"""
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    v = o - c
    b = float(np.dot(d, v))
    disc = b * b - (float(np.dot(v, v)) - radius * radius)
    if disc < 0:
        return None
    s = math.sqrt(disc)
    ts = [t for t in (-b - s, -b + s) if t >= 1e-4]
    if not ts:
        return None
    return min(ts)


def read_shoulder_head_pose(world) -> Dict[str, Any]:
    """当前双肩中点、head 光心与胸口。"""
    from behavior_interface.head_capture import get_head_sensor

    sl = world.shoulder_pose("left")
    sr = world.shoulder_pose("right")
    L = np.array([sl["x"], sl["y"], sl["z"]], dtype=np.float64)
    R = np.array([sr["x"], sr["y"], sr["z"]], dtype=np.float64)
    S_mid = 0.5 * (L + R)
    span = float(np.linalg.norm(R - L))

    head = get_head_sensor(world)
    if head is None:
        raise RuntimeError("no head sensor")
    pos, quat = head.get_position_orientation()
    H = np.asarray(pos, dtype=np.float64).reshape(3)
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    view = _camera_view_dir_world(quat)

    chest = world.chest_pose()
    pivot = np.array([chest["x"], chest["y"], chest["z"]], dtype=np.float64)
    tz0 = float(chest["theta_z_deg"])
    trunk_q = np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4)
    q1, q2, q3 = float(trunk_q[0]), float(trunk_q[1]), float(trunk_q[2])
    q4 = float(trunk_q[3]) if trunk_q.shape[0] > 3 else 0.0
    tz_min_urdf, tz_max_urdf = theta_z_limits_pitch_q3_then_q1(q1, q2, q3)
    tz_fk_now = fk_torso_link4_theta_z_deg(q1, q2, q3)
    yaw_rad = math.radians(float(chest["theta_x_deg"]))
    axis = _pitch_axis_world(yaw_rad)

    leg_l = float(np.linalg.norm(L - H))
    leg_r = float(np.linalg.norm(R - H))
    leg_head = 0.5 * (leg_l + leg_r)

    return {
        "left_shoulder": L,
        "right_shoulder": R,
        "shoulder_mid": S_mid,
        "shoulder_span_m": span,
        "head_pos": H,
        "head_quat": quat,
        "view_dir": view,
        "chest_pivot": pivot,
        "theta_z_now_deg": tz0,
        "theta_x_deg": float(chest["theta_x_deg"]),
        "theta_x_rad": yaw_rad,
        # 俯身 θz 钳到区间 [_THETA_Z_MIN_PITCH_DEG, _THETA_Z_MAX_PITCH_DEG]=[90,165]：
        # 下限取较大值禁止 forward 朝上（反折）；上限取较小值禁止俯过 165° 趴地。
        "theta_z_min_deg": max(_THETA_Z_MIN, tz_min_urdf, _THETA_Z_MIN_PITCH_DEG),
        "theta_z_max_deg": min(tz_max_urdf, _THETA_Z_MAX_PITCH_DEG),
        "theta_z_fk_now_deg": round(tz_fk_now, 2),
        "trunk_q1": q1,
        "trunk_q2": q2,
        "trunk_q3": q3,
        "trunk_q4": q4,
        "pitch_axis": axis,
        "leg_head_m": leg_head,
        "chest_z_now": float(chest["z"]),
    }


def solve_theta_z_ray_sphere(
    pose: Dict[str, Any],
    object_z: float,
    reach: Optional[float] = None,
    *,
    n_bisect: int = 32,
) -> Dict[str, Any]:
    """步骤 1–3：h 球 + head 视线交点高度对齐物体 z。"""
    R = float(reach if reach is not None else reach_sphere_radius())
    span = float(pose["shoulder_span_m"])
    h = isosceles_triangle_height(R, span)
    p = isosceles_triangle_height(float(pose["leg_head_m"]), span)

    S_mid = pose["shoulder_mid"]
    H0 = pose["head_pos"]
    v0 = pose["view_dir"]
    pivot = pose["chest_pivot"]
    axis = pose["pitch_axis"]
    tz0 = float(pose["theta_z_now_deg"])
    z_obj = float(object_z)

    def _intersection_at(tz: float) -> Optional[Tuple[float, np.ndarray]]:
        Hp, vd = _head_ray_at_theta_z(H0, v0, pivot, axis, tz0, tz)
        t = ray_sphere_forward_t(Hp, vd, S_mid, h)
        if t is None:
            return None
        I = Hp + t * vd
        return float(I[2]), I

    lo = float(pose.get("theta_z_min_deg", _THETA_Z_MIN))
    hi = float(pose.get("theta_z_max_deg", 165.0))
    f_lo = _intersection_at(lo)
    f_hi = _intersection_at(hi)
    if f_lo is None or f_hi is None:
        # 退化：用高度差 / h 估计
        dz = float(H0[2]) - z_obj
        Z = math.degrees(math.atan2(dz, h)) if h > 1e-6 else 0.0
        tz = float(np.clip(90.0 + Z, lo, hi))
        q2 = float(pose.get("trunk_q2", 0.0))
        q3 = float(pose.get("trunk_q3", 0.0))
        q1_pref = float(pose.get("trunk_q1", 0.0))
        q3_pref = float(pose.get("trunk_q3", 0.0))
        tz, feas = clamp_theta_z_pitch_q3_then_q1(
            tz, q1_pref, q2, q3_pref, prefer_q1=q1_pref, prefer_q3=q3_pref,
        )
        hit = _intersection_at(tz)
        return {
            "theta_z_deg": round(tz, 2),
            "fallback": "atan2_dz_h",
            "pitch_q1_feasibility": feas,
            "h_m": round(h, 4),
            "p_m": round(p, 4),
            "reach_m": R,
            "intersection_z": round(hit[0], 4) if hit else None,
            "object_z": z_obj,
            "err_z_m": round((hit[0] - z_obj) if hit else 999.0, 4),
        }

    z_lo = f_lo[0] - z_obj
    z_hi = f_hi[0] - z_obj
    best_tz = tz0
    best_err = 1e9
    best_I = None

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        f_mid = _intersection_at(mid)
        if f_mid is None:
            break
        z_mid = f_mid[0] - z_obj
        err = abs(z_mid)
        if err < best_err:
            best_err = err
            best_tz = mid
            best_I = f_mid[1]
        if abs(z_mid) < 0.002:
            best_tz = mid
            best_I = f_mid[1]
            best_err = err
            break
        if z_lo * z_mid <= 0:
            hi = mid
            z_hi = z_mid
        else:
            lo = mid
            z_lo = z_mid

    hit = _intersection_at(best_tz)
    if hit:
        best_I = hit[1]
        best_err = abs(hit[0] - z_obj)

    q2 = float(pose.get("trunk_q2", 0.0))
    q3 = float(pose.get("trunk_q3", 0.0))
    q1_pref = float(pose.get("trunk_q1", 0.0))
    q3_pref = float(pose.get("trunk_q3", 0.0))
    theta_z_geom, feas = clamp_theta_z_pitch_q3_then_q1(
        float(np.clip(best_tz, lo, hi)),
        q1_pref,
        q2,
        q3_pref,
        prefer_q1=q1_pref,
        prefer_q3=q3_pref,
    )
    if hit and feas.get("theta_z_saturated"):
        hit_sat = _intersection_at(theta_z_geom)
        if hit_sat is not None:
            best_I = hit_sat[1]
            best_err = abs(hit_sat[0] - z_obj)

    return {
        "theta_z_deg": round(theta_z_geom, 2),
        "theta_z_ray_bisect_deg": round(float(np.clip(best_tz, lo, hi)), 2),
        "theta_z_search_lo_deg": round(lo, 2),
        "theta_z_search_hi_deg": round(hi, 2),
        "pitch_q1_feasibility": feas,
        "h_m": round(h, 4),
        "p_m": round(p, 4),
        "reach_m": R,
        "intersection_z": round(hit[0], 4) if hit else None,
        "intersection_xyz": [round(float(x), 4) for x in best_I] if best_I is not None else None,
        "object_z": z_obj,
        "err_z_m": round(best_err, 4),
        "theta_z_now_deg": round(tz0, 2),
    }


def solve_theta_z_shoulder_forward(
    pose: Dict[str, Any],
    object_z: float,
    reach: Optional[float] = None,
    *,
    n_bisect: int = 32,
    base_link_z: float = 0.05,
) -> Dict[str, Any]:
    """按 shoulder_mid + h3 * chest_forward(theta_z) 对齐目标 z。"""
    R = float(reach if reach is not None else reach_sphere_radius())
    span = float(pose["shoulder_span_m"])
    h3 = isosceles_triangle_height(R, span)
    z_obj = float(object_z)

    q1_pref = float(pose.get("trunk_q1", 0.0))
    q2 = float(pose.get("trunk_q2", 0.0))
    q3_pref = float(pose.get("trunk_q3", 0.0))
    q4_pref = float(pose.get("trunk_q4", 0.0))
    q3_lo, q3_hi = _theta_z_limits_q3_only(q1_pref, q2)
    lo = max(float(pose.get("theta_z_min_deg", _THETA_Z_MIN)), q3_lo)
    hi = min(float(pose.get("theta_z_max_deg", 165.0)), q3_hi)
    tz0 = float(pose["theta_z_now_deg"])
    if hi < lo:
        lo, hi = min(q3_lo, q3_hi), max(q3_lo, q3_hi)

    def _point_at(tz: float) -> np.ndarray:
        return _shoulder_forward_point_at_theta(
            pose, float(tz), h3, base_link_z=base_link_z,
        )

    def _err(tz: float) -> float:
        return float(_point_at(float(tz))[2] - z_obj)

    z_lo = _err(lo)
    z_hi = _err(hi)
    best_tz = float(np.clip(tz0, lo, hi))
    best_I = _point_at(best_tz)
    best_err = abs(float(best_I[2] - z_obj))

    if z_lo == 0.0:
        best_tz = lo
        best_I = _point_at(best_tz)
        best_err = 0.0
    elif z_hi == 0.0:
        best_tz = hi
        best_I = _point_at(best_tz)
        best_err = 0.0
    elif z_lo * z_hi <= 0:
        a, b = lo, hi
        fa = z_lo
        for _ in range(n_bisect):
            mid = 0.5 * (a + b)
            fm = _err(mid)
            err = abs(fm)
            if err < best_err:
                best_tz = mid
                best_I = _point_at(mid)
                best_err = err
            if err < 0.002:
                break
            if fa * fm <= 0:
                b = mid
            else:
                a = mid
                fa = fm
    else:
        # No bracket in the URDF range: choose the boundary with smaller height error.
        for tz in (lo, hi):
            I = _point_at(tz)
            err = abs(float(I[2] - z_obj))
            if err < best_err:
                best_tz = tz
                best_I = I
                best_err = err

    theta_z_req = float(np.clip(best_tz, lo, hi))
    q3_sol = solve_q3_for_theta_z_holding_q12(
        theta_z_req, q1_pref, q2, prefer_q3=q3_pref,
    )
    if q3_sol is None:
        theta_z_geom = theta_z_req
        saturated = False
    else:
        theta_z_geom = float(fk_torso_link4_theta_z_deg(q1_pref, q2, q3_sol))
        saturated = abs(theta_z_geom - best_tz) > 0.5
    if abs(theta_z_geom - best_tz) > 1e-4:
        best_I = _point_at(theta_z_geom)
        best_err = abs(float(best_I[2] - z_obj))
    feas = {
        "theta_z_requested_deg": round(float(best_tz), 2),
        "theta_z_achieved_deg": round(theta_z_geom, 2),
        "theta_z_achievable_lo_deg": round(lo, 2),
        "theta_z_achievable_hi_deg": round(hi, 2),
        "pitch_phase": "q3_only",
        "q3_required_rad": round(float(q3_sol), 4) if q3_sol is not None else None,
        "q1_required_rad": round(q1_pref, 4),
        "theta_z_saturated": bool(saturated),
    }

    # 耦合正确的完整目标关节角：q1,q2,q4 沿用升降后配置，q3 用上面求解到的 q3_sol。
    # 该 q 经 FK 即得到规划的 chest_z + θz，供执行端「直驱到位再锁死」（避免解耦三段塌低/失锁）。
    q4 = float(pose.get("trunk_q4", 0.0))
    target_trunk_q = (
        [round(float(q1_pref), 5), round(float(q2), 5),
         round(float(q3_sol), 5), round(q4, 5)]
        if q3_sol is not None and np.isfinite(q3_sol)
        else None
    )

    return {
        "theta_z_deg": round(theta_z_geom, 2),
        "theta_z_ray_bisect_deg": round(float(best_tz), 2),
        "theta_z_search_lo_deg": round(lo, 2),
        "theta_z_search_hi_deg": round(hi, 2),
        "pitch_q1_feasibility": feas,
        "h_m": round(h3, 4),
        "h3_m": round(h3, 4),
        "reach_m": R,
        "intersection_z": round(float(best_I[2]), 4),
        "intersection_xyz": [round(float(x), 4) for x in best_I],
        "object_z": z_obj,
        "err_z_m": round(best_err, 4),
        "theta_z_now_deg": round(tz0, 2),
        "pitch_model": "shoulder_mid_forward_h3",
        "target_trunk_q": target_trunk_q,
        "bracketed": bool(z_lo * z_hi <= 0),
        "z_err_at_lo_m": round(z_lo, 4),
        "z_err_at_hi_m": round(z_hi, 4),
    }


def solve_theta_z_shoulder_forward_low_z_repair(
    pose: Dict[str, Any],
    object_z: float,
    reach: Optional[float] = None,
    *,
    n_bisect: int = 32,
    base_link_z: float = 0.05,
) -> Dict[str, Any]:
    """低位大俯角修补：q3 限到 60° 俯身，余量只调 q1 补射线高度。"""
    pitch = solve_theta_z_shoulder_forward(
        pose,
        object_z,
        reach=reach,
        n_bisect=n_bisect,
        base_link_z=base_link_z,
    )
    theta_q3_only = float(pitch.get("theta_z_deg", 90.0))
    lean_q3_only = max(0.0, theta_q3_only - 90.0)
    chest_z = float(pose.get("chest_z_now", 999.0))
    if (
        chest_z > _LOW_Z_Q1_REPAIR_CHEST_Z_M
        or lean_q3_only <= _LOW_Z_Q1_REPAIR_MAX_FORWARD_LEAN_DEG
    ):
        pitch["low_z_q1_repair"] = {
            "applied": False,
            "reason": (
                "not_low_z" if chest_z > _LOW_Z_Q1_REPAIR_CHEST_Z_M
                else "q3_lean_within_limit"
            ),
            "chest_z_now": round(chest_z, 4),
            "lean_q3_only_deg": round(lean_q3_only, 2),
            "lean_limit_deg": _LOW_Z_Q1_REPAIR_MAX_FORWARD_LEAN_DEG,
        }
        return pitch

    R = float(reach if reach is not None else reach_sphere_radius())
    span = float(pose["shoulder_span_m"])
    h3 = isosceles_triangle_height(R, span)
    q1_pref = float(pose.get("trunk_q1", 0.0))
    q2 = float(pose.get("trunk_q2", 0.0))
    q3_pref = float(pose.get("trunk_q3", 0.0))
    q4 = float(pose.get("trunk_q4", 0.0))
    theta_cap = 90.0 + _LOW_Z_Q1_REPAIR_MAX_FORWARD_LEAN_DEG
    q3_cap = solve_q3_for_theta_z_holding_q12(
        theta_cap,
        q1_pref,
        q2,
        prefer_q3=q3_pref,
    )
    if q3_cap is None:
        pitch["low_z_q1_repair"] = {
            "applied": False,
            "reason": "q3_cap_unreachable",
            "theta_cap_deg": theta_cap,
        }
        return pitch
    theta_cap_actual = float(fk_torso_link4_theta_z_deg(q1_pref, q2, q3_cap))
    repair = _solve_q1_for_forward_point_z(
        pose,
        object_z=float(object_z),
        h3=h3,
        q2=q2,
        q3=float(q3_cap),
        q4=q4,
        base_link_z=base_link_z,
        prefer_q1=q1_pref,
        theta_z_min_deg=_LOW_Z_Q1_REPAIR_MIN_THETA_Z_DEG,
        theta_z_max_deg=theta_cap,
        chest_z_min_m=_LOW_Z_Q1_REPAIR_MIN_CHEST_Z_M,
    )
    if repair is None:
        pitch["low_z_q1_repair"] = {
            "applied": False,
            "reason": "q1_solve_failed",
            "theta_cap_deg": round(theta_cap_actual, 2),
        }
        return pitch

    q1_sol, q_target, model, err_z = repair
    P, model = _shoulder_forward_point_at_trunk_q(
        pose,
        q_target,
        h3,
        base_link_z=base_link_z,
    )
    theta_final = float(model["theta_z_deg"])
    out = dict(pitch)
    feas = dict(pitch.get("pitch_q1_feasibility") or {})
    feas.update({
        "pitch_phase": "low_z_q3_cap_q1_repair",
        "q1_required_rad": round(float(q1_sol), 4),
        "q2_required_rad": round(q2, 4),
        "q3_required_rad": round(float(q3_cap), 4),
        "q4_required_rad": round(q4, 4),
        "theta_z_achieved_deg": round(theta_final, 2),
        "theta_z_q3_only_deg": round(theta_q3_only, 2),
        "theta_z_q3_cap_deg": round(theta_cap_actual, 2),
        "theta_z_saturated": False,
    })
    out.update({
        "theta_z_deg": round(theta_final, 2),
        "theta_z_q3_only_deg": round(theta_q3_only, 2),
        "theta_z_q3_cap_deg": round(theta_cap_actual, 2),
        "pitch_q1_feasibility": feas,
        "intersection_z": round(float(P[2]), 4),
        "intersection_xyz": [round(float(x), 4) for x in P],
        "err_z_m": round(float(err_z), 4),
        "pitch_model": "shoulder_mid_forward_h3_low_z_q3_cap_q1_repair",
        "target_trunk_q": [round(float(x), 5) for x in q_target.tolist()],
        "target_chest_z": round(float(model.get("chest_z_now", chest_z)), 4),
        "low_z_q1_repair": {
            "applied": True,
            "chest_z_now": round(chest_z, 4),
            "chest_z_threshold_m": _LOW_Z_Q1_REPAIR_CHEST_Z_M,
            "lean_q3_only_deg": round(lean_q3_only, 2),
            "lean_limit_deg": _LOW_Z_Q1_REPAIR_MAX_FORWARD_LEAN_DEG,
            "theta_z_q3_only_deg": round(theta_q3_only, 2),
            "theta_z_q3_cap_deg": round(theta_cap_actual, 2),
            "theta_z_final_deg": round(theta_final, 2),
            "q1_delta_rad": round(float(q1_sol - q1_pref), 4),
            "q3_delta_rad": round(float(q3_cap - q3_pref), 4),
            "err_z_m": round(float(err_z), 4),
            "theta_z_min_constraint_deg": _LOW_Z_Q1_REPAIR_MIN_THETA_Z_DEG,
            "theta_z_max_constraint_deg": round(theta_cap, 2),
            "chest_z_min_constraint_m": _LOW_Z_Q1_REPAIR_MIN_CHEST_Z_M,
            "theta_z_max_constraint_satisfied": bool(
                model.get("theta_z_max_constraint_satisfied", True)
            ),
        },
    })
    return out


def solve_q1_forward_height_with_current_q3(
    pose: Dict[str, Any],
    object_z: float,
    reach: Optional[float] = None,
    *,
    base_link_z: float = 0.05,
) -> Dict[str, Any]:
    """执行期第二段：锁当前 q2/q3/q4，只调 q1 补 shoulder-forward 点高度。"""
    R = float(reach if reach is not None else reach_sphere_radius())
    span = float(pose["shoulder_span_m"])
    h3 = isosceles_triangle_height(R, span)
    q1_pref = float(pose.get("trunk_q1", 0.0))
    q2 = float(pose.get("trunk_q2", 0.0))
    q3 = float(pose.get("trunk_q3", 0.0))
    q4 = float(pose.get("trunk_q4", 0.0))
    chest_z = float(pose.get("chest_z_now", 999.0))
    theta_cap = 90.0 + _LOW_Z_Q1_REPAIR_MAX_FORWARD_LEAN_DEG
    repair = _solve_q1_for_forward_point_z(
        pose,
        object_z=float(object_z),
        h3=h3,
        q2=q2,
        q3=q3,
        q4=q4,
        base_link_z=base_link_z,
        prefer_q1=q1_pref,
        theta_z_min_deg=_LOW_Z_Q1_REPAIR_MIN_THETA_Z_DEG,
        theta_z_max_deg=theta_cap,
        chest_z_min_m=_LOW_Z_Q1_REPAIR_MIN_CHEST_Z_M,
    )
    if repair is None:
        return {
            "ok": False,
            "error": "q1_solve_failed",
            "h3_m": round(h3, 4),
            "reach_m": R,
        }

    q1_sol, q_target, model, err_z = repair
    P, model = _shoulder_forward_point_at_trunk_q(
        pose,
        q_target,
        h3,
        base_link_z=base_link_z,
    )
    theta_final = float(model["theta_z_deg"])
    return {
        "ok": True,
        "theta_z_deg": round(theta_final, 2),
        "theta_z_now_deg": round(float(pose.get("theta_z_now_deg", theta_final)), 2),
        "target_trunk_q": [round(float(x), 5) for x in q_target.tolist()],
        "target_chest_z": round(float(model.get("chest_z_now", chest_z)), 4),
        "intersection_z": round(float(P[2]), 4),
        "intersection_xyz": [round(float(x), 4) for x in P],
        "object_z": float(object_z),
        "err_z_m": round(float(err_z), 4),
        "h3_m": round(h3, 4),
        "reach_m": R,
        "low_z_q1_repair": {
            "applied": True,
            "runtime_second_phase": True,
            "chest_z_now": round(chest_z, 4),
            "theta_z_final_deg": round(theta_final, 2),
            "theta_z_min_constraint_deg": _LOW_Z_Q1_REPAIR_MIN_THETA_Z_DEG,
            "theta_z_max_constraint_deg": round(theta_cap, 2),
            "chest_z_min_constraint_m": _LOW_Z_Q1_REPAIR_MIN_CHEST_Z_M,
            "theta_z_max_constraint_satisfied": bool(
                model.get("theta_z_max_constraint_satisfied", True)
            ),
            "q1_delta_rad": round(float(q1_sol - q1_pref), 4),
            "q2_locked_rad": round(q2, 4),
            "q3_locked_rad": round(q3, 4),
            "q4_locked_rad": round(q4, 4),
            "err_z_m": round(float(err_z), 4),
        },
    }


def plan_deltaz_for_chord(
    object_center: np.ndarray,
    robot_xy: np.ndarray,
    shoulder_z_now: float,
    chest_z_now: float,
    *,
    is_free_xy: Optional[Callable[[float, float], bool]] = None,
    nav_clearance_fn: Optional[Callable[[float, float], tuple]] = None,
    reach: Optional[float] = None,
    deltaz_span: float = 0.06,
    n_steps: int = 7,
) -> Tuple[float, float, Dict[str, Any], Dict[str, Any]]:
    """步骤 4：当前肩高切面弦球；返回 chest_z、delta_chest、底盘 plan。"""
    R = float(reach if reach is not None else reach_sphere_radius())
    best_plan: Dict[str, Any] = {}
    best_sc = 1e9
    best_cz = float(chest_z_now)
    best_dz = 0.0

    for dz in np.linspace(-deltaz_span, deltaz_span, n_steps):
        cz = float(np.clip(chest_z_now + dz, _CHEST_Z_MIN, _CHEST_Z_MAX))
        sh_z = float(shoulder_z_now) + float(cz - chest_z_now)
        plan = plan_base_chord_pose(
            object_center,
            robot_xy,
            shoulder_z=sh_z,
            is_free_xy=is_free_xy,
            nav_clearance_fn=nav_clearance_fn,
            reach_R=R,
        )
        if not plan.get("ok"):
            continue
        detail = plan.get("plan_detail") or {}
        chord_ok = bool(detail.get("chord_ok"))
        dists = detail.get("chord_dist") or {}
        nav = detail.get("nav_clearance") or {}
        err = abs(float(dists.get("left_dist", R)) - R) + abs(
            float(dists.get("right_dist", R)) - R
        )
        min_clr = float(nav.get("min_clearance_m", -1.0))
        sc = (
            -min_clr * 5.0
            + err
            + (0.0 if chord_ok else 1.2)
            + 0.2 * abs(dz)
            + (3.0 if not nav.get("all_free", True) else 0.0)
        )
        if sc < best_sc:
            best_sc = sc
            best_cz = cz
            best_dz = float(cz - chest_z_now)
            best_plan = plan

    meta = {
        "shoulder_z_slice_m": round(float(shoulder_z_now), 4),
        "delta_chest_z_m": round(best_dz, 4),
        "chest_z_target": round(best_cz, 4),
        "chord_scan_score": round(best_sc, 4),
    }
    return best_cz, best_dz, meta, best_plan


def _pose_shift_chest_z(pose: Dict[str, Any], delta_z: float) -> Dict[str, Any]:
    """按胸口升降量平移肩/头/胸口参考点（离线俯仰一阶近似）。"""
    out = dict(pose)
    dz = float(delta_z)
    for key in (
        "shoulder_mid", "head_pos", "chest_pivot",
        "left_shoulder", "right_shoulder",
    ):
        if key in out:
            v = np.asarray(out[key], dtype=np.float64).reshape(3).copy()
            v[2] += dz
            out[key] = v
    out["chest_z_now"] = float(pose.get("chest_z_now", 0.0)) + dz
    return out


def plan_geom_phases(
    world,
    object_center: np.ndarray,
    robot_xy: np.ndarray,
    *,
    is_free_xy: Optional[Callable[[float, float], bool]] = None,
    nav_clearance_fn: Optional[Callable[[float, float], tuple]] = None,
    reach: Optional[float] = None,
) -> Dict[str, Any]:
    """完整离线几何规划。

    俯仰 R_pitch：solve_theta_z_ray_sphere 用（默认 0.60m）。
    底盘 R_chord：与 R_pitch 相同（弦球取点 / xy 移动）。
    """
    R_pitch = float(reach if reach is not None else reach_sphere_radius())
    R_chord = R_pitch
    C = np.asarray(object_center, dtype=np.float64).reshape(3)
    robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(2)

    try:
        pose = read_shoulder_head_pose(world)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    sh_z_now = float(pose["shoulder_mid"][2])
    chest_now = float(pose["chest_z_now"])

    chest_z, deltaz, base_meta, plan = plan_deltaz_for_chord(
        C, robot_xy, sh_z_now, chest_now,
        is_free_xy=is_free_xy,
        nav_clearance_fn=nav_clearance_fn,
        reach=R_chord,
    )
    if not plan.get("ok"):
        return {"ok": False, "error": plan.get("error", "弦球规划失败")}

    sh_z_plan = float(sh_z_now) + float(chest_z - chest_now)

    # 俯仰：dz 确定后在目标胸口高度解射线×球（仅 z 平移肩/头，与规划 xy/yaw 无关）
    pose_at_chest = _pose_shift_chest_z(pose, float(chest_z - chest_now))
    pitch = solve_theta_z_ray_sphere(pose_at_chest, float(C[2]), reach=R_pitch)
    if float(pitch.get("err_z_m", 0.0)) > 0.015:
        pitch["theta_z_low_confidence"] = True
    theta_z_geom = float(pitch.get("theta_z_deg", pitch.get("theta_z_ray_bisect_deg", 90.0)))
    theta_z, feas = clamp_theta_z_pitch_q3_then_q1(
        theta_z_geom,
        float(pose["trunk_q1"]),
        float(pose["trunk_q2"]),
        float(pose["trunk_q3"]),
        prefer_q1=float(pose["trunk_q1"]),
        prefer_q3=float(pose["trunk_q3"]),
    )
    pitch = dict(pitch)
    pitch["theta_z_geom_deg"] = round(theta_z_geom, 2)
    pitch["pitch_q1_feasibility_final"] = feas
    pitch["theta_z_deg"] = round(theta_z, 2)

    bx, by = plan["base_xy"]
    yaw = float(plan["theta_x_deg"])

    detail = plan.get("plan_detail") or {}
    if detail.get("left_shoulder") and detail.get("right_shoulder"):
        left = np.asarray(detail["left_shoulder"], dtype=np.float64).reshape(3)
        right = np.asarray(detail["right_shoulder"], dtype=np.float64).reshape(3)
        chord_ok, chord_dist = verify_chord_on_sphere(C, left, right, R_chord)
    else:
        chord_ok = bool(detail.get("chord_ok", False))
        chord_dist = detail.get("chord_dist") or {}

    trunk_meta = {
        "phase": "geom_ray_sphere",
        "reach_pitch_m": R_pitch,
        "reach_chord_m": R_chord,
        "theta_z_geom_deg": pitch.get("theta_z_geom_deg"),
        "shoulder_span_m": round(float(pose["shoulder_span_m"]), 4),
        "chest_z_now": round(chest_now, 4),
        "theta_z_now_deg": pitch.get("theta_z_now_deg"),
        "pitch_geom": pitch,
        "base_geom": base_meta,
    }

    return {
        "ok": True,
        "chest_z": chest_z,
        "theta_z_deg": theta_z,
        "shoulder_z": sh_z_plan,
        "delta_chest_z_m": deltaz,
        "base_plan": plan,
        "base_xy": [float(bx), float(by)],
        "theta_x_deg": yaw,
        "trunk_meta": trunk_meta,
        "reach_pitch_m": R_pitch,
        "reach_chord_m": R_chord,
        "theta_z_geom_deg": pitch.get("theta_z_geom_deg"),
        "chord_ok_preview": chord_ok,
        "chord_dist_preview": chord_dist,
    }


def plan_geom_from_pose(
    pose: Dict[str, Any],
    object_center: np.ndarray,
    robot_xy: np.ndarray,
    *,
    is_free_xy: Optional[Callable[[float, float], bool]] = None,
    nav_clearance_fn: Optional[Callable[[float, float], tuple]] = None,
    reach: Optional[float] = None,
    base_link_z: float = 0.05,
) -> Dict[str, Any]:
    """预降/升降后的几何规划：q3-only 俯仰模型 → 肩高切面弦球底盘。"""
    R_pitch = float(reach if reach is not None else reach_sphere_radius())
    R_chord = R_pitch
    C = np.asarray(object_center, dtype=np.float64).reshape(3)
    robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(2)

    chest_now = float(pose["chest_z_now"])
    disable_lowz_q1 = os.environ.get("MOVE_TO_OBJECT_DISABLE_LOWZ_Q1_REPAIR", "").strip() in (
        "1", "true", "True", "yes", "on",
    )
    if disable_lowz_q1:
        pitch = solve_theta_z_shoulder_forward(
            pose, float(C[2]), reach=R_pitch, base_link_z=base_link_z,
        )
        pitch["low_z_q1_repair"] = {
            "applied": False,
            "reason": "disabled_by_env",
        }
    else:
        pitch = solve_theta_z_shoulder_forward_low_z_repair(
            pose, float(C[2]), reach=R_pitch, base_link_z=base_link_z,
        )
    if float(pitch.get("err_z_m", 0.0)) > 0.015:
        pitch["theta_z_low_confidence"] = True
    theta_z = float(pitch.get("theta_z_deg", 90.0))

    target_trunk_q = pitch.get("target_trunk_q")
    if target_trunk_q is not None:
        pose_after_pitch = _pose_after_trunk_q(
            pose,
            np.asarray(target_trunk_q, dtype=np.float64).reshape(4),
            base_link_z=base_link_z,
        )
    else:
        pose_after_pitch = _pose_after_q3_theta_z(pose, theta_z, base_link_z=base_link_z)
    sh_z_plan = float(np.asarray(pose_after_pitch["shoulder_mid"], dtype=np.float64)[2])
    chest_z_plan = float(pose_after_pitch.get("chest_z_now", chest_now))
    base_geom = {
        "shoulder_z_slice_m": round(sh_z_plan, 4),
        "delta_chest_z_m": round(chest_z_plan - chest_now, 4),
        "chest_z_target": round(chest_z_plan, 4),
        "source": (
            "low_z_q3_cap_q1_repair_after_lift"
            if target_trunk_q is not None
            else "q3_pitch_model_after_lift"
        ),
    }

    # 弦球放松（保持俯身不变）：先用 R_pitch 试解弦球；解不出来时，从 R_pitch 起
    # 每次 +1cm 放松弦球半径 R_chord，直到弦球有解，然后用该 xy/spin。
    # 关键：俯身后的肩高 sh_z_plan 固定，绝不因放松半径重算俯身（θz 不变）。
    # 仅在未显式指定 reach（reach is None）时自动放松；显式给定则尊重原值。
    def _chord_at(r: float) -> Dict[str, Any]:
        return plan_base_chord_pose(
            C,
            robot_xy,
            shoulder_z=sh_z_plan,
            is_free_xy=is_free_xy,
            nav_clearance_fn=nav_clearance_fn,
            reach_R=float(r),
        )

    plan = _chord_at(R_chord)
    chord_relax = {
        "applied": False,
        "reach_pitch_m": round(R_pitch, 4),
        "reach_chord_start_m": round(R_pitch, 4),
        "reach_chord_final_m": round(R_chord, 4),
        "steps": 0,
        "step_m": _CHORD_RELAX_STEP_M,
        "max_r_m": _CHORD_RELAX_MAX_R_M,
        "dz_m": round(float(sh_z_plan - C[2]), 4),
    }
    if (reach is None) and (not plan.get("ok")):
        r_try = R_chord
        steps = 0
        while (not plan.get("ok")) and (r_try < _CHORD_RELAX_MAX_R_M - 1e-9):
            r_try = round(r_try + _CHORD_RELAX_STEP_M, 4)
            steps += 1
            plan = _chord_at(r_try)
        if plan.get("ok"):
            R_chord = r_try
            chord_relax.update({
                "applied": True,
                "reach_chord_final_m": round(R_chord, 4),
                "steps": steps,
            })

    if not plan.get("ok"):
        chord_relax["exhausted"] = bool(reach is None)
        trunk_meta = {
            "phase": (
                "shoulder_forward_h3_low_z_q3_cap_q1_repair"
                if target_trunk_q is not None
                else "shoulder_forward_h3_q3_only"
            ),
            "reach_pitch_m": R_pitch,
            "reach_chord_m": R_chord,
            "theta_z_geom_deg": round(theta_z, 2),
            "shoulder_span_m": round(float(pose["shoulder_span_m"]), 4),
            "chest_z_now": round(chest_now, 4),
            "chest_z_after_q3_model": round(chest_z_plan, 4),
            "theta_z_now_deg": pitch.get("theta_z_now_deg"),
            "pitch_geom": pitch,
            "target_trunk_q": target_trunk_q,
            "base_geom": base_geom,
            "reach_chord_relax": chord_relax,
        }
        return {
            "ok": False,
            "error": plan.get("error", "弦球规划失败"),
            "chest_z": chest_z_plan,
            "theta_z_deg": theta_z,
            "shoulder_z": sh_z_plan,
            "delta_chest_z_m": float(chest_z_plan - chest_now),
            "trunk_meta": trunk_meta,
            "reach_pitch_m": R_pitch,
            "reach_chord_m": R_chord,
            "theta_z_geom_deg": round(theta_z, 2),
            "pitch_geom": pitch,
            "target_trunk_q": target_trunk_q,
            "base_geom": base_geom,
            "reach_chord_relax": chord_relax,
            "failure_dz_m": float(sh_z_plan - C[2]),
            "failure_used_pose_after_pitch": True,
        }

    bx, by = plan["base_xy"]
    yaw = float(plan["theta_x_deg"])
    detail = plan.get("plan_detail") or {}
    chord_dist = detail.get("chord_dist") or {}
    chord_ok = bool(detail.get("chord_ok"))

    trunk_meta = {
        "phase": (
            "shoulder_forward_h3_low_z_q3_cap_q1_repair"
            if target_trunk_q is not None
            else "shoulder_forward_h3_q3_only"
        ),
        "reach_pitch_m": R_pitch,
        "reach_chord_m": R_chord,
        "theta_z_geom_deg": round(theta_z, 2),
        "shoulder_span_m": round(float(pose["shoulder_span_m"]), 4),
        "chest_z_now": round(chest_now, 4),
        "chest_z_after_q3_model": round(chest_z_plan, 4),
        "theta_z_now_deg": pitch.get("theta_z_now_deg"),
        "pitch_geom": pitch,
        "target_trunk_q": target_trunk_q,
        "base_geom": {
            **base_geom,
        },
        "reach_chord_relax": chord_relax,
    }

    return {
        "ok": True,
        "chest_z": chest_z_plan,
        "theta_z_deg": theta_z,
        "shoulder_z": sh_z_plan,
        "delta_chest_z_m": float(chest_z_plan - chest_now),
        "base_plan": plan,
        "base_xy": [float(bx), float(by)],
        "theta_x_deg": yaw,
        "trunk_meta": trunk_meta,
        "reach_pitch_m": R_pitch,
        "reach_chord_m": R_chord,
        "theta_z_geom_deg": round(theta_z, 2),
        "chord_ok_preview": chord_ok,
        "chord_dist_preview": chord_dist,
        "reach_chord_relax": chord_relax,
    }
