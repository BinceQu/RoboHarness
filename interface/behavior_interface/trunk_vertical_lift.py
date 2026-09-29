"""R1Pro 腰部垂直升降 —— 正弦定理三角连杆模型。

几何（用户定义，与 move_to 内旧 IK 无关）：
  腰链三角形顶点：q1 轴、q2 轴、q3 轴。
  A = q3 轴 ↔ q2 轴 连杆长度（R1Pro torso_link3 段 ≈ 0.300 m）
  B = q2 轴 ↔ q1 轴 连杆长度（R1Pro torso_link2 段 ≈ 0.400 m）
  C = q3 轴 ↔ q1 轴 距离；垂直升降时取 C 为竖直分量（两轴水平对齐）。

正弦定理（角度用弧度，|q1|、|q3| 为关节角幅度）：
  A / sin|q1| = B / sin|q3| = C / sin(π - |q1| - |q3|)
  |q2| = |q1| + |q3|  （协同：弯腰时 q1>0, q3<0, q2 = -(|q1|+|q3|)）

目标胸口高度 z_tgt：
  C_tgt = C_curr + (z_tgt - z_curr)   # 高度差直接映射到 C
  对 C 插值 → 每点反解 (q1,q2,q3) → 下发 trunk 绝对角。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# R1Pro 腰段几何（USD/URDF：t2 段 0.4 m，t3 段 0.3 m）
R1PRO_WAIST_A_M = 0.300  # q3 轴 → q2 轴
R1PRO_WAIST_B_M = 0.400  # q2 轴 → q1 轴
# q1 枢轴在 base 系高度（torso_joint1 原点 z）
R1PRO_Q1_BASE_Z_M = 0.343
# torso_link4 相对 q3 累积末端的近似抬升（执行器链顶）
R1PRO_T4_LIFT_M = 0.100

# 与仿真一致的关节限位（eval_utils / tuck）
R1PRO_Q_LIMITS = (
    (-1.1345, 1.8326),   # q1
    (-2.7925, 2.5307),   # q2
    (-1.8326, 1.5708),   # q3
    (-3.0543, 3.0543),   # q4
)

CHEST_Z_MIN_M = 0.66
CHEST_Z_MAX_M = 1.15

# move_to_point 协同预降：肩高−目标 z 超过此值才触发，并尽量收到该值以内
POINT_PRE_DESCENT_DZ_MAX_M = 0.40

# 正弦定理流形 C 扫描缓存（关节限位内 A/sin|q1|=B/sin|q3|）
_SINE_LAW_MANIFOLD_CACHE: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class WaistTriangleGeom:
    """腰链三角几何常数。"""
    A: float = R1PRO_WAIST_A_M
    B: float = R1PRO_WAIST_B_M
    q1_base_z: float = R1PRO_Q1_BASE_Z_M
    t4_lift: float = R1PRO_T4_LIFT_M


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def vertical_C_from_joint_magnitudes(
    abs_q1: float,
    abs_q3: float,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> float:
    """正弦定理流形上由 |q1|、|q3| 算 C（仅用于流形反解/扫描，勿用于任意 trunk_q）。"""
    a = abs(float(abs_q1))
    b = abs(float(abs_q3))
    if a < 1e-5 and b < 1e-5:
        return geom.A + geom.B
    if a < 1e-5:
        a = 1e-5
    gamma = math.pi - a - b
    if gamma <= 1e-5:
        return geom.A + geom.B
    return geom.A * math.sin(gamma) / math.sin(a)


def fk_vertical_C_m(
    q1: float,
    q2: float,
    q3: float,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> float:
    """q1 轴与 q3 轴在 base 系竖直间距（与姿态无关的 FK 定义，直立时 ≈ A+B）。"""
    z1, z3, _ = _fk_chest_z_base(q1, q2, q3, geom)
    return float(z3 - z1)


def vertical_C_from_trunk_q(
    trunk_q: np.ndarray,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> float:
    """当前 trunk 姿态的竖直等效 C（FK：q3 轴 z − q1 轴 z）。"""
    q = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    return fk_vertical_C_m(float(q[0]), float(q[1]), float(q[2]), geom)


def _fk_chest_z_base(
    q1: float,
    q2: float,
    q3: float,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> Tuple[float, float, float]:
    """2D 矢状面 FK：返回 (q1轴z, q3轴z, 胸口近似z) base 系。"""
    th1 = q1
    th12 = q1 + q2
    z1 = geom.q1_base_z
    z2 = z1 + geom.B * math.cos(th1)
    z3 = z2 + geom.A * math.cos(th12)
    z_chest = z3 + geom.t4_lift * math.cos(th12 - q3)
    x1 = 0.0
    x2 = geom.B * math.sin(th1)
    x3 = x2 + geom.A * math.sin(th12)
    return z1, z3, z_chest


def scan_sine_law_manifold(
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    q_limits=R1PRO_Q_LIMITS,
    *,
    refresh: bool = False,
) -> Dict[str, Any]:
    """扫描正弦定理流形（A/sin|q1|=B/sin|q3| + 关节限位），返回 C / 胸口 z 可达范围。"""
    global _SINE_LAW_MANIFOLD_CACHE
    if _SINE_LAW_MANIFOLD_CACHE is not None and not refresh:
        return dict(_SINE_LAW_MANIFOLD_CACHE)

    A, B = geom.A, geom.B
    lo1, hi1 = q_limits[0]
    lo2, hi2 = q_limits[1]
    lo3, hi3 = q_limits[2]
    samples: List[Dict[str, float]] = []

    for a in np.linspace(0.001, 1.55, 4000):
        sb = B * math.sin(a) / A
        if abs(sb) > 1.0:
            continue
        b = math.asin(sb)
        gamma = math.pi - a - b
        if gamma <= 1e-8:
            continue
        C = A * math.sin(gamma) / math.sin(a)
        q1, q3 = float(a), float(-b)
        q2 = float(-(a + b))
        if not (lo1 <= q1 <= hi1 and lo2 <= q2 <= hi2 and lo3 <= q3 <= hi3):
            continue
        _, z3, z_chest = _fk_chest_z_base(q1, q2, q3, geom)
        z1 = geom.q1_base_z
        C = float(z3 - z1)
        samples.append({
            "C": C, "q1": q1, "q2": q2, "q3": q3, "z_chest_base": float(z_chest),
        })

    if not samples:
        out = {"ok": False, "error": "正弦定理流形扫描为空"}
        _SINE_LAW_MANIFOLD_CACHE = out
        return dict(out)

    Cs = [s["C"] for s in samples]
    zs = [s["z_chest_base"] for s in samples]
    imin = int(np.argmin(Cs))
    imax = int(np.argmax(Cs))
    out = {
        "ok": True,
        "C_min_m": float(min(Cs)),
        "C_max_m": float(max(Cs)),
        "z_chest_min_m": float(min(zs)),
        "z_chest_max_m": float(max(zs)),
        "q_at_C_min": [samples[imin]["q1"], samples[imin]["q2"], samples[imin]["q3"]],
        "q_at_C_max": [samples[imax]["q1"], samples[imax]["q2"], samples[imax]["q3"]],
        "n_samples": len(samples),
        "geom_A_m": geom.A,
        "geom_B_m": geom.B,
    }
    _SINE_LAW_MANIFOLD_CACHE = out
    return dict(out)


def estimate_chest_theta_z_deg(q1: float, q2: float, q3: float) -> float:
    """矢状面近似：胸口 forward 与世界 +Z 夹角（90°=水平）。

    注意：深弯腰时与 r1pro.urdf torso_link4 FK 偏差大；规划/执行请用 fk_torso_link4_theta_z_deg。
    """
    pitch = float(q1) + float(q2) - float(q3)
    fz = math.sin(pitch)
    return math.degrees(math.acos(max(-1.0, min(1.0, fz))))


# r1pro.urdf 腰链（与 world.chest_pose / torso_link4 局部 +X 一致）
_J1_ORIGIN = np.array([-0.079032, 0.0, 0.34265], dtype=np.float64)
_J2_ORIGIN = np.array([0.0, 0.0, 0.4], dtype=np.float64)
_J3_ORIGIN = np.array([0.0, 0.0001005, 0.3], dtype=np.float64)
_J4_ORIGIN = np.array([0.0, -0.00010015, 0.09962], dtype=np.float64)
_Q1_PITCH_SCAN_N = 128


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _chain44(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def fk_torso_link4_forward(
    q1: float,
    q2: float,
    q3: float,
    q4: float = 0.0,
) -> np.ndarray:
    """torso_link4 +X 方向（基座系，无底盘 yaw）。"""
    T = np.eye(4, dtype=np.float64)
    T = T @ _chain44(np.eye(3), _J1_ORIGIN) @ _chain44(_rot_y(float(q1)), np.zeros(3))
    T = T @ _chain44(np.eye(3), _J2_ORIGIN) @ _chain44(_rot_y(float(q2)), np.zeros(3))
    T = T @ _chain44(np.eye(3), _J3_ORIGIN) @ _chain44(_rot_y(-float(q3)), np.zeros(3))
    T = T @ _chain44(np.eye(3), _J4_ORIGIN) @ _chain44(_rot_z(float(q4)), np.zeros(3))
    fwd = T[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    n = float(np.linalg.norm(fwd))
    return fwd / max(n, 1e-9)


def fk_torso_link4_theta_z_deg(
    q1: float,
    q2: float,
    q3: float,
    q4: float = 0.0,
) -> float:
    """与 world.chest_pose().theta_z_deg 同定义（forward·+Z 极角）。"""
    fz = float(fk_torso_link4_forward(q1, q2, q3, q4)[2])
    return math.degrees(math.acos(max(-1.0, min(1.0, fz))))


def theta_z_limits_pitch_q1_only(
    q2: float,
    q3: float,
    q_limits=R1PRO_Q_LIMITS,
    *,
    n_scan: int = _Q1_PITCH_SCAN_N,
) -> Tuple[float, float]:
    """锁定 q2/q3、仅扫 q1 时，URDF q1 限位内可达 θz（r1pro.urdf FK）。"""
    lo1, hi1 = q_limits[0]
    q2f, q3f = float(q2), float(q3)
    tzs = [
        fk_torso_link4_theta_z_deg(float(q1), q2f, q3f)
        for q1 in np.linspace(lo1, hi1, max(32, int(n_scan)))
    ]
    return float(min(tzs)), float(max(tzs))


def solve_q1_for_theta_z_pitch_q1_only(
    theta_z_deg: float,
    q2: float,
    q3: float,
    *,
    prefer_q1: Optional[float] = None,
    q_limits=R1PRO_Q_LIMITS,
    tol_deg: float = 0.2,
    n_scan: int = _Q1_PITCH_SCAN_N,
) -> Optional[float]:
    """给定 θz 与锁定 q2/q3，在 URDF q1 限位内反解 q1（与 pitch_q1_only 执行一致）。

    与 solve_q3_for_theta_z_holding_q12 同理：只接受 forward fx≥0 的「未过折」解，
    避免 acos 折返把真实俯角 >180° 的镜像枝当成目标解。
    """
    lo1, hi1 = q_limits[0]
    target = float(theta_z_deg)
    q2f, q3f = float(q2), float(q3)
    best_q1: Optional[float] = None
    best_score = 1e18
    for q1 in np.linspace(lo1, hi1, max(32, int(n_scan))):
        q1f = float(q1)
        fwd = fk_torso_link4_forward(q1f, q2f, q3f)
        if float(fwd[0]) < 0.0:
            continue
        fz = max(-1.0, min(1.0, float(fwd[2])))
        tz = math.degrees(math.acos(fz))
        err = abs(tz - target)
        cont = 0.003 * abs(q1f - float(prefer_q1)) if prefer_q1 is not None else 0.0
        score = err + cont
        if score < best_score:
            best_score = score
            best_q1 = q1f
    if best_q1 is None or best_score > max(float(tol_deg), 2.5):
        return None
    return best_q1


def solve_q3_for_theta_z_holding_q12(
    theta_z_deg: float,
    q1: float,
    q2: float,
    *,
    prefer_q3: Optional[float] = None,
    q_limits=R1PRO_Q_LIMITS,
    tol_deg: float = 0.35,
    n_scan: int = _Q1_PITCH_SCAN_N,
) -> Optional[float]:
    """锁定 q1/q2，在 URDF q3 限位内反解 q3，使 torso_link4 θz 接近目标（胸廓垂直）。

    θz = acos(forward·Z) 值域只有 0–180°：胸口一旦转过竖直向下（forward 前向分量
    fx<0），真实俯角 196° 会被折返成 164°，与正确枝读数相同。这里只接受 fx≥0 的
    「未过折」解，否则规划会落在俯冲过头的镜像枝上（执行/校验用同一 acos 读数，
    全程发现不了）。
    """
    lo3, hi3 = q_limits[2]
    target = float(theta_z_deg)
    q1f, q2f = float(q1), float(q2)
    best_q3: Optional[float] = None
    best_score = 1e18
    for q3 in np.linspace(lo3, hi3, max(32, int(n_scan))):
        q3f = float(q3)
        fwd = fk_torso_link4_forward(q1f, q2f, q3f)
        if float(fwd[0]) < 0.0:
            # 过折枝：胸口指向后下方（真实俯角 >180°），acos 读数是镜像假象
            continue
        fz = max(-1.0, min(1.0, float(fwd[2])))
        tz = math.degrees(math.acos(fz))
        err = abs(tz - target)
        cont = 0.003 * abs(q3f - float(prefer_q3)) if prefer_q3 is not None else 0.0
        score = err + cont
        if score < best_score:
            best_score = score
            best_q3 = q3f
    if best_q3 is None or best_score > max(float(tol_deg), 2.5):
        return None
    return best_q3


def trunk_joint_limits_urdf() -> Dict[str, Any]:
    """R1Pro 腰链 URDF 限位（eval_utils.JOINT_RANGE / r1pro.urdf torso_joint1–4）。"""
    names = ("q1", "q2", "q3", "q4")
    out: Dict[str, Any] = {"source": "eval_utils.JOINT_RANGE['R1Pro']['torso']"}
    for i, nm in enumerate(names):
        lo, hi = R1PRO_Q_LIMITS[i]
        out[nm] = {
            "lo_rad": round(lo, 4),
            "hi_rad": round(hi, 4),
            "lo_deg": round(math.degrees(lo), 2),
            "hi_deg": round(math.degrees(hi), 2),
        }
    out["q1_backward"] = out["q1"]["lo_rad"]
    out["q1_backward_deg"] = out["q1"]["lo_deg"]
    out["q2_backward"] = out["q2"]["lo_rad"]
    out["q2_backward_deg"] = out["q2"]["lo_deg"]
    return out


def solve_q1_q3_for_theta_z_holding_q2(
    theta_z_deg: float,
    q2: float,
    *,
    prefer_q1: Optional[float] = None,
    prefer_q3: Optional[float] = None,
    bias_q3_fold: bool = False,
    q3_fold_weight: float = 0.08,
    q_limits=R1PRO_Q_LIMITS,
    tol_deg: float = 0.4,
    n_scan: int = _Q1_PITCH_SCAN_N,
) -> Optional[Tuple[float, float]]:
    """锁定 q2，扫 q1 并反解 q3，使 torso_link4 θz≈目标（胸廓垂直）。

    bias_q3_fold=True 时在多解中优先更负的 q3（折叠更深）。
    """
    lo1, hi1 = q_limits[0]
    target = float(theta_z_deg)
    q2f = float(q2)
    best: Optional[Tuple[float, float]] = None
    best_score = 1e18
    for q1 in np.linspace(lo1, hi1, max(48, int(n_scan))):
        q1f = float(q1)
        q3 = solve_q3_for_theta_z_holding_q12(
            target, q1f, q2f,
            prefer_q3=prefer_q3,
            q_limits=q_limits,
            tol_deg=tol_deg,
        )
        if q3 is None:
            continue
        tz = fk_torso_link4_theta_z_deg(q1f, q2f, q3)
        err = abs(tz - target)
        if err > max(float(tol_deg), 1.0):
            continue
        cont = 0.0
        if bias_q3_fold:
            if prefer_q1 is not None:
                cont += 0.015 * abs(q1f - float(prefer_q1))
            # q3 越负得分越低 → 优先折叠 q3
            cont += float(q3_fold_weight) * float(q3)
        else:
            if prefer_q1 is not None:
                cont += 0.02 * abs(q1f - float(prefer_q1))
            if prefer_q3 is not None:
                cont += 0.02 * abs(q3 - float(prefer_q3))
        score = err + cont
        if score < best_score:
            best_score = score
            best = (q1f, float(q3))
    return best


def solve_q2_q3_for_theta_z_holding_q1(
    theta_z_deg: float,
    q1: float,
    *,
    prefer_q2: Optional[float] = None,
    prefer_q3: Optional[float] = None,
    q_limits=R1PRO_Q_LIMITS,
    tol_deg: float = 0.4,
    n_scan: int = _Q1_PITCH_SCAN_N,
) -> Optional[Tuple[float, float]]:
    """锁定 q1（如后摆限位），扫 q2 并反解 q3，使 θz≈目标。"""
    lo2, hi2 = q_limits[1]
    target = float(theta_z_deg)
    q1f = float(q1)
    best: Optional[Tuple[float, float]] = None
    best_score = 1e18
    for q2 in np.linspace(lo2, hi2, max(48, int(n_scan))):
        q2f = float(q2)
        q3 = solve_q3_for_theta_z_holding_q12(
            target, q1f, q2f,
            prefer_q3=prefer_q3,
            q_limits=q_limits,
            tol_deg=tol_deg,
        )
        if q3 is None:
            continue
        tz = fk_torso_link4_theta_z_deg(q1f, q2f, q3)
        err = abs(tz - target)
        if err > max(float(tol_deg), 1.0):
            continue
        cont = 0.0
        if prefer_q2 is not None:
            cont += 0.02 * abs(q2f - float(prefer_q2))
        if prefer_q3 is not None:
            cont += 0.02 * abs(q3 - float(prefer_q3))
        score = err + cont
        if score < best_score:
            best_score = score
            best = (q2f, float(q3))
    return best


def plan_phase2_q2_backward_waypoints(
    trunk_q: np.ndarray,
    z_target_world: float,
    base_link_z: float,
    *,
    theta_z_hold_deg: float = 90.0,
    dq2_step_rad: float = -0.015,
    bias_q3_fold: bool = False,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    q_limits=R1PRO_Q_LIMITS,
    max_steps: int = 400,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """阶段2：q2 向后（减小）为主；每步反解 q1/q3 保持胸廓 θz≈90°。

    q2 向后 = 沿 torso_joint2 (+Y) 负向，趋向 URDF 下限 -2.7925 rad。
    """
    q = np.asarray(trunk_q, dtype=np.float64).reshape(4).copy()
    q4_lock = float(q[3])
    lo2, hi2 = q_limits[1]
    z_start = estimate_chest_z_world(q, base_link_z, geom)
    z_tgt = float(z_target_world)
    descending = z_tgt < z_start - 1e-4
    if not descending:
        return [q], {"ok": True, "skipped": True, "phase": "q2_backward_theta_guard"}

    dq2 = -abs(float(dq2_step_rad))
    limits_doc = trunk_joint_limits_urdf()
    meta: Dict[str, Any] = {
        "ok": True,
        "phase": "q2_backward_theta_guard",
        "z_start": round(z_start, 4),
        "z_target": round(z_tgt, 4),
        "theta_z_hold_deg": round(float(theta_z_hold_deg), 2),
        "dq2_rad": round(dq2, 5),
        "bias_q3_fold": bool(bias_q3_fold),
        "urdf_limits": limits_doc,
    }

    waypoints: List[np.ndarray] = [q.copy()]
    tz_trace: List[float] = []
    for step_i in range(max_steps):
        z_now = estimate_chest_z_world(q, base_link_z, geom)
        if z_now <= z_tgt + 1e-3:
            break
        nq2 = float(q[1]) + dq2
        if nq2 < lo2 + 1e-4:
            meta["limited_by"] = "q2_backward_limit"
            meta["q2_limit_rad"] = round(lo2, 4)
            break
        sol = solve_q1_q3_for_theta_z_holding_q2(
            float(theta_z_hold_deg), nq2,
            prefer_q1=float(q[0]),
            prefer_q3=float(q[2]),
            bias_q3_fold=bias_q3_fold,
            q_limits=q_limits,
        )
        if sol is None:
            meta["limited_by"] = "no_q1_q3_for_vertical"
            meta["failed_q2"] = round(nq2, 4)
            break
        nq1, nq3 = sol
        q = np.array([nq1, nq2, nq3, q4_lock], dtype=np.float64)
        waypoints.append(q.copy())
        tz_trace.append(theta_z_from_trunk_q(nq1, nq2, nq3, q4_lock))

    z_end = estimate_chest_z_world(q, base_link_z, geom)
    meta["z_end"] = round(z_end, 4)
    meta["n_waypoints"] = len(waypoints)
    meta["q_end"] = [round(float(x), 4) for x in q[:3]]
    if tz_trace:
        meta["theta_z_spread_deg"] = round(max(tz_trace) - min(tz_trace), 3)
        meta["theta_z_end_deg"] = round(tz_trace[-1], 2)
    if z_end > z_tgt + 0.012:
        meta["partial"] = True
    return waypoints, meta


def plan_phase2_q1q2_locked_q3_waypoints(
    trunk_q: np.ndarray,
    base_link_z: float,
    *,
    z_target_world: Optional[float] = None,
    dq_step_rad: float = 0.015,
    q1_limit_rad: Optional[float] = None,
    reverse_joint_dirs: bool = False,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    q_limits=R1PRO_Q_LIMITS,
    max_steps: int = 400,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """阶段2：|dq2|=|dq1|，dq3=0，直到 q1 触限（或 z 目标）。

    正向折姿下降：q2 向后(dq2<0)、q1 同幅增大(dq1=-dq2)，q3 锁定。
    reverse_joint_dirs=True：三关节步进方向全反（q2 回退/增大，q1 减小朝后摆限位）。
    """
    q = np.asarray(trunk_q, dtype=np.float64).reshape(4).copy()
    q4_lock = float(q[3])
    lo1, hi1 = q_limits[0]
    lo2, hi2 = q_limits[1]
    z_start = estimate_chest_z_world(q, base_link_z, geom)
    z_tgt = float(z_target_world) if z_target_world is not None else None
    d = abs(float(dq_step_rad))

    # 折姿：正向 q1→上限、q2 向后；反向 q1→后摆下限、q2 回退(增大)
    if reverse_joint_dirs:
        # 反向：q1→后摆下限，q2 回退(增大)，与 fold_sign=-1 阶段1末衔接
        q1_stop = float(q1_limit_rad) if q1_limit_rad is not None else float(lo1)
        dq1, dq2 = -d, d
    else:
        if q1_limit_rad is not None:
            q1_stop = float(q1_limit_rad)
        elif float(q[0]) >= 0.0:
            q1_stop = float(hi1)
        else:
            q1_stop = float(lo1)

        if q1_stop >= float(q[0]):
            dq1, dq2 = d, -d
        else:
            dq1, dq2 = -d, d

    meta: Dict[str, Any] = {
        "ok": True,
        "phase": "q1q2_locked_q3",
        "reverse_joint_dirs": bool(reverse_joint_dirs),
        "z_start": round(z_start, 4),
        "z_target": round(z_tgt, 4) if z_tgt is not None else None,
        "dq1_rad": round(dq1, 5),
        "dq2_rad": round(dq2, 5),
        "q1_stop_rad": round(q1_stop, 4),
        "coupling": "|dq2|=|dq1|, dq3=0",
    }

    waypoints: List[np.ndarray] = [q.copy()]
    for _ in range(max_steps):
        z_now = estimate_chest_z_world(q, base_link_z, geom)
        if z_tgt is not None and z_now <= z_tgt + 1e-3:
            break
        if abs(float(q[0]) - q1_stop) < 1e-4:
            meta["limited_by"] = "q1_at_limit"
            break

        nq1 = float(q[0]) + dq1
        nq2 = float(q[1]) + dq2
        nq3 = float(q[2])

        hit_q1 = (dq1 > 0 and nq1 >= q1_stop - 1e-4) or (
            dq1 < 0 and nq1 <= q1_stop + 1e-4
        )
        if hit_q1:
            nq1 = _clip(q1_stop, lo1, hi1)
            meta["limited_by"] = "q1_limit"
        if not (lo2 <= nq2 <= hi2):
            meta["limited_by"] = "q2_limit"
            break

        q = np.array([nq1, nq2, nq3, q4_lock], dtype=np.float64)
        waypoints.append(q.copy())
        if hit_q1:
            break

    z_end = estimate_chest_z_world(q, base_link_z, geom)
    meta["z_end"] = round(z_end, 4)
    meta["n_waypoints"] = len(waypoints)
    meta["q_end"] = [round(float(x), 4) for x in q[:3]]
    return waypoints, meta


def plan_phase3_q2q3_locked_q1_waypoints(
    trunk_q: np.ndarray,
    base_link_z: float,
    *,
    z_target_world: Optional[float] = None,
    dq_step_rad: float = 0.015,
    q3_limit_rad: Optional[float] = None,
    reverse_joint_dirs: bool = False,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    q_limits=R1PRO_Q_LIMITS,
    max_steps: int = 400,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """阶段3：|dq2|=|dq3|，dq1=0，直到 q3 触限（或 z 目标）。

    正向折姿下降：q3 更负、q2 同幅向后(dq2=dq3<0)，q1 锁定。
    reverse_joint_dirs=True：dq2/dq3 全反（q2 回退、q3 朝上限），q1 锁定。
    """
    q = np.asarray(trunk_q, dtype=np.float64).reshape(4).copy()
    q4_lock = float(q[3])
    lo2, hi2 = q_limits[1]
    lo3, hi3 = q_limits[2]
    z_start = estimate_chest_z_world(q, base_link_z, geom)
    z_tgt = float(z_target_world) if z_target_world is not None else None
    d = abs(float(dq_step_rad))

    if reverse_joint_dirs:
        # 反向：q3→上限，q2 同幅回退增大
        q3_stop = float(q3_limit_rad) if q3_limit_rad is not None else float(hi3)
        dq3, dq2 = d, d
    else:
        if q3_limit_rad is not None:
            q3_stop = float(q3_limit_rad)
        elif float(q[2]) <= 0.0:
            q3_stop = float(lo3)
        else:
            q3_stop = float(hi3)

        if q3_stop <= float(q[2]):
            dq3, dq2 = -d, -d
        else:
            dq3, dq2 = d, d

    meta: Dict[str, Any] = {
        "ok": True,
        "phase": "q2q3_locked_q1",
        "reverse_joint_dirs": bool(reverse_joint_dirs),
        "z_start": round(z_start, 4),
        "z_target": round(z_tgt, 4) if z_tgt is not None else None,
        "dq2_rad": round(dq2, 5),
        "dq3_rad": round(dq3, 5),
        "q3_stop_rad": round(q3_stop, 4),
        "q1_locked_rad": round(float(q[0]), 4),
        "coupling": "|dq2|=|dq3|, dq1=0",
    }

    waypoints: List[np.ndarray] = [q.copy()]
    for _ in range(max_steps):
        z_now = estimate_chest_z_world(q, base_link_z, geom)
        if z_tgt is not None and z_now <= z_tgt + 1e-3:
            break
        if abs(float(q[2]) - q3_stop) < 1e-4:
            meta["limited_by"] = "q3_at_limit"
            break

        nq1 = float(q[0])
        nq3 = float(q[2]) + dq3
        nq2 = float(q[1]) + dq2

        hit_q3 = (dq3 < 0 and nq3 <= q3_stop + 1e-4) or (
            dq3 > 0 and nq3 >= q3_stop - 1e-4
        )
        if hit_q3:
            nq3 = _clip(q3_stop, lo3, hi3)
            meta["limited_by"] = "q3_limit"
        if not (lo2 <= nq2 <= hi2):
            meta["limited_by"] = "q2_limit"
            break

        q = np.array([nq1, nq2, nq3, q4_lock], dtype=np.float64)
        waypoints.append(q.copy())
        if hit_q3:
            break

    z_end = estimate_chest_z_world(q, base_link_z, geom)
    meta["z_end"] = round(z_end, 4)
    meta["n_waypoints"] = len(waypoints)
    meta["q_end"] = [round(float(x), 4) for x in q[:3]]
    return waypoints, meta


def measure_q2_backward_vertical_range(
    trunk_q: Optional[np.ndarray] = None,
    *,
    base_link_z: float = 0.05,
    theta_z_hold_deg: float = 90.0,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> Dict[str, Any]:
    """FK 测量：阶段1 末 → q2 向后扫至限位；以及 q1=后摆限位时 (q2,q3) 垂直位。"""
    limits = trunk_joint_limits_urdf()
    q1_back = float(limits["q1_backward"])
    lo2 = float(limits["q2"]["lo_rad"])

    if trunk_q is None:
        trunk_q = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q0 = np.asarray(trunk_q, dtype=np.float64).reshape(4).copy()
    z_upright = estimate_chest_z_world(q0, base_link_z, geom)

    mf = scan_sine_law_manifold(geom)
    z_p1_min = float(base_link_z) + float(mf.get("z_chest_min_m", 0.72))
    wp_full, meta_full = plan_vertical_lift_waypoints_v2(
        q0, z_upright, CHEST_Z_MIN_M, base_link_z=base_link_z, n_steps=20,
    )
    p1_n = 0
    for ph in meta_full.get("phases", []):
        if ph.get("phase") == "sine_manifold":
            p1_n = int(ph.get("n_waypoints", 0))
    q_after_p1 = (
        wp_full[p1_n - 1] if p1_n > 0 and len(wp_full) >= p1_n else q0
    )
    z_after_p1 = estimate_chest_z_world(q_after_p1, base_link_z, geom)

    wp2, p2meta = plan_phase2_q2_backward_waypoints(
        q_after_p1, CHEST_Z_MIN_M, base_link_z,
        theta_z_hold_deg=theta_z_hold_deg,
    )
    z_p2_min = estimate_chest_z_world(wp2[-1], base_link_z, geom) if wp2 else z_after_p1

    q1_back_pose = solve_q2_q3_for_theta_z_holding_q1(
        theta_z_hold_deg, q1_back,
        prefer_q2=float(q_after_p1[1]),
        prefer_q3=float(q_after_p1[2]),
    )
    z_at_q1_back = None
    q_pose_q1_back = None
    if q1_back_pose is not None:
        q2b, q3b = q1_back_pose
        q_pose_q1_back = [round(q1_back, 4), round(q2b, 4), round(q3b, 4)]
        z_at_q1_back = round(
            estimate_chest_z_world(
                np.array([q1_back, q2b, q3b, q0[3]], dtype=np.float64),
                base_link_z, geom,
            ),
            4,
        )

    return {
        "ok": True,
        "urdf_limits": limits,
        "theta_z_hold_deg": theta_z_hold_deg,
        "upright": {
            "chest_z_world_m": round(z_upright, 4),
            "trunk_q": [round(float(x), 4) for x in q0[:3]],
        },
        "after_phase1_sine": {
            "chest_z_world_m": round(z_after_p1, 4),
            "trunk_q": [round(float(x), 4) for x in q_after_p1[:3]],
            "drop_from_upright_m": round(z_upright - z_after_p1, 4),
        },
        "phase2_q2_backward_to_limit": {
            "chest_z_min_world_m": round(z_p2_min, 4),
            "extra_drop_from_phase1_m": round(z_after_p1 - z_p2_min, 4),
            "total_drop_from_upright_m": round(z_upright - z_p2_min, 4),
            "q_end": p2meta.get("q_end"),
            "limited_by": p2meta.get("limited_by"),
            "n_waypoints": p2meta.get("n_waypoints"),
        },
        "at_q1_backward_limit_vertical": {
            "q1_rad": round(q1_back, 4),
            "q1_deg": round(math.degrees(q1_back), 2),
            "q2_q3_if_vertical": q_pose_q1_back,
            "chest_z_world_m": z_at_q1_back,
            "drop_from_upright_m": (
                round(z_upright - z_at_q1_back, 4) if z_at_q1_back is not None else None
            ),
        },
    }


def theta_z_from_trunk_q(q1: float, q2: float, q3: float, q4: float = 0.0) -> float:
    """URDF FK θz（与 world.chest_pose().theta_z_deg 同定义）。"""
    return fk_torso_link4_theta_z_deg(float(q1), float(q2), float(q3), float(q4))


def q3_pitch_step_dir(
    q1: float,
    q2: float,
    q3: float,
    target_theta_z_deg: float,
    q_limits=R1PRO_Q_LIMITS,
    *,
    probe_rad: float = 0.02,
    done_tol_deg: float = 0.35,
    improve_tol_deg: float = 0.04,
) -> int:
    """俯仰阶段 q3 优先：+1 应增 q3，-1 应减 q3，0 已到位或 q3 无法继续改进（改 q1）。"""
    lo3, hi3 = q_limits[2]
    target = float(target_theta_z_deg)
    q1f, q2f, q3f = float(q1), float(q2), float(q3)
    tz0 = fk_torso_link4_theta_z_deg(q1f, q2f, q3f)
    if abs(tz0 - target) <= float(done_tol_deg):
        return 0
    eps = float(probe_rad)
    q3p = min(hi3, q3f + eps)
    q3m = max(lo3, q3f - eps)
    e0 = abs(tz0 - target)
    ep = abs(fk_torso_link4_theta_z_deg(q1f, q2f, q3p) - target)
    em = abs(fk_torso_link4_theta_z_deg(q1f, q2f, q3m) - target)
    imp = float(improve_tol_deg)
    can_inc = q3f < hi3 - 1e-4 and ep < e0 - imp
    can_dec = q3f > lo3 + 1e-4 and em < e0 - imp
    if can_inc and not can_dec:
        return 1
    if can_dec and not can_inc:
        return -1
    if can_inc and can_dec:
        return 1 if ep <= em else -1
    return 0


def clamp_theta_z_pitch_q1_only(
    theta_z_deg: float,
    q2: float,
    q3: float,
    *,
    prefer_q1: Optional[float] = None,
    q_limits=R1PRO_Q_LIMITS,
) -> Tuple[float, Dict[str, Any]]:
    """将规划 θz 裁到 pitch_q1_only 在 URDF 内实际可达，并给出对应 q1。"""
    tz_min, tz_max = theta_z_limits_pitch_q1_only(q2, q3, q_limits)
    requested = float(theta_z_deg)
    bounded = max(tz_min, min(tz_max, requested))
    q1_sol = solve_q1_for_theta_z_pitch_q1_only(
        bounded, q2, q3, prefer_q1=prefer_q1, q_limits=q_limits,
    )
    if q1_sol is None:
        bounded = tz_max if requested >= 0.5 * (tz_min + tz_max) else tz_min
        q1_sol = solve_q1_for_theta_z_pitch_q1_only(
            bounded, q2, q3, prefer_q1=prefer_q1, q_limits=q_limits,
        )
    achieved = (
        fk_torso_link4_theta_z_deg(q1_sol, q2, q3)
        if q1_sol is not None
        else bounded
    )
    meta: Dict[str, Any] = {
        "theta_z_requested_deg": round(requested, 2),
        "theta_z_achieved_deg": round(achieved, 2),
        "theta_z_achievable_lo_deg": round(tz_min, 2),
        "theta_z_achievable_hi_deg": round(tz_max, 2),
        "q1_required_rad": round(q1_sol, 4) if q1_sol is not None else None,
        "theta_z_saturated": abs(requested - achieved) > 0.5,
    }
    return achieved, meta


def theta_z_limits_pitch_q3_then_q1(
    q1: float,
    q2: float,
    q3: float,
    q_limits=R1PRO_Q_LIMITS,
    *,
    n_scan: int = _Q1_PITCH_SCAN_N,
) -> Tuple[float, float]:
    """q3 优先、q3 饱和后再扫 q1 时，URDF 内可达 θz 范围。"""
    lo3, hi3 = q_limits[2]
    lo1, hi1 = q_limits[0]
    q1f, q2f, q3f = float(q1), float(q2), float(q3)
    n = max(32, int(n_scan))
    tzs = [
        fk_torso_link4_theta_z_deg(q1f, q2f, float(q3v))
        for q3v in np.linspace(lo3, hi3, n)
    ]
    for q3_sat in (lo3, hi3):
        tzs.extend(
            fk_torso_link4_theta_z_deg(float(q1v), q2f, q3_sat)
            for q1v in np.linspace(lo1, hi1, n)
        )
    return float(min(tzs)), float(max(tzs))


def clamp_theta_z_pitch_q3_then_q1(
    theta_z_deg: float,
    q1: float,
    q2: float,
    q3: float,
    *,
    prefer_q1: Optional[float] = None,
    prefer_q3: Optional[float] = None,
    q_limits=R1PRO_Q_LIMITS,
    q3_tol_deg: float = 0.35,
) -> Tuple[float, Dict[str, Any]]:
    """将规划 θz 裁到 q3 优先、q3 饱和后再动 q1 的实际可达范围。

    阶段 1：锁定 q1/q2，在 URDF q3 限位内反解 q3（torso_joint3: [-1.8326, 1.5708]）。
    阶段 2：q3 已到限位仍不足时，锁定 q2/q3，按 pitch_q1_only 反解 q1。
    """
    requested = float(theta_z_deg)
    lo3, hi3 = q_limits[2]
    q1f, q2f, q3f = float(q1), float(q2), float(q3)
    q3_pref = float(prefer_q3) if prefer_q3 is not None else q3f

    tz_min, tz_max = theta_z_limits_pitch_q3_then_q1(q1f, q2f, q3f, q_limits)
    bounded = max(tz_min, min(tz_max, requested))

    q3_sol = solve_q3_for_theta_z_holding_q12(
        bounded, q1f, q2f, prefer_q3=q3_pref, q_limits=q_limits,
    )
    if q3_sol is not None:
        q3_eff = max(lo3, min(hi3, q3_sol))
    else:
        q3_eff = q3f

    tz_q3 = fk_torso_link4_theta_z_deg(q1f, q2f, q3_eff)
    if abs(tz_q3 - bounded) <= max(float(q3_tol_deg), 0.35):
        meta: Dict[str, Any] = {
            "theta_z_requested_deg": round(requested, 2),
            "theta_z_achieved_deg": round(tz_q3, 2),
            "theta_z_achievable_lo_deg": round(tz_min, 2),
            "theta_z_achievable_hi_deg": round(tz_max, 2),
            "pitch_phase": "q3_only",
            "q3_required_rad": round(q3_eff, 4),
            "q1_required_rad": round(q1f, 4),
            "theta_z_saturated": abs(requested - tz_q3) > 0.5,
        }
        return tz_q3, meta

    achieved, meta_q1 = clamp_theta_z_pitch_q1_only(
        bounded, q2f, q3_eff, prefer_q1=prefer_q1, q_limits=q_limits,
    )
    meta = dict(meta_q1)
    meta.update({
        "theta_z_requested_deg": round(requested, 2),
        "theta_z_achievable_lo_deg": round(tz_min, 2),
        "theta_z_achievable_hi_deg": round(tz_max, 2),
        "pitch_phase": "q3_saturated_q1",
        "q3_saturated_rad": round(q3_eff, 4),
        "theta_z_saturated": abs(requested - achieved) > 0.5,
    })
    return achieved, meta


def solve_joints_for_vertical_C(
    C_target: float,
    *,
    fold_sign: float = 1.0,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    q_limits=R1PRO_Q_LIMITS,
    prev_q: Optional[Tuple[float, float, float]] = None,
) -> Optional[Tuple[float, float, float]]:
    """给定竖直等效 C，正弦定理反解 (q1, q2, q3)。

    fold_sign>0：弯腰（q1>0, q3<0）；与 R1Pro 典型前折一致。
    """
    C = float(C_target)
    if C <= 1e-4:
        return None
    A, B = geom.A, geom.B

    candidates: List[Tuple[float, float, float, float]] = []
    for a in np.linspace(0.02, 1.55, 3000):
        sb = B * math.sin(a) / A
        if abs(sb) > 1.0:
            continue
        for b in (math.asin(sb), math.pi - math.asin(sb)):
            if b <= 1e-5 or b >= math.pi - 1e-5:
                continue
            gamma = math.pi - a - b
            if gamma <= 1e-5:
                continue
            sa = math.sin(a)
            if abs(sa) < 1e-8:
                continue
            C_pred = A * math.sin(gamma) / sa
            err = abs(C_pred - C)
            if err <= 0.008:
                candidates.append((err, float(a), float(b), float(C_pred)))

    if not candidates:
        return None

    if prev_q is not None:
        pq = np.asarray(prev_q[:3], dtype=np.float64)
        candidates.sort(
            key=lambda t: (
                t[0],
                float(np.linalg.norm(np.array([t[1], -(t[1] + t[2]), -t[2]]) - pq)),
            )
        )
    else:
        candidates.sort(key=lambda t: t[0])
    _, a, b, _ = candidates[0]
    sign = 1.0 if float(fold_sign) >= 0 else -1.0
    q1 = sign * a
    q3 = -sign * b
    q2 = -sign * (a + b)

    lo1, hi1 = q_limits[0]
    lo2, hi2 = q_limits[1]
    lo3, hi3 = q_limits[2]
    if not (lo1 - 1e-3 <= q1 <= hi1 + 1e-3):
        return None
    if not (lo2 - 1e-3 <= q2 <= hi2 + 1e-3):
        return None
    if not (lo3 - 1e-3 <= q3 <= hi3 + 1e-3):
        return None

    return (
        _clip(q1, lo1, hi1),
        _clip(q2, lo2, hi2),
        _clip(q3, lo3, hi3),
    )


def estimate_chest_z_world(
    trunk_q: np.ndarray,
    base_link_z: float,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> float:
    """由腰关节角估算胸口世界 z（base_link_z + 胸口 base 高度）。"""
    q = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    _, _, z_chest_base = _fk_chest_z_base(float(q[0]), float(q[1]), float(q[2]), geom)
    return float(base_link_z) + z_chest_base


def plan_vertical_lift_waypoints(
    trunk_q: np.ndarray,
    z_start_world: float,
    z_end_world: float,
    *,
    base_link_z: float = 0.05,
    n_steps: int = 20,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    fold_sign: Optional[float] = None,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """高度插值 → 每点反解 q1,q2,q3；q4 保持。

    返回 (waypoints, meta)；失败时 waypoints=[] 且 meta['ok']=False。
    """
    q0 = np.asarray(trunk_q, dtype=np.float64).reshape(4).copy()
    z0 = float(z_start_world)
    z1 = float(z_end_world)
    z1 = _clip(z1, CHEST_Z_MIN_M, CHEST_Z_MAX_M)

    C_curr = vertical_C_from_trunk_q(q0, geom)
    C_tgt = float(C_curr + (z1 - z0))

    mf = scan_sine_law_manifold(geom)
    if not mf.get("ok"):
        return [], {**mf, "ok": False}
    C_min = float(mf["C_min_m"])
    C_max = float(mf["C_max_m"])

    meta: Dict[str, Any] = {
        "ok": True,
        "z_start": round(z0, 4),
        "z_end": round(z1, 4),
        "C_start": round(C_curr, 4),
        "C_end": round(C_tgt, 4),
        "C_range_m": [round(C_min, 4), round(C_max, 4)],
        "z_chest_range_base_m": [round(mf["z_chest_min_m"], 4), round(mf["z_chest_max_m"], 4)],
        "geom_A_m": geom.A,
        "geom_B_m": geom.B,
        "model": "sine_triangle_vertical_C",
    }

    if C_tgt < C_min - 0.005 or C_tgt > C_max + 0.005:
        max_drop_C = max(0.0, C_curr - C_min)
        max_rise_C = max(0.0, C_max - C_curr)
        meta["ok"] = False
        meta["error"] = (
            f"目标 C={C_tgt:.3f}m 超出正弦定理流形可达 [{C_min:.3f}, {C_max:.3f}]m "
            f"(当前 C={C_curr:.3f}m，本姿态最多再降 ΔC≈{max_drop_C:.3f}m / 升 ΔC≈{max_rise_C:.3f}m；"
            f"请求 Δz={z1-z0:+.3f}m 映射为 ΔC={C_tgt-C_curr:+.3f}m)"
        )
        return [], meta

    if fold_sign is None:
        fold_sign = 1.0 if float(q0[0]) >= 0 else -1.0
        if abs(float(q0[0])) < 0.05 and z1 < z0:
            fold_sign = 1.0

    n = max(2, int(n_steps))
    waypoints: List[np.ndarray] = []
    Cs = np.linspace(C_curr, C_tgt, n)

    prev_sol: Optional[Tuple[float, float, float]] = (
        float(q0[0]), float(q0[1]), float(q0[2])
    )
    for i, C_i in enumerate(Cs):
        sol = solve_joints_for_vertical_C(
            C_i, fold_sign=fold_sign, geom=geom, prev_q=prev_sol,
        )
        if sol is None:
            meta["ok"] = False
            meta["error"] = f"插值点 #{i} C={C_i:.3f}m 正弦定理无解"
            return [], meta
        q1, q2, q3 = sol
        prev_sol = (q1, q2, q3)
        q = q0.copy()
        q[0], q[1], q[2] = q1, q2, q3
        waypoints.append(q)

    z_est_end = estimate_chest_z_world(waypoints[-1], base_link_z, geom)
    meta["z_est_end"] = round(z_est_end, 4)
    meta["n_waypoints"] = len(waypoints)
    meta["q_end"] = [round(float(x), 4) for x in waypoints[-1][:3]]
    return waypoints, meta


def _sine_manifold_waypoints_between_C(
    trunk_q: np.ndarray,
    C_start: float,
    C_end: float,
    *,
    n_steps: int,
    fold_sign: float,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> Tuple[List[np.ndarray], Optional[float]]:
    """正弦定理流形上 C_start→C_end 插值路点；返回 (waypoints, 末段 q1 步进方向符号)。"""
    q0 = np.asarray(trunk_q, dtype=np.float64).reshape(4).copy()
    n = max(2, int(n_steps))
    Cs = np.linspace(float(C_start), float(C_end), n)
    waypoints: List[np.ndarray] = []
    prev_sol: Optional[Tuple[float, float, float]] = (
        float(q0[0]), float(q0[1]), float(q0[2])
    )
    dq1_sign: Optional[float] = None
    prev_q1: Optional[float] = None
    for C_i in Cs:
        sol = solve_joints_for_vertical_C(
            float(C_i), fold_sign=fold_sign, geom=geom, prev_q=prev_sol,
        )
        if sol is None:
            return [], dq1_sign
        q1, q2, q3 = sol
        if prev_q1 is not None and abs(q1 - prev_q1) > 1e-6:
            dq1_sign = 1.0 if q1 > prev_q1 else -1.0
        prev_q1 = q1
        prev_sol = (q1, q2, q3)
        q = q0.copy()
        q[0], q[1], q[2] = q1, q2, q3
        waypoints.append(q)
    return waypoints, dq1_sign


def _on_sine_manifold_q(
    q1: float,
    q2: float,
    q3: float,
    *,
    tol: float = 0.06,
) -> bool:
    """|q2| ≈ |q1|+|q3| 时视为仍在正弦定理协同流形上。"""
    return abs(abs(float(q2)) - (abs(float(q1)) + abs(float(q3)))) <= tol


def plan_q1q2_coupled_waypoints(
    trunk_q: np.ndarray,
    z_target_world: float,
    base_link_z: float,
    *,
    dq_sign_hint: float = 1.0,
    theta_z_hold_deg: Optional[float] = None,
    theta_z_tol_deg: float = 4.0,
    dq_step_rad: float = 0.015,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    q_limits=R1PRO_Q_LIMITS,
    max_steps: int = 400,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """阶段2：dq1=dq2（同幅同向），每步反解 q3 保持胸廓 θz≈常数（垂直于地面）。

    theta_z_hold_deg 默认取起点 URDF θz（阶段1末段姿态）。
    """
    q = np.asarray(trunk_q, dtype=np.float64).reshape(4).copy()
    q4_lock = float(q[3])
    z_start = estimate_chest_z_world(q, base_link_z, geom)
    z_tgt = float(z_target_world)
    descending = z_tgt < z_start - 1e-4
    tz_hold = (
        float(theta_z_hold_deg)
        if theta_z_hold_deg is not None
        else theta_z_from_trunk_q(q[0], q[1], q[2], q4_lock)
    )

    meta: Dict[str, Any] = {
        "ok": True,
        "phase": "q1q2_coupled_theta_guard",
        "z_start": round(z_start, 4),
        "z_target": round(z_tgt, 4),
        "theta_z_hold_deg": round(tz_hold, 2),
        "theta_z_tol_deg": round(float(theta_z_tol_deg), 2),
    }

    if abs(z_tgt - z_start) < 1e-4:
        meta["skipped"] = True
        return [q], meta

    step = abs(float(dq_step_rad))
    sign = 1.0 if float(dq_sign_hint) >= 0 else -1.0
    dq = sign * step if descending else -sign * step

    lo1, hi1 = q_limits[0]
    lo2, hi2 = q_limits[1]
    lo3, hi3 = q_limits[2]

    nq1, nq2 = float(q[0]) + dq, float(q[1]) + dq
    nq3 = solve_q3_for_theta_z_holding_q12(
        tz_hold, nq1, nq2, prefer_q3=float(q[2]), q_limits=q_limits,
    )
    if nq3 is None or not (
        lo1 <= nq1 <= hi1 and lo2 <= nq2 <= hi2 and lo3 <= nq3 <= hi3
    ):
        meta["ok"] = False
        meta["error"] = "阶段2 首步无法在保持 θz 下运动"
        return [], meta

    waypoints: List[np.ndarray] = [q.copy()]
    tz_trace: List[float] = [theta_z_from_trunk_q(q[0], q[1], q[2], q4_lock)]
    for _ in range(max_steps):
        z_now = estimate_chest_z_world(q, base_link_z, geom)
        if descending and z_now <= z_tgt + 1e-3:
            break
        if not descending and z_now >= z_tgt - 1e-3:
            break

        nq1 = float(q[0]) + dq
        nq2 = float(q[1]) + dq
        if not (lo1 <= nq1 <= hi1 and lo2 <= nq2 <= hi2):
            meta["limited_by"] = "joint_limits_q12"
            break

        nq3 = solve_q3_for_theta_z_holding_q12(
            tz_hold, nq1, nq2, prefer_q3=float(q[2]), q_limits=q_limits,
        )
        if nq3 is None or not (lo3 <= nq3 <= hi3):
            meta["limited_by"] = "joint_limits_q3_theta"
            break

        q_next = np.array([nq1, nq2, nq3, q4_lock], dtype=np.float64)
        q = q_next
        waypoints.append(q.copy())
        tz_trace.append(theta_z_from_trunk_q(nq1, nq2, nq3, q4_lock))

    z_end = estimate_chest_z_world(q, base_link_z, geom)
    meta["z_end"] = round(z_end, 4)
    meta["n_waypoints"] = len(waypoints)
    meta["dq_rad"] = round(dq, 5)
    meta["dq_sign_hint"] = sign
    meta["q_end"] = [round(float(x), 4) for x in q[:3]]
    if tz_trace:
        meta["theta_z_spread_deg"] = round(max(tz_trace) - min(tz_trace), 3)
        meta["theta_z_end_deg"] = round(tz_trace[-1], 2)
    if descending and z_end > z_tgt + 0.012:
        meta["partial"] = True
    if not descending and z_end < z_tgt - 0.012:
        meta["partial"] = True
    return waypoints, meta


def plan_vertical_lift_waypoints_v2(
    trunk_q: np.ndarray,
    z_start_world: float,
    z_end_world: float,
    *,
    base_link_z: float = 0.05,
    n_steps: int = 20,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    fold_sign: Optional[float] = None,
    dq_step_rad: float = 0.015,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """两阶段垂直升降：阶段1 正弦定理 C 流形；阶段2 q1/q2 同角协同（范围外）。

    上升为下降的逆序：先阶段2 回到流形边界，再阶段1 沿 C 上升。
    """
    q0 = np.asarray(trunk_q, dtype=np.float64).reshape(4).copy()
    z0 = float(z_start_world)
    z_req = float(z_end_world)
    user_descending = z_req < z0 - 1e-4
    z1 = _clip(z_req, CHEST_Z_MIN_M, CHEST_Z_MAX_M)
    if user_descending and z1 >= z0 - 1e-4:
        z1 = min(z0 - 1e-3, CHEST_Z_MIN_M)

    mf = scan_sine_law_manifold(geom)
    if not mf.get("ok"):
        return [], {**mf, "ok": False}

    C_min = float(mf["C_min_m"])
    C_max = float(mf["C_max_m"])
    z_manifold_min = float(base_link_z) + float(mf["z_chest_min_m"])
    z_manifold_max = float(base_link_z) + float(mf["z_chest_max_m"])
    q_at_C_min = mf.get("q_at_C_min")
    q_at_C_max = mf.get("q_at_C_max")

    C_curr = vertical_C_from_trunk_q(q0, geom)
    meta: Dict[str, Any] = {
        "ok": True,
        "model": "sine_triangle_vertical_C_plus_q12_q23_coupled",
        "z_start": round(z0, 4),
        "z_end": round(z1, 4),
        "C_start": round(C_curr, 4),
        "C_range_m": [round(C_min, 4), round(C_max, 4)],
        "z_manifold_world_m": [round(z_manifold_min, 4), round(z_manifold_max, 4)],
        "phases": [],
    }

    if abs(z1 - z0) < 1e-4:
        meta["skipped"] = True
        return [q0], meta

    if fold_sign is None:
        fold_sign = 1.0 if float(q0[0]) >= 0 else -1.0
        if abs(float(q0[0])) < 0.05 and z1 < z0:
            fold_sign = 1.0

    descending = z1 < z0

    # ── 上升且当前处于阶段2（偏离正弦流形）→ 先 q1q2 协同回到 C_min 边界 ──
    if not descending and not _on_sine_manifold_q(q0[0], q0[1], q0[2]):
        if q_at_C_min is not None:
            q_ref = np.array(
                [q_at_C_min[0], q_at_C_min[1], q_at_C_min[2], float(q0[3])],
                dtype=np.float64,
            )
            z_boundary = estimate_chest_z_world(q_ref, base_link_z, geom)
            if z0 < z_boundary - 1e-3:
                tz_hold = theta_z_from_trunk_q(q0[0], q0[1], q0[2], q0[3])
                wp2, m2 = plan_q1q2_coupled_waypoints(
                    q0, z_boundary, base_link_z,
                    dq_sign_hint=float(fold_sign),
                    theta_z_hold_deg=tz_hold,
                    dq_step_rad=dq_step_rad,
                    geom=geom,
                )
                meta["phases"].append(m2)
                if not m2.get("ok") or not wp2:
                    meta["ok"] = False
                    meta["error"] = m2.get("error", "阶段2 回升至流形边界失败")
                    return [], meta
                q0 = wp2[-1]
                C_curr = vertical_C_from_trunk_q(q0, geom)
                z0 = estimate_chest_z_world(q0, base_link_z, geom)

    waypoints: List[np.ndarray] = []
    dq1_sign = 1.0

    if descending:
        C_tgt_sine = max(C_min, C_curr + (z1 - z0))
        C_end_p1 = C_tgt_sine
        # 仅当 C 已在流形下限时跳过阶段1
        if C_end_p1 < C_curr - 1e-4 and C_curr > C_min + 0.01:
            n1 = max(8, int(n_steps * min(1.0, (C_curr - C_end_p1) / max(C_curr - C_min, 0.05))))
            wp1, dq1_sign = _sine_manifold_waypoints_between_C(
                q0, C_curr, C_end_p1,
                n_steps=n1, fold_sign=fold_sign, geom=geom,
            )
            if not wp1:
                meta["ok"] = False
                meta["error"] = "阶段1 正弦定理插值失败"
                return [], meta
            meta["phases"].append({
                "phase": "sine_manifold",
                "C_start": round(C_curr, 4),
                "C_end": round(C_end_p1, 4),
                "n_waypoints": len(wp1),
            })
            waypoints.extend(wp1)
            q0 = wp1[-1]
            z0 = estimate_chest_z_world(q0, base_link_z, geom)

        z_after_p1 = estimate_chest_z_world(q0, base_link_z, geom)
        if z1 < z_after_p1 - 0.008:
            wp2, m2 = plan_phase2_q1q2_locked_q3_waypoints(
                q0, base_link_z,
                z_target_world=z1,
                dq_step_rad=abs(float(dq_step_rad)),
            )
            meta["phases"].append(m2)
            if not m2.get("ok") or not wp2:
                meta["ok"] = False
                meta["error"] = m2.get("error", "阶段2 q1=q2 协同失败")
                return waypoints, meta
            if len(waypoints) > 0:
                waypoints.extend(wp2[1:])
            else:
                waypoints.extend(wp2)
            q0 = waypoints[-1]
            z_after_p2 = estimate_chest_z_world(q0, base_link_z, geom)
            if z1 < z_after_p2 - 0.008:
                wp3, m3 = plan_phase3_q2q3_locked_q1_waypoints(
                    q0, base_link_z,
                    z_target_world=z1,
                    dq_step_rad=abs(float(dq_step_rad)),
                )
                meta["phases"].append(m3)
                if not m3.get("ok") or not wp3:
                    meta["ok"] = False
                    meta["error"] = m3.get("error", "阶段3 q2=q3 协同失败")
                    return waypoints, meta
                if len(waypoints) > 0:
                    waypoints.extend(wp3[1:])
                else:
                    waypoints.extend(wp3)
    else:
        # 上升：阶段1 沿 C 至目标（流形内），必要时已在上方完成阶段2 回边界
        C_tgt_sine = min(C_max, C_curr + (z1 - z0))
        if C_tgt_sine > C_curr + 1e-4:
            n1 = max(8, int(n_steps))
            wp1, _ = _sine_manifold_waypoints_between_C(
                q0, C_curr, C_tgt_sine,
                n_steps=n1, fold_sign=fold_sign, geom=geom,
            )
            if not wp1:
                meta["ok"] = False
                meta["error"] = "阶段1 正弦定理上升插值失败"
                return [], meta
            meta["phases"].append({
                "phase": "sine_manifold",
                "C_start": round(C_curr, 4),
                "C_end": round(C_tgt_sine, 4),
                "n_waypoints": len(wp1),
            })
            waypoints.extend(wp1)
            q0 = wp1[-1]
            z0 = estimate_chest_z_world(q0, base_link_z, geom)

        if z1 > z0 + 0.008:
            meta["ok"] = False
            meta["error"] = (
                f"目标 z={z1:.3f}m 超出两阶段可达（当前 z≈{z0:.3f}m，"
                f"流形上限 z≈{z_manifold_max:.3f}m）"
            )
            return waypoints, meta

    if not waypoints:
        meta["ok"] = False
        meta["error"] = "未生成路点"
        return [], meta

    z_est_end = estimate_chest_z_world(waypoints[-1], base_link_z, geom)
    meta["C_end"] = round(vertical_C_from_trunk_q(waypoints[-1], geom), 4)
    meta["z_est_end"] = round(z_est_end, 4)
    meta["n_waypoints"] = len(waypoints)
    meta["q_end"] = [round(float(x), 4) for x in waypoints[-1][:3]]
    p2 = next(
        (p for p in meta.get("phases", [])
         if p.get("phase") in (
             "q1q2_coupled_theta_guard",
             "q2_backward_theta_guard",
             "q1q2_locked_q3",
         )),
        None,
    )
    if p2 is not None:
        meta["theta_z_hold_deg"] = p2.get("theta_z_hold_deg")
    return waypoints, meta


# 垂直轨迹规划缓存：同一起点姿态 + 目标 z 不重复 FK 求解
_VERTICAL_PLAN_CACHE: Dict[str, Tuple[List[np.ndarray], Dict[str, Any]]] = {}


def _vertical_plan_cache_key(
    trunk_q: np.ndarray,
    z_start_world: float,
    z_end_world: float,
    base_link_z: float,
    n_steps: int,
) -> str:
    q = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    parts = [
        "p23_coupled_v1",
        round(float(z_start_world), 4),
        round(float(z_end_world), 4),
        round(float(base_link_z), 4),
        int(n_steps),
    ] + [round(float(x), 4) for x in q[:4]]
    return "|".join(str(x) for x in parts)


def plan_vertical_lift_waypoints_v2_cached(
    trunk_q: np.ndarray,
    z_start_world: float,
    z_end_world: float,
    *,
    base_link_z: float = 0.05,
    n_steps: int = 20,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    fold_sign: Optional[float] = None,
    dq_step_rad: float = 0.015,
    use_cache: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """带内存缓存的 v2 规划；返回完整阶段1+阶段2 关节角轨迹（开环执行用）。"""
    key = _vertical_plan_cache_key(
        trunk_q, z_start_world, z_end_world, base_link_z, n_steps,
    )
    if use_cache and key in _VERTICAL_PLAN_CACHE:
        wp_cached, meta_cached = _VERTICAL_PLAN_CACHE[key]
        meta = dict(meta_cached)
        meta["cached"] = True
        meta["cache_key"] = key
        return [np.array(q, dtype=np.float64).copy() for q in wp_cached], meta

    wp, meta = plan_vertical_lift_waypoints_v2(
        trunk_q,
        z_start_world,
        z_end_world,
        base_link_z=base_link_z,
        n_steps=n_steps,
        geom=geom,
        fold_sign=fold_sign,
        dq_step_rad=dq_step_rad,
    )
    meta = dict(meta)
    meta["cached"] = False
    meta["cache_key"] = key
    if use_cache and meta.get("ok") and wp:
        _VERTICAL_PLAN_CACHE[key] = (
            [np.array(q, dtype=np.float64).copy() for q in wp],
            dict(meta),
        )
    return wp, meta


def clear_vertical_plan_cache() -> int:
    """清空垂直轨迹缓存，返回清除条数。"""
    n = len(_VERTICAL_PLAN_CACHE)
    _VERTICAL_PLAN_CACHE.clear()
    return n


def measure_two_phase_vertical_range(
    trunk_q: Optional[np.ndarray] = None,
    *,
    base_link_z: float = 0.05,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> Dict[str, Any]:
    """测量阶段1 / 阶段2 / 合计胸口 z 可达范围（FK 模型）。

    若给定 trunk_q，阶段2 从「当前姿态沿阶段1降至 C_min」或「已在流形外则直接阶段2」算起。
    """
    mf = scan_sine_law_manifold(geom, refresh=True)
    if not mf.get("ok"):
        return dict(mf)

    if trunk_q is None:
        trunk_q = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q0 = np.asarray(trunk_q, dtype=np.float64).reshape(4)

    z_start = estimate_chest_z_world(q0, base_link_z, geom)
    z_p1_min = float(base_link_z) + float(mf["z_chest_min_m"])
    z_p1_max = float(base_link_z) + float(mf["z_chest_max_m"])

    wp_full, meta_full = plan_vertical_lift_waypoints_v2(
        q0, z_start, CHEST_Z_MIN_M, base_link_z=base_link_z,
    )
    z_p2_min = float(meta_full.get("z_est_end", z_p1_min))
    if wp_full:
        z_p2_min = min(estimate_chest_z_world(q, base_link_z, geom) for q in wp_full)

    upright_q = np.array([0.0, 0.0, 0.0, float(q0[3])], dtype=np.float64)
    z_upright = estimate_chest_z_world(upright_q, base_link_z, geom)
    wp_up, _ = plan_vertical_lift_waypoints_v2(
        upright_q, z_upright, CHEST_Z_MIN_M, base_link_z=base_link_z,
    )
    z_upright_min = min(
        (estimate_chest_z_world(q, base_link_z, geom) for q in wp_up),
        default=z_p1_min,
    ) if wp_up else z_p1_min

    return {
        "ok": True,
        "from_pose": {
            "trunk_q": [round(float(x), 4) for x in q0[:4]],
            "chest_z_world_m": round(z_start, 4),
            "C_m": round(vertical_C_from_trunk_q(q0, geom), 4),
        },
        "phase1_sine_manifold": {
            "C_m": [mf["C_min_m"], mf["C_max_m"]],
            "z_chest_world_m": [round(z_p1_min, 4), round(z_p1_max, 4)],
            "drop_from_upright_m": round(z_upright - z_p1_min, 4),
        },
        "phase2_q1q2_coupled_extra_from_C_min": {
            "z_chest_world_m": round(z_p2_min, 4),
            "extra_drop_from_phase1_min_m": round(z_p1_min - z_p2_min, 4),
            "plan_meta": meta_full,
        },
        "phase1_plus_phase2": {
            "z_chest_min_world_m": round(z_p2_min, 4),
            "total_drop_from_current_m": round(z_start - z_p2_min, 4),
            "total_drop_from_upright_m": round(z_upright - z_upright_min, 4),
            "z_chest_min_from_upright_m": round(z_upright_min, 4),
        },
        "manifold": mf,
    }


# 肩中点相对胸口高度（与 move_to_object_geom 一致）
SHOULDER_ABOVE_CHEST_M = 0.303


def plan_pre_descent_for_point_dz(
    trunk_q: np.ndarray,
    chest_z_world: float,
    shoulder_z_world: float,
    target_z_world: float,
    base_link_z: float,
    *,
    dz_target_max_m: float = POINT_PRE_DESCENT_DZ_MAX_M,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """move_to_point 预降：肩高与目标 z 的 dz 过大时，协同 q1/q2/q3 垂直下降。

    dz = shoulder_z − target_z。若 dz ≤ dz_target_max 则跳过。
    否则尽量降到 dz≈dz_target_max；若行程不够则降到流形最低点。
    """
    chest_z = float(chest_z_world)
    shoulder_z = float(shoulder_z_world)
    target_z = float(target_z_world)
    dz0 = shoulder_z - target_z
    dz_cap = float(dz_target_max_m)

    meta: Dict[str, Any] = {
        "ok": True,
        "skipped": False,
        "dz_start_m": round(dz0, 4),
        "dz_target_max_m": round(dz_cap, 4),
        "model": "sine_triangle_vertical_C_plus_q12_q23_coupled",
    }

    if dz0 <= dz_cap + 1e-3:
        meta["skipped"] = True
        meta["reason"] = f"dz={dz0:.3f}m ≤ {dz_cap}m"
        return [], meta

    mf = scan_sine_law_manifold(geom)
    if not mf.get("ok"):
        meta["ok"] = False
        meta["error"] = mf.get("error", "流形扫描失败")
        return [], meta

    z_min_world = float(base_link_z) + float(mf["z_chest_min_m"])
    # 肩随胸口近似平移：降 (dz0−dz_cap) 即把 dz 收到 dz_cap
    drop_ideal = dz0 - dz_cap
    z_end_ideal = chest_z - drop_ideal
    z_end = max(z_min_world, min(chest_z - 1e-3, z_end_ideal))
    clipped_to_min = z_end <= z_min_world + 0.005 and z_end_ideal < z_min_world - 0.005

    if z_end >= chest_z - 0.008:
        meta["skipped"] = True
        meta["reason"] = "已在最低胸口高度，无法继续预降"
        return [], meta

    waypoints, vmeta = plan_vertical_lift_waypoints_v2(
        trunk_q,
        chest_z,
        z_end,
        base_link_z=base_link_z,
        n_steps=max(12, int(abs(chest_z - z_end) / 0.015) + 2),
    )
    if not vmeta.get("ok") or not waypoints:
        meta["ok"] = False
        meta["error"] = vmeta.get("error", "两阶段垂直预降规划失败")
        meta["vertical_lift_v2"] = vmeta
        return [], meta

    z_end_est = estimate_chest_z_world(waypoints[-1], base_link_z, geom)
    shoulder_drop = chest_z - z_end_est
    dz_end_est = dz0 - shoulder_drop
    meta.update({
        "chest_z_start": round(chest_z, 4),
        "chest_z_end_plan": round(z_end, 4),
        "chest_z_end_est": round(z_end_est, 4),
        "z_chest_min_world": round(z_min_world, 4),
        "clipped_to_min": clipped_to_min,
        "n_waypoints": len(waypoints),
        "dz_end_est_m": round(dz_end_est, 4),
        "vertical_lift_v2": vmeta,
        "model": "sine_triangle_vertical_C_plus_q12_q23_coupled",
    })
    if clipped_to_min and dz_end_est > dz_cap + 0.02:
        meta["partial"] = True
        meta["note"] = f"已降至最低点，dz≈{dz_end_est:.3f}m 仍 > {dz_cap}m"
    return waypoints, meta


def measure_C_range_report(
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    q_limits=R1PRO_Q_LIMITS,
) -> Dict[str, Any]:
    """测量 C 可达范围：正弦定理流形 + 关节网格 FK 包络（诊断用）。"""
    mf = scan_sine_law_manifold(geom, q_limits, refresh=True)
    lo1, hi1 = q_limits[0]
    lo2, hi2 = q_limits[1]
    lo3, hi3 = q_limits[2]
    fk_cs: List[float] = []
    for q1 in np.linspace(lo1, hi1, 50):
        for q3 in np.linspace(lo3, hi3, 50):
            q2_syn = -(abs(q1) + abs(q3))
            if lo2 <= q2_syn <= hi2:
                fk_cs.append(fk_vertical_C_m(q1, q2_syn, q3, geom))
            q2_alt = -q1
            if lo2 <= q2_alt <= hi2:
                fk_cs.append(fk_vertical_C_m(q1, q2_alt, q3, geom))

    return {
        "ok": True,
        "manifold_C_m": [mf.get("C_min_m"), mf.get("C_max_m")],
        "manifold_z_chest_base_m": [mf.get("z_chest_min_m"), mf.get("z_chest_max_m")],
        "fk_grid_C_m": [float(min(fk_cs)), float(max(fk_cs))] if fk_cs else None,
        "n_manifold_samples": mf.get("n_samples"),
        "upright_fk_C_m": fk_vertical_C_m(0.0, 0.0, 0.0, geom),
        "upright_q2_only_fk_C_m": fk_vertical_C_m(0.0, 0.04, 0.0, geom),
    }


def scan_reverse_sine_manifold_z_chain(
    base_link_z: float,
    *,
    q4: float = 0.0,
    fold_sign: float = -1.0,
    n_C: int = 320,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> List[Dict[str, float]]:
    """密集扫描反向正弦流形（fold_sign=-1）：C 从直立→C_min，θz≡90°。"""
    q0 = np.array([0.0, 0.0, 0.0, float(q4)], dtype=np.float64)
    mf = scan_sine_law_manifold(geom)
    if not mf.get("ok"):
        return []
    C0 = float(vertical_C_from_trunk_q(q0, geom))
    C_min = float(mf["C_min_m"])
    prev_sol: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    rows: List[Dict[str, float]] = []
    for C in np.linspace(C0, C_min, max(32, int(n_C))):
        sol = solve_joints_for_vertical_C(
            float(C), fold_sign=float(fold_sign), geom=geom, prev_q=prev_sol,
        )
        if sol is None:
            continue
        q1f, q2f, q3f = sol
        prev_sol = sol
        z_est = estimate_chest_z_world(
            np.array([q1f, q2f, q3f, float(q4)], dtype=np.float64),
            base_link_z,
            geom,
        )
        tz = fk_torso_link4_theta_z_deg(q1f, q2f, q3f, float(q4))
        rows.append({
            "C_m": float(C),
            "q1": float(q1f),
            "q2": float(q2f),
            "q3": float(q3f),
            "z": float(z_est),
            "theta_z_deg": float(tz),
            "on_sine_manifold": bool(_on_sine_manifold_q(q1f, q2f, q3f)),
        })
    rows.sort(key=lambda r: r["z"], reverse=True)
    return rows


def reverse_sine_manifold_z_bounds(
    base_link_z: float,
    *,
    q4: float = 0.0,
    fold_sign: float = -1.0,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> Tuple[float, float, float, float, np.ndarray]:
    """直立→反向正弦流形 C_min：返回 (z_start, z_end, C_start, C_min, q_at_C_min)。"""
    q0 = np.array([0.0, 0.0, 0.0, float(q4)], dtype=np.float64)
    z_start = estimate_chest_z_world(q0, base_link_z, geom)
    mf = scan_sine_law_manifold(geom)
    if not mf.get("ok"):
        return z_start, z_start, 0.7, 0.38, q0
    C0 = float(vertical_C_from_trunk_q(q0, geom))
    C_min = float(mf["C_min_m"])
    sol_min = solve_joints_for_vertical_C(C_min, fold_sign=float(fold_sign), geom=geom)
    if sol_min is None:
        return z_start, z_start, C0, C_min, q0
    q_end = np.array([sol_min[0], sol_min[1], sol_min[2], float(q4)], dtype=np.float64)
    z_end = estimate_chest_z_world(q_end, base_link_z, geom)
    return float(z_start), float(z_end), C0, C_min, q_end


def solve_q123_on_reverse_sine_manifold_for_z(
    z_target_world: float,
    base_link_z: float,
    *,
    z_start_world: float,
    C_start: float,
    C_min: float,
    q4: float = 0.0,
    prev_trunk_q: Optional[np.ndarray] = None,
    fold_sign: float = -1.0,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    z_tol_m: float = 0.012,
) -> Optional[Dict[str, Any]]:
    """在反向正弦流形上：C_tgt=C_start+(z_tgt-z_start)，fold_sign=-1 反解 q1/q2/q3（θz≡90°）。"""
    z_tgt = float(z_target_world)
    z0 = float(z_start_world)
    C_tgt = float(C_start) + (z_tgt - z0)
    C_tgt = _clip(C_tgt, float(C_min), float(C_start))
    pq = np.asarray(
        prev_trunk_q if prev_trunk_q is not None else [0.0, 0.0, 0.0, float(q4)],
        dtype=np.float64,
    ).reshape(4)
    sol = solve_joints_for_vertical_C(
        C_tgt,
        fold_sign=float(fold_sign),
        geom=geom,
        prev_q=tuple(float(x) for x in pq[:3]),
    )
    if sol is None:
        return None
    q1f, q2f, q3f = sol
    z_est = estimate_chest_z_world(
        np.array([q1f, q2f, q3f, float(q4)], dtype=np.float64),
        base_link_z,
        geom,
    )
    tz = fk_torso_link4_theta_z_deg(q1f, q2f, q3f, float(q4))
    z_err = abs(z_est - z_tgt)
    if z_err > max(float(z_tol_m), 0.015):
        return None
    return {
        "ok": True,
        "q1_rad": round(q1f, 5),
        "q2_rad": round(q2f, 5),
        "q3_rad": round(q3f, 5),
        "chest_z_world_m": round(z_est, 4),
        "theta_z_deg": round(tz, 3),
        "z_err_m": round(z_est - z_tgt, 4),
        "C_m": round(C_tgt, 4),
        "on_sine_manifold": bool(_on_sine_manifold_q(q1f, q2f, q3f)),
        "manifold_fold_sign": float(fold_sign),
    }


def reverse_upright_to_phase2_z_bounds(
    base_link_z: float,
    *,
    q4: float = 0.0,
    dq_step_rad: float = 0.02,
    fold_sign: float = -1.0,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> Tuple[float, float, np.ndarray]:
    """反向轨迹：直立 → 阶段1(C_min, fold=-1) → 阶段2末，返回 (z_start, z_end, q_phase2_end)。"""
    q0 = np.array([0.0, 0.0, 0.0, float(q4)], dtype=np.float64)
    z_start = estimate_chest_z_world(q0, base_link_z, geom)
    mf = scan_sine_law_manifold(geom)
    if not mf.get("ok"):
        return z_start, z_start, q0
    C0 = vertical_C_from_trunk_q(q0, geom)
    C_min = float(mf["C_min_m"])
    wp1, _ = _sine_manifold_waypoints_between_C(
        q0, C0, C_min, n_steps=32, fold_sign=float(fold_sign), geom=geom,
    )
    if not wp1:
        return z_start, z_start, q0
    wp2, _ = plan_phase2_q1q2_locked_q3_waypoints(
        wp1[-1], base_link_z,
        reverse_joint_dirs=True,
        dq_step_rad=abs(float(dq_step_rad)),
        geom=geom,
    )
    q_end = wp2[-1] if wp2 else wp1[-1]
    z_end = estimate_chest_z_world(q_end, base_link_z, geom)
    return float(z_start), float(z_end), q_end


def sample_reverse_vertical_z_ik_table(
    base_link_z: float,
    *,
    dz_step_m: float = 0.01,
    theta_z_deg: float = 90.0,
    q4: float = 0.0,
    dq_step_rad: float = 0.02,
    fold_sign: float = -1.0,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> Dict[str, Any]:
    """直立→反向正弦流形 C_min：1cm 胸口 z 间隔，fold_sign=-1 + θz≡90°。"""
    z_start, z_end_mf, C0, C_min, q_mf_end = reverse_sine_manifold_z_bounds(
        base_link_z, q4=q4, fold_sign=float(fold_sign), geom=geom,
    )
    _, z_end_p2, q_p2 = reverse_upright_to_phase2_z_bounds(
        base_link_z, q4=q4, dq_step_rad=dq_step_rad, fold_sign=float(fold_sign), geom=geom,
    )
    step = max(0.005, float(dz_step_m))
    z_lo = math.floor(z_end_mf / step) * step
    z_hi = math.ceil(z_start / step) * step
    chain = scan_reverse_sine_manifold_z_chain(
        base_link_z, q4=q4, fold_sign=float(fold_sign), geom=geom,
    )
    levels: List[float] = []
    z = round(z_hi, 4)
    while z >= z_lo - 1e-6:
        levels.append(round(z, 4))
        z = round(z - step, 4)

    prev = np.array([0.0, 0.0, 0.0, float(q4)], dtype=np.float64)
    samples: List[Dict[str, Any]] = []
    for i, z_tgt in enumerate(levels):
        upward = round(z_tgt - z_start, 4)
        row: Dict[str, Any] = {
            "i": i,
            "z_target_m": round(z_tgt, 4),
            "upward_m": upward,
            "theta_z_target_deg": round(float(theta_z_deg), 2),
        }
        sol = solve_q123_on_reverse_sine_manifold_for_z(
            z_tgt,
            base_link_z,
            z_start_world=z_start,
            C_start=C0,
            C_min=C_min,
            q4=q4,
            prev_trunk_q=prev,
            fold_sign=float(fold_sign),
            geom=geom,
        )
        if sol is None:
            row["ok"] = False
            row["error"] = "正弦流形反解失败"
            samples.append(row)
            continue
        prev = np.array(
            [sol["q1_rad"], sol["q2_rad"], sol["q3_rad"], float(q4)],
            dtype=np.float64,
        )
        row.update(sol)
        row["trunk_q"] = [
            sol["q1_rad"], sol["q2_rad"], sol["q3_rad"], round(float(q4), 4),
        ]
        samples.append(row)

    ok_rows = [r for r in samples if r.get("ok")]
    return {
        "ok": len(ok_rows) > 0,
        "model": "reverse_sine_manifold_fold_m1_theta90",
        "z_start_m": round(z_start, 4),
        "z_end_m": round(z_end_mf, 4),
        "z_end_manifold_Cmin_m": round(z_end_mf, 4),
        "z_end_phase2_openloop_m": round(z_end_p2, 4),
        "C_start_m": round(C0, 4),
        "C_min_m": round(C_min, 4),
        "dz_step_m": round(step, 4),
        "n_levels": len(levels),
        "n_solved": len(ok_rows),
        "theta_z_hold_deg": round(float(theta_z_deg), 2),
        "q_at_C_min_manifold": [round(float(x), 4) for x in q_mf_end[:3]],
        "q_phase2_end_openloop": [round(float(x), 4) for x in q_p2[:3]],
        "manifold_chain_n": len(chain),
        "samples": samples,
    }


def plan_phase2_reverse_theta90_waypoints(
    trunk_q_start: np.ndarray,
    base_link_z: float,
    *,
    theta_z_deg: float = 90.0,
    dq_step_rad: float = 0.012,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    q_limits=R1PRO_Q_LIMITS,
    max_steps: int = 200,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """阶段2（反向）：放弃正弦流形；|dq2|=|dq1|、每步反解 q3 保 θz≈90°，直至触限。

    从阶段1末（流形 C_min）接续；典型首先触 **q2 前摆上限**。
    """
    q = np.asarray(trunk_q_start, dtype=np.float64).reshape(4).copy()
    lo1, hi1 = q_limits[0]
    lo2, hi2 = q_limits[1]
    lo3, hi3 = q_limits[2]
    dq = abs(float(dq_step_rad))
    z_start = estimate_chest_z_world(q, base_link_z, geom)
    meta: Dict[str, Any] = {
        "ok": True,
        "phase": "reverse_theta90_q1q2_coupled_solve_q3",
        "theta_z_hold_deg": round(float(theta_z_deg), 2),
        "dq_rad": round(dq, 5),
        "coupling": "|dq2|=|dq1|, q3 from IK",
        "on_sine_manifold": False,
        "z_start": round(z_start, 4),
        "q_start": [round(float(x), 4) for x in q[:3]],
    }
    waypoints: List[np.ndarray] = [q.copy()]
    tz_trace: List[float] = [theta_z_from_trunk_q(q[0], q[1], q[2], q[3])]

    for _ in range(max_steps):
        nq1 = float(q[0]) - dq
        nq2 = float(q[1]) + dq
        if nq2 > hi2 + 1e-4:
            meta["limited_by"] = "q2_limit"
            meta["q2_limit_rad"] = round(hi2, 4)
            break
        if nq1 < lo1 - 1e-4:
            meta["limited_by"] = "q1_limit"
            meta["q1_limit_rad"] = round(lo1, 4)
            break
        nq3 = solve_q3_for_theta_z_holding_q12(
            float(theta_z_deg), nq1, nq2,
            prefer_q3=float(q[2]),
            q_limits=q_limits,
        )
        if nq3 is None:
            meta["limited_by"] = "no_q3_for_theta90"
            break
        if not (lo3 - 1e-4 <= float(nq3) <= hi3 + 1e-4):
            meta["limited_by"] = "q3_limit"
            meta["q3_limit_rad"] = round(lo3 if nq3 < lo3 else hi3, 4)
            break
        q = np.array([nq1, nq2, float(nq3), float(q[3])], dtype=np.float64)
        waypoints.append(q.copy())
        tz_trace.append(theta_z_from_trunk_q(nq1, nq2, float(nq3), float(q[3])))
        if nq2 >= hi2 - 1e-4:
            meta["limited_by"] = "q2_limit"
            meta["q2_limit_rad"] = round(hi2, 4)
            break
        if nq1 <= lo1 + 1e-4:
            meta["limited_by"] = "q1_limit"
            meta["q1_limit_rad"] = round(lo1, 4)
            break

    z_end = estimate_chest_z_world(waypoints[-1], base_link_z, geom)
    meta["z_end"] = round(z_end, 4)
    meta["z_drop_m"] = round(z_start - z_end, 4)
    meta["n_waypoints"] = len(waypoints)
    meta["q_end"] = [round(float(x), 4) for x in waypoints[-1][:3]]
    if tz_trace:
        meta["theta_z_end_deg"] = round(tz_trace[-1], 2)
        meta["theta_z_spread_deg"] = round(max(tz_trace) - min(tz_trace), 3)
    if "limited_by" not in meta:
        meta["limited_by"] = "max_steps"
    return waypoints, meta


def _interp_trunk_q_by_chest_z(
    waypoints: List[np.ndarray],
    z_target: float,
    base_link_z: float,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> Optional[np.ndarray]:
    """按胸口 z 在路点列中线性插值 trunk q。"""
    if not waypoints:
        return None
    zs = [estimate_chest_z_world(w, base_link_z, geom) for w in waypoints]
    z_tgt = float(z_target)
    if z_tgt >= zs[0] - 1e-4:
        return np.asarray(waypoints[0], dtype=np.float64).copy()
    if z_tgt <= zs[-1] + 1e-4:
        return np.asarray(waypoints[-1], dtype=np.float64).copy()
    for a, b, za, zb in zip(waypoints, waypoints[1:], zs, zs[1:]):
        if zb <= z_tgt <= za:
            t = (z_tgt - zb) / max(za - zb, 1e-6)
            qa = np.asarray(a, dtype=np.float64)
            qb = np.asarray(b, dtype=np.float64)
            return (qa * t + qb * (1.0 - t)).astype(np.float64)
    return np.asarray(waypoints[-1], dtype=np.float64).copy()


def sample_phase2_reverse_theta90_z_table(
    trunk_q_phase1_end: np.ndarray,
    base_link_z: float,
    *,
    z_upright_world: float,
    dz_step_m: float = 0.01,
    theta_z_deg: float = 90.0,
    dq_step_rad: float = 0.012,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
) -> Dict[str, Any]:
    """阶段2：θz≈90°、非流形；1cm z 间隔采样（从阶段1末→阶段2触限）。"""
    wp, p2meta = plan_phase2_reverse_theta90_waypoints(
        trunk_q_phase1_end, base_link_z,
        theta_z_deg=theta_z_deg,
        dq_step_rad=dq_step_rad,
        geom=geom,
    )
    z_start = float(p2meta.get("z_start", 0.0))
    z_end = float(p2meta.get("z_end", z_start))
    step = max(0.005, float(dz_step_m))
    z_lo = math.floor(z_end / step) * step
    z_hi = math.ceil(z_start / step) * step
    levels: List[float] = []
    z = round(z_hi, 4)
    while z >= z_lo - 1e-6:
        levels.append(round(z, 4))
        z = round(z - step, 4)

    z_up = float(z_upright_world)
    samples: List[Dict[str, Any]] = []
    for i, z_tgt in enumerate(levels):
        q_wp = _interp_trunk_q_by_chest_z(wp, z_tgt, base_link_z, geom)
        if q_wp is None:
            continue
        z_est = estimate_chest_z_world(q_wp, base_link_z, geom)
        tz = fk_torso_link4_theta_z_deg(float(q_wp[0]), float(q_wp[1]), float(q_wp[2]), float(q_wp[3]))
        samples.append({
            "i": i,
            "phase": 2,
            "z_target_m": round(z_tgt, 4),
            "upward_m": round(z_tgt - z_up, 4),
            "ok": True,
            "q1_rad": round(float(q_wp[0]), 5),
            "q2_rad": round(float(q_wp[1]), 5),
            "q3_rad": round(float(q_wp[2]), 5),
            "chest_z_world_m": round(z_est, 4),
            "theta_z_deg": round(tz, 3),
            "on_sine_manifold": bool(_on_sine_manifold_q(q_wp[0], q_wp[1], q_wp[2])),
            "trunk_q": [round(float(x), 5) for x in q_wp[:3]] + [round(float(q_wp[3]), 4)],
        })

    return {
        "ok": bool(samples),
        "model": "reverse_phase2_theta90_off_manifold",
        "z_start_m": round(z_start, 4),
        "z_end_m": round(z_end, 4),
        "z_upright_m": round(z_up, 4),
        "total_drop_from_upright_m": round(z_up - z_end, 4),
        "phase2_extra_drop_m": round(z_start - z_end, 4),
        "dz_step_m": round(step, 4),
        "n_levels": len(levels),
        "n_samples": len(samples),
        "limited_by": p2meta.get("limited_by"),
        "limit_detail": {
            "q1_limit_rad": p2meta.get("q1_limit_rad"),
            "q2_limit_rad": p2meta.get("q2_limit_rad"),
            "q3_limit_rad": p2meta.get("q3_limit_rad"),
        },
        "phase2_plan": p2meta,
        "q_phase1_end": [round(float(x), 4) for x in np.asarray(trunk_q_phase1_end).reshape(4)[:3]],
        "samples": samples,
    }


# 反向 upward：z≥边界用正弦流形 phase1，z<边界用 θz-only phase2
REVERSE_UPWARD_Z_PHASE_BOUNDARY_M = 0.73
REVERSE_UPWARD_Z_CM_STEP_M = 0.01

_REVERSE_UPWARD_LUT_CACHE: Dict[str, Dict[str, Any]] = {}


def snap_z_to_nearest_cm(z_world: float, step_m: float = REVERSE_UPWARD_Z_CM_STEP_M) -> float:
    """胸口 z 对齐到最近 1cm 整数高度。"""
    step = max(0.001, float(step_m))
    return round(round(float(z_world) / step) * step, 4)


def _validated_upward_table_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "skills", "test", "upward", name)


def _load_validated_upward_rows(
    *,
    q4: float,
    z_boundary: float,
) -> Optional[Dict[str, Any]]:
    """Load the user-validated 1 cm reverse-fold tables rendered in skills/test/upward."""
    p1_path = _validated_upward_table_path(os.path.join("phase1", "phase1_sine_manifold_table.json"))
    p2_path = _validated_upward_table_path(os.path.join("phase2", "phase2_theta90_table.json"))
    try:
        with open(p1_path, "r", encoding="utf-8") as f:
            table_p1 = json.load(f)
        with open(p2_path, "r", encoding="utf-8") as f:
            table_p2 = json.load(f)
    except Exception:
        return None

    if not table_p1.get("ok") or not table_p2.get("ok"):
        return None
    z_upright = float(table_p1["z_start_m"])
    zb = float(z_boundary)
    p1_rows: List[Dict[str, Any]] = []
    for r in table_p1.get("samples", []):
        if not r.get("ok"):
            continue
        zt = float(r["z_target_m"])
        if zt + 1e-4 < zb:
            continue
        tq = list(r.get("trunk_q") or [r["q1_rad"], r["q2_rad"], r["q3_rad"], q4])
        p1_rows.append({
            "phase": 1,
            "z_target_m": round(zt, 4),
            "upward_m": round(zt - z_upright, 4),
            "q1_rad": r["q1_rad"],
            "q2_rad": r["q2_rad"],
            "q3_rad": r["q3_rad"],
            "trunk_q": [float(tq[0]), float(tq[1]), float(tq[2]), float(q4)],
            "theta_z_deg": r.get("theta_z_deg"),
            "on_sine_manifold": bool(r.get("on_sine_manifold", True)),
            "source_table": "skills/test/upward/phase1",
        })

    p2_rows: List[Dict[str, Any]] = []
    for r in table_p2.get("samples", []):
        if not r.get("ok"):
            continue
        zt = float(r["z_target_m"])
        if zt >= zb - 1e-4:
            continue
        tq = list(r.get("trunk_q") or [r["q1_rad"], r["q2_rad"], r["q3_rad"], q4])
        p2_rows.append({
            "phase": 2,
            "z_target_m": round(zt, 4),
            "upward_m": round(zt - z_upright, 4),
            "q1_rad": r["q1_rad"],
            "q2_rad": r["q2_rad"],
            "q3_rad": r["q3_rad"],
            "trunk_q": [float(tq[0]), float(tq[1]), float(tq[2]), float(q4)],
            "theta_z_deg": r.get("theta_z_deg"),
            "on_sine_manifold": bool(r.get("on_sine_manifold", False)),
            "source_table": "skills/test/upward/phase2",
        })

    combined = p1_rows + p2_rows
    combined.sort(key=lambda r: float(r["z_target_m"]), reverse=True)
    if not combined:
        return None
    return {
        "z_upright": z_upright,
        "p1_rows": p1_rows,
        "p2_rows": p2_rows,
        "rows": combined,
        "phase2_limited_by": table_p2.get("limited_by"),
        "phase2_z_end_m": table_p2.get("z_end_m"),
        "source": "validated_png_json_tables",
    }


def _lut_row_trunk_q(row: Dict[str, Any], q4: float) -> np.ndarray:
    tq = row.get("trunk_q")
    if tq and len(tq) >= 3:
        return np.array(
            [float(tq[0]), float(tq[1]), float(tq[2]), float(q4)],
            dtype=np.float64,
        )
    return np.array(
        [float(row["q1_rad"]), float(row["q2_rad"]), float(row["q3_rad"]), float(q4)],
        dtype=np.float64,
    )


def _nearest_lut_row_index(rows: List[Dict[str, Any]], z_world: float) -> int:
    if not rows:
        return 0
    z = float(z_world)
    return int(min(range(len(rows)), key=lambda i: abs(float(rows[i]["z_target_m"]) - z)))


def get_reverse_upward_combined_lut(
    base_link_z: float,
    *,
    q4: float = 0.0,
    z_boundary: float = REVERSE_UPWARD_Z_PHASE_BOUNDARY_M,
    dz_step_m: float = REVERSE_UPWARD_Z_CM_STEP_M,
    theta_z_deg: float = 90.0,
    dq_step_rad: float = 0.012,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    use_cache: bool = True,
) -> Dict[str, Any]:
    """合并 phase1（z≥边界，正弦流形）+ phase2（z<边界，θz≈90°）1cm 查表。"""
    zb = float(z_boundary)
    step = max(0.005, float(dz_step_m))
    key = "|".join(
        str(round(x, 5))
        for x in (float(base_link_z), float(q4), zb, step, float(theta_z_deg))
    )
    if use_cache and key in _REVERSE_UPWARD_LUT_CACHE:
        return dict(_REVERSE_UPWARD_LUT_CACHE[key])

    validated = _load_validated_upward_rows(q4=q4, z_boundary=zb)
    if validated is not None:
        z_upright = float(validated["z_upright"])
        p1_rows = list(validated["p1_rows"])
        p2_rows = list(validated["p2_rows"])
        combined = list(validated["rows"])
        phase2_limited_by = validated.get("phase2_limited_by")
        phase2_z_end = validated.get("phase2_z_end_m")
        source = validated.get("source")
    else:
        table_p1 = sample_reverse_vertical_z_ik_table(
            base_link_z,
            dz_step_m=step,
            theta_z_deg=theta_z_deg,
            q4=q4,
            geom=geom,
        )
        ok_p1 = [r for r in table_p1.get("samples", []) if r.get("ok")]
        if not ok_p1:
            out = {"ok": False, "error": "阶段1 正弦流形查表失败", "cache_key": key}
            return out

        z_upright = float(table_p1["z_start_m"])
        p1_rows = []
        for r in ok_p1:
            zt = float(r["z_target_m"])
            if zt + 1e-4 < zb:
                continue
            p1_rows.append({
                "phase": 1,
                "z_target_m": round(zt, 4),
                "upward_m": round(zt - z_upright, 4),
                "q1_rad": r["q1_rad"],
                "q2_rad": r["q2_rad"],
                "q3_rad": r["q3_rad"],
                "trunk_q": r["trunk_q"],
                "theta_z_deg": r.get("theta_z_deg"),
                "on_sine_manifold": True,
            })

        q_p1_end = _lut_row_trunk_q(ok_p1[-1], q4)
        table_p2 = sample_phase2_reverse_theta90_z_table(
            q_p1_end,
            base_link_z,
            z_upright_world=z_upright,
            dz_step_m=step,
            theta_z_deg=theta_z_deg,
            dq_step_rad=dq_step_rad,
            geom=geom,
        )
        p2_rows = []
        for r in table_p2.get("samples", []):
            zt = float(r["z_target_m"])
            if zt >= zb - 1e-4:
                continue
            p2_rows.append({
                "phase": 2,
                "z_target_m": round(zt, 4),
                "upward_m": round(zt - z_upright, 4),
                "q1_rad": r["q1_rad"],
                "q2_rad": r["q2_rad"],
                "q3_rad": r["q3_rad"],
                "trunk_q": r["trunk_q"],
                "theta_z_deg": r.get("theta_z_deg"),
                "on_sine_manifold": r.get("on_sine_manifold", False),
            })

        combined = p1_rows + p2_rows
        combined.sort(key=lambda r: float(r["z_target_m"]), reverse=True)
        phase2_limited_by = table_p2.get("limited_by")
        phase2_z_end = table_p2.get("z_end_m")
        source = "generated_fallback"

    if not combined:
        out = {"ok": False, "error": "合并查表为空", "cache_key": key}
        return out

    z_min = float(min(r["z_target_m"] for r in combined))
    z_max = float(max(r["z_target_m"] for r in combined))
    out = {
        "ok": True,
        "cache_key": key,
        "model": "reverse_upward_phase1_manifold_phase2_theta90",
        "z_boundary_m": round(zb, 4),
        "z_upright_m": round(z_upright, 4),
        "z_min_m": round(z_min, 4),
        "z_max_m": round(z_max, 4),
        "dz_step_m": round(step, 4),
        "n_phase1": len(p1_rows),
        "n_phase2": len(p2_rows),
        "n_rows": len(combined),
        "phase2_limited_by": phase2_limited_by,
        "phase2_z_end_m": phase2_z_end,
        "source": source,
        "rows": combined,
    }
    if use_cache:
        _REVERSE_UPWARD_LUT_CACHE[key] = dict(out)
    return out


def plan_reverse_upward_trajectory_from_upward(
    z_curr_world: float,
    upward_m: float,
    base_link_z: float,
    *,
    q4: float = 0.0,
    z_boundary: float = REVERSE_UPWARD_Z_PHASE_BOUNDARY_M,
    dz_step_m: float = REVERSE_UPWARD_Z_CM_STEP_M,
    theta_z_deg: float = 90.0,
    geom: WaistTriangleGeom = WaistTriangleGeom(),
    use_cache: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """由 upward（相对直立）规划到最近 1cm 目标 z 的关节轨迹。

    z≥0.73m 走 phase1 正弦流形；z<0.73m 走 phase2（仅 θz≈90°）。
    """
    lut = get_reverse_upward_combined_lut(
        base_link_z,
        q4=q4,
        z_boundary=z_boundary,
        dz_step_m=dz_step_m,
        theta_z_deg=theta_z_deg,
        geom=geom,
        use_cache=use_cache,
    )
    meta: Dict[str, Any] = {
        "ok": False,
        "model": lut.get("model"),
        "z_boundary_m": lut.get("z_boundary_m"),
        "z_upright_m": lut.get("z_upright_m"),
        "upward_cmd_m": round(float(upward_m), 4),
        "z_curr_m": round(float(z_curr_world), 4),
        "cached_lut": bool(use_cache and lut.get("cache_key")),
    }
    if not lut.get("ok"):
        meta["error"] = lut.get("error", "查表失败")
        return [], meta

    z_up = float(lut["z_upright_m"])
    z_min = float(lut["z_min_m"])
    z_max = float(lut["z_max_m"])
    z_tgt_raw = z_up + float(upward_m)
    z_tgt = snap_z_to_nearest_cm(z_tgt_raw, dz_step_m)
    clamped = False
    if z_tgt < z_min - 1e-4:
        z_tgt = round(z_min, 4)
        clamped = True
    if z_tgt > z_max + 1e-4:
        z_tgt = round(z_max, 4)
        clamped = True

    rows = lut["rows"]
    i_start = _nearest_lut_row_index(rows, float(z_curr_world))
    i_end = _nearest_lut_row_index(rows, z_tgt)
    z_start_row = float(rows[i_start]["z_target_m"])
    z_end_row = float(rows[i_end]["z_target_m"])

    waypoints: List[np.ndarray] = []
    phases_used: List[int] = []
    if i_start == i_end:
        waypoints.append(_lut_row_trunk_q(rows[i_end], q4))
        phases_used.append(int(rows[i_end]["phase"]))
    elif i_start < i_end:
        for i in range(i_start + 1, i_end + 1):
            waypoints.append(_lut_row_trunk_q(rows[i], q4))
            phases_used.append(int(rows[i]["phase"]))
    else:
        for i in range(i_start - 1, i_end - 1, -1):
            waypoints.append(_lut_row_trunk_q(rows[i], q4))
            phases_used.append(int(rows[i]["phase"]))

    meta.update({
        "ok": bool(waypoints),
        "z_tgt_raw_m": round(z_tgt_raw, 4),
        "z_tgt_m": round(z_tgt, 4),
        "upward_snapped_m": round(z_tgt - z_up, 4),
        "z_clamped": clamped,
        "i_start": i_start,
        "i_end": i_end,
        "z_start_row_m": round(z_start_row, 4),
        "z_end_row_m": round(z_end_row, 4),
        "n_lut_rows": len(rows),
        "n_waypoints": len(waypoints),
        "phases_used": phases_used,
        "phase1_rows": lut.get("n_phase1"),
        "phase2_rows": lut.get("n_phase2"),
        "z_min_m": round(z_min, 4),
        "z_max_m": round(z_max, 4),
        "phase2_limited_by": lut.get("phase2_limited_by"),
        "direction": "down" if z_tgt < float(z_curr_world) - 1e-4 else (
            "up" if z_tgt > float(z_curr_world) + 1e-4 else "hold"
        ),
    })
    if not waypoints:
        if abs(z_tgt - float(z_curr_world)) <= float(dz_step_m) * 0.6:
            meta["ok"] = True
            meta["direction"] = "hold"
            meta["note"] = "已在目标高度附近，无需移动"
        else:
            meta["error"] = "无路点（起止索引相同且高度偏差过大）"
        return [], meta

    if abs(z_tgt_raw - z_tgt) > float(dz_step_m) * 0.51:
        meta["snap_note"] = (
            f"upward={float(upward_m):+.3f}m → z_raw={z_tgt_raw:.3f}m "
            f"对齐 1cm 格点 z={z_tgt:.3f}m"
        )
    if clamped:
        meta["clamp_note"] = (
            f"目标 z={z_tgt_raw:.3f}m 超出可行范围 "
            f"[{z_min:.3f}, {z_max:.3f}]，已钳制到 z={z_tgt:.3f}m"
        )
    meta["q_end"] = [round(float(x), 4) for x in waypoints[-1][:3]]
    meta["q_start"] = [
        round(float(x), 4)
        for x in _lut_row_trunk_q(rows[i_start], q4)[:4]
    ]
    meta["ok"] = True
    return waypoints, meta


def clear_reverse_upward_lut_cache() -> int:
    n = len(_REVERSE_UPWARD_LUT_CACHE)
    _REVERSE_UPWARD_LUT_CACHE.clear()
    return n
