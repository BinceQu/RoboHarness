"""以物体为中心的可达球 + 肩宽弦 —— 底盘 (x, y, theta_x) 规划。

用户几何（严格）：
  1. 以物体中心 O 为球心，半径 R = 肩→抓取工作区（统一 0.60m，无余量扣减）；
  2. 肩宽为球的一根弦（左右肩在球面上），弦中点到 O 的距离为 sqrt(R² − (肩宽/2)²)；
  3. 弦可在水平面内平移、绕竖直轴旋转（theta_x）；取弦中点沿「物体→机器人」方向落在球内，
     并令机器人朝向物体，尽量保证 head 能看到物体；
  4. 输出仅底盘 (x, y, theta_x_deg)，不含俯仰 trunk。
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# 与 grasp._suggest_base_pose 一致：肩→抓取工作区 / 弦球 / 等腰三角形统一半径
SHOULDER_TO_GRASP_WORK_M = 0.60
REACH_SPHERE_R_M = SHOULDER_TO_GRASP_WORK_M
# 已废弃扣减余量，保留字段名仅兼容旧日志（恒为 0）
REACH_SPHERE_MARGIN_M = 0.0
WRIST_TO_SHOULDER_M = SHOULDER_TO_GRASP_WORK_M
SINGULARITY_MARGIN_M = REACH_SPHERE_MARGIN_M
# 左右肩半宽（chest 局部 y，|±0.171|）
SHOULDER_HALF_WIDTH_M = 0.171
# base_link → 胸口投影前方偏移（与肩点估算一致）
CHEST_FORWARD_FROM_BASE_M = 0.079


def reach_sphere_radius() -> float:
    """弦球与等腰三角形 h 的统一半径（默认 0.60m，可用 REACH_SPHERE_R_M 覆盖）。"""
    return float(os.environ.get("REACH_SPHERE_R_M", str(REACH_SPHERE_R_M)))


def shoulder_span_m() -> float:
    return 2.0 * SHOULDER_HALF_WIDTH_M


def _unit2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def shoulder_positions_at_base(
    bx: float, by: float, yaw_rad: float, shoulder_z: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """给定底盘 (bx,by,yaw) 估算左右肩世界坐标与弦中点。"""
    lateral = np.array([-math.sin(yaw_rad), math.cos(yaw_rad)], dtype=np.float64)
    mid = np.array([bx, by, shoulder_z], dtype=np.float64)
    off = SHOULDER_HALF_WIDTH_M * lateral
    left = np.array([mid[0] + off[0], mid[1] + off[1], shoulder_z], dtype=np.float64)
    right = np.array([mid[0] - off[0], mid[1] - off[1], shoulder_z], dtype=np.float64)
    return left, right, mid


def chord_mid_horizontal_dist(R: float, half_chord: float, dz: float) -> Optional[float]:
    """弦中点相对 O 的水平距离：|S_mid-O|_3d² = R² − (肩宽/2)²，再减去 dz² 得水平分量。"""
    h3_sq = R * R - half_chord * half_chord
    if h3_sq < 0:
        return None
    horiz_sq = h3_sq - dz * dz
    if horiz_sq < 1e-6:
        return None
    return math.sqrt(horiz_sq)


def verify_chord_on_sphere(
    O: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    R: float,
    tol: float = 0.04,
) -> Tuple[bool, Dict[str, float]]:
    d_l = float(np.linalg.norm(left - O))
    d_r = float(np.linalg.norm(right - O))
    ok = abs(d_l - R) <= tol and abs(d_r - R) <= tol
    return ok, {"left_dist": d_l, "right_dist": d_r, "target_R": R}


def _pose_nav_clearance(
    nav_clearance_fn: Optional[Callable[[float, float], Tuple[bool, float, Optional[str]]]],
    is_free_xy,
    bx: float,
    by: float,
    smx: float,
    smy: float,
    left: np.ndarray,
    right: np.ndarray,
) -> Dict[str, Any]:
    """底盘+双肩采样点的最差 clearance（越小越危险）。"""
    sample = [
        (bx, by, "base"),
        (smx, smy, "shoulder_mid"),
        (float(left[0]), float(left[1]), "left_shoulder"),
        (float(right[0]), float(right[1]), "right_shoulder"),
    ]
    min_clr = 999.0
    all_free = True
    blockers: List[str] = []
    per_pt: Dict[str, float] = {}
    for px, py, tag in sample:
        if nav_clearance_fn is not None:
            free, clr, blk = nav_clearance_fn(px, py)
        elif is_free_xy is not None:
            free = bool(is_free_xy(px, py))
            clr = 0.05 if free else -0.05
            blk = None if free else "blocked"
        else:
            free, clr, blk = True, 999.0, None
        per_pt[tag] = round(float(clr), 4)
        min_clr = min(min_clr, float(clr))
        if not free:
            all_free = False
            if blk:
                blockers.append(str(blk))
    return {
        "min_clearance_m": round(min_clr, 4),
        "all_free": all_free,
        "blockers": list(dict.fromkeys(blockers)),
        "per_point": per_pt,
    }


def plan_base_chord_pose(
    object_center: np.ndarray,
    robot_xy: np.ndarray,
    *,
    shoulder_z: float,
    is_free_xy=None,
    nav_clearance_fn: Optional[
        Callable[[float, float], Tuple[bool, float, Optional[str]]]
    ] = None,
    approach_hint: Optional[np.ndarray] = None,
    reach_R: Optional[float] = None,
) -> Dict[str, Any]:
    """规划底盘目标：肩宽弦贴在以物体为中心的可达球上。

    Args:
        object_center: 物体中心世界坐标 (3,)
        robot_xy: 当前底盘 xy
        shoulder_z: 规划用肩高度（世界 z，通常 chest_z+0.303，不发给 move_to）
        is_free_xy: 可选 (x,y)->bool，free_region 检测
        approach_hint: 可选 xy 单位向量，弦中点相对 O 的「物体→机器人」方向
    """
    O = np.asarray(object_center, dtype=np.float64).reshape(3)
    robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    R = float(reach_R) if reach_R is not None else reach_sphere_radius()
    half_w = SHOULDER_HALF_WIDTH_M
    dz = float(shoulder_z - O[2])
    h_horiz = chord_mid_horizontal_dist(R, half_w, dz)
    if h_horiz is None:
        return {
            "ok": False,
            "error": (
                f"肩高 z={shoulder_z:.2f} 相对物体 z={O[2]:.2f} 使可达球半径 R={R:.2f}m "
                f"无法容纳肩宽弦 {shoulder_span_m():.2f}m"
            ),
        }
    h_mid_3d = math.sqrt(h_horiz * h_horiz + dz * dz)

    if approach_hint is not None:
        approach = _unit2(np.asarray(approach_hint, dtype=np.float64).reshape(2)[:2])
    else:
        approach = _unit2(robot_xy - O[:2])

    def _scan_candidates() -> Optional[Dict[str, Any]]:
        """13 个弦角候选：chord 合法前提下，优先选离当前机器人最近的弦位。"""
        local_best: Optional[Dict[str, Any]] = None
        local_key: Optional[Tuple[float, int, float, float]] = None
        for off_deg in (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120):
            ang = math.radians(off_deg)
            c, s = math.cos(ang), math.sin(ang)
            app = np.array([c * approach[0] - s * approach[1],
                            s * approach[0] + c * approach[1]], dtype=np.float64)
            app = _unit2(app)
            S_mid_xy = O[:2] + h_horiz * app
            yaw_rad = math.atan2(
                float(O[1] - S_mid_xy[1]), float(O[0] - S_mid_xy[0]),
            )
            smx, smy = float(S_mid_xy[0]), float(S_mid_xy[1])
            bx = smx - CHEST_FORWARD_FROM_BASE_M * math.cos(yaw_rad)
            by = smy - CHEST_FORWARD_FROM_BASE_M * math.sin(yaw_rad)
            left, right, mid = shoulder_positions_at_base(smx, smy, yaw_rad, shoulder_z)
            chord_ok, dists = verify_chord_on_sphere(O, left, right, R)
            if not chord_ok:
                continue
            nav = _pose_nav_clearance(
                nav_clearance_fn, is_free_xy,
                bx, by, smx, smy, left, right,
            )
            travel = float(np.linalg.norm(np.array([bx, by]) - robot_xy))
            chord_err = (
                abs(dists["left_dist"] - R) + abs(dists["right_dist"] - R)
            )
            min_clr = float(nav["min_clearance_m"])
            base_dist_o = float(np.linalg.norm(np.array([bx, by]) - O[:2]))
            # Selection policy: stay on the nearest reachable chord slice.
            # Clearance is reported for diagnostics and used only as a tiny
            # tie-break after distance, so it cannot pull the robot around to
            # the far side of the object.
            key = (
                round(travel, 6),
                0 if nav["all_free"] else 1,
                abs(float(off_deg)) * 0.001 + chord_err * 0.05,
                -min_clr,
            )
            if local_best is None or key < local_key:
                local_key = key
                local_best = {
                    "bx": bx, "by": by,
                    "theta_x_deg": math.degrees(yaw_rad),
                    "shoulder_mid": [float(mid[0]), float(mid[1]), float(mid[2])],
                    "left_shoulder": [float(left[0]), float(left[1]), float(left[2])],
                    "right_shoulder": [float(right[0]), float(right[1]), float(right[2])],
                    "approach_xy": [float(app[0]), float(app[1])],
                    "chord_ok": chord_ok,
                    "chord_dist": dists,
                    "travel_m": travel,
                    "base_dist_to_object_m": round(base_dist_o, 4),
                    "yaw_offset_deg": off_deg,
                    "nav_clearance": nav,
                    "pick_score": round(-float(travel), 4),
                    "pick_policy": "nearest_chord_travel_first",
                }
        return local_best

    best = _scan_candidates()
    if best is None:
        return {
            "ok": False,
            "error": "13 个弦角候选均无合法肩宽弦，或避障信息不可用",
        }
    return {
        "ok": True,
        "base_xy": [best["bx"], best["by"]],
        "theta_x_deg": round(best["theta_x_deg"], 2),
        "reach_sphere_R_m": round(R, 4),
        "shoulder_to_grasp_work_m": SHOULDER_TO_GRASP_WORK_M,
        "reach_sphere_margin_m": REACH_SPHERE_MARGIN_M,
        "wrist_to_shoulder_m": SHOULDER_TO_GRASP_WORK_M,
        "singularity_margin_m": REACH_SPHERE_MARGIN_M,
        "shoulder_span_m": shoulder_span_m(),
        "chord_mid_to_object_3d_m": round(h_mid_3d, 4),
        "chord_mid_to_object_horiz_m": round(h_horiz, 4),
        "shoulder_z_plan": round(shoulder_z, 4),
        "object_center": [float(O[0]), float(O[1]), float(O[2])],
        "plan_detail": best,
    }
