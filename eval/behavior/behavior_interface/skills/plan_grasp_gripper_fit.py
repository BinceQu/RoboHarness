"""由 VLM 3D 粗定位 + 物体点云，搜索 EEF pose（严格无碰撞，最大化开口内物体体积）。"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── R1Pro 夹爪几何：与 viz_eef_v2 OBJ 同一套 link 变换 + 三角柱楔形开口 ──
from behavior_interface.skills.plan_grasp_gripper_geom import (
    finger_inner_y_nominal,
    finger_z_tip,
    gap_center_local,
    gap_rail_metrics as _geom_gap_rail_metrics,
    gap_wedge_mask,
    gap_z_bounds,
    get_wedge_lut,
    hit_to_slider_rail_mm as _geom_hit_to_slider_rail_mm,
)

_lut_boot = get_wedge_lut()
GRIP_GAP_CENTER_LOCAL = gap_center_local()
GRIP_GAP_HALF = 0.042
VIZ_BALL_RADIUS = 0.022  # 球心在表面交点，半径使 ball 一半在内一半在外

# 掌盒：真机 z∈[-0.137,-0.06] y±0.054 x±0.03（仅掌部碰撞，指部走 OBJ 楔形）
_PALM_BOX = (-0.057, 0.057, -0.140, -0.058, 0.033)
MIN_GAP_POINTS = 4
MIN_GAP_EACH_SIDE = 2
# 指轨内侧面 Y（OBJ 夹持 z 带处内缘，约 0.46mm）
FINGER_INNER_Y = finger_inner_y_nominal()
_GAP_Z_TIP, _GAP_Z_OPEN = gap_z_bounds()
# 兼容旧引用：轴对齐盒由楔形 z/y/x 外包（审计字段用）
_GAP_BOX = (
    -FINGER_INNER_Y,
    FINGER_INNER_Y,
    _GAP_Z_TIP,
    _GAP_Z_OPEN,
    float(_lut_boot["x_half_gap"].max()),
)
# 指盒仅作 reach 长度回退；碰撞/体积已改用 OBJ
_FINGER_BOXES = (
    (FINGER_INNER_Y, 0.072, _GAP_Z_TIP, _GAP_Z_OPEN, 0.024),
    (-0.072, -FINGER_INNER_Y, _GAP_Z_TIP, _GAP_Z_OPEN, 0.024),
)
# VLM 3D 点仅作粗定位；锚点搜索半径随物体尺度自适应
HIT_SEARCH_RADIUS_MIN_M = 0.05
HIT_SEARCH_RADIUS_MAX_M = 0.14
# 沿 approach / 侧向微调
_OFFSET_STEPS_M = (
    0.0, -0.003, -0.006, -0.010, -0.015, -0.020, -0.025, -0.030, -0.035,
    0.003, 0.006, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050,
)
# EEF 局部 Y 侧移（两指连线方向）
_LATERAL_Y_STEPS_M = (-0.015, -0.008, 0.0, 0.008, 0.015)
# 粗搜 / 精搜
_COARSE_OFFSET_STEPS_M = (0.0, -0.010, -0.020, 0.010, 0.020, 0.030, 0.040)
_COARSE_PCD_MAX = 2500
_FINE_TOP_K = 8
_ROLLS_COARSE = 12
_ROLLS_FINE = 24
_FINE_ROLL_TOP = 5  # 每个 approach 精搜位置前先筛 top-N 组 (roll, ref_y)
# 开口邻域：只统计距开口中心此半径内的点（避免整物体投影进 EEF 框虚高）
_GAP_PROXIMITY_R = 0.045
_SCORE_PCD_MAX = 8000
# EEF 系固定体素边长：同一体素内多点只计一次，比裸 n_gap 更接近真实占据体积
_GAP_VOXEL_M = 0.008
MIN_GAP_VOXELS = 2


def gripper_reach_length_m() -> float:
    """
    R1Pro 夹爪有效长度：EEF 原点到物体侧指尖（OBJ mesh −Z 最远点）。
    用于判定 EEF 是否足够靠近物体表面，避免「夹空」pose。
    """
    return float(abs(finger_z_tip()))


def gripper_half_length_m() -> float:
    """夹爪长度的一半（米）。"""
    return gripper_reach_length_m() * 0.5


def _gap_voxel_metrics(q_gap: np.ndarray) -> Tuple[int, float, int]:
    """
    开口内占据体积（体素法）。
    depth 点云在世界系密度随距离变化，但在开口小盒内用固定边长体素可削弱
    「同表面多点重复计数」偏差，使不同位姿间体积可比。
    """
    if len(q_gap) == 0:
        return 0, 0.0, 0
    vox = np.floor(q_gap / _GAP_VOXEL_M).astype(np.int64)
    n_vox = int(np.unique(vox, axis=0).shape[0])
    vol_cm3 = float(n_vox * (_GAP_VOXEL_M ** 3) * 1e6)
    return n_vox, vol_cm3, int(len(q_gap))


def _hit_to_slider_rail_mm(q_hit: np.ndarray) -> float:
    """VLM/点击 3D 点到夹爪滑轨（OBJ 内侧面）的贴近度，单位 mm。"""
    q = np.asarray(q_hit, dtype=np.float64).reshape(3)
    z0, z1 = float(_GAP_Z_TIP), float(_GAP_Z_OPEN)
    if z0 <= float(q[2]) <= z1:
        return _geom_hit_to_slider_rail_mm(q)
    y_cl = float(np.clip(q[1], -FINGER_INNER_Y, FINGER_INNER_Y))
    z_cl = float(np.clip(q[2], z0, z1))
    return float(np.linalg.norm(q - np.array([q[0], y_cl, z_cl])) * 1000.0)


def _pose_sort_key(c: Optional[Dict[str, Any]]) -> tuple:
    """
    优化优先级：
      1. 开口占据体积（体素数）
      2. VLM 3D 点贴近滑轨
      3. 物体相对两指居中
      4. 无碰撞（ncol，靠 2/3 位姿调整消解，排序末位）
    """
    if c is None:
        return (1, 1, 0, 0.0, 999.0, 999.0, 1.0, 999, 999.0)
    return (
        0 if c.get("has_volume") else 1,
        0 if c.get("ncol", 0) == 0 else 1,   # 优先无穿模（几何已按真机标定，ncol 可信）
        -c.get("gap_voxel_n", 0),
        -c.get("gap_vol_cm3", 0.0),
        c.get("hit_slider_mm", 999.0),
        c.get("center_bias_mm", 999.0),
        c.get("y_balance", 1.0),
        c.get("ncol", 0),
        c.get("gap_err_mm", 999.0),
    )


def _pick_best(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """仅在 has_volume 候选中选最优；无有效体积则返回 None。"""
    if not candidates:
        return None
    with_vol = [c for c in candidates if c.get("has_volume")]
    if not with_vol:
        return None
    return min(with_vol, key=_pose_sort_key)


def grip_fit_has_volume(grip_fit: Optional[Dict[str, Any]]) -> bool:
    """统一判定开口是否具备有效占据体积（禁止 fallback 或无字段时误判为有效）。"""
    if not grip_fit:
        return False
    if grip_fit.get("approach_label") == "fallback_top_down":
        return False
    return bool(
        grip_fit.get("has_volume")
        and int(grip_fit.get("gap_voxel_n", 0)) >= MIN_GAP_VOXELS
        and float(grip_fit.get("gap_vol_cm3", 0.0)) > 0.0
    )


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        return np.array([
            (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s, 0.25 * s,
        ], dtype=np.float64)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        return np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    if m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        return np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                         (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
    return np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                     0.25 * s, (m[1, 0] - m[0, 1]) / s], dtype=np.float64)


def _eef_frame_from_approach_roll(approach: np.ndarray, roll: float,
                                   ref_y: Optional[np.ndarray] = None) -> np.ndarray:
    z = approach / (np.linalg.norm(approach) + 1e-9)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if ref_y is not None:
        y0 = ref_y - np.dot(ref_y, z) * z
    else:
        y0 = np.cross(z, up)
        if np.linalg.norm(y0) < 1e-3:
            y0 = np.cross(z, np.array([1.0, 0.0, 0.0]))
    y0 /= np.linalg.norm(y0) + 1e-9
    x0 = np.cross(y0, z)
    cr, sr = math.cos(roll), math.sin(roll)
    x = cr * x0 + sr * y0
    y = -sr * x0 + cr * y0
    return np.column_stack([x, y, z])


def _in_box(q: np.ndarray, ylo, yhi, zlo, zhi, xhalf) -> np.ndarray:
    return (
        (q[:, 0] >= -xhalf) & (q[:, 0] <= xhalf)
        & (q[:, 1] >= ylo) & (q[:, 1] <= yhi)
        & (q[:, 2] >= zlo) & (q[:, 2] <= zhi)
    )


def _solid_mask(q: np.ndarray) -> np.ndarray:
    """夹爪 solid 区域（仅真实碰撞，不膨胀）。"""
    hit = np.zeros(len(q), dtype=bool)
    for y0, y1, z0, z1, xh in _FINGER_BOXES:
        hit |= _in_box(q, y0, y1, z0, z1, xh)
    py0, py1, pz0, pz1, pxh = _PALM_BOX
    hit |= _in_box(q, py0, py1, pz0, pz1, pxh)
    return hit


def _gap_rail_metrics(q_gap: np.ndarray) -> Dict[str, float]:
    """开口内物体相对两指 OBJ 内侧面轨道的距离指标。"""
    return _geom_gap_rail_metrics(q_gap)


def _gap_mask(q: np.ndarray, *, for_volume: bool = False) -> np.ndarray:
    """爪间三角柱楔形开口（与红夹爪 OBJ 一致）。"""
    return gap_wedge_mask(q, for_volume=for_volume)


def _map_uv_to_depth(
    u: int,
    v: int,
    w: int,
    h: int,
    depth: np.ndarray,
) -> Tuple[int, int, int, int]:
    """将点击像素映射到 depth 分辨率；返回 (u_d, v_d, w_use, h_use)。"""
    dh, dw = int(depth.shape[0]), int(depth.shape[1])
    if dw == w and dh == h:
        return int(u), int(v), w, h
    u_d = int(round(np.clip(u * dw / max(w, 1), 0, dw - 1)))
    v_d = int(round(np.clip(v * dh / max(h, 1), 0, dh - 1)))
    return u_d, v_d, dw, dh


def _depth_backproject_uv(
    depth: np.ndarray,
    u: int,
    v: int,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
) -> Optional[np.ndarray]:
    """深度图反投影到世界坐标；点在点击射线上的深度交点，重投影与 (u,v) 一致。"""
    u_d, v_d, w_use, h_use = _map_uv_to_depth(u, v, w, h, depth)
    if v_d < 0 or v_d >= depth.shape[0] or u_d < 0 or u_d >= depth.shape[1]:
        return None
    d_val = float(depth[int(v_d), int(u_d)])
    if not (np.isfinite(d_val) and 0.05 < d_val < 50.0):
        return None
    fx = fl / ha * w_use
    cx, cy = w_use / 2.0, h_use / 2.0
    xc = (u_d - cx) / fx * d_val
    yc = -(v_d - cy) / fx * d_val
    zc = -d_val
    R_cam = _quat_to_mat(cam_quat)
    return cam_pos + R_cam @ np.array([xc, yc, zc], dtype=np.float64)


def _reproj_error_px(
    hit: np.ndarray,
    u: int,
    v: int,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
) -> Optional[float]:
    from behavior_interface.skills.vlm_grasp_verify import _world_to_pixel

    px = _world_to_pixel(cam_pos, cam_quat, hit, w, h, fl, ha)
    if px is None:
        return None
    return float(np.hypot(px[0] - u, px[1] - v))


def resolve_hit_on_surface(
    u: int,
    v: int,
    depth: np.ndarray,
    pcd: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    gta=None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    点击像素 (u,v) 的 3D 表面点：优先 depth 反投影（保证在点击射线上），
    点云射线求交仅作 fallback，且返回射线上的点而非偏离射线的点云顶点。
    """
    from behavior_interface.skills.vlm_grasp_verify import (
        _depth_to_world_point,
        _pixel_to_world_ray,
    )
    from behavior_interface.skills.vlm_lawn_dual import _ray_pcd_hit

    # 1) 冻结 depth 在点击像素反投影 — 与针孔模型一致，重投影误差≈0
    p_depth = _depth_backproject_uv(depth, u, v, cam_pos, cam_quat, w, h, fl, ha)
    if p_depth is not None:
        return p_depth, "depth_click_ray"

    # 2) 仿真实时 depth（capture 后 head 已动时）
    if gta is not None:
        p = _depth_to_world_point(gta, u, v, cam_pos, cam_quat, w, h, fl, ha)
        if p is not None:
            return np.asarray(p, dtype=np.float64), "depth_live"

    origin, ray_dir = _pixel_to_world_ray(
        cam_pos, cam_quat, u, v, w, h, fl, ha,
    )

    # 3) 点云射线求交（已在 _ray_pcd_hit 内投影到射线上）
    if len(pcd) >= 5:
        hp = _ray_pcd_hit(pcd, origin, ray_dir, max_perp=0.10)
        if hp is not None:
            err = _reproj_error_px(hp, u, v, cam_pos, cam_quat, w, h, fl, ha)
            if err is None or err <= 3.0:
                return hp.astype(np.float64), "pcd_ray_surface"
        hp = _ray_pcd_hit(pcd, origin, ray_dir, max_perp=0.25)
        if hp is not None:
            err = _reproj_error_px(hp, u, v, cam_pos, cam_quat, w, h, fl, ha)
            if err is None or err <= 5.0:
                return hp.astype(np.float64), "pcd_ray_surface_loose"

    if len(pcd) >= 1:
        hp = _ray_pcd_hit(pcd, origin, ray_dir, max_perp=9999.0)
        if hp is not None:
            return hp.astype(np.float64), "pcd_ray_fallback"

    return None, "failed"


def _fibonacci_sphere(n: int) -> np.ndarray:
    pts = []
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        th = ga * i
        pts.append([r * math.cos(th), r * math.sin(th), z])
    return np.asarray(pts, dtype=np.float64)


def _local_patch(pcd: np.ndarray, hit: np.ndarray, radius: float = 0.06) -> np.ndarray:
    if len(pcd) == 0:
        return pcd
    d = np.linalg.norm(pcd - hit, axis=1)
    patch = pcd[d < radius]
    if len(patch) < 20:
        patch = pcd[d < radius * 2.0]
    return patch


def _pca_axes(patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (短轴/法向, 长轴) 及中间轴。"""
    if len(patch) < 5:
        z = np.array([0.0, 0.0, 1.0])
        x = np.array([1.0, 0.0, 0.0])
        return z, x, np.array([0.0, 1.0, 0.0])
    c = patch.mean(axis=0)
    cov = np.cov((patch - c).T)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)
    return v[:, order[0]], v[:, order[2]], v[:, order[1]]


def _ref_y_candidates(t_long: np.ndarray, t_short: np.ndarray) -> List[Tuple[str, Optional[np.ndarray]]]:
    """绕 approach 旋转时的参考 Y：长轴 / 短轴 / 自由。"""
    return [
        ("long", t_long),
        ("short", t_short),
        ("free", None),
    ]


def _rank_key(ev: Optional[Dict[str, Any]]) -> tuple:
    return _pose_sort_key(ev)


def _top_approaches_from_coarse(
    coarse_rows: List[Tuple[np.ndarray, float, str, Optional[Dict[str, Any]]]],
    top_k: int,
) -> List[Tuple[np.ndarray, str]]:
    """每个 approach 方向取粗搜最优，再选 top-K 方向（不锁 roll）。"""
    best: Dict[str, Tuple[tuple, np.ndarray, str]] = {}
    for approach, _roll, alabel, ev in coarse_rows:
        if ev is None:
            continue
        rk = _rank_key(ev)
        if alabel not in best or rk < best[alabel][0]:
            best[alabel] = (rk, approach.copy(), alabel)
    ordered = sorted(best.values(), key=lambda x: x[0])
    return [(ap, lb) for _rk, ap, lb in ordered[:top_k]]


def _fine_search_for_approach(
    approach: np.ndarray,
    alabel: str,
    hit: np.ndarray,
    score_pcd: np.ndarray,
    rolls_fine: np.ndarray,
    ref_ys: List[Tuple[str, Optional[np.ndarray]]],
) -> List[Dict[str, Any]]:
    """
    精搜单个 approach：先快速扫 roll/ref_y（固定质心锚点），再对 top 组合做全位置搜索。
    """
    anchor = score_pcd.mean(axis=0)
    roll_rows: List[Tuple[str, Optional[np.ndarray], float, Dict[str, Any]]] = []
    for ry_label, ref_y in ref_ys:
        for roll in rolls_fine:
            R = _eef_frame_from_approach_roll(approach, float(roll), ref_y=ref_y)
            eef_probe = anchor - R @ GRIP_GAP_CENTER_LOCAL
            ev = _evaluate_pose(R, eef_probe, score_pcd, hit)
            roll_rows.append((ry_label, ref_y, float(roll), ev))
    roll_rows.sort(key=lambda row: _rank_key(row[3]))

    out: List[Dict[str, Any]] = []
    seen_roll: set = set()
    for ry_label, ref_y, roll, _ in roll_rows:
        key = (ry_label, round(math.degrees(roll)))
        if key in seen_roll:
            continue
        seen_roll.add(key)
        R = _eef_frame_from_approach_roll(approach, roll, ref_y=ref_y)
        ev = _search_best_eef_shift(R, hit, score_pcd)
        if ev is None:
            continue
        out.append({
            "pos": ev["pos"],
            "quat": _mat_to_quat_xyzw(R),
            "approach": R[:, 2].copy(),
            "roll_deg": math.degrees(roll),
            "approach_label": alabel,
            "ref_y_label": ry_label,
            "offset_m": ev["offset_m"],
            "lateral_y_m": ev.get("lateral_y_m", 0.0),
            "anchor_dist_mm": ev.get("anchor_dist_mm", 0.0),
            "n_gap": ev["n_gap"],
            "n_pos": ev["n_pos"],
            "n_neg": ev["n_neg"],
            "has_both": ev["has_both"],
            "ncol": ev["ncol"],
            "gap_err_mm": ev["gap_err_mm"],
            "bulk_dist_mm": ev.get("bulk_dist_mm", ev["gap_err_mm"]),
            "y_balance": ev["y_balance"],
            "center_bias_mm": ev["center_bias_mm"],
            "min_rail_clear_mm": ev["min_rail_clear_mm"],
            "gap_voxel_n": ev.get("gap_voxel_n", 0),
            "gap_vol_cm3": ev.get("gap_vol_cm3", 0.0),
            "has_volume": ev.get("has_volume", False),
            "hit_slider_mm": ev.get("hit_slider_mm", 999.0),
            "feasible": ev["feasible"],
        })
        if len(seen_roll) >= _FINE_ROLL_TOP:
            break
    return out


def extract_seg_object_mask(
    seg: np.ndarray,
    u: int,
    v: int,
    win: int = 80,
    *,
    tight: bool = False,
) -> Tuple[Optional[np.ndarray], List[int]]:
    """seg 物体掩码。tight=True：仅点击像素 instance（grasp_object 用，避免吞台面）。"""
    if seg.ndim == 3:
        seg = seg[..., 0]
    seg = seg.astype(np.int32)
    h, w = seg.shape
    u_c = int(np.clip(u, 0, w - 1))
    v_c = int(np.clip(v, 0, h - 1))
    if tight:
        primary = int(seg[v_c, u_c])
        if primary <= 0:
            pt_crop = seg[
                max(0, v_c - win):min(h, v_c + win),
                max(0, u_c - win):min(w, u_c + win),
            ]
            ids, counts = np.unique(pt_crop, return_counts=True)
            nz = ids != 0
            if not nz.any():
                return None, []
            primary = int(ids[nz][np.argmax(counts[nz])])
        # 点击邻域内与 primary 同 ID 的连通块（多 link 同物体时放宽）
        pt_crop = seg[
            max(0, v_c - win):min(h, v_c + win),
            max(0, u_c - win):min(w, u_c + win),
        ]
        n_primary = int((pt_crop == primary).sum())
        obj_ids = [primary]
        if n_primary > 0:
            ids, counts = np.unique(pt_crop, return_counts=True)
            for oid, cnt in zip(ids, counts):
                oid = int(oid)
                if oid <= 0 or oid == primary:
                    continue
                if cnt >= max(12, int(0.35 * n_primary)):
                    obj_ids.append(oid)
        obj_ids = sorted(set(obj_ids))
        return np.isin(seg, obj_ids), obj_ids

    cy, cx = h // 2, w // 2
    ph, pw = h // 3, w // 3
    center_crop = seg[max(0, cy - ph):cy + ph, max(0, cx - pw):cx + pw]
    pt_crop = seg[
        max(0, v_c - win):min(h, v_c + win),
        max(0, u_c - win):min(w, u_c + win),
    ]
    combined = np.concatenate([center_crop.ravel(), pt_crop.ravel()])
    ids, counts = np.unique(combined, return_counts=True)
    nz = ids != 0
    if not nz.any():
        return None, []
    ids_nz, counts_nz = ids[nz], counts[nz]
    thr = max(float(counts_nz.max()) * 0.05, 10.0)
    obj_ids = [int(i) for i in ids_nz[counts_nz >= thr]]
    return np.isin(seg, obj_ids), obj_ids


def build_object_pointcloud(
    depth: np.ndarray,
    seg: Optional[np.ndarray],
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    fl: float,
    ha: float,
    u: int,
    v: int,
    hit_ref: Optional[np.ndarray] = None,
    z_band: Optional[Tuple[float, float]] = None,
    *,
    tight_seg: bool = False,
    max_radius_from_ref: Optional[float] = None,
) -> Tuple[np.ndarray, List[int]]:
    from behavior_interface.skills.vlm_lawn_dual import _build_pointcloud

    obj_mask, obj_ids = (None, [])
    if seg is not None:
        obj_mask, obj_ids = extract_seg_object_mask(seg, u, v, tight=tight_seg)
    depth_use = depth.copy()
    if obj_mask is not None:
        depth_use[~obj_mask] = 0.0
    pts = _build_pointcloud(depth_use, cam_pos, cam_quat, fl, ha)
    if len(pts) == 0:
        return pts, obj_ids
    if hit_ref is not None:
        r = 0.15 if tight_seg else 0.35
        if max_radius_from_ref is not None:
            r = float(max_radius_from_ref)
        d = np.linalg.norm(pts - hit_ref, axis=1)
        pts = pts[d < r]
    if z_band is not None:
        lo, hi = z_band
        pts = pts[(pts[:, 2] > lo) & (pts[:, 2] < hi)]
    return pts, obj_ids


def _score_pointcloud(pcd: np.ndarray, seed: int = 0, max_pts: int = _SCORE_PCD_MAX) -> np.ndarray:
    """全物体点云（子采样）用于开口体积评分。"""
    if len(pcd) <= max_pts:
        return pcd
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pcd), max_pts, replace=False)
    return pcd[idx]


def _object_search_radius(hit: np.ndarray, pcd: np.ndarray) -> float:
    """根据 VLM 点附近物体尺度确定开口中心搜索半径。"""
    d = np.linalg.norm(pcd - hit, axis=1)
    near = d[d < 0.30]
    if len(near) < 12:
        return HIT_SEARCH_RADIUS_MIN_M
    ext = float(np.percentile(near, 88))
    return float(np.clip(ext, HIT_SEARCH_RADIUS_MIN_M, HIT_SEARCH_RADIUS_MAX_M))


def _gap_anchor_points(hit: np.ndarray, pcd: np.ndarray) -> np.ndarray:
    """在物体点云内搜索开口中心候选（含全物体质心，VLM 点仅作区域参考）。"""
    radius = _object_search_radius(hit, pcd)
    obj_centroid = pcd.mean(axis=0)
    d_hit = np.linalg.norm(pcd - hit, axis=1)
    near = pcd[d_hit <= radius]
    anchors: List[np.ndarray] = [hit.copy(), obj_centroid.copy()]

    if len(near) >= 8:
        local_c = near.mean(axis=0)
        if np.linalg.norm(local_c - obj_centroid) > 0.005:
            anchors.append(local_c)

        centered = near - local_c
        if len(centered) >= 5:
            cov = np.cov(centered.T)
            _, v = np.linalg.eigh(cov)
            t1, t2 = v[:, 1], v[:, 2]
            for scale in (0.015, 0.030, 0.050):
                for s1, s2 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    pt = local_c + (s1 * t1 + s2 * t2) * scale
                    if np.linalg.norm(pt - hit) <= max(radius, HIT_SEARCH_RADIUS_MAX_M):
                        anchors.append(pt)

    if len(pcd) > 20:
        step = max(len(pcd) // 10, 1)
        for i in range(0, len(pcd), step):
            pt = pcd[i]
            if np.linalg.norm(pt - hit) <= HIT_SEARCH_RADIUS_MAX_M:
                anchors.append(pt)

    unique: List[np.ndarray] = []
    for a in anchors:
        if all(np.linalg.norm(a - u) > 0.006 for u in unique):
            unique.append(a)
    return np.asarray(unique[:14], dtype=np.float64)


def _coarse_anchor_points(hit: np.ndarray, pcd: np.ndarray) -> np.ndarray:
    """粗搜：VLM 点 + 全物体质心 + 附近质心。"""
    anchors = [hit.copy(), pcd.mean(axis=0)]
    radius = _object_search_radius(hit, pcd)
    near = pcd[np.linalg.norm(pcd - hit, axis=1) <= radius]
    if len(near) >= 8:
        anchors.append(near.mean(axis=0))
    unique: List[np.ndarray] = []
    for a in anchors:
        if all(np.linalg.norm(a - u) > 0.005 for u in unique):
            unique.append(a)
    return np.asarray(unique, dtype=np.float64)


def _search_best_eef_shift(
    R: np.ndarray,
    hit: np.ndarray,
    score_pcd: np.ndarray,
    *,
    anchors: Optional[np.ndarray] = None,
    lateral_steps: Optional[Tuple[float, ...]] = None,
    offset_steps: Optional[Tuple[float, ...]] = None,
) -> Optional[Dict[str, Any]]:
    """搜索锚点/侧移/推进，最大化开口体素占据体积；无有效体积返回 None。"""
    approach_w = R[:, 2]
    lat_y = R[:, 1]
    use_anchors = anchors if anchors is not None else _gap_anchor_points(hit, score_pcd)
    use_lat = lateral_steps if lateral_steps is not None else _LATERAL_Y_STEPS_M
    use_off = offset_steps if offset_steps is not None else _OFFSET_STEPS_M
    cands: List[Dict[str, Any]] = []
    for anchor in use_anchors:
        base_eef = anchor - R @ GRIP_GAP_CENTER_LOCAL
        anchor_dist_mm = float(np.linalg.norm(anchor - hit) * 1000.0)
        for lat_y_m in use_lat:
            eef_lat = base_eef + float(lat_y_m) * lat_y
            for offset in use_off:
                eef_pos = eef_lat - float(offset) * approach_w
                ev = _evaluate_pose(R, eef_pos, score_pcd, hit)
                cands.append({
                    "pos": eef_pos,
                    "offset_m": float(offset),
                    "lateral_y_m": float(lat_y_m),
                    "anchor_dist_mm": anchor_dist_mm,
                    **ev,
                })
    return _pick_best(cands)


def _collision_mask(q: np.ndarray, in_gap: np.ndarray) -> np.ndarray:
    """
    碰撞判定：夹爪实体(指/掌)与物体点相交，但扣除落在开口 gap 内的点（那是要夹的）。
    几何已按真机标定，直接 solid & ~in_gap 即为穿模点（无需再按 ±Z 方向特判）。
    """
    return _solid_mask(q) & ~in_gap


def _evaluate_pose(
    R: np.ndarray,
    eef_pos: np.ndarray,
    patch: np.ndarray,
    hit: np.ndarray,
) -> Dict[str, Any]:
    """评估单个 EEF 姿态。"""
    gap_world = eef_pos + R @ GRIP_GAP_CENTER_LOCAL
    near = np.linalg.norm(patch - gap_world, axis=1) < _GAP_PROXIMITY_R
    q = (R.T @ (patch - eef_pos).T).T

    in_gap = _gap_mask(q) & near
    n_gap = int(in_gap.sum())
    gap_ys = q[in_gap, 1] if n_gap else np.array([])
    n_pos = int((gap_ys > 0.004).sum()) if len(gap_ys) else 0
    n_neg = int((gap_ys < -0.004).sum()) if len(gap_ys) else 0
    has_both = n_pos >= MIN_GAP_EACH_SIDE and n_neg >= MIN_GAP_EACH_SIDE

    ncol = int(_collision_mask(q, in_gap).sum())

    q_gap = q[in_gap]
    rail = _gap_rail_metrics(q_gap)
    gap_voxel_n, gap_vol_cm3, n_gap_raw = _gap_voxel_metrics(q_gap)

    q_hit = (R.T @ (hit - eef_pos).reshape(3)).reshape(3)
    hit_slider_mm = _hit_to_slider_rail_mm(q_hit)

    gap_err_mm = float(np.linalg.norm(gap_world - hit) * 1000.0)
    bulk_dist_mm = float(np.linalg.norm(gap_world - patch.mean(axis=0)) * 1000.0)

    has_volume = (
        gap_voxel_n >= MIN_GAP_VOXELS
        and n_gap >= MIN_GAP_POINTS
        and has_both
    )
    feasible = has_volume and ncol == 0
    return {
        "n_gap": n_gap,
        "n_gap_raw": n_gap_raw,
        "gap_voxel_n": gap_voxel_n,
        "gap_vol_cm3": gap_vol_cm3,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "has_both": has_both,
        "has_volume": has_volume,
        "ncol": ncol,
        "hit_slider_mm": hit_slider_mm,
        "gap_err_mm": gap_err_mm,
        "bulk_dist_mm": bulk_dist_mm,
        "feasible": feasible,
        **rail,
    }


def compute_eef_from_pcd_gripper_fit(
    hit_world: np.ndarray,
    pcd: np.ndarray,
    cam_pos: Optional[np.ndarray] = None,
    cam_quat: Optional[np.ndarray] = None,
    ctx=None,
) -> Optional[Dict[str, Any]]:
    """
    优化目标（按优先级）：
      1. 开口内两指中间占据体积最大（体素法）；无有效占据则 grasp 无效
      2. VLM/点击 3D 点尽量贴近夹爪滑轨（内侧面）
      3. 物体相对两指尽量居中
      4. 无碰撞 ncol=0（靠 2/3 位姿调整消解，排序末位 tie-break）
    体积度量：EEF 系 8mm 固定体素占据，削弱 depth 点云同面多点重复计数；
    远距表面世界系密度仍偏低，但开口盒内跨位姿可比性优于裸 n_gap。
    """
    hit = np.asarray(hit_world, dtype=np.float64).reshape(3)
    score_pcd = _score_pointcloud(np.asarray(pcd, dtype=np.float64))
    patch = _local_patch(score_pcd, hit, radius=0.07)
    if len(patch) < 8:
        patch = _local_patch(score_pcd, hit, radius=0.14)
    if len(patch) < 3:
        patch = score_pcd
    if len(score_pcd) < 3:
        if ctx:
            ctx.log("  [grip_fit] FAIL 点云过少(<3)，开口体积=0，grasp 无效")
        return None

    n_loc, t_long, t_short = _pca_axes(patch)
    if cam_pos is not None:
        to_cam = cam_pos - hit
        to_cam /= np.linalg.norm(to_cam) + 1e-9
        if np.dot(n_loc, to_cam) < 0:
            n_loc = -n_loc

    dirs: List[Tuple[np.ndarray, str]] = [
        (-n_loc, "pca_normal"),
        (np.array([0.0, 0.0, -1.0]), "top_down"),
        (t_long, "along_long"),
        (-t_long, "along_long_neg"),
    ]
    if cam_pos is not None:
        d = hit - cam_pos
        dn = float(np.linalg.norm(d))
        if dn > 1e-3:
            dirs.append((d / dn, "toward_cam"))
    for i, p in enumerate(_fibonacci_sphere(24)):
        dirs.append((p / (np.linalg.norm(p) + 1e-9), f"sphere_{i}"))

    rolls_coarse = np.linspace(0, 2 * math.pi, _ROLLS_COARSE, endpoint=False)
    rolls_fine = np.linspace(0, 2 * math.pi, _ROLLS_FINE, endpoint=False)
    ref_ys = _ref_y_candidates(t_long, t_short)
    coarse_pcd = _score_pointcloud(score_pcd, max_pts=_COARSE_PCD_MAX)

    # ── 粗搜：approach × roll（低分辨率位置）──
    coarse_rows: List[Tuple[np.ndarray, float, str, Dict[str, Any]]] = []
    coarse_anchors = _coarse_anchor_points(hit, score_pcd)
    for approach, alabel in dirs:
        approach = approach / (np.linalg.norm(approach) + 1e-9)
        for roll in rolls_coarse:
            for _ry_label, ref_y in ref_ys:
                R = _eef_frame_from_approach_roll(approach, float(roll), ref_y=ref_y)
                ev = _search_best_eef_shift(
                    R, hit, coarse_pcd,
                    anchors=coarse_anchors,
                    lateral_steps=(0.0,),
                    offset_steps=_COARSE_OFFSET_STEPS_M,
                )
                if ev is not None:
                    coarse_rows.append((approach, float(roll), alabel, ev))

    top_approaches = _top_approaches_from_coarse(coarse_rows, _FINE_TOP_K)
    if not top_approaches:
        # 粗搜无任何有效体积：退回 PCA/俯视等主方向做精搜，避免直接崩溃
        top_approaches = [(d / (np.linalg.norm(d) + 1e-9), lb) for d, lb in dirs[:4]]

    # ── 精搜：top-K approach × 全 roll 扫描 → top roll 做位置搜索 ──
    candidates: List[Dict[str, Any]] = []
    for approach, alabel in top_approaches:
        candidates.extend(
            _fine_search_for_approach(
                approach, alabel, hit, score_pcd, rolls_fine, ref_ys,
            )
        )

    if not candidates:
        if ctx:
            ctx.log("  [grip_fit] FAIL 搜索无候选，开口体积=0，grasp 无效")
        return None

    best = _pick_best(candidates)
    if best is None or not grip_fit_has_volume({
        "has_volume": best.get("has_volume"),
        "gap_voxel_n": best.get("gap_voxel_n", 0),
        "gap_vol_cm3": best.get("gap_vol_cm3", 0.0),
        "approach_label": best.get("approach_label", ""),
    }):
        if ctx:
            if best is None:
                ctx.log(
                    f"  [grip_fit] FAIL 全部候选开口体积=0 "
                    f"(需 vox>={MIN_GAP_VOXELS} 且两侧都有点)"
                )
            else:
                ctx.log(
                    f"  [grip_fit] FAIL 开口无有效占据体积 "
                    f"(vox={best.get('gap_voxel_n', 0)} "
                    f"vol={best.get('gap_vol_cm3', 0):.3f}cm³ "
                    f"pts={best.get('n_gap', 0)} ncol={best.get('ncol', 0)})"
                )
        return None

    if ctx:
        tag = "OK" if best.get("feasible") else (
            "WARN(有碰撞)" if best.get("ncol", 0) > 0 else "WARN"
        )
        ctx.log(
            f"  [grip_fit] {tag} {best['approach_label']} roll={best['roll_deg']:.0f}° "
            f"ref_y={best.get('ref_y_label','?')} "
            f"vox={best.get('gap_voxel_n', 0)} vol={best.get('gap_vol_cm3', 0):.2f}cm³ "
            f"pts={best.get('n_gap', 0)} ncol={best['ncol']} "
            f"slider={best.get('hit_slider_mm', 0):.1f}mm "
            f"center={best.get('center_bias_mm', 0):.1f}mm bal={best.get('y_balance', 0):.2f} "
            f"offset={best.get('offset_m', 0)*1000:.1f}mm lat={best.get('lateral_y_m', 0)*1000:.1f}mm "
            f"vlm_dist={best['gap_err_mm']:.1f}mm"
        )

    return {
        "pos": best["pos"].tolist(),
        "quat": best["quat"].tolist(),
        "approach": best["approach"].tolist(),
        "grip_fit": {
            "feasible": best["feasible"],
            "has_volume": best.get("has_volume", False),
            "n_collision": best["ncol"],
            "gap_voxel_n": best.get("gap_voxel_n", 0),
            "gap_vol_cm3": best.get("gap_vol_cm3", 0.0),
            "n_gap": best["n_gap"],
            "n_pos_side": best["n_pos"],
            "n_neg_side": best["n_neg"],
            "has_both_sides": best["has_both"],
            "hit_slider_mm": best.get("hit_slider_mm", 0.0),
            "y_balance": best.get("y_balance", 0.0),
            "center_bias_mm": best.get("center_bias_mm", 0.0),
            "min_rail_clear_mm": best.get("min_rail_clear_mm", 0.0),
            "gap_err_mm": best["gap_err_mm"],
            "vlm_dist_mm": best["gap_err_mm"],
            "bulk_dist_mm": best.get("bulk_dist_mm", 0.0),
            "anchor_dist_mm": best.get("anchor_dist_mm", 0.0),
            "lateral_y_m": best.get("lateral_y_m", 0.0),
            "offset_m": best.get("offset_m", 0.0),
            "approach_label": best["approach_label"],
            "ref_y_label": best.get("ref_y_label", ""),
            "roll_deg": best["roll_deg"],
            "voxel_m": _GAP_VOXEL_M,
            "n_score_pcd": len(score_pcd),
            "n_patch": len(patch),
        },
    }


def _fallback_topdown(hit: np.ndarray, ctx=None) -> None:
    """已废弃：无体积时不得返回假 grasp。"""
    if ctx:
        ctx.log("  [grip_fit] FAIL fallback_top_down 已禁用（开口体积=0）")
    return None
