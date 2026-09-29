"""grasp_object 模式：在目标物体上自动搜索最优夹取 pose（与 plan_grasp_point 区分）。

输入（二选一或组合）：
  - head 冻结视图 + 点击 (u,v)：仅用于 seg 实例分割，确定「夹哪个物体」
  - object_name：直接指定物体名（无点击时用 AABB 中心投影 seg）

规划流程（v13，与 unittest v3 同口径）：
  1. mesh 表面积均匀采样 100 点 + head seg 投影过滤（剔除被爆米花/他物遮挡的内壁点）
  2. 以每点为球心、夹爪开口宽度(~127mm) 为直径得 100 球
  3. 3mm 体素：目标 fill + 环境(桌面/邻物)表面 → 球∩占据体积
  4. 相交体积升序 → 最小 10 个表面点
  5. 10 点作夹爪 1/3 黄点锚点 × 正20面体 20 朝向 × 绕爪轴 8 等分自转 → 1600 pose
  6. fast_overlap 初筛 top60 → GPU 批量 v7 grasp_vol + overlap_vol
  7. overlap 用夹爪法向膨胀 3mm 重算；overlap_vol < 1cm³ 中取 grasp_vol 最大（同 v3，无 DLS）
  8. 返回 best pose；head 主视图叠影标记（同 v3 逻辑，不保存三视角）

与 plan_grasp_point：点击不指定夹取位置，锚点来自步骤 4 的表面点，非射线 3D 命中。
"""

from __future__ import annotations

# 进程日志 / plan_audit 中应出现此串，用于确认代码版本已加载
GRASP_OBJ_BUILD = "v13_unittest_v3_aligned"

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from behavior_interface.skills.gripper_camera_face import (
    apply_camera_face_forward_to_grasp_dict,
)
from behavior_interface.skills.plan_grasp_gripper_fit import (
    GRIP_GAP_CENTER_LOCAL,
    MIN_GAP_EACH_SIDE,
    MIN_GAP_POINTS,
    MIN_GAP_VOXELS,
    _FINGER_BOXES,
    _GAP_BOX,
    _GAP_PROXIMITY_R,
    _GAP_VOXEL_M,
    _PALM_BOX,
    _eef_frame_from_approach_roll,
    _fibonacci_sphere,
    _gap_mask,
    _gap_rail_metrics,
    _gap_voxel_metrics,
    _in_box,
    _mat_to_quat_xyzw,
    _pca_axes,
    _ref_y_candidates,
    _score_pointcloud,
    gripper_half_length_m,
    gripper_reach_length_m,
)

# 指部实体向外扩展 3mm（安全余量，仅用于穿模判据）
_FINGER_EXPAND_M = 0.003
# 指间体积/穿模邻域：仅统计距 gap_center 此半径内的 mesh 表面点（与 ncol 对齐）
_VOL_PROXIMITY_R = 0.018
_NCOL_PROXIMITY_R = _VOL_PROXIMITY_R
# 穿模体素边长（米）：同一体素内多点只计一次，抑制 mesh 过密虚高
_NCOL_VOXEL_M = 0.004
# 随机 EEF pose 采样数（旧路径回退）
_N_RANDOM_POSES = 8000
# 边界采样：200 个上表面边界点 × 每点 30 个 grasp 方向 = 6000 pose
_N_BOUNDARY_POINTS = 200
_N_GRASPS_PER_BOUNDARY = 30
_N_BOUNDARY_POSES = _N_BOUNDARY_POINTS * _N_GRASPS_PER_BOUNDARY
_BOUNDARY_SURFACE_CANDIDATES = 12000
# 同场景复用已验证边界点，避免每次 plan 重复黄球 render
_BOUNDARY_POINTS_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_BOUNDARY_APPROACH_JITTER_DEG = 8.0
# GPU/CPU 批量评估每批 pose 数
_BATCH_POSE_CHUNK = 128
# mesh 三角面均匀采样（专用于两爪间占据体积，不与 depth 混用）
_MESH_SURFACE_SAMPLES = 24000
_MESH_INTERIOR_SAMPLES = 5000
_MESH_PCD_MAX = 16000
# mesh 开口体积：OBJ 楔形约 1mm 宽，阈值按窄开口标定
_MESH_MIN_GAP_VOXELS = 2
_MESH_MIN_SIDE_VOX = 1
_MESH_MIN_CENTER_VOX = 1
_MESH_MIN_PINCH_CORE_VOX = 1
_MESH_MIN_Y_SPAN_M = 0.0003
_MESH_MIN_Z_SPAN_M = 0.0003
# 指间实体须为薄壁/局部表面：过大跨度表示袋腔/体内假体积
_MESH_MAX_GAP_Y_SPAN_M = 0.012
_MESH_MAX_GAP_Z_SPAN_M = 0.016
_MESH_RAIL_Y_THRESH = 0.0003
# 指轨内侧核心带（≈楔形半宽）
_PINCH_CORE_Y_MAX = 0.0006
_MESH_MIN_INTERIOR_PTS = 80
# gap_center 不得偏离 mesh 质心超过此距离（米）
_GAP_CENTER_MAX_DIST_M = 0.06
# 边界锚点须落在 mesh 外表面此距离内（米）
_BOUNDARY_MESH_ANCHOR_MAX_M = 0.018
# gap_center 邻域内须在指间核心带内的 depth/mesh 点数下限
_MIN_PINCH_NEAR_PTS = 4
# EEF 原点到物体表面最近点容差（相对夹爪半长，mm）
_SURF_REACH_TOL_MM = 2.0
# 采样进度日志间隔
_LOG_PROGRESS_EVERY = 1000
# 对 top 候选做位置微调次数
_N_REFINE = 16
_REFINE_JITTER_M = 0.010
# exec_move 与 eef.py 一致：safe = contact - back_m * pointing(quat)
_EXEC_BACK_M = 0.10
# 胸前 tuck 初值（与 eef._CHEST_TUCK_ARM 相同，供 IK 探针模拟 exec 路径）
_CHEST_TUCK_ARM = {
    "right": np.array([-0.73, 0.09, 1.245, -2.04, -0.18, 0.03, 0.08], dtype=np.float64),
    "left": np.array([-0.73, -0.09, -1.245, -2.04, 0.26, -0.03, -0.08], dtype=np.float64),
}
# 同步 DLS 探针：按指向对齐+体积排序后最多尝试多少个 pose
_DLS_PROBE_TOP = 12
# roll 对齐仅对精调后 top-K（控制 plan 耗时）
_ROLL_ALIGN_TOP = 18
# DLS 探针候选最低体积（相对 global max 比例，避免为可达性选空夹）
_PROBE_MIN_VOX_FRAC = 0.42
# 穿模体素数上限（0=严格无穿模；输出与 DLS 池均须满足）
_NCOL_SAFE_TOL = 0
# DLS/精调失败后回退到粗搜体积排序前列（避免只认 36 个精调结果）
_COARSE_FALLBACK_TOP = 100
# 朝外法向筛选：点相对物体质心沿法向一侧（剔除开口物体内壁三角面）
_OUTWARD_NORMAL_MIN_DOT = 0.004
# 表面须朝向 head 相机（剔除袋内壁等背向相机的面）
_CAMERA_FACING_MIN_DOT = 0.12
# 锚点吸附 mesh：薄袋口避免吸到内壁，仅在邻域上沿点中选
_MESH_SNAP_NEAR_M = 0.022
# 可达性：与 grasp.py 一致
_ARM_MIN_REACH = 0.18
_ARM_MAX_REACH = 1.05

# 边界模式（200×30）设计（见模块 docstring；与点击无关）：
# 1) mesh 外表面上沿点（outward+facing+深度可见）FPS，每点作对称轴 1/3 表面锚点
# 2) 均匀随机夹爪朝向（roll），eef = surf_pt - R @ gap_anchor
# 3) 筛选：不穿模 → 左右空隙均衡 → 体积 tie-break → DLS 可达
# 随机点云回退路径仍用体积优先排序。
GRASP_OBJ_FILTER_PRIORITY = (
    "anchor_on_surface_one_third",
    "ncol_no_penetration",
    "gap_lr_balance",
    "gap_voxel_volume",
)


def _triangulate_face_indices(
    counts: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """将 USD faceVertexCounts/Indices 转为三角面索引。"""
    tris: List[List[int]] = []
    i = 0
    for c in counts:
        face = indices[i:i + int(c)]
        i += int(c)
        if len(face) < 3:
            continue
        for j in range(1, len(face) - 1):
            tris.append([int(face[0]), int(face[j]), int(face[j + 1])])
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def _sample_points_on_triangles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    *,
    return_normals: bool = False,
) -> np.ndarray:
    """按三角面面积加权均匀采样表面点（世界系）。"""
    if len(triangles) == 0 or len(vertices) == 0:
        if return_normals:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
        return np.zeros((0, 3), dtype=np.float64)
    n_samples = max(1, int(n_samples))
    v0_all = vertices[triangles[:, 0]]
    v1_all = vertices[triangles[:, 1]]
    v2_all = vertices[triangles[:, 2]]
    face_n = np.cross(v1_all - v0_all, v2_all - v0_all)
    areas = 0.5 * np.linalg.norm(face_n, axis=1)
    fn_norm = np.linalg.norm(face_n, axis=1, keepdims=True) + 1e-12
    face_n_unit = face_n / fn_norm
    total = float(areas.sum())
    if total < 1e-12:
        idx = rng.choice(len(vertices), size=min(n_samples, len(vertices)), replace=True)
        pts = vertices[idx]
        if return_normals:
            return pts, np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1))
        return pts
    tri_idx = rng.choice(len(triangles), size=n_samples, p=areas / total)
    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)
    sr = np.sqrt(r1)
    bu = 1.0 - sr
    bv = sr * (1.0 - r2)
    bw = sr * r2
    tri = triangles[tri_idx]
    v0 = vertices[tri[:, 0]]
    v1 = vertices[tri[:, 1]]
    v2 = vertices[tri[:, 2]]
    pts = bu[:, None] * v0 + bv[:, None] * v1 + bw[:, None] * v2
    if return_normals:
        return pts, face_n_unit[tri_idx]
    return pts


_GRIPPER_LEN_M = gripper_reach_length_m()
_GRIPPER_HALF_MM = gripper_half_length_m() * 1000.0


def _eef_origin_surface_min_mm(q_local: np.ndarray) -> float:
    """EEF 原点到物体 mesh 点（EEF 系）的最小距离（mm）。"""
    if len(q_local) == 0:
        return 999.0
    return float(np.linalg.norm(q_local, axis=1).min() * 1000.0)


def _mesh_surface_reach_ok(surf_min_mm: float) -> bool:
    """EEF 原点须落在物体表面半夹爪reach 内，否则手指够不着实体。"""
    return surf_min_mm <= _GRIPPER_HALF_MM + _SURF_REACH_TOL_MM


def _gap_pinch_reach_ok(
    R: np.ndarray,
    eef_pos: np.ndarray,
    gap_world: np.ndarray,
    surf_src: np.ndarray,
    *,
    min_pts: int = _MIN_PINCH_NEAR_PTS,
) -> Tuple[bool, int]:
    """开口中心邻域内须有实体点落在指间核心带（拒绝悬空高 vox）。"""
    surf_src = np.asarray(surf_src, dtype=np.float64)
    if len(surf_src) < min_pts:
        return False, 0
    near = np.linalg.norm(surf_src - gap_world, axis=1) < _GAP_PROXIMITY_R
    if int(near.sum()) < min_pts:
        return False, 0
    q = (R.T @ (surf_src[near] - eef_pos).T).T
    n_core = int(_pinch_core_mask(q).sum())
    return n_core >= min_pts, n_core


def _pinch_core_mask(q: np.ndarray) -> np.ndarray:
    """对称轴锚点接触 z 带内、两指 OBJ 内侧面之间的核心窄带。"""
    if len(q) == 0:
        return np.zeros(0, dtype=bool)
    from behavior_interface.skills.plan_grasp_gripper_geom import gap_contact_z_bounds, get_wedge_lut

    lut = get_wedge_lut()
    z_tip, z_hi = gap_contact_z_bounds()
    z = q[:, 2]
    y_lo = np.interp(z, lut["z"], lut["y_inner_lo"])
    y_hi = np.interp(z, lut["z"], lut["y_inner_hi"])
    mid = 0.5 * (y_lo + y_hi)
    half = np.minimum(mid - y_lo, y_hi - mid)
    core_half = np.minimum(half, _PINCH_CORE_Y_MAX)
    in_y = (q[:, 1] >= mid - core_half) & (q[:, 1] <= mid + core_half)
    in_z = (z >= z_tip) & (z <= z_hi)
    return in_y & in_z


def _mesh_gap_volume_metrics(q_gap: np.ndarray) -> Dict[str, Any]:
    """
    基于 mesh 采样点的两爪间占据体积（EEF 开口盒内体素法 + 双侧/中心约束）。
    仅统计严格落在开口盒内的点，不用 depth 邻域过滤。
    """
    if len(q_gap) == 0:
        return {
            "gap_voxel_n": 0,
            "gap_vol_cm3": 0.0,
            "n_gap_raw": 0,
            "mesh_pos_vox": 0,
            "mesh_neg_vox": 0,
            "mesh_center_vox": 0,
            "mesh_y_span_mm": 0.0,
            "mesh_z_span_mm": 0.0,
            "mesh_pinch_core_vox": 0,
            "has_mesh_volume": False,
            "n_pos": 0,
            "n_neg": 0,
            "has_both": False,
        }
    vox = np.floor(q_gap / _GAP_VOXEL_M).astype(np.int64)
    uniq = np.unique(vox, axis=0)
    n_vox = int(len(uniq))
    vol_cm3 = float(n_vox * (_GAP_VOXEL_M ** 3) * 1e6)
    y_centers = (uniq[:, 1].astype(np.float64) + 0.5) * _GAP_VOXEL_M
    y_side = 0.0002
    pos_vox = int((y_centers > y_side).sum())
    neg_vox = int((y_centers < -y_side).sum())
    center_vox = int((np.abs(y_centers) <= y_side * 1.5).sum())
    y_span = float(q_gap[:, 1].max() - q_gap[:, 1].min())
    z_span = float(q_gap[:, 2].max() - q_gap[:, 2].min())
    core_pts = q_gap[_pinch_core_mask(q_gap)]
    if len(core_pts) > 0:
        core_vox = np.unique(np.floor(core_pts / _GAP_VOXEL_M).astype(np.int64), axis=0)
        pinch_core_vox = int(len(core_vox))
    else:
        pinch_core_vox = 0
    has_both_sides = pos_vox >= _MESH_MIN_SIDE_VOX and neg_vox >= _MESH_MIN_SIDE_VOX
    entity_span_ok = (
        y_span <= _MESH_MAX_GAP_Y_SPAN_M
        and z_span <= _MESH_MAX_GAP_Z_SPAN_M
        and y_span >= _MESH_MIN_Y_SPAN_M
    )
    # 开口体积：有表面点落入楔形即计体积；entity_span/双侧等仅作软排序，不作硬拒绝
    has_mesh_volume = n_vox >= 1 or n_gap_raw >= 1
    return {
        "gap_voxel_n": n_vox,
        "gap_vol_cm3": vol_cm3,
        "n_gap_raw": int(len(q_gap)),
        "mesh_pos_vox": pos_vox,
        "mesh_neg_vox": neg_vox,
        "mesh_center_vox": center_vox,
        "mesh_y_span_mm": y_span * 1000.0,
        "mesh_z_span_mm": z_span * 1000.0,
        "mesh_pinch_core_vox": pinch_core_vox,
        "has_mesh_volume": has_mesh_volume,
        "mesh_entity_span_ok": entity_span_ok,
        "n_pos": pos_vox,
        "n_neg": neg_vox,
        "has_both": has_both_sides,
    }


def _finger_penetration_mask(q: np.ndarray) -> np.ndarray:
    """穿模：锚点接触 z 带 + 与体积共用的开口楔形排除。"""
    from behavior_interface.skills.plan_grasp_gripper_geom import finger_penetration_mask

    return finger_penetration_mask(q, expand_m=_FINGER_EXPAND_M)


def _count_ncol_penetration(q_col: np.ndarray, near_col: np.ndarray) -> Tuple[int, int, int]:
    """返回 (ncol_vox, ncol_pts, near_n)。"""
    from behavior_interface.skills.plan_grasp_gripper_geom import count_penetration_voxels

    near_n = int(near_col.sum())
    pen = _finger_penetration_mask(q_col) & near_col
    ncol_vox, ncol_pts = count_penetration_voxels(
        q_col, pen, voxel_m=_NCOL_VOXEL_M,
    )
    return ncol_vox, ncol_pts, near_n


def _min_base_dist_mm(q: np.ndarray) -> float:
    """物体点到掌部（夹爪基座）外表的最小距离（mm）。"""
    py0, py1, pz0, pz1, pxh = _PALM_BOX
    dx = np.maximum(np.abs(q[:, 0]) - pxh, 0.0)
    dy = np.maximum(np.maximum(py0 - q[:, 1], 0.0), np.maximum(q[:, 1] - py1, 0.0))
    dz = np.maximum(np.maximum(pz0 - q[:, 2], 0.0), np.maximum(q[:, 2] - pz1, 0.0))
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    return float(dist.min() * 1000.0)


def _evaluate_grasp_object_pose(
    R: np.ndarray,
    eef_pos: np.ndarray,
    pcd: np.ndarray,
    *,
    vol_pcd: Optional[np.ndarray] = None,
    col_pcd: Optional[np.ndarray] = None,
    surf_pcd: Optional[np.ndarray] = None,
    use_mesh_volume: bool = False,
) -> Dict[str, Any]:
    """评估单个 EEF pose：mesh 开口体积 + 指部碰撞（col_pcd）。"""
    gap_world = _gap_world_from_eef(R, eef_pos)
    vol_src = np.asarray(vol_pcd if vol_pcd is not None else pcd, dtype=np.float64)
    col_src = np.asarray(col_pcd if col_pcd is not None else pcd, dtype=np.float64)
    if use_mesh_volume and surf_pcd is not None and len(surf_pcd) >= 8:
        surf_src = np.asarray(surf_pcd, dtype=np.float64)
    else:
        surf_src = np.asarray(pcd, dtype=np.float64)

    q_col = (R.T @ (col_src - eef_pos).T).T
    near_col = np.linalg.norm(col_src - gap_world, axis=1) < _NCOL_PROXIMITY_R
    ncol_exp, ncol_pts, ncol_near_n = _count_ncol_penetration(q_col, near_col)
    base_dist_mm = _min_base_dist_mm(q_col)

    q_vol = (R.T @ (vol_src - eef_pos).T).T
    vol_prox_r = _VOL_PROXIMITY_R if use_mesh_volume else _GAP_PROXIMITY_R
    near_vol = np.linalg.norm(vol_src - gap_world, axis=1) < vol_prox_r
    in_gap = _gap_mask(q_vol, for_volume=True) & near_vol
    n_gap = int(in_gap.sum())
    q_gap = q_vol[in_gap]
    rail = _gap_rail_metrics(q_gap)

    surf_reach_extra: Dict[str, Any] = {}
    if use_mesh_volume:
        q_surf = (R.T @ (surf_src - eef_pos).T).T
        surf_min_mm = _eef_origin_surface_min_mm(q_surf)
        mesh_m = _mesh_gap_volume_metrics(q_gap)
        # mesh 模式：爪间判定用楔形双侧+核心体素，不用 depth 邻域 pinch 误报
        surf_reach_ok = _mesh_surface_reach_ok(surf_min_mm)
        n_pinch_near = int(mesh_m.get("mesh_pinch_core_vox", 0))
        has_volume = bool(mesh_m["has_mesh_volume"])
        gap_voxel_n = mesh_m["gap_voxel_n"]
        gap_vol_cm3 = mesh_m["gap_vol_cm3"]
        n_gap_raw = mesh_m["n_gap_raw"]
        n_pos = mesh_m["n_pos"]
        n_neg = mesh_m["n_neg"]
        has_both = mesh_m["has_both"]
        surf_reach_extra = {
            "mesh_surf_min_mm": surf_min_mm,
            "gripper_half_mm": _GRIPPER_HALF_MM,
            "gripper_len_mm": _GRIPPER_LEN_M * 1000.0,
            "mesh_surf_reach_ok": surf_reach_ok,
            "mesh_pinch_near_pts": n_pinch_near,
        }
        extra = {
            k: mesh_m[k]
            for k in (
                "mesh_pos_vox", "mesh_neg_vox", "mesh_center_vox",
                "mesh_y_span_mm", "mesh_z_span_mm", "mesh_pinch_core_vox",
                "has_mesh_volume", "mesh_entity_span_ok",
            )
        }
        extra.update(surf_reach_extra)
    else:
        gap_ys = q_gap[:, 1] if n_gap else np.array([])
        n_pos = int((gap_ys > 0.004).sum()) if len(gap_ys) else 0
        n_neg = int((gap_ys < -0.004).sum()) if len(gap_ys) else 0
        has_both = n_pos >= MIN_GAP_EACH_SIDE and n_neg >= MIN_GAP_EACH_SIDE
        gap_voxel_n, gap_vol_cm3, n_gap_raw = _gap_voxel_metrics(q_gap)
        has_volume = (
            gap_voxel_n >= MIN_GAP_VOXELS
            and n_gap >= MIN_GAP_POINTS
            and has_both
        )
        extra = {"volume_source": "depth"}

    feasible = has_volume and ncol_exp <= _NCOL_SAFE_TOL
    return {
        "n_gap": n_gap,
        "n_gap_raw": n_gap_raw,
        "gap_voxel_n": gap_voxel_n,
        "gap_vol_cm3": gap_vol_cm3,
        "has_volume": has_volume,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "has_both": has_both,
        "ncol_exp": ncol_exp,
        "ncol_pts": ncol_pts,
        "ncol_near_n": ncol_near_n,
        "base_dist_mm": base_dist_mm,
        "feasible": feasible,
        "volume_source": "mesh_surface" if use_mesh_volume else "depth",
        **extra,
        **rail,
    }


def _is_shoulder_reachable(
    eef_pos: np.ndarray,
    shoulder: np.ndarray,
) -> Tuple[bool, str]:
    d = float(np.linalg.norm(eef_pos - shoulder))
    if d < _ARM_MIN_REACH:
        return False, f"too_close({d:.2f}m)"
    if d > _ARM_MAX_REACH:
        return False, f"too_far({d:.2f}m)"
    return True, f"ok(d={d:.2f}m)"


def _min_vox_threshold(global_max_vox: int) -> int:
    """相对全局最大体积的最低可接受体素数（用于 DLS 探针与退化拒绝）。"""
    return max(MIN_GAP_VOXELS, int(global_max_vox * _PROBE_MIN_VOX_FRAC))


def _ncol_ok(c: Dict[str, Any]) -> bool:
    return _scalar_int_field(c, "ncol_exp", 999) <= _NCOL_SAFE_TOL


def _best_effort_rank_key(c: Dict[str, Any]) -> tuple:
    """无穿模优先 → 指间开口体积最大（软排序，保证总有输出）。"""
    return (
        _scalar_int_field(c, "ncol_exp", 999),
        -_scalar_int_field(c, "gap_voxel_n", 0),
        -float(c.get("gap_vol_cm3", 0.0)),
        -_scalar_int_field(c, "mesh_pinch_core_vox", 0),
        0 if _scalar_bool_field(c, "mesh_entity_span_ok") else 1,
        float(c.get("y_balance", 1.0)),
        float(c.get("center_bias_mm", 999.0)),
        -min(_scalar_int_field(c, "n_pos", 0), _scalar_int_field(c, "n_neg", 0)),
        float(c.get("gap_dist_to_obj_mm", 999.0)),
    )


def _sort_best_effort_pool(
    candidates: List[Dict[str, Any]],
    *,
    prefer_ncol: bool = True,
) -> List[Dict[str, Any]]:
    """优先无穿模候选；无则全池按 ncol→vox 软排序。"""
    if not candidates:
        return []
    if prefer_ncol:
        ncol0 = [c for c in candidates if _ncol_ok(c)]
        if ncol0:
            pool = ncol0
        else:
            pool = list(candidates)
    else:
        pool = list(candidates)
    pool.sort(key=_best_effort_rank_key)
    return pool


def _pose_rank_key_boundary(c: Dict[str, Any]) -> tuple:
    """边界锚点模式：不穿模 → 表面实体跨度 → 左右均衡 → 体积 tie-break。"""
    n_pos = _scalar_int_field(c, "n_pos", 0)
    n_neg = _scalar_int_field(c, "n_neg", 0)
    return (
        0 if _scalar_bool_field(c, "has_volume") else 1,
        _scalar_int_field(c, "ncol_exp", 999),
        0 if _scalar_bool_field(c, "mesh_entity_span_ok") else 1,
        float(c.get("y_balance", 1.0)),
        float(c.get("center_bias_mm", 999.0)),
        -min(n_pos, n_neg),
        -_scalar_int_field(c, "gap_voxel_n", 0),
        -_scalar_int_field(c, "n_gap", 0),
        float(c.get("gap_dist_to_obj_mm", 999.0)),
        float(c.get("mesh_surf_min_mm", 999.0)),
    )


def _pose_rank_key(
    c: Dict[str, Any],
    *,
    boundary_mode: bool = False,
) -> tuple:
    """随机点云回退：体积优先；边界模式见 _pose_rank_key_boundary。"""
    if boundary_mode:
        return _pose_rank_key_boundary(c)
    return (
        0 if _scalar_bool_field(c, "has_volume") else 1,
        -_scalar_int_field(c, "gap_voxel_n", 0),
        -float(c.get("gap_vol_cm3", 0.0)),
        _scalar_int_field(c, "ncol_exp", 999),
        -_scalar_int_field(c, "mesh_pinch_core_vox", 0),
        -_scalar_int_field(c, "mesh_center_vox", 0),
        -_scalar_int_field(c, "n_gap", 0),
        float(c.get("gap_dist_to_obj_mm", 999.0)),
        float(c.get("mesh_surf_min_mm", 999.0)),
        -float(c.get("base_dist_mm", 0.0)),
        float(c.get("y_balance", 1.0)),
        float(c.get("center_bias_mm", 999.0)),
    )


def _as_obj_ref(obj_ref: Optional[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    """避免 `obj_ref or fallback` 在 numpy 数组上触发 ambiguous truth value。"""
    if obj_ref is None:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return np.asarray(obj_ref, dtype=np.float64).reshape(3)


def _gap_world_from_eef(R: np.ndarray, eef_pos: np.ndarray) -> np.ndarray:
    """EEF → 对称轴 1/3 表面锚点（世界系）。"""
    from behavior_interface.skills.plan_grasp_gripper_geom import gap_anchor_local_one_third_from_base

    return (
        np.asarray(eef_pos, dtype=np.float64).reshape(3)
        + np.asarray(R, dtype=np.float64) @ gap_anchor_local_one_third_from_base()
    )


def _sync_pose_gap_center(cand: Dict[str, Any]) -> Dict[str, Any]:
    """精调/roll 后同步 gap_center，避免 EEF 与锚点脱节导致可视化偏离物体。"""
    from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat

    R = _quat_to_mat(np.asarray(cand["quat"], dtype=np.float64))
    eef = np.asarray(cand["pos"], dtype=np.float64).reshape(3)
    out = dict(cand)
    out["gap_center"] = _gap_world_from_eef(R, eef)
    return out


def _pick_rim_mesh_snap_point(
    gap_c: np.ndarray,
    mesh_surf: np.ndarray,
    *,
    near_m: float = _MESH_SNAP_NEAR_M,
) -> Tuple[np.ndarray, float, str]:
    """薄开口物体：邻域内优先 z 较高的上沿外表面，避免吸到袋内壁。"""
    mesh_surf = np.asarray(mesh_surf, dtype=np.float64)
    gap_c = np.asarray(gap_c, dtype=np.float64).reshape(3)
    dists = np.linalg.norm(mesh_surf - gap_c, axis=1)
    near_m = float(near_m)
    near = mesh_surf[dists < near_m]
    if len(near) < 4:
        near = mesh_surf[dists < near_m * 2.0]
    if len(near) >= 4:
        z_med = float(np.percentile(near[:, 2], 52))
        top = near[near[:, 2] >= z_med - 0.004]
        if len(top) >= 1:
            j = int(np.argmin(np.linalg.norm(top - gap_c, axis=1)))
            snapped = top[j].copy()
            return snapped, float(np.linalg.norm(snapped - gap_c) * 1000.0), "rim_top"
    idx = int(np.argmin(dists))
    snapped = mesh_surf[idx].copy()
    return snapped, float(dists[idx] * 1000.0), "nearest"


def _project_anchor_to_boundary_mesh(
    anchor: np.ndarray,
    mesh_surf: Optional[np.ndarray],
) -> np.ndarray:
    """边界精调抖动后，把锚点投回上沿外表面（保持开口在袋口 rim）。"""
    if mesh_surf is None or len(mesh_surf) < 8:
        return np.asarray(anchor, dtype=np.float64).reshape(3)
    snapped, _, _ = _pick_rim_mesh_snap_point(anchor, mesh_surf, near_m=0.028)
    return snapped


def _eef_anchor_offset_mm(cand: Dict[str, Any]) -> float:
    """EEF 原点到对称轴 1/3 表面锚点的距离（mm）。"""
    from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat

    R = _quat_to_mat(np.asarray(cand["quat"], dtype=np.float64))
    eef = np.asarray(cand["pos"], dtype=np.float64).reshape(3)
    gap = np.asarray(cand.get("gap_center", _gap_world_from_eef(R, eef)), dtype=np.float64)
    return float(np.linalg.norm(gap - eef) * 1000.0)


def _snap_pose_to_mesh_surface(
    cand: Dict[str, Any],
    mesh_surf: Optional[np.ndarray],
) -> Dict[str, Any]:
    """将对称轴 1/3 锚点吸附到 mesh 外表面并重算 EEF：eef = snapped - R @ gap_anchor。"""
    if mesh_surf is None or len(mesh_surf) < 8:
        return cand
    from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat
    from behavior_interface.skills.plan_grasp_gripper_geom import gap_anchor_local_one_third_from_base

    mesh_surf = np.asarray(mesh_surf, dtype=np.float64)
    R = _quat_to_mat(np.asarray(cand["quat"], dtype=np.float64))
    gap_local = gap_anchor_local_one_third_from_base()
    gap_c = np.asarray(cand.get("gap_center", cand["pos"]), dtype=np.float64).reshape(3)
    snapped, snap_mm, snap_tag = _pick_rim_mesh_snap_point(gap_c, mesh_surf)
    eef = snapped - R @ gap_local
    out = dict(cand)
    out["gap_center"] = snapped
    out["pos"] = eef
    out["gap_dist_to_obj_mm"] = snap_mm
    out["mesh_snap_tag"] = snap_tag
    out["eef_anchor_dist_mm"] = _eef_anchor_offset_mm(out)
    return out


def _min_gap_dist_to_mesh_mm(gap_c: np.ndarray, mesh_pts: Optional[np.ndarray]) -> float:
    """开口锚点到 mesh 最近距离（mm）。"""
    if mesh_pts is None or len(mesh_pts) < 4:
        return 999.0
    gap_c = np.asarray(gap_c, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(np.asarray(mesh_pts, dtype=np.float64) - gap_c, axis=1).min() * 1000.0)


def _gap_on_object_ok(
    gap_c: np.ndarray,
    ref_pt: np.ndarray,
    obj_aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    *,
    hit_anchor: Optional[np.ndarray] = None,
    mesh_surf_pts: Optional[np.ndarray] = None,
    is_boundary: bool = False,
) -> bool:
    """开口中心须在物体上：边界模式要求贴近 mesh 外表面，否则 AABB/锚点距离。"""
    gap_c = np.asarray(gap_c, dtype=np.float64).reshape(3)
    if is_boundary:
        if mesh_surf_pts is not None and len(mesh_surf_pts) >= 8:
            return (
                _min_gap_dist_to_mesh_mm(gap_c, mesh_surf_pts)
                <= _BOUNDARY_MESH_ANCHOR_MAX_M * 1000.0
            )
        return True
    if obj_aabb is not None:
        lo, hi = obj_aabb
        lo = np.asarray(lo, dtype=np.float64) - 0.03
        hi = np.asarray(hi, dtype=np.float64) + 0.03
        if bool(np.all(gap_c >= lo) and np.all(gap_c <= hi)):
            return True
    refs = [np.asarray(ref_pt, dtype=np.float64).reshape(3)]
    if hit_anchor is not None:
        refs.append(np.asarray(hit_anchor, dtype=np.float64).reshape(3))
    return min(float(np.linalg.norm(gap_c - r)) for r in refs) <= _GAP_CENTER_MAX_DIST_M


def _obj_ref_for_region(
    obj_ref: Optional[np.ndarray],
    obj_aabb: Optional[Tuple[np.ndarray, np.ndarray]],
    fallback: np.ndarray,
) -> np.ndarray:
    """裁剪/距离校验用参考点：有 AABB 时用质心，避免错误 depth hit 导致 crop→0。"""
    if obj_aabb is not None:
        lo, hi = obj_aabb
        return (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64)) * 0.5
    return _as_obj_ref(obj_ref, fallback)


def _object_region_mask(
    pts: np.ndarray,
    obj_ref: np.ndarray,
    obj_aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    *,
    margin: float = 0.025,
) -> np.ndarray:
    """点是否属于目标物体区域（AABB 优先，否则球形）。"""
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    ref = np.asarray(obj_ref, dtype=np.float64).reshape(3)
    if obj_aabb is not None:
        lo, hi = obj_aabb
        lo = np.asarray(lo, dtype=np.float64) - margin
        hi = np.asarray(hi, dtype=np.float64) + margin
        return np.all(pts >= lo, axis=1) & np.all(pts <= hi, axis=1)
    r = _GAP_CENTER_MAX_DIST_M + 0.02
    return np.linalg.norm(pts - ref, axis=1) < r


def _crop_to_object_region(
    pcd: np.ndarray,
    obj_ref: np.ndarray,
    obj_aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    *,
    ctx=None,
) -> np.ndarray:
    m = _object_region_mask(pcd, obj_ref, obj_aabb)
    sub = pcd[m]
    if ctx is not None:
        ctx.log(
            f"  [grasp_obj] 物体区域裁剪 {len(pcd)} → {len(sub)} "
            f"ref={np.asarray(obj_ref).round(3).tolist()}"
        )
    return sub if len(sub) >= 8 else pcd


def _is_graspable_surface_mesh_path(path: str) -> bool:
    """判定 Mesh prim 是否为「可抓取的真实可见表面」。

    需排除两类非可见/非物体表面几何，否则锚点(黄球)与开口体积都会出错：
      1. *collision*：凸分解(VHACD)凸块，会用凸块填满袋口凹腔；
      2. meta__*/fillable/*：OmniGibson 元链接（fillable 可填充腔体等），
         是「不渲染」的语义标记，会在袋口上方封盖 → 黄球悬空、开口体积虚高。
    """
    p = str(path).lower()
    if "collision" in p:
        return False
    if "meta__" in p or "fillable" in p:
        return False
    return True


def _visual_mesh_prims(root, pxr, UsdGeom) -> list:
    """仅返回真实可见表面 Mesh prim（排除 collision 凸块与 meta 元链接）。

    无匹配时回退全部，避免极端物体取不到 mesh。
    """
    all_meshes = [p for p in pxr.Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
    visual = [p for p in all_meshes if _is_graspable_surface_mesh_path(p.GetPath())]
    return visual if visual else all_meshes


def _mesh_local_to_world(
    local_pts: np.ndarray,
    prim,
    normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """USD mesh 局部点/法向 → 世界系（与 mesh_prim_to_trimesh world_frame 一致）。"""
    from omnigibson.utils.usd_utils import PoseAPI
    from omnigibson.utils.transform_utils_np import transform_points

    path = str(prim.GetPath())
    mat = np.asarray(
        PoseAPI.get_world_pose_with_scale(path).detach().cpu().numpy(),
        dtype=np.float64,
    )
    world_pts = transform_points(local_pts, mat)
    if normals is None or len(normals) == 0:
        return world_pts, None
    rot = mat[:3, :3]
    wn = normals @ rot.T
    nn = np.linalg.norm(wn, axis=1, keepdims=True) + 1e-12
    return world_pts, wn / nn


def build_object_mesh_grasp_cloud(
    world,
    object_name: Optional[str],
    *,
    n_surface: int = _MESH_SURFACE_SAMPLES,
    n_interior: int = _MESH_INTERIOR_SAMPLES,
    max_pts: int = _MESH_PCD_MAX,
    obj_ref: Optional[np.ndarray] = None,
    obj_aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    seed: int = 0,
    ctx=None,
) -> Optional[Dict[str, Any]]:
    """
    从仿真 USD mesh 构建专用于两爪间体积评估的点云：
    三角面面积加权表面采样 +（封闭子网格）体内采样；不与 depth 混用。
    """
    if world is None or not object_name:
        return None
    from behavior_interface.skills.grasp import _resolve_object_handle

    obj = _resolve_object_handle(world, object_name)
    if obj is None:
        if ctx:
            ctx.log(f"  [grasp_obj] mesh: 未找到物体 {object_name}")
        return None
    try:
        import omnigibson as og
        import omnigibson.lazy as lazy

        pxr = lazy.pxr
        UsdGeom = pxr.UsdGeom
        stage = og.sim.stage
        prim_path = getattr(obj, "prim_path", None) or f"/World/{obj.name}"
        root = stage.GetPrimAtPath(str(prim_path))
        if not root.IsValid():
            return None

        mesh_prims = _visual_mesh_prims(root, pxr, UsdGeom)
        if not mesh_prims:
            if ctx:
                ctx.log(f"  [grasp_obj] mesh: {prim_path} 无 Mesh prim")
            return None

        rng = np.random.default_rng(seed)
        n_mesh = len(mesh_prims)
        surf_budget = max(800, int(n_surface // max(1, n_mesh)))
        interior_budget = max(200, int(n_interior // max(1, n_mesh)))
        surface_chunks: List[np.ndarray] = []
        interior_chunks: List[np.ndarray] = []

        for prim in mesh_prims:
            mesh = UsdGeom.Mesh(prim)
            local_pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            if len(local_pts) < 3:
                continue
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            tris = _triangulate_face_indices(counts, indices)
            world_pts, _ = _mesh_local_to_world(local_pts, prim)
            if len(tris) > 0:
                surface_chunks.append(
                    _sample_points_on_triangles(world_pts, tris, surf_budget, rng),
                )
            else:
                idx = rng.choice(len(world_pts), size=min(surf_budget, len(world_pts)), replace=True)
                surface_chunks.append(world_pts[idx])

            # 封闭子网格体内采样（爆米花等实体 prim）；失败则跳过
            try:
                from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh
                import trimesh

                tm = mesh_prim_to_trimesh_mesh(
                    prim, include_normals=False, include_texcoord=False, world_frame=True,
                )
                if tm.is_watertight and float(tm.volume) > 1e-8:
                    interior_chunks.append(
                        np.asarray(
                            trimesh.sample.volume_mesh(tm, interior_budget),
                            dtype=np.float64,
                        ),
                    )
            except Exception:
                pass

        if not surface_chunks and not interior_chunks:
            return None
        surface_all = (
            np.vstack(surface_chunks) if surface_chunks else np.zeros((0, 3), dtype=np.float64)
        )
        interior_all = (
            np.vstack(interior_chunks) if interior_chunks else np.zeros((0, 3), dtype=np.float64)
        )
        merged = np.vstack([c for c in (surface_chunks + interior_chunks) if len(c) > 0])
        region_ref = _obj_ref_for_region(obj_ref, obj_aabb, merged.mean(axis=0))
        if len(merged) > 0:
            m = _object_region_mask(merged, region_ref, obj_aabb, margin=0.02)
            merged = merged[m] if int(m.sum()) >= 8 else merged
            if len(surface_all) > 0:
                ms = _object_region_mask(surface_all, region_ref, obj_aabb, margin=0.02)
                surface_all = surface_all[ms] if int(ms.sum()) >= 8 else surface_all
            if len(interior_all) > 0:
                mi = _object_region_mask(interior_all, region_ref, obj_aabb, margin=0.02)
                interior_all = interior_all[mi] if int(mi.sum()) >= 8 else interior_all
        # 指间体积与穿模均仅用 USD mesh 外表面：禁止 interior/袋腔点计入开口体积
        vol_pts = surface_all if len(surface_all) >= 8 else merged
        col_pts = vol_pts
        if len(vol_pts) > max_pts:
            vol_pts = vol_pts[rng.choice(len(vol_pts), max_pts, replace=False)]
        if len(col_pts) > max_pts:
            col_pts = col_pts[rng.choice(len(col_pts), max_pts, replace=False)]
        surf_pts = surface_all
        surf_cap = min(12000, max_pts)
        if len(surf_pts) > surf_cap:
            surf_pts = surf_pts[rng.choice(len(surf_pts), surf_cap, replace=False)]
        if len(surf_pts) < 8:
            surf_pts = col_pts
        centroid = vol_pts.mean(axis=0) if len(vol_pts) >= 8 else merged.mean(axis=0)
        if ctx:
            ctx.log(
                f"  [grasp_obj] mesh体积云 vol={len(vol_pts)} col={len(col_pts)} "
                f"(surface={len(surface_all)} interior={len(interior_all)} "
                f"vol_src=surface vol_r={_VOL_PROXIMITY_R*1000:.0f}mm "
                f"max_span_y/z={_MESH_MAX_GAP_Y_SPAN_M*1000:.0f}/"
                f"{_MESH_MAX_GAP_Z_SPAN_M*1000:.0f}mm) "
                f"surf={len(surf_pts)} path={prim_path}"
            )
        return {
            "vol": vol_pts,
            "col": col_pts,
            "surface": surf_pts,
            "centroid": centroid,
            "n_interior": int(len(interior_all)),
            "volume_source_tag": "surface",
        }
    except Exception as e:
        if ctx:
            ctx.log(f"  [grasp_obj] mesh采样失败: {e}")
        return None


def sample_object_mesh_points(
    world,
    object_name: Optional[str],
    *,
    max_pts: int = _MESH_PCD_MAX,
    obj_ref: Optional[np.ndarray] = None,
    obj_aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ctx=None,
) -> Optional[np.ndarray]:
    """兼容旧接口：返回 mesh 体积评估点云。"""
    out = build_object_mesh_grasp_cloud(
        world, object_name,
        max_pts=max_pts, obj_ref=obj_ref, obj_aabb=obj_aabb, ctx=ctx,
    )
    if out is None:
        return None
    if isinstance(out, dict):
        return out.get("vol")
    return out


def _normalize_mesh_input(
    mesh_pcd: Optional[Any],
) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    Optional[np.ndarray], int, str,
]:
    """统一 mesh 输入：体积点云、碰撞点云、表面点云、质心。"""
    if mesh_pcd is None:
        return None, None, None, None, 0, "none"
    if isinstance(mesh_pcd, dict):
        vol = mesh_pcd.get("vol")
        col = mesh_pcd.get("col")
        if col is None:
            col = vol
        surf = mesh_pcd.get("surface")
        if surf is None:
            surf = col
        cen = mesh_pcd.get("centroid")
        n_int = int(mesh_pcd.get("n_interior", 0))
        tag = str(mesh_pcd.get("volume_source_tag", "mesh"))
        return vol, col, surf, cen, n_int, tag
    arr = np.asarray(mesh_pcd, dtype=np.float64)
    cen = arr.mean(axis=0) if len(arr) else None
    return arr, arr, arr, cen, 0, "merged"


def _pick_torch_device(ctx=None):
    """规划批量评估设备：显存充足用 GPU，否则 CPU torch 向量化。"""
    try:
        import torch
        if torch.cuda.is_available():
            # CUDA_VISIBLE_DEVICES uses process-local ordinals.  Resolve an
            # explicitly recorded physical owner before querying memory so an
            # unmasked compatibility caller cannot silently use GPU0.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            physical = os.environ.get("BEHAVIOR_INTERFACE_PHYSICAL_GPU", "").strip()
            local = None
            if visible:
                entries = [item.strip() for item in visible.split(",") if item.strip()]
                if physical and physical in entries:
                    local = entries.index(physical)
                elif len(entries) == 1:
                    local = 0
            elif physical.isdigit():
                owner = int(physical)
                if 0 <= owner < int(torch.cuda.device_count()):
                    local = owner
            if local is None:
                local = int(torch.cuda.current_device())
            device = torch.device(f"cuda:{local}")
            free, _ = torch.cuda.mem_get_info(device)
            if free > 400 * 1024 * 1024:
                if ctx:
                    ctx.log(f"  [grasp_obj] 批量评估 device={device} free={free // (1024 * 1024)}MB")
                return device
    except Exception:
        pass
    if ctx:
        ctx.log("  [grasp_obj] 批量评估 device=cpu（显存不足或不可用）")
    return __import__("torch").device("cpu")


def _torch_box_mask(q, ylo, yhi, zlo, zhi, xh):
    import torch
    return (
        (q[..., 0] >= -xh) & (q[..., 0] <= xh)
        & (q[..., 1] >= ylo) & (q[..., 1] <= yhi)
        & (q[..., 2] >= zlo) & (q[..., 2] <= zhi)
    )


_WEDGE_LUT_TORCH: Dict[str, Any] = {}


def _torch_wedge_lut(device, dtype):
    """缓存 OBJ 楔形 LUT 到 torch（批量评估用）。"""
    import torch
    from behavior_interface.skills.plan_grasp_gripper_geom import get_wedge_lut

    key = (str(device), str(dtype))
    if key not in _WEDGE_LUT_TORCH:
        from behavior_interface.skills.plan_grasp_gripper_geom import gap_contact_z_bounds

        lut = get_wedge_lut()
        z_contact_lo, z_contact_hi = gap_contact_z_bounds()
        _WEDGE_LUT_TORCH[key] = {
            "z": torch.as_tensor(lut["z"], device=device, dtype=dtype),
            "y_inner_lo": torch.as_tensor(lut["y_inner_lo"], device=device, dtype=dtype),
            "y_inner_hi": torch.as_tensor(lut["y_inner_hi"], device=device, dtype=dtype),
            "y_outer_f1": torch.as_tensor(lut["y_outer_f1"], device=device, dtype=dtype),
            "y_outer_f2": torch.as_tensor(lut["y_outer_f2"], device=device, dtype=dtype),
            "x_half_gap": torch.as_tensor(lut["x_half_gap"], device=device, dtype=dtype),
            "x_half_finger": torch.as_tensor(lut["x_half_finger"], device=device, dtype=dtype),
            "gap_width": torch.as_tensor(lut["gap_width"], device=device, dtype=dtype),
            "z_grasp_lo": float(lut["z_grasp_lo"]),
            "z_grasp_hi": float(lut["z_grasp_hi"]),
            "z_contact_lo": float(z_contact_lo),
            "z_contact_hi": float(z_contact_hi),
            "z_min": float(lut["z"].min()),
            "z_max": float(lut["z"].max()),
            "min_gap_width": 0.0003,
            "vol_y_inflate": 0.001,
        }
    return _WEDGE_LUT_TORCH[key]


def _torch_interp_z(z, x_pts, y_pts):
    import torch
    z_flat = z.reshape(-1)
    x0 = x_pts[0]
    x1 = x_pts[-1]
    z_cl = z_flat.clamp(x0, x1)
    idx = torch.searchsorted(x_pts, z_cl, right=True) - 1
    idx = idx.clamp(0, x_pts.numel() - 2)
    xa = x_pts[idx]
    xb = x_pts[idx + 1]
    ya = y_pts[idx]
    yb = y_pts[idx + 1]
    t = (z_cl - xa) / (xb - xa + 1e-9)
    out = ya + t * (yb - ya)
    return out.reshape(z.shape)


def _torch_gap_wedge_mask(q, *, for_volume: bool = False):
    import torch
    lut = _torch_wedge_lut(q.device, q.dtype)
    z = q[..., 2]
    y_lo = _torch_interp_z(z, lut["z"], lut["y_inner_lo"])
    y_hi = _torch_interp_z(z, lut["z"], lut["y_inner_hi"])
    xh = _torch_interp_z(z, lut["z"], lut["x_half_gap"])
    gap_w = _torch_interp_z(z, lut["z"], lut["gap_width"])
    inflate = lut["vol_y_inflate"] if for_volume else 0.0
    y_lo = y_lo - inflate
    y_hi = y_hi + inflate
    if for_volume:
        in_z = (z >= lut["z_contact_lo"]) & (z <= lut["z_contact_hi"])
    else:
        in_z = (z >= lut["z_grasp_lo"]) & (z <= lut["z_grasp_hi"])
    in_y = (q[..., 1] > y_lo) & (q[..., 1] < y_hi)
    in_x = torch.abs(q[..., 0]) <= xh
    open_slab = gap_w >= lut["min_gap_width"]
    return in_z & in_y & in_x & open_slab


def _torch_finger_body_mask(q, expand_m: float = 0.0):
    import torch
    lut = _torch_wedge_lut(q.device, q.dtype)
    z = q[..., 2]
    y_lo = _torch_interp_z(z, lut["z"], lut["y_inner_lo"])
    y_hi = _torch_interp_z(z, lut["z"], lut["y_inner_hi"])
    y1_max = _torch_interp_z(z, lut["z"], lut["y_outer_f1"])
    y2_min = _torch_interp_z(z, lut["z"], lut["y_outer_f2"])
    xh = _torch_interp_z(z, lut["z"], lut["x_half_finger"]) + expand_m
    em = float(expand_m)
    f1 = (
        (q[..., 1] >= y_hi - em * 0.5)
        & (q[..., 1] <= y1_max + em)
        & (torch.abs(q[..., 0]) <= xh)
        & (z >= lut["z_min"])
        & (z <= lut["z_max"])
    )
    f2 = (
        (q[..., 1] <= y_lo + em * 0.5)
        & (q[..., 1] >= y2_min - em)
        & (torch.abs(q[..., 0]) <= xh)
        & (z >= lut["z_min"])
        & (z <= lut["z_max"])
    )
    return (f1 | f2) & ~_torch_gap_wedge_mask(q)


def _torch_finger_penetration_mask(q, expand_m: float = 0.0):
    import torch
    lut = _torch_wedge_lut(q.device, q.dtype)
    z = q[..., 2]
    in_contact_z = (z >= lut["z_contact_lo"]) & (z <= lut["z_contact_hi"])
    y_lo = _torch_interp_z(z, lut["z"], lut["y_inner_lo"])
    y_hi = _torch_interp_z(z, lut["z"], lut["y_inner_hi"])
    y1_max = _torch_interp_z(z, lut["z"], lut["y_outer_f1"])
    y2_min = _torch_interp_z(z, lut["z"], lut["y_outer_f2"])
    xh = _torch_interp_z(z, lut["z"], lut["x_half_finger"]) + expand_m
    em = float(expand_m)
    f1 = (
        (q[..., 1] >= y_hi - em * 0.5)
        & (q[..., 1] <= y1_max + em)
        & (torch.abs(q[..., 0]) <= xh)
        & in_contact_z
    )
    f2 = (
        (q[..., 1] <= y_lo + em * 0.5)
        & (q[..., 1] >= y2_min - em)
        & (torch.abs(q[..., 0]) <= xh)
        & in_contact_z
    )
    return (f1 | f2) & ~_torch_gap_wedge_mask(q, for_volume=True)


def _batch_evaluate_grasp_poses(
    poses: List[Dict[str, Any]],
    score_pcd: np.ndarray,
    shoulder: np.ndarray,
    *,
    vol_pcd: Optional[np.ndarray] = None,
    col_pcd: Optional[np.ndarray] = None,
    surf_pcd: Optional[np.ndarray] = None,
    obj_ref: Optional[np.ndarray] = None,
    obj_aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    mesh_centroid: Optional[np.ndarray] = None,
    hit_anchor: Optional[np.ndarray] = None,
    ctx=None,
) -> List[Dict[str, Any]]:
    """GPU/CPU 批量评估：开口体积仅用 mesh(vol_pcd)，指碰用 col_pcd。"""
    import torch

    if not poses:
        return []
    device = _pick_torch_device(ctx)
    use_mesh_vol = vol_pcd is not None and len(vol_pcd) >= 8
    vol_src = np.asarray(vol_pcd, dtype=np.float64) if use_mesh_vol else score_pcd
    col_src = np.asarray(
        col_pcd if col_pcd is not None else (vol_pcd if use_mesh_vol else score_pcd),
        dtype=np.float64,
    )
    # surf_reach / pinch：mesh 模式用 mesh 表面点，避免 depth 邻域误报 pinch_near。
    if use_mesh_vol and surf_pcd is not None and len(surf_pcd) >= 8:
        surf_src = np.asarray(surf_pcd, dtype=np.float64)
    else:
        surf_src = np.asarray(score_pcd, dtype=np.float64)

    ref_pt = (
        np.asarray(mesh_centroid, dtype=np.float64).reshape(3)
        if mesh_centroid is not None
        else _obj_ref_for_region(obj_ref, obj_aabb, score_pcd.mean(axis=0))
    )
    if use_mesh_vol:
        vol_mask_np = np.ones(len(vol_src), dtype=bool)
        col_mask_np = np.ones(len(col_src), dtype=bool)
    else:
        vol_mask_np = _object_region_mask(vol_src, ref_pt, obj_aabb)
        col_mask_np = _object_region_mask(col_src, ref_pt, obj_aabb)

    vol_t = torch.as_tensor(vol_src, dtype=torch.float32, device=device)
    col_t = torch.as_tensor(col_src, dtype=torch.float32, device=device)
    surf_t = torch.as_tensor(surf_src, dtype=torch.float32, device=device)
    vol_mask_t = torch.as_tensor(vol_mask_np, dtype=torch.bool, device=device)
    col_mask_t = torch.as_tensor(col_mask_np, dtype=torch.bool, device=device)
    py0, py1, pz0, pz1, pxh = _PALM_BOX
    vol_prox_r = float(_VOL_PROXIMITY_R if use_mesh_vol else _GAP_PROXIMITY_R)
    ncol_prox_r = float(_NCOL_PROXIMITY_R)
    out: List[Dict[str, Any]] = []

    for start in range(0, len(poses), _BATCH_POSE_CHUNK):
        chunk = poses[start:start + _BATCH_POSE_CHUNK]
        R = torch.stack([
            torch.as_tensor(p["R"], dtype=torch.float32, device=device) for p in chunk
        ])
        eef = torch.stack([
            torch.as_tensor(p["eef_pos"], dtype=torch.float32, device=device) for p in chunk
        ])
        from behavior_interface.skills.plan_grasp_gripper_geom import gap_anchor_local_one_third_from_base

        gap_local_t = torch.as_tensor(
            gap_anchor_local_one_third_from_base(), dtype=torch.float32, device=device,
        )
        gap_world = eef + (R @ gap_local_t.unsqueeze(-1)).squeeze(-1)

        diff_col = col_t.unsqueeze(0) - eef.unsqueeze(1)
        q_col = torch.einsum("bij,bnj->bni", R.transpose(1, 2), diff_col)
        dist_col = torch.linalg.norm(col_t.unsqueeze(0) - gap_world.unsqueeze(1), dim=2)
        near_col = dist_col < ncol_prox_r
        finger_pen = _torch_finger_penetration_mask(q_col, expand_m=_FINGER_EXPAND_M)
        pen_mask = finger_pen & near_col & col_mask_t.unsqueeze(0)

        diff_vol = vol_t.unsqueeze(0) - eef.unsqueeze(1)
        q_vol = torch.einsum("bij,bnj->bni", R.transpose(1, 2), diff_vol)
        dist_vol = torch.linalg.norm(vol_t.unsqueeze(0) - gap_world.unsqueeze(1), dim=2)
        near_vol = dist_vol < vol_prox_r
        in_gap = _torch_gap_wedge_mask(q_vol, for_volume=True) & near_vol & vol_mask_t.unsqueeze(0)

        z0 = torch.tensor(0.0, device=device)
        dx = torch.maximum(torch.abs(q_col[..., 0]) - pxh, z0)
        dy = torch.maximum(torch.maximum(py0 - q_col[..., 1], z0), torch.maximum(q_col[..., 1] - py1, z0))
        dz = torch.maximum(torch.maximum(pz0 - q_col[..., 2], z0), torch.maximum(q_col[..., 2] - pz1, z0))
        base_dist_mm = torch.sqrt(dx * dx + dy * dy + dz * dz).amin(dim=1) * 1000.0

        diff_surf = surf_t.unsqueeze(0) - eef.unsqueeze(1)
        q_surf = torch.einsum("bij,bnj->bni", R.transpose(1, 2), diff_surf)
        surf_min_mm = torch.linalg.norm(q_surf, dim=2).amin(dim=1) * 1000.0
        dist_surf = torch.linalg.norm(surf_t.unsqueeze(0) - gap_world.unsqueeze(1), dim=2)
        near_surf = dist_surf < prox_r
        from behavior_interface.skills.plan_grasp_gripper_geom import gap_contact_z_bounds, get_wedge_lut
        wlut = get_wedge_lut()
        z_tip, z_hi_p = gap_contact_z_bounds()
        z_s = q_surf[..., 2]
        y_lo_p = _torch_interp_z(
            z_s,
            _torch_wedge_lut(q_surf.device, q_surf.dtype)["z"],
            _torch_wedge_lut(q_surf.device, q_surf.dtype)["y_inner_lo"],
        )
        y_hi_p = _torch_interp_z(
            z_s,
            _torch_wedge_lut(q_surf.device, q_surf.dtype)["z"],
            _torch_wedge_lut(q_surf.device, q_surf.dtype)["y_inner_hi"],
        )
        mid_p = 0.5 * (y_lo_p + y_hi_p)
        half_p = torch.minimum(mid_p - y_lo_p, y_hi_p - mid_p)
        core_half = torch.minimum(half_p, torch.tensor(_PINCH_CORE_Y_MAX, device=q_surf.device))
        pinch_core = (
            (q_surf[..., 1] >= mid_p - core_half)
            & (q_surf[..., 1] <= mid_p + core_half)
            & (z_s >= z_tip)
            & (z_s <= z_hi_p)
            & near_surf
        )
        n_pinch_near_t = pinch_core.sum(dim=1)
        min_pinch_t = torch.tensor(_MIN_PINCH_NEAR_PTS, dtype=torch.int64, device=device)
        surf_reach_ok = n_pinch_near_t >= min_pinch_t

        n_gap = in_gap.sum(dim=1)
        for bi, meta in enumerate(chunk):
            mask_i = in_gap[bi]
            n_g = int(n_gap[bi].item())
            q_gap_np = q_vol[bi, mask_i].detach().cpu().numpy()
            rail = _gap_rail_metrics(q_gap_np)
            if use_mesh_vol:
                mesh_m = _mesh_gap_volume_metrics(q_gap_np)
                s_min = float(surf_min_mm[bi].item())
                s_ok = _mesh_surface_reach_ok(s_min)
                has_volume = bool(mesh_m["has_mesh_volume"])
                gap_voxel_n = mesh_m["gap_voxel_n"]
                gap_vol_cm3 = mesh_m["gap_vol_cm3"]
                n_gap_raw = mesh_m["n_gap_raw"]
                n_pos = mesh_m["n_pos"]
                n_neg = mesh_m["n_neg"]
                has_both = mesh_m["has_both"]
                vol_extra = {
                    "volume_source": "mesh_surface",
                    **{k: mesh_m[k] for k in (
                        "mesh_pos_vox", "mesh_neg_vox", "mesh_center_vox",
                        "mesh_y_span_mm", "mesh_z_span_mm", "mesh_pinch_core_vox",
                        "has_mesh_volume", "mesh_entity_span_ok",
                    )},
                    "mesh_surf_min_mm": s_min,
                    "gripper_half_mm": _GRIPPER_HALF_MM,
                    "gripper_len_mm": _GRIPPER_LEN_M * 1000.0,
                    "mesh_surf_reach_ok": s_ok,
                    "mesh_pinch_near_pts": int(mesh_m.get("mesh_pinch_core_vox", 0)),
                }
            else:
                if n_g > 0:
                    ys = q_gap_np[:, 1]
                    n_pos = int((ys > 0.004).sum())
                    n_neg = int((ys < -0.004).sum())
                else:
                    n_pos = n_neg = 0
                has_both = n_pos >= MIN_GAP_EACH_SIDE and n_neg >= MIN_GAP_EACH_SIDE
                gap_voxel_n, gap_vol_cm3, n_gap_raw = _gap_voxel_metrics(q_gap_np)
                has_volume = (
                    gap_voxel_n >= MIN_GAP_VOXELS
                    and n_g >= MIN_GAP_POINTS
                    and has_both
                )
                vol_extra = {"volume_source": "depth"}

            eef_np = eef[bi].detach().cpu().numpy()
            ok, reason = _is_shoulder_reachable(eef_np, shoulder)
            gap_c = _gap_world_from_eef(
                meta["R"] if "R" in meta else np.eye(3), eef_np,
            )
            ref = np.asarray(ref_pt, dtype=np.float64).reshape(3)
            dist_pts = (
                surf_src if use_mesh_vol and len(surf_src) >= 8
                else (vol_src if use_mesh_vol else score_pcd)
            )
            gap_dist = _min_gap_dist_to_mesh_mm(gap_c, dist_pts)
            inside = _gap_on_object_ok(
                gap_c, ref, obj_aabb, hit_anchor=hit_anchor,
                mesh_surf_pts=surf_src if use_mesh_vol else None,
                is_boundary=(meta.get("approach_label") == "boundary"),
            )
            pen_i = pen_mask[bi]
            ncol_pts = int(pen_i.sum().item())
            ncol_near_n = int(near_col[bi].sum().item())
            if ncol_pts > 0:
                q_pen_np = q_col[bi, pen_i].detach().cpu().numpy()
                vox = np.floor(q_pen_np / _NCOL_VOXEL_M).astype(np.int64)
                ncol_exp = int(np.unique(vox, axis=0).shape[0])
            else:
                ncol_exp = 0
            out.append({
                **meta,
                "pos": eef_np,
                "gap_center": gap_c,
                "gap_dist_to_obj_mm": gap_dist * 1000.0,
                "gap_on_object": inside,
                "n_gap": n_g,
                "n_gap_raw": n_gap_raw,
                "gap_voxel_n": gap_voxel_n,
                "gap_vol_cm3": gap_vol_cm3,
                "has_volume": has_volume,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "has_both": has_both,
                "ncol_exp": int(ncol_exp),
                "ncol_pts": int(ncol_pts),
                "ncol_near_n": int(ncol_near_n),
                "base_dist_mm": float(base_dist_mm[bi].item()),
                "feasible": (
                    has_volume and inside and int(ncol_exp) <= _NCOL_SAFE_TOL
                ),
                "reachable": ok,
                "reach_reason": reason,
                **vol_extra,
                **rail,
            })
        step = min(start + _BATCH_POSE_CHUNK, len(poses))
        if step == _BATCH_POSE_CHUNK or step % _LOG_PROGRESS_EVERY == 0 or step == len(poses):
            n_vol = sum(1 for c in out if c.get("has_volume"))
            _grasp_obj_report(
                ctx,
                f"  [grasp_obj] 批量评估 {step}/{len(poses)} "
                f"has_volume={n_vol}",
                status=(step % _LOG_PROGRESS_EVERY == 0 or step == len(poses)),
            )
    return out


def _extract_object_surface_pool(
    world,
    object_name: str,
    n_samples: int,
    rng: np.random.Generator,
    *,
    obj_ref: Optional[np.ndarray] = None,
    obj_aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ctx=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """从 USD mesh 面积加权采样外表面点 + 朝外法向（世界系）。"""
    from behavior_interface.skills.grasp import _resolve_object_handle

    obj = _resolve_object_handle(world, object_name)
    if obj is None:
        return np.zeros((0, 3)), np.zeros((0, 3))
    try:
        import omnigibson as og
        import omnigibson.lazy as lazy

        pxr = lazy.pxr
        UsdGeom = pxr.UsdGeom
        stage = og.sim.stage
        prim_path = getattr(obj, "prim_path", None) or f"/World/{obj.name}"
        root = stage.GetPrimAtPath(str(prim_path))
        if not root.IsValid():
            return np.zeros((0, 3)), np.zeros((0, 3))
        mesh_prims = _visual_mesh_prims(root, pxr, UsdGeom)
        if not mesh_prims:
            return np.zeros((0, 3)), np.zeros((0, 3))

        n_mesh = len(mesh_prims)
        per_mesh = max(400, int(n_samples // max(1, n_mesh)))
        pts_all: List[np.ndarray] = []
        nrm_all: List[np.ndarray] = []
        for prim in mesh_prims:
            mesh = UsdGeom.Mesh(prim)
            local_pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            if len(local_pts) < 3:
                continue
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            tris = _triangulate_face_indices(counts, indices)
            if len(tris) == 0:
                continue
            world_pts, _ = _mesh_local_to_world(local_pts, prim)
            pts, face_nrms = _sample_points_on_triangles(
                world_pts, tris, per_mesh, rng, return_normals=True,
            )
            from omnigibson.utils.usd_utils import PoseAPI

            wmat = np.asarray(
                PoseAPI.get_world_pose_with_scale(str(prim.GetPath())).detach().cpu().numpy(),
                dtype=np.float64,
            )
            nrms = face_nrms @ wmat[:3, :3].T
            nn = np.linalg.norm(nrms, axis=1, keepdims=True) + 1e-12
            nrms = nrms / nn
            pts_all.append(pts)
            nrm_all.append(nrms)

        if not pts_all:
            return np.zeros((0, 3)), np.zeros((0, 3))
        pts = np.vstack(pts_all)
        nrms = np.vstack(nrm_all)
        ref = _obj_ref_for_region(obj_ref, obj_aabb, pts.mean(axis=0))
        m = _object_region_mask(pts, ref, obj_aabb, margin=0.02)
        if int(m.sum()) >= 8:
            pts, nrms = pts[m], nrms[m]
        outward = _filter_outward_surface_mask(pts, nrms, ref)
        if int(outward.sum()) >= 8:
            pts, nrms = pts[outward], nrms[outward]
        elif ctx:
            ctx.log(f"  [grasp_obj] WARN mesh朝外筛选不足，保留区域裁剪点 {len(pts)}")
        if ctx:
            ctx.log(f"  [grasp_obj] mesh外表面候选 {len(pts)} 点 path={prim_path}")
        return pts, nrms
    except Exception as e:
        if ctx:
            ctx.log(f"  [grasp_obj] mesh外表面采样失败: {e}")
        return np.zeros((0, 3)), np.zeros((0, 3))


def _fps_subsample_boundary(
    pts: np.ndarray,
    nrms: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    start_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """最远点采样，使边界点在物体表面空间均匀。"""
    n = min(n, len(pts))
    if n <= 0:
        return pts[:0], nrms[:0]
    if n >= len(pts):
        return pts, nrms
    start = int(start_idx) if start_idx is not None else int(rng.integers(len(pts)))
    start = max(0, min(start, len(pts) - 1))
    chosen = [start]
    dists = np.full(len(pts), np.inf, dtype=np.float64)
    for _ in range(n - 1):
        last = pts[chosen[-1]]
        d = np.linalg.norm(pts - last, axis=1)
        dists = np.minimum(dists, d)
        chosen.append(int(np.argmax(dists)))
    idx = np.asarray(chosen, dtype=np.int64)
    return pts[idx], nrms[idx]


def _filter_upper_boundary_mask(
    pts: np.ndarray,
    normals: np.ndarray,
    *,
    min_normal_z: float = 0.28,
) -> np.ndarray:
    """上表面边界：法向朝上且 z 位于物体上沿区域。"""
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    z_med = float(np.percentile(pts[:, 2], 62))
    return (normals[:, 2] >= min_normal_z) & (pts[:, 2] >= z_med)


def _filter_outward_surface_mask(
    pts: np.ndarray,
    normals: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """剔除内壁/朝内法向：点相对参考中心在法向外侧。"""
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    ref = np.asarray(ref, dtype=np.float64).reshape(3)
    to_pt = pts - ref.reshape(1, 3)
    return np.einsum("ij,ij->i", normals, to_pt) > _OUTWARD_NORMAL_MIN_DOT


def _filter_camera_facing_mask(
    pts: np.ndarray,
    normals: np.ndarray,
    cam_pos: np.ndarray,
    *,
    min_dot: float = _CAMERA_FACING_MIN_DOT,
) -> np.ndarray:
    """表面须朝向相机（开口袋剔除背对相机的内壁三角面）。"""
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    view = cam_pos.reshape(1, 3) - pts
    vn = np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
    view_u = view / vn
    return np.einsum("ij,ij->i", normals, view_u) > float(min_dot)


def _load_session_head_view(
    session: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any], int, int]:
    """读取 capture 冻结 head 视角 depth/seg/内外参。"""
    import json
    import os
    from behavior_interface.skills.plan_grasp_core import session_dir

    sid = session["session_id"]
    init_dir = session.get("init_dir") or os.path.join(session_dir(sid), "init")
    view = session.get("view", "head")
    meta_path = os.path.join(init_dir, f"camera_meta_{view}.json")
    if not os.path.isfile(meta_path):
        return None, None, {}, 0, 0
    with open(meta_path, encoding="utf-8") as f:
        cam_meta = json.load(f)
    depth_path = os.path.join(init_dir, "depth.npy")
    seg_path = os.path.join(init_dir, "seg.npy")
    depth = np.load(depth_path).astype(np.float64) if os.path.isfile(depth_path) else None
    seg = np.load(seg_path) if os.path.isfile(seg_path) else None
    from behavior_interface.head_capture import HEAD_IMAGE_HEIGHT, HEAD_IMAGE_WIDTH

    w = int(session.get("image_width") or cam_meta.get("image_width") or HEAD_IMAGE_WIDTH)
    h = int(session.get("image_height") or cam_meta.get("image_height") or HEAD_IMAGE_HEIGHT)
    return depth, seg, cam_meta, w, h


def _depth_visible_at_head(
    pt: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    depth: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    *,
    tol_m: float = 0.028,
) -> bool:
    """head 视角深度一致性：投影点深度与 3D 点距离相机一致。"""
    from behavior_interface.skills.vlm_grasp_verify import _world_to_pixel

    px = _world_to_pixel(cam_pos, cam_quat, pt, w, h, fl, ha)
    if px is None:
        return False
    u, v = int(px[0]), int(px[1])
    if u < 0 or v < 0 or v >= depth.shape[0] or u >= depth.shape[1]:
        return False
    d_val = float(depth[v, u])
    if not (np.isfinite(d_val) and 0.05 < d_val < 50.0):
        return False
    d_pt = float(np.linalg.norm(np.asarray(pt, dtype=np.float64) - cam_pos))
    return abs(d_pt - d_val) <= tol_m


def _sample_verified_boundary_points(
    world,
    object_name: str,
    session: Dict[str, Any],
    rng: np.random.Generator,
    *,
    n_points: int = _N_BOUNDARY_POINTS,
    obj_ref: Optional[np.ndarray] = None,
    obj_aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ctx=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """200 个上表面边界点：mesh 外表面 FPS + 深度预筛 + 黄球可见性（与点击无关）。"""
    cache_key = f"{GRASP_OBJ_BUILD}:{object_name}:{session.get('session_id', '')}"
    if cache_key in _BOUNDARY_POINTS_CACHE:
        cp, cn = _BOUNDARY_POINTS_CACHE[cache_key]
        if len(cp) >= n_points and ctx:
            ctx.log(f"  [grasp_obj] 边界点缓存命中 {len(cp)} pts key={cache_key}")
        return cp[:n_points], cn[:n_points]

    pts, nrms = _extract_object_surface_pool(
        world, object_name, _BOUNDARY_SURFACE_CANDIDATES, rng,
        obj_ref=obj_ref, obj_aabb=obj_aabb, ctx=ctx,
    )
    if len(pts) < n_points:
        if ctx:
            ctx.log(f"  [grasp_obj] WARN 外表面点不足 {len(pts)}<{n_points}")
        return pts, nrms

    depth, seg, cam_meta, w, h = _load_session_head_view(session)
    cam_pos = np.asarray(cam_meta.get("cam_pos", [0, 0, 0]), dtype=np.float64)
    cam_quat = np.asarray(cam_meta.get("cam_quat_xyzw", [0, 0, 0, 1]), dtype=np.float64)
    from behavior_interface.head_capture import HEAD_FOCAL_LENGTH, HEAD_HORIZONTAL_APERTURE

    fl = float(cam_meta.get("focal_length", HEAD_FOCAL_LENGTH))
    ha = float(cam_meta.get("horizontal_aperture", HEAD_HORIZONTAL_APERTURE))
    ref = _obj_ref_for_region(obj_ref, obj_aabb, pts.mean(axis=0))

    outward = _filter_outward_surface_mask(pts, nrms, ref)
    facing = _filter_camera_facing_mask(pts, nrms, cam_pos)
    outer_face = outward & facing
    if int(outer_face.sum()) >= n_points // 2:
        pts, nrms = pts[outer_face], nrms[outer_face]
        if ctx:
            ctx.log(
                f"  [grasp_obj] mesh外表面筛 outward+facing → {len(pts)} "
                f"(raw={len(outward)} facing={int(facing.sum())})"
            )
    elif ctx:
        ctx.log(
            f"  [grasp_obj] WARN outward+facing 不足 {int(outer_face.sum())}，"
            f"仅 outward={int(outward.sum())}"
        )
        if int(outward.sum()) >= n_points // 4:
            pts, nrms = pts[outward], nrms[outward]

    upper = _filter_upper_boundary_mask(pts, nrms)
    pts_u, nrms_u = pts[upper], nrms[upper]
    if len(pts_u) < n_points:
        pts_u, nrms_u = pts, nrms
        if ctx:
            ctx.log(f"  [grasp_obj] WARN 上表面不足，回退全表面 {len(pts_u)} 点")

    pre_idx: List[int] = []
    for i, p in enumerate(pts_u):
        if depth is not None and _depth_visible_at_head(
            p, cam_pos, cam_quat, depth, w, h, fl, ha,
        ):
            pre_idx.append(i)
    if len(pre_idx) < n_points:
        # 深度不可用或太严：按 z 降序取上沿点
        order = np.argsort(-pts_u[:, 2])
        pre_idx = order[: max(n_points * 3, n_points)].tolist()
        if ctx:
            ctx.log(
                f"  [grasp_obj] WARN 深度预筛仅 {len(pre_idx)}，改用 z 上沿排序"
            )

    pool_pts = pts_u[np.asarray(pre_idx, dtype=np.int64)]
    pool_nrms = nrms_u[np.asarray(pre_idx, dtype=np.int64)]
    if len(pool_pts) > n_points * 3:
        pool_pts, pool_nrms = _fps_subsample_boundary(
            pool_pts, pool_nrms, min(len(pool_pts), n_points * 4), rng,
        )
    order = list(range(len(pool_pts)))
    rng.shuffle(order)
    verified_pts: List[np.ndarray] = []
    verified_nrms: List[np.ndarray] = []
    for i in order:
        if len(verified_pts) >= n_points:
            break
        p = pool_pts[i]
        n = pool_nrms[i]
        # 仅用 capture 冻结深度预筛；不移动 head 传感器、不做 RTX 黄球重渲染
        if depth is not None and not _depth_visible_at_head(
            p, cam_pos, cam_quat, depth, w, h, fl, ha,
        ):
            continue
        verified_pts.append(p)
        verified_nrms.append(n)

    if len(verified_pts) < n_points:
        for i in order:
            if len(verified_pts) >= n_points:
                break
            p = pool_pts[i]
            if any(np.linalg.norm(p - v) < 1e-5 for v in verified_pts):
                continue
            verified_pts.append(p)
            verified_nrms.append(pool_nrms[i])

    out_pts = np.asarray(verified_pts[:n_points], dtype=np.float64)
    out_nrms = np.asarray(verified_nrms[:n_points], dtype=np.float64)
    if len(out_pts) >= min(8, n_points):
        _BOUNDARY_POINTS_CACHE[cache_key] = (out_pts.copy(), out_nrms.copy())
    if ctx:
        ctx.log(
            f"  [grasp_obj] 边界点 {len(out_pts)}/{n_points} "
            f"(上表面池={len(pts_u)} 深度预筛={len(pre_idx)}) "
            f"mean={out_pts.mean(axis=0).round(3).tolist() if len(out_pts) else '?'}"
        )
    return out_pts, out_nrms


def _perturb_rotation_matrix(R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """对基础姿态施加小角度随机扰动（等效四元数抖动）。"""
    ax = rng.normal(0.0, math.radians(_BOUNDARY_APPROACH_JITTER_DEG), size=3)
    angle = float(np.linalg.norm(ax))
    if angle < 1e-6:
        return R
    axis = ax / angle
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = axis
    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ], dtype=np.float64)
    dR = np.eye(3) + s * K + (1.0 - c) * (K @ K)
    return dR @ R


def _approach_directions_for_boundary(
    outward_normal: np.ndarray,
    n_dirs: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """每边界点 30 个 approach：主方向沿 -法向，其余在锥内随机。"""
    n_out = np.asarray(outward_normal, dtype=np.float64)
    n_out = n_out / (np.linalg.norm(n_out) + 1e-9)
    center = -n_out
    dirs: List[np.ndarray] = []
    sphere = _fibonacci_sphere(max(n_dirs, 8))
    for j in range(n_dirs):
        if j == 0:
            d = center.copy()
        elif j < n_dirs // 2:
            d = center + 0.32 * sphere[j % len(sphere)]
        else:
            d = center + 0.55 * rng.normal(size=3)
        d = d / (np.linalg.norm(d) + 1e-9)
        dirs.append(d)
    return dirs


def _sample_boundary_grasp_poses(
    boundary_pts: np.ndarray,
    boundary_normals: np.ndarray,
    rng: np.random.Generator,
    *,
    n_per_point: int = _N_GRASPS_PER_BOUNDARY,
    ref_ys: Optional[List[Tuple[str, Optional[np.ndarray]]]] = None,
) -> List[Tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
    """
    边界点 → N×M pose。

    mesh 表面点 surf_pt 锁为对称轴 1/3 锚点（两指中间、靠基座 1/3）；
    均匀随机夹爪朝向 R，eef = surf_pt - R @ gap_anchor。
    返回 (ori_hint, roll, surf_pt, R_matrix)。
    """
    _ = boundary_normals, ref_ys
    samples: List[Tuple[np.ndarray, float, np.ndarray, np.ndarray]] = []
    n_bp = len(boundary_pts)
    if n_bp == 0:
        return samples

    sphere = _fibonacci_sphere(max(64, n_per_point))
    for bi in range(n_bp):
        surf_pt = boundary_pts[bi]
        for j in range(n_per_point):
            if rng.random() < 0.15:
                approach = rng.normal(size=3)
            else:
                approach = sphere[j % len(sphere)].copy()
            approach = approach / (np.linalg.norm(approach) + 1e-9)
            roll = float(rng.uniform(0, 2.0 * math.pi))
            R = _eef_frame_from_approach_roll(approach, roll, ref_y=None)
            samples.append((approach, roll, surf_pt.copy(), R))
    return samples


def _sample_random_poses(
    pcd: np.ndarray,
    n_long: np.ndarray,
    n_short: np.ndarray,
    rng: np.random.Generator,
    n_total: int,
    hit_anchor: Optional[np.ndarray] = None,
    *,
    obj_ref: Optional[np.ndarray] = None,
    obj_aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> List[Tuple[np.ndarray, float, np.ndarray]]:
    """返回 (ori_hint, roll, gap_anchor) 列表；均匀球面朝向 + roll，锚点在物体点云内。"""
    if obj_ref is not None:
        m = _object_region_mask(pcd, obj_ref, obj_aabb)
        pool = pcd[m] if int(m.sum()) >= 8 else pcd
    else:
        pool = pcd

    if hit_anchor is not None:
        from behavior_interface.skills.plan_grasp_gripper_fit import _gap_anchor_points

        ha = np.asarray(hit_anchor, dtype=np.float64).reshape(3)
        anchor_pool = _gap_anchor_points(ha, pool)
        if len(anchor_pool) < n_total // 2:
            near = pool[np.linalg.norm(pool - ha, axis=1) < 0.12]
            if len(near) >= 4:
                anchor_pool = np.vstack([anchor_pool, near])
        anchors = anchor_pool[rng.choice(len(anchor_pool), size=min(n_total, len(anchor_pool)), replace=True)]
    else:
        anchors = pool[rng.choice(len(pool), size=min(n_total, len(pool)), replace=True)]
    if len(anchors) < n_total:
        center = (
            np.asarray(obj_ref, dtype=np.float64).reshape(3)
            if obj_ref is not None
            else (np.asarray(hit_anchor, dtype=np.float64).reshape(3) if hit_anchor is not None else pool.mean(axis=0))
        )
        extra = center + rng.normal(0, 0.008, size=(n_total - len(anchors), 3))
        anchors = np.vstack([anchors, extra])

    directions = _fibonacci_sphere(max(32, n_total // 64))
    samples: List[Tuple[np.ndarray, float, str, np.ndarray]] = []
    for i in range(n_total):
        if rng.random() < 0.18:
            approach = rng.normal(size=3)
            approach /= np.linalg.norm(approach) + 1e-9
        else:
            approach = directions[i % len(directions)]
            approach = approach / (np.linalg.norm(approach) + 1e-9)
        roll = float(rng.uniform(0, 2 * math.pi))
        anchor = anchors[i % len(anchors)].copy()
        # 沿物体尺度随机抖动锚点
        anchor += rng.normal(0, 0.008, size=3)
        samples.append((approach, roll, anchor))
    return samples


def _build_candidate(
    approach: np.ndarray,
    roll: float,
    ref_y_label: str,
    ref_y: Optional[np.ndarray],
    anchor: np.ndarray,
    pcd: np.ndarray,
    shoulder: np.ndarray,
    alabel: str = "random",
    *,
    vol_pcd: Optional[np.ndarray] = None,
    col_pcd: Optional[np.ndarray] = None,
    surf_pcd: Optional[np.ndarray] = None,
    use_mesh_volume: bool = False,
) -> Dict[str, Any]:
    from behavior_interface.skills.plan_grasp_gripper_geom import gap_anchor_local_one_third_from_base

    R = _eef_frame_from_approach_roll(approach, roll, ref_y=ref_y)
    eef_pos = anchor - R @ gap_anchor_local_one_third_from_base()
    ev = _evaluate_grasp_object_pose(
        R, eef_pos, pcd,
        vol_pcd=vol_pcd, col_pcd=col_pcd, surf_pcd=surf_pcd,
        use_mesh_volume=use_mesh_volume,
    )
    ok, reason = _is_shoulder_reachable(eef_pos, shoulder)
    gap_w = _gap_world_from_eef(R, eef_pos)
    inside = _gap_on_object_ok(
        gap_w, anchor, mesh_surf_pts=surf_pcd if use_mesh_volume else None,
        is_boundary=(alabel == "boundary"),
    )
    return {
        "pos": eef_pos,
        "quat": _mat_to_quat_xyzw(R),
        "approach": R[:, 2].copy(),
        "roll_deg": math.degrees(roll),
        "approach_label": alabel,
        "ref_y_label": ref_y_label,
        "gap_center": gap_w,
        "gap_on_object": inside,
        "gap_dist_to_obj_mm": _min_gap_dist_to_mesh_mm(
            gap_w,
            surf_pcd if use_mesh_volume and surf_pcd is not None else (
                vol_pcd if use_mesh_volume and vol_pcd is not None else pcd
            ),
        ),
        "reachable": ok,
        "reach_reason": reason,
        **ev,
    }


def _refine_candidate(
    cand: Dict[str, Any],
    pcd: np.ndarray,
    shoulder: np.ndarray,
    rng: np.random.Generator,
    *,
    vol_pcd: Optional[np.ndarray] = None,
    col_pcd: Optional[np.ndarray] = None,
    surf_pcd: Optional[np.ndarray] = None,
    use_mesh_volume: bool = False,
) -> Dict[str, Any]:
    """对 top 候选做小幅位置抖动；边界模式沿开口平面移锚点并保持贴在物体上。"""
    from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat
    from behavior_interface.skills.plan_grasp_gripper_geom import gap_anchor_local_one_third_from_base

    gap_local = gap_anchor_local_one_third_from_base()
    R = _quat_to_mat(np.asarray(cand["quat"]))
    best = _sync_pose_gap_center(cand)
    eef0 = np.asarray(best["pos"], dtype=np.float64)
    anchor0 = np.asarray(best["gap_center"], dtype=np.float64)
    is_boundary = best.get("approach_label") == "boundary"
    lat = R[:, 0]
    open_y = R[:, 1]
    for _ in range(_N_REFINE):
        if is_boundary:
            anchor = (
                anchor0
                + rng.normal(0, _REFINE_JITTER_M * 0.6) * lat
                + rng.normal(0, _REFINE_JITTER_M * 0.6) * open_y
            )
            eef_pos = anchor - R @ gap_local
        else:
            jitter = (
                rng.normal(0, _REFINE_JITTER_M) * open_y
                + rng.normal(0, _REFINE_JITTER_M) * R[:, 2]
                + rng.normal(0, _REFINE_JITTER_M * 0.5, size=3)
            )
            eef_pos = eef0 + jitter
            anchor = _gap_world_from_eef(R, eef_pos)
        ev = _evaluate_grasp_object_pose(
            R, eef_pos, pcd,
            vol_pcd=vol_pcd, col_pcd=col_pcd, surf_pcd=surf_pcd,
            use_mesh_volume=use_mesh_volume,
        )
        ok, reason = _is_shoulder_reachable(eef_pos, shoulder)
        trial = {
            **best,
            "pos": eef_pos,
            "gap_center": anchor,
            "reachable": ok,
            "reach_reason": reason,
            **ev,
        }
        if is_boundary and surf_pcd is not None and len(surf_pcd) >= 8:
            anchor = _project_anchor_to_boundary_mesh(anchor, surf_pcd)
            eef_pos = anchor - R @ gap_local
            if not _gap_on_object_ok(
                anchor, anchor, mesh_surf_pts=surf_pcd, is_boundary=True,
            ):
                continue
        if _pose_rank_key(trial) < _pose_rank_key(best):
            best = trial
    return _sync_pose_gap_center(best)


def _rot_mat_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=np.float64).reshape(3)
    ax = ax / (float(np.linalg.norm(ax)) + 1e-9)
    kx, ky, kz = ax
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _refine_roll_for_tuck_align(
    cand: Dict[str, Any],
    tuck_quat: np.ndarray,
    pcd: np.ndarray,
    shoulder: np.ndarray,
    *,
    vol_pcd: Optional[np.ndarray] = None,
    col_pcd: Optional[np.ndarray] = None,
    surf_pcd: Optional[np.ndarray] = None,
    use_mesh_volume: bool = False,
    n_roll: int = 16,
) -> Dict[str, Any]:
    """固定 gap_center，绕夹爪指向轴扫 roll，在保持体积前提下对齐胸前指向。"""
    from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat

    R0 = _quat_to_mat(np.asarray(cand["quat"]))
    anchor = np.asarray(cand["gap_center"], dtype=np.float64).reshape(3)
    axis = R0[:, 2].copy()
    best = cand
    best_key = (_pointing_delta_deg(cand["quat"], tuck_quat), _pose_rank_key(cand))

    for k in range(max(4, n_roll)):
        angle = 0.0 if k == 0 else (2.0 * math.pi * k / n_roll)
        R = _rot_mat_about_axis(axis, angle) @ R0
        from behavior_interface.skills.plan_grasp_gripper_geom import gap_anchor_local_one_third_from_base

        eef_pos = anchor - R @ gap_anchor_local_one_third_from_base()
        ev = _evaluate_grasp_object_pose(
            R, eef_pos, pcd,
            vol_pcd=vol_pcd, col_pcd=col_pcd, surf_pcd=surf_pcd,
            use_mesh_volume=use_mesh_volume,
        )
        if not ev.get("has_volume") or not ev.get("gap_on_object", True):
            continue
        ok, reason = _is_shoulder_reachable(eef_pos, shoulder)
        trial = {
            **cand,
            "pos": eef_pos,
            "quat": _mat_to_quat_xyzw(R),
            "approach": R[:, 2].copy(),
            "reachable": ok,
            "reach_reason": reason,
            **ev,
        }
        key = (_pointing_delta_deg(trial["quat"], tuck_quat), _pose_rank_key(trial))
        if key < best_key:
            best_key = key
            best = trial
    return _sync_pose_gap_center(best)


def _grasp_obj_report(ctx, msg: str, *, status: bool = False) -> None:
    """写 grasp_obj 规划进度到 web Log；可选同步 skill 状态栏。"""
    if ctx is None:
        return
    ctx.log(msg)
    if status and hasattr(ctx, "set_status"):
        ctx.set_status(msg.strip())


def _pick_best_pool(
    candidates: List[Dict[str, Any]],
    shoulder: np.ndarray,
    rng: np.random.Generator,
    ctx=None,
    *,
    boundary_mode: bool = False,
) -> Dict[str, Any]:
    """无穿模优先 + 开口体积最大；无 ncol=0 时软退化，禁止搜索失败。"""
    if not candidates:
        raise ValueError("无候选 pose")

    ncol0_n = sum(1 for c in candidates if _ncol_ok(c))
    pool = _sort_best_effort_pool(candidates)
    pool_tag = "ncol=0" if ncol0_n else "best_effort_min_ncol"
    if ncol0_n == 0 and ctx:
        min_nc = min(c.get("ncol_exp", 999) for c in candidates)
        ctx.log(
            f"  [grasp_obj] WARN 无 ncol=0，软退化选 min_ncol_vox={min_nc} "
            f"候选={len(candidates)}"
        )
    reach_pool = [c for c in pool if _scalar_bool_field(c, "reachable")] or pool
    chosen = reach_pool[0]
    global_max_vox = max(_scalar_int_field(c, "gap_voxel_n", 0) for c in candidates)

    if ctx:
        ctx.log(
            f"  [grasp_obj] 采样={len(candidates)} ncol0={ncol0_n} "
            f"pool={pool_tag}({len(pool)}) global_max_vox={global_max_vox} "
            f"pick vox={chosen.get('gap_voxel_n', 0)} "
            f"vol={chosen.get('gap_vol_cm3', 0):.2f}cm³ "
            f"src={chosen.get('volume_source', '?')} "
            f"ncol={chosen.get('ncol_exp', 0)} "
            f"reach={chosen.get('reachable')}({chosen.get('reach_reason', '')})"
        )
    return chosen


def _eef_pose_at_arm_qpos(
    world, arm: str, arm_q: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """短暂把手臂设到指定位形并读 EEF pose，随后恢复。"""
    from behavior_interface.skills.grasp import _get_arm_dof_idx

    saved = None
    robot = None
    try:
        robot = world.robot
        idx = _get_arm_dof_idx(world, arm)
        saved = robot.get_joint_positions().clone()
        q = saved.clone()
        aq = np.asarray(arm_q, dtype=np.float64).reshape(7)
        for i, j in enumerate(idx):
            q[int(j)] = float(aq[i])
        robot.set_joint_positions(q)
        eef = world.eef_pose(arm=arm)
        return (
            np.asarray(eef["pos"], dtype=np.float64).reshape(3),
            np.asarray(eef["quat"], dtype=np.float64).reshape(4),
        )
    except Exception:
        return None
    finally:
        if saved is not None and robot is not None:
            try:
                robot.set_joint_positions(saved)
            except Exception:
                pass


def _pointing_delta_deg(quat_a, quat_b) -> float:
    """两 EEF 姿态局部 +Z（夹爪指向）夹角（度）。"""
    from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat

    za = _quat_to_mat(np.asarray(quat_a))[:, 2]
    zb = _quat_to_mat(np.asarray(quat_b))[:, 2]
    za = za / (float(np.linalg.norm(za)) + 1e-9)
    zb = zb / (float(np.linalg.norm(zb)) + 1e-9)
    c = float(np.clip(np.dot(za, zb), -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def _scalar_int_field(c: Dict[str, Any], key: str, default: int = 0) -> int:
    """候选 dict 字段 → Python int（避免 gap_voxel_n 为 tensor 时排序崩溃）。"""
    from behavior_interface.skills.grasp_obj_pipeline_core import _py_int
    return _py_int(c.get(key, default), default)


def _scalar_bool_field(c: Dict[str, Any], key: str, default: bool = False) -> bool:
    from behavior_interface.skills.grasp_obj_pipeline_core import _py_bool
    v = c.get(key, default)
    return _py_bool(v) if v is not None else bool(default)


def _exec_path_rank_key(
    c: Dict[str, Any],
    tuck_pose: Optional[Tuple[np.ndarray, np.ndarray]],
    back_m: float,
    *,
    boundary_mode: bool = False,
) -> tuple:
    """exec 可达：边界模式优先左右均衡；否则体积优先。"""
    if tuck_pose is None:
        pd, tuck_dist = 999.0, 999.0
    else:
        tuck_pos, tuck_quat = tuck_pose
        pd = _pointing_delta_deg(c["quat"], tuck_quat)
        from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat

        pos = np.asarray(c["pos"], dtype=np.float64).reshape(3)
        pointing = _quat_to_mat(np.asarray(c["quat"]))[:, 2]
        pointing = pointing / (float(np.linalg.norm(pointing)) + 1e-9)
        safe = pos - float(back_m) * pointing
        tuck_dist = float(np.linalg.norm(safe - np.asarray(tuck_pos).reshape(3)))
    if boundary_mode:
        return (
            0 if _scalar_bool_field(c, "has_volume") else 1,
            _scalar_int_field(c, "ncol_exp", 999),
            float(c.get("y_balance", 1.0)),
            float(c.get("center_bias_mm", 999.0)),
            pd,
            -_scalar_int_field(c, "gap_voxel_n", 0),
            tuck_dist,
        )
    return (
        0 if _scalar_bool_field(c, "has_volume") else 1,
        -_scalar_int_field(c, "gap_voxel_n", 0),
        -float(c.get("gap_vol_cm3", 0.0)),
        _scalar_int_field(c, "ncol_exp", 999),
        pd,
        -_scalar_int_field(c, "mesh_pinch_core_vox", 0),
        tuck_dist,
        float(c.get("gap_dist_to_obj_mm", 999.0)),
        float(c.get("mesh_surf_min_mm", 999.0)),
    )


def _grasp_obj_filter_constants() -> Dict[str, Any]:
    """导出 plan 筛选链用到的阈值，供人工审计。"""
    return {
        "grasp_obj_build": GRASP_OBJ_BUILD,
        "filter_priority": list(GRASP_OBJ_FILTER_PRIORITY),
        "gap_geom": "anchor_contact_z_v2_surface_vol_best_effort",
        "sample_mode": "boundary_200x30",
        "n_boundary_points": _N_BOUNDARY_POINTS,
        "n_grasps_per_boundary": _N_GRASPS_PER_BOUNDARY,
        "n_random_poses": _N_RANDOM_POSES,
        "n_refine": _N_REFINE,
        "refine_jitter_m": _REFINE_JITTER_M,
        "ncol_safe_tol": _NCOL_SAFE_TOL,
        "ncol_proximity_mm": _NCOL_PROXIMITY_R * 1000.0,
        "ncol_voxel_mm": _NCOL_VOXEL_M * 1000.0,
        "gap_proximity_mm": _GAP_PROXIMITY_R * 1000.0,
        "ncol_col_source": "mesh_surface_only",
        "vol_proximity_mm": _VOL_PROXIMITY_R * 1000.0,
        "vol_source": "mesh_surface_only",
        "mesh_max_gap_y_span_mm": _MESH_MAX_GAP_Y_SPAN_M * 1000.0,
        "mesh_max_gap_z_span_mm": _MESH_MAX_GAP_Z_SPAN_M * 1000.0,
        "mesh_min_gap_voxels": _MESH_MIN_GAP_VOXELS,
        "mesh_min_side_vox": _MESH_MIN_SIDE_VOX,
        "mesh_min_center_vox": _MESH_MIN_CENTER_VOX,
        "mesh_min_pinch_core_vox": _MESH_MIN_PINCH_CORE_VOX,
        "mesh_min_y_span_mm": _MESH_MIN_Y_SPAN_M * 1000.0,
        "mesh_min_z_span_mm": _MESH_MIN_Z_SPAN_M * 1000.0,
        "gap_center_max_dist_mm": _GAP_CENTER_MAX_DIST_M * 1000.0,
        "boundary_mesh_anchor_max_mm": _BOUNDARY_MESH_ANCHOR_MAX_M * 1000.0,
        "min_pinch_near_pts": _MIN_PINCH_NEAR_PTS,
        "surf_reach_tol_mm": _SURF_REACH_TOL_MM,
        "gripper_half_mm": _GRIPPER_HALF_MM,
        "gripper_len_mm": _GRIPPER_LEN_M * 1000.0,
        "exec_back_m": _EXEC_BACK_M,
        "dls_probe_top": _DLS_PROBE_TOP,
        "roll_align_top": _ROLL_ALIGN_TOP,
        "probe_min_vox_frac": _PROBE_MIN_VOX_FRAC,
        "dls_safe_tol_mm": 100.0,
        "dls_cnt_tol_mm": 80.0,
        "dls_cnt_fallback_tol_mm": 130.0,
        "dls_min_pinch_probe": 2,
        "coarse_fallback_top": _COARSE_FALLBACK_TOP,
        "arm_min_reach_m": _ARM_MIN_REACH,
        "arm_max_reach_m": _ARM_MAX_REACH,
    }


def _candidate_audit_fields(c: Dict[str, Any], **extra) -> Dict[str, Any]:
    """单候选 pose 参与 filter 的字段（可 JSON 序列化）。"""
    def _arr(x):
        if x is None:
            return None
        a = np.asarray(x)
        return a.reshape(-1).tolist() if a.size else None

    out = {
        "pos": _arr(c.get("pos")),
        "quat": _arr(c.get("quat")),
        "gap_center": _arr(c.get("gap_center")),
        "gap_voxel_n": _scalar_int_field(c, "gap_voxel_n", 0),
        "gap_vol_cm3": float(c.get("gap_vol_cm3", 0.0)),
        "mesh_pinch_core_vox": _scalar_int_field(c, "mesh_pinch_core_vox", 0),
        "mesh_pos_vox": _scalar_int_field(c, "mesh_pos_vox", 0),
        "mesh_neg_vox": _scalar_int_field(c, "mesh_neg_vox", 0),
        "mesh_center_vox": _scalar_int_field(c, "mesh_center_vox", 0),
        "mesh_y_span_mm": float(c.get("mesh_y_span_mm", 0.0)),
        "mesh_z_span_mm": float(c.get("mesh_z_span_mm", 0.0)),
        "mesh_surf_min_mm": float(c.get("mesh_surf_min_mm", 0.0)),
        "mesh_surf_reach_ok": _scalar_bool_field(c, "mesh_surf_reach_ok"),
        "mesh_pinch_near_pts": _scalar_int_field(c, "mesh_pinch_near_pts", 0),
        "n_gap": _scalar_int_field(c, "n_gap", 0),
        "ncol_exp": _scalar_int_field(c, "ncol_exp", 0),
        "gap_on_object": _scalar_bool_field(c, "gap_on_object", True),
        "gap_dist_to_obj_mm": float(c.get("gap_dist_to_obj_mm", 0.0)),
        "has_volume": _scalar_bool_field(c, "has_volume"),
        "base_dist_mm": float(c.get("base_dist_mm", 0.0)),
        "y_balance": float(c.get("y_balance", 0.0)),
        "center_bias_mm": float(c.get("center_bias_mm", 0.0)),
        "min_rail_clear_mm": float(c.get("min_rail_clear_mm", 0.0)),
        "roll_deg": float(c.get("roll_deg", 0.0)),
        "ref_y_label": c.get("ref_y_label", ""),
        "volume_source": c.get("volume_source", ""),
        "reachable": _scalar_bool_field(c, "reachable"),
        "reach_reason": str(c.get("reach_reason", "")),
    }
    out.update(extra)
    return out


def _pick_exec_reachable_pool(
    candidates: List[Dict[str, Any]],
    world,
    arm: str,
    shoulder: np.ndarray,
    rng: np.random.Generator,
    ctx=None,
    *,
    back_m: float = _EXEC_BACK_M,
    max_probe: int = _DLS_PROBE_TOP,
    boundary_mode: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """在体积排序候选中，用同步 DLS 探针筛选 exec tuck→safe→contact 可达的 pose。

    与 exec_move 一致：safe 追 pos+quat，contact 仅追 pos。按夹爪指向与胸前对齐度排序，
    避免 to_safe 阶段为大姿态调整牺牲位置精度。
    """
    from behavior_interface.skills.grasp import _dls_probe_exec_path

    global_max_vox = max(_scalar_int_field(c, "gap_voxel_n", 0) for c in candidates)
    min_vox_probe = _min_vox_threshold(global_max_vox)
    ncol0_n = sum(1 for c in candidates if _ncol_ok(c))
    pool_cap = max(_DLS_PROBE_TOP * 4, 48)
    pool = _sort_best_effort_pool(candidates)[:pool_cap]
    pool_tag = "ncol=0" if ncol0_n else "best_effort_min_ncol"
    if ncol0_n == 0 and ctx:
        min_nc = min(c.get("ncol_exp", 999) for c in candidates)
        ctx.log(
            f"  [grasp_obj] WARN DLS池无 ncol=0，软退化 min_ncol_vox={min_nc} "
            f"候选={len(candidates)}"
        )
    vol_pool = [c for c in pool if _scalar_int_field(c, "gap_voxel_n", 0) >= min_vox_probe]
    if vol_pool:
        pool = vol_pool
        vol_tag = f"vox>={min_vox_probe}"
    else:
        vol_tag = "vox_any"

    chest_q = _CHEST_TUCK_ARM.get(arm, _CHEST_TUCK_ARM["right"])
    tuck_pose = _eef_pose_at_arm_qpos(world, arm, chest_q)

    pool.sort(key=lambda c: _exec_path_rank_key(
        c, tuck_pose, back_m, boundary_mode=boundary_mode,
    ))
    probe_n = min(max_probe, len(pool))

    if ctx:
        if tuck_pose is not None:
            tp, _ = tuck_pose
            tuck_s = np.asarray(tp).round(3).tolist()
        else:
            tuck_s = "?"
        ctx.log(
            f"  [grasp_obj] DLS探针 pool={pool_tag}({len(pool)}) {vol_tag} "
            f"probe≤{probe_n} global_max_vox={global_max_vox} tuck_eef={tuck_s}"
        )

    _SAFE_TOL = 0.10
    _CNT_TOL = 0.08
    _CNT_FALLBACK_TOL = 0.13
    _MIN_PINCH_PROBE = 2
    strict_pass: List[Tuple[Dict[str, Any], float, float, float, int]] = []
    safe_pass: List[Tuple[Dict[str, Any], float, float, float, int]] = []
    probe_results: List[Dict[str, Any]] = []

    audit_base: Dict[str, Any] = {
        "filter_constants": _grasp_obj_filter_constants(),
        "pool_tag": pool_tag,
        "pool_len": len(pool),
        "vol_tag": vol_tag,
        "global_max_vox": global_max_vox,
        "min_vox_probe": min_vox_probe,
        "probe_n": probe_n,
        "ncol0_in_pool": ncol0_n,
        "with_vol_len": len(candidates),
        "back_m": back_m,
        "tuck_eef": (
            np.asarray(tuck_pose[0]).round(4).tolist() if tuck_pose else None
        ),
        "tuck_quat": (
            np.asarray(tuck_pose[1]).round(4).tolist() if tuck_pose else None
        ),
    }

    def _finish(chosen: Dict[str, Any], pick_method: str, **pick_extra) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pd_pick = (
            _pointing_delta_deg(chosen["quat"], tuck_pose[1])
            if tuck_pose else None
        )
        audit = {
            **audit_base,
            "pick_method": pick_method,
            "pick": _candidate_audit_fields(
                chosen,
                pointing_delta_deg=pd_pick,
                **pick_extra,
            ),
            "probe_results": probe_results,
        }
        return chosen, audit

    for i, c in enumerate(pool[:probe_n]):
        if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
            ctx.raise_if_cancelled(f"DLS 探针 {i + 1}/{probe_n}")
        quat = np.asarray(c["quat"], dtype=np.float64).reshape(4)
        pd = _pointing_delta_deg(quat, tuck_pose[1]) if tuck_pose else 999.0
        _cancel = (
            (lambda: ctx.is_cancelled())
            if ctx is not None and hasattr(ctx, "is_cancelled") else None
        )
        safe_err, cnt_err = _dls_probe_exec_path(
            world, arm, c["pos"], quat.tolist(), chest_q,
            back_m=back_m, safe_tol=_SAFE_TOL, cnt_tol=_CNT_TOL,
            cancel_check=_cancel,
        )
        pinch = int(c.get("mesh_pinch_core_vox", 0))
        gvox = _scalar_int_field(c, "gap_voxel_n", 0)
        strict_ok = (
            safe_err <= _SAFE_TOL
            and cnt_err <= _CNT_TOL
            and gvox >= min_vox_probe
        )
        probe_results.append(_candidate_audit_fields(
            c,
            probe_rank=i + 1,
            pointing_delta_deg=round(pd, 2),
            dls_safe_mm=round(safe_err * 1000.0, 1),
            dls_cnt_mm=round(cnt_err * 1000.0, 1) if math.isfinite(cnt_err) else None,
            strict_pass=strict_ok,
            safe_pass=safe_err <= _SAFE_TOL and gvox >= min_vox_probe,
        ))
        if ctx:
            ctx.log(
                f"  [grasp_obj] DLS探针 rank={i+1} "
                f"vox={gvox} pinch={pinch} pd={pd:.0f}° "
                f"safe={safe_err*1000:.0f}mm cnt={cnt_err*1000:.0f}mm"
            )
        if safe_err <= _SAFE_TOL and gvox >= min_vox_probe:
            safe_pass.append((c, safe_err, cnt_err, pd, i + 1))
        if (
            safe_err <= _SAFE_TOL
            and cnt_err <= _CNT_TOL
            and gvox >= min_vox_probe
        ):
            strict_pass.append((c, safe_err, cnt_err, pd, i + 1))

    if strict_pass:
        strict_key = lambda x: (
            _scalar_int_field(x[0], "ncol_exp", 999),
            -_scalar_int_field(x[0], "gap_voxel_n", 0),
            0 if _scalar_bool_field(x[0], "mesh_entity_span_ok") else 1,
            float(x[0].get("y_balance", 1.0)),
            -x[3],
        )
        c, safe_err, cnt_err, pd, rk = min(strict_pass, key=strict_key)
        chosen = dict(c)
        chosen["reachable"] = True
        chosen["reach_reason"] = (
            f"dls_probe(rank={rk},pd={pd:.0f}°,vox={chosen['gap_voxel_n']},"
            f"safe={safe_err*1000:.0f}mm,cnt={cnt_err*1000:.0f}mm)"
        )
        if ctx:
            ctx.log(
                f"  [grasp_obj] DLS探针选中（严格通过"
                f"{'取左右最均衡' if boundary_mode else '取最大体积'}） rank={rk} "
                f"vox={chosen['gap_voxel_n']} pinch={chosen.get('mesh_pinch_core_vox', 0)} "
                f"pd={pd:.0f}° safe={safe_err*1000:.0f}mm cnt={cnt_err*1000:.0f}mm"
            )
        return _finish(
            chosen, "dls_strict_max_vox",
            probe_rank=rk, dls_safe_mm=round(safe_err * 1000, 1),
            dls_cnt_mm=round(cnt_err * 1000, 1),
        )

    if safe_pass:
        c, safe_err, cnt_err, pd, rk = min(
            safe_pass,
            key=lambda x: (
                _scalar_int_field(x[0], "ncol_exp", 999),
                -_scalar_int_field(x[0], "gap_voxel_n", 0),
                x[2],
                x[3],
            ),
        )
        chosen = dict(c)
        chosen["reachable"] = True
        chosen["reach_reason"] = (
            f"dls_fallback(rank={rk},pd={pd:.0f}°,"
            f"safe={safe_err*1000:.0f}mm,cnt={cnt_err*1000:.0f}mm)"
        )
        if ctx:
            ctx.log(
                f"  [grasp_obj] DLS fallback（safe过） rank={rk} "
                f"vox={chosen.get('gap_voxel_n', 0)} ncol={chosen.get('ncol_exp', 0)} "
                f"pd={pd:.0f}° safe={safe_err*1000:.0f}mm cnt={cnt_err*1000:.0f}mm"
            )
        return _finish(
            chosen, "dls_fallback_min_cnt",
            probe_rank=rk, dls_safe_mm=round(safe_err * 1000, 1),
            dls_cnt_mm=round(cnt_err * 1000, 1) if math.isfinite(cnt_err) else None,
        )

    # 最终软退化：DLS 全失败仍返回 pool 内 ncol 最小、vox 最大
    c = pool[0]
    chosen = dict(c)
    chosen["reachable"] = True
    chosen["reach_reason"] = (
        f"dls_degrade_best_effort(vox={chosen.get('gap_voxel_n', 0)},"
        f"ncol={chosen.get('ncol_exp', 0)})"
    )
    if ctx:
        ctx.log(
            f"  [grasp_obj] DLS 全失败，软退化 best_effort "
            f"vox={chosen.get('gap_voxel_n', 0)} ncol={chosen.get('ncol_exp', 0)}"
        )
    return _finish(chosen, "dls_degrade_best_effort")


def compute_eef_from_pcd_grasp_object(
    pcd: np.ndarray,
    shoulder: np.ndarray,
    *,
    hit_anchor: Optional[np.ndarray] = None,
    mesh_pcd: Optional[np.ndarray] = None,
    obj_ref: Optional[np.ndarray] = None,
    obj_aabb: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    seed: int = 42,
    n_random: Optional[int] = None,
    world=None,
    arm: str = "right",
    session: Optional[Dict[str, Any]] = None,
    object_name: Optional[str] = None,
    ctx=None,
    dual_arm_ik_filter: bool = False,
    plan_arm: str = "any",
) -> Optional[Dict[str, Any]]:
    """grasp_object v13：见模块 docstring 八步流程（unittest v3 口径）。"""
    shoulder = np.asarray(shoulder, dtype=np.float64).reshape(3)
    pcd_np = np.asarray(pcd, dtype=np.float64)

    # ── v13 主路径（world + object_name + session）────────────────
    if world is not None and object_name and session is not None:
        from behavior_interface.skills.grasp_obj_pipeline_core import (
            run_grasp_obj_filter_pipeline,
            run_grasp_obj_v12_pipeline,
        )

        if bool(dual_arm_ik_filter):
            result = run_grasp_obj_filter_pipeline(
                world, str(object_name), session, shoulder,
                seed=int(seed), arm=str(arm), ctx=ctx,
                plan_arm=plan_arm,
            )
        else:
            result = run_grasp_obj_v12_pipeline(
                world, str(object_name), session, shoulder,
                seed=int(seed), arm=str(arm), ctx=ctx,
            )
        if result is None:
            if ctx:
                ctx.log("  [grasp_obj] v13 pipeline 无可用 pose")
            return None
        best, pick_audit = result
        if bool(dual_arm_ik_filter):
            camera_face_audit = dict(best.get("camera_face") or {"skipped": True, "reason": "prefilter_applied"})
        else:
            best, camera_face_audit = apply_camera_face_forward_to_grasp_dict(best, world=world)
        pick_audit = dict(pick_audit)
        pick_audit["camera_face"] = camera_face_audit
        if ctx and not camera_face_audit.get("skipped"):
            ctx.log(
                "  [grasp_obj] wrist camera face "
                f"flip={camera_face_audit.get('flipped')} "
                f"angle {camera_face_audit.get('angle_before_deg', 0):.1f}°"
                f"→{camera_face_audit.get('angle_after_deg', 0):.1f}°"
            )
        centroid = pcd_np.mean(axis=0) if len(pcd_np) >= 1 else np.asarray(best["gap_center"])
        gf = {
            "feasible": bool(best.get("feasible", True)),
            "has_volume": float(best.get("grasp_vol_cm3", 0.0)) > 0.0,
            "n_collision": int(best.get("ncol_exp", 0)),
            "n_collision_expanded": int(best.get("ncol_exp", 0)),
            "n_gap": int(best.get("grasp_vox", best.get("open_intersect_vox", 0))),
            "gap_voxel_n": int(best.get("grasp_vox", best.get("open_intersect_vox", 0))),
            "gap_vol_cm3": float(best.get("grasp_vol_cm3", 0.0)),
            "n_pos_side": 0,
            "n_neg_side": 0,
            "has_both_sides": True,
            "base_dist_mm": float(best.get("base_dist_mm", 0.0)),
            "y_balance": float(best.get("y_balance", 1.0)),
            "center_bias_mm": float(best.get("center_bias_mm", 0.0)),
            "min_rail_clear_mm": float(best.get("min_rail_clear_mm", 0.0)),
            "approach_label": best.get("approach_label", "v12_icosa"),
            "ref_y_label": best.get("ref_y_label", "v12"),
            "roll_deg": float(best.get("roll_deg", 0.0)),
            "reachable": _scalar_bool_field(best, "reachable"),
            "reach_reason": str(best.get("reach_reason", "")),
            "n_score_pcd": int(len(pcd_np)),
            "n_mesh_pts": 0,
            "volume_source": best.get("volume_source", "v12_opening_voxel"),
            "mesh_vol_tag": "v12_trimesh",
            "n_pose_samples": 1600,
            "sample_mode": (
                "v14_surface_ball_icosa_dual_ik_filter"
                if bool(dual_arm_ik_filter)
                else "v12_surface_ball_icosa"
            ),
            "obj_centroid": centroid.tolist(),
            "gap_dist_to_obj_mm": float(best.get("anchor_dist_mm", 0.0)),
            "opening_vol_build": best.get("opening_vol_build"),
            "open_intersect_vox": int(best.get("open_intersect_vox", 0)),
            "open_intersect_vol_cm3": float(best.get("open_intersect_vol_cm3", 0.0)),
            "open_vol_method": best.get("open_vol_method", ""),
            "open_voxel_mm": float(best.get("open_voxel_mm", 3.0)),
            "opening_voxel_total": int(best.get("opening_voxel_total", 0)),
            "grasp_vol_cm3": float(best.get("grasp_vol_cm3", 0.0)),
            "overlap_vol_cm3": float(best.get("overlap_vol_cm3", 0.0)),
            "gripper_vol_cm3": float(best.get("gripper_vol_cm3", 0.0)),
            "overlap_frac": float(best.get("overlap_frac", 0.0)),
            "anchor_dist_mm": float(best.get("anchor_dist_mm", 0.0)),
            "marker_pt": (
                np.asarray(best["marker_pt"], dtype=np.float64).reshape(3).tolist()
                if best.get("marker_pt") is not None
                else np.asarray(best["gap_center"], dtype=np.float64).reshape(3).tolist()
            ),
            "camera_face": camera_face_audit,
        }
        recommended_arm = best.get("recommended_arm") or best.get("arm")
        return {
            "pos": np.asarray(best["pos"]).tolist(),
            "quat": np.asarray(best["quat"]).tolist(),
            "gap_center": np.asarray(best["gap_center"]).tolist(),
            "grip_fit": gf,
            "plan_audit": pick_audit,
            "camera_face": camera_face_audit,
            "recommended_arm": recommended_arm,
            "arm": recommended_arm,
            "selected_pose_ik": best.get("selected_pose_ik"),
            "selected_pose_ik_q": best.get("selected_pose_ik_q"),
            "ik_solution": best.get("ik_solution"),
            "ik_filter": best.get("ik_filter"),
            "ik_allerr": best.get("ik_allerr"),
        }

    # ── 旧路径回退（无 session / object_name 时）──────────────────
    region_ref = _obj_ref_for_region(obj_ref, obj_aabb, pcd_np.mean(axis=0))
    pcd_np = np.asarray(pcd, dtype=np.float64)
    pcd_np = _crop_to_object_region(pcd_np, region_ref, obj_aabb, ctx=ctx)

    mesh_vol_pcd: Optional[np.ndarray] = None
    mesh_col_pcd: Optional[np.ndarray] = None
    mesh_surf_pcd: Optional[np.ndarray] = None
    mesh_centroid: Optional[np.ndarray] = None
    mesh_n = 0
    mesh_interior_n = 0
    mesh_vol_tag = "none"
    use_mesh_volume = False
    mesh_vol_pcd, mesh_col_pcd, mesh_surf_pcd, mesh_centroid, mesh_interior_n, mesh_vol_tag = (
        _normalize_mesh_input(mesh_pcd)
    )
    mesh_eval_pcd = mesh_vol_pcd
    if mesh_eval_pcd is not None and len(mesh_eval_pcd) >= 8:
        mesh_n = len(mesh_eval_pcd)
        use_mesh_volume = True
    if mesh_col_pcd is None:
        mesh_col_pcd = mesh_vol_pcd

    if hit_anchor is not None:
        ha = np.asarray(hit_anchor, dtype=np.float64).reshape(3)
        near = pcd_np[np.linalg.norm(pcd_np - ha, axis=1) < 0.30]
        score_pcd = _score_pointcloud(near if len(near) >= 8 else pcd_np, seed=seed)
    else:
        score_pcd = _score_pointcloud(pcd_np, seed=seed)
    if len(score_pcd) < 8:
        if ctx:
            ctx.log("  [grasp_obj] 点云过少")
        return None

    shoulder = np.asarray(shoulder, dtype=np.float64).reshape(3)
    rng = np.random.default_rng(seed)
    centroid = score_pcd.mean(axis=0)
    _, t_long, t_short = _pca_axes(score_pcd)
    ref_ys = _ref_y_candidates(t_long, t_short)

    from behavior_interface.skills.plan_grasp_gripper_geom import gap_anchor_local_one_third_from_base

    gap_anchor_local = gap_anchor_local_one_third_from_base()
    use_boundary = (
        world is not None
        and object_name
        and session is not None
    )
    pose_metas: List[Dict[str, Any]] = []
    raw_samples: List[Any] = []

    if use_boundary:
        b_pts, b_nrms = _sample_verified_boundary_points(
            world, object_name, session, rng,
            n_points=_N_BOUNDARY_POINTS,
            obj_ref=region_ref, obj_aabb=obj_aabb,
            ctx=ctx,
        )
        if len(b_pts) < 8:
            if ctx:
                ctx.log("  [grasp_obj] WARN 边界点不足，回退旧随机采样")
            use_boundary = False
        else:
            boundary_samples = _sample_boundary_grasp_poses(
                b_pts, b_nrms, rng,
                n_per_point=_N_GRASPS_PER_BOUNDARY,
            )
            if ctx:
                p0 = b_pts[0]
                e0 = boundary_samples[0][2] - boundary_samples[0][3] @ gap_anchor_local
                ctx.log(
                    f"  [grasp_obj] 边界点 mean={b_pts.mean(axis=0).round(3).tolist()} "
                    f"span={(b_pts.max(0)-b_pts.min(0)).round(3).tolist()} "
                    f"pose0_anchor={np.asarray(p0).round(3).tolist()} "
                    f"pose0_eef={np.asarray(e0).round(3).tolist()} "
                    f"gap_local_z={gap_anchor_local[2]*1000:.1f}mm"
                )
            _grasp_obj_report(
                ctx,
                f"  [grasp_obj] 开始规划 boundary={len(b_pts)}×{_N_GRASPS_PER_BOUNDARY}"
                f"={len(boundary_samples)} 模式=对称轴1/3表面锚点+均匀随机朝向 "
                f"筛选=不穿模→左右均衡 depth_score={len(score_pcd)} "
                f"mesh_eval={mesh_n} seed={seed}",
                status=True,
            )
            for i, (approach, roll, anchor, R) in enumerate(boundary_samples):
                ry_label, _ = ref_ys[i % len(ref_ys)]
                eef_pos = anchor - R @ gap_anchor_local
                pose_metas.append({
                    "R": R,
                    "eef_pos": eef_pos,
                    "quat": _mat_to_quat_xyzw(R),
                    "approach": R[:, 2].copy(),
                    "roll_deg": math.degrees(roll),
                    "approach_label": "boundary",
                    "ref_y_label": ry_label,
                    "gap_center": anchor,
                })
            raw_samples = boundary_samples

    if not use_boundary:
        n_pose = int(n_random) if n_random is not None else _N_RANDOM_POSES
        n_pose = max(200, min(n_pose, _N_RANDOM_POSES))
        _grasp_obj_report(
            ctx,
            f"  [grasp_obj] 开始规划 depth_score={len(score_pcd)} "
            f"mesh_eval={mesh_n} interior={mesh_interior_n} vol_src={mesh_vol_tag} "
            f"grip_len={_GRIPPER_LEN_M * 1000:.0f}mm half={_GRIPPER_HALF_MM:.0f}mm "
            f"n_random={n_pose} seed={seed}",
            status=True,
        )
        raw_samples = _sample_random_poses(
            score_pcd, t_long, t_short, rng, n_pose, hit_anchor=hit_anchor,
            obj_ref=region_ref, obj_aabb=obj_aabb,
        )
        pose_metas = []
        for i, (approach, roll, anchor) in enumerate(raw_samples):
            ry_label, ref_y_val = ref_ys[i % len(ref_ys)]
            R = _eef_frame_from_approach_roll(approach, roll, ref_y=ref_y_val)
            eef_pos = anchor - R @ gap_anchor_local
            pose_metas.append({
                "R": R,
                "eef_pos": eef_pos,
                "quat": _mat_to_quat_xyzw(R),
                "approach": R[:, 2].copy(),
                "roll_deg": math.degrees(roll),
                "approach_label": "random",
                "ref_y_label": ry_label,
                "gap_center": anchor,
            })

    n_pose = len(pose_metas)

    try:
        candidates = _batch_evaluate_grasp_poses(
            pose_metas, score_pcd, shoulder,
            vol_pcd=mesh_eval_pcd, col_pcd=mesh_col_pcd, surf_pcd=mesh_surf_pcd,
            obj_ref=region_ref, obj_aabb=obj_aabb,
            mesh_centroid=mesh_centroid, hit_anchor=hit_anchor, ctx=ctx,
        )
    except Exception as e:
        if ctx:
            ctx.log(f"  [grasp_obj] 批量评估失败，回退逐 pose CPU: {e}")
        candidates = []
        for i in range(len(pose_metas)):
            pm = pose_metas[i]
            if use_boundary and len(raw_samples[i]) >= 4:
                approach, roll, anchor, R = raw_samples[i]
            else:
                approach, roll, anchor = raw_samples[i][:3]
                R = pm["R"]
            candidates.append(_build_candidate(
                approach, roll,
                pm["ref_y_label"],
                ref_ys[i % len(ref_ys)][1],
                anchor, score_pcd, shoulder,
                alabel=pm.get("approach_label", "random"),
                vol_pcd=mesh_eval_pcd, col_pcd=mesh_col_pcd, surf_pcd=mesh_surf_pcd,
                use_mesh_volume=use_mesh_volume,
            ))

    ranked = _sort_best_effort_pool(candidates)
    if not ranked:
        if ctx:
            ctx.log(f"  [grasp_obj] WARN 候选为空，无法规划")
        return None
    if ctx:
        ncol0_r = sum(1 for c in ranked if c.get("ncol_exp", 1) == 0)
        top5_vox = [int(c.get("gap_voxel_n", 0)) for c in ranked[:5]]
        top5_ncol = [int(c.get("ncol_exp", 0)) for c in ranked[:5]]
        gmax = max(int(c.get("gap_voxel_n", 0)) for c in ranked)
        top5_bal = [round(float(c.get("y_balance", 1)), 2) for c in ranked[:5]]
        ctx.log(
            f"  [grasp_obj] ranked={len(ranked)} ncol0={ncol0_r} "
            f"global_max_vox={gmax} top5_vox={top5_vox} top5_ncol={top5_ncol} "
            f"top5_bal={top5_bal}"
        )
    coarse_pick_pool = ranked[: min(_COARSE_FALLBACK_TOP, len(ranked))]
    refined_candidates: List[Dict[str, Any]] = []
    if use_boundary:
        if ctx:
            ctx.log(
                f"  [grasp_obj] 边界模式跳过精调/roll/tuck，直接从 ranked top-{len(coarse_pick_pool)} 选取"
            )
    else:
        top_k = ranked[: min(36, len(ranked))]
        _grasp_obj_report(
            ctx,
            f"  [grasp_obj] 精调 top-{len(top_k)} 体积候选（各 {_N_REFINE} 次抖动）"
            f" coarse_fb={len(coarse_pick_pool)}",
            status=True,
        )
        for j, c in enumerate(top_k):
            refined = _refine_candidate(
                c, score_pcd, shoulder, rng,
                vol_pcd=mesh_eval_pcd, col_pcd=mesh_col_pcd, surf_pcd=mesh_surf_pcd,
                use_mesh_volume=use_mesh_volume,
            )
            refined_candidates.append(refined)
            candidates.append(refined)

        if world is not None and refined_candidates:
            chest_q = _CHEST_TUCK_ARM.get(arm, _CHEST_TUCK_ARM["right"])
            tuck_pose = _eef_pose_at_arm_qpos(world, arm, chest_q)
            if tuck_pose is not None:
                _, tuck_quat = tuck_pose
                roll_src = sorted(
                    refined_candidates,
                    key=lambda c: _pose_rank_key(c, boundary_mode=False),
                )[: min(_ROLL_ALIGN_TOP, len(refined_candidates))]
                roll_src_ids = {id(c) for c in roll_src}
                roll_aligned = [
                    _refine_roll_for_tuck_align(
                        c, tuck_quat, score_pcd, shoulder,
                        vol_pcd=mesh_eval_pcd, col_pcd=mesh_col_pcd,
                        surf_pcd=mesh_surf_pcd, use_mesh_volume=use_mesh_volume,
                    )
                    for c in roll_src
                ]
                refined_candidates = roll_aligned + [
                    c for c in refined_candidates if id(c) not in roll_src_ids
                ]
                candidates.extend(roll_aligned)

    pick_audit: Dict[str, Any] = {}
    best: Optional[Dict[str, Any]] = None
    pick_stages: List[Tuple[str, List[Dict[str, Any]]]] = []
    if use_boundary:
        if coarse_pick_pool:
            pick_stages.append(("coarse_anchor", coarse_pick_pool))
    else:
        if refined_candidates:
            pick_stages.append(("refined", refined_candidates))
        if coarse_pick_pool:
            pick_stages.append(("coarse_top", coarse_pick_pool))
    if not pick_stages:
        pick_stages.append(("all", candidates))

    for stage_label, pool in pick_stages:
        if not pool:
            continue
        try:
            if world is not None:
                best, pick_audit = _pick_exec_reachable_pool(
                    pool, world, arm, shoulder, rng, ctx=ctx,
                    boundary_mode=use_boundary,
                )
            else:
                best = _pick_best_pool(
                    pool, shoulder, rng, ctx=ctx,
                    boundary_mode=use_boundary,
                )
                pick_audit = {"pick_method": "shoulder_heuristic_no_world"}
            pick_audit["pick_pool_stage"] = stage_label
            if ctx:
                ctx.log(
                    f"  [grasp_obj] 选取成功 stage={stage_label} "
                    f"method={pick_audit.get('pick_method')} "
                    f"vox={best.get('gap_voxel_n')} ncol={best.get('ncol_exp')} "
                    f"bal={best.get('y_balance', 0):.2f} "
                    f"ctr={best.get('center_bias_mm', 0):.1f}mm"
                )
            break
        except ValueError as e:
            if ctx:
                ctx.log(
                    f"  [grasp_obj] pick池 {stage_label}({len(pool)}) 失败: {e}"
                )
    if best is None and candidates:
        best = _sort_best_effort_pool(candidates)[0]
        pick_audit = {"pick_method": "global_best_effort_fallback"}
        if ctx:
            ctx.log(
                f"  [grasp_obj] WARN 各 pick 池失败，全局软退化 "
                f"vox={best.get('gap_voxel_n', 0)} ncol={best.get('ncol_exp', 0)}"
            )
    if best is None:
        return None

    # 对称轴锚点吸附 mesh 外表面并重算 eef（边界/随机均执行）
    best = _sync_pose_gap_center(best)
    if mesh_surf_pcd is not None and len(mesh_surf_pcd) >= 8:
        best = _snap_pose_to_mesh_surface(best, mesh_surf_pcd)
    best = _sync_pose_gap_center(best)
    best["eef_anchor_dist_mm"] = _eef_anchor_offset_mm(best)
    # snap 后复检：指间体积须仍为 bag mesh 外表面实体
    if use_mesh_volume and mesh_surf_pcd is not None and len(mesh_surf_pcd) >= 8:
        from behavior_interface.skills.plan_grasp_gripper_fit import _quat_to_mat
        R_fin = _quat_to_mat(np.asarray(best["quat"], dtype=np.float64))
        rev = _evaluate_grasp_object_pose(
            R_fin, np.asarray(best["pos"], dtype=np.float64),
            score_pcd,
            vol_pcd=mesh_eval_pcd, col_pcd=mesh_col_pcd, surf_pcd=mesh_surf_pcd,
            use_mesh_volume=True,
        )
        for k in (
            "has_volume", "gap_voxel_n", "gap_vol_cm3", "n_gap", "n_pos", "n_neg",
            "has_both", "mesh_pinch_core_vox", "mesh_y_span_mm", "mesh_z_span_mm",
            "mesh_entity_span_ok", "ncol_exp", "ncol_pts", "ncol_near_n",
            "volume_source",
        ):
            if k in rev:
                best[k] = rev[k]
        if not rev.get("has_volume") and ctx:
            ctx.log(
                f"  [grasp_obj] WARN snap后指间表面体积偏低 "
                f"vox={rev.get('gap_voxel_n', 0)} "
                f"y/z_span={rev.get('mesh_y_span_mm', 0):.1f}/"
                f"{rev.get('mesh_z_span_mm', 0):.1f}mm "
                f"仍输出 best_effort"
            )
    if ctx:
        from behavior_interface.skills.plan_grasp_gripper_geom import gap_anchor_local_one_third_from_base
        ga = gap_anchor_local_one_third_from_base()
        ctx.log(
            f"  [grasp_obj] 几何审计 gap_center={np.asarray(best['gap_center']).round(3).tolist()} "
            f"eef={np.asarray(best['pos']).round(3).tolist()} "
            f"eef_anchor_dist={best['eef_anchor_dist_mm']:.0f}mm "
            f"(对称轴1/3局部=[{ga[0]*1000:.1f},{ga[1]*1000:.1f},{ga[2]*1000:.1f}]mm) "
            f"vol_src={best.get('volume_source', '?')} "
            f"vox={best.get('gap_voxel_n', 0)} "
            f"y/z_span={best.get('mesh_y_span_mm', 0):.1f}/{best.get('mesh_z_span_mm', 0):.1f}mm "
            f"ncol_vox={best.get('ncol_exp', 0)} ncol_pts={best.get('ncol_pts', 0)} "
            f"near_n={best.get('ncol_near_n', 0)} "
            f"gap_dist={best.get('gap_dist_to_obj_mm', 0):.0f}mm "
            f"snap={best.get('mesh_snap_tag', 'n/a')}"
        )
    if mesh_surf_pcd is not None and len(mesh_surf_pcd) >= 8:
        best["gap_dist_to_obj_mm"] = _min_gap_dist_to_mesh_mm(
            best["gap_center"], mesh_surf_pcd,
        )

    best, camera_face_audit = apply_camera_face_forward_to_grasp_dict(best, world=world)
    if ctx and not camera_face_audit.get("skipped"):
        ctx.log(
            "  [grasp_obj] wrist camera face "
            f"flip={camera_face_audit.get('flipped')} "
            f"angle {camera_face_audit.get('angle_before_deg', 0):.1f}°"
            f"→{camera_face_audit.get('angle_after_deg', 0):.1f}°"
        )

    # grasp_vol / overlap_vol：与 v7 审计共用 compute_grasp_obj_v7_metrics（体素 3mm、可见 mesh）
    if world is not None and object_name:
        try:
            from behavior_interface.skills.plan_grasp_opening_volume import (
                compute_grasp_obj_v7_metrics,
            )
            vm = compute_grasp_obj_v7_metrics(
                world,
                object_name,
                np.asarray(best["pos"], dtype=np.float64),
                np.asarray(best["quat"], dtype=np.float64),
                np.asarray(best["gap_center"], dtype=np.float64),
            )
            best.update(vm)
            if ctx:
                ctx.log(
                    f"  [grasp_obj] v7 grasp_vol={best['grasp_vol_cm3']:.2f}cm3 "
                    f"overlap_vol={best['overlap_vol_cm3']:.2f}cm3 "
                    f"({100.0 * float(vm.get('overlap_frac', 0)):.0f}% of "
                    f"gripper {float(vm.get('gripper_vol_cm3', 0)):.1f}cm3) "
                    f"center->surf={best.get('anchor_dist_mm', 0):.2f}mm "
                    f"vox={vm.get('volume_voxel_mm')} build={vm.get('volume_v7_build')}"
                )
        except Exception as e:
            if ctx:
                ctx.log(f"  [grasp_obj] WARN v7 grasp/overlap 体积计算失败: {e}")

    ncol0_n = sum(1 for c in candidates if c.get("ncol_exp", 1) == 0)
    has_vol_n = sum(1 for c in candidates if c.get("has_volume"))
    on_obj_n = sum(
        1 for c in candidates
        if c.get("has_volume") and c.get("gap_on_object", True)
    )

    return {
        "pos": np.asarray(best["pos"]).tolist(),
        "quat": np.asarray(best["quat"]).tolist(),
        "gap_center": np.asarray(best["gap_center"]).tolist(),
        "grip_fit": {
            "feasible": best.get("feasible", False),
            "has_volume": bool(
                best.get("has_volume")
                or int(best.get("gap_voxel_n", 0)) >= 1
            ),
            "n_collision": best["ncol_exp"],
            "n_collision_expanded": best["ncol_exp"],
            "n_gap": best["n_gap"],
            "gap_voxel_n": best.get("gap_voxel_n", 0),
            "gap_vol_cm3": best.get("gap_vol_cm3", 0.0),
            "n_pos_side": best["n_pos"],
            "n_neg_side": best["n_neg"],
            "has_both_sides": best["has_both"],
            "base_dist_mm": best["base_dist_mm"],
            "y_balance": best.get("y_balance", 0.0),
            "center_bias_mm": best.get("center_bias_mm", 0.0),
            "min_rail_clear_mm": best.get("min_rail_clear_mm", 0.0),
            "approach_label": best["approach_label"],
            "ref_y_label": best.get("ref_y_label", ""),
            "roll_deg": best["roll_deg"],
            "reachable": best["reachable"],
            "reach_reason": best["reach_reason"],
            "n_score_pcd": len(score_pcd),
            "n_mesh_pts": mesh_n,
            "volume_source": best.get("volume_source", "depth"),
            "mesh_y_span_mm": best.get("mesh_y_span_mm", 0.0),
            "mesh_center_vox": best.get("mesh_center_vox", 0),
            "mesh_pinch_core_vox": best.get("mesh_pinch_core_vox", 0),
            "mesh_surf_min_mm": best.get("mesh_surf_min_mm", 0.0),
            "gripper_half_mm": best.get("gripper_half_mm", _GRIPPER_HALF_MM),
            "gripper_len_mm": best.get("gripper_len_mm", _GRIPPER_LEN_M * 1000.0),
            "mesh_surf_reach_ok": best.get("mesh_surf_reach_ok", False),
            "mesh_pinch_near_pts": best.get("mesh_pinch_near_pts", 0),
            "mesh_interior_n": mesh_interior_n,
            "mesh_vol_tag": mesh_vol_tag,
            "n_pose_samples": n_pose,
            "sample_mode": "boundary" if use_boundary else "random",
            "obj_centroid": centroid.tolist(),
            "gap_dist_to_obj_mm": best.get("gap_dist_to_obj_mm", 0.0),
            "opening_vol_build": best.get("opening_vol_build"),
            "open_intersect_vox": best.get("open_intersect_vox", 0),
            "open_intersect_vol_cm3": best.get("open_intersect_vol_cm3", 0.0),
            "open_intersect_vox_raw": best.get("open_intersect_vox_raw", 0),
            "open_intersect_vol_raw_cm3": best.get("open_intersect_vol_raw_cm3", 0.0),
            "open_vol_method": best.get("open_vol_method"),
            "open_voxel_mm": best.get("open_voxel_mm"),
            "opening_voxel_total": best.get("opening_voxel_total", 0),
            "open_intersect_frac": best.get("open_intersect_frac", 0.0),
            # v7 口径（plan 返回与标注主读这两项）
            "grasp_vol_cm3": float(best.get("grasp_vol_cm3", best.get("open_intersect_vol_cm3", 0.0))),
            "overlap_vol_cm3": float(best.get("overlap_vol_cm3", 0.0)),
            "gripper_vol_cm3": float(best.get("gripper_vol_cm3", 0.0)),
            "overlap_frac": float(best.get("overlap_frac", 0.0)),
            "anchor_dist_mm": float(best.get("anchor_dist_mm", best.get("gap_dist_to_obj_mm", 0.0))),
            "marker_pt": best.get("marker_pt"),
            "camera_face": camera_face_audit,
        },
        "plan_audit": {
            "grasp_obj_build": GRASP_OBJ_BUILD,
            "object_name": object_name,
            "seed": int(seed),
            "mesh_seed": 42,
            "n_candidates": len(candidates),
            "n_refined": len(refined_candidates),
            "n_has_volume": has_vol_n,
            "n_gap_on_object": on_obj_n,
            "n_ncol0": ncol0_n,
            "shoulder": np.asarray(shoulder).round(4).tolist(),
            "obj_centroid": centroid.tolist(),
            "pick": pick_audit,
            "camera_face": camera_face_audit,
        },
        "camera_face": camera_face_audit,
    }
