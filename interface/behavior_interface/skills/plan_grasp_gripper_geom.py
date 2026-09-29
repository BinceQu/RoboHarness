"""R1Pro 夹爪爪间几何：与 viz_eef_v2 OBJ 渲染同一套 link 变换，三角柱楔形开口。

体积统计、碰撞（不穿模）、轨道距离均基于两指内侧三角面之间的楔形域，
不再使用轴对齐 _GAP_BOX 长方体。
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from behavior_interface.gripper_geometry_calibration import (
    R1PRO_FINGER_LINK_Z_EEF_M,
)

# 与 viz_eef_v2._GRIPPER_LINKS 一致。R1Pro finger joint 是绝对位置：
# q=0 闭合，q=0.05 全开，分别沿 EEF +Y / -Y 移动。
_RY_PI_XYZW = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
_GRIPPER_Q_MIN_M = 0.0
_GRIPPER_Q_MAX_M = 0.05
_FINGER_LINK_SPECS = (
    ("right_gripper_finger_link1", +0.01345, +1.0),
    ("right_gripper_finger_link2", -0.01345, -1.0),
)
_OBJ_DIR = str(
    Path(os.environ.get("OMNIGIBSON_DATA_PATH", Path(__file__).resolve().parents[3] / "data"))
    / "omnigibson-robot-assets/source/r1pro/meshes"
)
# 楔形开口最小宽度（米）；低于此 z 切片视为指尖闭合
_MIN_GAP_WIDTH_M = 0.0003
# 体积统计 y 方向微膨胀（米）：抵消 mesh 离散，仍远小于旧 74mm 盒
_GAP_VOLUME_Y_INFLATE_M = 0.001
_Z_SLAB_BAND_M = 0.0015
_N_Z_BINS = 120

_LUT: Optional[Dict[str, Any]] = None
_FINGER_VERTS_EEF: Optional[Tuple[np.ndarray, np.ndarray]] = None
# 对称轴 1/3 锚点（EEF 局部），由两指 OBJ 推导后缓存
_GAP_ANCHOR_LOCAL: Optional[np.ndarray] = None


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def normalize_gripper_qpos(gripper_qpos=None) -> Tuple[float, float]:
    """Normalize one/two absolute finger positions into a clipped q1/q2 tuple."""
    if gripper_qpos is None:
        return _GRIPPER_Q_MAX_M, _GRIPPER_Q_MAX_M
    arr = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return _GRIPPER_Q_MAX_M, _GRIPPER_Q_MAX_M
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    if not np.all(np.isfinite(arr[:2])):
        raise ValueError(f"gripper qpos 包含非有限值: {arr[:2].tolist()}")
    q = np.clip(arr[:2], _GRIPPER_Q_MIN_M, _GRIPPER_Q_MAX_M)
    return round(float(q[0]), 6), round(float(q[1]), 6)


@lru_cache(maxsize=3)
def _load_obj_mesh(link_name: str) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(_OBJ_DIR, f"{link_name}.obj")
    verts: list = []
    faces: list = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                indices = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
                for index in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[index], indices[index + 1]])
    return (
        np.asarray(verts, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
    )


def _load_obj_verts(link_name: str) -> np.ndarray:
    return _load_obj_mesh(link_name)[0]


def _finger_link_translations_eef(gripper_qpos=None) -> Tuple[np.ndarray, np.ndarray]:
    q1, q2 = normalize_gripper_qpos(gripper_qpos)
    out = []
    for (_link_name, closed_y, axis_sign), q in zip(
        _FINGER_LINK_SPECS,
        (q1, q2),
    ):
        out.append(
            np.array(
                [
                    0.0,
                    closed_y + axis_sign * q,
                    R1PRO_FINGER_LINK_Z_EEF_M,
                ],
                dtype=np.float64,
            )
        )
    return out[0], out[1]


def _finger_verts_in_eef(gripper_qpos=None) -> Tuple[np.ndarray, np.ndarray]:
    global _FINGER_VERTS_EEF
    q1, q2 = normalize_gripper_qpos(gripper_qpos)
    is_default_open = q1 == _GRIPPER_Q_MAX_M and q2 == _GRIPPER_Q_MAX_M
    if is_default_open and _FINGER_VERTS_EEF is not None:
        return _FINGER_VERTS_EEF
    translations = _finger_link_translations_eef((q1, q2))
    out: list = []
    rotation = _quat_to_mat(_RY_PI_XYZW)
    for (link_name, _closed_y, _axis_sign), local_t in zip(
        _FINGER_LINK_SPECS,
        translations,
    ):
        v = _load_obj_verts(link_name)
        out.append((rotation @ v.T).T + local_t)
    result = (out[0], out[1])
    if is_default_open:
        _FINGER_VERTS_EEF = result
    return result


def _finger_tris_in_eef(gripper_qpos=None) -> Tuple[np.ndarray, np.ndarray]:
    translations = _finger_link_translations_eef(gripper_qpos)
    rotation = _quat_to_mat(_RY_PI_XYZW)
    out = []
    for (link_name, _closed_y, _axis_sign), local_t in zip(
        _FINGER_LINK_SPECS,
        translations,
    ):
        vertices, faces = _load_obj_mesh(link_name)
        vertices_eef = (rotation @ vertices.T).T + local_t
        out.append(vertices_eef[faces])
    return out[0], out[1]


def _slab_stats(
    verts: np.ndarray,
    zc: float,
    *,
    inner_side: str,
) -> Optional[Tuple[float, float, float, float]]:
    """单 z 切片：inner_y, outer_y, x_half, n_pts。"""
    band = _Z_SLAB_BAND_M
    m = (verts[:, 2] >= zc - band) & (verts[:, 2] <= zc + band)
    if int(m.sum()) < 3:
        return None
    ys = verts[m, 1]
    xs = verts[m, 0]
    if inner_side == "pos":
        inner_y = float(ys.min())
        outer_y = float(ys.max())
    else:
        inner_y = float(ys.max())
        outer_y = float(ys.min())
    x_half = float(np.percentile(np.abs(xs), 95))
    return inner_y, outer_y, x_half, float(m.sum())


def _triangle_section_stats(
    triangles: np.ndarray,
    zc: float,
    *,
    inner_side: str,
) -> Optional[Tuple[float, float, float, float]]:
    """Exact triangle-plane section at z=zc, avoiding sparse-vertex jumps."""
    z = triangles[:, :, 2]
    active = (np.min(z, axis=1) <= zc) & (np.max(z, axis=1) >= zc)
    tri = triangles[active]
    if len(tri) == 0:
        return None

    points = []
    for start, end in ((0, 1), (1, 2), (2, 0)):
        p0 = tri[:, start]
        p1 = tri[:, end]
        z0 = p0[:, 2]
        z1 = p1[:, 2]
        dz = z1 - z0
        crosses = (
            (np.minimum(z0, z1) <= zc)
            & (np.maximum(z0, z1) >= zc)
            & (np.abs(dz) > 1e-12)
        )
        if np.any(crosses):
            t = (zc - z0[crosses]) / dz[crosses]
            points.append(
                p0[crosses] + t[:, None] * (p1[crosses] - p0[crosses])
            )
    if not points:
        return None
    section = np.vstack(points)
    ys = section[:, 1]
    xs = section[:, 0]
    if inner_side == "pos":
        inner_y = float(ys.min())
        outer_y = float(ys.max())
    else:
        inner_y = float(ys.max())
        outer_y = float(ys.min())
    x_half = float(np.percentile(np.abs(xs), 95))
    return inner_y, outer_y, x_half, float(len(section))


def _lut_from_rows(
    rows: list,
    *,
    q1: float,
    q2: float,
    allow_empty_opening: bool = False,
) -> Dict[str, Any]:
    if not rows:
        raise RuntimeError("夹爪 OBJ 楔形 LUT 构建失败")

    arr = np.asarray(rows, dtype=np.float64)
    z_all = arr[:, 0]
    open_m = arr[:, 7] >= _MIN_GAP_WIDTH_M
    grasp_m = np.zeros_like(open_m)
    contact_indices = np.flatnonzero(open_m)
    if len(contact_indices):
        split_at = np.flatnonzero(np.diff(contact_indices) > 1) + 1
        runs = np.split(contact_indices, split_at)
        # Reject isolated body/root slivers and retain the main continuous
        # corridor between the fingers, including the curved fingertips.
        runs = [run for run in runs if len(run) >= 2]
        if runs:
            contact_run = max(
                runs,
                key=lambda run: (len(run), float(z_all[run[-1]])),
            )
            grasp_m[contact_run] = True
    if not grasp_m.any() and not allow_empty_opening:
        grasp_m = open_m
    if grasp_m.any():
        grasp = arr[grasp_m]
        z_grasp_lo = float(grasp[:, 0].min())
        z_grasp_hi = float(grasp[:, 0].max())
        gap_center = np.array([
            0.0,
            float((grasp[:, 1].mean() + grasp[:, 2].mean()) * 0.5),
            float(grasp[:, 0].mean()),
        ], dtype=np.float64)
        finger_inner_y = float(max(
            abs(grasp[:, 1]).max(),
            abs(grasp[:, 2]).max(),
        ))
    elif allow_empty_opening:
        z_grasp_lo = float("nan")
        z_grasp_hi = float("nan")
        gap_center = np.array([
            0.0,
            0.5 * (q1 - q2),
            float(z_all.mean()),
        ], dtype=np.float64)
        finger_inner_y = 0.0
    else:
        raise RuntimeError("夹爪 OBJ 楔形没有有效开口切片")

    return {
        "gripper_qpos_m": np.array([q1, q2], dtype=np.float64),
        "z": z_all,
        "y_inner_lo": arr[:, 1],
        "y_inner_hi": arr[:, 2],
        "y_outer_f1": arr[:, 3],
        "y_outer_f2": arr[:, 4],
        "x_half_gap": arr[:, 5],
        "x_half_finger": arr[:, 6],
        "gap_width": arr[:, 7],
        "z_grasp_lo": z_grasp_lo,
        "z_grasp_hi": z_grasp_hi,
        "gap_center_local": gap_center,
        "finger_inner_y": finger_inner_y,
        "finger_z_tip": float(z_all.min()),
        "finger_z_open": z_grasp_hi,
    }


def _build_wedge_lut(gripper_qpos=None) -> Dict[str, Any]:
    q1, q2 = normalize_gripper_qpos(gripper_qpos)
    f1, f2 = _finger_verts_in_eef((q1, q2))
    z_lo = max(float(f1[:, 2].min()), float(f2[:, 2].min()))
    z_hi = min(float(f1[:, 2].max()), float(f2[:, 2].max()))
    z_bins = np.linspace(z_lo, z_hi, _N_Z_BINS)

    rows: list = []
    for zc in z_bins:
        s1 = _slab_stats(f1, float(zc), inner_side="pos")
        s2 = _slab_stats(f2, float(zc), inner_side="neg")
        if s1 is None or s2 is None:
            continue
        y_hi, y1_max, xh1, _ = s1
        y_lo, y2_min, xh2, _ = s2
        gap_w = y_hi - y_lo
        rows.append((
            float(zc), y_lo, y_hi, y1_max, y2_min,
            min(xh1, xh2), max(xh1, xh2), gap_w,
        ))

    return _lut_from_rows(rows, q1=q1, q2=q2)


def _build_wrist_opening_lut(gripper_qpos=None) -> Dict[str, Any]:
    """Build the wrist blue-zone LUT from exact mesh plane intersections."""
    q1, q2 = normalize_gripper_qpos(gripper_qpos)
    f1, f2 = _finger_tris_in_eef((q1, q2))
    z_lo = max(float(f1[:, :, 2].min()), float(f2[:, :, 2].min()))
    z_hi = min(float(f1[:, :, 2].max()), float(f2[:, :, 2].max()))
    z_bins = np.linspace(z_lo, z_hi, _N_Z_BINS)

    rows = []
    for zc in z_bins:
        s1 = _triangle_section_stats(f1, float(zc), inner_side="pos")
        s2 = _triangle_section_stats(f2, float(zc), inner_side="neg")
        if s1 is None or s2 is None:
            continue
        y_hi, y1_max, xh1, _ = s1
        y_lo, y2_min, xh2, _ = s2
        rows.append((
            float(zc),
            y_lo,
            y_hi,
            y1_max,
            y2_min,
            min(xh1, xh2),
            max(xh1, xh2),
            y_hi - y_lo,
        ))
    return _lut_from_rows(
        rows,
        q1=q1,
        q2=q2,
        allow_empty_opening=True,
    )


@lru_cache(maxsize=64)
def _get_dynamic_wedge_lut(q1: float, q2: float) -> Dict[str, Any]:
    return _build_wedge_lut((q1, q2))


@lru_cache(maxsize=64)
def _get_cached_wrist_opening_lut(q1: float, q2: float) -> Dict[str, Any]:
    return _build_wrist_opening_lut((q1, q2))


def get_wedge_lut(gripper_qpos=None) -> Dict[str, Any]:
    global _LUT
    q1, q2 = normalize_gripper_qpos(gripper_qpos)
    if q1 == _GRIPPER_Q_MAX_M and q2 == _GRIPPER_Q_MAX_M:
        if _LUT is None:
            _LUT = _build_wedge_lut((q1, q2))
        return _LUT
    return _get_dynamic_wedge_lut(q1, q2)


def get_wrist_opening_lut(gripper_qpos=None) -> Dict[str, Any]:
    q1, q2 = normalize_gripper_qpos(gripper_qpos)
    return _get_cached_wrist_opening_lut(q1, q2)


def gap_center_local() -> np.ndarray:
    return get_wedge_lut()["gap_center_local"].copy()


def gap_contact_z_bounds(gripper_qpos=None) -> Tuple[float, float]:
    """Return the EEF-local fingertip contact band, excluding finger roots."""
    lut = get_wrist_opening_lut(gripper_qpos)
    return float(lut["z_grasp_lo"]), float(lut["z_grasp_hi"])


def gap_anchor_local_one_third_from_base() -> np.ndarray:
    """夹爪对称轴 (x≈0,y≈0) 上、靠基座 1/3 处（EEF 局部，单位米）。

    爪长：实际指关节平面 → 两指 mesh 最下端 z_tip（沿 -Z 爪尖方向）。
    物体表面采样点锁在此位置；eef = surf_pt - R @ 本向量（从锚点沿 +Z 回退）。
    """
    global _GAP_ANCHOR_LOCAL
    if _GAP_ANCHOR_LOCAL is not None:
        return _GAP_ANCHOR_LOCAL.copy()
    f1, f2 = _finger_verts_in_eef()
    z_tip = float(min(f1[:, 2].min(), f2[:, 2].min()))
    z_base = R1PRO_FINGER_LINK_Z_EEF_M
    z_at = z_base - (z_base - z_tip) / 3.0
    s1 = _slab_stats(f1, z_at, inner_side="pos")
    s2 = _slab_stats(f2, z_at, inner_side="neg")
    y_mid = 0.0
    if s1 is not None and s2 is not None:
        y_mid = 0.5 * (float(s1[0]) + float(s2[0]))
    _GAP_ANCHOR_LOCAL = np.array([0.0, y_mid, z_at], dtype=np.float64)
    return _GAP_ANCHOR_LOCAL.copy()


def gap_z_bounds() -> Tuple[float, float]:
    lut = get_wrist_opening_lut()
    return float(lut["z_grasp_lo"]), float(lut["z_grasp_hi"])


def finger_inner_y_nominal() -> float:
    return float(get_wedge_lut()["finger_inner_y"])


def finger_z_tip() -> float:
    return float(get_wedge_lut()["finger_z_tip"])


def _interp_cols(z: np.ndarray, lut: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[np.ndarray, ...]:
    z_bins = lut["z"]
    out = []
    for k in keys:
        out.append(np.interp(z, z_bins, lut[k]))
    return tuple(out)


def gap_opening_strict_mask(q: np.ndarray, gripper_qpos=None) -> np.ndarray:
    """Strict fingertip opening, excluding finger roots and robot body."""
    return gap_opening_strict_mask_from_lut(
        q,
        get_wrist_opening_lut(gripper_qpos),
    )


def gap_opening_strict_mask_from_lut(
    q: np.ndarray,
    lut: Dict[str, Any],
) -> np.ndarray:
    """Evaluate strict opening membership against an explicit geometry LUT."""
    if len(q) == 0:
        return np.zeros(0, dtype=bool)
    z = q[:, 2]
    y_lo, y_hi, xh, gap_w = _interp_cols(
        z, lut, ("y_inner_lo", "y_inner_hi", "x_half_gap", "gap_width"),
    )
    z_lo = float(lut["z_grasp_lo"])
    z_hi = float(lut["z_grasp_hi"])
    valid_z_band = math.isfinite(z_lo) and math.isfinite(z_hi) and z_hi >= z_lo
    in_z = (
        (z >= z_lo) & (z <= z_hi)
        if valid_z_band
        else np.zeros(len(q), dtype=bool)
    )
    in_y = (q[:, 1] > y_lo) & (q[:, 1] < y_hi)
    in_x = np.abs(q[:, 0]) <= xh
    open_slab = gap_w >= _MIN_GAP_WIDTH_M
    return in_z & in_y & in_x & open_slab


def gap_wedge_mask(q: np.ndarray, *, for_volume: bool = False) -> np.ndarray:
    """点落在两指内侧三角柱楔形开口内（与红夹爪 OBJ 一致）。

    for_volume=True 时在 y 向微膨胀并限定指尖 z 带，仅用于占据体积统计；
    碰撞/不穿模必须用 for_volume=False（严格内侧面）。
    """
    if len(q) == 0:
        return np.zeros(0, dtype=bool)
    lut = get_wedge_lut()
    z = q[:, 2]
    y_lo, y_hi, xh, gap_w = _interp_cols(
        z, lut, ("y_inner_lo", "y_inner_hi", "x_half_gap", "gap_width"),
    )
    inflate = _GAP_VOLUME_Y_INFLATE_M if for_volume else 0.0
    y_lo = y_lo - inflate
    y_hi = y_hi + inflate
    if for_volume:
        z_tip, z_hi = gap_contact_z_bounds()
        in_z = (z >= z_tip) & (z <= z_hi)
    else:
        in_z = (z >= lut["z_grasp_lo"]) & (z <= lut["z_grasp_hi"])
    in_y = (q[:, 1] > y_lo) & (q[:, 1] < y_hi)
    in_x = np.abs(q[:, 0]) <= xh
    open_slab = gap_w >= _MIN_GAP_WIDTH_M
    return in_z & in_y & in_x & open_slab


def finger_body_mask(q: np.ndarray, expand_m: float = 0.0) -> np.ndarray:
    """指部实体：在指 mesh 占据带内且不在楔形开口内的点（用于不穿模）。"""
    if len(q) == 0:
        return np.zeros(0, dtype=bool)
    lut = get_wedge_lut()
    z = q[:, 2]
    y_lo, y_hi, y1_max, y2_min, xh_f = _interp_cols(
        z, lut,
        ("y_inner_lo", "y_inner_hi", "y_outer_f1", "y_outer_f2", "x_half_finger"),
    )
    xh = xh_f + expand_m
    # +Y 指：从内侧面 y_hi 向外延伸到 y1_max
    f1 = (
        (q[:, 1] >= y_hi - expand_m * 0.5)
        & (q[:, 1] <= y1_max + expand_m)
        & (np.abs(q[:, 0]) <= xh)
        & (z >= lut["z"].min())
        & (z <= lut["z"].max())
    )
    # −Y 指：从内侧面 y_lo 向外延伸到 y2_min
    f2 = (
        (q[:, 1] <= y_lo + expand_m * 0.5)
        & (q[:, 1] >= y2_min - expand_m)
        & (np.abs(q[:, 0]) <= xh)
        & (z >= lut["z"].min())
        & (z <= lut["z"].max())
    )
    body = f1 | f2
    return body & ~gap_wedge_mask(q)


def finger_penetration_mask(q: np.ndarray, expand_m: float = 0.0) -> np.ndarray:
    """穿模判据：对称轴锚点接触 z 带内、指部实体、且不在开口体积楔形内。

    与体积统计共用 for_volume 楔形（含 y 微膨胀），避免薄壁同点既计体积又计穿模。
    """
    if len(q) == 0:
        return np.zeros(0, dtype=bool)
    lut = get_wedge_lut()
    z_tip, z_hi = gap_contact_z_bounds()
    z = q[:, 2]
    in_contact_z = (z >= z_tip) & (z <= z_hi)
    y_lo, y_hi, y1_max, y2_min, xh_f = _interp_cols(
        z, lut,
        ("y_inner_lo", "y_inner_hi", "y_outer_f1", "y_outer_f2", "x_half_finger"),
    )
    xh = xh_f + expand_m
    f1 = (
        (q[:, 1] >= y_hi - expand_m * 0.5)
        & (q[:, 1] <= y1_max + expand_m)
        & (np.abs(q[:, 0]) <= xh)
        & in_contact_z
    )
    f2 = (
        (q[:, 1] <= y_lo + expand_m * 0.5)
        & (q[:, 1] >= y2_min - expand_m)
        & (np.abs(q[:, 0]) <= xh)
        & in_contact_z
    )
    body = f1 | f2
    return body & ~gap_wedge_mask(q, for_volume=True)


def count_penetration_voxels(
    q: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_m: float = 0.004,
) -> Tuple[int, int]:
    """返回 (体素数, 原始点数)。mask 为 EEF 系穿模点布尔掩码。"""
    n_pts = int(mask.sum())
    if n_pts == 0:
        return 0, 0
    q_pen = np.asarray(q[mask], dtype=np.float64)
    vox = np.floor(q_pen / voxel_m).astype(np.int64)
    return int(np.unique(vox, axis=0).shape[0]), n_pts


def gap_rail_metrics(q_gap: np.ndarray) -> Dict[str, float]:
    """开口内物体到两指内侧轨（OBJ 内侧面）的距离。"""
    if len(q_gap) == 0:
        return {
            "min_rail_clear_mm": 0.0,
            "center_bias_mm": 999.0,
            "y_balance": 1.0,
        }
    lut = get_wedge_lut()
    z = q_gap[:, 2]
    y_lo, y_hi = _interp_cols(z, lut, ("y_inner_lo", "y_inner_hi"))
    ys = q_gap[:, 1]
    d_pos = y_hi - ys
    d_neg = ys - y_lo
    rail_clear = np.minimum(d_pos, d_neg)
    inside = (ys > y_lo) & (ys < y_hi)
    if inside.any():
        min_rail_clear_mm = float(rail_clear[inside].min() * 1000.0)
        center_bias_mm = float(np.abs(ys[inside]).mean() * 1000.0)
    else:
        min_rail_clear_mm = float(rail_clear.min() * 1000.0)
        center_bias_mm = float(np.abs(ys).mean() * 1000.0)
    mid_y = 0.5 * (y_lo + y_hi)
    n_pos = int((ys > mid_y + 1e-4).sum())
    n_neg = int((ys < mid_y - 1e-4).sum())
    denom = max(n_pos + n_neg, 1)
    y_balance = abs(n_pos - n_neg) / denom
    return {
        "min_rail_clear_mm": min_rail_clear_mm,
        "center_bias_mm": center_bias_mm,
        "y_balance": y_balance,
    }


def hit_to_slider_rail_mm(q_hit: np.ndarray) -> float:
    """VLM 点击 3D 点到夹爪滑轨（OBJ 内侧面）的贴近度（mm）。"""
    if len(q_hit) == 0:
        return 999.0
    lut = get_wedge_lut()
    z = float(q_hit[2])
    y_lo, y_hi = _interp_cols(np.array([z]), lut, ("y_inner_lo", "y_inner_hi"))
    y = float(q_hit[1])
    d_pos = float(y_hi[0] - y)
    d_neg = float(y - y_lo[0])
    return float(min(d_pos, d_neg) * 1000.0)
