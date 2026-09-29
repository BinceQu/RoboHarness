"""夹爪全开、全长度两指间通道 ∩ 物体 mesh 的体素相交体积。

开口域：保留旧版全长度楔形，但每个 (x, z) 体素列必须同时存在正、负 y
两侧手指实体，仅填充两指内侧面之间的空间，并剔除夹爪实体体素。这样不会把
单爪自身的空心结构误算成两爪之间的 grasp volume。
物体域：封闭 mesh 用 contains；开口薄壳用表面壳层（体素半边长）判定。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

OPENING_VOL_BUILD = "full_length_two_finger_corridor_eef_calibrated_voxel_v9"
_DEFAULT_VOXEL_MM = 3.0
_FULLY_OPEN_GRIPPER_QPOS_M = (0.05, 0.05)
# 薄壳占据须满足：最近面法向与开合方向(EEF y)夹角 ≤60° (|cos|≥0.5)
# 否则是「V腔贴在平行壁上」——两指合拢会滑脱，非有效夹持
_GRIP_ALIGN_COS = 0.5

# EEF 局部开口体素中心（与 pose 无关，构建一次缓存）
_OPENING_CENTERS_EEF: Dict[float, np.ndarray] = {}
_OBJECT_TM_CACHE: Dict[str, Any] = {}


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(4)]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _voxel_keys(points: np.ndarray, voxel_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.rint(points.reshape(-1, 3) / float(voxel_m)).astype(np.int64)


def _old_full_length_wedge_mask(
    points: np.ndarray,
    lut: Dict[str, Any],
) -> np.ndarray:
    """Evaluate the pre-change full-length sparse-vertex wedge."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    z = points[:, 2]
    z_bins = np.asarray(lut["z"], dtype=np.float64)
    y_lo = np.interp(z, z_bins, lut["y_inner_lo"])
    y_hi = np.interp(z, z_bins, lut["y_inner_hi"])
    x_half = np.interp(z, z_bins, lut["x_half_gap"])
    gap_width = np.interp(z, z_bins, lut["gap_width"])
    from behavior_interface.skills.plan_grasp_gripper_geom import (
        _MIN_GAP_WIDTH_M,
    )

    return (
        (z >= float(np.min(z_bins)))
        & (z <= float(np.max(z_bins)))
        & (points[:, 1] > y_lo)
        & (points[:, 1] < y_hi)
        & (np.abs(points[:, 0]) <= x_half)
        & (gap_width >= _MIN_GAP_WIDTH_M)
    )


def opening_voxel_centers_eef(voxel_m: float) -> np.ndarray:
    """在 EEF 系生成旧版全长度、但仅限两指之间的开口体素中心。"""
    voxel_m = float(voxel_m)
    if not math.isfinite(voxel_m) or voxel_m <= 0.0:
        raise ValueError(f"voxel_m 必须是正有限值，实际为 {voxel_m!r}")
    key = round(float(voxel_m), 6)
    if key in _OPENING_CENTERS_EEF:
        return _OPENING_CENTERS_EEF[key]

    from behavior_interface.skills.plan_grasp_gripper_geom import (
        get_wedge_lut,
    )

    components = gripper_component_voxels_eef(voxel_m)
    positive_keys = _voxel_keys(
        components["finger_positive_y"],
        voxel_m,
    )
    negative_keys = _voxel_keys(
        components["finger_negative_y"],
        voxel_m,
    )
    if len(positive_keys) == 0 or len(negative_keys) == 0:
        arr = np.zeros((0, 3), dtype=np.float64)
        _OPENING_CENTERS_EEF[key] = arr
        return arr

    positive_inner_y: Dict[Tuple[int, int], int] = {}
    negative_inner_y: Dict[Tuple[int, int], int] = {}
    for ix, iy, iz in positive_keys:
        column = (int(ix), int(iz))
        positive_inner_y[column] = min(
            int(iy),
            positive_inner_y.get(column, int(iy)),
        )
    for ix, iy, iz in negative_keys:
        column = (int(ix), int(iz))
        negative_inner_y[column] = max(
            int(iy),
            negative_inner_y.get(column, int(iy)),
        )

    corridor_keys = []
    shared_columns = sorted(
        set(positive_inner_y).intersection(negative_inner_y)
    )
    for ix, iz in shared_columns:
        y_lo = negative_inner_y[(ix, iz)]
        y_hi = positive_inner_y[(ix, iz)]
        corridor_keys.extend(
            (ix, iy, iz)
            for iy in range(y_lo + 1, y_hi)
        )
    if not corridor_keys:
        arr = np.zeros((0, 3), dtype=np.float64)
        _OPENING_CENTERS_EEF[key] = arr
        return arr

    corridor = (
        np.asarray(corridor_keys, dtype=np.float64) * voxel_m
    )
    corridor = corridor[
        _old_full_length_wedge_mask(
            corridor,
            get_wedge_lut(_FULLY_OPEN_GRIPPER_QPOS_M),
        )
    ]

    # The strict inner corridor should already avoid finger bodies. Removing the
    # complete solid set also excludes palm/root boundary voxels at coarse pitch.
    solid_keys = {
        tuple(value)
        for value in _voxel_keys(
            gripper_solid_voxels_eef(voxel_m),
            voxel_m,
        )
    }
    opening_keys = sorted(
        {
            tuple(value)
            for value in _voxel_keys(corridor, voxel_m)
        }.difference(solid_keys)
    )
    arr = (
        np.asarray(opening_keys, dtype=np.float64) * voxel_m
        if opening_keys
        else np.zeros((0, 3), dtype=np.float64)
    )
    _OPENING_CENTERS_EEF[key] = arr
    return arr


def load_object_trimesh_world(world, object_name: str, *, cache: bool = True):
    """合并物体全部 USD Mesh prim 为世界系 trimesh。"""
    if cache and object_name in _OBJECT_TM_CACHE:
        return _OBJECT_TM_CACHE[object_name]

    if world is None or not object_name:
        return None
    from behavior_interface.skills.grasp import _resolve_object_handle

    obj = _resolve_object_handle(world, object_name)
    if obj is None:
        return None
    try:
        import omnigibson as og
        import omnigibson.lazy as lazy
        import trimesh
        from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh

        pxr = lazy.pxr
        UsdGeom = pxr.UsdGeom
        stage = og.sim.stage
        prim_path = getattr(obj, "prim_path", None) or f"/World/{obj.name}"
        root = stage.GetPrimAtPath(str(prim_path))
        if not root.IsValid():
            return None
        # 仅用真实可见表面 mesh：排除碰撞凸块(填满凹腔) 与 meta 元链接(fillable 封盖袋口)
        def _ok(path: str) -> bool:
            p = str(path).lower()
            return ("collision" not in p) and ("meta__" not in p) and ("fillable" not in p)

        all_meshes = [p for p in pxr.Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
        mesh_prims = [p for p in all_meshes if _ok(p.GetPath())] or all_meshes
        if not mesh_prims:
            return None
        chunks = []
        for prim in mesh_prims:
            tm = mesh_prim_to_trimesh_mesh(
                prim, include_normals=False, include_texcoord=False, world_frame=True,
            )
            if tm is not None and len(tm.vertices) >= 3:
                chunks.append(tm)
        if not chunks:
            return None
        merged = chunks[0] if len(chunks) == 1 else trimesh.util.concatenate(chunks)
        if cache:
            _OBJECT_TM_CACHE[object_name] = merged
        return merged
    except Exception:
        return None


def _object_occupancy_mask(
    pts_world: np.ndarray,
    tm,
    voxel_m: float,
    open_dir_world: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """体素中心是否被物体占据。

    返回 (occ, occ_raw, method)：
      - occ      ：有效夹持占据（薄壳须叠加法向∥开合约束）
      - occ_raw  ：仅几何相交（不含法向约束，供对照）
    """
    if len(pts_world) == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, "empty"
    import trimesh

    # 封闭实心物体（苹果等）：体素在实体内部即占据，无需法向约束
    if bool(tm.is_watertight) and float(tm.volume) > 1e-10:
        inside = tm.contains(pts_world)
        return inside, inside, "watertight_contains"

    # 开口薄壳（爆米花袋等）：几何相交 + 壁面法向须与开合方向对齐
    _cp, dist, tri_id = trimesh.proximity.closest_point(tm, pts_world)
    shell_r = voxel_m * 0.5 * math.sqrt(3.0)
    near = dist <= shell_r
    if open_dir_world is None:
        return near, near, "surface_shell"
    yhat = np.asarray(open_dir_world, dtype=np.float64)
    yhat = yhat / (np.linalg.norm(yhat) + 1e-12)
    fn = np.asarray(tm.face_normals)[tri_id]
    cosang = np.abs(fn @ yhat)
    occ = near & (cosang >= _GRIP_ALIGN_COS)
    return occ, near, "surface_shell_grip_aligned"


def compute_opening_intersection_volume(
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    object_tm,
    *,
    voxel_mm: float = _DEFAULT_VOXEL_MM,
) -> Dict[str, Any]:
    """开口楔形体素 ∩ 物体 mesh → 开口相交体积。"""
    voxel_m = float(voxel_mm) / 1000.0
    eef_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
    R = _quat_to_mat(np.asarray(eef_quat, dtype=np.float64).reshape(4))
    centers_eef = opening_voxel_centers_eef(voxel_m)
    n_opening = int(len(centers_eef))
    if n_opening == 0 or object_tm is None:
        return {
            "opening_vol_build": OPENING_VOL_BUILD,
            "open_intersect_vox": 0,
            "open_intersect_vol_cm3": 0.0,
            "opening_voxel_total": n_opening,
            "open_vol_method": "none",
            "open_voxel_mm": voxel_mm,
            "object_watertight": False,
        }

    centers_world = (R @ centers_eef.T).T + eef_pos
    open_dir_world = R @ np.array([0.0, 1.0, 0.0])
    occ, occ_raw, method = _object_occupancy_mask(
        centers_world, object_tm, voxel_m, open_dir_world=open_dir_world,
    )
    n_hit = int(occ.sum())
    n_raw = int(occ_raw.sum())
    vol_m3 = n_hit * (voxel_m ** 3)
    return {
        "opening_vol_build": OPENING_VOL_BUILD,
        "open_intersect_vox": n_hit,
        "open_intersect_vol_cm3": float(vol_m3 * 1e6),
        "open_intersect_vox_raw": n_raw,
        "open_intersect_vol_raw_cm3": float(n_raw * voxel_m ** 3 * 1e6),
        "opening_voxel_total": n_opening,
        "open_vol_method": method,
        "open_voxel_mm": voxel_mm,
        "object_watertight": bool(getattr(object_tm, "is_watertight", False)),
        "open_intersect_frac": float(n_hit / max(n_opening, 1)),
    }


def annotate_opening_volume_on_image(
    img_path: str,
    *,
    open_intersect_vol_cm3: float,
    open_intersect_vox: int,
    raw_vol_cm3: float = 0.0,
    raw_vox: int = 0,
    legacy_gap_vol_cm3: float = 0.0,
    legacy_gap_vox: int = 0,
    ncol: int = 0,
    open_vol_method: str = "",
    out_path: Optional[str] = None,
) -> str:
    """在夹爪可视化图上叠加开口相交体积标注（有效夹持体积为主）。"""
    from PIL import Image, ImageDraw, ImageFont

    dst = out_path or img_path
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return img_path
    w, h = img.size
    draw = ImageDraw.Draw(img)
    pad = max(10, w // 80)
    fs = max(14, int(min(28, w / 55)))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", fs)
        font_big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", int(fs * 1.25),
        )
    except Exception:
        font = ImageFont.load_default()
        font_big = font
    lines = [
        (f"GRASP open_vol: {open_intersect_vol_cm3:.2f} cm3  vox={open_intersect_vox}", (0, 255, 0), font_big),
        (f"raw_surface(no align): {raw_vol_cm3:.2f} cm3  vox={raw_vox}", (255, 200, 0), font),
        (f"legacy_gap: {legacy_gap_vol_cm3:.2f} cm3   ncol={ncol}", (180, 180, 255), font),
    ]
    y = pad
    line_h = int(fs * 1.45)
    box_h = line_h * len(lines) + 10
    draw.rectangle((pad - 6, y - 4, w - pad, y + box_h), fill=(0, 0, 0))
    for text, color, fnt in lines:
        draw.text((pad, y), text, fill=color, font=fnt)
        y += line_h
    img.save(dst)
    return dst


# ── 夹爪实体 ∩ 物体 = overlap_vol ─────────────────────────────
_GRIPPER_VOX_EEF: Dict[float, np.ndarray] = {}
_GRIPPER_COMPONENT_VOX_EEF: Dict[float, Dict[str, np.ndarray]] = {}


def gripper_solid_voxels_eef(voxel_m: float) -> np.ndarray:
    """EEF 系夹爪实体体素中心（三爪 OBJ 体素化并填充），按 voxel 缓存。"""
    key = round(float(voxel_m), 6)
    if key in _GRIPPER_VOX_EEF:
        return _GRIPPER_VOX_EEF[key]
    from behavior_interface.skills.viz_eef_v2 import gripper_trimesh_eef

    tm = gripper_trimesh_eef()
    pts = np.zeros((0, 3), dtype=np.float64)
    if tm is not None:
        try:
            vg = tm.voxelized(pitch=float(voxel_m))
            try:
                vg = vg.fill()
            except Exception:
                pass
            pts = np.asarray(vg.points, dtype=np.float64)
        except Exception:
            pts = np.zeros((0, 3), dtype=np.float64)
    _GRIPPER_VOX_EEF[key] = pts
    return pts


def gripper_component_voxels_eef(voxel_m: float) -> Dict[str, np.ndarray]:
    """Return open-gripper and wrist-camera voxels separated by rigid link."""
    key = round(float(voxel_m), 6)
    cached = _GRIPPER_COMPONENT_VOX_EEF.get(key)
    if cached is not None:
        return cached

    from behavior_interface.skills.viz_eef_v2 import (
        _trimesh_eef_from_links,
        gripper_visual_links_eef,
    )

    component_names = {
        "right_gripper_link": "palm",
        "right_gripper_finger_link1": "finger_positive_y",
        "right_gripper_finger_link2": "finger_negative_y",
        "right_realsense_link": "camera",
    }
    components: Dict[str, np.ndarray] = {}
    for link_name, local_t, local_q in gripper_visual_links_eef(0.05):
        component_name = component_names.get(link_name)
        if component_name is None:
            continue
        tm = _trimesh_eef_from_links([(link_name, local_t, local_q)])
        pts = np.zeros((0, 3), dtype=np.float64)
        if tm is not None:
            try:
                vg = tm.voxelized(pitch=float(voxel_m))
                try:
                    vg = vg.fill()
                except Exception:
                    pass
                pts = np.asarray(vg.points, dtype=np.float64)
            except Exception:
                pts = np.zeros((0, 3), dtype=np.float64)
        components[component_name] = pts

    for component_name in component_names.values():
        components.setdefault(
            component_name,
            np.zeros((0, 3), dtype=np.float64),
        )
    _GRIPPER_COMPONENT_VOX_EEF[key] = components
    return components


def _points_inside_object(
    pts: np.ndarray,
    tm,
    obj_centroid: Optional[np.ndarray] = None,
    *,
    voxel_mm: float = _DEFAULT_VOXEL_MM,
) -> np.ndarray:
    """判定点是否与物体实体相交。

    封闭实心：contains。
    开口薄壳（袋/碗等）：仅距可见表面 ≤ 体素壳层半径算占据，**不把 fillable 内腔空域当实体**
    （与 _object_occupancy_mask 几何口径一致，避免袋内空气误判 overlap）。
    """
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    import trimesh

    if bool(getattr(tm, "is_watertight", False)) and float(getattr(tm, "volume", 0.0)) > 1e-10:
        try:
            return np.asarray(tm.contains(pts), dtype=bool)
        except Exception:
            pass
    voxel_m = float(voxel_mm) / 1000.0
    _cp, dist, _tri = trimesh.proximity.closest_point(tm, pts)
    shell_r = voxel_m * 0.5 * math.sqrt(3.0)
    return np.asarray(dist <= shell_r, dtype=bool)


def gripper_overlap_voxels_world(
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    object_tm,
    *,
    voxel_mm: float = _DEFAULT_VOXEL_MM,
    obj_centroid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """夹爪实体体素中落入物体实体侧的世界坐标 [N,3]（与 overlap_vol 同口径）。"""
    voxel_m = float(voxel_mm) / 1000.0
    eef_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
    R = _quat_to_mat(np.asarray(eef_quat, dtype=np.float64).reshape(4))
    vox = gripper_solid_voxels_eef(voxel_m)
    if len(vox) == 0 or object_tm is None:
        return np.zeros((0, 3), dtype=np.float64)
    pts_world = (R @ vox.T).T + eef_pos
    inside = _points_inside_object(
        pts_world, object_tm, obj_centroid, voxel_mm=voxel_mm)
    return np.asarray(pts_world[inside], dtype=np.float64)


def compute_gripper_overlap_volume(
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    object_tm,
    *,
    voxel_mm: float = _DEFAULT_VOXEL_MM,
    obj_centroid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """夹爪实体 ∩ 物体 的重叠体积（夹爪体素落入物体实体侧的体积）。"""
    voxel_m = float(voxel_mm) / 1000.0
    vox = gripper_solid_voxels_eef(voxel_m)
    n_total = int(len(vox))
    grip_vol_cm3 = float(n_total * (voxel_m ** 3) * 1e6)
    out = {
        "overlap_vox": 0,
        "overlap_vol_cm3": 0.0,
        "gripper_vol_cm3": grip_vol_cm3,
        "gripper_vox_total": n_total,
        "overlap_frac": 0.0,
        "overlap_voxel_mm": voxel_mm,
    }
    if n_total == 0 or object_tm is None:
        return out
    ovl_pts = gripper_overlap_voxels_world(
        eef_pos, eef_quat, object_tm,
        voxel_mm=voxel_mm, obj_centroid=obj_centroid)
    n_in = int(len(ovl_pts))
    out["overlap_vox"] = n_in
    out["overlap_vol_cm3"] = float(n_in * (voxel_m ** 3) * 1e6)
    out["overlap_frac"] = float(n_in / max(n_total, 1))
    return out


def clear_opening_volume_caches() -> None:
    """仿真物体变换后清缓存。

    夹爪体素 (_GRIPPER_VOX_EEF) 是 EEF 系下的常量几何，与物体变换无关，
    保留缓存避免每次 plan 重新体素化。
    """
    _OPENING_CENTERS_EEF.clear()
    _OBJECT_TM_CACHE.clear()


# v7 审计 / plan_grasp_object 共用口径（体素 3mm、可见 mesh、物心=物体根位姿）
V7_VOLUME_VOXEL_MM = _DEFAULT_VOXEL_MM
V7_VOLUME_BUILD = "v7_overlap_shell_not_cavity_v2"


def object_centroid_for_volume(world, object_name: str) -> Optional[np.ndarray]:
    """与 viz_grasp_obj_multiview v7 一致：物体根 link 世界位置，失败则用 AABB 中心。"""
    from behavior_interface.skills.grasp import _resolve_object_handle

    obj = _resolve_object_handle(world, object_name)
    if obj is None:
        return None
    try:
        return np.asarray(obj.get_position_orientation()[0], dtype=np.float64).reshape(3)
    except Exception:
        try:
            lo, hi = obj.aabb
            return (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64)) / 2.0
        except Exception:
            return None


def compute_grasp_obj_v7_metrics(
    world,
    object_name: str,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    gap_center: Optional[np.ndarray] = None,
    *,
    voxel_mm: float = V7_VOLUME_VOXEL_MM,
    clear_cache: bool = True,
) -> Dict[str, Any]:
    """grasp_vol + overlap_vol + anchor_dist；与 v7 head 2D 叠影审计同函数同参数。"""
    import trimesh

    if clear_cache:
        clear_opening_volume_caches()
    eef_p = np.asarray(eef_pos, dtype=np.float64).reshape(3)
    eef_q = np.asarray(eef_quat, dtype=np.float64).reshape(4)
    obj_c = object_centroid_for_volume(world, object_name)
    tm = load_object_trimesh_world(world, object_name)
    base = {
        "volume_v7_build": V7_VOLUME_BUILD,
        "volume_voxel_mm": float(voxel_mm),
        "grasp_vol_cm3": 0.0,
        "overlap_vol_cm3": 0.0,
        "gripper_vol_cm3": 0.0,
        "overlap_frac": 0.0,
        "anchor_dist_mm": 0.0,
        "marker_pt": None,
        "open_intersect_vol_cm3": 0.0,
    }
    if tm is None or obj_c is None:
        return base
    gv = compute_opening_intersection_volume(
        eef_p, eef_q, tm, voxel_mm=float(voxel_mm),
    )
    ovl = compute_gripper_overlap_volume(
        eef_p, eef_q, tm, voxel_mm=float(voxel_mm), obj_centroid=obj_c,
    )
    anchor_dist_mm = 0.0
    marker_pt = None
    if gap_center is not None:
        gap_c = np.asarray(gap_center, dtype=np.float64).reshape(1, 3)
        cp, d_a, _ = trimesh.proximity.closest_point(tm, gap_c)
        anchor_dist_mm = float(d_a[0] * 1000.0)
        marker_pt = np.asarray(cp[0], dtype=np.float64).tolist()
    return {
        **base,
        **gv,
        **ovl,
        "grasp_vol_cm3": float(gv.get("open_intersect_vol_cm3") or 0.0),
        "overlap_vol_cm3": float(ovl.get("overlap_vol_cm3") or 0.0),
        "anchor_dist_mm": anchor_dist_mm,
        "marker_pt": marker_pt,
        "obj_centroid": obj_c.tolist(),
    }
