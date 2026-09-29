"""grasp_obj 规划核心（与 diag_grasp_obj_pipeline unittest 同口径，供 plan_grasp_object 生产调用）。

流程（输入：head 冻结视角 + 点击 (u,v) 或 object_name 仅定目标物体）：

  0. object_name / seg 点云 → 解析目标物体 mesh（可见表面，排除 fillable/collision 填腔）
  1. mesh 表面积均匀采样池 → head 实例分割投影过滤 → 保留 100 个表面点
  2. 以每点为球心、夹爪开口宽度(~127mm) 为直径得 100 个球
  3. 3mm 体素占据：目标 fill 体素 + 环境(桌面/邻物)表面点 → 球∩占据体积
  4. 相交体积升序 → 取最小 10 点
  5. 10 点 × 正20面体 20 朝向 × 绕爪轴 8 等分自转 = 1600 EEF pose
     锚点 = 表面点 = 夹爪对称轴 1/3 黄点；eef = anchor - R @ gap_local
  6. fast_overlap 初筛 → top60 GPU 批量 v7（与 unittest v3 同口径）
  7. overlap 用夹爪法向膨胀 3mm 重算；overlap_vol < 1cm³ 中 grasp_vol 最大（同 v3 step8，无 DLS）
  8. 返回 best pose；head 主视图标记由 plan_eef_core 叠影（与 v3 同逻辑，不保存三视角）
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from behavior_interface.gripper_geometry_calibration import (
    R1PRO_GRIPPER_TO_EEF_FIXED_TRANSFORM,
)

from behavior_interface.head_capture import (
    HEAD_FOCAL_LENGTH,
    HEAD_HORIZONTAL_APERTURE,
    HEAD_IMAGE_HEIGHT,
    HEAD_IMAGE_WIDTH,
)

GRASP_OBJ_PIPELINE_BUILD = "v13_unittest_v3_aligned"
GRASP_OBJ_FILTER_PIPELINE_BUILD = "v20_filter_external_gpu_ik_budget120_ogfk_rank_top60"
N_SURFACE = 100
TOP_K_POINTS = 10
N_ICOSA_FACES = 20
N_ROLL = 8
N_POSES = TOP_K_POINTS * N_ICOSA_FACES * N_ROLL  # 1600
BALL_DIAMETER_MM = 126.9
VOXEL_MM = 3.0
ENV_COLLISION_MAX_CM3 = 0.5
PREFILTER_N = 60
# 与 unittest/diag v2/v3 一致
OVERLAP_VOL_MAX_CM3 = 1.0
OVERLAP_INFLATE_MM = 3.0
_GRIPPER_VOX_INFLATED_CACHE: Dict[tuple, np.ndarray] = {}

IK_FILTER_POS_TOL_M = 0.010
IK_FILTER_ORI_TOL_DEG = 3.0
IK_FILTER_BATCH_SIZE = 64
IK_FILTER_NUM_SEEDS = 24
IK_FILTER_IK_OPT_ITERS = 80
IK_FILTER_TOP_LOG_N = 20
IK_FILTER_OG_FK_TOP_N = 0
IK_FILTER_MAX_UNIQUE_TARGETS = 120
IK_FILTER_DEDUP_POS_DECIMALS = 4
IK_FILTER_DEDUP_QUAT_DECIMALS = 5
_STALE_PERSISTENT_IK_WORKERS = globals().get(
    "_PERSISTENT_IK_WORKERS",
    {},
)
_PERSISTENT_IK_WORKERS: Dict[tuple, Any] = {}

_IK_WORKER_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
)


def kdtree_query_workers() -> int:
    """Return the bounded host parallelism used by grasp KD-tree queries.

    ``cKDTree.query(workers=-1)`` expands to every visible CPU.  A grasp
    request is normally small and runs in the same process as the simulator;
    consuming the whole host here starves PhysX/Kit and other interfaces.  The
    cap remains operator-tunable for an isolated benchmark.
    """

    raw = os.environ.get("BEHAVIOR_KDTREE_WORKERS", "1").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1
    return max(1, min(value, 16))


def _ik_worker_environment(gpu: str) -> Dict[str, str]:
    """Create a single-card, bounded-CPU environment for cuRobo workers."""

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["IK_FILTER_CUDA_VISIBLE_DEVICES"] = str(gpu)
    thread_count = environment.get("BEHAVIOR_IK_CPU_THREADS", "1").strip()
    if not thread_count.isdigit() or int(thread_count) <= 0:
        raise ValueError("BEHAVIOR_IK_CPU_THREADS must be a positive integer")
    for variable in _IK_WORKER_THREAD_ENV_VARS:
        environment.setdefault(variable, thread_count)
    return environment


def _owned_ik_gpu() -> str:
    """Resolve the physical GPU for an external IK child.

    A single numeric CUDA mask is a physical id at the launcher boundary;
    derive it before the legacy GPU0 fallback so direct callers cannot send a
    child to a different card merely by omitting the IK-specific variable.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return (
        os.environ.get("IK_FILTER_CUDA_VISIBLE_DEVICES", "").strip()
        or os.environ.get("BEHAVIOR_INTERFACE_PHYSICAL_GPU", "").strip()
        or (visible if visible.isdigit() else "")
        or "0"
    )


def _py_float(v, default: float = 0.0) -> float:
    """强制转为 Python float（兼容 torch/numpy 标量）。"""
    if v is None:
        return float(default)
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except (ValueError, RuntimeError, TypeError):
            pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _py_int(v, default: int = 0) -> int:
    return int(round(_py_float(v, float(default))))


def _py_bool(v) -> bool:
    return _py_float(v, 0.0) > 0.0


def _sanitize_pose_metrics(pm: Dict[str, Any]) -> None:
    """batch GPU 后确保 pose 指标为原生 Python 标量，避免 DLS 池 bool(tensor) 崩溃。"""
    for k in (
        "grasp_vol_cm3", "overlap_vol_cm3", "gripper_vol_cm3", "overlap_frac",
        "anchor_dist_mm", "open_intersect_vol_cm3", "env_overlap_cm3",
    ):
        if k in pm:
            pm[k] = _py_float(pm[k])
    for k in ("grasp_vox", "overlap_vox", "open_intersect_vox", "opening_voxel_total",
              "env_overlap_vox", "pi", "ni", "ri"):
        if k in pm:
            pm[k] = _py_int(pm[k])
    if "eef_pos" in pm:
        pm["eef_pos"] = np.asarray(pm["eef_pos"], dtype=np.float64).reshape(3)
    elif "pos" in pm:
        pm["eef_pos"] = np.asarray(pm["pos"], dtype=np.float64).reshape(3)
    if "anchor" in pm:
        pm["anchor"] = np.asarray(pm["anchor"], dtype=np.float64).reshape(3)
    elif "gap_center" in pm:
        pm["anchor"] = np.asarray(pm["gap_center"], dtype=np.float64).reshape(3)
    if "R" in pm:
        pm["R"] = np.asarray(pm["R"], dtype=np.float64).reshape(3, 3)


def icosa_face_normals() -> np.ndarray:
    """正20面体 20 个顶点方向（单位向量）= 全空间均分朝向。"""
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    b, c = 1.0 / phi, phi
    verts: List[Tuple[float, float, float]] = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                verts.append((sx, sy, sz))
    for sy in (1.0, -1.0):
        for sz in (1.0, -1.0):
            verts.append((0.0, sy * b, sz * c))
    for sx in (1.0, -1.0):
        for sz in (1.0, -1.0):
            verts.append((sx * b, sz * c, 0.0))
    for sx in (1.0, -1.0):
        for sz in (1.0, -1.0):
            verts.append((sx * c, 0.0, sz * b))
    V = np.asarray(verts, dtype=np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return V


def _project_world(
    pts: np.ndarray, cam_pos: np.ndarray, cam_quat: np.ndarray,
    w: int, h: int, fl: float, ha: float,
):
    from behavior_interface.skills.viz_gripper_overlay import _project_points
    return _project_points(
        np.asarray(pts, dtype=np.float64).reshape(-1, 3),
        np.asarray(cam_pos, dtype=np.float64).reshape(3),
        np.asarray(cam_quat, dtype=np.float64).reshape(4),
        int(w), int(h), float(fl), float(ha),
    )


def head_seg_target_mask(
    head, target_prim_path: str, pts: np.ndarray,
    cam_pos, cam_quat, w: int, h: int, fl: float, ha: float,
):
    """表面点投影 head 实例分割，只保留命中目标 prim 的点。"""
    import omnigibson as og

    n = len(pts)
    keep = np.zeros(n, dtype=bool)
    reason = np.zeros(n, dtype=np.int8)
    have = list(getattr(head, "modalities", []))
    if "seg_instance_id" not in have:
        try:
            head.add_modality("seg_instance_id")
        except Exception:
            pass
    for _ in range(3):
        try:
            og.sim.render()
        except Exception:
            break
    try:
        obs, info = head.get_obs()
    except Exception as e:
        return keep, reason, {"error": f"get_obs 失败: {e}"}
    # 勿用 `a or b`：seg 常为 torch 张量，bool(tensor) 会报 ambiguous
    seg = obs.get("seg_instance_id")
    if seg is None:
        seg = obs.get("instance_id_segmentation")
    if seg is None:
        return keep, reason, {"error": "head 无 seg_instance_id"}
    if hasattr(seg, "detach"):
        seg = seg.detach().cpu().numpy()
    seg = np.asarray(seg)
    if seg.ndim == 3:
        seg = seg[..., 0]
    seg = seg.astype(np.int64)
    id2label: Dict[int, str] = {}
    raw = info.get("seg_instance_id") if isinstance(info, dict) else None
    if isinstance(raw, dict):
        for k, v in raw.items():
            id2label[int(k)] = str(v)
    tp = str(target_prim_path)
    target_ids = {i for i, lab in id2label.items() if lab == tp or lab.startswith(tp + "/")}
    uv, zc = _project_world(pts, cam_pos, cam_quat, w, h, fl, ha)
    for i in range(n):
        u, v, z = float(uv[i, 0]), float(uv[i, 1]), float(zc[i])
        if not (np.isfinite(u) and np.isfinite(v)) or z >= -1e-6:
            reason[i] = 1
            continue
        ui, vi = int(round(u)), int(round(v))
        if ui < 0 or vi < 0 or ui >= w or vi >= h:
            reason[i] = 1
            continue
        if int(seg[vi, ui]) in target_ids:
            keep[i] = True
        else:
            reason[i] = 2
    return keep, reason, {
        "target_prim": tp, "target_ids": sorted(target_ids),
        "n_keep": int(keep.sum()),
        "n_offscreen": int((reason == 1).sum()),
        "n_nontarget": int((reason == 2).sum()),
    }


def build_env_collision_tree(world, target_obj, voxel_m: float, *, near_m: float = 0.15):
    """目标附近环境物体表面点 KDTree（桌面/邻物）。"""
    from scipy.spatial import cKDTree
    import omnigibson as og
    import omnigibson.lazy as lazy
    import trimesh
    from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh

    pxr = lazy.pxr
    UsdGeom = pxr.UsdGeom
    stage = og.sim.stage
    try:
        tlo, thi = target_obj.aabb
        tlo = np.asarray(tlo, dtype=np.float64).reshape(3)
        thi = np.asarray(thi, dtype=np.float64).reshape(3)
    except Exception:
        return None, [], 0
    tlo_e, thi_e = tlo - near_m, thi + near_m
    tpath = str(getattr(target_obj, "prim_path", ""))
    try:
        objs = list(world.env.scene.objects)
    except Exception:
        try:
            objs = list(world.scene.objects)
        except Exception:
            return None, [], 0
    all_pts: List[np.ndarray] = []
    used: List[Dict[str, Any]] = []
    for o in objs:
        if o is target_obj:
            continue
        opath = str(getattr(o, "prim_path", ""))
        if opath and tpath and opath == tpath:
            continue
        try:
            lo, hi = o.aabb
            lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        except Exception:
            continue
        if np.any(hi < tlo_e) or np.any(lo > thi_e):
            continue
        root = stage.GetPrimAtPath(opath)
        if not root.IsValid():
            continue
        meshes = [p for p in pxr.Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
        if not meshes:
            continue
        col = [p for p in meshes if "collision" in str(p.GetPath()).lower()]
        use = col if col else [
            p for p in meshes
            if "meta__" not in str(p.GetPath()).lower()
            and "fillable" not in str(p.GetPath()).lower()
        ]
        chunks = []
        for prim in use:
            try:
                m = mesh_prim_to_trimesh_mesh(
                    prim, include_normals=False, include_texcoord=False, world_frame=True)
                if m is not None and len(m.vertices) >= 3:
                    chunks.append(m)
            except Exception:
                continue
        if not chunks:
            continue
        m = chunks[0] if len(chunks) == 1 else trimesh.util.concatenate(chunks)
        area = float(getattr(m, "area", 0.0)) or 1.0
        n_s = int(min(200000, max(2000, area / (voxel_m ** 2))))
        try:
            pts, _ = trimesh.sample.sample_surface(m, n_s)
            pts = np.asarray(pts, dtype=np.float64)
        except Exception:
            pts = np.asarray(m.vertices, dtype=np.float64)
        in_box = np.all((pts >= tlo_e) & (pts <= thi_e), axis=1)
        pts = pts[in_box]
        if len(pts):
            all_pts.append(pts)
            used.append({"name": getattr(o, "name", ""), "n_pts": int(len(pts))})
    if not all_pts:
        return None, [], 0
    return cKDTree(np.vstack(all_pts)), used, int(sum(u["n_pts"] for u in used))


def sample_surface_points_seg_filtered(
    world, object_name: str, tm, head, *,
    n_surface: int = N_SURFACE,
    cam_pos=None, cam_quat=None, w: int = HEAD_IMAGE_WIDTH, h: int = HEAD_IMAGE_HEIGHT,
    fl: float = HEAD_FOCAL_LENGTH, ha: float = HEAD_HORIZONTAL_APERTURE, ctx=None,
) -> Tuple[np.ndarray, Optional[str]]:
    """步骤1：表面积均匀采样 + head seg 过滤 → n_surface 点。

    返回 (pts, error)。seg 成功但保留 0 点时 error 非空（不再静默回退未过滤点）。
    """
    import trimesh
    from behavior_interface.skills.grasp import _resolve_object_handle

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("step1 表面采样")

    obj = _resolve_object_handle(world, object_name)
    pool_size = max(int(n_surface) * 15, 1500)
    pool_pts, _ = trimesh.sample.sample_surface(tm, pool_size)
    pool_pts = np.asarray(pool_pts, dtype=np.float64)
    seg_dbg: Dict[str, Any] = {}
    if head is not None and obj is not None:
        keep, _, seg_dbg = head_seg_target_mask(
            head, obj.prim_path, pool_pts, cam_pos, cam_quat, w, h, fl, ha)
        if seg_dbg.get("error"):
            if ctx:
                ctx.log(
                    f"  [grasp_obj/v13] seg 不可用: {seg_dbg['error']}，"
                    f"拒绝继续（避免未过滤表面点）")
            return np.zeros((0, 3), dtype=np.float64), (
                f"head 分割不可用: {seg_dbg['error']}")
        seg_applied = True
        valid = pool_pts[keep]
        if ctx:
            ctx.log(
                f"  [grasp_obj/v13] seg 池={pool_size} 保留={seg_dbg.get('n_keep')} "
                f"离屏={seg_dbg.get('n_offscreen')} 他物={seg_dbg.get('n_nontarget')} "
                f"target_ids={seg_dbg.get('target_ids')}")
        if len(valid) == 0:
            err = (
                f"seg 过滤后 0 个有效表面点（object={object_name} "
                f"prim={seg_dbg.get('target_prim')} "
                f"离屏={seg_dbg.get('n_offscreen')} 命中他物={seg_dbg.get('n_nontarget')}）"
                f"：物体在 head 视角不可见或 instance 未匹配，请先 move_to_object 再 capture"
            )
            if ctx:
                ctx.log(f"  [grasp_obj/v13] ERROR {err}")
            return np.zeros((0, 3), dtype=np.float64), err
    else:
        if ctx:
            ctx.log("  [grasp_obj/v13] ERROR 无 head 或物体句柄，无法 seg 过滤")
        return np.zeros((0, 3), dtype=np.float64), "无 head 相机或物体句柄，无法 seg 过滤表面点"

    if len(valid) >= n_surface:
        return valid[:n_surface], None
    if ctx:
        ctx.log(
            f"  [grasp_obj/v13] WARN seg 有效点 {len(valid)}<{n_surface}，使用全部有效点")
    return valid, None


def sphere_intersection_volumes(
    surf_pts: np.ndarray, tm, env_tree, *, voxel_m: float, radius_m: float,
) -> np.ndarray:
    """步骤3：每球心 query_ball 占据体素数 × 体素体积(cm³)。"""
    from scipy.spatial import cKDTree

    vg = tm.voxelized(pitch=voxel_m)
    try:
        vg = vg.fill()
    except Exception:
        pass
    obj_vox = np.asarray(vg.points, dtype=np.float64)
    if env_tree is not None and len(env_tree.data):
        occ_all = np.vstack([obj_vox, np.asarray(env_tree.data, dtype=np.float64)])
    else:
        occ_all = obj_vox
    if len(occ_all):
        keys = np.floor(occ_all / voxel_m).astype(np.int64)
        _, uniq_idx = np.unique(keys, axis=0, return_index=True)
        occ_pts = occ_all[uniq_idx]
    else:
        occ_pts = occ_all
    occ_tree = cKDTree(occ_pts) if len(occ_pts) else None
    vox_cm3 = voxel_m ** 3 * 1e6
    inter = np.zeros(len(surf_pts), dtype=np.float64)
    if occ_tree is not None:
        for i, c in enumerate(surf_pts):
            inter[i] = len(occ_tree.query_ball_point(c, radius_m)) * vox_cm3
    return inter


def generate_icosa_roll_poses(top_pts: np.ndarray, *, n_roll: int = N_ROLL) -> List[Dict[str, Any]]:
    """步骤5：10点×20面×8自转 → 1600 pose dict。"""
    from behavior_interface.skills.plan_grasp_gripper_fit import (
        _eef_frame_from_approach_roll, _mat_to_quat_xyzw)
    from behavior_interface.skills.plan_grasp_gripper_geom import gap_anchor_local_one_third_from_base

    normals = icosa_face_normals()
    gap_local = gap_anchor_local_one_third_from_base()
    rolls = [2.0 * np.pi * k / int(n_roll) for k in range(int(n_roll))]
    poses: List[Dict[str, Any]] = []
    for pi, anchor in enumerate(top_pts):
        for ni, nrm in enumerate(normals):
            approach = -nrm
            for ri, roll in enumerate(rolls):
                R = _eef_frame_from_approach_roll(approach, float(roll))
                eef_pos = anchor - R @ gap_local
                poses.append({
                    "pi": int(pi), "ni": int(ni), "ri": int(ri),
                    "anchor": np.asarray(anchor, dtype=np.float64).copy(),
                    "eef_pos": np.asarray(eef_pos, dtype=np.float64),
                    "R": R,
                    "quat": _mat_to_quat_xyzw(R),
                })
    return poses


def apply_camera_face_to_filter_poses(poses: List[Dict[str, Any]], world, ctx=None) -> None:
    """Apply final wrist-camera roll rule before IK so ranking matches exec pose."""
    if not poses:
        return
    try:
        from behavior_interface.skills.gripper_camera_face import ensure_camera_face_forward
        from behavior_interface.skills.plan_grasp_gripper_geom import gap_anchor_local_one_third_from_base
    except Exception as e:
        if ctx:
            ctx.log(f"  [grasp_obj_filter] WARN camera-face prefilter unavailable: {e}")
        return
    gap_local = gap_anchor_local_one_third_from_base()
    n_flip = 0
    n_skip = 0
    for p in poses:
        q, audit = ensure_camera_face_forward(p["quat"], world=world)
        p["camera_face"] = audit
        if audit.get("skipped"):
            n_skip += 1
            continue
        if audit.get("flipped"):
            n_flip += 1
        R = _quat_xyzw_to_mat_np(q)
        anchor = np.asarray(p["anchor"], dtype=np.float64).reshape(3)
        p["quat"] = q
        p["R"] = R
        p["eef_pos"] = anchor - R @ gap_local
    if ctx:
        ctx.log(
            f"  [grasp_obj_filter] camera-face 预处理 poses={len(poses)} "
            f"flipped={n_flip} skipped={n_skip}"
        )


def _ik_filter_pose_key(p: Dict[str, Any]) -> tuple:
    pos = np.asarray(p["eef_pos"], dtype=np.float64).reshape(3)
    quat = np.asarray(p["quat"], dtype=np.float64).reshape(4)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    if quat[3] < 0.0:
        quat = -quat
    return tuple(np.round(pos, IK_FILTER_DEDUP_POS_DECIMALS).tolist()) + tuple(
        np.round(quat, IK_FILTER_DEDUP_QUAT_DECIMALS).tolist()
    )


def _dedupe_ik_filter_poses(poses: List[Dict[str, Any]], ctx=None) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Deduplicate identical EEF targets before expensive dual-arm IK."""
    unique: List[Dict[str, Any]] = []
    unique_index: List[int] = []
    key_to_unique: Dict[tuple, int] = {}
    for p in poses:
        key = _ik_filter_pose_key(p)
        idx = key_to_unique.get(key)
        if idx is None:
            idx = len(unique)
            key_to_unique[key] = idx
            unique.append(p)
        unique_index.append(idx)
    if ctx and len(unique) != len(poses):
        ctx.log(
            f"  [grasp_obj_filter] IK EEF目标去重 {len(poses)}→{len(unique)} "
            f"(pos_dec={IK_FILTER_DEDUP_POS_DECIMALS} quat_dec={IK_FILTER_DEDUP_QUAT_DECIMALS})"
        )
    return unique, unique_index


def _expand_ik_filter_results(unique_results: List[Dict[str, Any]], unique_index: List[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx in unique_index:
        if 0 <= int(idx) < len(unique_results):
            out.append(dict(unique_results[int(idx)]))
        else:
            out.append({
                "ok": False,
                "pos_err_m": float("inf"),
                "ori_err_deg": float("inf"),
                "approach_err_deg": float("inf"),
                "error": "dedupe_expand_missing",
            })
    return out


def _external_ik_pose_request(pose: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize one EEF target plus optional warm starts and paired safe target."""
    item = {
        "eef_pos": [
            float(x)
            for x in np.asarray(pose["eef_pos"], dtype=np.float64).reshape(3)
        ],
        "quat": [
            float(x)
            for x in np.asarray(pose["quat"], dtype=np.float64).reshape(4)
        ],
    }
    raw_warm = pose.get("ik_warm_start_q_by_arm") or {}
    warm: Dict[str, List[float]] = {}
    for arm in ("left", "right"):
        q_arm = raw_warm.get(arm)
        if q_arm is None:
            continue
        q = np.asarray(q_arm, dtype=np.float64).reshape(-1)
        if len(q) == 7 and np.all(np.isfinite(q)):
            warm[arm] = [float(value) for value in q]
    if warm:
        item["warm_start_q_by_arm"] = warm
    raw_safe = pose.get("ik_paired_safe_pose")
    if isinstance(raw_safe, dict):
        safe = {
            "eef_pos": [
                float(x)
                for x in np.asarray(
                    raw_safe["eef_pos"], dtype=np.float64
                ).reshape(3)
            ],
            "quat": [
                float(x)
                for x in np.asarray(
                    raw_safe.get("quat", pose["quat"]),
                    dtype=np.float64,
                ).reshape(4)
            ],
            "pos_tol_m": float(raw_safe.get("pos_tol_m", 0.03)),
            "ori_tol_deg": float(raw_safe.get("ori_tol_deg", 10.0)),
            "final_branch_gap_rad": float(
                raw_safe.get("final_branch_gap_rad", 0.85)
            ),
        }
        item["paired_safe"] = safe
    return item


class _IKPoseSharedMemory:
    """Own a fixed-width shared-memory encoding of external IK pose requests."""

    VERSION = 1
    COLUMNS = 31
    DTYPE = np.dtype("<f8")

    def __init__(self, poses: List[Dict[str, Any]]):
        import os
        from multiprocessing import shared_memory

        records = np.full(
            (len(poses), self.COLUMNS),
            np.nan,
            dtype=self.DTYPE,
        )
        for index, pose in enumerate(poses):
            records[index, 0:3] = np.asarray(
                pose["eef_pos"],
                dtype=np.float64,
            ).reshape(3)
            records[index, 3:7] = np.asarray(
                pose["quat"],
                dtype=np.float64,
            ).reshape(4)
            warm = pose.get("warm_start_q_by_arm") or {}
            for arm, start in (("left", 7), ("right", 14)):
                q_arm = warm.get(arm)
                if q_arm is None:
                    continue
                q = np.asarray(q_arm, dtype=np.float64).reshape(-1)
                if len(q) == 7 and np.all(np.isfinite(q)):
                    records[index, start : start + 7] = q
            safe = pose.get("paired_safe")
            if isinstance(safe, dict):
                records[index, 21:24] = np.asarray(
                    safe["eef_pos"],
                    dtype=np.float64,
                ).reshape(3)
                records[index, 24:28] = np.asarray(
                    safe.get("quat", pose["quat"]),
                    dtype=np.float64,
                ).reshape(4)
                records[index, 28] = float(
                    safe.get("pos_tol_m", 0.03)
                )
                records[index, 29] = float(
                    safe.get("ori_tol_deg", 10.0)
                )
                records[index, 30] = float(
                    safe.get("final_branch_gap_rad", 0.85)
                )
        self._shm = shared_memory.SharedMemory(
            create=True,
            size=max(1, int(records.nbytes)),
        )
        if records.size:
            target = np.ndarray(
                records.shape,
                dtype=records.dtype,
                buffer=self._shm.buf,
            )
            target[...] = records
        self.descriptor = {
            "version": int(self.VERSION),
            "name": str(self._shm.name),
            "owner_pid": int(os.getpid()),
            "count": int(len(poses)),
            "columns": int(self.COLUMNS),
            "dtype": str(self.DTYPE.str),
            "bytes": int(records.nbytes),
        }
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._shm.close()
        finally:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


def _ik_filter_prefilter_sort_key(p: Dict[str, Any]) -> tuple:
    return (
        _py_int(p.get("env_overlap_vox", 0)),
        _py_int(p.get("fast_overlap", 0)),
        _py_int(p.get("pi", 0)),
        _py_int(p.get("ni", 0)),
        _py_int(p.get("ri", 0)),
    )


def _limit_ik_filter_candidates(
    poses: List[Dict[str, Any]],
    *,
    max_unique_targets: int,
    ctx=None,
) -> List[Dict[str, Any]]:
    """Bound expensive IK work to the best unique EEF targets after fast overlap."""
    if not poses:
        return []
    limit = int(max_unique_targets)
    if limit <= 0:
        limit = len(poses)
    selected: List[Dict[str, Any]] = []
    seen: set[tuple] = set()
    n_dup = 0
    for p in sorted(poses, key=_ik_filter_prefilter_sort_key):
        key = _ik_filter_pose_key(p)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        selected.append(p)
        if len(selected) >= limit:
            break
    if ctx:
        ctx.log(
            f"  [grasp_obj_filter] IK候选预算 fast_overlap排序 unique<={limit}: "
            f"poses={len(selected)}/{len(poses)} skipped_dup={n_dup}"
        )
    return selected


def gripper_solid_voxels_eef_inflated(
    voxel_m: float,
    inflate_m: Optional[float] = None,
) -> np.ndarray:
    """夹爪实体体素 + 外表面沿顶点法向膨胀（unittest v2/v3 overlap 口径）。"""
    if inflate_m is None:
        inflate_m = float(OVERLAP_INFLATE_MM) / 1000.0
    key = (round(float(voxel_m), 6), round(float(inflate_m), 6))
    if key in _GRIPPER_VOX_INFLATED_CACHE:
        return _GRIPPER_VOX_INFLATED_CACHE[key]
    from behavior_interface.skills.plan_grasp_opening_volume import gripper_solid_voxels_eef
    from behavior_interface.skills.viz_eef_v2 import gripper_trimesh_eef

    base_pts = gripper_solid_voxels_eef(voxel_m)
    tm = gripper_trimesh_eef()
    if tm is None or len(base_pts) == 0:
        _GRIPPER_VOX_INFLATED_CACHE[key] = base_pts
        return base_pts
    import trimesh

    verts = np.asarray(tm.vertices, dtype=np.float64)
    try:
        tm.fix_normals()
    except Exception:
        pass
    vn = np.asarray(tm.vertex_normals, dtype=np.float64)
    if len(vn) != len(verts):
        vn = np.zeros_like(verts)
    tm_out = tm.copy()
    tm_out.vertices = verts + float(inflate_m) * vn
    try:
        vg_out = tm_out.voxelized(pitch=float(voxel_m))
        try:
            vg_out = vg_out.fill()
        except Exception:
            pass
        pts_out = np.asarray(vg_out.points, dtype=np.float64)
    except Exception:
        pts_out = np.zeros((0, 3), dtype=np.float64)
    if len(pts_out) and len(base_pts):
        all_pts = np.vstack([base_pts, pts_out])
        keys = np.floor(all_pts / float(voxel_m)).astype(np.int64)
        _, uniq_idx = np.unique(keys, axis=0, return_index=True)
        merged = all_pts[uniq_idx]
    else:
        merged = pts_out if len(pts_out) else base_pts
    _GRIPPER_VOX_INFLATED_CACHE[key] = merged
    return merged


def patch_pose_overlap_inflated(
    pm: Dict[str, Any],
    gv_eef_inf: np.ndarray,
    object_tm,
    obj_centroid: np.ndarray,
    voxel_mm: float,
) -> None:
    """用膨胀后夹爪体素重算 overlap_vol（grasp_vol 保留 GPU/CPU v7 口径）。"""
    from behavior_interface.skills.plan_grasp_opening_volume import _points_inside_object

    voxel_m = float(voxel_mm) / 1000.0
    vox_cm3 = voxel_m ** 3 * 1e6
    if len(gv_eef_inf) == 0 or object_tm is None:
        pm["overlap_vol_cm3"] = 0.0
        pm["overlap_frac"] = 0.0
        pm["overlap_method"] = "inflate_gripper_normal"
        _sanitize_pose_metrics(pm)
        return
    R = np.asarray(pm["R"], dtype=np.float64)
    wv = (R @ gv_eef_inf.T).T + np.asarray(pm["eef_pos"], dtype=np.float64).reshape(3)
    inside = _points_inside_object(
        wv, object_tm, obj_centroid, voxel_mm=float(voxel_mm))
    n_in = int(np.asarray(inside, dtype=bool).sum())
    n_grip = int(len(gv_eef_inf))
    pm["overlap_vol_cm3"] = float(n_in * vox_cm3)
    pm["overlap_vox"] = n_in
    pm["overlap_frac"] = float(n_in / max(n_grip, 1))
    pm["gripper_vol_cm3"] = float(n_grip * vox_cm3)
    pm["overlap_method"] = f"inflate_gripper_normal_{int(round(float(OVERLAP_INFLATE_MM)))}mm"
    _sanitize_pose_metrics(pm)


def _apply_inflate_metrics_to_poses(
    poses: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    *,
    inflate_mm: float,
) -> None:
    """将 inflate overlap 指标写回 pose dict（保留 grasp_vol 等 GPU v7 字段）。"""
    tag = f"inflate_gripper_normal_{int(round(float(inflate_mm)))}mm"
    for pm, m in zip(poses, metrics):
        pm["overlap_vol_cm3"] = _py_float(m.get("overlap_vol_cm3", 0.0))
        pm["overlap_vox"] = _py_int(m.get("overlap_vox", 0))
        pm["overlap_frac"] = _py_float(m.get("overlap_frac", 0.0))
        pm["gripper_vol_cm3"] = _py_float(m.get("gripper_vol_cm3", 0.0))
        pm["overlap_method"] = m.get("overlap_method", tag)
        _sanitize_pose_metrics(pm)


def apply_overlap_inflate_to_poses(
    poses: List[Dict[str, Any]],
    object_tm,
    obj_centroid: np.ndarray,
    *,
    voxel_mm: float = VOXEL_MM,
    inflate_mm: float = OVERLAP_INFLATE_MM,
    ctx=None,
    use_gpu: bool = True,
    gpu_device_id: Optional[int] = None,
) -> None:
    """对 pose 列表批量应用膨胀 overlap（原地修改）；默认 GPU 壳层批量，失败回退 CPU。"""
    voxel_m = float(voxel_mm) / 1000.0
    inflate_m = float(inflate_mm) / 1000.0
    if inflate_m <= 0.0 or not poses:
        return
    n = len(poses)
    if use_gpu:
        try:
            import torch
            from behavior_interface.skills.grasp_obj_v7_gpu import (
                INFLATE_GPU_BUILD,
                batch_compute_inflate_overlap_gpu,
                pick_v7_gpu_device,
            )

            if torch.cuda.is_available():
                dev = pick_v7_gpu_device(gpu_device_id)
                from behavior_interface.skills.grasp_obj_v7_gpu import release_v7_gpu_memory

                metrics, meta = batch_compute_inflate_overlap_gpu(
                    poses, object_tm,
                    voxel_mm=float(voxel_mm), inflate_mm=float(inflate_mm),
                    device=dev)
                _apply_inflate_metrics_to_poses(
                    poses, metrics, inflate_mm=float(inflate_mm))
                release_v7_gpu_memory()
                if ctx:
                    ctx.log(
                        f"  [grasp_obj/v13] inflate overlap GPU {inflate_mm}mm × {n} pose "
                        f"{meta.get('elapsed_s', 0):.2f}s grip_vox={meta.get('n_grip_vox')} "
                        f"device={meta.get('device')} build={INFLATE_GPU_BUILD}")
                return
        except Exception as e:
            if ctx:
                ctx.log(f"  [grasp_obj/v13] WARN inflate GPU 回退 CPU: {e}")

    gv_inf = gripper_solid_voxels_eef_inflated(voxel_m, inflate_m=inflate_m)
    if ctx:
        ctx.log(
            f"  [grasp_obj/v13] inflate overlap CPU {inflate_mm}mm × {n} pose "
            f"(grip_vox={len(gv_inf)})")
    import time as _time
    t0 = _time.perf_counter()
    for i, pm in enumerate(poses):
        patch_pose_overlap_inflated(pm, gv_inf, object_tm, obj_centroid, float(voxel_mm))
        if ctx and n > 10 and (i + 1) % 10 == 0:
            ctx.log(f"  [grasp_obj/v13] inflate overlap CPU {i + 1}/{n}")
    if ctx:
        ctx.log(
            f"  [grasp_obj/v13] inflate overlap CPU 完成 {n} pose "
            f"{_time.perf_counter() - t0:.2f}s")


def object_voxel_kdtree(object_tm, voxel_m: float):
    """目标物体 3mm 体素占据 KDTree（与 diag step6 fast_overlap 同源）。"""
    from scipy.spatial import cKDTree

    vg = object_tm.voxelized(pitch=float(voxel_m))
    try:
        vg = vg.fill()
    except Exception:
        pass
    obj_vox = np.asarray(vg.points, dtype=np.float64)
    tree = cKDTree(obj_vox) if len(obj_vox) else None
    return tree, obj_vox


def mark_poses_overlap_env(
    poses: List[Dict[str, Any]],
    gv_eef: np.ndarray,
    obj_tree,
    env_tree,
    *,
    voxel_m: float,
) -> List[Dict[str, Any]]:
    """步骤6a：标注 fast_overlap + env_overlap（与 diag 一致，不剔除 pose）。

    数值口径与逐 pose 查询完全一致（同样的距离、同样的 shell_r 阈值），
    仅把 KDTree 查询按块合并成少量大查询：查询线程数由
    ``BEHAVIOR_KDTREE_WORKERS`` 限制，避免高负载机器上按整机核心数
    反复起停线程池。
    """
    shell_r = voxel_m * 0.5 * math.sqrt(3.0)
    vox_cm3 = voxel_m ** 3 * 1e6
    n_pose = len(poses)
    nv = int(len(gv_eef))
    if nv == 0:
        for pm in poses:
            pm["fast_overlap"] = 0
            pm["env_overlap_vox"] = 0
            pm["env_overlap_cm3"] = 0.0
        return poses
    fast = np.zeros(n_pose, dtype=np.int64)
    envc = np.zeros(n_pose, dtype=np.int64)
    # 每块约 2e6 个查询点：够大以摊薄线程池开销，够小以控制峰值内存
    chunk_poses = max(1, int(2_000_000 // nv))
    for s in range(0, n_pose, chunk_poses):
        sub = poses[s:s + chunk_poses]
        wv = np.concatenate(
            [(pm["R"] @ gv_eef.T).T + pm["eef_pos"] for pm in sub], axis=0)
        if obj_tree is not None:
            d, _ = obj_tree.query(wv, workers=kdtree_query_workers())
            fast[s:s + len(sub)] = (d <= shell_r).reshape(len(sub), nv).sum(axis=1)
        if env_tree is not None:
            de, _ = env_tree.query(wv, workers=kdtree_query_workers())
            envc[s:s + len(sub)] = (de <= shell_r).reshape(len(sub), nv).sum(axis=1)
    for i, pm in enumerate(poses):
        pm["fast_overlap"] = int(fast[i])
        pm["env_overlap_vox"] = int(envc[i])
        pm["env_overlap_cm3"] = float(envc[i] * vox_cm3)
    return poses


def step6_prefilter_v7_metrics(
    poses: List[Dict[str, Any]],
    object_tm,
    world,
    object_name: str,
    gv_eef: np.ndarray,
    env_tree=None,
    *,
    voxel_mm: float = VOXEL_MM,
    prefilter_n: int = PREFILTER_N,
    env_collision_max_cm3: float = ENV_COLLISION_MAX_CM3,
    ctx=None,
) -> List[Dict[str, Any]]:
    """步骤6（与 diag_grasp_obj_pipeline 同口径）：fast_overlap 初筛 + top-N CPU v7。"""
    from behavior_interface.skills.plan_grasp_opening_volume import compute_grasp_obj_v7_metrics

    voxel_m = float(voxel_mm) / 1000.0
    vox_cm3 = voxel_m ** 3 * 1e6
    obj_tree, _ = object_voxel_kdtree(object_tm, voxel_m)
    mark_poses_overlap_env(poses, gv_eef, obj_tree, env_tree, voxel_m=voxel_m)

    env_max_vox = max(2, int(float(env_collision_max_cm3) / vox_cm3))
    feasible = [p for p in poses if _py_int(p.get("env_overlap_vox", 0)) <= env_max_vox]
    if not feasible:
        if ctx:
            ctx.log("  [grasp_obj/v12] WARN 所有 pose 环境碰撞，放宽约束")
        feasible = list(poses)
    n_pre = min(int(prefilter_n), len(feasible))
    pre = sorted(feasible, key=lambda p: _py_int(p.get("fast_overlap", 0)))[:n_pre]
    if ctx:
        ctx.log(
            f"  [grasp_obj/v12] step6 fast_overlap→v7 可行={len(feasible)}/{len(poses)} "
            f"精确复算 top{n_pre} (diag 同口径)")

    for k, pm in enumerate(pre):
        vm = compute_grasp_obj_v7_metrics(
            world, object_name,
            np.asarray(pm["eef_pos"], dtype=np.float64),
            np.asarray(pm["quat"], dtype=np.float64),
            np.asarray(pm["anchor"], dtype=np.float64),
            voxel_mm=float(voxel_mm), clear_cache=False,
        )
        _apply_vm_to_pose(pm, vm, voxel_mm)
        if ctx and (k + 1) % 10 == 0:
            ctx.log(f"  [grasp_obj/v12] v7 精确复算 {k + 1}/{n_pre}")
    return pre


def step6_prefilter_v7_metrics_gpu(
    poses: List[Dict[str, Any]],
    object_tm,
    world,
    object_name: str,
    gv_eef: np.ndarray,
    env_tree=None,
    *,
    voxel_mm: float = VOXEL_MM,
    prefilter_n: int = PREFILTER_N,
    env_collision_max_cm3: float = ENV_COLLISION_MAX_CM3,
    ctx=None,
    gpu_device_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """步骤6 GPU 版：fast_overlap 初筛 + top-N GPU v7 批量复算。"""
    import time

    from behavior_interface.skills.grasp_obj_v7_gpu import (
        batch_compute_v7_metrics_gpu, pick_v7_gpu_device)

    voxel_m = float(voxel_mm) / 1000.0
    vox_cm3 = voxel_m ** 3 * 1e6
    obj_tree, _ = object_voxel_kdtree(object_tm, voxel_m)
    mark_poses_overlap_env(poses, gv_eef, obj_tree, env_tree, voxel_m=voxel_m)

    env_max_vox = max(2, int(float(env_collision_max_cm3) / vox_cm3))
    feasible = [p for p in poses if _py_int(p.get("env_overlap_vox", 0)) <= env_max_vox]
    if not feasible:
        if ctx:
            ctx.log("  [grasp_obj/v3] WARN 所有 pose 环境碰撞，放宽约束")
        feasible = list(poses)
    n_pre = min(int(prefilter_n), len(feasible))
    pre = sorted(feasible, key=lambda p: _py_int(p.get("fast_overlap", 0)))[:n_pre]
    dev = pick_v7_gpu_device(gpu_device_id)
    if ctx:
        ctx.log(
            f"  [grasp_obj/v3] step6 fast_overlap→GPU-v7 可行={len(feasible)}/{len(poses)} "
            f"精确复算 top{n_pre} device={dev}")

    t0 = time.perf_counter()
    gaps = np.stack([
        np.asarray(p["anchor"], dtype=np.float64).reshape(3) for p in pre
    ], axis=0)
    metrics, meta = batch_compute_v7_metrics_gpu(
        pre, object_tm, voxel_mm=float(voxel_mm),
        device=dev, gap_centers=gaps)
    for pm, vm in zip(pre, metrics):
        _apply_vm_to_pose(pm, vm, voxel_mm)
        pm["open_vol_method"] = vm.get("open_vol_method", "gpu_v7_surface")
    dt = time.perf_counter() - t0
    if ctx:
        ctx.log(
            f"  [grasp_obj/v3] GPU v7 批量完成 {len(pre)} pose "
            f"{dt:.2f}s (内核 {meta.get('elapsed_s', dt):.2f}s) "
            f"surf={meta.get('n_surface_samples')}")
    return pre


def step6_prefilter_dual_ik_v7_metrics_gpu(
    poses: List[Dict[str, Any]],
    object_tm,
    world,
    object_name: str,
    gv_eef: np.ndarray,
    env_tree=None,
    *,
    voxel_mm: float = VOXEL_MM,
    prefilter_n: int = PREFILTER_N,
    env_collision_max_cm3: float = ENV_COLLISION_MAX_CM3,
    ctx=None,
    gpu_device_id: Optional[int] = None,
    plan_arm: str = "any",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter by env fast-overlap, rank by batched left+right 6D IK, then GPU v7 top-N."""
    import os
    import time

    from behavior_interface.skills.grasp_obj_v7_gpu import (
        batch_compute_v7_metrics_gpu, pick_v7_gpu_device)

    t0 = time.perf_counter()
    voxel_m = float(voxel_mm) / 1000.0
    vox_cm3 = voxel_m ** 3 * 1e6
    obj_tree, _ = object_voxel_kdtree(object_tm, voxel_m)
    mark_poses_overlap_env(poses, gv_eef, obj_tree, env_tree, voxel_m=voxel_m)
    mark_dt = time.perf_counter() - t0

    env_max_vox = max(2, int(float(env_collision_max_cm3) / vox_cm3))
    feasible = [p for p in poses if _py_int(p.get("env_overlap_vox", 0)) <= env_max_vox]
    if not feasible:
        if ctx:
            ctx.log("  [grasp_obj_filter] WARN 所有 pose 环境碰撞，放宽到全部 pose 再做 IK 过滤")
        feasible = list(poses)

    if ctx:
        ctx.log(
            f"  [grasp_obj_filter] step6 fast_overlap 可行={len(feasible)}/{len(poses)} "
            f"(标注 {mark_dt:.1f}s) → 左右臂 GPU 6D IK rank(pos mm + ori deg)"
        )
    max_unique_targets = int(os.environ.get(
        "IK_FILTER_MAX_UNIQUE_TARGETS", str(IK_FILTER_MAX_UNIQUE_TARGETS)))
    ik_input = _limit_ik_filter_candidates(
        feasible, max_unique_targets=max_unique_targets, ctx=ctx)
    if not ik_input:
        return [], {
            "n_input": 0,
            "n_fast_overlap_feasible": int(len(feasible)),
            "n_ik_input": 0,
            "ik_max_unique_targets": int(max_unique_targets),
            "n_left_ok": 0,
            "n_right_ok": 0,
            "n_both_ok": 0,
            "n_ranked": 0,
            "n_gpu_v7": 0,
            "top20": [],
            "elapsed_total_s": float(time.perf_counter() - t0),
            "error": "no_ik_input_after_budget",
        }

    ik_ranked, ik_meta = filter_poses_dual_arm_ik(
        world, ik_input,
        pos_tol_m=IK_FILTER_POS_TOL_M,
        ori_tol_deg=IK_FILTER_ORI_TOL_DEG,
        ctx=ctx,
    )
    ik_meta["n_fast_overlap_feasible"] = int(len(feasible))
    ik_meta["n_ik_input"] = int(len(ik_input))
    ik_meta["ik_max_unique_targets"] = int(max_unique_targets)
    if not ik_ranked:
        ik_meta["n_gpu_v7"] = 0
        ik_meta["elapsed_total_s"] = float(time.perf_counter() - t0)
        return [], ik_meta

    plan_arm = str(plan_arm or "any").lower().strip()
    if plan_arm not in ("left", "right", "any"):
        plan_arm = "any"
    both_pool = [p for p in ik_ranked if p.get("dual_ik_ok")]
    left_pool = [p for p in ik_ranked if p.get("left_ik_ok")]
    right_pool = [p for p in ik_ranked if p.get("right_ik_ok")]
    recommended_arm: Optional[str] = None
    if plan_arm == "left":
        recommended_arm = "left"
        select_pool = left_pool
        selection = "forced_left_strict"
        if ctx:
            ctx.log(
                f"  [grasp_obj_filter] plan_arm=left: 从 left 严格可达池筛选 "
                f"left={len(left_pool)} both={len(both_pool)} right={len(right_pool)}"
            )
    elif plan_arm == "right":
        recommended_arm = "right"
        select_pool = right_pool
        selection = "forced_right_strict"
        if ctx:
            ctx.log(
                f"  [grasp_obj_filter] plan_arm=right: 从 right 严格可达池筛选 "
                f"right={len(right_pool)} both={len(both_pool)} left={len(left_pool)}"
            )
    elif len(both_pool) >= int(prefilter_n):
        select_pool = both_pool
        selection = "dual_strict"
    else:
        recommended_arm = "left" if len(left_pool) >= len(right_pool) else "right"
        select_pool = left_pool if recommended_arm == "left" else right_pool
        selection = f"single_{recommended_arm}_strict"
        if ctx:
            ctx.log(
                f"  [grasp_obj_filter] 双臂严格候选不足 top{prefilter_n}: "
                f"both={len(both_pool)} left={len(left_pool)} right={len(right_pool)} "
                f"→ recommended_arm={recommended_arm}"
            )
    ik_meta["plan_arm"] = plan_arm
    ik_meta["selection"] = selection
    ik_meta["recommended_arm"] = recommended_arm
    ik_meta["n_selection_pool"] = int(len(select_pool))
    if not select_pool:
        ik_meta["n_gpu_v7"] = 0
        ik_meta["elapsed_total_s"] = float(time.perf_counter() - t0)
        if ctx:
            if plan_arm in ("left", "right"):
                ctx.log(
                    f"  [grasp_obj_filter] FAIL plan_arm={plan_arm} 指定手臂没有严格可达候选；"
                    "保留 top20 最小误差用于诊断，不进入 GPU-v7/exec"
                )
            else:
                ctx.log(
                    "  [grasp_obj_filter] FAIL 左右单臂都没有严格可达候选；"
                    "保留 top20 最小误差用于诊断，不进入 GPU-v7/exec"
                )
        return [], ik_meta

    n_pre = min(int(prefilter_n), len(select_pool))
    pre = select_pool[:n_pre]
    dev = pick_v7_gpu_device(gpu_device_id)
    if ctx:
        ctx.log(
            f"  [grasp_obj_filter] IK ranked candidates={len(ik_ranked)}，"
            f"{selection} pool={len(select_pool)} IKallerr 最小 top{n_pre} "
            f"→ GPU-v7 device={dev}"
        )

    t_gpu = time.perf_counter()
    gaps = np.stack([
        np.asarray(p["anchor"], dtype=np.float64).reshape(3) for p in pre
    ], axis=0)
    metrics, meta = batch_compute_v7_metrics_gpu(
        pre, object_tm, voxel_mm=float(voxel_mm),
        device=dev, gap_centers=gaps)
    for pm, vm in zip(pre, metrics):
        _apply_vm_to_pose(pm, vm, voxel_mm)
        pm["open_vol_method"] = vm.get("open_vol_method", "gpu_v7_surface_filter_ik")
    gpu_dt = time.perf_counter() - t_gpu
    if ctx:
        ctx.log(
            f"  [grasp_obj_filter] GPU v7 批量完成 {len(pre)} pose "
            f"{gpu_dt:.2f}s (内核 {meta.get('elapsed_s', gpu_dt):.2f}s) "
            f"surf={meta.get('n_surface_samples')}"
        )
    ik_meta["n_gpu_v7"] = int(len(pre))
    ik_meta["gpu_v7_elapsed_s"] = float(gpu_dt)
    ik_meta["elapsed_total_s"] = float(time.perf_counter() - t0)
    return pre, ik_meta


def _extract_arm_q_from_curobo_js(
    *,
    world,
    mg,
    arm: str,
    js_obj,
    batch_i: int,
    saved_q,
):
    """Extract one arm's 7-DoF q from a cuRobo JointState in OG joint order."""
    robot = world.robot
    names_full = list(robot.joints.keys())
    try:
        arm_idx_full = [names_full.index(f"{arm}_arm_joint{i + 1}") for i in range(7)]
    except ValueError:
        return None, "missing_og_arm_joint"
    try:
        js_pos_full = js_obj.position.detach().cpu().float()
        js_names = list(js_obj.joint_names)
        if js_pos_full.dim() == 1:
            js_pos = js_pos_full
        else:
            rows = js_pos_full.reshape(-1, int(js_pos_full.shape[-1]))
            bi = max(0, min(int(batch_i), int(rows.shape[0]) - 1))
            js_pos = rows[bi]
    except Exception as e:
        return None, f"read_js_failed:{type(e).__name__}:{e}"

    by_name = {str(n): i for i, n in enumerate(js_names)}
    direct = []
    for jn in (f"{arm}_arm_joint{i + 1}" for i in range(7)):
        idx = by_name.get(jn)
        if idx is None or idx >= len(js_pos):
            direct = []
            break
        direct.append(float(js_pos[idx].item()))
    if len(direct) == 7:
        return np.asarray(direct, dtype=np.float64), f"direct_js_names n={len(js_names)}"

    q_full = saved_q.clone() if saved_q is not None else robot.get_joint_positions().clone()
    src_map = {str(n): i for i, n in enumerate(js_names)}
    wrote = 0
    for jn in names_full:
        idx = src_map.get(jn)
        if idx is not None and idx < len(js_pos):
            q_full[names_full.index(jn)] = js_pos[idx]
            wrote += 1
    if wrote == 0:
        dst_names = list(getattr(mg, "robot_joint_names", []) or [])
        if not dst_names:
            try:
                dst_names = list(mg.rollout_fn.kinematics.joint_names)
            except Exception:
                dst_names = []
        if len(dst_names) == int(js_pos.numel()):
            for i, jn in enumerate(dst_names):
                if jn in names_full:
                    q_full[names_full.index(jn)] = js_pos[i]
                    wrote += 1
    if wrote == 0:
        return None, f"no_name_overlap js_first={js_names[:4]}"
    return np.asarray([float(q_full[int(i)].item()) for i in arm_idx_full], dtype=np.float64), (
        f"full_reorder wrote={wrote}"
    )


def _get_curobo_filter_mg(world, *, batch_size: int = IK_FILTER_BATCH_SIZE, ctx=None):
    """Dedicated low-memory cuRobo IK-only MG for grasp-object ranking.

    Only the ARM embodiment is loaded.  The full OG wrapper normally loads all
    robot embodiments in ``robot.curobo_path``; doing that again inside the
    web process can OOM before any IK is actually solved.  The filter path only
    needs hand IK, so keep the MotionGenerator small and batch modestly.
    """
    robot_id = id(world.robot)
    batch_size = int(max(1, batch_size))
    cache = getattr(world, "_curobo_filter_mg_cache", None)
    if (
        isinstance(cache, dict)
        and cache.get("robot_id") == robot_id
        and int(cache.get("batch_size", 0)) == batch_size
        and int(cache.get("num_seeds", 0)) == IK_FILTER_NUM_SEEDS
        and int(cache.get("ik_opt_iters", 0)) == IK_FILTER_IK_OPT_ITERS
        and cache.get("mg") is not None
    ):
        return cache["mg"]

    from omnigibson.action_primitives.curobo import (
        CuRoboEmbodimentSelection,
        CuRoboMotionGenerator,
    )

    cfg_path_override = None
    try:
        import os as _os

        base_cfg = dict(world.robot.curobo_path)
        arm_path = base_cfg.get(CuRoboEmbodimentSelection.ARM)
        if arm_path:
            nt_path = arm_path.replace("_arm.yaml", "_arm_no_torso.yaml")
            if nt_path != arm_path and _os.path.exists(nt_path):
                arm_path = nt_path
            cfg_path_override = {CuRoboEmbodimentSelection.ARM: arm_path}
    except Exception as e:
        if ctx:
            ctx.log(f"  [grasp_obj_filter] WARN filter MG arm_no_torso cfg failed: {e}")

    try:
        import gc
        import torch as th

        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
            th.cuda.synchronize()
    except Exception:
        pass

    if ctx:
        ctx.log(
            f"  [grasp_obj_filter] init low-mem filter CuRobo MG "
            f"batch={batch_size} seeds={IK_FILTER_NUM_SEEDS} iters={IK_FILTER_IK_OPT_ITERS}"
        )
    candidates = []
    for bs in (batch_size, 12, 8, 4, 2, 1):
        if bs <= batch_size and bs not in candidates:
            candidates.append(int(bs))
    for bs_try in candidates:
        if bs_try > batch_size:
            continue
        try:
            mg = CuRoboMotionGenerator(
                robot=world.robot,
                robot_cfg_path=cfg_path_override,
                batch_size=int(bs_try),
                use_cuda_graph=False,
                debug=False,
                use_default_embodiment_only=False,
                collision_activation_distance=0.001,
                motion_cfg_kwargs={
                    "self_collision_check": False,
                    "num_ik_seeds": IK_FILTER_NUM_SEEDS,
                    "num_batch_ik_seeds": IK_FILTER_NUM_SEEDS,
                    "num_batch_trajopt_seeds": 1,
                    "num_trajopt_noisy_seeds": 1,
                    "num_trajopt_seeds": 1,
                    "num_graph_seeds": 1,
                    "ik_opt_iters": IK_FILTER_IK_OPT_ITERS,
                    "trajopt_tsteps": 8,
                    "finetune_trajopt_iters": 1,
                },
            )
            batch_size = int(bs_try)
            break
        except Exception as e:
            err_s = str(e).lower()
            if ctx:
                ctx.log(f"  [grasp_obj_filter] filter MG batch={bs_try} init failed: {type(e).__name__}: {e}")
            if "out of memory" not in err_s and "cuda" not in err_s and bs_try <= 16:
                raise
            try:
                import gc
                import torch as th

                gc.collect()
                if th.cuda.is_available():
                    th.cuda.empty_cache()
                    th.cuda.synchronize()
            except Exception:
                pass
    else:
        raise RuntimeError("failed to initialize filter CuRobo MG for all batch sizes")
    try:
        world._curobo_filter_mg_cache = {
            "robot_id": robot_id,
            "batch_size": batch_size,
            "num_seeds": IK_FILTER_NUM_SEEDS,
            "ik_opt_iters": IK_FILTER_IK_OPT_ITERS,
            "mg": mg,
        }
    except Exception:
        pass
    return mg


def _curobo_batch_err_vector(err, valid: int, *, prefer_max_over_links: bool = True):
    import torch as th

    valid = int(max(0, valid))
    if err is None or valid <= 0:
        return th.full((valid,), float("inf"))
    e = err.detach().float().reshape(tuple(err.shape))
    if e.dim() == 0:
        e = e.reshape(1)
    if e.dim() == 1:
        return e.flatten()[:valid]
    if e.shape[0] == valid:
        if e.dim() == 2:
            return (e.max(dim=1).values if prefer_max_over_links else e.min(dim=1).values)[:valid]
        return e.reshape(valid, -1).min(dim=1).values[:valid]
    if e.shape[-1] == valid:
        flat = e.reshape(-1, valid)
        return (flat.max(dim=0).values if prefer_max_over_links else flat.min(dim=0).values)[:valid]
    flat = e.flatten()
    if flat.numel() >= valid:
        return flat[:valid]
    out = th.full((valid,), float("inf"), dtype=flat.dtype, device=flat.device)
    out[: flat.numel()] = flat
    return out


def _curobo_batch_success_vector(success, valid: int):
    import torch as th

    valid = int(max(0, valid))
    if success is None:
        return th.zeros((valid,), dtype=th.bool)
    s = success.detach().bool()
    if s.dim() == 0:
        return s.reshape(1).repeat(valid)[:valid]
    if s.dim() == 1:
        if s.numel() >= valid:
            return s.flatten()[:valid]
        out = th.zeros((valid,), dtype=th.bool, device=s.device)
        out[: s.numel()] = s.flatten()
        return out
    if s.shape[0] == valid:
        return s.reshape(valid, -1).any(dim=1)[:valid]
    if s.shape[-1] == valid:
        return s.reshape(-1, valid).any(dim=0)[:valid]
    flat = s.flatten()
    if flat.numel() >= valid:
        return flat[:valid]
    out = th.zeros((valid,), dtype=th.bool, device=s.device)
    out[: flat.numel()] = flat
    return out


def _add_default_ee_current_pose_if_needed(world, mg, target_pos: Dict[str, Any], target_quat: Dict[str, Any],
                                           *, n: int, emb_sel) -> None:
    """Avoid OG's dummy zero-pose target when solving for a non-default EEF link."""
    import torch as th

    default_link = None
    try:
        default_link = mg.ee_link.get(emb_sel)
    except Exception:
        default_link = None
    if not default_link or default_link in target_pos:
        return
    default_arm = None
    for arm_name, link_name in world.robot.eef_link_names.items():
        if link_name == default_link:
            default_arm = arm_name
            break
    if default_arm is None:
        return
    ep = world.eef_pose(arm=default_arm)
    pos = th.tensor(list(ep["pos"]), dtype=th.float32).reshape(1, 3).repeat(int(n), 1)
    quat = th.tensor(list(ep["quat"]), dtype=th.float32).reshape(1, 4).repeat(int(n), 1)
    target_pos[default_link] = pos
    target_quat[default_link] = quat


def _curobo_batch_best_vectors(pe, re, err, success, valid: int):
    """Pick the best IK seed for each batch row and return pos/rot/success/seed vectors."""
    import torch as th

    valid = int(max(0, valid))
    if valid <= 0:
        zf = th.full((0,), float("inf"))
        zi = th.zeros((0,), dtype=th.long)
        zb = th.zeros((0,), dtype=th.bool)
        return zf, zf, zb, zi
    if pe is None:
        zf = th.full((valid,), float("inf"))
        zi = th.zeros((valid,), dtype=th.long)
        zb = th.zeros((valid,), dtype=th.bool)
        return zf, zf.clone(), zb, zi

    pe_t = pe.detach().float()
    re_t = re.detach().float() if re is not None else th.zeros_like(pe_t)
    err_t = err.detach().float() if err is not None else pe_t
    succ_t = success.detach().bool() if success is not None else th.zeros_like(err_t, dtype=th.bool)

    if err_t.dim() >= 2 and err_t.shape[0] >= valid:
        rows = err_t[:valid].reshape(valid, -1)
        succ_rows = succ_t[:valid].reshape(valid, -1) if succ_t.shape[0] >= valid else th.zeros_like(rows, dtype=th.bool)
        gated = rows.clone()
        if succ_rows.shape == gated.shape:
            gated[~succ_rows] = float("inf")
        best_seed = gated.argmin(dim=1)
        no_success = ~th.isfinite(gated.min(dim=1).values)
        if bool(no_success.any().item()):
            raw_seed = rows.argmin(dim=1)
            best_seed[no_success] = raw_seed[no_success]

        def _gather(metric, default_inf: bool = True):
            if metric is None:
                fill = float("inf") if default_inf else 0.0
                return th.full((valid,), fill, dtype=rows.dtype, device=rows.device)
            mt = metric.detach().float()
            if mt.dim() >= 2 and mt.shape[0] >= valid:
                mrows = mt[:valid].reshape(valid, -1)
                seed = th.clamp(best_seed, 0, mrows.shape[1] - 1)
                return mrows[th.arange(valid, device=mrows.device), seed]
            return _curobo_batch_err_vector(mt, valid)

        pos = _gather(pe_t)
        rot = _gather(re_t, default_inf=False)
        succ_best = (
            succ_rows[th.arange(valid, device=succ_rows.device), th.clamp(best_seed, 0, succ_rows.shape[1] - 1)]
            if succ_rows.shape == rows.shape
            else th.zeros((valid,), dtype=th.bool, device=rows.device)
        )
        return pos, rot, succ_best, best_seed.detach().long()

    pos = _curobo_batch_err_vector(pe_t, valid, prefer_max_over_links=True)
    rot = _curobo_batch_err_vector(re_t, valid, prefer_max_over_links=True)
    succ = _curobo_batch_success_vector(succ_t, valid)
    seed = th.zeros((valid,), dtype=th.long, device=pos.device)
    return pos, rot, succ, seed


def _extract_selected_arm_q_from_curobo_result(*, world, mg, arm: str, js_obj, batch_i: int, seed_i: int, saved_q):
    """Extract the selected IK seed as a 7-DoF arm q, if cuRobo exposes js_solution."""
    try:
        selected = js_obj[int(batch_i), int(seed_i)]
        q_arm, source = _extract_arm_q_from_curobo_js(
            world=world, mg=mg, arm=arm, js_obj=selected, batch_i=0, saved_q=saved_q)
        if q_arm is not None:
            return q_arm, f"selected_js[{batch_i},{seed_i}] {source}"
    except Exception:
        pass
    try:
        selected = js_obj[int(batch_i)]
        q_arm, source = _extract_arm_q_from_curobo_js(
            world=world, mg=mg, arm=arm, js_obj=selected, batch_i=int(seed_i), saved_q=saved_q)
        if q_arm is not None:
            return q_arm, f"selected_batch[{batch_i}] seed={seed_i} {source}"
    except Exception as e:
        return None, f"extract_selected_failed:{type(e).__name__}:{e}"
    return None, "extract_selected_failed"


def _current_q_by_joint_name(world) -> Dict[str, float]:
    robot = world.robot
    q = robot.get_joint_positions()
    names = list(robot.joints.keys())
    return {str(name): float(q[i].item()) for i, name in enumerate(names)}


def _arm_no_torso_curobo_yaml(world) -> str:
    import os as _os

    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    arm_path = dict(world.robot.curobo_path).get(CuRoboEmbodimentSelection.ARM)
    if not arm_path:
        raise RuntimeError("robot.curobo_path missing ARM config")
    nt_path = str(arm_path).replace("_arm.yaml", "_arm_no_torso.yaml")
    return nt_path if nt_path != arm_path and _os.path.exists(nt_path) else str(arm_path)


def _base_link_pose_for_curobo(world, robot_cfg_path: str, urdf_path: str | None = None) -> Dict[str, Any]:
    import yaml

    base_link = None
    if urdf_path:
        try:
            import xml.etree.ElementTree as ET

            root = ET.parse(str(urdf_path)).getroot()
            child_links = {
                j.find("child").attrib["link"]
                for j in root.findall("joint")
                if j.find("child") is not None
            }
            links = [lk.attrib["name"] for lk in root.findall("link")]
            roots = [lk for lk in links if lk not in child_links]
            if roots:
                base_link = roots[0]
        except Exception:
            base_link = None
    try:
        if not base_link:
            data = yaml.safe_load(open(robot_cfg_path, "r"))
            base_link = data["robot_cfg"]["kinematics"].get("base_link")
    except Exception:
        if not base_link:
            base_link = None
    if not base_link:
        base_link = "base_link"
    pos, quat = world.robot.links[str(base_link)].get_position_orientation()
    return {
        "name": str(base_link),
        "pos": [float(x) for x in pos],
        "quat": [float(x) for x in quat],
    }


def _quat_xyzw_to_mat_np(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def _mat_to_quat_xyzw_np(mat) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(max(tr + 1.0, 1e-12)) * 2.0
        q = np.array([
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ], dtype=np.float64)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
        q = np.array([
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[2, 1] - m[1, 2]) / s,
        ], dtype=np.float64)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
        q = np.array([
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
            (m[0, 2] - m[2, 0]) / s,
        ], dtype=np.float64)
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
        q = np.array([
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
            (m[1, 0] - m[0, 1]) / s,
        ], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def _pose_mat_np(pos, quat_xyzw) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _quat_xyzw_to_mat_np(quat_xyzw)
    out[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return out


def _fallback_eef_extra_link(arm: str, link_name: str) -> Dict[str, Any]:
    parent = f"{arm}_gripper_link"
    return {
        "parent_link_name": parent,
        "link_name": str(link_name),
        "fixed_transform": list(R1PRO_GRIPPER_TO_EEF_FIXED_TRANSFORM),
        "joint_type": "FIXED",
        "joint_name": f"{parent}_to_{link_name}_fixed_joint",
        "source": "fallback_r1pro_gripper_to_eef",
    }


def _canonicalize_r1pro_eef_fixed_transform(
    fixed_transform,
    *,
    position_tolerance_m: float = 2.0e-6,
    orientation_tolerance_deg: float = 1.0e-4,
) -> Tuple[List[float], Dict[str, Any]]:
    """Collapse simulator float jitter around the calibrated fixed transform."""
    raw = np.asarray(fixed_transform, dtype=np.float64).reshape(-1)
    nominal = np.asarray(
        R1PRO_GRIPPER_TO_EEF_FIXED_TRANSFORM,
        dtype=np.float64,
    )
    if raw.shape != (7,) or not np.all(np.isfinite(raw)):
        return [float(x) for x in nominal], {
            "canonicalized": True,
            "reason": "invalid_transform",
            "position_delta_m": None,
            "orientation_delta_deg": None,
        }

    raw_quat = raw[3:7]
    nominal_quat = nominal[3:7]
    raw_norm = float(np.linalg.norm(raw_quat))
    nominal_norm = float(np.linalg.norm(nominal_quat))
    if raw_norm <= 1.0e-12 or nominal_norm <= 1.0e-12:
        return [float(x) for x in raw], {
            "canonicalized": False,
            "reason": "invalid_quaternion",
            "position_delta_m": None,
            "orientation_delta_deg": None,
        }

    position_delta_m = float(np.linalg.norm(raw[:3] - nominal[:3]))
    quat_dot = float(
        abs(
            np.dot(
                raw_quat / raw_norm,
                nominal_quat / nominal_norm,
            )
        )
    )
    quat_dot = min(1.0, max(-1.0, quat_dot))
    orientation_delta_deg = float(
        math.degrees(2.0 * math.acos(quat_dot))
    )
    canonicalized = bool(
        position_delta_m <= float(position_tolerance_m)
        and orientation_delta_deg <= float(orientation_tolerance_deg)
    )
    selected = nominal if canonicalized else raw
    return [float(x) for x in selected], {
        "canonicalized": canonicalized,
        "reason": "within_calibration_tolerance" if canonicalized else "meaningful_delta",
        "position_delta_m": position_delta_m,
        "orientation_delta_deg": orientation_delta_deg,
        "position_tolerance_m": float(position_tolerance_m),
        "orientation_tolerance_deg": float(orientation_tolerance_deg),
    }


def _eef_extra_links_for_curobo_worker(world, ctx=None) -> Dict[str, Dict[str, Any]]:
    """Build URDF-mode cuRobo extra links for OG EEF links missing from R1Pro URDF."""
    out: Dict[str, Dict[str, Any]] = {}
    robot = world.robot
    for arm in ("left", "right"):
        link_name = str(robot.eef_link_names.get(arm, f"{arm}_eef_link"))
        parent = f"{arm}_gripper_link"
        extra = _fallback_eef_extra_link(arm, link_name)
        try:
            parent_link = robot.links.get(parent)
            child_link = robot.links.get(link_name)
            if parent_link is None or child_link is None:
                raise KeyError(f"missing link parent={parent_link is None} child={child_link is None}")
            p_pos, p_quat = parent_link.get_position_orientation()
            c_pos, c_quat = child_link.get_position_orientation()
            if hasattr(p_pos, "detach"):
                p_pos = p_pos.detach().cpu().numpy()
                p_quat = p_quat.detach().cpu().numpy()
                c_pos = c_pos.detach().cpu().numpy()
                c_quat = c_quat.detach().cpu().numpy()
            parent_tf = _pose_mat_np(p_pos, p_quat)
            child_tf = _pose_mat_np(c_pos, c_quat)
            rel = np.linalg.inv(parent_tf) @ child_tf
            q_xyzw = _mat_to_quat_xyzw_np(rel[:3, :3])
            extra["fixed_transform"] = [
                float(rel[0, 3]),
                float(rel[1, 3]),
                float(rel[2, 3]),
                float(q_xyzw[3]),
                float(q_xyzw[0]),
                float(q_xyzw[1]),
                float(q_xyzw[2]),
            ]
            extra["source"] = "sim_link_pose_parent_inv_child"
            canonical, _ = _canonicalize_r1pro_eef_fixed_transform(
                extra["fixed_transform"]
            )
            extra["fixed_transform"] = canonical
        except Exception as e:
            extra["fallback_reason"] = f"{type(e).__name__}: {e}"
            if ctx:
                ctx.log(
                    f"  [grasp_obj_filter] WARN {arm} EEF extra link 用 fallback "
                    f"parent={parent} child={link_name}: {e}"
                )
        out[arm] = extra
    return out


def _urdf_path_for_curobo_worker(world, robot_cfg_path: str) -> str:
    import os

    for attr in ("urdf_path", "_urdf_path"):
        try:
            p = getattr(world.robot, attr)
            if p and os.path.exists(str(p)):
                return str(p)
        except Exception:
            pass
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(robot_cfg_path), "..", "urdf"))
        preferred = os.path.join(root, "r1pro.urdf")
        if os.path.exists(preferred):
            return preferred
        for name in ("r1pro_with_meta_links.urdf", "r1pro_source.urdf", "r1_pro_with_gripper.urdf"):
            p = os.path.join(root, name)
            if os.path.exists(p):
                return p
    except Exception:
        pass
    raise RuntimeError("cannot locate R1Pro URDF for external IK worker")


def _normalize_active_ik_arms(active_arms) -> Tuple[str, ...]:
    requested = tuple(
        arm
        for arm in ("left", "right")
        if arm in set(active_arms or ("left", "right"))
    )
    return requested or ("left", "right")


def _inactive_ik_results(n: int, arm: str) -> List[Dict[str, Any]]:
    return [
        {
            "ok": False,
            "pos_err_m": float("inf"),
            "ori_err_deg": float("inf"),
            "approach_err_deg": float("inf"),
            "q_arm": None,
            "error": f"inactive_arm_not_requested:{arm}",
        }
        for _ in range(int(n))
    ]


def _persistent_ik_worker_signature(req: Dict[str, Any]) -> str:
    import hashlib
    import json

    static_req = {
        key: value
        for key, value in req.items()
        if key not in {
            "poses",
            # Request-level solve policy changes execution within an existing
            # solver but does not change its loaded robot configuration.
            "solver_policy",
            # This transform is consumed per request by world-to-base pose
            # conversion and does not alter the cuRobo solver configuration.
            "base_link_pose",
        }
    }
    payload = json.dumps(
        static_req,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class _PersistentExternalIKWorker:
    def __init__(
        self,
        *,
        arm: str,
        gpu: str,
        signature: str,
        command: List[str],
        env: Dict[str, str],
    ):
        import subprocess
        import threading

        self.arm = str(arm)
        self.gpu = str(gpu)
        self.signature = str(signature)
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            command,
            env=env,
            text=True,
            bufsize=1,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def alive(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        proc = self._proc
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def request(
        self,
        req: Dict[str, Any],
        *,
        timeout_s: float,
    ) -> Dict[str, Any]:
        import json
        import select
        import time

        with self._lock:
            if not self.alive():
                raise RuntimeError(
                    f"persistent IK worker arm={self.arm} exited "
                    f"rc={self._proc.returncode}"
                )
            request_id = f"{time.time_ns()}-{id(req)}"
            payload = {
                "request_id": request_id,
                "request": req,
            }
            if self._proc.stdin is None or self._proc.stdout is None:
                raise RuntimeError("persistent IK worker pipes unavailable")
            self._proc.stdin.write(
                json.dumps(payload, separators=(",", ":")) + "\n"
            )
            self._proc.stdin.flush()
            ready, _, _ = select.select(
                [self._proc.stdout],
                [],
                [],
                max(0.1, float(timeout_s)),
            )
            if not ready:
                raise TimeoutError(
                    f"persistent IK worker arm={self.arm} timed out "
                    f"after {float(timeout_s):.1f}s"
                )
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"persistent IK worker arm={self.arm} closed stdout "
                    f"rc={self._proc.poll()}"
                )
            response = json.loads(line)
            if response.get("request_id") != request_id:
                raise RuntimeError(
                    f"persistent IK worker arm={self.arm} response id mismatch"
                )
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"persistent IK worker arm={self.arm} invalid response"
                )
            if not result.get("ok"):
                raise RuntimeError(
                    f"persistent IK worker arm={self.arm} failed: "
                    f"{result.get('error')}\n{result.get('traceback', '')}"
                )
            return result


def _close_persistent_ik_workers(
    workers: Optional[Dict[tuple, Any]] = None,
) -> None:
    target = _PERSISTENT_IK_WORKERS if workers is None else workers
    for worker in list(target.values()):
        try:
            worker.close()
        except Exception:
            pass
    target.clear()


def _orphaned_persistent_ik_worker_pids(
    *,
    proc_root: str = "/proc",
    parent_pid: Optional[int] = None,
) -> List[int]:
    """Find untracked persistent IK workers owned directly by this process."""
    import os

    owner_pid = os.getpid() if parent_pid is None else int(parent_pid)
    worker_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "tools",
            "ik_filter_worker.py",
        )
    )
    found: List[int] = []
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return found
    for entry in entries:
        if not str(entry).isdigit():
            continue
        pid = int(entry)
        if pid == owner_pid:
            continue
        proc_dir = os.path.join(proc_root, str(pid))
        try:
            with open(
                os.path.join(proc_dir, "status"),
                "r",
                encoding="utf-8",
            ) as status_file:
                ppid = next(
                    int(line.split(":", 1)[1].strip())
                    for line in status_file
                    if line.startswith("PPid:")
                )
            with open(os.path.join(proc_dir, "cmdline"), "rb") as cmd_file:
                argv = [
                    arg.decode("utf-8", errors="replace")
                    for arg in cmd_file.read().split(b"\0")
                    if arg
                ]
        except (OSError, StopIteration, ValueError):
            continue
        if (
            ppid == owner_pid
            and worker_path in argv
            and "--persistent" in argv
        ):
            found.append(pid)
    return sorted(found)


def _terminate_orphaned_persistent_ik_workers() -> None:
    import os
    import signal
    import time

    pids = _orphaned_persistent_ik_worker_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 3.0
    remaining = set(pids)
    while remaining and time.monotonic() < deadline:
        remaining = {
            pid
            for pid in remaining
            if os.path.exists(f"/proc/{pid}")
        }
        if remaining:
            time.sleep(0.05)
    still_owned = set(_orphaned_persistent_ik_worker_pids())
    for pid in sorted(remaining & still_owned):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def shutdown_background_workers() -> None:
    """Release persistent subprocesses before this module is hot-reloaded."""
    _close_persistent_ik_workers()
    _terminate_orphaned_persistent_ik_workers()


def _persistent_ik_worker(
    *,
    arm: str,
    gpu: str,
    signature: str,
    worker_path: str,
    python_path: str,
    env: Dict[str, str],
) -> Tuple[_PersistentExternalIKWorker, bool]:
    import threading

    lock = globals().get("_PERSISTENT_IK_WORKERS_LOCK")
    if lock is None:
        lock = threading.Lock()
        globals()["_PERSISTENT_IK_WORKERS_LOCK"] = lock
    key = (str(gpu), str(arm))
    with lock:
        current = _PERSISTENT_IK_WORKERS.get(key)
        if (
            current is not None
            and current.signature == str(signature)
            and current.alive()
        ):
            return current, True
        if current is not None:
            current.close()
        worker = _PersistentExternalIKWorker(
            arm=arm,
            gpu=gpu,
            signature=signature,
            command=[
                python_path,
                "-u",
                worker_path,
                "--persistent",
                "--arm",
                arm,
            ],
            env=env,
        )
        _PERSISTENT_IK_WORKERS[key] = worker
        return worker, False


def _run_persistent_ik_arm(
    *,
    arm: str,
    gpu: str,
    signature: str,
    req: Dict[str, Any],
    worker_path: str,
    python_path: str,
    env: Dict[str, str],
    timeout_s: float,
) -> Tuple[Dict[str, Any], bool]:
    last_error: Optional[Exception] = None
    for attempt in range(2):
        worker, reused = _persistent_ik_worker(
            arm=arm,
            gpu=gpu,
            signature=signature,
            worker_path=worker_path,
            python_path=python_path,
            env=env,
        )
        try:
            return worker.request(req, timeout_s=timeout_s), reused
        except Exception as exc:
            last_error = exc
            key = (str(gpu), str(arm))
            current = _PERSISTENT_IK_WORKERS.pop(key, None)
            if current is not None:
                current.close()
            if attempt:
                break
    raise RuntimeError(
        f"persistent IK worker arm={arm} failed after restart: {last_error}"
    )


_close_persistent_ik_workers(_STALE_PERSISTENT_IK_WORKERS)
del _STALE_PERSISTENT_IK_WORKERS


def _run_external_gpu_ik_filter(
    world,
    poses: List[Dict[str, Any]],
    *,
    pos_tol_m: float,
    ori_tol_deg: float,
    active_arms: Tuple[str, ...] = ("left", "right"),
    ctx=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run pure IK in a separate CUDA process and return left/right per-pose metrics."""
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import time

    t0 = time.perf_counter()
    active_arms = _normalize_active_ik_arms(active_arms)
    unique_poses, unique_index = _dedupe_ik_filter_poses(poses, ctx=ctx)
    worker_policies = {
        str(pose.get("_ik_worker_policy") or "").strip()
        for pose in unique_poses
        if str(pose.get("_ik_worker_policy") or "").strip()
    }
    if len(worker_policies) > 1:
        raise ValueError(
            "mixed IK worker policies in one solve request: "
            f"{sorted(worker_policies)}"
        )
    worker_policy = (
        next(iter(worker_policies))
        if worker_policies
        else "baseline"
    )
    solver_config_policy = (
        worker_policy
        if worker_policy in {
            "cuda_graph_fixed64",
            "cuda_graph_fixed64_rewarm",
            "cuda_graph_split16_rewarm",
            "cuda_graph_split8_rewarm",
            "cuda_graph_warm32x6_rewarm",
        }
        else "baseline"
    )
    num_seeds = int(
        os.environ.get(
            "IK_FILTER_NUM_SEEDS",
            str(IK_FILTER_NUM_SEEDS),
        )
    )
    if worker_policy == "cuda_graph_warm32x6_rewarm":
        num_seeds = 6
    robot_cfg_path = _arm_no_torso_curobo_yaml(world)
    robot_urdf_path = _urdf_path_for_curobo_worker(world, robot_cfg_path)
    req = {
        "robot_cfg_path": robot_cfg_path,
        "robot_usd_path": str(world.robot.usd_path),
        "robot_urdf_path": robot_urdf_path,
        "base_link_pose": _base_link_pose_for_curobo(world, robot_cfg_path, robot_urdf_path),
        "eef_link_names": {arm: str(link) for arm, link in world.robot.eef_link_names.items()},
        "eef_extra_links": _eef_extra_links_for_curobo_worker(world, ctx=ctx),
        "q_by_name": _current_q_by_joint_name(world),
        "pos_tol_m": float(pos_tol_m),
        "ori_tol_deg": float(ori_tol_deg),
        "batch_size": int(os.environ.get("IK_FILTER_BATCH_SIZE", str(IK_FILTER_BATCH_SIZE))),
        "num_seeds": int(num_seeds),
        "ik_opt_iters": int(os.environ.get("IK_FILTER_IK_OPT_ITERS", str(IK_FILTER_IK_OPT_ITERS))),
        "solver_policy": str(worker_policy),
        "solver_config_policy": str(solver_config_policy),
        # Standalone worker does not run inside the Omniverse stage; cuRobo's
        # USD kinematics parser is registered by Isaac/OG lazy modules.  Use
        # the robot yaml kinematics here, then OG-FK revalidate the ranked top.
        "use_usd_kinematics": False,
        "poses": [_external_ik_pose_request(p) for p in unique_poses],
    }
    worker = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tools", "ik_filter_worker.py")
    )
    # The worker's CUDA mask is a physical id.  Derive it from a single-card
    # parent mask when callers bypass the shell launcher; otherwise an omitted
    # IK variable would silently send the child to GPU0.
    gpu = _owned_ik_gpu()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env = _ik_worker_environment(str(gpu))
    env["PYTHONPATH"] = repo_root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    py = os.environ.get("BEHAVIOR_PYTHON", sys.executable)
    worker_timeout = float(os.environ.get("IK_FILTER_WORKER_TIMEOUT_S", "180"))
    use_persistent = os.environ.get(
        "IK_FILTER_PERSISTENT_WORKER",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    signature = _persistent_ik_worker_signature(req)
    arm_data: Dict[str, Any] = {}
    worker_meta: Dict[str, Any] = {
        "n": int(len(unique_poses)),
        "parallel_arms": len(active_arms) > 1,
        "active_arms": list(active_arms),
        "persistent": bool(use_persistent),
        "pose_transport": "json",
        "solver_policy": str(worker_policy),
    }
    if ctx:
        ctx.log(
            f"  [grasp_obj_filter] external GPU IK worker gpu={gpu} "
            f"arms={','.join(active_arms)} poses={len(unique_poses)}/{len(poses)} "
            f"batch={req['batch_size']} seeds={req['num_seeds']} "
            f"persistent={int(use_persistent)}"
        )
    if use_persistent:
        import concurrent.futures

        shared_owner: Optional[_IKPoseSharedMemory] = None
        persistent_req = req
        use_shared_memory = os.environ.get(
            "IK_FILTER_SHARED_MEMORY",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        if use_shared_memory:
            try:
                shared_owner = _IKPoseSharedMemory(req["poses"])
                persistent_req = {
                    key: value
                    for key, value in req.items()
                    if key != "poses"
                }
                persistent_req["pose_shared_memory"] = dict(
                    shared_owner.descriptor
                )
                worker_meta["pose_transport"] = "shared_memory"
                worker_meta["pose_transport_bytes"] = int(
                    shared_owner.descriptor["bytes"]
                )
            except Exception as exc:
                worker_meta["pose_transport_fallback"] = (
                    f"{type(exc).__name__}: {exc}"
                )

        def run_arm(arm: str):
            return _run_persistent_ik_arm(
                arm=arm,
                gpu=str(gpu),
                signature=signature,
                req=persistent_req,
                worker_path=worker,
                python_path=py,
                env=env,
                timeout_s=worker_timeout,
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(active_arms)
            ) as executor:
                futures = {
                    arm: executor.submit(run_arm, arm)
                    for arm in active_arms
                }
                for arm, future in futures.items():
                    data, reused = future.result()
                    arm_data[arm] = data["arms"][arm]
                    arm_meta = dict(
                        (data.get("meta") or {}).get(arm, {})
                    )
                    arm_meta["persistent_reused"] = bool(reused)
                    worker_meta[arm] = arm_meta
        finally:
            if shared_owner is not None:
                shared_owner.close()
    else:
        with tempfile.TemporaryDirectory(prefix="grasp_obj_filter_ik_", dir="/tmp") as td:
            in_path = os.path.join(td, "request.json")
            with open(in_path, "w") as f:
                json.dump(req, f)
            procs: Dict[str, Any] = {}
            for arm in active_arms:
                out_path = os.path.join(td, f"result_{arm}.json")
                cmd = [py, "-u", worker, "--input", in_path, "--output", out_path, "--arm", arm]
                procs[arm] = {
                    "proc": subprocess.Popen(
                        cmd, env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE),
                    "out_path": out_path,
                }
            try:
                for arm, info in procs.items():
                    proc = info["proc"]
                    stdout, stderr = proc.communicate(timeout=worker_timeout)
                    if ctx and stdout:
                        diag_lines = [line for line in stdout.splitlines() if "GPU_DIAG" in line]
                        for line in diag_lines[-40:]:
                            ctx.log(f"  [grasp_obj_filter_worker/{arm}] {line[:4000]}")
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"external IK worker arm={arm} rc={proc.returncode} "
                            f"stdout={stdout[-2000:]} stderr={stderr[-4000:]}"
                        )
                    if not os.path.exists(info["out_path"]):
                        raise RuntimeError(
                            f"external IK worker arm={arm} produced no output "
                            f"stdout={stdout[-2000:]} stderr={stderr[-4000:]}"
                        )
                    with open(info["out_path"], "r") as result_file:
                        data = json.load(result_file)
                    if not data.get("ok"):
                        raise RuntimeError(
                            f"external IK worker arm={arm} failed: {data.get('error')}\n"
                            f"{data.get('traceback', '')}"
                        )
                    arm_data[arm] = data["arms"][arm]
                    worker_meta[arm] = (data.get("meta") or {}).get(arm, {})
            finally:
                for info in procs.values():
                    if info["proc"].poll() is None:
                        info["proc"].kill()
    left = (
        _expand_ik_filter_results(arm_data["left"], unique_index)
        if "left" in active_arms
        else _inactive_ik_results(len(poses), "left")
    )
    right = (
        _expand_ik_filter_results(arm_data["right"], unique_index)
        if "right" in active_arms
        else _inactive_ik_results(len(poses), "right")
    )
    meta = {
        "solver": "external_curobo_gpu_worker",
        "gpu": str(gpu),
        "elapsed_s": float(time.perf_counter() - t0),
        "n_requested": int(len(poses)),
        "n_unique": int(len(unique_poses)),
        "dedupe_pos_decimals": int(IK_FILTER_DEDUP_POS_DECIMALS),
        "dedupe_quat_decimals": int(IK_FILTER_DEDUP_QUAT_DECIMALS),
        "active_arms": list(active_arms),
        "persistent": bool(use_persistent),
        "solver_signature": signature[:16],
        "worker_meta": worker_meta,
    }
    return left, right, meta


def _load_filter_ik_solver(world, arm: str, *, pos_tol_m: float, ori_tol_deg: float, ctx=None):
    """Create a lightweight cuRobo IKSolver for one EEF link, with current locked joints.

    This deliberately avoids constructing a second MotionGen.  MotionGen allocates
    trajectory-optimization / MPPI buffers that are not needed for grasp-object
    candidate ranking and can OOM inside the already-running Isaac process.
    """
    import gc

    import torch as th
    import omnigibson.lazy as lazy

    robot = world.robot
    link_name = robot.eef_link_names[arm]
    robot_yaml = _arm_no_torso_curobo_yaml(world)
    q_by_name = _current_q_by_joint_name(world)
    tensor_args = lazy.curobo.types.base.TensorDeviceType(device=th.device("cuda:0"))
    content_path = lazy.curobo.types.file_path.ContentPath(
        robot_config_absolute_path=robot_yaml,
        robot_usd_absolute_path=robot.usd_path,
    )
    robot_cfg = lazy.curobo.cuda_robot_model.util.load_robot_yaml(content_path)["robot_cfg"]
    robot_cfg["kinematics"]["use_usd_kinematics"] = True
    robot_cfg["kinematics"]["ee_link"] = link_name
    for joint_name, lock_val in list(robot_cfg["kinematics"].get("lock_joints", {}).items()):
        if lock_val is None and joint_name in q_by_name:
            robot_cfg["kinematics"]["lock_joints"][joint_name] = float(q_by_name[joint_name])
    cspace = robot_cfg["kinematics"].get("cspace", {})
    if cspace.get("retract_config") is None:
        cspace["retract_config"] = [
            float(q_by_name.get(joint_name, 0.0))
            for joint_name in cspace.get("joint_names", [])
        ]

    try:
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
            th.cuda.synchronize()
    except Exception:
        pass
    if ctx:
        ctx.log(
            f"  [grasp_obj_filter] init lightweight GPU IKSolver arm={arm} "
            f"ee={link_name} batch={IK_FILTER_BATCH_SIZE} seeds={IK_FILTER_NUM_SEEDS} "
            f"iters={IK_FILTER_IK_OPT_ITERS}"
        )
    cfg = lazy.curobo.wrap.reacher.ik_solver.IKSolverConfig.load_from_robot_config(
        robot_cfg=robot_cfg,
        world_model=None,
        tensor_args=tensor_args,
        num_seeds=int(IK_FILTER_NUM_SEEDS),
        position_threshold=float(pos_tol_m),
        rotation_threshold=math.radians(float(ori_tol_deg)),
        use_cuda_graph=False,
        self_collision_check=False,
        self_collision_opt=False,
        use_particle_opt=False,
        collision_checker_type=None,
        grad_iters=int(IK_FILTER_IK_OPT_ITERS),
        high_precision=True,
        regularization=True,
        ee_link_name=link_name,
        project_pose_to_goal_frame=True,
        seed=1531,
    )
    solver = lazy.curobo.wrap.reacher.ik_solver.IKSolver(cfg)
    return solver, link_name, str(robot_cfg["kinematics"].get("base_link", ""))


def _curobo_world_pose_batch(world, *, base_link_name: str, positions: np.ndarray, quats_xyzw: np.ndarray,
                             tensor_args):
    import torch as th
    import omnigibson.lazy as lazy
    import omnigibson.utils.transform_utils as T

    pos_t = th.as_tensor(np.asarray(positions, dtype=np.float32), dtype=th.float32)
    quat_t = th.as_tensor(np.asarray(quats_xyzw, dtype=np.float32), dtype=th.float32)
    n = int(pos_t.shape[0])
    target_pose = th.zeros((n, 4, 4), dtype=th.float32)
    target_pose[:, 3, 3] = 1.0
    target_pose[:, :3, :3] = T.quat2mat(quat_t)
    target_pose[:, :3, 3] = pos_t
    robot_pos, robot_quat = world.robot.links[base_link_name].get_position_orientation()
    inv_robot_pose = T.pose_inv(T.pose2mat((robot_pos, robot_quat))).to(dtype=th.float32)
    target_pose = inv_robot_pose.view(1, 4, 4) @ target_pose
    local_pos = tensor_args.to_device(target_pose[:, :3, 3].contiguous())
    local_quat_xyzw = T.mat2quat(target_pose[:, :3, :3])
    local_quat_wxyz = tensor_args.to_device(local_quat_xyzw[:, [3, 0, 1, 2]].contiguous())
    return lazy.curobo.types.math.Pose(position=local_pos, quaternion=local_quat_wxyz)


def _active_q_batch_for_solver(world, solver, batch_n: int):
    import torch as th

    q_by_name = _current_q_by_joint_name(world)
    joint_names = list(solver.rollout_fn.kinematics.joint_names)
    q = th.tensor(
        [[float(q_by_name.get(jn, 0.0)) for jn in joint_names]],
        dtype=th.float32,
        device=solver.tensor_args.device,
    )
    return q.repeat(int(batch_n), 1).contiguous()


def _og_fk_validate_ranked_top(
    world,
    ranked: List[Dict[str, Any]],
    *,
    top_n: int,
    pos_tol_m: float = IK_FILTER_POS_TOL_M,
    ori_tol_deg: float = IK_FILTER_ORI_TOL_DEG,
    active_arms: Tuple[str, ...] = ("left", "right"),
    ctx=None,
) -> None:
    """Use OG FK to validate selected top IK candidates and overwrite their errors."""
    from behavior_interface.skills.eef import _eef_pose_err

    if not ranked or top_n <= 0:
        return
    active_arms = _normalize_active_ik_arms(active_arms)
    robot = world.robot
    saved = None
    names_full = list(robot.joints.keys())
    try:
        saved = robot.get_joint_positions().clone()
        arm_indices = {
            arm: [names_full.index(f"{arm}_arm_joint{i + 1}") for i in range(7)]
            for arm in ("left", "right")
        }
        n_done = 0
        for p in ranked[: int(top_n)]:
            eef_pos = np.asarray(p["eef_pos"], dtype=np.float64).reshape(3)
            quat = np.asarray(p["quat"], dtype=np.float64).reshape(4)
            allerr = 0.0
            for arm in active_arms:
                info = p.get("ik_filter", {}).get(arm, {})
                q_arm = info.get("q_arm")
                if q_arm is None:
                    info["og_pos_err_m"] = float("inf")
                    info["og_ori_err_deg"] = float("inf")
                    info["og_approach_err_deg"] = float("inf")
                    info["pos_err_m"] = float("inf")
                    info["ori_err_deg"] = float("inf")
                    info["approach_err_deg"] = float("inf")
                    info["ok"] = False
                    allerr = float("inf")
                    continue
                q_probe = saved.clone()
                for k, jidx in enumerate(arm_indices[arm]):
                    q_probe[int(jidx)] = float(q_arm[k])
                robot.set_joint_positions(q_probe)
                pos_err, ori_err, app_err = _eef_pose_err(world, arm, eef_pos, quat)
                info["og_pos_err_m"] = float(pos_err)
                info["og_ori_err_deg"] = float(ori_err)
                info["og_approach_err_deg"] = float(app_err)
                info["pos_err_m"] = float(pos_err)
                info["ori_err_deg"] = float(ori_err)
                info["approach_err_deg"] = float(app_err)
                info["ok"] = bool(
                    float(pos_err) <= float(pos_tol_m)
                    and float(ori_err) <= float(ori_tol_deg)
                )
                allerr += float(pos_err) * 1000.0 + float(ori_err)
            p["left_ik_ok"] = bool(p.get("ik_filter", {}).get("left", {}).get("ok"))
            p["right_ik_ok"] = bool(p.get("ik_filter", {}).get("right", {}).get("ok"))
            p["dual_ik_ok"] = bool(p["left_ik_ok"] and p["right_ik_ok"])
            p["ik_allerr"] = float(allerr)
            p["ik_validated_by_og_fk"] = True
            n_done += 1
        if ctx:
            ctx.log(f"  [grasp_obj_filter] OG FK 复核 top{n_done} IK候选并覆盖误差")
    finally:
        if saved is not None:
            try:
                robot.set_joint_positions(saved)
            except Exception:
                pass


def _batch_curobo_ik_errors_for_arm(
    world,
    arm: str,
    poses: List[Dict[str, Any]],
    *,
    pos_tol_m: float = IK_FILTER_POS_TOL_M,
    ori_tol_deg: float = IK_FILTER_ORI_TOL_DEG,
    max_attempts: int = 1,
    timeout_s: float = 20.0,
    ctx=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """cuRobo pure-GPU IK-only for many EEF poses, OG-FK-validated later."""
    import time

    import torch as th
    from behavior_interface.skills.eef import _quat_normalize_xyzw

    t0 = time.perf_counter()
    n = int(len(poses))
    out = [
        {
            "ok": False,
            "pos_err_m": float("inf"),
            "ori_err_deg": float("inf"),
            "approach_err_deg": float("inf"),
            "cu_pos_m": None,
            "cu_ori_deg": None,
            "cu_success": None,
            "error": "not_solved",
        }
        for _ in poses
    ]
    if n == 0:
        return out, {"n": 0, "ok": 0, "elapsed_s": 0.0}

    saved = None
    meta: Dict[str, Any] = {
        "arm": arm,
        "n": n,
        "ok": 0,
        "pos_tol_m": float(pos_tol_m),
        "ori_tol_deg": float(ori_tol_deg),
        "solver": "curobo_lightweight_iksolver_solve_batch",
    }
    solver = None
    try:
        saved = world.robot.get_joint_positions().clone()
        robot = world.robot
        solver, _link_name, base_link_name = _load_filter_ik_solver(
            world, arm, pos_tol_m=pos_tol_m, ori_tol_deg=ori_tol_deg, ctx=ctx)
        bs = int(max(1, IK_FILTER_BATCH_SIZE))
        meta["batch_size"] = bs
        meta["num_seeds"] = int(IK_FILTER_NUM_SEEDS)
        meta["ik_opt_iters"] = int(IK_FILTER_IK_OPT_ITERS)
        target_pos_all = np.stack([
            np.asarray(p["eef_pos"], dtype=np.float64).reshape(3) for p in poses
        ], axis=0)
        target_quat_all = np.stack([
            _quat_normalize_xyzw(p["quat"]) for p in poses
        ], axis=0)
        for offset in range(0, n, bs):
            end = min(n, offset + bs)
            valid = int(end - offset)
            goal_pose = _curobo_world_pose_batch(
                world,
                base_link_name=base_link_name,
                positions=target_pos_all[offset:end],
                quats_xyzw=target_quat_all[offset:end],
                tensor_args=solver.tensor_args,
            )
            retract = _active_q_batch_for_solver(world, solver, valid)
            r = solver.solve_batch(
                goal_pose,
                retract_config=retract,
                seed_config=None,
                return_seeds=1,
                num_seeds=int(IK_FILTER_NUM_SEEDS),
                use_nn_seed=False,
                newton_iters=int(IK_FILTER_IK_OPT_ITERS),
                link_poses=None,
            )
            pe = getattr(r, "position_error", None)
            re = getattr(r, "rotation_error", None)
            err = getattr(r, "error", None)
            js = getattr(r, "js_solution", None)
            if pe is None:
                continue
            succ = getattr(r, "success", None)
            pe_pb, re_pb, succ_pb, seed_pb = _curobo_batch_best_vectors(pe, re, err, succ, valid)
            for b in range(valid):
                idx = offset + b
                if idx >= n:
                    break
                pos_err = float(pe_pb[b].item()) if b < pe_pb.numel() else float("inf")
                ori_rad = float(re_pb[b].item()) if b < re_pb.numel() else float("inf")
                ori_err = float(math.degrees(ori_rad)) if math.isfinite(ori_rad) else float("inf")
                cu_success = (
                    bool(succ_pb[b].item())
                    if isinstance(succ_pb, th.Tensor) and b < succ_pb.numel()
                    else None
                )
                seed_i = int(seed_pb[b].item()) if isinstance(seed_pb, th.Tensor) and b < seed_pb.numel() else 0
                q_arm, q_source = (None, "no_js_solution")
                if js is not None:
                    q_arm, q_source = _extract_selected_arm_q_from_curobo_result(
                        world=world, mg=solver, arm=arm, js_obj=js,
                        batch_i=int(b), seed_i=seed_i, saved_q=saved)
                ok = bool(pos_err <= float(pos_tol_m) and ori_err <= float(ori_tol_deg))
                app_err = ori_err
                out[idx] = {
                    "ok": ok,
                    "pos_err_m": float(pos_err),
                    "ori_err_deg": float(ori_err),
                    "approach_err_deg": float(app_err),
                    "cu_pos_m": float(pos_err),
                    "cu_ori_deg": float(ori_err),
                    "cu_success": cu_success,
                    "q_arm": None if q_arm is None else q_arm.tolist(),
                    "seed_i": seed_i,
                    "source": f"curobo_iksolver offset={offset} batch={b} {q_source}",
                    "num_candidates": 1,
                }
                if ok:
                    meta["ok"] += 1
    except Exception as e:
        import traceback

        meta["error"] = f"{type(e).__name__}: {e}"
        meta["traceback"] = traceback.format_exc(limit=6)
        if ctx:
            ctx.log(f"  [grasp_obj_filter] {arm} IK 批量过滤异常: {meta['error']}")
    finally:
        if solver is not None:
            try:
                del solver
            except Exception:
                pass
        try:
            import gc
            gc.collect()
            if th.cuda.is_available():
                th.cuda.empty_cache()
                th.cuda.synchronize()
        except Exception:
            pass
        if saved is not None:
            try:
                world.robot.set_joint_positions(saved)
            except Exception:
                pass
        meta["elapsed_s"] = float(time.perf_counter() - t0)
    return out, meta


def filter_poses_dual_arm_ik(
    world,
    poses: List[Dict[str, Any]],
    *,
    pos_tol_m: float = IK_FILTER_POS_TOL_M,
    ori_tol_deg: float = IK_FILTER_ORI_TOL_DEG,
    max_attempts: int = 1,
    timeout_s: float = 20.0,
    keep_single_arm_results: bool = False,
    active_arms: Tuple[str, ...] = ("left", "right"),
    defer_og_fk_validation: bool = False,
    ctx=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rank poses by left+right 6D IK error; strict pass counts are reported separately."""
    import os
    import time

    t0 = time.perf_counter()
    active_arms = _normalize_active_ik_arms(active_arms)
    try:
        left, right, worker_meta = _run_external_gpu_ik_filter(
            world,
            poses,
            pos_tol_m=pos_tol_m,
            ori_tol_deg=ori_tol_deg,
            active_arms=active_arms,
            ctx=ctx,
        )
        lm = dict(worker_meta.get("worker_meta", {}).get("left", {}))
        rm = dict(worker_meta.get("worker_meta", {}).get("right", {}))
        lm.setdefault("elapsed_s", 0.0)
        rm.setdefault("elapsed_s", 0.0)
        lm["worker"] = worker_meta
        rm["worker"] = worker_meta
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        if ctx:
            ctx.log(f"  [grasp_obj_filter] external GPU IK worker 异常: {err}")
        if os.environ.get("IK_FILTER_ALLOW_INPROC_FALLBACK", "0") != "1":
            meta = {
                "n_input": int(len(poses)),
                "n_left_ok": 0,
                "n_right_ok": 0,
                "n_both_ok": 0,
                "n_ranked": 0,
                "n_og_fk_validated": 0,
                "pos_tol_m": float(pos_tol_m),
                "ori_tol_deg": float(ori_tol_deg),
                "max_attempts": int(max_attempts),
                "timeout_s": float(timeout_s),
                "active_arms": list(active_arms),
                "defer_og_fk_validation": bool(defer_og_fk_validation),
                "left": {"error": err, "elapsed_s": 0.0},
                "right": {"error": err, "elapsed_s": 0.0},
                "top20": [],
                "elapsed_s": float(time.perf_counter() - t0),
                "error": err,
            }
            return [], meta
        arm_results: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]] = {}
        for arm in active_arms:
            arm_results[arm] = _batch_curobo_ik_errors_for_arm(
                world,
                arm,
                poses,
                pos_tol_m=pos_tol_m,
                ori_tol_deg=ori_tol_deg,
                max_attempts=max_attempts,
                timeout_s=timeout_s,
                ctx=ctx,
            )
        left, lm = arm_results.get(
            "left",
            (_inactive_ik_results(len(poses), "left"), {"inactive": True}),
        )
        right, rm = arm_results.get(
            "right",
            (_inactive_ik_results(len(poses), "right"), {"inactive": True}),
        )
    ranked: List[Dict[str, Any]] = []
    for p, li, ri in zip(poses, left, right):
        l_pos_mm = float(li.get("pos_err_m", float("inf"))) * 1000.0
        r_pos_mm = float(ri.get("pos_err_m", float("inf"))) * 1000.0
        l_ori_deg = float(li.get("ori_err_deg", float("inf")))
        r_ori_deg = float(ri.get("ori_err_deg", float("inf")))
        left_finite = math.isfinite(l_pos_mm) and math.isfinite(l_ori_deg)
        right_finite = math.isfinite(r_pos_mm) and math.isfinite(r_ori_deg)
        finite_by_arm = {
            "left": left_finite,
            "right": right_finite,
        }
        error_by_arm = {
            "left": l_pos_mm + l_ori_deg,
            "right": r_pos_mm + r_ori_deg,
        }
        active_finite = [
            finite_by_arm[arm]
            for arm in active_arms
        ]
        if keep_single_arm_results:
            if not any(active_finite):
                continue
            allerr = sum(
                error_by_arm[arm]
                if finite_by_arm[arm]
                else 1.0e9
                for arm in active_arms
            )
        else:
            if not all(active_finite):
                continue
            allerr = sum(error_by_arm[arm] for arm in active_arms)
        if not math.isfinite(allerr):
            continue
        pp = dict(p)
        pp["ik_filter"] = {"left": li, "right": ri}
        pp["left_ik_ok"] = bool(li.get("ok"))
        pp["right_ik_ok"] = bool(ri.get("ok"))
        pp["dual_ik_ok"] = bool(li.get("ok") and ri.get("ok"))
        pp["ik_allerr"] = float(allerr)
        pp["ik_allerr_formula"] = " + ".join(
            f"{arm}_pos_mm + {arm}_ori_deg"
            for arm in active_arms
        )
        ranked.append(pp)
    ranked.sort(key=lambda p: (
        float(p.get("ik_allerr", float("inf"))),
        _py_int(p.get("fast_overlap", 0)),
    ))
    og_fk_env = int(os.environ.get("IK_FILTER_OG_FK_TOP_N", str(IK_FILTER_OG_FK_TOP_N)))
    og_fk_top_n = len(ranked) if og_fk_env <= 0 else max(PREFILTER_N, IK_FILTER_TOP_LOG_N, og_fk_env)
    worker_ranked_n = len(ranked)
    if not defer_og_fk_validation:
        _og_fk_validate_ranked_top(
            world,
            ranked,
            top_n=og_fk_top_n,
            pos_tol_m=float(pos_tol_m),
            ori_tol_deg=float(ori_tol_deg),
            active_arms=active_arms,
            ctx=ctx,
        )
        validated_ranked = [
            p
            for p in ranked
            if p.get("ik_validated_by_og_fk")
        ]
        if validated_ranked:
            ranked = validated_ranked
        elif ctx and worker_ranked_n:
            ctx.log(
                "  [grasp_obj_filter] WARN OG FK 未复核任何 IK 候选；"
                "为避免使用 worker raw 误差，本轮不返回可执行候选"
            )
            ranked = []
    ranked.sort(key=lambda p: (
        float(p.get("ik_allerr", float("inf"))),
        _py_int(p.get("fast_overlap", 0)),
    ))
    top20 = []
    for i, p in enumerate(ranked[:IK_FILTER_TOP_LOG_N], start=1):
        li = p["ik_filter"]["left"]
        ri = p["ik_filter"]["right"]
        item = {
            "rank": i,
            "allerr": round(float(p["ik_allerr"]), 3),
            "left_pos_mm": round(float(li.get("pos_err_m", float("inf"))) * 1000.0, 2),
            "left_ori_deg": round(float(li.get("ori_err_deg", float("inf"))), 2),
            "right_pos_mm": round(float(ri.get("pos_err_m", float("inf"))) * 1000.0, 2),
            "right_ori_deg": round(float(ri.get("ori_err_deg", float("inf"))), 2),
            "both_strict": bool(p.get("dual_ik_ok")),
            "fast_overlap": _py_int(p.get("fast_overlap", 0)),
        }
        top20.append(item)
    meta = {
        "n_input": int(len(poses)),
        "n_left_ok": int(sum(1 for p in ranked if p.get("left_ik_ok"))),
        "n_right_ok": int(sum(1 for p in ranked if p.get("right_ik_ok"))),
        "n_both_ok": int(sum(1 for p in ranked if p.get("dual_ik_ok"))),
        "n_ranked": int(len(ranked)),
        "n_worker_ranked": int(worker_ranked_n),
        "n_og_fk_validated": int(sum(1 for p in ranked if p.get("ik_validated_by_og_fk"))),
        "pos_tol_m": float(pos_tol_m),
        "ori_tol_deg": float(ori_tol_deg),
        "max_attempts": int(max_attempts),
        "timeout_s": float(timeout_s),
        "keep_single_arm_results": bool(keep_single_arm_results),
        "active_arms": list(active_arms),
        "defer_og_fk_validation": bool(defer_og_fk_validation),
        "left": lm,
        "right": rm,
        "top20": top20,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    if ctx:
        ctx.log(
            f"  [grasp_obj_filter] IK GPU rank input={len(poses)} ranked={meta['n_ranked']} "
            f"left={meta['n_left_ok']} right={meta['n_right_ok']} "
            f"both={meta['n_both_ok']} "
            f"elapsed={meta['elapsed_s']:.2f}s "
            f"(L {lm.get('elapsed_s', 0.0):.2f}s / R {rm.get('elapsed_s', 0.0):.2f}s) "
            f"arms={','.join(active_arms)} "
            f"reuse L/R={int(bool(lm.get('persistent_reused')))}/"
            f"{int(bool(rm.get('persistent_reused')))} "
            f"init L/R={int(bool(lm.get('solver_initialized_this_request')))}/"
            f"{int(bool(rm.get('solver_initialized_this_request')))}"
        )
        ctx.log(
            "  [grasp_obj_filter] IK top20 columns: "
            "rank, IKallerr, L_pos_mm, L_ori_deg, R_pos_mm, R_ori_deg, both_strict"
        )
        for item in top20:
            ctx.log(
                "  [grasp_obj_filter] IK top20: "
                "rank={rank:02d}, IKallerr={allerr:.3f}, "
                "L_pos_mm={left_pos_mm:.2f}, L_ori_deg={left_ori_deg:.2f}, "
                "R_pos_mm={right_pos_mm:.2f}, R_ori_deg={right_ori_deg:.2f}, "
                "both_strict={both_strict}".format(**item)
            )
    return ranked, meta


def _pose_ik_hard_constraint_summary(
    pm: Dict[str, Any],
    *,
    pos_tol_m: float = IK_FILTER_POS_TOL_M,
    ori_tol_deg: float = IK_FILTER_ORI_TOL_DEG,
) -> Dict[str, Any]:
    """Summarize which arms satisfy the selected pose's hard 6D IK constraint."""
    def _finite_round(v: float, ndigits: int = 3):
        return round(float(v), ndigits) if math.isfinite(float(v)) else None

    def _arm_summary(arm: str) -> Tuple[bool, Optional[float], Optional[float], Optional[list]]:
        info = (pm.get("ik_filter") or {}).get(arm) or {}
        pos_m = _py_float(info.get("pos_err_m"), float("inf"))
        ori_deg = _py_float(info.get("ori_err_deg"), float("inf"))
        ok = bool(
            math.isfinite(pos_m)
            and math.isfinite(ori_deg)
            and pos_m <= float(pos_tol_m)
            and ori_deg <= float(ori_tol_deg)
        )
        q_arm = None
        if ok and info.get("q_arm") is not None:
            try:
                q_arr = np.asarray(info.get("q_arm"), dtype=np.float64).reshape(7)
                q_arm = [round(float(x), 8) for x in q_arr.tolist()]
            except Exception:
                q_arm = None
        return ok, _finite_round(pos_m * 1000.0, 3), _finite_round(ori_deg, 3), q_arm

    left_ok, left_pos_mm, left_ori_deg, left_q_arm = _arm_summary("left")
    right_ok, right_pos_mm, right_ori_deg, right_q_arm = _arm_summary("right")
    both_ok = bool(left_ok and right_ok)
    if both_ok:
        solution = "both"
    elif left_ok:
        solution = "left"
    elif right_ok:
        solution = "right"
    else:
        solution = "none"
    return {
        "solution": solution,
        "left_ok": bool(left_ok),
        "right_ok": bool(right_ok),
        "both_ok": bool(both_ok),
        "left_pos_mm": left_pos_mm,
        "left_ori_deg": left_ori_deg,
        "right_pos_mm": right_pos_mm,
        "right_ori_deg": right_ori_deg,
        "pos_tol_mm": _finite_round(float(pos_tol_m) * 1000.0, 3),
        "ori_tol_deg": _finite_round(float(ori_tol_deg), 3),
        "hard_constraint": (
            f"pos<={float(pos_tol_m) * 1000.0:g}mm "
            f"&& ori<={float(ori_tol_deg):g}deg"
        ),
        "validated_by_og_fk": bool(pm.get("ik_validated_by_og_fk")),
        "ik_allerr": _finite_round(_py_float(pm.get("ik_allerr"), float("inf")), 3),
        "left_q_arm": left_q_arm,
        "right_q_arm": right_q_arm,
    }


def _selected_pose_ik_q_map(selected_pose_ik: Dict[str, Any]) -> Dict[str, Any]:
    """Return all stored final arm-q solutions carried by selected_pose_ik."""
    if not isinstance(selected_pose_ik, dict):
        return {}
    out: Dict[str, Any] = {}
    for arm in ("left", "right"):
        q = selected_pose_ik.get(f"{arm}_q_arm")
        if q is not None:
            out[arm] = q
    return out


def _format_ik_top20_for_error(ik_meta: Dict[str, Any], *, limit: int = 20) -> str:
    rows = ik_meta.get("top20") or []
    parts = []
    for item in rows[: int(limit)]:
        parts.append(
            "#{rank}:all={allerr} L={left_pos_mm}mm/{left_ori_deg}deg "
            "R={right_pos_mm}mm/{right_ori_deg}deg both={both_strict}".format(**item)
        )
    return "; ".join(parts)


def _apply_vm_to_pose(pm: Dict, vm: Dict, voxel_mm: float) -> None:
    pm["grasp_vol_cm3"] = _py_float(vm.get("grasp_vol_cm3", 0.0))
    pm["overlap_vol_cm3"] = _py_float(vm.get("overlap_vol_cm3", 0.0))
    pm["gripper_vol_cm3"] = _py_float(vm.get("gripper_vol_cm3", 0.0))
    pm["overlap_frac"] = _py_float(vm.get("overlap_frac", 0.0))
    pm["anchor_dist_mm"] = _py_float(vm.get("anchor_dist_mm", 0.0))
    vox_cm3 = (float(voxel_mm) / 1000.0) ** 3 * 1e6
    pm["grasp_vox"] = _py_int(pm["grasp_vol_cm3"] / max(vox_cm3, 1e-12))
    pm["overlap_vox"] = _py_int(pm["overlap_vol_cm3"] / max(vox_cm3, 1e-12))
    pm["open_intersect_vox"] = pm["grasp_vox"]
    pm["open_intersect_vol_cm3"] = pm["grasp_vol_cm3"]
    pm["open_vol_method"] = vm.get("open_vol_method", "cpu_v7")
    _sanitize_pose_metrics(pm)


def select_best_pose_low_overlap(
    pre: List[Dict[str, Any]],
    *,
    overlap_vol_max_cm3: float = OVERLAP_VOL_MAX_CM3,
    n_fallback: int = TOP_K_POINTS,
    ctx=None,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """步骤7/8：overlap_vol < 阈值（默认 0.01cm³）中 grasp_vol 最大；无则退化 min_overlap。"""
    if not pre:
        return None, []
    thr = float(overlap_vol_max_cm3)
    ovs = [_py_float(p.get("overlap_vol_cm3", 1e9)) for p in pre]
    if ctx:
        n_pass = sum(1 for ov in ovs if ov < thr)
        ov_min = min(ovs) if ovs else 0.0
        ctx.log(
            f"  [grasp_obj/v13] overlap 筛选 thr<{thr}cm³: "
            f"通过={n_pass}/{len(pre)} min_overlap={ov_min:.3f}cm³")
    cand = [p for p, ov in zip(pre, ovs) if ov < thr]
    if not cand:
        cand = sorted(
            pre, key=lambda p: _py_float(p.get("overlap_vol_cm3", 1e9)),
        )[: int(n_fallback)]
        if ctx:
            ctx.log(
                f"  [grasp_obj/v13] WARN 无 overlap<{thr}cm³，"
                f"退化 min_overlap top{len(cand)}")
    if not cand:
        return None, []
    best = max(cand, key=lambda p: _py_float(p.get("grasp_vol_cm3", 0.0)))
    if ctx:
        ctx.log(
            f"  [grasp_obj/v13] best grasp_vol={best['grasp_vol_cm3']:.2f} "
            f"overlap={best['overlap_vol_cm3']:.3f} "
            f"pi={best.get('pi')} ni={best.get('ni')} ri={best.get('ri')} "
            f"候选={len(cand)}/{len(pre)}")
    return best, cand


def pose_to_plan_candidate(pm: Dict[str, Any], shoulder: np.ndarray) -> Dict[str, Any]:
    """将 pipeline pose 转为 plan_grasp_object 候选 dict。"""
    _sanitize_pose_metrics(pm)
    gv = _py_float(pm.get("grasp_vol_cm3", 0.0))
    gvx = _py_int(pm.get("grasp_vox", pm.get("open_intersect_vox", 0)))
    return {
        "pos": np.asarray(pm["eef_pos"], dtype=np.float64),
        "quat": np.asarray(pm["quat"], dtype=np.float64),
        "gap_center": np.asarray(pm["anchor"], dtype=np.float64),
        "R": np.asarray(pm["R"], dtype=np.float64),
        "approach": np.asarray(pm["R"], dtype=np.float64)[:, 2],
        "approach_label": "v12_icosa",
        "ref_y_label": "v12",
        "roll_deg": 360.0 * _py_int(pm.get("ri", 0)) / float(N_ROLL),
        "grasp_vol_cm3": gv,
        "overlap_vol_cm3": _py_float(pm.get("overlap_vol_cm3", 0.0)),
        "gripper_vol_cm3": _py_float(pm.get("gripper_vol_cm3", 0.0)),
        "overlap_frac": _py_float(pm.get("overlap_frac", 0.0)),
        "anchor_dist_mm": _py_float(pm.get("anchor_dist_mm", 0.0)),
        "gap_voxel_n": gvx,
        "gap_vol_cm3": gv,
        "open_intersect_vox": _py_int(pm.get("open_intersect_vox", 0)),
        "open_intersect_vol_cm3": _py_float(pm.get("open_intersect_vol_cm3", 0.0)),
        "open_vol_method": pm.get("open_vol_method", "cpu_v7"),
        "open_voxel_mm": _py_float(pm.get("open_voxel_mm", VOXEL_MM)),
        "opening_voxel_total": _py_int(pm.get("opening_voxel_total", 0)),
        "has_volume": gv > 0.0,
        "ncol_exp": 0,
        "n_gap": gvx,
        "n_pos": 0,
        "n_neg": 0,
        "has_both": True,
        "gap_on_object": True,
        "volume_source": "v12_opening_voxel",
        "feasible": True,
        "reachable": False,
        "reach_reason": "pending_dls",
        "base_dist_mm": 0.0,
        "y_balance": 1.0,
        "center_bias_mm": 0.0,
        "min_rail_clear_mm": 0.0,
        "marker_pt": np.asarray(pm["anchor"], dtype=np.float64).reshape(3).tolist(),
        "pi": pm.get("pi"), "ni": pm.get("ni"), "ri": pm.get("ri"),
        "env_overlap_cm3": _py_float(pm.get("env_overlap_cm3", 0.0)),
        "mesh_entity_span_ok": True,
    }


def run_grasp_obj_v12_pipeline(
    world,
    object_name: str,
    session: Dict[str, Any],
    shoulder: np.ndarray,
    *,
    seed: int = 42,
    arm: str = "right",
    ctx=None,
    gpu_device_id: Optional[int] = None,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """执行 v13 全流程（unittest v3 口径），返回 (best_candidate_dict, plan_audit)。"""
    from behavior_interface.head_capture import (
        HEAD_FOCAL_LENGTH,
        HEAD_HORIZONTAL_APERTURE,
        HEAD_IMAGE_HEIGHT,
        HEAD_IMAGE_WIDTH,
        get_head_sensor,
        head_intrinsics_fallback,
        head_intrinsics_tuple,
    )
    from behavior_interface.skills.grasp import _resolve_object_handle
    from behavior_interface.skills.plan_grasp_opening_volume import (
        clear_opening_volume_caches,
        gripper_solid_voxels_eef, load_object_trimesh_world,
        object_centroid_for_volume)
    from behavior_interface.skills.plan_grasp_object import _load_session_head_view

    clear_opening_volume_caches()
    obj = _resolve_object_handle(world, object_name)
    tm = load_object_trimesh_world(world, object_name)
    obj_c = object_centroid_for_volume(world, object_name)
    if obj is None or tm is None or obj_c is None:
        if ctx:
            ctx.log("  [grasp_obj/v13] 物体/mesh/物心失败")
        return None

    head = get_head_sensor(world)
    depth, _, cam_meta, w, h = _load_session_head_view(session)
    if cam_meta:
        cam_pos = np.asarray(cam_meta.get("cam_pos", [0, 0, 0]), dtype=np.float64)
        cam_quat = np.asarray(cam_meta.get("cam_quat_xyzw", [0, 0, 0, 1]), dtype=np.float64)
        fl = float(cam_meta.get("focal_length", HEAD_FOCAL_LENGTH))
        ha = float(cam_meta.get("horizontal_aperture", HEAD_HORIZONTAL_APERTURE))
    elif head is not None:
        cp, cq = head.get_position_orientation()
        if hasattr(cp, "detach"):
            cp, cq = cp.detach().cpu().numpy(), cq.detach().cpu().numpy()
        cam_pos = np.asarray(cp, dtype=np.float64).reshape(3)
        cam_quat = np.asarray(cq, dtype=np.float64).reshape(4)
        fl, ha, w, h = head_intrinsics_tuple(head)
        if session.get("image_width") is not None:
            w = int(session["image_width"])
        if session.get("image_height") is not None:
            h = int(session["image_height"])
    else:
        cam_pos = cam_quat = None
        fl, ha, w, h = head_intrinsics_fallback()

    voxel_m = VOXEL_MM / 1000.0
    radius_m = BALL_DIAMETER_MM / 2000.0
    rng = np.random.default_rng(int(seed))

    if ctx:
        ctx.log(
            f"  [grasp_obj/v13] build={GRASP_OBJ_PIPELINE_BUILD} object={object_name} "
            f"步骤1-7 表面球筛→icosa×roll→fast_overlap→GPU-v7→inflate_overlap→best "
            f"thr_overlap<{OVERLAP_VOL_MAX_CM3}cm³ inflate={OVERLAP_INFLATE_MM}mm")

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("pipeline 启动")

    # 1-4
    surf_pts, seg_err = sample_surface_points_seg_filtered(
        world, object_name, tm, head,
        n_surface=N_SURFACE, cam_pos=cam_pos, cam_quat=cam_quat,
        w=w, h=h, fl=fl, ha=ha, ctx=ctx)
    if seg_err:
        from behavior_interface.errors import GraspObjPlanningError
        raise GraspObjPlanningError(seg_err)
    if len(surf_pts) < 3:
        from behavior_interface.errors import GraspObjPlanningError
        raise GraspObjPlanningError(
            f"有效表面点不足 ({len(surf_pts)}<3)，请 move_to_object 后重新 capture")

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("step3 球交体积")

    env_tree, env_used, env_npts = build_env_collision_tree(world, obj, voxel_m)
    inter_cm3 = sphere_intersection_volumes(
        surf_pts, tm, env_tree, voxel_m=voxel_m, radius_m=radius_m)
    top_idx = np.argsort(inter_cm3)[:TOP_K_POINTS]
    top_pts = surf_pts[top_idx]
    if ctx:
        ctx.log(
            f"  [grasp_obj/v13] 球交体积 top{TOP_K_POINTS}="
            f"{[round(float(inter_cm3[i]), 1) for i in top_idx]} "
            f"env_pts={env_npts}")

    # 5
    poses = generate_icosa_roll_poses(top_pts, n_roll=N_ROLL)
    if ctx:
        ctx.log(f"  [grasp_obj/v13] pose={len(poses)} ({TOP_K_POINTS}×{N_ICOSA_FACES}×{N_ROLL})")

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("step6 GPU-v7")

    # 6-7：与 unittest v3 step8 同口径 — GPU v7 → inflate overlap → 选 best（无 DLS、无三视角）
    gv_eef = gripper_solid_voxels_eef(voxel_m)
    pre = step6_prefilter_v7_metrics_gpu(
        poses, tm, world, object_name, gv_eef, env_tree,
        voxel_mm=VOXEL_MM,
        prefilter_n=PREFILTER_N,
        env_collision_max_cm3=ENV_COLLISION_MAX_CM3,
        ctx=ctx, gpu_device_id=gpu_device_id)
    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("inflate overlap")
    apply_overlap_inflate_to_poses(
        pre, tm, obj_c, voxel_mm=VOXEL_MM, inflate_mm=OVERLAP_INFLATE_MM,
        ctx=ctx, use_gpu=True, gpu_device_id=gpu_device_id)
    best_pm, cand = select_best_pose_low_overlap(
        pre,
        overlap_vol_max_cm3=OVERLAP_VOL_MAX_CM3,
        ctx=ctx,
    )
    if best_pm is None:
        from behavior_interface.errors import GraspObjPlanningError
        raise GraspObjPlanningError(
            f"无 overlap<{OVERLAP_VOL_MAX_CM3}cm³ 的抓取候选")

    best = pose_to_plan_candidate(best_pm, shoulder)
    best["reachable"] = True
    best["reach_reason"] = "v13_overlap_best"
    pick_audit = {"pick_method": "v13_overlap_best", "n_overlap_candidates": len(cand)}

    plan_audit: Dict[str, Any] = {
        "grasp_obj_build": GRASP_OBJ_PIPELINE_BUILD,
        "pipeline": "v13_gpu_inflate_overlap",
        "overlap_inflate_mm": float(OVERLAP_INFLATE_MM),
        "overlap_vol_max_cm3": float(OVERLAP_VOL_MAX_CM3),
        "env_collision_max_cm3": float(ENV_COLLISION_MAX_CM3),
        "use_gpu_step6": True,
        "object_name": object_name,
        "seed": int(seed),
        "n_surface": int(len(surf_pts)),
        "n_poses": int(len(poses)),
        "n_v7_prefilter": int(len(pre)),
        "n_overlap_candidates": int(len(cand)),
        "ball_diameter_mm": float(BALL_DIAMETER_MM),
        "voxel_mm": float(VOXEL_MM),
        "top10_inter_cm3": [float(inter_cm3[i]) for i in top_idx],
        "env_collision_objects": [u["name"] for u in env_used],
        "pick": pick_audit,
        "best_pi": best_pm.get("pi"),
        "best_ni": best_pm.get("ni"),
        "best_ri": best_pm.get("ri"),
    }
    if ctx:
        ctx.log(
            f"  [grasp_obj/v13] 最终 grasp_vol={best['grasp_vol_cm3']:.2f} "
            f"overlap_vol={best['overlap_vol_cm3']:.3f} "
            f"pi={best_pm.get('pi')} ni={best_pm.get('ni')} ri={best_pm.get('ri')}")
    return best, plan_audit


def run_grasp_obj_filter_pipeline(
    world,
    object_name: str,
    session: Dict[str, Any],
    shoulder: np.ndarray,
    *,
    seed: int = 42,
    arm: str = "right",
    ctx=None,
    gpu_device_id: Optional[int] = None,
    plan_arm: str = "any",
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """v15 filter: v13 grasp_obj, rank by left+right GPU IK before GPU top60."""
    from behavior_interface.head_capture import (
        HEAD_FOCAL_LENGTH,
        HEAD_HORIZONTAL_APERTURE,
        get_head_sensor,
        head_intrinsics_fallback,
        head_intrinsics_tuple,
    )
    from behavior_interface.skills.grasp import _resolve_object_handle
    from behavior_interface.skills.plan_grasp_opening_volume import (
        clear_opening_volume_caches,
        gripper_solid_voxels_eef, load_object_trimesh_world,
        object_centroid_for_volume)
    from behavior_interface.skills.plan_grasp_object import _load_session_head_view

    clear_opening_volume_caches()
    obj = _resolve_object_handle(world, object_name)
    tm = load_object_trimesh_world(world, object_name)
    obj_c = object_centroid_for_volume(world, object_name)
    if obj is None or tm is None or obj_c is None:
        if ctx:
            ctx.log("  [grasp_obj_filter] 物体/mesh/物心失败")
        return None

    head = get_head_sensor(world)
    depth, _, cam_meta, w, h = _load_session_head_view(session)
    if cam_meta:
        cam_pos = np.asarray(cam_meta.get("cam_pos", [0, 0, 0]), dtype=np.float64)
        cam_quat = np.asarray(cam_meta.get("cam_quat_xyzw", [0, 0, 0, 1]), dtype=np.float64)
        fl = float(cam_meta.get("focal_length", HEAD_FOCAL_LENGTH))
        ha = float(cam_meta.get("horizontal_aperture", HEAD_HORIZONTAL_APERTURE))
    elif head is not None:
        cp, cq = head.get_position_orientation()
        if hasattr(cp, "detach"):
            cp, cq = cp.detach().cpu().numpy(), cq.detach().cpu().numpy()
        cam_pos = np.asarray(cp, dtype=np.float64).reshape(3)
        cam_quat = np.asarray(cq, dtype=np.float64).reshape(4)
        fl, ha, w, h = head_intrinsics_tuple(head)
        if session.get("image_width") is not None:
            w = int(session["image_width"])
        if session.get("image_height") is not None:
            h = int(session["image_height"])
    else:
        cam_pos = cam_quat = None
        fl, ha, w, h = head_intrinsics_fallback()

    voxel_m = VOXEL_MM / 1000.0
    radius_m = BALL_DIAMETER_MM / 2000.0

    if ctx:
        plan_arm = str(plan_arm or "any").lower().strip()
        if plan_arm not in ("left", "right", "any"):
            plan_arm = "any"
        ctx.log(
            f"  [grasp_obj_filter] build={GRASP_OBJ_FILTER_PIPELINE_BUILD} object={object_name} "
            f"plan_arm={plan_arm} "
            "步骤1-5 同 grasp_obj/v13；step6 fast_overlap→双臂GPU IK rank→top60 GPU-v7"
        )
    else:
        plan_arm = str(plan_arm or "any").lower().strip()
        if plan_arm not in ("left", "right", "any"):
            plan_arm = "any"

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("grasp_obj_filter pipeline 启动")

    surf_pts, seg_err = sample_surface_points_seg_filtered(
        world, object_name, tm, head,
        n_surface=N_SURFACE, cam_pos=cam_pos, cam_quat=cam_quat,
        w=w, h=h, fl=fl, ha=ha, ctx=ctx)
    if seg_err:
        from behavior_interface.errors import GraspObjPlanningError
        raise GraspObjPlanningError(seg_err)
    if len(surf_pts) < 3:
        from behavior_interface.errors import GraspObjPlanningError
        raise GraspObjPlanningError(
            f"有效表面点不足 ({len(surf_pts)}<3)，请 move_to_object 后重新 capture")

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("grasp_obj_filter step3 球交体积")

    env_tree, env_used, env_npts = build_env_collision_tree(world, obj, voxel_m)
    inter_cm3 = sphere_intersection_volumes(
        surf_pts, tm, env_tree, voxel_m=voxel_m, radius_m=radius_m)
    top_idx = np.argsort(inter_cm3)[:TOP_K_POINTS]
    top_pts = surf_pts[top_idx]
    if ctx:
        ctx.log(
            f"  [grasp_obj_filter] 球交体积 top{TOP_K_POINTS}="
            f"{[round(float(inter_cm3[i]), 1) for i in top_idx]} "
            f"env_pts={env_npts}")

    poses = generate_icosa_roll_poses(top_pts, n_roll=N_ROLL)
    apply_camera_face_to_filter_poses(poses, world, ctx=ctx)
    if ctx:
        ctx.log(f"  [grasp_obj_filter] pose={len(poses)} ({TOP_K_POINTS}×{N_ICOSA_FACES}×{N_ROLL})")

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("grasp_obj_filter dual IK")

    gv_eef = gripper_solid_voxels_eef(voxel_m)
    pre, ik_meta = step6_prefilter_dual_ik_v7_metrics_gpu(
        poses, tm, world, object_name, gv_eef, env_tree,
        voxel_mm=VOXEL_MM,
        prefilter_n=PREFILTER_N,
        env_collision_max_cm3=ENV_COLLISION_MAX_CM3,
        ctx=ctx, gpu_device_id=gpu_device_id,
        plan_arm=plan_arm)
    if not pre:
        from behavior_interface.errors import GraspObjPlanningError
        top20_s = _format_ik_top20_for_error(ik_meta)
        raise GraspObjPlanningError(
            "grasp_obj_filter 没有可排序 IK pose "
            f"(ranked={ik_meta.get('n_ranked', 0)}, left={ik_meta.get('n_left_ok', 0)}, "
            f"right={ik_meta.get('n_right_ok', 0)}, both={ik_meta.get('n_both_ok', 0)}, "
            f"plan_arm={ik_meta.get('plan_arm', plan_arm)}, selection={ik_meta.get('selection')}, "
            f"elapsed={ik_meta.get('elapsed_s', 0.0):.2f}s)"
            + (f" top20={top20_s}" if top20_s else "")
        )

    if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
        ctx.raise_if_cancelled("grasp_obj_filter inflate overlap")
    apply_overlap_inflate_to_poses(
        pre, tm, obj_c, voxel_mm=VOXEL_MM, inflate_mm=OVERLAP_INFLATE_MM,
        ctx=ctx, use_gpu=True, gpu_device_id=gpu_device_id)
    best_pm, cand = select_best_pose_low_overlap(
        pre,
        overlap_vol_max_cm3=OVERLAP_VOL_MAX_CM3,
        ctx=ctx,
    )
    if best_pm is None:
        from behavior_interface.errors import GraspObjPlanningError
        raise GraspObjPlanningError(
            f"grasp_obj_filter IK过滤后无 overlap<{OVERLAP_VOL_MAX_CM3}cm³ 的抓取候选 "
            f"(IK both={ik_meta.get('n_both_ok', 0)}, plan_arm={ik_meta.get('plan_arm', plan_arm)})")

    selected_pose_ik = _pose_ik_hard_constraint_summary(best_pm)
    selected_pose_ik_q = _selected_pose_ik_q_map(selected_pose_ik)
    best = pose_to_plan_candidate(best_pm, shoulder)
    best_arm = ik_meta.get("recommended_arm") or str(arm)
    best["arm"] = best_arm
    best["recommended_arm"] = best_arm
    best["reachable"] = True
    best["reach_reason"] = "v15_dual_arm_gpu_ik_rank_overlap_best"
    best["ik_filter"] = best_pm.get("ik_filter")
    best["ik_allerr"] = best_pm.get("ik_allerr")
    best["selected_pose_ik"] = selected_pose_ik
    best["selected_pose_ik_q"] = selected_pose_ik_q
    best["ik_solution"] = selected_pose_ik.get("solution")
    best["camera_face"] = best_pm.get("camera_face")
    pick_audit = {
        "pick_method": "v15_dual_arm_gpu_ik_rank_overlap_best",
        "n_overlap_candidates": len(cand),
        "ik_filter": ik_meta,
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": selected_pose_ik_q,
    }

    plan_audit: Dict[str, Any] = {
        "grasp_obj_build": GRASP_OBJ_FILTER_PIPELINE_BUILD,
        "pipeline": "v15_dual_arm_gpu_ik_rank_gpu_inflate_overlap",
        "overlap_inflate_mm": float(OVERLAP_INFLATE_MM),
        "overlap_vol_max_cm3": float(OVERLAP_VOL_MAX_CM3),
        "env_collision_max_cm3": float(ENV_COLLISION_MAX_CM3),
        "use_gpu_step6": True,
        "dual_arm_ik_filter": True,
        "plan_arm": plan_arm,
        "recommended_arm": best_arm,
        "ik_filter": ik_meta,
        "selected_pose_ik": selected_pose_ik,
        "selected_pose_ik_q": selected_pose_ik_q,
        "object_name": object_name,
        "seed": int(seed),
        "n_surface": int(len(surf_pts)),
        "n_poses": int(len(poses)),
        "n_v7_prefilter": int(len(pre)),
        "n_overlap_candidates": int(len(cand)),
        "ball_diameter_mm": float(BALL_DIAMETER_MM),
        "voxel_mm": float(VOXEL_MM),
        "top10_inter_cm3": [float(inter_cm3[i]) for i in top_idx],
        "env_collision_objects": [u["name"] for u in env_used],
        "pick": pick_audit,
        "best_pi": best_pm.get("pi"),
        "best_ni": best_pm.get("ni"),
        "best_ri": best_pm.get("ri"),
    }
    if ctx:
        ctx.log(
            f"  [grasp_obj_filter] 最终 grasp_vol={best['grasp_vol_cm3']:.2f} "
            f"overlap_vol={best['overlap_vol_cm3']:.3f} "
            f"pi={best_pm.get('pi')} ni={best_pm.get('ni')} ri={best_pm.get('ri')} "
            f"IK both={ik_meta.get('n_both_ok', 0)} topGPU={ik_meta.get('n_gpu_v7', 0)} "
            f"plan_arm={ik_meta.get('plan_arm')} selection={ik_meta.get('selection')} "
            f"selected_pose_ik={selected_pose_ik.get('solution')} "
            f"L={selected_pose_ik.get('left_pos_mm')}mm/{selected_pose_ik.get('left_ori_deg')}deg "
            f"R={selected_pose_ik.get('right_pos_mm')}mm/{selected_pose_ik.get('right_ori_deg')}deg "
            f"default_exec_arm={best_arm}"
        )
    return best, plan_audit
