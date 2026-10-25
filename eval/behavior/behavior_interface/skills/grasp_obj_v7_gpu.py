"""grasp_obj v7 指标 GPU 批量复算（step6 瓶颈优化）。

CPU 口径：plan_grasp_opening_volume.compute_grasp_obj_v7_metrics（trimesh closest_point）。
GPU 口径：高密度表面采样 + 分块 cdist，与 CPU 在 3mm 壳层下数值对齐（bench 验证）。
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

V7_GPU_BUILD = "v3_surface_cdist_batch"
_DEFAULT_SURFACE_SAMPLES = 80000
_GRIP_ALIGN_COS = 0.5
_BATCH_POSE_CHUNK = 32
INFLATE_GPU_BUILD = "v3_inflate_shell_cdist_batch"
# 活跃中的 GPU 壳层缓存（bench/plan 后应 release_v7_gpu_memory）
_ACTIVE_SHELL_CACHES: List["V7ShellQueryCache"] = []


def release_v7_gpu_memory() -> None:
    """释放本 interface 所属卡上的 V7/inflate 缓存。

    Do not iterate over ``torch.cuda.device_count()`` here.  In an unmasked
    process that would flush allocator caches belonging to other interfaces
    (and can trigger synchronization on every card).  Fixed interface
    launchers expose one physical owner through ``CUDA_VISIBLE_DEVICES``;
    resolve that mapping to the process-local ordinal and touch only it.
    """
    import torch

    for cache in list(_ACTIVE_SHELL_CACHES):
        try:
            cache.dispose()
        except Exception:
            pass
    _ACTIVE_SHELL_CACHES.clear()
    if not torch.cuda.is_available():
        return

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
        owned = int(physical)
        if 0 <= owned < int(torch.cuda.device_count()):
            local = owned
    if local is None:
        # Legacy/unmasked callers have no explicit ownership metadata; use the
        # current CUDA context, never a process-wide device sweep.
        try:
            local = int(torch.cuda.current_device())
        except Exception:
            local = 0
    try:
        with torch.cuda.device(local):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def pick_v7_gpu_device(prefer_id: Optional[int] = None):
    """Return the interface-owned CUDA device, never a global free-memory pick."""
    import torch

    if not torch.cuda.is_available():
        return torch.device("cpu")
    n = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    physical = os.environ.get("BEHAVIOR_INTERFACE_PHYSICAL_GPU", "").strip()
    # Ownership metadata is authoritative.  In particular, a caller may pass
    # a physical ``prefer_id`` from a CLI while the process is masked to one
    # card; treating that number as a local ordinal would select another card
    # when an unmasked/multi-card environment happens to be inherited.
    entries = [item.strip() for item in visible.split(",") if item.strip()]
    if physical.isdigit():
        owner = int(physical)
        if entries:
            if physical in entries:
                return torch.device(f"cuda:{entries.index(physical)}")
            # A declared owner that is absent from the mask is a contradictory
            # environment.  Fail closed instead of guessing from prefer_id.
            raise RuntimeError(
                "BEHAVIOR_INTERFACE_PHYSICAL_GPU is absent from "
                f"CUDA_VISIBLE_DEVICES: owner={physical!r}, visible={visible!r}"
            )
        if 0 <= owner < n:
            return torch.device(f"cuda:{owner}")
        raise RuntimeError(
            f"BEHAVIOR_INTERFACE_PHYSICAL_GPU={owner} is outside CUDA device count {n}"
        )
    if len(entries) == 1 and entries[0].isdigit():
        # CUDA remaps the one visible physical card to local ordinal zero.
        return torch.device("cuda:0")
    if prefer_id is not None:
        pid = int(prefer_id)
        if 0 <= pid < n:
            return torch.device(f"cuda:{pid}")
    # Keep legacy callers deterministic when no ownership metadata exists;
    # this branch is intentionally not based on global free memory.
    if n > 0:
        return torch.device("cuda:0")
    return torch.device("cpu")


def _quat_to_mat_batch(quats: np.ndarray) -> np.ndarray:
    q = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


class V7ShellQueryCache:
    """物体薄壳查询缓存（CPU mesh + GPU 表面点）。"""

    def __init__(
        self,
        object_tm,
        *,
        voxel_mm: float = 3.0,
        device=None,
        n_surface_samples: int = _DEFAULT_SURFACE_SAMPLES,
    ):
        import torch
        import trimesh

        self.object_tm = object_tm
        self.voxel_mm = float(voxel_mm)
        self.voxel_m = self.voxel_mm / 1000.0
        self.shell_r = self.voxel_m * 0.5 * math.sqrt(3.0)
        self.vox_cm3 = self.voxel_m ** 3 * 1e6
        self.watertight = (
            bool(getattr(object_tm, "is_watertight", False))
            and float(getattr(object_tm, "volume", 0.0)) > 1e-10
        )
        self.device = device or pick_v7_gpu_device()
        n = max(5000, int(n_surface_samples))
        pts, face_idx = trimesh.sample.sample_surface(object_tm, n)
        pts = np.asarray(pts, dtype=np.float32)
        fn = np.asarray(object_tm.face_normals[np.asarray(face_idx, dtype=np.int64)], dtype=np.float32)
        self.n_surface_samples = int(len(pts))
        self.surf_pts = torch.from_numpy(pts).to(self.device)
        self.surf_fn = torch.from_numpy(fn).to(self.device)
        _ACTIVE_SHELL_CACHES.append(self)

    def clear_gpu(self) -> None:
        self.dispose()

    def dispose(self) -> None:
        """删除 GPU 张量并清空该 device 缓存。"""
        import torch

        try:
            if self in _ACTIVE_SHELL_CACHES:
                _ACTIVE_SHELL_CACHES.remove(self)
        except ValueError:
            pass
        for attr in ("surf_pts", "surf_fn"):
            if hasattr(self, attr):
                delattr(self, attr)
        if str(self.device).startswith("cuda"):
            try:
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                torch.cuda.empty_cache()


def _batch_min_dist_nn(
    pts: "torch.Tensor",
    surf_pts: "torch.Tensor",
    *,
    chunk_q: int = 2048,
    chunk_s: int = 16384,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """pts (B,N,3) → min_dist (B,N), argmin (B,N)。"""
    import torch

    B, N, _ = pts.shape
    S = surf_pts.shape[0]
    min_d = torch.full((B, N), float("inf"), device=pts.device, dtype=pts.dtype)
    min_i = torch.zeros((B, N), dtype=torch.long, device=pts.device)
    flat = pts.reshape(B * N, 3)
    flat_d = min_d.reshape(B * N)
    flat_i = min_i.reshape(B * N)
    for q0 in range(0, B * N, chunk_q):
        q1 = min(B * N, q0 + chunk_q)
        q_chunk = flat[q0:q1]
        best_d = torch.full((q1 - q0,), float("inf"), device=pts.device, dtype=pts.dtype)
        best_i = torch.zeros(q1 - q0, dtype=torch.long, device=pts.device)
        for s0 in range(0, S, chunk_s):
            s1 = min(S, s0 + chunk_s)
            s_chunk = surf_pts[s0:s1]
            d = torch.cdist(q_chunk, s_chunk)
            d_loc, i_loc = d.min(dim=1)
            upd = d_loc < best_d
            best_d = torch.where(upd, d_loc, best_d)
            best_i = torch.where(upd, i_loc + s0, best_i)
        flat_d[q0:q1] = best_d
        flat_i[q0:q1] = best_i
    return flat_d.reshape(B, N), flat_i.reshape(B, N)


def _shell_occ_gpu(
    pts_world: "torch.Tensor",
    cache: V7ShellQueryCache,
    open_dir: Optional["torch.Tensor"] = None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """返回 (occ, near) bool (B,N)。"""
    import torch

    min_d, min_i = _batch_min_dist_nn(pts_world, cache.surf_pts)
    near = min_d <= float(cache.shell_r)
    if open_dir is None:
        return near, near
    fn = cache.surf_fn[min_i]
    od = open_dir.unsqueeze(1).expand(-1, pts_world.shape[1], -1)
    cosang = (fn * od).sum(dim=-1).abs()
    occ = near & (cosang >= _GRIP_ALIGN_COS)
    return occ, near


def batch_compute_v7_metrics_gpu(
    poses: Sequence[Dict[str, Any]],
    object_tm,
    *,
    voxel_mm: float = 3.0,
    cache: Optional[V7ShellQueryCache] = None,
    device=None,
    gap_centers: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """GPU 批量 v7：grasp_vol + overlap_vol（与 CPU v7 同体素/壳层口径）。"""
    import torch
    from behavior_interface.skills.plan_grasp_opening_volume import (
        gripper_solid_voxels_eef, opening_voxel_centers_eef)

    t0 = time.perf_counter()
    if not poses:
        return [], {"method": "gpu_v7", "n_poses": 0, "elapsed_s": 0.0}

    voxel_m = float(voxel_mm) / 1000.0
    if cache is None:
        cache = V7ShellQueryCache(
            object_tm, voxel_mm=voxel_mm, device=device)
    dev = cache.device

    open_eef = opening_voxel_centers_eef(voxel_m)
    grip_eef = gripper_solid_voxels_eef(voxel_m)
    n_open = int(len(open_eef))
    n_grip = int(len(grip_eef))
    open_t = torch.as_tensor(open_eef, dtype=torch.float32, device=dev)
    grip_t = torch.as_tensor(grip_eef, dtype=torch.float32, device=dev)
    y_local = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=dev)

    out: List[Dict[str, Any]] = []
    timings: List[float] = []

    for start in range(0, len(poses), _BATCH_POSE_CHUNK):
        chunk = list(poses[start:start + _BATCH_POSE_CHUNK])
        tc0 = time.perf_counter()
        Rs = torch.as_tensor(
            np.stack([np.asarray(p["R"], dtype=np.float32) for p in chunk]),
            dtype=torch.float32, device=dev)
        eefs = torch.as_tensor(
            np.stack([np.asarray(p["eef_pos"], dtype=np.float32) for p in chunk]),
            dtype=torch.float32, device=dev)
        B = Rs.shape[0]

        open_w = torch.einsum("bij,nj->bni", Rs, open_t) + eefs.unsqueeze(1)
        grip_w = torch.einsum("bij,nj->bni", Rs, grip_t) + eefs.unsqueeze(1)
        open_dir = torch.einsum("bij,j->bi", Rs, y_local)

        occ_open, _ = _shell_occ_gpu(open_w, cache, open_dir)
        occ_grip, _ = _shell_occ_gpu(grip_w, cache, open_dir=None)

        n_grasp = occ_open.sum(dim=1).to(torch.float64)
        n_overlap = occ_grip.sum(dim=1).to(torch.float64)
        grasp_cm3 = (n_grasp * cache.vox_cm3).cpu().numpy()
        overlap_cm3 = (n_overlap * cache.vox_cm3).cpu().numpy()
        grip_cm3 = float(n_grip * cache.vox_cm3)
        timings.append(time.perf_counter() - tc0)

        for bi, pm in enumerate(chunk):
            og = float(grasp_cm3[bi])
            oo = float(overlap_cm3[bi])
            out.append({
                "grasp_vol_cm3": og,
                "overlap_vol_cm3": oo,
                "gripper_vol_cm3": grip_cm3,
                "overlap_frac": float(oo / max(grip_cm3, 1e-12)) if grip_cm3 > 0 else 0.0,
                "open_intersect_vol_cm3": og,
                "open_intersect_vox": int(round(og / max(cache.vox_cm3, 1e-12))),
                "overlap_vox": int(round(oo / max(cache.vox_cm3, 1e-12))),
                "open_vol_method": "gpu_v7_surface",
                "overlap_voxel_mm": voxel_mm,
                "volume_v7_build": V7_GPU_BUILD,
                "pi": pm.get("pi"),
                "ni": pm.get("ni"),
                "ri": pm.get("ri"),
            })

    # anchor_dist：仅 B 个 gap 点，CPU closest_point 足够快
    if gap_centers is not None and object_tm is not None:
        import trimesh

        gaps = np.asarray(gap_centers, dtype=np.float64).reshape(-1, 3)
        _cp, d_a, _ = trimesh.proximity.closest_point(object_tm, gaps)
        for i, vm in enumerate(out):
            if i < len(d_a):
                vm["anchor_dist_mm"] = float(d_a[i] * 1000.0)
                vm["marker_pt"] = np.asarray(_cp[i], dtype=np.float64).tolist()

    meta = {
        "method": "gpu_v7",
        "build": V7_GPU_BUILD,
        "device": str(dev),
        "n_poses": len(poses),
        "n_open_vox": n_open,
        "n_grip_vox": n_grip,
        "n_surface_samples": cache.n_surface_samples,
        "elapsed_s": time.perf_counter() - t0,
        "chunk_elapsed_s": timings,
    }
    return out, meta


def batch_compute_v7_metrics_cpu_batched(
    poses: Sequence[Dict[str, Any]],
    object_tm,
    *,
    voxel_mm: float = 3.0,
    gap_centers: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """CPU 批量精确版：单次 closest_point 覆盖多 pose（与逐 pose CPU 数值一致）。"""
    import trimesh
    from behavior_interface.skills.plan_grasp_opening_volume import (
        _GRIP_ALIGN_COS as _ALIGN_COS,
        _points_inside_object,
        gripper_solid_voxels_eef, opening_voxel_centers_eef)

    t0 = time.perf_counter()
    if not poses or object_tm is None:
        return [], {"method": "cpu_v7_batched", "n_poses": 0, "elapsed_s": 0.0}

    voxel_m = float(voxel_mm) / 1000.0
    vox_cm3 = voxel_m ** 3 * 1e6
    open_eef = opening_voxel_centers_eef(voxel_m)
    grip_eef = gripper_solid_voxels_eef(voxel_m)
    n_open = int(len(open_eef))
    n_grip = int(len(grip_eef))
    out: List[Dict[str, Any]] = []

    for start in range(0, len(poses), _BATCH_POSE_CHUNK):
        chunk = list(poses[start:start + _BATCH_POSE_CHUNK])
        open_blocks: List[np.ndarray] = []
        grip_blocks: List[np.ndarray] = []
        open_dirs: List[np.ndarray] = []
        for pm in chunk:
            R = np.asarray(pm["R"], dtype=np.float64)
            eef = np.asarray(pm["eef_pos"], dtype=np.float64).reshape(3)
            open_blocks.append((R @ open_eef.T).T + eef)
            grip_blocks.append((R @ grip_eef.T).T + eef)
            open_dirs.append(R @ np.array([0.0, 1.0, 0.0]))

        open_stack = np.vstack(open_blocks)
        grip_stack = np.vstack(grip_blocks)
        open_dir_rep = np.repeat(
            np.stack(open_dirs, axis=0), n_open, axis=0)
        shell_r = voxel_m * 0.5 * math.sqrt(3.0)
        _cp, dist, tri_id = trimesh.proximity.closest_point(object_tm, open_stack)
        near = np.asarray(dist <= shell_r, dtype=bool)
        fn = np.asarray(object_tm.face_normals)[tri_id]
        od = open_dir_rep / (np.linalg.norm(open_dir_rep, axis=1, keepdims=True) + 1e-12)
        cosang = np.abs(np.sum(fn * od, axis=1))
        occ_open = (near & (cosang >= _ALIGN_COS)).reshape(len(chunk), n_open)

        inside_grip = _points_inside_object(
            grip_stack, object_tm, None, voxel_mm=voxel_mm)
        inside_grip = inside_grip.reshape(len(chunk), n_grip)

        for bi, pm in enumerate(chunk):
            ng = int(occ_open[bi].sum())
            no = int(inside_grip[bi].sum())
            og = float(ng * vox_cm3)
            oo = float(no * vox_cm3)
            gg = float(n_grip * vox_cm3)
            out.append({
                "grasp_vol_cm3": og,
                "overlap_vol_cm3": oo,
                "gripper_vol_cm3": gg,
                "overlap_frac": float(oo / max(gg, 1e-12)),
                "open_intersect_vol_cm3": og,
                "open_vol_method": "cpu_v7_batched",
                "volume_v7_build": "cpu_batched_exact",
                "pi": pm.get("pi"),
                "ni": pm.get("ni"),
                "ri": pm.get("ri"),
            })

    if gap_centers is not None:
        gaps = np.asarray(gap_centers, dtype=np.float64).reshape(-1, 3)
        _cp, d_a, _ = trimesh.proximity.closest_point(object_tm, gaps)
        for i, vm in enumerate(out):
            if i < len(d_a):
                vm["anchor_dist_mm"] = float(d_a[i] * 1000.0)
                vm["marker_pt"] = np.asarray(_cp[i], dtype=np.float64).tolist()

    meta = {
        "method": "cpu_v7_batched",
        "n_poses": len(poses),
        "elapsed_s": time.perf_counter() - t0,
    }
    return out, meta


def compute_v7_metrics_cpu_single(
    world,
    object_name: str,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    gap_center: Optional[np.ndarray],
    *,
    voxel_mm: float = 3.0,
) -> Dict[str, Any]:
    """逐 pose CPU（现有生产口径）。"""
    from behavior_interface.skills.plan_grasp_opening_volume import (
        compute_grasp_obj_v7_metrics)

    return compute_grasp_obj_v7_metrics(
        world, object_name,
        np.asarray(eef_pos, dtype=np.float64),
        np.asarray(eef_quat, dtype=np.float64),
        None if gap_center is None else np.asarray(gap_center, dtype=np.float64),
        voxel_mm=float(voxel_mm),
        clear_cache=False,
    )


def bench_v7_cpu_vs_gpu(
    world,
    object_name: str,
    poses: Sequence[Dict[str, Any]],
    *,
    voxel_mm: float = 3.0,
    gpu_device_id: Optional[int] = None,
) -> Dict[str, Any]:
    """单步/多 pose bench：逐 pose CPU vs CPU-batch vs GPU-batch。"""
    from behavior_interface.skills.plan_grasp_opening_volume import (
        load_object_trimesh_world)

    tm = load_object_trimesh_world(world, object_name)
    if tm is None:
        return {"ok": False, "error": f"无法加载 mesh: {object_name}"}

    gaps = np.stack([
        np.asarray(p.get("anchor", p.get("eef_pos")), dtype=np.float64).reshape(3)
        for p in poses
    ], axis=0)

    # 逐 pose CPU（仅前 3 个，避免太慢）
    cpu_single: List[Dict[str, Any]] = []
    t_single0 = time.perf_counter()
    n_single = min(3, len(poses))
    for pm in poses[:n_single]:
        vm = compute_v7_metrics_cpu_single(
            world, object_name,
            pm["eef_pos"], pm.get("quat", pm.get("eef_quat")),
            pm.get("anchor"), voxel_mm=voxel_mm)
        cpu_single.append({
            "grasp_vol_cm3": float(vm.get("grasp_vol_cm3", 0.0)),
            "overlap_vol_cm3": float(vm.get("overlap_vol_cm3", 0.0)),
            "anchor_dist_mm": float(vm.get("anchor_dist_mm", 0.0)),
        })
    t_single = time.perf_counter() - t_single0

    cpu_batch, meta_cpu = batch_compute_v7_metrics_cpu_batched(
        poses, tm, voxel_mm=voxel_mm, gap_centers=gaps)
    dev = pick_v7_gpu_device(gpu_device_id)
    cache = V7ShellQueryCache(tm, voxel_mm=voxel_mm, device=dev)
    try:
        gpu_batch, meta_gpu = batch_compute_v7_metrics_gpu(
            poses, tm, voxel_mm=voxel_mm, cache=cache, gap_centers=gaps)
    finally:
        cache.dispose()
        release_v7_gpu_memory()

    def _diff(a: Dict, b: Dict) -> Dict[str, float]:
        return {
            "grasp_vol_cm3": abs(float(a.get("grasp_vol_cm3", 0)) - float(b.get("grasp_vol_cm3", 0))),
            "overlap_vol_cm3": abs(float(a.get("overlap_vol_cm3", 0)) - float(b.get("overlap_vol_cm3", 0))),
            "anchor_dist_mm": abs(float(a.get("anchor_dist_mm", 0)) - float(b.get("anchor_dist_mm", 0))),
        }

    diffs_cpu_gpu = [_diff(c, g) for c, g in zip(cpu_batch, gpu_batch)]
    diffs_single_batch = [_diff(s, cpu_batch[i]) for i, s in enumerate(cpu_single)]

    return {
        "ok": True,
        "object_name": object_name,
        "n_poses": len(poses),
        "voxel_mm": float(voxel_mm),
        "timing_s": {
            "cpu_single_first3": t_single,
            "cpu_single_per_pose_est": t_single / max(n_single, 1),
            "cpu_batched_all": meta_cpu["elapsed_s"],
            "gpu_batched_all": meta_gpu["elapsed_s"],
            "speedup_gpu_vs_cpu_single_est": (t_single / max(n_single, 1) * len(poses))
            / max(meta_gpu["elapsed_s"], 1e-6),
            "speedup_gpu_vs_cpu_batched": meta_cpu["elapsed_s"] / max(meta_gpu["elapsed_s"], 1e-6),
        },
        "gpu_meta": meta_gpu,
        "cpu_batch_meta": meta_cpu,
        "max_abs_diff_cpu_batch_vs_gpu": {
            k: max(d[k] for d in diffs_cpu_gpu) for k in diffs_cpu_gpu[0]
        } if diffs_cpu_gpu else {},
        "max_abs_diff_cpu_single_vs_batch": {
            k: max(d[k] for d in diffs_single_batch) for k in diffs_single_batch[0]
        } if diffs_single_batch else {},
        "sample_pose0": {
            "cpu_single": cpu_single[0] if cpu_single else None,
            "cpu_batch": cpu_batch[0] if cpu_batch else None,
            "gpu_batch": gpu_batch[0] if gpu_batch else None,
            "diff_cpu_batch_gpu": diffs_cpu_gpu[0] if diffs_cpu_gpu else None,
        },
    }


def batch_compute_inflate_overlap_gpu(
    poses: Sequence[Dict[str, Any]],
    object_tm,
    *,
    voxel_mm: float = 3.0,
    inflate_mm: float = 3.0,
    cache: Optional[V7ShellQueryCache] = None,
    device=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """GPU 批量：膨胀夹爪体素 ∩ 物体薄壳（与 CPU patch_pose_overlap_inflated 同壳层口径）。"""
    import torch
    from behavior_interface.skills.grasp_obj_pipeline_core import (
        gripper_solid_voxels_eef_inflated)

    t0 = time.perf_counter()
    if not poses or object_tm is None:
        return [], {"method": "gpu_inflate", "n_poses": 0, "elapsed_s": 0.0}

    voxel_m = float(voxel_mm) / 1000.0
    inflate_m = float(inflate_mm) / 1000.0
    if cache is None:
        cache = V7ShellQueryCache(object_tm, voxel_mm=voxel_mm, device=device)
    dev = cache.device
    gv_inf = gripper_solid_voxels_eef_inflated(voxel_m, inflate_m=inflate_m)
    n_grip = int(len(gv_inf))
    grip_cm3 = float(n_grip * cache.vox_cm3)
    if n_grip == 0:
        empty = [{
            "overlap_vol_cm3": 0.0,
            "overlap_vox": 0,
            "overlap_frac": 0.0,
            "gripper_vol_cm3": 0.0,
            "overlap_method": f"inflate_gripper_normal_{int(round(inflate_mm))}mm_gpu",
        } for _ in poses]
        return empty, {
            "method": "gpu_inflate", "n_poses": len(poses), "elapsed_s": 0.0,
            "n_grip_vox": 0, "build": INFLATE_GPU_BUILD,
        }

    grip_t = torch.as_tensor(gv_inf, dtype=torch.float32, device=dev)
    out: List[Dict[str, Any]] = []
    timings: List[float] = []
    tag = f"inflate_gripper_normal_{int(round(inflate_mm))}mm_gpu"

    for start in range(0, len(poses), _BATCH_POSE_CHUNK):
        chunk = list(poses[start:start + _BATCH_POSE_CHUNK])
        tc0 = time.perf_counter()
        Rs = torch.as_tensor(
            np.stack([np.asarray(p["R"], dtype=np.float32) for p in chunk]),
            dtype=torch.float32, device=dev)
        eefs = torch.as_tensor(
            np.stack([np.asarray(p["eef_pos"], dtype=np.float32) for p in chunk]),
            dtype=torch.float32, device=dev)
        grip_w = torch.einsum("bij,nj->bni", Rs, grip_t) + eefs.unsqueeze(1)
        near, _ = _shell_occ_gpu(grip_w, cache, open_dir=None)
        n_overlap = near.sum(dim=1).to(torch.float64)
        overlap_cm3 = (n_overlap * cache.vox_cm3).cpu().numpy()
        timings.append(time.perf_counter() - tc0)
        for bi in range(len(chunk)):
            oo = float(overlap_cm3[bi])
            nv = int(round(oo / max(cache.vox_cm3, 1e-12)))
            out.append({
                "overlap_vol_cm3": oo,
                "overlap_vox": nv,
                "overlap_frac": float(oo / max(grip_cm3, 1e-12)),
                "gripper_vol_cm3": grip_cm3,
                "overlap_method": tag,
            })

    meta = {
        "method": "gpu_inflate",
        "build": INFLATE_GPU_BUILD,
        "device": str(dev),
        "n_poses": len(poses),
        "n_grip_vox": n_grip,
        "inflate_mm": float(inflate_mm),
        "n_surface_samples": cache.n_surface_samples,
        "elapsed_s": time.perf_counter() - t0,
        "chunk_elapsed_s": timings,
    }
    return out, meta


def compute_inflate_overlap_cpu_single(
    pm: Dict[str, Any],
    object_tm,
    obj_centroid: np.ndarray,
    gv_eef_inf: np.ndarray,
    *,
    voxel_mm: float = 3.0,
) -> Dict[str, Any]:
    """单 pose CPU：patch_pose_overlap_inflated 口径。"""
    from behavior_interface.skills.grasp_obj_pipeline_core import (
        patch_pose_overlap_inflated)

    dup = dict(pm)
    dup["R"] = np.asarray(pm["R"], dtype=np.float64).copy()
    dup["eef_pos"] = np.asarray(pm["eef_pos"], dtype=np.float64).copy()
    patch_pose_overlap_inflated(
        dup, gv_eef_inf, object_tm, obj_centroid, float(voxel_mm))
    return {
        "overlap_vol_cm3": float(dup.get("overlap_vol_cm3", 0.0)),
        "overlap_vox": int(dup.get("overlap_vox", 0)),
        "overlap_frac": float(dup.get("overlap_frac", 0.0)),
        "gripper_vol_cm3": float(dup.get("gripper_vol_cm3", 0.0)),
        "overlap_method": dup.get("overlap_method", "cpu_inflate"),
    }


def batch_compute_inflate_overlap_cpu(
    poses: Sequence[Dict[str, Any]],
    object_tm,
    obj_centroid: np.ndarray,
    *,
    voxel_mm: float = 3.0,
    inflate_mm: float = 3.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """多 pose CPU 顺序复算（生产旧路径）。"""
    from behavior_interface.skills.grasp_obj_pipeline_core import (
        gripper_solid_voxels_eef_inflated)

    t0 = time.perf_counter()
    voxel_m = float(voxel_mm) / 1000.0
    gv_inf = gripper_solid_voxels_eef_inflated(voxel_m, inflate_m=float(inflate_mm) / 1000.0)
    out = [
        compute_inflate_overlap_cpu_single(
            pm, object_tm, obj_centroid, gv_inf, voxel_mm=float(voxel_mm))
        for pm in poses
    ]
    return out, {
        "method": "cpu_inflate_sequential",
        "n_poses": len(poses),
        "n_grip_vox": int(len(gv_inf)),
        "elapsed_s": time.perf_counter() - t0,
    }


def bench_inflate_overlap_cpu_vs_gpu(
    object_tm,
    obj_centroid: np.ndarray,
    poses: Sequence[Dict[str, Any]],
    *,
    voxel_mm: float = 3.0,
    inflate_mm: float = 3.0,
    gpu_device_id: Optional[int] = None,
    cpu_full_batch: bool = False,
    accuracy_n: int = 3,
) -> Dict[str, Any]:
    """inflate overlap：单 pose CPU /（可选）全量 CPU / GPU 批量 耗时与精度。

    默认不做全量 CPU 顺序（60 pose 可达数百秒且会阻塞仿真主线程）。
    """
    if object_tm is None or not poses:
        return {"ok": False, "error": "mesh 或 poses 为空"}

    from behavior_interface.skills.grasp_obj_pipeline_core import (
        gripper_solid_voxels_eef_inflated)

    voxel_m = float(voxel_mm) / 1000.0
    gv_inf = gripper_solid_voxels_eef_inflated(
        voxel_m, inflate_m=float(inflate_mm) / 1000.0)

    n_single = min(3, len(poses))
    t0 = time.perf_counter()
    cpu_single = [
        compute_inflate_overlap_cpu_single(
            poses[i], object_tm, obj_centroid, gv_inf, voxel_mm=float(voxel_mm))
        for i in range(n_single)
    ]
    t_single = time.perf_counter() - t0
    per_pose = t_single / max(n_single, 1)

    n_acc = min(int(accuracy_n), len(poses))
    if cpu_full_batch:
        cpu_batch, meta_cpu = batch_compute_inflate_overlap_cpu(
            poses, object_tm, obj_centroid,
            voxel_mm=float(voxel_mm), inflate_mm=float(inflate_mm))
        cpu_batched_all = float(meta_cpu["elapsed_s"])
    else:
        cpu_batch, meta_cpu = batch_compute_inflate_overlap_cpu(
            poses[:n_acc], object_tm, obj_centroid,
            voxel_mm=float(voxel_mm), inflate_mm=float(inflate_mm))
        cpu_batched_all = per_pose * len(poses)
        meta_cpu = {
            **meta_cpu,
            "note": f"accuracy_subset_n={n_acc}",
            "cpu_full_batch_est_s": cpu_batched_all,
        }

    dev = pick_v7_gpu_device(gpu_device_id)
    cache = V7ShellQueryCache(object_tm, voxel_mm=float(voxel_mm), device=dev)
    try:
        gpu_batch, meta_gpu = batch_compute_inflate_overlap_gpu(
            poses, object_tm, voxel_mm=float(voxel_mm), inflate_mm=float(inflate_mm),
            cache=cache)
    finally:
        cache.dispose()
        release_v7_gpu_memory()

    def _diff(a: Dict, b: Dict) -> Dict[str, float]:
        return {
            "overlap_vol_cm3": abs(
                float(a.get("overlap_vol_cm3", 0)) - float(b.get("overlap_vol_cm3", 0))),
            "overlap_vox": abs(
                int(a.get("overlap_vox", 0)) - int(b.get("overlap_vox", 0))),
            "overlap_frac": abs(
                float(a.get("overlap_frac", 0)) - float(b.get("overlap_frac", 0))),
        }

    n_cmp = min(len(cpu_batch), len(gpu_batch))
    diffs = [_diff(cpu_batch[i], gpu_batch[i]) for i in range(n_cmp)]
    diffs_single = [_diff(s, cpu_batch[i]) for i, s in enumerate(cpu_single)]

    return {
        "ok": True,
        "n_poses": len(poses),
        "voxel_mm": float(voxel_mm),
        "inflate_mm": float(inflate_mm),
        "n_grip_vox_inflated": int(len(gv_inf)),
        "cpu_full_batch": bool(cpu_full_batch),
        "timing_s": {
            "cpu_single_first3": t_single,
            "cpu_single_per_pose_est": per_pose,
            "cpu_batched_all": cpu_batched_all,
            "cpu_batched_measured": float(meta_cpu.get("elapsed_s", 0.0)),
            "gpu_batched_all": meta_gpu["elapsed_s"],
            "speedup_gpu_vs_cpu_single_est": (per_pose * len(poses))
            / max(meta_gpu["elapsed_s"], 1e-6),
            "speedup_gpu_vs_cpu_batched": cpu_batched_all / max(meta_gpu["elapsed_s"], 1e-6),
        },
        "gpu_meta": meta_gpu,
        "cpu_batch_meta": meta_cpu,
        "max_abs_diff_cpu_batch_vs_gpu": {
            k: max(d[k] for d in diffs) for k in diffs[0]
        } if diffs else {},
        "max_abs_diff_cpu_single_vs_batch": {
            k: max(d[k] for d in diffs_single) for k in diffs_single[0]
        } if diffs_single else {},
        "sample_pose0": {
            "cpu_single": cpu_single[0] if cpu_single else None,
            "cpu_batch": cpu_batch[0] if cpu_batch else None,
            "gpu_batch": gpu_batch[0] if gpu_batch else None,
            "diff_cpu_batch_gpu": diffs[0] if diffs else None,
        },
    }
