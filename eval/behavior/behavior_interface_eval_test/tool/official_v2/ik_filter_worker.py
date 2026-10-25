"""Standalone cuRobo IK ranking worker for grasp object filtering.

Runs in a separate process / CUDA_VISIBLE_DEVICES so Isaac's rendering process
does not lose its GPU memory to cuRobo solver initialization.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch as th


def _configure_cpu_thread_pools() -> None:
    """Bound Torch host pools without changing CUDA solver semantics.

    The parent passes thread defaults only to this isolated worker. Honor an
    explicit worker override first, then the conventional OpenMP/BLAS choice,
    so operators can raise the cap for a measured workload without modifying
    the simulator process.
    """

    thread_count = None
    for variable in (
        "BEHAVIOR_IK_CPU_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TBB_NUM_THREADS",
    ):
        raw = os.environ.get(variable, "").strip()
        if raw.isdigit() and int(raw) > 0:
            thread_count = int(raw)
            break
    if thread_count is None:
        thread_count = 1
    # These calls must happen before cuRobo starts any parallel work. The
    # inter-op setter can reject a late call in an embedded/reused process;
    # retaining the guard keeps diagnostics from turning a request into a
    # spurious failure.
    try:
        th.set_num_threads(thread_count)
    except (RuntimeError, ValueError):
        pass
    try:
        th.set_num_interop_threads(thread_count)
    except (RuntimeError, ValueError):
        pass


R1PRO_GRIPPER_TO_EEF_FIXED_TRANSFORM = (
    0.0,
    0.0,
    -0.06000,
    0.0,
    0.0,
    1.0,
    0.0,
)


def _enable_faulthandler() -> None:
    try:
        import faulthandler

        faulthandler.enable()
    except Exception:
        pass


def _log_gpu_diag(
    log_fn,
    event: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
    include_nvidia: bool = False,
) -> None:
    del include_nvidia
    payload = {"event": str(event), **dict(extra or {})}
    try:
        log_fn(f"GPU_DIAG {json.dumps(payload, sort_keys=True)}")
    except Exception:
        pass


IK_POSE_SHM_VERSION = 2
IK_POSE_SHM_V1_COLUMNS = 31
IK_POSE_SHM_DTYPE = np.dtype("<f8")
IK_FIXED_CUDA_GRAPH_POLICY = "cuda_graph_fixed64"
IK_FIXED_BATCH_NO_GRAPH_POLICY = "fixed64_no_graph"
IK_FIXED_BATCH_REWARM_NO_GRAPH_POLICY = "fixed64_rewarm_no_graph"
IK_FIXED_CUDA_GRAPH_REWARM_POLICY = "cuda_graph_fixed64_rewarm"
IK_FIXED_CUDA_GRAPH_SPLIT16_REWARM_POLICY = (
    "cuda_graph_split16_rewarm"
)
IK_FIXED_CUDA_GRAPH_SPLIT8_REWARM_POLICY = (
    "cuda_graph_split8_rewarm"
)
IK_FIXED_CUDA_GRAPH_WARM32X6_REWARM_POLICY = (
    "cuda_graph_warm32x6_rewarm"
)
IK_FIXED_CUDA_GRAPH_WARM32X6_COLD24_REWARM_POLICY = (
    "cuda_graph_warm32x6_cold24_rewarm"
)


def _fixed_cuda_graph_enabled(req: Dict[str, Any]) -> bool:
    return bool(
        str(req.get("solver_config_policy") or "").strip()
        in {
            IK_FIXED_CUDA_GRAPH_POLICY,
            IK_FIXED_CUDA_GRAPH_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_SPLIT16_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_SPLIT8_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_WARM32X6_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_WARM32X6_COLD24_REWARM_POLICY,
        }
    )


def _fixed_batch_enabled(req: Dict[str, Any]) -> bool:
    return bool(
        str(req.get("solver_policy") or "").strip()
        in {
            IK_FIXED_CUDA_GRAPH_POLICY,
            IK_FIXED_BATCH_NO_GRAPH_POLICY,
            IK_FIXED_BATCH_REWARM_NO_GRAPH_POLICY,
            IK_FIXED_CUDA_GRAPH_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_SPLIT16_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_SPLIT8_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_WARM32X6_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_WARM32X6_COLD24_REWARM_POLICY,
        }
    )


def _logical_shape_rewarm_enabled(req: Dict[str, Any]) -> bool:
    return bool(
        str(req.get("solver_policy") or "").strip()
        in {
            IK_FIXED_BATCH_REWARM_NO_GRAPH_POLICY,
            IK_FIXED_CUDA_GRAPH_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_SPLIT16_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_SPLIT8_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_WARM32X6_REWARM_POLICY,
            IK_FIXED_CUDA_GRAPH_WARM32X6_COLD24_REWARM_POLICY,
        }
    )


def _fixed_physical_batch_size(
    req: Dict[str, Any],
) -> Optional[int]:
    if not _fixed_batch_enabled(req):
        return None
    policy = str(req.get("solver_policy") or "").strip()
    if policy == IK_FIXED_CUDA_GRAPH_SPLIT16_REWARM_POLICY:
        return min(16, int(max(1, req["batch_size"])))
    if policy == IK_FIXED_CUDA_GRAPH_SPLIT8_REWARM_POLICY:
        return min(8, int(max(1, req["batch_size"])))
    if policy in {
        IK_FIXED_CUDA_GRAPH_WARM32X6_REWARM_POLICY,
        IK_FIXED_CUDA_GRAPH_WARM32X6_COLD24_REWARM_POLICY,
    }:
        return min(32, int(max(1, req["batch_size"])))
    return int(max(1, req["batch_size"]))


def _poses_from_shared_memory(
    descriptor: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Decode fixed-width pose records from a POSIX shared-memory segment."""
    from multiprocessing import shared_memory

    version = int(descriptor.get("version", 0))
    if version not in (1, IK_POSE_SHM_VERSION):
        raise ValueError(
            f"unsupported IK pose shared-memory version {version}"
        )
    count = int(descriptor["count"])
    columns = int(descriptor["columns"])
    arm_dof = 7 if version == 1 else int(descriptor.get("arm_dof", 0))
    expected_columns = (
        IK_POSE_SHM_V1_COLUMNS
        if version == 1
        else 17 + 2 * arm_dof
    )
    if arm_dof <= 0 or columns != expected_columns:
        raise ValueError(
            "invalid IK pose shared-memory layout "
            f"version={version} arm_dof={arm_dof} "
            f"columns={columns} expected={expected_columns}"
        )
    dtype = np.dtype(str(descriptor["dtype"]))
    if dtype != IK_POSE_SHM_DTYPE:
        raise ValueError(f"invalid IK pose shared-memory dtype {dtype}")
    shm = shared_memory.SharedMemory(
        name=str(descriptor["name"]),
        create=False,
    )
    try:
        records = np.ndarray(
            (count, columns),
            dtype=dtype,
            buffer=shm.buf,
        )
        poses: List[Dict[str, Any]] = []
        for record in records:
            pose: Dict[str, Any] = {
                "eef_pos": record[0:3].tolist(),
                "quat": record[3:7].tolist(),
            }
            warm: Dict[str, List[float]] = {}
            left_start = 7
            right_start = left_start + arm_dof
            safe_start = right_start + arm_dof
            if np.all(np.isfinite(record[left_start:right_start])):
                warm["left"] = record[left_start:right_start].tolist()
            if np.all(np.isfinite(record[right_start:safe_start])):
                warm["right"] = record[right_start:safe_start].tolist()
            if warm:
                pose["warm_start_q_by_arm"] = warm
            if np.all(np.isfinite(record[safe_start:safe_start + 7])):
                pose["paired_safe"] = {
                    "eef_pos": record[safe_start:safe_start + 3].tolist(),
                    "quat": record[safe_start + 3:safe_start + 7].tolist(),
                    "pos_tol_m": float(record[safe_start + 7]),
                    "ori_tol_deg": float(record[safe_start + 8]),
                    "final_branch_gap_rad": float(record[safe_start + 9]),
                }
            poses.append(pose)
    finally:
        shm.close()
        if int(descriptor.get("owner_pid", -1)) != os.getpid():
            try:
                from multiprocessing import resource_tracker

                resource_tracker.unregister(
                    shm._name,
                    "shared_memory",
                )
            except Exception:
                pass
    return poses, {
        "transport": "shared_memory",
        "version": version,
        "arm_dof": arm_dof,
        "count": count,
        "bytes": int(count * columns * dtype.itemsize),
    }


def _request_poses(
    req: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    descriptor = req.get("pose_shared_memory")
    if isinstance(descriptor, dict):
        return _poses_from_shared_memory(descriptor)
    return list(req.get("poses") or []), {
        "transport": "json",
        "count": int(len(req.get("poses") or [])),
        "bytes": None,
    }


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, th.Tensor):
        return _json_clean(value.detach().cpu().tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _quat_xyzw_to_mat(q: th.Tensor) -> th.Tensor:
    q = q / th.clamp(th.linalg.norm(q, dim=-1, keepdim=True), min=1.0e-9)
    x, y, z, w = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    out = th.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    out[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[..., 0, 1] = 2.0 * (xy - wz)
    out[..., 0, 2] = 2.0 * (xz + wy)
    out[..., 1, 0] = 2.0 * (xy + wz)
    out[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[..., 1, 2] = 2.0 * (yz - wx)
    out[..., 2, 0] = 2.0 * (xz - wy)
    out[..., 2, 1] = 2.0 * (yz + wx)
    out[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return out


def _mat_to_quat_xyzw(m: th.Tensor) -> th.Tensor:
    # Robust enough for batched rigid transforms here.
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    q = th.empty(m.shape[:-2] + (4,), dtype=m.dtype, device=m.device)
    mask = tr > 0.0
    s = th.sqrt(th.clamp(tr[mask] + 1.0, min=1.0e-12)) * 2.0
    q[mask, 3] = 0.25 * s
    q[mask, 0] = (m[mask, 2, 1] - m[mask, 1, 2]) / s
    q[mask, 1] = (m[mask, 0, 2] - m[mask, 2, 0]) / s
    q[mask, 2] = (m[mask, 1, 0] - m[mask, 0, 1]) / s
    rem = ~mask
    if bool(rem.any().item()):
        mr = m[rem]
        out = th.empty((mr.shape[0], 4), dtype=m.dtype, device=m.device)
        cond0 = (mr[:, 0, 0] > mr[:, 1, 1]) & (mr[:, 0, 0] > mr[:, 2, 2])
        cond1 = ~cond0 & (mr[:, 1, 1] > mr[:, 2, 2])
        cond2 = ~(cond0 | cond1)
        if bool(cond0.any().item()):
            mm = mr[cond0]
            ss = th.sqrt(th.clamp(1.0 + mm[:, 0, 0] - mm[:, 1, 1] - mm[:, 2, 2], min=1.0e-12)) * 2.0
            out[cond0, 3] = (mm[:, 2, 1] - mm[:, 1, 2]) / ss
            out[cond0, 0] = 0.25 * ss
            out[cond0, 1] = (mm[:, 0, 1] + mm[:, 1, 0]) / ss
            out[cond0, 2] = (mm[:, 0, 2] + mm[:, 2, 0]) / ss
        if bool(cond1.any().item()):
            mm = mr[cond1]
            ss = th.sqrt(th.clamp(1.0 + mm[:, 1, 1] - mm[:, 0, 0] - mm[:, 2, 2], min=1.0e-12)) * 2.0
            out[cond1, 3] = (mm[:, 0, 2] - mm[:, 2, 0]) / ss
            out[cond1, 0] = (mm[:, 0, 1] + mm[:, 1, 0]) / ss
            out[cond1, 1] = 0.25 * ss
            out[cond1, 2] = (mm[:, 1, 2] + mm[:, 2, 1]) / ss
        if bool(cond2.any().item()):
            mm = mr[cond2]
            ss = th.sqrt(th.clamp(1.0 + mm[:, 2, 2] - mm[:, 0, 0] - mm[:, 1, 1], min=1.0e-12)) * 2.0
            out[cond2, 3] = (mm[:, 1, 0] - mm[:, 0, 1]) / ss
            out[cond2, 0] = (mm[:, 0, 2] + mm[:, 2, 0]) / ss
            out[cond2, 1] = (mm[:, 1, 2] + mm[:, 2, 1]) / ss
            out[cond2, 2] = 0.25 * ss
        q[rem] = out
    return q / th.clamp(th.linalg.norm(q, dim=-1, keepdim=True), min=1.0e-9)


def _pose_inv_mat(pos: th.Tensor, quat_xyzw: th.Tensor) -> th.Tensor:
    rot = _quat_xyzw_to_mat(quat_xyzw.reshape(1, 4))[0]
    rt = rot.transpose(0, 1)
    out = th.eye(4, dtype=pos.dtype, device=pos.device)
    out[:3, :3] = rt
    out[:3, 3] = -(rt @ pos.reshape(3))
    return out


def _world_to_curobo_pose(req: Dict[str, Any], poses: List[Dict[str, Any]], tensor_args):
    from curobo.types.math import Pose

    local_pos, local_quat_wxyz = _world_to_curobo_pose_cpu(req, poses)
    return Pose(
        position=tensor_args.to_device(local_pos),
        quaternion=tensor_args.to_device(local_quat_wxyz),
    )


def _world_to_curobo_pose_cpu(
    req: Dict[str, Any],
    poses: List[Dict[str, Any]],
) -> Tuple[th.Tensor, th.Tensor]:
    """Preserve the established CPU float32 transform before device upload."""
    pos = th.as_tensor([p["eef_pos"] for p in poses], dtype=th.float32)
    quat = th.as_tensor([p["quat"] for p in poses], dtype=th.float32)
    n = int(pos.shape[0])
    tf = th.eye(4, dtype=th.float32).reshape(1, 4, 4).repeat(n, 1, 1)
    tf[:, :3, :3] = _quat_xyzw_to_mat(quat)
    tf[:, :3, 3] = pos
    base = req["base_link_pose"]
    inv = _pose_inv_mat(
        th.as_tensor(base["pos"], dtype=th.float32),
        th.as_tensor(base["quat"], dtype=th.float32),
    )
    local = inv.reshape(1, 4, 4) @ tf
    local_pos = local[:, :3, 3].contiguous()
    local_quat_xyzw = _mat_to_quat_xyzw(local[:, :3, :3])
    local_quat_wxyz = local_quat_xyzw[:, [3, 0, 1, 2]].contiguous()
    return local_pos, local_quat_wxyz


class _IKRuntimeTensorCache:
    """Reuse request tensors without growing peak VRAM.

    Explicit Halton replay remains opt-in because passing generated seeds through
    ``seed_config`` is not equivalent to cuRobo's internal seed path after a
    solver has warmed up.
    """

    def __init__(
        self,
        req: Dict[str, Any],
        *,
        solver,
        tensor_args,
        q_names: List[str],
        seed_replay_enabled: Optional[bool] = None,
    ):
        self.batch_size = int(max(1, req["batch_size"]))
        self.num_seeds = int(req["num_seeds"])
        self.dof = int(len(q_names))
        self.device = tensor_args.device
        self.goal_position = th.empty(
            (self.batch_size, 3),
            dtype=th.float32,
            device=self.device,
        )
        self.goal_quaternion = th.empty(
            (self.batch_size, 4),
            dtype=th.float32,
            device=self.device,
        )
        base_q = [
            float(req["q_by_name"].get(name, 0.0))
            for name in q_names
        ]
        self.cold_retract = th.tensor(
            base_q,
            dtype=th.float32,
            device=self.device,
        ).reshape(1, self.dof).repeat(self.batch_size, 1).contiguous()
        self.warm_retract = th.empty(
            (self.batch_size, self.dof),
            dtype=th.float32,
            device=self.device,
        )
        self._seed_cache: OrderedDict[
            Tuple[bytes, int, int],
            Tuple[np.ndarray, th.Tensor],
        ] = OrderedDict()
        self._seed_cache_limit = 32
        self.seed_replay_enabled = (
            os.environ.get(
                "IK_FILTER_SEED_REPLAY_CACHE",
                "0",
            ).strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            if seed_replay_enabled is None
            else bool(seed_replay_enabled)
        )
        self.goal_buffer_hits = 0
        self.goal_buffer_misses = 0
        self.retract_buffer_hits = 0
        self.seed_cache_hits = 0
        self.seed_cache_misses = 0
        self.solve_shape_hits = 0
        self.solve_shape_misses = 0
        self._last_solve_shape: Optional[int] = None
        self.logical_shape_rewarm_count = 0
        self._last_logical_solve_shape: Optional[int] = None
        self._solver = solver

    def stats(self) -> Dict[str, int]:
        return {
            "goal_buffer_hits": int(self.goal_buffer_hits),
            "goal_buffer_misses": int(self.goal_buffer_misses),
            "retract_buffer_hits": int(self.retract_buffer_hits),
            "seed_cache_hits": int(self.seed_cache_hits),
            "seed_cache_misses": int(self.seed_cache_misses),
            "seed_cache_entries": int(len(self._seed_cache)),
            "seed_replay_enabled": int(self.seed_replay_enabled),
            "solve_shape_hits": int(self.solve_shape_hits),
            "solve_shape_misses": int(self.solve_shape_misses),
            "logical_shape_rewarm_count": int(
                self.logical_shape_rewarm_count
            ),
        }

    def reset_request_shape_state(self) -> None:
        """Replay the same graph warm-up path as a fresh one-shot worker."""
        self._last_solve_shape = None
        self._last_logical_solve_shape = None

    @staticmethod
    def stats_delta(
        after: Dict[str, int],
        before: Dict[str, int],
    ) -> Dict[str, int]:
        return {
            key: int(after.get(key, 0) - before.get(key, 0))
            for key in after
            if key not in {
                "seed_cache_entries",
                "seed_replay_enabled",
            }
        } | {
            "seed_cache_entries": int(after.get("seed_cache_entries", 0)),
            "seed_replay_enabled": int(
                after.get("seed_replay_enabled", 0)
            ),
        }

    def goal_pose(
        self,
        req: Dict[str, Any],
        poses: List[Dict[str, Any]],
    ):
        from curobo.types.math import Pose

        n = int(len(poses))
        if n > self.batch_size:
            raise ValueError(
                f"IK runtime goal batch {n} exceeds {self.batch_size}"
            )
        local_pos, local_quat = _world_to_curobo_pose_cpu(req, poses)
        self.goal_position[:n].copy_(local_pos)
        self.goal_quaternion[:n].copy_(local_quat)
        self.goal_buffer_hits += 1
        return Pose(
            position=self.goal_position[:n],
            quaternion=self.goal_quaternion[:n],
        )

    def retract(
        self,
        warm_q: Optional[List[List[float]]],
        *,
        group_n: int,
    ) -> th.Tensor:
        self.retract_buffer_hits += 1
        if warm_q is None:
            return self.cold_retract[:group_n]
        warm_cpu = np.asarray(warm_q, dtype=np.float32).reshape(
            group_n,
            self.dof,
        )
        self.warm_retract[:group_n].copy_(
            th.from_numpy(warm_cpu),
        )
        return self.warm_retract[:group_n]

    @staticmethod
    def _generator_state_key(generator) -> bytes:
        state = generator.get_state().detach().cpu().contiguous()
        return state.numpy().tobytes()

    def _random_seeds(
        self,
        solver,
        *,
        batch_n: int,
        random_seed_count: int,
    ) -> th.Tensor:
        generator = solver.q_sample_gen._int_gen
        state_key = self._generator_state_key(generator)
        key = (state_key, int(batch_n), int(random_seed_count))
        cached = self._seed_cache.get(key)
        if cached is not None:
            random_cpu, post_state = cached
            generator.set_state(post_state)
            self._seed_cache.move_to_end(key)
            self.seed_cache_hits += 1
            return th.as_tensor(
                random_cpu,
                dtype=th.float32,
                device=self.device,
            ).contiguous()

        random = solver.generate_seed(
            num_seeds=int(random_seed_count),
            batch=int(batch_n),
            use_nn_seed=False,
        )
        post_state = generator.get_state().detach().cpu().clone()
        random_cpu = (
            random.detach().cpu().contiguous().numpy().copy()
        )
        self._seed_cache[key] = (random_cpu, post_state)
        self._seed_cache.move_to_end(key)
        while len(self._seed_cache) > self._seed_cache_limit:
            self._seed_cache.popitem(last=False)
        self.seed_cache_misses += 1
        return random

    def seeds(
        self,
        solver,
        *,
        retract: th.Tensor,
        use_warm_start: bool,
        group_n: int,
    ) -> Optional[th.Tensor]:
        if not self.seed_replay_enabled:
            if not use_warm_start:
                return None
            return retract[:, None, :].contiguous()
        warm_count = 1 if use_warm_start else 0
        random_count = int(self.num_seeds - warm_count)
        random = self._random_seeds(
            solver,
            batch_n=group_n,
            random_seed_count=random_count,
        )
        if not use_warm_start:
            return random
        return th.cat(
            (retract[:, None, :], random),
            dim=1,
        ).contiguous()

    def note_solve_shape(self, group_n: int) -> None:
        if self._last_solve_shape == int(group_n):
            self.solve_shape_hits += 1
        else:
            self.solve_shape_misses += 1
            self._last_solve_shape = int(group_n)

    def logical_shape_changed(self, group_n: int) -> bool:
        group_n = int(group_n)
        changed = self._last_logical_solve_shape != group_n
        self._last_logical_solve_shape = group_n
        if changed:
            self.logical_shape_rewarm_count += 1
        return changed


def _eef_extra_link_cfg(req: Dict[str, Any], arm: str, link_name: str) -> Dict[str, Any]:
    raw = (req.get("eef_extra_links") or {}).get(arm) or {}
    parent = str(raw.get("parent_link_name") or f"{arm}_gripper_link")
    fixed = raw.get("fixed_transform") or R1PRO_GRIPPER_TO_EEF_FIXED_TRANSFORM
    if len(fixed) != 7:
        fixed = R1PRO_GRIPPER_TO_EEF_FIXED_TRANSFORM
    return {
        "parent_link_name": parent,
        "link_name": str(link_name),
        "fixed_transform": [float(x) for x in fixed],
        "joint_type": "FIXED",
        "joint_name": str(raw.get("joint_name") or f"{parent}_to_{link_name}_fixed_joint"),
    }


def _urdf_names(urdf_path: str) -> Tuple[set[str], set[str], str | None]:
    root = ET.parse(str(urdf_path)).getroot()
    links = {str(x.attrib["name"]) for x in root.findall("link") if x.attrib.get("name")}
    joints = {str(x.attrib["name"]) for x in root.findall("joint") if x.attrib.get("name")}
    children = {
        str(child.attrib["link"])
        for joint in root.findall("joint")
        for child in [joint.find("child")]
        if child is not None and child.attrib.get("link")
    }
    roots = sorted(links - children)
    return links, joints, (roots[0] if roots else None)


def _filter_named_vector(vec: Any, names: List[str], keep: List[str]) -> Any:
    if vec is None:
        return None
    if not isinstance(vec, list):
        return vec
    if len(vec) == 1:
        return vec
    if len(vec) != len(names):
        return vec
    idx_by_name = {str(name): i for i, name in enumerate(names)}
    return [vec[idx_by_name[name]] for name in keep if name in idx_by_name]


def _lock_custom_j8_in_kinematics(
    kin: Dict[str, Any],
    req: Dict[str, Any],
    arm: str,
) -> None:
    if int(req.get("arm_dof", 7)) != 8:
        return
    joint_name = f"{arm}_arm_joint8"
    lock_joints = dict(kin.get("lock_joints") or {})
    lock_joints[joint_name] = 0.0
    kin["lock_joints"] = lock_joints

    cspace = dict(kin.get("cspace") or {})
    old_names = [str(name) for name in cspace.get("joint_names", [])]
    keep_names = [name for name in old_names if name != joint_name]
    for key in (
        "cspace_distance_weight",
        "null_space_weight",
        "max_acceleration",
        "max_jerk",
        "velocity_scale",
        "acceleration_scale",
        "jerk_scale",
        "position_limit_clip",
        "retract_config",
    ):
        if key in cspace:
            cspace[key] = _filter_named_vector(
                cspace.get(key),
                old_names,
                keep_names,
            )
    cspace["joint_names"] = keep_names
    kin["cspace"] = cspace

    q_by_name = req.setdefault("q_by_name", {})
    q_by_name[joint_name] = 0.0


def _set_urdf_arm_only_kinematics(
    robot_cfg: Dict[str, Any],
    req: Dict[str, Any],
    arm: str,
    link_name: str,
    urdf_path: str,
) -> Dict[str, Any]:
    """Make the OG cuRobo ARM yaml usable with standalone URDF IK only."""
    kin = robot_cfg["kinematics"]
    urdf_links, urdf_joints, urdf_root = _urdf_names(urdf_path)
    kin["base_link"] = str(urdf_root or "base_link")
    kin["urdf_path"] = urdf_path
    kin["external_asset_path"] = urdf_path
    kin["asset_root_path"] = os.path.dirname(urdf_path)
    kin["load_meshes"] = False

    ordered_extra_links: Dict[str, Any] = {}
    added_extra = None
    extra_link_names: set[str] = set()
    for extra_arm, extra_link_name in (req.get("eef_link_names") or {}).items():
        if str(extra_arm) != arm:
            continue
        extra_cfg = _eef_extra_link_cfg(req, str(extra_arm), str(extra_link_name))
        parent = str(extra_cfg.get("parent_link_name") or "")
        if parent in urdf_links:
            ordered_extra_links[str(extra_link_name)] = extra_cfg
            extra_link_names.add(str(extra_link_name))
            added_extra = extra_cfg
    kin["extra_links"] = ordered_extra_links

    # The shipped ARM yaml is USD-oriented and includes virtual base joints plus
    # collision spheres. The external worker only ranks IK poses, so strip that
    # collision model and keep the active URDF arm chain small and deterministic.
    kin["collision_link_names"] = []
    kin["collision_spheres"] = {}
    kin["extra_collision_spheres"] = {}
    kin["self_collision_ignore"] = {}
    kin["self_collision_buffer"] = {}
    kin["mesh_link_names"] = []
    kin["link_names"] = [str(link_name)]

    lock_joints = dict(kin.get("lock_joints") or {})
    q_by_name = req.get("q_by_name") or {}
    kin["lock_joints"] = {
        str(joint_name): (
            float(q_by_name[joint_name])
            if lock_val is None and joint_name in q_by_name
            else lock_val
        )
        for joint_name, lock_val in lock_joints.items()
        if str(joint_name) in urdf_joints
        and "gripper_finger" not in str(joint_name)
        and not (
            str(joint_name).startswith(("left_arm_joint", "right_arm_joint"))
            and not str(joint_name).startswith(f"{arm}_arm_joint")
        )
    }

    cspace = dict(kin.get("cspace") or {})
    old_names = [str(x) for x in cspace.get("joint_names", [])]
    arm_dof = int(req.get("arm_dof", 7))
    arm_joint_names = [
        f"{arm}_arm_joint{i}"
        for i in range(1, arm_dof + 1)
    ]
    keep_names = [name for name in arm_joint_names if name in urdf_joints]
    if not keep_names:
        keep_names = [name for name in old_names if name in urdf_joints and f"{arm}_" in name]
    for key in (
        "cspace_distance_weight",
        "null_space_weight",
        "max_acceleration",
        "max_jerk",
        "velocity_scale",
        "acceleration_scale",
        "jerk_scale",
        "position_limit_clip",
        "retract_config",
    ):
        if key in cspace:
            cspace[key] = _filter_named_vector(cspace.get(key), old_names, keep_names)
    cspace["joint_names"] = keep_names
    if cspace.get("retract_config") is None:
        cspace["retract_config"] = [float(q_by_name.get(name, 0.0)) for name in keep_names]
    kin["cspace"] = cspace
    return {
        "base_link": kin["base_link"],
        "extra_link": added_extra,
        "urdf_links": len(urdf_links),
        "urdf_joints": len(urdf_joints),
        "active_joints": keep_names,
    }


def _load_solver(req: Dict[str, Any], arm: str, *, log_fn=print):
    from curobo.cuda_robot_model.util import load_robot_yaml
    from curobo.types.base import TensorDeviceType
    from curobo.types.file_path import ContentPath
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    robot_cfg_path = req["robot_cfg_path"]
    usd_path = str(req.get("robot_usd_path") or req["robot_urdf_path"])
    link_name = req["eef_link_names"][arm]
    # include_nvidia=False：高负载下每次 nvidia-smi 子进程要 1~2s，纯诊断不值得
    _log_gpu_diag(
        log_fn,
        "ik_filter_worker.load_solver.before",
        extra={
            "arm": arm,
            "link": link_name,
            "batch_size": int(req.get("batch_size", 0)),
            "num_seeds": int(req.get("num_seeds", 0)),
            "ik_opt_iters": int(req.get("ik_opt_iters", 0)),
        },
        include_nvidia=False,
    )
    tensor_args = TensorDeviceType(device=th.device("cuda:0"))
    content_path = ContentPath(
        robot_config_absolute_path=robot_cfg_path,
        robot_usd_absolute_path=usd_path,
    )
    robot_cfg = load_robot_yaml(content_path)["robot_cfg"]
    use_usd = bool(req.get("use_usd_kinematics", True))
    kin = robot_cfg["kinematics"]
    kin["use_usd_kinematics"] = use_usd
    added_extra = None
    urdf_meta: Dict[str, Any] = {}
    if not use_usd:
        urdf_path = req.get("robot_urdf_path")
        if urdf_path:
            urdf_meta = _set_urdf_arm_only_kinematics(robot_cfg, req, arm, link_name, urdf_path)
            added_extra = urdf_meta.get("extra_link")
    _lock_custom_j8_in_kinematics(kin, req, arm)
    if urdf_meta:
        urdf_meta["active_joints"] = list(
            robot_cfg["kinematics"].get("cspace", {}).get("joint_names", [])
        )
    kin["ee_link"] = link_name
    q_by_name = req["q_by_name"]
    for joint_name, lock_val in list(robot_cfg["kinematics"].get("lock_joints", {}).items()):
        if lock_val is None and joint_name in q_by_name:
            robot_cfg["kinematics"]["lock_joints"][joint_name] = float(q_by_name[joint_name])
    cspace = robot_cfg["kinematics"].get("cspace", {})
    if cspace.get("retract_config") is None:
        cspace["retract_config"] = [
            float(q_by_name.get(joint_name, 0.0))
            for joint_name in cspace.get("joint_names", [])
        ]
    cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg=robot_cfg,
        world_model=None,
        tensor_args=tensor_args,
        num_seeds=int(req["num_seeds"]),
        position_threshold=float(req["pos_tol_m"]),
        rotation_threshold=math.radians(float(req["ori_tol_deg"])),
        use_cuda_graph=_fixed_cuda_graph_enabled(req),
        self_collision_check=False,
        self_collision_opt=False,
        use_particle_opt=False,
        collision_checker_type=None,
        grad_iters=int(req["ik_opt_iters"]),
        high_precision=True,
        regularization=True,
        ee_link_name=link_name,
        project_pose_to_goal_frame=True,
        seed=1531,
    )
    solver = IKSolver(cfg)
    _log_gpu_diag(
        log_fn,
        "ik_filter_worker.load_solver.after",
        extra={"arm": arm, "link": link_name},
        include_nvidia=False,
    )
    return solver, tensor_args, {
        "ee_link": str(link_name),
        "use_usd_kinematics": bool(use_usd),
        "use_cuda_graph": bool(_fixed_cuda_graph_enabled(req)),
        "fixed_solve_batch_size": (
            int(_fixed_physical_batch_size(req) or req["batch_size"])
            if _fixed_cuda_graph_enabled(req)
            else None
        ),
        "extra_link": added_extra,
        "urdf": urdf_meta,
    }


def _q_arm_from_js(
    req: Dict[str, Any],
    arm: str,
    js_obj,
    q_active_names: List[str],
    *,
    batch_i: int = 0,
    seed_i: int = 0,
) -> List[float] | None:
    arm_dof = int(req.get("arm_dof", 7))
    q_full = dict(req["q_by_name"])
    js_pos_full = js_obj.position.detach().cpu().float()
    js_names = [str(n) for n in js_obj.joint_names]
    n_names = len(js_names) if js_names else len(q_active_names)
    if js_pos_full.dim() == 1:
        js_pos = js_pos_full
    elif js_pos_full.dim() == 2:
        row_i = max(0, min(int(batch_i), int(js_pos_full.shape[0]) - 1))
        if int(js_pos_full.shape[-1]) == n_names:
            js_pos = js_pos_full[row_i]
        else:
            js_pos = js_pos_full.reshape(-1)
    elif js_pos_full.dim() >= 3:
        if int(js_pos_full.shape[-1]) == n_names:
            bi = max(0, min(int(batch_i), int(js_pos_full.shape[0]) - 1))
            si = max(0, min(int(seed_i), int(js_pos_full.shape[1]) - 1))
            js_pos = js_pos_full[bi, si]
        else:
            js_pos = js_pos_full.reshape(-1)
    else:
        return None
    if len(js_names) == int(js_pos.numel()):
        for i, name in enumerate(js_names):
            q_full[name] = float(js_pos[i].item())
    elif len(q_active_names) == int(js_pos.numel()):
        for i, name in enumerate(q_active_names):
            q_full[name] = float(js_pos[i].item())
    else:
        return None
    out = []
    for i in range(1, arm_dof + 1):
        name = f"{arm}_arm_joint{i}"
        if name not in q_full:
            return None
        out.append(0.0 if arm_dof == 8 and i == 8 else float(q_full[name]))
    return out


def _q_arms_from_js(
    req: Dict[str, Any],
    arm: str,
    js_obj,
    q_active_names: List[str],
    *,
    batch_n: int,
    seed_indices: List[int],
) -> List[List[float] | None]:
    """Extract one selected arm solution per batch row with one CPU transfer."""
    arm_dof = int(req.get("arm_dof", 7))
    if js_obj is None:
        return [None] * int(batch_n)
    js_pos_full = js_obj.position.detach().cpu().float()
    js_names = [str(n) for n in js_obj.joint_names]
    n_names = len(js_names) if js_names else len(q_active_names)
    output: List[List[float] | None] = []
    for batch_i in range(int(batch_n)):
        if js_pos_full.dim() == 1:
            js_pos = js_pos_full
        elif js_pos_full.dim() == 2:
            row_i = max(
                0,
                min(int(batch_i), int(js_pos_full.shape[0]) - 1),
            )
            if int(js_pos_full.shape[-1]) == n_names:
                js_pos = js_pos_full[row_i]
            else:
                js_pos = js_pos_full.reshape(-1)
        elif js_pos_full.dim() >= 3:
            if int(js_pos_full.shape[-1]) == n_names:
                row_i = max(
                    0,
                    min(int(batch_i), int(js_pos_full.shape[0]) - 1),
                )
                raw_seed = (
                    seed_indices[batch_i]
                    if batch_i < len(seed_indices)
                    else 0
                )
                seed_i = max(
                    0,
                    min(int(raw_seed), int(js_pos_full.shape[1]) - 1),
                )
                js_pos = js_pos_full[row_i, seed_i]
            else:
                js_pos = js_pos_full.reshape(-1)
        else:
            output.append(None)
            continue

        q_full = dict(req["q_by_name"])
        if len(js_names) == int(js_pos.numel()):
            for index, name in enumerate(js_names):
                q_full[name] = float(js_pos[index].item())
        elif len(q_active_names) == int(js_pos.numel()):
            for index, name in enumerate(q_active_names):
                q_full[name] = float(js_pos[index].item())
        else:
            output.append(None)
            continue
        q_arm = []
        for joint_index in range(1, arm_dof + 1):
            name = f"{arm}_arm_joint{joint_index}"
            if name not in q_full:
                q_arm = []
                break
            q_arm.append(
                0.0
                if arm_dof == 8 and joint_index == 8
                else float(q_full[name])
            )
        output.append(q_arm if len(q_arm) == arm_dof else None)
    return output


def _warm_start_q_for_pose(
    req: Dict[str, Any],
    arm: str,
    pose: Dict[str, Any],
    q_active_names: List[str],
) -> List[float] | None:
    arm_dof = int(req.get("arm_dof", 7))
    q_arm = (pose.get("warm_start_q_by_arm") or {}).get(arm)
    if q_arm is None:
        return None
    q_arm_array = np.asarray(q_arm, dtype=np.float64).reshape(-1)
    if len(q_arm_array) != arm_dof or not np.all(np.isfinite(q_arm_array)):
        return None
    if arm_dof == 8:
        q_arm_array = q_arm_array.copy()
        q_arm_array[7] = 0.0
    q_by_name = dict(req["q_by_name"])
    for index, value in enumerate(q_arm_array, start=1):
        q_by_name[f"{arm}_arm_joint{index}"] = float(value)
    return [float(q_by_name.get(name, 0.0)) for name in q_active_names]


def _fixed_shape_seed_config(
    solver,
    *,
    retract: th.Tensor,
    use_warm_start: bool,
    valid_n: int,
    solve_n: int,
    num_seeds: int,
) -> th.Tensor:
    """Build fixed-shape seeds with exact duplicate dummy problems."""
    valid_n = int(valid_n)
    solve_n = int(solve_n)
    num_seeds = int(num_seeds)
    if not (0 < valid_n <= solve_n):
        raise ValueError(
            f"invalid fixed-shape seed counts valid={valid_n} solve={solve_n}"
        )
    warm_count = 1 if use_warm_start else 0
    random_count = int(num_seeds - warm_count)
    if random_count < 0:
        raise ValueError(
            f"num_seeds={num_seeds} smaller than warm_count={warm_count}"
        )

    valid_random = (
        solver.generate_seed(
            num_seeds=random_count,
            batch=valid_n,
            use_nn_seed=False,
        )
        if random_count > 0
        else retract.new_empty((valid_n, 0, retract.shape[-1]))
    )
    random = valid_random
    if valid_n < solve_n:
        random = th.cat(
            (
                valid_random,
                valid_random[-1:].expand(
                    solve_n - valid_n,
                    -1,
                    -1,
                ),
            ),
            dim=0,
        )
    if not use_warm_start:
        return random.contiguous()
    warm_retract = retract[:valid_n]
    if valid_n < solve_n:
        warm_retract = th.cat(
            (
                warm_retract,
                warm_retract[-1:].expand(
                    solve_n - valid_n,
                    -1,
                ),
            ),
            dim=0,
        )
    return th.cat(
        (warm_retract[:, None, :], random),
        dim=1,
    ).contiguous()


def _fixed_shape_seed_configs(
    solver,
    *,
    retract: th.Tensor,
    use_warm_start: bool,
    valid_n: int,
    solve_n: int,
    num_seeds: int,
    total_num_seeds: int,
) -> List[th.Tensor]:
    """Build one or more fixed-shape passes from one deterministic seed draw."""
    num_seeds = int(num_seeds)
    total_num_seeds = int(total_num_seeds)
    if total_num_seeds == num_seeds:
        return [
            _fixed_shape_seed_config(
                solver,
                retract=retract,
                use_warm_start=use_warm_start,
                valid_n=valid_n,
                solve_n=solve_n,
                num_seeds=num_seeds,
            )
        ]
    if (
        num_seeds <= 0
        or total_num_seeds < num_seeds
        or total_num_seeds % num_seeds
    ):
        raise ValueError(
            "total_num_seeds must be a positive multiple of num_seeds: "
            f"{total_num_seeds} vs {num_seeds}"
        )

    valid_n = int(valid_n)
    solve_n = int(solve_n)
    if not (0 < valid_n <= solve_n):
        raise ValueError(
            f"invalid fixed-shape seed counts valid={valid_n} solve={solve_n}"
        )
    warm_count = 1 if use_warm_start else 0
    random_count = int(total_num_seeds - warm_count)
    valid_random = (
        solver.generate_seed(
            num_seeds=random_count,
            batch=valid_n,
            use_nn_seed=False,
        )
        if random_count > 0
        else retract.new_empty((valid_n, 0, retract.shape[-1]))
    )
    random = _pad_first_dim(valid_random, solve_n)
    if use_warm_start:
        warm_retract = _pad_first_dim(retract[:valid_n], solve_n)
        all_seeds = th.cat(
            (warm_retract[:, None, :], random),
            dim=1,
        ).contiguous()
    else:
        all_seeds = random.contiguous()
    if int(all_seeds.shape[1]) != total_num_seeds:
        raise RuntimeError(
            "unexpected total seed width "
            f"{int(all_seeds.shape[1])} != {total_num_seeds}"
        )
    return [
        all_seeds[:, start : start + num_seeds].contiguous()
        for start in range(0, total_num_seeds, num_seeds)
    ]


def _pad_first_dim(
    tensor: th.Tensor,
    target_n: int,
) -> th.Tensor:
    current_n = int(tensor.shape[0])
    target_n = int(target_n)
    if current_n <= 0 or target_n < current_n:
        raise ValueError(
            f"invalid first-dimension padding {current_n}->{target_n}"
        )
    if current_n == target_n:
        return tensor.contiguous()
    expand_shape = (
        target_n - current_n,
        *([-1] * (tensor.ndim - 1)),
    )
    return th.cat(
        (tensor, tensor[-1:].expand(*expand_shape)),
        dim=0,
    ).contiguous()


def _solve_pose_batch(
    req: Dict[str, Any],
    arm: str,
    poses: List[Dict[str, Any]],
    *,
    solver,
    tensor_args,
    q_names: List[str],
    pos_tol_m: float,
    ori_tol_deg: float,
    source_stage: str,
    runtime_cache: Optional[_IKRuntimeTensorCache] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    batch = int(max(1, req["batch_size"]))
    fixed_graph = _fixed_cuda_graph_enabled(req)
    fixed_batch = _fixed_batch_enabled(req)
    fixed_physical_batch = _fixed_physical_batch_size(req)
    logical_rewarm = _logical_shape_rewarm_enabled(req)
    total_num_seeds = (
        24
        if str(req.get("solver_policy") or "").strip()
        == IK_FIXED_CUDA_GRAPH_WARM32X6_COLD24_REWARM_POLICY
        else int(req["num_seeds"])
    )
    q_by_name = req["q_by_name"]
    out: List[Dict[str, Any]] = []
    warm_start_pose_count = 0
    for offset in range(0, len(poses), batch):
        chunk = poses[offset: offset + batch]
        b = len(chunk)
        warm_q = [
            _warm_start_q_for_pose(req, arm, pose, q_names)
            for pose in chunk
        ]
        warm_indices = [i for i, q in enumerate(warm_q) if q is not None]
        cold_indices = [i for i, q in enumerate(warm_q) if q is None]
        warm_start_pose_count += len(warm_indices)
        chunk_out: List[Dict[str, Any] | None] = [None] * b

        for use_warm_start, indices in (
            (True, warm_indices),
            (False, cold_indices),
        ):
            if not indices:
                continue
            group = [chunk[i] for i in indices]
            group_n = len(group)
            warm_group = (
                [warm_q[i] for i in indices]
                if use_warm_start
                else None
            )
            shape_changed = False
            if logical_rewarm:
                if runtime_cache is not None:
                    shape_changed = runtime_cache.logical_shape_changed(
                        group_n
                    )
                else:
                    previous_shape = getattr(
                        solver,
                        "_interface_last_logical_solve_shape",
                        None,
                    )
                    shape_changed = previous_shape != int(group_n)
                    solver._interface_last_logical_solve_shape = int(
                        group_n
                    )

            solve_units = []
            if fixed_batch:
                physical_batch = int(fixed_physical_batch or batch)
                if runtime_cache is None:
                    full_retract = th.tensor(
                        [
                            [
                                float(q_by_name.get(jn, 0.0))
                                for jn in q_names
                            ]
                        ],
                        dtype=th.float32,
                        device=tensor_args.device,
                    ).repeat(group_n, 1).contiguous()
                    if use_warm_start:
                        full_retract = th.tensor(
                            warm_group,
                            dtype=th.float32,
                            device=tensor_args.device,
                        ).contiguous()
                else:
                    full_retract = runtime_cache.retract(
                        warm_group,
                        group_n=group_n,
                    )
                full_seed_configs = _fixed_shape_seed_configs(
                    solver,
                    retract=full_retract,
                    use_warm_start=use_warm_start,
                    valid_n=group_n,
                    solve_n=group_n,
                    num_seeds=int(req["num_seeds"]),
                    total_num_seeds=total_num_seeds,
                )
                for sub_start in range(0, group_n, physical_batch):
                    sub_end = min(group_n, sub_start + physical_batch)
                    valid_n = int(sub_end - sub_start)
                    solve_group = list(group[sub_start:sub_end])
                    solve_group.extend(
                        [solve_group[-1]]
                        * (physical_batch - valid_n)
                    )
                    solve_units.append(
                        {
                            "poses": solve_group,
                            "valid_n": valid_n,
                            "chunk_indices": indices[sub_start:sub_end],
                            "retract": _pad_first_dim(
                                full_retract[sub_start:sub_end],
                                physical_batch,
                            ),
                            "seed_configs": [
                                _pad_first_dim(
                                    seed_config[sub_start:sub_end],
                                    physical_batch,
                                )
                                for seed_config in full_seed_configs
                            ],
                            "rewarm": bool(shape_changed),
                            "logical_n": int(group_n),
                        }
                    )
            else:
                solve_group = list(group)
                solve_n = int(group_n)
                if runtime_cache is None:
                    retract = th.tensor(
                        [
                            [
                                float(q_by_name.get(jn, 0.0))
                                for jn in q_names
                            ]
                        ],
                        dtype=th.float32,
                        device=tensor_args.device,
                    ).repeat(solve_n, 1).contiguous()
                    seed_config = None
                    if use_warm_start:
                        retract = th.tensor(
                            warm_group,
                            dtype=th.float32,
                            device=tensor_args.device,
                        ).contiguous()
                        seed_config = retract[:, None, :].contiguous()
                else:
                    retract = runtime_cache.retract(
                        warm_group,
                        group_n=solve_n,
                    )
                    seed_config = runtime_cache.seeds(
                        solver,
                        retract=retract,
                        use_warm_start=use_warm_start,
                        group_n=group_n,
                    )
                solve_units.append(
                    {
                        "poses": solve_group,
                        "valid_n": int(group_n),
                        "chunk_indices": indices,
                        "retract": retract,
                        "seed_configs": [seed_config],
                        "rewarm": False,
                        "logical_n": int(group_n),
                    }
                )

            for unit in solve_units:
                solve_group = unit["poses"]
                valid_n = int(unit["valid_n"])
                solve_n = int(len(solve_group))
                goal = (
                    runtime_cache.goal_pose(req, solve_group)
                    if runtime_cache is not None
                    else _world_to_curobo_pose(
                        req,
                        solve_group,
                        tensor_args,
                    )
                )
                if runtime_cache is not None:
                    runtime_cache.note_solve_shape(solve_n)
                if unit["rewarm"]:
                    # Every physical slice represents independent rows of the
                    # same logical solve. Recreate the two LBFGS warm-up passes
                    # for each slice when the baseline logical shape changes.
                    solver.solver._init_solver = False
                for seed_pass, seed_config in enumerate(
                    unit["seed_configs"]
                ):
                    res = solver.solve_batch(
                        goal,
                        retract_config=unit["retract"],
                        seed_config=seed_config,
                        return_seeds=1,
                        num_seeds=int(req["num_seeds"]),
                        use_nn_seed=False,
                        newton_iters=int(req["ik_opt_iters"]),
                        link_poses=None,
                    )
                    pe = (
                        res.position_error.detach()
                        .float()
                        .reshape(solve_n, -1)
                    )[:valid_n]
                    re = (
                        res.rotation_error.detach()
                        .float()
                        .reshape(solve_n, -1)
                    )[:valid_n]
                    err = (
                        res.error.detach().float().reshape(solve_n, -1)
                    )[:valid_n]
                    succ = (
                        res.success.detach().bool().reshape(solve_n, -1)
                    )[:valid_n]
                    gated = err.clone()
                    gated[~succ] = float("inf")
                    best = gated.argmin(dim=1)
                    no_success = ~th.isfinite(
                        gated.min(dim=1).values
                    )
                    if bool(no_success.any().item()):
                        best[no_success] = err.argmin(dim=1)[
                            no_success
                        ]
                    best_cpu = best.detach().long().cpu().tolist()
                    row_indices = th.arange(
                        valid_n,
                        device=best.device,
                    )
                    selected_pos = pe[
                        row_indices,
                        best.clamp(0, pe.shape[1] - 1),
                    ].detach().cpu().tolist()
                    selected_rot = re[
                        row_indices,
                        best.clamp(0, re.shape[1] - 1),
                    ].detach().cpu().tolist()
                    selected_error = err[
                        row_indices,
                        best.clamp(0, err.shape[1] - 1),
                    ].detach().cpu().tolist()
                    selected_success = succ[
                        row_indices,
                        best.clamp(0, succ.shape[1] - 1),
                    ].detach().cpu().tolist()
                    q_arms = _q_arms_from_js(
                        req,
                        arm,
                        res.js_solution,
                        q_names,
                        batch_n=solve_n,
                        seed_indices=best_cpu,
                    )[:valid_n]
                    for group_i, chunk_i in enumerate(
                        unit["chunk_indices"]
                    ):
                        si = (
                            int(seed_pass) * int(req["num_seeds"])
                            + int(best_cpu[group_i])
                        )
                        pos_m = float(selected_pos[group_i])
                        ori_rad = float(selected_rot[group_i])
                        solver_error = float(
                            selected_error[group_i]
                        )
                        solver_success = bool(
                            selected_success[group_i]
                        )
                        previous = chunk_out[chunk_i]
                        if previous is not None:
                            previous_success = bool(
                                previous["_solver_success"]
                            )
                            previous_error = float(
                                previous["_solver_error"]
                            )
                            if previous_success and not solver_success:
                                continue
                            if (
                                previous_success == solver_success
                                and solver_error >= previous_error
                            ):
                                continue
                        ok = bool(
                            pos_m <= float(pos_tol_m)
                            and math.degrees(ori_rad)
                            <= float(ori_tol_deg)
                        )
                        source = (
                            f"gpu_worker stage={source_stage} arm={arm} "
                            f"offset={offset} batch={chunk_i} "
                            f"warm={int(use_warm_start)} "
                            f"solve_n={solve_n} valid_n={valid_n} "
                            f"logical_n={unit['logical_n']} "
                            f"fixed_batch={int(fixed_batch)} "
                            f"cuda_graph={int(fixed_graph)}"
                        )
                        if len(unit["seed_configs"]) > 1:
                            source += (
                                f" seed_pass={seed_pass + 1}/"
                                f"{len(unit['seed_configs'])}"
                            )
                        chunk_out[chunk_i] = {
                            "ok": ok,
                            "pos_err_m": pos_m,
                            "ori_err_deg": math.degrees(ori_rad),
                            "approach_err_deg": math.degrees(ori_rad),
                            "cu_pos_m": pos_m,
                            "cu_ori_deg": math.degrees(ori_rad),
                            "cu_success": solver_success,
                            "q_arm": q_arms[group_i],
                            "seed_i": si,
                            "warm_start_used": bool(use_warm_start),
                            "warm_start_retract_used": bool(
                                use_warm_start
                            ),
                            "source": source,
                            "_solver_error": solver_error,
                            "_solver_success": solver_success,
                        }
        for item in chunk_out:
            if item is not None:
                item.pop("_solver_error", None)
                item.pop("_solver_success", None)
        out.extend(
            item
            if item is not None
            else {
                "ok": False,
                "pos_err_m": float("inf"),
                "ori_err_deg": float("inf"),
                "approach_err_deg": float("inf"),
                "error": "worker_group_result_missing",
            }
            for item in chunk_out
        )
    return out, int(warm_start_pose_count)


def _warm_only_probe_stage(
    req: Dict[str, Any],
    arm: str,
    poses: List[Dict[str, Any]],
    baseline_results: List[Dict[str, Any]],
    *,
    solver,
    tensor_args,
    q_names: List[str],
    pos_tol_m: float,
    ori_tol_deg: float,
    source_stage: str,
    branch_reference_q: Optional[List[List[float]]] = None,
    branch_limits: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compare the baseline 24-seed result with a warm-seed-only replay."""
    import time

    selected_indices = [
        index
        for index, pose in enumerate(poses)
        if _warm_start_q_for_pose(req, arm, pose, q_names) is not None
    ]
    if not selected_indices:
        return {
            "stage": str(source_stage),
            "input_count": int(len(poses)),
            "warm_pose_count": 0,
            "skipped": True,
        }

    probe_req = dict(req)
    probe_req["num_seeds"] = 1
    probe_poses = [poses[index] for index in selected_indices]
    started = time.perf_counter()
    probe_results, probe_warm_count = _solve_pose_batch(
        probe_req,
        arm,
        probe_poses,
        solver=solver,
        tensor_args=tensor_args,
        q_names=q_names,
        pos_tol_m=pos_tol_m,
        ori_tol_deg=ori_tol_deg,
        source_stage=f"{source_stage}_warm_only_probe",
        runtime_cache=None,
    )
    elapsed_s = time.perf_counter() - started

    comparisons: List[Dict[str, Any]] = []
    q_deltas: List[float] = []
    for probe_i, source_i in enumerate(selected_indices):
        baseline = baseline_results[source_i]
        probe = probe_results[probe_i]
        baseline_ok = bool(baseline.get("ok"))
        probe_ok = bool(probe.get("ok"))
        baseline_q_raw = baseline.get("q_arm")
        probe_q_raw = probe.get("q_arm")
        q_delta = float("inf")
        if baseline_q_raw is not None and probe_q_raw is not None:
            arm_dof = int(req.get("arm_dof", 7))
            baseline_q = np.asarray(
                baseline_q_raw,
                dtype=np.float64,
            ).reshape(arm_dof)
            probe_q = np.asarray(
                probe_q_raw,
                dtype=np.float64,
            ).reshape(arm_dof)
            q_delta = float(
                np.linalg.norm(probe_q - baseline_q, ord=np.inf)
            )
            if math.isfinite(q_delta):
                q_deltas.append(q_delta)

        baseline_pair_ok = baseline_ok
        probe_pair_ok = probe_ok
        probe_branch_gap = None
        if branch_reference_q is not None:
            arm_dof = int(req.get("arm_dof", 7))
            reference_q = np.asarray(
                branch_reference_q[source_i],
                dtype=np.float64,
            ).reshape(arm_dof)
            branch_limit = float(
                (branch_limits or [0.85] * len(poses))[source_i]
            )
            if probe_q_raw is not None:
                probe_branch_gap = float(
                    np.linalg.norm(
                        np.asarray(
                            probe_q_raw,
                            dtype=np.float64,
                        ).reshape(arm_dof)
                        - reference_q,
                        ord=np.inf,
                    )
                )
            probe_pair_ok = bool(
                probe_ok
                and probe_branch_gap is not None
                and math.isfinite(probe_branch_gap)
                and probe_branch_gap <= branch_limit
            )
        comparisons.append(
            {
                "source_index": int(source_i),
                "baseline_ok": bool(baseline_ok),
                "probe_ok": bool(probe_ok),
                "baseline_pair_ok": bool(baseline_pair_ok),
                "probe_pair_ok": bool(probe_pair_ok),
                "q_inf_delta_rad": (
                    float(q_delta) if math.isfinite(q_delta) else None
                ),
                "baseline_pos_err_m": float(
                    baseline.get("pos_err_m", float("inf"))
                ),
                "probe_pos_err_m": float(
                    probe.get("pos_err_m", float("inf"))
                ),
                "baseline_ori_err_deg": float(
                    baseline.get("ori_err_deg", float("inf"))
                ),
                "probe_ori_err_deg": float(
                    probe.get("ori_err_deg", float("inf"))
                ),
                "probe_branch_gap_rad": probe_branch_gap,
            }
        )

    pair_mismatches = [
        item
        for item in comparisons
        if item["baseline_pair_ok"] != item["probe_pair_ok"]
    ]
    q_close_1e3 = sum(
        float(item["q_inf_delta_rad"]) <= 1e-3
        for item in comparisons
        if item["q_inf_delta_rad"] is not None
    )
    q_close_1e2 = sum(
        float(item["q_inf_delta_rad"]) <= 1e-2
        for item in comparisons
        if item["q_inf_delta_rad"] is not None
    )
    return {
        "stage": str(source_stage),
        "input_count": int(len(poses)),
        "warm_pose_count": int(len(selected_indices)),
        "probe_warm_start_count": int(probe_warm_count),
        "elapsed_s": float(elapsed_s),
        "baseline_ok_count": int(
            sum(item["baseline_pair_ok"] for item in comparisons)
        ),
        "probe_ok_count": int(
            sum(item["probe_pair_ok"] for item in comparisons)
        ),
        "pair_ok_match_count": int(
            len(comparisons) - len(pair_mismatches)
        ),
        "pair_ok_match_ratio": float(
            (len(comparisons) - len(pair_mismatches))
            / max(len(comparisons), 1)
        ),
        "q_compared_count": int(len(q_deltas)),
        "q_close_1e3_count": int(q_close_1e3),
        "q_close_1e2_count": int(q_close_1e2),
        "q_inf_delta_rad_max": (
            float(max(q_deltas)) if q_deltas else None
        ),
        "q_inf_delta_rad_mean": (
            float(sum(q_deltas) / len(q_deltas))
            if q_deltas
            else None
        ),
        "mismatch_preview": pair_mismatches[:20],
    }


def _solve_arm_with_solver(
    req: Dict[str, Any],
    arm: str,
    poses: List[Dict[str, Any]],
    *,
    solver,
    tensor_args,
    solver_meta: Dict[str, Any],
    runtime_cache: Optional[_IKRuntimeTensorCache] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import time

    t0 = time.perf_counter()
    cuda_memory_before: Dict[str, float] = {}
    if th.cuda.is_available():
        th.cuda.reset_peak_memory_stats()
        cuda_memory_before = {
            "allocated_mib": float(th.cuda.memory_allocated() / (1024 ** 2)),
            "reserved_mib": float(th.cuda.memory_reserved() / (1024 ** 2)),
        }
    q_names = [str(n) for n in solver.rollout_fn.kinematics.joint_names]
    final_t0 = time.perf_counter()
    out, warm_start_pose_count = _solve_pose_batch(
        req,
        arm,
        poses,
        solver=solver,
        tensor_args=tensor_args,
        q_names=q_names,
        pos_tol_m=float(req["pos_tol_m"]),
        ori_tol_deg=float(req["ori_tol_deg"]),
        source_stage="final",
        runtime_cache=runtime_cache,
    )
    final_solve_s = time.perf_counter() - final_t0

    paired_requested = 0
    paired_solved = 0
    safe_prepare_t0 = time.perf_counter()
    safe_poses: List[Dict[str, Any]] = []
    safe_parent_indices: List[int] = []
    safe_cfg_by_parent: Dict[int, Dict[str, Any]] = {}
    for pose_i, (pose, final_result) in enumerate(zip(poses, out)):
        paired_cfg = pose.get("paired_safe")
        if not isinstance(paired_cfg, dict):
            continue
        paired_requested += 1
        final_q = final_result.get("q_arm")
        if not final_result.get("ok") or final_q is None:
            final_result["paired_safe"] = {
                "ok": False,
                "error": "final_endpoint_ik_failed",
                "source": "gpu_worker_paired_safe",
            }
            continue
        try:
            safe_pose = {
                "eef_pos": [
                    float(x)
                    for x in np.asarray(
                        paired_cfg["eef_pos"], dtype=np.float64
                    ).reshape(3)
                ],
                "quat": [
                    float(x)
                    for x in np.asarray(
                        paired_cfg.get("quat", pose["quat"]),
                        dtype=np.float64,
                    ).reshape(4)
                ],
                "warm_start_q_by_arm": {arm: list(final_q)},
            }
        except Exception as exc:
            final_result["paired_safe"] = {
                "ok": False,
                "error": f"invalid_paired_safe_target: {type(exc).__name__}: {exc}",
                "source": "gpu_worker_paired_safe",
            }
            continue
        safe_parent_indices.append(int(pose_i))
        safe_cfg_by_parent[int(pose_i)] = paired_cfg
        safe_poses.append(safe_pose)
    safe_prepare_s = time.perf_counter() - safe_prepare_t0

    safe_results: List[Dict[str, Any]] = []
    safe_warm_start_pose_count = 0
    safe_solve_s = 0.0
    if safe_poses:
        first_cfg = safe_cfg_by_parent[safe_parent_indices[0]]
        safe_t0 = time.perf_counter()
        safe_results, safe_warm_start_pose_count = _solve_pose_batch(
            req,
            arm,
            safe_poses,
            solver=solver,
            tensor_args=tensor_args,
            q_names=q_names,
            pos_tol_m=float(first_cfg.get("pos_tol_m", 0.03)),
            ori_tol_deg=float(first_cfg.get("ori_tol_deg", 10.0)),
            source_stage="safe",
            runtime_cache=runtime_cache,
        )
        safe_solve_s = time.perf_counter() - safe_t0
        paired_solved = len(safe_results)
        for parent_i, safe_result in zip(safe_parent_indices, safe_results):
            cfg = safe_cfg_by_parent[parent_i]
            final_result = out[parent_i]
            arm_dof = int(req.get("arm_dof", 7))
            final_q = np.asarray(
                final_result.get("q_arm"), dtype=np.float64
            ).reshape(arm_dof)
            safe_q_raw = safe_result.get("q_arm")
            branch_gap = float("inf")
            if safe_q_raw is not None:
                safe_q = np.asarray(
                    safe_q_raw,
                    dtype=np.float64,
                ).reshape(arm_dof)
                branch_gap = float(np.linalg.norm(safe_q - final_q, ord=np.inf))
            endpoint_ok = bool(safe_result.get("ok"))
            branch_limit = float(cfg.get("final_branch_gap_rad", 0.85))
            safe_result["endpoint_ok"] = endpoint_ok
            safe_result["safe_to_final_joint_gap_rad"] = branch_gap
            safe_result["final_branch_gap_limit_rad"] = branch_limit
            safe_result["same_final_branch"] = bool(
                math.isfinite(branch_gap) and branch_gap <= branch_limit
            )
            safe_result["ok"] = bool(
                endpoint_ok and safe_result["same_final_branch"]
            )
            if endpoint_ok and not safe_result["same_final_branch"]:
                safe_result["error"] = "safe_final_branch_gap"
            safe_result["source"] = (
                f"{safe_result.get('source', 'gpu_worker')} paired_from_final_q"
            )
            final_result["paired_safe"] = safe_result

    warm_probe: Dict[str, Any] = {"enabled": False}
    if str(req.get("solver_policy") or "baseline") == "warm_probe":
        warm_probe = {
            "enabled": True,
            "final": _warm_only_probe_stage(
                req,
                arm,
                poses,
                out,
                solver=solver,
                tensor_args=tensor_args,
                q_names=q_names,
                pos_tol_m=float(req["pos_tol_m"]),
                ori_tol_deg=float(req["ori_tol_deg"]),
                source_stage="final",
            ),
        }
        if safe_poses:
            warm_probe["safe"] = _warm_only_probe_stage(
                req,
                arm,
                safe_poses,
                safe_results,
                solver=solver,
                tensor_args=tensor_args,
                q_names=q_names,
                pos_tol_m=float(first_cfg.get("pos_tol_m", 0.03)),
                ori_tol_deg=float(first_cfg.get("ori_tol_deg", 10.0)),
                source_stage="safe",
                branch_reference_q=[
                    list(out[parent_i]["q_arm"])
                    for parent_i in safe_parent_indices
                ],
                branch_limits=[
                    float(
                        safe_cfg_by_parent[parent_i].get(
                            "final_branch_gap_rad",
                            0.85,
                        )
                    )
                    for parent_i in safe_parent_indices
                ],
            )

    cuda_memory: Dict[str, float] = {}
    if th.cuda.is_available():
        cuda_memory = {
            **cuda_memory_before,
            "allocated_after_mib": float(
                th.cuda.memory_allocated() / (1024 ** 2)
            ),
            "reserved_after_mib": float(
                th.cuda.memory_reserved() / (1024 ** 2)
            ),
            "max_allocated_mib": float(
                th.cuda.max_memory_allocated() / (1024 ** 2)
            ),
            "max_reserved_mib": float(
                th.cuda.max_memory_reserved() / (1024 ** 2)
            ),
        }
    meta = {
        "arm": arm,
        "n": len(poses),
        "ok": int(sum(1 for x in out if x.get("ok"))),
        "solver_return_seeds": 1,
        "seed_origin_observable": False,
        "warm_start_pose_count": int(warm_start_pose_count),
        "paired_safe_requested": int(paired_requested),
        "paired_safe_solved": int(paired_solved),
        "paired_safe_ok": int(sum(
            1
            for item in out
            if (item.get("paired_safe") or {}).get("ok")
        )),
        "paired_safe_warm_start_pose_count": int(
            safe_warm_start_pose_count
        ),
        "warm_only_probe": warm_probe,
        "final_solve_s": float(final_solve_s),
        "safe_prepare_s": float(safe_prepare_s),
        "safe_solve_s": float(safe_solve_s),
        "cuda_memory": cuda_memory,
        "elapsed_s": time.perf_counter() - t0,
        **solver_meta,
    }
    return out, meta


def _solve_arm(
    req: Dict[str, Any],
    arm: str,
    poses: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    solver, tensor_args, solver_meta = _load_solver(req, arm)
    try:
        runtime_cache = None
        runtime_cache_error = None
        if os.environ.get(
            "IK_FILTER_RUNTIME_CACHE",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}:
            try:
                runtime_cache = _IKRuntimeTensorCache(
                    req,
                    solver=solver,
                    tensor_args=tensor_args,
                    q_names=[
                        str(name)
                        for name in solver.rollout_fn.kinematics.joint_names
                    ],
                )
            except Exception as exc:
                runtime_cache_error = f"{type(exc).__name__}: {exc}"
        cache_before = runtime_cache.stats() if runtime_cache is not None else {}
        result, meta = _solve_arm_with_solver(
            req,
            arm,
            poses,
            solver=solver,
            tensor_args=tensor_args,
            solver_meta=solver_meta,
            runtime_cache=runtime_cache,
        )
        meta["runtime_cache_enabled"] = bool(runtime_cache is not None)
        if runtime_cache_error is not None:
            meta["runtime_cache_init_error"] = runtime_cache_error
        if runtime_cache is not None:
            meta["runtime_cache"] = _IKRuntimeTensorCache.stats_delta(
                runtime_cache.stats(),
                cache_before,
            )
        return result, meta
    finally:
        del solver


def _preload_forkserver_modules(req: Dict[str, Any]) -> None:
    """Load CPU-side modules once without creating a CUDA context."""
    from curobo.cuda_robot_model.util import load_robot_yaml  # noqa: F401
    from curobo.types.base import TensorDeviceType  # noqa: F401
    from curobo.types.file_path import ContentPath  # noqa: F401
    from curobo.wrap.reacher.ik_solver import (  # noqa: F401
        IKSolver,
        IKSolverConfig,
    )

    urdf_path = str(req.get("robot_urdf_path") or "")
    if urdf_path:
        _urdf_names(urdf_path)


def _set_parent_death_signal() -> None:
    """Ask Linux to terminate a prepared child with its forkserver parent."""
    try:
        import ctypes
        import signal

        parent_pid = os.getppid()
        libc = ctypes.CDLL(None, use_errno=True)
        if int(libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)) != 0:
            return
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        pass


def _prepared_solver_child(
    template_req: Dict[str, Any],
    arm: str,
    solver_signature: str,
    request_fd: int,
    result_fd: int,
    idle_timeout_s: float,
) -> None:
    """Initialize one solver, then consume exactly one future request."""
    import select
    import traceback

    _set_parent_death_signal()
    result: Optional[Dict[str, Any]] = None
    solver = None
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        solver, tensor_args, solver_meta = _load_solver(
            template_req,
            arm,
        )
        ready, _, _ = select.select(
            [request_fd],
            [],
            [],
            max(0.1, float(idle_timeout_s)),
        )
        if not ready:
            os._exit(124)
        with os.fdopen(request_fd, encoding="utf-8") as stream:
            envelope = json.load(stream)
        request_fd = -1
        requested_signature = str(
            envelope.get("solver_signature") or "legacy"
        )
        if requested_signature != str(solver_signature):
            raise RuntimeError(
                "prepared IK solver signature mismatch: "
                f"expected={solver_signature} got={requested_signature}"
            )
        req = envelope["request"]
        poses, transport_meta = _request_poses(req)
        runtime_cache = None
        runtime_cache_error = None
        if os.environ.get(
            "IK_FILTER_RUNTIME_CACHE",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}:
            try:
                runtime_cache = _IKRuntimeTensorCache(
                    req,
                    solver=solver,
                    tensor_args=tensor_args,
                    q_names=[
                        str(name)
                        for name in solver.rollout_fn.kinematics.joint_names
                    ],
                )
            except Exception as exc:
                runtime_cache_error = f"{type(exc).__name__}: {exc}"
        cache_before = runtime_cache.stats() if runtime_cache is not None else {}
        arm_results, arm_meta = _solve_arm_with_solver(
            req,
            arm,
            poses,
            solver=solver,
            tensor_args=tensor_args,
            solver_meta=solver_meta,
            runtime_cache=runtime_cache,
        )
        arm_meta["runtime_cache_enabled"] = bool(runtime_cache is not None)
        if runtime_cache_error is not None:
            arm_meta["runtime_cache_init_error"] = runtime_cache_error
        if runtime_cache is not None:
            arm_meta["runtime_cache"] = _IKRuntimeTensorCache.stats_delta(
                runtime_cache.stats(),
                cache_before,
            )
        arm_meta["pose_transport"] = transport_meta
        arm_meta["persistent_forkserver"] = True
        result = {
            "ok": True,
            "arms": {arm: arm_results},
            "meta": {
                "n": len(poses),
                arm: arm_meta,
            },
        }
    except Exception as exc:
        result = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    finally:
        if request_fd >= 0:
            try:
                os.close(request_fd)
            except OSError:
                pass
        if solver is not None:
            del solver
        if result is not None:
            try:
                with os.fdopen(result_fd, "w", encoding="utf-8") as stream:
                    json.dump(
                        _json_clean(result),
                        stream,
                        separators=(",", ":"),
                    )
                result_fd = -1
            except Exception:
                pass
        if result_fd >= 0:
            try:
                os.close(result_fd)
            except OSError:
                pass
        os._exit(0 if result is not None and result.get("ok") else 1)


def _spawn_prepared_solver(
    req: Dict[str, Any],
    arm: str,
    solver_signature: str,
) -> Dict[str, Any]:
    """Fork a child that prepares a fresh one-shot solver asynchronously."""
    request_read_fd, request_write_fd = os.pipe()
    result_read_fd, result_write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(request_write_fd)
        os.close(result_read_fd)
        idle_timeout_s = float(
            os.environ.get(
                "OFFICIAL_V2_LITE_PREPARED_IK_IDLE_TIMEOUT_S",
                "30",
            )
        )
        _prepared_solver_child(
            req,
            arm,
            solver_signature,
            request_read_fd,
            result_write_fd,
            idle_timeout_s,
        )
        os._exit(1)

    os.close(request_read_fd)
    os.close(result_write_fd)
    return {
        "pid": int(child_pid),
        "request_fd": int(request_write_fd),
        "result_fd": int(result_read_fd),
        "solver_signature": str(solver_signature),
    }


def _dispose_prepared_solver(prepared: Optional[Dict[str, Any]]) -> None:
    if not prepared:
        return
    for key in ("request_fd", "result_fd"):
        fd = int(prepared.get(key, -1))
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
            prepared[key] = -1
    pid = int(prepared.get("pid", -1))
    if pid <= 0:
        return
    try:
        waited_pid, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return
    if waited_pid == pid:
        return
    try:
        import signal

        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _prepared_solver_solve_request(
    prepared: Dict[str, Any],
    req: Dict[str, Any],
    solver_signature: str,
) -> str:
    """Dispatch one request to a matching prepared child and consume it."""
    if str(prepared.get("solver_signature")) != str(solver_signature):
        raise RuntimeError("prepared IK solver signature mismatch")
    payload = json.dumps(
        {
            "solver_signature": str(solver_signature),
            "request": req,
        },
        separators=(",", ":"),
    )
    write_error: Optional[Exception] = None
    request_fd = int(prepared.get("request_fd", -1))
    try:
        with os.fdopen(request_fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
    except Exception as exc:
        write_error = exc
    finally:
        prepared["request_fd"] = -1

    result_fd = int(prepared.get("result_fd", -1))
    try:
        with os.fdopen(result_fd, encoding="utf-8") as stream:
            result_json = stream.read()
    finally:
        prepared["result_fd"] = -1
    pid = int(prepared.get("pid", -1))
    try:
        _waited_pid, status = os.waitpid(pid, 0)
    except ChildProcessError:
        status = -1
    prepared["pid"] = -1
    if not result_json:
        detail = f" write_error={write_error}" if write_error else ""
        raise RuntimeError(
            f"prepared IK child arm result missing status={status}{detail}"
        )
    return result_json


def _forkserver_solve_request(
    req: Dict[str, Any],
    arm: str,
    poses: List[Dict[str, Any]],
    transport_meta: Dict[str, Any],
) -> str:
    """Solve in a fresh fork so every request has one-shot CUDA state."""
    import traceback

    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
            arm_results, arm_meta = _solve_arm(req, arm, poses)
            arm_meta["pose_transport"] = transport_meta
            arm_meta["persistent_forkserver"] = True
            result: Dict[str, Any] = {
                "ok": True,
                "arms": {arm: arm_results},
                "meta": {
                    "n": len(poses),
                    arm: arm_meta,
                },
            }
        except Exception as exc:
            result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        try:
            with os.fdopen(write_fd, "w", encoding="utf-8") as stream:
                json.dump(
                    _json_clean(result),
                    stream,
                    separators=(",", ":"),
                )
        finally:
            os._exit(0 if result.get("ok") else 1)

    os.close(write_fd)
    with os.fdopen(read_fd, encoding="utf-8") as stream:
        result_json = stream.read()
    _waited_pid, status = os.waitpid(child_pid, 0)
    if not result_json:
        raise RuntimeError(
            f"forkserver IK child arm={arm} produced no output "
            f"status={status}"
        )
    return result_json


def _persistent_forkserver_main(arm: str) -> None:
    """Persistent JSON transport with a fresh CUDA child per request."""
    import sys
    import traceback

    preloaded = False
    prepared: Optional[Dict[str, Any]] = None
    prepare_enabled = os.environ.get(
        "OFFICIAL_V2_LITE_PREPARED_IK",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request_id = None
            req: Optional[Dict[str, Any]] = None
            solver_signature = "legacy"
            try:
                envelope = json.loads(line)
                request_id = envelope.get("request_id")
                req = envelope["request"]
                solver_signature = str(
                    envelope.get("solver_signature") or "legacy"
                )
                if not preloaded:
                    _preload_forkserver_modules(req)
                    preloaded = True
                if bool(envelope.get("prepare_only")):
                    prepared_matches = bool(
                        prepared is not None
                        and str(prepared.get("solver_signature"))
                        == solver_signature
                    )
                    if not prepared_matches:
                        _dispose_prepared_solver(prepared)
                        prepared = _spawn_prepared_solver(
                            req,
                            arm,
                            solver_signature,
                        )
                    result_json = '{"ok":true,"prepared":true}'
                    response = (
                        '{"request_id":'
                        + json.dumps(
                            request_id,
                            separators=(",", ":"),
                        )
                        + ',"result":'
                        + result_json
                        + "}\n"
                    )
                    sys.stdout.write(response)
                    sys.stdout.flush()
                    continue
                use_prepared = bool(
                    prepared is not None
                    and str(prepared.get("solver_signature"))
                    == solver_signature
                )
                if use_prepared:
                    consumed = prepared
                    prepared = None
                    try:
                        result_json = _prepared_solver_solve_request(
                            consumed,
                            req,
                            solver_signature,
                        )
                    except Exception:
                        _dispose_prepared_solver(consumed)
                        poses, transport_meta = _request_poses(req)
                        result_json = _forkserver_solve_request(
                            req,
                            arm,
                            poses,
                            transport_meta,
                        )
                else:
                    _dispose_prepared_solver(prepared)
                    prepared = None
                    poses, transport_meta = _request_poses(req)
                    result_json = _forkserver_solve_request(
                        req,
                        arm,
                        poses,
                        transport_meta,
                    )
                result_ok = bool(json.loads(result_json).get("ok"))
                if prepare_enabled and result_ok:
                    try:
                        next_prepare = envelope.get("prepare_next")
                        next_request = req
                        next_signature = solver_signature
                        if isinstance(next_prepare, dict):
                            hinted_request = next_prepare.get("request")
                            hinted_signature = str(
                                next_prepare.get("solver_signature") or ""
                            )
                            if isinstance(hinted_request, dict) and hinted_signature:
                                next_request = hinted_request
                                next_signature = hinted_signature
                        prepared = _spawn_prepared_solver(
                            next_request,
                            arm,
                            next_signature,
                        )
                    except Exception:
                        prepared = None
            except Exception as exc:
                result_json = json.dumps(
                    _json_clean(
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=8),
                        }
                    ),
                    separators=(",", ":"),
                )
            response = (
                '{"request_id":'
                + json.dumps(request_id, separators=(",", ":"))
                + ',"result":'
                + result_json
                + "}\n"
            )
            sys.stdout.write(response)
            sys.stdout.flush()
    finally:
        _dispose_prepared_solver(prepared)


def _persistent_main(arm: str) -> None:
    import gc
    import sys
    import time
    import traceback

    def stderr_log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    solver = None
    tensor_args = None
    solver_meta: Dict[str, Any] = {}
    solver_init_s = 0.0
    runtime_cache: Optional[_IKRuntimeTensorCache] = None
    runtime_cache_init_error: Optional[str] = None
    solver_signature: Optional[str] = None
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request_id = None
            try:
                envelope = json.loads(line)
                request_id = envelope.get("request_id")
                req = envelope["request"]
                poses, transport_meta = _request_poses(req)
                requested_signature = str(
                    envelope.get("solver_signature") or "legacy"
                )
                initialized = solver is None
                exact_solver_reload = os.environ.get(
                    "OFFICIAL_V2_LITE_EXACT_SOLVER_RELOAD",
                    "1",
                ).strip().lower() not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
                reloaded = bool(
                    solver is not None
                    and (
                        requested_signature != solver_signature
                        or exact_solver_reload
                    )
                )
                if reloaded:
                    # Keep one solver resident at a time.  Retaining the CUDA
                    # process avoids repeated Python/CUDA startup while this
                    # explicit release preserves the one-shot peak-VRAM bound.
                    th.cuda.synchronize()
                    runtime_cache = None
                    del solver
                    solver = None
                    tensor_args = None
                    solver_meta = {}
                    gc.collect()
                    th.cuda.empty_cache()
                    runtime_cache_init_error = None
                if initialized or reloaded:
                    init_t0 = time.perf_counter()
                    solver, tensor_args, solver_meta = _load_solver(
                        req,
                        arm,
                        log_fn=stderr_log,
                    )
                    solver_init_s = time.perf_counter() - init_t0
                    use_runtime_cache = os.environ.get(
                        "IK_FILTER_RUNTIME_CACHE",
                        "1",
                    ).strip().lower() not in {
                        "0",
                        "false",
                        "no",
                        "off",
                    }
                    if use_runtime_cache:
                        try:
                            q_names = [
                                str(n)
                                for n in (
                                    solver.rollout_fn.kinematics.joint_names
                                )
                            ]
                            runtime_cache = _IKRuntimeTensorCache(
                                req,
                                solver=solver,
                                tensor_args=tensor_args,
                                q_names=q_names,
                            )
                        except Exception as cache_exc:
                            runtime_cache = None
                            runtime_cache_init_error = (
                                f"{type(cache_exc).__name__}: {cache_exc}"
                            )
                    solver_signature = requested_signature
                # A fresh one-shot IKSolver starts its Halton sequence at the
                # configured seed for every request. Reset it explicitly so
                # process reuse does not change candidate seeds or ranking.
                solver.reset_seed()
                if runtime_cache is not None:
                    runtime_cache.reset_request_shape_state()
                else:
                    solver._interface_last_logical_solve_shape = None
                cache_before = (
                    runtime_cache.stats()
                    if runtime_cache is not None
                    else {}
                )
                arm_results, arm_meta = _solve_arm_with_solver(
                    req,
                    arm,
                    poses,
                    solver=solver,
                    tensor_args=tensor_args,
                    solver_meta=solver_meta,
                    runtime_cache=runtime_cache,
                )
                arm_meta["persistent_worker"] = True
                arm_meta["solver_initialized_this_request"] = bool(
                    initialized
                )
                arm_meta["solver_reloaded_this_request"] = bool(reloaded)
                arm_meta["solver_init_s"] = (
                    float(solver_init_s)
                    if initialized or reloaded
                    else 0.0
                )
                arm_meta["pose_transport"] = transport_meta
                arm_meta["runtime_cache_enabled"] = bool(
                    runtime_cache is not None
                )
                if runtime_cache_init_error is not None:
                    arm_meta["runtime_cache_init_error"] = (
                        runtime_cache_init_error
                    )
                if runtime_cache is not None:
                    arm_meta["runtime_cache"] = (
                        _IKRuntimeTensorCache.stats_delta(
                            runtime_cache.stats(),
                            cache_before,
                        )
                    )
                result: Dict[str, Any] = {
                    "ok": True,
                    "arms": {arm: arm_results},
                    "meta": {
                        "n": len(poses),
                        arm: arm_meta,
                    },
                }
            except Exception as exc:
                result = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }
            response = {
                "request_id": request_id,
                "result": _json_clean(result),
            }
            print(
                json.dumps(response, separators=(",", ":")),
                flush=True,
            )
    finally:
        if solver is not None:
            del solver


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--persistent", action="store_true")
    # 单臂模式：父进程为左右臂各起一个 worker 并行求解（每臂求解本身与串行时逐位一致）
    ap.add_argument("--arm", choices=["left", "right", "both"], default="both")
    args = ap.parse_args()
    _configure_cpu_thread_pools()
    if args.persistent:
        if args.arm == "both":
            ap.error("--persistent requires --arm left or --arm right")
        use_forkserver = os.environ.get(
            "OFFICIAL_V2_LITE_FORKSERVER_IK",
            "1",
        ).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if use_forkserver:
            _persistent_forkserver_main(str(args.arm))
        else:
            _persistent_main(str(args.arm))
        return
    if not args.input or not args.output:
        ap.error("--input and --output are required without --persistent")
    _enable_faulthandler()
    with open(args.input, "r") as f:
        req = json.load(f)
    poses, transport_meta = _request_poses(req)
    arms = ("left", "right") if args.arm == "both" else (str(args.arm),)
    result: Dict[str, Any] = {"ok": True, "arms": {}, "meta": {"n": len(poses)}}
    _log_gpu_diag(
        print,
        "ik_filter_worker.start",
        extra={
            "poses": len(poses),
            "arm": args.arm,
            "batch_size": int(req.get("batch_size", 0)),
            "num_seeds": int(req.get("num_seeds", 0)),
        },
        include_nvidia=False,
    )
    try:
        for arm in arms:
            result["arms"][arm], result["meta"][arm] = _solve_arm(req, arm, poses)
            result["meta"][arm]["pose_transport"] = transport_meta
        _log_gpu_diag(
            print,
            "ik_filter_worker.done",
            extra={"poses": len(poses), "arm": args.arm},
            include_nvidia=False,
        )
    except Exception as e:
        import traceback
        _log_gpu_diag(
            print,
            "ik_filter_worker.exception",
            extra={"error": f"{type(e).__name__}: {e}", "poses": len(poses)},
        )
        result = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=8)}
    with open(args.output, "w") as f:
        json.dump(_json_clean(result), f)


if __name__ == "__main__":
    main()
