"""Pure local R1Pro kinematics and standalone cuRobo IK transport."""

from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import json
import math
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .grasp_geometry_local import mat_to_quat_xyzw, quat_to_mat_xyzw


IK_FILTER_POS_TOL_M = 0.010
IK_FILTER_ORI_TOL_DEG = 3.0
IK_FILTER_BATCH_SIZE = 64
IK_FILTER_NUM_SEEDS = 24
IK_FILTER_IK_OPT_ITERS = 80
IK_FILTER_TOP_LOG_N = 20
IK_FILTER_DEDUP_POS_DECIMALS = 4
IK_FILTER_DEDUP_QUAT_DECIMALS = 5
EEF_FIXED_TRANSFORM_WXYZ = (0.0, 0.0, -0.06, 0.0, 0.0, 1.0, 0.0)

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
_URDF_PATH = os.path.join(
    _ASSET_DIR,
    "r1pro_8dof_hf250_kinematics.urdf",
)
_CUROBO_PATH = os.path.join(
    _ASSET_DIR,
    "r1pro_description_curobo_arm_no_torso.yaml",
)
_WORKER_PATH = os.path.join(os.path.dirname(__file__), "ik_filter_worker.py")

_PERSISTENT_IK_WORKERS: Dict[Tuple[str, str], "_PersistentIKWorker"] = {}
_PERSISTENT_IK_WORKERS_LOCK = threading.Lock()
IK_POSE_SHM_VERSION = 2
IK_POSE_SHM_DTYPE = np.dtype("<f8")

# cuRobo executes its tensor work on CUDA, but importing Torch/cuBLAS and
# preparing a solver still creates host-side OpenMP/BLAS/TBB pools.  A worker
# is already isolated per arm, so letting each inherited pool size itself from
# the 128-core host only adds contention.  Keep the cap in the child
# environment; the parent interface retains its existing settings, and an
# explicitly supplied variable remains authoritative for diagnostics/tuning.
_IK_WORKER_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
)
_IK_WORKER_CPU_THREADS_ENV = "BEHAVIOR_IK_CPU_THREADS"


def _ik_worker_environment(gpu: str) -> Dict[str, str]:
    """Build a bounded environment for production IK child processes.

    ``setdefault`` deliberately preserves an operator's explicit thread
    choices.  The default applies only to the isolated worker, never to the
    simulator/interface process that owns the request.
    """

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["IK_FILTER_CUDA_VISIBLE_DEVICES"] = str(gpu)
    thread_count = environment.get(_IK_WORKER_CPU_THREADS_ENV, "1").strip()
    if not thread_count.isdigit() or int(thread_count) <= 0:
        raise ValueError(
            f"{_IK_WORKER_CPU_THREADS_ENV} must be a positive integer"
        )
    for variable in _IK_WORKER_THREAD_ENV_VARS:
        environment.setdefault(variable, thread_count)
    return environment


def _owned_ik_gpu() -> str:
    """Resolve the physical owner used to mask every IK child process."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return (
        os.environ.get("IK_FILTER_CUDA_VISIBLE_DEVICES", "").strip()
        or (visible if visible.isdigit() else "")
        or os.environ.get("BEHAVIOR_INTERFACE_PHYSICAL_GPU", "").strip()
        or os.environ.get("BEHAVIOR_EVAL_TEST_PHYSICAL_GPU", "").strip()
        or "0"
    )


@dataclass(frozen=True)
class LocalRobotState:
    arm_dof: int
    base_pos: np.ndarray
    base_quat: np.ndarray
    trunk_q: np.ndarray
    arm_left_q: np.ndarray
    arm_right_q: np.ndarray
    gripper_left_q: np.ndarray
    gripper_right_q: np.ndarray
    robot_forward: np.ndarray

    @classmethod
    def from_capture(
        cls,
        robot: Dict[str, Any],
        *,
        arm_dof: int,
    ) -> "LocalRobotState":
        base = dict(robot.get("base_pose") or {})
        yaw = float(
            base.get(
                "yaw",
                base.get("theta", base.get("heading", 0.0)),
            )
        )
        if base.get("quat") is not None:
            base_quat = np.asarray(base["quat"], dtype=np.float64).reshape(4)
        else:
            base_quat = np.array(
                [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)],
                dtype=np.float64,
            )
        if base.get("pos") is not None:
            base_pos = np.asarray(base["pos"], dtype=np.float64).reshape(3)
        else:
            base_pos = np.array(
                [
                    float(base.get("x", 0.0)),
                    float(base.get("y", 0.0)),
                    float(base.get("z", 0.0)),
                ],
                dtype=np.float64,
            )
        rotation = quat_to_mat_xyzw(base_quat)
        state = cls(
            arm_dof=int(arm_dof),
            base_pos=base_pos,
            base_quat=base_quat,
            trunk_q=_vector(robot.get("trunk_qpos"), 4),
            arm_left_q=_vector(robot.get("arm_left_qpos"), int(arm_dof)),
            arm_right_q=_vector(robot.get("arm_right_qpos"), int(arm_dof)),
            gripper_left_q=_vector(robot.get("gripper_left_qpos"), 2),
            gripper_right_q=_vector(robot.get("gripper_right_qpos"), 2),
            robot_forward=np.asarray(rotation[:, 0], dtype=np.float64),
        )
        chest_forward = link_transforms(state)["torso_link4"][:3, 0].copy()
        chest_forward[2] = 0.0
        norm = float(np.linalg.norm(chest_forward))
        if norm > 1e-9:
            object.__setattr__(
                state,
                "robot_forward",
                chest_forward / norm,
            )
        return state

    def q_by_name(self) -> Dict[str, float]:
        values = {
            "base_footprint_x_joint": 0.0,
            "base_footprint_y_joint": 0.0,
            "base_footprint_z_joint": 0.0,
            "base_footprint_rx_joint": 0.0,
            "base_footprint_ry_joint": 0.0,
            "base_footprint_rz_joint": 0.0,
        }
        values.update(
            {
                f"torso_joint{index + 1}": float(value)
                for index, value in enumerate(self.trunk_q)
            }
        )
        for arm, arm_q, gripper_q in (
            ("left", self.arm_left_q, self.gripper_left_q),
            ("right", self.arm_right_q, self.gripper_right_q),
        ):
            values.update(
                {
                    f"{arm}_arm_joint{index + 1}": (
                        0.0
                        if self.arm_dof == 8 and index == 7
                        else float(value)
                    )
                    for index, value in enumerate(arm_q)
                }
            )
            values[f"{arm}_gripper_finger_joint1"] = float(gripper_q[0])
            values[f"{arm}_gripper_finger_joint2"] = float(gripper_q[1])
        return values


def _vector(raw: Any, size: int) -> np.ndarray:
    value = np.asarray(raw if raw is not None else [], dtype=np.float64).reshape(-1)
    if len(value) < int(size):
        value = np.pad(value, (0, int(size) - len(value)))
    return value[: int(size)].copy()


def _rpy_rotation(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64).reshape(3)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    x, y, z = vector
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    one_minus = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ],
        dtype=np.float64,
    )


def _transform(
    position: Sequence[float] = (0.0, 0.0, 0.0),
    rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = (
        np.eye(3, dtype=np.float64)
        if rotation is None
        else np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    )
    result[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return result


@lru_cache(maxsize=1)
def _urdf_model() -> Tuple[str, Dict[str, List[Dict[str, Any]]], Dict[str, Tuple[float, float]]]:
    root = ET.parse(_URDF_PATH).getroot()
    links = {
        str(link.attrib["name"])
        for link in root.findall("link")
        if link.attrib.get("name")
    }
    children = set()
    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    limits: Dict[str, Tuple[float, float]] = {}
    for joint in root.findall("joint"):
        parent_node = joint.find("parent")
        child_node = joint.find("child")
        if parent_node is None or child_node is None:
            continue
        parent = str(parent_node.attrib["link"])
        child = str(child_node.attrib["link"])
        children.add(child)
        origin_node = joint.find("origin")
        xyz = [
            float(value)
            for value in (
                origin_node.attrib.get("xyz", "0 0 0").split()
                if origin_node is not None
                else ("0", "0", "0")
            )
        ]
        rpy = [
            float(value)
            for value in (
                origin_node.attrib.get("rpy", "0 0 0").split()
                if origin_node is not None
                else ("0", "0", "0")
            )
        ]
        axis_node = joint.find("axis")
        axis = [
            float(value)
            for value in (
                axis_node.attrib.get("xyz", "1 0 0").split()
                if axis_node is not None
                else ("1", "0", "0")
            )
        ]
        name = str(joint.attrib["name"])
        joint_type = str(joint.attrib.get("type", "fixed"))
        limit_node = joint.find("limit")
        if limit_node is not None:
            limits[name] = (
                float(limit_node.attrib.get("lower", "-inf")),
                float(limit_node.attrib.get("upper", "inf")),
            )
        by_parent.setdefault(parent, []).append(
            {
                "name": name,
                "type": joint_type,
                "child": child,
                "origin": _transform(xyz, _rpy_rotation(rpy)),
                "axis": np.asarray(axis, dtype=np.float64),
            }
        )
    roots = sorted(links - children)
    if not roots:
        raise RuntimeError("custom R1Pro URDF has no root link")
    return roots[0], by_parent, limits


def link_transforms(
    state: LocalRobotState,
    *,
    q_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    root, by_parent, _limits = _urdf_model()
    q_by_name = state.q_by_name()
    q_by_name.update(
        {
            str(name): float(value)
            for name, value in (q_overrides or {}).items()
        }
    )
    root_transform = _transform(
        state.base_pos,
        quat_to_mat_xyzw(state.base_quat),
    )
    transforms: Dict[str, np.ndarray] = {root: root_transform}
    stack = [root]
    while stack:
        parent = stack.pop()
        for joint in by_parent.get(parent, ()):
            motion = np.eye(4, dtype=np.float64)
            joint_type = str(joint["type"])
            value = float(q_by_name.get(str(joint["name"]), 0.0))
            if joint_type in ("revolute", "continuous"):
                motion[:3, :3] = _axis_rotation(joint["axis"], value)
            elif joint_type == "prismatic":
                motion[:3, 3] = (
                    np.asarray(joint["axis"], dtype=np.float64) * value
                )
            child = str(joint["child"])
            transforms[child] = (
                transforms[parent]
                @ np.asarray(joint["origin"], dtype=np.float64)
                @ motion
            )
            stack.append(child)
    return transforms


def eef_pose(
    state: LocalRobotState,
    arm: str,
    q_arm: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    side = str(arm)
    q = np.asarray(q_arm, dtype=np.float64).reshape(state.arm_dof)
    if state.arm_dof == 8:
        q = q.copy()
        q[7] = 0.0
    transforms = link_transforms(
        state,
        q_overrides={
            f"{side}_arm_joint{index + 1}": float(value)
            for index, value in enumerate(q)
        },
    )
    gripper = transforms[f"{side}_gripper_link"]
    eef_fixed = _transform(
        (0.0, 0.0, -0.06),
        quat_to_mat_xyzw((0.0, 1.0, 0.0, 0.0)),
    )
    result = gripper @ eef_fixed
    return result[:3, 3].copy(), mat_to_quat_xyzw(result[:3, :3])


def arm_joint_limits(
    state: LocalRobotState,
    arm: str,
) -> Tuple[np.ndarray, np.ndarray]:
    _root, _children, limits = _urdf_model()
    selected = [
        limits[f"{arm}_arm_joint{index + 1}"]
        for index in range(state.arm_dof)
    ]
    values = np.asarray(selected, dtype=np.float64)
    if state.arm_dof == 8:
        values[7] = [0.0, 0.0]
    return values[:, 0].copy(), values[:, 1].copy()


def pose_error(
    actual_pos: Sequence[float],
    actual_quat: Sequence[float],
    target_pos: Sequence[float],
    target_quat: Sequence[float],
) -> Tuple[float, float, float]:
    position_error = float(
        np.linalg.norm(
            np.asarray(actual_pos, dtype=np.float64).reshape(3)
            - np.asarray(target_pos, dtype=np.float64).reshape(3)
        )
    )
    actual_q = np.asarray(actual_quat, dtype=np.float64).reshape(4)
    target_q = np.asarray(target_quat, dtype=np.float64).reshape(4)
    actual_q /= max(float(np.linalg.norm(actual_q)), 1e-12)
    target_q /= max(float(np.linalg.norm(target_q)), 1e-12)
    dot = float(np.clip(abs(float(actual_q @ target_q)), 0.0, 1.0))
    orientation_error = float(math.degrees(2.0 * math.acos(dot)))
    actual_approach = quat_to_mat_xyzw(actual_q)[:, 2]
    target_approach = quat_to_mat_xyzw(target_q)[:, 2]
    approach_dot = float(
        np.clip(float(actual_approach @ target_approach), -1.0, 1.0)
    )
    approach_error = float(math.degrees(math.acos(approach_dot)))
    return position_error, orientation_error, approach_error


def _pose_key(pose: Dict[str, Any]) -> tuple:
    position = np.asarray(pose["eef_pos"], dtype=np.float64).reshape(3)
    quat = np.asarray(pose["quat"], dtype=np.float64).reshape(4)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    if quat[3] < 0.0:
        quat = -quat
    return tuple(
        np.round(position, IK_FILTER_DEDUP_POS_DECIMALS).tolist()
    ) + tuple(np.round(quat, IK_FILTER_DEDUP_QUAT_DECIMALS).tolist())


def _dedupe_poses(
    poses: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    unique: List[Dict[str, Any]] = []
    unique_index: List[int] = []
    index_by_key: Dict[tuple, int] = {}
    for pose in poses:
        key = _pose_key(pose)
        index = index_by_key.get(key)
        if index is None:
            index = len(unique)
            index_by_key[key] = index
            unique.append(pose)
        unique_index.append(index)
    return unique, unique_index


def _external_pose_request(
    pose: Dict[str, Any],
    *,
    arm_dof: int,
) -> Dict[str, Any]:
    item = {
        "eef_pos": [
            float(value)
            for value in np.asarray(
                pose["eef_pos"],
                dtype=np.float64,
            ).reshape(3)
        ],
        "quat": [
            float(value)
            for value in np.asarray(
                pose["quat"],
                dtype=np.float64,
            ).reshape(4)
        ],
    }
    warm_start = pose.get("ik_warm_start_q_by_arm") or {}
    warm: Dict[str, List[float]] = {}
    for arm in ("left", "right"):
        raw = warm_start.get(arm)
        if raw is None:
            continue
        q = np.asarray(raw, dtype=np.float64).reshape(-1)
        if len(q) == int(arm_dof) and np.all(np.isfinite(q)):
            q = q.copy()
            if int(arm_dof) == 8:
                q[7] = 0.0
            warm[arm] = [float(value) for value in q]
    if warm:
        item["warm_start_q_by_arm"] = warm
    safe = pose.get("ik_paired_safe_pose")
    if isinstance(safe, dict):
        item["paired_safe"] = {
            "eef_pos": [
                float(value)
                for value in np.asarray(
                    safe["eef_pos"],
                    dtype=np.float64,
                ).reshape(3)
            ],
            "quat": [
                float(value)
                for value in np.asarray(
                    safe.get("quat", pose["quat"]),
                    dtype=np.float64,
                ).reshape(4)
            ],
            "pos_tol_m": float(safe.get("pos_tol_m", 0.03)),
            "ori_tol_deg": float(safe.get("ori_tol_deg", 10.0)),
            "final_branch_gap_rad": float(
                safe.get("final_branch_gap_rad", 0.85)
            ),
        }
    return item


def _inactive_results(count: int, arm: str) -> List[Dict[str, Any]]:
    return [
        {
            "ok": False,
            "pos_err_m": float("inf"),
            "ori_err_deg": float("inf"),
            "approach_err_deg": float("inf"),
            "q_arm": None,
            "error": f"inactive_arm_not_requested:{arm}",
        }
        for _ in range(int(count))
    ]


def _persistent_worker_signature(request: Dict[str, Any]) -> str:
    """Hash only state which changes a loaded cuRobo solver."""
    static_request = {
        key: value
        for key, value in request.items()
        if key not in {
            "poses",
            "solver_policy",
            "base_link_pose",
        }
    }
    payload = json.dumps(
        static_request,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ik_request_in_shared_memory(
    request: Dict[str, Any],
) -> Tuple[Dict[str, Any], Any]:
    """Encode pose-only request data into an exact float64 shared array."""
    from multiprocessing import shared_memory

    poses = list(request.get("poses") or [])
    arm_dof = int(request.get("arm_dof", 0))
    if arm_dof <= 0:
        raise ValueError(f"invalid shared-memory arm_dof={arm_dof}")
    columns = 17 + 2 * arm_dof
    records = np.full(
        (len(poses), columns),
        np.nan,
        dtype=IK_POSE_SHM_DTYPE,
    )
    left_start = 7
    right_start = left_start + arm_dof
    safe_start = right_start + arm_dof
    for index, pose in enumerate(poses):
        records[index, 0:3] = np.asarray(
            pose["eef_pos"],
            dtype=np.float64,
        ).reshape(3)
        records[index, 3:7] = np.asarray(
            pose["quat"],
            dtype=np.float64,
        ).reshape(4)
        warm = dict(pose.get("warm_start_q_by_arm") or {})
        for arm, start in (("left", left_start), ("right", right_start)):
            raw = warm.get(arm)
            if raw is None:
                continue
            records[index, start:start + arm_dof] = np.asarray(
                raw,
                dtype=np.float64,
            ).reshape(arm_dof)
        safe = pose.get("paired_safe")
        if isinstance(safe, dict):
            records[index, safe_start:safe_start + 3] = np.asarray(
                safe["eef_pos"],
                dtype=np.float64,
            ).reshape(3)
            records[index, safe_start + 3:safe_start + 7] = np.asarray(
                safe["quat"],
                dtype=np.float64,
            ).reshape(4)
            records[index, safe_start + 7] = float(safe["pos_tol_m"])
            records[index, safe_start + 8] = float(safe["ori_tol_deg"])
            records[index, safe_start + 9] = float(
                safe["final_branch_gap_rad"]
            )

    shared = shared_memory.SharedMemory(create=True, size=max(records.nbytes, 1))
    if records.size:
        shared_records = np.ndarray(
            records.shape,
            dtype=records.dtype,
            buffer=shared.buf,
        )
        shared_records[:] = records
    transported = dict(request)
    transported["poses"] = []
    transported["pose_shared_memory"] = {
        "version": IK_POSE_SHM_VERSION,
        "name": shared.name,
        "owner_pid": os.getpid(),
        "count": len(poses),
        "columns": columns,
        "arm_dof": arm_dof,
        "dtype": IK_POSE_SHM_DTYPE.str,
    }
    return transported, shared


class _PersistentIKWorker:
    """One long-lived Python/CUDA process with at most one loaded solver."""

    def __init__(
        self,
        *,
        arm: str,
        gpu: str,
        python_path: str,
        environment: Dict[str, str],
    ) -> None:
        self.arm = str(arm)
        self.gpu = str(gpu)
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [
                python_path,
                "-u",
                _WORKER_PATH,
                "--persistent",
                "--arm",
                self.arm,
            ],
            env=environment,
            text=True,
            bufsize=1,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def alive(self) -> bool:
        return self._process.poll() is None

    def close(self) -> None:
        process = self._process
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=3.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _exchange(
        self,
        request: Dict[str, Any],
        *,
        solver_signature: str,
        timeout_s: float,
        prepare_only: bool,
        prepare_next: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if not self.alive():
                raise RuntimeError(
                    f"persistent IK worker arm={self.arm} exited "
                    f"rc={self._process.returncode}"
                )
            request_id = f"{time.time_ns()}-{id(request)}"
            envelope = {
                "request_id": request_id,
                "solver_signature": str(solver_signature),
                "request": request,
            }
            if prepare_only:
                envelope["prepare_only"] = True
            elif prepare_next is not None:
                envelope["prepare_next"] = prepare_next
            if self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("persistent IK worker pipes unavailable")
            self._process.stdin.write(
                json.dumps(envelope, separators=(",", ":")) + "\n"
            )
            self._process.stdin.flush()
            ready, _, _ = select.select(
                [self._process.stdout],
                [],
                [],
                max(0.1, float(timeout_s)),
            )
            if not ready:
                raise TimeoutError(
                    f"persistent IK worker arm={self.arm} timed out "
                    f"after {float(timeout_s):.1f}s"
                )
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"persistent IK worker arm={self.arm} closed stdout "
                    f"rc={self._process.poll()}"
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

    def request(
        self,
        request: Dict[str, Any],
        *,
        solver_signature: str,
        timeout_s: float,
        prepare_next: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._exchange(
            request,
            solver_signature=solver_signature,
            timeout_s=timeout_s,
            prepare_only=False,
            prepare_next=prepare_next,
        )

    def prepare(
        self,
        request: Dict[str, Any],
        *,
        solver_signature: str,
        timeout_s: float,
    ) -> None:
        self._exchange(
            request,
            solver_signature=solver_signature,
            timeout_s=timeout_s,
            prepare_only=True,
            prepare_next=None,
        )


def close_persistent_ik_workers() -> None:
    with _PERSISTENT_IK_WORKERS_LOCK:
        workers = list(_PERSISTENT_IK_WORKERS.values())
        _PERSISTENT_IK_WORKERS.clear()
    for worker in workers:
        worker.close()


def _persistent_ik_worker(
    *,
    arm: str,
    gpu: str,
    python_path: str,
    environment: Dict[str, str],
) -> Tuple[_PersistentIKWorker, bool]:
    key = (str(gpu), str(arm))
    with _PERSISTENT_IK_WORKERS_LOCK:
        worker = _PERSISTENT_IK_WORKERS.get(key)
        if worker is not None and worker.alive():
            return worker, True
        if worker is not None:
            worker.close()
        worker = _PersistentIKWorker(
            arm=arm,
            gpu=gpu,
            python_path=python_path,
            environment=environment,
        )
        _PERSISTENT_IK_WORKERS[key] = worker
        return worker, False


def _run_persistent_ik_arm(
    *,
    arm: str,
    gpu: str,
    python_path: str,
    environment: Dict[str, str],
    request: Dict[str, Any],
    solver_signature: str,
    timeout_s: float,
    prepare_next: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bool]:
    last_error: Optional[Exception] = None
    key = (str(gpu), str(arm))
    for attempt in range(2):
        worker, reused = _persistent_ik_worker(
            arm=arm,
            gpu=gpu,
            python_path=python_path,
            environment=environment,
        )
        try:
            return (
                worker.request(
                    request,
                    solver_signature=solver_signature,
                    timeout_s=timeout_s,
                    prepare_next=prepare_next,
                ),
                reused,
            )
        except Exception as exc:
            last_error = exc
            with _PERSISTENT_IK_WORKERS_LOCK:
                current = _PERSISTENT_IK_WORKERS.get(key)
                if current is worker:
                    _PERSISTENT_IK_WORKERS.pop(key, None)
                else:
                    current = None
            if current is not None:
                current.close()
            if attempt:
                break
    raise RuntimeError(
        f"persistent IK worker arm={arm} failed after restart: {last_error}"
    )


atexit.register(close_persistent_ik_workers)


def _build_external_ik_request(
    state: LocalRobotState,
    poses: List[Dict[str, Any]],
    *,
    pos_tol_m: float,
    ori_tol_deg: float,
    policy: str,
) -> Tuple[Dict[str, Any], str]:
    solver_config_policy = (
        policy
        if policy
        in {
            "cuda_graph_fixed64",
            "cuda_graph_fixed64_rewarm",
            "cuda_graph_split16_rewarm",
            "cuda_graph_split8_rewarm",
            "cuda_graph_warm32x6_rewarm",
        }
        else "baseline"
    )
    num_seeds = (
        6
        if policy == "cuda_graph_warm32x6_rewarm"
        else int(os.environ.get("IK_FILTER_NUM_SEEDS", IK_FILTER_NUM_SEEDS))
    )
    request = {
        "robot_cfg_path": _CUROBO_PATH,
        "robot_urdf_path": _URDF_PATH,
        "robot_usd_path": _URDF_PATH,
        "base_link_pose": {
            "name": "base_footprint_x",
            "pos": [float(value) for value in state.base_pos],
            "quat": [float(value) for value in state.base_quat],
        },
        "eef_link_names": {
            "left": "left_eef_link",
            "right": "right_eef_link",
        },
        "eef_extra_links": {
            arm: {
                "parent_link_name": f"{arm}_gripper_link",
                "link_name": f"{arm}_eef_link",
                "fixed_transform": list(EEF_FIXED_TRANSFORM_WXYZ),
                "joint_type": "FIXED",
                "joint_name": (
                    f"{arm}_gripper_link_to_{arm}_eef_link_fixed_joint"
                ),
                "source": "submission_static_r1pro_calibration",
            }
            for arm in ("left", "right")
        },
        "q_by_name": state.q_by_name(),
        "arm_dof": int(state.arm_dof),
        "pos_tol_m": float(pos_tol_m),
        "ori_tol_deg": float(ori_tol_deg),
        "batch_size": int(
            os.environ.get("IK_FILTER_BATCH_SIZE", IK_FILTER_BATCH_SIZE)
        ),
        "num_seeds": int(num_seeds),
        "ik_opt_iters": int(
            os.environ.get("IK_FILTER_IK_OPT_ITERS", IK_FILTER_IK_OPT_ITERS)
        ),
        "solver_policy": str(policy),
        "solver_config_policy": str(solver_config_policy),
        "use_usd_kinematics": False,
        "poses": [
            _external_pose_request(pose, arm_dof=state.arm_dof)
            for pose in poses
        ],
    }
    return request, str(solver_config_policy)


def prepare_external_ik(
    state: LocalRobotState,
    *,
    pos_tol_m: float,
    ori_tol_deg: float,
    active_arms: Tuple[str, ...],
    policy: str,
) -> None:
    """Asynchronously prepare the first fresh solver for a Lite plan."""
    use_persistent = os.environ.get(
        "OFFICIAL_V2_LITE_PERSISTENT_IK",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    use_forkserver = os.environ.get(
        "OFFICIAL_V2_LITE_FORKSERVER_IK",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if not use_persistent or not use_forkserver:
        return
    active = tuple(
        arm for arm in ("left", "right") if arm in set(active_arms)
    ) or ("left", "right")
    request, solver_config_policy = _build_external_ik_request(
        state,
        [],
        pos_tol_m=pos_tol_m,
        ori_tol_deg=ori_tol_deg,
        policy=policy,
    )
    persist_baseline = os.environ.get(
        "OFFICIAL_V2_LITE_PERSIST_BASELINE_IK",
        "0",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if solver_config_policy == "baseline" and not persist_baseline:
        return
    gpu = _owned_ik_gpu()
    python_path = os.environ.get("BEHAVIOR_PYTHON", sys.executable)
    timeout_s = float(os.environ.get("IK_FILTER_WORKER_TIMEOUT_S", "180"))
    environment = _ik_worker_environment(str(gpu))
    # ``gpu`` is a physical id at the interface boundary.  Mask the child to
    # that card so the worker's hard-coded local ``cuda:0`` resolves correctly.
    environment["IK_FILTER_CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    solver_signature = _persistent_worker_signature(request)

    def prepare_arm(arm: str) -> None:
        key = (str(gpu), str(arm))
        worker: Optional[_PersistentIKWorker] = None
        try:
            worker, _reused = _persistent_ik_worker(
                arm=arm,
                gpu=str(gpu),
                python_path=python_path,
                environment=environment,
            )
            worker.prepare(
                request,
                solver_signature=solver_signature,
                timeout_s=timeout_s,
            )
        except Exception:
            with _PERSISTENT_IK_WORKERS_LOCK:
                failed = _PERSISTENT_IK_WORKERS.get(key)
                if failed is worker:
                    _PERSISTENT_IK_WORKERS.pop(key, None)
                else:
                    failed = None
            if failed is not None:
                failed.close()

    for arm in active:
        threading.Thread(
            target=prepare_arm,
            args=(arm,),
            name=f"official-v2-lite-prepare-{arm}",
            daemon=True,
        ).start()


def run_external_ik(
    state: LocalRobotState,
    poses: List[Dict[str, Any]],
    *,
    pos_tol_m: float,
    ori_tol_deg: float,
    active_arms: Tuple[str, ...],
    ctx=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    started = time.perf_counter()
    active = tuple(
        arm for arm in ("left", "right") if arm in set(active_arms)
    ) or ("left", "right")
    unique, unique_index = _dedupe_poses(poses)
    policies = {
        str(pose.get("_ik_worker_policy") or "").strip()
        for pose in unique
        if str(pose.get("_ik_worker_policy") or "").strip()
    }
    if len(policies) > 1:
        raise ValueError(f"mixed IK worker policies: {sorted(policies)}")
    policy = next(iter(policies)) if policies else "baseline"
    request, solver_config_policy = _build_external_ik_request(
        state,
        unique,
        pos_tol_m=pos_tol_m,
        ori_tol_deg=ori_tol_deg,
        policy=policy,
    )
    gpu = _owned_ik_gpu()
    python_path = os.environ.get("BEHAVIOR_PYTHON", sys.executable)
    timeout_s = float(os.environ.get("IK_FILTER_WORKER_TIMEOUT_S", "180"))
    environment = _ik_worker_environment(str(gpu))
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    persist_baseline = os.environ.get(
        "OFFICIAL_V2_LITE_PERSIST_BASELINE_IK",
        "0",
    ).strip().lower() not in {"0", "false", "no", "off"}
    use_persistent = (
        (solver_config_policy != "baseline" or persist_baseline)
        and os.environ.get(
            "OFFICIAL_V2_LITE_PERSISTENT_IK",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
    )
    solver_signature = _persistent_worker_signature(request)
    prepare_next = None
    if use_persistent and solver_config_policy == "cuda_graph_split8_rewarm":
        next_request, _next_solver_config_policy = _build_external_ik_request(
            state,
            [],
            pos_tol_m=pos_tol_m,
            ori_tol_deg=ori_tol_deg,
            policy="cuda_graph_warm32x6_rewarm",
        )
        prepare_next = {
            "request": next_request,
            "solver_signature": _persistent_worker_signature(next_request),
        }
    transport_request = request
    pose_shared_memory = None
    shared_memory_enabled = (
        use_persistent
        and bool(unique)
        and os.environ.get(
            "OFFICIAL_V2_LITE_IK_SHARED_MEMORY",
            "0",
        ).strip().lower() not in {"0", "false", "no", "off"}
    )
    if shared_memory_enabled:
        transport_request, pose_shared_memory = _ik_request_in_shared_memory(
            request
        )
    arm_data: Dict[str, List[Dict[str, Any]]] = {}
    worker_meta: Dict[str, Any] = {
        "n": int(len(unique)),
        "parallel_arms": len(active) > 1,
        "active_arms": list(active),
        "persistent": bool(use_persistent),
        "pose_transport": (
            "shared_memory" if shared_memory_enabled else "json"
        ),
        "solver_policy": str(policy),
    }
    if ctx is not None:
        ctx.log(
            "  [official_v2/rgbd_lite] local cuRobo worker "
            f"gpu={gpu} arms={','.join(active)} "
            f"poses={len(unique)}/{len(poses)} "
            f"persistent={int(use_persistent)}"
        )
    try:
        if use_persistent:
            def run_arm(arm: str) -> Tuple[Dict[str, Any], bool]:
                return _run_persistent_ik_arm(
                    arm=arm,
                    gpu=str(gpu),
                    python_path=python_path,
                    environment=environment,
                    request=transport_request,
                    solver_signature=solver_signature,
                    timeout_s=timeout_s,
                    prepare_next=prepare_next,
                )

            def consume_arm_result(
                arm: str,
                result: Dict[str, Any],
                process_reused: bool,
            ) -> None:
                arm_data[arm] = list(result["arms"][arm])
                arm_meta = dict(
                    (result.get("meta") or {}).get(arm, {})
                )
                arm_meta["persistent_process_reused"] = bool(
                    process_reused
                )
                worker_meta[arm] = arm_meta

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(active),
            ) as executor:
                futures = {
                    arm: executor.submit(run_arm, arm)
                    for arm in active
                }
                for arm, future in futures.items():
                    result, process_reused = future.result()
                    consume_arm_result(
                        arm,
                        result,
                        process_reused,
                    )
        else:
            with tempfile.TemporaryDirectory(
                prefix="official_v2_rgbd_ik_",
                dir="/tmp",
            ) as temporary_dir:
                input_path = os.path.join(temporary_dir, "request.json")
                with open(input_path, "w", encoding="utf-8") as stream:
                    json.dump(request, stream)
                processes: Dict[str, Dict[str, Any]] = {}
                for arm in active:
                    output_path = os.path.join(
                        temporary_dir,
                        f"result_{arm}.json",
                    )
                    command = [
                        python_path,
                        "-u",
                        _WORKER_PATH,
                        "--input",
                        input_path,
                        "--output",
                        output_path,
                        "--arm",
                        arm,
                    ]
                    processes[arm] = {
                        "process": subprocess.Popen(
                            command,
                            env=environment,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        ),
                        "output_path": output_path,
                    }
                try:
                    for arm, info in processes.items():
                        process = info["process"]
                        stdout, stderr = process.communicate(timeout=timeout_s)
                        if process.returncode != 0:
                            raise RuntimeError(
                                f"local IK worker arm={arm} rc={process.returncode} "
                                f"stdout={stdout[-2000:]} stderr={stderr[-4000:]}"
                            )
                        with open(
                            info["output_path"],
                            encoding="utf-8",
                        ) as stream:
                            result = json.load(stream)
                        if not result.get("ok"):
                            raise RuntimeError(
                                f"local IK worker arm={arm} failed: "
                                f"{result.get('error')}"
                            )
                        arm_data[arm] = list(result["arms"][arm])
                        worker_meta[arm] = dict(
                            (result.get("meta") or {}).get(arm, {})
                        )
                finally:
                    for info in processes.values():
                        process = info["process"]
                        if process.poll() is None:
                            process.kill()
    finally:
        if pose_shared_memory is not None:
            pose_shared_memory.close()
            pose_shared_memory.unlink()

    def expand(unique_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [dict(unique_results[index]) for index in unique_index]

    left = (
        expand(arm_data["left"])
        if "left" in active
        else _inactive_results(len(poses), "left")
    )
    right = (
        expand(arm_data["right"])
        if "right" in active
        else _inactive_results(len(poses), "right")
    )
    return left, right, {
        "solver": "submission_local_curobo_gpu_worker",
        "gpu": str(gpu),
        "elapsed_s": float(time.perf_counter() - started),
        "n_requested": int(len(poses)),
        "n_unique": int(len(unique)),
        "dedupe_pos_decimals": IK_FILTER_DEDUP_POS_DECIMALS,
        "dedupe_quat_decimals": IK_FILTER_DEDUP_QUAT_DECIMALS,
        "active_arms": list(active),
        "persistent": bool(use_persistent),
        "solver_signature": solver_signature[:16],
        "worker_meta": worker_meta,
    }


def filter_poses_dual_arm_ik(
    state: LocalRobotState,
    poses: List[Dict[str, Any]],
    *,
    pos_tol_m: float = IK_FILTER_POS_TOL_M,
    ori_tol_deg: float = IK_FILTER_ORI_TOL_DEG,
    keep_single_arm_results: bool = False,
    active_arms: Tuple[str, ...] = ("left", "right"),
    ctx=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    started = time.perf_counter()
    active = tuple(
        arm for arm in ("left", "right") if arm in set(active_arms)
    ) or ("left", "right")
    try:
        left, right, worker = run_external_ik(
            state,
            poses,
            pos_tol_m=pos_tol_m,
            ori_tol_deg=ori_tol_deg,
            active_arms=active,
            ctx=ctx,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return [], {
            "n_input": int(len(poses)),
            "n_left_ok": 0,
            "n_right_ok": 0,
            "n_both_ok": 0,
            "n_ranked": 0,
            "n_local_fk_validated": 0,
            "pos_tol_m": float(pos_tol_m),
            "ori_tol_deg": float(ori_tol_deg),
            "active_arms": list(active),
            "left": {"error": error, "elapsed_s": 0.0},
            "right": {"error": error, "elapsed_s": 0.0},
            "top20": [],
            "elapsed_s": float(time.perf_counter() - started),
            "error": error,
        }
    ranked: List[Dict[str, Any]] = []
    for pose, left_info, right_info in zip(poses, left, right):
        finite = {
            "left": (
                math.isfinite(float(left_info.get("pos_err_m", float("inf"))))
                and math.isfinite(
                    float(left_info.get("ori_err_deg", float("inf")))
                )
            ),
            "right": (
                math.isfinite(float(right_info.get("pos_err_m", float("inf"))))
                and math.isfinite(
                    float(right_info.get("ori_err_deg", float("inf")))
                )
            ),
        }
        errors = {
            "left": (
                float(left_info.get("pos_err_m", float("inf"))) * 1000.0
                + float(left_info.get("ori_err_deg", float("inf")))
            ),
            "right": (
                float(right_info.get("pos_err_m", float("inf"))) * 1000.0
                + float(right_info.get("ori_err_deg", float("inf")))
            ),
        }
        if keep_single_arm_results:
            if not any(finite[arm] for arm in active):
                continue
            all_error = sum(
                errors[arm] if finite[arm] else 1.0e9
                for arm in active
            )
        else:
            if not all(finite[arm] for arm in active):
                continue
            all_error = sum(errors[arm] for arm in active)
        item = dict(pose)
        item["ik_filter"] = {
            "left": dict(left_info),
            "right": dict(right_info),
        }
        item["left_ik_ok"] = bool(left_info.get("ok"))
        item["right_ik_ok"] = bool(right_info.get("ok"))
        item["dual_ik_ok"] = bool(
            left_info.get("ok") and right_info.get("ok")
        )
        item["ik_allerr"] = float(all_error)
        item["ik_allerr_formula"] = " + ".join(
            f"{arm}_pos_mm + {arm}_ori_deg"
            for arm in active
        )
        ranked.append(item)
    ranked.sort(
        key=lambda pose: (
            float(pose.get("ik_allerr", float("inf"))),
            int(pose.get("fast_overlap", 0)),
        )
    )
    top20 = []
    for rank, pose in enumerate(ranked[:IK_FILTER_TOP_LOG_N], start=1):
        left_info = pose["ik_filter"]["left"]
        right_info = pose["ik_filter"]["right"]
        top20.append(
            {
                "rank": rank,
                "allerr": round(float(pose["ik_allerr"]), 3),
                "left_pos_mm": round(
                    float(left_info.get("pos_err_m", float("inf"))) * 1000.0,
                    2,
                ),
                "left_ori_deg": round(
                    float(left_info.get("ori_err_deg", float("inf"))),
                    2,
                ),
                "right_pos_mm": round(
                    float(right_info.get("pos_err_m", float("inf"))) * 1000.0,
                    2,
                ),
                "right_ori_deg": round(
                    float(right_info.get("ori_err_deg", float("inf"))),
                    2,
                ),
                "both_strict": bool(pose.get("dual_ik_ok")),
                "fast_overlap": int(pose.get("fast_overlap", 0)),
            }
        )
    worker_meta = worker.get("worker_meta") or {}
    return ranked, {
        "n_input": int(len(poses)),
        "n_left_ok": int(sum(bool(pose.get("left_ik_ok")) for pose in ranked)),
        "n_right_ok": int(
            sum(bool(pose.get("right_ik_ok")) for pose in ranked)
        ),
        "n_both_ok": int(sum(bool(pose.get("dual_ik_ok")) for pose in ranked)),
        "n_ranked": int(len(ranked)),
        "n_worker_ranked": int(len(ranked)),
        "n_local_fk_validated": 0,
        "pos_tol_m": float(pos_tol_m),
        "ori_tol_deg": float(ori_tol_deg),
        "keep_single_arm_results": bool(keep_single_arm_results),
        "active_arms": list(active),
        "left": dict(worker_meta.get("left") or {}),
        "right": dict(worker_meta.get("right") or {}),
        "worker": worker,
        "top20": top20,
        "elapsed_s": float(time.perf_counter() - started),
    }
