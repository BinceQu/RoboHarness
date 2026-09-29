"""Test-only parallel IK policy lanes for RGBD Lite benchmarks.

The public eval-test runtime keeps one fresh CUDA child per IK request.  This
installer is deliberately process-local: each lane retains the official fresh
CUDA child and solver reset semantics while allowing independent requests to
run concurrently. Planner inputs, solve policies, candidate ordering, and
result handling remain unchanged.
"""

from __future__ import annotations

import atexit
import copy
import concurrent.futures
from contextlib import contextmanager
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
import types
from typing import Any, Dict, Optional, Tuple

import numpy as np

from behavior_interface_eval_test.tool.official_v2 import grasp_kinematics_local
from behavior_interface_eval_test.tool.official_v2 import rgbd_grasp_lite
from behavior_interface_eval_test.tool.official_v2 import rgbd_grasp_planner
from behavior_interface_eval_test.tool.official_v2.mesh_occupancy import (
    MeshOccupancyEvaluator,
)


TEST_MODE_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_MODE"
POLICY_POOL_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_POLICY_POOL"
SKIP_REACHABLE_SE3_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_SKIP_REACHABLE_SE3"
)
SHARED_MEMORY_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_SHARED_MEMORY"
FORKSERVER_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_FORKSERVER"
EXACT_SOLVER_RELOAD_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_EXACT_SOLVER_RELOAD"
)
PHYSICAL_BATCH_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_PHYSICAL_BATCH"
SOLVER_RESET_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_SOLVER_RESET"
IK_TRACE_DIR_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_IK_TRACE_DIR"
RENDER_PROCESSES_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_RENDER_PROCESSES"
COMPONENT_CACHE_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_COMPONENT_CACHE"
POLICY_SHARD_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_POLICY_SHARD"
LANE_AFFINITY_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_LANE_AFFINITY"
PRE_IK_DEDUP_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_PRE_IK_DEDUP"
PRE_IK_COUNT_DEDUP_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_PRE_IK_COUNT_DEDUP"
)
PREPARED_SIGNATURE_POOL_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_PREPARED_SIGNATURE_POOL"
)
CAMERA_FACE_ORIENTATION_CACHE_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_CAMERA_FACE_ORIENTATION_CACHE"
)
NORMAL_ALIGNMENT_CACHE_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_NORMAL_ALIGNMENT_CACHE"
)
POSE_GENERATION_CACHE_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_POSE_GENERATION_CACHE"
)
WORKER_CPU_THREADS_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_WORKER_CPU_THREADS"
)
SIGNATURE_STICKY_LANES_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_SIGNATURE_STICKY_LANES"
)
CANDIDATE_COMPRESSION_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_COMPRESSION"
)
CANDIDATE_COMPRESSION_SCOPE_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_COMPRESSION_SCOPE"
)
CANDIDATE_COMPRESSION_STAGES_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_COMPRESSION_STAGES"
)
CANDIDATE_POST_LIMIT_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_POST_LIMIT"
)
CANDIDATE_STAGE_LIMIT_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_STAGE_LIMIT"
)
_TEST_WORKER_PATH = str(
    Path(__file__).with_name("rgbd_lite_ik_batch_worker.py")
)


class _PolicyWorkerLane:
    def __init__(
        self,
        worker: grasp_kinematics_local._PersistentIKWorker,
        affinity: Optional[int] = None,
        solver_signature: Optional[str] = None,
    ) -> None:
        self.worker = worker
        self.busy = True
        self.completed_requests = 0
        self.affinity = affinity
        self.solver_signature = solver_signature


_PolicyPoolKey = Tuple[str, ...]
_POLICY_WORKERS: Dict[_PolicyPoolKey, list[_PolicyWorkerLane]] = {}
_POLICY_WORKERS_LOCK = threading.Lock()
_POLICY_WORKERS_CONDITION = threading.Condition(_POLICY_WORKERS_LOCK)
_IK_TRACE_LOCK = threading.Lock()
_IK_TRACE_SEQUENCE = 0
_TEST_RENDER_EXECUTOR: Optional[concurrent.futures.ProcessPoolExecutor] = None
_TEST_RENDER_EXECUTOR_LOCK = threading.Lock()
_COMPONENT_CACHE: Dict[str, Dict[str, Any]] = {}
_COMPONENT_CACHE_LOCK = threading.Lock()
_REQUEST_AFFINITY: Dict[int, Tuple[Dict[str, Any], int]] = {}
_REQUEST_AFFINITY_LOCK = threading.Lock()
_PLANNER_CASE_LOCAL = threading.local()
_PRE_IK_FLIGHTS: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
_PRE_IK_FLIGHTS_LOCK = threading.Lock()
_CAMERA_FACE_ORIENTATION_RESULTS: Dict[
    Tuple[bytes, bytes],
    Tuple[np.ndarray, Dict[str, Any], np.ndarray],
] = {}
_CAMERA_FACE_ORIENTATION_RESULTS_LOCK = threading.Lock()
_NORMAL_ALIGNMENT_RESULTS: Dict[
    Tuple[bytes, bytes],
    Tuple[np.ndarray, float],
] = {}
_NORMAL_ALIGNMENT_RESULTS_LOCK = threading.Lock()
_RGBD_FILTER_POSE_TEMPLATES: Dict[
    Tuple[bytes, int],
    Tuple[Dict[str, Any], ...],
] = {}
_ORIENTATION_DELTA_SPECS: Dict[
    Tuple[bytes, bytes],
    Tuple[Tuple[float, float, float, np.ndarray], ...],
] = {}
_POSE_GENERATION_CACHE_LOCK = threading.Lock()
_INSTALLED = False
_ORIGINAL_RENDER_LOCAL_THREE_VIEWS_LITE = (
    rgbd_grasp_lite._render_local_three_views_lite
)
_ORIGINAL_LITE_RENDER_EXECUTOR = rgbd_grasp_lite._lite_render_executor
_ORIGINAL_RUN_PERSISTENT_IK_ARM = (
    grasp_kinematics_local._run_persistent_ik_arm
)
_ORIGINAL_PREPARE_EXTERNAL_IK = rgbd_grasp_lite.prepare_external_ik
_ORIGINAL_BUILD_EXTERNAL_IK_REQUEST = (
    grasp_kinematics_local._build_external_ik_request
)
_ORIGINAL_IK_REQUEST_IN_SHARED_MEMORY = (
    grasp_kinematics_local._ik_request_in_shared_memory
)
_ORIGINAL_IK_WORKER_PATH = grasp_kinematics_local._WORKER_PATH
_ORIGINAL_MESH_OCCUPANCY_BUILD_DESCRIPTOR = (
    MeshOccupancyEvaluator.__dict__["build"]
)
_ORIGINAL_MESH_OCCUPANCY_BUILD = MeshOccupancyEvaluator.build
_ORIGINAL_WATERTIGHT_COMPONENT_BOUNDS = (
    rgbd_grasp_lite._lite_watertight_component_bounds
)
_ORIGINAL_CAMERA_FACE_LITE = (
    rgbd_grasp_lite._apply_camera_face_preserving_anchor_lite
)
_ORIGINAL_ATTACH_NORMAL_ALIGNMENT = (
    rgbd_grasp_planner.attach_normal_alignment
)
_ORIGINAL_POSE_GRIPPER_VECTOR_WORLD = (
    rgbd_grasp_planner.pose_gripper_vector_world
)
_ORIGINAL_GENERATE_RGBD_FILTER_POSES = (
    rgbd_grasp_planner.generate_rgbd_filter_poses
)
_ORIGINAL_GENERATE_LOCAL_ORIENTATION_REFINEMENTS = (
    rgbd_grasp_planner.generate_local_orientation_refinements
)
_ORIGINAL_MAT_TO_QUAT_XYZW = rgbd_grasp_planner._mat_to_quat_xyzw
_ORIGINAL_COUNTS_FOR_POSES = rgbd_grasp_planner.DenseSceneOccupancy.counts_for_poses
_ORIGINAL_FILTER_SAFE_FINAL = (
    rgbd_grasp_planner.filter_poses_safe_final_dual_arm_ik
)
_ORIGINAL_PRODUCTION_RANK_DEDUPE = (
    rgbd_grasp_planner.rank_dedupe_top_ik_input
)
_ORIGINAL_PRODUCTION_RANK_MICRO = (
    rgbd_grasp_planner.rank_seed_balanced_micro_ik_input
)
_ORIGINAL_LITE_RANK_DEDUPE = (
    rgbd_grasp_lite.rank_dedupe_top_ik_input_lite
)
_ORIGINAL_LITE_COMPRESS_INITIAL_STAGE = bool(
    rgbd_grasp_lite.LITE_COMPRESS_INITIAL_STAGE
)
_ORIGINAL_LITE_COMPRESS_REFINEMENT_STAGES = bool(
    rgbd_grasp_lite.LITE_COMPRESS_REFINEMENT_STAGES
)
_ORIGINAL_LITE_POST_PARETO_IK_LIMIT = int(
    rgbd_grasp_lite.LITE_POST_PARETO_IK_LIMIT
)
_ORIGINAL_LITE_REACHABLE_STAGE_IK_LIMIT = int(
    rgbd_grasp_lite.LITE_REACHABLE_STAGE_IK_LIMIT
)


class _OriginalRankingProxy:
    """Expose frozen production helpers to process-local ranker clones."""

    rank_dedupe_top_ik_input = staticmethod(_ORIGINAL_PRODUCTION_RANK_DEDUPE)
    rank_seed_balanced_micro_ik_input = staticmethod(
        _ORIGINAL_PRODUCTION_RANK_MICRO
    )

    def __getattr__(self, name: str):
        return getattr(rgbd_grasp_planner, name)


_ORIGINAL_RANKING_PROXY = _OriginalRankingProxy()


def _clone_ranker_with_frozen_production(function):
    globals_copy = dict(function.__globals__)
    globals_copy["production"] = _ORIGINAL_RANKING_PROXY
    cloned = types.FunctionType(
        function.__code__,
        globals_copy,
        name=f"{function.__name__}_test_only",
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    cloned.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    cloned.__annotations__ = dict(function.__annotations__)
    return cloned


_TEST_INITIAL_RANKER = _clone_ranker_with_frozen_production(
    rgbd_grasp_lite.rank_dedupe_top_ik_input_lite
)
_TEST_MICRO_RANKER = _clone_ranker_with_frozen_production(
    rgbd_grasp_lite.rank_seed_balanced_micro_ik_input_lite
)
_TEST_LINEAGE_RANKER = _clone_ranker_with_frozen_production(
    rgbd_grasp_lite.rank_lineage_balanced_stage_ik_input_lite
)
_CANDIDATE_COMPRESSION_STAGES: set[str] = set()


def _candidate_stage_name(poses) -> str:
    rows = list(poses or [])
    if any(bool(pose.get("translation_refined", False)) for pose in rows):
        return "translation"
    generations = {
        int(pose.get("reachable_se3_generation", 0))
        for pose in rows
    }
    if 3 in generations:
        return "closure2"
    if 2 in generations:
        return "closure"
    if 1 in generations:
        return "interpolation"
    return "initial"


def _selective_lineage_ranker(*args, **kwargs):
    poses = args[0] if args else kwargs.get("poses")
    if _candidate_stage_name(poses) in _CANDIDATE_COMPRESSION_STAGES:
        return _TEST_LINEAGE_RANKER(*args, **kwargs)
    return _ORIGINAL_PRODUCTION_RANK_DEDUPE(*args, **kwargs)


def _enabled(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _configure_candidate_compression(
    enabled: bool,
    *,
    scope: str = "all",
    stages: Optional[set[str]] = None,
    post_limit: int = 64,
    stage_limit: int = 64,
) -> None:
    """Switch only this benchmark process to representative IK inputs."""
    global _CANDIDATE_COMPRESSION_STAGES
    normalized_scope = str(scope).strip().lower()
    if normalized_scope not in {"all", "initial", "refinement"}:
        raise ValueError(
            f"{CANDIDATE_COMPRESSION_SCOPE_ENV} must be all, initial, "
            "or refinement"
        )
    valid_stages = {
        "initial",
        "micro",
        "interpolation",
        "closure",
        "closure2",
        "translation",
    }
    if stages is None:
        if normalized_scope == "all":
            selected_stages = set(valid_stages)
        elif normalized_scope == "initial":
            selected_stages = {"initial"}
        else:
            selected_stages = valid_stages - {"initial"}
    else:
        selected_stages = {str(stage).strip().lower() for stage in stages}
    unknown_stages = selected_stages - valid_stages
    if unknown_stages:
        raise ValueError(
            f"{CANDIDATE_COMPRESSION_STAGES_ENV} has unknown stages: "
            f"{sorted(unknown_stages)}"
        )
    if not 1 <= int(post_limit) <= 120:
        raise ValueError(f"{CANDIDATE_POST_LIMIT_ENV} must be in [1, 120]")
    if not 1 <= int(stage_limit) <= 120:
        raise ValueError(f"{CANDIDATE_STAGE_LIMIT_ENV} must be in [1, 120]")
    if enabled:
        initial_enabled = "initial" in selected_stages
        refinement_enabled = bool(selected_stages - {"initial"})
        _CANDIDATE_COMPRESSION_STAGES = set(selected_stages)
        _TEST_MICRO_RANKER.__globals__["LITE_POST_PARETO_IK_LIMIT"] = int(
            post_limit
        )
        _TEST_LINEAGE_RANKER.__globals__[
            "LITE_REACHABLE_STAGE_IK_LIMIT"
        ] = int(stage_limit)
        rgbd_grasp_lite.LITE_POST_PARETO_IK_LIMIT = int(post_limit)
        rgbd_grasp_lite.LITE_REACHABLE_STAGE_IK_LIMIT = int(stage_limit)
        # The Lite clone dispatches its first rank call through the local
        # symbol only when this flag is true. A frozen production ranker here
        # therefore gives refinement-only experiments an unmodified initial
        # candidate set.
        rgbd_grasp_lite.LITE_COMPRESS_INITIAL_STAGE = True
        rgbd_grasp_lite.LITE_COMPRESS_REFINEMENT_STAGES = refinement_enabled
        rgbd_grasp_lite.rank_dedupe_top_ik_input_lite = (
            _TEST_INITIAL_RANKER
            if initial_enabled
            else _ORIGINAL_PRODUCTION_RANK_DEDUPE
        )
        rgbd_grasp_planner.rank_dedupe_top_ik_input = (
            _selective_lineage_ranker
            if refinement_enabled
            else _ORIGINAL_PRODUCTION_RANK_DEDUPE
        )
        rgbd_grasp_planner.rank_seed_balanced_micro_ik_input = (
            _TEST_MICRO_RANKER
            if "micro" in selected_stages
            else _ORIGINAL_PRODUCTION_RANK_MICRO
        )
        return

    _CANDIDATE_COMPRESSION_STAGES = set()
    rgbd_grasp_lite.LITE_COMPRESS_INITIAL_STAGE = (
        _ORIGINAL_LITE_COMPRESS_INITIAL_STAGE
    )
    rgbd_grasp_lite.LITE_COMPRESS_REFINEMENT_STAGES = (
        _ORIGINAL_LITE_COMPRESS_REFINEMENT_STAGES
    )
    rgbd_grasp_lite.LITE_POST_PARETO_IK_LIMIT = (
        _ORIGINAL_LITE_POST_PARETO_IK_LIMIT
    )
    rgbd_grasp_lite.LITE_REACHABLE_STAGE_IK_LIMIT = (
        _ORIGINAL_LITE_REACHABLE_STAGE_IK_LIMIT
    )
    rgbd_grasp_lite.rank_dedupe_top_ik_input_lite = (
        _ORIGINAL_LITE_RANK_DEDUPE
    )
    rgbd_grasp_planner.rank_dedupe_top_ik_input = (
        _ORIGINAL_PRODUCTION_RANK_DEDUPE
    )
    rgbd_grasp_planner.rank_seed_balanced_micro_ik_input = (
        _ORIGINAL_PRODUCTION_RANK_MICRO
    )


@contextmanager
def planner_case_scope(case_key: str):
    """Identify the two forced-arm runs that share one deterministic prefix."""
    previous = getattr(_PLANNER_CASE_LOCAL, "state", None)
    _PLANNER_CASE_LOCAL.state = {
        "case_key": str(case_key),
        "pre_ik": True,
        "ordinals": {},
    }
    try:
        yield
    finally:
        _PLANNER_CASE_LOCAL.state = previous


def _pre_ik_call_key(operation: str) -> Optional[Tuple[str, str, int]]:
    state = getattr(_PLANNER_CASE_LOCAL, "state", None)
    if not state or not state.get("pre_ik"):
        return None
    ordinals = state["ordinals"]
    ordinal = int(ordinals.get(operation, 0))
    ordinals[operation] = ordinal + 1
    return (str(state["case_key"]), str(operation), ordinal)


def _singleflight_copy(
    key: Tuple[str, str, int],
    compute,
):
    with _PRE_IK_FLIGHTS_LOCK:
        flight = _PRE_IK_FLIGHTS.get(key)
        if flight is None:
            flight = {
                "event": threading.Event(),
                "owner": True,
                "consumers": 0,
            }
            _PRE_IK_FLIGHTS[key] = flight
            owner = True
        else:
            owner = False
    if owner:
        try:
            flight["value"] = copy.deepcopy(compute())
        except BaseException as error:
            flight["error"] = error
        finally:
            flight["event"].set()
    else:
        flight["event"].wait()
    error = flight.get("error")
    if error is not None:
        raise error
    value = copy.deepcopy(flight["value"])
    with _PRE_IK_FLIGHTS_LOCK:
        flight["consumers"] = int(flight["consumers"]) + 1
        if int(flight["consumers"]) >= 2:
            _PRE_IK_FLIGHTS.pop(key, None)
    return value


def _cached_camera_face_lite(poses, *, world, ctx=None):
    key = _pre_ik_call_key("camera_face")
    if key is None or ctx is not None:
        return _ORIGINAL_CAMERA_FACE_LITE(poses, world=world, ctx=ctx)

    def compute():
        copied_poses = copy.deepcopy(poses)
        audit = _ORIGINAL_CAMERA_FACE_LITE(
            copied_poses,
            world=world,
            ctx=None,
        )
        return copied_poses, audit

    copied_poses, audit = _singleflight_copy(key, compute)
    if len(copied_poses) != len(poses):
        raise RuntimeError("pre-IK camera-face cache pose count changed")
    for destination, source in zip(poses, copied_poses):
        destination.clear()
        destination.update(source)
    return audit


def _copy_camera_face_audit(audit: Dict[str, Any]) -> Dict[str, Any]:
    """Copy the frozen audit schema without recursive generic dispatch."""
    result = dict(audit)
    for key in (
        "robot_forward_xy",
        "camera_normal_xy_before",
        "camera_normal_xy_after",
    ):
        value = result.get(key)
        if isinstance(value, list):
            result[key] = value.copy()
    return result


def _orientation_cached_camera_face_lite(poses, *, world, ctx=None):
    """Reuse exact camera-face results across stages and concurrent cases."""
    from behavior_interface_eval_test.tool.official_v2.grasp_geometry_local import (
        ensure_camera_face_forward,
    )

    forward = np.asarray(
        getattr(world, "robot_forward", [1.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(3)
    forward_key = np.ascontiguousarray(forward).tobytes()
    source_by_key: Dict[bytes, np.ndarray] = {}
    pose_keys = []
    for pose in poses:
        source_quat = np.asarray(pose["quat"], dtype=np.float64).reshape(4)
        key = np.ascontiguousarray(source_quat).tobytes()
        pose_keys.append(key)
        if key not in source_by_key:
            source_by_key[key] = source_quat.copy()

    orientation_cache: Dict[
        bytes,
        Tuple[np.ndarray, Dict[str, Any], np.ndarray],
    ] = {}
    # One lock per pose batch avoids thousands of contended lock round-trips.
    # It also serializes only the tiny cache-miss orientation calculations.
    with _CAMERA_FACE_ORIENTATION_RESULTS_LOCK:
        for key, source_quat in source_by_key.items():
            global_key = (forward_key, key)
            cached = _CAMERA_FACE_ORIENTATION_RESULTS.get(global_key)
            if cached is None:
                quat, audit = ensure_camera_face_forward(
                    source_quat,
                    forward=forward,
                )
                rotation = rgbd_grasp_planner._quat_to_mat_xyzw(quat)
                cached = (
                    np.asarray(quat, dtype=np.float64).copy(),
                    _copy_camera_face_audit(audit),
                    np.asarray(rotation, dtype=np.float64).copy(),
                )
                _CAMERA_FACE_ORIENTATION_RESULTS[global_key] = cached
            orientation_cache[key] = cached

    cached_rows = [orientation_cache[key] for key in pose_keys]
    if cached_rows:
        rotations = np.stack([row[2] for row in cached_rows], axis=0)
        anchors = np.stack(
            [np.asarray(pose["anchor"], dtype=np.float64) for pose in poses],
            axis=0,
        )
        anchor_locals = np.stack(
            [
                np.asarray(pose["anchor_local"], dtype=np.float64)
                for pose in poses
            ],
            axis=0,
        )
        eef_positions = anchors - np.matmul(
            rotations,
            anchor_locals[..., None],
        )[..., 0]
    else:
        rotations = np.empty((0, 3, 3), dtype=np.float64)
        eef_positions = np.empty((0, 3), dtype=np.float64)

    flipped = 0
    skipped = 0
    for index, (pose, cached) in enumerate(zip(poses, cached_rows)):
        cached_quat, cached_audit, cached_rotation = cached
        quat = cached_quat.copy()
        audit = _copy_camera_face_audit(cached_audit)
        rotation = rotations[index].copy()
        pose["quat"] = quat
        pose["R"] = rotation
        pose["eef_pos"] = eef_positions[index].copy()
        pose["camera_face"] = audit
        if audit.get("flipped"):
            pose.pop("ik_warm_start_q_by_arm", None)
        flipped += int(bool(audit.get("flipped")))
        skipped += int(bool(audit.get("skipped")))
    result = {
        "pose_count": len(poses),
        "flipped": flipped,
        "skipped": skipped,
    }
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] camera-face poses={len(poses)} "
            f"flipped={flipped} skipped={skipped}"
        )
    return result


def _replay_count_cache(occupancy, poses, offsets_eef, counts, metadata) -> None:
    device = str(metadata.get("device") or "")
    if device == "numpy":
        backend = "numpy"
        dtype = np.float64
    elif device == "none":
        return
    else:
        backend = device
        dtype = np.float32
    offsets = np.asarray(offsets_eef, dtype=np.float32).reshape(-1, 3)
    offset_key = occupancy._offset_query_key(offsets)
    count_cache = occupancy._count_cache.setdefault((backend, offset_key), {})
    for pose, count in zip(poses, np.asarray(counts).reshape(-1)):
        pose_key = occupancy._pose_query_key(pose, dtype=dtype)
        count_cache[pose_key] = int(count)


def _cached_counts_for_poses(
    self,
    poses,
    offsets_eef,
    *,
    batch_size=32,
    prefer_cuda=True,
):
    key = _pre_ik_call_key("counts_for_poses")
    if key is None:
        return _ORIGINAL_COUNTS_FOR_POSES(
            self,
            poses,
            offsets_eef,
            batch_size=batch_size,
            prefer_cuda=prefer_cuda,
        )

    result = _singleflight_copy(
        key,
        lambda: _ORIGINAL_COUNTS_FOR_POSES(
            self,
            poses,
            offsets_eef,
            batch_size=batch_size,
            prefer_cuda=prefer_cuda,
        ),
    )
    counts, metadata = result
    _replay_count_cache(self, poses, offsets_eef, counts, metadata)
    return counts, metadata


def _mark_post_prefix_filter(*args, **kwargs):
    state = getattr(_PLANNER_CASE_LOCAL, "state", None)
    if state is not None:
        state["pre_ik"] = False
    return _ORIGINAL_FILTER_SAFE_FINAL(*args, **kwargs)


def clear_test_pre_ik_cache() -> None:
    with _PRE_IK_FLIGHTS_LOCK:
        _PRE_IK_FLIGHTS.clear()


def clear_test_camera_face_cache() -> None:
    with _CAMERA_FACE_ORIENTATION_RESULTS_LOCK:
        _CAMERA_FACE_ORIENTATION_RESULTS.clear()


def _cached_normal_alignment(poses, outward_normal_world):
    """Reuse the official scalar result for duplicate pose orientations."""
    if outward_normal_world is None:
        return _ORIGINAL_ATTACH_NORMAL_ALIGNMENT(poses, outward_normal_world)

    outward = np.asarray(outward_normal_world, dtype=np.float64).reshape(3)
    outward /= max(float(np.linalg.norm(outward)), 1e-12)
    outward_key = np.ascontiguousarray(outward).tobytes()
    rotations_by_key: Dict[bytes, np.ndarray] = {}
    pose_keys = []
    for pose in poses:
        rotation = np.asarray(pose["R"], dtype=np.float64).reshape(3, 3)
        rotation_key = np.ascontiguousarray(rotation).tobytes()
        pose_keys.append(rotation_key)
        if rotation_key not in rotations_by_key:
            rotations_by_key[rotation_key] = rotation.copy()

    local_results: Dict[bytes, Tuple[np.ndarray, float]] = {}
    with _NORMAL_ALIGNMENT_RESULTS_LOCK:
        for rotation_key, rotation in rotations_by_key.items():
            global_key = (outward_key, rotation_key)
            cached = _NORMAL_ALIGNMENT_RESULTS.get(global_key)
            if cached is None:
                gripper_vector = _ORIGINAL_POSE_GRIPPER_VECTOR_WORLD(
                    {"R": rotation}
                )
                angle = float(
                    math.degrees(
                        math.acos(
                            float(
                                np.clip(
                                    outward @ gripper_vector,
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                )
                cached = (gripper_vector.copy(), angle)
                _NORMAL_ALIGNMENT_RESULTS[global_key] = cached
            local_results[rotation_key] = cached

    angles = []
    for pose, rotation_key in zip(poses, pose_keys):
        gripper_vector, angle = local_results[rotation_key]
        pose["gripper_vector_world"] = gripper_vector.copy()
        pose["normal_alignment_deg"] = angle
        angles.append(angle)
    values = np.asarray(angles, dtype=np.float64)
    return {
        "enabled": True,
        "pose_count": int(len(poses)),
        "outward_normal_world": outward.tolist(),
        "angle_min_deg": float(values.min()) if len(values) else None,
        "angle_median_deg": float(np.median(values)) if len(values) else None,
        "angle_max_deg": float(values.max()) if len(values) else None,
    }


def clear_test_normal_alignment_cache() -> None:
    with _NORMAL_ALIGNMENT_RESULTS_LOCK:
        _NORMAL_ALIGNMENT_RESULTS.clear()


def _cached_rgbd_filter_pose_templates(
    anchors,
    *,
    axial_z_m=rgbd_grasp_planner.AXIAL_Z_M,
    n_roll=rgbd_grasp_planner.N_ROLL,
):
    """Instantiate exact official pose templates at each requested anchor."""
    axial_values = np.asarray(axial_z_m, dtype=np.float64).reshape(-1)
    cache_key = (np.ascontiguousarray(axial_values).tobytes(), int(n_roll))
    with _POSE_GENERATION_CACHE_LOCK:
        templates = _RGBD_FILTER_POSE_TEMPLATES.get(cache_key)
        if templates is None:
            generated = _ORIGINAL_GENERATE_RGBD_FILTER_POSES(
                np.zeros((1, 3), dtype=np.float64),
                axial_z_m=axial_values,
                n_roll=int(n_roll),
            )
            templates = tuple(generated)
            _RGBD_FILTER_POSE_TEMPLATES[cache_key] = templates

    poses = []
    for anchor_index, anchor in enumerate(
        np.asarray(anchors, dtype=np.float64).reshape(-1, 3)
    ):
        for template in templates:
            pose = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in template.items()
            }
            rotation = pose["R"]
            anchor_local = pose["anchor_local"]
            pose["pi"] = int(anchor_index)
            pose["anchor"] = anchor.copy()
            pose["eef_pos"] = anchor - rotation @ anchor_local
            poses.append(pose)
    return poses


def _cached_orientation_delta_specs(tilt_deg, roll_deg):
    tilt_values = tuple(float(value) for value in tilt_deg)
    roll_values = tuple(float(value) for value in roll_deg)
    cache_key = (
        np.ascontiguousarray(
            np.asarray(tilt_values, dtype=np.float64)
        ).tobytes(),
        np.ascontiguousarray(
            np.asarray(roll_values, dtype=np.float64)
        ).tobytes(),
    )
    with _POSE_GENERATION_CACHE_LOCK:
        cached = _ORIENTATION_DELTA_SPECS.get(cache_key)
        if cached is not None:
            return cached

        def rotation_x(angle: float) -> np.ndarray:
            cosine, sine = math.cos(angle), math.sin(angle)
            return np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, cosine, -sine],
                    [0.0, sine, cosine],
                ],
                dtype=np.float64,
            )

        def rotation_y(angle: float) -> np.ndarray:
            cosine, sine = math.cos(angle), math.sin(angle)
            return np.asarray(
                [
                    [cosine, 0.0, sine],
                    [0.0, 1.0, 0.0],
                    [-sine, 0.0, cosine],
                ],
                dtype=np.float64,
            )

        def rotation_z(angle: float) -> np.ndarray:
            cosine, sine = math.cos(angle), math.sin(angle)
            return np.asarray(
                [
                    [cosine, -sine, 0.0],
                    [sine, cosine, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

        specs = []
        for x_deg in tilt_values:
            for y_deg in tilt_values:
                for z_deg in roll_values:
                    if (
                        abs(float(x_deg)) < 1e-12
                        and abs(float(y_deg)) < 1e-12
                        and abs(float(z_deg)) < 1e-12
                    ):
                        continue
                    delta = (
                        rotation_z(math.radians(float(z_deg)))
                        @ rotation_y(math.radians(float(y_deg)))
                        @ rotation_x(math.radians(float(x_deg)))
                    )
                    specs.append((x_deg, y_deg, z_deg, delta))
        cached = tuple(specs)
        _ORIENTATION_DELTA_SPECS[cache_key] = cached
        return cached


def _cached_local_orientation_refinements(
    seeds,
    *,
    tilt_deg=rgbd_grasp_planner.REFINEMENT_TILT_DEG,
    roll_deg=rgbd_grasp_planner.REFINEMENT_ROLL_DEG,
):
    """Generate official refinements while reusing fixed delta rotations."""
    specs = _cached_orientation_delta_specs(tilt_deg, roll_deg)
    refined = []
    refine_id = 0
    for seed_index, seed in enumerate(seeds):
        base_rotation = np.asarray(seed["R"], dtype=np.float64)
        anchor = np.asarray(seed["anchor"], dtype=np.float64)
        anchor_local = np.asarray(seed["anchor_local"], dtype=np.float64)
        for x_deg, y_deg, z_deg, delta in specs:
            rotation = base_rotation @ delta
            pose = dict(seed)
            pose["R"] = rotation
            pose["quat"] = _ORIGINAL_MAT_TO_QUAT_XYZW(rotation)
            pose["eef_pos"] = anchor - rotation @ anchor_local
            pose["anchor"] = anchor.copy()
            pose["anchor_local"] = anchor_local.copy()
            pose["refined"] = True
            pose["refine_id"] = int(refine_id)
            pose["refine_seed_index"] = int(seed_index)
            pose["refine_tilt_x_deg"] = float(x_deg)
            pose["refine_tilt_y_deg"] = float(y_deg)
            pose["refine_roll_deg"] = float(z_deg)
            refined.append(pose)
            refine_id += 1
    return refined


def clear_test_pose_generation_cache() -> None:
    with _POSE_GENERATION_CACHE_LOCK:
        _RGBD_FILTER_POSE_TEMPLATES.clear()
        _ORIENTATION_DELTA_SPECS.clear()


def _build_external_ik_request_with_affinity(*args, **kwargs):
    request, policy = _ORIGINAL_BUILD_EXTERNAL_IK_REQUEST(*args, **kwargs)
    if request.get("poses"):
        with _REQUEST_AFFINITY_LOCK:
            _REQUEST_AFFINITY[id(request)] = (
                request,
                int(threading.get_ident()),
            )
    return request, policy


def _consume_request_affinity(request: Dict[str, Any]) -> Optional[int]:
    with _REQUEST_AFFINITY_LOCK:
        entry = _REQUEST_AFFINITY.pop(id(request), None)
    if entry is None or entry[0] is not request:
        return None
    return int(entry[1])


def _ik_request_in_shared_memory_with_affinity(request: Dict[str, Any]):
    transport_request, shared_memory = _ORIGINAL_IK_REQUEST_IN_SHARED_MEMORY(
        request
    )
    with _REQUEST_AFFINITY_LOCK:
        entry = _REQUEST_AFFINITY.pop(id(request), None)
        if entry is not None and entry[0] is request:
            _REQUEST_AFFINITY[id(transport_request)] = (
                transport_request,
                int(entry[1]),
            )
    return transport_request, shared_memory


def clear_test_request_affinity() -> None:
    with _REQUEST_AFFINITY_LOCK:
        _REQUEST_AFFINITY.clear()


def _render_process_initializer() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["IK_FILTER_CUDA_VISIBLE_DEVICES"] = ""


def _test_render_executor() -> concurrent.futures.ProcessPoolExecutor:
    global _TEST_RENDER_EXECUTOR
    with _TEST_RENDER_EXECUTOR_LOCK:
        if _TEST_RENDER_EXECUTOR is None:
            workers = int(os.environ.get(RENDER_PROCESSES_ENV, "1"))
            _TEST_RENDER_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_render_process_initializer,
            )
        return _TEST_RENDER_EXECUTOR


def close_test_render_executor() -> None:
    global _TEST_RENDER_EXECUTOR
    try:
        rgbd_grasp_lite.flush_lite_render_tasks()
    finally:
        with _TEST_RENDER_EXECUTOR_LOCK:
            executor = _TEST_RENDER_EXECUTOR
            _TEST_RENDER_EXECUTOR = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)


def _component_cache_key(mesh) -> str:
    return str(getattr(mesh, "_official_v2_lite_scene_key", "") or "")


def _cached_component_record(mesh) -> Optional[Dict[str, Any]]:
    key = _component_cache_key(mesh)
    if not key:
        return None
    with _COMPONENT_CACHE_LOCK:
        cached = _COMPONENT_CACHE.get(key)
        if cached is not None:
            return cached
        watertight = bool(mesh.is_watertight)
        components = None
        all_watertight = False
        bounds = None
        if watertight:
            components = tuple(mesh.split(only_watertight=False))
            if not components:
                components = (mesh,)
            all_watertight = all(
                bool(component.is_watertight) for component in components
            )
            if all_watertight:
                bounds = np.asarray(
                    [component.bounds for component in components],
                    dtype=np.float64,
                ).reshape(-1, 2, 3)
        cached = {
            "watertight": watertight,
            "components": components,
            "all_watertight": all_watertight,
            "bounds": bounds,
        }
        _COMPONENT_CACHE[key] = cached
        return cached


def _cached_mesh_occupancy_build(
    cls,
    mesh,
    *,
    voxel_m: float,
    mesh_path=None,
    voxel_cache_path=None,
    query_bounds=None,
):
    record = _cached_component_record(mesh)
    if record is None or not record["all_watertight"]:
        return _ORIGINAL_MESH_OCCUPANCY_BUILD(
            mesh,
            voxel_m=voxel_m,
            mesh_path=mesh_path,
            voxel_cache_path=voxel_cache_path,
            query_bounds=query_bounds,
        )
    scoped_bounds = None
    if query_bounds is not None:
        scoped_bounds = np.asarray(query_bounds, dtype=np.float64).reshape(2, 3)
        if np.any(scoped_bounds[0] > scoped_bounds[1]):
            raise ValueError("query_bounds minimum exceeds maximum")
    components = tuple(record["components"])
    active_components = components
    if scoped_bounds is not None:
        active_components = tuple(
            component
            for component in components
            if bool(
                np.all(component.bounds[1] >= scoped_bounds[0] - 1e-12)
                and np.all(component.bounds[0] <= scoped_bounds[1] + 1e-12)
            )
        )
    source_count = len(components)
    if not active_components:
        return cls(
            mesh=mesh,
            voxel_m=float(voxel_m),
            method="empty_components",
            components=(),
            source_component_count=source_count,
            query_bounds=scoped_bounds,
        )
    if len(active_components) == 1:
        return cls(
            mesh=active_components[0],
            voxel_m=float(voxel_m),
            method="contains",
            components=active_components,
            source_component_count=source_count,
            query_bounds=scoped_bounds,
        )
    return cls(
        mesh=mesh,
        voxel_m=float(voxel_m),
        method="component_union_contains",
        components=active_components,
        source_component_count=source_count,
        query_bounds=scoped_bounds,
    )


def _cached_watertight_component_bounds(mesh):
    record = _cached_component_record(mesh)
    if record is None or not record["all_watertight"]:
        return _ORIGINAL_WATERTIGHT_COMPONENT_BOUNDS(mesh)
    return np.asarray(record["bounds"], dtype=np.float64).copy()


def clear_test_component_cache() -> None:
    with _COMPONENT_CACHE_LOCK:
        _COMPONENT_CACHE.clear()


def _worker_environment(gpu: str) -> Dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    worker_cpu_threads = int(os.environ.get(WORKER_CPU_THREADS_ENV, "0"))
    if worker_cpu_threads > 0:
        thread_count = str(worker_cpu_threads)
        environment["OMP_NUM_THREADS"] = thread_count
        environment["MKL_NUM_THREADS"] = thread_count
        environment["OPENBLAS_NUM_THREADS"] = thread_count
        environment["NUMEXPR_NUM_THREADS"] = thread_count
    return environment


def _lane_limit() -> int:
    return max(
        1,
        min(
            16,
            int(os.environ.get("OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES", "2")),
        ),
    )


def _policy_sharding_enabled() -> bool:
    return _enabled(os.environ.get(POLICY_SHARD_ENV, "0"))


def _signature_sticky_lanes_enabled() -> bool:
    return _enabled(os.environ.get(SIGNATURE_STICKY_LANES_ENV, "0"))


def _policy_pool_key(
    *,
    gpu: str,
    arm: str,
    solver_signature: str = "",
) -> _PolicyPoolKey:
    if _policy_sharding_enabled():
        return (str(gpu), str(arm), str(solver_signature))
    return (str(gpu), str(arm))


def _acquire_policy_worker(
    *,
    arm: str,
    gpu: str,
    timeout_s: float,
    solver_signature: str = "",
    affinity: Optional[int] = None,
) -> Tuple[_PolicyWorkerLane, bool]:
    key = _policy_pool_key(
        gpu=str(gpu),
        arm=str(arm),
        solver_signature=str(solver_signature),
    )
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while True:
        stale: Optional[_PolicyWorkerLane] = None
        with _POLICY_WORKERS_CONDITION:
            lanes = _POLICY_WORKERS.setdefault(key, [])
            alive_idle = []
            for lane in tuple(lanes):
                if lane.busy:
                    continue
                if lane.worker.alive():
                    alive_idle.append(lane)
                    continue
                lanes.remove(lane)
                stale = lane
                break
            matching = None
            unowned = None
            if stale is None:
                if _signature_sticky_lanes_enabled():
                    matching = next(
                        (
                            lane
                            for lane in alive_idle
                            if lane.solver_signature == str(solver_signature)
                        ),
                        None,
                    )
                if matching is None:
                    matching = next(
                        (
                            lane
                            for lane in alive_idle
                            if affinity is not None
                            and lane.affinity == affinity
                        ),
                        None,
                    )
                unowned = next(
                    (
                        lane
                        for lane in alive_idle
                        if lane.affinity is None
                        and (
                            not _signature_sticky_lanes_enabled()
                            or lane.solver_signature is None
                        )
                    ),
                    None,
                )
            selected = matching or unowned
            if stale is None and selected is not None:
                selected.busy = True
                selected.affinity = affinity
                return selected, bool(selected.completed_requests)
            if stale is None and len(lanes) < _lane_limit():
                worker = grasp_kinematics_local._PersistentIKWorker(
                    arm=str(arm),
                    gpu=str(gpu),
                    python_path=os.environ.get(
                        "BEHAVIOR_PYTHON",
                        sys.executable,
                    ),
                    environment=_worker_environment(str(gpu)),
                )
                lane = _PolicyWorkerLane(
                    worker,
                    affinity=affinity,
                    solver_signature=(
                        str(solver_signature)
                        if _signature_sticky_lanes_enabled()
                        else None
                    ),
                )
                lanes.append(lane)
                return lane, False
            if stale is None and alive_idle:
                selected = alive_idle[0]
                selected.busy = True
                selected.affinity = affinity
                return selected, bool(selected.completed_requests)
            if stale is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"test policy IK lane arm={arm} timed out waiting "
                        f"after {float(timeout_s):.1f}s"
                    )
                _POLICY_WORKERS_CONDITION.wait(
                    timeout=min(remaining, 0.5)
                )
        if stale is not None:
            stale.worker.close()


def _release_policy_worker(
    key: _PolicyPoolKey,
    lane: _PolicyWorkerLane,
    *,
    completed: bool,
    solver_signature: Optional[str] = None,
) -> None:
    with _POLICY_WORKERS_CONDITION:
        lanes = _POLICY_WORKERS.get(key)
        if lanes is not None and lane in lanes:
            if completed:
                lane.completed_requests += 1
                if _signature_sticky_lanes_enabled():
                    lane.solver_signature = str(solver_signature or "")
            lane.busy = False
        _POLICY_WORKERS_CONDITION.notify_all()


def _discard_policy_worker(
    key: _PolicyPoolKey,
    lane: _PolicyWorkerLane,
) -> None:
    with _POLICY_WORKERS_CONDITION:
        lanes = _POLICY_WORKERS.get(key)
        if lanes is not None and lane in lanes:
            lanes.remove(lane)
            if not lanes:
                _POLICY_WORKERS.pop(key, None)
        _POLICY_WORKERS_CONDITION.notify_all()
    lane.worker.close()


def _run_policy_ik_arm(
    *,
    arm: str,
    gpu: str,
    python_path: str,
    environment: Dict[str, str],
    request: Dict[str, Any],
    solver_signature: str,
    timeout_s: float,
    prepare_next: Optional[Dict[str, Any]] = None,
):
    del python_path, environment
    affinity = (
        _consume_request_affinity(request)
        if _enabled(os.environ.get(LANE_AFFINITY_ENV, "0"))
        else None
    )
    key = _policy_pool_key(
        gpu=str(gpu),
        arm=str(arm),
        solver_signature=str(solver_signature),
    )
    effective_prepare_next = prepare_next
    if _policy_sharding_enabled() or _signature_sticky_lanes_enabled():
        effective_prepare_next = {
            "request": request,
            "solver_signature": str(solver_signature),
        }
    last_error: Optional[Exception] = None
    for attempt in range(2):
        lane, reused = _acquire_policy_worker(
            arm=str(arm),
            gpu=str(gpu),
            timeout_s=float(timeout_s),
            solver_signature=str(solver_signature),
            affinity=affinity,
        )
        try:
            trace_path = _write_ik_trace_request(
                arm=arm,
                request=request,
                solver_signature=solver_signature,
            )
            result = lane.worker.request(
                request,
                solver_signature=str(solver_signature),
                timeout_s=float(timeout_s),
                prepare_next=effective_prepare_next,
            )
            _write_ik_trace_result(trace_path, result)
        except Exception as exc:
            last_error = exc
            _discard_policy_worker(key, lane)
            if attempt:
                break
        else:
            _release_policy_worker(
                key,
                lane,
                completed=True,
                solver_signature=str(solver_signature),
            )
            return result, reused
    raise RuntimeError(
        f"test policy IK worker arm={arm} failed after restart: {last_error}"
    )


def _write_ik_trace_request(
    *,
    arm: str,
    request: Dict[str, Any],
    solver_signature: str,
) -> Optional[Path]:
    raw_dir = os.environ.get(IK_TRACE_DIR_ENV, "").strip()
    if not raw_dir:
        return None
    if request.get("pose_shared_memory") is not None:
        raise RuntimeError(
            f"{IK_TRACE_DIR_ENV} requires "
            "OFFICIAL_V2_RGBD_LITE_TEST_SHARED_MEMORY=0"
        )
    global _IK_TRACE_SEQUENCE
    with _IK_TRACE_LOCK:
        sequence = _IK_TRACE_SEQUENCE
        _IK_TRACE_SEQUENCE += 1
    trace_dir = Path(raw_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    policy = str(request.get("solver_policy") or "baseline")
    path = trace_dir / f"{sequence:03d}_{arm}_{policy}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "sequence": sequence,
                "arm": str(arm),
                "solver_signature": str(solver_signature),
                "request": copy.deepcopy(request),
                "result": None,
            },
            ensure_ascii=True,
            allow_nan=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _write_ik_trace_result(
    path: Optional[Path],
    result: Dict[str, Any],
) -> None:
    if path is None:
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    record["result"] = copy.deepcopy(result)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            record,
            ensure_ascii=True,
            allow_nan=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_policy_ik(
    state,
    *,
    pos_tol_m: float,
    ori_tol_deg: float,
    active_arms: Tuple[str, ...],
    policy: str,
) -> None:
    del state, pos_tol_m, ori_tol_deg, active_arms, policy
    # Exact reload makes speculative preparation duplicate solver construction.
    # The first real requests create all required lanes concurrently instead.


def close_test_policy_workers() -> None:
    with _POLICY_WORKERS_CONDITION:
        workers = [
            lane.worker
            for lanes in _POLICY_WORKERS.values()
            for lane in lanes
        ]
        _POLICY_WORKERS.clear()
        _POLICY_WORKERS_CONDITION.notify_all()
    for worker in workers:
        worker.close()
    clear_test_request_affinity()
    clear_test_pre_ik_cache()


def install_policy_pool_runtime() -> Dict[str, str]:
    """Install the experiment in this process only, behind a test-mode gate."""
    global _INSTALLED
    if not _enabled(os.environ.get(TEST_MODE_ENV, "0")):
        raise RuntimeError(f"{TEST_MODE_ENV}=1 is required")

    policy_pool_enabled = _enabled(os.environ.get(POLICY_POOL_ENV, "1"))
    skip_reachable_se3 = _enabled(
        os.environ.get(SKIP_REACHABLE_SE3_ENV, "0")
    )
    shared_memory_enabled = _enabled(
        os.environ.get(SHARED_MEMORY_ENV, "1")
    )
    forkserver_enabled = _enabled(
        os.environ.get(FORKSERVER_ENV, "1")
    )
    exact_solver_reload = _enabled(
        os.environ.get(EXACT_SOLVER_RELOAD_ENV, "1")
    )
    physical_batch = int(os.environ.get(PHYSICAL_BATCH_ENV, "0"))
    if physical_batch < 0 or physical_batch > 64:
        raise ValueError(f"{PHYSICAL_BATCH_ENV} must be in [0, 64]")
    solver_reset = str(
        os.environ.get(SOLVER_RESET_ENV, "none")
    ).strip().lower()
    if solver_reset not in {"none", "optimizer", "graph", "optimizer_graph"}:
        raise ValueError(
            f"{SOLVER_RESET_ENV} must be none, optimizer, graph, or "
            "optimizer_graph"
        )
    render_processes = int(os.environ.get(RENDER_PROCESSES_ENV, "1"))
    if render_processes < 1 or render_processes > 16:
        raise ValueError(f"{RENDER_PROCESSES_ENV} must be in [1, 16]")
    component_cache_enabled = _enabled(
        os.environ.get(COMPONENT_CACHE_ENV, "0")
    )
    policy_shard_enabled = _policy_sharding_enabled()
    lane_affinity_enabled = _enabled(
        os.environ.get(LANE_AFFINITY_ENV, "0")
    )
    pre_ik_dedup_enabled = _enabled(
        os.environ.get(PRE_IK_DEDUP_ENV, "0")
    )
    pre_ik_count_dedup_enabled = _enabled(
        os.environ.get(PRE_IK_COUNT_DEDUP_ENV, "0")
    )
    prepared_signature_pool = int(
        os.environ.get(PREPARED_SIGNATURE_POOL_ENV, "0")
    )
    if prepared_signature_pool < 0 or prepared_signature_pool > 4:
        raise ValueError(
            f"{PREPARED_SIGNATURE_POOL_ENV} must be in [0, 4]"
        )
    camera_face_orientation_cache = _enabled(
        os.environ.get(CAMERA_FACE_ORIENTATION_CACHE_ENV, "0")
    )
    normal_alignment_cache = _enabled(
        os.environ.get(NORMAL_ALIGNMENT_CACHE_ENV, "0")
    )
    pose_generation_cache = _enabled(
        os.environ.get(POSE_GENERATION_CACHE_ENV, "0")
    )
    candidate_compression_enabled = _enabled(
        os.environ.get(CANDIDATE_COMPRESSION_ENV, "0")
    )
    candidate_compression_scope = str(
        os.environ.get(CANDIDATE_COMPRESSION_SCOPE_ENV, "all")
    ).strip().lower()
    candidate_stages_text = str(
        os.environ.get(CANDIDATE_COMPRESSION_STAGES_ENV, "")
    ).strip()
    candidate_compression_stages = (
        {
            stage.strip().lower()
            for stage in candidate_stages_text.split(",")
            if stage.strip()
        }
        if candidate_stages_text
        else None
    )
    candidate_post_limit = int(
        os.environ.get(CANDIDATE_POST_LIMIT_ENV, "64")
    )
    candidate_stage_limit = int(
        os.environ.get(CANDIDATE_STAGE_LIMIT_ENV, "64")
    )
    worker_cpu_threads = int(os.environ.get(WORKER_CPU_THREADS_ENV, "0"))
    if worker_cpu_threads < 0 or worker_cpu_threads > 64:
        raise ValueError(f"{WORKER_CPU_THREADS_ENV} must be in [0, 64]")
    signature_sticky_lanes = _signature_sticky_lanes_enabled()
    grasp_kinematics_local.close_persistent_ik_workers()
    close_test_policy_workers()
    close_test_render_executor()
    clear_test_component_cache()
    clear_test_camera_face_cache()
    clear_test_normal_alignment_cache()
    clear_test_pose_generation_cache()
    _configure_candidate_compression(
        candidate_compression_enabled,
        scope=candidate_compression_scope,
        stages=candidate_compression_stages,
        post_limit=candidate_post_limit,
        stage_limit=candidate_stage_limit,
    )
    settings = {
        "OFFICIAL_V2_LITE_SCENE_CACHE": "1",
        "OFFICIAL_V2_LITE_SCENE_CACHE_SIZE": os.environ.get(
            "OFFICIAL_V2_RGBD_LITE_TEST_SCENE_CACHE_SIZE",
            "5",
        ),
        "OFFICIAL_V2_LITE_OCCUPANCY_CACHE": "1",
        "OFFICIAL_V2_LITE_OCCUPANCY_CACHE_SIZE": os.environ.get(
            "OFFICIAL_V2_RGBD_LITE_TEST_OCCUPANCY_CACHE_SIZE",
            "32",
        ),
        "OFFICIAL_V2_LITE_OCCUPANCY_REGION_CACHE": os.environ.get(
            "OFFICIAL_V2_LITE_OCCUPANCY_REGION_CACHE",
            "0",
        ),
        "OFFICIAL_V2_LITE_OCCUPANCY_REGION_CACHE_SIZE": os.environ.get(
            "OFFICIAL_V2_RGBD_LITE_TEST_OCCUPANCY_REGION_CACHE_SIZE",
            "32",
        ),
        "OFFICIAL_V2_LITE_RENDER_CACHE": "1",
        "OFFICIAL_V2_LITE_RENDER_CACHE_SIZE": os.environ.get(
            "OFFICIAL_V2_RGBD_LITE_TEST_RENDER_CACHE_SIZE",
            "32",
        ),
        "OFFICIAL_V2_LITE_ASYNC_THREE_VIEW": os.environ.get(
            "OFFICIAL_V2_LITE_ASYNC_THREE_VIEW",
            "0",
        ),
        "OFFICIAL_V2_LITE_IK_SHARED_MEMORY": str(
            int(shared_memory_enabled)
        ),
        "OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES": os.environ.setdefault(
            "OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES",
            "2",
        ),
    }
    if policy_pool_enabled:
        settings.update(
            {
                "OFFICIAL_V2_LITE_PERSISTENT_IK": "1",
                "OFFICIAL_V2_LITE_FORKSERVER_IK": str(
                    int(forkserver_enabled)
                ),
                "OFFICIAL_V2_LITE_EXACT_SOLVER_RELOAD": str(
                    int(exact_solver_reload)
                ),
            }
        )
    os.environ.update(settings)
    grasp_kinematics_local._run_persistent_ik_arm = (
        _run_policy_ik_arm
        if policy_pool_enabled
        else _ORIGINAL_RUN_PERSISTENT_IK_ARM
    )
    grasp_kinematics_local._build_external_ik_request = (
        _build_external_ik_request_with_affinity
        if policy_pool_enabled and lane_affinity_enabled
        else _ORIGINAL_BUILD_EXTERNAL_IK_REQUEST
    )
    grasp_kinematics_local._ik_request_in_shared_memory = (
        _ik_request_in_shared_memory_with_affinity
        if policy_pool_enabled and lane_affinity_enabled
        else _ORIGINAL_IK_REQUEST_IN_SHARED_MEMORY
    )
    grasp_kinematics_local._WORKER_PATH = (
        _TEST_WORKER_PATH
        if policy_pool_enabled
        and (
            physical_batch > 0
            or solver_reset != "none"
            or prepared_signature_pool > 0
        )
        else _ORIGINAL_IK_WORKER_PATH
    )
    rgbd_grasp_lite.prepare_external_ik = (
        _prepare_policy_ik
        if policy_pool_enabled
        else _ORIGINAL_PREPARE_EXTERNAL_IK
    )
    rgbd_grasp_lite.LITE_SKIP_REACHABLE_SE3_IK_STAGES = (
        skip_reachable_se3
    )
    rgbd_grasp_lite._render_local_three_views_lite = (
        _ORIGINAL_RENDER_LOCAL_THREE_VIEWS_LITE
    )
    rgbd_grasp_lite._lite_render_executor = (
        _test_render_executor
        if render_processes > 1
        else _ORIGINAL_LITE_RENDER_EXECUTOR
    )
    MeshOccupancyEvaluator.build = (
        classmethod(_cached_mesh_occupancy_build)
        if component_cache_enabled
        else _ORIGINAL_MESH_OCCUPANCY_BUILD_DESCRIPTOR
    )
    rgbd_grasp_lite._lite_watertight_component_bounds = (
        _cached_watertight_component_bounds
        if component_cache_enabled
        else _ORIGINAL_WATERTIGHT_COMPONENT_BOUNDS
    )
    if pre_ik_dedup_enabled:
        rgbd_grasp_lite._apply_camera_face_preserving_anchor_lite = (
            _cached_camera_face_lite
        )
    elif camera_face_orientation_cache:
        rgbd_grasp_lite._apply_camera_face_preserving_anchor_lite = (
            _orientation_cached_camera_face_lite
        )
    else:
        rgbd_grasp_lite._apply_camera_face_preserving_anchor_lite = (
            _ORIGINAL_CAMERA_FACE_LITE
        )
    rgbd_grasp_planner.DenseSceneOccupancy.counts_for_poses = (
        _cached_counts_for_poses
        if pre_ik_dedup_enabled or pre_ik_count_dedup_enabled
        else _ORIGINAL_COUNTS_FOR_POSES
    )
    rgbd_grasp_planner.filter_poses_safe_final_dual_arm_ik = (
        _mark_post_prefix_filter
        if pre_ik_dedup_enabled or pre_ik_count_dedup_enabled
        else _ORIGINAL_FILTER_SAFE_FINAL
    )
    rgbd_grasp_planner.attach_normal_alignment = (
        _cached_normal_alignment
        if normal_alignment_cache
        else _ORIGINAL_ATTACH_NORMAL_ALIGNMENT
    )
    rgbd_grasp_planner.generate_rgbd_filter_poses = (
        _cached_rgbd_filter_pose_templates
        if pose_generation_cache
        else _ORIGINAL_GENERATE_RGBD_FILTER_POSES
    )
    rgbd_grasp_planner.generate_local_orientation_refinements = (
        _cached_local_orientation_refinements
        if pose_generation_cache
        else _ORIGINAL_GENERATE_LOCAL_ORIENTATION_REFINEMENTS
    )
    _INSTALLED = True
    settings[PHYSICAL_BATCH_ENV] = str(physical_batch)
    settings[SOLVER_RESET_ENV] = solver_reset
    settings[RENDER_PROCESSES_ENV] = str(render_processes)
    settings[COMPONENT_CACHE_ENV] = str(int(component_cache_enabled))
    settings[POLICY_SHARD_ENV] = str(int(policy_shard_enabled))
    settings[LANE_AFFINITY_ENV] = str(int(lane_affinity_enabled))
    settings[PRE_IK_DEDUP_ENV] = str(int(pre_ik_dedup_enabled))
    settings[PRE_IK_COUNT_DEDUP_ENV] = str(
        int(pre_ik_count_dedup_enabled)
    )
    settings[PREPARED_SIGNATURE_POOL_ENV] = str(prepared_signature_pool)
    settings[CAMERA_FACE_ORIENTATION_CACHE_ENV] = str(
        int(camera_face_orientation_cache)
    )
    settings[NORMAL_ALIGNMENT_CACHE_ENV] = str(
        int(normal_alignment_cache)
    )
    settings[POSE_GENERATION_CACHE_ENV] = str(
        int(pose_generation_cache)
    )
    settings[WORKER_CPU_THREADS_ENV] = str(worker_cpu_threads)
    settings[SIGNATURE_STICKY_LANES_ENV] = str(
        int(signature_sticky_lanes)
    )
    settings[CANDIDATE_COMPRESSION_ENV] = str(
        int(candidate_compression_enabled)
    )
    settings[CANDIDATE_COMPRESSION_SCOPE_ENV] = (
        candidate_compression_scope
    )
    settings[CANDIDATE_COMPRESSION_STAGES_ENV] = ",".join(
        sorted(_CANDIDATE_COMPRESSION_STAGES)
    )
    settings[CANDIDATE_POST_LIMIT_ENV] = str(candidate_post_limit)
    settings[CANDIDATE_STAGE_LIMIT_ENV] = str(candidate_stage_limit)
    settings["ik_worker_path"] = grasp_kinematics_local._WORKER_PATH
    return {
        "test_mode": "1",
        "runtime": (
            "exact_reload_parallel_ik_lanes"
            if policy_pool_enabled
            else "official_one_shot_ik"
        ),
        "policy_pool_enabled": str(int(policy_pool_enabled)),
        "shared_memory_enabled": str(int(shared_memory_enabled)),
        "forkserver_enabled": str(int(forkserver_enabled)),
        "exact_solver_reload": str(int(exact_solver_reload)),
        "render_policy": (
            "async_spawn_process_pool"
            if render_processes > 1
            else "async_serial_matplotlib"
        ),
        "lite_skip_reachable_se3_ik_stages": str(
            int(skip_reachable_se3)
        ),
        **settings,
    }


def installed() -> bool:
    return bool(_INSTALLED)


atexit.register(close_test_policy_workers)
atexit.register(close_test_render_executor)


__all__ = [
    "POLICY_POOL_ENV",
    "PHYSICAL_BATCH_ENV",
    "POLICY_SHARD_ENV",
    "LANE_AFFINITY_ENV",
    "PRE_IK_DEDUP_ENV",
    "PRE_IK_COUNT_DEDUP_ENV",
    "PREPARED_SIGNATURE_POOL_ENV",
    "CAMERA_FACE_ORIENTATION_CACHE_ENV",
    "NORMAL_ALIGNMENT_CACHE_ENV",
    "POSE_GENERATION_CACHE_ENV",
    "CANDIDATE_COMPRESSION_ENV",
    "CANDIDATE_COMPRESSION_SCOPE_ENV",
    "CANDIDATE_COMPRESSION_STAGES_ENV",
    "CANDIDATE_POST_LIMIT_ENV",
    "CANDIDATE_STAGE_LIMIT_ENV",
    "WORKER_CPU_THREADS_ENV",
    "SIGNATURE_STICKY_LANES_ENV",
    "IK_TRACE_DIR_ENV",
    "RENDER_PROCESSES_ENV",
    "SOLVER_RESET_ENV",
    "COMPONENT_CACHE_ENV",
    "EXACT_SOLVER_RELOAD_ENV",
    "FORKSERVER_ENV",
    "SHARED_MEMORY_ENV",
    "SKIP_REACHABLE_SE3_ENV",
    "TEST_MODE_ENV",
    "close_test_policy_workers",
    "close_test_render_executor",
    "clear_test_component_cache",
    "clear_test_camera_face_cache",
    "clear_test_request_affinity",
    "clear_test_pre_ik_cache",
    "install_policy_pool_runtime",
    "installed",
    "planner_case_scope",
]
