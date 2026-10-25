"""Isolated lite policy layered over the submission-local RGB-D planner.

Candidate compression is disabled by default. Sampling, ranking, IK stages,
local FK validation, occupancy, refinement, and final filtering retain the
original lite policy while every runtime dependency stays in this package.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import concurrent.futures
import hashlib
import math
import os
import threading
import time
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import rgbd_grasp_planner as production
from .grasp_geometry_local import gripper_geometry
from .grasp_kinematics_local import (
    IK_FILTER_ORI_TOL_DEG,
    IK_FILTER_POS_TOL_M,
    prepare_external_ik,
)
from .rgbd_projective_occupancy import (
    PROJECTIVE_GEOMETRY_VERSION as RGBD_LITE_GEOMETRY_VERSION,
    ProjectiveDepthScene,
    build_projective_local_occupancy,
    prepare_projective_scene_from_session,
)
from .rgbd_warp_occupancy import (
    WARP_OCCUPANCY_VERSION,
    build_warp_local_occupancy,
    clear_warp_mesh_cache,
)


RGBD_LITE_MESH_VERSION = RGBD_LITE_GEOMETRY_VERSION  # Compatibility alias.
LITE_GEOMETRY_BACKEND_ENV = "OFFICIAL_V2_LITE_GEOMETRY_BACKEND"
LITE_GEOMETRY_BACKEND_PROJECTIVE = "projective_cuda"
LITE_GEOMETRY_BACKEND_WARP = "v53_warp_cuda"
LITE_GEOMETRY_BACKEND_LEGACY = "legacy_v53_cpu"
LITE_POSITION_LIMIT_MM = 3.0
LITE_ANGLE_LIMIT_DEG = 6.0
LITE_MEMBERS_PER_CLUSTER = 2
LITE_POST_PARETO_IK_LIMIT = 64
LITE_REACHABLE_STAGE_IK_LIMIT = 64
# Lite is a distinct planner. A terminal Lite no-pose result is returned
# directly; callers can opt into the slower production retry for diagnostics.
LITE_FALLBACK_TO_FULL = False
LITE_FALLBACK_ENV = "PLAN_GRASP_RGBD_LITE_PRODUCTION_FALLBACK"
LITE_DIAGNOSTIC_CAPS = (32, 48, 64)
LITE_COMPRESS_INITIAL_STAGE = False
LITE_COMPRESS_REFINEMENT_STAGES = False
LITE_SKIP_REACHABLE_SE3_IK_STAGES = False
LITE_IK_WORKER_POLICY = "cuda_graph_split8_rewarm"
LITE_WARM_IK_WORKER_POLICY = "cuda_graph_warm32x6_rewarm"
LITE_GRIPPER_GEOMETRY_CACHE_VERSION = "rgbd_lite_gripper_geometry_v1"
LITE_OCCUPANCY_CACHE_VERSION = "rgbd_lite_dense_occupancy_v2_hit_validation"
LITE_OCCUPANCY_REGION_CACHE_VERSION = "rgbd_lite_occupancy_region_v2"
LITE_RENDER_CACHE_VERSION = "rgbd_lite_three_view_png_v1"
LITE_POSE_OVERLAP_FILTER_BUILD = (
    "rgbd_lite_pose_overlap_filter_v2_eef_envelope_center"
)

_V52_SCENE_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_V52_SCENE_CACHE_LOCK = threading.Lock()
_LITE_OCCUPANCY_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_LITE_OCCUPANCY_CACHE_LOCK = threading.Lock()
_LITE_OCCUPANCY_REGION_CACHE: "OrderedDict[str, Dict[str, Any]]" = (
    OrderedDict()
)
_LITE_RENDER_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
_LITE_RENDER_CACHE_LOCK = threading.Lock()
_LITE_RENDER_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_LITE_RENDER_FUTURES: List[concurrent.futures.Future] = []
_LITE_RENDER_EXECUTOR_LOCK = threading.Lock()


def _lite_geometry_backend() -> str:
    backend = os.environ.get(
        LITE_GEOMETRY_BACKEND_ENV,
        LITE_GEOMETRY_BACKEND_WARP,
    ).strip().lower()
    allowed = {
        LITE_GEOMETRY_BACKEND_PROJECTIVE,
        LITE_GEOMETRY_BACKEND_WARP,
        LITE_GEOMETRY_BACKEND_LEGACY,
    }
    if backend not in allowed:
        raise ValueError(
            f"invalid {LITE_GEOMETRY_BACKEND_ENV}={backend!r}; "
            f"expected one of {sorted(allowed)}"
        )
    return backend


def reconstruct_v52_scene_from_session(
    session: Dict[str, Any],
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
):
    """Prepare the configured Lite geometry representation.

    The historical function name is retained for benchmark and tool callers.
    V53 is imported only for mesh-backed modes. The default Warp mode keeps
    V53 semantics while replacing its CPU point-in-mesh stage with CUDA.
    """
    if _lite_geometry_backend() in {
        LITE_GEOMETRY_BACKEND_WARP,
        LITE_GEOMETRY_BACKEND_LEGACY,
    }:
        from .rgbd_scene_mesh_v53 import reconstruct_scene_mesh_from_session

        return reconstruct_scene_mesh_from_session(
            session,
            camera_pos=camera_pos,
            camera_quat_xyzw=camera_quat_xyzw,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            defer_topology_validation_to_warp_cuda=(
                _lite_geometry_backend() == LITE_GEOMETRY_BACKEND_WARP
            ),
        )
    scene_key = _v52_scene_cache_key(
        session,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
    )
    return prepare_projective_scene_from_session(
        session,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        scene_key=scene_key,
    )


def _lite_fallback_to_full_enabled() -> bool:
    value = os.environ.get(LITE_FALLBACK_ENV)
    if value is None:
        return bool(LITE_FALLBACK_TO_FULL)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _v52_scene_cache_key(
    session: Dict[str, Any],
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
) -> str:
    init_dir = str(session.get("init_dir") or "")
    depth_path = os.path.realpath(
        str(
            session.get("depth_path")
            or os.path.join(init_dir, "depth.npy")
        )
    )
    backend = _lite_geometry_backend()
    if backend == LITE_GEOMETRY_BACKEND_PROJECTIVE:
        depth_stat = os.stat(depth_path)
        payload = {
            "geometry_backend": backend,
            "geometry_version": RGBD_LITE_GEOMETRY_VERSION,
            "depth_path": depth_path,
            "depth_size": int(depth_stat.st_size),
            "depth_mtime_ns": int(depth_stat.st_mtime_ns),
        }
    else:
        rgb_path = os.path.realpath(
            str(session.get("rgb_path") or os.path.join(init_dir, "rgb.png"))
        )
        payload = {
            "geometry_backend": backend,
            "mesh_version": "v53",
            "rgb_path": rgb_path,
            "rgb_sha256": _file_sha256(rgb_path),
            "depth_path": depth_path,
            "depth_sha256": _file_sha256(depth_path),
        }
    payload.update({
        "camera_pos": [
            float(value).hex()
            for value in np.asarray(camera_pos, dtype=np.float64).reshape(3)
        ],
        "camera_quat_xyzw": [
            float(value).hex()
            for value in np.asarray(
                camera_quat_xyzw,
                dtype=np.float64,
            ).reshape(4)
        ],
        "focal_length": float(focal_length).hex(),
        "horizontal_aperture": float(horizontal_aperture).hex(),
    })
    return hashlib.sha256(
        repr(sorted(payload.items())).encode("utf-8")
    ).hexdigest()


def clear_v52_scene_cache() -> None:
    flush_lite_render_tasks()
    with _V52_SCENE_CACHE_LOCK:
        for reconstructed in _V52_SCENE_CACHE.values():
            scene = reconstructed[0]
            if isinstance(scene, ProjectiveDepthScene):
                scene.clear_device_cache()
        _V52_SCENE_CACHE.clear()
    with _LITE_OCCUPANCY_CACHE_LOCK:
        _LITE_OCCUPANCY_CACHE.clear()
        _LITE_OCCUPANCY_REGION_CACHE.clear()
    with _LITE_RENDER_CACHE_LOCK:
        _LITE_RENDER_CACHE.clear()
    clear_warp_mesh_cache()
    from .rgbd_scene_mesh_v12 import clear_support_structure_mask_cache

    clear_support_structure_mask_cache()


def _lite_render_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _LITE_RENDER_EXECUTOR
    with _LITE_RENDER_EXECUTOR_LOCK:
        if _LITE_RENDER_EXECUTOR is None:
            _LITE_RENDER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="official-v2-lite-render",
            )
        return _LITE_RENDER_EXECUTOR


def flush_lite_render_tasks() -> None:
    """Wait for byte-identical auxiliary three-view renders to finish."""
    with _LITE_RENDER_EXECUTOR_LOCK:
        pending = list(_LITE_RENDER_FUTURES)
        _LITE_RENDER_FUTURES.clear()
    for future in pending:
        future.result()


def _clone_v52_scene(reconstructed, *, scene_key: Optional[str] = None):
    """Return an isolated legacy mesh or shared immutable CUDA source."""
    mesh, rgb, depth, metadata = reconstructed
    if isinstance(mesh, ProjectiveDepthScene):
        if scene_key is not None:
            mesh.scene_key = str(scene_key)
        return mesh, rgb, depth, dict(metadata)
    if _lite_geometry_backend() == LITE_GEOMETRY_BACKEND_WARP:
        # V53 meshes are immutable after reconstruction. Sharing the object is
        # required so repeated clicks on one frozen frame reuse the CUDA BVH.
        if scene_key is not None:
            setattr(mesh, "_official_v2_lite_scene_key", scene_key)
        return mesh, rgb, depth, dict(metadata)
    local_mesh = mesh.copy(include_cache=False)
    if scene_key is not None:
        setattr(local_mesh, "_official_v2_lite_scene_key", scene_key)
    return (
        local_mesh,
        rgb,
        depth,
        dict(metadata),
    )


def _cached_reconstruct_v52_scene_from_session(
    session: Dict[str, Any],
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
):
    enabled = os.environ.get(
        "OFFICIAL_V2_LITE_SCENE_CACHE",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return reconstruct_v52_scene_from_session(
            session,
            camera_pos=camera_pos,
            camera_quat_xyzw=camera_quat_xyzw,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
        )

    key = _v52_scene_cache_key(
        session,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
    )
    limit = max(
        1,
        int(os.environ.get("OFFICIAL_V2_LITE_SCENE_CACHE_SIZE", "2")),
    )
    # Reconstruction is serialized in the evaluator already. Holding this
    # lock also makes concurrent plans for one frozen frame single-flight.
    with _V52_SCENE_CACHE_LOCK:
        cached = _V52_SCENE_CACHE.get(key)
        if cached is not None:
            _V52_SCENE_CACHE.move_to_end(key)
            return _clone_v52_scene(cached, scene_key=key)
        reconstructed = reconstruct_v52_scene_from_session(
            session,
            camera_pos=camera_pos,
            camera_quat_xyzw=camera_quat_xyzw,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
        )
        _V52_SCENE_CACHE[key] = reconstructed
        _V52_SCENE_CACHE.move_to_end(key)
        while len(_V52_SCENE_CACHE) > limit:
            _V52_SCENE_CACHE.popitem(last=False)
            with _LITE_OCCUPANCY_CACHE_LOCK:
                _LITE_OCCUPANCY_CACHE.clear()
        return _clone_v52_scene(reconstructed, scene_key=key)


def _lite_occupancy_cache_key(mesh, kwargs: Dict[str, Any]) -> str:
    scene_key = getattr(mesh, "_official_v2_lite_scene_key", None)
    if not scene_key:
        return ""
    digest = hashlib.sha256()
    digest.update(LITE_OCCUPANCY_CACHE_VERSION.encode("ascii"))
    digest.update(str(scene_key).encode("ascii"))

    def update_array(value) -> None:
        array = np.ascontiguousarray(
            np.asarray(value, dtype=np.float64)
        )
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())

    update_array(kwargs["hit"])
    update_array(kwargs.get("axial_z_m", production.AXIAL_Z_M))
    for part in kwargs["query_offsets"]:
        update_array(part)
    for value in (
        float(kwargs.get("anchor_radius_m", 0.0)),
        float(kwargs.get("voxel_m", production.VOXEL_M)),
        int(kwargs.get("contains_chunk", 200_000)),
        bool(kwargs.get("validate_hit", True)),
    ):
        digest.update(repr(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _lite_occupancy_region_cache_key(mesh, kwargs: Dict[str, Any]) -> str:
    scene_key = getattr(mesh, "_official_v2_lite_scene_key", None)
    if not scene_key:
        return ""
    digest = hashlib.sha256()
    digest.update(LITE_OCCUPANCY_REGION_CACHE_VERSION.encode("ascii"))
    digest.update(str(scene_key).encode("ascii"))

    def update_array(value) -> None:
        array = np.ascontiguousarray(
            np.asarray(value, dtype=np.float64)
        )
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())

    update_array(kwargs.get("axial_z_m", production.AXIAL_Z_M))
    for part in kwargs["query_offsets"]:
        update_array(part)
    for value in (
        float(kwargs.get("anchor_radius_m", 0.0)),
        float(kwargs.get("voxel_m", production.VOXEL_M)),
        int(kwargs.get("contains_chunk", 200_000)),
    ):
        digest.update(repr(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _lite_local_occupancy_grid_spec(
    kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    voxel_m = float(kwargs.get("voxel_m", production.VOXEL_M))
    maximum_radius = 0.0
    axial_z_m = np.asarray(
        kwargs.get("axial_z_m", production.AXIAL_Z_M),
        dtype=np.float64,
    ).reshape(-1)
    for axial_z in axial_z_m:
        anchor_local = np.asarray([0.0, 0.0, axial_z], dtype=np.float64)
        for local in kwargs["query_offsets"]:
            points = np.asarray(local, dtype=np.float64).reshape(-1, 3)
            if len(points):
                maximum_radius = max(
                    maximum_radius,
                    float(
                        np.linalg.norm(
                            points - anchor_local.reshape(1, 3),
                            axis=1,
                        ).max()
                    ),
                )
    radius = (
        float(kwargs.get("anchor_radius_m", 0.0))
        + maximum_radius
        + voxel_m * 2.0
    )
    center = np.asarray(kwargs["hit"], dtype=np.float64).reshape(3)
    origin = np.floor((center - radius) / voxel_m) * voxel_m
    upper = np.ceil((center + radius) / voxel_m) * voxel_m
    shape = np.rint((upper - origin) / voxel_m).astype(np.int64) + 1
    return origin, upper, shape, float(radius)


def _lite_watertight_component_bounds(mesh) -> Optional[np.ndarray]:
    """Cache the component contract used by MeshOccupancyEvaluator.build."""
    if not bool(getattr(mesh, "is_watertight", False)):
        return None
    components = tuple(mesh.split(only_watertight=False))
    if not components:
        components = (mesh,)
    if not all(bool(component.is_watertight) for component in components):
        return None
    return np.asarray(
        [component.bounds for component in components],
        dtype=np.float64,
    ).reshape(-1, 2, 3)


def _lite_local_component_metadata(
    component_bounds: Optional[np.ndarray],
    *,
    origin: np.ndarray,
    upper: np.ndarray,
) -> Dict[str, Any]:
    if component_bounds is None:
        return {}
    bounds = np.asarray(component_bounds, dtype=np.float64).reshape(-1, 2, 3)
    active = np.all(bounds[:, 1] >= origin - 1e-12, axis=1) & np.all(
        bounds[:, 0] <= upper + 1e-12,
        axis=1,
    )
    active_count = int(active.sum())
    source_count = int(len(bounds))
    if active_count == 0:
        method = "empty_components"
    elif active_count == 1:
        method = "contains"
    else:
        method = "component_union_contains"
    return {
        "occupancy_method": method,
        "occupancy_component_count": active_count,
        "occupancy_source_component_count": source_count,
        "occupancy_culled_component_count": source_count - active_count,
    }


def _extend_lite_occupancy_region(
    mesh,
    region: Dict[str, Any],
    *,
    required_origin: np.ndarray,
    required_upper: np.ndarray,
    voxel_m: float,
    contains_chunk: int,
) -> Dict[str, int]:
    from .mesh_occupancy import MeshOccupancyEvaluator

    old_origin = np.asarray(region["origin"], dtype=np.float64).reshape(3)
    old_occupancy = np.asarray(region["occupancy"], dtype=bool)
    old_upper = old_origin + (
        np.asarray(old_occupancy.shape, dtype=np.float64) - 1.0
    ) * float(voxel_m)
    new_origin = np.minimum(old_origin, required_origin)
    new_upper = np.maximum(old_upper, required_upper)
    new_shape = (
        np.rint((new_upper - new_origin) / float(voxel_m)).astype(np.int64)
        + 1
    )
    if tuple(new_shape.tolist()) == tuple(old_occupancy.shape):
        return {"extended_voxels": 0, "contains_query_points": 0}

    maximum_voxels = max(
        1,
        int(
            os.environ.get(
                "OFFICIAL_V2_LITE_OCCUPANCY_REGION_MAX_VOXELS",
                "16000000",
            )
        ),
    )
    total = int(np.prod(new_shape, dtype=np.int64))
    if total > maximum_voxels:
        raise OverflowError(
            f"occupancy region would contain {total} voxels"
        )

    expanded = np.zeros(tuple(new_shape.tolist()), dtype=bool)
    known = np.zeros(tuple(new_shape.tolist()), dtype=bool)
    insertion = np.rint(
        (old_origin - new_origin) / float(voxel_m)
    ).astype(np.int64)
    old_shape = np.asarray(old_occupancy.shape, dtype=np.int64)
    slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(insertion, old_shape)
    )
    expanded[slices] = old_occupancy
    known[slices] = True
    missing = np.flatnonzero(~known.reshape(-1))
    evaluator = MeshOccupancyEvaluator.build(
        mesh,
        voxel_m=float(voxel_m),
        query_bounds=np.stack((new_origin, new_upper), axis=0),
    )
    flat = expanded.reshape(-1)
    yz = int(new_shape[1] * new_shape[2])
    queried = 0
    chunk = max(1, int(contains_chunk))
    for start in range(0, len(missing), chunk):
        linear = missing[start : start + chunk]
        ix = linear // yz
        remainder = linear - ix * yz
        iy = remainder // int(new_shape[2])
        iz = remainder - iy * int(new_shape[2])
        points = (
            new_origin.reshape(1, 3)
            + np.column_stack((ix, iy, iz)) * float(voxel_m)
        )
        flat[linear] = evaluator.contains(points)
        queried += int(len(points))
    region["origin"] = new_origin
    region["occupancy"] = expanded
    return {
        "extended_voxels": int(len(missing)),
        "contains_query_points": int(queried),
    }


def _slice_lite_occupancy_region(
    region: Dict[str, Any],
    *,
    origin: np.ndarray,
    upper: np.ndarray,
    shape: np.ndarray,
    radius: float,
    voxel_m: float,
    extension: Dict[str, int],
    started: float,
) -> Any:
    region_origin = np.asarray(region["origin"], dtype=np.float64).reshape(3)
    offset = np.rint(
        (origin - region_origin) / float(voxel_m)
    ).astype(np.int64)
    slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(offset, shape)
    )
    occupancy = np.asarray(region["occupancy"], dtype=bool)[slices].copy()
    if tuple(occupancy.shape) != tuple(shape.tolist()):
        raise RuntimeError(
            f"invalid cached occupancy slice {occupancy.shape} != {tuple(shape)}"
        )
    cached_metadata = dict(region.get("metadata") or {})
    metadata = {
        **cached_metadata,
        **_lite_local_component_metadata(
            region.get("component_bounds"),
            origin=origin,
            upper=upper,
        ),
        "occupancy_query_bounds": np.stack((origin, upper), axis=0).tolist(),
        "origin": origin.tolist(),
        "shape": [int(value) for value in shape],
        "grid_voxels": int(np.prod(shape, dtype=np.int64)),
        "occupied_voxels": int(occupancy.sum()),
        "candidate_envelope_radius_m": float(radius),
        "elapsed_s": float(time.perf_counter() - started),
        "lite_region_cache": {
            "hit": True,
            "region_shape": [
                int(value) for value in region["occupancy"].shape
            ],
            **extension,
        },
    }
    return production.DenseSceneOccupancy(
        origin=origin.copy(),
        occupancy=occupancy,
        voxel_m=float(voxel_m),
        metadata=metadata,
    )


def _build_local_scene_occupancy_lite(mesh, **kwargs):
    enabled = os.environ.get(
        "OFFICIAL_V2_LITE_OCCUPANCY_CACHE",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    local_kwargs = dict(kwargs)
    local_kwargs["query_offsets"] = [
        np.asarray(part, dtype=np.float64).reshape(-1, 3)
        for part in kwargs["query_offsets"]
    ]
    if isinstance(mesh, ProjectiveDepthScene):
        cache_key = (
            _lite_occupancy_cache_key(mesh, local_kwargs)
            if enabled
            else ""
        )

        def build_projective():
            test_mode = os.environ.get(
                "OFFICIAL_V2_RGBD_LITE_TEST_MODE",
                "0",
            ).strip().lower() not in {"0", "false", "no", "off"}
            cpu_reference = os.environ.get(
                "OFFICIAL_V2_LITE_ALLOW_CPU_REFERENCE",
                "0",
            ).strip().lower() not in {"0", "false", "no", "off"}
            result = build_projective_local_occupancy(
                mesh,
                hit=local_kwargs["hit"],
                query_offsets=local_kwargs["query_offsets"],
                axial_z_m=np.asarray(
                    local_kwargs.get("axial_z_m", production.AXIAL_Z_M),
                    dtype=np.float64,
                ),
                anchor_radius_m=float(
                    local_kwargs.get("anchor_radius_m", 0.0)
                ),
                voxel_m=float(
                    local_kwargs.get("voxel_m", production.VOXEL_M)
                ),
                ctx=local_kwargs.get("ctx"),
                requested_device=os.environ.get(
                    "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE",
                    "cuda:0",
                ),
                allow_cpu_reference=bool(test_mode and cpu_reference),
                back_extrusion_m=float(
                    os.environ.get(
                        "OFFICIAL_V2_LITE_PROJECTIVE_BACK_EXTRUSION_M",
                        "0.060",
                    )
                ),
                front_tolerance_m=float(
                    os.environ.get(
                        "OFFICIAL_V2_LITE_PROJECTIVE_FRONT_TOLERANCE_M",
                        "0.002",
                    )
                ),
                pixel_radius=int(
                    os.environ.get(
                        "OFFICIAL_V2_LITE_PROJECTIVE_PIXEL_RADIUS",
                        "1",
                    )
                ),
                validate_hit=bool(local_kwargs.get("validate_hit", True)),
            )
            return production.DenseSceneOccupancy(
                origin=np.asarray(result.origin, dtype=np.float64).copy(),
                occupancy=np.asarray(result.occupancy, dtype=bool).copy(),
                voxel_m=float(result.voxel_m),
                metadata=dict(result.metadata),
            )

        if not cache_key:
            return build_projective()
        limit = max(
            1,
            int(os.environ.get("OFFICIAL_V2_LITE_OCCUPANCY_CACHE_SIZE", "4")),
        )
        with _LITE_OCCUPANCY_CACHE_LOCK:
            cached = _LITE_OCCUPANCY_CACHE.get(cache_key)
            if cached is None:
                result = build_projective()
                cached = (
                    np.asarray(result.origin, dtype=np.float64).copy(),
                    np.asarray(result.occupancy, dtype=bool).copy(),
                    float(result.voxel_m),
                    dict(result.metadata),
                )
                _LITE_OCCUPANCY_CACHE[cache_key] = cached
                _LITE_OCCUPANCY_CACHE.move_to_end(cache_key)
                while len(_LITE_OCCUPANCY_CACHE) > limit:
                    _LITE_OCCUPANCY_CACHE.popitem(last=False)
                return result
            _LITE_OCCUPANCY_CACHE.move_to_end(cache_key)
            origin, occupancy, voxel_m, metadata = cached
            return production.DenseSceneOccupancy(
                origin=origin.copy(),
                occupancy=occupancy.copy(),
                voxel_m=float(voxel_m),
                metadata=dict(metadata),
            )

    if _lite_geometry_backend() == LITE_GEOMETRY_BACKEND_WARP:
        cache_key = (
            _lite_occupancy_cache_key(mesh, local_kwargs)
            if enabled
            else ""
        )

        def build_warp():
            test_mode = os.environ.get(
                "OFFICIAL_V2_RGBD_LITE_TEST_MODE",
                "0",
            ).strip().lower() not in {"0", "false", "no", "off"}
            cpu_reference = os.environ.get(
                "OFFICIAL_V2_LITE_ALLOW_CPU_REFERENCE",
                "0",
            ).strip().lower() not in {"0", "false", "no", "off"}
            result = build_warp_local_occupancy(
                mesh,
                hit=local_kwargs["hit"],
                query_offsets=local_kwargs["query_offsets"],
                axial_z_m=np.asarray(
                    local_kwargs.get("axial_z_m", production.AXIAL_Z_M),
                    dtype=np.float64,
                ),
                anchor_radius_m=float(
                    local_kwargs.get("anchor_radius_m", 0.0)
                ),
                voxel_m=float(
                    local_kwargs.get("voxel_m", production.VOXEL_M)
                ),
                ctx=local_kwargs.get("ctx"),
                requested_device=os.environ.get(
                    "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE",
                    "cuda:0",
                ),
                allow_cpu_reference=bool(test_mode and cpu_reference),
                validate_hit=bool(local_kwargs.get("validate_hit", True)),
            )
            dense = production.DenseSceneOccupancy(
                origin=np.asarray(result.origin, dtype=np.float64).copy(),
                occupancy=np.asarray(result.occupancy, dtype=bool).copy(),
                voxel_m=float(result.voxel_m),
                metadata=dict(result.metadata),
            )
            if result.device_dense is not None:
                device_name = str(
                    result.metadata["pose_count_device_required"]
                )
                dense._device_cache[device_name] = {
                    "dense": result.device_dense,
                    "origin": result.device_origin,
                    "resources": result.device_resources,
                }
            return dense

        if not cache_key:
            return build_warp()
        limit = max(
            1,
            int(os.environ.get("OFFICIAL_V2_LITE_OCCUPANCY_CACHE_SIZE", "4")),
        )
        with _LITE_OCCUPANCY_CACHE_LOCK:
            cached = _LITE_OCCUPANCY_CACHE.get(cache_key)
            if cached is None:
                result = build_warp()
                cached = (
                    np.asarray(result.origin, dtype=np.float64).copy(),
                    np.asarray(result.occupancy, dtype=bool).copy(),
                    float(result.voxel_m),
                    dict(result.metadata),
                    {
                        key: dict(value)
                        for key, value in result._device_cache.items()
                    },
                )
                _LITE_OCCUPANCY_CACHE[cache_key] = cached
                _LITE_OCCUPANCY_CACHE.move_to_end(cache_key)
                while len(_LITE_OCCUPANCY_CACHE) > limit:
                    _LITE_OCCUPANCY_CACHE.popitem(last=False)
                return result
            _LITE_OCCUPANCY_CACHE.move_to_end(cache_key)
            origin, occupancy, voxel_m, metadata, device_cache = cached
            dense = production.DenseSceneOccupancy(
                origin=origin.copy(),
                occupancy=occupancy.copy(),
                voxel_m=float(voxel_m),
                metadata=dict(metadata),
            )
            dense._device_cache.update(
                {
                    key: dict(value)
                    for key, value in device_cache.items()
                }
            )
            return dense

    cache_key = (
        _lite_occupancy_cache_key(mesh, local_kwargs)
        if enabled
        else ""
    )
    legacy_kwargs = dict(local_kwargs)
    # The legacy mesh builder never had a clicked-surface contract. Keep its
    # established call signature while retaining validate_hit in the cache key.
    legacy_kwargs.pop("validate_hit", None)
    if not cache_key:
        return production.build_local_scene_occupancy(
            mesh,
            **legacy_kwargs,
        )
    limit = max(
        1,
        int(os.environ.get("OFFICIAL_V2_LITE_OCCUPANCY_CACHE_SIZE", "4")),
    )
    region_limit = max(
        1,
        int(
            os.environ.get(
                "OFFICIAL_V2_LITE_OCCUPANCY_REGION_CACHE_SIZE",
                "5",
            )
        ),
    )
    with _LITE_OCCUPANCY_CACHE_LOCK:
        cached = _LITE_OCCUPANCY_CACHE.get(cache_key)
        if cached is None:
            region_key = _lite_occupancy_region_cache_key(mesh, local_kwargs)
            region_enabled = os.environ.get(
                "OFFICIAL_V2_LITE_OCCUPANCY_REGION_CACHE",
                "1",
            ).strip().lower() not in {"0", "false", "no", "off"}
            region = (
                _LITE_OCCUPANCY_REGION_CACHE.get(region_key)
                if region_enabled and region_key
                else None
            )
            if region is None:
                result = production.build_local_scene_occupancy(
                    mesh,
                    **legacy_kwargs,
                )
                if region_enabled and region_key:
                    _LITE_OCCUPANCY_REGION_CACHE[region_key] = {
                        "origin": np.asarray(
                            result.origin,
                            dtype=np.float64,
                        ).copy(),
                        "occupancy": np.asarray(
                            result.occupancy,
                            dtype=bool,
                        ).copy(),
                        "metadata": dict(result.metadata),
                        "component_bounds": (
                            _lite_watertight_component_bounds(mesh)
                        ),
                    }
                    _LITE_OCCUPANCY_REGION_CACHE.move_to_end(region_key)
            else:
                region_started = time.perf_counter()
                origin, upper, shape, radius = (
                    _lite_local_occupancy_grid_spec(local_kwargs)
                )
                try:
                    extension = _extend_lite_occupancy_region(
                        mesh,
                        region,
                        required_origin=origin,
                        required_upper=upper,
                        voxel_m=float(
                            local_kwargs.get("voxel_m", production.VOXEL_M)
                        ),
                        contains_chunk=int(
                            local_kwargs.get("contains_chunk", 200_000)
                        ),
                    )
                    result = _slice_lite_occupancy_region(
                        region,
                        origin=origin,
                        upper=upper,
                        shape=shape,
                        radius=radius,
                        voxel_m=float(
                            local_kwargs.get("voxel_m", production.VOXEL_M)
                        ),
                        extension=extension,
                        started=region_started,
                    )
                    _LITE_OCCUPANCY_REGION_CACHE.move_to_end(region_key)
                except OverflowError:
                    result = production.build_local_scene_occupancy(
                        mesh,
                        **local_kwargs,
                    )
                    _LITE_OCCUPANCY_REGION_CACHE[region_key] = {
                        "origin": np.asarray(
                            result.origin,
                            dtype=np.float64,
                        ).copy(),
                        "occupancy": np.asarray(
                            result.occupancy,
                            dtype=bool,
                        ).copy(),
                        "metadata": dict(result.metadata),
                    }
                    _LITE_OCCUPANCY_REGION_CACHE.move_to_end(region_key)
            while len(_LITE_OCCUPANCY_REGION_CACHE) > region_limit:
                _LITE_OCCUPANCY_REGION_CACHE.popitem(last=False)
            cached = (
                np.asarray(result.origin, dtype=np.float64).copy(),
                np.asarray(result.occupancy, dtype=bool).copy(),
                float(result.voxel_m),
                dict(result.metadata),
            )
            _LITE_OCCUPANCY_CACHE[cache_key] = cached
            _LITE_OCCUPANCY_CACHE.move_to_end(cache_key)
            while len(_LITE_OCCUPANCY_CACHE) > limit:
                _LITE_OCCUPANCY_CACHE.popitem(last=False)
            return result
        _LITE_OCCUPANCY_CACHE.move_to_end(cache_key)
        origin, occupancy, voxel_m, metadata = cached
        return production.DenseSceneOccupancy(
            origin=origin.copy(),
            occupancy=occupancy.copy(),
            voxel_m=float(voxel_m),
            metadata=dict(metadata),
        )


def _lite_render_cache_key(mesh, kwargs: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(LITE_RENDER_CACHE_VERSION.encode("ascii"))

    def update_text(value: Any) -> None:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    def update_array(value: Any) -> None:
        array = np.ascontiguousarray(
            np.asarray(value, dtype=np.float64)
        )
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())

    scene_key = getattr(mesh, "_official_v2_lite_scene_key", None)
    if scene_key:
        update_text(scene_key)
    else:
        update_array(np.asarray(mesh.triangles, dtype=np.float64))
    update_array(kwargs["hit"])
    update_array(kwargs["anchors"])
    best_pose = kwargs["best_pose"]
    update_array(best_pose["R"])
    update_array(best_pose["eef_pos"])
    for key in (
        "axial_point",
        "grasp_vol_cm3",
        "overlap_vol_cm3",
        "inflated_overlap_vol_cm3",
        "normal_alignment_deg",
    ):
        value = best_pose.get(key, float("nan"))
        update_text(key)
        update_text(float(value).hex() if isinstance(value, float) else value)
    update_array(kwargs["gripper_voxels"])
    for name, points in kwargs["inflation_regions"].items():
        update_text(name)
        update_array(points)
    threshold_audit = kwargs["threshold_audit"]
    sorted_cm3 = threshold_audit.get("sorted_cm3")
    update_array([] if sorted_cm3 is None else sorted_cm3)
    for key in (
        "threshold_cm3",
        "passing_count",
        "input_count",
        "overlap_passing_count",
    ):
        value = threshold_audit[key]
        update_text(key)
        update_text(float(value).hex() if isinstance(value, float) else value)
    normal = kwargs.get("outward_normal_world")
    update_text("normal:none" if normal is None else "normal:present")
    if normal is not None:
        update_array(normal)
    update_text(production.BUILD)
    update_text(os.path.splitext(str(kwargs["output_path"]))[1].lower())
    return digest.hexdigest()


def _render_and_cache_lite_three_views(
    mesh,
    kwargs: Dict[str, Any],
    *,
    cache_key: str,
    cache_limit: int,
) -> Optional[str]:
    rendered = production.render_local_three_views(mesh, **kwargs)
    if not rendered or not os.path.isfile(rendered):
        return rendered
    with open(rendered, "rb") as stream:
        png_bytes = stream.read()
    with _LITE_RENDER_CACHE_LOCK:
        _LITE_RENDER_CACHE[cache_key] = png_bytes
        _LITE_RENDER_CACHE.move_to_end(cache_key)
        while len(_LITE_RENDER_CACHE) > int(cache_limit):
            _LITE_RENDER_CACHE.popitem(last=False)
    return rendered


def _render_local_three_views_lite(mesh, **kwargs) -> Optional[str]:
    if isinstance(mesh, ProjectiveDepthScene):
        enabled = os.environ.get(
            "OFFICIAL_V2_LITE_PROJECTIVE_THREE_VIEW",
            "0",
        ).strip().lower() not in {"0", "false", "no", "off"}
        if not enabled:
            return None
        raise RuntimeError(
            "projective occupancy has no triangle mesh for three-view rendering"
        )
    enabled = os.environ.get(
        "OFFICIAL_V2_LITE_RENDER_CACHE",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return production.render_local_three_views(mesh, **kwargs)
    cache_key = _lite_render_cache_key(mesh, kwargs)
    output_path = os.path.abspath(str(kwargs["output_path"]))
    limit = max(
        1,
        int(os.environ.get("OFFICIAL_V2_LITE_RENDER_CACHE_SIZE", "4")),
    )
    with _LITE_RENDER_CACHE_LOCK:
        png_bytes = _LITE_RENDER_CACHE.get(cache_key)
        if png_bytes is not None:
            _LITE_RENDER_CACHE.move_to_end(cache_key)
    if png_bytes is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as stream:
            stream.write(png_bytes)
        return output_path
    asynchronous = os.environ.get(
        "OFFICIAL_V2_LITE_ASYNC_THREE_VIEW",
        "0",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if asynchronous:
        future = _lite_render_executor().submit(
            _render_and_cache_lite_three_views,
            mesh,
            dict(kwargs),
            cache_key=cache_key,
            cache_limit=limit,
        )
        with _LITE_RENDER_EXECUTOR_LOCK:
            _LITE_RENDER_FUTURES.append(future)
        return output_path
    return _render_and_cache_lite_three_views(
        mesh,
        dict(kwargs),
        cache_key=cache_key,
        cache_limit=limit,
    )


def _v52_planner_build(build: str) -> str:
    source = str(build)
    old = "organized_shell_v8"
    backend = _lite_geometry_backend()
    if backend == LITE_GEOMETRY_BACKEND_PROJECTIVE:
        representative = "projective_cuda_depth_shell_v1_experimental"
    elif backend == LITE_GEOMETRY_BACKEND_WARP:
        representative = "representative_v53_warp_cuda_v1"
    else:
        representative = "representative_v53_legacy_cpu"
    if old in source:
        return source.replace(old, representative)
    return f"{source}_{representative}"


def _lite_gripper_geometry_fingerprint(voxel_m: float) -> str:
    digest = hashlib.sha256()
    digest.update(LITE_GRIPPER_GEOMETRY_CACHE_VERSION.encode("ascii"))
    digest.update(float(voxel_m).hex().encode("ascii"))
    geometry = gripper_geometry(float(voxel_m))
    for key in ("gripper_voxels", "opening_voxels", "axial_z_m"):
        digest.update(
            np.asarray(geometry[key], dtype=np.float64).tobytes()
        )
    return digest.hexdigest()[:20]


def _ensure_lite_gripper_geometry_cache(
    voxel_m: float,
) -> Dict[str, Any]:
    """Restore exact static gripper arrays after a skill-module reload."""
    voxel = float(voxel_m)
    geometry = gripper_geometry(voxel)
    fingerprint = _lite_gripper_geometry_fingerprint(voxel)
    return {
        "source": "submission_static_asset",
        "cache_hit": True,
        "fingerprint": fingerprint,
        "voxel_m": voxel,
        "gripper_voxels": int(len(geometry["gripper_voxels"])),
        "opening_voxels": int(len(geometry["opening_voxels"])),
    }


def evaluate_pose_overlap_volumes_lite(
    *,
    session: Mapping[str, Any],
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    poses: Sequence[Mapping[str, Any]],
    prepared_scene=None,
    ctx=None,
    check_cancelled=None,
    progress_callback=None,
) -> Dict[str, Any]:
    """Evaluate only the two Lite collision gates for fixed EEF poses.

    The strict grasp planner also queries the opening volume to rank grasp
    quality.  A move endpoint has no grasp-quality objective, so this helper
    deliberately omits that third query while retaining the exact 3 mm static
    gripper, component-aware inflation, occupancy implementation, and overlap
    thresholds used by ``plan_grasp_point_filter_rgbd_lite``.

    Original overlap is evaluated for every pose in one batch.  Inflated
    overlap is then evaluated only for poses that pass the original gate.  The
    short circuit is logically equivalent to evaluating both conjunction
    terms for every pose and avoids unnecessary transformed-voxel queries.
    """

    started = time.perf_counter()

    def checkpoint(stage):
        if check_cancelled is not None:
            check_cancelled()
        if progress_callback is not None:
            progress_callback(stage)

    checkpoint("input_validation")
    raw_poses = list(poses)
    if not raw_poses:
        raise ValueError("Lite pose overlap evaluation requires at least one pose")

    normalized_poses: List[Dict[str, Any]] = []
    for pose_index, raw_pose in enumerate(raw_poses):
        if not isinstance(raw_pose, Mapping):
            raise TypeError(f"Lite overlap pose {pose_index} must be an object")
        try:
            position = np.asarray(
                raw_pose["eef_pos"], dtype=np.float64
            ).reshape(3)
            quaternion = np.asarray(
                raw_pose["quat"], dtype=np.float64
            ).reshape(4)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Lite overlap pose {pose_index} requires eef_pos[3] and quat[4]"
            ) from exc
        if not np.all(np.isfinite(position)) or not np.all(
            np.isfinite(quaternion)
        ):
            raise ValueError(
                f"Lite overlap pose {pose_index} contains non-finite values"
            )
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= 1.0e-12:
            raise ValueError(f"Lite overlap pose {pose_index} has zero quaternion")
        quaternion = quaternion / quaternion_norm
        normalized_poses.append(
            {
                "eef_pos": position.copy(),
                "quat": quaternion.copy(),
                "R": production._quat_to_mat_xyzw(quaternion.copy()),
            }
        )

    camera_position = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    camera_quaternion = np.asarray(
        camera_quat_xyzw, dtype=np.float64
    ).reshape(4)
    if not np.all(np.isfinite(camera_position)) or not np.all(
        np.isfinite(camera_quaternion)
    ):
        raise ValueError("Lite overlap camera pose contains non-finite values")
    if float(np.linalg.norm(camera_quaternion)) <= 1.0e-12:
        raise ValueError("Lite overlap camera quaternion has zero norm")

    checkpoint("rgbd_reconstruction")
    reconstruction_started = time.perf_counter()
    if prepared_scene is None:
        prepared_scene = _cached_reconstruct_v52_scene_from_session(
            dict(session),
            camera_pos=camera_position.copy(),
            camera_quat_xyzw=camera_quaternion.copy(),
            focal_length=float(focal_length),
            horizontal_aperture=float(horizontal_aperture),
        )
    if not isinstance(prepared_scene, (tuple, list)) or len(prepared_scene) != 4:
        raise TypeError("Lite overlap prepared scene has an invalid contract")
    mesh, _rgb, _depth, reconstruction_metadata = prepared_scene
    reconstruction_elapsed_s = float(
        time.perf_counter() - reconstruction_started
    )

    checkpoint("static_gripper_geometry")
    geometry = gripper_geometry(production.VOXEL_M)
    gripper_voxels = np.asarray(
        geometry["gripper_voxels"], dtype=np.float64
    ).reshape(-1, 3)
    inflated_voxels, _inflation_regions, inflation_metadata = (
        production.inflated_gripper_voxels_eef(
            gripper_voxels,
            component_voxels=geometry["components"],
            voxel_m=production.VOXEL_M,
            camera_inflate_m=production.CAMERA_INFLATE_M,
            finger_inward_inflate_m=production.FINGER_INWARD_INFLATE_M,
            return_regions=True,
        )
    )
    inflated_voxels = np.asarray(
        inflated_voxels, dtype=np.float64
    ).reshape(-1, 3)
    if len(inflated_voxels) < len(gripper_voxels):
        raise RuntimeError("Lite inflated gripper is not a superset of the base")

    positions = np.stack(
        [pose["eef_pos"] for pose in normalized_poses], axis=0
    )
    envelope_center = np.mean(positions, axis=0)
    envelope_radius = float(
        np.max(np.linalg.norm(positions - envelope_center[None, :], axis=1))
    )
    checkpoint("occupancy_grid")
    occupancy = _build_local_scene_occupancy_lite(
        mesh,
        hit=envelope_center,
        query_offsets=(gripper_voxels, inflated_voxels),
        # These are already complete EEF poses; no grasp-axis samples are
        # needed merely to bound the local query grid.
        axial_z_m=np.asarray([0.0], dtype=np.float64),
        anchor_radius_m=envelope_radius,
        voxel_m=production.VOXEL_M,
        ctx=ctx,
        # This is an EEF-pose envelope center, not the user's RGB-D click.
        # Requiring a surface here incorrectly rejects safe poses in free space.
        validate_hit=False,
    )

    checkpoint("original_overlap")
    original_counts, original_query = occupancy.counts_for_poses(
        normalized_poses,
        gripper_voxels,
    )
    original_counts = np.asarray(original_counts, dtype=np.int64).reshape(-1)
    if original_counts.shape != (len(normalized_poses),) or np.any(
        original_counts < 0
    ):
        raise RuntimeError("Lite original overlap query returned invalid counts")

    voxel_cm3 = float(occupancy.voxel_m) ** 3 * 1.0e6
    original_volumes = original_counts.astype(np.float64) * voxel_cm3
    original_pass_mask = (
        original_volumes
        <= float(production.ORIGINAL_OVERLAP_MAX_CM3) + 1.0e-12
    )
    original_passing_indices = np.flatnonzero(original_pass_mask)
    inflated_counts_by_index: Dict[int, int] = {}
    if len(original_passing_indices):
        checkpoint("inflated_overlap")
        inflated_counts, inflated_query = occupancy.counts_for_poses(
            [normalized_poses[int(index)] for index in original_passing_indices],
            inflated_voxels,
        )
        inflated_counts = np.asarray(
            inflated_counts, dtype=np.int64
        ).reshape(-1)
        if inflated_counts.shape != (len(original_passing_indices),) or np.any(
            inflated_counts < 0
        ):
            raise RuntimeError("Lite inflated overlap query returned invalid counts")
        inflated_counts_by_index = {
            int(index): int(count)
            for index, count in zip(original_passing_indices, inflated_counts)
        }
    else:
        inflated_query = {
            "device": "none",
            "elapsed_s": 0.0,
            "pose_count": 0,
            "unique_query_pose_count": 0,
            "query_voxels_per_pose": int(len(inflated_voxels)),
            "skipped": True,
            "reason": "no_pose_passed_original_overlap",
        }

    pose_reports: List[Dict[str, Any]] = []
    passing_indices: List[int] = []
    for pose_index, (pose, original_count, original_volume) in enumerate(
        zip(normalized_poses, original_counts, original_volumes)
    ):
        original_ok = bool(original_pass_mask[pose_index])
        inflated_count = inflated_counts_by_index.get(pose_index)
        inflated_volume = (
            None
            if inflated_count is None
            else float(inflated_count * voxel_cm3)
        )
        inflated_ok = bool(
            original_ok
            and inflated_volume is not None
            and inflated_volume
            < float(production.INFLATED_OVERLAP_MAX_CM3)
        )
        ok = bool(original_ok and inflated_ok)
        if ok:
            passing_indices.append(int(pose_index))
        if not original_ok:
            reason = "original_overlap_exceeded"
        elif not inflated_ok:
            reason = "inflated_overlap_exceeded"
        else:
            reason = None
        pose_reports.append(
            {
                "pose_index": int(pose_index),
                "eef_position_robot_base_m": pose["eef_pos"].astype(float).tolist(),
                "eef_quaternion_xyzw": pose["quat"].astype(float).tolist(),
                "overlap_vox": int(original_count),
                "overlap_vol_cm3": float(original_volume),
                "original_overlap_ok": original_ok,
                "inflated_overlap_vox": (
                    None if inflated_count is None else int(inflated_count)
                ),
                "inflated_overlap_vol_cm3": inflated_volume,
                "inflated_overlap_ok": inflated_ok,
                "inflated_query_skipped": not original_ok,
                "inflated_query_skipped_reason": (
                    "original_overlap_failed" if not original_ok else None
                ),
                "ok": ok,
                "reason": reason,
            }
        )

    checkpoint("complete")
    reconstruction_report = dict(reconstruction_metadata or {})
    return {
        "ok": True,
        "build": LITE_POSE_OVERLAP_FILTER_BUILD,
        "source_planner": "plan_grasp_point_filter_rgbd_lite",
        "geometry_backend": _lite_geometry_backend(),
        "occupancy_center_semantics": "eef_pose_envelope_center",
        "occupancy_center_surface_validation": False,
        "pose_count": int(len(normalized_poses)),
        "passing_pose_count": int(len(passing_indices)),
        "rejected_pose_count": int(len(normalized_poses) - len(passing_indices)),
        "passing_pose_indices": passing_indices,
        "voxel_m": float(occupancy.voxel_m),
        "voxel_volume_cm3": voxel_cm3,
        "original_overlap_threshold_cm3": float(
            production.ORIGINAL_OVERLAP_MAX_CM3
        ),
        "original_overlap_comparison": "<=",
        "inflated_overlap_threshold_cm3": float(
            production.INFLATED_OVERLAP_MAX_CM3
        ),
        "inflated_overlap_comparison": "<",
        "camera_inflate_m": float(production.CAMERA_INFLATE_M),
        "finger_inward_inflate_m": float(
            production.FINGER_INWARD_INFLATE_M
        ),
        "gripper_voxels": int(len(gripper_voxels)),
        "inflated_gripper_voxels": int(len(inflated_voxels)),
        "candidate_envelope_center_robot_base_m": (
            envelope_center.astype(float).tolist()
        ),
        "candidate_envelope_radius_m": envelope_radius,
        "original_overlap_query": dict(original_query or {}),
        "inflated_overlap_query": dict(inflated_query or {}),
        "inflated_query_pose_count": int(len(original_passing_indices)),
        "query_short_circuit": (
            "inflated_overlap_only_after_original_overlap_pass"
        ),
        "filter_result_equivalent_to_full_conjunction": True,
        "grasp_opening_volume_queried": False,
        "inflation": dict(inflation_metadata or {}),
        "occupancy": dict(getattr(occupancy, "metadata", {}) or {}),
        "reconstruction": reconstruction_report,
        "reconstruction_elapsed_s": reconstruction_elapsed_s,
        "elapsed_s": float(time.perf_counter() - started),
        "poses": pose_reports,
        "allowed_inputs": [
            "frozen_evaluator_rgb",
            "frozen_evaluator_depth_linear",
            "frozen_evaluator_camera_relative_pose",
            "submission_local_static_gripper_geometry",
            "submission_local_planned_eef_pose",
        ],
        "segmentation_used": False,
        "object_identity_used": False,
        "simulator_mesh_used": False,
        "direct_simulator_mutation": False,
    }


def _apply_camera_face_preserving_anchor_lite(
    poses: List[Dict[str, Any]],
    *,
    world,
    ctx=None,
) -> Dict[str, int]:
    """Apply the production rule with one immutable forward lookup per batch."""
    from .grasp_geometry_local import ensure_camera_face_forward

    forward = np.asarray(
        getattr(world, "robot_forward", [1.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(3)

    flipped = 0
    skipped = 0
    for pose in poses:
        quat, audit = ensure_camera_face_forward(
            pose["quat"],
            forward=forward,
        )
        rotation = production._quat_to_mat_xyzw(quat)
        anchor = np.asarray(pose["anchor"], dtype=np.float64)
        anchor_local = np.asarray(
            pose["anchor_local"],
            dtype=np.float64,
        )
        pose["quat"] = quat
        pose["R"] = rotation
        pose["eef_pos"] = anchor - rotation @ anchor_local
        pose["camera_face"] = audit
        if audit.get("flipped"):
            pose.pop("ik_warm_start_q_by_arm", None)
        flipped += int(bool(audit.get("flipped")))
        skipped += int(bool(audit.get("skipped")))
    audit = {
        "pose_count": len(poses),
        "flipped": flipped,
        "skipped": skipped,
    }
    if ctx is not None:
        ctx.log(
            f"  [grasp_point_filter_rgbd] camera-face poses={len(poses)} "
            f"flipped={flipped} skipped={skipped}"
        )
    return audit


def _lite_ik_worker_policy(
    poses: Sequence[Dict[str, Any]],
    *,
    plan_arm: str,
) -> str:
    """Use the wider graph only when every requested solve has a warm seed."""
    requested_arm = str(plan_arm or "any").strip().lower()
    if requested_arm not in ("left", "right"):
        return LITE_IK_WORKER_POLICY
    rows = list(poses)
    if not rows:
        return LITE_IK_WORKER_POLICY
    for pose in rows:
        warm = pose.get("ik_warm_start_q_by_arm") or {}
        q_arm = warm.get(requested_arm)
        if q_arm is None:
            return LITE_IK_WORKER_POLICY
        q = np.asarray(q_arm, dtype=np.float64).reshape(-1)
        if len(q) not in (7, 8) or not np.all(np.isfinite(q)):
            return LITE_IK_WORKER_POLICY
    return LITE_WARM_IK_WORKER_POLICY


def _rotation_angle_deg(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> float:
    left_rotation = np.asarray(left["R"], dtype=np.float64).reshape(3, 3)
    right_rotation = np.asarray(right["R"], dtype=np.float64).reshape(3, 3)
    relative = left_rotation.T @ right_rotation
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _position_distance_mm(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left["eef_pos"], dtype=np.float64)
            - np.asarray(right["eef_pos"], dtype=np.float64)
        )
        * 1000.0
    )


def _same_representative_cell(
    pose: Dict[str, Any],
    representative: Dict[str, Any],
    *,
    position_limit_mm: float,
    angle_limit_deg: float,
) -> bool:
    if int(pose.get("ai", -1)) != int(representative.get("ai", -1)):
        return False
    return bool(
        _position_distance_mm(pose, representative)
        <= float(position_limit_mm) + 1e-12
        and _rotation_angle_deg(pose, representative)
        <= float(angle_limit_deg) + 1e-12
    )


def _strict_safe_pose(pose: Dict[str, Any]) -> bool:
    return bool(
        float(pose.get("env_overlap_cm3", float("inf")))
        <= production.ORIGINAL_OVERLAP_MAX_CM3 + 1e-12
        and float(
            pose.get(
                "prefilter_inflated_overlap_cm3",
                float("inf"),
            )
        )
        < production.INFLATED_OVERLAP_MAX_CM3
    )


def _source_label(pose: Dict[str, Any]) -> str:
    if (
        pose.get("pre_ik_translation_refined")
        or pose.get("translation_refined")
    ):
        return "translation"
    if pose.get("interpolated"):
        return "interpolated"
    if pose.get("refined"):
        return "refined"
    if pose.get("opening_contact_refined"):
        return "contact"
    return "raw"


def _metric_summary(pose: Dict[str, Any]) -> Dict[str, Any]:
    angle = float(pose.get("normal_alignment_deg", float("inf")))
    return {
        "base_rank": int(pose.get("lite_base_rank", -1)),
        "cluster_id": int(pose.get("lite_cluster_id", -1)),
        "cluster_member_rank": int(
            pose.get("lite_cluster_member_rank", -1)
        ),
        "ai": int(pose.get("ai", -1)),
        "source": _source_label(pose),
        "grasp_cm3": float(pose.get("prefilter_grasp_cm3", 0.0)),
        "original_overlap_cm3": float(
            pose.get("env_overlap_cm3", float("inf"))
        ),
        "inflated_overlap_cm3": float(
            pose.get(
                "prefilter_inflated_overlap_cm3",
                float("inf"),
            )
        ),
        "normal_alignment_deg": (
            angle if math.isfinite(angle) else None
        ),
    }


def _lineage_summary(
    pose: Dict[str, Any],
    *,
    rank: Optional[int] = None,
) -> Dict[str, Any]:
    summary = _metric_summary(pose)
    summary.update(
        {
            "rank": int(rank) if rank is not None else None,
            "micro_seed_index": int(
                pose.get("micro_seed_index", -1)
            ),
            "reachable_se3_pair_id": int(
                pose.get("reachable_se3_pair_id", -1)
            ),
            "reachable_se3_generation": int(
                pose.get("reachable_se3_generation", 0)
            ),
            "translation_seed_index": int(
                pose.get("translation_seed_index", -1)
            ),
            "translation_refine_generation": int(
                pose.get("translation_refine_generation", 0)
            ),
            "translation_refined": bool(
                pose.get("translation_refined", False)
            ),
            "micro_refined": bool(pose.get("micro_refined", False)),
            "reachable_se3_interpolated": bool(
                pose.get("reachable_se3_interpolated", False)
            ),
            "reachability_bridge": bool(
                pose.get("reachability_bridge", False)
            ),
        }
    )
    return summary


def _coverage_count(
    poses: Sequence[Dict[str, Any]],
    field: str,
) -> int:
    return int(
        len(
            {
                int(pose.get(field, -1))
                for pose in poses
                if int(pose.get(field, -1)) >= 0
            }
        )
    )


def _stage_coverage_summary(
    poses: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    rows = list(poses)
    lineage_fields = (
        "micro_seed_index",
        "reachable_se3_pair_id",
        "translation_seed_index",
    )
    total_coverage = {
        field: _coverage_count(rows, field)
        for field in lineage_fields
    }
    best_grasp = max(
        (
            float(pose.get("prefilter_grasp_cm3", 0.0))
            for pose in rows
        ),
        default=0.0,
    )
    finite_angles = [
        float(pose.get("normal_alignment_deg", float("inf")))
        for pose in rows
        if math.isfinite(
            float(pose.get("normal_alignment_deg", float("inf")))
        )
    ]
    cap_summaries: Dict[str, Dict[str, Any]] = {}
    for cap in LITE_DIAGNOSTIC_CAPS:
        capped = rows[: min(int(cap), len(rows))]
        capped_best_grasp = max(
            (
                float(pose.get("prefilter_grasp_cm3", 0.0))
                for pose in capped
            ),
            default=0.0,
        )
        capped_angles = [
            float(
                pose.get(
                    "normal_alignment_deg",
                    float("inf"),
                )
            )
            for pose in capped
            if math.isfinite(
                float(
                    pose.get(
                        "normal_alignment_deg",
                        float("inf"),
                    )
                )
            )
        ]
        cap_summaries[str(int(cap))] = {
            "selected_count": int(len(capped)),
            "coverage": {
                field: _coverage_count(capped, field)
                for field in lineage_fields
            },
            "coverage_ratio": {
                field: float(
                    _coverage_count(capped, field)
                    / max(total_coverage[field], 1)
                )
                for field in lineage_fields
            },
            "best_grasp_cm3": float(capped_best_grasp),
            "best_grasp_ratio": float(
                capped_best_grasp / max(best_grasp, 1e-12)
            ),
            "best_alignment_deg": (
                float(min(capped_angles)) if capped_angles else None
            ),
        }
    return {
        "input_count": int(len(rows)),
        "source_counts": dict(
            Counter(_source_label(pose) for pose in rows)
        ),
        "ai_count": _coverage_count(rows, "ai"),
        "coverage": total_coverage,
        "best_grasp_cm3": float(best_grasp),
        "best_alignment_deg": (
            float(min(finite_angles)) if finite_angles else None
        ),
        "caps": cap_summaries,
        "preview": [
            _lineage_summary(pose, rank=index)
            for index, pose in enumerate(rows[:20])
        ],
    }


def _record_stage_input(
    call_state: Dict[str, Any],
    stage_name: str,
    poses: Sequence[Dict[str, Any]],
) -> None:
    rows = list(poses)
    call_state["stage_pose_lists"][str(stage_name)] = rows
    call_state["stage_coverage"][str(stage_name)] = (
        _stage_coverage_summary(rows)
    )


def _float64_bytes(values, size: int) -> bytes:
    return np.asarray(values, dtype=np.float64).reshape(size).tobytes()


def _exact_ik_request_key(pose: Dict[str, Any]) -> tuple:
    """Key every value that can change a persistent-worker solve request."""
    safe = pose.get("ik_paired_safe_pose")
    if not isinstance(safe, dict):
        safe = production._paired_safe_target_for_pose(pose)
    warm = pose.get("ik_warm_start_q_by_arm") or {}

    def warm_key(arm: str):
        q_arm = warm.get(arm)
        if q_arm is None:
            return None
        q = np.asarray(q_arm, dtype=np.float64).reshape(-1)
        if len(q) not in (7, 8) or not np.all(np.isfinite(q)):
            return None
        return q.tobytes()

    return (
        _float64_bytes(pose["eef_pos"], 3),
        _float64_bytes(pose["quat"], 4),
        _float64_bytes(safe["eef_pos"], 3),
        _float64_bytes(safe.get("quat", pose["quat"]), 4),
        float(safe.get("pos_tol_m", 0.03)),
        float(safe.get("ori_tol_deg", 10.0)),
        float(safe.get("final_branch_gap_rad", 0.85)),
        warm_key("left"),
        warm_key("right"),
    )


def _ik_stage_reuse_summary(
    stage_pose_lists: Dict[str, Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Measure reusable IK work without changing any planner behavior."""
    pose_dedupe_key = production.pose_dedupe_key

    stage_rows = {
        str(stage): list(poses)
        for stage, poses in stage_pose_lists.items()
    }
    seen_geometry = set()
    seen_requests = set()
    stage_summary: Dict[str, Dict[str, Any]] = {}
    geometry_sets: Dict[str, set] = {}
    request_sets: Dict[str, set] = {}
    total_count = 0
    prior_geometry_reuse = 0
    prior_exact_request_reuse = 0
    geometry_only_reuse = 0

    for stage, poses in stage_rows.items():
        geometry_keys = [pose_dedupe_key(pose) for pose in poses]
        request_keys = [_exact_ik_request_key(pose) for pose in poses]
        geometry_set = set(geometry_keys)
        request_set = set(request_keys)
        geometry_sets[stage] = geometry_set
        request_sets[stage] = request_set

        stage_prior_geometry = sum(
            key in seen_geometry
            for key in geometry_keys
        )
        stage_prior_requests = sum(
            key in seen_requests
            for key in request_keys
        )
        stage_geometry_only = sum(
            geometry_key in seen_geometry
            and request_key not in seen_requests
            for geometry_key, request_key in zip(
                geometry_keys,
                request_keys,
            )
        )
        total_count += len(poses)
        prior_geometry_reuse += stage_prior_geometry
        prior_exact_request_reuse += stage_prior_requests
        geometry_only_reuse += stage_geometry_only
        stage_summary[stage] = {
            "input_count": int(len(poses)),
            "unique_geometry_count": int(len(geometry_set)),
            "unique_exact_request_count": int(len(request_set)),
            "within_stage_geometry_duplicate_count": int(
                len(poses) - len(geometry_set)
            ),
            "within_stage_exact_request_duplicate_count": int(
                len(poses) - len(request_set)
            ),
            "prior_stage_geometry_reuse_count": int(
                stage_prior_geometry
            ),
            "prior_stage_exact_request_reuse_count": int(
                stage_prior_requests
            ),
            "prior_stage_geometry_only_reuse_count": int(
                stage_geometry_only
            ),
        }
        seen_geometry.update(geometry_set)
        seen_requests.update(request_set)

    pairwise = []
    stage_names = list(stage_rows)
    for left_index, left in enumerate(stage_names):
        for right in stage_names[left_index + 1 :]:
            geometry_overlap = len(
                geometry_sets[left] & geometry_sets[right]
            )
            exact_overlap = len(
                request_sets[left] & request_sets[right]
            )
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "geometry_overlap_count": int(geometry_overlap),
                    "exact_request_overlap_count": int(exact_overlap),
                }
            )

    return {
        "stage_order": stage_names,
        "stages": stage_summary,
        "pairwise": pairwise,
        "total_pose_requests": int(total_count),
        "unique_geometry_count": int(len(seen_geometry)),
        "unique_exact_request_count": int(len(seen_requests)),
        "prior_stage_geometry_reuse_count": int(
            prior_geometry_reuse
        ),
        "prior_stage_exact_request_reuse_count": int(
            prior_exact_request_reuse
        ),
        "prior_stage_geometry_only_reuse_count": int(
            geometry_only_reuse
        ),
        "exact_request_reuse_ratio": float(
            prior_exact_request_reuse / max(total_count, 1)
        ),
    }


def _attach_lite_candidate_diagnostics(
    payload: Dict[str, Any],
    call_state: Dict[str, Any],
    *,
    fallback_used: bool,
) -> None:
    pose_dedupe_key = production.pose_dedupe_key

    lite_audit = payload.setdefault("plan_audit", {}).setdefault(
        "lite_planner",
        {},
    )
    lite_audit["stage_coverage"] = dict(
        call_state.get("stage_coverage") or {}
    )
    lite_audit["ik_request_reuse"] = _ik_stage_reuse_summary(
        call_state.get("stage_pose_lists") or {}
    )
    lite_audit["winner_lineage"] = None
    if fallback_used:
        return
    eef_pose = payload.get("eef_pose") or {}
    if eef_pose.get("pos") is None or eef_pose.get("quat") is None:
        return
    winner_key = pose_dedupe_key(
        {
            "eef_pos": eef_pose["pos"],
            "quat": eef_pose["quat"],
        }
    )
    matches = []
    for stage_name, poses in (
        call_state.get("stage_pose_lists") or {}
    ).items():
        for rank, pose in enumerate(poses):
            if pose_dedupe_key(pose) != winner_key:
                continue
            matches.append(
                {
                    "stage": str(stage_name),
                    **_lineage_summary(pose, rank=rank),
                }
            )
            break
    lite_audit["winner_lineage"] = {
        "matched": bool(matches),
        "matches": matches,
    }


def _secondary_member(
    members: Sequence[Dict[str, Any]],
    primary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    alternatives = [pose for pose in members if pose is not primary]
    if not alternatives:
        return None

    primary_source = _source_label(primary)

    def key(pose: Dict[str, Any]) -> Tuple[Any, ...]:
        position_ratio = _position_distance_mm(pose, primary) / max(
            LITE_POSITION_LIMIT_MM,
            1e-9,
        )
        angle_ratio = _rotation_angle_deg(pose, primary) / max(
            LITE_ANGLE_LIMIT_DEG,
            1e-9,
        )
        geometry_span = math.hypot(position_ratio, angle_ratio)
        angle = float(pose.get("normal_alignment_deg", float("inf")))
        return (
            geometry_span,
            int(_source_label(pose) != primary_source),
            int(pose.get("prefilter_grasp_vox", 0)),
            -int(
                pose.get(
                    "prefilter_inflated_overlap_vox",
                    np.iinfo(np.int32).max,
                )
            ),
            -int(pose.get("env_overlap_vox", np.iinfo(np.int32).max)),
            -angle if math.isfinite(angle) else -float("inf"),
            -int(pose.get("lite_base_rank", 0)),
        )

    return max(alternatives, key=key)


def select_lite_representatives(
    base_selected: Sequence[Dict[str, Any]],
    *,
    position_limit_mm: float = LITE_POSITION_LIMIT_MM,
    angle_limit_deg: float = LITE_ANGLE_LIMIT_DEG,
    members_per_cluster: int = LITE_MEMBERS_PER_CLUSTER,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compress production top candidates into same-ai local pose cells."""
    strict_safe: List[Dict[str, Any]] = []
    rejected_unsafe = 0
    for base_rank, pose in enumerate(base_selected):
        pose["lite_base_rank"] = int(base_rank)
        if not _strict_safe_pose(pose):
            rejected_unsafe += 1
            continue
        strict_safe.append(pose)

    clusters: List[List[Dict[str, Any]]] = []
    for pose in strict_safe:
        matching_cluster = None
        for cluster_id, members in enumerate(clusters):
            if _same_representative_cell(
                pose,
                members[0],
                position_limit_mm=position_limit_mm,
                angle_limit_deg=angle_limit_deg,
            ):
                matching_cluster = cluster_id
                break
        if matching_cluster is None:
            clusters.append([pose])
        else:
            clusters[matching_cluster].append(pose)

    selected: List[Dict[str, Any]] = []
    cluster_sizes: List[int] = []
    for cluster_id, members in enumerate(clusters):
        cluster_sizes.append(len(members))
        primary = members[0]
        keep = [primary]
        if int(members_per_cluster) >= 2:
            secondary = _secondary_member(members, primary)
            if secondary is not None:
                keep.append(secondary)
        if int(members_per_cluster) > 2:
            remaining = [
                pose
                for pose in members
                if not any(pose is kept for kept in keep)
            ]
            keep.extend(remaining[: int(members_per_cluster) - len(keep)])
        for member_rank, pose in enumerate(keep):
            pose["lite_cluster_id"] = int(cluster_id)
            pose["lite_cluster_member_rank"] = int(member_rank)
            pose["lite_cluster_size"] = int(len(members))
            selected.append(pose)

    selected.sort(key=lambda pose: int(pose.get("lite_base_rank", 0)))
    selected_source_counts = Counter(_source_label(pose) for pose in selected)
    selected_ai_counts = Counter(
        str(int(pose.get("ai", -1))) for pose in selected
    )
    return selected, {
        "enabled": True,
        "position_limit_mm": float(position_limit_mm),
        "angle_limit_deg": float(angle_limit_deg),
        "members_per_cluster": int(members_per_cluster),
        "base_selected_count": int(len(base_selected)),
        "base_strict_safe_count": int(len(strict_safe)),
        "rejected_unsafe_count": int(rejected_unsafe),
        "cluster_count": int(len(clusters)),
        "selected_count": int(len(selected)),
        "compression_ratio": float(
            len(selected) / max(len(base_selected), 1)
        ),
        "singleton_cluster_count": int(
            sum(size == 1 for size in cluster_sizes)
        ),
        "largest_cluster": int(max(cluster_sizes, default=0)),
        "cluster_size_quantiles": (
            np.quantile(
                cluster_sizes,
                [0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
            ).tolist()
            if cluster_sizes
            else []
        ),
        "selected_source_counts": dict(selected_source_counts),
        "selected_ai_counts": dict(selected_ai_counts),
        "selected_preview": [
            _metric_summary(pose) for pose in selected[:40]
        ],
    }


def rank_dedupe_top_ik_input_lite(
    poses: List[Dict[str, Any]],
    overlap_counts: np.ndarray,
    *,
    grasp_counts: Optional[np.ndarray] = None,
    inflated_counts: Optional[np.ndarray] = None,
    limit: int = 120,
    quality_quota: int = 90,
    alignment_quota: int = 20,
    normal_confidence: str = "none",
    weak_alignment_quota_cap: int = 12,
    alignment_grasp_ratio_override: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply production ranking, then retain two representatives per pose cell."""
    base_selected, base_audit = production.rank_dedupe_top_ik_input(
        poses,
        overlap_counts,
        grasp_counts=grasp_counts,
        inflated_counts=inflated_counts,
        limit=limit,
        quality_quota=quality_quota,
        alignment_quota=alignment_quota,
        normal_confidence=normal_confidence,
        weak_alignment_quota_cap=weak_alignment_quota_cap,
        alignment_grasp_ratio_override=alignment_grasp_ratio_override,
    )
    if grasp_counts is None or inflated_counts is None:
        audit = dict(base_audit)
        audit["lite_representative"] = {
            "enabled": False,
            "reason": "quality metrics unavailable",
        }
        return base_selected, audit

    selected, lite_audit = select_lite_representatives(base_selected)
    if not selected:
        audit = dict(base_audit)
        audit["lite_representative"] = {
            **lite_audit,
            "enabled": False,
            "reason": "no strict-safe representative; preserving production set",
        }
        return base_selected, audit

    audit = dict(base_audit)
    audit["base_quality_selected_count"] = int(
        base_audit.get("quality_selected_count", 0)
    )
    audit["base_alignment_added_count"] = int(
        base_audit.get("alignment_added_count", 0)
    )
    audit["base_selected_source_counts"] = dict(
        base_audit.get("selected_source_counts") or {}
    )
    audit["unique_selected_count"] = int(len(selected))
    audit["selected_strict_safe_count"] = int(len(selected))
    audit["quality_selected_count"] = int(len(selected))
    audit["alignment_added_count"] = 0
    audit["selected_source_counts"] = dict(
        lite_audit["selected_source_counts"]
    )
    audit["selected_strict_safe_grasp_max_cm3"] = max(
        (
            float(pose.get("prefilter_grasp_cm3", 0.0))
            for pose in selected
        ),
        default=0.0,
    )
    audit["lite_representative"] = lite_audit
    return selected, audit


def rank_seed_balanced_micro_ik_input_lite(
    poses: Sequence[Dict[str, Any]],
    overlap_counts: np.ndarray,
    grasp_counts: np.ndarray,
    inflated_counts: np.ndarray,
    *,
    limit: int = 120,
    per_seed_quality: int = 2,
    per_seed_alignment: int = 2,
    per_seed_bridge_midpoint: int = 2,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep every post-IK parent lineage while fitting one IK batch."""
    pose_dedupe_key = production.pose_dedupe_key
    base_selected, base_audit = production.rank_seed_balanced_micro_ik_input(
        poses,
        overlap_counts,
        grasp_counts,
        inflated_counts,
        limit=limit,
        per_seed_quality=per_seed_quality,
        per_seed_alignment=per_seed_alignment,
        per_seed_bridge_midpoint=per_seed_bridge_midpoint,
    )
    lite_limit = min(int(limit), int(LITE_POST_PARETO_IK_LIMIT))
    if len(base_selected) <= lite_limit:
        audit = dict(base_audit)
        audit["lite_parent_balanced"] = {
            "enabled": True,
            "base_selected_count": int(len(base_selected)),
            "selected_count": int(len(base_selected)),
            "limit": int(lite_limit),
            "parent_seed_count": int(
                len(
                    {
                        int(pose.get("micro_seed_index", -1))
                        for pose in base_selected
                    }
                )
            ),
            "selected_parent_seed_count": int(
                len(
                    {
                        int(pose.get("micro_seed_index", -1))
                        for pose in base_selected
                    }
                )
            ),
            "category_counts": {},
        }
        return base_selected, audit

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for pose in base_selected:
        groups.setdefault(
            int(pose.get("micro_seed_index", -1)),
            [],
        ).append(pose)

    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    category_counts = Counter()

    def add_pose(pose: Optional[Dict[str, Any]], category: str) -> None:
        if pose is None or len(selected) >= lite_limit:
            return
        key = pose_dedupe_key(pose)
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected.append(pose)
        category_counts[category] += 1

    def quality_key(pose: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            -int(pose.get("prefilter_grasp_vox", 0)),
            float(pose.get("normal_alignment_deg", float("inf"))),
            int(
                pose.get(
                    "prefilter_inflated_overlap_vox",
                    np.iinfo(np.int32).max,
                )
            ),
            int(pose.get("env_overlap_vox", np.iinfo(np.int32).max)),
        )

    def alignment_key(pose: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            float(pose.get("normal_alignment_deg", float("inf"))),
            -int(pose.get("prefilter_grasp_vox", 0)),
            int(
                pose.get(
                    "prefilter_inflated_overlap_vox",
                    np.iinfo(np.int32).max,
                )
            ),
            int(pose.get("env_overlap_vox", np.iinfo(np.int32).max)),
        )

    def bridge_key(pose: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            abs(float(pose.get("interpolation_t", 0.5)) - 0.5),
            *quality_key(pose),
        )

    # Three distinct semantic representatives per reachable parent.
    for seed_index in sorted(groups):
        seed_poses = groups[seed_index]
        add_pose(min(seed_poses, key=quality_key), "quality")
        add_pose(min(seed_poses, key=alignment_key), "alignment")
        bridge_poses = [
            pose
            for pose in seed_poses
            if bool(pose.get("reachability_bridge", False))
        ]
        add_pose(
            min(bridge_poses, key=bridge_key) if bridge_poses else None,
            "bridge",
        )

    # Preserve production preference for all remaining slots.
    for pose in base_selected:
        add_pose(pose, "production_fill")
        if len(selected) >= lite_limit:
            break

    base_rank = {
        pose_dedupe_key(pose): rank
        for rank, pose in enumerate(base_selected)
    }
    selected.sort(key=lambda pose: base_rank[pose_dedupe_key(pose)])
    selected_parent_count = len(
        {
            int(pose.get("micro_seed_index", -1))
            for pose in selected
        }
    )
    audit = dict(base_audit)
    audit["base_selected_count"] = int(len(base_selected))
    audit["selected_count"] = int(len(selected))
    audit["selected_parent_seed_count"] = int(selected_parent_count)
    audit["lite_parent_balanced"] = {
        "enabled": True,
        "base_selected_count": int(len(base_selected)),
        "selected_count": int(len(selected)),
        "limit": int(lite_limit),
        "parent_seed_count": int(len(groups)),
        "selected_parent_seed_count": int(selected_parent_count),
        "category_counts": dict(category_counts),
        "compression_ratio": float(
            len(selected) / max(len(base_selected), 1)
        ),
    }
    return selected, audit


def rank_lineage_balanced_stage_ik_input_lite(
    poses: List[Dict[str, Any]],
    overlap_counts: np.ndarray,
    *,
    grasp_counts: Optional[np.ndarray] = None,
    inflated_counts: Optional[np.ndarray] = None,
    limit: int = 120,
    quality_quota: int = 60,
    alignment_quota: int = 60,
    normal_confidence: str = "none",
    weak_alignment_quota_cap: int = 60,
    alignment_grasp_ratio_override: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compress a generated stage while preserving every parent lineage."""
    pose_dedupe_key = production.pose_dedupe_key
    base_selected, base_audit = production.rank_dedupe_top_ik_input(
        poses,
        overlap_counts,
        grasp_counts=grasp_counts,
        inflated_counts=inflated_counts,
        limit=limit,
        quality_quota=quality_quota,
        alignment_quota=alignment_quota,
        normal_confidence=normal_confidence,
        weak_alignment_quota_cap=weak_alignment_quota_cap,
        alignment_grasp_ratio_override=alignment_grasp_ratio_override,
    )
    lite_limit = min(
        int(limit),
        int(LITE_REACHABLE_STAGE_IK_LIMIT),
    )
    if len(base_selected) <= lite_limit:
        audit = dict(base_audit)
        audit["lite_lineage_balanced"] = {
            "enabled": True,
            "base_selected_count": int(len(base_selected)),
            "selected_count": int(len(base_selected)),
            "limit": int(lite_limit),
            "lineage_field": None,
            "lineage_count": 0,
            "selected_lineage_count": 0,
            "category_counts": {},
        }
        return base_selected, audit

    if any(
        bool(pose.get("reachable_se3_interpolated", False))
        and int(pose.get("reachable_se3_pair_id", -1)) >= 0
        for pose in base_selected
    ):
        lineage_field = "reachable_se3_pair_id"
    elif any(
        bool(pose.get("translation_refined", False))
        and int(pose.get("translation_seed_index", -1)) >= 0
        for pose in base_selected
    ):
        lineage_field = "translation_seed_index"
    else:
        lineage_field = next(
            (
                field
                for field in (
                    "reachable_se3_pair_id",
                    "micro_seed_index",
                )
                if any(
                    int(pose.get(field, -1)) >= 0
                    for pose in base_selected
                )
            ),
            None,
        )
    if lineage_field is None:
        audit = dict(base_audit)
        audit["lite_lineage_balanced"] = {
            "enabled": False,
            "reason": "no generated-stage lineage field",
            "base_selected_count": int(len(base_selected)),
            "selected_count": int(lite_limit),
            "limit": int(lite_limit),
        }
        return base_selected[:lite_limit], audit

    groups: Dict[int, List[Dict[str, Any]]] = {}
    ungrouped_index = -1
    for pose in base_selected:
        lineage = int(pose.get(lineage_field, -1))
        if lineage < 0:
            lineage = ungrouped_index
            ungrouped_index -= 1
        groups.setdefault(lineage, []).append(pose)

    if len(groups) > lite_limit:
        audit = dict(base_audit)
        audit["lite_lineage_balanced"] = {
            "enabled": False,
            "reason": "lineage count exceeds cap",
            "base_selected_count": int(len(base_selected)),
            "selected_count": int(lite_limit),
            "limit": int(lite_limit),
            "lineage_field": lineage_field,
            "lineage_count": int(len(groups)),
        }
        return base_selected[:lite_limit], audit

    base_rank = {
        pose_dedupe_key(pose): rank
        for rank, pose in enumerate(base_selected)
    }
    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    category_counts = Counter()

    def add_pose(
        pose: Optional[Dict[str, Any]],
        category: str,
    ) -> None:
        if pose is None or len(selected) >= lite_limit:
            return
        key = pose_dedupe_key(pose)
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected.append(pose)
        category_counts[category] += 1

    def quality_key(pose: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            -int(pose.get("prefilter_grasp_vox", 0)),
            float(pose.get("normal_alignment_deg", float("inf"))),
            int(
                pose.get(
                    "prefilter_inflated_overlap_vox",
                    np.iinfo(np.int32).max,
                )
            ),
            int(pose.get("env_overlap_vox", np.iinfo(np.int32).max)),
            int(base_rank[pose_dedupe_key(pose)]),
        )

    def alignment_key(pose: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            float(pose.get("normal_alignment_deg", float("inf"))),
            -int(pose.get("prefilter_grasp_vox", 0)),
            int(
                pose.get(
                    "prefilter_inflated_overlap_vox",
                    np.iinfo(np.int32).max,
                )
            ),
            int(pose.get("env_overlap_vox", np.iinfo(np.int32).max)),
            int(base_rank[pose_dedupe_key(pose)]),
        )

    def collision_key(pose: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            int(
                pose.get(
                    "prefilter_inflated_overlap_vox",
                    np.iinfo(np.int32).max,
                )
            ),
            int(pose.get("env_overlap_vox", np.iinfo(np.int32).max)),
            -int(pose.get("prefilter_grasp_vox", 0)),
            float(pose.get("normal_alignment_deg", float("inf"))),
            int(base_rank[pose_dedupe_key(pose)]),
        )

    def midpoint_key(pose: Dict[str, Any]) -> Tuple[Any, ...]:
        if pose.get("translation_offset_local_m") is not None:
            value = abs(
                float(
                    np.asarray(
                        pose["translation_offset_local_m"],
                        dtype=np.float64,
                    ).reshape(3)[2]
                )
            )
        else:
            value = abs(float(pose.get("interpolation_t", 0.5)) - 0.5)
        return (value, *quality_key(pose))

    for selector, category in (
        (quality_key, "quality"),
        (alignment_key, "alignment"),
        (collision_key, "collision"),
        (midpoint_key, "midpoint"),
    ):
        for lineage in sorted(groups):
            add_pose(min(groups[lineage], key=selector), category)

    for pose in base_selected:
        add_pose(pose, "production_fill")
        if len(selected) >= lite_limit:
            break

    selected.sort(
        key=lambda pose: base_rank[pose_dedupe_key(pose)]
    )
    selected_lineages = {
        int(pose.get(lineage_field, -1))
        for pose in selected
        if int(pose.get(lineage_field, -1)) >= 0
    }
    source_lineages = {
        int(pose.get(lineage_field, -1))
        for pose in base_selected
        if int(pose.get(lineage_field, -1)) >= 0
    }
    audit = dict(base_audit)
    audit["base_selected_count"] = int(len(base_selected))
    audit["unique_selected_count"] = int(len(selected))
    audit["selected_strict_safe_count"] = int(len(selected))
    audit["quality_selected_count"] = int(len(selected))
    audit["alignment_added_count"] = 0
    audit["selected_source_counts"] = dict(
        Counter(_source_label(pose) for pose in selected)
    )
    audit["selected_strict_safe_grasp_max_cm3"] = max(
        (
            float(pose.get("prefilter_grasp_cm3", 0.0))
            for pose in selected
        ),
        default=0.0,
    )
    audit["lite_lineage_balanced"] = {
        "enabled": True,
        "base_selected_count": int(len(base_selected)),
        "selected_count": int(len(selected)),
        "limit": int(lite_limit),
        "lineage_field": lineage_field,
        "lineage_count": int(len(source_lineages)),
        "selected_lineage_count": int(len(selected_lineages)),
        "category_counts": dict(category_counts),
        "compression_ratio": float(
            len(selected) / max(len(base_selected), 1)
        ),
    }
    return selected, audit


def _production_planner_clone_with_lite_ranker():
    production_planner = production.plan_grasp_point_filter_rgbd
    call_state = {
        "rank_calls": 0,
        "micro_rank_calls": 0,
        "stage_pose_lists": {},
        "stage_coverage": {},
        "skipped_se3_ik_stages": [],
        "ik_worker_policies": [],
        "active_ik_stage": "initial",
        "phase_timings": {},
        "warm_solver_prepare_requested": False,
        "warm_solver_prepare_elapsed_s": 0.0,
    }

    def record_timing(name: str, elapsed_s: float) -> None:
        timing = call_state["phase_timings"].setdefault(
            str(name),
            {"calls": 0, "elapsed_s": 0.0},
        )
        timing["calls"] += 1
        timing["elapsed_s"] += float(elapsed_s)

    def timed(name: str, function):
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                record_timing(
                    name,
                    time.perf_counter() - started,
                )

        return wrapper

    def rank_dispatch(*args, **kwargs):
        started = time.perf_counter()
        call_state["rank_calls"] += 1
        stage_names = (
            "initial",
            "reachable_se3_interpolation",
            "reachable_se3_closure",
        )
        stage_name = (
            stage_names[call_state["rank_calls"] - 1]
            if call_state["rank_calls"] <= len(stage_names)
            else f"rank_call_{call_state['rank_calls']}"
        )
        if (
            call_state["rank_calls"] == 1
            and LITE_COMPRESS_INITIAL_STAGE
        ):
            selected, audit = rank_dedupe_top_ik_input_lite(
                *args,
                **kwargs,
            )
        else:
            selected, audit = production.rank_dedupe_top_ik_input(
                *args,
                **kwargs,
            )
        try:
            _record_stage_input(call_state, stage_name, selected)
            call_state["active_ik_stage"] = stage_name
            return selected, audit
        finally:
            record_timing(
                f"rank.{stage_name}",
                time.perf_counter() - started,
            )

    def micro_rank_dispatch(*args, **kwargs):
        started = time.perf_counter()
        call_state["micro_rank_calls"] += 1
        selected, audit = production.rank_seed_balanced_micro_ik_input(
            *args,
            **kwargs,
        )
        _record_stage_input(
            call_state,
            "post_ik_pareto_refinement",
            selected,
        )
        try:
            call_state["active_ik_stage"] = (
                "post_ik_pareto_refinement"
            )
            return selected, audit
        finally:
            record_timing(
                "rank.post_ik_pareto_refinement",
                time.perf_counter() - started,
            )

    def execute_reachable_se3_stage_dispatch(*args, **kwargs):
        started = time.perf_counter()
        stage_name = str(
            kwargs.get("stage_name")
            or "reachable_se3_generated"
        )
        audit_stage_name = stage_name
        if "closure 2" in stage_name:
            audit_stage_name = "reachable_se3_closure2"
        if LITE_SKIP_REACHABLE_SE3_IK_STAGES:
            def skip_ranker(*rank_args, **rank_kwargs):
                selected, audit = production.rank_dedupe_top_ik_input(
                    *rank_args,
                    **rank_kwargs,
                )
                _record_stage_input(
                    call_state,
                    audit_stage_name,
                    selected,
                )
                audit = dict(audit)
                audit["lite_skip_reachable_se3_ik"] = {
                    "enabled": True,
                    "ranked_count": int(len(selected)),
                    "solver_input_count": 0,
                }
                return [], audit

            kwargs["ranker"] = skip_ranker
            call_state["skipped_se3_ik_stages"].append(
                audit_stage_name
            )
        else:
            kwargs.pop("ranker", None)
        stage_globals = dict(
            production.execute_reachable_se3_stage.__globals__
        )
        stage_globals["filter_poses_safe_final_dual_arm_ik"] = (
            filter_safe_final_dispatch
        )
        stage_runner = types.FunctionType(
            production.execute_reachable_se3_stage.__code__,
            stage_globals,
            name="execute_reachable_se3_stage_lite",
            argdefs=(
                production.execute_reachable_se3_stage.__defaults__
            ),
            closure=(
                production.execute_reachable_se3_stage.__closure__
            ),
        )
        stage_runner.__kwdefaults__ = dict(
            production.execute_reachable_se3_stage.__kwdefaults__
            or {}
        )
        previous_stage = call_state["active_ik_stage"]
        call_state["active_ik_stage"] = audit_stage_name
        try:
            result = stage_runner(*args, **kwargs)
        finally:
            call_state["active_ik_stage"] = previous_stage
        if not LITE_SKIP_REACHABLE_SE3_IK_STAGES:
            _record_stage_input(
                call_state,
                audit_stage_name,
                result.get("ik_input") or [],
            )
        record_timing(
            f"stage.{audit_stage_name}",
            time.perf_counter() - started,
        )
        return result

    def execute_generated_stage_dispatch(*args, **kwargs):
        started = time.perf_counter()
        kwargs.pop("ranker", None)
        poses = kwargs.get("poses") or []
        stage_name = str(
            kwargs.get("stage_name")
            or "generated_reachable_pose"
        )
        if "translation" in stage_name:
            stage_name = "reachable_translation_refinement"
        stage_globals = dict(
            production.execute_generated_reachable_pose_stage.__globals__
        )
        stage_globals["filter_poses_safe_final_dual_arm_ik"] = (
            filter_safe_final_dispatch
        )
        stage_runner = types.FunctionType(
            production.execute_generated_reachable_pose_stage.__code__,
            stage_globals,
            name="execute_generated_reachable_pose_stage_lite",
            argdefs=(
                production.execute_generated_reachable_pose_stage.__defaults__
            ),
            closure=(
                production.execute_generated_reachable_pose_stage.__closure__
            ),
        )
        stage_runner.__kwdefaults__ = dict(
            production.execute_generated_reachable_pose_stage.__kwdefaults__
            or {}
        )
        previous_stage = call_state["active_ik_stage"]
        call_state["active_ik_stage"] = stage_name
        try:
            result = stage_runner(*args, **kwargs)
        finally:
            call_state["active_ik_stage"] = previous_stage
        _record_stage_input(
            call_state,
            stage_name,
            result.get("ik_input") or [],
        )
        record_timing(
            f"stage.{stage_name}",
            time.perf_counter() - started,
        )
        return result

    def filter_safe_final_dispatch(*args, **kwargs):
        started = time.perf_counter()
        poses = (
            args[1]
            if len(args) >= 2
            else kwargs.get("poses")
        ) or []
        generations = {
            int(pose.get("reachable_se3_generation", 0))
            for pose in poses
        }
        if (
            LITE_SKIP_REACHABLE_SE3_IK_STAGES
            and generations
            and generations.issubset({1, 2})
        ):
            stage_name = (
                "reachable_se3_closure"
                if 2 in generations
                else "reachable_se3_interpolation"
            )
            call_state["skipped_se3_ik_stages"].append(stage_name)
            return [], {
                "skipped": True,
                "skip_reason": "lite_reachable_se3_ik_disabled",
                "lite_skip_reachable_se3_ik": True,
                "n_input": int(len(poses)),
                "n_solver_input": 0,
                "n_ranked": 0,
            }
        plan_arm = str(kwargs.get("plan_arm") or "any")
        policy = _lite_ik_worker_policy(
            poses,
            plan_arm=plan_arm,
        )
        kwargs["ik_worker_policy"] = policy
        call_state["ik_worker_policies"].append(
            {
                "stage": str(call_state["active_ik_stage"]),
                "policy": policy,
                "pose_count": int(len(poses)),
            }
        )
        try:
            result = production.filter_poses_safe_final_dual_arm_ik(
                *args,
                **kwargs,
            )
            if (
                call_state["active_ik_stage"] == "initial"
                and policy != LITE_WARM_IK_WORKER_POLICY
                and not call_state["warm_solver_prepare_requested"]
            ):
                prepare_started = time.perf_counter()
                requested_arm = str(kwargs.get("plan_arm") or "any")
                active_arms = (
                    (requested_arm,)
                    if requested_arm in ("left", "right")
                    else ("left", "right")
                )
                prepare_external_ik(
                    args[0] if args else kwargs["world"],
                    pos_tol_m=IK_FILTER_POS_TOL_M,
                    ori_tol_deg=IK_FILTER_ORI_TOL_DEG,
                    active_arms=active_arms,
                    policy=LITE_WARM_IK_WORKER_POLICY,
                )
                call_state["warm_solver_prepare_requested"] = True
                call_state["warm_solver_prepare_elapsed_s"] = float(
                    time.perf_counter() - prepare_started
                )
            return result
        finally:
            record_timing(
                f"ik.{call_state['active_ik_stage']}",
                time.perf_counter() - started,
            )

    isolated_globals = dict(production_planner.__globals__)
    isolated_globals["rank_dedupe_top_ik_input"] = rank_dispatch
    isolated_globals["rank_seed_balanced_micro_ik_input"] = (
        micro_rank_dispatch
    )
    isolated_globals["execute_reachable_se3_stage"] = (
        execute_reachable_se3_stage_dispatch
    )
    isolated_globals["execute_generated_reachable_pose_stage"] = (
        execute_generated_stage_dispatch
    )
    isolated_globals["filter_poses_safe_final_dual_arm_ik"] = (
        filter_safe_final_dispatch
    )
    isolated_globals["build_local_scene_occupancy"] = (
        _build_local_scene_occupancy_lite
    )
    isolated_globals["render_local_three_views"] = (
        _render_local_three_views_lite
    )
    for function_name in (
        "generate_rgbd_filter_poses",
        "attach_normal_alignment",
        "inflated_gripper_voxels_eef",
        "build_local_scene_occupancy",
        "select_local_refinement_seeds",
        "generate_local_orientation_refinements",
        "select_rescue_sampling_seeds",
        "generate_opening_contact_refinements",
        "generate_pre_ik_translation_refinements",
        "generate_quality_pose_interpolations",
        "generate_pareto_micro_refinements",
        "generate_reachability_bridge_interpolations",
        "generate_reachable_translation_refinements",
        "filter_final_overlap",
        "render_local_three_views",
    ):
        function = isolated_globals.get(function_name)
        if callable(function):
            isolated_globals[function_name] = timed(
                function_name,
                function,
            )
    isolated_globals["apply_camera_face_preserving_anchor"] = timed(
        "apply_camera_face_preserving_anchor",
        _apply_camera_face_preserving_anchor_lite,
    )
    build_suffix = (
        "lite_skip_reachable_se3_ik"
        if LITE_SKIP_REACHABLE_SE3_IK_STAGES
        else f"lite_full_candidate_{LITE_IK_WORKER_POLICY}"
    )
    isolated_globals["BUILD"] = (
        f"{_v52_planner_build(production.BUILD)}_{build_suffix}"
    )
    cloned = types.FunctionType(
        production_planner.__code__,
        isolated_globals,
        name="plan_grasp_point_filter_rgbd_lite_core",
        argdefs=production_planner.__defaults__,
        closure=production_planner.__closure__,
    )
    cloned.__kwdefaults__ = dict(production_planner.__kwdefaults__ or {})
    cloned.__annotations__ = dict(production_planner.__annotations__)
    return cloned, call_state


def _production_planner_clone_with_exact_runtime_reuse():
    """Clone production policy while replacing runtime-only mesh operations."""
    production_planner = production.plan_grasp_point_filter_rgbd
    isolated_globals = dict(production_planner.__globals__)
    isolated_globals["build_local_scene_occupancy"] = (
        _build_local_scene_occupancy_lite
    )
    isolated_globals["render_local_three_views"] = (
        _render_local_three_views_lite
    )
    isolated_globals["BUILD"] = _v52_planner_build(production.BUILD)
    cloned = types.FunctionType(
        production_planner.__code__,
        isolated_globals,
        name="plan_grasp_point_filter_rgbd_exact_runtime_reuse",
        argdefs=production_planner.__defaults__,
        closure=production_planner.__closure__,
    )
    cloned.__kwdefaults__ = dict(production_planner.__kwdefaults__ or {})
    cloned.__annotations__ = dict(production_planner.__annotations__)
    return cloned


def plan_grasp_point_filter_rgbd_lite(
    *,
    world,
    session: Dict[str, Any],
    u: int,
    v: int,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    plan_arm: str,
    seed: int,
    ctx=None,
    prepared_scene=None,
    prepared_target=None,
) -> Dict[str, Any]:
    """Run the isolated Lite planner with an optional production retry."""
    # Production quaternion helpers normalize ndarray inputs in place. Keep
    # each plan isolated so repeated calls on one frozen capture start from
    # byte-identical camera calibration.
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3).copy()
    cam_quat = np.asarray(cam_quat, dtype=np.float64).reshape(4).copy()
    requested_arm = str(plan_arm or "any").strip().lower()
    active_arms = (
        (requested_arm,)
        if requested_arm in ("left", "right")
        else ("left", "right")
    )
    geometry_backend = _lite_geometry_backend()
    requested_geometry_version = (
        RGBD_LITE_GEOMETRY_VERSION
        if geometry_backend == LITE_GEOMETRY_BACKEND_PROJECTIVE
        else "v53"
    )
    prepare_external_ik(
        world,
        pos_tol_m=IK_FILTER_POS_TOL_M,
        ori_tol_deg=IK_FILTER_ORI_TOL_DEG,
        active_arms=active_arms,
        policy=LITE_IK_WORKER_POLICY,
    )
    reconstruction_elapsed_s = 0.0
    if prepared_scene is None:
        reconstruction_started = time.perf_counter()
        if ctx is not None:
            if geometry_backend == LITE_GEOMETRY_BACKEND_PROJECTIVE:
                ctx.log(
                    "  [grasp_point_filter_rgbd_lite] preparing frozen "
                    "depth for local CUDA projective occupancy "
                    "(whole-scene mesh disabled)"
                )
            elif geometry_backend == LITE_GEOMETRY_BACKEND_WARP:
                ctx.log(
                    "  [grasp_point_filter_rgbd_lite] reconstructing "
                    "representative V53 scene mesh for Warp CUDA occupancy"
                )
            else:
                ctx.log(
                    "  [grasp_point_filter_rgbd_lite] reconstructing "
                    "legacy representative V53 scene mesh"
                )
        prepared_scene = _cached_reconstruct_v52_scene_from_session(
            session,
            camera_pos=cam_pos,
            camera_quat_xyzw=cam_quat,
            focal_length=fl,
            horizontal_aperture=ha,
        )
        reconstruction_elapsed_s = float(
            time.perf_counter() - reconstruction_started
        )
        if ctx is not None:
            reconstruction = dict(prepared_scene[3])
            if geometry_backend == LITE_GEOMETRY_BACKEND_PROJECTIVE:
                ctx.log(
                    "  [grasp_point_filter_rgbd_lite] projective depth "
                    f"ready shape={reconstruction.get('depth_shape')} "
                    f"valid={reconstruction.get('depth_valid_pixels')} "
                    "mesh_built=False trimesh_contains=False "
                    f"elapsed={reconstruction_elapsed_s:.3f}s"
                )
            else:
                ctx.log(
                    "  [grasp_point_filter_rgbd_lite] V53 mesh ready "
                    f"effective={reconstruction.get('effective_mesh_version')} "
                    f"expert={reconstruction.get('expert_applied')} "
                    f"fallback={reconstruction.get('fallback')} "
                    f"vertices={reconstruction.get('mesh_vertices')} "
                    f"faces={reconstruction.get('mesh_faces')} "
                    f"elapsed={reconstruction_elapsed_s:.3f}s"
                )
    else:
        prepared_metadata = dict(prepared_scene[3])
        prepared_version = prepared_metadata.get(
            "geometry_version",
            prepared_metadata.get("mesh_version"),
        )
        if prepared_version != requested_geometry_version:
            raise ValueError(
                "plan_grasp_point_filter_rgbd_lite requires a "
                f"{requested_geometry_version} prepared_scene, got "
                f"{prepared_version!r}"
            )
    if (
        geometry_backend == LITE_GEOMETRY_BACKEND_PROJECTIVE
        and not isinstance(prepared_scene[0], ProjectiveDepthScene)
    ):
        raise TypeError(
            "projective_cuda prepared_scene must contain ProjectiveDepthScene"
        )
    geometry_cache_started = time.perf_counter()
    geometry_cache = _ensure_lite_gripper_geometry_cache(
        production.VOXEL_M
    )
    geometry_cache_elapsed_s = float(
        time.perf_counter() - geometry_cache_started
    )
    lite_planner, call_state = _production_planner_clone_with_lite_ranker()
    fallback_to_full = _lite_fallback_to_full_enabled()
    fallback_error = None
    planner_started = time.perf_counter()
    try:
        payload = lite_planner(
            world=world,
            session=session,
            u=u,
            v=v,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            w=w,
            h=h,
            fl=fl,
            ha=ha,
            plan_arm=plan_arm,
            seed=seed,
            ctx=ctx,
            prepared_scene=prepared_scene,
            prepared_target=prepared_target,
        )
        fallback_used = False
    except production.GraspObjPlanningError as exc:
        if not fallback_to_full:
            raise
        fallback_error = str(exc)
        prepare_external_ik(
            world,
            pos_tol_m=IK_FILTER_POS_TOL_M,
            ori_tol_deg=IK_FILTER_ORI_TOL_DEG,
            active_arms=active_arms,
            policy="baseline",
        )
        if ctx is not None:
            ctx.log(
                "  [grasp_point_filter_rgbd_lite] representative attempt "
                f"failed; retrying production top120: {fallback_error}"
            )
        fallback_planner = (
            _production_planner_clone_with_exact_runtime_reuse()
        )
        payload = fallback_planner(
            world=world,
            session=session,
            u=u,
            v=v,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            w=w,
            h=h,
            fl=fl,
            ha=ha,
            plan_arm=plan_arm,
            seed=seed,
            ctx=ctx,
            prepared_scene=prepared_scene,
            prepared_target=prepared_target,
        )
        fallback_used = True
    planner_elapsed_s = float(time.perf_counter() - planner_started)

    payload["mode"] = "grasp_point_filter_rgbd_lite"
    for candidate in payload.get("candidates") or []:
        meta = candidate.get("meta")
        if isinstance(meta, dict):
            meta["mode"] = "grasp_point_filter_rgbd_lite"
    plan_audit = payload.setdefault("plan_audit", {})
    reconstruction_audit = dict(plan_audit.get("reconstruction") or {})
    reconstructed_version = reconstruction_audit.get(
        "geometry_version",
        reconstruction_audit.get("mesh_version"),
    )
    if reconstructed_version == requested_geometry_version:
        representative_build = _v52_planner_build(
            str(plan_audit.get("build") or production.BUILD)
        )
        plan_audit["build"] = representative_build
        grip_fit = payload.get("grip_fit")
        if isinstance(grip_fit, dict):
            grip_fit["build"] = representative_build
            if geometry_backend == LITE_GEOMETRY_BACKEND_PROJECTIVE:
                grip_fit["volume_source"] = (
                    "rgbd_projective_cuda_v1_dense_occupancy_experimental"
                )
            elif geometry_backend == LITE_GEOMETRY_BACKEND_WARP:
                grip_fit["volume_source"] = (
                    "rgbd_v53_warp_cuda_winding_dense_occupancy"
                )
            else:
                grip_fit["volume_source"] = "rgbd_v53_dense_occupancy_legacy_cpu"
    plan_audit["lite_planner"] = {
        "enabled": True,
        "position_limit_mm": LITE_POSITION_LIMIT_MM,
        "angle_limit_deg": LITE_ANGLE_LIMIT_DEG,
        "members_per_cluster": LITE_MEMBERS_PER_CLUSTER,
        "post_pareto_ik_limit": LITE_POST_PARETO_IK_LIMIT,
        "reachable_stage_ik_limit": LITE_REACHABLE_STAGE_IK_LIMIT,
        "initial_stage_compression_enabled": bool(
            LITE_COMPRESS_INITIAL_STAGE
        ),
        "refinement_stage_compression_enabled": bool(
            LITE_COMPRESS_REFINEMENT_STAGES
        ),
        "skip_reachable_se3_ik_stages": bool(
            LITE_SKIP_REACHABLE_SE3_IK_STAGES
        ),
        "skipped_se3_ik_stages": list(
            dict.fromkeys(call_state["skipped_se3_ik_stages"])
        ),
        "ik_worker_policy": str(LITE_IK_WORKER_POLICY),
        "warm_ik_worker_policy": str(LITE_WARM_IK_WORKER_POLICY),
        "ik_worker_policies": list(
            call_state["ik_worker_policies"]
        ),
        "warm_solver_prepare_requested": bool(
            call_state["warm_solver_prepare_requested"]
        ),
        "warm_solver_prepare_elapsed_s": float(
            call_state["warm_solver_prepare_elapsed_s"]
        ),
        "core_planner_elapsed_s": planner_elapsed_s,
        "scene_geometry_backend": geometry_backend,
        "scene_geometry_requested_version": requested_geometry_version,
        "occupancy_requested_version": (
            WARP_OCCUPANCY_VERSION
            if geometry_backend == LITE_GEOMETRY_BACKEND_WARP
            else requested_geometry_version
        ),
        "scene_mesh_requested_version": (
            None
            if geometry_backend == LITE_GEOMETRY_BACKEND_PROJECTIVE
            else "v53"
        ),
        "scene_mesh_built": bool(
            reconstruction_audit.get("scene_mesh_built", True)
        ),
        "scene_reconstruction_elapsed_s": reconstruction_elapsed_s,
        "scene_reconstruction_prepared_once": True,
        "scene_reconstruction_reused_by_fallback": bool(fallback_used),
        "gripper_geometry_cache": {
            **geometry_cache,
            "elapsed_s": geometry_cache_elapsed_s,
        },
        "phase_timings": {
            name: {
                "calls": int(values["calls"]),
                "elapsed_s": float(values["elapsed_s"]),
            }
            for name, values in call_state["phase_timings"].items()
        },
        "rank_calls_in_lite_attempt": int(call_state["rank_calls"]),
        "micro_rank_calls_in_lite_attempt": int(
            call_state["micro_rank_calls"]
        ),
        "fallback_to_production_enabled": bool(fallback_to_full),
        "fallback_used": bool(fallback_used),
        "lite_failure": fallback_error,
    }
    _attach_lite_candidate_diagnostics(
        payload,
        call_state,
        fallback_used=fallback_used,
    )
    return payload


__all__ = [
    "LITE_POSE_OVERLAP_FILTER_BUILD",
    "clear_v52_scene_cache",
    "evaluate_pose_overlap_volumes_lite",
    "plan_grasp_point_filter_rgbd_lite",
]
