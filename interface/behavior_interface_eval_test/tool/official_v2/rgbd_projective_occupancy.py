"""GPU-only local occupancy directly from a frozen metric-depth image.

The RGB-D Lite planner only queries a small envelope around one depth hit.  A
whole-scene watertight triangle mesh is therefore an unnecessary intermediate:
the same conservative camera-ray shell can be evaluated projectively on the
local 3 mm grid.  Runtime callers are fail-closed when CUDA is unavailable;
the CPU device exists only for small deterministic unit-test references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import threading
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .grasp_geometry_local import quat_to_mat_xyzw


PROJECTIVE_GEOMETRY_VERSION = "projective_cuda_v1"
PROJECTIVE_GEOMETRY_BUILD = "rgbd_lite_projective_cuda_depth_shell_v1"
DEFAULT_BACK_EXTRUSION_M = 0.060
DEFAULT_FRONT_TOLERANCE_M = 0.002
DEFAULT_PIXEL_RADIUS = 1
DEFAULT_CHUNK_VOXELS = 262_144
DEFAULT_MAX_GRID_VOXELS = 4_000_000
DEFAULT_MIN_FREE_CUDA_MIB = 256
MIN_VALID_DEPTH_M = 0.05
MAX_VALID_DEPTH_M = 50.0


class ProjectiveOccupancyError(RuntimeError):
    """Base error for fail-closed projective occupancy construction."""


class ProjectiveOccupancyUnavailable(ProjectiveOccupancyError):
    """The configured CUDA backend cannot be used safely."""


class ProjectiveOccupancyContractError(ProjectiveOccupancyError):
    """The RGB-D hit and generated occupancy disagree."""


@dataclass
class ProjectiveDepthScene:
    """Immutable RGB-D geometry inputs with a lazily cached depth tensor."""

    depth: np.ndarray
    camera_pos: np.ndarray
    camera_quat_xyzw: np.ndarray
    focal_length: float
    horizontal_aperture: float
    scene_key: str
    metadata: Dict[str, Any]
    geometry_kind: str = PROJECTIVE_GEOMETRY_VERSION
    _device_depth_cache: Dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _device_cache_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    @property
    def _official_v2_lite_scene_key(self) -> str:
        return str(self.scene_key)

    @property
    def is_watertight(self) -> bool:
        return False

    @property
    def vertices(self) -> np.ndarray:
        return np.empty((0, 3), dtype=np.float64)

    @property
    def faces(self) -> np.ndarray:
        return np.empty((0, 3), dtype=np.int64)

    @property
    def triangles(self) -> np.ndarray:
        return np.empty((0, 3, 3), dtype=np.float64)

    def depth_tensor(self, torch, device):
        key = str(device)
        with self._device_cache_lock:
            cached = self._device_depth_cache.get(key)
            if cached is None:
                cached = torch.as_tensor(
                    np.ascontiguousarray(self.depth, dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                ).contiguous()
                self._device_depth_cache[key] = cached
            return cached

    def clear_device_cache(self) -> None:
        with self._device_cache_lock:
            self._device_depth_cache.clear()


@dataclass
class ProjectiveOccupancyGrid:
    origin: np.ndarray
    occupancy: np.ndarray
    voxel_m: float
    metadata: Dict[str, Any]


def _normalized_depth(depth: np.ndarray) -> np.ndarray:
    array = np.asarray(depth)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"depth must be HxW, got {array.shape}")
    result = np.ascontiguousarray(array, dtype=np.float32)
    if result.size == 0:
        raise ValueError("depth image is empty")
    return result


def prepare_projective_scene_from_session(
    session: Dict[str, Any],
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    scene_key: str,
) -> Tuple[ProjectiveDepthScene, None, np.ndarray, Dict[str, Any]]:
    """Load only the frozen metric depth needed by projective occupancy."""
    started = time.perf_counter()
    depth_path = os.path.realpath(str(session.get("depth_path") or ""))
    if not depth_path or not os.path.isfile(depth_path):
        raise FileNotFoundError(f"frozen depth file not found: {depth_path!r}")
    depth = _normalized_depth(np.load(depth_path, allow_pickle=False))
    camera_position = np.asarray(camera_pos, dtype=np.float64).reshape(3).copy()
    camera_quaternion = (
        np.asarray(camera_quat_xyzw, dtype=np.float64).reshape(4).copy()
    )
    if not np.all(np.isfinite(camera_position)):
        raise ValueError("camera_pos contains non-finite values")
    if not np.all(np.isfinite(camera_quaternion)):
        raise ValueError("camera_quat_xyzw contains non-finite values")
    quaternion_norm = float(np.linalg.norm(camera_quaternion))
    if quaternion_norm <= 1e-12:
        raise ValueError("camera_quat_xyzw has zero norm")
    camera_quaternion /= quaternion_norm
    focal = float(focal_length)
    aperture = float(horizontal_aperture)
    if not np.isfinite(focal) or not np.isfinite(aperture):
        raise ValueError("camera intrinsics contain non-finite values")
    if focal <= 0.0 or aperture <= 0.0:
        raise ValueError("camera focal length and aperture must be positive")

    valid = (
        np.isfinite(depth)
        & (depth > float(MIN_VALID_DEPTH_M))
        & (depth < float(MAX_VALID_DEPTH_M))
    )
    metadata = {
        "build": PROJECTIVE_GEOMETRY_BUILD,
        "geometry_version": PROJECTIVE_GEOMETRY_VERSION,
        "effective_geometry_version": PROJECTIVE_GEOMETRY_VERSION,
        "geometry_representation": "frozen_metric_depth_projective_shell",
        "mesh_version": None,
        "effective_mesh_version": "not_built",
        "scene_mesh_built": False,
        "expert_applied": False,
        "fallback": None,
        "forbidden_inputs_used": [],
        "allowed_inputs": [
            "frozen_head_metric_depth",
            "frozen_head_camera_intrinsics",
            "frozen_head_camera_extrinsics",
        ],
        "depth_path": depth_path,
        "depth_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "depth_valid_pixels": int(valid.sum()),
        "depth_valid_fraction": float(valid.mean()),
        "camera_convention": "USD +X right +Y up -Z forward",
        "whole_scene_mesh_skipped": True,
        "whole_scene_mesh_skip_reason": (
            "local grasp occupancy is evaluated directly from frozen depth"
        ),
        "trimesh_contains_used": False,
        "pyembree_used": False,
        "elapsed_s": float(time.perf_counter() - started),
    }
    scene = ProjectiveDepthScene(
        depth=depth,
        camera_pos=camera_position,
        camera_quat_xyzw=camera_quaternion,
        focal_length=focal,
        horizontal_aperture=aperture,
        scene_key=str(scene_key),
        metadata=dict(metadata),
    )
    return scene, None, depth, metadata


def local_grid_spec(
    *,
    hit: Sequence[float],
    query_offsets: Iterable[np.ndarray],
    axial_z_m: np.ndarray,
    anchor_radius_m: float,
    voxel_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return the complete axis-aligned candidate query envelope."""
    voxel = float(voxel_m)
    if not np.isfinite(voxel) or voxel <= 0.0:
        raise ValueError("voxel_m must be positive and finite")
    parts = [
        np.asarray(part, dtype=np.float64).reshape(-1, 3)
        for part in query_offsets
    ]
    maximum_radius = 0.0
    for axial_z in np.asarray(axial_z_m, dtype=np.float64).reshape(-1):
        anchor_local = np.asarray([0.0, 0.0, axial_z], dtype=np.float64)
        for points in parts:
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
    radius = float(anchor_radius_m) + maximum_radius + voxel * 2.0
    center = np.asarray(hit, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center)):
        raise ValueError("hit contains non-finite values")
    origin = np.floor((center - radius) / voxel) * voxel
    upper = np.ceil((center + radius) / voxel) * voxel
    shape = np.rint((upper - origin) / voxel).astype(np.int64) + 1
    if np.any(shape <= 0):
        raise ValueError(f"invalid local occupancy shape {shape.tolist()}")
    return origin, upper, shape, float(radius)


def _torch_device(
    *,
    requested_device: Optional[str],
    allow_cpu_reference: bool,
):
    try:
        import torch
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"PyTorch is required for projective occupancy: {exc}"
        ) from exc

    requested = str(
        requested_device
        or os.environ.get("OFFICIAL_V2_LITE_OCCUPANCY_DEVICE", "cuda:0")
    ).strip()
    try:
        device = torch.device(requested)
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"invalid occupancy device {requested!r}: {exc}"
        ) from exc
    if device.type == "cpu":
        if not bool(allow_cpu_reference):
            raise ProjectiveOccupancyUnavailable(
                "CPU projective occupancy is disabled in runtime; CUDA is required"
            )
        return torch, device
    if device.type != "cuda":
        raise ProjectiveOccupancyUnavailable(
            f"unsupported occupancy device {device}; expected cuda:*"
        )
    if not torch.cuda.is_available():
        raise ProjectiveOccupancyUnavailable(
            "CUDA projective occupancy requested but torch.cuda.is_available() is false"
        )
    try:
        torch.empty(1, dtype=torch.uint8, device=device)
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"cannot initialize projective occupancy on {device}: {exc}"
        ) from exc
    return torch, device


def _hit_contract(
    occupancy: np.ndarray,
    *,
    origin: np.ndarray,
    hit: np.ndarray,
    voxel_m: float,
    maximum_distance_m: float,
) -> Dict[str, Any]:
    shape = np.asarray(occupancy.shape, dtype=np.int64)
    center_index = np.rint(
        (np.asarray(hit, dtype=np.float64) - origin) / float(voxel_m)
    ).astype(np.int64)
    radius_cells = max(
        1,
        int(np.ceil(float(maximum_distance_m) / float(voxel_m))),
    )
    lower = np.maximum(center_index - radius_cells, 0)
    upper = np.minimum(center_index + radius_cells + 1, shape)
    local = occupancy[
        int(lower[0]) : int(upper[0]),
        int(lower[1]) : int(upper[1]),
        int(lower[2]) : int(upper[2]),
    ]
    occupied_local = np.column_stack(np.nonzero(local))
    nearest_m = None
    if len(occupied_local):
        grid_index = occupied_local + lower.reshape(1, 3)
        points = origin.reshape(1, 3) + grid_index * float(voxel_m)
        nearest_m = float(
            np.linalg.norm(points - np.asarray(hit).reshape(1, 3), axis=1).min()
        )
    ok = nearest_m is not None and nearest_m <= float(maximum_distance_m) + 1e-12
    return {
        "ok": bool(ok),
        "nearest_occupied_distance_mm": (
            float(nearest_m * 1000.0) if nearest_m is not None else None
        ),
        "maximum_distance_mm": float(maximum_distance_m * 1000.0),
        "search_radius_cells": int(radius_cells),
        "local_occupied_voxels": int(len(occupied_local)),
    }


def build_projective_local_occupancy(
    scene: ProjectiveDepthScene,
    *,
    hit: Sequence[float],
    query_offsets: Iterable[np.ndarray],
    axial_z_m: np.ndarray,
    anchor_radius_m: float,
    voxel_m: float,
    ctx=None,
    requested_device: Optional[str] = None,
    allow_cpu_reference: bool = False,
    back_extrusion_m: float = DEFAULT_BACK_EXTRUSION_M,
    front_tolerance_m: float = DEFAULT_FRONT_TOLERANCE_M,
    pixel_radius: int = DEFAULT_PIXEL_RADIUS,
    chunk_voxels: Optional[int] = None,
    max_grid_voxels: Optional[int] = None,
    validate_hit: bool = True,
) -> ProjectiveOccupancyGrid:
    """Build local dense occupancy with projective depth-shell tests on Torch."""
    started = time.perf_counter()
    if not isinstance(scene, ProjectiveDepthScene):
        raise TypeError("scene must be ProjectiveDepthScene")
    query_parts = [
        np.asarray(part, dtype=np.float64).reshape(-1, 3)
        for part in query_offsets
    ]
    origin, upper, shape, radius = local_grid_spec(
        hit=hit,
        query_offsets=query_parts,
        axial_z_m=axial_z_m,
        anchor_radius_m=float(anchor_radius_m),
        voxel_m=float(voxel_m),
    )
    total = int(np.prod(shape, dtype=np.int64))
    maximum = int(
        max_grid_voxels
        if max_grid_voxels is not None
        else os.environ.get(
            "OFFICIAL_V2_LITE_GPU_OCCUPANCY_MAX_VOXELS",
            str(DEFAULT_MAX_GRID_VOXELS),
        )
    )
    if maximum <= 0:
        raise ValueError("max_grid_voxels must be positive")
    if total > maximum:
        raise ProjectiveOccupancyContractError(
            f"local occupancy grid has {total} voxels, limit is {maximum}"
        )
    extrusion = float(back_extrusion_m)
    front_tolerance = float(front_tolerance_m)
    if not np.isfinite(extrusion) or extrusion <= 0.0:
        raise ValueError("back_extrusion_m must be positive and finite")
    if not np.isfinite(front_tolerance) or front_tolerance < 0.0:
        raise ValueError("front_tolerance_m must be non-negative and finite")
    neighborhood = int(pixel_radius)
    if neighborhood < 0 or neighborhood > 3:
        raise ValueError("pixel_radius must be in [0, 3]")

    torch, device = _torch_device(
        requested_device=requested_device,
        allow_cpu_reference=bool(allow_cpu_reference),
    )
    chunk = int(
        chunk_voxels
        if chunk_voxels is not None
        else os.environ.get(
            "OFFICIAL_V2_LITE_GPU_OCCUPANCY_CHUNK_VOXELS",
            str(DEFAULT_CHUNK_VOXELS),
        )
    )
    if chunk <= 0:
        raise ValueError("chunk_voxels must be positive")
    chunk = min(chunk, total)

    free_bytes = None
    total_bytes = None
    if device.type == "cuda":
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except Exception as exc:
            raise ProjectiveOccupancyUnavailable(
                f"cannot query CUDA memory on {device}: {exc}"
            ) from exc
        reserve_mib = int(
            os.environ.get(
                "OFFICIAL_V2_LITE_GPU_OCCUPANCY_MIN_FREE_MIB",
                str(DEFAULT_MIN_FREE_CUDA_MIB),
            )
        )
        reserve_bytes = max(0, reserve_mib) * 1024 * 1024
        usable_bytes = int(free_bytes) - reserve_bytes
        # Integer indices, world/camera points, masks, and allocator overhead.
        conservative_bytes_per_chunk_voxel = 160
        memory_limited_chunk = usable_bytes // conservative_bytes_per_chunk_voxel
        if memory_limited_chunk < 16_384:
            raise ProjectiveOccupancyUnavailable(
                f"insufficient free CUDA memory on {device}: "
                f"{int(free_bytes) / 1024**2:.1f} MiB free, "
                f"{reserve_mib} MiB reserved"
            )
        chunk = min(chunk, int(memory_limited_chunk))

    depth = scene.depth_tensor(torch, device)
    height, width = [int(value) for value in scene.depth.shape]
    focal_px = (
        float(scene.focal_length)
        / float(scene.horizontal_aperture)
        * float(width)
    )
    rotation = torch.as_tensor(
        quat_to_mat_xyzw(scene.camera_quat_xyzw.copy()),
        dtype=torch.float32,
        device=device,
    )
    camera_position = torch.as_tensor(
        scene.camera_pos,
        dtype=torch.float32,
        device=device,
    )
    grid_origin = torch.as_tensor(
        origin,
        dtype=torch.float32,
        device=device,
    )
    occupancy_device = torch.empty(total, dtype=torch.bool, device=device)
    yz = int(shape[1] * shape[2])
    shape_z = int(shape[2])
    query_batches = 0
    cuda_start = None
    cuda_end = None
    if device.type == "cuda":
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()

    with torch.inference_mode():
        for start in range(0, total, chunk):
            if ctx is not None and hasattr(ctx, "raise_if_cancelled"):
                ctx.raise_if_cancelled("rgbd CUDA projective occupancy")
            stop = min(total, start + chunk)
            linear = torch.arange(
                start,
                stop,
                dtype=torch.int64,
                device=device,
            )
            ix = torch.div(linear, yz, rounding_mode="floor")
            remainder = linear - ix * yz
            iy = torch.div(remainder, shape_z, rounding_mode="floor")
            iz = remainder - iy * shape_z
            grid_index = torch.stack((ix, iy, iz), dim=1).to(torch.float32)
            world = grid_origin.reshape(1, 3) + grid_index * float(voxel_m)
            camera = (world - camera_position.reshape(1, 3)) @ rotation
            query_depth = -camera[:, 2]
            positive = query_depth > float(MIN_VALID_DEPTH_M)
            denominator = torch.clamp(query_depth, min=float(MIN_VALID_DEPTH_M))
            pixel_u = torch.round(
                float(focal_px) * camera[:, 0] / denominator + width / 2.0
            ).to(torch.int64)
            pixel_v = torch.round(
                height / 2.0
                - float(focal_px) * camera[:, 1] / denominator
            ).to(torch.int64)
            selected_depth = torch.zeros(
                stop - start,
                dtype=torch.float32,
                device=device,
            )
            selected_valid = torch.zeros(
                stop - start,
                dtype=torch.bool,
                device=device,
            )
            offsets = [(0, 0)]
            offsets.extend(
                sorted(
                    (
                        (offset_v, offset_u)
                        for offset_v in range(-neighborhood, neighborhood + 1)
                        for offset_u in range(-neighborhood, neighborhood + 1)
                        if offset_u != 0 or offset_v != 0
                    ),
                    key=lambda item: (
                        item[0] * item[0] + item[1] * item[1],
                        abs(item[0]) + abs(item[1]),
                        item[0],
                        item[1],
                    ),
                )
            )
            for offset_v, offset_u in offsets:
                sample_u = pixel_u + int(offset_u)
                sample_v = pixel_v + int(offset_v)
                in_image = (
                    positive
                    & (sample_u >= 0)
                    & (sample_u < width)
                    & (sample_v >= 0)
                    & (sample_v < height)
                )
                flat_pixel = (
                    torch.clamp(sample_v, 0, height - 1) * width
                    + torch.clamp(sample_u, 0, width - 1)
                )
                observed_depth = depth.reshape(-1)[flat_pixel]
                valid_depth = (
                    torch.isfinite(observed_depth)
                    & (observed_depth > float(MIN_VALID_DEPTH_M))
                    & (observed_depth < float(MAX_VALID_DEPTH_M))
                )
                use = (~selected_valid) & in_image & valid_depth
                selected_depth = torch.where(
                    use,
                    observed_depth,
                    selected_depth,
                )
                selected_valid |= use
            delta = query_depth - selected_depth
            occupied = (
                selected_valid
                & (delta >= -front_tolerance)
                & (delta <= extrusion + front_tolerance)
            )
            occupancy_device[start:stop] = occupied
            query_batches += 1

    if cuda_end is not None:
        cuda_end.record()
    occupied_voxels = int(occupancy_device.sum().item())
    occupancy = (
        occupancy_device.reshape(tuple(int(value) for value in shape))
        .detach()
        .cpu()
        .numpy()
        .astype(bool, copy=True)
    )
    cuda_kernel_s = None
    if cuda_start is not None and cuda_end is not None:
        cuda_end.synchronize()
        cuda_kernel_s = float(cuda_start.elapsed_time(cuda_end) / 1000.0)

    hit_array = np.asarray(hit, dtype=np.float64).reshape(3)
    hit_max_distance_m = max(float(voxel_m) * 2.5, front_tolerance * 2.0)
    hit_audit = _hit_contract(
        occupancy,
        origin=origin,
        hit=hit_array,
        voxel_m=float(voxel_m),
        maximum_distance_m=hit_max_distance_m,
    )
    if validate_hit and not hit_audit["ok"]:
        raise ProjectiveOccupancyContractError(
            "clicked RGB-D surface is missing from projective occupancy "
            f"(nearest={hit_audit['nearest_occupied_distance_mm']}mm, "
            f"limit={hit_audit['maximum_distance_mm']:.3f}mm)"
        )

    metadata = {
        "geometry_version": PROJECTIVE_GEOMETRY_VERSION,
        "geometry_representation": "frozen_metric_depth_projective_shell",
        "occupancy_method": (
            "cuda_projective_depth_shell"
            if device.type == "cuda"
            else "torch_cpu_projective_depth_shell_reference"
        ),
        "occupancy_device": str(device),
        "pose_count_device_required": (
            str(device) if device.type == "cuda" else None
        ),
        "cpu_runtime_fallback_allowed": bool(allow_cpu_reference),
        "occupancy_dtype": "bool",
        "occupancy_query_bounds": np.stack((origin, upper), axis=0).tolist(),
        "origin": origin.tolist(),
        "shape": [int(value) for value in shape],
        "grid_voxels": int(total),
        "occupied_voxels": int(occupied_voxels),
        "candidate_envelope_radius_m": float(radius),
        "voxel_m": float(voxel_m),
        "back_extrusion_mm": float(extrusion * 1000.0),
        "front_tolerance_mm": float(front_tolerance * 1000.0),
        "pixel_neighborhood_radius": int(neighborhood),
        "pixel_neighborhood_policy": (
            "center_depth_then_nearest_valid_fallback_no_cross_edge_union"
        ),
        "chunk_voxels": int(chunk),
        "query_batches": int(query_batches),
        "cuda_kernel_s": cuda_kernel_s,
        "elapsed_s": float(time.perf_counter() - started),
        "cuda_free_mib_before": (
            float(free_bytes / 1024**2) if free_bytes is not None else None
        ),
        "cuda_total_mib": (
            float(total_bytes / 1024**2) if total_bytes is not None else None
        ),
        "regular_grid_strategy": "projective_depth_gpu_chunks",
        "regular_grid_query_points": int(total),
        "mesh_split_calls": 0,
        "trimesh_contains_calls": 0,
        "pyembree_used": False,
        "visible_free_space_rule": (
            "voxels in front of observed depth beyond tolerance are empty"
        ),
        "hit_contract": hit_audit,
    }
    if ctx is not None and hasattr(ctx, "log"):
        ctx.log(
            "  [grasp_point_filter_rgbd] occupancy 3mm "
            f"shape={metadata['shape']} "
            f"occupied={occupied_voxels}/{total} "
            f"method={metadata['occupancy_method']} "
            f"device={device} elapsed={metadata['elapsed_s']:.3f}s "
            f"cuda_kernel={cuda_kernel_s}"
        )
    return ProjectiveOccupancyGrid(
        origin=origin,
        occupancy=occupancy,
        voxel_m=float(voxel_m),
        metadata=metadata,
    )


__all__ = [
    "DEFAULT_BACK_EXTRUSION_M",
    "DEFAULT_FRONT_TOLERANCE_M",
    "PROJECTIVE_GEOMETRY_BUILD",
    "PROJECTIVE_GEOMETRY_VERSION",
    "ProjectiveDepthScene",
    "ProjectiveOccupancyContractError",
    "ProjectiveOccupancyError",
    "ProjectiveOccupancyGrid",
    "ProjectiveOccupancyUnavailable",
    "build_projective_local_occupancy",
    "local_grid_spec",
    "prepare_projective_scene_from_session",
]
