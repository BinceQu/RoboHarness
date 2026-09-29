"""CUDA-only topology operations used while constructing V53 meshes."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .rgbd_projective_occupancy import (
    ProjectiveOccupancyContractError,
    ProjectiveOccupancyUnavailable,
)


CUDA_MESH_OPS_VERSION = "rgbd_v53_cuda_mesh_ops_v1"
DEFAULT_MAX_CORNERS = 6_000_000
DEFAULT_COMPONENT_ITERATIONS = 64
DEFAULT_MIN_FREE_CUDA_MIB = 512


def _device(
    requested_device: Optional[str],
    *,
    allow_cpu_reference: bool,
):
    try:
        import torch
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"Torch is required for V53 CUDA mesh operations: {exc}"
        ) from exc
    name = str(
        requested_device
        or os.environ.get("OFFICIAL_V2_LITE_OCCUPANCY_DEVICE", "cuda:0")
    ).strip()
    device = torch.device(name)
    if device.type == "cpu":
        if not allow_cpu_reference:
            raise ProjectiveOccupancyUnavailable(
                "CPU V53 mesh operations are disabled in runtime"
            )
    elif device.type != "cuda":
        raise ProjectiveOccupancyUnavailable(
            f"unsupported V53 mesh operation device {device}"
        )
    try:
        torch.empty(1, dtype=torch.uint8, device=device)
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"V53 mesh operation device {device} is unavailable: {exc}"
        ) from exc
    return torch, device


def _memory_preflight(torch, device, *, corners: int) -> Dict[str, Any]:
    if device.type != "cuda":
        return {
            "cuda_free_mib_before": None,
            "estimated_required_mib": None,
        }
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    reserve_mib = max(
        0,
        int(
            os.environ.get(
                "OFFICIAL_V2_LITE_GPU_OCCUPANCY_MIN_FREE_MIB",
                str(DEFAULT_MIN_FREE_CUDA_MIB),
            )
        ),
    )
    estimated_bytes = int(corners * 224 + 64 * 1024**2)
    if int(free_bytes) - reserve_mib * 1024**2 < estimated_bytes:
        raise ProjectiveOccupancyUnavailable(
            f"insufficient CUDA memory for V53 fan splitting on {device}: "
            f"{free_bytes / 1024**2:.1f} MiB free, "
            f"{estimated_bytes / 1024**2:.1f} MiB estimated, "
            f"{reserve_mib} MiB reserved"
        )
    return {
        "cuda_free_mib_before": float(free_bytes / 1024**2),
        "estimated_required_mib": float(estimated_bytes / 1024**2),
    }


def split_nonmanifold_vertex_fans_cuda(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    requested_device: Optional[str] = None,
    allow_cpu_reference: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Duplicate disconnected incident fans without CPU graph traversal.

    Triangle corners are graph nodes. Corners at the same endpoint of a
    shared edge are joined, so each connected corner component is exactly one
    incident fan. Output ordering matches the historical CPU implementation:
    original vertices first, then extra fans ordered by source vertex and the
    minimum source face in the fan.
    """
    started = time.perf_counter()
    source_points = np.ascontiguousarray(
        np.asarray(points, dtype=np.float64).reshape(-1, 3)
    )
    source_faces = np.ascontiguousarray(
        np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    )
    if not len(source_points) or not len(source_faces):
        raise ProjectiveOccupancyContractError(
            "V53 fan splitting requires non-empty points and triangles"
        )
    if int(source_faces.min()) < 0 or int(source_faces.max()) >= len(
        source_points
    ):
        raise ProjectiveOccupancyContractError(
            "V53 fan splitting received invalid triangle indices"
        )
    corner_count = int(3 * len(source_faces))
    maximum_corners = int(
        os.environ.get(
            "OFFICIAL_V2_V53_GPU_FAN_MAX_CORNERS",
            str(DEFAULT_MAX_CORNERS),
        )
    )
    if maximum_corners <= 0 or corner_count > maximum_corners:
        raise ProjectiveOccupancyContractError(
            f"V53 fan graph has {corner_count} corners, limit is "
            f"{maximum_corners}"
        )

    torch, device = _device(
        requested_device,
        allow_cpu_reference=bool(allow_cpu_reference),
    )
    memory = _memory_preflight(torch, device, corners=corner_count)
    try:
        with torch.inference_mode():
            point_tensor = torch.as_tensor(
                source_points,
                dtype=torch.float64,
                device=device,
            )
            faces = torch.as_tensor(
                source_faces,
                dtype=torch.int64,
                device=device,
            )
            corner = torch.arange(
                corner_count,
                dtype=torch.int64,
                device=device,
            ).reshape(-1, 3)
            directed_vertices = torch.cat(
                (
                    faces[:, (0, 1)],
                    faces[:, (1, 2)],
                    faces[:, (2, 0)],
                ),
                dim=0,
            )
            directed_corners = torch.cat(
                (
                    corner[:, (0, 1)],
                    corner[:, (1, 2)],
                    corner[:, (2, 0)],
                ),
                dim=0,
            )
            edge_low = torch.minimum(
                directed_vertices[:, 0],
                directed_vertices[:, 1],
            )
            edge_high = torch.maximum(
                directed_vertices[:, 0],
                directed_vertices[:, 1],
            )
            edge_key = edge_low * int(len(source_points)) + edge_high
            sorted_key, order = torch.sort(edge_key)
            sorted_vertices = directed_vertices[order]
            sorted_corners = directed_corners[order]
            sorted_low = edge_low[order]
            low_corner = torch.where(
                sorted_vertices[:, 0] == sorted_low,
                sorted_corners[:, 0],
                sorted_corners[:, 1],
            )
            high_corner = torch.where(
                sorted_vertices[:, 0] == sorted_low,
                sorted_corners[:, 1],
                sorted_corners[:, 0],
            )
            same_edge = sorted_key[1:] == sorted_key[:-1]
            adjacent_a = torch.cat(
                (low_corner[:-1][same_edge], high_corner[:-1][same_edge])
            )
            adjacent_b = torch.cat(
                (low_corner[1:][same_edge], high_corner[1:][same_edge])
            )

            parent = torch.arange(
                corner_count,
                dtype=torch.int64,
                device=device,
            )
            maximum_iterations = max(
                1,
                int(
                    os.environ.get(
                        "OFFICIAL_V2_V53_GPU_FAN_COMPONENT_ITERATIONS",
                        str(DEFAULT_COMPONENT_ITERATIONS),
                    )
                ),
            )
            converged = False
            for iterations in range(1, maximum_iterations + 1):
                root_a = parent[adjacent_a]
                root_b = parent[adjacent_b]
                higher = torch.maximum(root_a, root_b)
                lower = torch.minimum(root_a, root_b)
                updated = parent.clone()
                updated.scatter_reduce_(
                    0,
                    higher,
                    lower,
                    reduce="amin",
                    include_self=True,
                )
                updated = updated[updated]
                if bool(torch.equal(updated, parent)):
                    parent = updated
                    converged = True
                    break
                parent = updated
            if not converged:
                raise ProjectiveOccupancyContractError(
                    "V53 CUDA fan component labeling did not converge in "
                    f"{maximum_iterations} iterations"
                )
            for _ in range(4):
                parent = parent[parent]

            flat_vertices = faces.reshape(-1)
            face_ids = torch.arange(
                len(faces),
                dtype=torch.int64,
                device=device,
            ).repeat_interleave(3)
            roots = torch.unique(parent)
            root_min_face = torch.full(
                (corner_count,),
                len(faces),
                dtype=torch.int64,
                device=device,
            )
            root_min_face.scatter_reduce_(
                0,
                parent,
                face_ids,
                reduce="amin",
                include_self=True,
            )
            root_vertex = flat_vertices[roots]
            fan_sort_key = (
                root_vertex * (int(len(faces)) + 1)
                + root_min_face[roots]
            )
            fan_order = torch.argsort(fan_sort_key)
            ordered_roots = roots[fan_order]
            ordered_vertices = root_vertex[fan_order]
            first_fan = torch.ones(
                len(ordered_roots),
                dtype=torch.bool,
                device=device,
            )
            if len(ordered_roots) > 1:
                first_fan[1:] = (
                    ordered_vertices[1:] != ordered_vertices[:-1]
                )
            duplicate_fan = ~first_fan
            duplicate_vertices = ordered_vertices[duplicate_fan]
            fan_output = torch.full(
                (corner_count,),
                -1,
                dtype=torch.int64,
                device=device,
            )
            ordered_output = ordered_vertices.clone()
            ordered_output[duplicate_fan] = (
                int(len(source_points))
                + torch.arange(
                    int(duplicate_fan.sum().item()),
                    dtype=torch.int64,
                    device=device,
                )
            )
            fan_output[ordered_roots] = ordered_output
            output_faces = fan_output[parent].reshape(-1, 3)
            output_points = torch.cat(
                (point_tensor, point_tensor[duplicate_vertices]),
                dim=0,
            )
            split_vertices = int(torch.unique(duplicate_vertices).numel())
            added_vertex_fans = int(len(duplicate_vertices))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            output_points_np = output_points.cpu().numpy()
            output_faces_np = output_faces.cpu().numpy()
    except ProjectiveOccupancyContractError:
        raise
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"V53 fan splitting failed on {device}: {exc}"
        ) from exc

    return (
        np.ascontiguousarray(output_points_np, dtype=np.float64),
        np.ascontiguousarray(output_faces_np, dtype=np.int64),
        {
            "split_nonmanifold_vertices": split_vertices,
            "added_vertex_fans": added_vertex_fans,
            "fan_split_method": "torch_corner_component_union_find",
            "fan_split_device": str(device),
            "fan_split_component_iterations": int(iterations),
            "fan_split_elapsed_s": float(time.perf_counter() - started),
            **memory,
        },
    )


__all__ = [
    "CUDA_MESH_OPS_VERSION",
    "split_nonmanifold_vertex_fans_cuda",
]
