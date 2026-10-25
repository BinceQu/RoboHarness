#!/usr/bin/env python3
"""Reusable evaluator occupancy matching the production mesh contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import trimesh


_MAX_DIRECT_SUBDIVIDE_VERTEX_ESTIMATE = 20_000_000
_CHUNK_SUBDIVIDE_VERTEX_ESTIMATE = 1_000_000
_ORTHOGONAL_RAY_SAMPLES_PER_CELL = 3


def _mesh_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _subdivide_vertex_estimate(
    mesh: trimesh.Trimesh,
    *,
    pitch: float,
    edge_factor: float = 2.0,
) -> tuple[np.ndarray, int]:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    edges = np.stack(
        (
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ),
        axis=1,
    )
    longest = np.linalg.norm(edges, axis=2).max(axis=1)
    max_edge = float(pitch) / float(edge_factor)
    levels = np.maximum(
        np.ceil(
            np.log2(
                np.maximum(longest / max(max_edge, 1e-12), 1.0)
            )
        ).astype(np.int64),
        0,
    )
    per_face = 3 * np.power(4, levels, dtype=np.int64)
    return per_face, int(per_face.sum())


def _chunked_subdivide_filled_voxel_points(
    mesh: trimesh.Trimesh,
    *,
    pitch: float,
    edge_factor: float = 2.0,
    chunk_vertex_estimate: int = _CHUNK_SUBDIVIDE_VERTEX_ESTIMATE,
) -> np.ndarray:
    """Match trimesh subdivide voxelization without one giant subdivision."""
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    per_face, _total_estimate = _subdivide_vertex_estimate(
        mesh,
        pitch=float(pitch),
        edge_factor=float(edge_factor),
    )
    max_edge = float(pitch) / float(edge_factor)
    global_max_iter = int(
        np.ceil(
            np.log2(
                max(
                    float(
                        np.linalg.norm(
                            np.asarray(mesh.vertices)[mesh.edges[:, 0]]
                            - np.asarray(mesh.vertices)[mesh.edges[:, 1]],
                            axis=1,
                        ).max()
                    )
                    / max(max_edge, 1e-12),
                    1.0,
                )
            )
        )
    )
    hit_parts: list[np.ndarray] = []
    pending_estimate = 0
    start = 0
    while start < len(triangles):
        stop = start
        pending_estimate = 0
        while stop < len(triangles):
            face_estimate = int(per_face[stop])
            if stop > start and (
                pending_estimate + face_estimate
                > int(chunk_vertex_estimate)
            ):
                break
            pending_estimate += face_estimate
            stop += 1
            if pending_estimate >= int(chunk_vertex_estimate):
                break

        chunk_triangles = triangles[start:stop]
        chunk_vertices = chunk_triangles.reshape(-1, 3)
        chunk_faces = np.arange(
            len(chunk_vertices),
            dtype=np.int64,
        ).reshape(-1, 3)
        subdivided_vertices, _subdivided_faces = (
            trimesh.remesh.subdivide_to_size(
                chunk_vertices,
                chunk_faces,
                max_edge=max_edge,
                max_iter=global_max_iter,
            )
        )
        hit_parts.append(
            np.round(subdivided_vertices / float(pitch)).astype(
                np.int64
            )
        )
        start = stop

    occupied_index = np.unique(
        np.concatenate(hit_parts, axis=0),
        axis=0,
    )
    origin_index = occupied_index.min(axis=0)
    local_index = occupied_index - origin_index
    shape = local_index.max(axis=0) + 1
    dense = np.zeros(tuple(shape.tolist()), dtype=bool)
    dense[tuple(local_index.T)] = True
    filled = ndimage.binary_fill_holes(dense)
    filled_index = np.column_stack(np.nonzero(filled))
    return (
        filled_index.astype(np.float64)
        + origin_index.reshape(1, 3)
    ) * float(pitch)


def _orthogonal_ray_filled_voxel_points(
    mesh: trimesh.Trimesh,
    *,
    pitch: float,
    samples_per_cell: int = _ORTHOGONAL_RAY_SAMPLES_PER_CELL,
) -> np.ndarray:
    """Build a closed surface from rays along all three voxel axes."""
    per_cell = int(samples_per_cell)
    if per_cell < 1:
        raise ValueError("samples_per_cell must be positive")
    hit_parts: list[np.ndarray] = []
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    for ray_axis in range(3):
        other_axes = [
            axis for axis in range(3) if axis != ray_axis
        ]
        permutation = other_axes + [ray_axis]
        inverse = np.argsort(permutation)
        permuted = trimesh.Trimesh(
            vertices=vertices[:, permutation],
            faces=faces,
            process=False,
        )
        ray_voxels = permuted.voxelized(
            pitch=float(pitch),
            method="ray",
            per_cell=[per_cell, per_cell],
        )
        ray_points = np.asarray(
            ray_voxels.points,
            dtype=np.float64,
        )[:, inverse]
        hit_parts.append(
            np.round(ray_points / float(pitch)).astype(np.int64)
        )

    occupied_index = np.unique(
        np.concatenate(hit_parts, axis=0),
        axis=0,
    )
    origin_index = occupied_index.min(axis=0)
    local_index = occupied_index - origin_index
    shape = local_index.max(axis=0) + 1
    dense = np.zeros(tuple(shape.tolist()), dtype=bool)
    dense[tuple(local_index.T)] = True
    filled = ndimage.binary_fill_holes(dense)
    filled_index = np.column_stack(np.nonzero(filled))
    return (
        filled_index.astype(np.float64)
        + origin_index.reshape(1, 3)
    ) * float(pitch)


@dataclass
class MeshOccupancyEvaluator:
    """Query mesh occupancy without rebuilding fallback voxels per sphere."""

    mesh: trimesh.Trimesh
    voxel_m: float
    method: str
    voxel_points: np.ndarray | None = None
    voxelization_method: str | None = None
    components: tuple[trimesh.Trimesh, ...] | None = None
    source_component_count: int | None = None
    query_bounds: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._tree = (
            cKDTree(self.voxel_points)
            if self.voxel_points is not None and len(self.voxel_points)
            else None
        )
        self._component_bounds = (
            np.asarray(
                [component.bounds for component in self.components],
                dtype=np.float64,
            )
            if self.components
            else None
        )

    @classmethod
    def build(
        cls,
        mesh: trimesh.Trimesh,
        *,
        voxel_m: float,
        mesh_path: Path | None = None,
        voxel_cache_path: Path | None = None,
        query_bounds: np.ndarray | None = None,
    ) -> "MeshOccupancyEvaluator":
        scoped_bounds = None
        if query_bounds is not None:
            scoped_bounds = np.asarray(
                query_bounds,
                dtype=np.float64,
            ).reshape(2, 3)
            if np.any(scoped_bounds[0] > scoped_bounds[1]):
                raise ValueError("query_bounds minimum exceeds maximum")

        if bool(mesh.is_watertight):
            components = tuple(mesh.split(only_watertight=False))
            if not components:
                components = (mesh,)
            source_component_count = int(len(components))
            if all(
                bool(component.is_watertight)
                for component in components
            ):
                active_components = components
                if scoped_bounds is not None:
                    active_components = tuple(
                        component
                        for component in components
                        if bool(
                            np.all(
                                component.bounds[1]
                                >= scoped_bounds[0] - 1e-12
                            )
                            and np.all(
                                component.bounds[0]
                                <= scoped_bounds[1] + 1e-12
                            )
                        )
                    )
                if not active_components:
                    return cls(
                        mesh=mesh,
                        voxel_m=float(voxel_m),
                        method="empty_components",
                        components=(),
                        source_component_count=source_component_count,
                        query_bounds=scoped_bounds,
                    )
                if len(active_components) == 1:
                    active_mesh = active_components[0]
                    return cls(
                        mesh=active_mesh,
                        voxel_m=float(voxel_m),
                        method="contains",
                        components=active_components,
                        source_component_count=source_component_count,
                        query_bounds=scoped_bounds,
                    )
                return cls(
                    mesh=mesh,
                    voxel_m=float(voxel_m),
                    method="component_union_contains",
                    components=active_components,
                    source_component_count=source_component_count,
                    query_bounds=scoped_bounds,
                )

        expected_hash = (
            _mesh_sha256(mesh_path)
            if mesh_path is not None and mesh_path.is_file()
            else None
        )
        voxel_points: np.ndarray | None = None
        voxelization_method: str | None = None
        if voxel_cache_path is not None and voxel_cache_path.is_file():
            cache = np.load(voxel_cache_path)
            cache_pitch = float(cache["voxel_m"])
            cache_hash = (
                str(cache["mesh_sha256"].item())
                if "mesh_sha256" in cache
                else None
            )
            if (
                abs(cache_pitch - float(voxel_m)) <= 1e-12
                and (expected_hash is None or cache_hash == expected_hash)
            ):
                voxel_points = np.asarray(
                    cache["voxel_points"],
                    dtype=np.float64,
                ).reshape(-1, 3)
                voxelization_method = (
                    str(cache["voxelization_method"].item())
                    if "voxelization_method" in cache
                    else "cached_unknown"
                )

        if voxel_points is None:
            _per_face, vertex_estimate = _subdivide_vertex_estimate(
                mesh,
                pitch=float(voxel_m),
            )
            if vertex_estimate > _MAX_DIRECT_SUBDIVIDE_VERTEX_ESTIMATE:
                voxel_points = _orthogonal_ray_filled_voxel_points(
                    mesh,
                    pitch=float(voxel_m),
                )
                voxelization_method = (
                    "orthogonal_ray_holes_per_cell_"
                    f"{_ORTHOGONAL_RAY_SAMPLES_PER_CELL}"
                )
            else:
                voxels = mesh.voxelized(pitch=float(voxel_m))
                try:
                    voxels = voxels.fill()
                except Exception:
                    pass
                voxel_points = np.asarray(
                    voxels.points,
                    dtype=np.float64,
                )
                voxelization_method = "trimesh_subdivide_holes"
            if voxel_cache_path is not None:
                voxel_cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    voxel_cache_path,
                    voxel_points=voxel_points,
                    voxel_m=np.array(float(voxel_m)),
                    mesh_sha256=np.array(expected_hash or ""),
                    voxelization_method=np.array(
                        voxelization_method
                    ),
                )

        method = (
            "filled_voxel_nearest"
            if len(voxel_points)
            else "empty_voxelized"
        )
        return cls(
            mesh=mesh,
            voxel_m=float(voxel_m),
            method=method,
            voxel_points=voxel_points,
            voxelization_method=voxelization_method,
            query_bounds=scoped_bounds,
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.method == "contains":
            return np.asarray(self.mesh.contains(values), dtype=bool)
        if self.method == "component_union_contains":
            occupied = np.zeros(len(values), dtype=bool)
            assert self.components is not None
            assert self._component_bounds is not None
            for component, bounds in zip(
                self.components,
                self._component_bounds,
            ):
                candidates = (
                    (~occupied)
                    & np.all(values >= bounds[0] - 1e-12, axis=1)
                    & np.all(values <= bounds[1] + 1e-12, axis=1)
                )
                if not np.any(candidates):
                    continue
                occupied[candidates] = np.asarray(
                    component.contains(values[candidates]),
                    dtype=bool,
                )
            return occupied
        if self._tree is None:
            return np.zeros(len(values), dtype=bool)
        distance, _ = self._tree.query(values, k=1)
        threshold = float(self.voxel_m) * math.sqrt(3.0) * 0.55
        return np.asarray(distance <= threshold, dtype=bool)

    def overlap_counts(
        self,
        centers: np.ndarray,
        *,
        offsets: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        local = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
        counts = np.zeros(len(values), dtype=np.int32)
        for start in range(0, len(values), int(batch_size)):
            stop = min(len(values), start + int(batch_size))
            query = (
                values[start:stop, None, :] + local[None, :, :]
            ).reshape(-1, 3)
            occupied = self.contains(query)
            counts[start:stop] = occupied.reshape(
                stop - start,
                len(local),
            ).sum(axis=1)
        return counts

    def metadata(self) -> dict[str, Any]:
        return {
            "occupancy_method": self.method,
            "occupancy_voxel_m": float(self.voxel_m),
            "occupancy_voxel_point_count": (
                int(len(self.voxel_points))
                if self.voxel_points is not None
                else None
            ),
            "occupancy_voxelization_method": (
                self.voxelization_method
            ),
            "occupancy_component_count": (
                int(len(self.components))
                if self.components is not None
                else 1
            ),
            "occupancy_source_component_count": (
                int(self.source_component_count)
                if self.source_component_count is not None
                else (
                    int(len(self.components))
                    if self.components is not None
                    else 1
                )
            ),
            "occupancy_culled_component_count": (
                int(self.source_component_count - len(self.components))
                if (
                    self.source_component_count is not None
                    and self.components is not None
                )
                else 0
            ),
            "occupancy_query_bounds": (
                np.asarray(self.query_bounds, dtype=np.float64).tolist()
                if self.query_bounds is not None
                else None
            ),
        }
