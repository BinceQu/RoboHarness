"""Continuous queries of the frozen official X5 inner-triangle volume.

The X-Z grid below is only a triangle *index*, not a stepped boundary. Every
answer interpolates the original triangle planes. No erosion, voxel rounding,
scene geometry, segmentation, or simulator access enters membership queries.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from tools.util.x5_grasp_geometry import (
    VOXEL_M,
    x5_dynamic_opening_mesh,
    x5_gripper_command_fraction,
    x5_gripper_geometry,
    x5_gripper_joint_position_m,
)

OPENING_QUERY_VERSION = "official_x5_continuous_inner_triangle_intervals_v1"


class InnerTriangleIntervals:
    """Indexed barycentric lower/upper inner surfaces of the frozen asset."""

    def __init__(self) -> None:
        geometry = x5_gripper_geometry()
        vertices = np.asarray(geometry["exact_opening_mesh_vertices"])
        faces = np.asarray(geometry["exact_opening_mesh_faces"])
        sides = np.asarray(geometry["exact_opening_mesh_vertex_side"])
        lower_indices = np.flatnonzero(sides < 0)
        upper_indices = np.flatnonzero(sides > 0)
        upper_by_xz = {tuple(vertices[i, [0, 2]]): i for i in upper_indices}
        if len(upper_by_xz) != len(upper_indices):
            raise ValueError("X5 exact upper surface has duplicate X-Z vertices")
        mates = np.full(len(vertices), -1, dtype=np.int64)
        for i in lower_indices:
            mates[i] = upper_by_xz.get(tuple(vertices[i, [0, 2]]), -1)
        if np.any(mates[lower_indices] < 0):
            raise ValueError("X5 exact inner surfaces are not paired in X-Z")
        lower_faces = faces[np.all(sides[faces] < 0, axis=1)]
        upper_faces = faces[np.all(sides[faces] > 0, axis=1)]
        if {tuple(sorted(f)) for f in mates[lower_faces]} != {
            tuple(sorted(f)) for f in upper_faces
        }:
            raise ValueError("X5 exact inner surfaces have different triangulations")
        xz = vertices[lower_faces][:, :, [0, 2]]
        matrix = np.stack((xz[:, 1] - xz[:, 0], xz[:, 2] - xz[:, 0]), axis=2)
        valid = np.abs(np.linalg.det(matrix)) > 1e-18
        self.origin = xz[valid, 0]
        self.inverse = np.linalg.inv(matrix[valid])
        self.lower_y = vertices[lower_faces[valid], 1]
        self.upper_y = vertices[mates[lower_faces[valid]], 1]
        if np.any(self.upper_y <= self.lower_y):
            raise ValueError("X5 exact reference inner surfaces cross")
        self.bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        self.open_m = float(geometry["metadata"]["gripper_open_m"])
        # A bucket contains all triangles whose bounds intersect it; it does
        # not clip or approximate the triangle. Query resolution is continuous.
        self.buckets: dict[tuple[int, int], np.ndarray] = {}
        bucket_lists: dict[tuple[int, int], list[int]] = {}
        for index, triangle in enumerate(xz[valid]):
            low = np.floor((triangle.min(axis=0) - 1e-10) / VOXEL_M).astype(int)
            high = np.floor((triangle.max(axis=0) + 1e-10) / VOXEL_M).astype(int)
            for ix in range(low[0], high[0] + 1):
                for iz in range(low[1], high[1] + 1):
                    bucket_lists.setdefault((ix, iz), []).append(index)
        self.buckets = {
            key: np.asarray(value, dtype=int) for key, value in bucket_lists.items()
        }

    def y_bounds(
        self, points_eef: Any, gripper_qpos: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_eef, dtype=np.float64).reshape(-1, 3)
        lower = np.full(len(points), np.nan)
        upper = np.full(len(points), np.nan)
        closure = self.open_m - x5_gripper_joint_position_m(gripper_qpos)
        eligible = np.all(np.isfinite(points), axis=1)
        eligible &= np.all(points[:, [0, 2]] >= self.bounds[0, [0, 2]] - 1e-10, axis=1)
        eligible &= np.all(points[:, [0, 2]] <= self.bounds[1, [0, 2]] + 1e-10, axis=1)
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return lower, upper
        keys = np.floor(points[indices][:, [0, 2]] / VOXEL_M).astype(np.int64)
        unique, groups = np.unique(keys, axis=0, return_inverse=True)
        order = np.argsort(groups, kind="stable")
        boundaries = np.r_[0, np.cumsum(np.bincount(groups))]
        for group, key in enumerate(unique):
            triangles = self.buckets.get(tuple(key))
            if triangles is None:
                continue
            selected = indices[order[boundaries[group] : boundaries[group + 1]]]
            # Bound memory even for a dense wrist frame or repeated points.
            for start in range(0, len(selected), 4096):
                chunk = selected[start : start + 4096]
                delta = points[chunk][:, None, [0, 2]] - self.origin[triangles][None]
                bc = np.einsum("tij,ptj->pti", self.inverse[triangles], delta)
                weights = np.concatenate(
                    (1.0 - bc.sum(axis=2, keepdims=True), bc), axis=2
                )
                inside = np.all(weights >= -1e-9, axis=2)
                present = inside.any(axis=1)
                if not np.any(present):
                    continue
                row = np.flatnonzero(present)
                column = np.argmax(inside[present], axis=1)
                w = weights[row, column]
                tri = triangles[column]
                lower[chunk[row]] = (
                    np.einsum("ij,ij->i", w, self.lower_y[tri]) + closure
                )
                upper[chunk[row]] = (
                    np.einsum("ij,ij->i", w, self.upper_y[tri]) - closure
                )
        return lower, upper

    def contains(self, points_eef: Any, gripper_qpos: Any) -> np.ndarray:
        points = np.asarray(points_eef, dtype=np.float64).reshape(-1, 3)
        result = np.zeros(len(points), dtype=bool)
        closure = self.open_m - x5_gripper_joint_position_m(gripper_qpos)
        eligible = np.all(np.isfinite(points), axis=1)
        eligible &= points[:, 1] > self.bounds[0, 1] + closure
        eligible &= points[:, 1] < self.bounds[1, 1] - closure
        selected = np.flatnonzero(eligible)
        lower, upper = self.y_bounds(points[selected], gripper_qpos)
        result[selected] = (points[selected, 1] > lower) & (points[selected, 1] < upper)
        return result


@lru_cache(maxsize=1)
def exact_opening_query() -> InnerTriangleIntervals:
    return InnerTriangleIntervals()


def points_inside_exact_x5_opening(points_eef: Any, gripper_qpos: Any) -> np.ndarray:
    return exact_opening_query().contains(points_eef, gripper_qpos)


@lru_cache(maxsize=32)
def _opening_voxels(command: float) -> np.ndarray:
    bounds = exact_opening_query().bounds
    axes = [
        np.arange(
            int(np.ceil(bounds[0, i] / VOXEL_M)),
            int(np.floor(bounds[1, i] / VOXEL_M)) + 1,
        )
        for i in range(3)
    ]
    candidates = (
        np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3) * VOXEL_M
    )
    result = candidates[points_inside_exact_x5_opening(candidates, [command])]
    result.setflags(write=False)
    return result


def exact_x5_opening_voxels(gripper_qpos: Any) -> np.ndarray:
    """3 mm quadrature centers classified by exact triangles, with no erosion."""
    return _opening_voxels(x5_gripper_command_fraction(gripper_qpos))


@lru_cache(maxsize=1)
def exact_x5_planning_geometry() -> dict[str, Any]:
    """Use exact opening quadrature; leave the frozen legacy asset untouched."""
    legacy = x5_gripper_geometry()
    voxels = exact_x5_opening_voxels([1.0])
    mesh = x5_dynamic_opening_mesh([1.0])
    metadata = dict(
        legacy["metadata"],
        opening_query_version=OPENING_QUERY_VERSION,
        opening_voxel_definition="3mm_centers_inside_exact_inner_triangles_no_erosion",
        legacy_opening_voxel_count=len(legacy["opening_voxels"]),
        opening_voxel_count=len(voxels),
        opening_bounds_eef_m=[
            mesh["vertices"].min(axis=0).tolist(),
            mesh["vertices"].max(axis=0).tolist(),
        ],
    )
    axial = np.linspace(float(voxels[:, 0].min()), float(voxels[:, 0].max()), 7)
    axial.setflags(write=False)
    return {
        **legacy,
        "opening_voxels": voxels,
        "axial_x_m": axial,
        "metadata": metadata,
    }
