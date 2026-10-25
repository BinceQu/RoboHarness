"""Explicit opt-in continuous-cell cover; never enabled by default.

Cell/primitive intersection replaces three normal samples plus blanket query
dilation in offline comparisons or the planner's operator-selected route.
The builder does not decide runtime activation. It covers the chosen RGB-D
interpolation/uncertainty prior, not unknown physical object interiors. The
existing source cells are retained. No click, object labels, new camera or
simulator state is accepted here. Failed geometry/budget checks raise rather
than returning a partially rasterized scene that could appear collision-free.
"""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from tools.util import rgbd_scene_mesh_robodojo_v1 as mesh

BUILD = "continuous_rgbd_prism_cell_cover_experiment_v1"
_PRISM_EDGES = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5))
_PRISM_FACES = ((0, 1, 2), (3, 4, 5), (0, 1, 3), (1, 2, 4), (2, 0, 5))


class CellCover:
    """Conservative closed-cell supercover of segments and convex prisms.

    SAT uses all prism face normals and prism-edge/cube-edge cross products.
    Exact boundary touches may add a cell; there is no neighbor dilation.
    """

    def __init__(
        self, origin, voxel_m, *, max_candidate_cells=60_000_000, chunk_size=256
    ):
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3).copy()
        self.voxel_m = float(voxel_m)
        if (
            not np.isfinite(self.origin).all()
            or not math.isfinite(self.voxel_m)
            or self.voxel_m <= 0.0
        ):
            raise ValueError("finite origin and positive voxel size required")
        self.budget = int(max_candidate_cells)
        self.chunk_size = int(chunk_size)
        if self.budget < 1 or self.chunk_size < 1:
            raise ValueError("positive cover budget/chunk size required")
        self.candidate_count = 0
        self.primitive_count = 0

    def _batches(self, vertices):
        vertices = (np.asarray(vertices, dtype=np.float64) - self.origin) / self.voxel_m
        if (
            vertices.ndim != 3
            or vertices.shape[-1] != 3
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("primitive vertices must be finite NxKx3")
        if np.any(np.abs(vertices) >= (1 << 20) - 2):
            raise ValueError("primitive exceeds the packed voxel coordinate range")
        # 1e-8 voxel units, not an extra metric collision clearance.
        lower = np.floor(vertices.min(axis=1) - 1e-8).astype(np.int64)
        upper = np.floor(vertices.max(axis=1) + 1e-8).astype(np.int64)
        sizes = upper - lower + 1
        if np.any(sizes > 1024):
            raise ValueError("primitive is too large for bounded voxelization")
        counts = np.prod(sizes, axis=1, dtype=np.int64)
        required = int(counts.sum())
        if self.candidate_count + required > self.budget:
            raise RuntimeError(
                f"continuous_cover_budget_exceeded: {self.candidate_count + required}>{self.budget}"
            )
        if len(counts) and counts.max() > 1_000_000:
            raise RuntimeError("continuous_cover_single_primitive_budget_exceeded")
        self.candidate_count += required
        self.primitive_count += len(vertices)
        for first in range(0, len(vertices), self.chunk_size):
            last = min(first + self.chunk_size, len(vertices))
            count = counts[first:last]
            ids = np.repeat(np.arange(last - first), count)
            start = np.r_[0, np.cumsum(count)[:-1]]
            offset = np.arange(int(count.sum())) - start[ids]
            size = sizes[first:last][ids]
            index = np.column_stack(
                (
                    offset // (size[:, 1] * size[:, 2]),
                    (offset // size[:, 2]) % size[:, 1],
                    offset % size[:, 2],
                )
            )
            index += lower[first:last][ids]
            yield vertices[first:last], ids, index

    def segments(self, start, end):
        vertices = np.stack(
            (np.asarray(start).reshape(-1, 3), np.asarray(end).reshape(-1, 3)), axis=1
        )
        keys = []
        for values, ids, index in self._batches(vertices):
            low, high = np.zeros(len(index)), np.ones(len(index))
            alive = np.ones(len(index), bool)
            delta = values[:, 1] - values[:, 0]
            for axis in range(3):
                begin, direction = values[ids, 0, axis], delta[ids, axis]
                parallel = np.abs(direction) < 1e-14
                alive &= ~parallel | (
                    (begin >= index[:, axis] - 1e-8)
                    & (begin <= index[:, axis] + 1.0 + 1e-8)
                )
                first = np.zeros(len(index))
                second = np.zeros(len(index))
                np.divide(index[:, axis] - begin, direction, out=first, where=~parallel)
                np.divide(
                    index[:, axis] + 1.0 - begin, direction, out=second, where=~parallel
                )
                low = np.maximum(
                    low, np.where(parallel, -np.inf, np.minimum(first, second))
                )
                high = np.minimum(
                    high, np.where(parallel, np.inf, np.maximum(first, second))
                )
            alive &= low <= high + 1e-8
            keys.append(mesh._pack_indices(index[alive]))
        return self._indices(keys)

    def prisms(self, front, back):
        front = np.asarray(front, dtype=np.float64).reshape(-1, 3, 3)
        back = np.asarray(back, dtype=np.float64).reshape(-1, 3, 3)
        if front.shape != back.shape:
            raise ValueError("front/back triangles must match")
        vertices = np.concatenate((front, back), axis=1)
        keys = []
        for values, ids, index in self._batches(vertices):
            face_axes = [
                np.cross(values[:, b] - values[:, a], values[:, c] - values[:, a])
                for a, b, c in _PRISM_FACES
            ]
            # Each quadrilateral must be planar and bound a convex prism.
            # Unknown/malformed solids are not silently approximated.
            for face_index, ((a, _, _), normal) in enumerate(
                zip(_PRISM_FACES, face_axes)
            ):
                norm = np.linalg.norm(normal, axis=1)
                if np.any(norm < 1e-12):
                    raise ValueError("degenerate continuous prism")
                distance = (
                    np.einsum("nvi,ni->nv", values - values[:, a, None], normal)
                    / norm[:, None]
                )
                if face_index >= 2 and np.any(
                    np.abs(distance[:, (4, 5, 3)[face_index - 2]]) > 1e-7
                ):
                    raise ValueError("nonplanar continuous prism side")
                if np.any(
                    (distance.max(axis=1) > 1e-7) & (distance.min(axis=1) < -1e-7)
                ):
                    raise ValueError("nonconvex or nonplanar continuous prism")
            alive = np.ones(len(index), bool)
            axes = list(face_axes)
            for a, b in _PRISM_EDGES:
                edge = values[:, b] - values[:, a]
                axes.extend(np.cross(edge, basis) for basis in np.eye(3))
            # Cube face axes already passed via the conservative AABB.
            for axis in axes:
                chosen = np.flatnonzero(alive)
                if not len(chosen):
                    break
                norm = np.linalg.norm(axis, axis=1)
                unit = axis / np.maximum(norm, 1e-30)[:, None]
                projections = np.einsum("nvi,ni->nv", values, unit)
                source = ids[chosen]
                center = np.einsum("ni,ni->n", index[chosen] + 0.5, unit[source])
                half = 0.5 * np.abs(unit[source]).sum(axis=1)
                separated = (center + half < projections.min(axis=1)[source] - 1e-8) | (
                    center - half > projections.max(axis=1)[source] + 1e-8
                )
                alive[chosen[separated]] = False
            keys.append(mesh._pack_indices(index[alive]))
        return self._indices(keys)

    @staticmethod
    def _indices(keys):
        return (
            mesh._unpack_indices(np.unique(np.concatenate(keys)))
            if keys
            else np.empty((0, 3), dtype=np.int64)
        )


def continuous_collision_candidate(snapshot, result):
    """Keep v1 grasp material; replace its collision sampling in an experiment.

    Full-view accepted triangles get the maximum incident-vertex uncertainty
    band along their face normal. Original normal segments and original source
    occupied cells are included too. Unreferenced/boundary points are retained.
    The 12 mm completion is still an explicitly unobserved collision prior.
    """
    points = np.asarray(result.observed_points_world)
    if not np.array_equal(points, result.surface_mesh.vertices):
        raise ValueError(
            "requires unpartitioned whole-view result, not a clicked/robot-filtered subset"
        )
    depth = np.asarray(snapshot.depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth >= 0.03) & (depth <= 8.0)
    world = mesh._camera_points(
        np.where(valid, depth, 0.0),
        snapshot.intrinsic_matrix,
        snapshot.extrinsic_matrix,
    )[0]
    selected = np.zeros(depth.shape, bool)
    pixels = result.observed_pixel_uv
    selected[pixels[:, 1], pixels[:, 0]] = True
    camera = np.asarray(snapshot.extrinsic_matrix)[:3, 3]
    normal_grid, fallback, normal_audit = mesh._surface_normal_evidence(
        world, valid, selected, camera
    )
    normals = normal_grid[selected]
    ray = points - camera
    ray /= np.maximum(np.linalg.norm(ray, axis=1), 1e-12)[:, None]
    incidence = np.abs(np.einsum("ni,ni->n", normals, ray))
    band = (
        np.clip(
            0.8
            + 220.0 * mesh._valid_depth_gradient(depth, valid)[selected]
            + 2.5 * (1.0 - incidence),
            0.8,
            6.0,
        )
        / 1000.0
    )
    band[fallback[selected]] = 0.006
    separation = result.metadata["support_separation"]
    material = np.ones(len(points), bool)
    if separation.get("applied"):
        a, b, c = separation["plane_coefficients_z_from_xy"]
        material &= (
            np.abs(points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c))
            > separation["source_inlier_tolerance_m"]
        )
    collision = result.collision_occupancy
    cover = CellCover(collision.origin_world, collision.voxel_size_m)
    # Preserve all r=0 observations/old completion cells, regardless of SAT.
    index_parts = [collision.occupied_indices]
    index_parts.append(
        cover.segments(
            points[material] - normals[material] * band[material, None],
            points[material] + normals[material] * band[material, None],
        )
    )
    faces = result.surface_mesh.faces
    faces = faces[material[faces].all(axis=1)]
    triangle = points[faces]
    normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    normal_norm = np.linalg.norm(normal, axis=1)
    usable = normal_norm > 1e-12
    triangle, faces, normal = triangle[usable], faces[usable], normal[usable]
    normal /= normal_norm[usable, None]
    face_band = band[faces].max(axis=1)
    offset = normal[:, None, :] * face_band[:, None, None]
    index_parts.append(cover.prisms(triangle - offset, triangle + offset))
    completion = result.metadata["completion_candidates"][0]
    completion_audit = {"applied": False, "reason": "baseline_completion_not_applied"}
    if completion.get("applied") and completion.get("thickness_mm", 0.0) > 0.0:
        extrusion = float(completion["thickness_mm"]) / 1000.0
        back = triangle + ray[faces] * extrusion
        completion_indices = cover.prisms(triangle, back)
        point_ray_indices = cover.segments(
            points[material], points[material] + ray[material] * extrusion
        )
        completion_indices = np.concatenate((completion_indices, point_ray_indices))
        centers = (
            collision.origin_world + (completion_indices + 0.5) * collision.voxel_size_m
        )
        guard = mesh.visible_free_space_guard(
            centers, depth, snapshot.intrinsic_matrix, snapshot.extrinsic_matrix
        )
        completion_audit = {
            "applied": bool(guard["ok"]),
            "visibility_guard": guard,
            "guard_scope": "all_candidate_voxel_centers_not_entire_cell_free_space_proof",
            "thickness_mm": completion["thickness_mm"],
        }
        if not guard["ok"]:
            raise RuntimeError(
                "continuous_completion_rejected_by_visible_free_space_guard"
            )
        index_parts.append(completion_indices)
    candidate = mesh.VoxelOccupancy(
        collision.voxel_size_m,
        collision.origin_world,
        np.concatenate(index_parts),
        "continuous_observed_prisms_plus_explicit_collision_prior_experiment",
        0,
    )
    audit = {
        "build": BUILD,
        "enabled_by_default": False,
        "activation_decided_by": "explicit_caller_not_geometry_builder",
        "forbidden_inputs_used": [],
        "inputs": ["frozen_public_rgb", "frozen_public_depth", "camera_calibration"],
        "used_click": False,
        "whole_view": True,
        "normal_estimation": normal_audit,
        "accepted_material_triangles": len(triangle),
        "degenerate_triangles_without_prism": int((~usable).sum()),
        "original_source_cells_removed": 0,
        "source_cells_before": len(collision.occupied_indices),
        "cells_after": len(candidate.occupied_indices),
        "candidate_cells_tested": cover.candidate_count,
        "primitive_count": cover.primitive_count,
        "query_radius_cells": 0,
        "surface_material_score_changed": False,
        "uncertainty_max_face_band_mm": float(face_band.max() * 1000.0)
        if len(face_band)
        else None,
        "completion": completion_audit,
        "true_object_thickness": "unknown_single_view",
        "unknown_boundary_space": "not_completed_beyond_explicit_prior",
        "physical_clearance_proven": False,
    }
    metadata = {
        **result.metadata,
        "continuous_collision_experiment": audit,
        "collision_occupancy": candidate.metadata(),
    }
    return replace(result, collision_occupancy=candidate, metadata=metadata), audit
