"""Remove only neighborhood padding proved free by the frozen public depth.

This is not a thinner scene shell and does not remove a single source voxel.
An r=1 query adds cells outside the explicit observed/completed occupancy.
Such an added cell can contradict visible free space. We veto it only when
its entire projected bounding rectangle (plus a pixel guard) has valid depth
beyond the furthest of its eight corners plus a metric clearance. Unknown,
occluded, boundary-crossing and original occupied cells remain occupied.

All queries use one immutable depth/calibration pair. No simulator data,
object identity, click-conditioned reconstruction, contact or live capture.
This is currently an offline experiment, not wired into the production planner.
Visibility is relative to the finite public raster, not a guarantee about
unobserved subpixel geometry or a physical-contact oracle.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from tools.util.rgbd_scene_mesh_robodojo_v1 import (
    VoxelOccupancy,
    _pack_indices,
    _project_world,
    _unpack_indices,
)


BUILD = "whole_voxel_visible_free_padding_guard_v1"
CLEARANCE_M = 0.008
PIXEL_MARGIN = 3
MAX_RECTANGLE_PIXELS = 64
_CORNERS = np.asarray(list(itertools.product((0.0, 1.0), repeat=3)))


class WholeVoxelFreeSpace:
    """Conservative, finite-resolution visibility proof for a whole grid cell."""

    def __init__(self, depth, intrinsic, extrinsic):
        self.depth = np.array(depth, dtype=np.float64, copy=True)
        self.intrinsic = np.array(intrinsic, dtype=np.float64, copy=True).reshape(3, 3)
        self.extrinsic = np.array(extrinsic, dtype=np.float64, copy=True).reshape(4, 4)
        if self.depth.ndim != 2 or not self.depth.size:
            raise ValueError("padding guard requires a nonempty HxW depth")
        if (
            not np.isfinite(self.intrinsic).all()
            or not np.isfinite(self.extrinsic).all()
            or min(self.intrinsic[0, 0], self.intrinsic[1, 1]) <= 0
        ):
            raise ValueError("padding guard requires finite calibrated cameras")
        if not np.allclose(self.intrinsic[2], [0.0, 0.0, 1.0], atol=1e-10):
            raise ValueError("padding guard requires optical pinhole intrinsics")
        valid = np.isfinite(self.depth) & (self.depth >= 0.03) & (self.depth <= 8.0)
        # Unknown depths are zero, so even one unknown ray defeats the proof.
        self._horizontal_min = [np.where(valid, self.depth, 0.0)]
        for level in range(1, 7):
            step = 1 << (level - 1)
            previous = self._horizontal_min[-1]
            shifted = np.zeros_like(previous)
            if step < previous.shape[1]:
                shifted[:, :-step] = previous[:, step:]
            self._horizontal_min.append(np.minimum(previous, shifted))

    def _rectangle_minimum(self, left, top, right, bottom):
        """Exact minimum over inclusive rectangles, not just their corners."""
        width = right - left + 1
        level = np.floor(np.log2(width)).astype(int)
        output = np.full(len(left), np.inf)
        for power in np.unique(level):
            ids = np.flatnonzero(level == power)
            window = self._horizontal_min[int(power)]
            far = right[ids] - (1 << int(power)) + 1
            for dy in range(int(np.max(bottom[ids] - top[ids])) + 1):
                active = bottom[ids] >= top[ids] + dy
                subset = ids[active]
                row = top[subset] + dy
                value = np.minimum(window[row, left[subset]], window[row, far[active]])
                output[subset] = np.minimum(output[subset], value)
        return output

    def known_free(self, indices, origin, voxel_size_m):
        cells = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        free = np.zeros(len(cells), dtype=bool)
        voxel = float(voxel_size_m)
        if not np.isfinite(voxel) or voxel <= 0:
            raise ValueError("padding guard voxel must be finite and positive")
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        if not np.isfinite(origin).all():
            raise ValueError("padding guard grid origin must be finite")
        if not len(cells):
            return free
        # Positive-depth perspective extrema of a convex cell occur at its
        # vertices. A rectangle containing all eight projected vertices also
        # contains the projection of every point inside the cell.
        corners = origin + (cells[:, None, :] + _CORNERS) * voxel
        uv, depth = _project_world(
            corners.reshape(-1, 3), self.intrinsic, self.extrinsic
        )
        uv = uv.reshape(-1, 8, 2)
        depth = depth.reshape(-1, 8)
        h, w = self.depth.shape
        valid = (
            np.isfinite(uv).all(axis=(1, 2))
            & np.isfinite(depth).all(axis=1)
            & (depth > 0.0).all(axis=1)
        )
        ids = np.flatnonzero(valid)
        if not len(ids):
            return free
        low = np.floor(uv[ids].min(axis=1)) - PIXEL_MARGIN
        high = np.ceil(uv[ids].max(axis=1)) + PIXEL_MARGIN
        valid_rectangle = (
            (low >= 0.0).all(axis=1)
            & (high[:, 0] < w)
            & (high[:, 1] < h)
            & ((high - low + 1.0) <= MAX_RECTANGLE_PIXELS).all(axis=1)
        )
        ids = ids[valid_rectangle]
        low = low[valid_rectangle].astype(int)
        high = high[valid_rectangle].astype(int)
        if len(ids):
            minimum = self._rectangle_minimum(
                low[:, 0], low[:, 1], high[:, 0], high[:, 1]
            )
            free[ids] = (minimum > 0.0) & (
                np.max(depth[ids], axis=1) + CLEARANCE_M < minimum
            )
        return free


class VisibilityGuardedCollisionOccupancy:
    """Preserve the voxel set and veto only provably-free added query cells."""

    def __init__(self, base: VoxelOccupancy, depth, intrinsic, extrinsic):
        self.base = base
        self.guard = WholeVoxelFreeSpace(depth, intrinsic, extrinsic)
        self._free_by_key: dict[int, bool] = {}
        self.audit: dict[str, Any] = {
            "build": BUILD,
            "effective_policy": "whole_voxel_visible_free_padding_guard",
            "fallback_reason": None,
            "clearance_m": CLEARANCE_M,
            "clearance_coordinate": "optical_camera_z",
            "pixel_margin": PIXEL_MARGIN,
            "proof": "all_projected_cell_rays_clear_beyond_furthest_corner",
            "proof_scope": "frozen_finite_resolution_depth_not_continuous_physical_truth",
            "unknown_policy": "retain_padding",
            "source_occupied_voxels_removed": 0,
            "original_query_radius_cells": base.query_radius_cells,
            "unique_padding_cells_checked": 0,
            "unique_padding_cells_proven_free": 0,
            "allowed_inputs": [
                "frozen_public_depth",
                "camera_intrinsics",
                "camera_extrinsics",
            ],
            "forbidden_inputs_used": [],
            "used_click": False,
            "physical_contact_truth_used": False,
        }

    def __getattr__(self, name):
        return getattr(self.base, name)

    def metadata(self):
        return {**self.base.metadata(), "padding_visibility_guard": dict(self.audit)}

    def contains_world(self, points_world, *, query_radius_cells=None):
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        occupied = self.base.contains_world(
            points, query_radius_cells=query_radius_cells
        )
        radius = (
            self.base.query_radius_cells
            if query_radius_cells is None
            else max(0, int(query_radius_cells))
        )
        if not radius or not occupied.any():
            return occupied
        # Source occupancy is an explicit prior and is never vetoed, even if
        # its uncertainty band itself reaches visible free space.
        exact = self.base.contains_world(points, query_radius_cells=0)
        query_ids = np.flatnonzero(occupied & ~exact)
        if not len(query_ids):
            return occupied
        cells = np.floor(
            (points[query_ids] - self.base.origin_world) / self.base.voxel_size_m
        ).astype(np.int64)
        keys, inverse = np.unique(_pack_indices(cells), return_inverse=True)
        missing = np.asarray(
            [key for key in keys if int(key) not in self._free_by_key], dtype=np.uint64
        )
        if len(missing):
            free = self.guard.known_free(
                _unpack_indices(missing), self.base.origin_world, self.base.voxel_size_m
            )
            self._free_by_key.update(
                (int(key), bool(value)) for key, value in zip(missing, free)
            )
            self.audit["unique_padding_cells_checked"] += len(missing)
            self.audit["unique_padding_cells_proven_free"] += int(free.sum())
        veto = np.asarray([self._free_by_key[int(key)] for key in keys], dtype=bool)
        occupied[query_ids[veto[inverse]]] = False
        return occupied
