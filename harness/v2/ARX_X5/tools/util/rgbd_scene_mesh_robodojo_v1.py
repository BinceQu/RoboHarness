"""Auditable whole-view RGB-D geometry for the RoboDojo official adapter.

This module is deliberately independent of the BEHAVIOR checkout and of the
RoboDojo evaluator internals.  It ports the useful geometry ideas from the
BEHAVIOR v12/v52/v53 implementations while keeping two different geometric
objects:

* ``surface_mesh``/``surface_occupancy`` describe only RGB-D-supported visible
  surface evidence plus a measured uncertainty band;
* ``collision_occupancy`` is a conservative, camera-ray completion shell and
  is never reported as the object's true material volume.

The organized grid is retained for efficient construction, but a hard
all-edges triangle gate is not allowed to silently erase a valid boundary.
Rejected edges are recorded and a local, image-wide planar repair is applied
when the four-pixel cell provides enough geometric evidence.  Open boundaries
are valid output and are reported in the audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


MESH_VERSION = "robodojo_v1"
RGBD_SCENE_MESH_BUILD = "robodojo_rgbd_whole_view_surface_repair_finite_neighbors_v2"
VOXEL_M = 0.003
OPTICAL_TO_USD_AXES = np.array([1.0, -1.0, -1.0], dtype=np.float64)

DEFAULT_PIXEL_STRIDE = 1
DEFAULT_MIN_DEPTH_M = 0.03
DEFAULT_MAX_DEPTH_M = 8.0
DEFAULT_SURFACE_EDGE_SCALE = 3.5
DEFAULT_SURFACE_EDGE_SLACK_M = 0.002
DEFAULT_MAX_DEPTH_JUMP_M = 0.035
DEFAULT_REPAIR_MAX_DEPTH_JUMP_M = 0.120
DEFAULT_REPAIR_PLANE_RESIDUAL_M = 0.012
DEFAULT_REPAIR_MAX_EDGE_M = 0.090
DEFAULT_REPAIR_RGB_BOUNDARY = 42.0
DEFAULT_SURFACE_BAND_MIN_MM = 0.8
DEFAULT_SURFACE_BAND_MAX_MM = 6.0
DEFAULT_COLLISION_EXTRUSION_MM = 12.0
DEFAULT_SUPPORT_BIN_M = 0.001


@dataclass
class TriangleMesh:
    """Small trimesh-compatible subset used by the adapter and renderers."""

    vertices: np.ndarray
    faces: np.ndarray

    @property
    def is_watertight(self) -> bool:
        if self.faces.size == 0:
            return False
        edges = np.concatenate(
            (
                self.faces[:, [0, 1]],
                self.faces[:, [1, 2]],
                self.faces[:, [2, 0]],
            ),
            axis=0,
        )
        edges.sort(axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return bool(np.all(counts == 2))


@dataclass
class VoxelOccupancy:
    """Integer voxel set with a bounded neighborhood query."""

    voxel_size_m: float
    origin_world: np.ndarray
    occupied_indices: np.ndarray
    source: str
    query_radius_cells: int = 1

    def __post_init__(self) -> None:
        self.voxel_size_m = float(self.voxel_size_m)
        self.origin_world = np.asarray(self.origin_world, dtype=np.float64).reshape(3)
        indices = np.asarray(self.occupied_indices, dtype=np.int64)
        if indices.size == 0:
            indices = np.zeros((0, 3), dtype=np.int64)
        # Deduplicate through the packed scalar key.  Sorting one uint64 array
        # is materially cheaper than sorting a three-column structured array
        # for the hundreds of thousands of cells in a camera frame.
        self._keys = np.unique(_pack_indices(indices.reshape(-1, 3)))
        self.occupied_indices = _unpack_indices(self._keys)
        self.query_radius_cells = max(0, int(self.query_radius_cells))
        # Keep the index compact.  Earlier versions materialized every
        # neighborhood cell here; a 700k-cell real frame then expanded to
        # roughly 19 million uint64 keys before the first planner query.  The
        # sorted packed keys support exact vectorized lower-bound lookups, so
        # only the small query batch is expanded at call time.
        self._query_keys = self._keys
        self._query_offsets = _voxel_offsets(self.query_radius_cells)
        self._hash_table, self._hash_mask = _build_hash_table(self._keys)
        self._local_query_table: np.ndarray | None = None
        self._local_query_mask: int | None = None
        self._local_query_bounds: tuple[np.ndarray, np.ndarray] | None = None

    def prepare_query_region(
        self,
        center_world: Sequence[float],
        radius_m: float,
        *,
        query_radius_cells: int | None = None,
    ) -> dict[str, Any]:
        """Build a compact exact neighborhood index for a bounded query region.

        X5 evaluates many poses around one clicked anchor.  Expanding the
        entire scene by a 27-cell neighborhood is needlessly large, while
        querying the compact scene index 27 times per pose is CPU-heavy.  This
        method expands only occupied cells that can intersect the declared
        region and stores them in a compact hash table.  Queries outside the
        region continue to use the global exact index.
        """
        center = np.asarray(center_world, dtype=np.float64).reshape(3)
        radius = float(radius_m)
        if not np.all(np.isfinite(center)) or not math.isfinite(radius) or radius < 0.0:
            raise ValueError("query region center/radius must be finite")
        cells = (
            self.query_radius_cells
            if query_radius_cells is None
            else max(0, int(query_radius_cells))
        )
        # Include one voxel diagonal so floor quantization at the spherical
        # boundary cannot select an occupied neighbor just outside the box.
        half_cells = int(math.ceil(radius / self.voxel_size_m)) + cells + 1
        center_index = np.floor(
            (center - self.origin_world) / self.voxel_size_m
        ).astype(np.int64)
        lower = center_index - half_cells
        upper = center_index + half_cells
        if len(self.occupied_indices):
            occupied = self.occupied_indices
            keep = np.all(
                (occupied >= lower.reshape(1, 3)) & (occupied <= upper.reshape(1, 3)),
                axis=1,
            )
            local_indices = occupied[keep]
        else:
            local_indices = np.zeros((0, 3), dtype=np.int64)
        if cells and len(local_indices):
            offsets = _voxel_offsets(cells)
            expanded = np.concatenate(
                [
                    _pack_indices(local_indices + offset.reshape(1, 3))
                    for offset in offsets
                ]
            )
            local_keys = np.unique(expanded)
        else:
            local_keys = _pack_indices(local_indices)
        table, mask = _build_hash_table(local_keys)
        self._local_query_table = table
        self._local_query_mask = mask
        self._local_query_bounds = (lower, upper)
        return {
            "center_world": center.tolist(),
            "radius_m": radius,
            "radius_cells": half_cells,
            "occupied_source_count": int(len(local_indices)),
            "expanded_query_key_count": int(len(local_keys)),
            "table_size": int(len(table)),
            "query_radius_cells": int(cells),
        }

    @property
    def volume_m3(self) -> float:
        return float(len(self.occupied_indices)) * self.voxel_size_m**3

    @property
    def volume_cm3(self) -> float:
        return self.volume_m3 * 1.0e6

    def contains_world(
        self,
        points_world: np.ndarray,
        *,
        query_radius_cells: int | None = None,
    ) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        result = np.zeros(len(points), dtype=bool)
        if not len(points) or not len(self.occupied_indices):
            return result
        finite = np.all(np.isfinite(points), axis=1)
        if not np.any(finite):
            return result
        radius = (
            self.query_radius_cells
            if query_radius_cells is None
            else max(0, int(query_radius_cells))
        )
        query = np.floor(
            (points[finite] - self.origin_world.reshape(1, 3)) / self.voxel_size_m
        ).astype(np.int64)
        matched = np.zeros(len(query), dtype=bool)
        handled = np.zeros(len(query), dtype=bool)
        if (
            radius == self.query_radius_cells
            and self._local_query_table is not None
            and self._local_query_mask is not None
            and self._local_query_bounds is not None
        ):
            lower, upper = self._local_query_bounds
            inside = np.all(
                (query >= lower.reshape(1, 3)) & (query <= upper.reshape(1, 3)),
                axis=1,
            )
            if np.any(inside):
                matched[inside] = _hash_contains(
                    _pack_indices(query[inside]),
                    self._local_query_table,
                    self._local_query_mask,
                )
                handled[inside] = True
        # The local index is optional (e.g. for a renderer querying an
        # arbitrary image-space set), so retain an exact global fallback for
        # every query cell outside its declared bounds.
        if np.any(~handled):
            offsets = (
                self._query_offsets
                if radius == self.query_radius_cells
                else _voxel_offsets(radius)
            )
            matched[~handled] = _contains_with_offsets(
                query[~handled],
                self._keys,
                self._hash_table,
                self._hash_mask,
                offsets,
            )
        result[np.flatnonzero(finite)] = matched
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "voxel_size_m": self.voxel_size_m,
            "occupied_voxel_count": int(len(self.occupied_indices)),
            "volume_cm3": self.volume_cm3,
            "origin_world": self.origin_world.tolist(),
            "source": self.source,
            "query_radius_cells": int(self.query_radius_cells),
        }


@dataclass
class RGBDSceneMeshResult:
    """Geometry and audit data returned by the RoboDojo mesh route."""

    mesh: TriangleMesh
    surface_mesh: TriangleMesh
    surface_occupancy: VoxelOccupancy
    collision_occupancy: VoxelOccupancy
    metadata: dict[str, Any]
    observed_points_world: np.ndarray | None = None
    observed_pixel_uv: np.ndarray | None = None

    def anchor_audit(
        self,
        point_world: Sequence[float],
        *,
        max_distance_m: float = 0.018,
    ) -> dict[str, Any]:
        point = np.asarray(point_world, dtype=np.float64).reshape(3)
        points = np.asarray(
            self.observed_points_world
            if self.observed_points_world is not None
            else [],
            dtype=np.float64,
        )
        if points.size == 0:
            return {
                "status": "surface_missing",
                "nearest_surface_distance_m": None,
                "nearest_surface_pixel_uv": None,
                "max_allowed_distance_m": float(max_distance_m),
            }
        points = points.reshape(-1, 3)
        best_distance = float("inf")
        best_index = -1
        # Chunking prevents a large frame from creating a second giant matrix.
        for start in range(0, len(points), 65536):
            block = points[start : start + 65536]
            distances = np.linalg.norm(block - point.reshape(1, 3), axis=1)
            local = int(np.argmin(distances))
            value = float(distances[local])
            if value < best_distance:
                best_distance = value
                best_index = start + local
        pixels = np.asarray(
            self.observed_pixel_uv if self.observed_pixel_uv is not None else [],
            dtype=np.int64,
        )
        pixel = None
        if best_index >= 0 and pixels.size:
            pixel = pixels.reshape(-1, 2)[best_index].astype(int).tolist()
        return {
            "status": "ok"
            if best_distance <= float(max_distance_m)
            else "surface_missing",
            "nearest_surface_distance_m": best_distance,
            "nearest_surface_distance_mm": best_distance * 1000.0,
            "nearest_surface_pixel_uv": pixel,
            "max_allowed_distance_m": float(max_distance_m),
        }


class MeshSceneOccupancy:
    """DepthRayOccupancy-compatible query wrapper for a mesh result."""

    def __init__(
        self,
        result: RGBDSceneMeshResult,
        *,
        support_z: float | None = None,
        target_anchor: np.ndarray | None = None,
        target_radius_m: float = 0.110,
        support_margin_m: float = 0.003,
        surface_query_radius_cells: int | None = None,
    ) -> None:
        self.result = result
        self.surface = result.surface_occupancy
        self.collision = result.collision_occupancy
        self.support_z = (
            None
            if support_z is None or not math.isfinite(float(support_z))
            else float(support_z)
        )
        self.target_anchor = (
            None
            if target_anchor is None
            else np.asarray(target_anchor, dtype=np.float64).reshape(3)
        )
        self.target_radius_m = float(target_radius_m)
        self.support_margin_m = float(support_margin_m)
        self.surface_query_radius_cells = surface_query_radius_cells

    def prepare_query_region(
        self,
        center_world: Sequence[float],
        radius_m: float,
    ) -> dict[str, Any]:
        """Prepare the same bounded lookup region for both occupancy layers."""
        collision_radius = getattr(self, "collision_query_radius_cells", None)
        return {
            "surface": self.surface.prepare_query_region(center_world, radius_m),
            "collision": self.collision.prepare_query_region(
                center_world,
                radius_m,
                **(
                    {}
                    if collision_radius is None
                    else {"query_radius_cells": int(collision_radius)}
                ),
            ),
        }

    def occupied_mask(
        self,
        points_eef: np.ndarray,
        rotation: np.ndarray,
        position: np.ndarray,
        *,
        surface_only: bool = False,
        target_anchor: np.ndarray | None = None,
        target_radius_m: float | None = None,
    ) -> np.ndarray:
        points = np.asarray(points_eef, dtype=np.float64).reshape(-1, 3)
        rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        points_world = points @ rot.T + pos.reshape(1, 3)
        occupancy = self.surface if surface_only else self.collision
        if surface_only:
            query_radius = self.surface_query_radius_cells
        else:
            # None keeps the occupancy's own radius. The grasp planner sets 0
            # because the 15 mm inflated finger shell is already the clearance.
            query_radius = getattr(self, "collision_query_radius_cells", None)
        occupied = occupancy.contains_world(
            points_world,
            query_radius_cells=query_radius,
        )
        if not surface_only:
            return occupied
        anchor = (
            self.target_anchor
            if target_anchor is None
            else np.asarray(target_anchor, dtype=np.float64).reshape(3)
        )
        if self.support_z is not None:
            # Infer which side of the support plane contains the clicked
            # surface. Depending on camera/world orientation, a foreground
            # protrusion can have either a greater or a smaller world-z value.
            # A click inside the support uncertainty band has no separable
            # object evidence and must not score nearby robot geometry.
            anchor_delta = (
                None if anchor is None else float(anchor[2]) - float(self.support_z)
            )
            if anchor_delta is None or abs(anchor_delta) < self.support_margin_m:
                return np.zeros(len(points_world), dtype=bool)
            side = 1.0 if anchor_delta > 0.0 else -1.0
            occupied &= (
                side * (points_world[:, 2] - float(self.support_z))
                >= self.support_margin_m
            )
        radius = (
            self.target_radius_m if target_radius_m is None else float(target_radius_m)
        )
        if anchor is not None and math.isfinite(radius) and radius > 0.0:
            occupied &= (
                np.linalg.norm(points_world - anchor.reshape(1, 3), axis=1) <= radius
            )
        return occupied

    def count(self, *args: Any, **kwargs: Any) -> int:
        return int(np.count_nonzero(self.occupied_mask(*args, **kwargs)))


def _pack_indices(indices: np.ndarray) -> np.ndarray:
    """Pack signed voxel coordinates into collision-free uint64 keys."""
    values = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    offset = 1 << 20
    limit = (1 << 21) - 1
    shifted = values + offset
    if np.any(shifted < 0) or np.any(shifted > limit):
        raise ValueError("world voxel index exceeds the supported +/-3 km range")
    return (
        (shifted[:, 0].astype(np.uint64) << np.uint64(42))
        | (shifted[:, 1].astype(np.uint64) << np.uint64(21))
        | shifted[:, 2].astype(np.uint64)
    )


def _unpack_indices(keys: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_pack_indices` for audit/volume metadata."""
    values = np.asarray(keys, dtype=np.uint64).reshape(-1)
    offset = np.int64(1 << 20)
    mask = np.uint64((1 << 21) - 1)
    result = np.empty((len(values), 3), dtype=np.int64)
    result[:, 0] = ((values >> np.uint64(42)) & mask).astype(np.int64) - offset
    result[:, 1] = ((values >> np.uint64(21)) & mask).astype(np.int64) - offset
    result[:, 2] = (values & mask).astype(np.int64) - offset
    return result


_VOXEL_OFFSET_CACHE: dict[int, np.ndarray] = {}
_HASH_EMPTY = np.uint64(0xFFFFFFFFFFFFFFFF)


def _voxel_offsets(radius: int) -> np.ndarray:
    """Return deterministic Chebyshev-neighborhood offsets for a radius."""
    radius = max(0, int(radius))
    cached = _VOXEL_OFFSET_CACHE.get(radius)
    if cached is not None:
        return cached
    offsets = np.asarray(
        [
            (dx, dy, dz)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
        ],
        dtype=np.int64,
    )
    _VOXEL_OFFSET_CACHE[radius] = offsets
    return offsets


def _mix_uint64(values: np.ndarray) -> np.ndarray:
    """Deterministic splitmix-style hash with uint64 wraparound."""
    result = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        result ^= result >> np.uint64(30)
        result *= np.uint64(0xBF58476D1CE4E5B9)
        result ^= result >> np.uint64(27)
        result *= np.uint64(0x94D049BB133111EB)
        result ^= result >> np.uint64(31)
    return result


def _build_hash_table(keys: np.ndarray) -> tuple[np.ndarray, int]:
    """Build a compact exact linear-probing table from unique uint64 keys."""
    values = np.asarray(keys, dtype=np.uint64).reshape(-1)
    table_size = 8
    while table_size < max(8, 2 * len(values)):
        table_size <<= 1
    table = np.full(table_size, _HASH_EMPTY, dtype=np.uint64)
    if not len(values):
        return table, table_size - 1
    mask = table_size - 1
    pending = values.copy()
    slots = (_mix_uint64(pending) & np.uint64(mask)).astype(np.int64)
    rounds = 0
    while len(pending):
        rounds += 1
        if rounds > table_size:
            raise RuntimeError("voxel hash table insertion did not converge")
        order = np.argsort(slots, kind="stable")
        sorted_slots = slots[order]
        first = np.r_[True, sorted_slots[1:] != sorted_slots[:-1]]
        leaders = order[first]
        leader_slots = sorted_slots[first]
        can_place = table[leader_slots] == _HASH_EMPTY
        if np.any(can_place):
            table[leader_slots[can_place]] = pending[leaders[can_place]]
        placed = np.zeros(len(pending), dtype=bool)
        placed[leaders[can_place]] = True
        if np.all(placed):
            break
        pending = pending[~placed]
        slots = (slots[~placed] + 1) & mask
    return table, mask


def _hash_contains(
    query_keys: np.ndarray,
    table: np.ndarray,
    mask: int,
) -> np.ndarray:
    """Exact vectorized membership lookup for packed voxel keys."""
    query = np.asarray(query_keys, dtype=np.uint64).reshape(-1)
    result = np.zeros(len(query), dtype=bool)
    if not len(query):
        return result
    slots = (_mix_uint64(query) & np.uint64(mask)).astype(np.int64)
    active = np.ones(len(query), dtype=bool)
    for _ in range(len(table)):
        if not np.any(active):
            break
        values = table[slots]
        equal = active & (values == query)
        empty = active & (values == _HASH_EMPTY)
        result[equal] = True
        active &= ~(equal | empty)
        if np.any(active):
            slots[active] = (slots[active] + 1) & int(mask)
    if np.any(active):
        raise RuntimeError("voxel hash table lookup did not converge")
    return result


def _contains_with_offsets(
    query: np.ndarray,
    keys: np.ndarray,
    table: np.ndarray,
    mask: int,
    offsets: np.ndarray,
) -> np.ndarray:
    """Check a query-cell neighborhood without materializing scene neighbors."""
    values = np.asarray(query, dtype=np.int64).reshape(-1, 3)
    if not len(values):
        return np.zeros(0, dtype=bool)
    if len(offsets) == 1:
        return _hash_contains(_pack_indices(values + offsets[0]), table, mask)
    # A single vectorized lookup amortizes the hash-probe bookkeeping across
    # the neighborhood.  The temporary contains only the caller's query batch.
    expanded = values.reshape(1, -1, 3) + offsets.reshape(-1, 1, 3)
    matched = _hash_contains(
        _pack_indices(expanded.reshape(-1, 3)), table, mask
    ).reshape(len(offsets), len(values))
    return np.any(matched, axis=0)


def _finite_depth(depth: Any) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"depth must be HxW, got {values.shape}")
    return values


def _camera_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metric = _finite_depth(depth)
    height, width = metric.shape
    rows, cols = np.mgrid[:height, :width]
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if not math.isfinite(fx) or not math.isfinite(fy) or fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    optical = np.stack(
        (
            (cols - cx) * metric / fx,
            (rows - cy) * metric / fy,
            metric,
        ),
        axis=-1,
    )
    usd = optical * OPTICAL_TO_USD_AXES.reshape(1, 1, 3)
    rotation = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)[:3, :3]
    translation = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)[:3, 3]
    world = usd @ rotation.T + translation.reshape(1, 1, 3)
    return world, optical, rows, cols


def _edge_ok(
    first: np.ndarray,
    second: np.ndarray,
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    *,
    pixel_span: float,
    focal_px: float,
    scale: float,
    slack_m: float,
    max_depth_jump_m: float,
) -> np.ndarray:
    distance = np.linalg.norm(first - second, axis=-1)
    expected = pixel_span * np.maximum(first_depth, second_depth) / max(focal_px, 1e-9)
    return (distance <= scale * expected + slack_m) & (
        np.abs(first_depth - second_depth) <= max_depth_jump_m
    )


def _rgb_delta(
    rgb: np.ndarray, first_uv: np.ndarray, second_uv: np.ndarray
) -> np.ndarray:
    a = rgb[first_uv[..., 1], first_uv[..., 0]].astype(np.float64)
    b = rgb[second_uv[..., 1], second_uv[..., 0]].astype(np.float64)
    return np.linalg.norm(a - b, axis=-1)


def _surface_normals(
    world: np.ndarray,
    valid: np.ndarray,
    selected: np.ndarray,
    camera_position: np.ndarray,
) -> np.ndarray:
    return _surface_normal_evidence(world, valid, selected, camera_position)[0]


def _surface_normal_evidence(
    world: np.ndarray,
    valid: np.ndarray,
    selected: np.ndarray,
    camera_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Finite local differences, never image-wrap or a derivative across a hole.

    One-sided derivatives are supported at a frame/depth boundary. If one
    tangent cannot be measured, a camera-facing ray is only an orientation
    fallback, not an observed surface normal; the caller widens its band.
    """
    height, width = valid.shape
    normals = np.zeros((height, width, 3), dtype=np.float64)
    tangent_u = np.zeros_like(normals)
    tangent_v = np.zeros_like(normals)
    count_u = np.zeros((height, width), dtype=np.uint8)
    count_v = np.zeros((height, width), dtype=np.uint8)
    for axis, tangent, count in ((1, tangent_u, count_u), (0, tangent_v, count_v)):
        lower = [slice(None), slice(None)]
        upper = [slice(None), slice(None)]
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        lower, upper = tuple(lower), tuple(upper)
        pair_valid = valid[lower] & valid[upper]
        delta = np.zeros_like(world[lower])
        np.subtract(world[upper], world[lower], out=delta, where=pair_valid[..., None])
        tangent[lower] += delta
        tangent[upper] += delta
        count[lower] += pair_valid
        count[upper] += pair_valid
    cross = np.cross(tangent_u, tangent_v)
    norm = np.linalg.norm(cross, axis=-1)
    good = selected & (count_u > 0) & (count_v > 0) & np.isfinite(norm) & (norm > 1e-9)
    normals[good] = cross[good] / norm[good, None]
    fallback = selected & ~good
    if np.any(fallback):
        view = camera_position.reshape(1, 3) - world[fallback]
        view_norm = np.linalg.norm(view, axis=1)
        valid_view = view_norm > 1e-9
        fallback_indices = np.flatnonzero(fallback)
        values = np.zeros((len(fallback_indices), 3), dtype=np.float64)
        values[valid_view] = view[valid_view] / view_norm[valid_view, None]
        normals.reshape(-1, 3)[fallback_indices] = values
    # Orient every normal toward the camera, matching the public anchor normal
    # convention and keeping the band symmetric when the cross product flips.
    selected_indices = np.flatnonzero(selected)
    if len(selected_indices):
        vectors = camera_position.reshape(1, 3) - world.reshape(-1, 3)[selected_indices]
        signs = np.einsum("ij,ij->i", normals.reshape(-1, 3)[selected_indices], vectors)
        flip = signs < 0.0
        flat = normals.reshape(-1, 3)
        flat[selected_indices[flip]] *= -1.0
    return (
        normals,
        fallback,
        {
            "method": "finite_immediate_neighbor_differences",
            "central_difference_points": int(
                np.count_nonzero(good & (count_u == 2) & (count_v == 2))
            ),
            "one_sided_difference_points": int(
                np.count_nonzero(good & ((count_u < 2) | (count_v < 2)))
            ),
            "unobserved_normal_camera_ray_fallback_points": int(
                np.count_nonzero(fallback)
            ),
            "uses_wrapped_image_neighbors": False,
            "derivatives_cross_invalid_depth": False,
            "fallback_uncertainty_policy": "maximum_configured_observed_band_not_known_thickness",
        },
    )


def _valid_depth_gradient(metric: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Symmetric gradient evidence from valid adjacent pairs only."""
    gradient = np.zeros_like(metric, dtype=np.float64)
    for axis in (0, 1):
        lower = [slice(None), slice(None)]
        upper = [slice(None), slice(None)]
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        lower, upper = tuple(lower), tuple(upper)
        delta = np.zeros_like(metric[lower], dtype=np.float64)
        np.subtract(
            metric[upper], metric[lower], out=delta, where=valid[upper] & valid[lower]
        )
        np.abs(delta, out=delta)
        np.maximum(gradient[lower], delta, out=gradient[lower])
        np.maximum(gradient[upper], delta, out=gradient[upper])
    return gradient


def _mesh_boundary_stats(faces: np.ndarray) -> dict[str, Any]:
    if len(faces) == 0:
        return {
            "boundary_edge_count": 0,
            "nonmanifold_edge_count": 0,
            "face_count": 0,
        }
    edges = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "boundary_edge_count": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edge_count": int(np.count_nonzero(counts > 2)),
        "face_count": int(len(faces)),
    }


def _voxel_indices_from_points(
    points: np.ndarray,
    *,
    origin: np.ndarray,
    voxel_size_m: float,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(values), axis=1)
    if not np.any(finite):
        return np.zeros((0, 3), dtype=np.int64)
    indices = np.floor(
        (values[finite] - origin.reshape(1, 3)) / float(voxel_size_m)
    ).astype(np.int64)
    # Scalar-key deduplication avoids the relatively expensive row-wise sort
    # used by ``np.unique(..., axis=0)`` while preserving exact integer cells.
    return _unpack_indices(np.unique(_pack_indices(indices)))


def _project_world(
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    usd = (points - transform[:3, 3]) @ transform[:3, :3]
    depth = -usd[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 1e-6)
    uv[valid, 0] = (
        float(intrinsic[0, 2]) + float(intrinsic[0, 0]) * usd[valid, 0] / depth[valid]
    )
    uv[valid, 1] = (
        float(intrinsic[1, 2]) - float(intrinsic[1, 1]) * usd[valid, 1] / depth[valid]
    )
    return uv, depth


def visible_free_space_guard(
    candidate_points_world: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    *,
    clearance_m: float = 0.008,
    clearance_pixels: float = 3.0,
    maximum_violation_fraction: float = 0.05,
    minimum_violation_pixels: int = 64,
) -> dict[str, Any]:
    """Audit whether a completion occupies RGB-D-observed free space."""
    metric = _finite_depth(depth)
    height, width = metric.shape
    uv, candidate_depth = _project_world(candidate_points_world, intrinsic, extrinsic)
    valid = (
        np.isfinite(candidate_depth)
        & (candidate_depth > 0.0)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= -float(clearance_pixels))
        & (uv[:, 0] <= width - 1 + float(clearance_pixels))
        & (uv[:, 1] >= -float(clearance_pixels))
        & (uv[:, 1] <= height - 1 + float(clearance_pixels))
    )
    if not np.any(valid):
        return {
            "ok": True,
            "violation_fraction": 0.0,
            "violation_pixels": 0,
            "projected_candidate_count": 0,
            "reason": "no_projected_candidate_points",
        }
    rows = np.rint(uv[valid, 1]).astype(np.int64)
    cols = np.rint(uv[valid, 0]).astype(np.int64)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    valid_indices = np.flatnonzero(valid)[inside]
    if not len(valid_indices):
        return {
            "ok": True,
            "violation_fraction": 0.0,
            "violation_pixels": 0,
            "projected_candidate_count": 0,
            "reason": "candidate_outside_observed_image",
        }
    rows = rows[inside]
    cols = cols[inside]
    observed = metric[rows, cols]
    candidate = candidate_depth[valid_indices]
    observed_valid = np.isfinite(observed) & (observed > 0.0)
    violations = observed_valid & (candidate < observed - float(clearance_m))
    # Count image pixels rather than duplicate 3-D samples.  A dense shell can
    # project many points to one pixel; treating those samples as independent
    # would make the guard depend on sampling density instead of visibility.
    projected_keys = rows.astype(np.int64) * int(width) + cols.astype(np.int64)
    unique_projected = np.unique(projected_keys)
    violation_keys = np.unique(projected_keys[violations])
    fraction = float(len(violation_keys) / max(1, len(unique_projected)))
    count = int(len(violation_keys))
    return {
        "ok": bool(
            fraction <= float(maximum_violation_fraction)
            or count < int(minimum_violation_pixels)
        ),
        "violation_fraction": fraction,
        "violation_pixels": count,
        "projected_candidate_count": int(len(candidate)),
        "projected_unique_pixel_count": int(len(unique_projected)),
        "observed_valid_projection_count": int(np.count_nonzero(observed_valid)),
        "clearance_m": float(clearance_m),
        "clearance_pixels": float(clearance_pixels),
        "maximum_violation_fraction": float(maximum_violation_fraction),
        "minimum_violation_pixels": int(minimum_violation_pixels),
        "reason": "within_limit"
        if (
            fraction <= float(maximum_violation_fraction)
            or count < int(minimum_violation_pixels)
        )
        else "observed_free_space_occlusion",
    }


def _support_plane_audit(
    world: np.ndarray,
    valid: np.ndarray,
    *,
    bin_m: float = DEFAULT_SUPPORT_BIN_M,
) -> tuple[float | None, dict[str, Any]]:
    points_all = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    valid_flat = np.asarray(valid, dtype=bool).reshape(-1)
    valid_flat &= np.isfinite(points_all).all(axis=1)
    points = points_all[valid_flat]
    values = points[:, 2]
    audit: dict[str, Any] = {
        "source": "world_z_robust_mode_from_frozen_rgbd",
        "bin_m": float(bin_m),
        "uses_segmentation": False,
        "uses_simulator_geometry": False,
    }
    if len(values) < 32:
        audit["reason"] = "too_few_points"
        return None, audit
    low, high = np.percentile(values, [1.0, 99.0])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        audit["reason"] = "nonfinite_height_range"
        return None, audit
    # A degenerate range is a valid observation of a flat support plane.  Use
    # its median directly instead of dropping the entire frame.
    if high - low <= 1e-9:
        center = float(np.median(values))
    else:
        edges = np.arange(
            math.floor(float(low) / bin_m),
            math.ceil(float(high) / bin_m) + 2,
            dtype=np.float64,
        ) * float(bin_m)
        if len(edges) < 2:
            edges = np.asarray([low, low + float(bin_m)], dtype=np.float64)
        histogram, _ = np.histogram(values, bins=edges)
        if not np.any(histogram):
            audit["reason"] = "empty_height_histogram"
            return None, audit
        index = int(np.argmax(histogram))
        center = float((edges[index] + edges[index + 1]) * 0.5)
    inliers = np.abs(values - center) <= max(0.003, 2.5 * float(bin_m))
    ratio = float(np.count_nonzero(inliers) / max(1, len(values)))
    # Fit a locally planar support surface from the robust height inliers.  The
    # scalar z intercept remains the planner-compatible value, while the
    # normal and residuals make the assumption auditable to downstream users.
    support_points = points[inliers]
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    coefficients = np.array([0.0, 0.0, center], dtype=np.float64)
    residuals = np.abs(values - center)
    if len(support_points) >= 3:
        design = np.column_stack(
            (support_points[:, 0], support_points[:, 1], np.ones(len(support_points)))
        )
        try:
            fitted, *_ = np.linalg.lstsq(design, support_points[:, 2], rcond=None)
            if np.all(np.isfinite(fitted)):
                coefficients = np.asarray(fitted, dtype=np.float64)
                normal = np.asarray([-fitted[0], -fitted[1], 1.0], dtype=np.float64)
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                residuals = np.abs(
                    support_points[:, 2]
                    - (
                        fitted[0] * support_points[:, 0]
                        + fitted[1] * support_points[:, 1]
                        + fitted[2]
                    )
                )
        except (np.linalg.LinAlgError, ValueError):
            pass
    audit.update(
        {
            "reason": "dominant_world_z_mode_planar_refinement",
            "z_m": float(coefficients[2]),
            "plane_coefficients_z_from_xy": coefficients.tolist(),
            "normal_world": normal.tolist(),
            "residual_median_mm": float(np.median(residuals) * 1000.0),
            "residual_p95_mm": float(np.percentile(residuals, 95.0) * 1000.0),
            "fit_method": "robust_height_mode_then_least_squares_plane",
            "inlier_count": int(np.count_nonzero(inliers)),
            "point_count": int(len(values)),
            "inlier_fraction": ratio,
            "confidence": "strong" if ratio >= 0.20 else "weak",
        }
    )
    return float(coefficients[2]), audit


def _make_surface_mesh(
    rgb: np.ndarray,
    metric_depth: np.ndarray,
    world: np.ndarray,
    valid: np.ndarray,
    *,
    pixel_stride: int,
    intrinsic: np.ndarray,
    camera_position: np.ndarray,
    surface_edge_scale: float,
    surface_edge_slack_m: float,
    max_depth_jump_m: float,
    repair_max_depth_jump_m: float,
    repair_plane_residual_m: float,
    repair_max_edge_m: float,
    repair_rgb_boundary: float,
) -> tuple[TriangleMesh, np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = metric_depth.shape
    focal_px = float((float(intrinsic[0, 0]) + float(intrinsic[1, 1])) * 0.5)
    selected = np.zeros((height, width), dtype=bool)
    selected[::pixel_stride, ::pixel_stride] = True
    selected &= valid
    # Adaptive sampling is global: depth/RGB edges are never selected only
    # because they are close to the user's click.
    if pixel_stride > 1:
        depth_delta = _valid_depth_gradient(metric_depth, valid)
        rgb_float = np.asarray(rgb[:, :, :3], dtype=np.float64)
        rgb_delta = np.zeros_like(metric_depth)
        rgb_delta[:-1] = np.maximum(
            rgb_delta[:-1], np.linalg.norm(rgb_float[1:] - rgb_float[:-1], axis=2)
        )
        rgb_delta[:, :-1] = np.maximum(
            rgb_delta[:, :-1],
            np.linalg.norm(rgb_float[:, 1:] - rgb_float[:, :-1], axis=2),
        )
        selected |= valid & ((depth_delta >= 0.010) | (rgb_delta >= 32.0))
    rows, cols = np.nonzero(selected)
    vertex_index = np.full((height, width), -1, dtype=np.int32)
    vertex_index[rows, cols] = np.arange(len(rows), dtype=np.int32)
    vertices = world[selected].astype(np.float64, copy=False)
    pixel_uv = np.column_stack((cols, rows)).astype(np.int64)
    faces_rows: list[np.ndarray] = []
    edge_counts = {
        "horizontal_total": 0,
        "horizontal_rejected": 0,
        "vertical_total": 0,
        "vertical_rejected": 0,
        "diagonal_total": 0,
        "diagonal_rejected": 0,
        "triangle_standard": 0,
        "triangle_repaired": 0,
        "triangle_rejected": 0,
    }
    rgb_array = np.asarray(rgb[:, :, :3])
    for row in range(height - 1):
        a = vertex_index[row, :-1]
        b = vertex_index[row, 1:]
        c = vertex_index[row + 1, :-1]
        d = vertex_index[row + 1, 1:]
        quad = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        if not np.any(quad):
            continue
        columns = np.flatnonzero(quad)
        pa = world[row, columns]
        pb = world[row, columns + 1]
        pc = world[row + 1, columns]
        pd = world[row + 1, columns + 1]
        da = metric_depth[row, columns]
        db = metric_depth[row, columns + 1]
        dc = metric_depth[row + 1, columns]
        dd = metric_depth[row + 1, columns + 1]
        horizontal_top = _edge_ok(
            pa,
            pb,
            da,
            db,
            pixel_span=float(pixel_stride),
            focal_px=focal_px,
            scale=surface_edge_scale,
            slack_m=surface_edge_slack_m,
            max_depth_jump_m=max_depth_jump_m,
        )
        horizontal_bottom = _edge_ok(
            pc,
            pd,
            dc,
            dd,
            pixel_span=float(pixel_stride),
            focal_px=focal_px,
            scale=surface_edge_scale,
            slack_m=surface_edge_slack_m,
            max_depth_jump_m=max_depth_jump_m,
        )
        vertical_left = _edge_ok(
            pa,
            pc,
            da,
            dc,
            pixel_span=float(pixel_stride),
            focal_px=focal_px,
            scale=surface_edge_scale,
            slack_m=surface_edge_slack_m,
            max_depth_jump_m=max_depth_jump_m,
        )
        vertical_right = _edge_ok(
            pb,
            pd,
            db,
            dd,
            pixel_span=float(pixel_stride),
            focal_px=focal_px,
            scale=surface_edge_scale,
            slack_m=surface_edge_slack_m,
            max_depth_jump_m=max_depth_jump_m,
        )
        diagonal_main = _edge_ok(
            pa,
            pd,
            da,
            dd,
            pixel_span=float(pixel_stride) * math.sqrt(2.0),
            focal_px=focal_px,
            scale=surface_edge_scale,
            slack_m=surface_edge_slack_m,
            max_depth_jump_m=max_depth_jump_m,
        )
        diagonal_other = _edge_ok(
            pb,
            pc,
            db,
            dc,
            pixel_span=float(pixel_stride) * math.sqrt(2.0),
            focal_px=focal_px,
            scale=surface_edge_scale,
            slack_m=surface_edge_slack_m,
            max_depth_jump_m=max_depth_jump_m,
        )
        edge_counts["horizontal_total"] += int(2 * len(columns))
        edge_counts["horizontal_rejected"] += int(
            np.count_nonzero(~horizontal_top) + np.count_nonzero(~horizontal_bottom)
        )
        edge_counts["vertical_total"] += int(2 * len(columns))
        edge_counts["vertical_rejected"] += int(
            np.count_nonzero(~vertical_left) + np.count_nonzero(~vertical_right)
        )
        edge_counts["diagonal_total"] += int(2 * len(columns))
        edge_counts["diagonal_rejected"] += int(
            np.count_nonzero(~diagonal_main) + np.count_nonzero(~diagonal_other)
        )

        # Evaluate each triangle's own edges.  This preserves a valid triangle
        # when the opposite diagonal is the only discontinuous edge.
        tri_one_edges = np.column_stack((horizontal_top, vertical_right, diagonal_main))
        tri_two_edges = np.column_stack(
            (diagonal_main, horizontal_bottom, vertical_left)
        )
        tri_one_ok = np.all(tri_one_edges, axis=1)
        tri_two_ok = np.all(tri_two_edges, axis=1)
        quad_points = np.stack((pa, pb, pc, pd), axis=1)
        plane_normal = np.cross(pb - pa, pc - pa)
        plane_norm = np.linalg.norm(plane_normal, axis=1)
        plane_unit = plane_normal / np.maximum(plane_norm, 1e-12)[:, None]
        residual = np.abs(np.einsum("ij,ij->i", pd - pa, plane_unit))
        max_edge = np.maximum.reduce(
            (
                np.linalg.norm(pb - pa, axis=1),
                np.linalg.norm(pc - pa, axis=1),
                np.linalg.norm(pd - pb, axis=1),
                np.linalg.norm(pd - pc, axis=1),
            )
        )
        depth_span = np.maximum.reduce((da, db, dc, dd)) - np.minimum.reduce(
            (da, db, dc, dd)
        )
        color_boundary = np.maximum.reduce(
            (
                _rgb_delta(
                    rgb_array,
                    np.column_stack((columns, np.full(len(columns), row))),
                    np.column_stack((columns + 1, np.full(len(columns), row))),
                ),
                _rgb_delta(
                    rgb_array,
                    np.column_stack((columns, np.full(len(columns), row))),
                    np.column_stack((columns, np.full(len(columns), row + 1))),
                ),
                _rgb_delta(
                    rgb_array,
                    np.column_stack((columns + 1, np.full(len(columns), row))),
                    np.column_stack((columns + 1, np.full(len(columns), row + 1))),
                ),
                _rgb_delta(
                    rgb_array,
                    np.column_stack((columns, np.full(len(columns), row + 1))),
                    np.column_stack((columns + 1, np.full(len(columns), row + 1))),
                ),
            )
        )
        repairable = (
            (plane_norm > 1e-9)
            & (residual <= float(repair_plane_residual_m))
            & (depth_span <= float(repair_max_depth_jump_m))
            # A grazing wall can legitimately span many pixel-widths in
            # world space.  The planar residual and RGB boundary checks are
            # the discontinuity guards; this cap only prevents arbitrarily
            # large quads from being repaired.
            & (max_edge <= float(repair_max_edge_m))
            & (color_boundary < float(repair_rgb_boundary))
        )
        tri_one_repair = (~tri_one_ok) & repairable
        tri_two_repair = (~tri_two_ok) & repairable
        tri_one_keep = tri_one_ok | tri_one_repair
        tri_two_keep = tri_two_ok | tri_two_repair
        edge_counts["triangle_standard"] += int(
            np.count_nonzero(tri_one_ok) + np.count_nonzero(tri_two_ok)
        )
        edge_counts["triangle_repaired"] += int(
            np.count_nonzero(tri_one_repair) + np.count_nonzero(tri_two_repair)
        )
        edge_counts["triangle_rejected"] += int(
            np.count_nonzero(~tri_one_keep) + np.count_nonzero(~tri_two_keep)
        )
        if np.any(tri_one_keep):
            keep = columns[tri_one_keep]
            faces_rows.append(
                np.column_stack((a[keep], b[keep], d[keep])).astype(np.int64)
            )
        if np.any(tri_two_keep):
            keep = columns[tri_two_keep]
            faces_rows.append(
                np.column_stack((a[keep], d[keep], c[keep])).astype(np.int64)
            )
    faces = (
        np.concatenate(faces_rows, axis=0)
        if faces_rows
        else np.zeros((0, 3), dtype=np.int64)
    )
    # Orient faces toward the camera without dropping open-boundary faces.
    if len(faces):
        triangles = vertices[faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        view = camera_position.reshape(1, 3) - triangles.mean(axis=1)
        flip = np.einsum("ij,ij->i", normals, view) < 0.0
        faces[flip, 0], faces[flip, 1] = faces[flip, 1].copy(), faces[flip, 0].copy()
    mesh = TriangleMesh(vertices=vertices, faces=faces)
    edge_counts.update(_mesh_boundary_stats(faces))
    edge_counts["selected_vertex_count"] = int(len(vertices))
    edge_counts["unreferenced_vertex_count"] = int(
        len(vertices) - len(np.unique(faces.reshape(-1)))
        if len(faces)
        else len(vertices)
    )
    return mesh, selected, pixel_uv, edge_counts


def reconstruct_rgbd_scene_mesh(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    intrinsic_matrix: np.ndarray | None = None,
    extrinsic_matrix: np.ndarray | None = None,
    camera_pos: Sequence[float] | None = None,
    camera_quat_xyzw: Sequence[float] | None = None,
    focal_length: float | None = None,
    horizontal_aperture: float | None = None,
    pixel_stride: int = DEFAULT_PIXEL_STRIDE,
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
    max_depth_m: float = DEFAULT_MAX_DEPTH_M,
    surface_edge_scale: float = DEFAULT_SURFACE_EDGE_SCALE,
    surface_edge_slack_mm: float = 2.0,
    max_depth_jump_mm: float = 35.0,
    repair_max_depth_jump_mm: float = DEFAULT_REPAIR_MAX_DEPTH_JUMP_M * 1000.0,
    repair_plane_residual_mm: float = DEFAULT_REPAIR_PLANE_RESIDUAL_M * 1000.0,
    repair_max_edge_mm: float = DEFAULT_REPAIR_MAX_EDGE_M * 1000.0,
    repair_rgb_boundary: float = DEFAULT_REPAIR_RGB_BOUNDARY,
    surface_band_min_mm: float = DEFAULT_SURFACE_BAND_MIN_MM,
    surface_band_max_mm: float = DEFAULT_SURFACE_BAND_MAX_MM,
    collision_extrusion_mm: float = DEFAULT_COLLISION_EXTRUSION_MM,
    voxel_size_m: float = VOXEL_M,
    analytic_support_plane: bool = False,
) -> RGBDSceneMeshResult:
    """Build an auditable whole-view surface and conservative collision shell.

    The preferred RoboDojo call supplies ``intrinsic_matrix`` and
    ``extrinsic_matrix``.  The BEHAVIOR-style camera arguments are accepted as
    a convenience for offline cross-checks, but no BEHAVIOR module is imported.
    """
    color = np.asarray(rgb)
    metric = _finite_depth(depth)
    if color.ndim != 3 or color.shape[2] < 3:
        raise ValueError(f"rgb must be HxWx3, got {color.shape}")
    if color.shape[:2] != metric.shape:
        raise ValueError(
            f"rgb/depth shape mismatch: {color.shape[:2]} vs {metric.shape}"
        )
    if intrinsic_matrix is None:
        if focal_length is None or horizontal_aperture is None:
            raise ValueError("camera intrinsics are required")
        focal_px = (
            float(focal_length) / float(horizontal_aperture) * float(metric.shape[1])
        )
        intrinsic = np.array(
            [
                [focal_px, 0.0, metric.shape[1] / 2.0],
                [0.0, focal_px, metric.shape[0] / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    else:
        intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    if extrinsic_matrix is None:
        if camera_pos is None or camera_quat_xyzw is None:
            raise ValueError("camera extrinsics are required")
        quat = np.asarray(camera_quat_xyzw, dtype=np.float64).reshape(4)
        qx, qy, qz, qw = quat / max(float(np.linalg.norm(quat)), 1e-12)
        rotation = np.array(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float64,
        )
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    else:
        extrinsic = np.asarray(extrinsic_matrix, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(extrinsic)):
        raise ValueError("camera matrices must be finite")
    stride = max(1, int(pixel_stride))
    if float(voxel_size_m) <= 0.0:
        raise ValueError("voxel_size_m must be positive")
    if float(surface_band_min_mm) <= 0.0 or float(surface_band_max_mm) < float(
        surface_band_min_mm
    ):
        raise ValueError("invalid surface uncertainty band")
    if float(collision_extrusion_mm) < 0.0:
        raise ValueError("collision extrusion must be nonnegative")
    if float(repair_max_edge_mm) <= 0.0:
        raise ValueError("repair_max_edge_mm must be positive")
    # Invalid depth remains invalid evidence, but need not participate in
    # inf * 0 camera-matrix arithmetic. Do not mutate the frozen input.
    metric_valid = (
        np.isfinite(metric)
        & (metric >= float(min_depth_m))
        & (metric <= float(max_depth_m))
    )
    world, optical, rows, cols = _camera_points(
        np.where(metric_valid, metric, 0.0), intrinsic, extrinsic
    )
    valid = (
        np.isfinite(metric)
        & (metric >= float(min_depth_m))
        & (metric <= float(max_depth_m))
        & np.isfinite(world).all(axis=2)
    )
    camera_position = extrinsic[:3, 3].copy()
    surface_mesh, selected, pixel_uv, edge_audit = _make_surface_mesh(
        color,
        metric,
        world,
        valid,
        pixel_stride=stride,
        intrinsic=intrinsic,
        camera_position=camera_position,
        surface_edge_scale=float(surface_edge_scale),
        surface_edge_slack_m=float(surface_edge_slack_mm) / 1000.0,
        max_depth_jump_m=float(max_depth_jump_mm) / 1000.0,
        repair_max_depth_jump_m=float(repair_max_depth_jump_mm) / 1000.0,
        repair_plane_residual_m=float(repair_plane_residual_mm) / 1000.0,
        repair_max_edge_m=float(repair_max_edge_mm) / 1000.0,
        repair_rgb_boundary=float(repair_rgb_boundary),
    )
    selected_points = world[selected]
    selected_depth = metric[selected]
    normals_grid, normal_fallback, normal_audit = _surface_normal_evidence(
        world, valid, selected, camera_position
    )
    selected_normals = normals_grid[selected]
    # A local band grows with depth gradient and grazing incidence, but remains
    # a narrow observed layer.  It is not a hidden-object thickness estimate.
    local_gradient = _valid_depth_gradient(metric, valid)
    incidence = np.abs(
        np.einsum(
            "ij,ij->i",
            selected_normals,
            (camera_position.reshape(1, 3) - selected_points)
            / np.maximum(
                np.linalg.norm(camera_position.reshape(1, 3) - selected_points, axis=1)[
                    :, None
                ],
                1e-9,
            ),
        )
    )
    band_mm = np.clip(
        float(surface_band_min_mm)
        + 220.0 * local_gradient[selected]
        + 2.5 * (1.0 - incidence),
        float(surface_band_min_mm),
        float(surface_band_max_mm),
    )
    band_mm[normal_fallback[selected]] = float(surface_band_max_mm)
    origin = (
        np.floor(np.min(selected_points, axis=0) / float(voxel_size_m))
        * float(voxel_size_m)
        - float(voxel_size_m)
        if len(selected_points)
        else np.zeros(3, dtype=np.float64)
    )
    support_z, support_audit = _support_plane_audit(world, valid)
    material_mask = np.ones(len(selected_points), dtype=bool)
    support_separation = {"requested": bool(analytic_support_plane), "applied": False}
    if (
        analytic_support_plane
        and support_audit.get("confidence") == "strong"
        and float(support_audit.get("residual_p95_mm", math.inf)) <= 1.5
    ):
        plane = np.asarray(support_audit["plane_coefficients_z_from_xy"])
        plane_height = (
            selected_points[:, 0] * plane[0]
            + selected_points[:, 1] * plane[1]
            + plane[2]
        )
        # Separate *source observations* before either uncertainty dilation
        # or completion. Filtering the already-expanded cells by z leaves
        # the upper half of the tabletop band falsely labelled as material.
        material_mask = np.abs(selected_points[:, 2] - plane_height) > 0.0015
        support_separation.update(
            applied=True,
            plane_coefficients_z_from_xy=plane.tolist(),
            normal_world=support_audit["normal_world"],
            source_inlier_tolerance_m=0.0015,
            removed_source_points=int(np.count_nonzero(~material_mask)),
            collision_policy="analytic_solid_halfspace_checked_by_exact_gripper_mesh",
        )
    band_offsets = (
        np.column_stack(
            (
                selected_points - selected_normals * (band_mm / 1000.0)[:, None],
                selected_points,
                selected_points + selected_normals * (band_mm / 1000.0)[:, None],
            )
        )[material_mask].reshape(-1, 3)
        if len(selected_points)
        else np.zeros((0, 3), dtype=np.float64)
    )
    surface_indices = _voxel_indices_from_points(
        band_offsets, origin=origin, voxel_size_m=float(voxel_size_m)
    )
    surface_occupancy = VoxelOccupancy(
        voxel_size_m=float(voxel_size_m),
        origin_world=origin,
        occupied_indices=surface_indices,
        source=(
            "observed_non_support_surface_plus_local_uncertainty_band"
            if support_separation["applied"]
            else "observed_surface_plus_local_uncertainty_band"
        ),
        query_radius_cells=1,
    )
    # Build the conservative shell from a bounded subset of points.  The
    # subsampling is uniform over the whole image and therefore cannot be a
    # click-local completion.  Its thickness is explicit and separately named.
    material_points = selected_points[material_mask]
    if len(material_points) > 150000:
        shell_step = int(math.ceil(len(material_points) / 150000.0))
        shell_points = material_points[::shell_step]
    else:
        shell_points = material_points
    ray = shell_points - camera_position.reshape(1, 3)
    ray_norm = np.linalg.norm(ray, axis=1)
    ray /= np.maximum(ray_norm, 1e-9)[:, None]
    extrusion_steps = max(
        0, int(math.ceil(float(collision_extrusion_mm) / 1000.0 / float(voxel_size_m)))
    )
    completion_points = np.zeros((0, 3), dtype=np.float64)
    if len(shell_points) and extrusion_steps:
        distances = np.arange(1, extrusion_steps + 1, dtype=np.float64) * float(
            voxel_size_m
        )
        completion_points = (
            shell_points[:, None, :] + ray[:, None, :] * distances.reshape(1, -1, 1)
        ).reshape(-1, 3)
    # The guard applies to unobserved completion only.  The measured surface
    # band is allowed to sit at the observed depth; if completion violates
    # observed free space, reject that completion and retain the band as a
    # conservative-but-honest collision fallback.
    guard = visible_free_space_guard(
        completion_points[:: max(1, len(completion_points) // 100000)],
        metric,
        intrinsic,
        extrinsic,
    )
    completion_applied = bool(guard.get("ok", False) and len(completion_points))
    collision_points = (
        np.concatenate((band_offsets, completion_points), axis=0)
        if completion_applied and len(completion_points)
        else band_offsets
    )
    collision_source = (
        "guarded_camera_ray_conservative_collision_shell"
        if completion_applied
        else "visibility_guard_fallback_observed_surface_band"
    )
    collision_indices = _voxel_indices_from_points(
        collision_points,
        origin=origin,
        voxel_size_m=float(voxel_size_m),
    )
    collision_occupancy = VoxelOccupancy(
        voxel_size_m=float(voxel_size_m),
        origin_world=origin,
        occupied_indices=collision_indices,
        source=collision_source,
        query_radius_cells=1,
    )
    valid_fraction = float(np.count_nonzero(valid) / max(1, valid.size))
    band_percentiles = (
        np.percentile(band_mm, [5.0, 50.0, 95.0]).tolist()
        if len(band_mm)
        else [None, None, None]
    )
    projection_uv, projection_depth = _project_world(
        selected_points,
        intrinsic,
        extrinsic,
    )
    projection_valid = np.isfinite(projection_uv).all(axis=1) & np.isfinite(
        projection_depth
    )
    projection_error = (
        np.linalg.norm(
            projection_uv[projection_valid] - pixel_uv[projection_valid],
            axis=1,
        )
        if np.any(projection_valid)
        else np.zeros(0, dtype=np.float64)
    )
    metadata: dict[str, Any] = {
        "build": RGBD_SCENE_MESH_BUILD,
        "mesh_version": MESH_VERSION,
        "requested_version": MESH_VERSION,
        "effective_version": MESH_VERSION,
        "route": "whole_view_observed_surface_adaptive_repair_guarded_structural_completion",
        "fallback_reason": None,
        "allowed_inputs": [
            "rgb",
            "metric_depth",
            "camera_intrinsics",
            "camera_extrinsics",
        ],
        "forbidden_inputs_used": [],
        "used_click": False,
        "used_segmentation": False,
        "used_object_identity": False,
        "used_simulator_geometry": False,
        "camera_convention": {
            "extrinsic": "usd_camera_xform_to_world",
            "depth": "optical_x_right_y_down_z_forward",
            "optical_to_usd_axes": OPTICAL_TO_USD_AXES.tolist(),
        },
        "depth_shape": [int(metric.shape[0]), int(metric.shape[1])],
        "valid_depth_fraction": valid_fraction,
        "pixel_stride": int(stride),
        "adaptive_sampling": bool(stride > 1),
        "observed_point_count": int(len(selected_points)),
        "surface_mesh": {
            "vertices_world_count": int(len(surface_mesh.vertices)),
            "faces_count": int(len(surface_mesh.faces)),
            "open_boundary_stats": {
                "boundary_edge_count": int(edge_audit.get("boundary_edge_count", 0)),
                "nonmanifold_edge_count": int(
                    edge_audit.get("nonmanifold_edge_count", 0)
                ),
                "face_count": int(
                    edge_audit.get("face_count", len(surface_mesh.faces))
                ),
            },
        },
        "edge_rejection_counts": edge_audit,
        "local_repair_counts": {
            "repaired_triangles": int(edge_audit.get("triangle_repaired", 0)),
            "repaired_vertex_points_retained": int(len(selected_points)),
            "repair_max_depth_jump_mm": float(repair_max_depth_jump_mm),
            "repair_plane_residual_mm": float(repair_plane_residual_mm),
            "repair_max_edge_mm": float(repair_max_edge_mm),
            "repair_rgb_boundary_threshold": float(repair_rgb_boundary),
        },
        "support_plane": support_audit,
        "normal_estimation": normal_audit,
        "support_plane_z_m": support_z,
        "support_separation": support_separation,
        "uncertainty": {
            "local_band_mm_percentiles": band_percentiles,
            "local_band_mm_min": float(np.min(band_mm)) if len(band_mm) else None,
            "local_band_mm_max": float(np.max(band_mm)) if len(band_mm) else None,
            "thickness_status": "unknown_single_view",
            "thickness_lower_bound_mm": float(np.min(band_mm))
            if len(band_mm)
            else None,
            "thickness_upper_bound_mm": None,
            "single_view_backside_observable": False,
        },
        "surface_occupancy": surface_occupancy.metadata(),
        "collision_occupancy": collision_occupancy.metadata(),
        "completion_sources": [
            "observed_surface_plus_local_uncertainty_band",
            collision_source,
        ],
        "completion_candidates": [
            {
                "name": "camera_ray_shell",
                "applicable": True,
                "applied": completion_applied,
                "thickness_mm": float(collision_extrusion_mm),
                "thickness_semantics": "conservative_collision_prior_not_true_volume",
                "visibility_guard": guard,
                "fallback_reason": (
                    None
                    if completion_applied
                    else "completion_rejected_by_visible_free_space_guard"
                ),
            },
            {
                "name": "support_heightfield_or_planar_prism",
                "applicable": bool(support_z is not None),
                "applied": False,
                "reason": "deferred_without_observed_hole_evidence",
                "visibility_guard": None,
            },
        ],
        "free_space_violation_fraction": float(guard.get("violation_fraction", 0.0)),
        "free_space_violation_pixels": int(guard.get("violation_pixels", 0)),
        "projection_residual_stats": {
            "reprojection_error_px": float(np.max(projection_error))
            if len(projection_error)
            else None,
            "reprojection_error_mean_px": float(np.mean(projection_error))
            if len(projection_error)
            else None,
            "reprojection_error_p95_px": float(np.percentile(projection_error, 95.0))
            if len(projection_error)
            else None,
            "projected_point_count": int(len(projection_error)),
            "source": "analytic_backprojection_roundtrip_contract",
        },
        "grasp_volume_semantics": "surface_occupancy_intersection_with_exact_x5_opening",
        "collision_overlap_semantics": "collision_occupancy_intersection_with_official_x5_mesh",
    }
    # Keep the lightweight mesh in the public ``mesh`` slot for callers that
    # use the BEHAVIOR result shape.  It intentionally remains open when the
    # RGB-D evidence has open boundaries.
    return RGBDSceneMeshResult(
        mesh=surface_mesh,
        surface_mesh=surface_mesh,
        surface_occupancy=surface_occupancy,
        collision_occupancy=collision_occupancy,
        metadata=metadata,
        observed_points_world=selected_points,
        observed_pixel_uv=pixel_uv,
    )


def reconstruct_scene_mesh_from_snapshot(
    snapshot: Any, **options: Any
) -> RGBDSceneMeshResult:
    """Convenience adapter for the immutable RoboDojo ``Snapshot`` object."""
    if getattr(snapshot, "depth", None) is None:
        raise ValueError("snapshot has no depth")
    if (
        getattr(snapshot, "intrinsic_matrix", None) is None
        or getattr(snapshot, "extrinsic_matrix", None) is None
    ):
        raise ValueError("snapshot has no camera matrices")
    return reconstruct_rgbd_scene_mesh(
        snapshot.rgb,
        snapshot.depth,
        intrinsic_matrix=snapshot.intrinsic_matrix,
        extrinsic_matrix=snapshot.extrinsic_matrix,
        **options,
    )


__all__ = [
    "DEFAULT_COLLISION_EXTRUSION_MM",
    "DEFAULT_PIXEL_STRIDE",
    "MESH_VERSION",
    "MeshSceneOccupancy",
    "RGBDSceneMeshResult",
    "RGBD_SCENE_MESH_BUILD",
    "TriangleMesh",
    "VOXEL_M",
    "VoxelOccupancy",
    "reconstruct_rgbd_scene_mesh",
    "reconstruct_scene_mesh_from_snapshot",
    "visible_free_space_guard",
]
