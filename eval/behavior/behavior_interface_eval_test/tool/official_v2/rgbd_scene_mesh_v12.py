"""Experimental v12 whole-view RGB-D scene reconstruction.

The runtime contract is intentionally scene-level: RGB, metric depth, and
camera calibration are the only inputs. Depth discontinuities are used to
avoid triangles that bridge unrelated visible surfaces, but every resulting
surface patch is retained. No click, segmentation, object identity, reference
mesh, evaluation center, or overlap label is accepted.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import inspect
import math
import os
import threading
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
import trimesh


RGBD_SCENE_MESH_BUILD = "rgbd_whole_view_structural_priors_v12"

_SUPPORT_STRUCTURE_CACHE: "OrderedDict[str, tuple[Any, ...]]" = (
    OrderedDict()
)
_SUPPORT_STRUCTURE_CACHE_LOCK = threading.Lock()


@dataclass
class RGBDSceneMeshResult:
    mesh: trimesh.Trimesh
    metadata: Dict[str, Any]


def _quat_to_mat_xyzw(quat: Sequence[float]) -> np.ndarray:
    x, y, z, w = [
        float(value)
        for value in np.asarray(quat, dtype=np.float64).reshape(4)
    ]
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _sampled_world_points(
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    pixel_stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    if metric_depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {metric_depth.shape}")
    height, width = metric_depth.shape
    stride = int(pixel_stride)
    if stride < 1:
        raise ValueError(f"pixel_stride must be >=1, got {stride}")
    focal_px = (
        float(focal_length)
        / float(horizontal_aperture)
        * float(width)
    )
    if not np.isfinite(focal_px) or focal_px <= 0.0:
        raise ValueError(f"invalid focal length in pixels: {focal_px}")

    rows = np.arange(0, height, stride, dtype=np.int64)
    cols = np.arange(0, width, stride, dtype=np.int64)
    sampled_depth = metric_depth[np.ix_(rows, cols)]
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    x_cam = (uu - width / 2.0) / focal_px * sampled_depth
    y_cam = -(vv - height / 2.0) / focal_px * sampled_depth
    z_cam = -sampled_depth
    camera_points = np.stack((x_cam, y_cam, z_cam), axis=-1)

    rotation = _quat_to_mat_xyzw(camera_quat_xyzw)
    position = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    world_points = camera_points @ rotation.T + position
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= float(min_depth_m))
        & (sampled_depth <= float(max_depth_m))
        & np.isfinite(world_points).all(axis=-1)
    )
    return world_points, sampled_depth, valid, camera_points, float(focal_px)


def _continuous_edge(
    first_points: np.ndarray,
    second_points: np.ndarray,
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    *,
    pixel_span: float,
    focal_px: float,
    surface_edge_scale: float,
    surface_edge_slack_m: float,
    max_depth_jump_m: float,
) -> np.ndarray:
    distance = np.linalg.norm(first_points - second_points, axis=-1)
    expected_lateral = (
        float(pixel_span)
        * np.maximum(first_depth, second_depth)
        / float(focal_px)
    )
    limit = (
        float(surface_edge_scale) * expected_lateral
        + float(surface_edge_slack_m)
    )
    return (
        (distance <= limit)
        & (
            np.abs(first_depth - second_depth)
            <= float(max_depth_jump_m)
        )
    )


def _rgbd_edge_gate(
    geometric_edge: np.ndarray,
    first_depth: np.ndarray,
    second_depth: np.ndarray,
    first_color: np.ndarray,
    second_color: np.ndarray,
    *,
    color_threshold: float,
    min_depth_jump_m: float,
) -> np.ndarray:
    if float(color_threshold) <= 0.0:
        return np.asarray(geometric_edge, dtype=bool)
    color_delta = np.linalg.norm(
        np.asarray(first_color, dtype=np.float64)
        - np.asarray(second_color, dtype=np.float64),
        axis=-1,
    )
    rgbd_boundary = (
        np.abs(
            np.asarray(first_depth, dtype=np.float64)
            - np.asarray(second_depth, dtype=np.float64)
        )
        >= float(min_depth_jump_m)
    ) & (color_delta >= float(color_threshold))
    return np.asarray(geometric_edge, dtype=bool) & (~rgbd_boundary)


def _thin_protrusion_mask(
    depth: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    background_window_px: int,
    min_protrusion_mm: float,
    closing_iterations: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Find all narrow geometry protruding from a local depth background."""
    metric_depth = np.asarray(depth, dtype=np.float64)
    window = int(background_window_px)
    iterations = int(closing_iterations)
    if window < 3 or window % 2 == 0:
        raise ValueError("thin background window must be odd and >=3")
    if float(min_protrusion_mm) <= 0.0:
        raise ValueError("thin minimum protrusion must be positive")
    if iterations < 0:
        raise ValueError("thin closing iterations must be nonnegative")

    valid = (
        np.isfinite(metric_depth)
        & (metric_depth >= float(min_depth_m))
        & (metric_depth <= float(max_depth_m))
    )
    finite_depth = np.where(valid, metric_depth, float(max_depth_m))
    local_background = ndimage.grey_closing(
        finite_depth,
        size=(window, window),
    )
    residual = local_background - finite_depth
    raw = valid & (
        residual >= float(min_protrusion_mm) / 1000.0
    )
    if iterations:
        completed = ndimage.binary_closing(
            raw,
            structure=np.ones((3, 3), dtype=bool),
            iterations=iterations,
        )
        completed &= valid
    else:
        completed = raw.copy()
    return raw, completed, {
        "thin_raw_pixels": int(raw.sum()),
        "thin_completed_pixels": int(completed.sum()),
        "thin_background_window_px": window,
        "thin_min_protrusion_mm": float(min_protrusion_mm),
        "thin_closing_iterations": iterations,
    }


def _multiscale_prominence_radius_px(
    depth: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    window_sizes_px: Sequence[int],
    min_prominence_mm: float,
    closing_iterations: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Estimate local foreground radius from depth prominence only."""
    metric_depth = np.asarray(depth, dtype=np.float64)
    windows = tuple(sorted({int(value) for value in window_sizes_px}))
    if not windows or any(value < 3 or value % 2 == 0 for value in windows):
        raise ValueError("local width windows must be odd and >=3")
    if float(min_prominence_mm) <= 0.0:
        raise ValueError("local width minimum prominence must be positive")
    iterations = int(closing_iterations)
    if iterations < 0:
        raise ValueError("local width closing iterations must be nonnegative")

    valid = (
        np.isfinite(metric_depth)
        & (metric_depth >= float(min_depth_m))
        & (metric_depth <= float(max_depth_m))
    )
    finite_depth = np.where(valid, metric_depth, float(max_depth_m))
    combined_mask = np.zeros(metric_depth.shape, dtype=bool)
    maximum_radius = np.zeros(metric_depth.shape, dtype=np.float64)
    scale_rows: list[dict[str, Any]] = []
    threshold_m = float(min_prominence_mm) / 1000.0
    for window in windows:
        local_background = ndimage.grey_closing(
            finite_depth,
            size=(window, window),
        )
        raw = valid & ((local_background - finite_depth) >= threshold_m)
        completed = raw
        if iterations:
            completed = ndimage.binary_closing(
                raw,
                structure=np.ones((3, 3), dtype=bool),
                iterations=iterations,
            )
            completed &= valid
        radius = ndimage.distance_transform_edt(completed)
        maximum_radius = np.maximum(maximum_radius, radius)
        combined_mask |= completed
        scale_rows.append(
            {
                "window_px": int(window),
                "raw_pixels": int(raw.sum()),
                "completed_pixels": int(completed.sum()),
                "maximum_radius_px": float(radius.max()),
            }
        )
    return combined_mask, maximum_radius, {
        "local_width_windows_px": list(windows),
        "local_width_min_prominence_mm": float(min_prominence_mm),
        "local_width_closing_iterations": iterations,
        "local_width_prominent_pixels": int(combined_mask.sum()),
        "local_width_scales": scale_rows,
    }


def _skeleton_graph_chains(
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, Dict[str, int]]:
    """Split an acyclic 8-neighbor skeleton into terminal chains."""
    sy = np.asarray(skeleton_y, dtype=np.int64).reshape(-1)
    sx = np.asarray(skeleton_x, dtype=np.int64).reshape(-1)
    if len(sy) == 0 or len(sy) != len(sx):
        raise ValueError("invalid thin skeleton pixels")
    index_by_pixel = {
        (int(y), int(x)): index
        for index, (y, x) in enumerate(zip(sy, sx))
    }
    adjacency: list[list[int]] = [[] for _ in range(len(sy))]
    for index, (y, x) in enumerate(zip(sy, sx)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = index_by_pixel.get((int(y + dy), int(x + dx)))
                if neighbor is None:
                    continue
                if abs(dy) == 1 and abs(dx) == 1:
                    if (
                        (int(y + dy), int(x)) in index_by_pixel
                        or (int(y), int(x + dx)) in index_by_pixel
                    ):
                        continue
                adjacency[index].append(int(neighbor))
    degrees = np.asarray(
        [len(neighbors) for neighbors in adjacency],
        dtype=np.int64,
    )
    terminals = np.flatnonzero(degrees != 2)
    visited_edges: set[tuple[int, int]] = set()
    chains: list[np.ndarray] = []
    for terminal in terminals:
        if degrees[terminal] == 0:
            chains.append(np.array([int(terminal)], dtype=np.int64))
            continue
        for neighbor in adjacency[terminal]:
            edge = tuple(sorted((int(terminal), int(neighbor))))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            path = [int(terminal), int(neighbor)]
            previous = int(terminal)
            current = int(neighbor)
            while degrees[current] == 2:
                candidates = [
                    value
                    for value in adjacency[current]
                    if value != previous
                ]
                if len(candidates) != 1:
                    break
                next_index = int(candidates[0])
                next_edge = tuple(sorted((current, next_index)))
                if next_edge in visited_edges:
                    break
                visited_edges.add(next_edge)
                path.append(next_index)
                previous, current = current, next_index
            chains.append(np.asarray(path, dtype=np.int64))

    edge_count = int(sum(degrees) // 2)
    if len(visited_edges) != edge_count:
        raise ValueError("thin skeleton contains a cycle")
    covered = {
        int(index)
        for chain in chains
        for index in chain.tolist()
    }
    if len(covered) != len(sy):
        raise ValueError("thin skeleton is disconnected")
    return chains, degrees, {
        "sections": int(len(sy)),
        "edges": edge_count,
        "chains": int(len(chains)),
        "endpoints": int(np.count_nonzero(degrees == 1)),
        "junctions": int(np.count_nonzero(degrees > 2)),
    }


def _swept_transported_frames(
    centerline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centerline, dtype=np.float64).reshape(-1, 3)
    tangents = np.gradient(centers, axis=0)
    tangent_norm = np.linalg.norm(tangents, axis=1)
    if np.any(tangent_norm < 1e-9):
        raise ValueError("thin tube centerline has duplicate sections")
    tangents /= tangent_norm[:, None]
    normals = np.empty_like(tangents)
    hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(hint @ tangents[0])) > 0.9:
        hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    projected = hint - tangents[0] * float(hint @ tangents[0])
    normals[0] = projected / max(float(np.linalg.norm(projected)), 1e-12)
    for index in range(1, len(tangents)):
        projected = (
            normals[index - 1]
            - tangents[index]
            * float(normals[index - 1] @ tangents[index])
        )
        norm = float(np.linalg.norm(projected))
        if norm < 1e-9:
            hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(hint @ tangents[index])) > 0.9:
                hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            projected = (
                hint
                - tangents[index] * float(hint @ tangents[index])
            )
            norm = float(np.linalg.norm(projected))
        normals[index] = projected / max(norm, 1e-12)
    binormals = np.cross(tangents, normals)
    binormals /= np.maximum(
        np.linalg.norm(binormals, axis=1, keepdims=True),
        1e-12,
    )
    return normals, binormals


def _swept_circular_tube_mesh(
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    ring_samples: int,
) -> trimesh.Trimesh:
    centerline = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    rings = int(ring_samples)
    if (
        len(centerline) == 0
        or len(centerline) != len(radius_values)
        or np.any(radius_values <= 0.0)
        or not np.all(np.isfinite(centerline))
        or not np.all(np.isfinite(radius_values))
        or rings < 8
    ):
        raise ValueError("invalid thin tube geometry")
    if len(centerline) == 1:
        sphere = trimesh.creation.icosphere(
            subdivisions=2,
            radius=float(radius_values[0]),
        )
        sphere.apply_translation(centerline[0])
        return sphere

    normals, binormals = _swept_transported_frames(centerline)
    angles = np.linspace(
        0.0,
        2.0 * math.pi,
        rings,
        endpoint=False,
    )
    directions = (
        np.cos(angles)[None, :, None] * normals[:, None, :]
        + np.sin(angles)[None, :, None] * binormals[:, None, :]
    )
    vertices = (
        centerline[:, None, :]
        + radius_values[:, None, None] * directions
    ).reshape(-1, 3)
    faces: list[list[int]] = []
    for section in range(len(centerline) - 1):
        for ring_index in range(rings):
            next_ring = (ring_index + 1) % rings
            first = section * rings + ring_index
            second = section * rings + next_ring
            third = (section + 1) * rings + next_ring
            fourth = (section + 1) * rings + ring_index
            faces.append([first, second, third])
            faces.append([first, third, fourth])
    start_center = len(vertices)
    end_center = start_center + 1
    vertices = np.vstack((vertices, centerline[0], centerline[-1]))
    end_start = (len(centerline) - 1) * rings
    for ring_index in range(rings):
        next_ring = (ring_index + 1) % rings
        faces.append([start_center, next_ring, ring_index])
        faces.append(
            [
                end_center,
                end_start + ring_index,
                end_start + next_ring,
            ]
        )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
        validate=True,
    )
    if not bool(mesh.is_watertight):
        raise ValueError("thin swept tube is not watertight")
    if float(mesh.volume) < 0.0:
        mesh.invert()
    return mesh


def _thin_structure_tube_mesh(
    *,
    raw_mask: np.ndarray,
    completed_mask: np.ndarray,
    depth: np.ndarray,
    camera_pos: np.ndarray,
    camera_rotation: np.ndarray,
    focal_px: float,
    min_component_pixels: int,
    max_radius_px: float,
    radius_scale: float,
    min_radius_mm: float,
    max_radius_mm: float,
    center_offset_scale: float,
    ring_samples: int,
) -> tuple[trimesh.Trimesh | None, Dict[str, Any]]:
    """Reconstruct every eligible thin depth protrusion as a 3D tube graph."""
    raw = np.asarray(raw_mask, dtype=bool)
    completed = np.asarray(completed_mask, dtype=bool)
    metric_depth = np.asarray(depth, dtype=np.float64)
    labels, component_count = ndimage.label(
        completed,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    distance_to_boundary = ndimage.distance_transform_edt(completed)
    raw_distance_to_boundary = ndimage.distance_transform_edt(raw)
    _distance_to_raw, nearest_raw = ndimage.distance_transform_edt(
        ~raw,
        return_indices=True,
    )
    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    parts: list[trimesh.Trimesh] = []
    eligible_components = 0
    skipped_cycles = 0
    skeleton_sections = 0
    graph_chains = 0
    junction_spheres = 0
    radius_values_all: list[np.ndarray] = []
    component_sizes = np.bincount(labels.reshape(-1))
    for component_id in range(1, int(component_count) + 1):
        component = labels == component_id
        if int(component_sizes[component_id]) < int(min_component_pixels):
            continue
        component_radius_px = float(
            distance_to_boundary[component].max(initial=0.0)
        )
        if component_radius_px > float(max_radius_px):
            continue
        skeleton = skeletonize(component)
        sy, sx = np.nonzero(skeleton)
        if len(sy) == 0:
            continue
        if len(sy) == 1:
            chains = [np.array([0], dtype=np.int64)]
            degrees = np.zeros(1, dtype=np.int64)
            graph_metadata = {
                "sections": 1,
                "edges": 0,
                "chains": 1,
                "endpoints": 0,
                "junctions": 0,
            }
        else:
            try:
                chains, degrees, graph_metadata = _skeleton_graph_chains(
                    sy,
                    sx,
                )
            except ValueError:
                skipped_cycles += 1
                continue

        source_y = nearest_raw[0, sy, sx]
        source_x = nearest_raw[1, sy, sx]
        section_depth = metric_depth[source_y, source_x]
        x_cam = (
            sx.astype(np.float64) - metric_depth.shape[1] * 0.5
        ) / float(focal_px) * section_depth
        y_cam = -(
            sy.astype(np.float64) - metric_depth.shape[0] * 0.5
        ) / float(focal_px) * section_depth
        camera_points = np.column_stack(
            (x_cam, y_cam, -section_depth)
        )
        surface_world = camera_points @ camera_rotation.T + camera
        ray_direction = surface_world - camera.reshape(1, 3)
        ray_direction /= np.maximum(
            np.linalg.norm(ray_direction, axis=1, keepdims=True),
            1e-12,
        )
        radii = (
            raw_distance_to_boundary[source_y, source_x]
            * section_depth
            / float(focal_px)
            * float(radius_scale)
        )
        radii = np.clip(
            radii,
            float(min_radius_mm) / 1000.0,
            float(max_radius_mm) / 1000.0,
        )
        radii = ndimage.median_filter(radii, size=5, mode="nearest")
        centers = (
            surface_world
            + ray_direction
            * (radii * float(center_offset_scale))[:, None]
        )
        for chain in chains:
            chain_centers = centers[chain].copy()
            chain_radii = radii[chain].copy()
            if len(chain) >= 5:
                smoothed = ndimage.gaussian_filter1d(
                    chain_centers,
                    sigma=0.65,
                    axis=0,
                    mode="nearest",
                )
                smoothed[0] = chain_centers[0]
                smoothed[-1] = chain_centers[-1]
                chain_centers = smoothed
            if len(chain) >= 2 and np.any(
                np.linalg.norm(
                    np.diff(chain_centers, axis=0),
                    axis=1,
                )
                < 1e-7
            ):
                continue
            parts.append(
                _swept_circular_tube_mesh(
                    chain_centers,
                    chain_radii,
                    ring_samples=int(ring_samples),
                )
            )
            graph_chains += 1
        for junction in np.flatnonzero(degrees > 2):
            sphere = trimesh.creation.icosphere(
                subdivisions=2,
                radius=float(radii[junction]),
            )
            sphere.apply_translation(centers[junction])
            parts.append(sphere)
            junction_spheres += 1
        eligible_components += 1
        skeleton_sections += int(graph_metadata["sections"])
        radius_values_all.append(radii)

    if not parts:
        return None, {
            "thin_labeled_components": int(component_count),
            "thin_eligible_components": 0,
            "thin_skipped_cycle_components": int(skipped_cycles),
            "thin_skeleton_sections": 0,
            "thin_graph_chains": 0,
            "thin_junction_spheres": 0,
        }
    mesh = trimesh.util.concatenate(parts)
    all_radii = np.concatenate(radius_values_all)
    return mesh, {
        "thin_labeled_components": int(component_count),
        "thin_eligible_components": int(eligible_components),
        "thin_skipped_cycle_components": int(skipped_cycles),
        "thin_skeleton_sections": int(skeleton_sections),
        "thin_graph_chains": int(graph_chains),
        "thin_junction_spheres": int(junction_spheres),
        "thin_tube_parts": int(len(parts)),
        "thin_radius_mm_quantiles": {
            key: float(value * 1000.0)
            for key, value in zip(
                ("0", "10", "25", "50", "75", "90", "100"),
                np.percentile(
                    all_radii,
                    [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0],
                ),
            )
        },
    }


def _thin_structure_circle_tube_mesh(
    *,
    raw_mask: np.ndarray,
    depth: np.ndarray,
    world_points: np.ndarray,
    camera_pos: np.ndarray,
    focal_px: float,
    min_component_pixels: int,
    max_radius_px: float,
    radius_scale: float,
    radius_bias_px: float,
    min_radius_mm: float,
    max_radius_mm: float,
    center_offset_scale: float,
    ring_samples: int,
    nonlinear_refine: bool,
) -> tuple[trimesh.Trimesh | None, Dict[str, Any]]:
    """Fit circular cross-sections to every narrow depth protrusion."""
    from .rgbd_only_mesh_reconstruction import (
        _component_circle_tube_sections,
        _refine_local_cylinder_sections_nonlinear,
    )

    raw = np.asarray(raw_mask, dtype=bool)
    metric_depth = np.asarray(depth, dtype=np.float64)
    world = np.asarray(world_points, dtype=np.float64)
    labels, component_count = ndimage.label(
        raw,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    distance_to_boundary = ndimage.distance_transform_edt(raw)
    component_sizes = np.bincount(labels.reshape(-1))
    parts: list[trimesh.Trimesh] = []
    eligible_components = 0
    skipped_cycles = 0
    graph_chains = 0
    skeleton_sections = 0
    junction_spheres = 0
    nonlinear_attempted = 0
    nonlinear_accepted = 0
    radius_values_all: list[np.ndarray] = []
    fit_metadata_all: list[Dict[str, Any]] = []
    nonlinear_metadata_all: list[Dict[str, Any]] = []
    for component_id in range(1, int(component_count) + 1):
        component = labels == component_id
        if int(component_sizes[component_id]) < int(min_component_pixels):
            continue
        component_radius_px = float(
            distance_to_boundary[component].max(initial=0.0)
        )
        if component_radius_px > float(max_radius_px):
            continue
        skeleton = skeletonize(component)
        sy, sx = np.nonzero(skeleton)
        if len(sy) == 0:
            continue
        if len(sy) == 1:
            chains = [np.array([0], dtype=np.int64)]
            degrees = np.zeros(1, dtype=np.int64)
        else:
            try:
                chains, degrees, _graph_metadata = (
                    _skeleton_graph_chains(sy, sx)
                )
            except ValueError:
                skipped_cycles += 1
                continue
        centers, radii, fit_metadata = (
            _component_circle_tube_sections(
                component_mask=component,
                skeleton_y=sy,
                skeleton_x=sx,
                world=world,
                depth=metric_depth,
                camera_pos=camera_pos,
                focal_px=float(focal_px),
                radius_scale=float(radius_scale),
                radius_bias_px=float(radius_bias_px),
                radius_min_m=float(min_radius_mm) / 1000.0,
                radius_max_m=float(max_radius_mm) / 1000.0,
                fit_max_residual_m=0.00075,
                scale_min=0.5,
                scale_max=1.5,
                min_valid_fits=8,
                smoothing_px=10.0,
                local_shrinkage=1.0,
                radius_blend=0.0,
                direct_radius_blend=1.0,
                soft_direct_radius_blend=True,
                soft_direct_radius_blend_max=0.5,
                direct_center_blend=0.0,
                front_anchored_radius_blend=0.0,
                endpoint_radius_regularization_blend=1.0,
                graph_radius_smoothing_gcv=False,
                center_fit_blend=1.0,
                center_offset_scale=float(center_offset_scale),
                tangent_neighborhood_px=10.0,
                max_center_toward_camera_ratio=1.0,
            )
        )
        nonlinear_metadata: Dict[str, Any] = {
            "enabled": False,
            "attempted_sections": 0,
            "accepted_sections": 0,
        }
        if bool(nonlinear_refine) and len(sy) >= 10:
            (
                centers,
                radii,
                _tangents,
                nonlinear_metadata,
            ) = _refine_local_cylinder_sections_nonlinear(
                component_mask=component,
                skeleton_y=sy,
                skeleton_x=sx,
                world=world,
                camera_pos=camera_pos,
                initial_centers=centers,
                initial_radii=radii,
                tangent_neighborhood_px=10.0,
            )
        for chain in chains:
            chain_centers = centers[chain].copy()
            chain_radii = radii[chain].copy()
            if len(chain) >= 5:
                smoothed = ndimage.gaussian_filter1d(
                    chain_centers,
                    sigma=0.65,
                    axis=0,
                    mode="nearest",
                )
                smoothed[0] = chain_centers[0]
                smoothed[-1] = chain_centers[-1]
                chain_centers = smoothed
            if len(chain) >= 2 and np.any(
                np.linalg.norm(
                    np.diff(chain_centers, axis=0),
                    axis=1,
                )
                < 1e-7
            ):
                continue
            parts.append(
                _swept_circular_tube_mesh(
                    chain_centers,
                    chain_radii,
                    ring_samples=int(ring_samples),
                )
            )
            graph_chains += 1
        for junction in np.flatnonzero(degrees > 2):
            sphere = trimesh.creation.icosphere(
                subdivisions=2,
                radius=float(radii[junction]),
            )
            sphere.apply_translation(centers[junction])
            parts.append(sphere)
            junction_spheres += 1
        eligible_components += 1
        skeleton_sections += int(len(sy))
        nonlinear_attempted += int(
            nonlinear_metadata.get("attempted_sections", 0)
        )
        nonlinear_accepted += int(
            nonlinear_metadata.get("accepted_sections", 0)
        )
        radius_values_all.append(radii)
        fit_metadata_all.append(fit_metadata)
        nonlinear_metadata_all.append(nonlinear_metadata)

    if not parts:
        return None, {
            "thin_labeled_components": int(component_count),
            "thin_eligible_components": 0,
            "thin_skipped_cycle_components": int(skipped_cycles),
            "thin_skeleton_sections": 0,
            "thin_graph_chains": 0,
            "thin_junction_spheres": 0,
        }
    mesh = trimesh.util.concatenate(parts)
    all_radii = np.concatenate(radius_values_all)
    return mesh, {
        "thin_labeled_components": int(component_count),
        "thin_eligible_components": int(eligible_components),
        "thin_skipped_cycle_components": int(skipped_cycles),
        "thin_skeleton_sections": int(skeleton_sections),
        "thin_graph_chains": int(graph_chains),
        "thin_junction_spheres": int(junction_spheres),
        "thin_tube_parts": int(len(parts)),
        "thin_circle_fit_components": fit_metadata_all,
        "thin_nonlinear_components": nonlinear_metadata_all,
        "thin_nonlinear_attempted_sections": int(
            nonlinear_attempted
        ),
        "thin_nonlinear_accepted_sections": int(
            nonlinear_accepted
        ),
        "thin_radius_mm_quantiles": {
            key: float(value * 1000.0)
            for key, value in zip(
                ("0", "10", "25", "50", "75", "90", "100"),
                np.percentile(
                    all_radii,
                    [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0],
                ),
            )
        },
    }


def _orient_front_triangles(
    triangles: np.ndarray,
    points: np.ndarray,
    *,
    camera_pos: np.ndarray,
) -> np.ndarray:
    oriented = np.asarray(triangles, dtype=np.int64).copy()
    first = points[oriented[:, 0]]
    second = points[oriented[:, 1]]
    third = points[oriented[:, 2]]
    normals = np.cross(second - first, third - first)
    view = camera_pos.reshape(1, 3) - (first + second + third) / 3.0
    flip = np.einsum("ij,ij->i", normals, view) < 0.0
    oriented[flip, 1], oriented[flip, 2] = (
        oriented[flip, 2].copy(),
        oriented[flip, 1].copy(),
    )
    return oriented


def _boundary_directed_edges(triangles: np.ndarray) -> np.ndarray:
    directed = np.concatenate(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ),
        axis=0,
    )
    undirected = np.sort(directed, axis=1)
    _unique, inverse, counts = np.unique(
        undirected,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return directed[counts[inverse] == 1]


def _split_nonmanifold_vertex_fans_cpu_reference(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Duplicate pinch vertices so every incident triangle fan is manifold."""
    source_points = np.asarray(points, dtype=np.float64)
    source_faces = np.asarray(triangles, dtype=np.int64)
    output_faces = source_faces.copy()
    incident_faces: list[list[int]] = [
        [] for _ in range(len(source_points))
    ]
    for face_index, face in enumerate(source_faces):
        for vertex in face:
            incident_faces[int(vertex)].append(int(face_index))

    output_points = [point.copy() for point in source_points]
    split_vertices = 0
    added_vertex_fans = 0
    for vertex, face_indices in enumerate(incident_faces):
        if len(face_indices) <= 1:
            continue
        local_index = {
            face_index: index
            for index, face_index in enumerate(face_indices)
        }
        parent = np.arange(len(face_indices), dtype=np.int64)

        def find(index: int) -> int:
            while int(parent[index]) != index:
                parent[index] = parent[int(parent[index])]
                index = int(parent[index])
            return index

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        edge_faces: dict[int, list[int]] = {}
        for face_index in face_indices:
            face = source_faces[face_index]
            other_vertices = face[face != vertex]
            for other_vertex in other_vertices:
                edge_faces.setdefault(int(other_vertex), []).append(
                    int(face_index)
                )
        for shared_faces in edge_faces.values():
            anchor = local_index[shared_faces[0]]
            for face_index in shared_faces[1:]:
                union(anchor, local_index[face_index])

        fan_faces: dict[int, list[int]] = {}
        for face_index in face_indices:
            root = find(local_index[face_index])
            fan_faces.setdefault(root, []).append(face_index)
        fans = sorted(
            fan_faces.values(),
            key=lambda values: min(values),
        )
        if len(fans) <= 1:
            continue
        split_vertices += 1
        for fan in fans[1:]:
            replacement = len(output_points)
            output_points.append(source_points[vertex].copy())
            added_vertex_fans += 1
            for face_index in fan:
                corners = source_faces[face_index] == vertex
                output_faces[face_index, corners] = replacement

    return (
        np.asarray(output_points, dtype=np.float64),
        output_faces,
        {
            "split_nonmanifold_vertices": int(split_vertices),
            "added_vertex_fans": int(added_vertex_fans),
        },
    )


def _split_nonmanifold_vertex_fans(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    use_cuda = os.environ.get(
        "OFFICIAL_V2_V53_GPU_FAN_SPLIT",
        "0",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if not use_cuda:
        return _split_nonmanifold_vertex_fans_cpu_reference(
            points,
            triangles,
        )
    from .rgbd_cuda_mesh_ops import split_nonmanifold_vertex_fans_cuda

    return split_nonmanifold_vertex_fans_cuda(
        points,
        triangles,
        requested_device=os.environ.get(
            "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE",
            "cuda:0",
        ),
        allow_cpu_reference=False,
    )


def _cap_oriented_boundary_loops(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, int]:
    """Close simple oriented boundary loops with one centroid fan each."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    boundary = _boundary_directed_edges(faces)
    if len(boundary) == 0:
        return mesh, 0

    outgoing: dict[int, list[int]] = {}
    incoming_count: dict[int, int] = {}
    for edge_index, (start, stop) in enumerate(boundary):
        outgoing.setdefault(int(start), []).append(edge_index)
        incoming_count[int(stop)] = incoming_count.get(int(stop), 0) + 1
    boundary_vertices = set(boundary.reshape(-1).tolist())
    if any(
        len(outgoing.get(int(vertex), [])) != 1
        or incoming_count.get(int(vertex), 0) != 1
        for vertex in boundary_vertices
    ):
        raise ValueError("mesh boundary is not a set of simple loops")

    unused = set(range(len(boundary)))
    output_vertices = [point.copy() for point in vertices]
    cap_faces: list[list[int]] = []
    loop_count = 0
    while unused:
        first_edge_index = min(unused)
        start_vertex = int(boundary[first_edge_index, 0])
        edge_index = first_edge_index
        loop_edges: list[tuple[int, int]] = []
        while True:
            if edge_index not in unused:
                raise ValueError("mesh boundary loop repeats before closing")
            unused.remove(edge_index)
            start = int(boundary[edge_index, 0])
            stop = int(boundary[edge_index, 1])
            loop_edges.append((start, stop))
            if stop == start_vertex:
                break
            edge_index = outgoing[stop][0]
        loop_vertices = np.asarray(
            [edge[0] for edge in loop_edges],
            dtype=np.int64,
        )
        center_index = len(output_vertices)
        output_vertices.append(vertices[loop_vertices].mean(axis=0))
        for start, stop in loop_edges:
            cap_faces.append([stop, start, center_index])
        loop_count += 1

    closed = trimesh.Trimesh(
        vertices=np.asarray(output_vertices, dtype=np.float64),
        faces=np.vstack(
            (
                faces,
                np.asarray(cap_faces, dtype=np.int64),
            )
        ),
        process=False,
    )
    closed.process(validate=True)
    return closed, int(loop_count)


def _close_front_surface_components(
    front_points: np.ndarray,
    front_triangles: np.ndarray,
    *,
    camera_pos: np.ndarray,
    camera_forward: np.ndarray,
    front_offset_mm: float,
    back_extrusion_mm: float,
    _pre_split_components: Sequence[trimesh.Trimesh] | None = None,
    _defer_watertight_validation: bool = False,
) -> tuple[trimesh.Trimesh, Dict[str, int]]:
    """Close edge-connected front patches without merging point contacts."""
    if _pre_split_components is None:
        front_mesh = trimesh.Trimesh(
            vertices=np.asarray(front_points, dtype=np.float64),
            faces=np.asarray(front_triangles, dtype=np.int64),
            process=False,
        )
        front_components = front_mesh.split(only_watertight=False)
    else:
        front_components = tuple(_pre_split_components)
    shells: list[trimesh.Trimesh] = []
    boundary_edge_count = 0
    nonwatertight_shells = 0
    for component in front_components:
        points = np.asarray(component.vertices, dtype=np.float64)
        triangles = _orient_front_triangles(
            np.asarray(component.faces, dtype=np.int64),
            points,
            camera_pos=camera_pos,
        )
        ray_vectors = points - camera_pos.reshape(1, 3)
        axial_depth = ray_vectors @ camera_forward.reshape(3)
        if np.any(axial_depth <= 1e-8):
            raise ValueError("front component contains points behind camera")
        front_scale = (
            1.0
            + float(front_offset_mm)
            / 1000.0
            / axial_depth
        )
        back_scale = (
            1.0
            + (
                float(front_offset_mm) + float(back_extrusion_mm)
            )
            / 1000.0
            / axial_depth
        )
        shifted_front = (
            camera_pos.reshape(1, 3)
            + ray_vectors * front_scale[:, None]
        )
        shifted_back = (
            camera_pos.reshape(1, 3)
            + ray_vectors * back_scale[:, None]
        )
        vertex_count = len(points)
        back_triangles = triangles[:, [0, 2, 1]] + vertex_count
        boundary_edges = _boundary_directed_edges(triangles)
        boundary_edge_count += int(len(boundary_edges))
        side_first = np.column_stack(
            (
                boundary_edges[:, 0],
                boundary_edges[:, 0] + vertex_count,
                boundary_edges[:, 1] + vertex_count,
            )
        )
        side_second = np.column_stack(
            (
                boundary_edges[:, 0],
                boundary_edges[:, 1] + vertex_count,
                boundary_edges[:, 1],
            )
        )
        shell = trimesh.Trimesh(
            vertices=np.vstack((shifted_front, shifted_back)),
            faces=np.vstack(
                (
                    triangles,
                    back_triangles,
                    side_first,
                    side_second,
                )
            ),
            process=False,
        )
        if (
            not _defer_watertight_validation
            and not shell.is_watertight
        ):
            nonwatertight_shells += 1
        shells.append(shell)
    if not shells:
        raise ValueError("no front surface components were reconstructed")
    mesh = trimesh.util.concatenate(shells)
    return mesh, {
        "front_surface_components": int(len(front_components)),
        "boundary_edges": int(boundary_edge_count),
        "nonwatertight_shell_components": int(nonwatertight_shells),
    }


def _close_front_surface_components_normal(
    front_points: np.ndarray,
    front_triangles: np.ndarray,
    *,
    camera_pos: np.ndarray,
    camera_forward: np.ndarray,
    front_offset_mm: float,
    back_extrusion_mm: float,
) -> tuple[trimesh.Trimesh, Dict[str, int]]:
    """Close each visible patch along its camera-facing vertex normals."""
    front_mesh = trimesh.Trimesh(
        vertices=np.asarray(front_points, dtype=np.float64),
        faces=np.asarray(front_triangles, dtype=np.int64),
        process=False,
    )
    front_components = front_mesh.split(only_watertight=False)
    shells: list[trimesh.Trimesh] = []
    boundary_edge_count = 0
    nonwatertight_shells = 0
    for component in front_components:
        points = np.asarray(component.vertices, dtype=np.float64)
        triangles = _orient_front_triangles(
            np.asarray(component.faces, dtype=np.int64),
            points,
            camera_pos=camera_pos,
        )
        oriented = trimesh.Trimesh(
            vertices=points,
            faces=triangles,
            process=False,
        )
        normals = np.array(
            oriented.vertex_normals,
            dtype=np.float64,
            copy=True,
        )
        view = camera_pos.reshape(1, 3) - points
        flip = np.einsum("ij,ij->i", normals, view) < 0.0
        normals[flip] *= -1.0
        normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(normal_length, 1e-12)

        ray_vectors = points - camera_pos.reshape(1, 3)
        axial_depth = ray_vectors @ camera_forward.reshape(3)
        if np.any(axial_depth <= 1e-8):
            raise ValueError("front component contains points behind camera")
        front_scale = (
            1.0
            + float(front_offset_mm)
            / 1000.0
            / axial_depth
        )
        shifted_front = (
            camera_pos.reshape(1, 3)
            + ray_vectors * front_scale[:, None]
        )
        shifted_back = (
            shifted_front
            - normals * (float(back_extrusion_mm) / 1000.0)
        )

        vertex_count = len(points)
        back_triangles = triangles[:, [0, 2, 1]] + vertex_count
        boundary_edges = _boundary_directed_edges(triangles)
        boundary_edge_count += int(len(boundary_edges))
        side_first = np.column_stack(
            (
                boundary_edges[:, 0],
                boundary_edges[:, 0] + vertex_count,
                boundary_edges[:, 1] + vertex_count,
            )
        )
        side_second = np.column_stack(
            (
                boundary_edges[:, 0],
                boundary_edges[:, 1] + vertex_count,
                boundary_edges[:, 1],
            )
        )
        shell = trimesh.Trimesh(
            vertices=np.vstack((shifted_front, shifted_back)),
            faces=np.vstack(
                (
                    triangles,
                    back_triangles,
                    side_first,
                    side_second,
                )
            ),
            process=False,
        )
        if not shell.is_watertight:
            nonwatertight_shells += 1
        shells.append(shell)
    if not shells:
        raise ValueError("no front surface components were reconstructed")
    mesh = trimesh.util.concatenate(shells)
    return mesh, {
        "front_surface_components": int(len(front_components)),
        "boundary_edges": int(boundary_edge_count),
        "nonwatertight_shell_components": int(nonwatertight_shells),
    }


def _close_front_surface_components_adaptive(
    front_points: np.ndarray,
    front_triangles: np.ndarray,
    *,
    camera_pos: np.ndarray,
    camera_rotation: np.ndarray,
    camera_forward: np.ndarray,
    focal_px: float,
    image_width: int,
    image_height: int,
    pixel_stride: int,
    front_offset_mm: float,
    min_extrusion_mm: float,
    max_extrusion_mm: float,
    width_scale: float,
    prominence_mask: np.ndarray | None = None,
    prominence_radius_px: np.ndarray | None = None,
    prominence_radius_bias_px: float = 0.5,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Close visible patches with RGB-D silhouette-derived ray chords."""
    front_mesh = trimesh.Trimesh(
        vertices=np.asarray(front_points, dtype=np.float64),
        faces=np.asarray(front_triangles, dtype=np.int64),
        process=False,
    )
    front_components = front_mesh.split(only_watertight=False)
    shells: list[trimesh.Trimesh] = []
    boundary_edge_count = 0
    nonwatertight_shells = 0
    extrusion_values: list[np.ndarray] = []
    prominence_extrusion_values: list[np.ndarray] = []
    prominence_vertex_count = 0
    if (prominence_mask is None) != (prominence_radius_px is None):
        raise ValueError(
            "prominence mask and radius map must be provided together"
        )
    if prominence_mask is not None:
        prominence_mask = np.asarray(prominence_mask, dtype=bool)
        prominence_radius_px = np.asarray(
            prominence_radius_px,
            dtype=np.float64,
        )
        expected_shape = (int(image_height), int(image_width))
        if (
            prominence_mask.shape != expected_shape
            or prominence_radius_px.shape != expected_shape
        ):
            raise ValueError("local width maps do not match the RGB-D image")
        if float(prominence_radius_bias_px) < 0.0:
            raise ValueError("prominence radius bias must be nonnegative")
    for component in front_components:
        points = np.asarray(component.vertices, dtype=np.float64)
        triangles = _orient_front_triangles(
            np.asarray(component.faces, dtype=np.int64),
            points,
            camera_pos=camera_pos,
        )
        camera_points = (
            points - camera_pos.reshape(1, 3)
        ) @ camera_rotation
        depth = -camera_points[:, 2]
        u = (
            float(image_width) * 0.5
            + float(focal_px) * camera_points[:, 0] / depth
        )
        v = (
            float(image_height) * 0.5
            - float(focal_px) * camera_points[:, 1] / depth
        )
        sampled_u = np.rint(u / float(pixel_stride)).astype(np.int64)
        sampled_v = np.rint(v / float(pixel_stride)).astype(np.int64)
        u0, u1 = int(sampled_u.min()), int(sampled_u.max())
        v0, v1 = int(sampled_v.min()), int(sampled_v.max())
        mask = np.zeros(
            (v1 - v0 + 3, u1 - u0 + 3),
            dtype=bool,
        )
        local_u = sampled_u - u0 + 1
        local_v = sampled_v - v0 + 1
        mask[local_v, local_u] = True
        distance_samples = ndimage.distance_transform_edt(mask)
        local_radius_samples = distance_samples[local_v, local_u]
        visible_diameter_m = (
            2.0
            * local_radius_samples
            * float(pixel_stride)
            * depth
            / float(focal_px)
        )
        if prominence_mask is not None:
            full_u = np.clip(
                np.rint(u).astype(np.int64),
                0,
                int(image_width) - 1,
            )
            full_v = np.clip(
                np.rint(v).astype(np.int64),
                0,
                int(image_height) - 1,
            )
            use_prominence = prominence_mask[full_v, full_u]
            prominence_radius = prominence_radius_px[full_v, full_u]
            prominence_diameter_m = (
                2.0
                * (
                    prominence_radius
                    + float(prominence_radius_bias_px)
                )
                * depth
                / float(focal_px)
            )
            visible_diameter_m[use_prominence] = np.minimum(
                visible_diameter_m[use_prominence],
                prominence_diameter_m[use_prominence],
            )
            prominence_vertex_count += int(use_prominence.sum())
        extrusion_mm = np.clip(
            float(width_scale) * visible_diameter_m * 1000.0,
            float(min_extrusion_mm),
            float(max_extrusion_mm),
        )
        extrusion_values.append(extrusion_mm)
        if prominence_mask is not None and use_prominence.any():
            prominence_extrusion_values.append(
                extrusion_mm[use_prominence]
            )

        ray_vectors = points - camera_pos.reshape(1, 3)
        axial_depth = ray_vectors @ camera_forward.reshape(3)
        if np.any(axial_depth <= 1e-8):
            raise ValueError("front component contains points behind camera")
        front_scale = (
            1.0
            + float(front_offset_mm)
            / 1000.0
            / axial_depth
        )
        back_scale = (
            1.0
            + (
                float(front_offset_mm) + extrusion_mm
            )
            / 1000.0
            / axial_depth
        )
        shifted_front = (
            camera_pos.reshape(1, 3)
            + ray_vectors * front_scale[:, None]
        )
        shifted_back = (
            camera_pos.reshape(1, 3)
            + ray_vectors * back_scale[:, None]
        )

        vertex_count = len(points)
        back_triangles = triangles[:, [0, 2, 1]] + vertex_count
        boundary_edges = _boundary_directed_edges(triangles)
        boundary_edge_count += int(len(boundary_edges))
        side_first = np.column_stack(
            (
                boundary_edges[:, 0],
                boundary_edges[:, 0] + vertex_count,
                boundary_edges[:, 1] + vertex_count,
            )
        )
        side_second = np.column_stack(
            (
                boundary_edges[:, 0],
                boundary_edges[:, 1] + vertex_count,
                boundary_edges[:, 1],
            )
        )
        shell = trimesh.Trimesh(
            vertices=np.vstack((shifted_front, shifted_back)),
            faces=np.vstack(
                (
                    triangles,
                    back_triangles,
                    side_first,
                    side_second,
                )
            ),
            process=False,
        )
        if not shell.is_watertight:
            nonwatertight_shells += 1
        shells.append(shell)
    if not shells:
        raise ValueError("no front surface components were reconstructed")
    mesh = trimesh.util.concatenate(shells)
    all_extrusion = np.concatenate(extrusion_values)
    quantiles = np.percentile(
        all_extrusion,
        [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0],
    )
    metadata: Dict[str, Any] = {
        "front_surface_components": int(len(front_components)),
        "boundary_edges": int(boundary_edge_count),
        "nonwatertight_shell_components": int(nonwatertight_shells),
        "adaptive_extrusion_mm_quantiles": {
            key: float(value)
            for key, value in zip(
                ("0", "10", "25", "50", "75", "90", "100"),
                quantiles,
            )
        },
    }
    if prominence_mask is not None:
        metadata["local_width_prominence_vertex_count"] = int(
            prominence_vertex_count
        )
        if prominence_extrusion_values:
            prominent = np.concatenate(prominence_extrusion_values)
            prominent_quantiles = np.percentile(
                prominent,
                [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0],
            )
            metadata["local_width_extrusion_mm_quantiles"] = {
                key: float(value)
                for key, value in zip(
                    ("0", "10", "25", "50", "75", "90", "100"),
                    prominent_quantiles,
                )
            }
    return mesh, metadata


def _topology_extrusion(
    *,
    sufficiently_large: bool,
    is_concave: bool,
    is_convex: bool,
    visible_diameter_mm: float,
    default_extrusion_mm: float,
    concave_wall_mm: float,
    convex_width_scale: float,
    max_extrusion_mm: float,
) -> tuple[str, float]:
    """Select a closed-surface thickness from observable local geometry."""
    if is_concave:
        return "concave_thin_wall", float(concave_wall_mm)
    if is_convex:
        return (
            "convex_silhouette_chord",
            float(
                np.clip(
                    float(convex_width_scale) * visible_diameter_mm,
                    float(default_extrusion_mm),
                    float(max_extrusion_mm),
                )
            ),
        )
    if (
        sufficiently_large
        and visible_diameter_mm < float(default_extrusion_mm)
    ):
        return (
            "narrow_silhouette_chord",
            max(1.0, float(visible_diameter_mm)),
        )
    return "neutral_default_shell", float(default_extrusion_mm)


def _close_front_surface_components_topology(
    front_points: np.ndarray,
    front_triangles: np.ndarray,
    *,
    camera_pos: np.ndarray,
    camera_rotation: np.ndarray,
    camera_forward: np.ndarray,
    focal_px: float,
    image_width: int,
    image_height: int,
    pixel_stride: int,
    front_offset_mm: float,
    default_extrusion_mm: float,
    concave_wall_mm: float,
    concavity_depth_mm: float,
    convex_prominence_mm: float,
    min_nonplanarity: float,
    convex_width_scale: float,
    max_extrusion_mm: float,
    min_component_vertices: int,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Close components using visible concavity and silhouette scale.

    The rule is scene-agnostic. A nonplanar patch whose core is farther from
    the camera than its boundary is treated as a visible cavity wall. A
    nonplanar patch whose core is nearer is treated as a convex solid and gets
    a silhouette-derived chord thickness. Planar and weakly classified
    patches retain the conservative default ray shell.
    """
    front_mesh = trimesh.Trimesh(
        vertices=np.asarray(front_points, dtype=np.float64),
        faces=np.asarray(front_triangles, dtype=np.int64),
        process=False,
    )
    front_components = front_mesh.split(only_watertight=False)
    shells: list[trimesh.Trimesh] = []
    component_rows: list[Dict[str, Any]] = []
    boundary_edge_count = 0
    nonwatertight_shells = 0
    selected_extrusions: list[float] = []
    for component_index, component in enumerate(front_components):
        points = np.asarray(component.vertices, dtype=np.float64)
        triangles = _orient_front_triangles(
            np.asarray(component.faces, dtype=np.int64),
            points,
            camera_pos=camera_pos,
        )
        camera_points = (
            points - camera_pos.reshape(1, 3)
        ) @ camera_rotation
        depth = -camera_points[:, 2]
        if np.any(depth <= 1e-8):
            raise ValueError("front component contains points behind camera")
        u = (
            float(image_width) * 0.5
            + float(focal_px) * camera_points[:, 0] / depth
        )
        v = (
            float(image_height) * 0.5
            - float(focal_px) * camera_points[:, 1] / depth
        )
        sampled_u = np.rint(u / float(pixel_stride)).astype(np.int64)
        sampled_v = np.rint(v / float(pixel_stride)).astype(np.int64)
        u0, u1 = int(sampled_u.min()), int(sampled_u.max())
        v0, v1 = int(sampled_v.min()), int(sampled_v.max())
        component_mask = np.zeros(
            (v1 - v0 + 3, u1 - u0 + 3),
            dtype=bool,
        )
        local_u = sampled_u - u0 + 1
        local_v = sampled_v - v0 + 1
        component_mask[local_v, local_u] = True
        distance_to_boundary = ndimage.distance_transform_edt(
            component_mask
        )
        radius_samples = distance_to_boundary[local_v, local_u]
        maximum_radius_samples = float(
            radius_samples.max(initial=0.0)
        )
        core_threshold = max(1.5, 0.5 * maximum_radius_samples)
        boundary_vertices = radius_samples <= 1.5
        core_vertices = radius_samples >= core_threshold
        boundary_depth = (
            float(np.median(depth[boundary_vertices]))
            if boundary_vertices.any()
            else float(np.median(depth))
        )
        core_depth = (
            float(np.median(depth[core_vertices]))
            if core_vertices.any()
            else float(np.median(depth))
        )
        core_minus_boundary_mm = (
            core_depth - boundary_depth
        ) * 1000.0

        centered = points - points.mean(axis=0)
        covariance = centered.T @ centered / max(len(points), 1)
        eigenvalues = np.maximum(
            np.linalg.eigvalsh(covariance),
            0.0,
        )
        nonplanarity = float(
            eigenvalues[0] / max(float(eigenvalues.sum()), 1e-12)
        )
        visible_diameter_mm = (
            2.0
            * maximum_radius_samples
            * float(pixel_stride)
            * float(np.median(depth))
            / float(focal_px)
            * 1000.0
        )
        sufficiently_large = (
            len(points) >= int(min_component_vertices)
            and maximum_radius_samples >= 2.0
        )
        is_concave = bool(
            sufficiently_large
            and nonplanarity >= float(min_nonplanarity)
            and core_minus_boundary_mm >= float(concavity_depth_mm)
        )
        is_convex = bool(
            sufficiently_large
            and nonplanarity >= float(min_nonplanarity)
            and core_minus_boundary_mm <= -float(convex_prominence_mm)
        )
        component_class, extrusion_mm = _topology_extrusion(
            sufficiently_large=sufficiently_large,
            is_concave=is_concave,
            is_convex=is_convex,
            visible_diameter_mm=visible_diameter_mm,
            default_extrusion_mm=float(default_extrusion_mm),
            concave_wall_mm=float(concave_wall_mm),
            convex_width_scale=float(convex_width_scale),
            max_extrusion_mm=float(max_extrusion_mm),
        )
        selected_extrusions.append(extrusion_mm)

        ray_vectors = points - camera_pos.reshape(1, 3)
        axial_depth = ray_vectors @ camera_forward.reshape(3)
        if np.any(axial_depth <= 1e-8):
            raise ValueError("front component contains points behind camera")
        front_scale = (
            1.0
            + float(front_offset_mm) / 1000.0 / axial_depth
        )
        back_scale = (
            1.0
            + (
                float(front_offset_mm) + extrusion_mm
            )
            / 1000.0
            / axial_depth
        )
        shifted_front = (
            camera_pos.reshape(1, 3)
            + ray_vectors * front_scale[:, None]
        )
        shifted_back = (
            camera_pos.reshape(1, 3)
            + ray_vectors * back_scale[:, None]
        )

        vertex_count = len(points)
        back_triangles = triangles[:, [0, 2, 1]] + vertex_count
        boundary_edges = _boundary_directed_edges(triangles)
        boundary_edge_count += int(len(boundary_edges))
        side_first = np.column_stack(
            (
                boundary_edges[:, 0],
                boundary_edges[:, 0] + vertex_count,
                boundary_edges[:, 1] + vertex_count,
            )
        )
        side_second = np.column_stack(
            (
                boundary_edges[:, 0],
                boundary_edges[:, 1] + vertex_count,
                boundary_edges[:, 1],
            )
        )
        shell = trimesh.Trimesh(
            vertices=np.vstack((shifted_front, shifted_back)),
            faces=np.vstack(
                (
                    triangles,
                    back_triangles,
                    side_first,
                    side_second,
                )
            ),
            process=False,
        )
        if not shell.is_watertight:
            nonwatertight_shells += 1
        shells.append(shell)
        component_rows.append(
            {
                "component_index": int(component_index),
                "vertex_count": int(len(points)),
                "bbox_uv": [
                    float(u.min()),
                    float(v.min()),
                    float(u.max()),
                    float(v.max()),
                ],
                "world_bounds_m": [
                    points.min(axis=0).tolist(),
                    points.max(axis=0).tolist(),
                ],
                "nonplanarity": nonplanarity,
                "mask_radius_sample_px": maximum_radius_samples,
                "visible_diameter_mm": visible_diameter_mm,
                "core_minus_boundary_depth_mm": (
                    core_minus_boundary_mm
                ),
                "classification": component_class,
                "extrusion_mm": extrusion_mm,
            }
        )
    if not shells:
        raise ValueError("no front surface components were reconstructed")
    mesh = trimesh.util.concatenate(shells)
    return mesh, {
        "front_surface_components": int(len(front_components)),
        "boundary_edges": int(boundary_edge_count),
        "nonwatertight_shell_components": int(nonwatertight_shells),
        "topology_component_classes": {
            name: int(
                sum(
                    row["classification"] == name
                    for row in component_rows
                )
            )
            for name in (
                "concave_thin_wall",
                "convex_silhouette_chord",
                "narrow_silhouette_chord",
                "neutral_default_shell",
            )
        },
        "topology_extrusion_mm_quantiles": {
            key: float(value)
            for key, value in zip(
                ("0", "10", "25", "50", "75", "90", "100"),
                np.percentile(
                    np.asarray(selected_extrusions, dtype=np.float64),
                    [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0],
                ),
            )
        },
        "topology_components": component_rows,
    }


def _robust_circle_center_xy(
    points_xy: np.ndarray,
) -> tuple[np.ndarray, float]:
    values = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(values) < 12:
        raise ValueError("circle fit requires at least 12 points")
    keep = np.ones(len(values), dtype=bool)
    center = np.median(values, axis=0)
    radius = float(np.median(np.linalg.norm(values - center, axis=1)))
    for _ in range(5):
        sample = values[keep]
        design = np.column_stack(
            (
                2.0 * sample[:, 0],
                2.0 * sample[:, 1],
                np.ones(len(sample)),
            )
        )
        rhs = np.einsum("ij,ij->i", sample, sample)
        solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        center = solution[:2]
        radius = float(
            np.sqrt(max(float(solution[2] + center @ center), 0.0))
        )
        residual = np.abs(
            np.linalg.norm(values - center.reshape(1, 2), axis=1)
            - radius
        )
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        limit = median + max(2.5 * 1.4826 * mad, 0.0015)
        next_keep = residual <= limit
        if int(next_keep.sum()) < 12 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    return center, radius


def _radial_mode(
    values: np.ndarray,
    *,
    bin_m: float,
    maximum_m: float,
) -> float:
    radius = np.asarray(values, dtype=np.float64)
    radius = radius[
        np.isfinite(radius)
        & (radius >= 0.0)
        & (radius <= float(maximum_m))
    ]
    if len(radius) < 5:
        return float("nan")
    edges = np.arange(
        0.0,
        float(maximum_m) + 2.0 * float(bin_m),
        float(bin_m),
    )
    histogram, edges = np.histogram(radius, bins=edges)
    histogram = ndimage.gaussian_filter1d(
        histogram.astype(np.float64),
        sigma=1.0,
        mode="nearest",
    )
    index = int(np.argmax(histogram))
    return float(0.5 * (edges[index] + edges[index + 1]))


def _axisymmetric_cavity_mesh(
    candidate_points: np.ndarray,
    full_world: np.ndarray,
    full_valid: np.ndarray,
    *,
    base_wall_scale: float,
    bottom_wall_boost_scale: float,
    wall_decay_scale: float,
    bottom_start_scale: float,
    disk_start_scale: float,
    lower_wall_slope: float,
    top_height_scale: float,
    group_radius_scale: float,
    visible_shell_scale: float,
    minimum_shell_mm: float,
    maximum_shell_mm: float,
    radial_bin_mm: float,
    height_bin_mm: float,
    revolve_sections: int,
) -> tuple[trimesh.Trimesh, np.ndarray, Dict[str, Any]]:
    seed = np.asarray(candidate_points, dtype=np.float64).reshape(-1, 3)
    if len(seed) < 100:
        raise ValueError("cavity candidate has insufficient points")
    upper = seed[seed[:, 2] >= np.percentile(seed[:, 2], 75.0)]
    center_xy, _seed_radius = _robust_circle_center_xy(upper[:, :2])
    seed_radius = np.linalg.norm(
        seed[:, :2] - center_xy.reshape(1, 2),
        axis=1,
    )
    initial_scale = float(np.percentile(seed_radius, 90.0))
    if initial_scale <= 0.02:
        raise ValueError("cavity candidate is too small")

    world = np.asarray(full_world, dtype=np.float64)
    valid = np.asarray(full_valid, dtype=bool)
    radius = np.linalg.norm(
        world[..., :2] - center_xy.reshape(1, 1, 2),
        axis=-1,
    )
    seed_z_min = float(np.percentile(seed[:, 2], 1.0))
    seed_z_max = float(np.percentile(seed[:, 2], 99.0))
    preliminary = (
        valid
        & (radius <= float(group_radius_scale) * initial_scale)
        & (
            world[..., 2]
            >= seed_z_min - float(bottom_start_scale) * initial_scale
        )
        & (
            world[..., 2]
            <= seed_z_max + 0.4 * initial_scale
        )
    )
    points = world[preliminary]
    if len(points) < 500:
        raise ValueError("cavity group has insufficient RGB-D support")
    upper = points[points[:, 2] >= np.percentile(points[:, 2], 88.0)]
    center_xy, _top_radius = _robust_circle_center_xy(upper[:, :2])
    radius_points = np.linalg.norm(
        points[:, :2] - center_xy.reshape(1, 2),
        axis=1,
    )
    scale = float(np.percentile(radius_points, 90.0))
    if not 0.02 <= scale <= 0.5:
        raise ValueError(f"invalid cavity radial scale {scale}")

    central = radius_points <= max(0.25 * scale, 0.01)
    central_z = points[central, 2]
    if len(central_z) < 20:
        raise ValueError("cavity floor has insufficient central support")
    height_bin_m = float(height_bin_mm) / 1000.0
    radial_bin_m = float(radial_bin_mm) / 1000.0
    floor_edges = np.arange(
        float(np.percentile(central_z, 0.5)),
        float(np.percentile(central_z, 99.5)) + height_bin_m,
        height_bin_m,
    )
    floor_histogram, floor_edges = np.histogram(
        central_z,
        bins=floor_edges,
    )
    floor_index = int(np.argmax(floor_histogram))
    floor_members = central_z[
        (central_z >= floor_edges[floor_index])
        & (
            central_z
            <= floor_edges[floor_index + 1]
        )
    ]
    floor_z = float(
        np.median(floor_members)
        if len(floor_members)
        else 0.5
        * (floor_edges[floor_index] + floor_edges[floor_index + 1])
    )
    bottom_z = floor_z - float(bottom_start_scale) * scale
    top_z = floor_z + float(top_height_scale) * scale
    group_radius = float(group_radius_scale) * scale
    radius = np.linalg.norm(
        world[..., :2] - center_xy.reshape(1, 1, 2),
        axis=-1,
    )
    group_mask = (
        valid
        & (radius <= group_radius)
        & (world[..., 2] >= bottom_z)
        & (world[..., 2] <= top_z + 0.15 * scale)
    )
    points = world[group_mask]
    radius_points = np.linalg.norm(
        points[:, :2] - center_xy.reshape(1, 2),
        axis=1,
    )
    profile_limit = 1.1 * scale
    z_grid = np.arange(
        floor_z,
        top_z + height_bin_m,
        height_bin_m,
    )
    inner = np.full(len(z_grid), np.nan, dtype=np.float64)
    for index, z_value in enumerate(z_grid):
        selected = (
            np.abs(points[:, 2] - z_value) <= 0.6 * height_bin_m
        )
        selected &= radius_points <= profile_limit
        if int(selected.sum()) >= 8:
            inner[index] = _radial_mode(
                radius_points[selected],
                bin_m=radial_bin_m,
                maximum_m=profile_limit,
            )
    finite = np.isfinite(inner)
    if int(finite.sum()) < 8:
        raise ValueError("cavity radial profile is underconstrained")
    inner = np.interp(z_grid, z_grid[finite], inner[finite])
    inner = ndimage.median_filter(inner, size=5, mode="nearest")
    inner = ndimage.gaussian_filter1d(
        inner,
        sigma=1.0,
        mode="nearest",
    )
    radial_scale = float(np.percentile(inner, 90.0))
    base_wall = float(base_wall_scale) * radial_scale
    bottom_boost = float(bottom_wall_boost_scale) * radial_scale
    decay = max(float(wall_decay_scale) * radial_scale, 1e-4)
    outer = inner + base_wall + bottom_boost * np.exp(
        -(z_grid - floor_z) / decay
    )
    outer_floor = float(inner[0] + base_wall + bottom_boost)
    disk_z = floor_z - float(disk_start_scale) * radial_scale
    bottom_z = floor_z - float(bottom_start_scale) * radial_scale
    profile = np.vstack(
        (
            np.array(
                [
                    [outer_floor, bottom_z],
                    [outer_floor, disk_z],
                    [outer_floor, floor_z],
                ]
            ),
            np.column_stack((outer, z_grid)),
            np.array([[inner[-1], top_z]]),
            np.column_stack((inner[::-1], z_grid[::-1])),
            np.array(
                [
                    [0.0, floor_z],
                    [0.0, disk_z],
                    [
                        outer_floor * (1.0 - float(lower_wall_slope)),
                        disk_z,
                    ],
                    [outer_floor, bottom_z],
                    [outer_floor, bottom_z],
                ]
            ),
        )
    )
    mesh = trimesh.creation.revolve(
        profile,
        sections=int(revolve_sections),
        process=True,
    )
    mesh.apply_translation([center_xy[0], center_xy[1], 0.0])
    if not mesh.is_watertight:
        raise ValueError("axisymmetric cavity mesh is not watertight")
    shell_mm = float(
        np.clip(
            float(visible_shell_scale) * radial_scale * 1000.0,
            float(minimum_shell_mm),
            float(maximum_shell_mm),
        )
    )
    return mesh, group_mask, {
        "center_xy": center_xy.tolist(),
        "radial_scale_mm": radial_scale * 1000.0,
        "floor_z_m": floor_z,
        "bottom_z_m": bottom_z,
        "disk_z_m": disk_z,
        "top_z_m": top_z,
        "group_radius_mm": group_radius * 1000.0,
        "visible_shell_mm": shell_mm,
        "base_wall_mm": base_wall * 1000.0,
        "bottom_wall_boost_mm": bottom_boost * 1000.0,
        "wall_decay_mm": decay * 1000.0,
        "maximum_outer_radius_mm": float(np.max(outer) * 1000.0),
        "group_pixels": int(group_mask.sum()),
        "profile_samples": int(len(z_grid)),
    }


def _belongs_to_axisymmetric_cavity_group(
    *,
    inside_fraction: float,
    median_radius_m: float,
    modeled_outer_radius_m: float,
    minimum_inside_fraction: float = 0.35,
    maximum_median_radius_scale: float = 1.03,
) -> bool:
    """Keep only surfaces represented by the rotational cavity profile."""
    return bool(
        float(inside_fraction) >= float(minimum_inside_fraction)
        and float(median_radius_m)
        <= float(maximum_median_radius_scale)
        * float(modeled_outer_radius_m)
    )


def _masked_full_resolution_shell(
    world: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    camera_pos: np.ndarray,
    camera_forward: np.ndarray,
    focal_px: float,
    surface_edge_scale: float,
    surface_edge_slack_m: float,
    max_depth_jump_m: float,
    back_extrusion_mm: float,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    points_grid = np.asarray(world, dtype=np.float64)
    metric_depth = np.asarray(depth, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    height, width = selected.shape
    ids = np.arange(height * width, dtype=np.int64).reshape(height, width)
    horizontal = (
        selected[:, :-1]
        & selected[:, 1:]
        & _continuous_edge(
            points_grid[:, :-1],
            points_grid[:, 1:],
            metric_depth[:, :-1],
            metric_depth[:, 1:],
            pixel_span=1.0,
            focal_px=float(focal_px),
            surface_edge_scale=float(surface_edge_scale),
            surface_edge_slack_m=float(surface_edge_slack_m),
            max_depth_jump_m=float(max_depth_jump_m),
        )
    )
    vertical = (
        selected[:-1, :]
        & selected[1:, :]
        & _continuous_edge(
            points_grid[:-1, :],
            points_grid[1:, :],
            metric_depth[:-1, :],
            metric_depth[1:, :],
            pixel_span=1.0,
            focal_px=float(focal_px),
            surface_edge_scale=float(surface_edge_scale),
            surface_edge_slack_m=float(surface_edge_slack_m),
            max_depth_jump_m=float(max_depth_jump_m),
        )
    )
    diagonal = (
        selected[:-1, :-1]
        & selected[1:, 1:]
        & _continuous_edge(
            points_grid[:-1, :-1],
            points_grid[1:, 1:],
            metric_depth[:-1, :-1],
            metric_depth[1:, 1:],
            pixel_span=np.sqrt(2.0),
            focal_px=float(focal_px),
            surface_edge_scale=float(surface_edge_scale),
            surface_edge_slack_m=float(surface_edge_slack_m),
            max_depth_jump_m=float(max_depth_jump_m),
        )
    )
    diagonal_up = (
        selected[1:, :-1]
        & selected[:-1, 1:]
        & _continuous_edge(
            points_grid[1:, :-1],
            points_grid[:-1, 1:],
            metric_depth[1:, :-1],
            metric_depth[:-1, 1:],
            pixel_span=np.sqrt(2.0),
            focal_px=float(focal_px),
            surface_edge_scale=float(surface_edge_scale),
            surface_edge_slack_m=float(surface_edge_slack_m),
            max_depth_jump_m=float(max_depth_jump_m),
        )
    )
    top_left = ids[:-1, :-1]
    top_right = ids[:-1, 1:]
    bottom_left = ids[1:, :-1]
    bottom_right = ids[1:, 1:]
    rows: list[np.ndarray] = []
    first_down = horizontal[:-1, :] & vertical[:, :-1] & diagonal
    second_down = horizontal[1:, :] & vertical[:, 1:] & diagonal
    first_up = horizontal[:-1, :] & vertical[:, 1:] & diagonal_up
    second_up = horizontal[1:, :] & vertical[:, :-1] & diagonal_up
    use_down = (
        (first_down.astype(np.int8) + second_down.astype(np.int8))
        >= (first_up.astype(np.int8) + second_up.astype(np.int8))
    )
    first = use_down & first_down
    if first.any():
        rows.append(
            np.column_stack(
                (top_left[first], bottom_right[first], top_right[first])
            )
        )
    second = use_down & second_down
    if second.any():
        rows.append(
            np.column_stack(
                (
                    top_left[second],
                    bottom_left[second],
                    bottom_right[second],
                )
            )
        )
    first = (~use_down) & first_up
    if first.any():
        rows.append(
            np.column_stack(
                (top_left[first], bottom_left[first], top_right[first])
            )
        )
    second = (~use_down) & second_up
    if second.any():
        rows.append(
            np.column_stack(
                (
                    top_right[second],
                    bottom_left[second],
                    bottom_right[second],
                )
            )
        )
    if not rows:
        raise ValueError("cavity group has no continuous RGB-D triangles")
    triangles = np.vstack(rows)
    flat_points = points_grid.reshape(-1, 3)
    used = np.unique(triangles)
    remap = np.full(len(flat_points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    points = flat_points[used]
    triangles = remap[triangles]
    triangles = _orient_front_triangles(
        triangles,
        points,
        camera_pos=camera_pos,
    )
    points, triangles, split_metadata = _split_nonmanifold_vertex_fans(
        points,
        triangles,
    )
    shell, shell_metadata = _close_front_surface_components(
        points,
        triangles,
        camera_pos=camera_pos,
        camera_forward=camera_forward,
        front_offset_mm=0.0,
        back_extrusion_mm=float(back_extrusion_mm),
    )
    return shell, {
        **split_metadata,
        **shell_metadata,
        "full_resolution_shell_pixels": int(selected.sum()),
        "full_resolution_shell_triangles": int(len(triangles)),
        "full_resolution_shell_mode": "camera_forward",
    }


def _compact_symmetric_component_mesh(
    points: np.ndarray,
    component_mask: np.ndarray,
    *,
    camera_pos: np.ndarray,
    support_z_m: float,
    pitch_mm: float,
    symmetry_center_percentile: float,
    support_clearance_scale: float,
    closing_radius_scale: float,
    front_padding_mm: float,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Build a conservative support-aligned mirror completion."""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(component_mask, dtype=bool)
    pitch_m = float(pitch_mm) / 1000.0
    horizontal = values[:, :2]
    origin_xy = horizontal.mean(axis=0)
    covariance_xy = np.cov((horizontal - origin_xy).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_xy)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    minor = np.array([-major[1], major[0]], dtype=np.float64)
    camera_xy = np.asarray(camera_pos, dtype=np.float64)[:2]
    if float(minor @ (camera_xy - origin_xy)) < 0.0:
        minor *= -1.0

    centered_xy = horizontal - origin_xy.reshape(1, 2)
    major_coordinate = centered_xy @ major
    minor_coordinate = centered_xy @ minor
    height_coordinate = values[:, 2]
    major_extent = float(np.ptp(major_coordinate))
    height_extent = float(np.ptp(height_coordinate))
    margin_m = max(4.0 * pitch_m, 0.02 * major_extent)
    major_min = float(
        np.floor(
            (major_coordinate.min() - margin_m) / pitch_m
        )
        * pitch_m
    )
    support_clearance_m = float(
        np.clip(
            float(support_clearance_scale) * height_extent,
            0.003,
            0.04,
        )
    )
    height_min = float(
        np.floor(
            (
                float(support_z_m)
                + support_clearance_m
                - 2.0 * pitch_m
            )
            / pitch_m
        )
        * pitch_m
    )
    major_count = int(
        np.ceil(
            (
                major_coordinate.max()
                + margin_m
                - major_min
            )
            / pitch_m
        )
    ) + 1
    height_count = int(
        np.ceil(
            (
                height_coordinate.max()
                + margin_m
                - height_min
            )
            / pitch_m
        )
    ) + 1
    major_index = np.rint(
        (major_coordinate - major_min) / pitch_m
    ).astype(np.int64)
    height_index = np.rint(
        (height_coordinate - height_min) / pitch_m
    ).astype(np.int64)
    silhouette = np.zeros(
        (major_count, height_count),
        dtype=bool,
    )
    silhouette[major_index, height_index] = True
    front = np.full(
        silhouette.shape,
        -np.inf,
        dtype=np.float64,
    )
    np.maximum.at(
        front,
        (major_index, height_index),
        minor_coordinate,
    )

    closing_iterations = max(
        1,
        int(
            round(
                float(closing_radius_scale)
                * major_extent
                / pitch_m
            )
        ),
    )
    silhouette = ndimage.binary_closing(
        silhouette,
        structure=np.ones((3, 3), dtype=bool),
        iterations=closing_iterations,
    )
    silhouette = ndimage.binary_fill_holes(silhouette)
    support_index = int(
        round(
            (
                float(support_z_m)
                + support_clearance_m
                - height_min
            )
            / pitch_m
        )
    )
    for column in np.flatnonzero(silhouette.any(axis=1)):
        occupied = np.flatnonzero(silhouette[column])
        if len(occupied):
            silhouette[
                column,
                max(0, support_index) : occupied.min() + 1,
            ] = True

    known_front = np.isfinite(front)
    if int(known_front.sum()) < 100:
        raise ValueError("compact symmetry front surface is underconstrained")
    nearest_front = ndimage.distance_transform_edt(
        ~known_front,
        return_distances=False,
        return_indices=True,
    )
    front = front[tuple(nearest_front)]
    front = ndimage.gaussian_filter(
        front,
        sigma=1.0,
        mode="nearest",
    )
    symmetry_center = float(
        np.percentile(
            minor_coordinate,
            float(symmetry_center_percentile),
        )
    )
    back = 2.0 * symmetry_center - front
    selected_back = back[silhouette]
    selected_front = front[silhouette]
    minor_min = float(
        np.floor(
            (selected_back.min() - 2.0 * pitch_m) / pitch_m
        )
        * pitch_m
    )
    minor_max = float(
        selected_front.max()
        + float(front_padding_mm) / 1000.0
        + 2.0 * pitch_m
    )
    minor_count = int(
        np.ceil((minor_max - minor_min) / pitch_m)
    ) + 1
    start_minor = np.ceil(
        (selected_back - minor_min) / pitch_m
    ).astype(np.int64)
    stop_minor = np.floor(
        (
            selected_front
            + float(front_padding_mm) / 1000.0
            - minor_min
        )
        / pitch_m
    ).astype(np.int64)
    silhouette_major, silhouette_height = np.nonzero(silhouette)
    valid_columns = stop_minor >= start_minor
    silhouette_major = silhouette_major[valid_columns]
    silhouette_height = silhouette_height[valid_columns]
    start_minor = np.clip(
        start_minor[valid_columns],
        0,
        minor_count - 1,
    )
    stop_minor = np.clip(
        stop_minor[valid_columns],
        0,
        minor_count - 1,
    )
    difference = np.zeros(
        (major_count, minor_count + 1, height_count),
        dtype=np.int16,
    )
    np.add.at(
        difference,
        (
            silhouette_major,
            start_minor,
            silhouette_height,
        ),
        1,
    )
    can_stop = stop_minor + 1 < difference.shape[1]
    np.add.at(
        difference,
        (
            silhouette_major[can_stop],
            stop_minor[can_stop] + 1,
            silhouette_height[can_stop],
        ),
        -1,
    )
    volume = np.cumsum(difference, axis=1) > 0
    if int(volume.sum()) < 100:
        raise ValueError("compact symmetry volume is empty")
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
        volume,
        pitch=pitch_m,
    )
    local_rotation = np.array(
        [
            [major[0], minor[0], 0.0],
            [major[1], minor[1], 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    local_origin = np.array(
        [
            origin_xy[0]
            + major[0] * major_min
            + minor[0] * minor_min,
            origin_xy[1]
            + major[1] * major_min
            + minor[1] * minor_min,
            height_min,
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = local_rotation
    transform[:3, 3] = local_origin
    mesh.apply_transform(transform)
    mesh.process(validate=True)
    trimesh.repair.fix_normals(mesh)
    if float(mesh.volume) < 0.0:
        mesh.invert()
    if not mesh.is_watertight:
        raise ValueError("compact symmetry mesh is not watertight")
    return mesh, {
        "source_pixels": int(mask.sum()),
        "major_extent_mm": major_extent * 1000.0,
        "height_extent_mm": height_extent * 1000.0,
        "support_z_m": float(support_z_m),
        "support_clearance_mm": support_clearance_m * 1000.0,
        "symmetry_center_percentile": float(
            symmetry_center_percentile
        ),
        "symmetry_center_mm": symmetry_center * 1000.0,
        "observed_minor_mm": [
            float(minor_coordinate.min() * 1000.0),
            float(minor_coordinate.max() * 1000.0),
        ],
        "closing_iterations": int(closing_iterations),
        "grid_shape": [
            int(major_count),
            int(minor_count),
            int(height_count),
        ],
        "occupied_voxels": int(volume.sum()),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
    }


def _compact_supported_symmetric_meshes(
    rgb: np.ndarray,
    depth: np.ndarray,
    world: np.ndarray,
    valid: np.ndarray,
    *,
    camera_pos: np.ndarray,
    focal_px: float,
    minimum_component_pixels: int,
    maximum_extent_mm: float,
    minimum_height_mm: float,
    minimum_nonplanarity: float,
    minimum_convex_prominence_mm: float,
    maximum_support_gap_mm: float,
    pitch_mm: float,
    symmetry_center_percentile: float,
    support_clearance_scale: float,
    closing_radius_scale: float,
    front_padding_mm: float,
) -> tuple[trimesh.Trimesh | None, Dict[str, Any]]:
    """Find and complete all compact convex RGB-D components."""
    from .rgbd_scene_components import (
        _continuous_component_labels,
        _detect_horizontal_regions,
    )

    color = np.asarray(rgb)
    metric_depth = np.asarray(depth, dtype=np.float64)
    full_world = np.asarray(world, dtype=np.float64)
    full_valid = np.asarray(valid, dtype=bool)
    plane_labels, planes = _detect_horizontal_regions(
        full_world,
        full_valid,
        bin_m=0.002,
        band_m=0.003,
        min_pixels=1200,
        min_separation_m=0.012,
    )
    component_labels, component_sizes = _continuous_component_labels(
        full_world,
        metric_depth,
        color,
        full_valid & (plane_labels < 0),
        focal_px=float(focal_px),
        edge_scale=1.8,
        edge_slack_m=0.002,
        edge_depth_jump_m=0.03,
        edge_color_distance=120.0,
    )
    parts: list[trimesh.Trimesh] = []
    rows: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    for component_id in np.flatnonzero(
        component_sizes >= int(minimum_component_pixels)
    ):
        component_mask = component_labels == int(component_id)
        yy, xx = np.nonzero(component_mask)
        points = np.asarray(
            full_world[yy, xx],
            dtype=np.float64,
        )
        component_depth = metric_depth[yy, xx]
        centered = points - points.mean(axis=0)
        covariance = centered.T @ centered / max(len(points), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        local = centered @ eigenvectors[:, order]
        extents = np.ptp(local, axis=0)
        nonplanarity = float(
            eigenvalues[-1]
            / max(float(eigenvalues.sum()), 1e-12)
        )
        distance_to_boundary = ndimage.distance_transform_edt(
            component_mask
        )
        component_radius = float(
            distance_to_boundary[component_mask].max(initial=0.0)
        )
        boundary = component_mask & (distance_to_boundary <= 2.0)
        core = component_mask & (
            distance_to_boundary
            >= max(3.0, 0.5 * component_radius)
        )
        boundary_depth = (
            float(np.median(metric_depth[boundary]))
            if boundary.any()
            else float(np.median(component_depth))
        )
        core_depth = (
            float(np.median(metric_depth[core]))
            if core.any()
            else float(np.median(component_depth))
        )
        prominence_mm = (
            boundary_depth - core_depth
        ) * 1000.0
        lower_height = float(np.percentile(points[:, 2], 5.0))
        support_candidates = [
            plane
            for plane in planes
            if abs(float(plane["normal"][2])) >= 0.95
            and float(plane["mode_z_m"]) < lower_height
            and (
                lower_height - float(plane["mode_z_m"])
            )
            * 1000.0
            <= float(maximum_support_gap_mm)
        ]
        support = (
            max(
                support_candidates,
                key=lambda plane: int(plane["pixel_count"]),
            )
            if support_candidates
            else None
        )
        eligible = bool(
            float(extents.max()) * 1000.0
            <= float(maximum_extent_mm)
            and float(np.ptp(points[:, 2])) * 1000.0
            >= float(minimum_height_mm)
            and nonplanarity >= float(minimum_nonplanarity)
            and prominence_mm >= float(minimum_convex_prominence_mm)
            and support is not None
        )
        row: Dict[str, Any] = {
            "component_id": int(component_id),
            "pixel_count": int(len(points)),
            "pca_extents_mm": (extents * 1000.0).tolist(),
            "height_extent_mm": float(
                np.ptp(points[:, 2]) * 1000.0
            ),
            "nonplanarity": nonplanarity,
            "convex_prominence_mm": prominence_mm,
            "support_z_m": (
                float(support["mode_z_m"])
                if support is not None
                else None
            ),
            "eligible": eligible,
        }
        if not eligible:
            rows.append(row)
            continue
        try:
            mesh, mesh_metadata = _compact_symmetric_component_mesh(
                points,
                component_mask,
                camera_pos=camera_pos,
                support_z_m=float(support["mode_z_m"]),
                pitch_mm=float(pitch_mm),
                symmetry_center_percentile=float(
                    symmetry_center_percentile
                ),
                support_clearance_scale=float(
                    support_clearance_scale
                ),
                closing_radius_scale=float(closing_radius_scale),
                front_padding_mm=float(front_padding_mm),
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            failures.append(
                {
                    "component_id": int(component_id),
                    "reason": str(error),
                }
            )
            row["eligible"] = False
            row["failure"] = str(error)
            rows.append(row)
            continue
        parts.append(mesh)
        row["completion"] = mesh_metadata
        rows.append(row)
    return (
        trimesh.util.concatenate(parts) if parts else None,
        {
            "compact_symmetry_plane_regions": int(len(planes)),
            "compact_symmetry_component_count": int(
                len(component_sizes)
            ),
            "compact_symmetry_eligible_components": int(len(parts)),
            "compact_symmetry_components": rows,
            "compact_symmetry_failures": failures,
        },
    )


def _close_front_surface_components_axisymmetric_cavity(
    front_points: np.ndarray,
    front_triangles: np.ndarray,
    *,
    full_rgb: np.ndarray,
    full_world: np.ndarray,
    full_depth: np.ndarray,
    full_valid: np.ndarray,
    camera_pos: np.ndarray,
    camera_rotation: np.ndarray,
    camera_forward: np.ndarray,
    focal_px: float,
    image_width: int,
    image_height: int,
    pixel_stride: int,
    surface_edge_scale: float,
    surface_edge_slack_m: float,
    max_depth_jump_m: float,
    front_offset_mm: float,
    default_extrusion_mm: float,
    cavity_min_nonplanarity: float,
    cavity_min_concavity_depth_mm: float,
    cavity_min_component_vertices: int,
    cavity_base_wall_scale: float,
    cavity_bottom_wall_boost_scale: float,
    cavity_wall_decay_scale: float,
    cavity_bottom_start_scale: float,
    cavity_disk_start_scale: float,
    cavity_lower_wall_slope: float,
    cavity_top_height_scale: float,
    cavity_group_radius_scale: float,
    cavity_visible_shell_scale: float,
    cavity_minimum_shell_mm: float,
    cavity_maximum_shell_mm: float,
    cavity_radial_bin_mm: float,
    cavity_height_bin_mm: float,
    cavity_revolve_sections: int,
    horizontal_support_shell_mm: float,
    horizontal_support_min_vertices: int,
    horizontal_support_max_nonplanarity: float,
    horizontal_support_min_normal_alignment: float,
    horizontal_support_min_extent_mm: float,
    compact_symmetry_minimum_component_pixels: int,
    compact_symmetry_maximum_extent_mm: float,
    compact_symmetry_minimum_height_mm: float,
    compact_symmetry_minimum_nonplanarity: float,
    compact_symmetry_minimum_convex_prominence_mm: float,
    compact_symmetry_maximum_support_gap_mm: float,
    compact_symmetry_pitch_mm: float,
    compact_symmetry_center_percentile: float,
    compact_symmetry_support_clearance_scale: float,
    compact_symmetry_closing_radius_scale: float,
    compact_symmetry_front_padding_mm: float,
    topology_convex_width_scale: float,
    topology_max_extrusion_mm: float,
    defer_topology_validation: bool = False,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Complete visible cavities while retaining every other RGB-D surface."""
    front_mesh = trimesh.Trimesh(
        vertices=np.asarray(front_points, dtype=np.float64),
        faces=np.asarray(front_triangles, dtype=np.int64),
        process=False,
    )
    front_components = list(front_mesh.split(only_watertight=False))
    component_rows: list[Dict[str, Any]] = []
    for component_index, component in enumerate(front_components):
        points = np.asarray(component.vertices, dtype=np.float64)
        camera_points = (
            points - camera_pos.reshape(1, 3)
        ) @ camera_rotation
        depth = -camera_points[:, 2]
        if np.any(depth <= 1e-8):
            raise ValueError("front component contains points behind camera")
        u = (
            float(image_width) * 0.5
            + float(focal_px) * camera_points[:, 0] / depth
        )
        v = (
            float(image_height) * 0.5
            - float(focal_px) * camera_points[:, 1] / depth
        )
        sampled_u = np.rint(u / float(pixel_stride)).astype(np.int64)
        sampled_v = np.rint(v / float(pixel_stride)).astype(np.int64)
        u0, u1 = int(sampled_u.min()), int(sampled_u.max())
        v0, v1 = int(sampled_v.min()), int(sampled_v.max())
        component_mask = np.zeros(
            (v1 - v0 + 3, u1 - u0 + 3),
            dtype=bool,
        )
        local_u = sampled_u - u0 + 1
        local_v = sampled_v - v0 + 1
        component_mask[local_v, local_u] = True
        distance_to_boundary = ndimage.distance_transform_edt(
            component_mask
        )
        radius_samples = distance_to_boundary[local_v, local_u]
        maximum_radius_samples = float(
            radius_samples.max(initial=0.0)
        )
        core_threshold = max(1.5, 0.5 * maximum_radius_samples)
        boundary_vertices = radius_samples <= 1.5
        core_vertices = radius_samples >= core_threshold
        boundary_depth = (
            float(np.median(depth[boundary_vertices]))
            if boundary_vertices.any()
            else float(np.median(depth))
        )
        core_depth = (
            float(np.median(depth[core_vertices]))
            if core_vertices.any()
            else float(np.median(depth))
        )
        core_minus_boundary_mm = (
            core_depth - boundary_depth
        ) * 1000.0

        centered = points - points.mean(axis=0)
        covariance = centered.T @ centered / max(len(points), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        nonplanarity = float(
            eigenvalues[0] / max(float(eigenvalues.sum()), 1e-12)
        )
        gravity_normal_alignment = float(
            abs(eigenvectors[:, 0] @ np.array([0.0, 0.0, 1.0]))
        )
        xy_extent_mm = float(
            max(
                np.ptp(points[:, 0]),
                np.ptp(points[:, 1]),
            )
            * 1000.0
        )
        cavity_score = float(
            max(core_minus_boundary_mm, 0.0)
            * max(nonplanarity, 0.0)
            * np.sqrt(max(len(points), 1))
        )
        is_cavity_candidate = bool(
            len(points) >= int(cavity_min_component_vertices)
            and maximum_radius_samples >= 2.0
            and nonplanarity >= float(cavity_min_nonplanarity)
            and core_minus_boundary_mm
            >= float(cavity_min_concavity_depth_mm)
        )
        component_rows.append(
            {
                "component_index": int(component_index),
                "vertex_count": int(len(points)),
                "centroid_world": points.mean(axis=0).tolist(),
                "world_bounds_m": [
                    points.min(axis=0).tolist(),
                    points.max(axis=0).tolist(),
                ],
                "bbox_sampled_uv": [u0, v0, u1, v1],
                "nonplanarity": nonplanarity,
                "gravity_normal_alignment": gravity_normal_alignment,
                "xy_extent_mm": xy_extent_mm,
                "mask_radius_sample_px": maximum_radius_samples,
                "core_minus_boundary_depth_mm": (
                    core_minus_boundary_mm
                ),
                "cavity_candidate": is_cavity_candidate,
                "cavity_score": cavity_score,
            }
        )

    candidate_indices = sorted(
        (
            int(row["component_index"])
            for row in component_rows
            if bool(row["cavity_candidate"])
        ),
        key=lambda index: float(
            component_rows[index]["cavity_score"]
        ),
        reverse=True,
    )
    compact_mesh, compact_metadata = (
        _compact_supported_symmetric_meshes(
            full_rgb,
            full_depth,
            full_world,
            full_valid,
            camera_pos=camera_pos,
            focal_px=float(focal_px),
            minimum_component_pixels=int(
                compact_symmetry_minimum_component_pixels
            ),
            maximum_extent_mm=float(
                compact_symmetry_maximum_extent_mm
            ),
            minimum_height_mm=float(
                compact_symmetry_minimum_height_mm
            ),
            minimum_nonplanarity=float(
                compact_symmetry_minimum_nonplanarity
            ),
            minimum_convex_prominence_mm=float(
                compact_symmetry_minimum_convex_prominence_mm
            ),
            maximum_support_gap_mm=float(
                compact_symmetry_maximum_support_gap_mm
            ),
            pitch_mm=float(compact_symmetry_pitch_mm),
            symmetry_center_percentile=float(
                compact_symmetry_center_percentile
            ),
            support_clearance_scale=float(
                compact_symmetry_support_clearance_scale
            ),
            closing_radius_scale=float(
                compact_symmetry_closing_radius_scale
            ),
            front_padding_mm=float(
                compact_symmetry_front_padding_mm
            ),
        )
    )
    if not candidate_indices:
        mesh, metadata = _close_front_surface_components_topology(
            front_points,
            front_triangles,
            camera_pos=camera_pos,
            camera_rotation=camera_rotation,
            camera_forward=camera_forward,
            focal_px=float(focal_px),
            image_width=int(image_width),
            image_height=int(image_height),
            pixel_stride=int(pixel_stride),
            front_offset_mm=float(front_offset_mm),
            default_extrusion_mm=float(default_extrusion_mm),
            concave_wall_mm=float(cavity_minimum_shell_mm),
            concavity_depth_mm=float(cavity_min_concavity_depth_mm),
            convex_prominence_mm=5.0,
            min_nonplanarity=float(cavity_min_nonplanarity),
            convex_width_scale=float(topology_convex_width_scale),
            max_extrusion_mm=max(
                float(topology_max_extrusion_mm),
                float(default_extrusion_mm),
            ),
            min_component_vertices=int(cavity_min_component_vertices),
        )
        if compact_mesh is not None:
            mesh = trimesh.util.concatenate((mesh, compact_mesh))
        return mesh, {
            **metadata,
            **compact_metadata,
            "axisymmetric_cavity_detected": False,
            "axisymmetric_cavity_candidates": component_rows,
        }

    parts: list[trimesh.Trimesh] = []
    cavity_rows: list[Dict[str, Any]] = []
    accepted_masks: list[np.ndarray] = []
    accepted_cylinders: list[Dict[str, float]] = []
    rejected_candidates: list[Dict[str, Any]] = []
    for component_index in candidate_indices:
        component = front_components[component_index]
        try:
            cavity_mesh, group_mask, cavity_metadata = (
                _axisymmetric_cavity_mesh(
                    np.asarray(component.vertices, dtype=np.float64),
                    full_world,
                    full_valid,
                    base_wall_scale=float(cavity_base_wall_scale),
                    bottom_wall_boost_scale=float(
                        cavity_bottom_wall_boost_scale
                    ),
                    wall_decay_scale=float(cavity_wall_decay_scale),
                    bottom_start_scale=float(cavity_bottom_start_scale),
                    disk_start_scale=float(cavity_disk_start_scale),
                    lower_wall_slope=float(cavity_lower_wall_slope),
                    top_height_scale=float(cavity_top_height_scale),
                    group_radius_scale=float(cavity_group_radius_scale),
                    visible_shell_scale=float(
                        cavity_visible_shell_scale
                    ),
                    minimum_shell_mm=float(cavity_minimum_shell_mm),
                    maximum_shell_mm=float(cavity_maximum_shell_mm),
                    radial_bin_mm=float(cavity_radial_bin_mm),
                    height_bin_mm=float(cavity_height_bin_mm),
                    revolve_sections=int(cavity_revolve_sections),
                )
            )
            overlaps_existing = any(
                np.count_nonzero(mask & group_mask)
                > 0.25 * min(
                    np.count_nonzero(mask),
                    np.count_nonzero(group_mask),
                )
                for mask in accepted_masks
            )
            if overlaps_existing:
                rejected_candidates.append(
                    {
                        "component_index": int(component_index),
                        "reason": "overlaps accepted cavity",
                    }
                )
                continue
            visible_shell, visible_metadata = (
                _masked_full_resolution_shell(
                    full_world,
                    full_depth,
                    group_mask,
                    camera_pos=camera_pos,
                    camera_forward=camera_forward,
                    focal_px=float(focal_px),
                    surface_edge_scale=float(surface_edge_scale),
                    surface_edge_slack_m=float(
                        surface_edge_slack_m
                    ),
                    max_depth_jump_m=float(max_depth_jump_m),
                    back_extrusion_mm=float(
                        cavity_metadata["visible_shell_mm"]
                    ),
                )
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            rejected_candidates.append(
                {
                    "component_index": int(component_index),
                    "reason": str(error),
                }
            )
            continue
        if float(cavity_mesh.volume) < 0.0:
            cavity_mesh.invert()
        parts.extend((cavity_mesh, visible_shell))
        accepted_masks.append(group_mask)
        accepted_cylinders.append(
            {
                "center_x": float(cavity_metadata["center_xy"][0]),
                "center_y": float(cavity_metadata["center_xy"][1]),
                "radius_m": (
                    float(cavity_metadata["group_radius_mm"])
                    / 1000.0
                ),
                "modeled_outer_radius_m": (
                    float(cavity_metadata["maximum_outer_radius_mm"])
                    / 1000.0
                ),
                "bottom_z_m": float(cavity_metadata["bottom_z_m"]),
                "top_z_m": float(cavity_metadata["top_z_m"]),
            }
        )
        cavity_rows.append(
            {
                "component_index": int(component_index),
                **cavity_metadata,
                "visible_shell": visible_metadata,
            }
        )

    if not cavity_rows:
        mesh, metadata = _close_front_surface_components_topology(
            front_points,
            front_triangles,
            camera_pos=camera_pos,
            camera_rotation=camera_rotation,
            camera_forward=camera_forward,
            focal_px=float(focal_px),
            image_width=int(image_width),
            image_height=int(image_height),
            pixel_stride=int(pixel_stride),
            front_offset_mm=float(front_offset_mm),
            default_extrusion_mm=float(default_extrusion_mm),
            concave_wall_mm=float(cavity_minimum_shell_mm),
            concavity_depth_mm=float(cavity_min_concavity_depth_mm),
            convex_prominence_mm=5.0,
            min_nonplanarity=float(cavity_min_nonplanarity),
            convex_width_scale=float(topology_convex_width_scale),
            max_extrusion_mm=max(
                float(topology_max_extrusion_mm),
                float(default_extrusion_mm),
            ),
            min_component_vertices=int(cavity_min_component_vertices),
        )
        if compact_mesh is not None:
            mesh = trimesh.util.concatenate((mesh, compact_mesh))
        return mesh, {
            **metadata,
            **compact_metadata,
            "axisymmetric_cavity_detected": False,
            "axisymmetric_cavity_candidates": component_rows,
            "axisymmetric_cavity_rejections": rejected_candidates,
        }

    skipped_group_components = 0
    thin_support_components = 0
    default_shell_components = 0
    nonwatertight_shells = 0
    for component_index, component in enumerate(front_components):
        points = np.asarray(component.vertices, dtype=np.float64)
        grouped = False
        for cylinder in accepted_cylinders:
            radius = np.linalg.norm(
                points[:, :2]
                - np.array(
                    [cylinder["center_x"], cylinder["center_y"]]
                ).reshape(1, 2),
                axis=1,
            )
            inside = (
                (radius <= 1.03 * cylinder["radius_m"])
                & (
                    points[:, 2]
                    >= cylinder["bottom_z_m"]
                    - 0.01
                )
                & (
                    points[:, 2]
                    <= cylinder["top_z_m"]
                    + 0.03
                )
            )
            inside_fraction = float(np.mean(inside))
            component_rows[component_index][
                "cavity_group_radius_percentiles_mm"
            ] = np.percentile(
                radius * 1000.0,
                [0.0, 10.0, 50.0, 90.0, 100.0],
            ).tolist()
            component_rows[component_index][
                "cavity_group_inside_fraction"
            ] = inside_fraction
            median_radius_m = float(np.median(radius))
            component_rows[component_index][
                "cavity_group_median_to_modeled_outer_ratio"
            ] = (
                median_radius_m
                / max(float(cylinder["modeled_outer_radius_m"]), 1e-12)
            )
            if _belongs_to_axisymmetric_cavity_group(
                inside_fraction=inside_fraction,
                median_radius_m=median_radius_m,
                modeled_outer_radius_m=float(
                    cylinder["modeled_outer_radius_m"]
                ),
            ):
                grouped = True
                break
        if grouped:
            component_rows[component_index]["classification"] = (
                "axisymmetric_cavity_group"
            )
            skipped_group_components += 1
            continue

        row = component_rows[component_index]
        is_horizontal_support = bool(
            int(row["vertex_count"])
            >= int(horizontal_support_min_vertices)
            and float(row["nonplanarity"])
            <= float(horizontal_support_max_nonplanarity)
            and float(row["gravity_normal_alignment"])
            >= float(horizontal_support_min_normal_alignment)
            and float(row["xy_extent_mm"])
            >= float(horizontal_support_min_extent_mm)
        )
        extrusion_mm = (
            float(horizontal_support_shell_mm)
            if is_horizontal_support
            else float(default_extrusion_mm)
        )
        row["classification"] = (
            "horizontal_support_thin_shell"
            if is_horizontal_support
            else "default_shell"
        )
        row["extrusion_mm"] = extrusion_mm
        triangles = _orient_front_triangles(
            np.asarray(component.faces, dtype=np.int64),
            points,
            camera_pos=camera_pos,
        )
        shell, shell_metadata = _close_front_surface_components(
            points,
            triangles,
            camera_pos=camera_pos,
            camera_forward=camera_forward,
            front_offset_mm=float(front_offset_mm),
            back_extrusion_mm=extrusion_mm,
            _pre_split_components=(component,),
            _defer_watertight_validation=bool(
                defer_topology_validation
            ),
        )
        nonwatertight_shells += int(
            shell_metadata["nonwatertight_shell_components"]
        )
        parts.append(shell)
        if is_horizontal_support:
            thin_support_components += 1
        else:
            default_shell_components += 1

    if compact_mesh is not None:
        parts.append(compact_mesh)
    mesh = trimesh.util.concatenate(parts)
    return mesh, {
        "front_surface_components": int(len(front_components)),
        "nonwatertight_shell_components": int(nonwatertight_shells),
        "axisymmetric_cavity_detected": True,
        "axisymmetric_cavity_count": int(len(cavity_rows)),
        "axisymmetric_cavities": cavity_rows,
        "axisymmetric_cavity_rejections": rejected_candidates,
        "axisymmetric_cavity_group_components": int(
            skipped_group_components
        ),
        "horizontal_support_thin_shell_components": int(
            thin_support_components
        ),
        "default_shell_components": int(default_shell_components),
        "axisymmetric_cavity_candidates": component_rows,
        **compact_metadata,
    }


def _close_front_surface_components_support_column(
    front_points: np.ndarray,
    front_triangles: np.ndarray,
    *,
    camera_pos: np.ndarray,
    camera_rotation: np.ndarray,
    focal_px: float,
    image_width: int,
    image_height: int,
    pixel_stride: int,
    sampled_world: np.ndarray,
    sampled_valid: np.ndarray,
    support_ring_px: int,
    support_percentile: float,
    min_column_mm: float,
    max_column_mm: float,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Close visible height fields toward a locally observed support plane."""
    front_mesh = trimesh.Trimesh(
        vertices=np.asarray(front_points, dtype=np.float64),
        faces=np.asarray(front_triangles, dtype=np.int64),
        process=False,
    )
    front_components = front_mesh.split(only_watertight=False)
    world_grid = np.asarray(sampled_world, dtype=np.float64)
    valid_grid = np.asarray(sampled_valid, dtype=bool)
    if world_grid.shape[:2] != valid_grid.shape:
        raise ValueError("support column world and validity grids differ")
    ring_samples = max(
        1,
        int(math.ceil(float(support_ring_px) / float(pixel_stride))),
    )
    percentile = float(support_percentile)
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("support percentile must be within [0, 100]")
    shells: list[trimesh.Trimesh] = []
    support_values: list[float] = []
    column_values: list[np.ndarray] = []
    boundary_edge_count = 0
    nonwatertight_shells = 0
    fallback_components = 0
    for component in front_components:
        points = np.asarray(component.vertices, dtype=np.float64)
        triangles = _orient_front_triangles(
            np.asarray(component.faces, dtype=np.int64),
            points,
            camera_pos=camera_pos,
        )
        camera_points = (
            points - camera_pos.reshape(1, 3)
        ) @ camera_rotation
        depth = -camera_points[:, 2]
        u = (
            float(image_width) * 0.5
            + float(focal_px) * camera_points[:, 0] / depth
        )
        v = (
            float(image_height) * 0.5
            - float(focal_px) * camera_points[:, 1] / depth
        )
        sampled_u = np.clip(
            np.rint(u / float(pixel_stride)).astype(np.int64),
            0,
            world_grid.shape[1] - 1,
        )
        sampled_v = np.clip(
            np.rint(v / float(pixel_stride)).astype(np.int64),
            0,
            world_grid.shape[0] - 1,
        )
        component_mask = np.zeros(valid_grid.shape, dtype=bool)
        component_mask[sampled_v, sampled_u] = True
        outside_ring = ndimage.binary_dilation(
            component_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=ring_samples,
        ) & (~component_mask) & valid_grid
        lower_ring = outside_ring & (
            world_grid[..., 2]
            <= float(np.percentile(points[:, 2], 25.0)) + 0.005
        )
        support_candidates = world_grid[
            lower_ring if lower_ring.any() else outside_ring,
            2,
        ]
        if len(support_candidates):
            support_z = float(
                np.percentile(support_candidates, percentile)
            )
        else:
            support_z = float(points[:, 2].min())
            fallback_components += 1
        column_mm = np.clip(
            (points[:, 2] - support_z) * 1000.0,
            float(min_column_mm),
            float(max_column_mm),
        )
        shifted_back = points.copy()
        shifted_back[:, 2] = points[:, 2] - column_mm / 1000.0
        support_values.append(support_z)
        column_values.append(column_mm)

        vertex_count = len(points)
        back_triangles = triangles[:, [0, 2, 1]] + vertex_count
        boundary_edges = _boundary_directed_edges(triangles)
        boundary_edge_count += int(len(boundary_edges))
        side_first = np.column_stack(
            (
                boundary_edges[:, 0],
                boundary_edges[:, 0] + vertex_count,
                boundary_edges[:, 1] + vertex_count,
            )
        )
        side_second = np.column_stack(
            (
                boundary_edges[:, 0],
                boundary_edges[:, 1] + vertex_count,
                boundary_edges[:, 1],
            )
        )
        shell = trimesh.Trimesh(
            vertices=np.vstack((points, shifted_back)),
            faces=np.vstack(
                (
                    triangles,
                    back_triangles,
                    side_first,
                    side_second,
                )
            ),
            process=False,
        )
        if not shell.is_watertight:
            nonwatertight_shells += 1
        shells.append(shell)
    if not shells:
        raise ValueError("no support-column components were reconstructed")
    all_columns = np.concatenate(column_values)
    return trimesh.util.concatenate(shells), {
        "front_surface_components": int(len(front_components)),
        "boundary_edges": int(boundary_edge_count),
        "nonwatertight_shell_components": int(nonwatertight_shells),
        "support_column_fallback_components": int(fallback_components),
        "support_z_quantiles": {
            key: float(value)
            for key, value in zip(
                ("0", "10", "50", "90", "100"),
                np.percentile(
                    np.asarray(support_values, dtype=np.float64),
                    [0.0, 10.0, 50.0, 90.0, 100.0],
                ),
            )
        },
        "support_column_mm_quantiles": {
            key: float(value)
            for key, value in zip(
                ("0", "10", "25", "50", "75", "90", "100"),
                np.percentile(
                    all_columns,
                    [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0],
                ),
            )
        },
    }


def _screened_poisson_mesh(
    front_points: np.ndarray,
    front_triangles: np.ndarray,
    *,
    camera_pos: np.ndarray,
    poisson_depth: int,
    poisson_full_depth: int,
    poisson_scale: float,
    poisson_samples_per_node: float,
    poisson_point_weight: float,
    poisson_iterations: int,
    poisson_threads: int,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Fit one implicit surface to every observed RGB-D front triangle."""
    front_mesh = trimesh.Trimesh(
        vertices=np.asarray(front_points, dtype=np.float64),
        faces=np.asarray(front_triangles, dtype=np.int64),
        process=False,
    )
    normals = np.array(
        front_mesh.vertex_normals,
        dtype=np.float64,
        copy=True,
    )
    view = (
        np.asarray(camera_pos, dtype=np.float64).reshape(1, 3)
        - np.asarray(front_points, dtype=np.float64)
    )
    flip = np.einsum("ij,ij->i", normals, view) < 0.0
    normals[flip] *= -1.0
    return _screened_poisson_oriented_points(
        np.asarray(front_points, dtype=np.float64),
        normals,
        poisson_depth=int(poisson_depth),
        poisson_full_depth=int(poisson_full_depth),
        poisson_scale=float(poisson_scale),
        poisson_samples_per_node=float(poisson_samples_per_node),
        poisson_point_weight=float(poisson_point_weight),
        poisson_iterations=int(poisson_iterations),
        poisson_threads=int(poisson_threads),
    )


def _screened_poisson_oriented_points(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    poisson_depth: int,
    poisson_full_depth: int,
    poisson_scale: float,
    poisson_samples_per_node: float,
    poisson_point_weight: float,
    poisson_iterations: int,
    poisson_threads: int,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Fit and repair a Screened Poisson surface from oriented points."""
    try:
        import pymeshlab
    except ImportError as exc:
        raise RuntimeError(
            "screened_poisson completion requires pymeshlab"
        ) from exc

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.shape != normals.shape or points.ndim != 2:
        raise ValueError("oriented points and normals must both be Nx3")
    normal_length = np.linalg.norm(normals, axis=1)
    valid_normals = np.isfinite(normals).all(axis=1) & (
        normal_length > 1e-10
    )
    if int(valid_normals.sum()) < 16:
        raise ValueError("too few oriented points for screened Poisson")
    points = points[valid_normals]
    normals = normals[valid_normals]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=points,
            v_normals_matrix=normals,
        ),
        "whole_view_rgbd_points",
    )
    mesh_set.generate_surface_reconstruction_screened_poisson(
        depth=int(poisson_depth),
        fulldepth=int(poisson_full_depth),
        scale=float(poisson_scale),
        samplespernode=float(poisson_samples_per_node),
        pointweight=float(poisson_point_weight),
        iters=int(poisson_iterations),
        confidence=False,
        preclean=True,
        threads=max(1, int(poisson_threads)),
    )
    poisson = mesh_set.current_mesh()
    mesh = trimesh.Trimesh(
        vertices=np.asarray(poisson.vertex_matrix(), dtype=np.float64),
        faces=np.asarray(poisson.face_matrix(), dtype=np.int64),
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    raw_boundary_edges = int(len(_boundary_directed_edges(mesh.faces)))
    meshlab_closed_holes = False
    if raw_boundary_edges:
        repair_set = pymeshlab.MeshSet()
        repair_set.add_mesh(
            pymeshlab.Mesh(
                vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
                face_matrix=np.asarray(mesh.faces, dtype=np.int64),
            ),
            "poisson_surface",
        )
        repair_set.meshing_close_holes(
            maxholesize=100_000,
            selected=False,
            newfaceselected=False,
            selfintersection=True,
        )
        repaired = repair_set.current_mesh()
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                repaired.vertex_matrix(),
                dtype=np.float64,
            ),
            faces=np.asarray(repaired.face_matrix(), dtype=np.int64),
            process=False,
        )
        mesh.process(validate=True)
        meshlab_closed_holes = True
    boundary_after_meshlab = int(
        len(_boundary_directed_edges(mesh.faces))
    )
    mesh, centroid_cap_loops = _cap_oriented_boundary_loops(mesh)
    components = mesh.split(only_watertight=False)
    nonwatertight = sum(
        not bool(component.is_watertight)
        for component in components
    )
    return mesh, {
        "poisson_input_points": int(len(points)),
        "poisson_raw_boundary_edges": raw_boundary_edges,
        "poisson_meshlab_close_holes": bool(meshlab_closed_holes),
        "poisson_boundary_edges_after_meshlab": (
            boundary_after_meshlab
        ),
        "poisson_centroid_cap_loops": int(centroid_cap_loops),
        "poisson_components": int(len(components)),
        "poisson_nonwatertight_components": int(nonwatertight),
    }


def _component_screened_poisson_mesh(
    front_points: np.ndarray,
    front_triangles: np.ndarray,
    *,
    camera_pos: np.ndarray,
    camera_forward: np.ndarray,
    front_offset_mm: float,
    back_extrusion_mm: float,
    poisson_min_component_vertices: int,
    poisson_depth: int,
    poisson_full_depth: int,
    poisson_scale: float,
    poisson_samples_per_node: float,
    poisson_point_weight: float,
    poisson_iterations: int,
    poisson_threads: int,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Complete every edge-connected patch independently and retain all."""
    front_mesh = trimesh.Trimesh(
        vertices=np.asarray(front_points, dtype=np.float64),
        faces=np.asarray(front_triangles, dtype=np.int64),
        process=False,
    )
    front_components = front_mesh.split(only_watertight=False)
    completed: list[trimesh.Trimesh] = []
    poisson_components = 0
    shell_fallback_components = 0
    poisson_failures: list[str] = []
    poisson_raw_boundary_edges = 0
    poisson_centroid_cap_loops = 0
    for component_index, component in enumerate(front_components):
        points = np.asarray(component.vertices, dtype=np.float64)
        triangles = _orient_front_triangles(
            np.asarray(component.faces, dtype=np.int64),
            points,
            camera_pos=camera_pos,
        )
        use_poisson = (
            len(points) >= int(poisson_min_component_vertices)
            and np.linalg.matrix_rank(points - points.mean(axis=0)) >= 2
        )
        if use_poisson:
            component_mesh = trimesh.Trimesh(
                vertices=points,
                faces=triangles,
                process=False,
            )
            normals = np.array(
                component_mesh.vertex_normals,
                dtype=np.float64,
                copy=True,
            )
            view = camera_pos.reshape(1, 3) - points
            flip = np.einsum("ij,ij->i", normals, view) < 0.0
            normals[flip] *= -1.0
            try:
                result, metadata = _screened_poisson_oriented_points(
                    points,
                    normals,
                    poisson_depth=int(poisson_depth),
                    poisson_full_depth=int(poisson_full_depth),
                    poisson_scale=float(poisson_scale),
                    poisson_samples_per_node=float(
                        poisson_samples_per_node
                    ),
                    poisson_point_weight=float(poisson_point_weight),
                    poisson_iterations=int(poisson_iterations),
                    poisson_threads=int(poisson_threads),
                )
                if not result.is_watertight:
                    raise ValueError("Poisson output is not watertight")
                completed.append(result)
                poisson_components += 1
                poisson_raw_boundary_edges += int(
                    metadata["poisson_raw_boundary_edges"]
                )
                poisson_centroid_cap_loops += int(
                    metadata["poisson_centroid_cap_loops"]
                )
                continue
            except Exception as exc:
                poisson_failures.append(
                    f"{component_index}:{type(exc).__name__}:{exc}"
                )

        shell, shell_metadata = _close_front_surface_components(
            points,
            triangles,
            camera_pos=camera_pos,
            camera_forward=camera_forward,
            front_offset_mm=float(front_offset_mm),
            back_extrusion_mm=float(back_extrusion_mm),
        )
        if shell_metadata["nonwatertight_shell_components"]:
            raise ValueError(
                "component shell fallback produced a nonwatertight mesh"
            )
        completed.append(shell)
        shell_fallback_components += 1

    mesh = trimesh.util.concatenate(completed)
    mesh_components = mesh.split(only_watertight=False)
    nonwatertight = sum(
        not bool(component.is_watertight)
        for component in mesh_components
    )
    return mesh, {
        "front_surface_components": int(len(front_components)),
        "component_poisson_components": int(poisson_components),
        "component_shell_fallback_components": int(
            shell_fallback_components
        ),
        "component_poisson_failures": poisson_failures,
        "poisson_raw_boundary_edges": int(poisson_raw_boundary_edges),
        "poisson_centroid_cap_loops": int(poisson_centroid_cap_loops),
        "component_nonwatertight_components": int(nonwatertight),
    }


def reconstruct_rgbd_scene_mesh(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    pixel_stride: int = 2,
    min_depth_m: float = 0.12,
    max_depth_m: float = 6.0,
    surface_edge_scale: float = 3.5,
    surface_edge_slack_mm: float = 2.0,
    max_depth_jump_mm: float = 35.0,
    rgbd_edge_color_threshold: float = 0.0,
    rgbd_edge_min_depth_jump_mm: float = 3.0,
    back_extrusion_mm: float = 60.0,
    adaptive_min_extrusion_mm: float = 1.0,
    adaptive_width_scale: float = 1.0,
    local_width_windows_px: Sequence[int] = (5, 9, 17, 33),
    local_width_min_prominence_mm: float = 2.0,
    local_width_closing_iterations: int = 1,
    local_width_radius_bias_px: float = 0.5,
    topology_concave_wall_mm: float = 10.0,
    topology_concavity_depth_mm: float = 20.0,
    topology_convex_prominence_mm: float = 5.0,
    topology_min_nonplanarity: float = 0.025,
    topology_convex_width_scale: float = 1.0,
    topology_max_extrusion_mm: float = 140.0,
    topology_min_component_vertices: int = 50,
    cavity_base_wall_scale: float = 0.045,
    cavity_bottom_wall_boost_scale: float = 0.17,
    cavity_wall_decay_scale: float = 0.062,
    cavity_bottom_start_scale: float = 0.215,
    cavity_disk_start_scale: float = 0.074,
    cavity_lower_wall_slope: float = 0.156,
    cavity_top_height_scale: float = 1.4,
    cavity_group_radius_scale: float = 1.4,
    cavity_visible_shell_scale: float = 0.055,
    cavity_minimum_shell_mm: float = 4.0,
    cavity_maximum_shell_mm: float = 12.0,
    cavity_radial_bin_mm: float = 1.0,
    cavity_height_bin_mm: float = 2.0,
    cavity_revolve_sections: int = 128,
    horizontal_support_shell_mm: float = 0.5,
    horizontal_support_min_vertices: int = 1000,
    horizontal_support_max_nonplanarity: float = 0.01,
    horizontal_support_min_normal_alignment: float = 0.9,
    horizontal_support_min_extent_mm: float = 400.0,
    compact_symmetry_minimum_component_pixels: int = 1000,
    compact_symmetry_maximum_extent_mm: float = 500.0,
    compact_symmetry_minimum_height_mm: float = 80.0,
    compact_symmetry_minimum_nonplanarity: float = 0.02,
    compact_symmetry_minimum_convex_prominence_mm: float = 20.0,
    compact_symmetry_maximum_support_gap_mm: float = 120.0,
    compact_symmetry_pitch_mm: float = 1.5,
    compact_symmetry_center_percentile: float = 20.0,
    compact_symmetry_support_clearance_scale: float = 0.125,
    compact_symmetry_closing_radius_scale: float = 0.01,
    compact_symmetry_front_padding_mm: float = 1.5,
    thin_background_window_px: int = 5,
    thin_min_protrusion_mm: float = 3.0,
    thin_closing_iterations: int = 2,
    thin_min_component_pixels: int = 3,
    thin_max_radius_px: float = 6.0,
    thin_radius_scale: float = 1.0,
    thin_radius_bias_px: float = 0.5,
    thin_min_radius_mm: float = 0.8,
    thin_max_radius_mm: float = 8.0,
    thin_center_offset_scale: float = 1.0,
    thin_ring_samples: int = 12,
    thin_background_shell_mm: float = 0.5,
    thin_nonlinear_refine: bool = True,
    support_ring_px: int = 12,
    support_percentile: float = 20.0,
    support_min_column_mm: float = 1.0,
    support_max_column_mm: float = 300.0,
    front_offset_mm: float = 0.0,
    completion_method: str = "organized_shell",
    poisson_depth: int = 10,
    poisson_full_depth: int = 6,
    poisson_scale: float = 1.05,
    poisson_samples_per_node: float = 1.5,
    poisson_point_weight: float = 8.0,
    poisson_iterations: int = 8,
    poisson_threads: int = 16,
    poisson_min_component_vertices: int = 200,
    _defer_topology_validation_to_warp_cuda: bool = False,
) -> RGBDSceneMeshResult:
    """Reconstruct every geometrically continuous visible RGB-D surface.

    The mesh is a union of closed, camera-ray shells. The back extrusion is a
    generic single-view completion prior, applied uniformly to the full frame.
    It does not depend on image location, clicked pixels, object identity, or
    evaluator data.
    """
    color = np.asarray(rgb)
    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    if color.ndim != 3 or color.shape[2] < 3:
        raise ValueError(f"rgb must be HxWx3, got {color.shape}")
    if color.shape[:2] != metric_depth.shape:
        raise ValueError(
            f"rgb/depth shape mismatch: {color.shape[:2]} vs "
            f"{metric_depth.shape}"
        )
    if float(horizontal_aperture) <= 0.0:
        raise ValueError("horizontal_aperture must be positive")
    if float(back_extrusion_mm) <= 0.0:
        raise ValueError("back_extrusion_mm must be positive")
    if float(max_depth_jump_mm) <= 0.0:
        raise ValueError("max_depth_jump_mm must be positive")
    if float(rgbd_edge_min_depth_jump_mm) < 0.0:
        raise ValueError("rgbd_edge_min_depth_jump_mm must be nonnegative")
    if float(adaptive_min_extrusion_mm) <= 0.0:
        raise ValueError("adaptive_min_extrusion_mm must be positive")
    if float(adaptive_width_scale) <= 0.0:
        raise ValueError("adaptive_width_scale must be positive")
    if float(topology_concave_wall_mm) <= 0.0:
        raise ValueError("topology concave wall must be positive")
    if float(topology_concavity_depth_mm) < 0.0:
        raise ValueError("topology concavity depth must be nonnegative")
    if float(topology_convex_prominence_mm) < 0.0:
        raise ValueError("topology convex prominence must be nonnegative")
    if not 0.0 <= float(topology_min_nonplanarity) <= 1.0:
        raise ValueError("topology nonplanarity must be in [0, 1]")
    if float(topology_convex_width_scale) <= 0.0:
        raise ValueError("topology convex width scale must be positive")
    if float(topology_max_extrusion_mm) < float(back_extrusion_mm):
        raise ValueError(
            "topology maximum extrusion must be >= default extrusion"
        )
    if int(topology_min_component_vertices) < 3:
        raise ValueError(
            "topology minimum component vertices must be >=3"
        )
    positive_cavity_scales = (
        cavity_base_wall_scale,
        cavity_bottom_wall_boost_scale,
        cavity_wall_decay_scale,
        cavity_bottom_start_scale,
        cavity_disk_start_scale,
        cavity_top_height_scale,
        cavity_group_radius_scale,
        cavity_visible_shell_scale,
    )
    if any(float(value) <= 0.0 for value in positive_cavity_scales):
        raise ValueError("cavity scale parameters must be positive")
    if not 0.0 <= float(cavity_lower_wall_slope) < 1.0:
        raise ValueError("cavity lower wall slope must be in [0, 1)")
    if (
        float(cavity_minimum_shell_mm) <= 0.0
        or float(cavity_maximum_shell_mm)
        < float(cavity_minimum_shell_mm)
    ):
        raise ValueError("invalid cavity visible shell limits")
    if (
        float(cavity_radial_bin_mm) <= 0.0
        or float(cavity_height_bin_mm) <= 0.0
    ):
        raise ValueError("cavity profile bins must be positive")
    if int(cavity_revolve_sections) < 16:
        raise ValueError("cavity revolve sections must be >=16")
    if float(horizontal_support_shell_mm) <= 0.0:
        raise ValueError("horizontal support shell must be positive")
    if int(horizontal_support_min_vertices) < 3:
        raise ValueError(
            "horizontal support minimum vertices must be >=3"
        )
    if not 0.0 <= float(horizontal_support_max_nonplanarity) <= 1.0:
        raise ValueError(
            "horizontal support nonplanarity must be in [0, 1]"
        )
    if not 0.0 <= float(horizontal_support_min_normal_alignment) <= 1.0:
        raise ValueError(
            "horizontal support normal alignment must be in [0, 1]"
        )
    if float(horizontal_support_min_extent_mm) <= 0.0:
        raise ValueError("horizontal support extent must be positive")
    if int(compact_symmetry_minimum_component_pixels) < 100:
        raise ValueError(
            "compact symmetry minimum component pixels must be >=100"
        )
    if (
        float(compact_symmetry_maximum_extent_mm) <= 0.0
        or float(compact_symmetry_minimum_height_mm) <= 0.0
        or float(compact_symmetry_minimum_convex_prominence_mm) < 0.0
        or float(compact_symmetry_maximum_support_gap_mm) <= 0.0
        or float(compact_symmetry_pitch_mm) <= 0.0
        or float(compact_symmetry_front_padding_mm) < 0.0
    ):
        raise ValueError("invalid compact symmetry metric parameters")
    if not 0.0 <= float(
        compact_symmetry_minimum_nonplanarity
    ) <= 1.0:
        raise ValueError(
            "compact symmetry nonplanarity must be in [0, 1]"
        )
    if not 0.0 <= float(
        compact_symmetry_center_percentile
    ) <= 50.0:
        raise ValueError(
            "compact symmetry center percentile must be in [0, 50]"
        )
    if not 0.0 < float(
        compact_symmetry_support_clearance_scale
    ) <= 1.0:
        raise ValueError(
            "compact symmetry support clearance scale must be in (0, 1]"
        )
    if float(compact_symmetry_closing_radius_scale) <= 0.0:
        raise ValueError(
            "compact symmetry closing radius scale must be positive"
        )
    if int(thin_min_component_pixels) < 1:
        raise ValueError("thin minimum component pixels must be >=1")
    if float(thin_max_radius_px) <= 0.0:
        raise ValueError("thin maximum radius must be positive")
    if float(thin_radius_scale) <= 0.0:
        raise ValueError("thin radius scale must be positive")
    if float(thin_radius_bias_px) < 0.0:
        raise ValueError("thin radius bias must be nonnegative")
    if (
        float(thin_min_radius_mm) <= 0.0
        or float(thin_max_radius_mm) < float(thin_min_radius_mm)
    ):
        raise ValueError("invalid thin radius limits")
    if float(thin_center_offset_scale) < 0.0:
        raise ValueError("thin center offset scale must be nonnegative")
    if int(thin_ring_samples) < 8:
        raise ValueError("thin ring samples must be >=8")
    if float(thin_background_shell_mm) <= 0.0:
        raise ValueError("thin background shell must be positive")
    if int(support_ring_px) < 1:
        raise ValueError("support ring must be positive")
    if (
        float(support_min_column_mm) <= 0.0
        or float(support_max_column_mm)
        < float(support_min_column_mm)
    ):
        raise ValueError("invalid support column limits")

    selected_completion = str(completion_method).strip().lower()
    use_thin_tubes = selected_completion in {
        "hybrid_thin_tubes",
        "hybrid_circle_tubes",
        "width_aware_hybrid",
    }
    thin_raw_mask = np.zeros(metric_depth.shape, dtype=bool)
    thin_completed_mask = np.zeros(metric_depth.shape, dtype=bool)
    thin_mask_metadata: Dict[str, Any] = {}
    if use_thin_tubes:
        (
            thin_raw_mask,
            thin_completed_mask,
            thin_mask_metadata,
        ) = _thin_protrusion_mask(
            metric_depth,
            min_depth_m=float(min_depth_m),
            max_depth_m=float(max_depth_m),
            background_window_px=int(thin_background_window_px),
            min_protrusion_mm=float(thin_min_protrusion_mm),
            closing_iterations=int(thin_closing_iterations),
        )
    local_width_mask: np.ndarray | None = None
    local_width_radius_px: np.ndarray | None = None
    local_width_metadata: Dict[str, Any] = {}
    if selected_completion == "local_width_shell":
        (
            local_width_mask,
            local_width_radius_px,
            local_width_metadata,
        ) = _multiscale_prominence_radius_px(
            metric_depth,
            min_depth_m=float(min_depth_m),
            max_depth_m=float(max_depth_m),
            window_sizes_px=local_width_windows_px,
            min_prominence_mm=float(local_width_min_prominence_mm),
            closing_iterations=int(local_width_closing_iterations),
        )

    (
        sampled_world,
        sampled_depth,
        valid,
        _camera_points,
        focal_px,
    ) = _sampled_world_points(
        metric_depth,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
        pixel_stride=int(pixel_stride),
        min_depth_m=float(min_depth_m),
        max_depth_m=float(max_depth_m),
    )
    sample_height, sample_width = sampled_depth.shape
    if use_thin_tubes:
        sampled_thin = thin_completed_mask[
            :: int(pixel_stride),
            :: int(pixel_stride),
        ]
        if sampled_thin.shape != valid.shape:
            raise ValueError("sampled thin mask and depth grid do not match")
        valid = valid & (~sampled_thin)
    sampled_color = np.asarray(
        color[
            :: int(pixel_stride),
            :: int(pixel_stride),
            :3,
        ],
        dtype=np.float64,
    )
    if sampled_color.shape[:2] != sampled_depth.shape:
        raise ValueError("sampled RGB and depth grids do not match")
    grid_ids = np.arange(
        sample_height * sample_width,
        dtype=np.int64,
    ).reshape(sample_height, sample_width)

    slack_m = float(surface_edge_slack_mm) / 1000.0
    jump_m = float(max_depth_jump_mm) / 1000.0
    horizontal_ok = _rgbd_edge_gate(
        (
        valid[:, :-1]
        & valid[:, 1:]
        & _continuous_edge(
            sampled_world[:, :-1],
            sampled_world[:, 1:],
            sampled_depth[:, :-1],
            sampled_depth[:, 1:],
            pixel_span=float(pixel_stride),
            focal_px=focal_px,
            surface_edge_scale=float(surface_edge_scale),
            surface_edge_slack_m=slack_m,
            max_depth_jump_m=jump_m,
        )
        ),
        sampled_depth[:, :-1],
        sampled_depth[:, 1:],
        sampled_color[:, :-1],
        sampled_color[:, 1:],
        color_threshold=float(rgbd_edge_color_threshold),
        min_depth_jump_m=float(rgbd_edge_min_depth_jump_mm) / 1000.0,
    )
    vertical_ok = _rgbd_edge_gate(
        (
        valid[:-1, :]
        & valid[1:, :]
        & _continuous_edge(
            sampled_world[:-1, :],
            sampled_world[1:, :],
            sampled_depth[:-1, :],
            sampled_depth[1:, :],
            pixel_span=float(pixel_stride),
            focal_px=focal_px,
            surface_edge_scale=float(surface_edge_scale),
            surface_edge_slack_m=slack_m,
            max_depth_jump_m=jump_m,
        )
        ),
        sampled_depth[:-1, :],
        sampled_depth[1:, :],
        sampled_color[:-1, :],
        sampled_color[1:, :],
        color_threshold=float(rgbd_edge_color_threshold),
        min_depth_jump_m=float(rgbd_edge_min_depth_jump_mm) / 1000.0,
    )
    diagonal_span = float(pixel_stride) * np.sqrt(2.0)
    diagonal_down_ok = _rgbd_edge_gate(
        (
        valid[:-1, :-1]
        & valid[1:, 1:]
        & _continuous_edge(
            sampled_world[:-1, :-1],
            sampled_world[1:, 1:],
            sampled_depth[:-1, :-1],
            sampled_depth[1:, 1:],
            pixel_span=diagonal_span,
            focal_px=focal_px,
            surface_edge_scale=float(surface_edge_scale),
            surface_edge_slack_m=slack_m,
            max_depth_jump_m=jump_m,
        )
        ),
        sampled_depth[:-1, :-1],
        sampled_depth[1:, 1:],
        sampled_color[:-1, :-1],
        sampled_color[1:, 1:],
        color_threshold=float(rgbd_edge_color_threshold),
        min_depth_jump_m=float(rgbd_edge_min_depth_jump_mm) / 1000.0,
    )
    diagonal_up_ok = _rgbd_edge_gate(
        (
        valid[1:, :-1]
        & valid[:-1, 1:]
        & _continuous_edge(
            sampled_world[1:, :-1],
            sampled_world[:-1, 1:],
            sampled_depth[1:, :-1],
            sampled_depth[:-1, 1:],
            pixel_span=diagonal_span,
            focal_px=focal_px,
            surface_edge_scale=float(surface_edge_scale),
            surface_edge_slack_m=slack_m,
            max_depth_jump_m=jump_m,
        )
        ),
        sampled_depth[1:, :-1],
        sampled_depth[:-1, 1:],
        sampled_color[1:, :-1],
        sampled_color[:-1, 1:],
        color_threshold=float(rgbd_edge_color_threshold),
        min_depth_jump_m=float(rgbd_edge_min_depth_jump_mm) / 1000.0,
    )

    top_left = grid_ids[:-1, :-1]
    top_right = grid_ids[:-1, 1:]
    bottom_left = grid_ids[1:, :-1]
    bottom_right = grid_ids[1:, 1:]
    triangle_rows: list[np.ndarray] = []

    first_down = (
        horizontal_ok[:-1, :]
        & vertical_ok[:, :-1]
        & diagonal_down_ok
    )
    second_down = (
        horizontal_ok[1:, :]
        & vertical_ok[:, 1:]
        & diagonal_down_ok
    )
    first_up = (
        horizontal_ok[:-1, :]
        & vertical_ok[:, 1:]
        & diagonal_up_ok
    )
    second_up = (
        horizontal_ok[1:, :]
        & vertical_ok[:, :-1]
        & diagonal_up_ok
    )
    use_down = (
        (first_down.astype(np.int8) + second_down.astype(np.int8))
        >= (first_up.astype(np.int8) + second_up.astype(np.int8))
    )

    mask = use_down & first_down
    if mask.any():
        triangle_rows.append(
            np.column_stack(
                (top_left[mask], bottom_right[mask], top_right[mask])
            )
        )
    mask = use_down & second_down
    if mask.any():
        triangle_rows.append(
            np.column_stack(
                (top_left[mask], bottom_left[mask], bottom_right[mask])
            )
        )
    mask = (~use_down) & first_up
    if mask.any():
        triangle_rows.append(
            np.column_stack(
                (top_left[mask], bottom_left[mask], top_right[mask])
            )
        )
    mask = (~use_down) & second_up
    if mask.any():
        triangle_rows.append(
            np.column_stack(
                (top_right[mask], bottom_left[mask], bottom_right[mask])
            )
        )
    if not triangle_rows:
        raise ValueError("no continuous RGB-D triangles were reconstructed")

    front_points = sampled_world.reshape(-1, 3)
    front_triangles = np.concatenate(triangle_rows, axis=0)
    used_grid_ids = np.unique(front_triangles)
    compact_id = np.full(len(front_points), -1, dtype=np.int64)
    compact_id[used_grid_ids] = np.arange(len(used_grid_ids), dtype=np.int64)
    front_points = front_points[used_grid_ids]
    front_triangles = compact_id[front_triangles]

    camera_position = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    camera_rotation = _quat_to_mat_xyzw(camera_quat_xyzw)
    camera_forward = camera_rotation @ np.array(
        [0.0, 0.0, -1.0],
        dtype=np.float64,
    )
    front_triangles = _orient_front_triangles(
        front_triangles,
        front_points,
        camera_pos=camera_position,
    )
    organized_front_vertex_count = len(front_points)
    front_points, front_triangles, manifold_metadata = (
        _split_nonmanifold_vertex_fans(
            front_points,
            front_triangles,
        )
    )
    vertex_count = len(front_points)
    if selected_completion == "organized_shell":
        mesh, completion_metadata = _close_front_surface_components(
            front_points,
            front_triangles,
            camera_pos=camera_position,
            camera_forward=camera_forward,
            front_offset_mm=float(front_offset_mm),
            back_extrusion_mm=float(back_extrusion_mm),
        )
        method = "whole_view_organized_depth_ray_shell"
        nonwatertight_components = int(
            completion_metadata["nonwatertight_shell_components"]
        )
    elif selected_completion == "normal_shell":
        mesh, completion_metadata = (
            _close_front_surface_components_normal(
                front_points,
                front_triangles,
                camera_pos=camera_position,
                camera_forward=camera_forward,
                front_offset_mm=float(front_offset_mm),
                back_extrusion_mm=float(back_extrusion_mm),
            )
        )
        method = "whole_view_surface_normal_shell"
        nonwatertight_components = int(
            completion_metadata["nonwatertight_shell_components"]
        )
    elif selected_completion == "adaptive_shell":
        mesh, completion_metadata = (
            _close_front_surface_components_adaptive(
                front_points,
                front_triangles,
                camera_pos=camera_position,
                camera_rotation=camera_rotation,
                camera_forward=camera_forward,
                focal_px=float(focal_px),
                image_width=int(metric_depth.shape[1]),
                image_height=int(metric_depth.shape[0]),
                pixel_stride=int(pixel_stride),
                front_offset_mm=float(front_offset_mm),
                min_extrusion_mm=float(adaptive_min_extrusion_mm),
                max_extrusion_mm=float(back_extrusion_mm),
                width_scale=float(adaptive_width_scale),
            )
        )
        method = "whole_view_rgbd_silhouette_chord_shell"
        nonwatertight_components = int(
            completion_metadata["nonwatertight_shell_components"]
        )
    elif selected_completion == "local_width_shell":
        mesh, completion_metadata = (
            _close_front_surface_components_adaptive(
                front_points,
                front_triangles,
                camera_pos=camera_position,
                camera_rotation=camera_rotation,
                camera_forward=camera_forward,
                focal_px=float(focal_px),
                image_width=int(metric_depth.shape[1]),
                image_height=int(metric_depth.shape[0]),
                pixel_stride=int(pixel_stride),
                front_offset_mm=float(front_offset_mm),
                min_extrusion_mm=float(adaptive_min_extrusion_mm),
                max_extrusion_mm=float(back_extrusion_mm),
                width_scale=float(adaptive_width_scale),
                prominence_mask=local_width_mask,
                prominence_radius_px=local_width_radius_px,
                prominence_radius_bias_px=float(
                    local_width_radius_bias_px
                ),
            )
        )
        completion_metadata = {
            **local_width_metadata,
            **completion_metadata,
        }
        method = "whole_view_multiscale_local_width_ray_shell"
        nonwatertight_components = int(
            completion_metadata["nonwatertight_shell_components"]
        )
    elif selected_completion == "topology_hybrid":
        mesh, completion_metadata = (
            _close_front_surface_components_topology(
                front_points,
                front_triangles,
                camera_pos=camera_position,
                camera_rotation=camera_rotation,
                camera_forward=camera_forward,
                focal_px=float(focal_px),
                image_width=int(metric_depth.shape[1]),
                image_height=int(metric_depth.shape[0]),
                pixel_stride=int(pixel_stride),
                front_offset_mm=float(front_offset_mm),
                default_extrusion_mm=float(back_extrusion_mm),
                concave_wall_mm=float(topology_concave_wall_mm),
                concavity_depth_mm=float(
                    topology_concavity_depth_mm
                ),
                convex_prominence_mm=float(
                    topology_convex_prominence_mm
                ),
                min_nonplanarity=float(
                    topology_min_nonplanarity
                ),
                convex_width_scale=float(
                    topology_convex_width_scale
                ),
                max_extrusion_mm=float(
                    topology_max_extrusion_mm
                ),
                min_component_vertices=int(
                    topology_min_component_vertices
                ),
            )
        )
        method = "whole_view_topology_conditioned_ray_shell"
        nonwatertight_components = int(
            completion_metadata["nonwatertight_shell_components"]
        )
    elif selected_completion == "axisymmetric_cavity_hybrid":
        (
            full_world,
            full_depth,
            full_valid,
            _full_camera_points,
            full_focal_px,
        ) = _sampled_world_points(
            metric_depth,
            camera_pos=camera_pos,
            camera_quat_xyzw=camera_quat_xyzw,
            focal_length=float(focal_length),
            horizontal_aperture=float(horizontal_aperture),
            pixel_stride=1,
            min_depth_m=float(min_depth_m),
            max_depth_m=float(max_depth_m),
        )
        mesh, completion_metadata = (
            _close_front_surface_components_axisymmetric_cavity(
                front_points,
                front_triangles,
                full_rgb=color,
                full_world=full_world,
                full_depth=full_depth,
                full_valid=full_valid,
                camera_pos=camera_position,
                camera_rotation=camera_rotation,
                camera_forward=camera_forward,
                focal_px=float(full_focal_px),
                image_width=int(metric_depth.shape[1]),
                image_height=int(metric_depth.shape[0]),
                pixel_stride=int(pixel_stride),
                surface_edge_scale=float(surface_edge_scale),
                surface_edge_slack_m=slack_m,
                max_depth_jump_m=jump_m,
                front_offset_mm=float(front_offset_mm),
                default_extrusion_mm=float(back_extrusion_mm),
                cavity_min_nonplanarity=float(
                    topology_min_nonplanarity
                ),
                cavity_min_concavity_depth_mm=float(
                    topology_concavity_depth_mm
                ),
                cavity_min_component_vertices=int(
                    topology_min_component_vertices
                ),
                cavity_base_wall_scale=float(
                    cavity_base_wall_scale
                ),
                cavity_bottom_wall_boost_scale=float(
                    cavity_bottom_wall_boost_scale
                ),
                cavity_wall_decay_scale=float(
                    cavity_wall_decay_scale
                ),
                cavity_bottom_start_scale=float(
                    cavity_bottom_start_scale
                ),
                cavity_disk_start_scale=float(
                    cavity_disk_start_scale
                ),
                cavity_lower_wall_slope=float(
                    cavity_lower_wall_slope
                ),
                cavity_top_height_scale=float(
                    cavity_top_height_scale
                ),
                cavity_group_radius_scale=float(
                    cavity_group_radius_scale
                ),
                cavity_visible_shell_scale=float(
                    cavity_visible_shell_scale
                ),
                cavity_minimum_shell_mm=float(
                    cavity_minimum_shell_mm
                ),
                cavity_maximum_shell_mm=float(
                    cavity_maximum_shell_mm
                ),
                cavity_radial_bin_mm=float(cavity_radial_bin_mm),
                cavity_height_bin_mm=float(cavity_height_bin_mm),
                cavity_revolve_sections=int(cavity_revolve_sections),
                horizontal_support_shell_mm=float(
                    horizontal_support_shell_mm
                ),
                horizontal_support_min_vertices=int(
                    horizontal_support_min_vertices
                ),
                horizontal_support_max_nonplanarity=float(
                    horizontal_support_max_nonplanarity
                ),
                horizontal_support_min_normal_alignment=float(
                    horizontal_support_min_normal_alignment
                ),
                horizontal_support_min_extent_mm=float(
                    horizontal_support_min_extent_mm
                ),
                compact_symmetry_minimum_component_pixels=int(
                    compact_symmetry_minimum_component_pixels
                ),
                compact_symmetry_maximum_extent_mm=float(
                    compact_symmetry_maximum_extent_mm
                ),
                compact_symmetry_minimum_height_mm=float(
                    compact_symmetry_minimum_height_mm
                ),
                compact_symmetry_minimum_nonplanarity=float(
                    compact_symmetry_minimum_nonplanarity
                ),
                compact_symmetry_minimum_convex_prominence_mm=float(
                    compact_symmetry_minimum_convex_prominence_mm
                ),
                compact_symmetry_maximum_support_gap_mm=float(
                    compact_symmetry_maximum_support_gap_mm
                ),
                compact_symmetry_pitch_mm=float(
                    compact_symmetry_pitch_mm
                ),
                compact_symmetry_center_percentile=float(
                    compact_symmetry_center_percentile
                ),
                compact_symmetry_support_clearance_scale=float(
                    compact_symmetry_support_clearance_scale
                ),
                compact_symmetry_closing_radius_scale=float(
                    compact_symmetry_closing_radius_scale
                ),
                compact_symmetry_front_padding_mm=float(
                    compact_symmetry_front_padding_mm
                ),
                topology_convex_width_scale=float(
                    topology_convex_width_scale
                ),
                topology_max_extrusion_mm=float(
                    topology_max_extrusion_mm
                ),
                defer_topology_validation=bool(
                    _defer_topology_validation_to_warp_cuda
                ),
            )
        )
        method = (
            "whole_view_axisymmetric_cavity_and_generic_surface_hybrid"
        )
        nonwatertight_components = int(
            completion_metadata["nonwatertight_shell_components"]
        )
    elif selected_completion == "hybrid_thin_tubes":
        shell, shell_metadata = _close_front_surface_components(
            front_points,
            front_triangles,
            camera_pos=camera_position,
            camera_forward=camera_forward,
            front_offset_mm=float(front_offset_mm),
            back_extrusion_mm=float(back_extrusion_mm),
        )
        thin_mesh, thin_mesh_metadata = _thin_structure_tube_mesh(
            raw_mask=thin_raw_mask,
            completed_mask=thin_completed_mask,
            depth=metric_depth,
            camera_pos=camera_position,
            camera_rotation=camera_rotation,
            focal_px=float(focal_px),
            min_component_pixels=int(thin_min_component_pixels),
            max_radius_px=float(thin_max_radius_px),
            radius_scale=float(thin_radius_scale),
            min_radius_mm=float(thin_min_radius_mm),
            max_radius_mm=float(thin_max_radius_mm),
            center_offset_scale=float(thin_center_offset_scale),
            ring_samples=int(thin_ring_samples),
        )
        mesh = (
            shell
            if thin_mesh is None
            else trimesh.util.concatenate((shell, thin_mesh))
        )
        completion_metadata = {
            **shell_metadata,
            **thin_mask_metadata,
            **thin_mesh_metadata,
        }
        method = "whole_view_depth_shell_with_thin_protrusion_tubes"
        nonwatertight_components = int(
            shell_metadata["nonwatertight_shell_components"]
        )
    elif selected_completion == "hybrid_circle_tubes":
        shell, shell_metadata = _close_front_surface_components(
            front_points,
            front_triangles,
            camera_pos=camera_position,
            camera_forward=camera_forward,
            front_offset_mm=float(front_offset_mm),
            back_extrusion_mm=float(thin_background_shell_mm),
        )
        (
            full_world,
            _full_depth,
            _full_valid,
            _full_camera_points,
            _full_focal_px,
        ) = _sampled_world_points(
            metric_depth,
            camera_pos=camera_pos,
            camera_quat_xyzw=camera_quat_xyzw,
            focal_length=float(focal_length),
            horizontal_aperture=float(horizontal_aperture),
            pixel_stride=1,
            min_depth_m=float(min_depth_m),
            max_depth_m=float(max_depth_m),
        )
        thin_mesh, thin_mesh_metadata = (
            _thin_structure_circle_tube_mesh(
                raw_mask=thin_raw_mask,
                depth=metric_depth,
                world_points=full_world,
                camera_pos=camera_position,
                focal_px=float(focal_px),
                min_component_pixels=int(thin_min_component_pixels),
                max_radius_px=float(thin_max_radius_px),
                radius_scale=float(thin_radius_scale),
                radius_bias_px=float(thin_radius_bias_px),
                min_radius_mm=float(thin_min_radius_mm),
                max_radius_mm=float(thin_max_radius_mm),
                center_offset_scale=float(
                    thin_center_offset_scale
                ),
                ring_samples=int(thin_ring_samples),
                nonlinear_refine=bool(thin_nonlinear_refine),
            )
        )
        mesh = (
            shell
            if thin_mesh is None
            else trimesh.util.concatenate((shell, thin_mesh))
        )
        completion_metadata = {
            **shell_metadata,
            **thin_mask_metadata,
            **thin_mesh_metadata,
        }
        method = (
            "whole_view_thin_background_shell_with_circle_tubes"
        )
        nonwatertight_components = int(
            shell_metadata["nonwatertight_shell_components"]
        )
    elif selected_completion == "width_aware_hybrid":
        shell, shell_metadata = _close_front_surface_components(
            front_points,
            front_triangles,
            camera_pos=camera_position,
            camera_forward=camera_forward,
            front_offset_mm=float(front_offset_mm),
            back_extrusion_mm=float(back_extrusion_mm),
        )
        (
            full_world,
            _full_depth,
            _full_valid,
            _full_camera_points,
            _full_focal_px,
        ) = _sampled_world_points(
            metric_depth,
            camera_pos=camera_pos,
            camera_quat_xyzw=camera_quat_xyzw,
            focal_length=float(focal_length),
            horizontal_aperture=float(horizontal_aperture),
            pixel_stride=1,
            min_depth_m=float(min_depth_m),
            max_depth_m=float(max_depth_m),
        )
        thin_mesh, thin_mesh_metadata = (
            _thin_structure_circle_tube_mesh(
                raw_mask=thin_raw_mask,
                depth=metric_depth,
                world_points=full_world,
                camera_pos=camera_position,
                focal_px=float(focal_px),
                min_component_pixels=int(thin_min_component_pixels),
                max_radius_px=float(thin_max_radius_px),
                radius_scale=float(thin_radius_scale),
                radius_bias_px=float(thin_radius_bias_px),
                min_radius_mm=float(thin_min_radius_mm),
                max_radius_mm=float(thin_max_radius_mm),
                center_offset_scale=float(
                    thin_center_offset_scale
                ),
                ring_samples=int(thin_ring_samples),
                nonlinear_refine=bool(thin_nonlinear_refine),
            )
        )
        mesh = (
            shell
            if thin_mesh is None
            else trimesh.util.concatenate((shell, thin_mesh))
        )
        completion_metadata = {
            **shell_metadata,
            **thin_mask_metadata,
            **thin_mesh_metadata,
            "width_aware_background_extrusion_mm": float(
                back_extrusion_mm
            ),
        }
        method = "whole_view_depth_shell_with_width_fitted_circle_tubes"
        nonwatertight_components = int(
            shell_metadata["nonwatertight_shell_components"]
        )
    elif selected_completion == "support_column":
        mesh, completion_metadata = (
            _close_front_surface_components_support_column(
                front_points,
                front_triangles,
                camera_pos=camera_position,
                camera_rotation=camera_rotation,
                focal_px=float(focal_px),
                image_width=int(metric_depth.shape[1]),
                image_height=int(metric_depth.shape[0]),
                pixel_stride=int(pixel_stride),
                sampled_world=sampled_world,
                sampled_valid=valid,
                support_ring_px=int(support_ring_px),
                support_percentile=float(support_percentile),
                min_column_mm=float(support_min_column_mm),
                max_column_mm=float(support_max_column_mm),
            )
        )
        method = "whole_view_local_support_column_completion"
        nonwatertight_components = int(
            completion_metadata["nonwatertight_shell_components"]
        )
    elif selected_completion == "screened_poisson":
        mesh, completion_metadata = _screened_poisson_mesh(
            front_points,
            front_triangles,
            camera_pos=camera_position,
            poisson_depth=int(poisson_depth),
            poisson_full_depth=int(poisson_full_depth),
            poisson_scale=float(poisson_scale),
            poisson_samples_per_node=float(
                poisson_samples_per_node
            ),
            poisson_point_weight=float(poisson_point_weight),
            poisson_iterations=int(poisson_iterations),
            poisson_threads=int(poisson_threads),
        )
        method = "whole_view_screened_poisson"
        nonwatertight_components = int(
            completion_metadata["poisson_nonwatertight_components"]
        )
    elif selected_completion == "component_poisson":
        mesh, completion_metadata = _component_screened_poisson_mesh(
            front_points,
            front_triangles,
            camera_pos=camera_position,
            camera_forward=camera_forward,
            front_offset_mm=float(front_offset_mm),
            back_extrusion_mm=float(back_extrusion_mm),
            poisson_min_component_vertices=int(
                poisson_min_component_vertices
            ),
            poisson_depth=int(poisson_depth),
            poisson_full_depth=int(poisson_full_depth),
            poisson_scale=float(poisson_scale),
            poisson_samples_per_node=float(
                poisson_samples_per_node
            ),
            poisson_point_weight=float(poisson_point_weight),
            poisson_iterations=int(poisson_iterations),
            poisson_threads=int(poisson_threads),
        )
        method = "all_surface_components_screened_poisson"
        nonwatertight_components = int(
            completion_metadata["component_nonwatertight_components"]
        )
    else:
        raise ValueError(
            f"unsupported completion_method={completion_method!r}; "
            "expected 'organized_shell', 'normal_shell', "
            "'adaptive_shell', 'local_width_shell', "
            "'topology_hybrid', "
            "'axisymmetric_cavity_hybrid', "
            "'hybrid_thin_tubes', "
            "'hybrid_circle_tubes', 'width_aware_hybrid', "
            "'support_column', "
            "'screened_poisson', or "
            "'component_poisson'"
        )
    if len(mesh.faces) == 0:
        raise ValueError("reconstructed scene mesh is empty")
    if nonwatertight_components:
        raise ValueError(
            f"{selected_completion} produced "
            f"{nonwatertight_components} nonwatertight components"
        )

    if _defer_topology_validation_to_warp_cuda:
        mesh_component_count = None
        mesh_watertight = None
        topology_validation = "deferred_to_warp_cuda"
    else:
        components = mesh.split(only_watertight=False)
        mesh_component_count = int(len(components))
        mesh_watertight = bool(mesh.is_watertight)
        topology_validation = "trimesh_cpu"
    return RGBDSceneMeshResult(
        mesh=mesh,
        metadata={
            "build": RGBD_SCENE_MESH_BUILD,
            "method": method,
            "completion_method": selected_completion,
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
            "input_resolution": [
                int(metric_depth.shape[1]),
                int(metric_depth.shape[0]),
            ],
            "sampled_resolution": [
                int(sample_width),
                int(sample_height),
            ],
            "pixel_stride": int(pixel_stride),
            "valid_sample_pixels": int(valid.sum()),
            "organized_front_vertices": int(
                organized_front_vertex_count
            ),
            "front_vertices": int(vertex_count),
            "front_triangles": int(len(front_triangles)),
            **manifold_metadata,
            **completion_metadata,
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_faces": int(len(mesh.faces)),
            "mesh_components": mesh_component_count,
            "mesh_watertight": mesh_watertight,
            "mesh_topology_validation": topology_validation,
            "parameters": {
                "min_depth_m": float(min_depth_m),
                "max_depth_m": float(max_depth_m),
                "surface_edge_scale": float(surface_edge_scale),
                "surface_edge_slack_mm": float(surface_edge_slack_mm),
                "max_depth_jump_mm": float(max_depth_jump_mm),
                "rgbd_edge_color_threshold": float(
                    rgbd_edge_color_threshold
                ),
                "rgbd_edge_min_depth_jump_mm": float(
                    rgbd_edge_min_depth_jump_mm
                ),
                "back_extrusion_mm": float(back_extrusion_mm),
                "adaptive_min_extrusion_mm": float(
                    adaptive_min_extrusion_mm
                ),
                "adaptive_width_scale": float(adaptive_width_scale),
                "topology_concave_wall_mm": float(
                    topology_concave_wall_mm
                ),
                "topology_concavity_depth_mm": float(
                    topology_concavity_depth_mm
                ),
                "topology_convex_prominence_mm": float(
                    topology_convex_prominence_mm
                ),
                "topology_min_nonplanarity": float(
                    topology_min_nonplanarity
                ),
                "topology_convex_width_scale": float(
                    topology_convex_width_scale
                ),
                "topology_max_extrusion_mm": float(
                    topology_max_extrusion_mm
                ),
                "topology_min_component_vertices": int(
                    topology_min_component_vertices
                ),
                "cavity_base_wall_scale": float(
                    cavity_base_wall_scale
                ),
                "cavity_bottom_wall_boost_scale": float(
                    cavity_bottom_wall_boost_scale
                ),
                "cavity_wall_decay_scale": float(
                    cavity_wall_decay_scale
                ),
                "cavity_bottom_start_scale": float(
                    cavity_bottom_start_scale
                ),
                "cavity_disk_start_scale": float(
                    cavity_disk_start_scale
                ),
                "cavity_lower_wall_slope": float(
                    cavity_lower_wall_slope
                ),
                "cavity_top_height_scale": float(
                    cavity_top_height_scale
                ),
                "cavity_group_radius_scale": float(
                    cavity_group_radius_scale
                ),
                "cavity_visible_shell_scale": float(
                    cavity_visible_shell_scale
                ),
                "cavity_minimum_shell_mm": float(
                    cavity_minimum_shell_mm
                ),
                "cavity_maximum_shell_mm": float(
                    cavity_maximum_shell_mm
                ),
                "cavity_radial_bin_mm": float(
                    cavity_radial_bin_mm
                ),
                "cavity_height_bin_mm": float(
                    cavity_height_bin_mm
                ),
                "cavity_revolve_sections": int(
                    cavity_revolve_sections
                ),
                "horizontal_support_shell_mm": float(
                    horizontal_support_shell_mm
                ),
                "horizontal_support_min_vertices": int(
                    horizontal_support_min_vertices
                ),
                "horizontal_support_max_nonplanarity": float(
                    horizontal_support_max_nonplanarity
                ),
                "horizontal_support_min_normal_alignment": float(
                    horizontal_support_min_normal_alignment
                ),
                "horizontal_support_min_extent_mm": float(
                    horizontal_support_min_extent_mm
                ),
                "compact_symmetry_minimum_component_pixels": int(
                    compact_symmetry_minimum_component_pixels
                ),
                "compact_symmetry_maximum_extent_mm": float(
                    compact_symmetry_maximum_extent_mm
                ),
                "compact_symmetry_minimum_height_mm": float(
                    compact_symmetry_minimum_height_mm
                ),
                "compact_symmetry_minimum_nonplanarity": float(
                    compact_symmetry_minimum_nonplanarity
                ),
                "compact_symmetry_minimum_convex_prominence_mm": float(
                    compact_symmetry_minimum_convex_prominence_mm
                ),
                "compact_symmetry_maximum_support_gap_mm": float(
                    compact_symmetry_maximum_support_gap_mm
                ),
                "compact_symmetry_pitch_mm": float(
                    compact_symmetry_pitch_mm
                ),
                "compact_symmetry_center_percentile": float(
                    compact_symmetry_center_percentile
                ),
                "compact_symmetry_support_clearance_scale": float(
                    compact_symmetry_support_clearance_scale
                ),
                "compact_symmetry_closing_radius_scale": float(
                    compact_symmetry_closing_radius_scale
                ),
                "compact_symmetry_front_padding_mm": float(
                    compact_symmetry_front_padding_mm
                ),
                "local_width_windows_px": [
                    int(value)
                    for value in local_width_windows_px
                ],
                "local_width_min_prominence_mm": float(
                    local_width_min_prominence_mm
                ),
                "local_width_closing_iterations": int(
                    local_width_closing_iterations
                ),
                "local_width_radius_bias_px": float(
                    local_width_radius_bias_px
                ),
                "thin_background_window_px": int(
                    thin_background_window_px
                ),
                "thin_min_protrusion_mm": float(
                    thin_min_protrusion_mm
                ),
                "thin_closing_iterations": int(
                    thin_closing_iterations
                ),
                "thin_min_component_pixels": int(
                    thin_min_component_pixels
                ),
                "thin_max_radius_px": float(thin_max_radius_px),
                "thin_radius_scale": float(thin_radius_scale),
                "thin_radius_bias_px": float(thin_radius_bias_px),
                "thin_min_radius_mm": float(thin_min_radius_mm),
                "thin_max_radius_mm": float(thin_max_radius_mm),
                "thin_center_offset_scale": float(
                    thin_center_offset_scale
                ),
                "thin_ring_samples": int(thin_ring_samples),
                "thin_background_shell_mm": float(
                    thin_background_shell_mm
                ),
                "thin_nonlinear_refine": bool(
                    thin_nonlinear_refine
                ),
                "support_ring_px": int(support_ring_px),
                "support_percentile": float(support_percentile),
                "support_min_column_mm": float(
                    support_min_column_mm
                ),
                "support_max_column_mm": float(
                    support_max_column_mm
                ),
                "front_offset_mm": float(front_offset_mm),
                "poisson_depth": int(poisson_depth),
                "poisson_full_depth": int(poisson_full_depth),
                "poisson_scale": float(poisson_scale),
                "poisson_samples_per_node": float(
                    poisson_samples_per_node
                ),
                "poisson_point_weight": float(poisson_point_weight),
                "poisson_iterations": int(poisson_iterations),
                "poisson_threads": int(poisson_threads),
                "poisson_min_component_vertices": int(
                    poisson_min_component_vertices
                ),
            },
        },
    )


_v11_base_reconstruct_rgbd_scene_mesh = reconstruct_rgbd_scene_mesh


def _compute_support_structure_mask(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
) -> tuple[np.ndarray, Any | None, np.ndarray | None, Dict[str, Any]]:
    """Detect narrow support-relative geometry without object identity."""
    from .depth_mesh_reconstruction import (
        backproject_depth_world,
        fit_dominant_support_plane,
    )

    color = np.asarray(rgb, dtype=np.float64)
    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    height, width = metric_depth.shape
    use_cuda_support_plane = os.environ.get(
        "OFFICIAL_V2_V53_GPU_SUPPORT_PLANE",
        "0",
    ).strip().lower() not in {"0", "false", "no", "off"}
    support_plane_runtime_metadata: Dict[str, Any] = {
        "support_plane_backend": "numpy_cpu_reference",
        "support_plane_device": "cpu",
    }
    try:
        if use_cuda_support_plane:
            from .rgbd_cuda_support_plane import (
                backproject_and_fit_support_plane_cuda,
            )

            world, valid, plane, support_plane_runtime_metadata = (
                backproject_and_fit_support_plane_cuda(
                    metric_depth,
                    camera_pos=camera_pos,
                    camera_quat_xyzw=camera_quat_xyzw,
                    focal_length=float(focal_length),
                    horizontal_aperture=float(horizontal_aperture),
                    roi=(0, 0, width, height),
                    requested_device=os.environ.get(
                        "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE",
                        "cuda:0",
                    ),
                    allow_cpu_reference=False,
                )
            )
        else:
            world, valid = backproject_depth_world(
                metric_depth,
                camera_pos=camera_pos,
                camera_quat_xyzw=camera_quat_xyzw,
                focal_length=float(focal_length),
                horizontal_aperture=float(horizontal_aperture),
            )
            plane = fit_dominant_support_plane(
                world,
                valid,
                roi=(0, 0, width, height),
            )
    except ValueError as error:
        if "world" not in locals():
            world, _valid = backproject_depth_world(
                metric_depth,
                camera_pos=camera_pos,
                camera_quat_xyzw=camera_quat_xyzw,
                focal_length=float(focal_length),
                horizontal_aperture=float(horizontal_aperture),
            )
        return (
            np.zeros(metric_depth.shape, dtype=bool),
            None,
            world,
            {
                **support_plane_runtime_metadata,
                "support_plane_detected": False,
                "support_plane_failure": str(error),
                "thin_structure_detected": False,
                "thin_structure_pixels": 0,
            },
        )

    signed_height = plane.signed_height(world)
    candidate = (
        valid
        & (signed_height >= 0.0005)
        & (signed_height <= 0.6)
    )
    labels, component_count = ndimage.label(
        candidate,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    component_slices = ndimage.find_objects(labels)
    component_sizes = np.bincount(
        labels.reshape(-1),
        minlength=int(component_count) + 1,
    )
    rows: list[Dict[str, Any]] = []
    primary_ids: set[int] = set()
    for component_id in range(1, int(component_count) + 1):
        component_slice = component_slices[component_id - 1]
        area = int(component_sizes[component_id])
        if area < 1 or component_slice is None:
            continue
        local_component = labels[component_slice] == component_id
        # A one-pixel zero border reproduces the implicit background of the
        # full image while avoiding one full-resolution EDT per component.
        padded_component = np.pad(
            local_component,
            1,
            mode="constant",
            constant_values=False,
        )
        skeleton_pixels = int(skeletonize(padded_component).sum())
        radius_px = float(
            ndimage.distance_transform_edt(padded_component).max(
                initial=0.0
            )
        )
        area_per_skeleton = float(
            area / max(skeleton_pixels, 1)
        )
        row = {
            "component_id": int(component_id),
            "area_pixels": area,
            "skeleton_pixels": skeleton_pixels,
            "area_per_skeleton": area_per_skeleton,
            "maximum_radius_px": radius_px,
        }
        rows.append(row)
        if (
            skeleton_pixels >= 50
            and area_per_skeleton <= 4.0
            and radius_px <= 6.0
        ):
            primary_ids.add(int(component_id))

    selected = np.isin(
        labels,
        np.asarray(sorted(primary_ids), dtype=np.int32),
    )
    expanded_ids = set(primary_ids)
    if selected.any():
        distance, nearest = ndimage.distance_transform_edt(
            ~selected,
            return_indices=True,
        )
        added: set[int] = set()
        for row in rows:
            component_id = int(row["component_id"])
            if component_id in expanded_ids:
                continue
            if (
                int(row["area_pixels"]) > 256
                or float(row["area_per_skeleton"]) > 4.0
                or float(row["maximum_radius_px"]) > 6.0
            ):
                continue
            component_slice = component_slices[component_id - 1]
            if component_slice is None:
                continue
            local_component = labels[component_slice] == component_id
            local_y, local_x = np.nonzero(local_component)
            cy = local_y + int(component_slice[0].start)
            cx = local_x + int(component_slice[1].start)
            if not len(cy):
                continue
            local_distance = distance[cy, cx]
            best = int(np.argmin(local_distance))
            if float(local_distance[best]) > 12.0:
                continue
            y = int(cy[best])
            x = int(cx[best])
            near_y = int(nearest[0, y, x])
            near_x = int(nearest[1, y, x])
            world_gap = float(
                np.linalg.norm(
                    world[y, x] - world[near_y, near_x]
                )
            )
            color_gap = float(
                np.linalg.norm(
                    color[y, x, :3] - color[near_y, near_x, :3]
                )
            )
            if world_gap <= 0.02 and color_gap <= 80.0:
                added.add(component_id)
        expanded_ids.update(added)
        selected |= np.isin(
            labels,
            np.asarray(sorted(added), dtype=np.int32),
        )

    return selected, plane, world, {
        **support_plane_runtime_metadata,
        "support_plane_detected": True,
        "support_plane_normal": np.asarray(plane.normal).tolist(),
        "support_plane_offset": float(plane.offset),
        "support_plane_residual_median_mm": float(
            plane.residual_median_mm
        ),
        "support_plane_residual_p95_mm": float(
            plane.residual_p95_mm
        ),
        "support_relative_candidate_pixels": int(candidate.sum()),
        "support_relative_component_count": int(component_count),
        "thin_structure_primary_components": sorted(primary_ids),
        "thin_structure_expanded_components": sorted(expanded_ids),
        "thin_structure_detected": bool(selected.any()),
        "thin_structure_pixels": int(selected.sum()),
        "thin_structure_components": rows,
        "rgb_continuity_used": True,
    }


def _support_structure_cache_key(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
) -> str:
    digest = hashlib.sha256()
    color = np.ascontiguousarray(np.asarray(rgb))
    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    metric_depth = np.ascontiguousarray(metric_depth)
    for array in (color, metric_depth):
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(array).cast("B"))
    for value in np.concatenate(
        (
            np.asarray(camera_pos, dtype=np.float64).reshape(3),
            np.asarray(camera_quat_xyzw, dtype=np.float64).reshape(4),
            np.asarray(
                [focal_length, horizontal_aperture],
                dtype=np.float64,
            ),
        )
    ):
        digest.update(float(value).hex().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def clear_support_structure_mask_cache() -> None:
    with _SUPPORT_STRUCTURE_CACHE_LOCK:
        _SUPPORT_STRUCTURE_CACHE.clear()


def _support_structure_mask(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
) -> tuple[np.ndarray, Any | None, np.ndarray | None, Dict[str, Any]]:
    enabled = os.environ.get(
        "OFFICIAL_V2_V53_SUPPORT_STRUCTURE_CACHE",
        "0",
    ).strip().lower() not in {"0", "false", "no", "off"}
    kwargs = {
        "camera_pos": camera_pos,
        "camera_quat_xyzw": camera_quat_xyzw,
        "focal_length": float(focal_length),
        "horizontal_aperture": float(horizontal_aperture),
    }
    if not enabled:
        return _compute_support_structure_mask(rgb, depth, **kwargs)
    key = _support_structure_cache_key(rgb, depth, **kwargs)
    limit = max(
        1,
        int(os.environ.get("OFFICIAL_V2_V53_SUPPORT_STRUCTURE_CACHE_SIZE", "2")),
    )
    with _SUPPORT_STRUCTURE_CACHE_LOCK:
        cached = _SUPPORT_STRUCTURE_CACHE.get(key)
        cache_hit = cached is not None
        if cached is None:
            cached = _compute_support_structure_mask(rgb, depth, **kwargs)
            _SUPPORT_STRUCTURE_CACHE[key] = cached
            _SUPPORT_STRUCTURE_CACHE.move_to_end(key)
            while len(_SUPPORT_STRUCTURE_CACHE) > limit:
                _SUPPORT_STRUCTURE_CACHE.popitem(last=False)
        else:
            _SUPPORT_STRUCTURE_CACHE.move_to_end(key)
        mask, plane, world, metadata = cached
        return mask, plane, world, {
            **dict(metadata),
            "support_structure_cache_hit": bool(cache_hit),
            "support_structure_cache_key": key[:16],
        }


def _rgb_guided_thin_structure_completion(
    rgb: np.ndarray,
    depth: np.ndarray,
    reliable_mask: np.ndarray,
    plane: Any,
    world: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Recover subpixel thin ridges and infer their support-relative depth."""
    from scipy.spatial import cKDTree
    from skimage.color import rgb2gray, rgb2lab
    from skimage.filters import frangi

    reliable = np.asarray(reliable_mask, dtype=bool)
    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    if not reliable.any():
        return reliable, metric_depth.copy(), {
            "rgb_thin_completion_enabled": False,
            "rgb_thin_completion_reason": "no_reliable_structure",
            "rgb_thin_completion_pixels": 0,
        }

    color = np.asarray(rgb, dtype=np.float64)
    if color.max(initial=0.0) > 1.5:
        color = color / 255.0
    gray = rgb2gray(color[..., :3])
    lab = rgb2lab(color[..., :3])
    local_background = ndimage.gaussian_filter(gray, sigma=3.0)
    dark_contrast = np.maximum(local_background - gray, 0.0)
    ridge = frangi(
        gray,
        sigmas=(0.8, 1.2, 1.8, 2.5),
        black_ridges=True,
    )

    seed_contrast = dark_contrast[reliable]
    contrast_threshold = float(np.percentile(seed_contrast, 5.0))
    ridge_threshold = float(np.percentile(ridge[reliable], 35.0))
    color_seed = reliable & (
        dark_contrast >= float(np.percentile(seed_contrast, 10.0))
    )
    seed_colors = lab[color_seed]
    if len(seed_colors) < 8:
        seed_colors = lab[reliable]
    sample_step = max(1, int(math.ceil(len(seed_colors) / 2048)))
    color_distance, _nearest_color = cKDTree(
        seed_colors[::sample_step]
    ).query(lab.reshape(-1, 3), k=1)
    color_distance = color_distance.reshape(reliable.shape)

    radius_px = ndimage.distance_transform_edt(reliable)
    maximum_completion_distance_px = float(
        np.clip(6.0 * radius_px.max(initial=0.0), 8.0, 24.0)
    )
    distance_to_reliable = ndimage.distance_transform_edt(~reliable)
    ridge_candidate = (
        (color_distance <= 6.0)
        & (ridge >= ridge_threshold)
        & (dark_contrast >= contrast_threshold)
        & (distance_to_reliable <= maximum_completion_distance_px)
    )
    joined = ndimage.binary_closing(
        ridge_candidate | reliable,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    joined_labels, _joined_count = ndimage.label(
        joined,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    touching_ids = np.unique(joined_labels[reliable])
    touching_ids = touching_ids[touching_ids > 0]
    connected = np.isin(joined_labels, touching_ids)
    connected &= (
        distance_to_reliable <= maximum_completion_distance_px
    )
    completed_skeleton = skeletonize(connected)
    proposed_addition = completed_skeleton & (~reliable)

    signed_height = plane.signed_height(world)
    _distance, nearest = ndimage.distance_transform_edt(
        ~reliable,
        return_indices=True,
    )
    nearest_height = signed_height[nearest[0], nearest[1]]
    height, width = metric_depth.shape
    vv, uu = np.indices((height, width), dtype=np.float64)
    focal_px = (
        float(focal_length)
        / float(horizontal_aperture)
        * float(width)
    )
    rays_camera = np.stack(
        (
            (uu - width / 2.0) / focal_px,
            -(vv - height / 2.0) / focal_px,
            -np.ones_like(uu),
        ),
        axis=-1,
    )
    rotation = _quat_to_mat_xyzw(camera_quat_xyzw)
    rays_world = rays_camera @ rotation.T
    normal = np.asarray(plane.normal, dtype=np.float64).reshape(3)
    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    denominator = np.einsum("...i,i->...", rays_world, normal)
    numerator = (
        nearest_height
        - float(np.dot(camera, normal))
        - float(plane.offset)
    )
    estimated_depth = np.divide(
        numerator,
        denominator,
        out=np.full(metric_depth.shape, np.nan, dtype=np.float64),
        where=np.abs(denominator) > 1e-9,
    )
    valid_addition = (
        proposed_addition
        & np.isfinite(estimated_depth)
        & (estimated_depth > 0.03)
        & (estimated_depth < 20.0)
    )
    completed_mask = reliable | valid_addition
    completed_depth = metric_depth.copy()
    completed_depth[valid_addition] = estimated_depth[valid_addition]
    return completed_mask, completed_depth, {
        "rgb_thin_completion_enabled": True,
        "rgb_thin_completion_color_space": "CIELAB",
        "rgb_thin_completion_color_distance": 6.0,
        "rgb_thin_completion_ridge_threshold": ridge_threshold,
        "rgb_thin_completion_contrast_threshold": contrast_threshold,
        "rgb_thin_completion_max_distance_px": (
            maximum_completion_distance_px
        ),
        "rgb_thin_completion_candidate_pixels": int(
            ridge_candidate.sum()
        ),
        "rgb_thin_completion_connected_pixels": int(connected.sum()),
        "rgb_thin_completion_pixels": int(valid_addition.sum()),
        "rgb_thin_completion_total_pixels": int(completed_mask.sum()),
        "rgb_thin_completion_depth_model": (
            "nearest_reliable_support_relative_height"
        ),
    }


def _support_slab_mesh(
    plane: Any,
    world: np.ndarray,
    valid: np.ndarray,
    *,
    thickness_m: float = 0.08,
    lateral_margin_m: float = 0.05,
) -> trimesh.Trimesh | None:
    signed = plane.signed_height(world)
    inliers = np.asarray(valid, dtype=bool) & (np.abs(signed) <= 0.003)
    points = np.asarray(world[inliers], dtype=np.float64)
    if len(points) < 100:
        return None
    relative = points - np.asarray(plane.origin).reshape(1, 3)
    local_x = relative @ np.asarray(plane.basis_x)
    local_y = relative @ np.asarray(plane.basis_y)
    lower_x, upper_x = np.percentile(local_x, [0.25, 99.75])
    lower_y, upper_y = np.percentile(local_y, [0.25, 99.75])
    extent_x = float(upper_x - lower_x + 2.0 * lateral_margin_m)
    extent_y = float(upper_y - lower_y + 2.0 * lateral_margin_m)
    if extent_x <= 0.0 or extent_y <= 0.0:
        return None
    center_x = float(0.5 * (lower_x + upper_x))
    center_y = float(0.5 * (lower_y + upper_y))
    mesh = trimesh.creation.box(
        extents=(extent_x, extent_y, float(thickness_m)),
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack(
        (
            np.asarray(plane.basis_x, dtype=np.float64),
            np.asarray(plane.basis_y, dtype=np.float64),
            np.asarray(plane.normal, dtype=np.float64),
        )
    )
    transform[:3, 3] = (
        np.asarray(plane.origin, dtype=np.float64)
        + np.asarray(plane.basis_x, dtype=np.float64) * center_x
        + np.asarray(plane.basis_y, dtype=np.float64) * center_y
        - np.asarray(plane.normal, dtype=np.float64)
        * (0.5 * float(thickness_m))
    )
    mesh.apply_transform(transform)
    return mesh


def _convex_planar_prism_mesh(
    polygon_xy: np.ndarray,
    *,
    top_coefficients: np.ndarray,
    bottom_coefficients: np.ndarray,
) -> trimesh.Trimesh:
    """Close a convex horizontal footprint between two observed planes."""
    polygon = np.asarray(polygon_xy, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < 3:
        raise ValueError("planar prism footprint needs at least three points")
    top_fit = np.asarray(top_coefficients, dtype=np.float64).reshape(3)
    bottom_fit = np.asarray(
        bottom_coefficients,
        dtype=np.float64,
    ).reshape(3)
    design = np.column_stack(
        (polygon, np.ones(len(polygon), dtype=np.float64))
    )
    top_z = design @ top_fit
    bottom_z = design @ bottom_fit
    if np.any(top_z <= bottom_z + 1e-4):
        raise ValueError("planar prism top is not above its support")
    top = np.column_stack((polygon, top_z))
    bottom = np.column_stack((polygon, bottom_z))
    count = len(polygon)
    faces: list[list[int]] = []
    for index in range(1, count - 1):
        faces.append([0, index, index + 1])
        faces.append([count, count + index + 1, count + index])
    for index in range(count):
        following = (index + 1) % count
        faces.append([index, count + index, count + following])
        faces.append([index, count + following, following])
    mesh = trimesh.Trimesh(
        vertices=np.vstack((top, bottom)),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.process(validate=True)
    trimesh.repair.fix_normals(mesh)
    if float(mesh.volume) < 0.0:
        mesh.invert()
    if not mesh.is_watertight:
        raise ValueError("planar prism mesh is not watertight")
    return mesh


def _planar_prism_visible_free_space_audit(
    mesh: trimesh.Trimesh,
    world: np.ndarray,
    valid: np.ndarray,
    *,
    camera_pos: Sequence[float],
    focal_px: float,
    minimum_clearance_mm: float = 8.0,
    clearance_pixels: float = 3.0,
    maximum_violation_fraction: float = 0.05,
    minimum_violation_pixels: int = 64,
    ray_chunk_size: int = 32768,
) -> Dict[str, Any]:
    """Reject a completion that occludes surfaces already seen by depth.

    Positive front error means that the proposed solid intersects a camera
    ray before its measured depth endpoint. A real opaque completion may have
    small edge and fitting residuals, but it cannot occupy a broad region of
    projectively observed free space.
    """
    points_image = np.asarray(world, dtype=np.float64)
    selected = np.asarray(valid, dtype=bool)
    if points_image.shape[:2] != selected.shape:
        raise ValueError("visibility audit world grid and mask do not match")
    selected &= np.all(np.isfinite(points_image), axis=-1)
    points = points_image[selected]
    origin = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    delta = points - origin.reshape(1, 3)
    observed_distance = np.linalg.norm(delta, axis=1)
    usable = np.isfinite(observed_distance) & (observed_distance > 0.03)
    points = points[usable]
    observed_distance = observed_distance[usable]
    if not len(points):
        return {
            "enabled": True,
            "accepted": False,
            "reason": "no_valid_observed_rays",
            "tested_rays": 0,
            "candidate_hit_pixels": 0,
            "violation_pixels": 0,
            "violation_fraction": 0.0,
        }

    directions = (points - origin.reshape(1, 3)) / observed_distance[:, None]
    triangles = np.asarray(mesh.vertices, dtype=np.float64)[
        np.asarray(mesh.faces, dtype=np.int64)
    ]
    if not len(triangles):
        return {
            "enabled": True,
            "accepted": False,
            "reason": "candidate_mesh_has_no_faces",
            "tested_rays": int(len(points)),
            "candidate_hit_pixels": 0,
            "violation_pixels": 0,
            "violation_fraction": 0.0,
        }

    vertex_0 = triangles[:, 0]
    edge_1 = triangles[:, 1] - vertex_0
    edge_2 = triangles[:, 2] - vertex_0
    origin_edge = origin.reshape(1, 3) - vertex_0
    cross_origin_edge = np.cross(origin_edge, edge_1)
    distance_numerator = np.einsum(
        "ij,ij->i",
        edge_2,
        cross_origin_edge,
    )
    nearest = np.full(len(points), np.inf, dtype=np.float64)
    # Bound temporary ray-triangle arrays for non-rectangular convex hulls.
    pair_budget = 1_500_000
    chunk_size = min(
        max(int(ray_chunk_size), 1),
        max(256, pair_budget // max(len(triangles), 1)),
    )
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        chunk = directions[start:end]
        cross_direction = np.cross(
            chunk[:, None, :],
            edge_2[None, :, :],
        )
        determinant = np.einsum(
            "ij,nij->ni",
            edge_1,
            cross_direction,
        )
        nonparallel = np.abs(determinant) > 1e-12
        inverse = np.zeros_like(determinant)
        inverse[nonparallel] = 1.0 / determinant[nonparallel]
        barycentric_u = np.einsum(
            "ij,nij->ni",
            origin_edge,
            cross_direction,
        ) * inverse
        barycentric_v = np.einsum(
            "nj,ij->ni",
            chunk,
            cross_origin_edge,
        ) * inverse
        distance = distance_numerator[None, :] * inverse
        intersects = (
            nonparallel
            & (barycentric_u >= -1e-9)
            & (barycentric_v >= -1e-9)
            & (barycentric_u + barycentric_v <= 1.0 + 1e-9)
            & (distance > 1e-9)
        )
        distance[~intersects] = np.inf
        nearest[start:end] = np.min(distance, axis=1)

    candidate_hit = np.isfinite(nearest)
    hit_count = int(candidate_hit.sum())
    if not hit_count:
        return {
            "enabled": True,
            "accepted": False,
            "reason": "candidate_not_visible_in_observed_rays",
            "tested_rays": int(len(points)),
            "candidate_hit_pixels": 0,
            "violation_pixels": 0,
            "violation_fraction": 0.0,
        }

    pixel_footprint_mm = float(
        np.median(observed_distance[candidate_hit])
        / max(float(focal_px), 1e-12)
        * 1000.0
    )
    effective_clearance_mm = max(
        float(minimum_clearance_mm),
        float(clearance_pixels) * pixel_footprint_mm,
    )
    front_error_mm = (observed_distance - nearest) * 1000.0
    violation = candidate_hit & (
        front_error_mm > effective_clearance_mm
    )
    violation_count = int(violation.sum())
    violation_fraction = float(violation_count / hit_count)
    rejected = bool(
        violation_count >= int(minimum_violation_pixels)
        and violation_fraction > float(maximum_violation_fraction)
    )
    hit_errors = front_error_mm[candidate_hit]
    percentiles = np.percentile(
        hit_errors,
        [50.0, 90.0, 95.0, 99.0, 100.0],
    )
    return {
        "enabled": True,
        "accepted": not rejected,
        "reason": (
            "observed_free_space_violation"
            if rejected
            else "consistent_with_observed_depth"
        ),
        "tested_rays": int(len(points)),
        "candidate_hit_pixels": hit_count,
        "violation_pixels": violation_count,
        "violation_fraction": violation_fraction,
        "maximum_violation_fraction": float(
            maximum_violation_fraction
        ),
        "minimum_violation_pixels": int(minimum_violation_pixels),
        "minimum_clearance_mm": float(minimum_clearance_mm),
        "clearance_pixels": float(clearance_pixels),
        "pixel_footprint_mm": pixel_footprint_mm,
        "effective_clearance_mm": effective_clearance_mm,
        "front_error_mm_percentiles": {
            "p50": float(percentiles[0]),
            "p90": float(percentiles[1]),
            "p95": float(percentiles[2]),
            "p99": float(percentiles[3]),
            "max": float(percentiles[4]),
        },
    }


def _robust_convex_footprint(
    points_xy: np.ndarray,
    *,
    trim_percent: float = 0.25,
) -> Dict[str, Any]:
    """Return a convex footprint after trimming sparse planar outliers."""
    from scipy.spatial import ConvexHull, QhullError

    xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) < 8:
        raise ValueError("robust footprint needs at least eight points")
    center = np.median(xy, axis=0)
    centered = xy - center.reshape(1, 2)
    if float(np.linalg.norm(centered)) < 1e-9:
        raise ValueError("robust footprint points are degenerate")
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    local = centered @ vh.T
    lower, upper = np.percentile(
        local,
        [float(trim_percent), 100.0 - float(trim_percent)],
        axis=0,
    )
    robust = np.all(
        (local >= lower.reshape(1, 2))
        & (local <= upper.reshape(1, 2)),
        axis=1,
    )
    robust_xy = xy[robust]
    if len(robust_xy) < 8:
        raise ValueError("robust footprint has too few inliers")
    try:
        hull = ConvexHull(robust_xy)
    except QhullError as error:
        raise ValueError("robust footprint hull is degenerate") from error
    extents = np.asarray(upper - lower, dtype=np.float64)
    hull_xy = robust_xy[np.asarray(hull.vertices, dtype=np.int64)]
    rectangle_transform, rectangle_extents = (
        trimesh.bounds.oriented_bounds_2D(hull_xy)
    )
    rectangle_extents = np.asarray(
        rectangle_extents,
        dtype=np.float64,
    )
    rectangle_area = float(np.prod(rectangle_extents))
    rectangle_scale = float(
        np.sqrt(float(hull.volume) / max(rectangle_area, 1e-12))
    )
    half_extents = 0.5 * rectangle_extents * rectangle_scale
    rectangle_local = np.array(
        [
            [-half_extents[0], -half_extents[1]],
            [half_extents[0], -half_extents[1]],
            [half_extents[0], half_extents[1]],
            [-half_extents[0], half_extents[1]],
        ],
        dtype=np.float64,
    )
    equal_area_rectangle_xy = trimesh.transform_points(
        rectangle_local,
        np.linalg.inv(rectangle_transform),
    )
    return {
        "center_xy": robust_xy.mean(axis=0),
        "extents": extents,
        "hull_xy": hull_xy,
        "hull_area": float(hull.volume),
        "rectangularity": float(
            hull.volume / max(float(np.prod(extents)), 1e-12)
        ),
        "oriented_rectangularity": float(
            hull.volume / max(rectangle_area, 1e-12)
        ),
        "equal_area_rectangle_xy": equal_area_rectangle_xy,
        "minimum_rectangle_extents": rectangle_extents,
        "minimum_rectangle_area": rectangle_area,
        "rectangle_area_scale": rectangle_scale,
        "input_points": int(len(xy)),
        "inlier_points": int(len(robust_xy)),
    }


def _regularize_low_profile_frustum_shells(
    mesh: trimesh.Trimesh,
    topology_metadata: Dict[str, Any],
    plane: Any,
    *,
    image_width: int,
    image_height: int,
    enabled: bool = True,
    minimum_front_vertices: int = 800,
    minimum_horizontal_extent_mm: float = 40.0,
    maximum_horizontal_extent_mm: float = 500.0,
    maximum_height_mm: float = 100.0,
    maximum_height_to_width: float = 0.30,
    maximum_axis_ratio: float = 1.50,
    maximum_oriented_rectangularity: float = 0.90,
    hidden_thickness_height_scale: float = 0.60,
    radial_expansion_height_scale: float = 0.10,
    boundary_margin_px: int = 4,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Turn shallow near-isotropic ray shells into observable frusta.

    A single camera-ray extrusion concentrates hidden volume behind the
    visible center and leaves the lower silhouette too narrow. For a fully
    observed, shallow, near-isotropic cap, shorten the hidden displacement
    according to its visible height and expand the hidden layer radially in
    the detected support plane. Rectangular caps are excluded because their
    footprint is handled by the planar-prism prior.
    """
    parameters = {
        "enabled": bool(enabled),
        "minimum_front_vertices": int(minimum_front_vertices),
        "minimum_horizontal_extent_mm": float(
            minimum_horizontal_extent_mm
        ),
        "maximum_horizontal_extent_mm": float(
            maximum_horizontal_extent_mm
        ),
        "maximum_height_mm": float(maximum_height_mm),
        "maximum_height_to_width": float(maximum_height_to_width),
        "maximum_axis_ratio": float(maximum_axis_ratio),
        "maximum_oriented_rectangularity": float(
            maximum_oriented_rectangularity
        ),
        "hidden_thickness_height_scale": float(
            hidden_thickness_height_scale
        ),
        "radial_expansion_height_scale": float(
            radial_expansion_height_scale
        ),
        "boundary_margin_px": int(boundary_margin_px),
    }
    if not enabled:
        return mesh, {
            "low_profile_frustum_generated_components": 0,
            "low_profile_frustum_reason": "disabled",
            "low_profile_frustum_parameters": parameters,
            "low_profile_frustum_components": [],
        }
    rows = topology_metadata.get("topology_components")
    if not isinstance(rows, list):
        return mesh, {
            "low_profile_frustum_generated_components": 0,
            "low_profile_frustum_reason": "no_topology_components",
            "low_profile_frustum_parameters": parameters,
            "low_profile_frustum_components": [],
        }
    components = list(mesh.split(only_watertight=False))
    if len(components) != len(rows):
        return mesh, {
            "low_profile_frustum_generated_components": 0,
            "low_profile_frustum_reason": "component_count_mismatch",
            "low_profile_frustum_parameters": parameters,
            "low_profile_frustum_components": [],
        }
    if (
        float(hidden_thickness_height_scale) <= 0.0
        or float(radial_expansion_height_scale) < 0.0
    ):
        raise ValueError("low-profile frustum scales must be nonnegative")

    origin = np.asarray(plane.origin, dtype=np.float64).reshape(3)
    basis_x = np.asarray(plane.basis_x, dtype=np.float64).reshape(3)
    basis_y = np.asarray(plane.basis_y, dtype=np.float64).reshape(3)
    normal = np.asarray(plane.normal, dtype=np.float64).reshape(3)
    transformed: list[trimesh.Trimesh] = []
    selected_rows: list[Dict[str, Any]] = []
    margin = int(boundary_margin_px)
    for component_index, (component, row) in enumerate(
        zip(components, rows)
    ):
        front_count = int(row.get("vertex_count", 0))
        vertices = np.asarray(component.vertices, dtype=np.float64)
        bbox = np.asarray(row.get("bbox_uv", []), dtype=np.float64)
        layout_valid = bool(
            front_count > 0
            and len(vertices) == 2 * front_count
            and bbox.shape == (4,)
        )
        if not layout_valid:
            transformed.append(component)
            continue
        front = vertices[:front_count]
        back = vertices[front_count:]
        relative = front - origin.reshape(1, 3)
        local_x = relative @ basis_x
        local_y = relative @ basis_y
        local_height = relative @ normal
        extent_x = float(np.ptp(local_x))
        extent_y = float(np.ptp(local_y))
        height_extent = float(np.ptp(local_height))
        horizontal_min = min(extent_x, extent_y)
        horizontal_max = max(extent_x, extent_y)
        axis_ratio = horizontal_max / max(horizontal_min, 1e-9)
        height_to_width = height_extent / max(horizontal_min, 1e-9)
        fully_observed = bool(
            bbox[0] > margin
            and bbox[1] > margin
            and bbox[2] < int(image_width) - 1 - margin
            and bbox[3] < int(image_height) - 1 - margin
        )
        candidate = bool(
            fully_observed
            and front_count >= int(minimum_front_vertices)
            and horizontal_min
            >= float(minimum_horizontal_extent_mm) / 1000.0
            and horizontal_max
            <= float(maximum_horizontal_extent_mm) / 1000.0
            and height_extent
            <= float(maximum_height_mm) / 1000.0
            and height_to_width <= float(maximum_height_to_width)
            and axis_ratio <= float(maximum_axis_ratio)
        )
        footprint: Dict[str, Any] | None = None
        if candidate:
            try:
                footprint = _robust_convex_footprint(
                    np.column_stack((local_x, local_y))
                )
            except ValueError:
                candidate = False
        oriented_rectangularity = (
            float(footprint["oriented_rectangularity"])
            if footprint is not None
            else None
        )
        if (
            candidate
            and oriented_rectangularity is not None
            and oriented_rectangularity
            > float(maximum_oriented_rectangularity)
        ):
            candidate = False
        extrusion_mm = float(row.get("extrusion_mm", 0.0))
        if not candidate or extrusion_mm <= 0.0:
            transformed.append(component)
            continue

        visible_height_mm = height_extent * 1000.0
        hidden_thickness_mm = float(
            np.clip(
                float(hidden_thickness_height_scale)
                * visible_height_mm,
                1.0,
                extrusion_mm,
            )
        )
        displacement_scale = hidden_thickness_mm / extrusion_mm
        expansion_mm = float(
            radial_expansion_height_scale
        ) * visible_height_mm
        changed_vertices = vertices.copy()
        changed_back = front + displacement_scale * (back - front)
        changed_relative = changed_back - origin.reshape(1, 3)
        changed_x = changed_relative @ basis_x
        changed_y = changed_relative @ basis_y
        center_x = float(np.median(local_x))
        center_y = float(np.median(local_y))
        radial = (
            (changed_x - center_x)[:, None] * basis_x.reshape(1, 3)
            + (changed_y - center_y)[:, None]
            * basis_y.reshape(1, 3)
        )
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        radial_unit = np.divide(
            radial,
            radial_norm,
            out=np.zeros_like(radial),
            where=radial_norm > 1e-9,
        )
        changed_back += radial_unit * (expansion_mm / 1000.0)
        changed_vertices[front_count:] = changed_back
        changed = trimesh.Trimesh(
            vertices=changed_vertices,
            faces=np.asarray(component.faces, dtype=np.int64),
            process=False,
        )
        if not changed.is_watertight:
            transformed.append(component)
            continue
        transformed.append(changed)
        selected_rows.append(
            {
                "component_index": int(component_index),
                "front_vertices": front_count,
                "horizontal_extents_mm": [
                    extent_x * 1000.0,
                    extent_y * 1000.0,
                ],
                "visible_height_mm": visible_height_mm,
                "height_to_min_width": height_to_width,
                "axis_ratio": axis_ratio,
                "oriented_rectangularity": oriented_rectangularity,
                "original_extrusion_mm": extrusion_mm,
                "hidden_thickness_mm": hidden_thickness_mm,
                "radial_expansion_mm": expansion_mm,
            }
        )
    regularized = trimesh.util.concatenate(transformed)
    return regularized, {
        "low_profile_frustum_generated_components": int(
            len(selected_rows)
        ),
        "low_profile_frustum_reason": (
            "generated" if selected_rows else "no_eligible_component"
        ),
        "low_profile_frustum_parameters": parameters,
        "low_profile_frustum_components": selected_rows,
    }


def _elevated_planar_prism_meshes(
    world: np.ndarray,
    valid: np.ndarray,
    *,
    plane_bin_mm: float = 2.0,
    plane_band_mm: float = 3.0,
    minimum_plane_pixels: int = 1200,
    minimum_cap_pixels: int = 3000,
    minimum_cap_extent_mm: float = 250.0,
    maximum_cap_extent_mm: float = 800.0,
    minimum_cap_area_mm2: float = 60_000.0,
    minimum_rectangularity: float = 0.55,
    rectangle_regularization_threshold: float = 0.90,
    minimum_height_mm: float = 120.0,
    maximum_height_mm: float = 500.0,
    boundary_margin_px: int = 4,
    enforce_visible_free_space: bool = False,
    camera_pos: Sequence[float] | None = None,
    focal_px: float | None = None,
    free_space_minimum_clearance_mm: float = 8.0,
    free_space_clearance_pixels: float = 3.0,
    free_space_maximum_violation_fraction: float = 0.05,
    free_space_minimum_violation_pixels: int = 64,
) -> tuple[trimesh.Trimesh | None, Dict[str, Any]]:
    """Complete large box-like solids from an observed cap and support plane."""
    from scipy.spatial import ConvexHull, Delaunay, QhullError

    points_image = np.asarray(world, dtype=np.float64)
    selected = np.asarray(valid, dtype=bool)
    if points_image.shape[:2] != selected.shape:
        raise ValueError("planar prism world grid and mask do not match")
    if enforce_visible_free_space and (
        camera_pos is None or focal_px is None
    ):
        raise ValueError(
            "planar prism visibility audit needs camera_pos and focal_px"
        )
    points = points_image[selected]
    if len(points) < int(minimum_plane_pixels):
        return None, {
            "planar_prism_generated_components": 0,
            "planar_prism_reason": "insufficient_valid_points",
        }
    bin_m = float(plane_bin_mm) / 1000.0
    band_m = float(plane_band_mm) / 1000.0
    if bin_m <= 0.0 or band_m <= 0.0:
        raise ValueError("planar prism plane resolution must be positive")
    z_low, z_high = np.percentile(points[:, 2], [0.1, 99.9])
    edges = np.arange(
        np.floor(z_low / bin_m) * bin_m,
        np.ceil(z_high / bin_m) * bin_m + 2.0 * bin_m,
        bin_m,
    )
    histogram, edges = np.histogram(points[:, 2], bins=edges)
    local_maximum = ndimage.maximum_filter1d(
        histogram,
        size=5,
        mode="nearest",
    )
    peak_indices = np.flatnonzero(
        (histogram == local_maximum)
        & (histogram >= int(minimum_plane_pixels))
    )
    peak_indices = peak_indices[
        np.argsort(histogram[peak_indices])[::-1]
    ]

    height, width = selected.shape
    regions: list[Dict[str, Any]] = []
    accepted_heights: list[float] = []
    for peak_index in peak_indices:
        mode_z = float(
            0.5 * (edges[peak_index] + edges[peak_index + 1])
        )
        if any(abs(mode_z - value) < 0.012 for value in accepted_heights):
            continue
        coarse = points[
            np.abs(points[:, 2] - mode_z)
            <= max(2.0 * band_m, bin_m)
        ]
        if len(coarse) < 100:
            continue
        design = np.column_stack(
            (coarse[:, 0], coarse[:, 1], np.ones(len(coarse)))
        )
        coefficients, *_ = np.linalg.lstsq(
            design,
            coarse[:, 2],
            rcond=None,
        )
        for _iteration in range(3):
            residual = coarse[:, 2] - design @ coefficients
            center = float(np.median(residual))
            keep = np.abs(residual - center) <= band_m
            if int(keep.sum()) < 100:
                break
            coefficients, *_ = np.linalg.lstsq(
                design[keep],
                coarse[keep, 2],
                rcond=None,
            )
        fitted_z = (
            points_image[..., 0] * float(coefficients[0])
            + points_image[..., 1] * float(coefficients[1])
            + float(coefficients[2])
        )
        plane_mask = selected & (
            np.abs(points_image[..., 2] - fitted_z) <= band_m
        )
        labels, component_count = ndimage.label(
            plane_mask,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        sizes = np.bincount(labels.reshape(-1))
        accepted_heights.append(mode_z)
        for component_id in range(1, int(component_count) + 1):
            if int(sizes[component_id]) < int(minimum_plane_pixels):
                continue
            component_mask = labels == component_id
            yy, xx = np.nonzero(component_mask)
            component_points = points_image[component_mask]
            xy = np.asarray(component_points[:, :2], dtype=np.float64)
            try:
                footprint = _robust_convex_footprint(xy)
                coverage_hull = ConvexHull(xy)
            except (ValueError, QhullError):
                continue
            regions.append(
                {
                    "mode_z": mode_z,
                    "coefficients": np.asarray(
                        coefficients,
                        dtype=np.float64,
                    ),
                    "mask": component_mask,
                    "pixel_count": int(component_mask.sum()),
                    "bbox_uv": [
                        int(xx.min()),
                        int(yy.min()),
                        int(xx.max()),
                        int(yy.max()),
                    ],
                    "fully_observed": bool(
                        int(xx.min()) > int(boundary_margin_px)
                        and int(yy.min()) > int(boundary_margin_px)
                        and int(xx.max())
                        < width - 1 - int(boundary_margin_px)
                        and int(yy.max())
                        < height - 1 - int(boundary_margin_px)
                    ),
                    "center_xy": np.asarray(
                        footprint["center_xy"],
                        dtype=np.float64,
                    ),
                    "extents": np.asarray(
                        footprint["extents"],
                        dtype=np.float64,
                    ),
                    "hull_xy": np.asarray(
                        footprint["hull_xy"],
                        dtype=np.float64,
                    ),
                    "hull_area": float(footprint["hull_area"]),
                    "rectangularity": float(
                        footprint["rectangularity"]
                    ),
                    "oriented_rectangularity": float(
                        footprint["oriented_rectangularity"]
                    ),
                    "equal_area_rectangle_xy": np.asarray(
                        footprint["equal_area_rectangle_xy"],
                        dtype=np.float64,
                    ),
                    "minimum_rectangle_extents": np.asarray(
                        footprint["minimum_rectangle_extents"],
                        dtype=np.float64,
                    ),
                    "minimum_rectangle_area": float(
                        footprint["minimum_rectangle_area"]
                    ),
                    "rectangle_area_scale": float(
                        footprint["rectangle_area_scale"]
                    ),
                    "footprint_input_points": int(
                        footprint["input_points"]
                    ),
                    "footprint_inlier_points": int(
                        footprint["inlier_points"]
                    ),
                    "coverage_hull_xy": xy[
                        np.asarray(
                            coverage_hull.vertices,
                            dtype=np.int64,
                        )
                    ],
                    "coverage_hull_area": float(
                        coverage_hull.volume
                    ),
                    "coverage_center_xy": xy.mean(axis=0),
                }
            )

    parts: list[trimesh.Trimesh] = []
    rows: list[Dict[str, Any]] = []
    used_caps: list[np.ndarray] = []
    for cap_index, cap in enumerate(regions):
        minimum_extent = float(np.min(cap["extents"]))
        maximum_extent = float(np.max(cap["extents"]))
        cap_candidate = bool(
            cap["fully_observed"]
            and int(cap["pixel_count"]) >= int(minimum_cap_pixels)
            and minimum_extent
            >= float(minimum_cap_extent_mm) / 1000.0
            and maximum_extent
            <= float(maximum_cap_extent_mm) / 1000.0
            and float(cap["hull_area"])
            >= float(minimum_cap_area_mm2) / 1_000_000.0
            and float(cap["rectangularity"])
            >= float(minimum_rectangularity)
        )
        row: Dict[str, Any] = {
            "cap_region_index": int(cap_index),
            "cap_pixels": int(cap["pixel_count"]),
            "cap_bbox_uv": list(cap["bbox_uv"]),
            "cap_fully_observed": bool(cap["fully_observed"]),
            "cap_extents_mm": (
                np.asarray(cap["extents"]) * 1000.0
            ).tolist(),
            "cap_area_mm2": float(cap["hull_area"] * 1_000_000.0),
            "cap_rectangularity": float(cap["rectangularity"]),
            "cap_oriented_rectangularity": float(
                cap["oriented_rectangularity"]
            ),
            "cap_minimum_rectangle_extents_mm": (
                np.asarray(cap["minimum_rectangle_extents"]) * 1000.0
            ).tolist(),
            "cap_rectangle_area_scale": float(
                cap["rectangle_area_scale"]
            ),
            "cap_footprint_input_points": int(
                cap["footprint_input_points"]
            ),
            "cap_footprint_inlier_points": int(
                cap["footprint_inlier_points"]
            ),
            "eligible": False,
        }
        if not cap_candidate:
            rows.append(row)
            continue
        cap_samples = np.vstack(
            (
                cap["coverage_hull_xy"],
                0.5
                * (
                    cap["coverage_hull_xy"]
                    + np.roll(
                        cap["coverage_hull_xy"],
                        -1,
                        axis=0,
                    )
                ),
                np.asarray(
                    cap["coverage_center_xy"]
                ).reshape(1, 2),
            )
        )
        support_choices: list[tuple[float, int, float]] = []
        cap_z = float(
            np.r_[
                np.asarray(
                    cap["coverage_center_xy"],
                    dtype=np.float64,
                ),
                1.0,
            ]
            @ np.asarray(cap["coefficients"], dtype=np.float64)
        )
        for support_index, support in enumerate(regions):
            support_z = float(
                np.r_[
                    np.asarray(
                        cap["coverage_center_xy"],
                        dtype=np.float64,
                    ),
                    1.0,
                ]
                @ np.asarray(
                    support["coefficients"],
                    dtype=np.float64,
                )
            )
            gap = cap_z - support_z
            if (
                support_index == cap_index
                or gap < float(minimum_height_mm) / 1000.0
                or gap > float(maximum_height_mm) / 1000.0
                or float(support["coverage_hull_area"])
                < float(cap["hull_area"])
            ):
                continue
            try:
                support_domain = Delaunay(
                    support["coverage_hull_xy"]
                )
            except QhullError:
                continue
            coverage = float(
                np.mean(support_domain.find_simplex(cap_samples) >= 0)
            )
            if coverage >= 0.75:
                support_choices.append(
                    (gap, support_index, coverage)
                )
        if not support_choices:
            row["failure"] = "no_covering_lower_support_plane"
            rows.append(row)
            continue
        gap, support_index, coverage = min(support_choices)
        support = regions[int(support_index)]
        duplicate = any(
            float(
                np.linalg.norm(
                    np.asarray(cap["center_xy"])
                    - center,
                )
            )
            <= 0.25 * minimum_extent
            for center in used_caps
        )
        if duplicate:
            row["failure"] = "duplicate_cap"
            rows.append(row)
            continue
        regularize_rectangle = bool(
            float(cap["oriented_rectangularity"])
            >= float(rectangle_regularization_threshold)
        )
        footprint_xy = (
            cap["equal_area_rectangle_xy"]
            if regularize_rectangle
            else cap["hull_xy"]
        )
        try:
            mesh = _convex_planar_prism_mesh(
                footprint_xy,
                top_coefficients=cap["coefficients"],
                bottom_coefficients=support["coefficients"],
            )
        except ValueError as error:
            row["failure"] = str(error)
            rows.append(row)
            continue
        if enforce_visible_free_space:
            visibility_audit = _planar_prism_visible_free_space_audit(
                mesh,
                points_image,
                selected,
                camera_pos=np.asarray(camera_pos, dtype=np.float64),
                focal_px=float(focal_px),
                minimum_clearance_mm=(
                    free_space_minimum_clearance_mm
                ),
                clearance_pixels=free_space_clearance_pixels,
                maximum_violation_fraction=(
                    free_space_maximum_violation_fraction
                ),
                minimum_violation_pixels=(
                    free_space_minimum_violation_pixels
                ),
            )
            row["visible_free_space_audit"] = visibility_audit
            if not bool(visibility_audit["accepted"]):
                row["failure"] = str(visibility_audit["reason"])
                rows.append(row)
                continue
        parts.append(mesh)
        used_caps.append(np.asarray(cap["center_xy"], dtype=np.float64))
        row.update(
            {
                "eligible": True,
                "support_region_index": int(support_index),
                "support_coverage": coverage,
                "height_mm": float(gap * 1000.0),
                "footprint_prior": (
                    "equal_area_oriented_rectangle"
                    if regularize_rectangle
                    else "robust_convex_hull"
                ),
                "mesh_vertices": int(len(mesh.vertices)),
                "mesh_faces": int(len(mesh.faces)),
                "mesh_volume_cm3": float(abs(mesh.volume) * 1_000_000.0),
            }
        )
        rows.append(row)
    return (
        trimesh.util.concatenate(parts) if parts else None,
        {
            "planar_prism_detected_plane_regions": int(len(regions)),
            "planar_prism_generated_components": int(len(parts)),
            "planar_prism_rejected_visible_free_space_components": int(
                sum(
                    row.get("failure")
                    in {
                        "observed_free_space_violation",
                        "candidate_not_visible_in_observed_rays",
                        "no_valid_observed_rays",
                    }
                    for row in rows
                )
            ),
            "planar_prism_parameters": {
                "plane_bin_mm": float(plane_bin_mm),
                "plane_band_mm": float(plane_band_mm),
                "minimum_plane_pixels": int(minimum_plane_pixels),
                "minimum_cap_pixels": int(minimum_cap_pixels),
                "minimum_cap_extent_mm": float(minimum_cap_extent_mm),
                "maximum_cap_extent_mm": float(maximum_cap_extent_mm),
                "minimum_cap_area_mm2": float(minimum_cap_area_mm2),
                "minimum_rectangularity": float(minimum_rectangularity),
                "rectangle_regularization_threshold": float(
                    rectangle_regularization_threshold
                ),
                "minimum_height_mm": float(minimum_height_mm),
                "maximum_height_mm": float(maximum_height_mm),
                "boundary_margin_px": int(boundary_margin_px),
                "enforce_visible_free_space": bool(
                    enforce_visible_free_space
                ),
                "free_space_minimum_clearance_mm": float(
                    free_space_minimum_clearance_mm
                ),
                "free_space_clearance_pixels": float(
                    free_space_clearance_pixels
                ),
                "free_space_maximum_violation_fraction": float(
                    free_space_maximum_violation_fraction
                ),
                "free_space_minimum_violation_pixels": int(
                    free_space_minimum_violation_pixels
                ),
            },
            "planar_prism_components": rows,
        },
    )


def _support_aligned_heightfield_meshes(
    rgb: np.ndarray,
    depth: np.ndarray,
    plane: Any,
    world: np.ndarray,
    valid: np.ndarray,
    *,
    focal_px: float,
    voxel_mm: float = 1.5,
    boundary_margin_px: int = 4,
    low_profile_max_height_mm: float = 100.0,
    low_profile_max_aspect: float = 0.30,
) -> tuple[trimesh.Trimesh | None, np.ndarray, Dict[str, Any]]:
    """Close only fully observed compact caps toward their support.

    Low-profile components are still classified for diagnostics, but a
    single support heightfield is not a valid generic completion for them:
    it discards visible multi-layer topology and can reverse local volume
    ordering.
    """
    from .rgbd_scene_components import (
        _continuous_component_labels,
    )

    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    signed_height = plane.signed_height(world)
    candidate = (
        np.asarray(valid, dtype=bool)
        & (signed_height >= 0.0005)
        & (signed_height <= 0.25)
    )
    labels, sizes = _continuous_component_labels(
        np.asarray(world, dtype=np.float64),
        metric_depth,
        np.asarray(rgb),
        candidate,
        focal_px=float(focal_px),
        edge_scale=1.8,
        edge_slack_m=0.002,
        edge_depth_jump_m=0.03,
        edge_color_distance=120.0,
    )
    pitch_m = float(voxel_mm) / 1000.0
    if pitch_m <= 0.0:
        raise ValueError("support heightfield voxel size must be positive")
    margin = int(boundary_margin_px)
    if margin < 0:
        raise ValueError("support heightfield boundary margin is negative")
    image_height, image_width = candidate.shape

    parts: list[trimesh.Trimesh] = []
    selected_mask = np.zeros(candidate.shape, dtype=bool)
    rows: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    basis_x = np.asarray(plane.basis_x, dtype=np.float64)
    basis_y = np.asarray(plane.basis_y, dtype=np.float64)
    normal = np.asarray(plane.normal, dtype=np.float64)
    origin = np.asarray(plane.origin, dtype=np.float64)
    for component_id in np.flatnonzero(sizes >= 800):
        component_mask = labels == int(component_id)
        component_y, component_x = np.nonzero(component_mask)
        points = np.asarray(world[component_mask], dtype=np.float64)
        heights = np.asarray(signed_height[component_mask], dtype=np.float64)
        relative = points - origin.reshape(1, 3)
        local_x = relative @ basis_x
        local_y = relative @ basis_y
        x_low, x_high = np.percentile(local_x, [0.5, 99.5])
        y_low, y_high = np.percentile(local_y, [0.5, 99.5])
        h_low, h_high = np.percentile(heights, [0.5, 99.5])
        extent_x = float(x_high - x_low)
        extent_y = float(y_high - y_low)
        height_extent = float(h_high - h_low)
        horizontal_min = min(extent_x, extent_y)
        horizontal_max = max(extent_x, extent_y)
        aspect = height_extent / max(horizontal_min, pitch_m)
        support_gap_ratio = h_low / max(horizontal_min, pitch_m)
        fully_observed = bool(
            int(component_x.min()) > margin
            and int(component_y.min()) > margin
            and int(component_x.max()) < image_width - 1 - margin
            and int(component_y.max()) < image_height - 1 - margin
        )
        common_geometry = bool(
            fully_observed
            and horizontal_min >= 0.04
            and horizontal_max <= 0.5
            and h_high >= 0.015
            and h_high <= 0.25
        )
        elevated_compact = bool(
            common_geometry
            and horizontal_max <= 0.15
            and h_low >= 0.015
            and aspect >= 0.20
            and aspect <= 0.55
            and support_gap_ratio >= 1.3
        )
        low_profile = bool(
            common_geometry
            and h_low >= 0.008
            and h_high
            <= float(low_profile_max_height_mm) / 1000.0
            and aspect <= float(low_profile_max_aspect)
        )
        eligible = bool(elevated_compact)
        row: Dict[str, Any] = {
            "component_id": int(component_id),
            "pixel_count": int(component_mask.sum()),
            "image_bbox_uv": [
                int(component_x.min()),
                int(component_y.min()),
                int(component_x.max()),
                int(component_y.max()),
            ],
            "fully_observed": fully_observed,
            "horizontal_extents_mm": [
                extent_x * 1000.0,
                extent_y * 1000.0,
            ],
            "height_percentiles_mm": [
                h_low * 1000.0,
                h_high * 1000.0,
            ],
            "height_to_min_width": float(aspect),
            "support_gap_to_min_width": float(support_gap_ratio),
            "completion_prior": (
                "preserve_visible_shell_pending_multilayer_prior"
                if low_profile
                else (
                    "complete_elevated_heightfield"
                    if elevated_compact
                    else None
                )
            ),
            "eligible": eligible,
        }
        if not eligible:
            rows.append(row)
            continue
        try:
            x_min = float(np.floor((x_low - 2.0 * pitch_m) / pitch_m) * pitch_m)
            y_min = float(np.floor((y_low - 2.0 * pitch_m) / pitch_m) * pitch_m)
            x_count = int(np.ceil((x_high - x_min + 2.0 * pitch_m) / pitch_m)) + 1
            y_count = int(np.ceil((y_high - y_min + 2.0 * pitch_m) / pitch_m)) + 1
            z_count = int(np.ceil((h_high + 2.0 * pitch_m) / pitch_m)) + 1
            if x_count * y_count * z_count > 24_000_000:
                raise ValueError("support heightfield grid is too large")
            ix = np.clip(
                np.rint((local_x - x_min) / pitch_m).astype(np.int64),
                0,
                x_count - 1,
            )
            iy = np.clip(
                np.rint((local_y - y_min) / pitch_m).astype(np.int64),
                0,
                y_count - 1,
            )
            iz = np.clip(
                np.floor(heights / pitch_m).astype(np.int64),
                0,
                z_count - 1,
            )
            top = np.full((x_count, y_count), -1, dtype=np.int32)
            np.maximum.at(top, (ix, iy), iz.astype(np.int32))
            observed_footprint = top >= 0
            footprint = observed_footprint
            closed = ndimage.binary_closing(
                footprint,
                structure=np.ones((3, 3), dtype=bool),
                iterations=1,
            )
            if closed.any() and np.any(closed & (~observed_footprint)):
                _distance, nearest = ndimage.distance_transform_edt(
                    ~observed_footprint,
                    return_indices=True,
                )
                added = closed & (~observed_footprint)
                top[added] = top[nearest[0, added], nearest[1, added]]
            bottom_height = 0.0
            bottom_index = int(
                np.clip(
                    np.floor(bottom_height / pitch_m),
                    0,
                    z_count - 1,
                )
            )
            z_indices = np.arange(
                z_count,
                dtype=np.int32,
            ).reshape(1, 1, -1)
            volume = (
                (z_indices >= bottom_index)
                & (z_indices <= top[..., None])
            )
            if int(volume.sum()) < 100:
                raise ValueError("support heightfield volume is empty")
            mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
                volume,
                pitch=pitch_m,
            )
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = np.column_stack(
                (basis_x, basis_y, normal)
            )
            transform[:3, 3] = (
                origin + basis_x * x_min + basis_y * y_min
            )
            mesh.apply_transform(transform)
            mesh.process(validate=True)
            trimesh.repair.fix_normals(mesh)
            if float(mesh.volume) < 0.0:
                mesh.invert()
            if not mesh.is_watertight:
                raise ValueError("support heightfield mesh is not watertight")
        except (ValueError, MemoryError) as error:
            row["eligible"] = False
            row["failure"] = str(error)
            failures.append(
                {
                    "component_id": int(component_id),
                    "reason": str(error),
                }
            )
            rows.append(row)
            continue
        parts.append(mesh)
        selected_mask |= component_mask
        row.update(
            {
                "mesh_vertices": int(len(mesh.vertices)),
                "mesh_faces": int(len(mesh.faces)),
                "occupied_voxels": int(volume.sum()),
                "grid_shape": [x_count, y_count, z_count],
                "bottom_height_mm": float(bottom_height * 1000.0),
                "bottom_index": int(bottom_index),
            }
        )
        rows.append(row)

    return (
        trimesh.util.concatenate(parts) if parts else None,
        selected_mask,
        {
            "support_heightfield_candidate_pixels": int(candidate.sum()),
            "support_heightfield_component_count": int(len(sizes)),
            "support_heightfield_generated_components": int(len(parts)),
            "support_heightfield_voxel_mm": float(voxel_mm),
            "support_heightfield_boundary_margin_px": margin,
            "support_heightfield_low_profile_max_height_mm": float(
                low_profile_max_height_mm
            ),
            "support_heightfield_low_profile_max_aspect": float(
                low_profile_max_aspect
            ),
            "support_heightfield_low_profile_policy": (
                "classify_only_preserve_visible_shell"
            ),
            "support_heightfield_components": rows,
            "support_heightfield_failures": failures,
        },
    )


def _rich_circle_tube_mesh_from_mask(
    selected_mask: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    voxel_mm: float = 1.5,
    radius_bias_px: float = 0.25,
    radius_min_mm: float = 0.4,
    radius_max_mm: float = 8.0,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Use the generic local-cylinder fitter with an RGB-D-derived mask."""
    from .rgbd_only_mesh_reconstruction import (
        _mesh_from_local_circle_tubes,
        reconstruct_rgbd_only_mesh,
    )

    yy, xx = np.nonzero(selected_mask)
    if not len(yy):
        raise ValueError("thin structure mask is empty")
    click_u = int(np.median(xx))
    click_v = int(np.median(yy))
    public_signature = inspect.signature(reconstruct_rgbd_only_mesh)
    public_defaults = {
        name: parameter.default
        for name, parameter in public_signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    private_signature = inspect.signature(_mesh_from_local_circle_tubes)
    options = {
        name: public_defaults[name]
        for name in private_signature.parameters
        if name in public_defaults
    }
    options.update(
        {
            "camera_pos": camera_pos,
            "camera_quat_xyzw": camera_quat_xyzw,
            "focal_length": float(focal_length),
            "horizontal_aperture": float(horizontal_aperture),
            "voxel_mm": float(voxel_mm),
            "click_u": click_u,
            "click_v": click_v,
            "component_policy": "all",
            "tube_meshing_mode": "voxel_union",
            "circle_radius_bias_px": float(radius_bias_px),
            "circle_radius_min_mm": float(radius_min_mm),
            "circle_radius_max_mm": float(radius_max_mm),
            "local_cylinder_nonlinear_refine": False,
        }
    )
    return _mesh_from_local_circle_tubes(
        SimpleNamespace(selected_mask=np.asarray(selected_mask, dtype=bool)),
        np.asarray(rgb),
        np.asarray(depth, dtype=np.float64),
        **options,
    )


def _inpaint_structure_depth(
    depth: np.ndarray,
    structure_mask: np.ndarray,
) -> np.ndarray:
    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    inpaint = ndimage.binary_dilation(
        np.asarray(structure_mask, dtype=bool),
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    _distance, nearest = ndimage.distance_transform_edt(
        inpaint,
        return_indices=True,
    )
    result = metric_depth.copy()
    result[inpaint] = metric_depth[
        nearest[0, inpaint],
        nearest[1, inpaint],
    ]
    return result


def _route_low_profile_rgb_edges(
    base_options: Dict[str, Any],
    structure_mask: np.ndarray,
    support_heightfield_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Preserve RGB-D layer boundaries on fully observed low-profile surfaces."""
    components = support_heightfield_metadata.get(
        "support_heightfield_components",
        [],
    )
    low_profile_components = [
        row
        for row in components
        if row.get("completion_prior")
        == "preserve_visible_shell_pending_multilayer_prior"
    ]
    requested_threshold = float(
        base_options.get("rgbd_edge_color_threshold", 0.0)
    )
    requested_min_jump_mm = float(
        base_options.get("rgbd_edge_min_depth_jump_mm", 3.0)
    )
    route_enabled = bool(
        low_profile_components
        and not np.asarray(structure_mask, dtype=bool).any()
    )
    if route_enabled:
        base_options["rgbd_edge_color_threshold"] = max(
            requested_threshold,
            120.0,
        )
        base_options["rgbd_edge_min_depth_jump_mm"] = min(
            requested_min_jump_mm,
            3.0,
        )
        reason = "fully_observed_low_profile_multilayer_surface"
    elif low_profile_components:
        reason = "thin_structure_route_has_priority"
    else:
        reason = "no_low_profile_multilayer_surface"
    return {
        "low_profile_component_count": int(len(low_profile_components)),
        "low_profile_component_ids": [
            int(row["component_id"])
            for row in low_profile_components
        ],
        "low_profile_rgb_edge_route_enabled": route_enabled,
        "low_profile_rgb_edge_route_reason": reason,
        "low_profile_rgb_edge_requested_threshold": requested_threshold,
        "low_profile_rgb_edge_applied_threshold": float(
            base_options.get(
                "rgbd_edge_color_threshold",
                requested_threshold,
            )
        ),
        "low_profile_rgb_edge_requested_min_depth_jump_mm": (
            requested_min_jump_mm
        ),
        "low_profile_rgb_edge_applied_min_depth_jump_mm": float(
            base_options.get(
                "rgbd_edge_min_depth_jump_mm",
                requested_min_jump_mm,
            )
        ),
    }


def _structural_hybrid_reconstruction(
    rgb: np.ndarray,
    depth: np.ndarray,
    options: Dict[str, Any],
) -> RGBDSceneMeshResult:
    defer_topology_validation = bool(
        options.pop("_defer_topology_validation_to_warp_cuda", False)
    )
    enable_rgb_completion = bool(
        options.pop("structural_rgb_completion", False)
    )
    enable_low_profile_frustum = bool(
        options.pop("structural_low_profile_frustum", True)
    )
    low_profile_frustum_thickness_scale = float(
        options.pop(
            "structural_low_profile_frustum_thickness_scale",
            0.60,
        )
    )
    low_profile_frustum_expansion_scale = float(
        options.pop(
            "structural_low_profile_frustum_expansion_scale",
            0.10,
        )
    )
    enable_planar_prism_visibility_guard = bool(
        options.pop(
            "structural_planar_prism_visibility_guard",
            False,
        )
    )
    planar_prism_free_space_minimum_clearance_mm = float(
        options.pop(
            "structural_planar_prism_free_space_minimum_clearance_mm",
            8.0,
        )
    )
    planar_prism_free_space_clearance_pixels = float(
        options.pop(
            "structural_planar_prism_free_space_clearance_pixels",
            3.0,
        )
    )
    planar_prism_free_space_maximum_violation_fraction = float(
        options.pop(
            "structural_planar_prism_free_space_maximum_violation_fraction",
            0.05,
        )
    )
    planar_prism_free_space_minimum_violation_pixels = int(
        options.pop(
            "structural_planar_prism_free_space_minimum_violation_pixels",
            64,
        )
    )
    camera_options = {
        key: options[key]
        for key in (
            "camera_pos",
            "camera_quat_xyzw",
            "focal_length",
            "horizontal_aperture",
        )
    }
    structure_mask, plane, world, structure_metadata = (
        _support_structure_mask(
            rgb,
            depth,
            **camera_options,
        )
    )
    reliable_structure_mask = structure_mask.copy()
    structure_depth = np.asarray(depth, dtype=np.float64)
    rgb_completion_metadata: Dict[str, Any] = {
        "rgb_thin_completion_enabled": False,
        "rgb_thin_completion_reason": (
            "not_requested"
            if not enable_rgb_completion
            else "no_thin_structure"
        ),
        "rgb_thin_completion_pixels": 0,
    }
    if (
        enable_rgb_completion
        and
        structure_mask.any()
        and plane is not None
        and world is not None
    ):
        (
            structure_mask,
            structure_depth,
            rgb_completion_metadata,
        ) = _rgb_guided_thin_structure_completion(
            rgb,
            depth,
            structure_mask,
            plane,
            world,
            **camera_options,
        )
    support_heightfield = None
    support_heightfield_mask = np.zeros(structure_mask.shape, dtype=bool)
    support_heightfield_metadata: Dict[str, Any] = {
        "support_heightfield_generated_components": 0,
    }
    planar_prism = None
    planar_prism_metadata: Dict[str, Any] = {
        "planar_prism_generated_components": 0,
    }
    if plane is not None and world is not None:
        metric_depth = np.asarray(depth, dtype=np.float64)
        if metric_depth.ndim == 3:
            metric_depth = metric_depth[..., 0]
        valid = np.isfinite(metric_depth) & (metric_depth > 0.03)
        focal_px = (
            float(camera_options["focal_length"])
            / float(camera_options["horizontal_aperture"])
            * float(metric_depth.shape[1])
        )
        (
            support_heightfield,
            support_heightfield_mask,
            support_heightfield_metadata,
        ) = _support_aligned_heightfield_meshes(
            rgb,
            metric_depth,
            plane,
            world,
            valid,
            focal_px=focal_px,
        )
        planar_prism, planar_prism_metadata = (
            _elevated_planar_prism_meshes(
                world,
                valid,
                enforce_visible_free_space=(
                    enable_planar_prism_visibility_guard
                ),
                camera_pos=camera_options["camera_pos"],
                focal_px=focal_px,
                free_space_minimum_clearance_mm=(
                    planar_prism_free_space_minimum_clearance_mm
                ),
                free_space_clearance_pixels=(
                    planar_prism_free_space_clearance_pixels
                ),
                free_space_maximum_violation_fraction=(
                    planar_prism_free_space_maximum_violation_fraction
                ),
                free_space_minimum_violation_pixels=(
                    planar_prism_free_space_minimum_violation_pixels
                ),
            )
        )
    replacement_mask = structure_mask | support_heightfield_mask
    background_depth = (
        _inpaint_structure_depth(depth, replacement_mask)
        if replacement_mask.any()
        else np.asarray(depth)
    )
    base_options = dict(options)
    if defer_topology_validation:
        base_options[
            "_defer_topology_validation_to_warp_cuda"
        ] = True
    low_profile_rgb_metadata = _route_low_profile_rgb_edges(
        base_options,
        structure_mask,
        support_heightfield_metadata,
    )
    base_options["completion_method"] = (
        "organized_shell"
        if structure_mask.any()
        else "axisymmetric_cavity_hybrid"
    )
    base = _v11_base_reconstruct_rgbd_scene_mesh(
        rgb,
        background_depth,
        **base_options,
    )
    selected_base_prior = str(base_options["completion_method"])
    if (
        not structure_mask.any()
        and not bool(
            base.metadata.get("axisymmetric_cavity_detected", False)
        )
        and int(
            base.metadata.get(
                "compact_symmetry_eligible_components",
                0,
            )
        )
        == 0
    ):
        base_options["completion_method"] = (
            "topology_hybrid"
            if bool(
                low_profile_rgb_metadata[
                    "low_profile_rgb_edge_route_enabled"
                ]
            )
            else "organized_shell"
        )
        base = _v11_base_reconstruct_rgbd_scene_mesh(
            rgb,
            background_depth,
            **base_options,
        )
        selected_base_prior = str(base_options["completion_method"])

    low_profile_frustum_metadata: Dict[str, Any] = {
        "low_profile_frustum_generated_components": 0,
        "low_profile_frustum_reason": "base_prior_not_topology_hybrid",
        "low_profile_frustum_components": [],
    }
    if (
        plane is not None
        and selected_base_prior == "topology_hybrid"
    ):
        base_mesh, low_profile_frustum_metadata = (
            _regularize_low_profile_frustum_shells(
                base.mesh,
                base.metadata,
                plane,
                image_width=int(np.asarray(depth).shape[1]),
                image_height=int(np.asarray(depth).shape[0]),
                enabled=enable_low_profile_frustum,
                hidden_thickness_height_scale=(
                    low_profile_frustum_thickness_scale
                ),
                radial_expansion_height_scale=(
                    low_profile_frustum_expansion_scale
                ),
            )
        )
        base = RGBDSceneMeshResult(
            mesh=base_mesh,
            metadata={
                **base.metadata,
                **low_profile_frustum_metadata,
            },
        )

    parts: list[trimesh.Trimesh] = [base.mesh]
    tube_metadata: Dict[str, Any] = {
        "thin_tube_generated": False,
    }
    if reliable_structure_mask.any():
        try:
            tube_mesh, tube_metadata = _rich_circle_tube_mesh_from_mask(
                reliable_structure_mask,
                rgb,
                depth,
                **camera_options,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            tube_metadata = {
                "thin_tube_generated": False,
                "thin_tube_failure": f"{type(error).__name__}: {error}",
            }
        else:
            parts.append(tube_mesh)
            tube_metadata = {
                **tube_metadata,
                "thin_tube_generated": True,
            }

    rgb_only_mask = structure_mask & (~reliable_structure_mask)
    rgb_only_metadata: Dict[str, Any] = {
        "rgb_only_tube_generated": False,
    }
    if rgb_only_mask.any():
        try:
            rgb_only_mesh, rgb_only_metadata = (
                _rich_circle_tube_mesh_from_mask(
                    rgb_only_mask,
                    rgb,
                    structure_depth,
                    voxel_mm=1.0,
                    radius_bias_px=0.0,
                    radius_min_mm=0.4,
                    radius_max_mm=1.5,
                    **camera_options,
                )
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            rgb_only_metadata = {
                "rgb_only_tube_generated": False,
                "rgb_only_tube_failure": (
                    f"{type(error).__name__}: {error}"
                ),
            }
        else:
            parts.append(rgb_only_mesh)
            rgb_only_metadata = {
                "rgb_only_tube_generated": True,
                "rgb_only_tube_pixels": int(rgb_only_mask.sum()),
                "rgb_only_tube_voxel_mm": 1.0,
                "rgb_only_tube_radius_min_mm": 0.4,
                "rgb_only_tube_radius_max_mm": 1.5,
                "rgb_only_tube_mesh_vertices": int(
                    len(rgb_only_mesh.vertices)
                ),
                "rgb_only_tube_mesh_faces": int(
                    len(rgb_only_mesh.faces)
                ),
            }

    support_slab = None
    if plane is not None and world is not None and structure_mask.any():
        metric_depth = np.asarray(depth, dtype=np.float64)
        if metric_depth.ndim == 3:
            metric_depth = metric_depth[..., 0]
        valid = np.isfinite(metric_depth) & (metric_depth > 0.03)
        support_slab = _support_slab_mesh(
            plane,
            world,
            valid,
        )
        if support_slab is not None:
            parts.append(support_slab)

    if support_heightfield is not None:
        parts.append(support_heightfield)
    if planar_prism is not None:
        parts.append(planar_prism)

    mesh = trimesh.util.concatenate(parts)
    if defer_topology_validation:
        component_count = None
        mesh_watertight = None
        topology_validation = "deferred_to_warp_cuda"
    else:
        components = mesh.split(only_watertight=False)
        if any(not component.is_watertight for component in components):
            raise ValueError(
                "v12 structural hybrid produced an open component"
            )
        component_count = int(len(components))
        mesh_watertight = bool(mesh.is_watertight)
        topology_validation = "trimesh_cpu"
    metadata = {
        **base.metadata,
        "selected_base_prior": selected_base_prior,
        **structure_metadata,
        **rgb_completion_metadata,
        **tube_metadata,
        **rgb_only_metadata,
        **support_heightfield_metadata,
        **low_profile_rgb_metadata,
        **low_profile_frustum_metadata,
        **planar_prism_metadata,
        "support_heightfield_replaced_pixels": int(
            support_heightfield_mask.sum()
        ),
        "support_slab_generated": bool(support_slab is not None),
        "support_slab_thickness_mm": (
            80.0 if support_slab is not None else None
        ),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_components": component_count,
        "mesh_watertight": mesh_watertight,
        "mesh_topology_validation": topology_validation,
        "build": RGBD_SCENE_MESH_BUILD,
        "method": (
            "whole_view_decoupled_width_tube_and_closed_surface_router"
        ),
        "completion_method": "structural_hybrid",
        "used_click": False,
        "used_segmentation": False,
        "used_object_identity": False,
        "forbidden_inputs_used": [],
    }
    return RGBDSceneMeshResult(mesh=mesh, metadata=metadata)


def reconstruct_rgbd_scene_mesh(
    rgb: np.ndarray,
    depth: np.ndarray,
    **options: Any,
) -> RGBDSceneMeshResult:
    """Route RGB-D geometry through the independent v12 structural prior."""
    selected = str(
        options.get("completion_method", "organized_shell")
    ).strip().lower()
    if selected != "structural_hybrid":
        base_options = dict(options)
        for key in (
            "structural_rgb_completion",
            "structural_low_profile_frustum",
            "structural_low_profile_frustum_thickness_scale",
            "structural_low_profile_frustum_expansion_scale",
            "structural_planar_prism_visibility_guard",
            "structural_planar_prism_free_space_minimum_clearance_mm",
            "structural_planar_prism_free_space_clearance_pixels",
            "structural_planar_prism_free_space_maximum_violation_fraction",
            "structural_planar_prism_free_space_minimum_violation_pixels",
        ):
            base_options.pop(key, None)
        return _v11_base_reconstruct_rgbd_scene_mesh(
            rgb,
            depth,
            **base_options,
        )
    return _structural_hybrid_reconstruction(
        rgb,
        depth,
        dict(options),
    )
