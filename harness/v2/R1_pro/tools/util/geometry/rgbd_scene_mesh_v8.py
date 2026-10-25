"""Frozen v8 whole-view RGB-D scene reconstruction.

The runtime contract is intentionally scene-level: RGB, metric depth, and
camera calibration are the only inputs. Depth discontinuities are used to
avoid triangles that bridge unrelated visible surfaces, but every resulting
surface patch is retained. No click, segmentation, object identity, reference
mesh, evaluation center, or overlap label is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Sequence

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
import trimesh


RGBD_SCENE_MESH_BUILD = "rgbd_whole_view_organized_shell_v8"


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


def _split_nonmanifold_vertex_fans(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
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
) -> tuple[trimesh.Trimesh, Dict[str, int]]:
    """Close edge-connected front patches without merging point contacts."""
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

    components = mesh.split(only_watertight=False)
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
            "mesh_components": int(len(components)),
            "mesh_watertight": bool(mesh.is_watertight),
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
