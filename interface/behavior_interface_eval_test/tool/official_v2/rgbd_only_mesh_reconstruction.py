"""RGBD-only mesh reconstruction experiments.

The functions in this module intentionally accept only RGB, metric depth,
camera calibration, and a target click. Reference meshes, evaluation centers,
segmentation, object names, and overlap labels are not inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Sequence

import numpy as np
from scipy import ndimage
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from skimage import measure
from skimage.morphology import skeletonize
import trimesh

from .depth_mesh_reconstruction import (
    DepthMeshReconstruction,
    _depth_component_labels,
    backproject_depth_world,
    reconstruct_supported_object_mesh,
)


RGBD_ONLY_MESH_BUILD = "rgbd_only_geometric_completion_v27"


@dataclass
class RGBDOnlyMeshResult:
    mesh: trimesh.Trimesh
    extraction: DepthMeshReconstruction
    metadata: Dict[str, Any]


def _component_endpoint_samples(
    component_mask: np.ndarray,
    *,
    world_points: np.ndarray,
    rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return endpoint-biased RGBD samples for one image component."""
    mask = np.asarray(component_mask, dtype=bool)
    if not mask.any():
        raise ValueError("component endpoint sampling requires pixels")
    skeleton = skeletonize(mask)
    neighbor_count = ndimage.convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    endpoint_mask = skeleton & (neighbor_count <= 2)
    if not endpoint_mask.any():
        endpoint_mask = skeleton
    yy, xx = np.nonzero(endpoint_mask)
    return (
        np.asarray(world_points, dtype=np.float64)[yy, xx],
        np.column_stack((xx, yy)).astype(np.float64),
        np.asarray(rgb, dtype=np.float64)[yy, xx],
    )


def _expand_component_mask_graph(
    *,
    candidate_mask: np.ndarray,
    initial_mask: np.ndarray,
    world_points: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    focal_px: float,
    max_gap_3d_m: float,
    max_gap_px: float,
    max_color_delta: float,
    max_hops: int,
    max_added_pixel_ratio: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Expand a clicked mask through conservative endpoint RGBD contacts."""
    candidate = np.asarray(candidate_mask, dtype=bool)
    initial = np.asarray(initial_mask, dtype=bool)
    points = np.asarray(world_points, dtype=np.float64)
    color = np.asarray(rgb, dtype=np.float64)
    metric_depth = np.asarray(depth, dtype=np.float64)
    if candidate.shape != initial.shape or candidate.shape != metric_depth.shape:
        raise ValueError("component graph masks and depth must share a shape")
    if points.shape != candidate.shape + (3,):
        raise ValueError("world_points must be HxWx3")
    if color.shape[:2] != candidate.shape or color.shape[-1] < 3:
        raise ValueError("rgb must be HxWx3")
    if (
        float(focal_px) <= 0.0
        or float(max_gap_3d_m) <= 0.0
        or float(max_gap_px) <= 0.0
        or float(max_color_delta) < 0.0
        or int(max_hops) < 0
        or float(max_added_pixel_ratio) < 0.0
    ):
        raise ValueError("invalid component graph expansion parameters")

    candidate = candidate | initial
    labels, _yy, _xx, component_for_node = _depth_component_labels(
        candidate,
        world_points=points,
        rgb=color,
        depth=metric_depth,
        focal_px=float(focal_px),
    )
    component_count = (
        int(component_for_node.max()) + 1
        if len(component_for_node)
        else 0
    )
    if component_count == 0:
        return initial.copy(), {
            "candidate_pixels": int(candidate.sum()),
            "initial_pixels": int(initial.sum()),
            "expanded_pixels": int(initial.sum()),
            "component_count": 0,
            "initial_components": [],
            "selected_components": [],
            "accepted_edges": [],
            "hop_count": 0,
            "pixel_budget": int(initial.sum()),
        }

    component_masks = [labels == value for value in range(component_count)]
    component_sizes = np.asarray(
        [int(mask.sum()) for mask in component_masks],
        dtype=np.int64,
    )
    surface_samples = []
    for mask in component_masks:
        surface_y, surface_x = np.nonzero(mask)
        surface_samples.append(
            (
                points[surface_y, surface_x],
                np.column_stack((surface_x, surface_y)).astype(
                    np.float64
                ),
                color[surface_y, surface_x],
            )
        )
    endpoint_samples = [
        _component_endpoint_samples(
            mask,
            world_points=points,
            rgb=color,
        )
        for mask in component_masks
    ]
    selected_components = {
        component
        for component, mask in enumerate(component_masks)
        if bool(np.any(mask & initial))
    }
    initial_components = set(selected_components)
    if not selected_components:
        raise ValueError("initial mask has no candidate RGBD component")

    initial_pixels = int(initial.sum())
    pixel_budget = int(
        math.ceil(
            initial_pixels * (1.0 + float(max_added_pixel_ratio))
        )
    )
    selected_pixels = int(
        component_sizes[list(selected_components)].sum()
    )
    accepted_edges: list[Dict[str, Any]] = []
    hop_count = 0
    for hop in range(1, int(max_hops) + 1):
        additions: list[tuple[float, int, Dict[str, Any]]] = []
        for candidate_component in range(component_count):
            if candidate_component in selected_components:
                continue
            candidate_world, candidate_uv, candidate_rgb = endpoint_samples[
                candidate_component
            ]
            best_edge: tuple[float, Dict[str, Any]] | None = None
            for selected_component in selected_components:
                contact_modes = [
                    (
                        "endpoint",
                        endpoint_samples[selected_component],
                        endpoint_samples[candidate_component],
                        float(max_gap_px),
                        0.0,
                    ),
                    (
                        "surface_junction",
                        surface_samples[selected_component],
                        surface_samples[candidate_component],
                        min(4.0, float(max_gap_px)),
                        0.15,
                    ),
                ]
                for (
                    contact_kind,
                    selected_samples,
                    candidate_samples,
                    contact_gap_px,
                    contact_penalty,
                ) in contact_modes:
                    selected_world, selected_uv, selected_rgb = (
                        selected_samples
                    )
                    candidate_world, candidate_uv, candidate_rgb = (
                        candidate_samples
                    )
                    selected_tree = cKDTree(selected_world)
                    distance_3d, nearest_selected = selected_tree.query(
                        candidate_world,
                        k=1,
                    )
                    candidate_index = int(np.argmin(distance_3d))
                    selected_index = int(nearest_selected[candidate_index])
                    gap_3d = float(distance_3d[candidate_index])
                    gap_px = float(
                        np.linalg.norm(
                            candidate_uv[candidate_index]
                            - selected_uv[selected_index]
                        )
                    )
                    color_delta = float(
                        np.linalg.norm(
                            candidate_rgb[candidate_index, :3]
                            - selected_rgb[selected_index, :3]
                        )
                    )
                    if (
                        gap_3d > float(max_gap_3d_m)
                        or gap_px > contact_gap_px
                        or color_delta > float(max_color_delta)
                    ):
                        continue
                    normalized_cost = (
                        gap_3d / float(max_gap_3d_m)
                        + gap_px / contact_gap_px
                        + color_delta
                        / max(float(max_color_delta), 1e-12)
                        + contact_penalty
                    )
                    edge = {
                        "hop": int(hop),
                        "contact_kind": contact_kind,
                        "from_component": int(selected_component),
                        "to_component": int(candidate_component),
                        "gap_3d_mm": float(gap_3d * 1000.0),
                        "gap_px": gap_px,
                        "color_delta": color_delta,
                        "from_uv": selected_uv[selected_index].tolist(),
                        "to_uv": candidate_uv[candidate_index].tolist(),
                        "to_component_pixels": int(
                            component_sizes[candidate_component]
                        ),
                        "normalized_cost": float(normalized_cost),
                    }
                    if (
                        best_edge is None
                        or normalized_cost < best_edge[0]
                    ):
                        best_edge = (normalized_cost, edge)
            if best_edge is not None:
                additions.append(
                    (
                        best_edge[0],
                        candidate_component,
                        best_edge[1],
                    )
                )
        if not additions:
            break
        added_this_hop = 0
        for _cost, component, edge in sorted(additions):
            component_pixels = int(component_sizes[component])
            if selected_pixels + component_pixels > pixel_budget:
                continue
            selected_components.add(int(component))
            selected_pixels += component_pixels
            accepted_edges.append(edge)
            added_this_hop += 1
        if added_this_hop == 0:
            break
        hop_count = hop

    expanded = np.isin(labels, list(selected_components))
    return expanded, {
        "candidate_pixels": int(candidate.sum()),
        "initial_pixels": initial_pixels,
        "expanded_pixels": int(expanded.sum()),
        "added_pixels": int(expanded.sum() - initial_pixels),
        "component_count": component_count,
        "initial_components": sorted(int(v) for v in initial_components),
        "selected_components": sorted(
            int(v) for v in selected_components
        ),
        "accepted_edges": accepted_edges,
        "hop_count": int(hop_count),
        "pixel_budget": pixel_budget,
    }


def _complete_extraction_mask_from_rgbd_graph(
    extraction: DepthMeshReconstruction,
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    max_gap_3d_mm: float,
    max_gap_px: float,
    max_color_delta: float,
    max_hops: int,
    max_added_pixel_ratio: float,
) -> tuple[DepthMeshReconstruction, Dict[str, Any]]:
    """Recover visible target fragments omitted by the click component gate."""
    color = np.asarray(rgb, dtype=np.float64)
    metric_depth = np.asarray(depth, dtype=np.float64)
    world, valid = backproject_depth_world(
        metric_depth,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
    )
    height = np.asarray(extraction.height_image_m, dtype=np.float64)
    h, w = metric_depth.shape[:2]
    roi = extraction.metadata.get("roi_native")
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError("extraction metadata has no roi_native")
    x0, y0, x1, y1 = [int(value) for value in roi]
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[
        max(0, y0) : min(h, y1),
        max(0, x0) : min(w, x1),
    ] = True
    floor_rgb = np.asarray(
        extraction.rgb_floor_median,
        dtype=np.float64,
    ).reshape(1, 1, 3)
    rgb_delta = np.linalg.norm(color[..., :3] - floor_rgb, axis=-1)
    geometric_candidate = (
        valid
        & roi_mask
        & (height >= 0.003)
        & (height <= 0.120)
        & ((height >= 0.007) | (rgb_delta >= 16.0))
    )
    selected_colors = color[
        np.asarray(extraction.selected_mask, dtype=bool)
    ][..., :3]
    if len(selected_colors) == 0:
        raise ValueError("RGBD mask completion requires selected target pixels")
    unique_target_colors = np.unique(
        np.rint(selected_colors).astype(np.int16),
        axis=0,
    ).astype(np.float64)
    low_height_pool = (
        valid
        & roi_mask
        & (height >= 0.0)
        & (height < 0.003)
        & (rgb_delta >= 16.0)
    )
    low_height_target_distance = np.full((h, w), np.inf, dtype=np.float64)
    if np.any(low_height_pool):
        target_color_tree = cKDTree(unique_target_colors)
        low_height_target_distance[low_height_pool] = (
            target_color_tree.query(
                color[low_height_pool, :3],
                k=1,
            )[0]
        )
    low_height_color_threshold = min(float(max_color_delta), 32.0)
    low_height_candidate = low_height_pool & (
        low_height_target_distance <= low_height_color_threshold
    )
    candidate = geometric_candidate | low_height_candidate
    focal_px = (
        float(focal_length)
        / float(horizontal_aperture)
        * float(w)
    )
    expanded, metadata = _expand_component_mask_graph(
        candidate_mask=candidate,
        initial_mask=extraction.selected_mask,
        world_points=world,
        rgb=color,
        depth=metric_depth,
        focal_px=focal_px,
        max_gap_3d_m=float(max_gap_3d_mm) / 1000.0,
        max_gap_px=float(max_gap_px),
        max_color_delta=float(max_color_delta),
        max_hops=int(max_hops),
        max_added_pixel_ratio=float(max_added_pixel_ratio),
    )
    metadata.update(
        {
            "geometric_candidate_pixels": int(
                geometric_candidate.sum()
            ),
            "low_height_pool_pixels": int(low_height_pool.sum()),
            "low_height_color_candidate_pixels": int(
                low_height_candidate.sum()
            ),
            "low_height_added_pixels": int(
                np.count_nonzero(
                    expanded
                    & low_height_candidate
                    & ~np.asarray(extraction.selected_mask, dtype=bool)
                )
            ),
            "low_height_target_color_threshold": float(
                low_height_color_threshold
            ),
            "target_unique_color_count": int(
                len(unique_target_colors)
            ),
        }
    )
    expanded_metadata = {
        **extraction.metadata,
        "rgbd_mask_completion": metadata,
        "selected_pixels_before_rgbd_mask_completion": int(
            np.asarray(extraction.selected_mask, dtype=bool).sum()
        ),
        "selected_pixels_after_rgbd_mask_completion": int(expanded.sum()),
    }
    completed = replace(
        extraction,
        selected_mask=expanded,
        selected_surface_points=world[expanded],
        selected_heights_m=height[expanded],
        metadata=expanded_metadata,
    )
    return completed, metadata


def _image_seed_component_labels(
    selected_mask: np.ndarray,
    *,
    click_u: int,
    click_v: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge depth fragments inside the clicked 2D-connected target."""
    selected = np.asarray(selected_mask, dtype=bool)
    if selected.ndim != 2 or not np.any(selected):
        raise ValueError("invalid image-seed component mask")
    image_labels, component_count = ndimage.label(
        selected,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if int(component_count) <= 0:
        raise ValueError("image-seed mask has no connected component")
    yy, xx = np.nonzero(selected)
    nearest = int(
        np.argmin(
            (xx - int(click_u)) ** 2 + (yy - int(click_v)) ** 2
        )
    )
    clicked_label = int(image_labels[yy[nearest], xx[nearest]])
    target = image_labels == clicked_label
    target_y, target_x = np.nonzero(target)
    labels = np.full(selected.shape, -1, dtype=np.int32)
    labels[target] = 0
    component_for_node = np.zeros(len(target_x), dtype=np.int32)
    return labels, target_y, target_x, component_for_node


def _effective_bridge_contact_margin_m(
    *,
    base_margin_mm: float,
    voxel_margin_scale: float,
    voxel_mm: float,
) -> float:
    base = float(base_margin_mm)
    scale = float(voxel_margin_scale)
    resolution = float(voxel_mm)
    if base < 0.0 or scale < 0.0 or resolution <= 0.0:
        raise ValueError(
            "bridge contact margin inputs must be nonnegative and "
            "voxel_mm must be positive"
        )
    return (base + scale * resolution) / 1000.0


def _axis_capsule_bridge_samples(
    *,
    first_axis_world: np.ndarray,
    second_axis_world: np.ndarray,
    first_radius_m: float,
    second_radius_m: float,
    voxel_m: float,
    radius_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first_axis_world, dtype=np.float64).reshape(3)
    second = np.asarray(second_axis_world, dtype=np.float64).reshape(3)
    first_radius = float(first_radius_m)
    second_radius = float(second_radius_m)
    resolution = float(voxel_m)
    scale = float(radius_scale)
    if (
        not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
        or first_radius <= 0.0
        or second_radius <= 0.0
        or resolution <= 0.0
        or scale <= 0.0
    ):
        raise ValueError("invalid axis capsule bridge inputs")
    axis_gap = float(np.linalg.norm(second - first))
    steps = max(
        2,
        int(math.ceil(axis_gap / (0.75 * resolution))) + 1,
    )
    alpha = np.linspace(0.0, 1.0, steps)
    centers = (
        first.reshape(1, 3)
        + alpha[:, None] * (second - first).reshape(1, 3)
    )
    radii = (
        first_radius
        + alpha * (second_radius - first_radius)
    ) * scale
    return centers, radii


def _skeleton_graph_capsule_samples(
    *,
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    voxel_m: float,
    spacing_scale: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Densify fitted tube sections only along 8-neighbor skeleton edges."""
    sy = np.asarray(skeleton_y, dtype=np.int64).reshape(-1)
    sx = np.asarray(skeleton_x, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    resolution = float(voxel_m)
    scale = float(spacing_scale)
    if (
        len(sy) == 0
        or len(sy) != len(sx)
        or len(sy) != len(center_values)
        or len(sy) != len(radius_values)
        or not np.all(np.isfinite(center_values))
        or not np.all(np.isfinite(radius_values))
        or np.any(radius_values <= 0.0)
        or resolution <= 0.0
        or scale <= 0.0
    ):
        raise ValueError("invalid skeleton graph capsule inputs")

    index_by_pixel = {
        (int(y), int(x)): index
        for index, (y, x) in enumerate(zip(sy, sx))
    }
    center_parts = [center_values]
    radius_parts = [radius_values]
    edge_count = 0
    interpolated_count = 0
    forward_offsets = ((0, 1), (1, -1), (1, 0), (1, 1))
    for first, (y, x) in enumerate(zip(sy, sx)):
        for dy, dx in forward_offsets:
            second = index_by_pixel.get((int(y + dy), int(x + dx)))
            if second is None:
                continue
            edge_count += 1
            first_center = center_values[first]
            second_center = center_values[second]
            gap = float(np.linalg.norm(second_center - first_center))
            if gap <= 1e-12:
                continue
            sample_count = max(
                2,
                int(math.ceil(gap / (scale * resolution))) + 1,
            )
            alpha = np.linspace(0.0, 1.0, sample_count)[1:-1]
            if len(alpha) == 0:
                continue
            center_parts.append(
                first_center.reshape(1, 3)
                + alpha[:, None]
                * (second_center - first_center).reshape(1, 3)
            )
            radius_parts.append(
                radius_values[first]
                + alpha
                * (radius_values[second] - radius_values[first])
            )
            interpolated_count += int(len(alpha))
    return (
        np.concatenate(center_parts, axis=0),
        np.concatenate(radius_parts, axis=0),
        {
            "skeleton_graph_edges": int(edge_count),
            "interpolated_samples": int(interpolated_count),
        },
    )


def _skeleton_chain_order(
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Order a non-branching 8-neighbor skeleton from one endpoint."""
    sy = np.asarray(skeleton_y, dtype=np.int64).reshape(-1)
    sx = np.asarray(skeleton_x, dtype=np.int64).reshape(-1)
    if len(sy) == 0 or len(sy) != len(sx):
        raise ValueError("invalid skeleton chain pixels")
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
    if np.any(degrees > 2):
        raise ValueError("skeleton contains a branch junction")
    endpoints = np.flatnonzero(degrees == 1)
    if len(sy) == 1:
        order = np.array([0], dtype=np.int64)
    elif len(endpoints) != 2:
        raise ValueError("skeleton is not an open chain")
    else:
        ordered = [int(endpoints[0])]
        previous = -1
        current = int(endpoints[0])
        while True:
            candidates = [
                neighbor
                for neighbor in adjacency[current]
                if neighbor != previous
            ]
            if not candidates:
                break
            if len(candidates) != 1:
                raise ValueError("ambiguous skeleton chain traversal")
            next_index = int(candidates[0])
            ordered.append(next_index)
            previous, current = current, next_index
        order = np.asarray(ordered, dtype=np.int64)
    if len(order) != len(sy):
        raise ValueError("skeleton chain is disconnected")
    return order, {
        "chain_sections": int(len(order)),
        "chain_endpoint_count": int(len(endpoints)),
        "chain_max_degree": int(degrees.max(initial=0)),
    }


def _skeleton_graph_chains(
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, Dict[str, Any]]:
    """Split a skeleton graph into maximal chains between graph terminals."""
    sy = np.asarray(skeleton_y, dtype=np.int64).reshape(-1)
    sx = np.asarray(skeleton_x, dtype=np.int64).reshape(-1)
    if len(sy) == 0 or len(sy) != len(sx):
        raise ValueError("invalid skeleton graph pixels")
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
                next_candidates = [
                    value
                    for value in adjacency[current]
                    if value != previous
                ]
                if len(next_candidates) != 1:
                    break
                next_index = int(next_candidates[0])
                next_edge = tuple(sorted((current, next_index)))
                if next_edge in visited_edges:
                    break
                visited_edges.add(next_edge)
                path.append(next_index)
                previous, current = current, next_index
            chains.append(np.asarray(path, dtype=np.int64))

    edge_count = int(sum(degrees) // 2)
    if len(visited_edges) != edge_count:
        raise ValueError("skeleton graph contains an unsupported cycle")
    covered = set(
        int(index)
        for chain in chains
        for index in chain.tolist()
    )
    if len(covered) != len(sy):
        raise ValueError("skeleton graph is disconnected")
    return chains, degrees, {
        "graph_sections": int(len(sy)),
        "graph_edges": edge_count,
        "graph_chains": int(len(chains)),
        "graph_endpoint_count": int(np.count_nonzero(degrees == 1)),
        "graph_junction_count": int(np.count_nonzero(degrees > 2)),
        "graph_max_degree": int(degrees.max(initial=0)),
    }


def _prune_short_skeleton_spurs(
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    *,
    enabled: bool,
    maximum_spur_edges: int = 6,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Remove only short terminal-to-junction artifacts from a skeleton."""
    sy = np.asarray(skeleton_y, dtype=np.int64).reshape(-1)
    sx = np.asarray(skeleton_x, dtype=np.int64).reshape(-1)
    maximum_edges = int(maximum_spur_edges)
    if (
        len(sy) == 0
        or len(sy) != len(sx)
        or maximum_edges < 1
    ):
        raise ValueError("invalid skeleton spur pruning inputs")
    metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": bool(enabled and len(sy) >= 8),
        "maximum_spur_edges": maximum_edges,
        "initial_sections": int(len(sy)),
        "final_sections": int(len(sy)),
        "removed_sections": 0,
        "removed_spurs": 0,
        "iterations": 0,
    }
    if not metadata["enabled"]:
        return sy.copy(), sx.copy(), metadata

    current_y = sy.copy()
    current_x = sx.copy()
    removed_spurs = 0
    iterations = 0
    while len(current_y) >= 3:
        try:
            chains, degrees, _graph_metadata = (
                _skeleton_graph_chains(current_y, current_x)
            )
        except ValueError:
            break
        remove = np.zeros(len(current_y), dtype=bool)
        for chain in chains:
            if len(chain) < 2:
                continue
            first = int(chain[0])
            last = int(chain[-1])
            first_terminal = int(degrees[first]) == 1
            last_terminal = int(degrees[last]) == 1
            first_junction = int(degrees[first]) > 2
            last_junction = int(degrees[last]) > 2
            terminal_to_junction = bool(
                (first_terminal and last_junction)
                or (last_terminal and first_junction)
            )
            edge_count = int(len(chain) - 1)
            if (
                not terminal_to_junction
                or edge_count > maximum_edges
            ):
                continue
            junction = last if last_junction else first
            removable = chain[chain != junction]
            remove[np.asarray(removable, dtype=np.int64)] = True
            removed_spurs += 1
        if not np.any(remove):
            break
        current_y = current_y[~remove]
        current_x = current_x[~remove]
        iterations += 1

    return current_y, current_x, {
        **metadata,
        "final_sections": int(len(current_y)),
        "removed_sections": int(len(sy) - len(current_y)),
        "removed_spurs": int(removed_spurs),
        "iterations": int(iterations),
    }


def _swept_transported_frames(
    centerline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(centerline, dtype=np.float64).reshape(-1, 3)
    if len(centers) < 2 or not np.all(np.isfinite(centers)):
        raise ValueError("invalid swept frame centerline")
    tangents = np.gradient(centers, axis=0)
    tangent_norm = np.linalg.norm(tangents, axis=1)
    if np.any(tangent_norm < 1e-9):
        raise ValueError("swept tube centerline has duplicate sections")
    tangents /= tangent_norm[:, None]
    normals = np.empty_like(tangents)
    hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(hint @ tangents[0])) > 0.9:
        hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    first_normal = hint - tangents[0] * float(hint @ tangents[0])
    normals[0] = first_normal / (
        np.linalg.norm(first_normal) + 1e-12
    )
    for index in range(1, len(tangents)):
        transported = (
            normals[index - 1]
            - tangents[index]
            * float(normals[index - 1] @ tangents[index])
        )
        transported_norm = float(np.linalg.norm(transported))
        if transported_norm < 1e-9:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(fallback @ tangents[index])) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            transported = (
                fallback
                - tangents[index]
                * float(fallback @ tangents[index])
            )
            transported_norm = float(np.linalg.norm(transported))
        normals[index] = transported / (transported_norm + 1e-12)
    binormals = np.cross(tangents, normals)
    binormals /= (
        np.linalg.norm(binormals, axis=1, keepdims=True) + 1e-12
    )
    return tangents, normals, binormals


def _swept_circular_tube_mesh(
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    ring_samples: int = 24,
    ring_phase_rad: float = 0.0,
) -> trimesh.Trimesh:
    """Create a watertight flat-capped tube along an ordered centerline."""
    centerline = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    rings = int(ring_samples)
    ring_phase = float(ring_phase_rad)
    if (
        len(centerline) == 0
        or len(centerline) != len(radius_values)
        or not np.all(np.isfinite(centerline))
        or not np.all(np.isfinite(radius_values))
        or np.any(radius_values <= 0.0)
        or rings < 8
        or not math.isfinite(ring_phase)
    ):
        raise ValueError("invalid swept tube inputs")
    if len(centerline) == 1:
        mesh = trimesh.creation.icosphere(
            subdivisions=3,
            radius=float(radius_values[0]),
        )
        mesh.apply_translation(centerline[0])
        return mesh

    _tangents, normals, binormals = _swept_transported_frames(centerline)

    angles = np.linspace(
        0.0,
        2.0 * math.pi,
        rings,
        endpoint=False,
    ) + ring_phase
    ring_direction = (
        np.cos(angles)[None, :, None] * normals[:, None, :]
        + np.sin(angles)[None, :, None] * binormals[:, None, :]
    )
    vertices = (
        centerline[:, None, :]
        + radius_values[:, None, None] * ring_direction
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
    mesh.remove_unreferenced_vertices()
    if not bool(mesh.is_watertight):
        raise ValueError("swept tube mesh is not watertight")
    if float(mesh.volume) < 0.0:
        mesh.invert()
    return mesh


def _skeleton_graph_swept_tube_mesh(
    *,
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    ring_samples: int = 24,
    ring_phase_rad: float = 0.0,
    endpoint_extension_vectors: Dict[int, np.ndarray] | None = None,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Sweep every skeleton chain and union branch junctions exactly."""
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    chains, degrees, metadata = _skeleton_graph_chains(
        skeleton_y,
        skeleton_x,
    )
    if len(center_values) != len(radius_values) or len(center_values) != len(
        degrees
    ):
        raise ValueError("skeleton swept graph values do not align")
    extension_vectors = endpoint_extension_vectors or {}
    for index, vector in extension_vectors.items():
        if (
            int(index) < 0
            or int(index) >= len(center_values)
            or int(degrees[int(index)]) != 1
            or np.asarray(vector, dtype=np.float64).shape != (3,)
            or not np.all(np.isfinite(vector))
        ):
            raise ValueError("invalid swept endpoint extension vector")
    parts: list[trimesh.Trimesh] = []
    extended_terminal_count = 0
    for chain in chains:
        chain_centers = center_values[chain]
        chain_radii = radius_values[chain]
        first = int(chain[0])
        last = int(chain[-1])
        if first in extension_vectors:
            chain_centers = np.vstack(
                (
                    center_values[first]
                    + np.asarray(
                        extension_vectors[first],
                        dtype=np.float64,
                    ),
                    chain_centers,
                )
            )
            chain_radii = np.concatenate(
                ((radius_values[first],), chain_radii)
            )
            extended_terminal_count += 1
        if last in extension_vectors:
            chain_centers = np.vstack(
                (
                    chain_centers,
                    center_values[last]
                    + np.asarray(
                        extension_vectors[last],
                        dtype=np.float64,
                    ),
                )
            )
            chain_radii = np.concatenate(
                (chain_radii, (radius_values[last],))
            )
            extended_terminal_count += 1
        parts.append(
            _swept_circular_tube_mesh(
                chain_centers,
                chain_radii,
                ring_samples=int(ring_samples),
                ring_phase_rad=float(ring_phase_rad),
            )
        )
    junctions = np.flatnonzero(degrees > 2)
    for junction in junctions:
        sphere = trimesh.creation.icosphere(
            subdivisions=2,
            radius=float(radius_values[junction]),
        )
        sphere.apply_translation(center_values[junction])
        parts.append(sphere)
    if not parts:
        raise ValueError("skeleton swept graph produced no tube parts")
    if len(parts) == 1:
        mesh = parts[0]
        boolean_applied = False
    else:
        try:
            mesh = trimesh.boolean.union(parts, engine="manifold")
        except Exception as exc:
            raise ValueError(
                f"skeleton swept graph boolean union failed: {exc}"
            ) from exc
        boolean_applied = True
    if not isinstance(mesh, trimesh.Trimesh) or not bool(mesh.is_watertight):
        raise ValueError("skeleton swept graph is not watertight")
    return mesh, {
        **metadata,
        "eligible": True,
        "ring_samples": int(ring_samples),
        "ring_phase_rad": float(ring_phase_rad),
        "boolean_part_count": int(len(parts)),
        "boolean_applied": boolean_applied,
        "extended_terminal_count": int(extended_terminal_count),
    }


def _polygon_section_radial_factor(
    angle: np.ndarray,
    *,
    ring_samples: int,
    edge_normal_phase_rad: float,
) -> np.ndarray:
    rings = int(ring_samples)
    phase = float(edge_normal_phase_rad)
    values = np.asarray(angle, dtype=np.float64)
    if rings < 8 or not math.isfinite(phase):
        raise ValueError("invalid polygon radial factor inputs")
    half_sector = math.pi / float(rings)
    sector = 2.0 * half_sector
    local = (
        np.mod(values - phase + half_sector, sector) - half_sector
    )
    return math.cos(half_sector) / np.cos(local)


def _fit_polygon_section_candidate(
    *,
    radial_distance: np.ndarray,
    radial_angle: np.ndarray,
    base_radius: np.ndarray,
    fold_ids: np.ndarray,
    ring_samples: int,
    phase_grid_size: int = 129,
) -> Dict[str, Any]:
    rho = np.asarray(radial_distance, dtype=np.float64).reshape(-1)
    angle = np.asarray(radial_angle, dtype=np.float64).reshape(-1)
    radius = np.asarray(base_radius, dtype=np.float64).reshape(-1)
    folds = np.asarray(fold_ids, dtype=np.int64).reshape(-1)
    rings = int(ring_samples)
    grid_size = int(phase_grid_size)
    if (
        len(rho) == 0
        or len(angle) != len(rho)
        or len(radius) != len(rho)
        or len(folds) != len(rho)
        or not np.all(np.isfinite(rho))
        or not np.all(np.isfinite(angle))
        or not np.all(np.isfinite(radius))
        or np.any(rho <= 0.0)
        or np.any(radius <= 0.0)
        or rings < 8
        or grid_size < 17
    ):
        raise ValueError("invalid polygon section candidate inputs")
    phase_grid = np.linspace(
        0.0,
        2.0 * math.pi / float(rings),
        grid_size,
        endpoint=False,
    )

    def fit(training: np.ndarray) -> tuple[float, float, float]:
        best: tuple[float, float, float] | None = None
        for phase in phase_grid:
            factor = _polygon_section_radial_factor(
                angle,
                ring_samples=rings,
                edge_normal_phase_rad=float(phase),
            )
            model = radius * factor
            scale = float(
                np.median(
                    rho[training]
                    / np.maximum(model[training], 1e-12)
                )
            )
            error = np.abs(rho[training] - scale * model[training])
            score = float(
                np.median(error) + 0.2 * np.percentile(error, 90.0)
            )
            candidate = (score, float(phase), scale)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        return best

    all_points = np.ones(len(rho), dtype=bool)
    full_score, full_phase, full_scale = fit(all_points)
    fold_rows: list[Dict[str, Any]] = []
    for fold in np.unique(folds):
        validation = folds == fold
        training = ~validation
        if (
            np.count_nonzero(validation) == 0
            or np.count_nonzero(training) == 0
        ):
            continue
        _training_score, phase, scale = fit(training)
        prediction = (
            scale
            * radius[validation]
            * _polygon_section_radial_factor(
                angle[validation],
                ring_samples=rings,
                edge_normal_phase_rad=phase,
            )
        )
        error = np.abs(rho[validation] - prediction)
        median_error = float(np.median(error))
        p90_error = float(np.percentile(error, 90.0))
        fold_rows.append(
            {
                "fold": int(fold),
                "edge_normal_phase_rad": phase,
                "radius_scale": scale,
                "median_error_mm": median_error * 1000.0,
                "p90_error_mm": p90_error * 1000.0,
                "score_mm": (
                    median_error + 0.2 * p90_error
                )
                * 1000.0,
            }
        )
    return {
        "ring_samples": rings,
        "fold_count": int(len(fold_rows)),
        "score_mm": float(
            np.mean([row["score_mm"] for row in fold_rows])
            if fold_rows
            else math.inf
        ),
        "full_fit": {
            "score_mm": full_score * 1000.0,
            "edge_normal_phase_rad": full_phase,
            "vertex_phase_rad": (
                full_phase - math.pi / float(rings)
            ),
            "radius_scale": full_scale,
        },
        "folds": fold_rows,
    }


def _select_polygonal_swept_ring_model(
    *,
    chain_order: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    surface_points: np.ndarray,
    surface_point_fold_ids: np.ndarray,
    requested_ring_samples: int,
    enabled: bool,
    candidate_ring_samples: Sequence[int] = (
        8,
        12,
        16,
        20,
        24,
        32,
        48,
        64,
    ),
    minimum_mean_gain_m: float = 5e-6,
    minimum_each_fold_gain_m: float = 4e-6,
) -> tuple[int, float, float, Dict[str, Any]]:
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    points = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    folds = np.asarray(
        surface_point_fold_ids,
        dtype=np.int64,
    ).reshape(-1)
    requested = int(requested_ring_samples)
    candidates = sorted(
        {
            int(value)
            for value in (*candidate_ring_samples, requested)
            if int(value) >= 8
        }
    )
    minimum_mean_gain = float(minimum_mean_gain_m)
    minimum_fold_gain = float(minimum_each_fold_gain_m)
    if (
        len(order) != len(center_values)
        or len(order) != len(radius_values)
        or len(order) < 3
        or len(np.unique(order)) != len(order)
        or np.min(order) < 0
        or np.max(order) >= len(order)
        or len(points) != len(folds)
        or not np.all(np.isfinite(center_values))
        or not np.all(np.isfinite(radius_values))
        or np.any(radius_values <= 0.0)
        or not np.all(np.isfinite(points))
        or requested < 8
        or len(candidates) < 2
        or minimum_mean_gain < 0.0
        or minimum_fold_gain < 0.0
    ):
        raise ValueError("invalid polygonal swept ring selection inputs")
    active = bool(
        enabled
        and len(order) >= 24
        and len(points) >= 48
        and len(np.unique(folds)) >= 3
    )
    metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": active,
        "accepted": False,
        "requested_ring_samples": requested,
        "effective_ring_samples": requested,
        "effective_ring_phase_rad": 0.0,
        "effective_radius_scale": 1.0,
        "minimum_mean_gain_mm": minimum_mean_gain * 1000.0,
        "minimum_each_fold_gain_mm": minimum_fold_gain * 1000.0,
        "surface_points": int(len(points)),
        "sections": int(len(order)),
        "candidates": [],
    }
    if not active:
        return requested, 0.0, 1.0, metadata

    ordered_centers = center_values[order]
    ordered_radii = radius_values[order]
    tangents, normals, binormals = _swept_transported_frames(
        ordered_centers
    )
    _distance, nearest = cKDTree(ordered_centers).query(points, k=1)
    nearest = np.asarray(nearest, dtype=np.int64)
    relative = points - ordered_centers[nearest]
    axial = np.einsum("ij,ij->i", relative, tangents[nearest])
    radial = relative - axial[:, None] * tangents[nearest]
    radial_distance = np.linalg.norm(radial, axis=1)
    valid = radial_distance > 1e-9
    if np.count_nonzero(valid) < 48:
        return requested, 0.0, 1.0, {
            **metadata,
            "enabled": False,
            "valid_surface_points": int(np.count_nonzero(valid)),
        }
    radial = radial[valid]
    nearest = nearest[valid]
    radial_distance = radial_distance[valid]
    radial_angle = np.arctan2(
        np.einsum("ij,ij->i", radial, binormals[nearest]),
        np.einsum("ij,ij->i", radial, normals[nearest]),
    )
    base_radius = ordered_radii[nearest]
    valid_folds = folds[valid]
    candidate_rows = [
        _fit_polygon_section_candidate(
            radial_distance=radial_distance,
            radial_angle=radial_angle,
            base_radius=base_radius,
            fold_ids=valid_folds,
            ring_samples=ring_samples,
        )
        for ring_samples in candidates
    ]
    usable = [
        row
        for row in candidate_rows
        if int(row["fold_count"]) >= 3
        and math.isfinite(float(row["score_mm"]))
    ]
    if len(usable) < 2:
        return requested, 0.0, 1.0, {
            **metadata,
            "enabled": False,
            "valid_surface_points": int(len(radial_distance)),
            "candidates": candidate_rows,
        }
    selected = min(usable, key=lambda row: float(row["score_mm"]))
    smooth_min_samples = max(32, requested)
    smooth_candidates = [
        row
        for row in usable
        if int(row["ring_samples"]) >= smooth_min_samples
    ]
    if not smooth_candidates:
        return requested, 0.0, 1.0, {
            **metadata,
            "enabled": False,
            "valid_surface_points": int(len(radial_distance)),
            "candidates": candidate_rows,
        }
    smooth = min(
        smooth_candidates,
        key=lambda row: float(row["score_mm"]),
    )
    selected_fold_scores = np.asarray(
        [float(row["score_mm"]) for row in selected["folds"]],
        dtype=np.float64,
    )
    smooth_fold_scores = np.min(
        np.vstack(
            [
                np.asarray(
                    [float(fold["score_mm"]) for fold in row["folds"]],
                    dtype=np.float64,
                )
                for row in smooth_candidates
            ]
        ),
        axis=0,
    )
    fold_gain_mm = smooth_fold_scores - selected_fold_scores
    mean_gain_mm = float(
        float(smooth["score_mm"]) - float(selected["score_mm"])
    )
    minimum_fold_gain_mm = float(np.min(fold_gain_mm))
    full_fit = selected["full_fit"]
    radius_scale = float(full_fit["radius_scale"])
    accepted = bool(
        int(selected["ring_samples"]) < smooth_min_samples
        and mean_gain_mm >= minimum_mean_gain * 1000.0
        and minimum_fold_gain_mm >= minimum_fold_gain * 1000.0
        and 0.97 <= radius_scale <= 1.05
    )
    effective_ring_samples = (
        int(selected["ring_samples"]) if accepted else requested
    )
    effective_phase = (
        float(full_fit["vertex_phase_rad"]) if accepted else 0.0
    )
    effective_radius_scale = radius_scale if accepted else 1.0
    metadata.update(
        {
            "accepted": accepted,
            "valid_surface_points": int(len(radial_distance)),
            "smooth_min_ring_samples": int(smooth_min_samples),
            "selected_ring_samples": int(selected["ring_samples"]),
            "selected_score_mm": float(selected["score_mm"]),
            "smooth_ring_samples": int(smooth["ring_samples"]),
            "smooth_score_mm": float(smooth["score_mm"]),
            "mean_gain_mm": mean_gain_mm,
            "minimum_fold_gain_mm": minimum_fold_gain_mm,
            "fold_gain_mm": fold_gain_mm.tolist(),
            "selected_full_fit": full_fit,
            "effective_ring_samples": effective_ring_samples,
            "effective_ring_phase_rad": effective_phase,
            "effective_radius_scale": effective_radius_scale,
            "candidates": candidate_rows,
        }
    )
    return (
        effective_ring_samples,
        effective_phase,
        effective_radius_scale,
        metadata,
    )


def _polygon_facet_bspline_basis(
    parameters: np.ndarray,
    *,
    control_count: int,
) -> np.ndarray:
    values = np.clip(
        np.asarray(parameters, dtype=np.float64),
        0.0,
        1.0,
    )
    degree = 3
    interior_count = int(control_count) - degree - 1
    interior = (
        np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
        if interior_count > 0
        else np.empty(0, dtype=np.float64)
    )
    knots = np.concatenate(
        (
            np.zeros(degree + 1, dtype=np.float64),
            interior,
            np.ones(degree + 1, dtype=np.float64),
        )
    )
    spline = BSpline(
        knots,
        np.eye(int(control_count), dtype=np.float64),
        degree,
        axis=0,
        extrapolate=False,
    )
    return np.asarray(spline(values), dtype=np.float64)


def _polygon_facet_interpolate_rows(
    section_s: np.ndarray,
    values: np.ndarray,
    point_s: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(point_s, section_s, values[:, axis])
            for axis in range(values.shape[1])
        ]
    )


def _polygon_facet_point_geometry(
    *,
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> Dict[str, np.ndarray]:
    section_s, point_s, _distance = _fixed_centerline_point_coordinates(
        points,
        centers,
    )
    tangents, normals, binormals = _swept_transported_frames(centers)
    point_centers = _polygon_facet_interpolate_rows(
        section_s,
        centers,
        point_s,
    )
    point_tangents = _polygon_facet_interpolate_rows(
        section_s,
        tangents,
        point_s,
    )
    point_tangents /= (
        np.linalg.norm(point_tangents, axis=1, keepdims=True) + 1e-12
    )
    point_normals = _polygon_facet_interpolate_rows(
        section_s,
        normals,
        point_s,
    )
    point_normals /= (
        np.linalg.norm(point_normals, axis=1, keepdims=True) + 1e-12
    )
    point_binormals = np.cross(point_tangents, point_normals)
    point_binormals /= (
        np.linalg.norm(point_binormals, axis=1, keepdims=True) + 1e-12
    )
    point_normals = np.cross(point_binormals, point_tangents)
    point_normals /= (
        np.linalg.norm(point_normals, axis=1, keepdims=True) + 1e-12
    )
    point_radii = np.interp(point_s, section_s, radii)
    relative = points - point_centers
    axial = np.einsum("ij,ij->i", relative, point_tangents)
    radial = relative - axial[:, None] * point_tangents
    return {
        "section_s": section_s,
        "point_s": point_s,
        "normals": normals,
        "binormals": binormals,
        "point_centers": point_centers,
        "point_normals": point_normals,
        "point_binormals": point_binormals,
        "point_radii": point_radii,
        "normal_coordinate": np.einsum(
            "ij,ij->i",
            radial,
            point_normals,
        ),
        "binormal_coordinate": np.einsum(
            "ij,ij->i",
            radial,
            point_binormals,
        ),
    }


def _polygon_facet_active_normals(
    normal_coordinate: np.ndarray,
    binormal_coordinate: np.ndarray,
    *,
    ring_samples: int,
    edge_normal_phase_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.arctan2(binormal_coordinate, normal_coordinate)
    sector = 2.0 * math.pi / float(ring_samples)
    facet_index = np.floor(
        (angle - float(edge_normal_phase_rad)) / sector + 0.5
    )
    facet_angle = (
        float(edge_normal_phase_rad) + facet_index * sector
    )
    return np.cos(facet_angle), np.sin(facet_angle)


def _polygon_facet_surface_residual(
    *,
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    ring_samples: int,
    edge_normal_phase_rad: float,
) -> np.ndarray:
    geometry = _polygon_facet_point_geometry(
        points=points,
        centers=centers,
        radii=radii,
    )
    cosine, sine = _polygon_facet_active_normals(
        geometry["normal_coordinate"],
        geometry["binormal_coordinate"],
        ring_samples=int(ring_samples),
        edge_normal_phase_rad=float(edge_normal_phase_rad),
    )
    return (
        cosine * geometry["normal_coordinate"]
        + sine * geometry["binormal_coordinate"]
        - geometry["point_radii"]
        * math.cos(math.pi / float(ring_samples))
    )


def _polygon_facet_ray_residual(
    *,
    points: np.ndarray,
    camera_pos: Sequence[float],
    centers: np.ndarray,
    radii: np.ndarray,
    ring_samples: int,
    edge_normal_phase_rad: float,
) -> np.ndarray:
    geometry = _polygon_facet_point_geometry(
        points=points,
        centers=centers,
        radii=radii,
    )
    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    observed_vector = points - camera.reshape(1, 3)
    observed_distance = np.linalg.norm(observed_vector, axis=1)
    ray_direction = observed_vector / (
        observed_distance[:, None] + 1e-12
    )
    rings = int(ring_samples)
    edge_angles = (
        float(edge_normal_phase_rad)
        + np.linspace(0.0, 2.0 * math.pi, rings, endpoint=False)
    )
    edge_cosine = np.cos(edge_angles)
    edge_sine = np.sin(edge_angles)
    camera_relative = (
        camera.reshape(1, 3) - geometry["point_centers"]
    )
    origin_normal = np.einsum(
        "ij,ij->i",
        camera_relative,
        geometry["point_normals"],
    )
    origin_binormal = np.einsum(
        "ij,ij->i",
        camera_relative,
        geometry["point_binormals"],
    )
    direction_normal = np.einsum(
        "ij,ij->i",
        ray_direction,
        geometry["point_normals"],
    )
    direction_binormal = np.einsum(
        "ij,ij->i",
        ray_direction,
        geometry["point_binormals"],
    )
    denominator = (
        direction_normal[:, None] * edge_cosine[None, :]
        + direction_binormal[:, None] * edge_sine[None, :]
    )
    apothem = (
        geometry["point_radii"]
        * math.cos(math.pi / float(rings))
    )
    numerator = (
        apothem[:, None]
        - origin_normal[:, None] * edge_cosine[None, :]
        - origin_binormal[:, None] * edge_sine[None, :]
    )
    parameter = numerator / np.where(
        np.abs(denominator) > 1e-12,
        denominator,
        np.nan,
    )
    entry = np.max(
        np.where(denominator < -1e-12, parameter, -np.inf),
        axis=1,
    )
    exit_parameter = np.min(
        np.where(denominator > 1e-12, parameter, np.inf),
        axis=1,
    )
    valid = (
        np.isfinite(entry)
        & np.isfinite(exit_parameter)
        & (entry <= exit_parameter + 1e-9)
        & (exit_parameter > 0.0)
    )
    residual = np.full(len(points), math.inf, dtype=np.float64)
    residual[valid] = (
        np.maximum(entry[valid], 0.0) - observed_distance[valid]
    )
    return residual


def _polygon_facet_residual_score(residual: np.ndarray) -> float:
    absolute = np.abs(np.asarray(residual, dtype=np.float64).reshape(-1))
    finite = absolute[np.isfinite(absolute)]
    if len(finite) == 0:
        return math.inf
    return float(
        np.median(finite) + 0.2 * np.percentile(finite, 90.0)
    )


def _fit_polygon_facet_linear_model(
    *,
    centers: np.ndarray,
    radii: np.ndarray,
    points: np.ndarray,
    selected: np.ndarray,
    ring_samples: int,
    edge_normal_phase_rad: float,
    control_spacing_sections: float,
    center_prior_sigma_m: float,
    radius_prior_sigma_m: float,
    curvature_weight: float = 1.0,
    irls_iterations: int = 6,
    data_scale_m: float = 4e-5,
    maximum_center_shift_m: float = 0.0025,
    maximum_radius_shift_m: float = 0.001,
    endpoint_taper_sections: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    base_centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    base_radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    surface = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    chosen = np.asarray(selected, dtype=bool).reshape(-1)
    geometry = _polygon_facet_point_geometry(
        points=surface,
        centers=base_centers,
        radii=base_radii,
    )
    section_s = geometry["section_s"]
    point_s = geometry["point_s"]
    control_count = int(
        np.clip(
            round(len(base_centers) / float(control_spacing_sections)),
            8,
            32,
        )
    )
    point_basis = _polygon_facet_bspline_basis(
        point_s,
        control_count=control_count,
    )
    section_basis = _polygon_facet_bspline_basis(
        section_s,
        control_count=control_count,
    )
    cosine, sine = _polygon_facet_active_normals(
        geometry["normal_coordinate"],
        geometry["binormal_coordinate"],
        ring_samples=int(ring_samples),
        edge_normal_phase_rad=float(edge_normal_phase_rad),
    )
    half_cosine = math.cos(math.pi / float(ring_samples))
    target = (
        cosine * geometry["normal_coordinate"]
        + sine * geometry["binormal_coordinate"]
        - half_cosine * geometry["point_radii"]
    )
    data_matrix = np.hstack(
        (
            cosine[:, None] * point_basis,
            sine[:, None] * point_basis,
            half_cosine * point_basis,
        )
    )
    endpoint_margin = float(endpoint_taper_sections) / max(
        len(base_centers) - 1,
        1,
    )
    interior = (
        (point_s >= endpoint_margin)
        & (point_s <= 1.0 - endpoint_margin)
    )
    training = chosen & interior
    if np.count_nonzero(training) < max(48, 3 * control_count):
        training = chosen
    if np.count_nonzero(training) < 3 * control_count:
        raise ValueError("too few polygon facet observations")

    second_difference = np.zeros(
        (control_count - 2, control_count),
        dtype=np.float64,
    )
    row = np.arange(control_count - 2)
    second_difference[row, row] = 1.0
    second_difference[row, row + 1] = -2.0
    second_difference[row, row + 2] = 1.0
    zero = np.zeros_like(second_difference)
    center_prior = np.zeros(
        (2 * control_count, 3 * control_count),
        dtype=np.float64,
    )
    center_prior[:control_count, :control_count] = np.eye(control_count)
    center_prior[
        control_count:,
        control_count : 2 * control_count,
    ] = np.eye(control_count)
    radius_prior = np.zeros(
        (control_count, 3 * control_count),
        dtype=np.float64,
    )
    radius_prior[:, 2 * control_count :] = np.eye(control_count)
    curvature = np.block(
        [
            [second_difference, zero, zero],
            [zero, second_difference, zero],
            [zero, zero, second_difference],
        ]
    )
    regularization = np.vstack(
        (
            center_prior / float(center_prior_sigma_m),
            radius_prior / float(radius_prior_sigma_m),
            float(curvature_weight)
            * curvature
            / float(center_prior_sigma_m),
        )
    )
    fit_matrix = data_matrix[training] / float(data_scale_m)
    fit_target = target[training] / float(data_scale_m)
    weights = np.ones(len(fit_target), dtype=np.float64)
    parameters = np.zeros(3 * control_count, dtype=np.float64)
    for _iteration in range(int(irls_iterations)):
        root_weight = np.sqrt(weights)
        matrix = np.vstack(
            (
                fit_matrix * root_weight[:, None],
                regularization,
            )
        )
        right = np.concatenate(
            (
                fit_target * root_weight,
                np.zeros(len(regularization), dtype=np.float64),
            )
        )
        parameters, *_ = np.linalg.lstsq(matrix, right, rcond=None)
        residual = (
            fit_matrix @ parameters - fit_target
        ) * float(data_scale_m)
        weights = np.minimum(
            1.0,
            float(data_scale_m) / np.maximum(np.abs(residual), 1e-12),
        )

    first = control_count
    second = 2 * control_count
    normal_offset = section_basis @ parameters[:first]
    binormal_offset = section_basis @ parameters[first:second]
    radius_shift = section_basis @ parameters[second:]
    endpoint_distance = (
        np.minimum(section_s, 1.0 - section_s)
        * max(len(base_centers) - 1, 1)
    )
    endpoint_window = np.clip(
        endpoint_distance / max(float(endpoint_taper_sections), 1e-12),
        0.0,
        1.0,
    )
    endpoint_window = (
        endpoint_window**2 * (3.0 - 2.0 * endpoint_window)
    )
    normal_offset *= endpoint_window
    binormal_offset *= endpoint_window
    radius_shift *= endpoint_window
    center_shift_norm = np.hypot(normal_offset, binormal_offset)
    center_scale = np.minimum(
        1.0,
        float(maximum_center_shift_m)
        / np.maximum(center_shift_norm, 1e-12),
    )
    normal_offset *= center_scale
    binormal_offset *= center_scale
    radius_shift = np.clip(
        radius_shift,
        -float(maximum_radius_shift_m),
        float(maximum_radius_shift_m),
    )
    fitted_centers = (
        base_centers
        + normal_offset[:, None] * geometry["normals"]
        + binormal_offset[:, None] * geometry["binormals"]
    )
    fitted_radii = np.maximum(base_radii + radius_shift, 0.0004)
    center_shift = np.linalg.norm(
        fitted_centers - base_centers,
        axis=1,
    )
    return fitted_centers, fitted_radii, {
        "control_points": control_count,
        "training_points": int(np.count_nonzero(training)),
        "median_center_shift_mm": float(
            np.median(center_shift) * 1000.0
        ),
        "maximum_center_shift_mm": float(
            np.max(center_shift) * 1000.0
        ),
        "median_radius_shift_mm": float(
            np.median(fitted_radii - base_radii) * 1000.0
        ),
        "minimum_radius_shift_mm": float(
            np.min(fitted_radii - base_radii) * 1000.0
        ),
        "maximum_radius_shift_mm": float(
            np.max(fitted_radii - base_radii) * 1000.0
        ),
        "endpoint_taper_sections": float(endpoint_taper_sections),
    }


def _refine_polygonal_swept_facets(
    *,
    ordered_centers: np.ndarray,
    ordered_radii: np.ndarray,
    surface_points: np.ndarray,
    surface_point_fold_ids: np.ndarray,
    camera_pos: Sequence[float],
    ring_samples: int,
    vertex_phase_rad: float,
    enabled: bool,
    control_spacing_candidates: Sequence[float] = (6.0, 8.0, 12.0, 16.0),
    center_prior_sigma_mm_candidates: Sequence[float] = (
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ),
    radius_prior_sigma_mm_candidates: Sequence[float] = (
        0.1,
        0.25,
        0.5,
        1.0,
    ),
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    centers = np.asarray(ordered_centers, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(ordered_radii, dtype=np.float64).reshape(-1)
    points = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    folds = np.asarray(
        surface_point_fold_ids,
        dtype=np.int64,
    ).reshape(-1)
    rings = int(ring_samples)
    edge_phase = float(vertex_phase_rad) + math.pi / float(rings)
    active = bool(
        enabled
        and len(centers) >= 24
        and len(points) >= 48
        and len(points) == len(folds)
        and len(np.unique(folds)) >= 3
        and rings >= 8
    )
    metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": active,
        "accepted": False,
        "sections": int(len(centers)),
        "surface_points": int(len(points)),
        "ring_samples": rings,
        "vertex_phase_rad": float(vertex_phase_rad),
        "edge_normal_phase_rad": edge_phase,
        "candidates": [],
    }
    if not active:
        return centers.copy(), radii.copy(), metadata

    baseline_facet = _polygon_facet_surface_residual(
        points=points,
        centers=centers,
        radii=radii,
        ring_samples=rings,
        edge_normal_phase_rad=edge_phase,
    )
    baseline_ray = _polygon_facet_ray_residual(
        points=points,
        camera_pos=camera_pos,
        centers=centers,
        radii=radii,
        ring_samples=rings,
        edge_normal_phase_rad=edge_phase,
    )
    candidate_rows: list[Dict[str, Any]] = []
    unique_folds = np.unique(folds)
    for spacing in control_spacing_candidates:
        for center_sigma_mm in center_prior_sigma_mm_candidates:
            for radius_sigma_mm in radius_prior_sigma_mm_candidates:
                fold_rows: list[Dict[str, Any]] = []
                for fold in unique_folds:
                    validation = folds == fold
                    try:
                        candidate_centers, candidate_radii, _fit = (
                            _fit_polygon_facet_linear_model(
                                centers=centers,
                                radii=radii,
                                points=points,
                                selected=~validation,
                                ring_samples=rings,
                                edge_normal_phase_rad=edge_phase,
                                control_spacing_sections=float(spacing),
                                center_prior_sigma_m=(
                                    float(center_sigma_mm) / 1000.0
                                ),
                                radius_prior_sigma_m=(
                                    float(radius_sigma_mm) / 1000.0
                                ),
                            )
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        fold_rows = []
                        break
                    facet_score = _polygon_facet_residual_score(
                        _polygon_facet_surface_residual(
                            points=points,
                            centers=candidate_centers,
                            radii=candidate_radii,
                            ring_samples=rings,
                            edge_normal_phase_rad=edge_phase,
                        )[validation]
                    )
                    ray_score = _polygon_facet_residual_score(
                        _polygon_facet_ray_residual(
                            points=points,
                            camera_pos=camera_pos,
                            centers=candidate_centers,
                            radii=candidate_radii,
                            ring_samples=rings,
                            edge_normal_phase_rad=edge_phase,
                        )[validation]
                    )
                    baseline_facet_score = _polygon_facet_residual_score(
                        baseline_facet[validation]
                    )
                    baseline_ray_score = _polygon_facet_residual_score(
                        baseline_ray[validation]
                    )
                    fold_rows.append(
                        {
                            "fold": int(fold),
                            "facet_score_mm": facet_score * 1000.0,
                            "facet_gain_mm": (
                                baseline_facet_score - facet_score
                            )
                            * 1000.0,
                            "ray_score_mm": ray_score * 1000.0,
                            "ray_gain_mm": (
                                baseline_ray_score - ray_score
                            )
                            * 1000.0,
                        }
                    )
                if len(fold_rows) != len(unique_folds):
                    continue
                candidate_rows.append(
                    {
                        "control_spacing_sections": float(spacing),
                        "center_prior_sigma_mm": float(center_sigma_mm),
                        "radius_prior_sigma_mm": float(radius_sigma_mm),
                        "mean_joint_score_mm": float(
                            np.mean(
                                [
                                    row["facet_score_mm"]
                                    + row["ray_score_mm"]
                                    for row in fold_rows
                                ]
                            )
                        ),
                        "mean_facet_gain_mm": float(
                            np.mean(
                                [
                                    row["facet_gain_mm"]
                                    for row in fold_rows
                                ]
                            )
                        ),
                        "minimum_facet_gain_mm": float(
                            np.min(
                                [
                                    row["facet_gain_mm"]
                                    for row in fold_rows
                                ]
                            )
                        ),
                        "mean_ray_gain_mm": float(
                            np.mean(
                                [
                                    row["ray_gain_mm"]
                                    for row in fold_rows
                                ]
                            )
                        ),
                        "minimum_ray_gain_mm": float(
                            np.min(
                                [
                                    row["ray_gain_mm"]
                                    for row in fold_rows
                                ]
                            )
                        ),
                        "folds": fold_rows,
                    }
                )
    if not candidate_rows:
        return centers.copy(), radii.copy(), {
            **metadata,
            "enabled": False,
            "reason": "no_valid_cv_candidate",
        }
    selected = min(
        candidate_rows,
        key=lambda row: float(row["mean_joint_score_mm"]),
    )
    accepted = bool(
        float(selected["mean_facet_gain_mm"]) > 0.0
        and float(selected["minimum_facet_gain_mm"]) >= 0.0
        and float(selected["mean_ray_gain_mm"]) > 0.0
        and float(selected["minimum_ray_gain_mm"]) >= 0.0
    )
    metadata.update(
        {
            "accepted": accepted,
            "baseline_facet_score_mm": (
                _polygon_facet_residual_score(baseline_facet) * 1000.0
            ),
            "baseline_ray_score_mm": (
                _polygon_facet_residual_score(baseline_ray) * 1000.0
            ),
            "selected": selected,
            "candidates": candidate_rows,
        }
    )
    if not accepted:
        return centers.copy(), radii.copy(), metadata
    fitted_centers, fitted_radii, full_fit = (
        _fit_polygon_facet_linear_model(
            centers=centers,
            radii=radii,
            points=points,
            selected=np.ones(len(points), dtype=bool),
            ring_samples=rings,
            edge_normal_phase_rad=edge_phase,
            control_spacing_sections=float(
                selected["control_spacing_sections"]
            ),
            center_prior_sigma_m=(
                float(selected["center_prior_sigma_mm"]) / 1000.0
            ),
            radius_prior_sigma_m=(
                float(selected["radius_prior_sigma_mm"]) / 1000.0
            ),
        )
    )
    metadata.update(
        {
            "full_fit": full_fit,
            "fitted_facet_score_mm": (
                _polygon_facet_residual_score(
                    _polygon_facet_surface_residual(
                        points=points,
                        centers=fitted_centers,
                        radii=fitted_radii,
                        ring_samples=rings,
                        edge_normal_phase_rad=edge_phase,
                    )
                )
                * 1000.0
            ),
            "fitted_ray_score_mm": (
                _polygon_facet_residual_score(
                    _polygon_facet_ray_residual(
                        points=points,
                        camera_pos=camera_pos,
                        centers=fitted_centers,
                        radii=fitted_radii,
                        ring_samples=rings,
                        edge_normal_phase_rad=edge_phase,
                    )
                )
                * 1000.0
            ),
        }
    )
    return fitted_centers, fitted_radii, metadata


def _capped_cylinder_signed_field(
    points: np.ndarray,
    *,
    start: np.ndarray,
    direction: np.ndarray,
    length_m: float,
    radius_m: float,
    axial_margin_m: float,
) -> np.ndarray:
    """Return a conservative signed field for a finite oriented cylinder."""
    values = np.asarray(points, dtype=np.float64)
    origin = np.asarray(start, dtype=np.float64).reshape(3)
    axis = np.asarray(direction, dtype=np.float64).reshape(3)
    length = float(length_m)
    radius = float(radius_m)
    margin = float(axial_margin_m)
    axis_norm = float(np.linalg.norm(axis))
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(axis))
        or axis_norm <= 1e-12
        or length <= 0.0
        or radius <= 0.0
        or margin < 0.0
    ):
        raise ValueError("invalid capped-cylinder field inputs")
    axis = axis / axis_norm
    relative = values - origin
    axial = np.einsum("...i,i->...", relative, axis)
    radial = relative - axial[..., None] * axis
    radial_clearance = radius - np.linalg.norm(radial, axis=-1)
    front_clearance = axial + margin
    back_clearance = length + margin - axial
    return np.minimum(
        radial_clearance,
        np.minimum(front_clearance, back_clearance),
    )


def _capped_cylinder_axial_margin_m(
    *,
    voxel_m: float,
    radius_m: float,
    radius_scale: float,
) -> float:
    """Return flat-cap axial padding with a half-voxel raster allowance."""
    resolution = float(voxel_m)
    radius = float(radius_m)
    scale = float(radius_scale)
    if resolution <= 0.0 or radius <= 0.0 or scale < 0.0:
        raise ValueError("invalid capped-cylinder axial margin inputs")
    return 0.5 * resolution + scale * radius


def _direct_circle_fit_confidence(
    *,
    valid_fits: int,
    skeleton_points: int,
    residual_median_m: float,
    fitted_radius_median_m: float,
) -> tuple[bool, Dict[str, float]]:
    """Gate direct fitted radii using only RGBD fit coverage and residual."""
    valid = int(valid_fits)
    total = int(skeleton_points)
    residual = float(residual_median_m)
    radius = float(fitted_radius_median_m)
    if (
        valid < 0
        or total <= 0
        or valid > total
        or not math.isfinite(residual)
        or not math.isfinite(radius)
        or residual < 0.0
        or radius <= 0.0
    ):
        raise ValueError("invalid direct circle fit confidence inputs")
    fit_fraction = float(valid / total)
    normalized_residual = float(residual / radius)
    minimum_fit_fraction = 0.90
    maximum_normalized_residual = 0.03
    accepted = bool(
        fit_fraction >= minimum_fit_fraction
        and normalized_residual <= maximum_normalized_residual
    )
    return accepted, {
        "fit_fraction": fit_fraction,
        "normalized_residual": normalized_residual,
        "minimum_fit_fraction": minimum_fit_fraction,
        "maximum_normalized_residual": maximum_normalized_residual,
    }


def _select_soft_direct_radius_blend(
    *,
    requested_blend: float,
    direct_fit_accepted: bool,
    fit_fraction: float,
    normalized_residual: float,
    fitted_to_image_radius_ratio: float,
    maximum_soft_blend: float,
    enabled: bool,
) -> tuple[float, Dict[str, Any]]:
    """Conservatively supplement an uncertain silhouette radius fit."""
    requested = float(requested_blend)
    fraction = float(fit_fraction)
    residual = float(normalized_residual)
    radius_ratio = float(fitted_to_image_radius_ratio)
    maximum = float(maximum_soft_blend)
    if (
        not 0.0 <= requested <= 1.0
        or not 0.0 <= fraction <= 1.0
        or not math.isfinite(residual)
        or residual < 0.0
        or not math.isfinite(radius_ratio)
        or radius_ratio <= 0.0
        or not 0.0 <= maximum <= 1.0
    ):
        raise ValueError("invalid soft direct radius blend inputs")
    minimum_fraction = 0.65
    full_fraction = 0.90
    maximum_residual = 0.03
    minimum_radius_ratio = 1.05
    eligible = bool(
        enabled
        and not direct_fit_accepted
        and fraction >= minimum_fraction
        and residual <= maximum_residual
        and radius_ratio >= minimum_radius_ratio
    )
    coverage_weight = float(
        np.clip(
            (fraction - minimum_fraction)
            / (full_fraction - minimum_fraction),
            0.0,
            1.0,
        )
    )
    effective = (
        requested * maximum * coverage_weight if eligible else 0.0
    )
    return effective, {
        "requested": bool(enabled),
        "enabled": bool(eligible and effective > 0.0),
        "effective_blend": effective,
        "maximum_soft_blend": maximum,
        "coverage_weight": coverage_weight,
        "minimum_fit_fraction": minimum_fraction,
        "full_fit_fraction": full_fraction,
        "maximum_normalized_residual": maximum_residual,
        "minimum_fitted_to_image_radius_ratio": minimum_radius_ratio,
        "direct_fit_accepted": bool(direct_fit_accepted),
        "fit_fraction": fraction,
        "normalized_residual": residual,
        "fitted_to_image_radius_ratio": radius_ratio,
    }


def _select_adaptive_direct_center_blend(
    *,
    requested_blend: float,
    direct_fit_accepted: bool,
    nonlinear_refinement_applied: bool,
    observations_per_section: float,
    preselected_adaptive: bool,
    enabled: bool,
) -> tuple[float, Dict[str, Any]]:
    """Use direct centers only for dense fits lacking a local update."""
    requested = float(requested_blend)
    observation_density = float(observations_per_section)
    if (
        not 0.0 <= requested <= 1.0
        or not math.isfinite(observation_density)
        or observation_density < 0.0
    ):
        raise ValueError("invalid adaptive direct center blend inputs")
    minimum_density = 3.0
    adaptive = bool(
        enabled
        and preselected_adaptive
        and requested >= 0.5
        and direct_fit_accepted
        and not nonlinear_refinement_applied
        and observation_density >= minimum_density
    )
    effective = 1.0 if adaptive else requested
    return effective, {
        "requested": bool(enabled),
        "enabled": adaptive,
        "requested_blend": requested,
        "effective_blend": effective,
        "direct_fit_accepted": bool(direct_fit_accepted),
        "nonlinear_refinement_applied": bool(
            nonlinear_refinement_applied
        ),
        "preselected_adaptive": bool(preselected_adaptive),
        "observations_per_section": observation_density,
        "minimum_observations_per_section": minimum_density,
    }


def _select_supported_direct_center_blend(
    *,
    requested_blend: float,
    direct_fit_accepted: bool,
    fitted_to_image_radius_ratio: float,
    initial_ray_perp_p10: float | None,
    observations_per_section: float,
    enabled: bool,
) -> tuple[float, Dict[str, Any]]:
    """Seed supported direct fits with a conservative fitted center."""
    requested = float(requested_blend)
    radius_ratio = float(fitted_to_image_radius_ratio)
    observation_density = float(observations_per_section)
    if (
        not 0.0 <= requested <= 1.0
        or not math.isfinite(radius_ratio)
        or radius_ratio < 0.0
        or (bool(direct_fit_accepted) and radius_ratio <= 0.0)
        or not math.isfinite(observation_density)
        or observation_density < 0.0
    ):
        raise ValueError("invalid supported direct center blend inputs")
    if initial_ray_perp_p10 is None:
        ray_perp = None
    else:
        ray_perp = float(initial_ray_perp_p10)
        if (
            not math.isfinite(ray_perp)
            or not 0.0 <= ray_perp <= 1.0
        ):
            raise ValueError("invalid supported direct center view input")
    minimum_density = 3.0
    minimum_radius_ratio = 1.02
    minimum_ray_perp = 0.4
    adaptive = bool(
        enabled
        and requested == 0.0
        and direct_fit_accepted
        and radius_ratio >= minimum_radius_ratio
        and ray_perp is not None
        and ray_perp >= minimum_ray_perp
    )
    if adaptive and observation_density >= minimum_density:
        effective = 0.5
        mode = "dense_supported"
    elif adaptive:
        effective = 0.1
        mode = "sparse_supported"
    else:
        effective = requested
        mode = "requested"
    return effective, {
        "requested": bool(enabled),
        "enabled": adaptive,
        "requested_blend": requested,
        "effective_blend": effective,
        "mode": mode,
        "direct_fit_accepted": bool(direct_fit_accepted),
        "fitted_to_image_radius_ratio": radius_ratio,
        "initial_ray_perp_p10": ray_perp,
        "observations_per_section": observation_density,
        "minimum_observations_per_section": minimum_density,
        "minimum_fitted_to_image_radius_ratio": minimum_radius_ratio,
        "minimum_ray_perp_p10": minimum_ray_perp,
    }


def _accept_nonlinear_refinement_result(
    *,
    accepted_sections: int,
    accepted_fraction: float,
    minimum_component_fraction: float,
    local_accepted_sections: bool,
) -> bool:
    """Choose whether already-validated local cylinder updates survive."""
    accepted = int(accepted_sections)
    fraction = float(accepted_fraction)
    minimum = float(minimum_component_fraction)
    if (
        accepted < 0
        or not math.isfinite(fraction)
        or not math.isfinite(minimum)
        or fraction < 0.0
        or fraction > 1.0
        or minimum < 0.0
        or minimum > 1.0
    ):
        raise ValueError("invalid nonlinear refinement acceptance inputs")
    if bool(local_accepted_sections):
        return accepted > 0
    return accepted > 0 and fraction >= minimum


def _accept_adaptive_sparse_nonlinear_refinement(
    *,
    accepted_sections: int,
    accepted_fraction: float,
    initial_ray_perp_p10: float | None,
    minimum_sections: int = 8,
    minimum_fraction: float = 0.35,
    maximum_ray_perp_p10: float = 0.4,
) -> bool:
    """Keep sparse local fits when the initial tube is view-degenerate."""
    accepted = int(accepted_sections)
    fraction = float(accepted_fraction)
    minimum_count = int(minimum_sections)
    minimum_ratio = float(minimum_fraction)
    maximum_ray_perp = float(maximum_ray_perp_p10)
    if (
        accepted < 0
        or not math.isfinite(fraction)
        or fraction < 0.0
        or fraction > 1.0
        or minimum_count < 1
        or not math.isfinite(minimum_ratio)
        or minimum_ratio < 0.0
        or minimum_ratio > 1.0
        or not math.isfinite(maximum_ray_perp)
        or maximum_ray_perp < 0.0
        or maximum_ray_perp > 1.0
    ):
        raise ValueError("invalid adaptive nonlinear refinement inputs")
    if initial_ray_perp_p10 is None:
        return False
    ray_perp = float(initial_ray_perp_p10)
    if not math.isfinite(ray_perp) or ray_perp < 0.0 or ray_perp > 1.0:
        raise ValueError("invalid initial ray perpendicular percentile")
    return bool(
        accepted >= minimum_count
        and fraction >= minimum_ratio
        and ray_perp < maximum_ray_perp
    )


def _accept_degenerate_view_front_anchor(
    *,
    accepted_sections: int,
    accepted_fraction: float,
    initial_ray_perp_p10: float | None,
    minimum_fraction: float = 0.35,
    maximum_ray_perp_p10: float = 0.4,
) -> bool:
    """Use metric-depth front anchors only for supported axial views."""
    return _accept_adaptive_sparse_nonlinear_refinement(
        accepted_sections=int(accepted_sections),
        accepted_fraction=float(accepted_fraction),
        initial_ray_perp_p10=initial_ray_perp_p10,
        minimum_fraction=float(minimum_fraction),
        maximum_ray_perp_p10=float(maximum_ray_perp_p10),
    )


def _select_adaptive_radius_curvature_sigma(
    *,
    requested_sigma: float,
    direct_fit_accepted: bool,
    accepted_fraction: float,
    initial_ray_perp_p10: float | None,
    front_anchor_policy_accepted: bool = False,
    observations_per_section: float = 0.0,
) -> tuple[float, Dict[str, Any]]:
    """Strengthen radius smoothness for unsupported axial observations."""
    requested = float(requested_sigma)
    fraction = float(accepted_fraction)
    observation_density = float(observations_per_section)
    if (
        not math.isfinite(requested)
        or requested < 0.0
        or not math.isfinite(fraction)
        or fraction < 0.0
        or fraction > 1.0
        or not math.isfinite(observation_density)
        or observation_density < 0.0
    ):
        raise ValueError("invalid adaptive radius curvature inputs")
    if initial_ray_perp_p10 is None:
        ray_perp = None
    else:
        ray_perp = float(initial_ray_perp_p10)
        if (
            not math.isfinite(ray_perp)
            or ray_perp < 0.0
            or ray_perp > 1.0
        ):
            raise ValueError("invalid radius curvature ray percentile")
    unsupported_axial = bool(
        requested > 0.0
        and not direct_fit_accepted
        and fraction < 0.35
        and ray_perp is not None
        and ray_perp < 0.4
    )
    sparse_front_anchored_direct = bool(
        requested > 0.0
        and direct_fit_accepted
        and front_anchor_policy_accepted
        and observation_density < 2.25
    )
    if unsupported_axial:
        effective = min(requested, 0.01)
        mode = "unsupported_axial"
    elif sparse_front_anchored_direct:
        effective = min(requested, 0.02)
        mode = "sparse_front_anchored_direct"
    else:
        effective = requested
        mode = "requested"
    adaptive = bool(abs(effective - requested) > 1e-12)
    return effective, {
        "requested_sigma": requested,
        "effective_sigma": effective,
        "adaptive": adaptive,
        "mode": mode,
        "maximum_supported_fraction": 0.35,
        "maximum_ray_perp_p10": 0.4,
        "direct_fit_accepted": bool(direct_fit_accepted),
        "accepted_fraction": fraction,
        "initial_ray_perp_p10": ray_perp,
        "front_anchor_policy_accepted": bool(
            front_anchor_policy_accepted
        ),
        "observations_per_section": observation_density,
        "maximum_sparse_front_anchor_observations_per_section": 2.25,
    }


def _select_adaptive_global_sections_per_control(
    *,
    requested_sections_per_control: float,
    direct_fit_accepted: bool,
    accepted_fraction: float,
    initial_ray_perp_p10: float | None,
    nonlinear_refinement_applied: bool,
    observations_per_section: float,
    front_anchor_policy_accepted: bool,
    enabled: bool,
) -> tuple[float, Dict[str, Any]]:
    """Adapt global spline dimension to RGBD cross-section observability."""
    requested = float(requested_sections_per_control)
    fraction = float(accepted_fraction)
    observation_density = float(observations_per_section)
    if (
        not math.isfinite(requested)
        or requested < 4.0
        or not math.isfinite(fraction)
        or fraction < 0.0
        or fraction > 1.0
        or not math.isfinite(observation_density)
        or observation_density < 0.0
    ):
        raise ValueError("invalid adaptive global control inputs")
    if initial_ray_perp_p10 is None:
        ray_perp = None
    else:
        ray_perp = float(initial_ray_perp_p10)
        if (
            not math.isfinite(ray_perp)
            or ray_perp < 0.0
            or ray_perp > 1.0
        ):
            raise ValueError("invalid global control ray percentile")
    low_support = bool(
        enabled
        and not direct_fit_accepted
        and fraction < 0.35
        and ray_perp is not None
    )
    dense_direct_without_local_refinement = bool(
        enabled
        and direct_fit_accepted
        and not nonlinear_refinement_applied
        and observation_density >= 3.0
    )
    sparse_front_anchored_direct = bool(
        direct_fit_accepted
        and front_anchor_policy_accepted
        and nonlinear_refinement_applied
        and observation_density < 2.25
    )
    if dense_direct_without_local_refinement:
        effective = requested * 1.25
        mode = "lower_dimension_dense_direct_without_local_refinement"
    elif sparse_front_anchored_direct:
        effective = max(4.0, requested * 0.75)
        mode = "higher_dimension_sparse_front_anchored_direct"
    elif not low_support:
        effective = requested
        mode = "requested"
    elif ray_perp < 0.4:
        effective = max(4.0, requested * 0.75)
        mode = "higher_dimension_axial"
    else:
        effective = requested * 1.25
        mode = "lower_dimension_oblique"
    return effective, {
        "requested": bool(enabled),
        "enabled": bool(
            low_support
            or dense_direct_without_local_refinement
            or sparse_front_anchored_direct
        ),
        "requested_sections_per_control": requested,
        "effective_sections_per_control": effective,
        "mode": mode,
        "maximum_supported_fraction": 0.35,
        "axial_ray_perp_p10": 0.4,
        "direct_fit_accepted": bool(direct_fit_accepted),
        "accepted_fraction": fraction,
        "initial_ray_perp_p10": ray_perp,
        "nonlinear_refinement_applied": bool(
            nonlinear_refinement_applied
        ),
        "observations_per_section": observation_density,
        "minimum_dense_observations_per_section": 3.0,
        "front_anchor_policy_accepted": bool(
            front_anchor_policy_accepted
        ),
        "maximum_sparse_front_anchor_observations_per_section": 2.25,
    }


def _select_adaptive_point_parameter_refinement_sections(
    *,
    requested_sections: int,
    direct_fit_accepted: bool,
    front_anchor_policy_accepted: bool,
    nonlinear_refinement_applied: bool,
    observations_per_section: float,
    enabled: bool,
) -> tuple[int, Dict[str, Any]]:
    """Enable local 3D point binding when image binding is trustworthy."""
    requested = int(requested_sections)
    observation_density = float(observations_per_section)
    if (
        requested < 0
        or not math.isfinite(observation_density)
        or observation_density < 0.0
    ):
        raise ValueError("invalid adaptive point parameter inputs")
    dense_direct_without_local_refinement = bool(
        direct_fit_accepted
        and not nonlinear_refinement_applied
        and observation_density >= 3.0
    )
    unanchored_direct = bool(
        enabled
        and requested == 0
        and direct_fit_accepted
        and not front_anchor_policy_accepted
        and not dense_direct_without_local_refinement
    )
    anchored_indirect_supported = bool(
        enabled
        and requested == 0
        and not direct_fit_accepted
        and front_anchor_policy_accepted
        and nonlinear_refinement_applied
    )
    adaptive = bool(
        unanchored_direct or anchored_indirect_supported
    )
    effective = 2 if adaptive else requested
    if anchored_indirect_supported:
        mode = "front_anchored_indirect_supported"
    elif unanchored_direct:
        mode = "unanchored_direct"
    else:
        mode = "requested"
    return effective, {
        "requested": bool(enabled),
        "enabled": adaptive,
        "requested_sections": requested,
        "effective_sections": effective,
        "mode": mode,
        "direct_fit_accepted": bool(direct_fit_accepted),
        "front_anchor_policy_accepted": bool(
            front_anchor_policy_accepted
        ),
        "nonlinear_refinement_applied": bool(
            nonlinear_refinement_applied
        ),
        "observations_per_section": observation_density,
        "minimum_dense_observations_per_section": 3.0,
    }


def _select_adaptive_circle_radius_bias_px(
    *,
    requested_bias_px: float,
    direct_fit_accepted: bool,
    direct_fit_fraction: float,
    initial_ray_perp_p10: float | None,
    enabled: bool,
) -> tuple[float, Dict[str, Any]]:
    """Adapt silhouette erosion correction from RGBD observability."""
    requested = float(requested_bias_px)
    fit_fraction = float(direct_fit_fraction)
    if (
        not math.isfinite(requested)
        or requested < 0.0
        or not math.isfinite(fit_fraction)
        or fit_fraction < 0.0
        or fit_fraction > 1.0
    ):
        raise ValueError("invalid adaptive circle radius bias inputs")
    if initial_ray_perp_p10 is None:
        ray_perp = None
    else:
        ray_perp = float(initial_ray_perp_p10)
        if (
            not math.isfinite(ray_perp)
            or ray_perp < 0.0
            or ray_perp > 1.0
        ):
            raise ValueError("invalid circle radius bias ray percentile")
    adaptive = bool(
        enabled
        and abs(requested - 0.25) <= 1e-12
        and not direct_fit_accepted
        and ray_perp is not None
    )
    effective = requested
    mode = "requested"
    if adaptive and ray_perp < 0.4:
        if fit_fraction >= 0.9:
            effective = 0.0
            mode = "axial_high_arc_coverage"
        elif fit_fraction >= 0.85:
            effective = 0.125
            mode = "axial_near_complete_arc_coverage"
    elif adaptive and fit_fraction < 0.8:
        effective = 0.375
        mode = "oblique_sparse_arc_coverage"
    changed = bool(abs(effective - requested) > 1e-12)
    return effective, {
        "requested": bool(enabled),
        "enabled": changed,
        "requested_bias_px": requested,
        "effective_bias_px": effective,
        "mode": mode,
        "direct_fit_accepted": bool(direct_fit_accepted),
        "direct_fit_fraction": fit_fraction,
        "initial_ray_perp_p10": ray_perp,
        "axial_ray_perp_p10": 0.4,
        "high_arc_coverage": 0.9,
        "near_complete_arc_coverage": 0.85,
        "sparse_arc_coverage": 0.8,
    }


def _minimum_covering_arc_span_rad(points_xy: np.ndarray) -> float:
    """Return the smallest circular arc containing all nonzero 2D rays."""
    values = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("arc span requires at least two finite 2D points")
    norms = np.linalg.norm(values, axis=1)
    valid = norms > 1e-12
    if int(np.count_nonzero(valid)) < 2:
        raise ValueError("arc span requires at least two nonzero rays")
    angles = np.mod(
        np.arctan2(values[valid, 1], values[valid, 0]),
        2.0 * math.pi,
    )
    angles.sort()
    wrapped = np.concatenate((angles, angles[:1] + 2.0 * math.pi))
    largest_gap = float(np.max(np.diff(wrapped)))
    return float(2.0 * math.pi - largest_gap)


def _fit_front_anchored_circle_radius(
    section_xy: np.ndarray,
    *,
    minimum_depth_offset_m: float,
    radius_min_m: float,
    radius_max_m: float,
) -> tuple[float, float]:
    """Fit radius with the skeleton sample fixed as the frontmost point."""
    values = np.asarray(section_xy, dtype=np.float64).reshape(-1, 2)
    minimum_offset = float(minimum_depth_offset_m)
    radius_min = float(radius_min_m)
    radius_max = float(radius_max_m)
    if (
        len(values) < 3
        or not np.all(np.isfinite(values))
        or minimum_offset <= 0.0
        or radius_min <= 0.0
        or radius_max <= radius_min
    ):
        raise ValueError("invalid front-anchored circle fit inputs")
    x = values[:, 0]
    y = values[:, 1]
    usable = x < -minimum_offset
    if int(np.count_nonzero(usable)) < 2:
        raise ValueError("front-anchored circle has too little depth sag")
    x = x[usable]
    y = y[usable]
    radius_samples = -(x * x + y * y) / (2.0 * x)
    radius_samples = radius_samples[
        np.isfinite(radius_samples)
        & (radius_samples >= radius_min)
        & (radius_samples <= radius_max)
    ]
    if len(radius_samples) < 2:
        raise ValueError("front-anchored circle has no valid radii")
    radius = float(np.median(radius_samples))
    radial_distance = np.sqrt(
        (values[:, 0] + radius) ** 2 + values[:, 1] ** 2
    )
    residual = float(np.median(np.abs(radial_distance - radius)))
    return radius, residual


def _front_anchored_fit_confidence(
    *,
    valid_fits: int,
    skeleton_points: int,
    residual_median_m: float,
    fitted_radius_median_m: float,
) -> tuple[bool, Dict[str, float]]:
    """Gate the constrained fit by support coverage and normalized residual."""
    valid = int(valid_fits)
    total = int(skeleton_points)
    residual = float(residual_median_m)
    radius = float(fitted_radius_median_m)
    if (
        valid < 0
        or total <= 0
        or valid > total
        or not math.isfinite(residual)
        or not math.isfinite(radius)
        or residual < 0.0
        or radius <= 0.0
    ):
        raise ValueError("invalid front-anchored fit confidence inputs")
    fit_fraction = float(valid / total)
    normalized_residual = float(residual / radius)
    minimum_fit_fraction = 0.65
    maximum_normalized_residual = 0.12
    accepted = bool(
        fit_fraction >= minimum_fit_fraction
        and normalized_residual <= maximum_normalized_residual
    )
    return accepted, {
        "fit_fraction": fit_fraction,
        "normalized_residual": normalized_residual,
        "minimum_fit_fraction": minimum_fit_fraction,
        "maximum_normalized_residual": maximum_normalized_residual,
    }


def _select_adaptive_front_radius_blend(
    *,
    requested_blend: float,
    front_fit_accepted: bool,
    direct_fit_accepted: bool,
    front_radius_median_m: float,
    image_radius_median_m: float,
) -> tuple[float, Dict[str, Any]]:
    """Apply only moderate front-anchored corrections automatically."""
    requested = float(requested_blend)
    front_radius = float(front_radius_median_m)
    image_radius = float(image_radius_median_m)
    if (
        not 0.0 <= requested <= 1.0
        or not math.isfinite(front_radius)
        or not math.isfinite(image_radius)
        or front_radius <= 0.0
        or image_radius <= 0.0
    ):
        raise ValueError("invalid adaptive front radius blend inputs")
    ratio = float(front_radius / image_radius)
    adaptive = bool(
        requested == 0.0
        and front_fit_accepted
        and not direct_fit_accepted
        and 1.05 <= ratio <= 1.25
    )
    effective = (
        requested
        if requested > 0.0 and front_fit_accepted
        else (0.5 if adaptive else 0.0)
    )
    return effective, {
        "requested_blend": requested,
        "effective_blend": effective,
        "adaptive": adaptive,
        "front_to_image_radius_ratio": ratio,
        "minimum_adaptive_ratio": 1.05,
        "maximum_adaptive_ratio": 1.25,
        "front_fit_accepted": bool(front_fit_accepted),
        "direct_fit_accepted": bool(direct_fit_accepted),
    }


def _blend_direct_circle_centers(
    *,
    skeleton_world: np.ndarray,
    fallback_direction: np.ndarray,
    radii: np.ndarray,
    local_center_offset: np.ndarray,
    fitted_radii: np.ndarray,
    center_offset_scale: float,
    scale_min: float,
    scale_max: float,
    requested_blend: float,
    direct_fit_accepted: bool,
) -> tuple[np.ndarray, Dict[str, float | bool]]:
    """Blend RGBD-fitted section centers with the silhouette fallback."""
    surface = np.asarray(skeleton_world, dtype=np.float64).reshape(-1, 3)
    direction = np.asarray(
        fallback_direction,
        dtype=np.float64,
    ).reshape(-1, 3)
    radius = np.asarray(radii, dtype=np.float64).reshape(-1)
    offsets = np.asarray(
        local_center_offset,
        dtype=np.float64,
    ).reshape(-1, 3)
    fit_radius = np.asarray(
        fitted_radii,
        dtype=np.float64,
    ).reshape(-1)
    blend = float(requested_blend)
    offset_scale = float(center_offset_scale)
    lower_scale = float(scale_min)
    upper_scale = float(scale_max)
    if (
        len(surface) == 0
        or len(surface) != len(direction)
        or len(surface) != len(radius)
        or len(surface) != len(offsets)
        or len(surface) != len(fit_radius)
        or not np.all(np.isfinite(surface))
        or not np.all(np.isfinite(direction))
        or not np.all(np.isfinite(radius))
        or not np.all(np.isfinite(offsets))
        or not np.all(np.isfinite(fit_radius))
        or np.any(radius <= 0.0)
        or np.any(fit_radius <= 0.0)
        or offset_scale < 0.0
        or lower_scale <= 0.0
        or upper_scale < lower_scale
        or not 0.0 <= blend <= 1.0
    ):
        raise ValueError("invalid direct circle center inputs")
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
    fallback_offset = direction * (radius * offset_scale)[:, None]
    offset_norm = np.linalg.norm(offsets, axis=1)
    offset_direction = offsets / (offset_norm[:, None] + 1e-12)
    bounded_norm = np.clip(
        offset_norm,
        lower_scale * fit_radius,
        upper_scale * fit_radius,
    )
    bounded_offset = (
        offset_direction * (bounded_norm * offset_scale)[:, None]
    )
    effective_blend = blend if bool(direct_fit_accepted) else 0.0
    completed_offset = (
        (1.0 - effective_blend) * fallback_offset
        + effective_blend * bounded_offset
    )
    centers = surface + completed_offset
    return centers, {
        "requested_direct_center_blend": blend,
        "direct_center_blend": effective_blend,
        "direct_center_fit_accepted": bool(direct_fit_accepted),
        "fitted_center_offset_min_mm": float(offset_norm.min() * 1000.0),
        "fitted_center_offset_median_mm": float(
            np.median(offset_norm) * 1000.0
        ),
        "fitted_center_offset_max_mm": float(offset_norm.max() * 1000.0),
        "bounded_center_offset_min_mm": float(
            bounded_norm.min() * 1000.0
        ),
        "bounded_center_offset_median_mm": float(
            np.median(bounded_norm) * 1000.0
        ),
        "bounded_center_offset_max_mm": float(
            bounded_norm.max() * 1000.0
        ),
    }


def _regularize_skeleton_endpoint_radii(
    *,
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    radii: np.ndarray,
    requested_blend: float,
    direct_fit_accepted: bool,
    endpoint_span_px: float = 18.0,
    fit_span_px: float = 24.0,
    trusted_radii: np.ndarray | None = None,
    initial_ray_perp_p10: float | None = None,
    require_consistent_extrapolation: bool = False,
) -> tuple[np.ndarray, Dict[str, float | int | bool]]:
    """Extrapolate reliable interior radii across cap-biased endpoints."""
    sy = np.asarray(skeleton_y, dtype=np.int64).reshape(-1)
    sx = np.asarray(skeleton_x, dtype=np.int64).reshape(-1)
    values = np.asarray(radii, dtype=np.float64).reshape(-1)
    blend = float(requested_blend)
    endpoint_span = float(endpoint_span_px)
    fit_span = float(fit_span_px)
    trusted = (
        np.asarray(trusted_radii, dtype=np.float64).reshape(-1)
        if trusted_radii is not None
        else values.copy()
    )
    ray_perp_p10 = (
        float(initial_ray_perp_p10)
        if initial_ray_perp_p10 is not None
        else None
    )
    if (
        len(values) == 0
        or len(values) != len(sy)
        or len(values) != len(sx)
        or len(trusted) != len(values)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(trusted))
        or np.any(values <= 0.0)
        or np.any(trusted <= 0.0)
        or not 0.0 <= blend <= 1.0
        or endpoint_span <= 0.0
        or fit_span <= 0.0
        or (
            ray_perp_p10 is not None
            and (
                not math.isfinite(ray_perp_p10)
                or ray_perp_p10 < 0.0
                or ray_perp_p10 > 1.0
            )
        )
        or (
            bool(require_consistent_extrapolation)
            and initial_ray_perp_p10 is None
        )
    ):
        raise ValueError("invalid endpoint radius regularization inputs")
    effective_blend = blend if bool(direct_fit_accepted) else 0.0
    if effective_blend <= 0.0:
        return values.copy(), {
            "requested_endpoint_radius_blend": blend,
            "endpoint_radius_blend": 0.0,
            "endpoint_radius_fit_accepted": bool(direct_fit_accepted),
            "endpoint_span_px": endpoint_span,
            "fit_span_px": fit_span,
            "endpoint_count": 0,
            "regularized_endpoint_count": 0,
            "adjusted_sections": 0,
            "max_radius_adjustment_mm": 0.0,
            "consistent_extrapolation_required": bool(
                require_consistent_extrapolation
            ),
            "consistent_endpoint_count": 0,
        }

    index_by_pixel = {
        (int(y), int(x)): index
        for index, (y, x) in enumerate(zip(sy, sx))
    }
    adjacency: list[list[int]] = [[] for _ in range(len(values))]
    for index, (y, x) in enumerate(zip(sy, sx)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = index_by_pixel.get((int(y + dy), int(x + dx)))
                if neighbor is not None:
                    adjacency[index].append(int(neighbor))

    original = values.copy()
    result = values.copy()
    endpoint_indices = [
        index
        for index, neighbors in enumerate(adjacency)
        if len(neighbors) == 1
    ]
    regularized_endpoints = 0
    overshoot_capped_endpoints = 0
    consistent_endpoints = 0
    endpoint_diagnostics: list[Dict[str, float | int]] = []
    for endpoint in endpoint_indices:
        path = [int(endpoint)]
        distance = [0.0]
        previous = -1
        current = int(endpoint)
        distance_limit = endpoint_span + fit_span + 3.0
        while distance[-1] < distance_limit:
            candidates = [
                neighbor
                for neighbor in adjacency[current]
                if neighbor != previous
            ]
            if len(candidates) != 1:
                break
            next_index = int(candidates[0])
            step = float(
                math.hypot(
                    int(sy[next_index]) - int(sy[current]),
                    int(sx[next_index]) - int(sx[current]),
                )
            )
            path.append(next_index)
            distance.append(distance[-1] + step)
            previous, current = current, next_index

        path_indices = np.asarray(path, dtype=np.int64)
        path_distance = np.asarray(distance, dtype=np.float64)
        fit = (
            (path_distance >= endpoint_span)
            & (path_distance <= endpoint_span + fit_span)
        )
        if int(np.count_nonzero(fit)) < 5:
            continue
        coefficients = np.polyfit(
            path_distance[fit],
            original[path_indices[fit]],
            1,
        )
        predicted = np.polyval(coefficients, path_distance)
        near_fit = (
            (path_distance >= 0.45 * endpoint_span)
            & (path_distance <= endpoint_span + 0.25 * fit_span)
        )
        near_slope = 0.0
        if int(np.count_nonzero(near_fit)) >= 5:
            near_coefficients = np.polyfit(
                path_distance[near_fit],
                original[path_indices[near_fit]],
                1,
            )
            near_slope = float(near_coefficients[0])
        target = path_distance < endpoint_span
        if not np.any(target):
            continue
        target_indices = path_indices[target]
        target_weight = np.clip(
            1.0 - path_distance[target] / endpoint_span,
            0.0,
            1.0,
        )
        interior_median = float(np.median(original[path_indices[fit]]))
        upward_target = np.maximum(
            original[target_indices],
            predicted[target],
        )
        overshoot_capped = bool(
            original[endpoint] < interior_median
            and predicted[0] > interior_median
            and near_slope / interior_median > 0.003
            and coefficients[0] / interior_median < -0.005
        )
        if overshoot_capped:
            upward_target = np.minimum(
                upward_target,
                interior_median,
            )
        upward_target = np.clip(
            upward_target,
            0.67 * interior_median,
            1.5 * interior_median,
        )
        endpoint_erosion_ratio = float(
            original[endpoint] / max(trusted[endpoint], 1e-12)
        )
        slope_supported = bool(
            coefficients[0] / interior_median <= -0.0005
        )
        erosion_supported = bool(
            endpoint_erosion_ratio < 0.85
            and ray_perp_p10 is not None
            and ray_perp_p10 < 0.8
        )
        raw_endpoint_growth_supported = bool(
            predicted[0] > original[endpoint] + 1e-9
        )
        interior_floor_applied = bool(
            require_consistent_extrapolation
            and erosion_supported
            and not slope_supported
            and raw_endpoint_growth_supported
        )
        if interior_floor_applied:
            upward_target = np.maximum(
                upward_target,
                interior_median,
            )
        maximum_candidate_adjustment = float(
            np.max(upward_target - original[target_indices])
        )
        magnitude_supported = bool(
            maximum_candidate_adjustment >= 0.000025
        )
        endpoint_growth_supported = raw_endpoint_growth_supported
        consistent_extrapolation = bool(
            not overshoot_capped
            and (slope_supported or erosion_supported)
            and magnitude_supported
            and endpoint_growth_supported
        )
        if (
            bool(require_consistent_extrapolation)
            and not consistent_extrapolation
        ):
            upward_target = original[target_indices]
        endpoint_result = (
            original[target_indices]
            + effective_blend
            * target_weight
            * (upward_target - original[target_indices])
        )
        result[target_indices] = np.maximum(
            result[target_indices],
            endpoint_result,
        )
        endpoint_diagnostics.append(
            {
                "endpoint_index": int(endpoint),
                "path_sections": int(len(path_indices)),
                "fit_sections": int(np.count_nonzero(fit)),
                "endpoint_original_mm": float(
                    original[endpoint] * 1000.0
                ),
                "endpoint_predicted_mm": float(
                    predicted[0] * 1000.0
                ),
                "endpoint_result_mm": float(
                    result[endpoint] * 1000.0
                ),
                "interior_median_mm": float(
                    interior_median * 1000.0
                ),
                "fit_slope_mm_per_px": float(
                    coefficients[0] * 1000.0
                ),
                "near_slope_mm_per_px": float(
                    near_slope * 1000.0
                ),
                "near_slope_ratio_per_px": float(
                    near_slope / interior_median
                ),
                "fit_slope_ratio_per_px": float(
                    coefficients[0] / interior_median
                ),
                "overshoot_capped": overshoot_capped,
                "trusted_endpoint_radius_mm": float(
                    trusted[endpoint] * 1000.0
                ),
                "endpoint_erosion_ratio": endpoint_erosion_ratio,
                "slope_supported": slope_supported,
                "erosion_supported": erosion_supported,
                "interior_floor_applied": interior_floor_applied,
                "maximum_candidate_adjustment_mm": float(
                    maximum_candidate_adjustment * 1000.0
                ),
                "magnitude_supported": magnitude_supported,
                "endpoint_growth_supported": (
                    endpoint_growth_supported
                ),
                "consistent_extrapolation": (
                    consistent_extrapolation
                ),
            }
        )
        overshoot_capped_endpoints += int(overshoot_capped)
        consistent_endpoints += int(consistent_extrapolation)
        regularized_endpoints += 1

    adjustment = result - original
    return result, {
        "requested_endpoint_radius_blend": blend,
        "endpoint_radius_blend": effective_blend,
        "endpoint_radius_fit_accepted": bool(direct_fit_accepted),
        "endpoint_span_px": endpoint_span,
        "fit_span_px": fit_span,
        "endpoint_count": int(len(endpoint_indices)),
        "regularized_endpoint_count": int(regularized_endpoints),
        "overshoot_capped_endpoint_count": int(
            overshoot_capped_endpoints
        ),
        "consistent_extrapolation_required": bool(
            require_consistent_extrapolation
        ),
        "consistent_endpoint_count": int(consistent_endpoints),
        "adjusted_sections": int(np.count_nonzero(adjustment > 1e-12)),
        "max_radius_adjustment_mm": float(adjustment.max() * 1000.0),
        "endpoint_diagnostics": endpoint_diagnostics,
    }


def _preserve_front_surface_after_radius_change(
    *,
    chain_order: np.ndarray,
    centers: np.ndarray,
    original_radii: np.ndarray,
    adjusted_radii: np.ndarray,
    camera_pos: Sequence[float],
    enabled: bool,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Move centers away from the camera by each radius increase."""
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    original = np.asarray(
        original_radii,
        dtype=np.float64,
    ).reshape(-1)
    adjusted = np.asarray(
        adjusted_radii,
        dtype=np.float64,
    ).reshape(-1)
    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    if (
        len(order) != len(center_values)
        or len(original) != len(center_values)
        or len(adjusted) != len(center_values)
        or len(order) == 0
        or len(np.unique(order)) != len(order)
        or np.min(order) < 0
        or np.max(order) >= len(order)
        or not np.all(np.isfinite(center_values))
        or not np.all(np.isfinite(original))
        or not np.all(np.isfinite(adjusted))
        or not np.all(np.isfinite(camera))
        or np.any(original <= 0.0)
        or np.any(adjusted <= 0.0)
        or np.any(adjusted + 1e-12 < original)
    ):
        raise ValueError("invalid front-surface radius adjustment inputs")
    delta = adjusted - original
    active = bool(enabled and np.any(delta > 1e-12))
    metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": active,
        "adjusted_sections": int(np.count_nonzero(delta > 1e-12)),
        "max_center_shift_mm": 0.0,
    }
    if not active:
        return center_values.copy(), metadata

    ordered_centers = center_values[order]
    tangents = np.gradient(ordered_centers, axis=0)
    tangents /= np.linalg.norm(
        tangents,
        axis=1,
        keepdims=True,
    ) + 1e-12
    toward_camera = camera.reshape(1, 3) - ordered_centers
    toward_camera -= (
        np.einsum("ij,ij->i", toward_camera, tangents)[:, None]
        * tangents
    )
    toward_camera /= np.linalg.norm(
        toward_camera,
        axis=1,
        keepdims=True,
    ) + 1e-12
    shifted = center_values.copy()
    shifted[order] = (
        ordered_centers - toward_camera * delta[order, None]
    )
    metadata["max_center_shift_mm"] = float(
        np.max(delta) * 1000.0
    )
    return shifted, metadata


def _restore_severely_collapsed_endpoint_radii(
    *,
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    fitted_radii: np.ndarray,
    trusted_radii: np.ndarray,
    enabled: bool,
    collapse_ratio: float = 0.67,
    endpoint_span_px: float = 18.0,
    fit_span_px: float = 24.0,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Preserve trusted endpoint lower bounds after a global surface fit."""
    sy = np.asarray(skeleton_y, dtype=np.int64).reshape(-1)
    sx = np.asarray(skeleton_x, dtype=np.int64).reshape(-1)
    fitted = np.asarray(fitted_radii, dtype=np.float64).reshape(-1)
    trusted = np.asarray(trusted_radii, dtype=np.float64).reshape(-1)
    threshold = float(collapse_ratio)
    endpoint_span = float(endpoint_span_px)
    fit_span = float(fit_span_px)
    if (
        len(fitted) == 0
        or len(fitted) != len(sy)
        or len(fitted) != len(sx)
        or len(fitted) != len(trusted)
        or not np.all(np.isfinite(fitted))
        or not np.all(np.isfinite(trusted))
        or np.any(fitted <= 0.0)
        or np.any(trusted <= 0.0)
        or not 0.0 < threshold < 1.0
        or endpoint_span <= 0.0
        or fit_span <= 0.0
    ):
        raise ValueError("invalid collapsed endpoint restoration inputs")
    base_metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": bool(enabled),
        "collapse_ratio": threshold,
        "endpoint_span_px": endpoint_span,
        "fit_span_px": fit_span,
        "endpoint_count": 0,
        "restored_endpoint_count": 0,
        "adjusted_sections": 0,
        "max_radius_adjustment_mm": 0.0,
        "endpoint_diagnostics": [],
    }
    if not enabled:
        return fitted.copy(), base_metadata

    index_by_pixel = {
        (int(y), int(x)): index
        for index, (y, x) in enumerate(zip(sy, sx))
    }
    adjacency: list[list[int]] = [[] for _ in range(len(fitted))]
    for index, (y, x) in enumerate(zip(sy, sx)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = index_by_pixel.get((int(y + dy), int(x + dx)))
                if neighbor is not None:
                    adjacency[index].append(int(neighbor))

    endpoints = [
        index
        for index, neighbors in enumerate(adjacency)
        if len(neighbors) == 1
    ]
    result = fitted.copy()
    diagnostics: list[Dict[str, Any]] = []
    restored_endpoints = 0
    for endpoint in endpoints:
        path = [int(endpoint)]
        distances = [0.0]
        previous = -1
        current = int(endpoint)
        distance_limit = endpoint_span + fit_span + 3.0
        while distances[-1] < distance_limit:
            candidates = [
                neighbor
                for neighbor in adjacency[current]
                if neighbor != previous
            ]
            if len(candidates) != 1:
                break
            next_index = int(candidates[0])
            distances.append(
                distances[-1]
                + float(
                    math.hypot(
                        int(sy[next_index]) - int(sy[current]),
                        int(sx[next_index]) - int(sx[current]),
                    )
                )
            )
            path.append(next_index)
            previous, current = current, next_index

        path_indices = np.asarray(path, dtype=np.int64)
        path_distance = np.asarray(distances, dtype=np.float64)
        fit = (
            (path_distance >= endpoint_span)
            & (path_distance <= endpoint_span + fit_span)
        )
        if int(np.count_nonzero(fit)) < 5:
            continue
        interior_median = float(
            np.median(fitted[path_indices[fit]])
        )
        endpoint_ratio = float(
            fitted[endpoint] / max(interior_median, 1e-12)
        )
        collapsed = bool(endpoint_ratio < threshold)
        target = path_distance < endpoint_span
        if collapsed and np.any(target):
            target_indices = path_indices[target]
            target_weight = np.clip(
                1.0 - path_distance[target] / endpoint_span,
                0.0,
                1.0,
            )
            lower_bound = np.maximum(
                fitted[target_indices],
                trusted[target_indices],
            )
            restored = (
                fitted[target_indices]
                + target_weight
                * (lower_bound - fitted[target_indices])
            )
            result[target_indices] = np.maximum(
                result[target_indices],
                restored,
            )
            restored_endpoints += 1
        diagnostics.append(
            {
                "endpoint_index": int(endpoint),
                "endpoint_radius_mm": float(
                    fitted[endpoint] * 1000.0
                ),
                "trusted_endpoint_radius_mm": float(
                    trusted[endpoint] * 1000.0
                ),
                "interior_median_mm": float(
                    interior_median * 1000.0
                ),
                "endpoint_to_interior_ratio": endpoint_ratio,
                "collapsed": collapsed,
            }
        )

    adjustment = result - fitted
    return result, {
        **base_metadata,
        "endpoint_count": int(len(endpoints)),
        "restored_endpoint_count": int(restored_endpoints),
        "adjusted_sections": int(np.count_nonzero(adjustment > 1e-12)),
        "max_radius_adjustment_mm": float(
            np.max(adjustment) * 1000.0
        ),
        "endpoint_diagnostics": diagnostics,
    }


def _regularize_skeleton_graph_radii_gcv(
    *,
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    radii: np.ndarray,
    enabled: bool,
    direct_fit_accepted: bool,
    minimum_sections: int = 12,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Denoise log radii along skeleton chains with data-selected smoothing."""
    sy = np.asarray(skeleton_y, dtype=np.int64).reshape(-1)
    sx = np.asarray(skeleton_x, dtype=np.int64).reshape(-1)
    values = np.asarray(radii, dtype=np.float64).reshape(-1)
    if (
        len(values) == 0
        or len(values) != len(sy)
        or len(values) != len(sx)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or int(minimum_sections) < 3
    ):
        raise ValueError("invalid skeleton graph radius inputs")
    active = bool(
        enabled
        and direct_fit_accepted
        and len(values) >= int(minimum_sections)
    )
    base_metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": active,
        "direct_fit_accepted": bool(direct_fit_accepted),
        "minimum_sections": int(minimum_sections),
        "sections": int(len(values)),
        "regularized_sections": 0,
        "selected_lambda": 0.0,
        "gcv_score": None,
        "candidate_count": 0,
        "roughness_before": 0.0,
        "roughness_after": 0.0,
        "max_radius_adjustment_mm": 0.0,
    }
    if not active:
        return values.copy(), base_metadata

    index_by_pixel = {
        (int(y), int(x)): index
        for index, (y, x) in enumerate(zip(sy, sx))
    }
    adjacency: list[list[tuple[int, float]]] = [
        [] for _ in range(len(values))
    ]
    for index, (y, x) in enumerate(zip(sy, sx)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = index_by_pixel.get((int(y + dy), int(x + dx)))
                if neighbor is None:
                    continue
                adjacency[index].append(
                    (int(neighbor), float(math.hypot(dy, dx)))
                )

    curvature_rows: list[np.ndarray] = []
    for index, neighbors in enumerate(adjacency):
        if len(neighbors) != 2:
            continue
        (first, first_distance), (second, second_distance) = neighbors
        total_distance = first_distance + second_distance
        if total_distance <= 0.0:
            continue
        row = np.zeros(len(values), dtype=np.float64)
        row[index] = -1.0
        row[first] = second_distance / total_distance
        row[second] = first_distance / total_distance
        curvature_rows.append(row)
    if len(curvature_rows) < 3:
        return values.copy(), base_metadata

    curvature = np.stack(curvature_rows, axis=0)
    penalty = curvature.T @ curvature
    observation = np.log(values)
    identity = np.eye(len(values), dtype=np.float64)
    candidate_lambdas = np.asarray(
        [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        dtype=np.float64,
    )
    best_score = math.inf
    best_lambda = 0.0
    best_smoothed = observation.copy()
    for smoothing_lambda in candidate_lambdas:
        system = identity + float(smoothing_lambda) * penalty
        try:
            smoother = np.linalg.solve(system, identity)
        except np.linalg.LinAlgError:
            continue
        smoothed = smoother @ observation
        residual = observation - smoothed
        effective_fraction = 1.0 - float(
            np.trace(smoother) / len(values)
        )
        if effective_fraction <= 1e-9:
            score = math.inf
        else:
            score = float(
                np.mean(residual * residual)
                / (effective_fraction * effective_fraction)
            )
        if score < best_score:
            best_score = score
            best_lambda = float(smoothing_lambda)
            best_smoothed = smoothed

    result = np.exp(best_smoothed)
    adjustment = result - values
    before = curvature @ observation
    after = curvature @ best_smoothed
    return result, {
        **base_metadata,
        "enabled": True,
        "regularized_sections": int(
            np.count_nonzero(np.abs(adjustment) > 1e-12)
        ),
        "selected_lambda": best_lambda,
        "gcv_score": (
            float(best_score) if math.isfinite(best_score) else None
        ),
        "candidate_count": int(len(candidate_lambdas)),
        "curvature_constraints": int(len(curvature_rows)),
        "roughness_before": float(np.mean(before * before)),
        "roughness_after": float(np.mean(after * after)),
        "max_radius_adjustment_mm": float(
            np.max(np.abs(adjustment)) * 1000.0
        ),
    }


def _regularize_swept_chain_geometry_gcv(
    *,
    chain_order: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Denoise one ordered swept tube using RGBD-only GCV selection."""
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    if (
        len(order) != len(center_values)
        or len(order) != len(radius_values)
        or len(order) == 0
        or len(np.unique(order)) != len(order)
        or np.min(order) < 0
        or np.max(order) >= len(order)
        or not np.all(np.isfinite(center_values))
        or not np.all(np.isfinite(radius_values))
        or np.any(radius_values <= 0.0)
    ):
        raise ValueError("invalid swept chain geometry inputs")
    base_metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": bool(enabled and len(order) >= 12),
        "sections": int(len(order)),
        "selected_center_lambda": 0.0,
        "selected_radius_lambda": 0.0,
        "center_gcv_score": None,
        "radius_gcv_score": None,
        "center_roughness_before": 0.0,
        "center_roughness_after": 0.0,
        "radius_roughness_before": 0.0,
        "radius_roughness_after": 0.0,
        "max_center_adjustment_mm": 0.0,
        "max_radius_adjustment_mm": 0.0,
    }
    if not base_metadata["enabled"]:
        return center_values.copy(), radius_values.copy(), base_metadata

    count = len(order)
    curvature = np.zeros((count - 2, count), dtype=np.float64)
    rows = np.arange(count - 2)
    curvature[rows, rows] = 0.5
    curvature[rows, rows + 1] = -1.0
    curvature[rows, rows + 2] = 0.5
    penalty = curvature.T @ curvature
    identity = np.eye(count, dtype=np.float64)
    candidate_lambdas = np.asarray(
        [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        dtype=np.float64,
    )

    ordered_centers = center_values[order]
    ordered_log_radii = np.log(radius_values[order])

    def select(
        observation: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        values = np.asarray(observation, dtype=np.float64)
        best_score = math.inf
        best_lambda = 0.0
        best_smoothed = values.copy()
        for smoothing_lambda in candidate_lambdas:
            system = identity + float(smoothing_lambda) * penalty
            try:
                smoother = np.linalg.solve(system, identity)
            except np.linalg.LinAlgError:
                continue
            smoothed = smoother @ values
            residual = values - smoothed
            denominator = 1.0 - float(np.trace(smoother) / count)
            if denominator <= 1e-9:
                continue
            score = float(
                np.mean(residual * residual)
                / (denominator * denominator)
            )
            if score < best_score:
                best_score = score
                best_lambda = float(smoothing_lambda)
                best_smoothed = smoothed
        return best_smoothed, best_lambda, best_score

    smooth_centers, center_lambda, center_score = select(
        ordered_centers
    )
    smooth_log_radii, radius_lambda, radius_score = select(
        ordered_log_radii
    )
    smooth_radii = np.exp(smooth_log_radii)

    result_centers = center_values.copy()
    result_radii = radius_values.copy()
    result_centers[order] = smooth_centers
    result_radii[order] = smooth_radii
    center_before = curvature @ ordered_centers
    center_after = curvature @ smooth_centers
    radius_before = curvature @ ordered_log_radii
    radius_after = curvature @ smooth_log_radii
    return result_centers, result_radii, {
        **base_metadata,
        "selected_center_lambda": center_lambda,
        "selected_radius_lambda": radius_lambda,
        "center_gcv_score": (
            float(center_score) if math.isfinite(center_score) else None
        ),
        "radius_gcv_score": (
            float(radius_score) if math.isfinite(radius_score) else None
        ),
        "center_roughness_before": float(
            np.mean(center_before * center_before)
        ),
        "center_roughness_after": float(
            np.mean(center_after * center_after)
        ),
        "radius_roughness_before": float(
            np.mean(radius_before * radius_before)
        ),
        "radius_roughness_after": float(
            np.mean(radius_after * radius_after)
        ),
        "max_center_adjustment_mm": float(
            np.max(
                np.linalg.norm(
                    smooth_centers - ordered_centers,
                    axis=1,
                )
            )
            * 1000.0
        ),
        "max_radius_adjustment_mm": float(
            np.max(np.abs(smooth_radii - radius_values[order]))
            * 1000.0
        ),
    }


def _optimize_swept_chain_surface_global(
    *,
    chain_order: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    point_parameters: np.ndarray,
    surface_points: np.ndarray,
    enabled: bool,
    radius_curvature_sigma: float = 0.0,
    sections_per_control: float = 8.0,
    center_prior_sigma_m: float = 0.0015,
    front_anchor_points: np.ndarray | None = None,
    camera_pos: Sequence[float] | None = None,
    front_anchor_sigma_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit a low-dimensional swept tube to all visible RGBD points."""
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    point_s = np.asarray(point_parameters, dtype=np.float64).reshape(-1)
    points = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    absolute_radius_curvature_sigma = float(radius_curvature_sigma)
    control_spacing = float(sections_per_control)
    requested_center_prior_sigma = float(center_prior_sigma_m)
    anchor_sigma = float(front_anchor_sigma_m)
    anchor_enabled = bool(anchor_sigma > 0.0)
    anchors = (
        np.asarray(front_anchor_points, dtype=np.float64).reshape(-1, 3)
        if front_anchor_points is not None
        else np.empty((0, 3), dtype=np.float64)
    )
    camera = (
        np.asarray(camera_pos, dtype=np.float64).reshape(3)
        if camera_pos is not None
        else np.zeros(3, dtype=np.float64)
    )
    if (
        len(order) != len(center_values)
        or len(order) != len(radius_values)
        or len(order) == 0
        or len(np.unique(order)) != len(order)
        or np.min(order) < 0
        or np.max(order) >= len(order)
        or len(point_s) != len(points)
        or not np.all(np.isfinite(center_values))
        or not np.all(np.isfinite(radius_values))
        or np.any(radius_values <= 0.0)
        or not np.all(np.isfinite(point_s))
        or not np.all(np.isfinite(points))
        or np.any(point_s < 0.0)
        or np.any(point_s > 1.0)
        or absolute_radius_curvature_sigma < 0.0
        or not math.isfinite(control_spacing)
        or control_spacing < 4.0
        or not math.isfinite(requested_center_prior_sigma)
        or requested_center_prior_sigma <= 0.0
        or anchor_sigma < 0.0
        or (
            anchor_enabled
            and (
                len(anchors) != len(center_values)
                or not np.all(np.isfinite(anchors))
                or camera_pos is None
                or not np.all(np.isfinite(camera))
            )
        )
    ):
        raise ValueError("invalid global swept surface fit inputs")
    active = bool(enabled and len(order) >= 24 and len(points) >= 48)
    base_metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": active,
        "accepted": False,
        "sections": int(len(order)),
        "surface_points": int(len(points)),
        "control_points": 0,
        "base_residual_median_mm": None,
        "fitted_residual_median_mm": None,
        "residual_gain": None,
        "max_center_shift_mm": 0.0,
        "radius_ratio_min": 1.0,
        "radius_ratio_median": 1.0,
        "radius_ratio_max": 1.0,
        "radius_curvature_sigma": absolute_radius_curvature_sigma,
        "sections_per_control": control_spacing,
        "center_prior_sigma_mm": (
            requested_center_prior_sigma * 1000.0
        ),
        "radius_curvature_regularization_enabled": bool(
            absolute_radius_curvature_sigma > 0.0
        ),
        "front_anchor_enabled": anchor_enabled,
        "front_anchor_sigma_mm": anchor_sigma * 1000.0,
        "base_front_anchor_residual_median_mm": None,
        "fitted_front_anchor_residual_median_mm": None,
    }
    if not active:
        return center_values.copy(), radius_values.copy(), base_metadata

    ordered_centers = center_values[order]
    ordered_radii = radius_values[order]
    section_step = np.linalg.norm(
        np.diff(ordered_centers, axis=0),
        axis=1,
    )
    section_s = np.concatenate(([0.0], np.cumsum(section_step)))
    if float(section_s[-1]) <= 1e-9:
        return center_values.copy(), radius_values.copy(), base_metadata
    section_s /= float(section_s[-1])

    observations_per_section = float(len(points) / len(order))
    control_count = int(
        np.clip(round(len(order) / control_spacing), 8, 20)
    )
    degree = 3
    interior_count = control_count - degree - 1
    interior = (
        np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
        if interior_count > 0
        else np.empty(0, dtype=np.float64)
    )
    knots = np.concatenate(
        (
            np.zeros(degree + 1, dtype=np.float64),
            interior,
            np.ones(degree + 1, dtype=np.float64),
        )
    )
    basis_spline = BSpline(
        knots,
        np.eye(control_count, dtype=np.float64),
        degree,
        axis=0,
        extrapolate=False,
    )
    section_basis = np.asarray(
        basis_spline(section_s),
        dtype=np.float64,
    )
    point_basis = np.asarray(
        basis_spline(point_s),
        dtype=np.float64,
    )
    point_derivative = np.asarray(
        basis_spline.derivative(1)(point_s),
        dtype=np.float64,
    )
    section_derivative = np.asarray(
        basis_spline.derivative(1)(section_s),
        dtype=np.float64,
    )
    curvature = np.zeros(
        (control_count - 2, control_count),
        dtype=np.float64,
    )
    rows = np.arange(control_count - 2)
    curvature[rows, rows] = 0.5
    curvature[rows, rows + 1] = -1.0
    curvature[rows, rows + 2] = 0.5
    ridge = 0.03
    system = (
        section_basis.T @ section_basis
        + ridge * curvature.T @ curvature
        + 1e-8 * np.eye(control_count)
    )
    initial_control_centers = np.linalg.solve(
        system,
        section_basis.T @ ordered_centers,
    )
    initial_control_log_radii = np.linalg.solve(
        system,
        section_basis.T @ np.log(ordered_radii),
    )

    def decode(
        parameters: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        split = control_count * 3
        control_centers = parameters[:split].reshape(control_count, 3)
        control_log_radii = parameters[split:]
        return control_centers, control_log_radii

    initial_parameters = np.concatenate(
        (
            initial_control_centers.reshape(-1),
            initial_control_log_radii,
        )
    )

    def data_residual(parameters: np.ndarray) -> np.ndarray:
        control_centers, control_log_radii = decode(parameters)
        point_centers = point_basis @ control_centers
        point_tangents = point_derivative @ control_centers
        point_tangents /= (
            np.linalg.norm(point_tangents, axis=1, keepdims=True)
            + 1e-12
        )
        relative = points - point_centers
        axial = np.einsum("ij,ij->i", relative, point_tangents)
        radial = np.linalg.norm(
            relative - axial[:, None] * point_tangents,
            axis=1,
        )
        point_radii = np.exp(point_basis @ control_log_radii)
        return radial - point_radii

    def front_anchor_residual(parameters: np.ndarray) -> np.ndarray:
        control_centers, control_log_radii = decode(parameters)
        section_centers = section_basis @ control_centers
        section_tangents = section_derivative @ control_centers
        section_tangents /= (
            np.linalg.norm(section_tangents, axis=1, keepdims=True)
            + 1e-12
        )
        view = camera.reshape(1, 3) - section_centers
        view -= (
            np.einsum("ij,ij->i", view, section_tangents)[:, None]
            * section_tangents
        )
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
        section_radii = np.exp(section_basis @ control_log_radii)
        predicted_front = section_centers + view * section_radii[:, None]
        delta = predicted_front - anchors[order]
        delta -= (
            np.einsum("ij,ij->i", delta, section_tangents)[:, None]
            * section_tangents
        )
        return delta.reshape(-1)

    center_prior_sigma = requested_center_prior_sigma
    log_radius_prior_sigma = 0.20
    center_delta_curvature_sigma = 0.0005
    log_radius_delta_curvature_sigma = 0.08
    data_sigma = 0.00015

    def objective(parameters: np.ndarray) -> np.ndarray:
        control_centers, control_log_radii = decode(parameters)
        center_delta = control_centers - initial_control_centers
        radius_delta = (
            control_log_radii - initial_control_log_radii
        )
        residual_parts = [
            data_residual(parameters) / data_sigma,
            center_delta.reshape(-1) / center_prior_sigma,
            radius_delta / log_radius_prior_sigma,
            (curvature @ center_delta).reshape(-1)
            / center_delta_curvature_sigma,
            (curvature @ radius_delta)
            / log_radius_delta_curvature_sigma,
        ]
        if absolute_radius_curvature_sigma > 0.0:
            residual_parts.append(
                (curvature @ control_log_radii)
                / absolute_radius_curvature_sigma
            )
        if anchor_enabled:
            residual_parts.append(
                front_anchor_residual(parameters) / anchor_sigma
            )
        return np.concatenate(residual_parts)

    center_lower = (
        initial_control_centers - 0.003
    ).reshape(-1)
    center_upper = (
        initial_control_centers + 0.003
    ).reshape(-1)
    radius_lower = initial_control_log_radii - math.log(1.6)
    radius_upper = initial_control_log_radii + math.log(1.6)
    lower = np.concatenate((center_lower, radius_lower))
    upper = np.concatenate((center_upper, radius_upper))
    base_residual = np.abs(data_residual(initial_parameters))
    base_front_anchor_residual = (
        np.linalg.norm(
            front_anchor_residual(initial_parameters).reshape(-1, 3),
            axis=1,
        )
        if anchor_enabled
        else np.empty(0, dtype=np.float64)
    )
    try:
        optimization = least_squares(
            objective,
            initial_parameters,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=120,
        )
    except Exception:
        return center_values.copy(), radius_values.copy(), {
            **base_metadata,
            "control_points": control_count,
            "base_residual_median_mm": float(
                np.median(base_residual) * 1000.0
            ),
        }

    fitted_control_centers, fitted_control_log_radii = decode(
        optimization.x
    )
    fitted_ordered_centers = section_basis @ fitted_control_centers
    fitted_ordered_radii = np.exp(
        section_basis @ fitted_control_log_radii
    )
    prior_ordered_centers = section_basis @ initial_control_centers
    prior_ordered_radii = np.exp(
        section_basis @ initial_control_log_radii
    )
    fitted_residual = np.abs(data_residual(optimization.x))
    fitted_front_anchor_residual = (
        np.linalg.norm(
            front_anchor_residual(optimization.x).reshape(-1, 3),
            axis=1,
        )
        if anchor_enabled
        else np.empty(0, dtype=np.float64)
    )
    raw_center_shift = np.linalg.norm(
        fitted_ordered_centers - ordered_centers,
        axis=1,
    )
    raw_radius_ratio = fitted_ordered_radii / ordered_radii
    prior_center_shift = np.linalg.norm(
        fitted_ordered_centers - prior_ordered_centers,
        axis=1,
    )
    prior_radius_ratio = fitted_ordered_radii / prior_ordered_radii
    base_median = float(np.median(base_residual))
    fitted_median = float(np.median(fitted_residual))
    residual_gain = base_median / max(fitted_median, 1e-12)
    accepted = bool(
        optimization.success
        and residual_gain >= 1.15
        and float(np.max(prior_center_shift)) <= 0.0035
        and float(np.min(prior_radius_ratio)) >= 0.6
        and float(np.max(prior_radius_ratio)) <= 1.7
    )
    result_centers = center_values.copy()
    result_radii = radius_values.copy()
    if accepted:
        result_centers[order] = fitted_ordered_centers
        result_radii[order] = fitted_ordered_radii
    return result_centers, result_radii, {
        **base_metadata,
        "accepted": accepted,
        "control_points": control_count,
        "observations_per_section": observations_per_section,
        "sections_per_control": control_spacing,
        "optimizer_success": bool(optimization.success),
        "optimizer_cost": float(optimization.cost),
        "optimizer_nfev": int(optimization.nfev),
        "base_residual_median_mm": base_median * 1000.0,
        "fitted_residual_median_mm": fitted_median * 1000.0,
        "residual_gain": residual_gain,
        "max_center_shift_mm": float(
            np.max(raw_center_shift) * 1000.0
        ),
        "radius_ratio_min": float(np.min(raw_radius_ratio)),
        "radius_ratio_median": float(np.median(raw_radius_ratio)),
        "radius_ratio_max": float(np.max(raw_radius_ratio)),
        "prior_max_center_shift_mm": float(
            np.max(prior_center_shift) * 1000.0
        ),
        "prior_radius_ratio_min": float(
            np.min(prior_radius_ratio)
        ),
        "prior_radius_ratio_median": float(
            np.median(prior_radius_ratio)
        ),
        "prior_radius_ratio_max": float(
            np.max(prior_radius_ratio)
        ),
        "base_front_anchor_residual_median_mm": (
            float(np.median(base_front_anchor_residual) * 1000.0)
            if anchor_enabled
            else None
        ),
        "fitted_front_anchor_residual_median_mm": (
            float(np.median(fitted_front_anchor_residual) * 1000.0)
            if anchor_enabled
            else None
        ),
    }


def _swept_surface_point_residuals(
    *,
    chain_order: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    point_parameters: np.ndarray,
    surface_points: np.ndarray,
) -> np.ndarray:
    """Evaluate visible radial residuals against sampled swept sections."""
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    point_s = np.asarray(point_parameters, dtype=np.float64).reshape(-1)
    points = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    if (
        len(order) != len(center_values)
        or len(order) != len(radius_values)
        or len(order) < 2
        or len(np.unique(order)) != len(order)
        or np.min(order) < 0
        or np.max(order) >= len(order)
        or len(point_s) != len(points)
        or not np.all(np.isfinite(center_values))
        or not np.all(np.isfinite(radius_values))
        or np.any(radius_values <= 0.0)
        or not np.all(np.isfinite(point_s))
        or not np.all(np.isfinite(points))
        or np.any(point_s < 0.0)
        or np.any(point_s > 1.0)
    ):
        raise ValueError("invalid swept surface residual inputs")
    ordered_centers = center_values[order]
    ordered_radii = radius_values[order]
    section_step = np.linalg.norm(
        np.diff(ordered_centers, axis=0),
        axis=1,
    )
    section_s = np.concatenate(([0.0], np.cumsum(section_step)))
    if float(section_s[-1]) <= 1e-9:
        raise ValueError("degenerate swept section chain")
    section_s /= float(section_s[-1])
    point_centers = np.column_stack(
        [
            np.interp(point_s, section_s, ordered_centers[:, axis])
            for axis in range(3)
        ]
    )
    section_tangents = np.gradient(ordered_centers, section_s, axis=0)
    section_tangents /= (
        np.linalg.norm(section_tangents, axis=1, keepdims=True)
        + 1e-12
    )
    point_tangents = np.column_stack(
        [
            np.interp(point_s, section_s, section_tangents[:, axis])
            for axis in range(3)
        ]
    )
    point_tangents /= (
        np.linalg.norm(point_tangents, axis=1, keepdims=True)
        + 1e-12
    )
    point_radii = np.exp(
        np.interp(point_s, section_s, np.log(ordered_radii))
    )
    relative = points - point_centers
    axial = np.einsum("ij,ij->i", relative, point_tangents)
    radial = np.linalg.norm(
        relative - axial[:, None] * point_tangents,
        axis=1,
    )
    return np.abs(radial - point_radii)


def _fixed_centerline_point_coordinates(
    points: np.ndarray,
    ordered_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segments = np.diff(ordered_centers, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 1e-9:
        raise ValueError("fixed centerline has zero length")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    section_s = cumulative / total_length
    relative = points[:, None, :] - ordered_centers[:-1][None, :, :]
    segment_parameter = np.clip(
        np.einsum("nsi,si->ns", relative, segments)
        / np.maximum(segment_lengths[None, :] ** 2, 1e-12),
        0.0,
        1.0,
    )
    projection = (
        ordered_centers[:-1][None, :, :]
        + segment_parameter[:, :, None] * segments[None, :, :]
    )
    distance_sq = np.einsum(
        "nsi,nsi->ns",
        points[:, None, :] - projection,
        points[:, None, :] - projection,
    )
    nearest_segment = np.argmin(distance_sq, axis=1)
    row = np.arange(len(points), dtype=np.int64)
    nearest_parameter = segment_parameter[row, nearest_segment]
    point_s = (
        cumulative[nearest_segment]
        + nearest_parameter * segment_lengths[nearest_segment]
    ) / total_length
    return section_s, point_s, np.sqrt(distance_sq[row, nearest_segment])


def _estimate_fixed_centerline_radius_profile(
    *,
    section_s: np.ndarray,
    point_s: np.ndarray,
    radial_distance: np.ndarray,
    ordered_original: np.ndarray,
    quantile: float,
    bandwidth_sections: float,
    blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_bandwidth = float(bandwidth_sections) / max(
        len(ordered_original) - 1,
        1,
    )
    estimated = np.empty(len(ordered_original), dtype=np.float64)
    effective_samples = np.empty(len(ordered_original), dtype=np.float64)
    for section, parameter in enumerate(section_s):
        weights = np.exp(
            -0.5 * ((point_s - parameter) / normalized_bandwidth) ** 2
        )
        keep = weights > 1e-4
        local_values = radial_distance[keep]
        local_weights = weights[keep]
        if len(local_values) == 0:
            estimated[section] = ordered_original[section]
            effective_samples[section] = 0.0
            continue
        value_order = np.argsort(local_values)
        sorted_values = local_values[value_order]
        sorted_weights = local_weights[value_order]
        threshold = float(quantile) * float(np.sum(sorted_weights))
        selected = int(
            np.searchsorted(
                np.cumsum(sorted_weights),
                threshold,
                side="left",
            )
        )
        estimated[section] = sorted_values[
            min(selected, len(sorted_values) - 1)
        ]
        effective_samples[section] = float(
            np.sum(local_weights) ** 2
            / max(np.sum(local_weights * local_weights), 1e-12)
        )
    result = ordered_original + float(blend) * (
        estimated - ordered_original
    )
    return (
        np.clip(result, 0.67 * ordered_original, 1.5 * ordered_original),
        effective_samples,
    )


def _refine_swept_radii_from_fixed_centerline(
    *,
    chain_order: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    surface_points: np.ndarray,
    enabled: bool,
    quantile: float = 0.60,
    bandwidth_sections: float = 6.0,
    blend: float = 1.0,
    surface_point_fold_ids: np.ndarray | None = None,
    cross_validate: bool = False,
    cross_validation_min_gain_m: float = 1e-6,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Estimate a radius field and reject it on held-out RGBD residuals."""
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    points = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    selected_quantile = float(quantile)
    bandwidth = float(bandwidth_sections)
    selected_blend = float(blend)
    minimum_gain = float(cross_validation_min_gain_m)
    folds = (
        None
        if surface_point_fold_ids is None
        else np.asarray(surface_point_fold_ids, dtype=np.int64).reshape(-1)
    )
    if (
        len(order) != len(center_values)
        or len(order) != len(radius_values)
        or len(order) < 3
        or len(np.unique(order)) != len(order)
        or np.min(order) < 0
        or np.max(order) >= len(order)
        or not np.all(np.isfinite(center_values))
        or not np.all(np.isfinite(radius_values))
        or np.any(radius_values <= 0.0)
        or not np.all(np.isfinite(points))
        or not 0.0 < selected_quantile < 1.0
        or not math.isfinite(bandwidth)
        or bandwidth <= 0.0
        or not math.isfinite(selected_blend)
        or not 0.0 <= selected_blend <= 1.0
        or not math.isfinite(minimum_gain)
        or minimum_gain < 0.0
        or (folds is not None and len(folds) != len(points))
        or (cross_validate and folds is None)
    ):
        raise ValueError("invalid fixed-centerline radius inputs")
    active = bool(enabled and len(points) >= 2 * len(order))
    metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": active,
        "applied": False,
        "sections": int(len(order)),
        "surface_points": int(len(points)),
        "quantile": selected_quantile,
        "bandwidth_sections": bandwidth,
        "blend": selected_blend,
        "adjusted_sections": 0,
        "median_adjustment_mm": 0.0,
        "max_absolute_adjustment_mm": 0.0,
        "cross_validation": {
            "requested": bool(cross_validate),
            "enabled": False,
            "accepted": not bool(cross_validate),
            "minimum_gain_mm": minimum_gain * 1000.0,
        },
    }
    if not active:
        return radius_values.copy(), metadata

    ordered_centers = center_values[order]
    ordered_original = radius_values[order]
    try:
        section_s, point_s, radial_distance = (
            _fixed_centerline_point_coordinates(points, ordered_centers)
        )
    except ValueError:
        return radius_values.copy(), {**metadata, "enabled": False}
    ordered_candidate, effective_samples = (
        _estimate_fixed_centerline_radius_profile(
            section_s=section_s,
            point_s=point_s,
            radial_distance=radial_distance,
            ordered_original=ordered_original,
            quantile=selected_quantile,
            bandwidth_sections=bandwidth,
            blend=selected_blend,
        )
    )

    cross_validation_metadata: Dict[str, Any] = {
        **metadata["cross_validation"],
    }
    accepted = True
    if cross_validate:
        unique_folds = np.unique(folds)
        baseline_losses: list[float] = []
        candidate_losses: list[float] = []
        for fold in unique_folds:
            validation = folds == fold
            training = ~validation
            if (
                np.count_nonzero(validation) == 0
                or np.count_nonzero(training) == 0
            ):
                continue
            fold_candidate, _effective = (
                _estimate_fixed_centerline_radius_profile(
                    section_s=section_s,
                    point_s=point_s[training],
                    radial_distance=radial_distance[training],
                    ordered_original=ordered_original,
                    quantile=selected_quantile,
                    bandwidth_sections=bandwidth,
                    blend=selected_blend,
                )
            )
            validation_s = point_s[validation]
            validation_radial = radial_distance[validation]
            baseline_prediction = np.interp(
                validation_s,
                section_s,
                ordered_original,
            )
            candidate_prediction = np.interp(
                validation_s,
                section_s,
                fold_candidate,
            )
            baseline_losses.append(
                float(
                    np.mean(
                        np.abs(validation_radial - baseline_prediction)
                    )
                )
            )
            candidate_losses.append(
                float(
                    np.mean(
                        np.abs(validation_radial - candidate_prediction)
                    )
                )
            )
        cv_enabled = len(baseline_losses) >= 2
        baseline_loss = (
            float(np.mean(baseline_losses)) if cv_enabled else math.inf
        )
        candidate_loss = (
            float(np.mean(candidate_losses)) if cv_enabled else math.inf
        )
        gain = baseline_loss - candidate_loss
        accepted = bool(cv_enabled and gain >= minimum_gain)
        cross_validation_metadata = {
            **cross_validation_metadata,
            "enabled": cv_enabled,
            "accepted": accepted,
            "fold_count": int(len(baseline_losses)),
            "baseline_mae_mm": (
                baseline_loss * 1000.0 if cv_enabled else None
            ),
            "candidate_mae_mm": (
                candidate_loss * 1000.0 if cv_enabled else None
            ),
            "gain_mm": gain * 1000.0 if cv_enabled else None,
        }
    if not accepted:
        return radius_values.copy(), {
            **metadata,
            "cross_validation": cross_validation_metadata,
        }

    result = radius_values.copy()
    result[order] = ordered_candidate
    adjustment = ordered_candidate - ordered_original
    return result, {
        **metadata,
        "applied": True,
        "adjusted_sections": int(
            np.count_nonzero(np.abs(adjustment) > 1e-12)
        ),
        "median_adjustment_mm": float(
            np.median(adjustment) * 1000.0
        ),
        "max_absolute_adjustment_mm": float(
            np.max(np.abs(adjustment)) * 1000.0
        ),
        "effective_sample_count_min": float(
            np.min(effective_samples)
        ),
        "effective_sample_count_median": float(
            np.median(effective_samples)
        ),
        "cross_validation": cross_validation_metadata,
    }


def _select_global_sections_per_control_cv(
    *,
    chain_order: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    point_parameters: np.ndarray,
    surface_points: np.ndarray,
    requested_sections_per_control: float,
    radius_curvature_sigma: float,
    enabled: bool,
    fold_count: int = 3,
    block_sections: int = 4,
) -> tuple[float, Dict[str, Any]]:
    """Choose global spline complexity from blocked visible-point holdouts."""
    requested = float(requested_sections_per_control)
    folds = int(fold_count)
    block = int(block_sections)
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    point_s = np.asarray(point_parameters, dtype=np.float64).reshape(-1)
    points = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    if (
        not math.isfinite(requested)
        or requested < 4.0
        or folds < 2
        or block < 1
    ):
        raise ValueError("invalid global control cross-validation inputs")
    base_metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": bool(
            enabled and len(order) >= 24 and len(points) >= 72
        ),
        "requested_sections_per_control": requested,
        "selected_sections_per_control": requested,
        "fold_count": folds,
        "block_sections": block,
        "candidates": [],
    }
    if not base_metadata["enabled"]:
        base_metadata["reason"] = "disabled_or_insufficient_visible_data"
        return requested, base_metadata
    if (
        len(point_s) != len(points)
        or not np.all(np.isfinite(point_s))
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("invalid global control cross-validation inputs")
    candidate_values = sorted(
        {
            max(4.0, requested * scale)
            for scale in (0.75, 1.0, 1.25)
        }
    )
    ordered_centers = np.asarray(centers, dtype=np.float64)[order]
    section_step = np.linalg.norm(
        np.diff(ordered_centers, axis=0),
        axis=1,
    )
    section_s = np.concatenate(([0.0], np.cumsum(section_step)))
    if float(section_s[-1]) <= 1e-9:
        return requested, base_metadata
    section_s /= float(section_s[-1])
    section_midpoint = 0.5 * (section_s[:-1] + section_s[1:])
    point_section = np.searchsorted(
        section_midpoint,
        point_s,
        side="right",
    )
    point_fold = (point_section // block) % folds

    best_spacing = requested
    best_score = math.inf
    candidate_metadata: list[Dict[str, Any]] = []
    for spacing in candidate_values:
        fold_residuals: list[np.ndarray] = []
        fold_radius_predictions: list[np.ndarray] = []
        accepted_folds = 0
        for fold in range(folds):
            validation = point_fold == fold
            training = ~validation
            if (
                int(np.count_nonzero(validation)) < 8
                or int(np.count_nonzero(training)) < 48
            ):
                continue
            fitted_centers, fitted_radii, fit_metadata = (
                _optimize_swept_chain_surface_global(
                    chain_order=order,
                    centers=centers,
                    radii=radii,
                    point_parameters=point_s[training],
                    surface_points=points[training],
                    enabled=True,
                    radius_curvature_sigma=float(
                        radius_curvature_sigma
                    ),
                    sections_per_control=spacing,
                )
            )
            if not bool(fit_metadata.get("accepted", False)):
                continue
            fold_residuals.append(
                _swept_surface_point_residuals(
                    chain_order=order,
                    centers=fitted_centers,
                    radii=fitted_radii,
                    point_parameters=point_s[validation],
                    surface_points=points[validation],
                )
            )
            fold_radius_predictions.append(
                np.asarray(fitted_radii, dtype=np.float64)[order]
            )
            accepted_folds += 1
        if accepted_folds == folds:
            residual = np.concatenate(fold_residuals)
            median_residual = float(np.median(residual))
            p90_residual = float(np.percentile(residual, 90.0))
            radius_stack = np.stack(fold_radius_predictions, axis=0)
            radius_instability = float(
                np.median(np.std(radius_stack, axis=0))
            )
            score = (
                median_residual
                + 0.20 * p90_residual
                + 0.50 * radius_instability
            )
        else:
            median_residual = math.inf
            p90_residual = math.inf
            radius_instability = math.inf
            score = math.inf
        candidate_metadata.append(
            {
                "sections_per_control": spacing,
                "accepted_folds": accepted_folds,
                "median_holdout_residual_mm": (
                    median_residual * 1000.0
                ),
                "p90_holdout_residual_mm": p90_residual * 1000.0,
                "median_radius_instability_mm": (
                    radius_instability * 1000.0
                ),
                "score_mm": score * 1000.0,
            }
        )
        if score < best_score:
            best_score = score
            best_spacing = spacing
    return best_spacing, {
        **base_metadata,
        "selected_sections_per_control": best_spacing,
        "candidates": candidate_metadata,
    }


def _select_global_center_prior_sigma_cv(
    *,
    chain_order: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    point_parameters: np.ndarray,
    surface_points: np.ndarray,
    requested_center_prior_sigma_m: float,
    radius_curvature_sigma: float,
    sections_per_control: float,
    enabled: bool,
    fold_count: int = 3,
    block_sections: int = 4,
    minimum_score_gain_m: float = 1e-6,
) -> tuple[float, Dict[str, Any]]:
    """Choose center freedom from blocked held-out RGBD surface points."""
    requested = float(requested_center_prior_sigma_m)
    radius_sigma = float(radius_curvature_sigma)
    spacing = float(sections_per_control)
    folds = int(fold_count)
    block = int(block_sections)
    minimum_gain = float(minimum_score_gain_m)
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    point_s = np.asarray(point_parameters, dtype=np.float64).reshape(-1)
    points = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    if (
        not math.isfinite(requested)
        or requested <= 0.0
        or not math.isfinite(radius_sigma)
        or radius_sigma < 0.0
        or not math.isfinite(spacing)
        or spacing < 4.0
        or folds < 2
        or block < 1
        or not math.isfinite(minimum_gain)
        or minimum_gain < 0.0
        or len(order) < 24
        or len(point_s) != len(points)
        or not np.all(np.isfinite(point_s))
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("invalid global center-prior CV inputs")
    candidate_values = sorted(
        {
            requested * scale
            for scale in (0.5, 2.0 / 3.0, 1.0, 4.0 / 3.0, 2.0)
        }
    )
    base_metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": bool(enabled and len(points) >= 72),
        "requested_sigma_mm": requested * 1000.0,
        "selected_sigma_mm": requested * 1000.0,
        "minimum_score_gain_mm": minimum_gain * 1000.0,
        "score_gain_mm": 0.0,
        "fold_count": folds,
        "block_sections": block,
        "candidates": [],
    }
    if not base_metadata["enabled"]:
        return requested, base_metadata

    ordered_centers = np.asarray(centers, dtype=np.float64)[order]
    section_step = np.linalg.norm(
        np.diff(ordered_centers, axis=0),
        axis=1,
    )
    section_s = np.concatenate(([0.0], np.cumsum(section_step)))
    if float(section_s[-1]) <= 1e-9:
        return requested, base_metadata
    section_s /= float(section_s[-1])
    section_midpoint = 0.5 * (section_s[:-1] + section_s[1:])
    point_section = np.searchsorted(
        section_midpoint,
        point_s,
        side="right",
    )
    point_fold = (point_section // block) % folds

    candidate_metadata: list[Dict[str, Any]] = []
    score_by_sigma: Dict[float, float] = {}
    for center_sigma in candidate_values:
        fold_residuals: list[np.ndarray] = []
        fold_radius_predictions: list[np.ndarray] = []
        accepted_folds = 0
        for fold in range(folds):
            validation = point_fold == fold
            training = ~validation
            if (
                int(np.count_nonzero(validation)) < 8
                or int(np.count_nonzero(training)) < 48
            ):
                continue
            fitted_centers, fitted_radii, fit_metadata = (
                _optimize_swept_chain_surface_global(
                    chain_order=order,
                    centers=centers,
                    radii=radii,
                    point_parameters=point_s[training],
                    surface_points=points[training],
                    enabled=True,
                    radius_curvature_sigma=radius_sigma,
                    sections_per_control=spacing,
                    center_prior_sigma_m=center_sigma,
                )
            )
            if not bool(fit_metadata.get("accepted", False)):
                continue
            fold_residuals.append(
                _swept_surface_point_residuals(
                    chain_order=order,
                    centers=fitted_centers,
                    radii=fitted_radii,
                    point_parameters=point_s[validation],
                    surface_points=points[validation],
                )
            )
            fold_radius_predictions.append(
                np.asarray(fitted_radii, dtype=np.float64)[order]
            )
            accepted_folds += 1
        if accepted_folds == folds:
            residual = np.concatenate(fold_residuals)
            median_residual = float(np.median(residual))
            p90_residual = float(np.percentile(residual, 90.0))
            radius_stack = np.stack(fold_radius_predictions, axis=0)
            radius_instability = float(
                np.median(np.std(radius_stack, axis=0))
            )
            score = (
                median_residual
                + 0.20 * p90_residual
                + 0.50 * radius_instability
            )
        else:
            median_residual = math.inf
            p90_residual = math.inf
            radius_instability = math.inf
            score = math.inf
        score_by_sigma[center_sigma] = score
        candidate_metadata.append(
            {
                "center_prior_sigma_mm": center_sigma * 1000.0,
                "accepted_folds": accepted_folds,
                "median_holdout_residual_mm": (
                    median_residual * 1000.0
                ),
                "p90_holdout_residual_mm": p90_residual * 1000.0,
                "median_radius_instability_mm": (
                    radius_instability * 1000.0
                ),
                "score_mm": score * 1000.0,
            }
        )

    requested_score = score_by_sigma.get(requested, math.inf)
    best_sigma = min(score_by_sigma, key=score_by_sigma.get)
    best_score = score_by_sigma[best_sigma]
    score_gain = requested_score - best_score
    selected = (
        best_sigma
        if (
            math.isfinite(best_score)
            and math.isfinite(requested_score)
            and score_gain >= minimum_gain
        )
        else requested
    )
    return selected, {
        **base_metadata,
        "selected_sigma_mm": selected * 1000.0,
        "score_gain_mm": (
            score_gain * 1000.0 if math.isfinite(score_gain) else None
        ),
        "candidates": candidate_metadata,
    }


def _refine_surface_point_parameters_local_3d(
    *,
    chain_order: np.ndarray,
    centers: np.ndarray,
    initial_point_parameters: np.ndarray,
    surface_points: np.ndarray,
    max_shift_sections: int,
    blend: float = 1.0,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Project visible points onto nearby 3D centerline segments."""
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    initial_s = np.asarray(
        initial_point_parameters,
        dtype=np.float64,
    ).reshape(-1)
    points = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    shift_limit = int(max_shift_sections)
    applied_blend = float(blend)
    if (
        len(order) != len(center_values)
        or len(order) < 2
        or len(np.unique(order)) != len(order)
        or np.min(order) < 0
        or np.max(order) >= len(order)
        or len(initial_s) != len(points)
        or not np.all(np.isfinite(center_values))
        or not np.all(np.isfinite(initial_s))
        or not np.all(np.isfinite(points))
        or np.any(initial_s < 0.0)
        or np.any(initial_s > 1.0)
        or shift_limit < 0
        or not math.isfinite(applied_blend)
        or not 0.0 <= applied_blend <= 1.0
    ):
        raise ValueError("invalid local 3D point parameter inputs")
    base_metadata: Dict[str, Any] = {
        "requested_max_shift_sections": shift_limit,
        "enabled": bool(shift_limit > 0 and len(points) > 0),
        "point_count": int(len(points)),
        "blend": applied_blend,
        "median_shift_sections": 0.0,
        "p90_shift_sections": 0.0,
        "max_shift_sections": 0.0,
        "median_axis_distance_before_mm": None,
        "median_axis_distance_after_mm": None,
    }
    if not base_metadata["enabled"]:
        return initial_s.copy(), base_metadata

    ordered_centers = center_values[order]
    segment = ordered_centers[1:] - ordered_centers[:-1]
    segment_length_sq = np.einsum("ij,ij->i", segment, segment)
    section_step = np.sqrt(np.maximum(segment_length_sq, 0.0))
    section_s = np.concatenate(([0.0], np.cumsum(section_step)))
    if float(section_s[-1]) <= 1e-9:
        return initial_s.copy(), base_metadata
    section_s /= float(section_s[-1])
    section_midpoint = 0.5 * (section_s[:-1] + section_s[1:])
    initial_section = np.searchsorted(
        section_midpoint,
        initial_s,
        side="right",
    )
    projected_s = initial_s.copy()
    before_distance = np.empty(len(points), dtype=np.float64)
    section_shift = np.empty(len(points), dtype=np.float64)
    for point_index, point in enumerate(points):
        nearest_section = int(initial_section[point_index])
        first_segment = max(0, nearest_section - shift_limit)
        last_segment = min(
            len(segment) - 1,
            nearest_section + shift_limit,
        )
        segment_index = np.arange(
            first_segment,
            last_segment + 1,
            dtype=np.int64,
        )
        relative = point.reshape(1, 3) - ordered_centers[segment_index]
        parameter = np.clip(
            np.einsum(
                "ij,ij->i",
                relative,
                segment[segment_index],
            )
            / np.maximum(segment_length_sq[segment_index], 1e-12),
            0.0,
            1.0,
        )
        projection = (
            ordered_centers[segment_index]
            + parameter[:, None] * segment[segment_index]
        )
        distance_sq = np.einsum(
            "ij,ij->i",
            point.reshape(1, 3) - projection,
            point.reshape(1, 3) - projection,
        )
        local_best = int(np.argmin(distance_sq))
        selected_segment = int(segment_index[local_best])
        selected_parameter = float(parameter[local_best])
        projected_value = (
            (1.0 - selected_parameter) * section_s[selected_segment]
            + selected_parameter * section_s[selected_segment + 1]
        )
        projected_s[point_index] = projected_value
        initial_center = np.column_stack(
            [
                np.interp(
                    [initial_s[point_index]],
                    section_s,
                    ordered_centers[:, axis],
                )
                for axis in range(3)
            ]
        ).reshape(3)
        before_distance[point_index] = float(
            np.linalg.norm(point - initial_center)
        )
        section_shift[point_index] = abs(
            selected_segment + selected_parameter - nearest_section
        )
    refined_s = (
        initial_s + applied_blend * (projected_s - initial_s)
    )
    final_centers = np.column_stack(
        [
            np.interp(
                refined_s,
                section_s,
                ordered_centers[:, axis],
            )
            for axis in range(3)
        ]
    )
    after_distance = np.linalg.norm(points - final_centers, axis=1)
    return refined_s, {
        **base_metadata,
        "median_shift_sections": float(np.median(section_shift)),
        "p90_shift_sections": float(
            np.percentile(section_shift, 90.0)
        ),
        "max_shift_sections": float(np.max(section_shift)),
        "median_axis_distance_before_mm": float(
            np.median(before_distance) * 1000.0
        ),
        "median_axis_distance_after_mm": float(
            np.median(after_distance) * 1000.0
        ),
    }


def _repair_axial_swept_endpoint_radii(
    *,
    chain_order: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    camera_pos: Sequence[float],
    enabled: bool,
    ray_perp_threshold: float = 0.75,
    fit_sections: int = 24,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Extrapolate radii over end-on tube sections with weak visible arcs."""
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    threshold = float(ray_perp_threshold)
    if (
        len(order) != len(center_values)
        or len(order) != len(radius_values)
        or len(order) == 0
        or len(np.unique(order)) != len(order)
        or np.min(order) < 0
        or np.max(order) >= len(order)
        or not np.all(np.isfinite(center_values))
        or not np.all(np.isfinite(radius_values))
        or np.any(radius_values <= 0.0)
        or not np.all(np.isfinite(camera))
        or not 0.0 < threshold < 1.0
        or int(fit_sections) < 5
    ):
        raise ValueError("invalid axial endpoint radius repair inputs")
    metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": bool(enabled and len(order) >= 24),
        "ray_perp_threshold": threshold,
        "fit_sections": int(fit_sections),
        "repaired_endpoints": 0,
        "adjusted_sections": 0,
        "max_radius_adjustment_mm": 0.0,
        "endpoint_diagnostics": [],
    }
    if not metadata["enabled"]:
        return radius_values.copy(), metadata

    ordered_centers = center_values[order]
    ordered_radii = radius_values[order]
    tangent = np.gradient(ordered_centers, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
    view_ray = ordered_centers - camera.reshape(1, 3)
    view_ray /= np.linalg.norm(view_ray, axis=1, keepdims=True) + 1e-12
    ray_perp = np.linalg.norm(
        view_ray
        - np.einsum("ij,ij->i", view_ray, tangent)[:, None] * tangent,
        axis=1,
    )
    arc_step = np.linalg.norm(
        np.diff(ordered_centers, axis=0),
        axis=1,
    )
    arc = np.concatenate(([0.0], np.cumsum(arc_step)))

    repaired = ordered_radii.copy()
    endpoint_diagnostics: list[Dict[str, Any]] = []
    for endpoint_name, reverse in (("start", False), ("end", True)):
        local_radii = repaired[::-1].copy() if reverse else repaired.copy()
        local_perp = ray_perp[::-1] if reverse else ray_perp
        local_arc = (
            float(arc[-1]) - arc[::-1] if reverse else arc.copy()
        )
        endpoint_perp = float(np.median(local_perp[: min(5, len(order))]))
        diagnostic: Dict[str, Any] = {
            "endpoint": endpoint_name,
            "endpoint_ray_perp_median": endpoint_perp,
            "triggered": False,
            "unreliable_sections": 0,
            "fit_start": None,
            "fit_stop": None,
            "max_radius_adjustment_mm": 0.0,
        }
        if endpoint_perp >= threshold:
            endpoint_diagnostics.append(diagnostic)
            continue

        stable = ndimage.uniform_filter1d(
            local_perp,
            size=5,
            mode="nearest",
        )
        candidates = np.flatnonzero(stable >= threshold)
        candidates = candidates[candidates >= 4]
        if len(candidates) == 0:
            endpoint_diagnostics.append(diagnostic)
            continue
        unreliable_stop = int(candidates[0])
        maximum_unreliable = max(6, int(round(0.35 * len(order))))
        unreliable_stop = min(unreliable_stop, maximum_unreliable)
        fit_start = min(unreliable_stop + 2, len(order) - 5)
        fit_stop = min(fit_start + int(fit_sections), len(order))
        if fit_stop - fit_start < 5:
            endpoint_diagnostics.append(diagnostic)
            continue

        fit_x = local_arc[fit_start:fit_stop]
        fit_y = local_radii[fit_start:fit_stop]
        coefficients = np.polyfit(fit_x, fit_y, 1)
        predicted = np.polyval(
            coefficients,
            local_arc[:unreliable_stop],
        )
        interior_median = float(np.median(fit_y))
        predicted = np.clip(
            predicted,
            0.65 * interior_median,
            1.5 * interior_median,
        )
        target = np.maximum(
            local_radii[:unreliable_stop],
            predicted,
        )
        adjustment = target - local_radii[:unreliable_stop]
        local_radii[:unreliable_stop] = target
        repaired = local_radii[::-1].copy() if reverse else local_radii
        diagnostic.update(
            {
                "triggered": True,
                "unreliable_sections": int(unreliable_stop),
                "fit_start": int(fit_start),
                "fit_stop": int(fit_stop),
                "interior_radius_median_mm": interior_median * 1000.0,
                "max_radius_adjustment_mm": float(
                    np.max(adjustment) * 1000.0
                ),
            }
        )
        endpoint_diagnostics.append(diagnostic)

    result = radius_values.copy()
    result[order] = repaired
    adjustment = result - radius_values
    repaired_count = int(
        sum(bool(entry["triggered"]) for entry in endpoint_diagnostics)
    )
    return result, {
        **metadata,
        "repaired_endpoints": repaired_count,
        "adjusted_sections": int(
            np.count_nonzero(adjustment > 1e-12)
        ),
        "max_radius_adjustment_mm": float(
            np.max(adjustment) * 1000.0
        ),
        "endpoint_diagnostics": endpoint_diagnostics,
    }


def _extend_swept_chain_endpoints_to_visible_support(
    *,
    chain_order: np.ndarray,
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    component_mask: np.ndarray,
    world: np.ndarray,
    enabled: bool,
    blend: float = 1.0,
    maximum_extension_radius_ratio: float = 1.5,
    preserve_original_endpoint_ring: bool = False,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Move flat-cap centers to the RGBD-visible longitudinal boundary."""
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    sy = np.asarray(skeleton_y, dtype=np.int64).reshape(-1)
    sx = np.asarray(skeleton_x, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    mask = np.asarray(component_mask, dtype=bool)
    world_points = np.asarray(world, dtype=np.float64)
    requested_blend = float(blend)
    maximum_ratio = float(maximum_extension_radius_ratio)
    if (
        len(order) != len(center_values)
        or len(sy) != len(center_values)
        or len(sx) != len(center_values)
        or len(radius_values) != len(center_values)
        or len(order) == 0
        or len(np.unique(order)) != len(order)
        or np.min(order) < 0
        or np.max(order) >= len(order)
        or mask.ndim != 2
        or world_points.shape != (*mask.shape, 3)
        or not np.all(np.isfinite(center_values))
        or not np.all(np.isfinite(radius_values))
        or np.any(radius_values <= 0.0)
        or not 0.0 <= requested_blend <= 1.0
        or maximum_ratio <= 0.0
    ):
        raise ValueError("invalid visible endpoint extension inputs")
    metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": bool(enabled and len(order) >= 5),
        "blend": requested_blend,
        "maximum_extension_radius_ratio": maximum_ratio,
        "preserve_original_endpoint_ring": bool(
            preserve_original_endpoint_ring
        ),
        "endpoint_count": 0,
        "extended_endpoints": 0,
        "max_extension_mm": 0.0,
        "endpoint_diagnostics": [],
    }
    if not metadata["enabled"]:
        return center_values.copy(), metadata

    component_y, component_x = np.nonzero(mask)
    component_uv = np.column_stack(
        (component_x, component_y)
    ).astype(np.float64)
    component_world = world_points[component_y, component_x]
    finite_component = np.all(np.isfinite(component_world), axis=1)
    component_uv = component_uv[finite_component]
    component_world = component_world[finite_component]
    if len(component_uv) == 0:
        return center_values.copy(), {
            **metadata,
            "enabled": False,
        }

    distance_px = ndimage.distance_transform_edt(mask)
    ordered_uv = np.column_stack(
        (sx[order], sy[order])
    ).astype(np.float64)
    ordered_centers = center_values[order]
    ordered_radii = radius_values[order]
    result = center_values.copy()
    diagnostics: list[Dict[str, Any]] = []
    extensions: list[float] = []
    for endpoint_name, reverse in (("start", False), ("end", True)):
        local_uv = ordered_uv[::-1] if reverse else ordered_uv
        local_centers = (
            ordered_centers[::-1] if reverse else ordered_centers
        )
        local_radii = (
            ordered_radii[::-1] if reverse else ordered_radii
        )
        local_order = order[::-1] if reverse else order
        endpoint_uv = local_uv[0]
        endpoint_center = local_centers[0]
        endpoint_radius = float(local_radii[0])
        endpoint_skeleton_index = int(local_order[0])
        radius_px = float(
            distance_px[
                int(sy[endpoint_skeleton_index]),
                int(sx[endpoint_skeleton_index]),
            ]
        )
        path_step_px = np.linalg.norm(
            np.diff(local_uv, axis=0),
            axis=1,
        )
        path_distance_px = np.concatenate(
            ([0.0], np.cumsum(path_step_px))
        )
        tangent_span_px = max(4.0, 2.5 * radius_px)
        interior_candidates = np.flatnonzero(
            path_distance_px >= tangent_span_px
        )
        diagnostic: Dict[str, Any] = {
            "endpoint": endpoint_name,
            "endpoint_index": endpoint_skeleton_index,
            "radius_mm": endpoint_radius * 1000.0,
            "radius_px": radius_px,
            "tangent_span_px": tangent_span_px,
            "triggered": False,
            "reason": None,
        }
        if len(interior_candidates) == 0:
            diagnostic["reason"] = "insufficient_tangent_span"
            diagnostics.append(diagnostic)
            continue
        interior = int(interior_candidates[0])
        outward_uv = endpoint_uv - local_uv[interior]
        outward_uv_norm = float(np.linalg.norm(outward_uv))
        outward_world = endpoint_center - local_centers[interior]
        outward_world_norm = float(np.linalg.norm(outward_world))
        if outward_uv_norm < 1e-6 or outward_world_norm < 1e-9:
            diagnostic["reason"] = "degenerate_endpoint_tangent"
            diagnostics.append(diagnostic)
            continue
        outward_uv /= outward_uv_norm
        outward_world /= outward_world_norm

        relative_uv = component_uv - endpoint_uv.reshape(1, 2)
        axial_uv = relative_uv @ outward_uv
        lateral_uv = np.abs(
            relative_uv[:, 0] * outward_uv[1]
            - relative_uv[:, 1] * outward_uv[0]
        )
        support_half_width_px = max(1.5, 1.25 * radius_px)
        support_length_px = max(3.0, 3.0 * radius_px)
        local_support = (
            (axial_uv > 0.0)
            & (axial_uv <= support_length_px)
            & (lateral_uv <= support_half_width_px)
        )
        support_count = int(np.count_nonzero(local_support))
        diagnostic.update(
            {
                "support_half_width_px": support_half_width_px,
                "support_length_px": support_length_px,
                "support_count": support_count,
            }
        )
        if support_count == 0:
            diagnostic["reason"] = "no_outward_visible_support"
            diagnostics.append(diagnostic)
            continue

        center_span_m = float(
            np.dot(
                endpoint_center - local_centers[interior],
                outward_world,
            )
        )
        center_span_px = float(
            np.dot(
                endpoint_uv - local_uv[interior],
                outward_uv,
            )
        )
        meters_per_axial_pixel = (
            center_span_m / max(center_span_px, 1e-9)
        )
        maximum_axial_px = float(np.max(axial_uv[local_support]))
        image_extension_m = (
            maximum_axial_px + 0.5
        ) * meters_per_axial_pixel
        axial_world = (
            component_world[local_support] - endpoint_center.reshape(1, 3)
        ) @ outward_world
        world_extension_m = float(np.max(axial_world)) + (
            0.5 * meters_per_axial_pixel
        )
        positive_estimates = [
            estimate
            for estimate in (image_extension_m, world_extension_m)
            if np.isfinite(estimate) and estimate > 0.0
        ]
        diagnostic.update(
            {
                "meters_per_axial_pixel_mm": (
                    meters_per_axial_pixel * 1000.0
                ),
                "maximum_axial_support_px": maximum_axial_px,
                "image_extension_mm": image_extension_m * 1000.0,
                "world_extension_mm": world_extension_m * 1000.0,
            }
        )
        if not positive_estimates:
            diagnostic["reason"] = "support_not_beyond_cap_center"
            diagnostics.append(diagnostic)
            continue
        raw_extension = float(min(positive_estimates))
        maximum_extension = maximum_ratio * endpoint_radius
        extension = min(raw_extension, maximum_extension)
        extension *= requested_blend
        diagnostic.update(
            {
                "raw_extension_mm": raw_extension * 1000.0,
                "maximum_extension_mm": maximum_extension * 1000.0,
                "extension_mm": extension * 1000.0,
            }
        )
        if extension <= 1e-9:
            diagnostic["reason"] = "zero_blended_extension"
            diagnostics.append(diagnostic)
            continue
        result[endpoint_skeleton_index] = (
            endpoint_center
            if preserve_original_endpoint_ring
            else endpoint_center + extension * outward_world
        )
        extension_vector = extension * outward_world
        diagnostic["triggered"] = True
        diagnostic["reason"] = "visible_support"
        diagnostic["extension_vector_world"] = extension_vector.tolist()
        diagnostic["extended_center_world"] = (
            endpoint_center + extension_vector
        ).tolist()
        diagnostics.append(diagnostic)
        extensions.append(extension)

    return result, {
        **metadata,
        "endpoint_count": 2,
        "extended_endpoints": int(len(extensions)),
        "max_extension_mm": (
            float(max(extensions) * 1000.0) if extensions else 0.0
        ),
        "endpoint_diagnostics": diagnostics,
    }


def _refine_degenerate_endpoint_centers_from_curve_ray(
    *,
    chain_order: np.ndarray,
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    image_width: int,
    image_height: int,
    eligible_endpoint_indices: set[int],
    enabled: bool,
    blend: float = 0.75,
    polynomial_degree: int = 4,
    fit_start_px: float = 3.0,
    fit_stop_px: float = 12.0,
    maximum_curve_ray_miss_radius_ratio: float = 0.6,
    minimum_shift_radius_ratio: float = 2.0,
    maximum_shift_radius_ratio: float = 5.0,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Recover axial endpoint centers from an interior curve and image ray."""
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    sy = np.asarray(skeleton_y, dtype=np.int64).reshape(-1)
    sx = np.asarray(skeleton_x, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radius_values = np.asarray(radii, dtype=np.float64).reshape(-1)
    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    quaternion = np.asarray(
        camera_quat_xyzw,
        dtype=np.float64,
    ).reshape(4)
    selected_blend = float(blend)
    degree = int(polynomial_degree)
    fit_start = float(fit_start_px)
    fit_stop = float(fit_stop_px)
    maximum_miss_ratio = float(
        maximum_curve_ray_miss_radius_ratio
    )
    minimum_shift_ratio = float(minimum_shift_radius_ratio)
    maximum_shift_ratio = float(maximum_shift_radius_ratio)
    width = int(image_width)
    height = int(image_height)
    if (
        not np.all(np.isfinite(camera))
        or not np.all(np.isfinite(quaternion))
        or float(focal_length) <= 0.0
        or float(horizontal_aperture) <= 0.0
        or width <= 0
        or height <= 0
        or not 0.0 <= selected_blend <= 1.0
        or degree < 1
        or fit_start < 0.0
        or fit_stop <= fit_start
        or maximum_miss_ratio <= 0.0
        or minimum_shift_ratio <= 0.0
        or maximum_shift_ratio < minimum_shift_ratio
    ):
        raise ValueError("invalid degenerate endpoint curve inputs")
    metadata: Dict[str, Any] = {
        "requested": bool(enabled),
        "enabled": bool(enabled and eligible_endpoint_indices),
        "blend": selected_blend,
        "polynomial_degree": degree,
        "fit_start_px": fit_start,
        "fit_stop_px": fit_stop,
        "maximum_curve_ray_miss_radius_ratio": maximum_miss_ratio,
        "minimum_shift_radius_ratio": minimum_shift_ratio,
        "maximum_shift_radius_ratio": maximum_shift_ratio,
        "eligible_endpoint_indices": sorted(
            int(index) for index in eligible_endpoint_indices
        ),
        "refined_endpoints": 0,
        "max_shift_mm": 0.0,
        "endpoint_diagnostics": [],
    }
    if not metadata["enabled"]:
        return center_values.copy(), metadata
    if len(order) < degree + 3:
        metadata["enabled"] = False
        metadata["reason"] = "insufficient_curve_sections"
        return center_values.copy(), metadata
    if (
        len(order) != len(center_values)
        or len(order) != len(radius_values)
        or len(sy) != len(center_values)
        or len(sx) != len(center_values)
        or len(np.unique(order)) != len(order)
        or np.min(order) < 0
        or np.max(order) >= len(order)
        or not np.all(np.isfinite(center_values))
        or not np.all(np.isfinite(radius_values))
        or np.any(radius_values <= 0.0)
        or any(
            index < 0 or index >= len(center_values)
            for index in eligible_endpoint_indices
        )
    ):
        raise ValueError("invalid degenerate endpoint curve inputs")

    focal_px = (
        float(focal_length)
        / float(horizontal_aperture)
        * float(width)
    )
    rotation = Rotation.from_quat(quaternion).as_matrix()
    ordered_uv = np.column_stack(
        (sx[order], sy[order])
    ).astype(np.float64)
    ordered_centers = center_values[order]
    ordered_radii = radius_values[order]
    result = center_values.copy()
    diagnostics: list[Dict[str, Any]] = []
    shifts: list[float] = []
    for endpoint_name, reverse in (("start", False), ("end", True)):
        local_uv = ordered_uv[::-1] if reverse else ordered_uv
        local_centers = (
            ordered_centers[::-1] if reverse else ordered_centers
        )
        local_radii = (
            ordered_radii[::-1] if reverse else ordered_radii
        )
        local_order = order[::-1] if reverse else order
        endpoint_index = int(local_order[0])
        diagnostic: Dict[str, Any] = {
            "endpoint": endpoint_name,
            "endpoint_index": endpoint_index,
            "eligible": endpoint_index in eligible_endpoint_indices,
            "triggered": False,
            "reason": None,
        }
        if endpoint_index not in eligible_endpoint_indices:
            diagnostic["reason"] = "endpoint_not_eligible"
            diagnostics.append(diagnostic)
            continue

        path_step = np.linalg.norm(
            np.diff(local_uv, axis=0),
            axis=1,
        )
        path_distance = np.concatenate(
            ([0.0], np.cumsum(path_step))
        )
        fit = (
            (path_distance >= fit_start)
            & (path_distance <= fit_stop)
        )
        diagnostic["fit_sections"] = int(np.count_nonzero(fit))
        if int(np.count_nonzero(fit)) < degree + 2:
            diagnostic["reason"] = "insufficient_curve_sections"
            diagnostics.append(diagnostic)
            continue
        coefficients = [
            np.polyfit(
                path_distance[fit],
                local_centers[fit, axis],
                degree,
            )
            for axis in range(3)
        ]
        endpoint_uv = local_uv[0]
        ray_camera = np.array(
            [
                (endpoint_uv[0] - 0.5 * width) / focal_px,
                -(endpoint_uv[1] - 0.5 * height) / focal_px,
                -1.0,
            ],
            dtype=np.float64,
        )
        ray_world = rotation @ ray_camera
        ray_world /= np.linalg.norm(ray_world) + 1e-12

        def curve(parameter: float) -> np.ndarray:
            return np.asarray(
                [
                    np.polyval(axis_coefficients, parameter)
                    for axis_coefficients in coefficients
                ],
                dtype=np.float64,
            )

        def ray_distance_sq(parameter: float) -> float:
            curve_point = curve(parameter)
            relative = curve_point - camera
            axial = float(relative @ ray_world)
            perpendicular = relative - axial * ray_world
            return float(perpendicular @ perpendicular)

        optimization = minimize_scalar(
            ray_distance_sq,
            bounds=(-12.0, fit_start),
            method="bounded",
        )
        curve_point = curve(float(optimization.x))
        ray_parameter = float((curve_point - camera) @ ray_world)
        candidate = camera + ray_parameter * ray_world
        curve_ray_miss = math.sqrt(max(float(optimization.fun), 0.0))
        shift = float(np.linalg.norm(candidate - local_centers[0]))
        endpoint_radius = float(local_radii[0])
        miss_ratio = curve_ray_miss / endpoint_radius
        shift_ratio = shift / endpoint_radius
        diagnostic.update(
            {
                "curve_parameter_px": float(optimization.x),
                "curve_ray_miss_mm": curve_ray_miss * 1000.0,
                "curve_ray_miss_radius_ratio": miss_ratio,
                "candidate_shift_mm": shift * 1000.0,
                "candidate_shift_radius_ratio": shift_ratio,
            }
        )
        if float(optimization.x) > 1e-3:
            diagnostic["reason"] = "curve_does_not_extrapolate"
        elif miss_ratio > maximum_miss_ratio:
            diagnostic["reason"] = "curve_ray_miss_too_large"
        elif shift_ratio < minimum_shift_ratio:
            diagnostic["reason"] = "candidate_shift_too_small"
        elif shift_ratio > maximum_shift_ratio:
            diagnostic["reason"] = "candidate_shift_too_large"
        else:
            result[endpoint_index] = (
                local_centers[0]
                + selected_blend * (candidate - local_centers[0])
            )
            diagnostic["triggered"] = True
            diagnostic["reason"] = "curve_ray_supported"
            diagnostic["candidate_center_world"] = candidate.tolist()
            diagnostic["refined_center_world"] = result[
                endpoint_index
            ].tolist()
            shifts.append(selected_blend * shift)
        diagnostics.append(diagnostic)

    return result, {
        **metadata,
        "refined_endpoints": int(len(shifts)),
        "max_shift_mm": (
            float(max(shifts) * 1000.0) if shifts else 0.0
        ),
        "endpoint_diagnostics": diagnostics,
    }


def _swept_refinement_continuity_gate(
    *,
    chain_order: np.ndarray,
    initial_radii: np.ndarray,
    refined_radii: np.ndarray,
    base_residual_m: float,
    refined_residual_m: float,
) -> tuple[bool, Dict[str, Any]]:
    """Balance local radial fit gain against global radius roughness."""
    order = np.asarray(chain_order, dtype=np.int64).reshape(-1)
    initial = np.asarray(initial_radii, dtype=np.float64).reshape(-1)
    refined = np.asarray(refined_radii, dtype=np.float64).reshape(-1)
    base_residual = float(base_residual_m)
    refined_residual = float(refined_residual_m)
    if (
        len(order) < 3
        or len(initial) != len(refined)
        or np.any(order < 0)
        or np.any(order >= len(initial))
        or not np.all(np.isfinite(initial))
        or not np.all(np.isfinite(refined))
        or np.any(initial <= 0.0)
        or np.any(refined <= 0.0)
        or not math.isfinite(base_residual)
        or not math.isfinite(refined_residual)
        or base_residual <= 0.0
        or refined_residual <= 0.0
    ):
        raise ValueError("invalid swept refinement continuity inputs")

    initial_log = np.log(initial[order])
    refined_log = np.log(refined[order])
    initial_curvature = np.abs(
        initial_log[1:-1]
        - 0.5 * (initial_log[:-2] + initial_log[2:])
    )
    refined_curvature = np.abs(
        refined_log[1:-1]
        - 0.5 * (refined_log[:-2] + refined_log[2:])
    )
    initial_p90 = float(np.percentile(initial_curvature, 90.0))
    refined_p90 = float(np.percentile(refined_curvature, 90.0))
    roughness_ratio = refined_p90 / max(initial_p90, 1e-9)
    residual_gain = base_residual / refined_residual
    allowed_ratio = math.sqrt(max(residual_gain, 1.0))
    accepted = bool(roughness_ratio <= allowed_ratio)
    return accepted, {
        "accepted": accepted,
        "initial_log_radius_curvature_p90": initial_p90,
        "refined_log_radius_curvature_p90": refined_p90,
        "roughness_ratio": roughness_ratio,
        "base_residual_mm": base_residual * 1000.0,
        "refined_residual_mm": refined_residual * 1000.0,
        "residual_gain": residual_gain,
        "allowed_roughness_ratio": allowed_ratio,
    }


def _disk_structure(radius_cells: int) -> np.ndarray:
    radius = max(0, int(radius_cells))
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _fill_new_footprint_cells(
    top_height: np.ndarray,
    old_footprint: np.ndarray,
    new_footprint: np.ndarray,
) -> np.ndarray:
    fill = np.asarray(new_footprint, dtype=bool) & ~np.asarray(
        old_footprint,
        dtype=bool,
    )
    if not np.any(fill):
        return top_height
    nearest = ndimage.distance_transform_edt(
        ~old_footprint,
        return_distances=False,
        return_indices=True,
    )
    result = np.asarray(top_height, dtype=np.float32).copy()
    result[fill] = result[
        nearest[0][fill],
        nearest[1][fill],
    ]
    return result


def _local_skeleton_tangent_uv(
    skeleton_uv: np.ndarray,
    endpoint_uv: np.ndarray,
    *,
    neighborhood_px: float,
) -> np.ndarray | None:
    points = np.asarray(skeleton_uv, dtype=np.float64).reshape(-1, 2)
    endpoint = np.asarray(endpoint_uv, dtype=np.float64).reshape(2)
    if len(points) < 2:
        return None
    distance = np.linalg.norm(points - endpoint.reshape(1, 2), axis=1)
    local = points[distance <= float(neighborhood_px)]
    if len(local) < 2:
        local = points[np.argsort(distance)[: min(6, len(points))]]
    centered = local - local.mean(axis=0, keepdims=True)
    if float(np.linalg.norm(centered)) < 1e-9:
        return None
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = np.asarray(vh[0], dtype=np.float64)
    tangent /= np.linalg.norm(tangent) + 1e-12
    return tangent


def _select_component_bridges(
    *,
    labels: np.ndarray,
    selected_components: set[int],
    world: np.ndarray,
    color: np.ndarray,
    skeleton_uv_by_component: Dict[int, np.ndarray],
    chord_map: np.ndarray,
    footprint_radius_map: np.ndarray,
    tube_radius_map: np.ndarray,
    tube_axis_map: np.ndarray,
    max_gap_3d_m: float,
    max_gap_px: float,
    max_color_delta: float,
    min_tangent_alignment: float,
    tangent_neighborhood_px: float,
    selection_mode: str,
    contact_margin_m: float,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Select conservative RGBD-only continuation edges between fragments."""
    mode = str(selection_mode).strip().lower()
    if mode not in {"tangent", "contact", "both"}:
        raise ValueError(
            f"unsupported bridge selection mode={selection_mode!r}"
        )
    components = sorted(int(value) for value in selected_components)
    component_samples: Dict[int, Dict[str, np.ndarray]] = {}
    for component in components:
        cy, cx = np.nonzero(labels == component)
        if len(cx) == 0:
            continue
        component_samples[component] = {
            "y": cy,
            "x": cx,
            "world": np.asarray(world[cy, cx], dtype=np.float64),
        }

    candidates: list[Dict[str, Any]] = []
    candidate_diagnostics: list[Dict[str, Any]] = []
    for index, first in enumerate(components):
        first_samples = component_samples.get(first)
        if first_samples is None:
            continue
        first_tree = cKDTree(first_samples["world"])
        for second in components[index + 1 :]:
            second_samples = component_samples.get(second)
            if second_samples is None:
                continue
            distance, first_index = first_tree.query(
                second_samples["world"],
                k=1,
            )
            second_index = int(np.argmin(distance))
            first_index_value = int(first_index[second_index])
            gap_3d = float(distance[second_index])
            if gap_3d > float(max_gap_3d_m):
                continue

            first_y = int(first_samples["y"][first_index_value])
            first_x = int(first_samples["x"][first_index_value])
            second_y = int(second_samples["y"][second_index])
            second_x = int(second_samples["x"][second_index])
            first_uv = np.array([first_x, first_y], dtype=np.float64)
            second_uv = np.array([second_x, second_y], dtype=np.float64)
            gap_uv = second_uv - first_uv
            gap_px = float(np.linalg.norm(gap_uv))
            if gap_px > float(max_gap_px) or gap_px < 1e-9:
                continue
            color_delta = float(
                np.linalg.norm(
                    color[first_y, first_x] - color[second_y, second_x]
                )
            )
            if color_delta > float(max_color_delta):
                continue

            gap_direction = gap_uv / gap_px
            first_tangent = _local_skeleton_tangent_uv(
                skeleton_uv_by_component[first],
                first_uv,
                neighborhood_px=float(tangent_neighborhood_px),
            )
            second_tangent = _local_skeleton_tangent_uv(
                skeleton_uv_by_component[second],
                second_uv,
                neighborhood_px=float(tangent_neighborhood_px),
            )
            first_alignment = (
                abs(float(first_tangent @ gap_direction))
                if first_tangent is not None
                else 1.0
            )
            second_alignment = (
                abs(float(second_tangent @ gap_direction))
                if second_tangent is not None
                else 1.0
            )
            tangent_alignment = (
                abs(float(first_tangent @ second_tangent))
                if first_tangent is not None and second_tangent is not None
                else 1.0
            )
            minimum_alignment = min(
                first_alignment,
                second_alignment,
                tangent_alignment,
            )
            first_tube_radius = float(
                tube_radius_map[first_y, first_x]
            )
            second_tube_radius = float(
                tube_radius_map[second_y, second_x]
            )
            contact_limit = (
                first_tube_radius
                + second_tube_radius
                + float(contact_margin_m)
            )
            first_axis_world = np.asarray(
                tube_axis_map[first_y, first_x],
                dtype=np.float64,
            )
            second_axis_world = np.asarray(
                tube_axis_map[second_y, second_x],
                dtype=np.float64,
            )
            axis_gap_3d = float(
                np.linalg.norm(second_axis_world - first_axis_world)
            )
            radius_contact = bool(
                np.isfinite(first_tube_radius)
                and np.isfinite(second_tube_radius)
                and np.all(np.isfinite(first_axis_world))
                and np.all(np.isfinite(second_axis_world))
                and gap_3d <= contact_limit
            )
            tangent_continuation = bool(
                minimum_alignment >= float(min_tangent_alignment)
            )
            selected_by_mode = bool(
                (mode == "tangent" and tangent_continuation)
                or (mode == "contact" and radius_contact)
                or (
                    mode == "both"
                    and (tangent_continuation or radius_contact)
                )
            )
            candidate_diagnostics.append(
                {
                    "components": [int(first), int(second)],
                    "first_yx": [first_y, first_x],
                    "second_yx": [second_y, second_x],
                    "gap_3d_mm": gap_3d * 1000.0,
                    "gap_px": gap_px,
                    "color_delta": color_delta,
                    "first_tube_radius_mm": (
                        first_tube_radius * 1000.0
                    ),
                    "second_tube_radius_mm": (
                        second_tube_radius * 1000.0
                    ),
                    "contact_limit_mm": contact_limit * 1000.0,
                    "axis_gap_3d_mm": axis_gap_3d * 1000.0,
                    "radius_contact": radius_contact,
                    "tangent_continuation": tangent_continuation,
                    "selected_by_mode": selected_by_mode,
                    "first_alignment": first_alignment,
                    "second_alignment": second_alignment,
                    "tangent_alignment": tangent_alignment,
                }
            )
            if not selected_by_mode:
                continue
            if radius_contact and not tangent_continuation:
                bridge_kind = "radius_contact"
            elif tangent_continuation:
                bridge_kind = "tangent_continuation"
            else:
                bridge_kind = "radius_contact"

            candidates.append(
                {
                    "first_component": int(first),
                    "second_component": int(second),
                    "first_yx": [first_y, first_x],
                    "second_yx": [second_y, second_x],
                    "first_world": np.asarray(
                        world[first_y, first_x],
                        dtype=np.float64,
                    ),
                    "second_world": np.asarray(
                        world[second_y, second_x],
                        dtype=np.float64,
                    ),
                    "first_axis_world": first_axis_world,
                    "second_axis_world": second_axis_world,
                    "first_chord_m": float(chord_map[first_y, first_x]),
                    "second_chord_m": float(chord_map[second_y, second_x]),
                    "first_footprint_radius_m": float(
                        footprint_radius_map[first_y, first_x]
                    ),
                    "second_footprint_radius_m": float(
                        footprint_radius_map[second_y, second_x]
                    ),
                    "first_tube_radius_m": first_tube_radius,
                    "second_tube_radius_m": second_tube_radius,
                    "contact_limit_m": contact_limit,
                    "radius_contact": radius_contact,
                    "tangent_continuation": tangent_continuation,
                    "bridge_kind": bridge_kind,
                    "gap_3d_m": gap_3d,
                    "gap_px": gap_px,
                    "color_delta": color_delta,
                    "first_alignment": first_alignment,
                    "second_alignment": second_alignment,
                    "tangent_alignment": tangent_alignment,
                    "cost": (
                        gap_3d
                        / max(
                            contact_limit
                            if bridge_kind == "radius_contact"
                            else float(max_gap_3d_m),
                            1e-12,
                        )
                        + gap_px / max(float(max_gap_px), 1e-12)
                        + color_delta / max(float(max_color_delta), 1e-12)
                        + (
                            3.0 * (1.0 - minimum_alignment)
                            if bridge_kind == "tangent_continuation"
                            else 0.0
                        )
                    ),
                }
            )

    parent = {component: component for component in components}

    def find(component: int) -> int:
        while parent[component] != component:
            parent[component] = parent[parent[component]]
            component = parent[component]
        return component

    selected_edges: list[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda value: value["cost"]):
        first = int(candidate["first_component"])
        second = int(candidate["second_component"])
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            continue
        parent[second_root] = first_root
        selected_edges.append(candidate)
    candidate_diagnostics.sort(
        key=lambda value: (
            float(value["gap_3d_mm"]),
            float(value["gap_px"]),
            value["components"],
        )
    )
    return selected_edges, candidate_diagnostics


def _resample_component_spanning_edges(
    *,
    labels: np.ndarray,
    selected_components: set[int],
    world: np.ndarray,
    color: np.ndarray,
    depth: np.ndarray,
    focal_px: float,
    chord_map: np.ndarray,
    footprint_radius_map: np.ndarray,
    voxel_m: float,
    spacing_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Densify only RGBD graph edges needed to connect each component."""
    active = np.isin(
        labels,
        np.asarray(sorted(selected_components), dtype=np.int32),
    )
    active &= np.isfinite(chord_map) & np.isfinite(footprint_radius_map)
    yy, xx = np.nonzero(active)
    if len(xx) == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, np.zeros(0), np.zeros(0), {
            "tree_edges": 0,
            "resampled_edges": 0,
            "resampled_samples": 0,
        }

    node_at = np.full(active.shape, -1, dtype=np.int32)
    node_at[yy, xx] = np.arange(len(xx), dtype=np.int32)
    points = np.asarray(world, dtype=np.float64)
    colors = np.asarray(color, dtype=np.float64)
    metric_depth = np.asarray(depth, dtype=np.float64)
    candidates: list[tuple[float, int, int]] = []
    h, w = active.shape
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy < 0 or (dy == 0 and dx <= 0):
                continue
            y2 = yy + dy
            x2 = xx + dx
            in_image = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
            first = np.flatnonzero(in_image)
            second = node_at[y2[in_image], x2[in_image]]
            has_neighbor = second >= 0
            first = first[has_neighbor]
            second = second[has_neighbor]
            if len(first) == 0:
                continue
            same_component = (
                labels[yy[first], xx[first]]
                == labels[yy[second], xx[second]]
            )
            first = first[same_component]
            second = second[same_component]
            if len(first) == 0:
                continue

            first_points = points[yy[first], xx[first]]
            second_points = points[yy[second], xx[second]]
            distance_3d = np.linalg.norm(
                first_points - second_points,
                axis=1,
            )
            color_distance = np.linalg.norm(
                colors[yy[first], xx[first]]
                - colors[yy[second], xx[second]],
                axis=1,
            )
            pixel_span = math.sqrt(float(dx * dx + dy * dy))
            meters_per_pixel = np.maximum(
                metric_depth[yy[first], xx[first]],
                metric_depth[yy[second], xx[second]],
            ) / max(float(focal_px), 1e-9)
            edge_limit = (
                1.4 * pixel_span * meters_per_pixel + 0.0015
            )
            keep = (
                np.isfinite(distance_3d)
                & (distance_3d <= edge_limit)
                & (color_distance <= 120.0)
            )
            candidates.extend(
                (
                    float(gap),
                    int(first_node),
                    int(second_node),
                )
                for gap, first_node, second_node in zip(
                    distance_3d[keep],
                    first[keep],
                    second[keep],
                )
            )

    parent = np.arange(len(xx), dtype=np.int32)
    size = np.ones(len(xx), dtype=np.int32)

    def find(node: int) -> int:
        while int(parent[node]) != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(first: int, second: int) -> bool:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return False
        if int(size[first_root]) < int(size[second_root]):
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size[second_root]
        return True

    surface_parts: list[np.ndarray] = []
    chord_parts: list[np.ndarray] = []
    radius_parts: list[np.ndarray] = []
    tree_edges = 0
    resampled_edges = 0
    resampled_gaps: list[float] = []
    for gap, first, second in sorted(candidates):
        if not union(first, second):
            continue
        tree_edges += 1
        first_radius = float(
            footprint_radius_map[yy[first], xx[first]]
        )
        second_radius = float(
            footprint_radius_map[yy[second], xx[second]]
        )
        if gap <= first_radius + second_radius:
            continue
        maximum_spacing = max(
            0.75 * float(voxel_m),
            float(spacing_scale) * min(first_radius, second_radius),
        )
        segment_count = max(
            2,
            int(math.ceil(gap / maximum_spacing)),
        )
        alpha = np.linspace(
            0.0,
            1.0,
            segment_count + 1,
        )[1:-1]
        if len(alpha) == 0:
            continue
        first_surface = points[yy[first], xx[first]]
        second_surface = points[yy[second], xx[second]]
        first_chord = float(chord_map[yy[first], xx[first]])
        second_chord = float(chord_map[yy[second], xx[second]])
        surface_parts.append(
            first_surface.reshape(1, 3)
            + alpha[:, None]
            * (second_surface - first_surface).reshape(1, 3)
        )
        interpolated_radius = (
            first_radius + alpha * (second_radius - first_radius)
        )
        interpolated_chord = (
            first_chord + alpha * (second_chord - first_chord)
        )
        chord_parts.append(
            np.minimum(
                interpolated_chord,
                2.0 * interpolated_radius,
            )
        )
        radius_parts.append(interpolated_radius)
        resampled_edges += 1
        resampled_gaps.append(float(gap))

    if not surface_parts:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, np.zeros(0), np.zeros(0), {
            "tree_edges": int(tree_edges),
            "resampled_edges": 0,
            "resampled_samples": 0,
            "spacing_scale": float(spacing_scale),
        }
    surfaces = np.concatenate(surface_parts, axis=0)
    chords = np.concatenate(chord_parts, axis=0)
    radii = np.concatenate(radius_parts, axis=0)
    return surfaces, chords, radii, {
        "tree_edges": int(tree_edges),
        "resampled_edges": int(resampled_edges),
        "resampled_samples": int(len(surfaces)),
        "spacing_scale": float(spacing_scale),
        "chord_cap_footprint_diameters": 2.0,
        "resampled_gap_median_mm": float(
            np.median(resampled_gaps) * 1000.0
        ),
        "resampled_gap_max_mm": float(
            np.max(resampled_gaps) * 1000.0
        ),
    }


def _component_curvature_radius_scale(
    *,
    component_mask: np.ndarray,
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    world: np.ndarray,
    depth: np.ndarray,
    camera_pos: Sequence[float],
    focal_px: float,
    fit_max_residual_m: float,
    scale_min: float,
    scale_max: float,
    min_valid_fits: int,
    smoothing_px: float,
    local_shrinkage: float,
    max_center_toward_camera_ratio: float,
) -> tuple[float, np.ndarray, Dict[str, Any]]:
    """Estimate tube-radius correction from the visible RGBD surface arc."""
    cy, cx = np.nonzero(component_mask)
    skeleton_uv = np.column_stack((skeleton_x, skeleton_y)).astype(
        np.float64
    )
    skeleton_world = np.asarray(
        world[skeleton_y, skeleton_x],
        dtype=np.float64,
    )
    component_uv = np.column_stack((cx, cy)).astype(np.float64)
    component_world = np.asarray(world[cy, cx], dtype=np.float64)
    distance_to_edge = ndimage.distance_transform_edt(component_mask)
    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)

    ratios: list[float] = []
    fitted_radii: list[float] = []
    image_radii: list[float] = []
    residuals: list[float] = []
    ratio_by_skeleton = np.full(len(skeleton_uv), np.nan, dtype=np.float64)
    for index, (uv, surface_point) in enumerate(
        zip(skeleton_uv, skeleton_world)
    ):
        skeleton_distance = np.linalg.norm(
            skeleton_uv - uv.reshape(1, 2),
            axis=1,
        )
        local_skeleton = np.flatnonzero(skeleton_distance <= 10.0)
        if len(local_skeleton) < 3:
            continue

        local_world = skeleton_world[local_skeleton]
        centered_world = local_world - local_world.mean(
            axis=0,
            keepdims=True,
        )
        if float(np.linalg.norm(centered_world)) < 1e-9:
            continue
        _u, _s, world_vh = np.linalg.svd(
            centered_world,
            full_matrices=False,
        )
        tangent_world = np.asarray(world_vh[0], dtype=np.float64)
        tangent_world /= np.linalg.norm(tangent_world) + 1e-12

        local_uv = skeleton_uv[local_skeleton]
        centered_uv = local_uv - local_uv.mean(axis=0, keepdims=True)
        if float(np.linalg.norm(centered_uv)) < 1e-9:
            continue
        _u, _s, uv_vh = np.linalg.svd(centered_uv, full_matrices=False)
        tangent_uv = np.asarray(uv_vh[0], dtype=np.float64)
        tangent_uv /= np.linalg.norm(tangent_uv) + 1e-12
        normal_uv = np.array(
            [-tangent_uv[1], tangent_uv[0]],
            dtype=np.float64,
        )

        delta_uv = component_uv - uv.reshape(1, 2)
        along = np.abs(delta_uv @ tangent_uv)
        across = np.abs(delta_uv @ normal_uv)
        local_image_radius_px = float(
            distance_to_edge[int(skeleton_y[index]), int(skeleton_x[index])]
        )
        cross_section = (
            (along <= 2.25)
            & (across <= max(5.0, local_image_radius_px + 2.0))
        )
        section_points = component_world[cross_section]
        if len(section_points) < 4:
            continue

        view_normal = camera - surface_point
        view_normal -= (
            tangent_world * float(view_normal @ tangent_world)
        )
        view_norm = float(np.linalg.norm(view_normal))
        if view_norm < 1e-9:
            continue
        basis_x = view_normal / view_norm
        basis_y = np.cross(tangent_world, basis_x)
        basis_y /= np.linalg.norm(basis_y) + 1e-12
        relative = section_points - surface_point.reshape(1, 3)
        section_xy = np.column_stack(
            (relative @ basis_x, relative @ basis_y)
        )
        design = np.column_stack(
            (
                2.0 * section_xy[:, 0],
                2.0 * section_xy[:, 1],
                np.ones(len(section_xy)),
            )
        )
        target = np.einsum("ij,ij->i", section_xy, section_xy)
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        center_xy = np.asarray(solution[:2], dtype=np.float64)
        radius_sq = float(solution[2] + center_xy @ center_xy)
        if radius_sq <= 0.0:
            continue
        fitted_radius = math.sqrt(radius_sq)
        radial_residual = np.abs(
            np.linalg.norm(
                section_xy - center_xy.reshape(1, 2),
                axis=1,
            )
            - fitted_radius
        )
        residual = float(np.median(radial_residual))
        image_radius = (
            local_image_radius_px
            * float(depth[int(skeleton_y[index]), int(skeleton_x[index])])
            / max(float(focal_px), 1e-12)
        )
        if (
            fitted_radius < 0.0004
            or fitted_radius > 0.008
            or image_radius < 0.0002
            or residual > float(fit_max_residual_m)
            or float(np.linalg.norm(center_xy))
            > 2.5 * fitted_radius + 0.001
            or float(center_xy[0])
            > float(max_center_toward_camera_ratio) * fitted_radius
        ):
            continue
        ratios.append(fitted_radius / image_radius)
        ratio_by_skeleton[index] = fitted_radius / image_radius
        fitted_radii.append(fitted_radius)
        image_radii.append(image_radius)
        residuals.append(residual)

    if len(ratios) < int(min_valid_fits):
        return 1.0, np.ones(len(skeleton_uv), dtype=np.float64), {
            "valid_fits": int(len(ratios)),
            "required_valid_fits": int(min_valid_fits),
            "max_center_toward_camera_ratio": float(
                max_center_toward_camera_ratio
            ),
            "raw_scale": 1.0,
            "clipped_scale": 1.0,
        }
    ratio_values = np.asarray(ratios, dtype=np.float64)
    lower, upper = np.percentile(ratio_values, [15.0, 85.0])
    robust = ratio_values[
        (ratio_values >= float(lower))
        & (ratio_values <= float(upper))
    ]
    if len(robust) == 0:
        robust = ratio_values
    raw_scale = float(np.median(robust))
    clipped_scale = float(
        np.clip(raw_scale, float(scale_min), float(scale_max))
    )
    valid_indices = np.flatnonzero(np.isfinite(ratio_by_skeleton))
    valid_tree = cKDTree(skeleton_uv[valid_indices])
    local_scale = np.empty(len(skeleton_uv), dtype=np.float64)
    for index, uv in enumerate(skeleton_uv):
        neighbors = valid_tree.query_ball_point(
            uv,
            r=float(smoothing_px),
        )
        if not neighbors:
            _distance, nearest = valid_tree.query(uv, k=1)
            neighbors = [int(nearest)]
        values = ratio_by_skeleton[
            valid_indices[np.asarray(neighbors, dtype=np.int64)]
        ]
        local_scale[index] = float(
            np.clip(
                np.median(values),
                float(scale_min),
                float(scale_max),
            )
        )
    local_scale = (
        clipped_scale
        + float(local_shrinkage) * (local_scale - clipped_scale)
    )
    return clipped_scale, local_scale, {
        "valid_fits": int(len(ratios)),
        "required_valid_fits": int(min_valid_fits),
        "max_center_toward_camera_ratio": float(
            max_center_toward_camera_ratio
        ),
        "raw_scale": raw_scale,
        "clipped_scale": clipped_scale,
        "fitted_radius_median_mm": float(
            np.median(fitted_radii) * 1000.0
        ),
        "image_radius_median_mm": float(
            np.median(image_radii) * 1000.0
        ),
        "fit_residual_median_mm": float(
            np.median(residuals) * 1000.0
        ),
        "local_scale_min": float(local_scale.min()),
        "local_scale_median": float(np.median(local_scale)),
        "local_scale_max": float(local_scale.max()),
        "local_shrinkage": float(local_shrinkage),
    }


def _component_circle_tube_sections(
    *,
    component_mask: np.ndarray,
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    world: np.ndarray,
    depth: np.ndarray,
    camera_pos: Sequence[float],
    focal_px: float,
    radius_scale: float,
    radius_bias_px: float,
    radius_min_m: float,
    radius_max_m: float,
    fit_max_residual_m: float,
    scale_min: float,
    scale_max: float,
    min_valid_fits: int,
    smoothing_px: float,
    local_shrinkage: float,
    radius_blend: float,
    direct_radius_blend: float,
    soft_direct_radius_blend: bool,
    soft_direct_radius_blend_max: float,
    direct_center_blend: float,
    front_anchored_radius_blend: float,
    endpoint_radius_regularization_blend: float,
    graph_radius_smoothing_gcv: bool,
    center_fit_blend: float,
    center_offset_scale: float,
    tangent_neighborhood_px: float,
    max_center_toward_camera_ratio: float,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Infer local circular tube sections from a visible RGBD surface arc."""
    cy, cx = np.nonzero(component_mask)
    skeleton_uv = np.column_stack((skeleton_x, skeleton_y)).astype(
        np.float64
    )
    skeleton_world = np.asarray(
        world[skeleton_y, skeleton_x],
        dtype=np.float64,
    )
    component_uv = np.column_stack((cx, cy)).astype(np.float64)
    component_world = np.asarray(world[cy, cx], dtype=np.float64)
    distance_to_edge = ndimage.distance_transform_edt(component_mask)
    image_radius_px = np.maximum(
        distance_to_edge[skeleton_y, skeleton_x]
        - float(radius_bias_px),
        0.25,
    )
    image_radius = (
        image_radius_px
        * np.asarray(depth[skeleton_y, skeleton_x], dtype=np.float64)
        / max(float(focal_px), 1e-12)
        * float(radius_scale)
    )
    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    fallback_direction = skeleton_world - camera.reshape(1, 3)
    fallback_direction /= (
        np.linalg.norm(fallback_direction, axis=1, keepdims=True) + 1e-12
    )

    ratio_by_skeleton = np.full(
        len(skeleton_uv),
        np.nan,
        dtype=np.float64,
    )
    fitted_radius_by_skeleton = np.full(
        len(skeleton_uv),
        np.nan,
        dtype=np.float64,
    )
    direction_by_skeleton = np.full(
        (len(skeleton_uv), 3),
        np.nan,
        dtype=np.float64,
    )
    center_offset_by_skeleton = np.full(
        (len(skeleton_uv), 3),
        np.nan,
        dtype=np.float64,
    )
    front_radius_by_skeleton = np.full(
        len(skeleton_uv),
        np.nan,
        dtype=np.float64,
    )
    front_direction_by_skeleton = np.full(
        (len(skeleton_uv), 3),
        np.nan,
        dtype=np.float64,
    )
    fitted_radii: list[float] = []
    image_radii: list[float] = []
    residuals: list[float] = []
    center_offset_norms: list[float] = []
    arc_spans_deg: list[float] = []
    front_radii: list[float] = []
    front_residuals: list[float] = []
    for index, (uv, surface_point) in enumerate(
        zip(skeleton_uv, skeleton_world)
    ):
        skeleton_distance = np.linalg.norm(
            skeleton_uv - uv.reshape(1, 2),
            axis=1,
        )
        local_skeleton = np.flatnonzero(
            skeleton_distance <= float(tangent_neighborhood_px)
        )
        if len(local_skeleton) < 3:
            continue

        local_world = skeleton_world[local_skeleton]
        centered_world = local_world - local_world.mean(
            axis=0,
            keepdims=True,
        )
        if float(np.linalg.norm(centered_world)) < 1e-9:
            continue
        _u, _s, world_vh = np.linalg.svd(
            centered_world,
            full_matrices=False,
        )
        tangent_world = np.asarray(world_vh[0], dtype=np.float64)
        tangent_world /= np.linalg.norm(tangent_world) + 1e-12

        local_uv = skeleton_uv[local_skeleton]
        centered_uv = local_uv - local_uv.mean(axis=0, keepdims=True)
        if float(np.linalg.norm(centered_uv)) < 1e-9:
            continue
        _u, _s, uv_vh = np.linalg.svd(centered_uv, full_matrices=False)
        tangent_uv = np.asarray(uv_vh[0], dtype=np.float64)
        tangent_uv /= np.linalg.norm(tangent_uv) + 1e-12
        normal_uv = np.array(
            [-tangent_uv[1], tangent_uv[0]],
            dtype=np.float64,
        )

        delta_uv = component_uv - uv.reshape(1, 2)
        along = np.abs(delta_uv @ tangent_uv)
        across = np.abs(delta_uv @ normal_uv)
        local_image_radius_px = float(
            distance_to_edge[
                int(skeleton_y[index]),
                int(skeleton_x[index]),
            ]
        )
        cross_section = (
            (along <= 2.25)
            & (across <= max(5.0, local_image_radius_px + 2.0))
        )
        section_points = component_world[cross_section]
        if len(section_points) < 4:
            continue

        view_normal = camera - surface_point
        view_normal -= tangent_world * float(view_normal @ tangent_world)
        view_norm = float(np.linalg.norm(view_normal))
        if view_norm < 1e-9:
            continue
        basis_x = view_normal / view_norm
        basis_y = np.cross(tangent_world, basis_x)
        basis_y /= np.linalg.norm(basis_y) + 1e-12
        relative = section_points - surface_point.reshape(1, 3)
        section_xy = np.column_stack(
            (relative @ basis_x, relative @ basis_y)
        )
        meters_per_pixel = float(
            depth[int(skeleton_y[index]), int(skeleton_x[index])]
        ) / max(float(focal_px), 1e-12)
        try:
            front_radius, front_residual = (
                _fit_front_anchored_circle_radius(
                    section_xy,
                    minimum_depth_offset_m=max(
                        0.15 * meters_per_pixel,
                        0.00015,
                    ),
                    radius_min_m=float(radius_min_m),
                    radius_max_m=float(radius_max_m),
                )
            )
        except ValueError:
            pass
        else:
            if front_residual <= float(fit_max_residual_m):
                front_radius_by_skeleton[index] = front_radius
                front_direction_by_skeleton[index] = -basis_x
                front_radii.append(front_radius)
                front_residuals.append(front_residual)

        design = np.column_stack(
            (
                2.0 * section_xy[:, 0],
                2.0 * section_xy[:, 1],
                np.ones(len(section_xy)),
            )
        )
        target = np.einsum("ij,ij->i", section_xy, section_xy)
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        center_xy = np.asarray(solution[:2], dtype=np.float64)
        radius_sq = float(solution[2] + center_xy @ center_xy)
        if radius_sq <= 0.0:
            continue
        fitted_radius = math.sqrt(radius_sq)
        radial_residual = np.abs(
            np.linalg.norm(
                section_xy - center_xy.reshape(1, 2),
                axis=1,
            )
            - fitted_radius
        )
        residual = float(np.median(radial_residual))
        center_offset = (
            center_xy[0] * basis_x + center_xy[1] * basis_y
        )
        center_offset_norm = float(np.linalg.norm(center_offset))
        local_image_radius = float(image_radius[index])
        if (
            fitted_radius < 0.0004
            or fitted_radius > 0.008
            or local_image_radius < 0.0002
            or residual > float(fit_max_residual_m)
            or center_offset_norm > 2.5 * fitted_radius + 0.001
            or float(center_xy[0])
            > float(max_center_toward_camera_ratio) * fitted_radius
            or center_offset_norm < 1e-9
        ):
            continue
        fitted_direction = center_offset / center_offset_norm
        if float(fitted_direction @ fallback_direction[index]) < 0.0:
            fitted_direction = -fitted_direction
            center_offset = -center_offset
        ratio_by_skeleton[index] = fitted_radius / local_image_radius
        fitted_radius_by_skeleton[index] = fitted_radius
        direction_by_skeleton[index] = fitted_direction
        center_offset_by_skeleton[index] = center_offset
        fitted_radii.append(fitted_radius)
        image_radii.append(local_image_radius)
        residuals.append(residual)
        center_offset_norms.append(center_offset_norm)
        arc_spans_deg.append(
            math.degrees(
                _minimum_covering_arc_span_rad(
                    section_xy - center_xy.reshape(1, 2)
                )
            )
        )

    valid_indices = np.flatnonzero(np.isfinite(ratio_by_skeleton))
    if len(valid_indices) < int(min_valid_fits):
        radii = np.clip(
            image_radius,
            float(radius_min_m),
            float(radius_max_m),
        )
        centers = (
            skeleton_world
            + fallback_direction
            * (
                radii * float(center_offset_scale)
            )[:, None]
        )
        metadata: Dict[str, Any] = {
            "valid_fits": int(len(valid_indices)),
            "required_valid_fits": int(min_valid_fits),
            "raw_scale": 1.0,
            "clipped_scale": 1.0,
            "fallback_only": True,
            "radius_min_mm": float(radii.min() * 1000.0),
            "radius_median_mm": float(np.median(radii) * 1000.0),
            "radius_max_mm": float(radii.max() * 1000.0),
        }
        if arc_spans_deg:
            metadata.update(
                {
                    "fit_arc_span_p10_deg": float(
                        np.percentile(arc_spans_deg, 10.0)
                    ),
                    "fit_arc_span_median_deg": float(
                        np.median(arc_spans_deg)
                    ),
                    "fit_arc_span_p90_deg": float(
                        np.percentile(arc_spans_deg, 90.0)
                    ),
                }
            )
        return centers, radii, metadata

    ratio_values = ratio_by_skeleton[valid_indices]
    lower, upper = np.percentile(ratio_values, [15.0, 85.0])
    robust = ratio_values[
        (ratio_values >= float(lower))
        & (ratio_values <= float(upper))
    ]
    if len(robust) == 0:
        robust = ratio_values
    raw_scale = float(np.median(robust))
    clipped_scale = float(
        np.clip(raw_scale, float(scale_min), float(scale_max))
    )

    valid_tree = cKDTree(skeleton_uv[valid_indices])
    local_scale = np.empty(len(skeleton_uv), dtype=np.float64)
    local_fitted_radius = np.empty(len(skeleton_uv), dtype=np.float64)
    local_direction = np.empty_like(fallback_direction)
    local_center_offset = np.empty_like(fallback_direction)
    for index, uv in enumerate(skeleton_uv):
        neighbors = valid_tree.query_ball_point(
            uv,
            r=float(smoothing_px),
        )
        if not neighbors:
            _distance, nearest = valid_tree.query(uv, k=1)
            neighbors = [int(nearest)]
        source_indices = valid_indices[
            np.asarray(neighbors, dtype=np.int64)
        ]
        local_scale[index] = float(
            np.clip(
                np.median(ratio_by_skeleton[source_indices]),
                float(scale_min),
                float(scale_max),
            )
        )
        local_fitted_radius[index] = float(
            np.median(fitted_radius_by_skeleton[source_indices])
        )
        local_center_offset[index] = np.median(
            center_offset_by_skeleton[source_indices],
            axis=0,
        )
        direction = np.median(
            direction_by_skeleton[source_indices],
            axis=0,
        )
        if float(direction @ fallback_direction[index]) < 0.0:
            direction = -direction
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-9:
            direction = fallback_direction[index]
        else:
            direction /= direction_norm
        local_direction[index] = direction

    requested_front_blend = float(front_anchored_radius_blend)
    if not 0.0 <= requested_front_blend <= 1.0:
        raise ValueError(
            "front_anchored_radius_blend must be within [0, 1]"
        )
    front_valid_indices = np.flatnonzero(
        np.isfinite(front_radius_by_skeleton)
    )
    if len(front_valid_indices):
        (
            front_fit_accepted,
            front_fit_confidence,
        ) = _front_anchored_fit_confidence(
            valid_fits=len(front_valid_indices),
            skeleton_points=len(skeleton_uv),
            residual_median_m=float(np.median(front_residuals)),
            fitted_radius_median_m=float(np.median(front_radii)),
        )
        front_tree = cKDTree(skeleton_uv[front_valid_indices])
        local_front_radius = np.empty(
            len(skeleton_uv),
            dtype=np.float64,
        )
        local_front_direction = np.empty_like(fallback_direction)
        for index, uv in enumerate(skeleton_uv):
            neighbors = front_tree.query_ball_point(
                uv,
                r=float(smoothing_px),
            )
            if not neighbors:
                _distance, nearest = front_tree.query(uv, k=1)
                neighbors = [int(nearest)]
            source_indices = front_valid_indices[
                np.asarray(neighbors, dtype=np.int64)
            ]
            local_front_radius[index] = float(
                np.median(
                    front_radius_by_skeleton[source_indices]
                )
            )
            front_direction = np.median(
                front_direction_by_skeleton[source_indices],
                axis=0,
            )
            if (
                float(front_direction @ fallback_direction[index])
                < 0.0
            ):
                front_direction = -front_direction
            front_direction_norm = float(
                np.linalg.norm(front_direction)
            )
            if front_direction_norm < 1e-9:
                front_direction = fallback_direction[index]
            else:
                front_direction /= front_direction_norm
            local_front_direction[index] = front_direction
    else:
        front_fit_accepted = False
        front_fit_confidence = {
            "fit_fraction": 0.0,
            "normalized_residual": math.inf,
            "minimum_fit_fraction": 0.65,
            "maximum_normalized_residual": 0.12,
        }
        local_front_radius = image_radius.copy()
        local_front_direction = fallback_direction.copy()
    bounded_front_radius = np.clip(
        local_front_radius,
        image_radius * float(scale_min),
        image_radius * float(scale_max),
    )

    local_scale = (
        clipped_scale
        + float(local_shrinkage) * (local_scale - clipped_scale)
    )
    applied_scale = (
        1.0 + float(radius_blend) * (local_scale - 1.0)
    )
    requested_direct_blend = float(direct_radius_blend)
    if not 0.0 <= requested_direct_blend <= 1.0:
        raise ValueError("direct_radius_blend must be within [0, 1]")
    (
        direct_fit_accepted,
        direct_fit_confidence,
    ) = _direct_circle_fit_confidence(
        valid_fits=len(valid_indices),
        skeleton_points=len(skeleton_uv),
        residual_median_m=float(np.median(residuals)),
        fitted_radius_median_m=float(np.median(fitted_radii)),
    )
    soft_direct_blend, soft_direct_blend_metadata = (
        _select_soft_direct_radius_blend(
            requested_blend=requested_direct_blend,
            direct_fit_accepted=bool(direct_fit_accepted),
            fit_fraction=float(
                direct_fit_confidence["fit_fraction"]
            ),
            normalized_residual=float(
                direct_fit_confidence["normalized_residual"]
            ),
            fitted_to_image_radius_ratio=float(
                np.median(fitted_radii)
                / max(float(np.median(image_radii)), 1e-12)
            ),
            maximum_soft_blend=float(
                soft_direct_radius_blend_max
            ),
            enabled=bool(soft_direct_radius_blend),
        )
    )
    direct_blend = (
        requested_direct_blend
        if direct_fit_accepted
        else soft_direct_blend
    )
    front_blend, adaptive_front_blend_metadata = (
        _select_adaptive_front_radius_blend(
            requested_blend=requested_front_blend,
            front_fit_accepted=bool(front_fit_accepted),
            direct_fit_accepted=bool(direct_fit_accepted),
            front_radius_median_m=(
                float(np.median(front_radii))
                if front_radii
                else float(np.median(image_radius))
            ),
            image_radius_median_m=float(np.median(image_radii)),
        )
    )
    scaled_image_radius = image_radius * applied_scale
    bounded_fitted_radius = np.clip(
        local_fitted_radius,
        image_radius * float(scale_min),
        image_radius * float(scale_max),
    )
    radii = np.clip(
        (1.0 - direct_blend) * scaled_image_radius
        + direct_blend * bounded_fitted_radius,
        float(radius_min_m),
        float(radius_max_m),
    )
    radii = np.clip(
        (1.0 - front_blend) * radii
        + front_blend * bounded_front_radius,
        float(radius_min_m),
        float(radius_max_m),
    )
    radii, endpoint_radius_metadata = (
        _regularize_skeleton_endpoint_radii(
            skeleton_y=skeleton_y,
            skeleton_x=skeleton_x,
            radii=radii,
            requested_blend=float(
                endpoint_radius_regularization_blend
            ),
            direct_fit_accepted=bool(direct_fit_accepted),
        )
    )
    radii, graph_radius_metadata = (
        _regularize_skeleton_graph_radii_gcv(
            skeleton_y=skeleton_y,
            skeleton_x=skeleton_x,
            radii=radii,
            enabled=bool(graph_radius_smoothing_gcv),
            direct_fit_accepted=bool(direct_fit_accepted),
        )
    )
    direction = (
        (1.0 - float(center_fit_blend)) * fallback_direction
        + float(center_fit_blend) * local_direction
    )
    direction = (
        (1.0 - front_blend) * direction
        + front_blend * local_front_direction
    )
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
    centers, direct_center_metadata = _blend_direct_circle_centers(
        skeleton_world=skeleton_world,
        fallback_direction=direction,
        radii=radii,
        local_center_offset=local_center_offset,
        fitted_radii=bounded_fitted_radius,
        center_offset_scale=float(center_offset_scale),
        scale_min=float(scale_min),
        scale_max=float(scale_max),
        requested_blend=float(direct_center_blend),
        direct_fit_accepted=bool(direct_fit_accepted),
    )
    return centers, radii, {
        "valid_fits": int(len(valid_indices)),
        "required_valid_fits": int(min_valid_fits),
        "raw_scale": raw_scale,
        "clipped_scale": clipped_scale,
        "fallback_only": False,
        "fitted_radius_median_mm": float(
            np.median(fitted_radii) * 1000.0
        ),
        "image_radius_median_mm": float(
            np.median(image_radii) * 1000.0
        ),
        "fit_residual_median_mm": float(
            np.median(residuals) * 1000.0
        ),
        "fit_arc_span_p10_deg": float(
            np.percentile(arc_spans_deg, 10.0)
        ),
        "fit_arc_span_median_deg": float(
            np.median(arc_spans_deg)
        ),
        "fit_arc_span_p90_deg": float(
            np.percentile(arc_spans_deg, 90.0)
        ),
        "center_offset_median_mm": float(
            np.median(center_offset_norms) * 1000.0
        ),
        "local_scale_min": float(local_scale.min()),
        "local_scale_median": float(np.median(local_scale)),
        "local_scale_max": float(local_scale.max()),
        "applied_scale_min": float(applied_scale.min()),
        "applied_scale_median": float(np.median(applied_scale)),
        "applied_scale_max": float(applied_scale.max()),
        "local_fitted_radius_min_mm": float(
            local_fitted_radius.min() * 1000.0
        ),
        "local_fitted_radius_median_mm": float(
            np.median(local_fitted_radius) * 1000.0
        ),
        "local_fitted_radius_max_mm": float(
            local_fitted_radius.max() * 1000.0
        ),
        "bounded_fitted_radius_min_mm": float(
            bounded_fitted_radius.min() * 1000.0
        ),
        "bounded_fitted_radius_median_mm": float(
            np.median(bounded_fitted_radius) * 1000.0
        ),
        "bounded_fitted_radius_max_mm": float(
            bounded_fitted_radius.max() * 1000.0
        ),
        "radius_min_mm": float(radii.min() * 1000.0),
        "radius_median_mm": float(np.median(radii) * 1000.0),
        "radius_max_mm": float(radii.max() * 1000.0),
        "local_shrinkage": float(local_shrinkage),
        "radius_blend": float(radius_blend),
        "requested_direct_radius_blend": requested_direct_blend,
        "direct_radius_blend": direct_blend,
        "soft_direct_radius_blend": soft_direct_blend_metadata,
        "direct_fit_accepted": bool(direct_fit_accepted),
        "direct_fit_confidence": direct_fit_confidence,
        "front_anchored_valid_fits": int(len(front_valid_indices)),
        "front_anchored_radius_median_mm": (
            float(np.median(front_radii) * 1000.0)
            if front_radii
            else None
        ),
        "front_anchored_residual_median_mm": (
            float(np.median(front_residuals) * 1000.0)
            if front_residuals
            else None
        ),
        "front_anchored_fit_accepted": bool(front_fit_accepted),
        "front_anchored_fit_confidence": front_fit_confidence,
        "requested_front_anchored_radius_blend": float(
            requested_front_blend
        ),
        "front_anchored_radius_blend": float(front_blend),
        "adaptive_front_anchored_radius_blend": (
            adaptive_front_blend_metadata
        ),
        **direct_center_metadata,
        **endpoint_radius_metadata,
        "graph_radius_smoothing": graph_radius_metadata,
        "center_fit_blend": float(center_fit_blend),
        "center_offset_scale": float(center_offset_scale),
        "tangent_neighborhood_px": float(tangent_neighborhood_px),
    }


def _skeleton_world_tangents(
    skeleton_uv: np.ndarray,
    skeleton_world: np.ndarray,
    *,
    neighborhood_px: float = 10.0,
) -> np.ndarray:
    """Estimate an unoriented 3D tangent at every image skeleton point."""
    uv = np.asarray(skeleton_uv, dtype=np.float64).reshape(-1, 2)
    world = np.asarray(skeleton_world, dtype=np.float64).reshape(-1, 3)
    tangents = np.zeros_like(world)
    for index, point_uv in enumerate(uv):
        distance = np.linalg.norm(
            uv - point_uv.reshape(1, 2),
            axis=1,
        )
        local_indices = np.flatnonzero(
            distance <= float(neighborhood_px)
        )
        if len(local_indices) < 3:
            local_indices = np.argsort(distance)[: min(6, len(uv))]
        local_world = world[local_indices]
        centered = local_world - local_world.mean(axis=0, keepdims=True)
        if float(np.linalg.norm(centered)) < 1e-9:
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            tangent = np.asarray(vh[0], dtype=np.float64)
        tangent /= np.linalg.norm(tangent) + 1e-12
        tangents[index] = tangent
    return tangents


def _initial_skeleton_ray_perp_p10(
    *,
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    world: np.ndarray,
    camera_pos: Sequence[float],
    tangent_neighborhood_px: float,
) -> float:
    """Return a pre-refinement view observability percentile."""
    sy = np.asarray(skeleton_y, dtype=np.int64).reshape(-1)
    sx = np.asarray(skeleton_x, dtype=np.int64).reshape(-1)
    points = np.asarray(world, dtype=np.float64)
    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    if (
        len(sy) == 0
        or len(sy) != len(sx)
        or points.ndim != 3
        or points.shape[-1] != 3
        or np.min(sy) < 0
        or np.max(sy) >= points.shape[0]
        or np.min(sx) < 0
        or np.max(sx) >= points.shape[1]
        or not np.all(np.isfinite(camera))
    ):
        raise ValueError("invalid initial skeleton observability inputs")
    skeleton_world = np.asarray(points[sy, sx], dtype=np.float64)
    skeleton_uv = np.column_stack((sx, sy)).astype(np.float64)
    tangents = _skeleton_world_tangents(
        skeleton_uv,
        skeleton_world,
        neighborhood_px=float(tangent_neighborhood_px),
    )
    view_ray = skeleton_world - camera.reshape(1, 3)
    view_ray /= np.linalg.norm(view_ray, axis=1, keepdims=True) + 1e-12
    ray_perp = np.linalg.norm(
        view_ray
        - np.einsum("ij,ij->i", view_ray, tangents)[:, None]
        * tangents,
        axis=1,
    )
    return float(np.percentile(ray_perp, 10.0))


def _refine_local_cylinder_sections_nonlinear(
    *,
    component_mask: np.ndarray,
    skeleton_y: np.ndarray,
    skeleton_x: np.ndarray,
    world: np.ndarray,
    camera_pos: Sequence[float],
    initial_centers: np.ndarray,
    initial_radii: np.ndarray,
    tangent_neighborhood_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Refine local tube sections against an RGBD surface patch."""
    skeleton_uv = np.column_stack((skeleton_x, skeleton_y)).astype(
        np.float64
    )
    skeleton_world = np.asarray(
        world[skeleton_y, skeleton_x],
        dtype=np.float64,
    )
    component_y, component_x = np.nonzero(component_mask)
    component_uv = np.column_stack(
        (component_x, component_y)
    ).astype(np.float64)
    component_world = np.asarray(
        world[component_y, component_x],
        dtype=np.float64,
    )
    centers = np.asarray(initial_centers, dtype=np.float64).copy()
    radii = np.asarray(initial_radii, dtype=np.float64).copy()
    tangents = _skeleton_world_tangents(
        skeleton_uv,
        skeleton_world,
        neighborhood_px=float(tangent_neighborhood_px),
    )
    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    initial_view_ray = skeleton_world - camera.reshape(1, 3)
    initial_view_ray /= (
        np.linalg.norm(initial_view_ray, axis=1, keepdims=True) + 1e-12
    )
    initial_ray_perp = np.linalg.norm(
        initial_view_ray
        - np.einsum(
            "ij,ij->i",
            initial_view_ray,
            tangents,
        )[:, None]
        * tangents,
        axis=1,
    )

    patch_half_length_px = 6.0
    patch_half_width_px = 6.0
    max_center_shift_m = 0.0015
    min_radius_ratio = 2.0 / 3.0
    max_radius_ratio = 1.5
    max_tangent_change_deg = 4.0
    min_residual_gain = 1.5
    max_refined_residual_m = 0.00015

    accepted = 0
    attempted = 0
    base_residuals: list[float] = []
    refined_residuals: list[float] = []
    radius_ratios: list[float] = []
    center_shifts: list[float] = []
    tangent_changes: list[float] = []
    for index, (
        uv,
        initial_center,
        initial_radius,
        initial_tangent,
    ) in enumerate(
        zip(
            skeleton_uv,
            centers.copy(),
            radii.copy(),
            tangents,
        )
    ):
        tangent_uv = _local_skeleton_tangent_uv(
            skeleton_uv,
            uv,
            neighborhood_px=float(tangent_neighborhood_px),
        )
        if tangent_uv is None:
            continue
        normal_uv = np.array(
            [-tangent_uv[1], tangent_uv[0]],
            dtype=np.float64,
        )
        delta_uv = component_uv - uv.reshape(1, 2)
        patch_mask = (
            (np.abs(delta_uv @ tangent_uv) <= patch_half_length_px)
            & (np.abs(delta_uv @ normal_uv) <= patch_half_width_px)
        )
        patch = component_world[patch_mask]
        if len(patch) < 10:
            continue

        view_basis = camera - initial_center
        view_basis -= (
            initial_tangent
            * float(view_basis @ initial_tangent)
        )
        view_norm = float(np.linalg.norm(view_basis))
        if view_norm < 1e-9:
            continue
        basis_first = view_basis / view_norm
        basis_second = np.cross(initial_tangent, basis_first)
        basis_second /= np.linalg.norm(basis_second) + 1e-12

        def decode(
            parameters: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, float]:
            tangent = (
                initial_tangent
                + parameters[2] * basis_first
                + parameters[3] * basis_second
            )
            tangent /= np.linalg.norm(tangent) + 1e-12
            center = (
                initial_center
                + parameters[0] * basis_first
                + parameters[1] * basis_second
            )
            radius = float(
                initial_radius * math.exp(float(parameters[4]))
            )
            return center, tangent, radius

        def signed_residual(
            center: np.ndarray,
            tangent: np.ndarray,
            radius: float,
        ) -> np.ndarray:
            relative = patch - center.reshape(1, 3)
            axial = relative @ tangent
            radial = np.linalg.norm(
                relative - axial[:, None] * tangent,
                axis=1,
            )
            return radial - float(radius)

        def objective(parameters: np.ndarray) -> np.ndarray:
            return signed_residual(*decode(parameters))

        attempted += 1
        try:
            optimization = least_squares(
                objective,
                np.zeros(5, dtype=np.float64),
                bounds=(
                    np.array(
                        [-0.004, -0.004, -0.6, -0.6, -0.7],
                        dtype=np.float64,
                    ),
                    np.array(
                        [0.004, 0.004, 0.6, 0.6, 0.7],
                        dtype=np.float64,
                    ),
                ),
                loss="soft_l1",
                f_scale=0.00015,
                max_nfev=60,
            )
        except Exception:
            continue
        refined_center, refined_tangent, refined_radius = decode(
            optimization.x
        )
        if (
            not np.all(np.isfinite(refined_center))
            or not np.all(np.isfinite(refined_tangent))
            or not math.isfinite(refined_radius)
        ):
            continue
        base_residual = float(
            np.median(
                np.abs(
                    signed_residual(
                        initial_center,
                        initial_tangent,
                        float(initial_radius),
                    )
                )
            )
        )
        refined_residual = float(
            np.median(
                np.abs(
                    signed_residual(
                        refined_center,
                        refined_tangent,
                        refined_radius,
                    )
                )
            )
        )
        radius_ratio = float(
            refined_radius / max(float(initial_radius), 1e-12)
        )
        center_shift = float(
            np.linalg.norm(refined_center - initial_center)
        )
        tangent_change = math.degrees(
            math.acos(
                float(
                    np.clip(
                        abs(
                            float(
                                refined_tangent @ initial_tangent
                            )
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        residual_gain = base_residual / max(
            refined_residual,
            1e-12,
        )
        trusted = bool(
            min_radius_ratio <= radius_ratio <= max_radius_ratio
            and center_shift <= max_center_shift_m
            and tangent_change <= max_tangent_change_deg
            and residual_gain >= min_residual_gain
            and refined_residual <= max_refined_residual_m
        )
        if not trusted:
            continue
        centers[index] = refined_center
        radii[index] = refined_radius
        tangents[index] = refined_tangent
        accepted += 1
        base_residuals.append(base_residual)
        refined_residuals.append(refined_residual)
        radius_ratios.append(radius_ratio)
        center_shifts.append(center_shift)
        tangent_changes.append(tangent_change)

    return centers, radii, tangents, {
        "enabled": True,
        "attempted_sections": int(attempted),
        "accepted_sections": int(accepted),
        "accepted_fraction": float(
            accepted / max(attempted, 1)
        ),
        "initial_ray_perp_p10": (
            float(np.percentile(initial_ray_perp, 10.0))
            if len(initial_ray_perp)
            else None
        ),
        "patch_half_length_px": patch_half_length_px,
        "patch_half_width_px": patch_half_width_px,
        "max_center_shift_mm": max_center_shift_m * 1000.0,
        "min_radius_ratio": min_radius_ratio,
        "max_radius_ratio": max_radius_ratio,
        "max_tangent_change_deg": max_tangent_change_deg,
        "min_residual_gain": min_residual_gain,
        "max_refined_residual_mm": (
            max_refined_residual_m * 1000.0
        ),
        "base_residual_median_mm": (
            float(np.median(base_residuals) * 1000.0)
            if base_residuals
            else None
        ),
        "refined_residual_median_mm": (
            float(np.median(refined_residuals) * 1000.0)
            if refined_residuals
            else None
        ),
        "radius_ratio_median": (
            float(np.median(radius_ratios))
            if radius_ratios
            else None
        ),
        "center_shift_median_mm": (
            float(np.median(center_shifts) * 1000.0)
            if center_shifts
            else None
        ),
        "tangent_change_median_deg": (
            float(np.median(tangent_changes))
            if tangent_changes
            else None
        ),
    }


def _mesh_from_discrete_occupancy_components(
    occupancy: np.ndarray,
    *,
    lower: np.ndarray,
    voxel_m: float,
    smoothing_sigma_voxels: float = 0.35,
) -> tuple[trimesh.Trimesh, int, int]:
    """Mesh 6-connected voxel bodies separately to avoid corner contacts."""
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, component_count = ndimage.label(
        np.asarray(occupancy, dtype=bool),
        structure=structure,
    )
    meshes: list[trimesh.Trimesh] = []
    for component in range(1, int(component_count) + 1):
        indices = np.argwhere(labels == component)
        if len(indices) == 0:
            continue
        start = np.maximum(indices.min(axis=0) - 1, 0)
        stop = np.minimum(
            indices.max(axis=0) + 2,
            np.asarray(labels.shape, dtype=np.int64),
        )
        slices = tuple(
            slice(int(start[axis]), int(stop[axis]))
            for axis in range(3)
        )
        local = labels[slices] == component
        local = np.pad(local, 2, mode="constant", constant_values=False)
        field = ndimage.gaussian_filter(
            local.astype(np.float32),
            sigma=float(smoothing_sigma_voxels),
            mode="constant",
            cval=0.0,
        )
        if float(field.max()) <= 0.5 or float(field.min()) >= 0.5:
            continue
        vertices, faces, _normals, _values = measure.marching_cubes(
            field,
            level=0.5,
            spacing=(float(voxel_m),) * 3,
            allow_degenerate=False,
            method="lewiner",
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
            validate=False,
        )
        mesh.vertices += (
            np.asarray(lower, dtype=np.float64)
            + (start.astype(np.float64) - 2.0) * float(voxel_m)
        )
        meshes.append(mesh)
    if not meshes:
        raise ValueError("occupancy has no meshable 6-connected components")
    vertices = []
    faces = []
    vertex_offset = 0
    for component_mesh in meshes:
        component_vertices = np.asarray(
            component_mesh.vertices,
            dtype=np.float64,
        )
        component_faces = np.asarray(
            component_mesh.faces,
            dtype=np.int64,
        )
        vertices.append(component_vertices)
        faces.append(component_faces + int(vertex_offset))
        vertex_offset += int(len(component_vertices))
    mesh = trimesh.Trimesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
        process=False,
        validate=False,
    )
    nonwatertight_components = sum(
        not bool(component_mesh.is_watertight)
        for component_mesh in meshes
    )
    return mesh, int(component_count), int(nonwatertight_components)


def _mesh_from_implicit_union_field(
    field: np.ndarray,
    *,
    lower: np.ndarray,
    voxel_m: float,
) -> tuple[trimesh.Trimesh, int, int]:
    """Extract a continuous zero-level set from a union-of-balls field."""
    values = np.asarray(field, dtype=np.float32)
    positive = values >= 0.0
    if not np.any(positive):
        raise ValueError("implicit ray field has no nonnegative samples")
    if np.all(positive):
        raise ValueError("implicit ray field has no exterior samples")
    vertices, faces, _normals, _samples = measure.marching_cubes(
        values,
        level=0.0,
        spacing=(float(voxel_m),) * 3,
        allow_degenerate=False,
        method="lewiner",
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False,
    )
    mesh.vertices += np.asarray(lower, dtype=np.float64)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    _labels, component_count = ndimage.label(
        positive,
        structure=structure,
    )
    mesh_components = mesh.split(
        only_watertight=False,
    )
    nonwatertight_components = sum(
        not bool(component.is_watertight)
        for component in mesh_components
    )
    return mesh, int(component_count), int(nonwatertight_components)


def _mesh_from_local_circle_tubes(
    extraction: DepthMeshReconstruction,
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    voxel_mm: float,
    click_u: int,
    click_v: int,
    component_policy: str,
    component_gap_3d_mm: float,
    component_gap_px: float,
    component_color_delta: float,
    circle_radius_scale: float,
    circle_radius_bias_px: float,
    circle_adaptive_radius_bias: bool,
    circle_radius_min_mm: float,
    circle_radius_max_mm: float,
    circle_center_fit_blend: float,
    circle_center_offset_scale: float,
    circle_tangent_neighborhood_px: float,
    curvature_radius_blend: float,
    curvature_scale_min: float,
    curvature_scale_max: float,
    curvature_fit_max_residual_mm: float,
    curvature_min_valid_fits: int,
    curvature_smoothing_px: float,
    curvature_local_shrinkage: float,
    curvature_max_center_toward_camera_ratio: float,
    circle_direct_radius_blend: float,
    circle_soft_direct_radius_blend: bool,
    circle_soft_direct_radius_blend_max: float,
    circle_direct_center_blend: float,
    circle_adaptive_direct_center_blend: bool,
    circle_front_anchored_radius_blend: float,
    circle_endpoint_radius_regularization_blend: float,
    circle_graph_radius_smoothing_gcv: bool,
    circle_chain_geometry_smoothing_gcv: bool,
    circle_global_surface_fit: bool,
    circle_global_radius_curvature_sigma: float,
    circle_global_sections_per_control: float,
    circle_global_center_prior_sigma_mm: float,
    circle_global_adaptive_sections_per_control: bool,
    circle_global_control_cv: bool,
    circle_global_center_prior_cv: bool,
    circle_global_point_parameter_refinement_sections: int,
    circle_global_point_parameter_refinement_blend: float,
    circle_global_adaptive_point_parameter_refinement: bool,
    circle_global_front_anchor_sigma_mm: float,
    circle_global_front_anchor_min_fraction: float,
    circle_global_front_anchor_max_ray_perp_p10: float,
    circle_fixed_centerline_radius_refinement: bool,
    circle_fixed_centerline_radius_quantile: float,
    circle_fixed_centerline_radius_bandwidth_sections: float,
    circle_fixed_centerline_radius_blend: float,
    circle_degenerate_endpoint_curve_refinement: bool,
    circle_degenerate_endpoint_curve_blend: float,
    circle_prune_short_skeleton_spurs: bool,
    circle_axial_endpoint_radius_repair: bool,
    circle_visible_endpoint_extension: bool,
    circle_visible_endpoint_extension_blend: float,
    circle_visible_endpoint_extension_max_radius_ratio: float,
    circle_visible_endpoint_preserve_original_ring: bool,
    tube_meshing_mode: str,
    swept_ring_samples: int,
    circle_polygonal_ring_selection: bool,
    circle_polygonal_facet_inversion: bool,
    local_cylinder_nonlinear_refine: bool,
    local_cylinder_nonlinear_local_accepts: bool,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Complete the target as locally fitted circular tubes."""
    selected = np.asarray(extraction.selected_mask, dtype=bool)
    color = np.asarray(rgb, dtype=np.float64)
    metric_depth = np.asarray(depth, dtype=np.float64)
    h, w = metric_depth.shape[:2]
    focal_px = (
        float(focal_length)
        / float(horizontal_aperture)
        * float(w)
    )
    world, _valid = backproject_depth_world(
        metric_depth,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
    )
    policy = str(component_policy).strip().lower()
    if policy == "seed_image":
        labels, yy, xx, component_for_node = (
            _image_seed_component_labels(
                selected,
                click_u=int(click_u),
                click_v=int(click_v),
            )
        )
    else:
        labels, yy, xx, component_for_node = _depth_component_labels(
            selected,
            world_points=world,
            rgb=color,
            depth=metric_depth,
            focal_px=focal_px,
        )
    component_sizes = np.bincount(component_for_node).astype(np.int64)
    if len(component_sizes) == 0:
        raise ValueError("RGBD target has no 3D-connected components")
    requested_meshing_mode = str(tube_meshing_mode).strip().lower()
    if requested_meshing_mode not in {"voxel_union", "swept_surface"}:
        raise ValueError(
            f"unsupported tube_meshing_mode={tube_meshing_mode!r}"
        )

    nearest_click_node = int(
        np.argmin(
            (xx - int(click_u)) ** 2 + (yy - int(click_v)) ** 2
        )
    )
    seed_component = int(component_for_node[nearest_click_node])
    if policy == "all":
        selected_components = set(range(len(component_sizes)))
    elif policy in {"seed_only", "seed_image"}:
        selected_components = {seed_component}
    elif policy == "seed_direct":
        selected_components = {seed_component}
        seed_mask = component_for_node == seed_component
        seed_y = yy[seed_mask]
        seed_x = xx[seed_mask]
        seed_world = world[seed_y, seed_x]
        seed_tree = cKDTree(seed_world)
        for component in range(len(component_sizes)):
            if component == seed_component:
                continue
            candidate_mask = component_for_node == component
            candidate_y = yy[candidate_mask]
            candidate_x = xx[candidate_mask]
            candidate_world = world[candidate_y, candidate_x]
            distance, seed_index = seed_tree.query(candidate_world, k=1)
            best = int(np.argmin(distance))
            paired_seed = int(seed_index[best])
            uv_gap = float(
                np.linalg.norm(
                    np.array(
                        [candidate_x[best], candidate_y[best]],
                        dtype=np.float64,
                    )
                    - np.array(
                        [seed_x[paired_seed], seed_y[paired_seed]],
                        dtype=np.float64,
                    )
                )
            )
            color_gap = float(
                np.linalg.norm(
                    color[candidate_y[best], candidate_x[best]]
                    - color[seed_y[paired_seed], seed_x[paired_seed]]
                )
            )
            if (
                float(distance[best])
                <= float(component_gap_3d_mm) / 1000.0
                and uv_gap <= float(component_gap_px)
                and color_gap <= float(component_color_delta)
            ):
                selected_components.add(component)
    else:
        raise ValueError(
            f"unsupported component_policy={component_policy!r}; expected "
            "'all', 'seed_only', 'seed_image', or 'seed_direct'"
        )

    voxel_m = float(voxel_mm) / 1000.0
    center_parts: list[np.ndarray] = []
    radius_parts: list[np.ndarray] = []
    component_metadata: Dict[int, Dict[str, Any]] = {}
    skeleton_sizes: list[int] = []
    graph_edge_count = 0
    graph_interpolated_count = 0
    swept_meshes: list[trimesh.Trimesh] = []
    swept_chain_sections = 0
    swept_fallback_reason: str | None = None
    for component in range(len(component_sizes)):
        if component not in selected_components:
            continue
        component_mask = labels == component
        skeleton = skeletonize(component_mask)
        sy, sx = np.nonzero(skeleton)
        if len(sx) == 0:
            continue
        sy, sx, skeleton_spur_metadata = (
            _prune_short_skeleton_spurs(
                sy,
                sx,
                enabled=bool(circle_prune_short_skeleton_spurs),
            )
        )
        observation_density = float(
            np.count_nonzero(component_mask) / max(len(sx), 1)
        )
        def fit_component_sections(
            radius_bias_px: float,
            direct_center_blend: float,
        ) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            return _component_circle_tube_sections(
                component_mask=component_mask,
                skeleton_y=sy,
                skeleton_x=sx,
                world=world,
                depth=metric_depth,
                camera_pos=camera_pos,
                focal_px=focal_px,
                radius_scale=float(circle_radius_scale),
                radius_bias_px=float(radius_bias_px),
                radius_min_m=float(circle_radius_min_mm) / 1000.0,
                radius_max_m=float(circle_radius_max_mm) / 1000.0,
                fit_max_residual_m=(
                    float(curvature_fit_max_residual_mm) / 1000.0
                ),
                scale_min=float(curvature_scale_min),
                scale_max=float(curvature_scale_max),
                min_valid_fits=int(curvature_min_valid_fits),
                smoothing_px=float(curvature_smoothing_px),
                local_shrinkage=float(curvature_local_shrinkage),
                radius_blend=float(curvature_radius_blend),
                direct_radius_blend=float(circle_direct_radius_blend),
                soft_direct_radius_blend=bool(
                    circle_soft_direct_radius_blend
                ),
                soft_direct_radius_blend_max=float(
                    circle_soft_direct_radius_blend_max
                ),
                direct_center_blend=float(direct_center_blend),
                front_anchored_radius_blend=float(
                    circle_front_anchored_radius_blend
                ),
                endpoint_radius_regularization_blend=float(
                    circle_endpoint_radius_regularization_blend
                ),
                graph_radius_smoothing_gcv=bool(
                    circle_graph_radius_smoothing_gcv
                ),
                center_fit_blend=float(circle_center_fit_blend),
                center_offset_scale=float(circle_center_offset_scale),
                tangent_neighborhood_px=float(
                    circle_tangent_neighborhood_px
                ),
                max_center_toward_camera_ratio=float(
                    curvature_max_center_toward_camera_ratio
                ),
            )

        centers, radii, metadata = fit_component_sections(
            float(circle_radius_bias_px),
            float(circle_direct_center_blend),
        )
        initial_ray_perp_p10 = _initial_skeleton_ray_perp_p10(
            skeleton_y=sy,
            skeleton_x=sx,
            world=world,
            camera_pos=camera_pos,
            tangent_neighborhood_px=float(
                circle_tangent_neighborhood_px
            ),
        )
        (
            effective_radius_bias_px,
            radius_bias_policy,
        ) = _select_adaptive_circle_radius_bias_px(
            requested_bias_px=float(circle_radius_bias_px),
            direct_fit_accepted=bool(
                metadata.get("direct_fit_accepted", False)
            ),
            direct_fit_fraction=float(
                metadata.get("direct_fit_confidence", {}).get(
                    "fit_fraction",
                    0.0,
                )
            ),
            initial_ray_perp_p10=initial_ray_perp_p10,
            enabled=bool(circle_adaptive_radius_bias),
        )
        if radius_bias_policy["enabled"]:
            centers, radii, metadata = fit_component_sections(
                effective_radius_bias_px,
                float(circle_direct_center_blend),
            )
        metadata["adaptive_radius_bias_policy"] = radius_bias_policy
        (
            effective_initial_direct_center_blend,
            supported_direct_center_blend_policy,
        ) = _select_supported_direct_center_blend(
            requested_blend=float(circle_direct_center_blend),
            direct_fit_accepted=bool(
                metadata.get("direct_fit_accepted", False)
            ),
            fitted_to_image_radius_ratio=float(
                metadata.get("fitted_radius_median_mm", 0.0)
                / max(
                    float(
                        metadata.get("image_radius_median_mm", 0.0)
                    ),
                    1e-12,
                )
            ),
            initial_ray_perp_p10=initial_ray_perp_p10,
            observations_per_section=observation_density,
            enabled=bool(circle_adaptive_direct_center_blend),
        )
        if supported_direct_center_blend_policy["enabled"]:
            centers, radii, metadata = fit_component_sections(
                effective_radius_bias_px,
                effective_initial_direct_center_blend,
            )
            metadata["adaptive_radius_bias_policy"] = (
                radius_bias_policy
            )
        metadata["supported_direct_center_blend_policy"] = (
            supported_direct_center_blend_policy
        )
        chain_order: np.ndarray | None = None
        chain_metadata: Dict[str, Any] = {"eligible": False}
        swept_section_diagnostics: Dict[str, Any] = {
            "available": False,
        }
        if requested_meshing_mode == "swept_surface":
            try:
                graph_chains, _graph_degrees, chain_metadata = (
                    _skeleton_graph_chains(
                        sy,
                        sx,
                    )
                )
                if (
                    len(graph_chains) == 1
                    and len(graph_chains[0]) == len(sx)
                ):
                    chain_order = graph_chains[0]
            except ValueError as exc:
                swept_fallback_reason = str(exc)
            else:
                chain_metadata["eligible"] = True
        if chain_order is not None:
            try:
                verified_order, _verified_metadata = (
                    _skeleton_chain_order(
                        sy,
                        sx,
                    )
                )
            except ValueError:
                chain_order = None
            else:
                chain_order = verified_order
        initial_centers = centers.copy()
        initial_radii = radii.copy()
        if chain_order is not None:
            swept_section_diagnostics = {
                "available": True,
                "chain_order": chain_order.astype(np.int64).tolist(),
                "skeleton_uv": np.column_stack(
                    (sx[chain_order], sy[chain_order])
                )
                .astype(np.int64)
                .tolist(),
                "initial_centers_world": (
                    initial_centers[chain_order].tolist()
                ),
                "initial_radii_mm": (
                    initial_radii[chain_order] * 1000.0
                ).tolist(),
            }
        nonlinear_metadata: Dict[str, Any] = {
            "enabled": bool(local_cylinder_nonlinear_refine),
            "attempted_sections": 0,
            "accepted_sections": 0,
            "accepted_fraction": 0.0,
            "applied_to_component": False,
        }
        nonlinear_applied = False
        if bool(local_cylinder_nonlinear_refine):
            (
                refined_centers,
                refined_radii,
                _refined_tangents,
                nonlinear_metadata,
            ) = _refine_local_cylinder_sections_nonlinear(
                component_mask=component_mask,
                skeleton_y=sy,
                skeleton_x=sx,
                world=world,
                camera_pos=camera_pos,
                initial_centers=centers,
                initial_radii=radii,
                tangent_neighborhood_px=float(
                    circle_tangent_neighborhood_px
                ),
            )
            nonlinear_applied = _accept_nonlinear_refinement_result(
                accepted_sections=int(
                    nonlinear_metadata["accepted_sections"]
                ),
                accepted_fraction=float(
                    nonlinear_metadata["accepted_fraction"]
                ),
                minimum_component_fraction=0.5,
                local_accepted_sections=bool(
                    local_cylinder_nonlinear_local_accepts
                ),
            )
            adaptive_sparse_applied = bool(
                not nonlinear_applied
                and _accept_adaptive_sparse_nonlinear_refinement(
                    accepted_sections=int(
                        nonlinear_metadata["accepted_sections"]
                    ),
                    accepted_fraction=float(
                        nonlinear_metadata["accepted_fraction"]
                    ),
                    initial_ray_perp_p10=nonlinear_metadata.get(
                        "initial_ray_perp_p10"
                    ),
                )
            )
            nonlinear_applied = bool(
                nonlinear_applied or adaptive_sparse_applied
            )
            if (
                nonlinear_applied
                and requested_meshing_mode == "swept_surface"
                and chain_order is None
            ):
                nonlinear_applied = False
                nonlinear_metadata["swept_continuity_gate"] = {
                    "accepted": False,
                    "reason": "branched_graph_requires_global_refinement",
                }
            if nonlinear_applied and chain_order is not None:
                base_residual_mm = nonlinear_metadata.get(
                    "base_residual_median_mm"
                )
                refined_residual_mm = nonlinear_metadata.get(
                    "refined_residual_median_mm"
                )
                if (
                    base_residual_mm is not None
                    and refined_residual_mm is not None
                ):
                    (
                        continuity_accepted,
                        continuity_metadata,
                    ) = _swept_refinement_continuity_gate(
                        chain_order=chain_order,
                        initial_radii=initial_radii,
                        refined_radii=refined_radii,
                        base_residual_m=(
                            float(base_residual_mm) / 1000.0
                        ),
                        refined_residual_m=(
                            float(refined_residual_mm) / 1000.0
                        ),
                    )
                    nonlinear_metadata["swept_continuity_gate"] = (
                        continuity_metadata
                    )
                    nonlinear_applied = bool(continuity_accepted)
            nonlinear_metadata.update(
                {
                    "application_mode": (
                        "local_accepted_sections"
                        if bool(
                            local_cylinder_nonlinear_local_accepts
                        )
                        else (
                            "adaptive_sparse_sections"
                            if adaptive_sparse_applied
                            else "component_fraction_gate"
                        )
                    ),
                    "adaptive_sparse_applied": adaptive_sparse_applied,
                    "adaptive_sparse_minimum_sections": 8,
                    "adaptive_sparse_minimum_fraction": 0.35,
                    "adaptive_sparse_maximum_ray_perp_p10": 0.4,
                    "applied_to_component": bool(nonlinear_applied),
                }
            )
            if nonlinear_applied:
                centers = refined_centers
                radii = refined_radii
            else:
                centers = initial_centers
                radii = initial_radii

        (
            effective_direct_center_blend,
            adaptive_direct_center_blend_policy,
        ) = _select_adaptive_direct_center_blend(
            requested_blend=effective_initial_direct_center_blend,
            direct_fit_accepted=bool(
                metadata.get("direct_fit_accepted", False)
            ),
            nonlinear_refinement_applied=bool(nonlinear_applied),
            observations_per_section=observation_density,
            preselected_adaptive=bool(
                supported_direct_center_blend_policy["enabled"]
            ),
            enabled=bool(circle_adaptive_direct_center_blend),
        )
        if adaptive_direct_center_blend_policy["enabled"]:
            centers, radii, metadata = fit_component_sections(
                effective_radius_bias_px,
                effective_direct_center_blend,
            )
            metadata["adaptive_radius_bias_policy"] = (
                radius_bias_policy
            )
            metadata["supported_direct_center_blend_policy"] = (
                supported_direct_center_blend_policy
            )
        metadata["adaptive_direct_center_blend_policy"] = (
            adaptive_direct_center_blend_policy
        )

        chain_geometry_metadata: Dict[str, Any] = {
            "requested": bool(circle_chain_geometry_smoothing_gcv),
            "enabled": False,
        }
        if chain_order is not None:
            (
                centers,
                radii,
                chain_geometry_metadata,
            ) = _regularize_swept_chain_geometry_gcv(
                chain_order=chain_order,
                centers=centers,
                radii=radii,
                enabled=bool(circle_chain_geometry_smoothing_gcv),
            )
            swept_section_diagnostics.update(
                {
                    "preglobal_centers_world": (
                        centers[chain_order].tolist()
                    ),
                    "preglobal_radii_mm": (
                        radii[chain_order] * 1000.0
                    ).tolist(),
                }
            )

        global_surface_fit_metadata: Dict[str, Any] = {
            "requested": bool(circle_global_surface_fit),
            "enabled": False,
            "accepted": False,
        }
        surface_points_for_radius_refinement: np.ndarray | None = None
        surface_fold_ids_for_radius_refinement: np.ndarray | None = None
        if chain_order is not None:
            ordered_centers = centers[chain_order]
            ordered_uv = np.column_stack(
                (
                    sx[chain_order],
                    sy[chain_order],
                )
            ).astype(np.float64)
            ordered_step = np.linalg.norm(
                np.diff(ordered_centers, axis=0),
                axis=1,
            )
            ordered_s = np.concatenate(
                ([0.0], np.cumsum(ordered_step))
            )
            if float(ordered_s[-1]) > 1e-9:
                ordered_s /= float(ordered_s[-1])
                component_y, component_x = np.nonzero(component_mask)
                component_uv = np.column_stack(
                    (component_x, component_y)
                ).astype(np.float64)
                _distance, nearest_ordered = cKDTree(
                    ordered_uv
                ).query(component_uv, k=1)
                point_parameters = ordered_s[
                    np.asarray(nearest_ordered, dtype=np.int64)
                ]
                surface_points = np.asarray(
                    world[component_y, component_x],
                    dtype=np.float64,
                )
                surface_points_for_radius_refinement = surface_points
                surface_fold_ids_for_radius_refinement = (
                    component_x + 2 * component_y
                ) % 4
                preglobal_centers = centers.copy()
                preglobal_radii = radii.copy()
                prospective_anchor_policy_accepted = bool(
                    float(circle_global_front_anchor_sigma_mm) > 0.0
                    and _accept_degenerate_view_front_anchor(
                        accepted_sections=int(
                            nonlinear_metadata["accepted_sections"]
                        ),
                        accepted_fraction=float(
                            nonlinear_metadata["accepted_fraction"]
                        ),
                        initial_ray_perp_p10=nonlinear_metadata.get(
                            "initial_ray_perp_p10"
                        ),
                        minimum_fraction=float(
                            circle_global_front_anchor_min_fraction
                        ),
                        maximum_ray_perp_p10=float(
                            circle_global_front_anchor_max_ray_perp_p10
                        ),
                    )
                )
                (
                    effective_point_parameter_refinement_sections,
                    adaptive_point_parameter_policy,
                ) = _select_adaptive_point_parameter_refinement_sections(
                    requested_sections=int(
                        circle_global_point_parameter_refinement_sections
                    ),
                    direct_fit_accepted=bool(
                        metadata.get("direct_fit_accepted", False)
                    ),
                    front_anchor_policy_accepted=(
                        prospective_anchor_policy_accepted
                    ),
                    nonlinear_refinement_applied=bool(
                        nonlinear_applied
                    ),
                    observations_per_section=float(
                        len(surface_points) / len(chain_order)
                    ),
                    enabled=bool(
                        circle_global_surface_fit
                        and circle_global_adaptive_point_parameter_refinement
                    ),
                )
                (
                    point_parameters,
                    point_parameter_refinement,
                ) = _refine_surface_point_parameters_local_3d(
                    chain_order=chain_order,
                    centers=preglobal_centers,
                    initial_point_parameters=point_parameters,
                    surface_points=surface_points,
                    max_shift_sections=int(
                        effective_point_parameter_refinement_sections
                    ),
                    blend=float(
                        min(
                            circle_global_point_parameter_refinement_blend,
                            0.25,
                        )
                        if adaptive_point_parameter_policy["mode"]
                        == "front_anchored_indirect_supported"
                        else circle_global_point_parameter_refinement_blend
                    ),
                )
                point_parameter_refinement[
                    "adaptive_policy"
                ] = adaptive_point_parameter_policy
                (
                    effective_radius_curvature_sigma,
                    radius_curvature_policy,
                ) = _select_adaptive_radius_curvature_sigma(
                    requested_sigma=float(
                        circle_global_radius_curvature_sigma
                    ),
                    direct_fit_accepted=bool(
                        metadata.get("direct_fit_accepted", False)
                    ),
                    accepted_fraction=float(
                        nonlinear_metadata["accepted_fraction"]
                    ),
                    initial_ray_perp_p10=nonlinear_metadata.get(
                        "initial_ray_perp_p10"
                    ),
                    front_anchor_policy_accepted=(
                        prospective_anchor_policy_accepted
                    ),
                    observations_per_section=float(
                        len(surface_points) / len(chain_order)
                    ),
                )
                (
                    adaptive_sections_per_control,
                    adaptive_control_spacing_policy,
                ) = _select_adaptive_global_sections_per_control(
                    requested_sections_per_control=float(
                        circle_global_sections_per_control
                    ),
                    direct_fit_accepted=bool(
                        metadata.get("direct_fit_accepted", False)
                    ),
                    accepted_fraction=float(
                        nonlinear_metadata["accepted_fraction"]
                    ),
                    initial_ray_perp_p10=nonlinear_metadata.get(
                        "initial_ray_perp_p10"
                    ),
                    nonlinear_refinement_applied=bool(
                        nonlinear_applied
                    ),
                    observations_per_section=float(
                        len(surface_points) / len(chain_order)
                    ),
                    front_anchor_policy_accepted=(
                        prospective_anchor_policy_accepted
                    ),
                    enabled=bool(
                        circle_global_surface_fit
                        and circle_global_adaptive_sections_per_control
                    ),
                )
                (
                    effective_sections_per_control,
                    control_spacing_policy,
                ) = _select_global_sections_per_control_cv(
                    chain_order=chain_order,
                    centers=preglobal_centers,
                    radii=preglobal_radii,
                    point_parameters=point_parameters,
                    surface_points=surface_points,
                    requested_sections_per_control=(
                        adaptive_sections_per_control
                    ),
                    radius_curvature_sigma=(
                        effective_radius_curvature_sigma
                    ),
                    enabled=bool(
                        circle_global_surface_fit
                        and circle_global_control_cv
                    ),
                )
                (
                    effective_center_prior_sigma_m,
                    center_prior_cv,
                ) = _select_global_center_prior_sigma_cv(
                    chain_order=chain_order,
                    centers=preglobal_centers,
                    radii=preglobal_radii,
                    point_parameters=point_parameters,
                    surface_points=surface_points,
                    requested_center_prior_sigma_m=(
                        float(circle_global_center_prior_sigma_mm)
                        / 1000.0
                    ),
                    radius_curvature_sigma=(
                        effective_radius_curvature_sigma
                    ),
                    sections_per_control=(
                        effective_sections_per_control
                    ),
                    enabled=bool(
                        circle_global_surface_fit
                        and circle_global_center_prior_cv
                        and not prospective_anchor_policy_accepted
                    ),
                )
                (
                    baseline_centers,
                    baseline_radii,
                    baseline_global_metadata,
                ) = _optimize_swept_chain_surface_global(
                    chain_order=chain_order,
                    centers=preglobal_centers,
                    radii=preglobal_radii,
                    point_parameters=point_parameters,
                    surface_points=surface_points,
                    enabled=bool(circle_global_surface_fit),
                    radius_curvature_sigma=(
                        effective_radius_curvature_sigma
                    ),
                    sections_per_control=(
                        effective_sections_per_control
                    ),
                    center_prior_sigma_m=effective_center_prior_sigma_m,
                )
                anchor_policy_accepted = (
                    prospective_anchor_policy_accepted
                )
                anchor_candidate_metadata: Dict[str, Any] | None = None
                centers = baseline_centers
                radii = baseline_radii
                global_surface_fit_metadata = baseline_global_metadata
                selected_solution = "unanchored_baseline"
                if anchor_policy_accepted:
                    (
                        anchor_centers,
                        anchor_radii,
                        anchor_candidate_metadata,
                    ) = _optimize_swept_chain_surface_global(
                        chain_order=chain_order,
                        centers=preglobal_centers,
                        radii=preglobal_radii,
                        point_parameters=point_parameters,
                        surface_points=surface_points,
                        enabled=bool(circle_global_surface_fit),
                        radius_curvature_sigma=(
                            effective_radius_curvature_sigma
                        ),
                        sections_per_control=(
                            effective_sections_per_control
                        ),
                        center_prior_sigma_m=(
                            effective_center_prior_sigma_m
                        ),
                        front_anchor_points=np.asarray(
                            world[sy, sx],
                            dtype=np.float64,
                        ),
                        camera_pos=camera_pos,
                        front_anchor_sigma_m=(
                            float(
                                circle_global_front_anchor_sigma_mm
                            )
                            / 1000.0
                        ),
                    )
                    if bool(
                        anchor_candidate_metadata.get(
                            "accepted",
                            False,
                        )
                    ):
                        centers = anchor_centers
                        radii = anchor_radii
                        global_surface_fit_metadata = (
                            anchor_candidate_metadata
                        )
                        selected_solution = "front_anchored"
                global_surface_fit_metadata = {
                    **global_surface_fit_metadata,
                    "selected_solution": selected_solution,
                    "radius_curvature_policy": (
                        radius_curvature_policy
                    ),
                    "control_spacing_policy": control_spacing_policy,
                    "center_prior_cv": center_prior_cv,
                    "adaptive_control_spacing_policy": (
                        adaptive_control_spacing_policy
                    ),
                    "point_parameter_refinement": (
                        point_parameter_refinement
                    ),
                    "front_anchor_policy": {
                        "accepted": anchor_policy_accepted,
                        "minimum_sections": 8,
                        "minimum_fraction": float(
                            circle_global_front_anchor_min_fraction
                        ),
                        "maximum_ray_perp_p10": 0.4,
                        "requested_maximum_ray_perp_p10": float(
                            circle_global_front_anchor_max_ray_perp_p10
                        ),
                        "accepted_sections": int(
                            nonlinear_metadata["accepted_sections"]
                        ),
                        "accepted_fraction": float(
                            nonlinear_metadata["accepted_fraction"]
                        ),
                        "initial_ray_perp_p10": (
                            float(
                                nonlinear_metadata[
                                    "initial_ray_perp_p10"
                                ]
                            )
                            if nonlinear_metadata.get(
                                "initial_ray_perp_p10"
                            )
                            is not None
                            else None
                        ),
                    },
                    "unanchored_baseline": baseline_global_metadata,
                    "front_anchor_candidate": (
                        anchor_candidate_metadata
                    ),
                }

        post_global_endpoint_metadata: Dict[str, Any] = {
            "requested": bool(
                circle_endpoint_radius_regularization_blend > 0.0
            ),
            "enabled": False,
        }
        if chain_order is not None:
            post_global_endpoint_enabled = bool(
                circle_global_surface_fit
                and global_surface_fit_metadata.get(
                    "accepted",
                    False,
                )
                and metadata.get("direct_fit_accepted", False)
                and nonlinear_metadata.get(
                    "initial_ray_perp_p10"
                )
                is not None
            )
            pre_post_global_radii = radii.copy()
            radii, post_global_endpoint_metadata = (
                _regularize_skeleton_endpoint_radii(
                    skeleton_y=sy,
                    skeleton_x=sx,
                    radii=radii,
                    requested_blend=(
                        float(
                            circle_endpoint_radius_regularization_blend
                        )
                        if post_global_endpoint_enabled
                        else 0.0
                    ),
                    direct_fit_accepted=bool(
                        metadata.get("direct_fit_accepted", False)
                    ),
                    trusted_radii=initial_radii,
                    initial_ray_perp_p10=nonlinear_metadata.get(
                        "initial_ray_perp_p10"
                    ),
                    require_consistent_extrapolation=(
                        post_global_endpoint_enabled
                    ),
                )
            )
            center_shift_supported = bool(
                any(
                    diagnostic.get("consistent_extrapolation", False)
                    and diagnostic.get("erosion_supported", False)
                    for diagnostic in post_global_endpoint_metadata.get(
                        "endpoint_diagnostics",
                        [],
                    )
                )
            )
            (
                centers,
                post_global_center_shift_metadata,
            ) = _preserve_front_surface_after_radius_change(
                chain_order=chain_order,
                centers=centers,
                original_radii=pre_post_global_radii,
                adjusted_radii=radii,
                camera_pos=camera_pos,
                enabled=bool(
                    post_global_endpoint_enabled
                    and center_shift_supported
                ),
            )
            post_global_endpoint_metadata.update(
                {
                    "requested": bool(
                        circle_endpoint_radius_regularization_blend
                        > 0.0
                    ),
                    "enabled": post_global_endpoint_enabled,
                    "front_surface_center_shift_supported": (
                        center_shift_supported
                    ),
                    "front_surface_center_shift": (
                        post_global_center_shift_metadata
                    ),
                }
            )

        collapsed_endpoint_metadata: Dict[str, Any] = {
            "requested": bool(circle_global_surface_fit),
            "enabled": False,
        }
        if chain_order is not None:
            radii, collapsed_endpoint_metadata = (
                _restore_severely_collapsed_endpoint_radii(
                    skeleton_y=sy,
                    skeleton_x=sx,
                    fitted_radii=radii,
                    trusted_radii=initial_radii,
                    enabled=bool(
                        circle_global_surface_fit
                        and global_surface_fit_metadata.get(
                            "accepted",
                            False,
                        )
                        and metadata.get("direct_fit_accepted", False)
                    ),
                )
            )
            collapsed_endpoint_metadata[
                "direct_fit_accepted"
            ] = bool(
                        metadata.get("direct_fit_accepted", False)
            )

        axial_endpoint_metadata: Dict[str, Any] = {
            "requested": bool(circle_axial_endpoint_radius_repair),
            "enabled": False,
        }
        if chain_order is not None:
            radii, axial_endpoint_metadata = (
                _repair_axial_swept_endpoint_radii(
                    chain_order=chain_order,
                    centers=centers,
                    radii=radii,
                    camera_pos=camera_pos,
                    enabled=bool(
                        circle_axial_endpoint_radius_repair
                    ),
                )
            )

        fixed_centerline_radius_metadata: Dict[str, Any] = {
            "requested": bool(
                circle_fixed_centerline_radius_refinement
            ),
            "enabled": False,
        }
        if (
            chain_order is not None
            and surface_points_for_radius_refinement is not None
        ):
            fixed_centerline_enabled = bool(
                circle_fixed_centerline_radius_refinement
                and global_surface_fit_metadata.get("accepted", False)
                and global_surface_fit_metadata.get("selected_solution")
                == "unanchored_baseline"
            )
            (
                radii,
                fixed_centerline_radius_metadata,
            ) = _refine_swept_radii_from_fixed_centerline(
                chain_order=chain_order,
                centers=centers,
                radii=radii,
                surface_points=surface_points_for_radius_refinement,
                enabled=fixed_centerline_enabled,
                quantile=float(
                    circle_fixed_centerline_radius_quantile
                ),
                bandwidth_sections=float(
                    circle_fixed_centerline_radius_bandwidth_sections
                ),
                blend=float(circle_fixed_centerline_radius_blend),
                surface_point_fold_ids=(
                    surface_fold_ids_for_radius_refinement
                ),
                cross_validate=True,
                cross_validation_min_gain_m=1e-6,
            )
            fixed_centerline_radius_metadata.update(
                {
                    "policy_enabled": fixed_centerline_enabled,
                    "selected_solution": (
                        global_surface_fit_metadata.get(
                            "selected_solution"
                        )
                    ),
                }
            )

        visible_endpoint_metadata: Dict[str, Any] = {
            "requested": bool(circle_visible_endpoint_extension),
            "enabled": False,
        }
        degenerate_endpoint_curve_metadata: Dict[str, Any] = {
            "requested": bool(
                circle_degenerate_endpoint_curve_refinement
            ),
            "enabled": False,
        }
        if chain_order is not None:
            pre_visible_centers = centers.copy()
            pre_visible_radii = radii.copy()
            centers, visible_endpoint_metadata = (
                _extend_swept_chain_endpoints_to_visible_support(
                    chain_order=chain_order,
                    skeleton_y=sy,
                    skeleton_x=sx,
                    centers=centers,
                    radii=radii,
                    component_mask=component_mask,
                    world=world,
                    enabled=bool(circle_visible_endpoint_extension),
                    blend=float(
                        circle_visible_endpoint_extension_blend
                    ),
                    maximum_extension_radius_ratio=float(
                        circle_visible_endpoint_extension_max_radius_ratio
                    ),
                    preserve_original_endpoint_ring=bool(
                        circle_visible_endpoint_preserve_original_ring
                    ),
                )
            )
            unsupported_endpoint_indices = {
                int(diagnostic["endpoint_index"])
                for diagnostic in visible_endpoint_metadata.get(
                    "endpoint_diagnostics",
                    [],
                )
                if not bool(diagnostic.get("triggered"))
            }
            degenerate_curve_policy_enabled = bool(
                circle_degenerate_endpoint_curve_refinement
                and global_surface_fit_metadata.get("accepted", False)
                and global_surface_fit_metadata.get("selected_solution")
                == "front_anchored"
                and not bool(metadata.get("direct_fit_accepted", False))
            )
            (
                centers,
                degenerate_endpoint_curve_metadata,
            ) = _refine_degenerate_endpoint_centers_from_curve_ray(
                chain_order=chain_order,
                skeleton_y=sy,
                skeleton_x=sx,
                centers=centers,
                radii=radii,
                camera_pos=camera_pos,
                camera_quat_xyzw=camera_quat_xyzw,
                focal_length=float(focal_length),
                horizontal_aperture=float(horizontal_aperture),
                image_width=int(w),
                image_height=int(h),
                eligible_endpoint_indices=unsupported_endpoint_indices,
                enabled=degenerate_curve_policy_enabled,
                blend=float(circle_degenerate_endpoint_curve_blend),
            )
            degenerate_endpoint_curve_metadata.update(
                {
                    "policy_enabled": degenerate_curve_policy_enabled,
                    "selected_solution": (
                        global_surface_fit_metadata.get(
                            "selected_solution"
                        )
                    ),
                    "direct_fit_accepted": bool(
                        metadata.get("direct_fit_accepted", False)
                    ),
                }
            )
            swept_section_diagnostics.update(
                {
                    "pre_visible_centers_world": (
                        pre_visible_centers[chain_order].tolist()
                    ),
                    "pre_visible_radii_mm": (
                        pre_visible_radii[chain_order] * 1000.0
                    ).tolist(),
                    "final_centers_world": (
                        centers[chain_order].tolist()
                    ),
                    "final_radii_mm": (
                        radii[chain_order] * 1000.0
                    ).tolist(),
                }
            )

        polygonal_ring_metadata: Dict[str, Any] = {
            "requested": bool(circle_polygonal_ring_selection),
            "enabled": False,
            "accepted": False,
            "requested_ring_samples": int(swept_ring_samples),
            "effective_ring_samples": int(swept_ring_samples),
            "effective_ring_phase_rad": 0.0,
            "effective_radius_scale": 1.0,
        }
        effective_ring_samples = int(swept_ring_samples)
        effective_ring_phase = 0.0
        effective_ring_radius_scale = 1.0
        if (
            chain_order is not None
            and surface_points_for_radius_refinement is not None
            and surface_fold_ids_for_radius_refinement is not None
        ):
            (
                effective_ring_samples,
                effective_ring_phase,
                effective_ring_radius_scale,
                polygonal_ring_metadata,
            ) = _select_polygonal_swept_ring_model(
                chain_order=chain_order,
                centers=centers,
                radii=radii,
                surface_points=surface_points_for_radius_refinement,
                surface_point_fold_ids=(
                    surface_fold_ids_for_radius_refinement
                ),
                requested_ring_samples=int(swept_ring_samples),
                enabled=bool(
                    circle_polygonal_ring_selection
                    and requested_meshing_mode == "swept_surface"
                ),
            )
            if swept_section_diagnostics.get("available", False):
                swept_section_diagnostics.update(
                    {
                        "mesh_radii_mm": (
                            radii[chain_order]
                            * effective_ring_radius_scale
                            * 1000.0
                        ).tolist(),
                        "mesh_ring_samples": int(
                            effective_ring_samples
                        ),
                        "mesh_ring_phase_rad": float(
                            effective_ring_phase
                        ),
                    }
                )
        meshing_radii = radii * effective_ring_radius_scale
        polygonal_facet_metadata: Dict[str, Any] = {
            "requested": bool(circle_polygonal_facet_inversion),
            "enabled": False,
            "accepted": False,
        }
        if (
            chain_order is not None
            and bool(polygonal_ring_metadata.get("accepted", False))
            and surface_points_for_radius_refinement is not None
            and surface_fold_ids_for_radius_refinement is not None
        ):
            (
                fitted_ordered_centers,
                fitted_ordered_radii,
                polygonal_facet_metadata,
            ) = _refine_polygonal_swept_facets(
                ordered_centers=centers[chain_order],
                ordered_radii=meshing_radii[chain_order],
                surface_points=surface_points_for_radius_refinement,
                surface_point_fold_ids=(
                    surface_fold_ids_for_radius_refinement
                ),
                camera_pos=camera_pos,
                ring_samples=int(effective_ring_samples),
                vertex_phase_rad=float(effective_ring_phase),
                enabled=bool(
                    circle_polygonal_facet_inversion
                    and requested_meshing_mode == "swept_surface"
                ),
            )
            if bool(polygonal_facet_metadata.get("accepted", False)):
                centers = centers.copy()
                radii = radii.copy()
                meshing_radii = meshing_radii.copy()
                centers[chain_order] = fitted_ordered_centers
                meshing_radii[chain_order] = fitted_ordered_radii
                radii[chain_order] = (
                    fitted_ordered_radii
                    / max(float(effective_ring_radius_scale), 1e-12)
                )
                if swept_section_diagnostics.get("available", False):
                    swept_section_diagnostics.update(
                        {
                            "final_centers_world": (
                                fitted_ordered_centers.tolist()
                            ),
                            "mesh_radii_mm": (
                                fitted_ordered_radii * 1000.0
                            ).tolist(),
                        }
                    )

        if (
            requested_meshing_mode == "swept_surface"
            and bool(chain_metadata.get("eligible"))
        ):
            try:
                endpoint_extension_vectors = {
                    int(diagnostic["endpoint_index"]): np.asarray(
                        diagnostic["extension_vector_world"],
                        dtype=np.float64,
                    )
                    for diagnostic in visible_endpoint_metadata.get(
                        "endpoint_diagnostics",
                        [],
                    )
                    if bool(diagnostic.get("triggered"))
                    and bool(
                        visible_endpoint_metadata.get(
                            "preserve_original_endpoint_ring",
                            False,
                        )
                    )
                }
                swept_mesh, swept_graph_metadata = (
                    _skeleton_graph_swept_tube_mesh(
                        skeleton_y=sy,
                        skeleton_x=sx,
                        centers=centers,
                        radii=meshing_radii,
                        ring_samples=int(effective_ring_samples),
                        ring_phase_rad=float(effective_ring_phase),
                        endpoint_extension_vectors=(
                            endpoint_extension_vectors
                        ),
                    )
                )
            except ValueError as exc:
                swept_fallback_reason = str(exc)
            else:
                chain_metadata = swept_graph_metadata
                swept_meshes.append(swept_mesh)
                swept_chain_sections += int(len(sx))
        (
            graph_centers,
            graph_radii,
            graph_metadata,
        ) = _skeleton_graph_capsule_samples(
            skeleton_y=sy,
            skeleton_x=sx,
            centers=centers,
            radii=radii,
            voxel_m=voxel_m,
        )
        center_parts.append(graph_centers)
        radius_parts.append(graph_radii)
        graph_edge_count += int(graph_metadata["skeleton_graph_edges"])
        graph_interpolated_count += int(
            graph_metadata["interpolated_samples"]
        )
        skeleton_sizes.append(int(len(sx)))
        component_metadata[component] = {
            **metadata,
            "skeleton_points": int(len(sx)),
            "nonlinear_refinement": nonlinear_metadata,
            "chain_geometry_smoothing": chain_geometry_metadata,
            "global_surface_fit": global_surface_fit_metadata,
            "post_global_endpoint_radius_regularization": (
                post_global_endpoint_metadata
            ),
            "collapsed_endpoint_radius_restoration": (
                collapsed_endpoint_metadata
            ),
            "skeleton_spur_pruning": skeleton_spur_metadata,
            "axial_endpoint_radius_repair": axial_endpoint_metadata,
            "fixed_centerline_radius_refinement": (
                fixed_centerline_radius_metadata
            ),
            "visible_endpoint_extension": visible_endpoint_metadata,
            "degenerate_endpoint_curve_refinement": (
                degenerate_endpoint_curve_metadata
            ),
            "polygonal_ring_selection": polygonal_ring_metadata,
            "polygonal_facet_inversion": polygonal_facet_metadata,
            "swept_section_diagnostics": swept_section_diagnostics,
            "swept_chain": chain_metadata,
            **graph_metadata,
        }

    if not center_parts:
        raise ValueError("RGBD target has no circle-tube completion samples")
    centers = np.concatenate(center_parts, axis=0)
    radii = np.concatenate(radius_parts, axis=0)
    swept_surface_active = bool(
        requested_meshing_mode == "swept_surface"
        and len(swept_meshes) == len(center_parts)
    )
    if swept_surface_active:
        mesh = trimesh.util.concatenate(swept_meshes)
        mesh_components = mesh.split(only_watertight=False)
        nonwatertight_components = int(
            sum(
                not bool(component.is_watertight)
                for component in mesh_components
            )
        )
        return mesh, {
            "build": RGBD_ONLY_MESH_BUILD,
            "method": "local_rgbd_swept_tube_completion",
            "tube_meshing_mode": "swept_surface",
            "tube_meshing_fallback_reason": None,
            "selected_rgbd_points": int(swept_chain_sections),
            "rgbd_components": int(len(component_sizes)),
            "seed_component": int(seed_component),
            "component_policy": policy,
            "selected_components": sorted(
                int(component) for component in selected_components
            ),
            "selected_component_count": int(len(selected_components)),
            "rgbd_component_pixels_desc": sorted(
                [int(value) for value in component_sizes],
                reverse=True,
            ),
            "rgbd_component_skeleton_points": skeleton_sizes,
            "circle_tube_components": {
                str(component): metadata
                for component, metadata in component_metadata.items()
            },
            "voxel_mm": float(voxel_mm),
            "grid_shape": None,
            "occupied_voxels": None,
            "occupancy_component_count": int(len(mesh_components)),
            "nonwatertight_occupancy_components": (
                nonwatertight_components
            ),
            "radius_min_mm": float(radii.min() * 1000.0),
            "radius_median_mm": float(np.median(radii) * 1000.0),
            "radius_max_mm": float(radii.max() * 1000.0),
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_faces": int(len(mesh.faces)),
            "mesh_watertight": bool(mesh.is_watertight),
            "mesh_volume_cm3": float(abs(mesh.volume) * 1e6),
            "parameters": {
                "tube_meshing_mode": "swept_surface",
                "ring_samples": int(swept_ring_samples),
                "circle_polygonal_ring_selection": bool(
                    circle_polygonal_ring_selection
                ),
                "circle_polygonal_facet_inversion": bool(
                    circle_polygonal_facet_inversion
                ),
                "local_cylinder_nonlinear_refine": bool(
                    local_cylinder_nonlinear_refine
                ),
                "circle_graph_radius_smoothing_gcv": bool(
                    circle_graph_radius_smoothing_gcv
                ),
                "circle_chain_geometry_smoothing_gcv": bool(
                    circle_chain_geometry_smoothing_gcv
                ),
                "circle_global_surface_fit": bool(
                    circle_global_surface_fit
                ),
                "circle_soft_direct_radius_blend": bool(
                    circle_soft_direct_radius_blend
                ),
                "circle_soft_direct_radius_blend_max": float(
                    circle_soft_direct_radius_blend_max
                ),
                "circle_adaptive_direct_center_blend": bool(
                    circle_adaptive_direct_center_blend
                ),
                "circle_global_radius_curvature_sigma": float(
                    circle_global_radius_curvature_sigma
                ),
                "circle_global_sections_per_control": float(
                    circle_global_sections_per_control
                ),
                "circle_global_center_prior_sigma_mm": float(
                    circle_global_center_prior_sigma_mm
                ),
                "circle_global_adaptive_sections_per_control": bool(
                    circle_global_adaptive_sections_per_control
                ),
                "circle_global_control_cv": bool(
                    circle_global_control_cv
                ),
                "circle_global_center_prior_cv": bool(
                    circle_global_center_prior_cv
                ),
                "circle_global_point_parameter_refinement_sections": int(
                    circle_global_point_parameter_refinement_sections
                ),
                "circle_global_point_parameter_refinement_blend": float(
                    circle_global_point_parameter_refinement_blend
                ),
                "circle_global_adaptive_point_parameter_refinement": bool(
                    circle_global_adaptive_point_parameter_refinement
                ),
                "circle_global_front_anchor_sigma_mm": float(
                    circle_global_front_anchor_sigma_mm
                ),
                "circle_global_front_anchor_min_fraction": float(
                    circle_global_front_anchor_min_fraction
                ),
                "circle_global_front_anchor_max_ray_perp_p10": float(
                    circle_global_front_anchor_max_ray_perp_p10
                ),
                "circle_fixed_centerline_radius_refinement": bool(
                    circle_fixed_centerline_radius_refinement
                ),
                "circle_fixed_centerline_radius_quantile": float(
                    circle_fixed_centerline_radius_quantile
                ),
                "circle_fixed_centerline_radius_bandwidth_sections": float(
                    circle_fixed_centerline_radius_bandwidth_sections
                ),
                "circle_fixed_centerline_radius_blend": float(
                    circle_fixed_centerline_radius_blend
                ),
                "circle_degenerate_endpoint_curve_refinement": bool(
                    circle_degenerate_endpoint_curve_refinement
                ),
                "circle_degenerate_endpoint_curve_blend": float(
                    circle_degenerate_endpoint_curve_blend
                ),
                "circle_prune_short_skeleton_spurs": bool(
                    circle_prune_short_skeleton_spurs
                ),
                "circle_axial_endpoint_radius_repair": bool(
                    circle_axial_endpoint_radius_repair
                ),
                "circle_visible_endpoint_extension": bool(
                    circle_visible_endpoint_extension
                ),
                "circle_visible_endpoint_extension_blend": float(
                    circle_visible_endpoint_extension_blend
                ),
                "circle_visible_endpoint_extension_max_radius_ratio": float(
                    circle_visible_endpoint_extension_max_radius_ratio
                ),
                "circle_visible_endpoint_preserve_original_ring": bool(
                    circle_visible_endpoint_preserve_original_ring
                ),
            },
            "forbidden_inputs_used": [],
        }
    effective_meshing_mode = "voxel_union"
    margin = float(radii.max()) + 2.0 * voxel_m
    lower = np.floor((centers.min(axis=0) - margin) / voxel_m) * voxel_m
    upper = np.ceil((centers.max(axis=0) + margin) / voxel_m) * voxel_m
    shape = np.rint((upper - lower) / voxel_m).astype(np.int64) + 1
    if np.any(shape <= 2) or int(np.prod(shape)) > 80_000_000:
        raise ValueError(f"invalid circle-tube voxel grid: {shape.tolist()}")
    occupancy = np.zeros(tuple(int(value) for value in shape), dtype=bool)

    for center, radius in zip(centers, radii):
        center_index = np.rint((center - lower) / voxel_m).astype(
            np.int64
        )
        reach = int(math.ceil(float(radius) / voxel_m)) + 1
        axes = [
            np.arange(
                max(0, int(center_index[axis]) - reach),
                min(
                    int(shape[axis]),
                    int(center_index[axis]) + reach + 1,
                ),
            )
            for axis in range(3)
        ]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        local_points = (
            np.stack((gx, gy, gz), axis=-1).astype(np.float64) * voxel_m
            + lower
        )
        hit = (
            np.einsum(
                "...i,...i->...",
                local_points - center,
                local_points - center,
            )
            <= float(radius) ** 2
        )
        occupancy[np.ix_(*axes)] |= hit

    if int(occupancy.sum()) < 8:
        raise ValueError(
            f"circle-tube occupancy too small: "
            f"{int(occupancy.sum())} voxels"
        )
    (
        mesh,
        occupancy_component_count,
        nonwatertight_occupancy_components,
    ) = _mesh_from_discrete_occupancy_components(
        occupancy,
        lower=lower,
        voxel_m=voxel_m,
    )
    return mesh, {
        "build": RGBD_ONLY_MESH_BUILD,
        "method": "local_rgbd_circle_tube_completion",
        "tube_meshing_mode": effective_meshing_mode,
        "tube_meshing_fallback_reason": (
            swept_fallback_reason
            if requested_meshing_mode == "swept_surface"
            else None
        ),
        "selected_rgbd_points": int(len(centers)),
        "rgbd_components": int(len(component_sizes)),
        "seed_component": int(seed_component),
        "component_policy": policy,
        "selected_components": sorted(
            int(component) for component in selected_components
        ),
        "selected_component_count": int(len(selected_components)),
        "rgbd_component_pixels_desc": sorted(
            [int(value) for value in component_sizes],
            reverse=True,
        ),
        "rgbd_component_skeleton_points": skeleton_sizes,
        "skeleton_graph_edges": int(graph_edge_count),
        "skeleton_graph_interpolated_samples": int(
            graph_interpolated_count
        ),
        "circle_tube_components": {
            str(component): metadata
            for component, metadata in component_metadata.items()
        },
        "voxel_mm": float(voxel_mm),
        "grid_shape": [int(value) for value in shape],
        "occupied_voxels": int(occupancy.sum()),
        "occupancy_component_count": int(occupancy_component_count),
        "nonwatertight_occupancy_components": int(
            nonwatertight_occupancy_components
        ),
        "radius_min_mm": float(radii.min() * 1000.0),
        "radius_median_mm": float(np.median(radii) * 1000.0),
        "radius_max_mm": float(radii.max() * 1000.0),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_volume_cm3": float(abs(mesh.volume) * 1e6),
        "parameters": {
                "circle_radius_scale": float(circle_radius_scale),
                "circle_radius_bias_px": float(circle_radius_bias_px),
                "circle_adaptive_radius_bias": bool(
                    circle_adaptive_radius_bias
                ),
                "circle_radius_min_mm": float(circle_radius_min_mm),
            "circle_radius_max_mm": float(circle_radius_max_mm),
            "circle_center_fit_blend": float(circle_center_fit_blend),
            "circle_center_offset_scale": float(
                circle_center_offset_scale
            ),
            "circle_tangent_neighborhood_px": float(
                circle_tangent_neighborhood_px
            ),
            "curvature_radius_blend": float(curvature_radius_blend),
            "curvature_scale_min": float(curvature_scale_min),
            "curvature_scale_max": float(curvature_scale_max),
            "curvature_fit_max_residual_mm": float(
                curvature_fit_max_residual_mm
            ),
            "curvature_min_valid_fits": int(
                curvature_min_valid_fits
            ),
            "curvature_smoothing_px": float(curvature_smoothing_px),
            "curvature_local_shrinkage": float(
                curvature_local_shrinkage
            ),
            "circle_direct_radius_blend": float(
                circle_direct_radius_blend
            ),
            "circle_direct_center_blend": float(
                circle_direct_center_blend
            ),
            "circle_front_anchored_radius_blend": float(
                circle_front_anchored_radius_blend
            ),
            "circle_endpoint_radius_regularization_blend": float(
                circle_endpoint_radius_regularization_blend
            ),
            "circle_graph_radius_smoothing_gcv": bool(
                circle_graph_radius_smoothing_gcv
            ),
            "curvature_max_center_toward_camera_ratio": float(
                curvature_max_center_toward_camera_ratio
            ),
        },
        "forbidden_inputs_used": [],
    }


def _mesh_from_support_height_field(
    extraction: DepthMeshReconstruction,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    focal_px: float,
    voxel_mm: float,
    footprint_scale: float,
    footprint_min_mm: float,
    lateral_expand_mm: float,
    closing_radius_mm: float,
    backfill_scale: float,
    backfill_step_mm: float,
    height_scale: float,
    height_offset_mm: float,
    base_extension_mm: float,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    selected = np.asarray(extraction.selected_mask, dtype=bool)
    sy, sx = np.nonzero(selected)
    if len(sx) < 3:
        raise ValueError(
            f"clicked RGBD target has too few samples: {len(sx)}"
        )

    voxel_m = float(voxel_mm) / 1000.0
    footprint_min_m = float(footprint_min_mm) / 1000.0
    lateral_expand_m = float(lateral_expand_mm) / 1000.0
    closing_radius_m = float(closing_radius_mm) / 1000.0
    backfill_step_m = max(
        float(backfill_step_mm) / 1000.0,
        voxel_m,
    )
    height_offset_m = float(height_offset_mm) / 1000.0
    base_extension_m = max(0.0, float(base_extension_mm) / 1000.0)

    points = np.asarray(
        extraction.selected_surface_points,
        dtype=np.float64,
    )
    heights = np.clip(
        np.asarray(extraction.selected_heights_m, dtype=np.float64)
        * float(height_scale)
        + height_offset_m,
        voxel_m * 0.5,
        None,
    )
    selected_depth = np.asarray(depth, dtype=np.float64)[sy, sx]
    if len(points) != len(selected_depth):
        raise ValueError(
            f"selected point/depth mismatch: {len(points)} vs "
            f"{len(selected_depth)}"
        )

    plane = extraction.plane
    projected = plane.project_to_plane(points)
    relative = projected - plane.origin.reshape(1, 3)
    sample_xy = np.column_stack(
        (
            relative @ plane.basis_x,
            relative @ plane.basis_y,
        )
    )
    sample_radius = np.maximum(
        selected_depth / max(float(focal_px), 1e-12)
        * float(footprint_scale),
        footprint_min_m,
    )
    sample_radius += lateral_expand_m

    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    view_on_plane = points - camera.reshape(1, 3)
    view_on_plane -= (
        view_on_plane @ plane.normal
    )[:, None] * plane.normal.reshape(1, 3)
    view_norm = np.linalg.norm(view_on_plane, axis=1)
    view_direction_world = np.zeros_like(view_on_plane)
    valid_direction = view_norm > 1e-9
    view_direction_world[valid_direction] = (
        view_on_plane[valid_direction]
        / view_norm[valid_direction, None]
    )
    view_direction_xy = np.column_stack(
        (
            view_direction_world @ plane.basis_x,
            view_direction_world @ plane.basis_y,
        )
    )

    backfill_distance = np.maximum(
        heights * float(backfill_scale),
        0.0,
    )
    margin = (
        float(sample_radius.max())
        + float(backfill_distance.max())
        + 2.0 * voxel_m
    )
    lower_xy = (
        np.floor((sample_xy.min(axis=0) - margin) / voxel_m)
        * voxel_m
    )
    upper_xy = (
        np.ceil((sample_xy.max(axis=0) + margin) / voxel_m)
        * voxel_m
    )
    xy_shape = (
        np.rint((upper_xy - lower_xy) / voxel_m).astype(np.int64) + 1
    )
    if np.any(xy_shape <= 2) or int(np.prod(xy_shape)) > 40_000_000:
        raise ValueError(
            f"invalid RGBD height-field grid: {xy_shape.tolist()}"
        )

    top_height = np.zeros(tuple(int(v) for v in xy_shape), dtype=np.float32)
    footprint = np.zeros_like(top_height, dtype=bool)
    splat_count = 0
    for xy, direction, height, radius, distance in zip(
        sample_xy,
        view_direction_xy,
        heights,
        sample_radius,
        backfill_distance,
    ):
        steps = max(1, int(math.ceil(float(distance) / backfill_step_m)) + 1)
        offsets = np.linspace(0.0, float(distance), steps)
        for offset in offsets:
            shifted_xy = xy + direction * offset
            center_index = np.rint(
                (shifted_xy - lower_xy) / voxel_m
            ).astype(np.int64)
            reach = int(math.ceil(float(radius) / voxel_m)) + 1
            axes = [
                np.arange(
                    max(0, int(center_index[axis]) - reach),
                    min(
                        int(xy_shape[axis]),
                        int(center_index[axis]) + reach + 1,
                    ),
                )
                for axis in range(2)
            ]
            gx, gy = np.meshgrid(*axes, indexing="ij")
            local_xy = (
                np.stack((gx, gy), axis=-1).astype(np.float64) * voxel_m
                + lower_xy
            )
            hit = (
                np.einsum(
                    "...i,...i->...",
                    local_xy - shifted_xy,
                    local_xy - shifted_xy,
                )
                <= float(radius) ** 2
            )
            target = np.ix_(*axes)
            local_top = top_height[target]
            local_top[hit] = np.maximum(local_top[hit], float(height))
            top_height[target] = local_top
            local_footprint = footprint[target]
            local_footprint |= hit
            footprint[target] = local_footprint
            splat_count += 1

    closing_cells = int(math.ceil(closing_radius_m / voxel_m))
    if closing_cells > 0:
        closed = ndimage.binary_closing(
            footprint,
            structure=_disk_structure(closing_cells),
        )
        top_height = _fill_new_footprint_cells(
            top_height,
            footprint,
            closed,
        )
        footprint = closed

    lower_z = -base_extension_m
    upper_z = float(top_height.max()) + 2.0 * voxel_m
    z_levels = max(
        3,
        int(math.ceil((upper_z - lower_z) / voxel_m)) + 1,
    )
    z_values = lower_z + np.arange(z_levels, dtype=np.float64) * voxel_m
    occupancy = (
        footprint[:, :, None]
        & (z_values.reshape(1, 1, -1) <= top_height[:, :, None])
    )
    if int(occupancy.sum()) < 8:
        raise ValueError(
            f"RGBD height-field occupancy too small: "
            f"{int(occupancy.sum())} voxels"
        )

    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
        occupancy,
        pitch=voxel_m,
    )
    local = np.asarray(mesh.vertices, dtype=np.float64)
    local[:, 0] += float(lower_xy[0])
    local[:, 1] += float(lower_xy[1])
    local[:, 2] += float(lower_z)
    mesh.vertices = (
        plane.origin.reshape(1, 3)
        + local[:, 0:1] * plane.basis_x.reshape(1, 3)
        + local[:, 1:2] * plane.basis_y.reshape(1, 3)
        + local[:, 2:3] * plane.normal.reshape(1, 3)
    )
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    return mesh, {
        "build": RGBD_ONLY_MESH_BUILD,
        "method": "support_plane_height_field",
        "selected_rgbd_points": int(len(points)),
        "voxel_mm": float(voxel_mm),
        "grid_shape": [
            int(xy_shape[0]),
            int(xy_shape[1]),
            int(z_levels),
        ],
        "occupied_voxels": int(occupancy.sum()),
        "footprint_cells": int(footprint.sum()),
        "splat_count": int(splat_count),
        "height_min_mm": float(heights.min() * 1000.0),
        "height_median_mm": float(np.median(heights) * 1000.0),
        "height_max_mm": float(heights.max() * 1000.0),
        "sample_radius_min_mm": float(sample_radius.min() * 1000.0),
        "sample_radius_median_mm": float(
            np.median(sample_radius) * 1000.0
        ),
        "sample_radius_max_mm": float(sample_radius.max() * 1000.0),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_volume_cm3": float(abs(mesh.volume) * 1e6),
        "parameters": {
            "footprint_scale": float(footprint_scale),
            "footprint_min_mm": float(footprint_min_mm),
            "lateral_expand_mm": float(lateral_expand_mm),
            "closing_radius_mm": float(closing_radius_mm),
            "backfill_scale": float(backfill_scale),
            "backfill_step_mm": float(backfill_step_mm),
            "height_scale": float(height_scale),
            "height_offset_mm": float(height_offset_mm),
            "base_extension_mm": float(base_extension_mm),
        },
        "forbidden_inputs_used": [],
    }


def _mesh_from_ray_chord_volume(
    extraction: DepthMeshReconstruction,
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    voxel_mm: float,
    footprint_scale: float,
    footprint_min_mm: float,
    chord_scale: float,
    chord_min_mm: float,
    chord_max_mm: float,
    surface_offset_mm: float,
    radius_gain: float,
    radius_bias_px: float,
    inset_bias_px: float,
    click_u: int,
    click_v: int,
    component_policy: str,
    component_gap_3d_mm: float,
    component_gap_px: float,
    component_color_delta: float,
    resample_component_edges: bool,
    component_edge_spacing_scale: float,
    bridge_components: bool,
    bridge_selection_mode: str,
    bridge_contact_margin_mm: float,
    bridge_contact_voxel_margin_scale: float,
    bridge_geometry_mode: str,
    bridge_axis_radius_scale: float,
    bridge_tangent_alignment: float,
    bridge_tangent_neighborhood_px: float,
    bridge_chord_scale: float,
    bridge_footprint_scale: float,
    curvature_radius_blend: float,
    curvature_scale_min: float,
    curvature_scale_max: float,
    curvature_fit_max_residual_mm: float,
    curvature_min_valid_fits: int,
    curvature_full_blend_skeleton_points: int,
    curvature_local_scales: bool,
    curvature_smoothing_px: float,
    curvature_local_shrinkage: float,
    curvature_max_center_toward_camera_ratio: float,
    chord_model: str,
    local_cylinder_chord_blend: float,
    local_cylinder_require_fit: bool,
    local_cylinder_max_radial_error_mm: float,
    local_cylinder_max_radial_error_ratio: float,
    local_cylinder_min_ray_perp: float,
    local_cylinder_max_chord_ratio: float,
    local_cylinder_axial_p10_threshold: float,
    local_cylinder_axial_chord_blend: float,
    local_cylinder_axial_chord_scale: float,
    local_cylinder_axial_require_single_component: bool,
    local_cylinder_nonlinear_refine: bool,
    local_cylinder_nonlinear_local_accepts: bool,
    circle_radius_scale: float,
    circle_radius_bias_px: float,
    circle_radius_min_mm: float,
    circle_radius_max_mm: float,
    circle_direct_radius_blend: float,
    circle_direct_center_blend: float,
    circle_front_anchored_radius_blend: float,
    circle_endpoint_radius_regularization_blend: float,
    circle_graph_radius_smoothing_gcv: bool,
    circle_center_fit_blend: float,
    circle_center_offset_scale: float,
    circle_tangent_neighborhood_px: float,
    ray_primitive_mode: str,
    capped_cylinder_axial_margin_radius_scale: float,
    ray_meshing_mode: str,
    binary_smoothing_sigma_voxels: float,
    external_chord_map: np.ndarray | None = None,
    external_chord_confidence_map: np.ndarray | None = None,
    external_chord_blend: float = 0.0,
) -> tuple[trimesh.Trimesh, Dict[str, Any]]:
    """Complete silhouette pixels along the camera ray using circular chords."""
    selected = np.asarray(extraction.selected_mask, dtype=bool)
    color = np.asarray(rgb, dtype=np.float64)
    metric_depth = np.asarray(depth, dtype=np.float64)
    h, w = metric_depth.shape[:2]
    focal_px = (
        float(focal_length)
        / float(horizontal_aperture)
        * float(w)
    )
    world, _valid = backproject_depth_world(
        metric_depth,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
    )
    labels, yy, xx, component_for_node = _depth_component_labels(
        selected,
        world_points=world,
        rgb=color,
        depth=metric_depth,
        focal_px=focal_px,
    )
    component_sizes = np.bincount(component_for_node).astype(np.int64)
    if len(component_sizes) == 0:
        raise ValueError("RGBD target has no 3D-connected components")

    nearest_click_node = int(
        np.argmin(
            (xx - int(click_u)) ** 2 + (yy - int(click_v)) ** 2
        )
    )
    seed_component = int(component_for_node[nearest_click_node])
    policy = str(component_policy).strip().lower()
    if policy == "all":
        selected_components = set(range(len(component_sizes)))
    elif policy == "seed_only":
        selected_components = {seed_component}
    elif policy == "seed_direct":
        selected_components = {seed_component}
        seed_mask = component_for_node == seed_component
        seed_y = yy[seed_mask]
        seed_x = xx[seed_mask]
        seed_world = world[seed_y, seed_x]
        seed_tree = cKDTree(seed_world)
        for component in range(len(component_sizes)):
            if component == seed_component:
                continue
            candidate_mask = component_for_node == component
            candidate_y = yy[candidate_mask]
            candidate_x = xx[candidate_mask]
            candidate_world = world[candidate_y, candidate_x]
            distance, seed_index = seed_tree.query(candidate_world, k=1)
            best = int(np.argmin(distance))
            paired_seed = int(seed_index[best])
            uv_gap = float(
                np.linalg.norm(
                    np.array(
                        [candidate_x[best], candidate_y[best]],
                        dtype=np.float64,
                    )
                    - np.array(
                        [seed_x[paired_seed], seed_y[paired_seed]],
                        dtype=np.float64,
                    )
                )
            )
            color_gap = float(
                np.linalg.norm(
                    color[candidate_y[best], candidate_x[best]]
                    - color[seed_y[paired_seed], seed_x[paired_seed]]
                )
            )
            if (
                float(distance[best])
                <= float(component_gap_3d_mm) / 1000.0
                and uv_gap <= float(component_gap_px)
                and color_gap <= float(component_color_delta)
            ):
                selected_components.add(component)
    else:
        raise ValueError(
            f"unsupported component_policy={component_policy!r}; expected "
            "'all', 'seed_only', or 'seed_direct'"
        )

    surface_parts = []
    chord_parts = []
    radius_parts = []
    raw_chord_parts = []
    skeleton_sizes = []
    skeleton_uv_by_component: Dict[int, np.ndarray] = {}
    chord_map = np.full(metric_depth.shape, np.nan, dtype=np.float64)
    footprint_radius_map = np.full(
        metric_depth.shape,
        np.nan,
        dtype=np.float64,
    )
    tube_radius_map = np.full(
        metric_depth.shape,
        np.nan,
        dtype=np.float64,
    )
    tube_axis_map = np.full(
        (*metric_depth.shape, 3),
        np.nan,
        dtype=np.float64,
    )
    curvature_components: Dict[int, Dict[str, Any]] = {}
    local_cylinder_components: Dict[int, Dict[str, Any]] = {}
    external_chord_values: list[np.ndarray] = []
    external_confidence_values: list[np.ndarray] = []
    external_map = (
        None
        if external_chord_map is None
        else np.asarray(external_chord_map, dtype=np.float64)
    )
    if external_map is not None and external_map.shape != metric_depth.shape:
        raise ValueError(
            "external_chord_map must match the depth image shape"
        )
    external_confidence = (
        None
        if external_chord_confidence_map is None
        else np.asarray(
            external_chord_confidence_map,
            dtype=np.float64,
        )
    )
    if (
        external_confidence is not None
        and external_confidence.shape != metric_depth.shape
    ):
        raise ValueError(
            "external_chord_confidence_map must match the depth image shape"
        )
    if external_confidence is not None:
        finite_confidence = external_confidence[
            np.isfinite(external_confidence)
        ]
        if np.any(
            (finite_confidence < 0.0) | (finite_confidence > 1.0)
        ):
            raise ValueError(
                "external_chord_confidence_map must be in [0, 1]"
            )
    external_blend = float(external_chord_blend)
    if not 0.0 <= external_blend <= 1.0:
        raise ValueError("external_chord_blend must be in [0, 1]")
    selected_chord_model = str(chord_model).strip().lower()
    if selected_chord_model not in {"silhouette", "local_cylinder"}:
        raise ValueError(
            f"unsupported chord_model={chord_model!r}; expected "
            "'silhouette' or 'local_cylinder'"
        )
    for component in range(len(component_sizes)):
        if component not in selected_components:
            continue
        component_mask = labels == component
        cy, cx = np.nonzero(component_mask)
        if len(cx) == 0:
            continue
        skeleton = skeletonize(component_mask)
        sy, sx = np.nonzero(skeleton)
        if len(sx) == 0:
            continue
        skeleton_uv_by_component[component] = np.column_stack((sx, sy))

        distance_to_edge = ndimage.distance_transform_edt(component_mask)
        skeleton_radius_px = distance_to_edge[sy, sx]
        nearest_skeleton = cKDTree(
            np.column_stack((sx, sy)).astype(np.float64)
        ).query(
            np.column_stack((cx, cy)).astype(np.float64),
            k=1,
        )[1]
        local_radius_px = skeleton_radius_px[nearest_skeleton]
        raw_inset_px = np.minimum(
            distance_to_edge[cy, cx],
            local_radius_px,
        )
        raw_chord_px = 2.0 * np.sqrt(
            np.maximum(
                2.0 * local_radius_px * raw_inset_px
                - raw_inset_px * raw_inset_px,
                0.0,
            )
        )
        effective_radius_px = np.maximum(
            local_radius_px * float(radius_gain) - float(radius_bias_px),
            0.25,
        )
        inset_px = np.clip(
            distance_to_edge[cy, cx] - float(inset_bias_px),
            0.25,
            effective_radius_px,
        )
        chord_px = 2.0 * np.sqrt(
            np.maximum(
                2.0 * effective_radius_px * inset_px
                - inset_px * inset_px,
                0.0,
            )
        )
        source_depth = metric_depth[cy, cx]
        meters_per_pixel = source_depth / max(focal_px, 1e-12)
        raw_chord = raw_chord_px * meters_per_pixel
        corrected_chord = chord_px * meters_per_pixel
        (
            curvature_scale,
            local_curvature_scale,
            curvature_metadata,
        ) = (
            _component_curvature_radius_scale(
                component_mask=component_mask,
                skeleton_y=sy,
                skeleton_x=sx,
                world=world,
                depth=metric_depth,
                camera_pos=camera_pos,
                focal_px=focal_px,
                fit_max_residual_m=(
                    float(curvature_fit_max_residual_mm) / 1000.0
                ),
                scale_min=float(curvature_scale_min),
                scale_max=float(curvature_scale_max),
                min_valid_fits=int(curvature_min_valid_fits),
                smoothing_px=float(curvature_smoothing_px),
                local_shrinkage=float(curvature_local_shrinkage),
                max_center_toward_camera_ratio=float(
                    curvature_max_center_toward_camera_ratio
                ),
            )
        )
        if bool(curvature_local_scales):
            length_confidence = 1.0
            adaptive_blend = float(curvature_radius_blend)
            applied_curvature_scale = (
                1.0
                + adaptive_blend
                * (
                    local_curvature_scale[nearest_skeleton]
                    - 1.0
                )
            )
        else:
            length_confidence = min(
                1.0,
                float(curvature_full_blend_skeleton_points)
                / max(float(len(sx)), 1.0),
            )
            adaptive_blend = (
                float(curvature_radius_blend) * length_confidence
            )
            applied_curvature_scale = (
                1.0
                + adaptive_blend
                * (float(curvature_scale) - 1.0)
            )
        corrected_chord *= applied_curvature_scale
        component_chord_scale = float(chord_scale)
        curvature_components[component] = {
            **curvature_metadata,
            "skeleton_points": int(len(sx)),
            "length_confidence": float(length_confidence),
            "adaptive_blend": float(adaptive_blend),
            "applied_scale_min": float(
                np.min(applied_curvature_scale)
            ),
            "applied_scale_median": float(
                np.median(applied_curvature_scale)
            ),
            "applied_scale_max": float(
                np.max(applied_curvature_scale)
            ),
        }
        if selected_chord_model == "local_cylinder":
            (
                tube_centers,
                tube_radii,
                tube_metadata,
            ) = _component_circle_tube_sections(
                component_mask=component_mask,
                skeleton_y=sy,
                skeleton_x=sx,
                world=world,
                depth=metric_depth,
                camera_pos=camera_pos,
                focal_px=focal_px,
                radius_scale=float(circle_radius_scale),
                radius_bias_px=float(circle_radius_bias_px),
                radius_min_m=float(circle_radius_min_mm) / 1000.0,
                radius_max_m=float(circle_radius_max_mm) / 1000.0,
                fit_max_residual_m=(
                    float(curvature_fit_max_residual_mm) / 1000.0
                ),
                scale_min=float(curvature_scale_min),
                scale_max=float(curvature_scale_max),
                min_valid_fits=int(curvature_min_valid_fits),
                smoothing_px=float(curvature_smoothing_px),
                local_shrinkage=float(curvature_local_shrinkage),
                radius_blend=float(curvature_radius_blend),
                direct_radius_blend=float(
                    circle_direct_radius_blend
                ),
                soft_direct_radius_blend=True,
                soft_direct_radius_blend_max=0.5,
                direct_center_blend=float(
                    circle_direct_center_blend
                ),
                front_anchored_radius_blend=float(
                    circle_front_anchored_radius_blend
                ),
                endpoint_radius_regularization_blend=float(
                    circle_endpoint_radius_regularization_blend
                ),
                graph_radius_smoothing_gcv=bool(
                    circle_graph_radius_smoothing_gcv
                ),
                center_fit_blend=float(circle_center_fit_blend),
                center_offset_scale=float(circle_center_offset_scale),
                tangent_neighborhood_px=float(
                    circle_tangent_neighborhood_px
                ),
                max_center_toward_camera_ratio=float(
                    curvature_max_center_toward_camera_ratio
                ),
            )
            if bool(local_cylinder_nonlinear_refine):
                initial_tube_centers = tube_centers.copy()
                initial_tube_radii = tube_radii.copy()
                (
                    tube_centers,
                    tube_radii,
                    tube_tangents,
                    nonlinear_metadata,
                ) = _refine_local_cylinder_sections_nonlinear(
                    component_mask=component_mask,
                    skeleton_y=sy,
                    skeleton_x=sx,
                    world=world,
                    camera_pos=camera_pos,
                    initial_centers=tube_centers,
                    initial_radii=tube_radii,
                    tangent_neighborhood_px=float(
                        circle_tangent_neighborhood_px
                    ),
                )
                initial_ray_perp_p10 = nonlinear_metadata[
                    "initial_ray_perp_p10"
                ]
                well_conditioned_fragmented = bool(
                    len(selected_components) > 1
                    and initial_ray_perp_p10 is not None
                    and float(initial_ray_perp_p10) >= 0.65
                    and int(
                        nonlinear_metadata["accepted_sections"]
                    )
                    >= 8
                )
                minimum_accepted_fraction = (
                    0.25 if well_conditioned_fragmented else 0.5
                )
                nonlinear_applied = (
                    _accept_nonlinear_refinement_result(
                        accepted_sections=int(
                            nonlinear_metadata["accepted_sections"]
                        ),
                        accepted_fraction=float(
                            nonlinear_metadata["accepted_fraction"]
                        ),
                        minimum_component_fraction=float(
                            minimum_accepted_fraction
                        ),
                        local_accepted_sections=bool(
                            local_cylinder_nonlinear_local_accepts
                        ),
                    )
                )
                if not nonlinear_applied:
                    tube_centers = initial_tube_centers
                    tube_radii = initial_tube_radii
                    tube_tangents = _skeleton_world_tangents(
                        np.column_stack((sx, sy)),
                        np.asarray(
                            world[sy, sx],
                            dtype=np.float64,
                        ),
                        neighborhood_px=float(
                            circle_tangent_neighborhood_px
                        ),
                    )
                nonlinear_metadata.update(
                    {
                        "minimum_component_accepted_fraction": (
                            minimum_accepted_fraction
                        ),
                        "application_mode": (
                            "local_accepted_sections"
                            if bool(
                                local_cylinder_nonlinear_local_accepts
                            )
                            else "component_fraction_gate"
                        ),
                        "adaptive_low_fraction_gate": (
                            well_conditioned_fragmented
                        ),
                        "adaptive_min_ray_perp_p10": 0.65,
                        "adaptive_min_accepted_sections": 8,
                        "applied_to_component": nonlinear_applied,
                        "tangent_applied_downstream": nonlinear_applied,
                    }
                )
            else:
                tube_tangents = _skeleton_world_tangents(
                    np.column_stack((sx, sy)),
                    np.asarray(
                        world[sy, sx],
                        dtype=np.float64,
                    ),
                    neighborhood_px=float(
                        circle_tangent_neighborhood_px
                    ),
                )
                nonlinear_metadata = {
                    "enabled": False,
                    "attempted_sections": 0,
                    "accepted_sections": 0,
                    "accepted_fraction": 0.0,
                    "minimum_component_accepted_fraction": 0.5,
                    "application_mode": (
                        "local_accepted_sections"
                        if bool(local_cylinder_nonlinear_local_accepts)
                        else "component_fraction_gate"
                    ),
                    "applied_to_component": False,
                    "tangent_applied_downstream": False,
                }
            surface_points = np.asarray(
                world[cy, cx],
                dtype=np.float64,
            )
            camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
            pixel_ray = surface_points - camera.reshape(1, 3)
            pixel_ray /= (
                np.linalg.norm(pixel_ray, axis=1, keepdims=True) + 1e-12
            )
            local_center = tube_centers[nearest_skeleton]
            local_radius = tube_radii[nearest_skeleton]
            local_tangent = tube_tangents[nearest_skeleton]
            tube_radius_map[cy, cx] = local_radius
            axial_offset = np.einsum(
                "ij,ij->i",
                surface_points - local_center,
                local_tangent,
            )
            local_axis_point = (
                local_center + axial_offset[:, None] * local_tangent
            )
            tube_axis_map[cy, cx] = local_axis_point
            relative = surface_points - local_axis_point
            relative_perp = (
                relative
                - np.einsum(
                    "ij,ij->i",
                    relative,
                    local_tangent,
                )[:, None]
                * local_tangent
            )
            ray_perp = (
                pixel_ray
                - np.einsum(
                    "ij,ij->i",
                    pixel_ray,
                    local_tangent,
                )[:, None]
                * local_tangent
            )
            quadratic_a = np.einsum(
                "ij,ij->i",
                ray_perp,
                ray_perp,
            )
            quadratic_b = 2.0 * np.einsum(
                "ij,ij->i",
                relative_perp,
                ray_perp,
            )
            quadratic_c = (
                np.einsum(
                    "ij,ij->i",
                    relative_perp,
                    relative_perp,
                )
                - local_radius * local_radius
            )
            discriminant = (
                quadratic_b * quadratic_b
                - 4.0 * quadratic_a * quadratic_c
            )
            exact_chord = np.full(len(cx), np.nan, dtype=np.float64)
            valid_exact = (
                np.isfinite(discriminant)
                & (discriminant >= 0.0)
                & (quadratic_a > 1e-9)
            )
            valid_indices = np.flatnonzero(valid_exact)
            rejected_low_ray_perp = 0
            rejected_chord_ratio = 0
            ray_perp_values = np.zeros(0, dtype=np.float64)
            chord_ratio_values = np.zeros(0, dtype=np.float64)
            if len(valid_indices):
                root = (
                    -quadratic_b[valid_indices]
                    + np.sqrt(discriminant[valid_indices])
                ) / (2.0 * quadratic_a[valid_indices])
                surface_radius = np.linalg.norm(
                    relative_perp[valid_indices],
                    axis=1,
                )
                radial_error = np.abs(
                    surface_radius - local_radius[valid_indices]
                )
                ray_perp_norm = np.sqrt(
                    quadratic_a[valid_indices]
                )
                chord_ratio = root / np.maximum(
                    corrected_chord[valid_indices],
                    1e-9,
                )
                ray_perp_values = ray_perp_norm[
                    np.isfinite(ray_perp_norm)
                ]
                chord_ratio_values = chord_ratio[
                    np.isfinite(chord_ratio)
                    & (chord_ratio > 0.0)
                ]
                ray_perp_ok = (
                    ray_perp_norm
                    >= float(local_cylinder_min_ray_perp)
                )
                chord_ratio_ok = (
                    chord_ratio
                    <= float(local_cylinder_max_chord_ratio)
                )
                rejected_low_ray_perp = int(
                    np.count_nonzero(~ray_perp_ok)
                )
                rejected_chord_ratio = int(
                    np.count_nonzero(~chord_ratio_ok)
                )
                stable = (
                    np.isfinite(root)
                    & (root > 0.0)
                    & ray_perp_ok
                    & chord_ratio_ok
                    & (
                        radial_error
                        <= np.maximum(
                            float(
                                local_cylinder_max_radial_error_mm
                            )
                            / 1000.0,
                            float(
                                local_cylinder_max_radial_error_ratio
                            )
                            * local_radius[valid_indices],
                        )
                    )
                )
                exact_chord[
                    valid_indices[stable]
                ] = root[stable]
            usable_exact = np.isfinite(exact_chord)
            if (
                bool(local_cylinder_require_fit)
                and bool(tube_metadata["fallback_only"])
            ):
                usable_exact[:] = False
            ray_perp_p10 = (
                float(np.percentile(ray_perp_values, 10.0))
                if len(ray_perp_values)
                else None
            )
            axial_completion_active = bool(
                float(local_cylinder_axial_p10_threshold) >= 0.0
                and ray_perp_p10 is not None
                and ray_perp_p10
                < float(local_cylinder_axial_p10_threshold)
                and (
                    not bool(
                        local_cylinder_axial_require_single_component
                    )
                    or len(selected_components) == 1
                )
            )
            if axial_completion_active:
                blend = float(local_cylinder_axial_chord_blend)
                component_chord_scale = float(
                    local_cylinder_axial_chord_scale
                )
            else:
                blend = float(local_cylinder_chord_blend)
            corrected_chord[usable_exact] = (
                (1.0 - blend) * corrected_chord[usable_exact]
                + blend * exact_chord[usable_exact]
            )
            local_cylinder_components[component] = {
                **tube_metadata,
                "nonlinear_refinement": nonlinear_metadata,
                "surface_pixels": int(len(cx)),
                "usable_exact_chords": int(
                    np.count_nonzero(usable_exact)
                ),
                "usable_exact_fraction": float(
                    np.count_nonzero(usable_exact) / max(len(cx), 1)
                ),
                "require_fit": bool(local_cylinder_require_fit),
                "min_ray_perp": float(
                    local_cylinder_min_ray_perp
                ),
                "max_chord_ratio": float(
                    local_cylinder_max_chord_ratio
                ),
                "applied_chord_blend": float(blend),
                "applied_chord_scale": float(
                    component_chord_scale
                ),
                "axial_completion_active": bool(
                    axial_completion_active
                ),
                "axial_single_component_ok": bool(
                    len(selected_components) == 1
                ),
                "rejected_low_ray_perp": int(
                    rejected_low_ray_perp
                ),
                "rejected_chord_ratio": int(
                    rejected_chord_ratio
                ),
                "ray_perp_p10": ray_perp_p10,
                "ray_perp_median": (
                    float(np.median(ray_perp_values))
                    if len(ray_perp_values)
                    else None
                ),
                "chord_ratio_median": (
                    float(np.median(chord_ratio_values))
                    if len(chord_ratio_values)
                    else None
                ),
                "chord_ratio_p90": (
                    float(np.percentile(chord_ratio_values, 90.0))
                    if len(chord_ratio_values)
                    else None
                ),
                "exact_chord_min_mm": (
                    float(np.nanmin(exact_chord) * 1000.0)
                    if np.any(usable_exact)
                    else None
                ),
                "exact_chord_median_mm": (
                    float(np.nanmedian(exact_chord) * 1000.0)
                    if np.any(usable_exact)
                    else None
                ),
                "exact_chord_max_mm": (
                    float(np.nanmax(exact_chord) * 1000.0)
                    if np.any(usable_exact)
                    else None
                ),
            }
        if external_map is not None and external_blend > 0.0:
            prior_chord = external_map[cy, cx]
            prior_confidence = (
                np.ones(len(cx), dtype=np.float64)
                if external_confidence is None
                else np.nan_to_num(
                    external_confidence[cy, cx],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
            effective_blend = external_blend * prior_confidence
            usable_prior = (
                np.isfinite(prior_chord)
                & (prior_chord > 0.0)
                & (effective_blend > 0.0)
            )
            corrected_chord[usable_prior] = (
                (1.0 - effective_blend[usable_prior])
                * corrected_chord[usable_prior]
                + effective_blend[usable_prior]
                * prior_chord[usable_prior]
            )
            if np.any(usable_prior):
                external_chord_values.append(prior_chord[usable_prior])
                external_confidence_values.append(
                    prior_confidence[usable_prior]
                )
        chord = np.clip(
            corrected_chord * float(component_chord_scale),
            float(chord_min_mm) / 1000.0,
            float(chord_max_mm) / 1000.0,
        )
        footprint_radius = np.maximum(
            meters_per_pixel * float(footprint_scale),
            float(footprint_min_mm) / 1000.0,
        )
        chord_map[cy, cx] = chord
        footprint_radius_map[cy, cx] = footprint_radius
        missing_tube_radius = ~np.isfinite(tube_radius_map[cy, cx])
        if np.any(missing_tube_radius):
            fallback_tube_radius = np.maximum(
                0.5 * chord[missing_tube_radius],
                footprint_radius[missing_tube_radius],
            )
            component_tube_radius = tube_radius_map[cy, cx]
            component_tube_radius[missing_tube_radius] = (
                fallback_tube_radius
            )
            tube_radius_map[cy, cx] = component_tube_radius
        missing_tube_axis = ~np.all(
            np.isfinite(tube_axis_map[cy, cx]),
            axis=1,
        )
        if np.any(missing_tube_axis):
            component_surface = np.asarray(
                world[cy, cx],
                dtype=np.float64,
            )
            component_ray = component_surface - np.asarray(
                camera_pos,
                dtype=np.float64,
            ).reshape(1, 3)
            component_ray /= (
                np.linalg.norm(
                    component_ray,
                    axis=1,
                    keepdims=True,
                )
                + 1e-12
            )
            fallback_axis = (
                component_surface
                + 0.5 * chord[:, None] * component_ray
            )
            component_tube_axis = tube_axis_map[cy, cx]
            component_tube_axis[missing_tube_axis] = fallback_axis[
                missing_tube_axis
            ]
            tube_axis_map[cy, cx] = component_tube_axis
        surface_parts.append(world[cy, cx])
        chord_parts.append(chord)
        radius_parts.append(footprint_radius)
        raw_chord_parts.append(raw_chord)
        skeleton_sizes.append(int(len(sx)))

    if not surface_parts:
        raise ValueError("RGBD target has no chord-completion samples")
    surface = np.concatenate(surface_parts, axis=0)
    chord = np.concatenate(chord_parts, axis=0)
    footprint_radius = np.concatenate(radius_parts, axis=0)
    raw_chord = np.concatenate(raw_chord_parts, axis=0)
    voxel_m = float(voxel_mm) / 1000.0

    multiple_components_ok = len(selected_components) > 1
    component_resampling_active = bool(
        resample_component_edges and multiple_components_ok
    )
    component_resampling_metadata: Dict[str, Any] = {
        "requested": bool(resample_component_edges),
        "enabled": component_resampling_active,
        "require_multiple_components": True,
        "multiple_components_ok": bool(multiple_components_ok),
        "tree_edges": 0,
        "resampled_edges": 0,
        "resampled_samples": 0,
        "spacing_scale": float(component_edge_spacing_scale),
    }
    if component_resampling_active:
        (
            resampled_surface,
            resampled_chord,
            resampled_radius,
            component_resampling_metadata,
        ) = _resample_component_spanning_edges(
            labels=labels,
            selected_components=selected_components,
            world=world,
            color=color,
            depth=metric_depth,
            focal_px=focal_px,
            chord_map=chord_map,
            footprint_radius_map=footprint_radius_map,
            voxel_m=voxel_m,
            spacing_scale=float(component_edge_spacing_scale),
        )
        component_resampling_metadata.update(
            {
                "requested": True,
                "enabled": True,
                "require_multiple_components": True,
                "multiple_components_ok": True,
            }
        )
        if len(resampled_surface):
            surface = np.concatenate(
                (surface, resampled_surface),
                axis=0,
            )
            chord = np.concatenate((chord, resampled_chord), axis=0)
            footprint_radius = np.concatenate(
                (footprint_radius, resampled_radius),
                axis=0,
            )

    bridge_edges: list[Dict[str, Any]] = []
    bridge_candidate_diagnostics: list[Dict[str, Any]] = []
    bridge_sample_count = 0
    bridge_axis_sample_count = 0
    selected_bridge_geometry = str(bridge_geometry_mode).strip().lower()
    if selected_bridge_geometry not in {
        "ray_chord",
        "axis_capsule",
        "both",
    }:
        raise ValueError(
            f"unsupported bridge_geometry_mode={bridge_geometry_mode!r}"
        )
    effective_bridge_contact_margin_m = (
        _effective_bridge_contact_margin_m(
            base_margin_mm=float(bridge_contact_margin_mm),
            voxel_margin_scale=float(
                bridge_contact_voxel_margin_scale
            ),
            voxel_mm=float(voxel_mm),
        )
    )
    effective_bridge_contact_margin_mm = (
        effective_bridge_contact_margin_m * 1000.0
    )
    if bool(bridge_components):
        (
            bridge_edges,
            bridge_candidate_diagnostics,
        ) = _select_component_bridges(
            labels=labels,
            selected_components=selected_components,
            world=world,
            color=color,
            skeleton_uv_by_component=skeleton_uv_by_component,
            chord_map=chord_map,
            footprint_radius_map=footprint_radius_map,
            tube_radius_map=tube_radius_map,
            tube_axis_map=tube_axis_map,
            max_gap_3d_m=float(component_gap_3d_mm) / 1000.0,
            max_gap_px=float(component_gap_px),
            max_color_delta=float(component_color_delta),
            min_tangent_alignment=float(bridge_tangent_alignment),
            tangent_neighborhood_px=float(
                bridge_tangent_neighborhood_px
            ),
            selection_mode=str(bridge_selection_mode),
            contact_margin_m=effective_bridge_contact_margin_m,
        )
        bridge_surface_parts = []
        bridge_chord_parts = []
        bridge_radius_parts = []
        for edge in bridge_edges:
            if selected_bridge_geometry in {"ray_chord", "both"}:
                steps = max(
                    2,
                    int(math.ceil(float(edge["gap_px"]))) + 1,
                )
                alpha = np.linspace(0.0, 1.0, steps + 1)[1:-1]
                if len(alpha):
                    first_world = np.asarray(
                        edge["first_world"],
                        dtype=np.float64,
                    )
                    second_world = np.asarray(
                        edge["second_world"],
                        dtype=np.float64,
                    )
                    bridge_surface_parts.append(
                        first_world.reshape(1, 3)
                        + alpha[:, None]
                        * (second_world - first_world).reshape(1, 3)
                    )
                    bridge_chord_parts.append(
                        (
                            float(edge["first_chord_m"])
                            + alpha
                            * (
                                float(edge["second_chord_m"])
                                - float(edge["first_chord_m"])
                            )
                        )
                        * float(bridge_chord_scale)
                    )
                    bridge_radius_parts.append(
                        (
                            float(
                                edge["first_footprint_radius_m"]
                            )
                            + alpha
                            * (
                                float(
                                    edge[
                                        "second_footprint_radius_m"
                                    ]
                                )
                                - float(
                                    edge[
                                        "first_footprint_radius_m"
                                    ]
                                )
                            )
                        )
                        * float(bridge_footprint_scale)
                    )
            if selected_bridge_geometry in {"axis_capsule", "both"}:
                axis_centers, axis_radii = (
                    _axis_capsule_bridge_samples(
                        first_axis_world=edge["first_axis_world"],
                        second_axis_world=edge["second_axis_world"],
                        first_radius_m=float(
                            edge["first_tube_radius_m"]
                        ),
                        second_radius_m=float(
                            edge["second_tube_radius_m"]
                        ),
                        voxel_m=voxel_m,
                        radius_scale=float(bridge_axis_radius_scale),
                    )
                )
                bridge_surface_parts.append(axis_centers)
                bridge_chord_parts.append(
                    np.zeros(len(axis_centers), dtype=np.float64)
                )
                bridge_radius_parts.append(axis_radii)
                bridge_axis_sample_count += int(len(axis_centers))
        if bridge_surface_parts:
            bridge_surface = np.concatenate(bridge_surface_parts, axis=0)
            bridge_chord = np.concatenate(bridge_chord_parts, axis=0)
            bridge_radius = np.concatenate(bridge_radius_parts, axis=0)
            bridge_sample_count = int(len(bridge_surface))
            surface = np.concatenate((surface, bridge_surface), axis=0)
            chord = np.concatenate((chord, bridge_chord), axis=0)
            footprint_radius = np.concatenate(
                (footprint_radius, bridge_radius),
                axis=0,
            )

    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    ray = surface - camera.reshape(1, 3)
    ray /= np.linalg.norm(ray, axis=1, keepdims=True) + 1e-12
    surface = surface + ray * (float(surface_offset_mm) / 1000.0)
    endpoints = surface + ray * chord[:, None]

    margin = float(footprint_radius.max()) + 2.0 * voxel_m
    lower = (
        np.floor(
            (np.minimum(surface.min(axis=0), endpoints.min(axis=0)) - margin)
            / voxel_m
        )
        * voxel_m
    )
    upper = (
        np.ceil(
            (np.maximum(surface.max(axis=0), endpoints.max(axis=0)) + margin)
            / voxel_m
        )
        * voxel_m
    )
    shape = np.rint((upper - lower) / voxel_m).astype(np.int64) + 1
    if np.any(shape <= 2) or int(np.prod(shape)) > 80_000_000:
        raise ValueError(f"invalid ray-chord voxel grid: {shape.tolist()}")
    selected_meshing_mode = str(ray_meshing_mode).strip().lower()
    if selected_meshing_mode == "binary":
        occupancy = np.zeros(
            tuple(int(value) for value in shape),
            dtype=bool,
        )
        implicit_field = None
    elif selected_meshing_mode == "sdf":
        occupancy = None
        implicit_field = np.full(
            tuple(int(value) for value in shape),
            -max(float(footprint_radius.max()), 2.0 * voxel_m),
            dtype=np.float32,
        )
    else:
        raise ValueError(
            f"unsupported ray_meshing_mode={ray_meshing_mode!r}; expected "
            "'binary' or 'sdf'"
        )
    selected_ray_primitive = str(ray_primitive_mode).strip().lower()
    if selected_ray_primitive not in {"capsule", "capped_cylinder"}:
        raise ValueError(
            f"unsupported ray_primitive_mode={ray_primitive_mode!r}; "
            "expected 'capsule' or 'capped_cylinder'"
        )

    splat_count = 0
    for start, direction, length, radius in zip(
        surface,
        ray,
        chord,
        footprint_radius,
    ):
        if (
            selected_ray_primitive == "capped_cylinder"
            and float(length) > 1e-12
        ):
            end = start + direction * float(length)
            primitive_lower = np.minimum(start, end) - float(radius)
            primitive_upper = np.maximum(start, end) + float(radius)
            lower_index = np.floor(
                (primitive_lower - lower) / voxel_m
            ).astype(np.int64)
            upper_index = np.ceil(
                (primitive_upper - lower) / voxel_m
            ).astype(np.int64)
            axes = [
                np.arange(
                    max(0, int(lower_index[axis])),
                    min(int(shape[axis]), int(upper_index[axis]) + 1),
                )
                for axis in range(3)
            ]
            gx, gy, gz = np.meshgrid(*axes, indexing="ij")
            local_points = (
                np.stack((gx, gy, gz), axis=-1).astype(np.float64)
                * voxel_m
                + lower
            )
            local_field = _capped_cylinder_signed_field(
                local_points,
                start=start,
                direction=direction,
                length_m=float(length),
                radius_m=float(radius),
                axial_margin_m=_capped_cylinder_axial_margin_m(
                    voxel_m=voxel_m,
                    radius_m=float(radius),
                    radius_scale=float(
                        capped_cylinder_axial_margin_radius_scale
                    ),
                ),
            )
            target = np.ix_(*axes)
            if occupancy is not None:
                occupancy[target] |= local_field >= 0.0
            else:
                current = implicit_field[target]
                implicit_field[target] = np.maximum(
                    current,
                    local_field.astype(np.float32),
                )
            splat_count += 1
            continue
        steps = (
            1
            if float(length) <= 1e-12
            else max(2, int(math.ceil(float(length) / voxel_m)) + 1)
        )
        for distance in np.linspace(0.0, float(length), steps):
            center = start + direction * distance
            center_index = np.rint((center - lower) / voxel_m).astype(
                np.int64
            )
            reach = int(math.ceil(float(radius) / voxel_m)) + 1
            axes = [
                np.arange(
                    max(0, int(center_index[axis]) - reach),
                    min(
                        int(shape[axis]),
                        int(center_index[axis]) + reach + 1,
                    ),
                )
                for axis in range(3)
            ]
            gx, gy, gz = np.meshgrid(*axes, indexing="ij")
            local_points = (
                np.stack((gx, gy, gz), axis=-1).astype(np.float64) * voxel_m
                + lower
            )
            distance_sq = np.einsum(
                "...i,...i->...",
                local_points - center,
                local_points - center,
            )
            target = np.ix_(*axes)
            if occupancy is not None:
                occupancy[target] |= distance_sq <= float(radius) ** 2
            else:
                local_field = (
                    float(radius) - np.sqrt(distance_sq)
                ).astype(np.float32)
                current = implicit_field[target]
                implicit_field[target] = np.maximum(
                    current,
                    local_field,
                )
            splat_count += 1

    if occupancy is not None:
        positive_voxels = int(occupancy.sum())
        if positive_voxels < 8:
            raise ValueError(
                f"ray-chord occupancy too small: "
                f"{positive_voxels} voxels"
            )
        (
            mesh,
            occupancy_component_count,
            nonwatertight_occupancy_components,
        ) = _mesh_from_discrete_occupancy_components(
            occupancy,
            lower=lower,
            voxel_m=voxel_m,
            smoothing_sigma_voxels=float(
                binary_smoothing_sigma_voxels
            ),
        )
    else:
        positive_voxels = int(np.count_nonzero(implicit_field >= 0.0))
        if positive_voxels < 8:
            raise ValueError(
                f"ray-chord implicit field too small: "
                f"{positive_voxels} nonnegative voxels"
            )
        (
            mesh,
            occupancy_component_count,
            nonwatertight_occupancy_components,
        ) = _mesh_from_implicit_union_field(
            implicit_field,
            lower=lower,
            voxel_m=voxel_m,
        )
    return mesh, {
        "build": RGBD_ONLY_MESH_BUILD,
        "method": "camera_ray_circular_chord_completion",
        "selected_rgbd_points": int(len(surface)),
        "rgbd_components": int(len(component_sizes)),
        "seed_component": int(seed_component),
        "component_policy": policy,
        "selected_components": sorted(
            int(component) for component in selected_components
        ),
        "selected_component_count": int(len(selected_components)),
        "rgbd_component_pixels_desc": sorted(
            [int(value) for value in component_sizes],
            reverse=True,
        ),
        "rgbd_component_skeleton_points": skeleton_sizes,
        "voxel_mm": float(voxel_mm),
        "grid_shape": [int(value) for value in shape],
        "occupied_voxels": int(positive_voxels),
        "ray_primitive_mode": selected_ray_primitive,
        "capped_cylinder_axial_margin_radius_scale": float(
            capped_cylinder_axial_margin_radius_scale
        ),
        "ray_meshing_mode": selected_meshing_mode,
        "occupancy_component_count": int(occupancy_component_count),
        "nonwatertight_occupancy_components": int(
            nonwatertight_occupancy_components
        ),
        "splat_count": int(splat_count),
        "bridge_edge_count": int(len(bridge_edges)),
        "bridge_sample_count": int(bridge_sample_count),
        "bridge_axis_sample_count": int(bridge_axis_sample_count),
        "bridge_candidate_diagnostics": bridge_candidate_diagnostics,
        "component_edge_resampling": component_resampling_metadata,
        "bridge_edges": [
            {
                "components": [
                    int(edge["first_component"]),
                    int(edge["second_component"]),
                ],
                "first_yx": [
                    int(value) for value in edge["first_yx"]
                ],
                "second_yx": [
                    int(value) for value in edge["second_yx"]
                ],
                "gap_3d_mm": float(edge["gap_3d_m"] * 1000.0),
                "gap_px": float(edge["gap_px"]),
                "color_delta": float(edge["color_delta"]),
                "bridge_kind": str(edge["bridge_kind"]),
                "first_tube_radius_mm": float(
                    edge["first_tube_radius_m"] * 1000.0
                ),
                "second_tube_radius_mm": float(
                    edge["second_tube_radius_m"] * 1000.0
                ),
                "contact_limit_mm": float(
                    edge["contact_limit_m"] * 1000.0
                ),
                "axis_gap_3d_mm": float(
                    np.linalg.norm(
                        np.asarray(edge["second_axis_world"])
                        - np.asarray(edge["first_axis_world"])
                    )
                    * 1000.0
                ),
                "radius_contact": bool(edge["radius_contact"]),
                "tangent_continuation": bool(
                    edge["tangent_continuation"]
                ),
                "first_alignment": float(edge["first_alignment"]),
                "second_alignment": float(edge["second_alignment"]),
                "tangent_alignment": float(edge["tangent_alignment"]),
            }
            for edge in bridge_edges
        ],
        "curvature_radius_components": {
            str(component): metadata
            for component, metadata in curvature_components.items()
        },
        "chord_model": selected_chord_model,
        "local_cylinder_components": {
            str(component): metadata
            for component, metadata in local_cylinder_components.items()
        },
        "raw_chord_min_mm": float(raw_chord.min() * 1000.0),
        "raw_chord_median_mm": float(np.median(raw_chord) * 1000.0),
        "raw_chord_max_mm": float(raw_chord.max() * 1000.0),
        "external_chord": {
            "provided": external_map is not None,
            "blend": external_blend,
            "confidence_provided": external_confidence is not None,
            "usable_pixels": int(
                sum(len(values) for values in external_chord_values)
            ),
            "confidence_min": (
                float(
                    min(
                        values.min()
                        for values in external_confidence_values
                    )
                )
                if external_confidence_values
                else None
            ),
            "confidence_median": (
                float(
                    np.median(
                        np.concatenate(external_confidence_values)
                    )
                )
                if external_confidence_values
                else None
            ),
            "confidence_max": (
                float(
                    max(
                        values.max()
                        for values in external_confidence_values
                    )
                )
                if external_confidence_values
                else None
            ),
            "min_mm": (
                float(
                    min(values.min() for values in external_chord_values)
                    * 1000.0
                )
                if external_chord_values
                else None
            ),
            "median_mm": (
                float(
                    np.median(np.concatenate(external_chord_values))
                    * 1000.0
                )
                if external_chord_values
                else None
            ),
            "max_mm": (
                float(
                    max(values.max() for values in external_chord_values)
                    * 1000.0
                )
                if external_chord_values
                else None
            ),
        },
        "chord_min_mm": float(chord.min() * 1000.0),
        "chord_median_mm": float(np.median(chord) * 1000.0),
        "chord_max_mm": float(chord.max() * 1000.0),
        "footprint_radius_min_mm": float(
            footprint_radius.min() * 1000.0
        ),
        "footprint_radius_median_mm": float(
            np.median(footprint_radius) * 1000.0
        ),
        "footprint_radius_max_mm": float(
            footprint_radius.max() * 1000.0
        ),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_volume_cm3": float(abs(mesh.volume) * 1e6),
        "parameters": {
            "footprint_scale": float(footprint_scale),
            "footprint_min_mm": float(footprint_min_mm),
            "chord_scale": float(chord_scale),
            "chord_min_mm": float(chord_min_mm),
            "chord_max_mm": float(chord_max_mm),
            "surface_offset_mm": float(surface_offset_mm),
            "radius_gain": float(radius_gain),
            "radius_bias_px": float(radius_bias_px),
            "inset_bias_px": float(inset_bias_px),
            "component_gap_3d_mm": float(component_gap_3d_mm),
            "component_gap_px": float(component_gap_px),
            "component_color_delta": float(component_color_delta),
            "resample_component_edges": bool(
                resample_component_edges
            ),
            "component_edge_spacing_scale": float(
                component_edge_spacing_scale
            ),
            "bridge_components": bool(bridge_components),
            "bridge_selection_mode": str(bridge_selection_mode),
            "bridge_contact_margin_mm": float(
                bridge_contact_margin_mm
            ),
            "bridge_contact_voxel_margin_scale": float(
                bridge_contact_voxel_margin_scale
            ),
            "effective_bridge_contact_margin_mm": float(
                effective_bridge_contact_margin_mm
            ),
            "bridge_geometry_mode": selected_bridge_geometry,
            "bridge_axis_radius_scale": float(
                bridge_axis_radius_scale
            ),
            "bridge_tangent_alignment": float(
                bridge_tangent_alignment
            ),
            "bridge_tangent_neighborhood_px": float(
                bridge_tangent_neighborhood_px
            ),
            "bridge_chord_scale": float(bridge_chord_scale),
            "bridge_footprint_scale": float(bridge_footprint_scale),
            "curvature_radius_blend": float(curvature_radius_blend),
            "curvature_scale_min": float(curvature_scale_min),
            "curvature_scale_max": float(curvature_scale_max),
            "curvature_fit_max_residual_mm": float(
                curvature_fit_max_residual_mm
            ),
            "curvature_min_valid_fits": int(
                curvature_min_valid_fits
            ),
            "curvature_full_blend_skeleton_points": int(
                curvature_full_blend_skeleton_points
            ),
            "curvature_local_scales": bool(curvature_local_scales),
            "curvature_smoothing_px": float(curvature_smoothing_px),
            "curvature_local_shrinkage": float(
                curvature_local_shrinkage
            ),
            "curvature_max_center_toward_camera_ratio": float(
                curvature_max_center_toward_camera_ratio
            ),
            "chord_model": selected_chord_model,
            "local_cylinder_chord_blend": float(
                local_cylinder_chord_blend
            ),
            "local_cylinder_require_fit": bool(
                local_cylinder_require_fit
            ),
            "local_cylinder_max_radial_error_mm": float(
                local_cylinder_max_radial_error_mm
            ),
            "local_cylinder_max_radial_error_ratio": float(
                local_cylinder_max_radial_error_ratio
            ),
            "local_cylinder_min_ray_perp": float(
                local_cylinder_min_ray_perp
            ),
            "local_cylinder_max_chord_ratio": float(
                local_cylinder_max_chord_ratio
            ),
            "local_cylinder_axial_p10_threshold": float(
                local_cylinder_axial_p10_threshold
            ),
            "local_cylinder_axial_chord_blend": float(
                local_cylinder_axial_chord_blend
            ),
            "local_cylinder_axial_chord_scale": float(
                local_cylinder_axial_chord_scale
            ),
            "local_cylinder_axial_require_single_component": bool(
                local_cylinder_axial_require_single_component
            ),
            "local_cylinder_nonlinear_refine": bool(
                local_cylinder_nonlinear_refine
            ),
            "local_cylinder_nonlinear_local_accepts": bool(
                local_cylinder_nonlinear_local_accepts
            ),
            "circle_radius_scale": float(circle_radius_scale),
            "circle_radius_bias_px": float(circle_radius_bias_px),
            "circle_radius_min_mm": float(circle_radius_min_mm),
            "circle_radius_max_mm": float(circle_radius_max_mm),
            "circle_direct_radius_blend": float(
                circle_direct_radius_blend
            ),
            "circle_direct_center_blend": float(
                circle_direct_center_blend
            ),
            "circle_front_anchored_radius_blend": float(
                circle_front_anchored_radius_blend
            ),
            "circle_endpoint_radius_regularization_blend": float(
                circle_endpoint_radius_regularization_blend
            ),
            "circle_graph_radius_smoothing_gcv": bool(
                circle_graph_radius_smoothing_gcv
            ),
            "circle_center_fit_blend": float(circle_center_fit_blend),
            "circle_center_offset_scale": float(
                circle_center_offset_scale
            ),
            "circle_tangent_neighborhood_px": float(
                circle_tangent_neighborhood_px
            ),
            "ray_primitive_mode": selected_ray_primitive,
            "capped_cylinder_axial_margin_radius_scale": float(
                capped_cylinder_axial_margin_radius_scale
            ),
            "ray_meshing_mode": selected_meshing_mode,
            "binary_smoothing_sigma_voxels": float(
                binary_smoothing_sigma_voxels
            ),
            "external_chord_blend": external_blend,
        },
        "forbidden_inputs_used": [],
    }


def reconstruct_rgbd_only_mesh(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    click_u: int,
    click_v: int,
    roi_radius_px: int = 190,
    extraction_voxel_mm: float = 1.25,
    output_voxel_mm: float = 1.25,
    mask_completion_mode: str = "off",
    mask_completion_gap_3d_mm: float = 20.0,
    mask_completion_gap_px: float = 8.0,
    mask_completion_color_delta: float = 80.0,
    mask_completion_max_hops: int = 8,
    mask_completion_max_added_pixel_ratio: float = 2.0,
    method: str = "ray_chord",
    footprint_scale: float = 0.55,
    footprint_min_mm: float = 0.65,
    lateral_expand_mm: float = 0.0,
    closing_radius_mm: float = 1.25,
    backfill_scale: float = 0.0,
    backfill_step_mm: float = 1.25,
    height_scale: float = 1.0,
    height_offset_mm: float = 0.0,
    base_extension_mm: float = 0.0,
    chord_scale: float = 1.0,
    chord_min_mm: float = 0.5,
    chord_max_mm: float = 12.0,
    surface_offset_mm: float = 0.0,
    radius_gain: float = 1.3,
    radius_bias_px: float = 0.8,
    inset_bias_px: float = 0.25,
    component_policy: str = "all",
    component_gap_3d_mm: float = 12.0,
    component_gap_px: float = 8.0,
    component_color_delta: float = 40.0,
    resample_component_edges: bool = False,
    component_edge_spacing_scale: float = 1.5,
    bridge_components: bool = False,
    bridge_selection_mode: str = "tangent",
    bridge_contact_margin_mm: float = 1.5,
    bridge_contact_voxel_margin_scale: float = 0.5,
    bridge_geometry_mode: str = "ray_chord",
    bridge_axis_radius_scale: float = 1.0,
    bridge_tangent_alignment: float = 0.85,
    bridge_tangent_neighborhood_px: float = 10.0,
    bridge_chord_scale: float = 1.0,
    bridge_footprint_scale: float = 1.0,
    curvature_radius_blend: float = 0.0,
    curvature_scale_min: float = 0.5,
    curvature_scale_max: float = 1.5,
    curvature_fit_max_residual_mm: float = 0.75,
    curvature_min_valid_fits: int = 8,
    curvature_full_blend_skeleton_points: int = 80,
    curvature_local_scales: bool = False,
    curvature_smoothing_px: float = 10.0,
    curvature_local_shrinkage: float = 1.0,
    curvature_max_center_toward_camera_ratio: float = 1.0,
    circle_radius_scale: float = 1.0,
    circle_radius_bias_px: float = 0.25,
    circle_adaptive_radius_bias: bool = True,
    circle_radius_min_mm: float = 0.4,
    circle_radius_max_mm: float = 8.0,
    circle_direct_radius_blend: float = 1.0,
    circle_soft_direct_radius_blend: bool = True,
    circle_soft_direct_radius_blend_max: float = 0.5,
    circle_direct_center_blend: float = 0.0,
    circle_adaptive_direct_center_blend: bool = True,
    circle_front_anchored_radius_blend: float = 0.0,
    circle_endpoint_radius_regularization_blend: float = 1.0,
    circle_graph_radius_smoothing_gcv: bool = False,
    circle_chain_geometry_smoothing_gcv: bool = False,
    circle_global_surface_fit: bool = False,
    circle_global_radius_curvature_sigma: float = 0.0,
    circle_global_sections_per_control: float = 8.0,
    circle_global_center_prior_sigma_mm: float = 1.5,
    circle_global_adaptive_sections_per_control: bool = True,
    circle_global_control_cv: bool = False,
    circle_global_center_prior_cv: bool = False,
    circle_global_point_parameter_refinement_sections: int = 0,
    circle_global_point_parameter_refinement_blend: float = 0.75,
    circle_global_adaptive_point_parameter_refinement: bool = True,
    circle_global_front_anchor_sigma_mm: float = 0.25,
    circle_global_front_anchor_min_fraction: float = 0.25,
    circle_global_front_anchor_max_ray_perp_p10: float = 0.4,
    circle_fixed_centerline_radius_refinement: bool = True,
    circle_fixed_centerline_radius_quantile: float = 0.60,
    circle_fixed_centerline_radius_bandwidth_sections: float = 6.0,
    circle_fixed_centerline_radius_blend: float = 1.0,
    circle_degenerate_endpoint_curve_refinement: bool = True,
    circle_degenerate_endpoint_curve_blend: float = 0.75,
    circle_prune_short_skeleton_spurs: bool = False,
    circle_axial_endpoint_radius_repair: bool = False,
    circle_visible_endpoint_extension: bool = True,
    circle_visible_endpoint_extension_blend: float = 0.8,
    circle_visible_endpoint_extension_max_radius_ratio: float = 1.0,
    circle_visible_endpoint_preserve_original_ring: bool = False,
    swept_ring_samples: int = 24,
    circle_polygonal_ring_selection: bool = True,
    circle_polygonal_facet_inversion: bool = True,
    circle_center_fit_blend: float = 1.0,
    circle_center_offset_scale: float = 1.0,
    circle_tangent_neighborhood_px: float = 10.0,
    chord_model: str = "silhouette",
    local_cylinder_chord_blend: float = 1.0,
    local_cylinder_require_fit: bool = False,
    local_cylinder_max_radial_error_mm: float = 1.5,
    local_cylinder_max_radial_error_ratio: float = 0.75,
    local_cylinder_min_ray_perp: float = 0.0,
    local_cylinder_max_chord_ratio: float = 1_000_000.0,
    local_cylinder_axial_p10_threshold: float = -1.0,
    local_cylinder_axial_chord_blend: float = 1.0,
    local_cylinder_axial_chord_scale: float = 1.0,
    local_cylinder_axial_require_single_component: bool = False,
    local_cylinder_nonlinear_refine: bool = False,
    local_cylinder_nonlinear_local_accepts: bool = False,
    ray_primitive_mode: str = "capsule",
    capped_cylinder_axial_margin_radius_scale: float = 0.0,
    ray_meshing_mode: str = "binary",
    binary_smoothing_sigma_voxels: float = 0.35,
    external_chord_map: np.ndarray | None = None,
    external_chord_confidence_map: np.ndarray | None = None,
    external_chord_blend: float = 0.0,
) -> RGBDOnlyMeshResult:
    """Reconstruct a watertight mesh without any evaluator-owned data."""
    color = np.asarray(rgb)
    metric_depth = np.asarray(depth, dtype=np.float64)
    extraction = reconstruct_supported_object_mesh(
        color,
        metric_depth,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
        click_u=int(click_u),
        click_v=int(click_v),
        roi_radius_px=int(roi_radius_px),
        voxel_mm=float(extraction_voxel_mm),
    )
    selected_mask_completion = str(mask_completion_mode).strip().lower()
    mask_completion_metadata: Dict[str, Any] = {
        "mode": selected_mask_completion,
        "initial_pixels": int(extraction.selected_mask.sum()),
        "expanded_pixels": int(extraction.selected_mask.sum()),
        "added_pixels": 0,
    }
    if selected_mask_completion == "rgbd_graph":
        extraction, mask_completion_metadata = (
            _complete_extraction_mask_from_rgbd_graph(
                extraction,
                color,
                metric_depth,
                camera_pos=camera_pos,
                camera_quat_xyzw=camera_quat_xyzw,
                focal_length=float(focal_length),
                horizontal_aperture=float(horizontal_aperture),
                max_gap_3d_mm=float(mask_completion_gap_3d_mm),
                max_gap_px=float(mask_completion_gap_px),
                max_color_delta=float(mask_completion_color_delta),
                max_hops=int(mask_completion_max_hops),
                max_added_pixel_ratio=float(
                    mask_completion_max_added_pixel_ratio
                ),
            )
        )
        mask_completion_metadata = {
            "mode": selected_mask_completion,
            **mask_completion_metadata,
        }
    elif selected_mask_completion != "off":
        raise ValueError(
            f"unsupported mask_completion_mode={mask_completion_mode!r}; "
            "expected 'off' or 'rgbd_graph'"
        )
    selected_method = str(method).strip().lower()
    if selected_method == "ray_chord":
        mesh, mesh_metadata = _mesh_from_ray_chord_volume(
            extraction,
            color,
            metric_depth,
            camera_pos=camera_pos,
            camera_quat_xyzw=camera_quat_xyzw,
            focal_length=float(focal_length),
            horizontal_aperture=float(horizontal_aperture),
            voxel_mm=float(output_voxel_mm),
            footprint_scale=float(footprint_scale),
            footprint_min_mm=float(footprint_min_mm),
            chord_scale=float(chord_scale),
            chord_min_mm=float(chord_min_mm),
            chord_max_mm=float(chord_max_mm),
            surface_offset_mm=float(surface_offset_mm),
            radius_gain=float(radius_gain),
            radius_bias_px=float(radius_bias_px),
            inset_bias_px=float(inset_bias_px),
            click_u=int(click_u),
            click_v=int(click_v),
            component_policy=str(component_policy),
            component_gap_3d_mm=float(component_gap_3d_mm),
            component_gap_px=float(component_gap_px),
            component_color_delta=float(component_color_delta),
            resample_component_edges=bool(resample_component_edges),
            component_edge_spacing_scale=float(
                component_edge_spacing_scale
            ),
            bridge_components=bool(bridge_components),
            bridge_selection_mode=str(bridge_selection_mode),
            bridge_contact_margin_mm=float(
                bridge_contact_margin_mm
            ),
            bridge_contact_voxel_margin_scale=float(
                bridge_contact_voxel_margin_scale
            ),
            bridge_geometry_mode=str(bridge_geometry_mode),
            bridge_axis_radius_scale=float(bridge_axis_radius_scale),
            bridge_tangent_alignment=float(bridge_tangent_alignment),
            bridge_tangent_neighborhood_px=float(
                bridge_tangent_neighborhood_px
            ),
            bridge_chord_scale=float(bridge_chord_scale),
            bridge_footprint_scale=float(bridge_footprint_scale),
            curvature_radius_blend=float(curvature_radius_blend),
            curvature_scale_min=float(curvature_scale_min),
            curvature_scale_max=float(curvature_scale_max),
            curvature_fit_max_residual_mm=float(
                curvature_fit_max_residual_mm
            ),
            curvature_min_valid_fits=int(curvature_min_valid_fits),
            curvature_full_blend_skeleton_points=int(
                curvature_full_blend_skeleton_points
            ),
            curvature_local_scales=bool(curvature_local_scales),
            curvature_smoothing_px=float(curvature_smoothing_px),
            curvature_local_shrinkage=float(
                curvature_local_shrinkage
            ),
            curvature_max_center_toward_camera_ratio=float(
                curvature_max_center_toward_camera_ratio
            ),
            chord_model=str(chord_model),
            local_cylinder_chord_blend=float(
                local_cylinder_chord_blend
            ),
            local_cylinder_require_fit=bool(
                local_cylinder_require_fit
            ),
            local_cylinder_max_radial_error_mm=float(
                local_cylinder_max_radial_error_mm
            ),
            local_cylinder_max_radial_error_ratio=float(
                local_cylinder_max_radial_error_ratio
            ),
            local_cylinder_min_ray_perp=float(
                local_cylinder_min_ray_perp
            ),
            local_cylinder_max_chord_ratio=float(
                local_cylinder_max_chord_ratio
            ),
            local_cylinder_axial_p10_threshold=float(
                local_cylinder_axial_p10_threshold
            ),
            local_cylinder_axial_chord_blend=float(
                local_cylinder_axial_chord_blend
            ),
            local_cylinder_axial_chord_scale=float(
                local_cylinder_axial_chord_scale
            ),
            local_cylinder_axial_require_single_component=bool(
                local_cylinder_axial_require_single_component
            ),
            local_cylinder_nonlinear_refine=bool(
                local_cylinder_nonlinear_refine
            ),
            local_cylinder_nonlinear_local_accepts=bool(
                local_cylinder_nonlinear_local_accepts
            ),
            circle_radius_scale=float(circle_radius_scale),
            circle_radius_bias_px=float(circle_radius_bias_px),
            circle_radius_min_mm=float(circle_radius_min_mm),
            circle_radius_max_mm=float(circle_radius_max_mm),
            circle_direct_radius_blend=float(
                circle_direct_radius_blend
            ),
            circle_soft_direct_radius_blend=bool(
                circle_soft_direct_radius_blend
            ),
            circle_soft_direct_radius_blend_max=float(
                circle_soft_direct_radius_blend_max
            ),
            circle_direct_center_blend=float(
                circle_direct_center_blend
            ),
            circle_adaptive_direct_center_blend=bool(
                circle_adaptive_direct_center_blend
            ),
            circle_front_anchored_radius_blend=float(
                circle_front_anchored_radius_blend
            ),
            circle_endpoint_radius_regularization_blend=float(
                circle_endpoint_radius_regularization_blend
            ),
            circle_graph_radius_smoothing_gcv=bool(
                circle_graph_radius_smoothing_gcv
            ),
            circle_center_fit_blend=float(circle_center_fit_blend),
            circle_center_offset_scale=float(
                circle_center_offset_scale
            ),
            circle_tangent_neighborhood_px=float(
                circle_tangent_neighborhood_px
            ),
            ray_primitive_mode=str(ray_primitive_mode),
            capped_cylinder_axial_margin_radius_scale=float(
                capped_cylinder_axial_margin_radius_scale
            ),
            ray_meshing_mode=str(ray_meshing_mode),
            binary_smoothing_sigma_voxels=float(
                binary_smoothing_sigma_voxels
            ),
            external_chord_map=external_chord_map,
            external_chord_confidence_map=(
                external_chord_confidence_map
            ),
            external_chord_blend=float(external_chord_blend),
        )
    elif selected_method in {"circle_tube", "swept_tube"}:
        mesh, mesh_metadata = _mesh_from_local_circle_tubes(
            extraction,
            color,
            metric_depth,
            camera_pos=camera_pos,
            camera_quat_xyzw=camera_quat_xyzw,
            focal_length=float(focal_length),
            horizontal_aperture=float(horizontal_aperture),
            voxel_mm=float(output_voxel_mm),
            click_u=int(click_u),
            click_v=int(click_v),
            component_policy=str(component_policy),
            component_gap_3d_mm=float(component_gap_3d_mm),
            component_gap_px=float(component_gap_px),
            component_color_delta=float(component_color_delta),
            circle_radius_scale=float(circle_radius_scale),
            circle_radius_bias_px=float(circle_radius_bias_px),
            circle_adaptive_radius_bias=bool(
                circle_adaptive_radius_bias
            ),
            circle_radius_min_mm=float(circle_radius_min_mm),
            circle_radius_max_mm=float(circle_radius_max_mm),
            circle_direct_radius_blend=float(
                circle_direct_radius_blend
            ),
            circle_soft_direct_radius_blend=bool(
                circle_soft_direct_radius_blend
            ),
            circle_soft_direct_radius_blend_max=float(
                circle_soft_direct_radius_blend_max
            ),
            circle_direct_center_blend=float(
                circle_direct_center_blend
            ),
            circle_adaptive_direct_center_blend=bool(
                circle_adaptive_direct_center_blend
            ),
            circle_front_anchored_radius_blend=float(
                circle_front_anchored_radius_blend
            ),
            circle_endpoint_radius_regularization_blend=float(
                circle_endpoint_radius_regularization_blend
            ),
            circle_graph_radius_smoothing_gcv=bool(
                circle_graph_radius_smoothing_gcv
            ),
            circle_chain_geometry_smoothing_gcv=bool(
                circle_chain_geometry_smoothing_gcv
            ),
            circle_global_surface_fit=bool(
                circle_global_surface_fit
            ),
            circle_global_radius_curvature_sigma=float(
                circle_global_radius_curvature_sigma
            ),
            circle_global_sections_per_control=float(
                circle_global_sections_per_control
            ),
            circle_global_center_prior_sigma_mm=float(
                circle_global_center_prior_sigma_mm
            ),
            circle_global_adaptive_sections_per_control=bool(
                circle_global_adaptive_sections_per_control
            ),
            circle_global_control_cv=bool(
                circle_global_control_cv
            ),
            circle_global_center_prior_cv=bool(
                circle_global_center_prior_cv
            ),
            circle_global_point_parameter_refinement_sections=int(
                circle_global_point_parameter_refinement_sections
            ),
            circle_global_point_parameter_refinement_blend=float(
                circle_global_point_parameter_refinement_blend
            ),
            circle_global_adaptive_point_parameter_refinement=bool(
                circle_global_adaptive_point_parameter_refinement
            ),
            circle_global_front_anchor_sigma_mm=float(
                circle_global_front_anchor_sigma_mm
            ),
            circle_global_front_anchor_min_fraction=float(
                circle_global_front_anchor_min_fraction
            ),
            circle_global_front_anchor_max_ray_perp_p10=float(
                circle_global_front_anchor_max_ray_perp_p10
            ),
            circle_fixed_centerline_radius_refinement=bool(
                circle_fixed_centerline_radius_refinement
            ),
            circle_fixed_centerline_radius_quantile=float(
                circle_fixed_centerline_radius_quantile
            ),
            circle_fixed_centerline_radius_bandwidth_sections=float(
                circle_fixed_centerline_radius_bandwidth_sections
            ),
            circle_fixed_centerline_radius_blend=float(
                circle_fixed_centerline_radius_blend
            ),
            circle_degenerate_endpoint_curve_refinement=bool(
                circle_degenerate_endpoint_curve_refinement
            ),
            circle_degenerate_endpoint_curve_blend=float(
                circle_degenerate_endpoint_curve_blend
            ),
            circle_prune_short_skeleton_spurs=bool(
                circle_prune_short_skeleton_spurs
            ),
            circle_axial_endpoint_radius_repair=bool(
                circle_axial_endpoint_radius_repair
            ),
            circle_visible_endpoint_extension=bool(
                circle_visible_endpoint_extension
            ),
            circle_visible_endpoint_extension_blend=float(
                circle_visible_endpoint_extension_blend
            ),
            circle_visible_endpoint_extension_max_radius_ratio=float(
                circle_visible_endpoint_extension_max_radius_ratio
            ),
            circle_visible_endpoint_preserve_original_ring=bool(
                circle_visible_endpoint_preserve_original_ring
            ),
            swept_ring_samples=int(swept_ring_samples),
            circle_polygonal_ring_selection=bool(
                circle_polygonal_ring_selection
            ),
            circle_polygonal_facet_inversion=bool(
                circle_polygonal_facet_inversion
            ),
            circle_center_fit_blend=float(circle_center_fit_blend),
            circle_center_offset_scale=float(
                circle_center_offset_scale
            ),
            circle_tangent_neighborhood_px=float(
                circle_tangent_neighborhood_px
            ),
            curvature_radius_blend=float(curvature_radius_blend),
            curvature_scale_min=float(curvature_scale_min),
            curvature_scale_max=float(curvature_scale_max),
            curvature_fit_max_residual_mm=float(
                curvature_fit_max_residual_mm
            ),
            curvature_min_valid_fits=int(curvature_min_valid_fits),
            curvature_smoothing_px=float(curvature_smoothing_px),
            curvature_local_shrinkage=float(
                curvature_local_shrinkage
            ),
            curvature_max_center_toward_camera_ratio=float(
                curvature_max_center_toward_camera_ratio
            ),
            tube_meshing_mode=(
                "swept_surface"
                if selected_method == "swept_tube"
                else "voxel_union"
            ),
            local_cylinder_nonlinear_refine=bool(
                local_cylinder_nonlinear_refine
            ),
            local_cylinder_nonlinear_local_accepts=bool(
                local_cylinder_nonlinear_local_accepts
            ),
        )
    elif selected_method == "height_field":
        image_width = int(metric_depth.shape[1])
        focal_px = (
            float(focal_length)
            / float(horizontal_aperture)
            * float(image_width)
        )
        mesh, mesh_metadata = _mesh_from_support_height_field(
            extraction,
            metric_depth,
            camera_pos=camera_pos,
            focal_px=focal_px,
            voxel_mm=float(output_voxel_mm),
            footprint_scale=float(footprint_scale),
            footprint_min_mm=float(footprint_min_mm),
            lateral_expand_mm=float(lateral_expand_mm),
            closing_radius_mm=float(closing_radius_mm),
            backfill_scale=float(backfill_scale),
            backfill_step_mm=float(backfill_step_mm),
            height_scale=float(height_scale),
            height_offset_mm=float(height_offset_mm),
            base_extension_mm=float(base_extension_mm),
        )
    else:
        raise ValueError(
            f"unsupported RGBD-only method={method!r}; expected "
            "'ray_chord', 'circle_tube', 'swept_tube', or 'height_field'"
        )
    return RGBDOnlyMeshResult(
        mesh=mesh,
        extraction=extraction,
        metadata={
            **mesh_metadata,
            "allowed_inputs": [
                "rgb",
                "metric_depth",
                "camera_intrinsics",
                "camera_extrinsics",
                "target_click_uv",
            ],
            "used_segmentation": False,
            "extraction_build": extraction.metadata["build"],
            "mask_completion": mask_completion_metadata,
        },
    )
