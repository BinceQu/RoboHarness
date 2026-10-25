"""Frozen V13 thin-structure reconstruction for the official RGB-D runtime.

This module is a runtime-only extraction of the canonical V13 implementation.
It consumes RGB, metric depth, and camera calibration only.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.color import rgb2lab
from skimage.morphology import skeletonize
import trimesh

from .depth_mesh_reconstruction import backproject_depth_world
from .rgbd_only_mesh_reconstruction import (
    _depth_component_labels,
    _initial_skeleton_ray_perp_p10,
    _mesh_from_local_circle_tubes,
    _mesh_from_ray_chord_volume,
    reconstruct_rgbd_only_mesh,
)
from .rgbd_scene_mesh_v12 import _support_structure_mask

def _private_defaults() -> dict[str, Any]:
    public_signature = inspect.signature(reconstruct_rgbd_only_mesh)
    public_defaults = {
        name: parameter.default
        for name, parameter in public_signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    private_signature = inspect.signature(_mesh_from_ray_chord_volume)
    return {
        name: public_defaults[name]
        for name in private_signature.parameters
        if name in public_defaults
    }

def _reconstruct_mask(
    mask: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    options: dict[str, Any],
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("cannot reconstruct an empty thin mask")
    local_options = dict(options)
    local_options["click_u"] = int(np.median(xx))
    local_options["click_v"] = int(np.median(yy))
    return _mesh_from_ray_chord_volume(
        SimpleNamespace(selected_mask=mask),
        rgb,
        depth,
        **local_options,
    )

_ray_defaults = _private_defaults

def _crossing_radius(
    offsets: np.ndarray,
    membership: np.ndarray,
    *,
    sign: int,
) -> float | None:
    center = int(np.argmin(np.abs(offsets)))
    indices = (
        np.arange(center, len(offsets), dtype=np.int64)
        if sign > 0
        else np.arange(center, -1, -1, dtype=np.int64)
    )
    for inner, outer in zip(indices[:-1], indices[1:]):
        inner_value = float(membership[inner])
        outer_value = float(membership[outer])
        if outer_value < 0.5 <= inner_value:
            denominator = max(inner_value - outer_value, 1e-12)
            fraction = (inner_value - 0.5) / denominator
            crossing = float(
                offsets[inner]
                + fraction * (offsets[outer] - offsets[inner])
            )
            return abs(crossing)
    return None

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = 0.5 * float(np.sum(ordered_weights))
    index = int(
        np.searchsorted(
            np.cumsum(ordered_weights),
            threshold,
            side="left",
        )
    )
    return float(ordered_values[min(index, len(ordered_values) - 1)])

def rgb_profile_chord_maps(
    mask: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera: dict[str, Any],
    profile_radius_px: float = 7.0,
    profile_step_px: float = 0.25,
    profile_sigma_px: float = 0.55,
    minimum_component_pixels: int = 8,
    minimum_observations_per_section: float = 0.0,
    minimum_contrast_lab: float = 2.0,
    maximum_radius_px: float = 5.0,
    maximum_depth_radius_ratio: float = 3.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    selected = np.asarray(mask, dtype=bool)
    metric_depth = np.asarray(depth, dtype=np.float64)
    color = np.asarray(rgb, dtype=np.float64)
    if color.max(initial=0.0) > 1.5:
        color /= 255.0
    lab = rgb2lab(color[..., :3])
    lab = np.stack(
        [
            ndimage.gaussian_filter(
                lab[..., channel],
                sigma=float(profile_sigma_px),
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    world, _valid = backproject_depth_world(metric_depth, **camera)
    focal_px = (
        float(camera["focal_length"])
        / float(camera["horizontal_aperture"])
        * float(metric_depth.shape[1])
    )
    labels, _yy, _xx, component_for_node = _depth_component_labels(
        selected,
        world_points=world,
        rgb=rgb,
        depth=metric_depth,
        focal_px=focal_px,
    )
    component_sizes = np.bincount(component_for_node).astype(np.int64)
    offsets = np.arange(
        -float(profile_radius_px),
        float(profile_radius_px) + 0.5 * float(profile_step_px),
        float(profile_step_px),
        dtype=np.float64,
    )
    chord_map = np.full(metric_depth.shape, np.nan, dtype=np.float64)
    confidence_map = np.zeros(metric_depth.shape, dtype=np.float64)
    radius_map_px = np.full(metric_depth.shape, np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for component_id, pixel_count in enumerate(component_sizes):
        component = labels == int(component_id)
        cy, cx = np.nonzero(component)
        if int(pixel_count) < int(minimum_component_pixels):
            continue
        skeleton = skeletonize(component)
        sy, sx = np.nonzero(skeleton)
        if not len(sx):
            continue
        observations_per_section = float(pixel_count / max(len(sx), 1))
        if observations_per_section < float(
            minimum_observations_per_section
        ):
            continue
        skeleton_uv = np.column_stack((sx, sy)).astype(np.float64)
        depth_radius = ndimage.distance_transform_edt(component)[sy, sx]
        raw_radius = np.full(len(sx), np.nan, dtype=np.float64)
        raw_confidence = np.zeros(len(sx), dtype=np.float64)
        normal_uv = np.full((len(sx), 2), np.nan, dtype=np.float64)

        for index, (x, y) in enumerate(skeleton_uv):
            distance = np.linalg.norm(
                skeleton_uv - np.array([x, y], dtype=np.float64),
                axis=1,
            )
            local = skeleton_uv[distance <= 10.0]
            if len(local) < 3:
                continue
            centered = local - local.mean(axis=0, keepdims=True)
            if float(np.linalg.norm(centered)) < 1e-9:
                continue
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            tangent = np.asarray(vh[0], dtype=np.float64)
            tangent /= np.linalg.norm(tangent) + 1e-12
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            sample_x = x + offsets * normal[0]
            sample_y = y + offsets * normal[1]
            profile = np.column_stack(
                [
                    ndimage.map_coordinates(
                        lab[..., channel],
                        [sample_y, sample_x],
                        order=1,
                        mode="nearest",
                    )
                    for channel in range(3)
                ]
            )
            core = np.median(profile[np.abs(offsets) <= 0.5], axis=0)
            negative_background = np.median(
                profile[
                    (offsets >= -float(profile_radius_px))
                    & (offsets <= -0.6 * float(profile_radius_px))
                ],
                axis=0,
            )
            positive_background = np.median(
                profile[
                    (offsets >= 0.6 * float(profile_radius_px))
                    & (offsets <= float(profile_radius_px))
                ],
                axis=0,
            )
            negative_contrast = float(
                np.linalg.norm(core - negative_background)
            )
            positive_contrast = float(
                np.linalg.norm(core - positive_background)
            )
            minimum_contrast = min(
                negative_contrast,
                positive_contrast,
            )
            if minimum_contrast < float(minimum_contrast_lab):
                continue

            membership = np.empty(len(offsets), dtype=np.float64)
            for side, background in (
                (offsets < 0.0, negative_background),
                (offsets >= 0.0, positive_background),
            ):
                distance_to_core = np.linalg.norm(
                    profile[side] - core.reshape(1, 3),
                    axis=1,
                )
                distance_to_background = np.linalg.norm(
                    profile[side] - background.reshape(1, 3),
                    axis=1,
                )
                membership[side] = distance_to_background / (
                    distance_to_core + distance_to_background + 1e-12
                )
            negative_radius = _crossing_radius(
                offsets,
                membership,
                sign=-1,
            )
            positive_radius = _crossing_radius(
                offsets,
                membership,
                sign=1,
            )
            if negative_radius is None or positive_radius is None:
                continue
            radius = 0.5 * (negative_radius + positive_radius)
            if not 0.25 <= radius <= float(maximum_radius_px):
                continue

            mean_contrast = 0.5 * (
                negative_contrast + positive_contrast
            )
            contrast_confidence = 1.0 - np.exp(
                -minimum_contrast / 5.0
            )
            background_agreement = np.exp(
                -float(
                    np.linalg.norm(
                        negative_background - positive_background
                    )
                )
                / max(2.0 * mean_contrast, 1e-12)
            )
            symmetry_confidence = np.exp(
                -abs(negative_radius - positive_radius)
                / max(negative_radius + positive_radius, 0.25)
            )
            raw_radius[index] = min(
                radius,
                float(maximum_depth_radius_ratio)
                * max(float(depth_radius[index]), 0.25),
            )
            raw_confidence[index] = float(
                np.clip(
                    contrast_confidence
                    * np.sqrt(
                        background_agreement * symmetry_confidence
                    ),
                    0.0,
                    1.0,
                )
            )
            normal_uv[index] = normal

        valid = np.flatnonzero(np.isfinite(raw_radius))
        if not len(valid):
            continue
        valid_tree = cKDTree(skeleton_uv[valid])
        smooth_radius = raw_radius.copy()
        smooth_confidence = raw_confidence.copy()
        for index, uv in enumerate(skeleton_uv):
            neighbors = valid_tree.query_ball_point(uv, r=4.0)
            if not neighbors:
                _distance, nearest = valid_tree.query(uv, k=1)
                neighbors = [int(nearest)]
            source = valid[np.asarray(neighbors, dtype=np.int64)]
            weights = np.maximum(raw_confidence[source], 1e-3)
            smooth_radius[index] = _weighted_median(
                raw_radius[source],
                weights,
            )
            smooth_confidence[index] = float(
                np.median(raw_confidence[source])
            )
            if not np.all(np.isfinite(normal_uv[index])):
                normal_uv[index] = normal_uv[source[np.argmax(weights)]]

        nearest_skeleton = cKDTree(skeleton_uv).query(
            np.column_stack((cx, cy)).astype(np.float64),
            k=1,
        )[1]
        local_radius_px = smooth_radius[nearest_skeleton]
        local_confidence = smooth_confidence[nearest_skeleton]
        local_normal = normal_uv[nearest_skeleton]
        delta_uv = (
            np.column_stack((cx, cy)).astype(np.float64)
            - skeleton_uv[nearest_skeleton]
        )
        across_px = np.abs(
            np.einsum("ij,ij->i", delta_uv, local_normal)
        )
        valid_pixel = (
            np.isfinite(local_radius_px)
            & np.isfinite(local_confidence)
            & np.all(np.isfinite(local_normal), axis=1)
            & (across_px <= local_radius_px)
        )
        chord_px = 2.0 * np.sqrt(
            np.maximum(
                local_radius_px * local_radius_px
                - across_px * across_px,
                0.0,
            )
        )
        chord_m = (
            chord_px
            * metric_depth[cy, cx]
            / max(float(focal_px), 1e-12)
        )
        chord_map[cy[valid_pixel], cx[valid_pixel]] = chord_m[valid_pixel]
        confidence_map[cy[valid_pixel], cx[valid_pixel]] = (
            local_confidence[valid_pixel]
        )
        radius_map_px[cy[valid_pixel], cx[valid_pixel]] = (
            local_radius_px[valid_pixel]
        )
        rows.append(
            {
                "component_id": int(component_id),
                "pixels": int(pixel_count),
                "skeleton_pixels": int(len(sx)),
                "observations_per_section": observations_per_section,
                "valid_width_sections": int(len(valid)),
                "valid_width_fraction": float(len(valid) / max(len(sx), 1)),
                "rgb_radius_p10_px": float(
                    np.percentile(raw_radius[valid], 10.0)
                ),
                "rgb_radius_median_px": float(
                    np.median(raw_radius[valid])
                ),
                "rgb_radius_p90_px": float(
                    np.percentile(raw_radius[valid], 90.0)
                ),
                "confidence_median": float(
                    np.median(raw_confidence[valid])
                ),
            }
        )

    return chord_map, confidence_map, radius_map_px, {
        "components": rows,
        "components_with_width": int(len(rows)),
        "estimated_pixels": int(np.isfinite(chord_map).sum()),
        "confidence_median": (
            float(np.median(confidence_map[confidence_map > 0.0]))
            if np.any(confidence_map > 0.0)
            else None
        ),
        "radius_median_px": (
            float(np.nanmedian(radius_map_px))
            if np.any(np.isfinite(radius_map_px))
            else None
        ),
    }

@dataclass(frozen=True)
class RGBDThinStructureResult:
    mesh: trimesh.Trimesh | None
    mask: np.ndarray
    metadata: dict[str, Any]

def _tube_defaults() -> dict[str, Any]:
    public_signature = inspect.signature(reconstruct_rgbd_only_mesh)
    public_defaults = {
        name: parameter.default
        for name, parameter in public_signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    private_signature = inspect.signature(_mesh_from_local_circle_tubes)
    return {
        name: public_defaults[name]
        for name in private_signature.parameters
        if name in public_defaults
    }

def _component_partition(
    mask: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    world: np.ndarray,
    *,
    camera_pos: Sequence[float],
    focal_px: float,
    axial_min_pixels: int,
    axial_ray_perp_p10: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
    list[dict[str, Any]],
]:
    labels, _yy, _xx, component_for_node = _depth_component_labels(
        mask,
        world_points=world,
        rgb=rgb,
        depth=depth,
        focal_px=float(focal_px),
    )
    component_sizes = np.bincount(component_for_node).astype(np.int64)
    axial_ids: list[int] = []
    rows: list[dict[str, Any]] = []
    for component_id, pixel_count in enumerate(component_sizes):
        component = labels == int(component_id)
        skeleton = skeletonize(component)
        sy, sx = np.nonzero(skeleton)
        if not len(sx):
            continue
        ray_perp_p10 = _initial_skeleton_ray_perp_p10(
            skeleton_y=sy,
            skeleton_x=sx,
            world=world,
            camera_pos=camera_pos,
            tangent_neighborhood_px=10.0,
        )
        axial = bool(
            int(pixel_count) >= int(axial_min_pixels)
            and ray_perp_p10 is not None
            and float(ray_perp_p10) < float(axial_ray_perp_p10)
        )
        if axial:
            axial_ids.append(int(component_id))
        rows.append(
            {
                "component_id": int(component_id),
                "pixels": int(pixel_count),
                "skeleton_pixels": int(len(sx)),
                "ray_perp_p10": (
                    None
                    if ray_perp_p10 is None
                    else float(ray_perp_p10)
                ),
                "finite_tube_selected": axial,
            }
        )
    axial_mask = np.isin(labels, np.asarray(axial_ids, dtype=np.int64))
    return labels, component_sizes, axial_mask, axial_ids, rows

def _finite_tube_mesh(
    axial_mask: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera: dict[str, Any],
    voxel_mm: float,
    radius_bias_px: float,
    radius_min_mm: float,
    radius_max_mm: float,
    curvature_radius_blend: float,
    center_direction_blend: float,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    yy, xx = np.nonzero(axial_mask)
    if not len(xx):
        raise ValueError("finite tube requires a nonempty axial mask")
    options = _tube_defaults()
    options.update(
        {
            **camera,
            "voxel_mm": float(voxel_mm),
            "click_u": int(np.median(xx)),
            "click_v": int(np.median(yy)),
            "component_policy": "all",
            "tube_meshing_mode": "voxel_union",
            "circle_radius_scale": 1.0,
            "circle_radius_bias_px": float(radius_bias_px),
            "circle_radius_min_mm": float(radius_min_mm),
            "circle_radius_max_mm": float(radius_max_mm),
            "circle_adaptive_radius_bias": False,
            "circle_direct_radius_blend": 0.0,
            "circle_direct_center_blend": 0.0,
            "circle_adaptive_direct_center_blend": False,
            "circle_center_fit_blend": float(center_direction_blend),
            "circle_front_anchored_radius_blend": 0.0,
            "circle_graph_radius_smoothing_gcv": False,
            "circle_chain_geometry_smoothing_gcv": False,
            "circle_global_surface_fit": False,
            "curvature_radius_blend": float(curvature_radius_blend),
            "curvature_min_valid_fits": 8,
            "local_cylinder_nonlinear_refine": False,
        }
    )
    mesh, metadata = _mesh_from_local_circle_tubes(
        SimpleNamespace(selected_mask=axial_mask),
        rgb,
        depth,
        **options,
    )
    return mesh, {
        **metadata,
        "finite_tube_generated": True,
    }

def _tube_ray_chords(
    tube_mesh: trimesh.Trimesh,
    axial_mask: np.ndarray,
    world: np.ndarray,
    camera_pos: Sequence[float],
    *,
    entry_tolerance_mm: float,
    maximum_chord_mm: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ay, ax = np.nonzero(axial_mask)
    surface = np.asarray(world[ay, ax], dtype=np.float64)
    camera_world = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    directions = surface - camera_world.reshape(1, 3)
    surface_range = np.linalg.norm(directions, axis=1)
    directions /= surface_range[:, None] + 1e-12
    origins = np.repeat(camera_world.reshape(1, 3), len(surface), axis=0)
    locations, ray_indices, _triangle_indices = (
        tube_mesh.ray.intersects_location(
            origins,
            directions,
            multiple_hits=True,
        )
    )
    hit_range = np.linalg.norm(
        locations - origins[ray_indices],
        axis=1,
    )
    chord_map = np.full(axial_mask.shape, np.nan, dtype=np.float64)
    confidence_map = np.zeros(axial_mask.shape, dtype=np.float64)
    tolerance_m = float(entry_tolerance_mm) / 1000.0
    maximum_chord_m = float(maximum_chord_mm) / 1000.0
    accepted_entry_residuals: list[float] = []
    accepted_chords: list[float] = []

    for local_ray in np.unique(ray_indices):
        selected_hits = np.sort(hit_range[ray_indices == int(local_ray)])
        if len(selected_hits) < 2:
            continue
        observed = float(surface_range[int(local_ray)])
        candidates: list[tuple[float, float]] = []
        for hit_index in range(0, len(selected_hits) - 1, 2):
            entry = float(selected_hits[hit_index])
            exit_range = float(selected_hits[hit_index + 1])
            chord = exit_range - observed
            if 0.0 < chord <= maximum_chord_m:
                candidates.append((abs(entry - observed), chord))
        if not candidates:
            continue
        entry_residual, chord = min(candidates)
        if entry_residual > tolerance_m:
            continue
        y = int(ay[int(local_ray)])
        x = int(ax[int(local_ray)])
        chord_map[y, x] = chord
        confidence_map[y, x] = float(
            np.exp(
                -0.5
                * (entry_residual / max(tolerance_m, 1e-12)) ** 2
            )
        )
        accepted_entry_residuals.append(entry_residual)
        accepted_chords.append(chord)

    return chord_map, confidence_map, {
        "candidate_rays": int(len(surface)),
        "ray_intersections": int(len(locations)),
        "accepted_rays": int(len(accepted_chords)),
        "accepted_fraction": float(
            len(accepted_chords) / max(len(surface), 1)
        ),
        "entry_tolerance_mm": float(entry_tolerance_mm),
        "entry_residual_median_mm": (
            float(np.median(accepted_entry_residuals) * 1000.0)
            if accepted_entry_residuals
            else None
        ),
        "chord_median_mm": (
            float(np.median(accepted_chords) * 1000.0)
            if accepted_chords
            else None
        ),
        "chord_p90_mm": (
            float(np.percentile(accepted_chords, 90.0) * 1000.0)
            if accepted_chords
            else None
        ),
        "chord_max_mm": (
            float(np.max(accepted_chords) * 1000.0)
            if accepted_chords
            else None
        ),
    }

def _apply_rgb_width_prior(
    mask: np.ndarray,
    labels: np.ndarray,
    axial_ids: Sequence[int],
    axial_mask: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    chord_map: np.ndarray,
    confidence_map: np.ndarray,
    *,
    camera: dict[str, Any],
    tube_radius_bias_px: float,
    confidence_scale: float,
    minimum_observations_per_section: float,
    shape_factor_minimum: float,
    shape_factor_maximum: float,
) -> dict[str, Any]:
    (
        rgb_chord,
        rgb_confidence,
        rgb_radius_px,
        rgb_metadata,
    ) = rgb_profile_chord_maps(
        mask,
        rgb,
        depth,
        camera=camera,
        minimum_observations_per_section=float(
            minimum_observations_per_section
        ),
    )
    rgb_confidence = np.clip(
        rgb_confidence * float(confidence_scale),
        0.0,
        1.0,
    )
    shape_rows: list[dict[str, Any]] = []
    for component_id in axial_ids:
        component = labels == int(component_id)
        shape_use = (
            component
            & np.isfinite(chord_map)
            & (chord_map > 0.0)
            & (confidence_map > 0.0)
            & np.isfinite(rgb_chord)
            & (rgb_chord > 0.0)
            & (rgb_confidence > 0.0)
        )
        if not shape_use.any():
            continue
        distance_to_edge = ndimage.distance_transform_edt(component)
        sy, sx = np.nonzero(skeletonize(component))
        if not len(sx):
            continue
        shape_y, shape_x = np.nonzero(shape_use)
        nearest_skeleton = cKDTree(
            np.column_stack((sx, sy)).astype(np.float64)
        ).query(
            np.column_stack((shape_x, shape_y)).astype(np.float64),
            k=1,
        )[1]
        depth_radius_px = np.maximum(
            distance_to_edge[sy, sx][nearest_skeleton]
            - float(tube_radius_bias_px),
            0.25,
        )
        log_signal = np.log(
            rgb_radius_px[shape_use] / depth_radius_px
        )
        center_log_signal = float(np.median(log_signal))
        normalized_factor = np.clip(
            np.exp(log_signal - center_log_signal),
            float(shape_factor_minimum),
            float(shape_factor_maximum),
        )
        shape_confidence = (
            rgb_confidence[shape_use] * confidence_map[shape_use]
        )
        applied_factor = np.exp(
            shape_confidence * np.log(normalized_factor)
        )
        chord_map[shape_use] *= applied_factor
        shape_rows.append(
            {
                "component_id": int(component_id),
                "adjusted_pixels": int(shape_use.sum()),
                "raw_rgb_depth_radius_ratio_median": float(
                    np.exp(center_log_signal)
                ),
                "applied_factor_p10": float(
                    np.percentile(applied_factor, 10.0)
                ),
                "applied_factor_median": float(
                    np.median(applied_factor)
                ),
                "applied_factor_p90": float(
                    np.percentile(applied_factor, 90.0)
                ),
            }
        )

    transverse_mask = mask & (~axial_mask)
    transverse_use = (
        transverse_mask
        & np.isfinite(rgb_chord)
        & (rgb_chord > 0.0)
        & (rgb_confidence > 0.0)
    )
    chord_map[transverse_use] = rgb_chord[transverse_use]
    confidence_map[transverse_use] = rgb_confidence[transverse_use]
    return {
        **rgb_metadata,
        "enabled": True,
        "confidence_scale": float(confidence_scale),
        "transverse_accepted_pixels": int(transverse_use.sum()),
        "axial_shape_mode": "component_normalized_rgb_depth_radius_ratio",
        "axial_shape_components": shape_rows,
    }

def reconstruct_rgbd_thin_structure_mesh(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    axial_ray_perp_p10: float = 0.4,
    axial_min_pixels: int = 20,
    tube_voxel_mm: float = 1.25,
    tube_radius_bias_px: float = 0.5,
    tube_radius_min_mm: float = 0.4,
    tube_radius_max_mm: float = 8.0,
    tube_curvature_radius_blend: float = 0.5,
    tube_center_direction_blend: float = 1.0,
    ray_voxel_mm: float = 1.25,
    ray_footprint_scale: float = 0.55,
    ray_footprint_min_mm: float = 0.65,
    fusion_entry_tolerance_mm: float = 2.5,
    fusion_chord_max_mm: float = 120.0,
    rgb_width_confidence_scale: float = 1.5,
    rgb_width_min_observations_per_section: float = 2.0,
    rgb_width_axial_shape_min_factor: float = 2.0 / 3.0,
    rgb_width_axial_shape_max_factor: float = 1.5,
) -> RGBDThinStructureResult:
    """Reconstruct all detected thin structures without instance inputs."""
    color = np.asarray(rgb)
    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    camera = {
        "camera_pos": camera_pos,
        "camera_quat_xyzw": camera_quat_xyzw,
        "focal_length": float(focal_length),
        "horizontal_aperture": float(horizontal_aperture),
    }
    mask, _plane, initial_world, mask_metadata = _support_structure_mask(
        color,
        metric_depth,
        **camera,
    )
    if not mask.any():
        return RGBDThinStructureResult(
            mesh=None,
            mask=mask,
            metadata={
                "build": "rgbd_thin_structure_v13",
                "method": "finite_tube_rgb_width_ray_chord",
                **mask_metadata,
                "thin_mesh_generated": False,
                "forbidden_inputs_used": [],
            },
        )

    world = initial_world
    if world is None:
        world, _valid = backproject_depth_world(metric_depth, **camera)
    focal_px = (
        float(focal_length)
        / float(horizontal_aperture)
        * float(metric_depth.shape[1])
    )
    (
        labels,
        component_sizes,
        axial_mask,
        axial_ids,
        component_rows,
    ) = _component_partition(
        mask,
        color,
        metric_depth,
        world,
        camera_pos=camera_pos,
        focal_px=focal_px,
        axial_min_pixels=int(axial_min_pixels),
        axial_ray_perp_p10=float(axial_ray_perp_p10),
    )

    chord_map = np.full(mask.shape, np.nan, dtype=np.float64)
    confidence_map = np.zeros(mask.shape, dtype=np.float64)
    tube_metadata: dict[str, Any] = {
        "finite_tube_generated": False,
    }
    fusion_metadata: dict[str, Any] = {
        "candidate_rays": 0,
        "accepted_rays": 0,
    }
    if axial_mask.any():
        tube_mesh, tube_metadata = _finite_tube_mesh(
            axial_mask,
            color,
            metric_depth,
            camera=camera,
            voxel_mm=float(tube_voxel_mm),
            radius_bias_px=float(tube_radius_bias_px),
            radius_min_mm=float(tube_radius_min_mm),
            radius_max_mm=float(tube_radius_max_mm),
            curvature_radius_blend=float(
                tube_curvature_radius_blend
            ),
            center_direction_blend=float(
                tube_center_direction_blend
            ),
        )
        chord_map, confidence_map, fusion_metadata = _tube_ray_chords(
            tube_mesh,
            axial_mask,
            world,
            camera_pos,
            entry_tolerance_mm=float(fusion_entry_tolerance_mm),
            maximum_chord_mm=float(fusion_chord_max_mm),
        )

    rgb_width_metadata = _apply_rgb_width_prior(
        mask,
        labels,
        axial_ids,
        axial_mask,
        color,
        metric_depth,
        chord_map,
        confidence_map,
        camera=camera,
        tube_radius_bias_px=float(tube_radius_bias_px),
        confidence_scale=float(rgb_width_confidence_scale),
        minimum_observations_per_section=float(
            rgb_width_min_observations_per_section
        ),
        shape_factor_minimum=float(
            rgb_width_axial_shape_min_factor
        ),
        shape_factor_maximum=float(
            rgb_width_axial_shape_max_factor
        ),
    )
    ray_options = _ray_defaults()
    ray_options.update(
        {
            **camera,
            "voxel_mm": float(ray_voxel_mm),
            "footprint_scale": float(ray_footprint_scale),
            "footprint_min_mm": float(ray_footprint_min_mm),
            "component_policy": "all",
            "chord_model": "silhouette",
            "chord_max_mm": float(fusion_chord_max_mm),
            "ray_meshing_mode": "sdf",
            "ray_primitive_mode": "capsule",
            "external_chord_map": chord_map,
            "external_chord_confidence_map": confidence_map,
            "external_chord_blend": 1.0,
        }
    )
    mesh, ray_metadata = _reconstruct_mask(
        mask,
        color,
        metric_depth,
        ray_options,
    )
    metadata = {
        "build": "rgbd_thin_structure_v13",
        "method": "finite_tube_rgb_width_ray_chord",
        **mask_metadata,
        "component_count": int(len(component_sizes)),
        "component_diagnostics": component_rows,
        "axial_component_ids": axial_ids,
        "axial_component_count": int(len(axial_ids)),
        "axial_pixels": int(axial_mask.sum()),
        "transverse_pixels": int((mask & (~axial_mask)).sum()),
        "tube": tube_metadata,
        "finite_tube_chord_fusion": fusion_metadata,
        "rgb_width": rgb_width_metadata,
        "ray_chord": ray_metadata,
        "thin_mesh_generated": True,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_volume_cm3": float(abs(mesh.volume) * 1e6),
        "allowed_inputs": [
            "rgb",
            "metric_depth",
            "camera_intrinsics",
            "camera_extrinsics",
        ],
        "used_click": False,
        "used_segmentation": False,
        "used_object_identity": False,
        "forbidden_inputs_used": [],
    }
    return RGBDThinStructureResult(
        mesh=mesh,
        mask=mask,
        metadata=metadata,
    )

__all__ = ["RGBDThinStructureResult", "reconstruct_rgbd_thin_structure_mesh"]
