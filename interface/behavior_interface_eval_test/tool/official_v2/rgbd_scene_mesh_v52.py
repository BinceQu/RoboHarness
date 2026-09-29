"""Runtime-only RGB-D scene reconstruction using representative V52.

V52 starts from the frozen V13 structural-hybrid reconstruction and applies
the canonical boundary-uncertainty expert only when its RGB-D applicability
gate succeeds. No click, segmentation, object identity, evaluator samples, or
reference geometry are accepted by this module.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import interpolate, ndimage
from scipy.spatial import cKDTree
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
import trimesh

from .rgbd_scene_components import _continuous_component_labels
from .rgbd_scene_mesh_v12 import (
    RGBDSceneMeshResult,
    _boundary_directed_edges,
    _continuous_edge,
    _orient_front_triangles,
    _quat_to_mat_xyzw,
    _split_nonmanifold_vertex_fans,
    _support_structure_mask,
)
from .rgbd_scene_mesh_v13 import (
    reconstruct_rgbd_scene_mesh as _reconstruct_v13_scene_mesh,
)


MESH_VERSION = "v52"
RGBD_BOUNDARY_UNCERTAINTY_BAND_BUILD = (
    "rgbd_boundary_uncertainty_band_v52"
)
RGBD_SCENE_MESH_BUILD = "rgbd_representative_v52_runtime"
V13_RECONSTRUCTION_KWARGS: Dict[str, Any] = {
    "completion_method": "structural_hybrid",
    "pixel_stride": 3,
    "back_extrusion_mm": 60.0,
    "surface_edge_scale": 3.5,
    "surface_edge_slack_mm": 2.0,
    "max_depth_jump_mm": 35.0,
}

def _supported_low_profile_layers(
    rgb: np.ndarray,
    depth: np.ndarray,
    world: np.ndarray,
    valid: np.ndarray,
    plane: Any,
    *,
    focal_px: float,
    minimum_pixels: int = 1000,
    minimum_extent_mm: float = 40.0,
    maximum_extent_mm: float = 500.0,
    minimum_lower_height_mm: float = 20.0,
    maximum_upper_height_mm: float = 100.0,
    maximum_height_to_width: float = 0.30,
    maximum_axis_ratio: float = 1.50,
    boundary_margin_px: int = 4,
) -> tuple[list[np.ndarray], list[Dict[str, Any]]]:
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
    image_height, image_width = candidate.shape
    masks: list[np.ndarray] = []
    rows: list[Dict[str, Any]] = []
    origin = np.asarray(plane.origin, dtype=np.float64)
    basis_x = np.asarray(plane.basis_x, dtype=np.float64)
    basis_y = np.asarray(plane.basis_y, dtype=np.float64)
    for component_id in np.flatnonzero(
        sizes >= int(minimum_pixels)
    ):
        component_mask = labels == int(component_id)
        yy, xx = np.nonzero(component_mask)
        points = np.asarray(world[component_mask], dtype=np.float64)
        heights = np.asarray(
            signed_height[component_mask],
            dtype=np.float64,
        )
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
        axis_ratio = horizontal_max / max(horizontal_min, 1e-9)
        height_to_width = height_extent / max(horizontal_min, 1e-9)
        fully_observed = bool(
            int(xx.min()) > int(boundary_margin_px)
            and int(yy.min()) > int(boundary_margin_px)
            and int(xx.max())
            < image_width - 1 - int(boundary_margin_px)
            and int(yy.max())
            < image_height - 1 - int(boundary_margin_px)
        )
        eligible = bool(
            fully_observed
            and horizontal_min
            >= float(minimum_extent_mm) / 1000.0
            and horizontal_max
            <= float(maximum_extent_mm) / 1000.0
            and h_low >= float(minimum_lower_height_mm) / 1000.0
            and h_high <= float(maximum_upper_height_mm) / 1000.0
            and height_to_width <= float(maximum_height_to_width)
            and axis_ratio <= float(maximum_axis_ratio)
        )
        rows.append(
            {
                "component_id": int(component_id),
                "pixel_count": int(component_mask.sum()),
                "image_bbox_uv": [
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max()),
                    int(yy.max()),
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
                "height_to_min_width": height_to_width,
                "axis_ratio": axis_ratio,
                "eligible": eligible,
            }
        )
        if eligible:
            masks.append(component_mask)
    return masks, rows

def _plane_xy(
    points: np.ndarray,
    plane: Any,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    relative = values - np.asarray(plane.origin, dtype=np.float64)
    return np.column_stack(
        (
            relative @ np.asarray(plane.basis_x, dtype=np.float64),
            relative @ np.asarray(plane.basis_y, dtype=np.float64),
        )
    )

def _fit_boundary_circle(
    world: np.ndarray,
    mask: np.ndarray,
    plane: Any,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    boundary = np.asarray(mask, dtype=bool) ^ ndimage.binary_erosion(
        np.asarray(mask, dtype=bool),
        iterations=2,
    )
    values = _plane_xy(np.asarray(world)[boundary], plane)
    if len(values) < 24:
        raise ValueError("radial layer boundary has insufficient pixels")
    keep = np.ones(len(values), dtype=bool)
    center = np.median(values, axis=0)
    radius = float(
        np.median(np.linalg.norm(values - center.reshape(1, 2), axis=1))
    )
    for _iteration in range(6):
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
        next_keep = residual <= float(
            np.quantile(residual[keep], 0.75)
        )
        if int(next_keep.sum()) < 24 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    residual = np.abs(
        np.linalg.norm(values - center.reshape(1, 2), axis=1)
        - radius
    )
    return center, radius, {
        "center_xy_m": center.tolist(),
        "radius_mm": radius * 1000.0,
        "boundary_pixels": int(len(values)),
        "fit_inlier_pixels": int(keep.sum()),
        "residual_percentiles_mm": (
            np.percentile(residual, [25.0, 50.0, 75.0, 90.0])
            * 1000.0
        ).tolist(),
    }

def _polar_height_evidence(
    world: np.ndarray,
    mask: np.ndarray,
    plane: Any,
    *,
    center_xy: np.ndarray,
    radius_m: float,
    angular_samples: int = 720,
    radial_samples: int = 120,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    points = np.asarray(world)[np.asarray(mask, dtype=bool)]
    xy = _plane_xy(points, plane)
    height = np.asarray(
        plane.signed_height(points),
        dtype=np.float64,
    )
    theta = np.linspace(
        -np.pi,
        np.pi,
        int(angular_samples),
        endpoint=False,
    )
    radius = np.linspace(
        0.08 * float(radius_m),
        0.92 * float(radius_m),
        int(radial_samples),
    )
    theta_grid, radius_grid = np.meshgrid(theta, radius)
    query = np.column_stack(
        (
            float(center_xy[0])
            + radius_grid.ravel() * np.cos(theta_grid.ravel()),
            float(center_xy[1])
            + radius_grid.ravel() * np.sin(theta_grid.ravel()),
        )
    )
    polar = interpolate.LinearNDInterpolator(
        xy,
        height,
        fill_value=np.nan,
    )(query).reshape(len(radius), len(theta))
    nearest = interpolate.NearestNDInterpolator(
        xy,
        height,
    )(query).reshape(len(radius), len(theta))
    polar = np.where(np.isfinite(polar), polar, nearest)
    polar = ndimage.gaussian_filter(
        polar,
        sigma=(1.2, 0.7),
        mode="wrap",
    )

    broad = ndimage.percentile_filter(
        polar,
        percentile=65.0,
        size=(1, 25),
        mode="wrap",
    )
    valley_mm = np.clip((broad - polar) * 1000.0, 0.0, 3.0)
    score = (
        0.55 * np.quantile(valley_mm, 0.40, axis=0)
        + 0.30 * np.quantile(valley_mm, 0.60, axis=0)
        + 0.15 * np.mean(valley_mm > 0.35, axis=0)
    )
    score = ndimage.gaussian_filter1d(
        score,
        sigma=1.5,
        mode="wrap",
    )
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    normalized = (score - median) / max(mad, 1e-6)
    return theta, normalized, {
        "angular_samples": int(len(theta)),
        "radial_samples": int(len(radius)),
        "radial_range_mm": [
            float(radius[0] * 1000.0),
            float(radius[-1] * 1000.0),
        ],
        "score_percentiles": np.percentile(
            normalized,
            [0.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0],
        ).tolist(),
    }

def _periodic_radial_structure(
    theta: np.ndarray,
    score: np.ndarray,
    *,
    minimum_sectors: int = 3,
    maximum_sectors: int = 16,
) -> dict[str, Any]:
    angles = np.asarray(theta, dtype=np.float64)
    evidence = np.asarray(score, dtype=np.float64)
    sample_count = int(len(evidence))
    local_half_width = max(2, int(round(sample_count / 120.0)))
    rows: list[dict[str, Any]] = []
    for sectors in range(int(minimum_sectors), int(maximum_sectors) + 1):
        step = float(sample_count) / float(sectors)
        best: tuple[float, float, np.ndarray, np.ndarray] | None = None
        for phase in np.linspace(0.0, step, 160, endpoint=False):
            values: list[float] = []
            indices: list[int] = []
            for sector_index in range(sectors):
                center = int(round(phase + sector_index * step))
                window = (
                    np.arange(
                        center - local_half_width,
                        center + local_half_width + 1,
                    )
                    % sample_count
                )
                selected = int(window[np.argmax(evidence[window])])
                values.append(float(evidence[selected]))
                indices.append(selected)
            ray_values = np.asarray(values, dtype=np.float64)
            quality = float(
                0.55 * np.mean(np.clip(ray_values, 0.0, None))
                + 0.45 * np.quantile(ray_values, 0.25)
                - 0.08 * float(sectors)
            )
            candidate = (
                quality,
                float(phase),
                ray_values,
                np.asarray(indices, dtype=np.int64),
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
        assert best is not None
        quality, phase, ray_values, indices = best
        rows.append(
            {
                "sectors": int(sectors),
                "quality": float(quality),
                "phase_sample": float(phase),
                "ray_values": ray_values.tolist(),
                "ray_angles_deg": (
                    np.degrees(angles[indices]) % 360.0
                ).tolist(),
                "minimum_ray_value": float(ray_values.min()),
            }
        )

    strongest = max(rows, key=lambda row: float(row["quality"]))
    quality_floor = 0.88 * float(strongest["quality"])
    evidence_floor = max(
        1.5,
        float(np.percentile(evidence, 75.0)),
    )
    credible = [
        row
        for row in rows
        if float(row["quality"]) >= quality_floor
        and float(row["minimum_ray_value"]) >= evidence_floor
    ]
    selected = (
        max(credible, key=lambda row: int(row["sectors"]))
        if credible
        else strongest
    )
    selected = dict(selected)
    selected["credible"] = bool(credible)
    selected["strongest_quality"] = float(strongest["quality"])
    selected["quality_ratio"] = float(
        float(selected["quality"])
        / max(float(strongest["quality"]), 1e-9)
    )
    selected["evidence_floor"] = evidence_floor
    selected["candidate_fits"] = rows
    return selected

def _support_prominence(
    world: np.ndarray,
    mask: np.ndarray,
    plane: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = np.asarray(mask, dtype=bool)
    height = np.full(selected.shape, np.nan, dtype=np.float64)
    height[selected] = np.asarray(
        plane.signed_height(np.asarray(world)[selected]),
        dtype=np.float64,
    )
    _distance, nearest = ndimage.distance_transform_edt(
        ~selected,
        return_indices=True,
    )
    filled = height.copy()
    filled[~selected] = height[
        nearest[0, ~selected],
        nearest[1, ~selected],
    ]
    equivalent_radius_px = float(
        np.sqrt(float(selected.sum()) / np.pi)
    )
    filter_radius_px = max(
        3,
        int(round(0.08 * equivalent_radius_px)),
    )
    base = ndimage.percentile_filter(
        filled,
        percentile=25.0,
        size=2 * filter_radius_px + 1,
        mode="nearest",
    )
    base = ndimage.gaussian_filter(
        base,
        sigma=max(0.75, 0.15 * filter_radius_px),
        mode="nearest",
    )
    prominence = np.maximum(height - base, 0.0)
    values = prominence[selected]
    return prominence, {
        "equivalent_radius_px": equivalent_radius_px,
        "filter_radius_px": int(filter_radius_px),
        "prominence_percentiles_mm": (
            np.percentile(
                values,
                [0.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0],
            )
            * 1000.0
        ).tolist(),
    }

def _grid_observed_top(
    xy: np.ndarray,
    height: np.ndarray,
    *,
    pitch_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    x_low, x_high = np.percentile(xy[:, 0], [0.1, 99.9])
    y_low, y_high = np.percentile(xy[:, 1], [0.1, 99.9])
    x_min = float(np.floor((x_low - 2.0 * pitch_m) / pitch_m) * pitch_m)
    y_min = float(np.floor((y_low - 2.0 * pitch_m) / pitch_m) * pitch_m)
    x_count = int(
        np.ceil((x_high - x_min + 2.0 * pitch_m) / pitch_m)
    ) + 1
    y_count = int(
        np.ceil((y_high - y_min + 2.0 * pitch_m) / pitch_m)
    ) + 1
    ix = np.clip(
        np.rint((xy[:, 0] - x_min) / pitch_m).astype(np.int64),
        0,
        x_count - 1,
    )
    iy = np.clip(
        np.rint((xy[:, 1] - y_min) / pitch_m).astype(np.int64),
        0,
        y_count - 1,
    )
    top = np.full((x_count, y_count), -np.inf, dtype=np.float64)
    np.maximum.at(top, (ix, iy), height)
    observed = np.isfinite(top)
    footprint = ndimage.binary_closing(
        observed,
        structure=np.ones((3, 3), dtype=bool),
        iterations=2,
    )
    footprint = ndimage.binary_fill_holes(footprint)
    if int(footprint.sum()) < 100:
        raise ValueError("radial wedge footprint is empty")
    _distance, nearest = ndimage.distance_transform_edt(
        ~observed,
        return_indices=True,
    )
    top[footprint & (~observed)] = top[
        nearest[0, footprint & (~observed)],
        nearest[1, footprint & (~observed)],
    ]
    return top, footprint, x_min, y_min

def _angular_sidewall_thickness(
    *,
    rgb: np.ndarray,
    world: np.ndarray,
    valid: np.ndarray,
    layer_mask: np.ndarray,
    plane: Any,
    center_xy: np.ndarray,
    radius_m: float,
    seam_angles_deg: list[float],
    base_top: np.ndarray,
    footprint: np.ndarray,
    grid_xx: np.ndarray,
    grid_yy: np.ndarray,
    global_outer_thickness_m: float,
    pixel_footprint_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    boundaries = np.sort(
        np.asarray(seam_angles_deg, dtype=np.float64) % 360.0
    )
    if len(boundaries) < 3:
        raise ValueError("sidewall profile requires radial sectors")
    extended = np.concatenate((boundaries, boundaries[:1] + 360.0))
    sector_centers_deg = (
        boundaries + 0.5 * (extended[1:] - boundaries)
    ) % 360.0

    origin = np.asarray(plane.origin, dtype=np.float64)
    basis_x = np.asarray(plane.basis_x, dtype=np.float64)
    basis_y = np.asarray(plane.basis_y, dtype=np.float64)
    normal = np.asarray(plane.normal, dtype=np.float64)
    relative = np.asarray(world, dtype=np.float64) - origin
    image_x = relative @ basis_x
    image_y = relative @ basis_y
    image_height = relative @ normal
    image_radius = np.hypot(
        image_x - float(center_xy[0]),
        image_y - float(center_xy[1]),
    )
    image_angle = (
        np.degrees(
            np.arctan2(
                image_y - float(center_xy[1]),
                image_x - float(center_xy[0]),
            )
        )
        % 360.0
    )
    image_sector = np.searchsorted(
        extended,
        image_angle,
        side="right",
    ) - 1
    image_sector %= len(boundaries)

    selected = np.asarray(layer_mask, dtype=bool)
    distance_px, nearest = ndimage.distance_transform_edt(
        ~selected,
        return_indices=True,
    )
    nearest_y, nearest_x = nearest
    nearest_height = image_height[nearest_y, nearest_x]
    nearest_world = np.asarray(world)[nearest_y, nearest_x]
    world_gap = np.linalg.norm(
        np.asarray(world) - nearest_world,
        axis=2,
    )
    lab = rgb2lab(np.asarray(rgb, dtype=np.float64) / 255.0)
    color_gap = np.linalg.norm(
        lab - lab[nearest_y, nearest_x],
        axis=2,
    )
    equivalent_radius_px = float(
        np.sqrt(float(selected.sum()) / np.pi)
    )
    adjacency_px = max(
        3.0,
        0.12 * equivalent_radius_px,
    )
    layer_low = float(
        np.percentile(image_height[selected], 0.5)
    )
    sidewall = (
        np.asarray(valid, dtype=bool)
        & (~selected)
        & (distance_px <= adjacency_px)
        & (world_gap <= 0.18 * float(radius_m))
        & (image_height >= 0.5 * pixel_footprint_m)
        & (
            image_height
            < nearest_height - 0.5 * pixel_footprint_m
        )
        & (
            image_height
            < layer_low + pixel_footprint_m
        )
        & (color_gap <= 45.0)
        & (image_radius >= 0.70 * float(radius_m))
        & (image_radius <= 1.15 * float(radius_m))
    )

    grid_radius = np.hypot(
        grid_xx - float(center_xy[0]),
        grid_yy - float(center_xy[1]),
    )
    grid_angle = (
        np.degrees(
            np.arctan2(
                grid_yy - float(center_xy[1]),
                grid_xx - float(center_xy[0]),
            )
        )
        % 360.0
    )
    grid_sector = np.searchsorted(
        extended,
        grid_angle,
        side="right",
    ) - 1
    grid_sector %= len(boundaries)

    observed_rows: list[dict[str, Any]] = []
    for sector_index in range(len(boundaries)):
        outer = (
            footprint
            & (grid_sector == int(sector_index))
            & (grid_radius >= 0.72 * float(radius_m))
        )
        lower = image_height[
            sidewall & (image_sector == int(sector_index))
        ]
        if int(outer.sum()) < 16 or len(lower) < 32:
            continue
        outer_base = float(np.median(base_top[outer]))
        lower_height = float(np.percentile(lower, 75.0))
        observed_thickness = outer_base - lower_height
        if not (
            2.0 * pixel_footprint_m
            <= observed_thickness
            <= 1.5 * float(global_outer_thickness_m)
        ):
            continue
        observed_rows.append(
            {
                "sector_index": int(sector_index),
                "sector_center_deg": float(
                    sector_centers_deg[sector_index]
                ),
                "sidewall_pixels": int(len(lower)),
                "outer_base_height_mm": outer_base * 1000.0,
                "lower_height_q75_mm": lower_height * 1000.0,
                "observed_thickness_mm": (
                    observed_thickness * 1000.0
                ),
            }
        )
    if len(observed_rows) < 3:
        raise ValueError("visible sidewall arc has insufficient sectors")
    observed_angle = np.deg2rad(
        np.asarray(
            [row["sector_center_deg"] for row in observed_rows],
            dtype=np.float64,
        )
    )
    observed_thickness = (
        np.asarray(
            [row["observed_thickness_mm"] for row in observed_rows],
            dtype=np.float64,
        )
        / 1000.0
    )
    design = np.column_stack(
        (
            np.ones(len(observed_angle)),
            np.cos(observed_angle),
            np.sin(observed_angle),
        )
    )
    coefficient, *_ = np.linalg.lstsq(
        design,
        observed_thickness,
        rcond=None,
    )
    all_angle = np.deg2rad(sector_centers_deg)
    profile = np.column_stack(
        (
            np.ones(len(all_angle)),
            np.cos(all_angle),
            np.sin(all_angle),
        )
    ) @ coefficient
    profile = np.clip(
        profile,
        2.0 * pixel_footprint_m,
        1.5 * float(global_outer_thickness_m),
    )
    return profile, grid_sector, {
        "sector_boundaries_deg": boundaries.tolist(),
        "sector_centers_deg": sector_centers_deg.tolist(),
        "visible_sectors": observed_rows,
        "first_harmonic_coefficients_mm": (
            coefficient * 1000.0
        ).tolist(),
        "completed_outer_thickness_mm": (
            profile * 1000.0
        ).tolist(),
        "sidewall_candidate_pixels": int(sidewall.sum()),
        "adjacency_px": float(adjacency_px),
    }

def _radial_wedge_mesh(
    world: np.ndarray,
    mask: np.ndarray,
    plane: Any,
    *,
    rgb: np.ndarray,
    valid: np.ndarray,
    center_xy: np.ndarray,
    radius_m: float,
    seam_angles_deg: list[float],
    voxel_mm: float,
    radial_power: float,
    outer_thickness_scale: float,
    center_thickness_pixels: float,
    base_percentile: float,
    seam_gap_pixels: float,
    sectorwise_base: bool,
    sidewall_profile_weight: float,
    median_depth_m: float,
    focal_px: float,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    points = np.asarray(world)[np.asarray(mask, dtype=bool)]
    xy = _plane_xy(points, plane)
    height = np.asarray(
        plane.signed_height(points),
        dtype=np.float64,
    )
    pitch_m = float(voxel_mm) / 1000.0
    if pitch_m <= 0.0:
        raise ValueError("radial wedge voxel size must be positive")
    top, footprint, x_min, y_min = _grid_observed_top(
        xy,
        height,
        pitch_m=pitch_m,
    )
    x_count, y_count = footprint.shape
    grid_x = x_min + np.arange(x_count, dtype=np.float64) * pitch_m
    grid_y = y_min + np.arange(y_count, dtype=np.float64) * pitch_m
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="ij")
    normalized_radius = np.clip(
        np.hypot(
            xx - float(center_xy[0]),
            yy - float(center_xy[1]),
        )
        / max(float(radius_m), pitch_m),
        0.0,
        1.25,
    )

    filled_top = top.copy()
    finite = np.isfinite(filled_top)
    _distance, nearest = ndimage.distance_transform_edt(
        ~finite,
        return_indices=True,
    )
    filled_top[~finite] = filled_top[
        nearest[0, ~finite],
        nearest[1, ~finite],
    ]
    footprint_radius_cells = max(
        3,
        int(round(0.08 * float(radius_m) / pitch_m)),
    )
    filter_size = 2 * footprint_radius_cells + 1
    if bool(sectorwise_base) and len(seam_angles_deg) >= 3:
        angle_grid = np.degrees(
            np.arctan2(
                yy - float(center_xy[1]),
                xx - float(center_xy[0]),
            )
        ) % 360.0
        boundaries = np.sort(
            np.asarray(seam_angles_deg, dtype=np.float64) % 360.0
        )
        extended = np.concatenate((boundaries, boundaries[:1] + 360.0))
        sector_index = np.searchsorted(
            extended,
            angle_grid,
            side="right",
        ) - 1
        sector_index %= len(boundaries)
        base_top = np.empty_like(filled_top)
        for index in range(len(boundaries)):
            sector = footprint & (sector_index == int(index))
            if int(sector.sum()) < 16:
                base_top[sector] = filled_top[sector]
                continue
            sector_values = np.full_like(filled_top, np.nan)
            sector_values[sector] = filled_top[sector]
            finite_sector = np.isfinite(sector_values)
            _sector_distance, sector_nearest = (
                ndimage.distance_transform_edt(
                    ~finite_sector,
                    return_indices=True,
                )
            )
            sector_filled = sector_values.copy()
            sector_filled[~finite_sector] = sector_values[
                sector_nearest[0, ~finite_sector],
                sector_nearest[1, ~finite_sector],
            ]
            sector_base = ndimage.percentile_filter(
                sector_filled,
                percentile=float(base_percentile),
                size=filter_size,
                mode="nearest",
            )
            sector_base = ndimage.gaussian_filter(
                sector_base,
                sigma=max(0.75, 0.15 * footprint_radius_cells),
                mode="nearest",
            )
            base_top[sector] = sector_base[sector]
        base_top[~footprint] = filled_top[~footprint]
    else:
        base_top = ndimage.percentile_filter(
            filled_top,
            percentile=float(base_percentile),
            size=filter_size,
            mode="nearest",
        )
        base_top = ndimage.gaussian_filter(
            base_top,
            sigma=max(0.75, 0.15 * footprint_radius_cells),
            mode="nearest",
        )

    pixel_footprint_m = float(median_depth_m) / float(focal_px)
    h_low, h_high = np.percentile(height, [0.5, 99.5])
    visible_height_m = max(float(h_high - h_low), pixel_footprint_m)
    center_thickness_m = max(
        pitch_m,
        float(center_thickness_pixels) * pixel_footprint_m,
    )
    outer_thickness_m = max(
        center_thickness_m + pitch_m,
        float(outer_thickness_scale) * visible_height_m,
    )
    sidewall_metadata: dict[str, Any] = {
        "enabled": False,
        "blend_weight": float(sidewall_profile_weight),
    }
    outer_thickness = np.full(
        footprint.shape,
        outer_thickness_m,
        dtype=np.float64,
    )
    if float(sidewall_profile_weight) > 0.0:
        try:
            sidewall_profile, grid_sector, fitted_metadata = (
                _angular_sidewall_thickness(
                    rgb=rgb,
                    world=world,
                    valid=valid,
                    layer_mask=mask,
                    plane=plane,
                    center_xy=center_xy,
                    radius_m=radius_m,
                    seam_angles_deg=seam_angles_deg,
                    base_top=base_top,
                    footprint=footprint,
                    grid_xx=xx,
                    grid_yy=yy,
                    global_outer_thickness_m=outer_thickness_m,
                    pixel_footprint_m=pixel_footprint_m,
                )
            )
        except ValueError as error:
            sidewall_metadata["reason"] = str(error)
        else:
            completed = sidewall_profile[grid_sector]
            outer_thickness = (
                (1.0 - float(sidewall_profile_weight))
                * outer_thickness_m
                + float(sidewall_profile_weight) * completed
            )
            sidewall_metadata = {
                **fitted_metadata,
                "enabled": True,
                "blend_weight": float(sidewall_profile_weight),
            }
    thickness = center_thickness_m + (
        outer_thickness - center_thickness_m
    ) * np.power(
        np.clip(normalized_radius, 0.0, 1.0),
        float(radial_power),
    )
    bottom = np.minimum(
        base_top - thickness,
        filled_top - pitch_m,
    )

    gap_m = max(0.0, float(seam_gap_pixels) * pixel_footprint_m)
    if gap_m > 0.0 and seam_angles_deg:
        radial_distance = normalized_radius * float(radius_m)
        gap = np.zeros_like(footprint)
        for angle_deg in seam_angles_deg:
            angle = np.deg2rad(float(angle_deg))
            perpendicular = np.abs(
                -(xx - float(center_xy[0])) * np.sin(angle)
                + (yy - float(center_xy[1])) * np.cos(angle)
            )
            forward = (
                (xx - float(center_xy[0])) * np.cos(angle)
                + (yy - float(center_xy[1])) * np.sin(angle)
            )
            gap |= (
                (perpendicular <= 0.5 * gap_m)
                & (forward >= 0.04 * float(radius_m))
                & (radial_distance <= 1.03 * float(radius_m))
            )
        footprint &= ~gap

    z_min = float(
        np.floor(
            (float(np.min(bottom[footprint])) - 2.0 * pitch_m)
            / pitch_m
        )
        * pitch_m
    )
    z_max = float(np.max(filled_top[footprint]) + 2.0 * pitch_m)
    z_count = int(np.ceil((z_max - z_min) / pitch_m)) + 1
    if x_count * y_count * z_count > 16_000_000:
        raise ValueError("radial wedge voxel grid is too large")
    bottom_index = np.floor((bottom - z_min) / pitch_m).astype(np.int32)
    top_index = np.ceil((filled_top - z_min) / pitch_m).astype(np.int32)
    z_index = np.arange(z_count, dtype=np.int32).reshape(1, 1, -1)
    volume = (
        footprint[..., None]
        & (z_index >= bottom_index[..., None])
        & (z_index <= top_index[..., None])
    )
    if int(volume.sum()) < 100:
        raise ValueError("radial wedge volume is empty")
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
        volume,
        pitch=pitch_m,
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
        + np.asarray(plane.basis_x, dtype=np.float64) * x_min
        + np.asarray(plane.basis_y, dtype=np.float64) * y_min
        + np.asarray(plane.normal, dtype=np.float64) * z_min
    )
    mesh.apply_transform(transform)
    mesh.process(validate=True)
    trimesh.repair.fix_normals(mesh)
    if float(mesh.volume) < 0.0:
        mesh.invert()
    if not mesh.is_watertight:
        raise ValueError("radial wedge mesh is not watertight")
    return mesh, {
        "source_pixels": int(np.asarray(mask, dtype=bool).sum()),
        "voxel_mm": float(voxel_mm),
        "radial_power": float(radial_power),
        "outer_thickness_scale": float(outer_thickness_scale),
        "center_thickness_pixels": float(center_thickness_pixels),
        "base_percentile": float(base_percentile),
        "seam_gap_pixels": float(seam_gap_pixels),
        "sectorwise_base": bool(sectorwise_base),
        "sidewall_thickness": sidewall_metadata,
        "pixel_footprint_mm": pixel_footprint_m * 1000.0,
        "visible_height_percentiles_mm": [
            float(h_low * 1000.0),
            float(h_high * 1000.0),
        ],
        "center_thickness_mm": center_thickness_m * 1000.0,
        "outer_thickness_mm": outer_thickness_m * 1000.0,
        "thickness_percentiles_mm": (
            np.percentile(
                thickness[footprint],
                [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0],
            )
            * 1000.0
        ).tolist(),
        "base_top_percentiles_mm": (
            np.percentile(
                base_top[footprint],
                [0.5, 10.0, 50.0, 90.0, 99.5],
            )
            * 1000.0
        ).tolist(),
        "bottom_percentiles_mm": (
            np.percentile(
                bottom[footprint],
                [0.5, 10.0, 50.0, 90.0, 99.5],
            )
            * 1000.0
        ).tolist(),
        "grid_shape": [int(x_count), int(y_count), int(z_count)],
        "occupied_voxels": int(volume.sum()),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_volume_cm3": abs(float(mesh.volume)) * 1e6,
    }

def _write_detection_overlay(
    rgb: np.ndarray,
    world: np.ndarray,
    mask: np.ndarray,
    plane: Any,
    *,
    center_xy: np.ndarray,
    radius_m: float,
    seam_angles_deg: list[float],
    path: Path,
) -> None:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    points_xy = _plane_xy(np.asarray(world)[mask], plane)
    center_index = int(
        np.argmin(
            np.linalg.norm(
                points_xy - np.asarray(center_xy).reshape(1, 2),
                axis=1,
            )
        )
    )
    center_uv = (int(xx[center_index]), int(yy[center_index]))
    draw.ellipse(
        (
            center_uv[0] - 5,
            center_uv[1] - 5,
            center_uv[0] + 5,
            center_uv[1] + 5,
        ),
        fill=(0, 255, 0, 220),
    )
    for angle_deg in seam_angles_deg:
        angle = np.deg2rad(float(angle_deg))
        target_xy = np.asarray(center_xy, dtype=np.float64) + (
            float(radius_m)
            * np.asarray([np.cos(angle), np.sin(angle)])
        )
        target_index = int(
            np.argmin(
                np.linalg.norm(
                    points_xy - target_xy.reshape(1, 2),
                    axis=1,
                )
            )
        )
        target_uv = (int(xx[target_index]), int(yy[target_index]))
        draw.line(
            (center_uv, target_uv),
            fill=(255, 0, 255, 190),
            width=2,
        )
    outline = np.asarray(mask, dtype=bool) ^ ndimage.binary_erosion(mask)
    output = np.asarray(image.convert("RGB")).copy()
    output[outline] = np.asarray([0, 255, 255], dtype=np.uint8)
    Image.fromarray(output).save(path)

def _fit_visible_radial_power(
    normalized_radius: np.ndarray,
    height_m: np.ndarray,
    *,
    bin_count: int = 16,
) -> tuple[float, dict[str, Any]]:
    radius = np.asarray(normalized_radius, dtype=np.float64).reshape(-1)
    height = np.asarray(height_m, dtype=np.float64).reshape(-1)
    valid = (
        np.isfinite(radius)
        & np.isfinite(height)
        & (radius >= 0.0)
        & (radius <= 1.05)
    )
    radius = radius[valid]
    height = height[valid]
    if len(radius) < 128:
        raise ValueError("radial power requires at least 128 visible points")

    edges = np.linspace(0.0, 1.0, int(bin_count) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.full(int(bin_count), np.nan, dtype=np.float64)
    counts = np.zeros(int(bin_count), dtype=np.int64)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (radius >= low) & (radius < high)
        counts[index] = int(selected.sum())
        if counts[index] >= 8:
            profile[index] = float(np.median(height[selected]))

    finite = np.isfinite(profile)
    fitting = finite & (centers >= 0.10) & (centers <= 0.92)
    if int(fitting.sum()) < 8:
        raise ValueError("radial power has insufficient occupied bins")
    profile[finite] = ndimage.gaussian_filter1d(
        profile[finite],
        sigma=0.65,
        mode="nearest",
    )
    fitting_centers = centers[fitting]
    fitting_profile = profile[fitting]
    weights = np.maximum(counts[fitting].astype(np.float64), 1.0)
    candidate_powers = np.linspace(0.35, 1.50, 231)
    losses: list[float] = []
    fits: list[tuple[float, float]] = []
    for candidate_power in candidate_powers:
        radial_term = np.power(fitting_centers, candidate_power)
        design = np.column_stack(
            (np.ones(len(radial_term)), radial_term)
        )
        weighted_design = design * np.sqrt(weights).reshape(-1, 1)
        weighted_target = fitting_profile * np.sqrt(weights)
        coefficient, *_ = np.linalg.lstsq(
            weighted_design,
            weighted_target,
            rcond=None,
        )
        predicted = design @ coefficient
        losses.append(
            float(
                np.average(
                    (fitting_profile - predicted) ** 2,
                    weights=weights,
                )
            )
        )
        fits.append((float(coefficient[0]), float(coefficient[1])))
    selected_index = int(np.argmin(np.asarray(losses)))
    fitted_power = float(candidate_powers[selected_index])
    low_height, amplitude = fits[selected_index]
    high_height = low_height + amplitude
    noise = float(
        np.median(
            np.abs(
                fitting_profile
                - ndimage.median_filter(
                    profile,
                    size=3,
                    mode="nearest",
                )[fitting]
            )
        )
    )
    if amplitude <= max(1e-4, 3.0 * noise):
        power = 1.0
        confidence = 0.0
    else:
        power = fitted_power
        confidence = float(
            np.clip(
                1.0 - noise / max(amplitude, 1e-9),
                0.0,
                1.0,
            )
        )
    return power, {
        "bin_centers": centers.tolist(),
        "bin_counts": counts.tolist(),
        "median_height_mm": (profile * 1000.0).tolist(),
        "inner_height_mm": low_height * 1000.0,
        "outer_height_mm": high_height * 1000.0,
        "visible_amplitude_mm": amplitude * 1000.0,
        "profile_noise_mm": noise * 1000.0,
        "fitted_power": float(power),
        "confidence": confidence,
    }

def _derive_sidewall_prior(
    observed_thickness_m: np.ndarray,
    *,
    harmonic_intercept_m: float,
    visible_sector_count: int,
    sector_count: int,
    visible_height_m: float,
    pixel_footprint_m: float,
) -> dict[str, float]:
    thickness = np.asarray(
        observed_thickness_m,
        dtype=np.float64,
    ).reshape(-1)
    thickness = thickness[np.isfinite(thickness) & (thickness > 0.0)]
    if len(thickness) < 3:
        raise ValueError("ordinary closure needs three visible sidewall sectors")
    if int(sector_count) < 3:
        raise ValueError("ordinary closure needs a periodic radial boundary")

    center_thickness_m = max(
        2.0 * float(pixel_footprint_m),
        float(np.quantile(thickness, 0.25)),
    )
    outer_thickness_m = max(
        center_thickness_m + float(pixel_footprint_m),
        float(harmonic_intercept_m),
    )
    visible_fraction = float(
        np.clip(
            float(visible_sector_count) / float(sector_count),
            0.0,
            1.0,
        )
    )
    return {
        "center_thickness_m": center_thickness_m,
        "outer_thickness_m": outer_thickness_m,
        "outer_thickness_scale": (
            outer_thickness_m / max(float(visible_height_m), 1e-9)
        ),
        "sidewall_profile_weight": visible_fraction**2,
        "visible_sector_fraction": visible_fraction,
        "seam_gap_pixels": 1.0,
    }

def _observable_parameters(
    *,
    rgb: np.ndarray,
    world: np.ndarray,
    valid: np.ndarray,
    mask: np.ndarray,
    plane: Any,
    center_xy: np.ndarray,
    radius_m: float,
    seam_angles_deg: list[float],
    voxel_mm: float,
    median_depth_m: float,
    focal_px: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    points = np.asarray(world)[np.asarray(mask, dtype=bool)]
    xy = _plane_xy(points, plane)
    height = np.asarray(
        plane.signed_height(points),
        dtype=np.float64,
    )
    normalized_radius = np.linalg.norm(
        xy - np.asarray(center_xy, dtype=np.float64).reshape(1, 2),
        axis=1,
    ) / max(float(radius_m), 1e-9)
    radial_power, radial_metadata = _fit_visible_radial_power(
        normalized_radius,
        height,
    )

    pitch_m = float(voxel_mm) / 1000.0
    top, footprint, x_min, y_min = _grid_observed_top(
        xy,
        height,
        pitch_m=pitch_m,
    )
    finite = np.isfinite(top)
    _distance, nearest = ndimage.distance_transform_edt(
        ~finite,
        return_indices=True,
    )
    filled_top = top.copy()
    filled_top[~finite] = top[
        nearest[0, ~finite],
        nearest[1, ~finite],
    ]
    footprint_radius_cells = max(
        3,
        int(round(0.08 * float(radius_m) / pitch_m)),
    )
    base_top = ndimage.percentile_filter(
        filled_top,
        percentile=25.0,
        size=2 * footprint_radius_cells + 1,
        mode="nearest",
    )
    base_top = ndimage.gaussian_filter(
        base_top,
        sigma=max(0.75, 0.15 * footprint_radius_cells),
        mode="nearest",
    )
    grid_x = x_min + np.arange(top.shape[0], dtype=np.float64) * pitch_m
    grid_y = y_min + np.arange(top.shape[1], dtype=np.float64) * pitch_m
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="ij")

    pixel_footprint_m = float(median_depth_m) / float(focal_px)
    h_low, h_high = np.percentile(height, [0.5, 99.5])
    visible_height_m = max(
        float(h_high - h_low),
        pixel_footprint_m,
    )
    _, _, sidewall_metadata = _angular_sidewall_thickness(
        rgb=rgb,
        world=world,
        valid=valid,
        layer_mask=mask,
        plane=plane,
        center_xy=center_xy,
        radius_m=radius_m,
        seam_angles_deg=seam_angles_deg,
        base_top=base_top,
        footprint=footprint,
        grid_xx=xx,
        grid_yy=yy,
        global_outer_thickness_m=visible_height_m,
        pixel_footprint_m=pixel_footprint_m,
    )
    observed_thickness_m = np.asarray(
        [
            row["observed_thickness_mm"]
            for row in sidewall_metadata["visible_sectors"]
        ],
        dtype=np.float64,
    ) / 1000.0
    harmonic_intercept_m = (
        float(sidewall_metadata["first_harmonic_coefficients_mm"][0])
        / 1000.0
    )
    sidewall_prior = _derive_sidewall_prior(
        observed_thickness_m,
        harmonic_intercept_m=harmonic_intercept_m,
        visible_sector_count=len(sidewall_metadata["visible_sectors"]),
        sector_count=len(seam_angles_deg),
        visible_height_m=visible_height_m,
        pixel_footprint_m=pixel_footprint_m,
    )
    parameters = {
        "radial_power": float(radial_power),
        "outer_thickness_scale": float(
            sidewall_prior["outer_thickness_scale"]
        ),
        "center_thickness_pixels": float(
            sidewall_prior["center_thickness_m"] / pixel_footprint_m
        ),
        "base_percentile": 25.0,
        "seam_gap_pixels": float(sidewall_prior["seam_gap_pixels"]),
        "sidewall_profile_weight": float(
            sidewall_prior["sidewall_profile_weight"]
        ),
    }
    return parameters, {
        "runtime_inputs": {
            "rgb": True,
            "metric_depth": True,
            "camera_calibration": True,
            "segmentation": False,
            "object_identity": False,
            "evaluator_centers": False,
            "reference_counts": False,
            "gt_mesh": False,
        },
        "radial_profile": radial_metadata,
        "sidewall_measurement": sidewall_metadata,
        "visible_height_mm": visible_height_m * 1000.0,
        "pixel_footprint_mm": pixel_footprint_m * 1000.0,
        "derived_center_thickness_mm": (
            sidewall_prior["center_thickness_m"] * 1000.0
        ),
        "derived_outer_thickness_mm": (
            sidewall_prior["outer_thickness_m"] * 1000.0
        ),
        "visible_sector_fraction": sidewall_prior[
            "visible_sector_fraction"
        ],
        "derived_parameters": parameters,
    }

def _robust_scale(values: np.ndarray) -> float:
    sample = np.asarray(values, dtype=np.float64)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    return max(1.4826 * mad, 1e-9)

def _detect_outer_ridge_start(
    normalized_radius: np.ndarray,
    height_m: np.ndarray,
    rgb_values: np.ndarray,
    *,
    bin_count: int = 40,
) -> tuple[float, dict[str, Any]]:
    radius = np.asarray(normalized_radius, dtype=np.float64).reshape(-1)
    height = np.asarray(height_m, dtype=np.float64).reshape(-1)
    colors = np.asarray(rgb_values, dtype=np.uint8).reshape(-1, 3)
    valid = (
        np.isfinite(radius)
        & np.isfinite(height)
        & (radius >= 0.04)
        & (radius <= 1.02)
    )
    radius = radius[valid]
    height = height[valid]
    colors = colors[valid]
    if len(radius) < 256:
        raise ValueError("ridge detection needs 256 visible samples")
    lab = rgb2lab(colors.astype(np.float64) / 255.0)

    edges = np.linspace(0.04, 0.96, int(bin_count) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profiles = np.full((4, int(bin_count)), np.nan, dtype=np.float64)
    counts = np.zeros(int(bin_count), dtype=np.int64)
    channels = np.column_stack((height * 1000.0, lab))
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (radius >= low) & (radius < high)
        counts[index] = int(selected.sum())
        if counts[index] >= 8:
            profiles[:, index] = np.median(
                channels[selected],
                axis=0,
            )
    finite = np.all(np.isfinite(profiles), axis=0)
    if int(finite.sum()) < int(0.75 * bin_count):
        raise ValueError("ridge annular profile is too sparse")
    for channel in range(len(profiles)):
        profiles[channel, finite] = ndimage.gaussian_filter1d(
            profiles[channel, finite],
            sigma=1.0,
            mode="nearest",
        )
    gradients = np.gradient(profiles, centers, axis=1)
    height_gradient = gradients[0]
    color_gradient = np.linalg.norm(
        gradients[1:]
        / np.asarray(
            [
                _robust_scale(gradients[1]),
                _robust_scale(gradients[2]),
                _robust_scale(gradients[3]),
            ],
            dtype=np.float64,
        ).reshape(3, 1),
        axis=0,
    )
    height_score = np.clip(
        height_gradient / _robust_scale(height_gradient),
        0.0,
        None,
    )
    color_score = color_gradient / _robust_scale(color_gradient)
    score = height_score + 0.30 * color_score
    eligible = (
        finite
        & (centers >= 0.58)
        & (centers <= 0.88)
        & (height_gradient > 0.0)
    )
    if not np.any(eligible):
        raise ValueError("no observable positive outer ridge transition")
    eligible_indices = np.flatnonzero(eligible)
    selected_index = int(
        eligible_indices[np.argmax(score[eligible_indices])]
    )
    ridge_start = float(centers[selected_index])
    return ridge_start, {
        "bin_centers": centers.tolist(),
        "bin_counts": counts.tolist(),
        "height_profile_mm": profiles[0].tolist(),
        "lab_profiles": profiles[1:].tolist(),
        "height_gradient": height_gradient.tolist(),
        "color_gradient_score": color_score.tolist(),
        "combined_score": score.tolist(),
        "ridge_start_radius": ridge_start,
        "selected_bin": selected_index,
    }

def _prominence_mixture_threshold(
    prominence_m: np.ndarray,
    *,
    pixel_footprint_m: float,
) -> tuple[float, dict[str, Any]]:
    values = np.asarray(prominence_m, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values >= 0.0)]
    if len(values) < 128:
        raise ValueError("prominence mixture needs 128 samples")
    split = float(threshold_otsu(values))
    high = values[values > split]
    if len(high) < 16:
        threshold = max(split, 2.0 * float(pixel_footprint_m))
    else:
        threshold = max(
            float(np.median(high)),
            2.0 * float(pixel_footprint_m),
        )
    return threshold, {
        "otsu_split_mm": split * 1000.0,
        "high_class_fraction": float(len(high) / len(values)),
        "high_class_percentiles_mm": (
            np.percentile(high, [25.0, 50.0, 75.0]) * 1000.0
        ).tolist()
        if len(high)
        else [],
        "ordinary_detail_threshold_mm": threshold * 1000.0,
    }

def _shallow_low_profile_component(
    mesh: trimesh.Trimesh,
    metadata: dict[str, Any],
    *,
    shell_mm: float,
) -> tuple[list[trimesh.Trimesh], dict[str, Any]]:
    rows = metadata.get("low_profile_frustum_components")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("expected exactly one low-profile component")
    row = dict(rows[0])
    component_index = int(row["component_index"])
    front_count = int(row["front_vertices"])
    components = list(mesh.split(only_watertight=False))
    component = components[component_index]
    vertices = np.asarray(component.vertices, dtype=np.float64)
    if len(vertices) != 2 * front_count:
        raise ValueError("low-profile front/back layout is unavailable")
    front = vertices[:front_count]
    back = vertices[front_count:]
    displacement = back - front
    distance = np.linalg.norm(displacement, axis=1)
    unit = np.divide(
        displacement,
        distance[:, None],
        out=np.zeros_like(displacement),
        where=distance[:, None] > 1e-9,
    )
    changed_vertices = vertices.copy()
    changed_vertices[front_count:] = (
        front + unit * (float(shell_mm) / 1000.0)
    )
    changed = trimesh.Trimesh(
        vertices=changed_vertices,
        faces=np.asarray(component.faces, dtype=np.int64),
        process=False,
    )
    if not changed.is_watertight:
        raise ValueError("shallow low-profile shell is not watertight")
    components[component_index] = changed
    return components, {
        "component_index": component_index,
        "front_vertices": front_count,
        "surface_shell_mm": float(shell_mm),
        "original_back_distance_percentiles_mm": (
            np.percentile(distance, [5.0, 50.0, 95.0]) * 1000.0
        ).tolist(),
    }

def _front_surface(
    world: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    focal_px: float,
    surface_edge_scale: float,
    surface_edge_slack_m: float,
    max_depth_jump_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    first_down = horizontal[:-1, :] & vertical[:, :-1] & diagonal
    second_down = horizontal[1:, :] & vertical[:, 1:] & diagonal
    first_up = horizontal[:-1, :] & vertical[:, 1:] & diagonal_up
    second_up = horizontal[1:, :] & vertical[:, :-1] & diagonal_up
    use_down = (
        first_down.astype(np.int8) + second_down.astype(np.int8)
        >= first_up.astype(np.int8) + second_up.astype(np.int8)
    )
    rows: list[np.ndarray] = []
    use = use_down & first_down
    if use.any():
        rows.append(
            np.column_stack(
                (top_left[use], bottom_right[use], top_right[use])
            )
        )
    use = use_down & second_down
    if use.any():
        rows.append(
            np.column_stack(
                (top_left[use], bottom_left[use], bottom_right[use])
            )
        )
    use = (~use_down) & first_up
    if use.any():
        rows.append(
            np.column_stack(
                (top_left[use], bottom_left[use], top_right[use])
            )
        )
    use = (~use_down) & second_up
    if use.any():
        rows.append(
            np.column_stack(
                (top_right[use], bottom_left[use], bottom_right[use])
            )
        )
    if not rows:
        raise ValueError("ordinary surface has no continuous RGB-D triangles")
    triangles = np.vstack(rows)
    flat_points = points_grid.reshape(-1, 3)
    used = np.unique(triangles)
    remap = np.full(len(flat_points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return flat_points[used], remap[triangles], used

def _moderated_parameters(
    derivation: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Derive closure parameters without using evaluator information."""
    pixel_footprint_m = (
        float(derivation["pixel_footprint_mm"]) / 1000.0
    )
    visible_height_m = float(derivation["visible_height_mm"]) / 1000.0
    sidewall = dict(derivation["sidewall_measurement"])
    observed_thickness_m = np.asarray(
        [
            row["observed_thickness_mm"]
            for row in sidewall["visible_sectors"]
        ],
        dtype=np.float64,
    ) / 1000.0
    if len(observed_thickness_m) < 3:
        raise ValueError("moderated shell needs three visible sidewalls")
    ordinary_thickness_m = max(
        2.0 * pixel_footprint_m,
        float(np.quantile(observed_thickness_m, 0.25)),
    )
    harmonic_outer_m = (
        float(sidewall["first_harmonic_coefficients_mm"][0]) / 1000.0
    )
    outer_thickness_m = max(
        harmonic_outer_m,
        ordinary_thickness_m + pixel_footprint_m,
    )
    visible_amplitude_m = max(
        float(derivation["radial_profile"]["visible_amplitude_mm"])
        / 1000.0,
        0.0,
    )
    center_thickness_m = max(
        pixel_footprint_m,
        ordinary_thickness_m - 0.5 * visible_amplitude_m,
    )
    fitted_power = float(
        derivation["radial_profile"]["fitted_power"]
    )
    radial_power = float(
        np.clip(np.sqrt(max(fitted_power, 1e-6)), 0.50, 1.0)
    )
    visible_fraction = float(derivation["visible_sector_fraction"])
    sidewall_weight = float(
        np.clip(visible_fraction, 0.0, 1.0) ** 3
    )
    parameters = {
        "radial_power": radial_power,
        "outer_thickness_scale": (
            outer_thickness_m / max(visible_height_m, 1e-9)
        ),
        "center_thickness_pixels": 1.0,
        "base_percentile": 25.0,
        "seam_gap_pixels": 1.0,
        "sidewall_profile_weight": sidewall_weight,
        "center_thickness_m": center_thickness_m,
        "outer_thickness_m": outer_thickness_m,
        "ordinary_thickness_m": ordinary_thickness_m,
    }
    return parameters, {
        "source": "rgbd_observable_geometry",
        "ordinary_prior": "visible_sidewall_lower_quartile",
        "center_prior": (
            "ordinary_thickness_minus_half_visible_radial_amplitude"
        ),
        "outer_prior": "visible_sidewall_first_harmonic_intercept",
        "radial_power_prior": (
            "geometric_blend_of_visible_profile_and_linear_transition"
        ),
        "sidewall_confidence_prior": "visible_sector_fraction_cubed",
        "fitted_visible_radial_power": fitted_power,
        "derived_radial_power": radial_power,
        "visible_radial_amplitude_mm": visible_amplitude_m * 1000.0,
        "ordinary_thickness_mm": ordinary_thickness_m * 1000.0,
        "center_thickness_mm": center_thickness_m * 1000.0,
        "outer_thickness_mm": outer_thickness_m * 1000.0,
        "visible_sector_fraction": visible_fraction,
        "sidewall_profile_weight": sidewall_weight,
    }

def _variable_camera_ray_shell(
    world: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    depth_map_m: np.ndarray,
    *,
    camera_pos: np.ndarray,
    camera_forward: np.ndarray,
    focal_px: float,
    surface_edge_scale: float = 1.8,
    surface_edge_slack_m: float = 0.002,
    max_depth_jump_m: float = 0.03,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    points, triangles, used = _front_surface(
        world,
        depth,
        mask,
        focal_px=float(focal_px),
        surface_edge_scale=float(surface_edge_scale),
        surface_edge_slack_m=float(surface_edge_slack_m),
        max_depth_jump_m=float(max_depth_jump_m),
    )
    camera_position = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    forward = np.asarray(camera_forward, dtype=np.float64).reshape(3)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    triangles = _orient_front_triangles(
        triangles,
        points,
        camera_pos=camera_position,
    )
    original_points = points.copy()
    original_depth = np.asarray(
        depth_map_m,
        dtype=np.float64,
    ).reshape(-1)[used]
    points, triangles, split_metadata = _split_nonmanifold_vertex_fans(
        points,
        triangles,
    )
    nearest = cKDTree(original_points).query(points, k=1)[1]
    split_depth = original_depth[np.asarray(nearest, dtype=np.int64)]

    front_mesh = trimesh.Trimesh(
        vertices=points,
        faces=triangles,
        process=False,
    )
    shells: list[trimesh.Trimesh] = []
    boundary_count = 0
    nonwatertight = 0
    for component in front_mesh.split(only_watertight=False):
        front = np.asarray(component.vertices, dtype=np.float64)
        faces = _orient_front_triangles(
            np.asarray(component.faces, dtype=np.int64),
            front,
            camera_pos=camera_position,
        )
        source = cKDTree(points).query(front, k=1)[1]
        local_depth = split_depth[np.asarray(source, dtype=np.int64)]
        ray_vectors = front - camera_position.reshape(1, 3)
        axial_depth = ray_vectors @ forward.reshape(3)
        if np.any(axial_depth <= 1e-8):
            raise ValueError("front component contains points behind camera")
        back_scale = 1.0 + local_depth / axial_depth
        back = (
            camera_position.reshape(1, 3)
            + ray_vectors * back_scale[:, None]
        )
        vertex_count = len(front)
        boundary = _boundary_directed_edges(faces)
        boundary_count += int(len(boundary))
        back_faces = faces[:, [0, 2, 1]] + vertex_count
        side_a = np.column_stack(
            (
                boundary[:, 0],
                boundary[:, 0] + vertex_count,
                boundary[:, 1] + vertex_count,
            )
        )
        side_b = np.column_stack(
            (
                boundary[:, 0],
                boundary[:, 1] + vertex_count,
                boundary[:, 1],
            )
        )
        shell = trimesh.Trimesh(
            vertices=np.vstack((front, back)),
            faces=np.vstack((faces, back_faces, side_a, side_b)),
            process=False,
        )
        if not shell.is_watertight:
            nonwatertight += 1
        shells.append(shell)
    if not shells:
        raise ValueError("variable camera shell is empty")
    return trimesh.util.concatenate(shells), {
        **split_metadata,
        "front_surface_components": int(len(shells)),
        "boundary_edges": int(boundary_count),
        "nonwatertight_shell_components": int(nonwatertight),
        "source_pixels": int(np.asarray(mask, dtype=bool).sum()),
        "depth_mm_quantiles": (
            np.percentile(
                np.asarray(depth_map_m)[np.asarray(mask, dtype=bool)],
                [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0],
            )
            * 1000.0
        ).tolist(),
        "mode": "variable_camera_ray",
    }

def _expand_outer_ring(
    mesh: trimesh.Trimesh,
    *,
    plane: Any,
    center_xy: np.ndarray,
    radius_m: float,
    ridge_start: float,
    expansion_m: float,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    if (
        float(radius_m) <= 0.0
        or not 0.0 <= float(ridge_start) < 1.0
        or float(expansion_m) < 0.0
    ):
        raise ValueError("invalid outer-ring expansion parameters")
    result = mesh.copy()
    vertices = np.asarray(result.vertices, dtype=np.float64).copy()
    origin = np.asarray(plane.origin, dtype=np.float64).reshape(3)
    basis_x = np.asarray(plane.basis_x, dtype=np.float64).reshape(3)
    basis_y = np.asarray(plane.basis_y, dtype=np.float64).reshape(3)
    relative = vertices - origin.reshape(1, 3)
    x = relative @ basis_x
    y = relative @ basis_y
    delta_x = x - float(center_xy[0])
    delta_y = y - float(center_xy[1])
    radius = np.hypot(delta_x, delta_y)
    normalized = radius / float(radius_m)
    transition = np.clip(
        (normalized - float(ridge_start))
        / max(1.0 - float(ridge_start), 1e-9),
        0.0,
        1.0,
    )
    transition = transition * transition * (3.0 - 2.0 * transition)
    displacement = float(expansion_m) * transition
    unit_x = np.divide(
        delta_x,
        radius,
        out=np.zeros_like(delta_x),
        where=radius > 1e-9,
    )
    unit_y = np.divide(
        delta_y,
        radius,
        out=np.zeros_like(delta_y),
        where=radius > 1e-9,
    )
    vertices += (
        basis_x.reshape(1, 3) * (unit_x * displacement)[:, None]
        + basis_y.reshape(1, 3) * (unit_y * displacement)[:, None]
    )
    result.vertices = vertices
    if not result.is_watertight:
        raise ValueError("outer-ring expansion opened the wedge")
    return result, {
        "ridge_start_radius": float(ridge_start),
        "maximum_expansion_mm": float(expansion_m) * 1000.0,
        "transition": "smoothstep_outer_ring",
        "moved_vertex_count": int((transition > 0.0).sum()),
        "full_expansion_vertex_count": int((transition >= 1.0).sum()),
    }

def _boundary_transition_starts(
    *,
    radius_m: float,
    ridge_start: float,
    residual_percentiles_mm: np.ndarray,
) -> dict[str, float]:
    radius = float(radius_m)
    residual_mm = np.asarray(
        residual_percentiles_mm,
        dtype=np.float64,
    )
    if (
        radius <= 0.0
        or not 0.0 <= float(ridge_start) < 1.0
        or residual_mm.shape != (4,)
        or np.any(~np.isfinite(residual_mm))
        or np.any(residual_mm < 0.0)
    ):
        raise ValueError("invalid boundary uncertainty statistics")

    def from_residual(value_mm: float) -> float:
        normalized = 1.0 - float(value_mm) / (1000.0 * radius)
        return float(
            np.clip(
                normalized,
                float(ridge_start),
                1.0 - 1e-6,
            )
        )

    return {
        "ridge": float(ridge_start),
        "median_band": from_residual(float(residual_mm[1])),
        "p75_band": from_residual(float(residual_mm[2])),
        "p90_band": from_residual(float(residual_mm[3])),
    }

def _candidate_row(
    *,
    name: str,
    output_dir: Path,
    base_components: list[trimesh.Trimesh],
    wedge: trimesh.Trimesh,
    shell: trimesh.Trimesh | None,
    common: dict[str, Any],
    shell_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    parts = [*base_components, wedge]
    if shell is not None:
        parts.append(shell)
    mesh = trimesh.util.concatenate(parts)
    mesh_path = output_dir / f"{name}.ply"
    mesh.export(mesh_path)
    return {
        "name": name,
        "mesh_path": str(mesh_path),
        "runtime_eligible": True,
        "build": RGBD_BOUNDARY_UNCERTAINTY_BAND_BUILD,
        "reconstruction_inputs": {
            "rgb": True,
            "metric_depth": True,
            "camera_calibration": True,
            "baseline_rgbd_mesh": True,
            "segmentation": False,
            "object_identity": False,
            "evaluator_centers": False,
            "reference_counts": False,
            "gt_mesh": False,
        },
        **common,
        "variable_shell": shell_metadata,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_volume_cm3": abs(float(mesh.volume)) * 1e6,
    }

def _reconstruct_candidates(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    camera: dict[str, Any],
    baseline_mesh: trimesh.Trimesh,
    baseline_metadata: dict[str, Any],
    voxel_mm: float,
    surface_shell_mm: float,
    overlap_voxels: float,
    output_dir: Path,
) -> list[dict[str, Any]]:
    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    _thin_mask, plane, world, support_metadata = (
        _support_structure_mask(
            rgb,
            metric_depth,
            camera_pos=camera["pos"],
            camera_quat_xyzw=camera["quat"],
            focal_length=float(camera["focal_length"]),
            horizontal_aperture=float(camera["horizontal_aperture"]),
        )
    )
    if plane is None or world is None:
        raise ValueError("support plane was not detected")
    valid = np.isfinite(metric_depth) & (metric_depth > 0.03)
    focal_px = (
        float(camera["focal_length"])
        / float(camera["horizontal_aperture"])
        * float(metric_depth.shape[1])
    )
    masks, layer_rows = _supported_low_profile_layers(
        rgb,
        metric_depth,
        world,
        valid,
        plane,
        focal_px=float(focal_px),
    )
    if len(masks) != 1:
        raise ValueError(
            f"expected one supported radial layer, got {len(masks)}"
        )
    layer_mask = masks[0]
    center_xy, radius_m, circle_metadata = _fit_boundary_circle(
        world,
        layer_mask,
        plane,
    )
    theta, evidence, evidence_metadata = _polar_height_evidence(
        world,
        layer_mask,
        plane,
        center_xy=center_xy,
        radius_m=radius_m,
    )
    periodic = _periodic_radial_structure(theta, evidence)
    if not bool(periodic["credible"]):
        raise ValueError("supported layer has no credible radial periodicity")
    seam_angles_deg = [
        float(value) for value in periodic["ray_angles_deg"]
    ]
    median_depth_m = float(np.median(metric_depth[layer_mask]))
    _old_parameters, derivation = _observable_parameters(
        rgb=rgb,
        world=world,
        valid=valid,
        mask=layer_mask,
        plane=plane,
        center_xy=center_xy,
        radius_m=radius_m,
        seam_angles_deg=seam_angles_deg,
        voxel_mm=float(voxel_mm),
        median_depth_m=median_depth_m,
        focal_px=float(focal_px),
    )
    parameters, moderated_metadata = _moderated_parameters(
        derivation
    )
    wedge, wedge_metadata = _radial_wedge_mesh(
        world,
        layer_mask,
        plane,
        rgb=rgb,
        valid=valid,
        center_xy=center_xy,
        radius_m=radius_m,
        seam_angles_deg=seam_angles_deg,
        voxel_mm=float(voxel_mm),
        radial_power=float(parameters["radial_power"]),
        outer_thickness_scale=float(
            parameters["outer_thickness_scale"]
        ),
        center_thickness_pixels=1.0,
        base_percentile=25.0,
        seam_gap_pixels=1.0,
        sectorwise_base=False,
        sidewall_profile_weight=float(
            parameters["sidewall_profile_weight"]
        ),
        median_depth_m=median_depth_m,
        focal_px=float(focal_px),
    )

    prominence, prominence_metadata = _support_prominence(
        world,
        layer_mask,
        plane,
    )
    pixel_footprint_m = (
        float(derivation["pixel_footprint_mm"]) / 1000.0
    )
    prominence_threshold_m, threshold_metadata = (
        _prominence_mixture_threshold(
            prominence[layer_mask],
            pixel_footprint_m=pixel_footprint_m,
        )
    )
    ordinary_mask = (
        layer_mask
        & np.isfinite(prominence)
        & (prominence <= prominence_threshold_m)
    )
    ordinary_mask = ndimage.binary_opening(
        ordinary_mask,
        structure=np.ones((2, 2), dtype=bool),
    )
    relative = (
        np.asarray(world, dtype=np.float64)
        - np.asarray(plane.origin, dtype=np.float64).reshape(1, 1, 3)
    )
    points_xy = np.stack(
        (
            relative @ np.asarray(plane.basis_x, dtype=np.float64),
            relative @ np.asarray(plane.basis_y, dtype=np.float64),
        ),
        axis=-1,
    )
    normalized_radius = np.linalg.norm(
        points_xy - center_xy.reshape(1, 1, 2),
        axis=-1,
    ) / max(float(radius_m), 1e-9)
    layer_points = np.asarray(world)[layer_mask]
    layer_height = np.asarray(
        plane.signed_height(layer_points),
        dtype=np.float64,
    )
    ridge_start, ridge_metadata = _detect_outer_ridge_start(
        normalized_radius[layer_mask],
        layer_height,
        np.asarray(rgb)[layer_mask],
    )
    camera_rotation = _quat_to_mat_xyzw(camera["quat"])
    camera_forward = camera_rotation @ np.asarray(
        [0.0, 0.0, -1.0],
        dtype=np.float64,
    )

    base_components, shallow_metadata = (
        _shallow_low_profile_component(
            baseline_mesh,
            baseline_metadata,
            shell_mm=float(surface_shell_mm),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_detection_overlay(
        rgb,
        world,
        layer_mask,
        plane,
        center_xy=center_xy,
        radius_m=radius_m,
        seam_angles_deg=seam_angles_deg,
        path=output_dir / "boundary_uncertainty_band_overlay.png",
    )
    common = {
        "support_plane": support_metadata,
        "supported_layer_candidates": layer_rows,
        "boundary_circle": circle_metadata,
        "radial_evidence": evidence_metadata,
        "radial_periodicity": periodic,
        "parameter_derivation": derivation,
        "outer_ring_parameter_derivation": moderated_metadata,
        "ridge_detection": ridge_metadata,
        "prominence": prominence_metadata,
        "prominence_mixture": threshold_metadata,
        "radial_wedge": wedge_metadata,
        "surface_shell": shallow_metadata,
        "overlap_voxels": float(overlap_voxels),
    }
    overlap_m = (
        float(overlap_voxels) * float(voxel_mm) / 1000.0
    )
    ordinary_thickness_m = float(parameters["ordinary_thickness_m"])
    shell_depth = np.full_like(
        normalized_radius,
        ordinary_thickness_m + overlap_m,
        dtype=np.float64,
    )
    shell, shell_metadata = _variable_camera_ray_shell(
        world,
        metric_depth,
        ordinary_mask,
        shell_depth,
        camera_pos=np.asarray(camera["pos"], dtype=np.float64),
        camera_forward=camera_forward,
        focal_px=float(focal_px),
    )
    residual_mm = np.asarray(
        circle_metadata["residual_percentiles_mm"],
        dtype=np.float64,
    )
    transition_starts = _boundary_transition_starts(
        radius_m=radius_m,
        ridge_start=ridge_start,
        residual_percentiles_mm=residual_mm,
    )
    rows: list[dict[str, Any]] = []
    for mode, transition_start in transition_starts.items():
        expanded_wedge, expansion_metadata = _expand_outer_ring(
            wedge,
            plane=plane,
            center_xy=center_xy,
            radius_m=radius_m,
            ridge_start=transition_start,
            expansion_m=float(residual_mm[1]) / 1000.0,
        )
        rows.append(
            _candidate_row(
                name=f"v52_boundary_uncertainty_{mode}",
                output_dir=output_dir,
                base_components=base_components,
                wedge=expanded_wedge,
                shell=shell,
                common={
                    **common,
                    "outer_ring_expansion": expansion_metadata,
                    "boundary_uncertainty_band": {
                        "source": "visible_circle_fit_residuals",
                        "residual_percentiles_mm": residual_mm.tolist(),
                        "transition_start": float(transition_start),
                    },
                },
                shell_metadata={
                    **shell_metadata,
                    "transition_mode": "constant",
                    "center_thickness_mm": ordinary_thickness_m * 1000.0,
                    "outer_thickness_mm": ordinary_thickness_m * 1000.0,
                },
            )
        )
    return rows

def reconstruct_rgbd_scene_mesh(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
) -> RGBDSceneMeshResult:
    """Reconstruct one whole RGB-D view with canonical representative V52."""
    total_started = time.perf_counter()
    base_started = time.perf_counter()
    base = _reconstruct_v13_scene_mesh(
        rgb,
        depth,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
        **V13_RECONSTRUCTION_KWARGS,
    )
    base_elapsed_s = float(time.perf_counter() - base_started)
    camera = {
        "pos": np.asarray(camera_pos, dtype=np.float64).reshape(3).tolist(),
        "quat": np.asarray(camera_quat_xyzw, dtype=np.float64).reshape(4).tolist(),
        "focal_length": float(focal_length),
        "horizontal_aperture": float(horizontal_aperture),
    }

    def _v52_workdir():
        # Re-select an existing temp root if an external owner removed its leaf.
        if tempfile.tempdir is not None and not Path(tempfile.tempdir).is_dir():
            tempfile.tempdir = None
        return tempfile.TemporaryDirectory(prefix="official_v2_rgbd_v52_")

    def _apply_v52_expert(temporary_dir: str) -> tuple[trimesh.Trimesh, dict[str, Any]]:
        rows = _reconstruct_candidates(
            rgb=np.asarray(rgb),
            depth=np.asarray(depth),
            camera=camera,
            baseline_mesh=base.mesh,
            baseline_metadata=base.metadata,
            voxel_mm=1.5,
            surface_shell_mm=0.5,
            overlap_voxels=0.0,
            output_dir=Path(temporary_dir),
        )
        selected = next(
            row
            for row in rows
            if row["name"] == "v52_boundary_uncertainty_p90_band"
        )
        loaded = trimesh.load(
            selected["mesh_path"],
            force="mesh",
            process=False,
        )
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError("V52 output is not a Trimesh")
        metadata = dict(selected)
        metadata.pop("mesh_path", None)
        return loaded, metadata

    expert_started = time.perf_counter()
    selected_metadata: dict[str, Any] | None = None
    try:
        try:
            with _v52_workdir() as temporary_dir:
                mesh, selected_metadata = _apply_v52_expert(temporary_dir)
        except FileNotFoundError:
            with _v52_workdir() as temporary_dir:
                mesh, selected_metadata = _apply_v52_expert(temporary_dir)
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        IndexError,
        StopIteration,
    ) as error:
        mesh = base.mesh.copy()
        representative = {
            "representative_version": MESH_VERSION,
            "effective_mesh_version": "v13_fallback",
            "expert_applied": False,
            "applicability": "not_applicable",
            "fallback": "frozen_v13_mesh",
            "reason": f"{type(error).__name__}: {error}",
        }
    else:
        representative = {
            "representative_version": MESH_VERSION,
            "effective_mesh_version": MESH_VERSION,
            "expert_applied": True,
            "applicability": "applicable",
            "fallback": None,
            "reason": None,
            "selected_candidate": selected_metadata,
        }
    expert_elapsed_s = float(time.perf_counter() - expert_started)
    metadata = {
        **base.metadata,
        "parent_build": base.metadata.get("build"),
        "build": RGBD_SCENE_MESH_BUILD,
        "mesh_version": MESH_VERSION,
        **representative,
        "base_mesh_version": "v13",
        "base_reconstruction_build": base.metadata.get("build"),
        "base_reconstruction_kwargs": dict(V13_RECONSTRUCTION_KWARGS),
        "v52_parameters": {
            "voxel_mm": 1.5,
            "surface_shell_mm": 0.5,
            "overlap_voxels": 0.0,
            "selected_candidate": "v52_boundary_uncertainty_p90_band",
        },
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_watertight": bool(mesh.is_watertight),
        "timings_s": {
            "v13_base": base_elapsed_s,
            "v52_expert": expert_elapsed_s,
            "total": float(time.perf_counter() - total_started),
        },
        "allowed_inputs": [
            "rgb",
            "metric_depth",
            "camera_intrinsics",
            "camera_extrinsics",
        ],
        "used_click": False,
        "used_segmentation": False,
        "used_object_identity": False,
        "used_evaluator_centers": False,
        "used_reference_counts": False,
        "used_gt_mesh": False,
        "forbidden_inputs_used": [],
    }
    return RGBDSceneMeshResult(mesh=mesh, metadata=metadata)


def reconstruct_scene_mesh_from_session(
    session: Dict[str, Any],
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
) -> tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a frozen capture and reconstruct one reusable V52 scene mesh."""
    init_dir = str(session.get("init_dir") or "")
    rgb_path = str(
        session.get("rgb_path") or os.path.join(init_dir, "rgb.png")
    )
    depth_path = str(
        session.get("depth_path") or os.path.join(init_dir, "depth.npy")
    )
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"missing frozen RGB: {rgb_path}")
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"missing frozen depth: {depth_path}")
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path)
    started = time.perf_counter()
    result = reconstruct_rgbd_scene_mesh(
        rgb,
        depth,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=float(focal_length),
        horizontal_aperture=float(horizontal_aperture),
    )
    metadata = dict(result.metadata)
    metadata["elapsed_s"] = float(time.perf_counter() - started)
    return result.mesh, rgb, depth, metadata


__all__ = [
    "MESH_VERSION",
    "RGBD_SCENE_MESH_BUILD",
    "V13_RECONSTRUCTION_KWARGS",
    "reconstruct_rgbd_scene_mesh",
    "reconstruct_scene_mesh_from_session",
]
