"""RGB-D scene component helpers used by the frozen V12/V52 runtime."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

def _fit_horizontal_plane(
    samples: np.ndarray,
    *,
    mode_z: float,
    coarse_band_m: float,
    inlier_band_m: float,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    coarse = samples[
        np.abs(samples[:, 2] - float(mode_z)) <= float(coarse_band_m)
    ]
    if len(coarse) < 100:
        return None
    design = np.column_stack(
        (coarse[:, 0], coarse[:, 1], np.ones(len(coarse)))
    )
    coefficient, *_ = np.linalg.lstsq(design, coarse[:, 2], rcond=None)
    for _ in range(3):
        residual = coarse[:, 2] - design @ coefficient
        center = float(np.median(residual))
        keep = np.abs(residual - center) <= float(inlier_band_m)
        if int(keep.sum()) < 100:
            break
        coefficient, *_ = np.linalg.lstsq(
            design[keep],
            coarse[keep, 2],
            rcond=None,
        )
    a, b, c = [float(value) for value in coefficient]
    scale = float(np.sqrt(a * a + b * b + 1.0))
    normal = np.array([-a, -b, 1.0], dtype=np.float64) / scale
    offset = -c / scale
    signed = samples @ normal + offset
    return normal, float(offset), signed

def _detect_horizontal_regions(
    world: np.ndarray,
    valid: np.ndarray,
    *,
    bin_m: float,
    band_m: float,
    min_pixels: int,
    min_separation_m: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = np.full(valid.shape, -1, dtype=np.int32)
    points = np.asarray(world[valid], dtype=np.float64)
    z = points[:, 2]
    low, high = np.percentile(z, [0.1, 99.9])
    edges = np.arange(
        np.floor(low / bin_m) * bin_m,
        np.ceil(high / bin_m) * bin_m + bin_m,
        bin_m,
    )
    histogram, edges = np.histogram(z, bins=edges)
    maximum = ndimage.maximum_filter1d(histogram, size=5, mode="nearest")
    candidates = np.flatnonzero(
        (histogram == maximum) & (histogram >= int(min_pixels))
    )
    candidates = candidates[
        np.argsort(histogram[candidates])[::-1]
    ]

    regions: list[dict[str, Any]] = []
    accepted_heights: list[float] = []
    for histogram_index in candidates:
        mode_z = float(
            0.5 * (edges[histogram_index] + edges[histogram_index + 1])
        )
        if any(
            abs(mode_z - height) < float(min_separation_m)
            for height in accepted_heights
        ):
            continue
        fitted = _fit_horizontal_plane(
            points,
            mode_z=mode_z,
            coarse_band_m=max(2.0 * band_m, bin_m),
            inlier_band_m=band_m,
        )
        if fitted is None:
            continue
        normal, offset, _signed_samples = fitted
        signed_image = np.einsum("...i,i->...", world, normal) + offset
        plane_mask = valid & (np.abs(signed_image) <= band_m)
        image_labels, count = ndimage.label(
            plane_mask,
            structure=np.ones((3, 3), dtype=np.int8),
        )
        component_sizes = np.bincount(image_labels.ravel())
        component_ids = np.flatnonzero(
            component_sizes >= int(min_pixels)
        )
        component_ids = component_ids[component_ids != 0]
        if not len(component_ids):
            continue
        accepted_heights.append(mode_z)
        for component_id in component_ids:
            mask = image_labels == int(component_id)
            region_id = len(regions)
            labels[mask & (labels < 0)] = region_id
            yy, xx = np.nonzero(mask)
            residual_mm = np.abs(signed_image[mask]) * 1000.0
            regions.append(
                {
                    "region_id": int(region_id),
                    "mode_z_m": mode_z,
                    "normal": normal.tolist(),
                    "offset": float(offset),
                    "pixel_count": int(mask.sum()),
                    "image_bbox_uv": [
                        int(xx.min()),
                        int(yy.min()),
                        int(xx.max()),
                        int(yy.max()),
                    ],
                    "residual_mm_median": float(
                        np.median(residual_mm)
                    ),
                    "residual_mm_p95": float(
                        np.percentile(residual_mm, 95.0)
                    ),
                }
            )
    return labels, regions

def _continuous_component_labels(
    world: np.ndarray,
    depth: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    focal_px: float,
    edge_scale: float,
    edge_slack_m: float,
    edge_depth_jump_m: float,
    edge_color_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(mask, dtype=bool)
    yy, xx = np.nonzero(selected)
    node_at = np.full(selected.shape, -1, dtype=np.int32)
    node_at[yy, xx] = np.arange(len(yy), dtype=np.int32)
    edge_first: list[np.ndarray] = []
    edge_second: list[np.ndarray] = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        first_y = yy
        first_x = xx
        second_y = yy + dy
        second_x = xx + dx
        inside = (
            (second_y >= 0)
            & (second_y < selected.shape[0])
            & (second_x >= 0)
            & (second_x < selected.shape[1])
        )
        first = np.flatnonzero(inside)
        second = node_at[second_y[inside], second_x[inside]]
        present = second >= 0
        first = first[present]
        second = second[present]
        if not len(first):
            continue
        p1 = world[first_y[first], first_x[first]]
        p2 = world[second_y[first], second_x[first]]
        d1 = depth[first_y[first], first_x[first]]
        d2 = depth[second_y[first], second_x[first]]
        color_distance = np.linalg.norm(
            np.asarray(
                rgb[first_y[first], first_x[first], :3],
                dtype=np.float64,
            )
            - np.asarray(
                rgb[second_y[first], second_x[first], :3],
                dtype=np.float64,
            ),
            axis=1,
        )
        pixel_span = float(np.hypot(dx, dy))
        expected = (
            pixel_span * np.maximum(d1, d2) / max(float(focal_px), 1e-9)
        )
        continuous = (
            np.linalg.norm(p1 - p2, axis=1)
            <= float(edge_scale) * expected + float(edge_slack_m)
        )
        continuous &= np.abs(d1 - d2) <= float(edge_depth_jump_m)
        continuous &= color_distance <= float(edge_color_distance)
        edge_first.append(first[continuous])
        edge_second.append(second[continuous])
    if not edge_first:
        labels = np.full(selected.shape, -1, dtype=np.int32)
        labels[yy, xx] = np.arange(len(yy), dtype=np.int32)
        return labels, np.ones(len(yy), dtype=np.int64)
    first = np.concatenate(edge_first)
    second = np.concatenate(edge_second)
    adjacency = sparse.coo_matrix(
        (
            np.ones(2 * len(first), dtype=np.uint8),
            (
                np.concatenate((first, second)),
                np.concatenate((second, first)),
            ),
        ),
        shape=(len(yy), len(yy)),
    ).tocsr()
    count, component = csgraph.connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    sizes = np.bincount(component, minlength=count)
    labels = np.full(selected.shape, -1, dtype=np.int32)
    labels[yy, xx] = component.astype(np.int32, copy=False)
    return labels, sizes
