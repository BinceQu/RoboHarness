"""Depth-only local watertight reconstruction and mesh overlap diagnostics.

The reconstruction targets objects resting on a dominant support plane:
1. back-project frozen linear depth into world coordinates;
2. fit the dominant local support plane without segmentation;
3. use height above the plane plus RGB contrast to isolate the clicked
   connected protrusion;
4. preserve its local RGBD-connected components, skeletonize each component,
   estimate tube radius from metric image width, and mesh the resulting voxel
   volume.

This is a geometry prior for branches, twigs, cables, and similar thin objects.
It is not a general single-view completion method for arbitrary solid objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


DEPTH_MESH_BUILD = "rgb_depth_3d_components_calibrated_tubes_v3"


@dataclass
class SupportPlane:
    normal: np.ndarray
    offset: float
    origin: np.ndarray
    basis_x: np.ndarray
    basis_y: np.ndarray
    inlier_count: int
    residual_median_mm: float
    residual_p95_mm: float

    def signed_height(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        return pts @ self.normal + float(self.offset)

    def project_to_plane(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        h = self.signed_height(pts)
        return pts - h[..., None] * self.normal


@dataclass
class DepthMeshReconstruction:
    mesh: trimesh.Trimesh
    plane: SupportPlane
    selected_mask: np.ndarray
    height_image_m: np.ndarray
    selected_surface_points: np.ndarray
    selected_heights_m: np.ndarray
    centerline_points: np.ndarray
    centerline_radii_m: np.ndarray
    rgb_floor_median: np.ndarray
    metadata: Dict[str, Any]


def quat_to_mat_xyzw(quat: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in np.asarray(quat, dtype=np.float64).reshape(4)]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def backproject_depth_world(
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project USD linear -Z depth to a dense world-coordinate image."""
    dep = np.asarray(depth, dtype=np.float64)
    if dep.ndim == 3:
        dep = dep[..., 0]
    if dep.ndim != 2:
        raise ValueError(f"depth must be HxW, got {dep.shape}")
    h, w = dep.shape
    fx = float(focal_length) / float(horizontal_aperture) * float(w)
    if not np.isfinite(fx) or fx <= 0.0:
        raise ValueError(f"invalid camera intrinsics fx={fx}")

    vv, uu = np.indices((h, w), dtype=np.float64)
    x_cam = (uu - w / 2.0) / fx * dep
    y_cam = -(vv - h / 2.0) / fx * dep
    z_cam = -dep
    cam_points = np.stack((x_cam, y_cam, z_cam), axis=-1)

    rotation = quat_to_mat_xyzw(camera_quat_xyzw)
    position = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    world = cam_points @ rotation.T + position
    valid = np.isfinite(dep) & (dep > 0.03) & (dep < 20.0)
    valid &= np.isfinite(world).all(axis=-1)
    return world, valid


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n /= np.linalg.norm(n) + 1e-12
    x_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    basis_x = x_world - n * float(np.dot(x_world, n))
    if np.linalg.norm(basis_x) < 1e-4:
        y_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        basis_x = y_world - n * float(np.dot(y_world, n))
    basis_x /= np.linalg.norm(basis_x) + 1e-12
    basis_y = np.cross(n, basis_x)
    basis_y /= np.linalg.norm(basis_y) + 1e-12
    return basis_x, basis_y


def fit_dominant_support_plane(
    world_points: np.ndarray,
    valid_mask: np.ndarray,
    *,
    roi: Tuple[int, int, int, int],
    histogram_bin_m: float = 0.002,
    inlier_band_m: float = 0.003,
) -> SupportPlane:
    """Fit z=ax+by+c around the dominant local height mode."""
    points = np.asarray(world_points, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    x0, y0, x1, y1 = [int(v) for v in roi]
    local = points[y0:y1, x0:x1]
    local_valid = valid[y0:y1, x0:x1]
    samples = local[local_valid]
    if len(samples) < 100:
        raise ValueError(f"support-plane ROI has too few valid points: {len(samples)}")

    z = samples[:, 2]
    lo, hi = np.percentile(z, [1.0, 99.0])
    if hi - lo < histogram_bin_m:
        mode_z = float(np.median(z))
    else:
        edges = np.arange(
            math.floor(lo / histogram_bin_m) * histogram_bin_m,
            math.ceil(hi / histogram_bin_m) * histogram_bin_m + histogram_bin_m,
            histogram_bin_m,
        )
        hist, edges = np.histogram(z, bins=edges)
        mode_i = int(np.argmax(hist))
        mode_z = float(0.5 * (edges[mode_i] + edges[mode_i + 1]))

    coarse = samples[np.abs(z - mode_z) <= max(inlier_band_m * 2.0, histogram_bin_m)]
    if len(coarse) < 100:
        raise ValueError(f"support-plane mode has too few points: {len(coarse)}")

    design = np.column_stack((coarse[:, 0], coarse[:, 1], np.ones(len(coarse))))
    coef, *_ = np.linalg.lstsq(design, coarse[:, 2], rcond=None)
    predicted = design @ coef
    residual = coarse[:, 2] - predicted
    keep = np.abs(residual - np.median(residual)) <= inlier_band_m
    inliers = coarse[keep]
    if len(inliers) < 100:
        inliers = coarse

    design = np.column_stack((inliers[:, 0], inliers[:, 1], np.ones(len(inliers))))
    coef, *_ = np.linalg.lstsq(design, inliers[:, 2], rcond=None)
    a, b, c = [float(v) for v in coef]
    scale = math.sqrt(a * a + b * b + 1.0)
    normal = np.array([-a, -b, 1.0], dtype=np.float64) / scale
    offset = -c / scale
    if normal[2] < 0.0:
        normal = -normal
        offset = -offset
    basis_x, basis_y = _plane_basis(normal)
    origin = -offset * normal

    signed = inliers @ normal + offset
    abs_mm = np.abs(signed) * 1000.0
    return SupportPlane(
        normal=normal,
        offset=float(offset),
        origin=origin,
        basis_x=basis_x,
        basis_y=basis_y,
        inlier_count=int(len(inliers)),
        residual_median_mm=float(np.median(abs_mm)),
        residual_p95_mm=float(np.percentile(abs_mm, 95.0)),
    )


def _depth_component_labels(
    mask: np.ndarray,
    *,
    world_points: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    focal_px: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Label components whose neighboring pixels are continuous in RGBD."""
    raw = np.asarray(mask, dtype=bool)
    if not raw.any():
        return (
            np.full(raw.shape, -1, dtype=np.int32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int32),
        )
    h, w = raw.shape
    yy, xx = np.nonzero(raw)
    n = int(len(xx))
    node_at = np.full((h, w), -1, dtype=np.int32)
    node_at[yy, xx] = np.arange(n, dtype=np.int32)
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)

    def find(node: int) -> int:
        while int(parent[node]) != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(first: int, second: int) -> None:
        a, b = find(first), find(second)
        if a == b:
            return
        if int(size[a]) < int(size[b]):
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    color = np.asarray(rgb, dtype=np.float64)
    dep = np.asarray(depth, dtype=np.float64)
    points = np.asarray(world_points, dtype=np.float64)
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

            p1 = points[yy[first], xx[first]]
            p2 = points[yy[second], xx[second]]
            distance_3d = np.linalg.norm(p1 - p2, axis=1)
            color_distance = np.linalg.norm(
                color[yy[first], xx[first]] - color[yy[second], xx[second]],
                axis=1,
            )
            pixel_span = math.sqrt(float(dx * dx + dy * dy))
            meters_per_pixel = np.maximum(
                dep[yy[first], xx[first]],
                dep[yy[second], xx[second]],
            ) / max(float(focal_px), 1e-9)
            edge_limit = 1.4 * pixel_span * meters_per_pixel + 0.0015
            keep = (distance_3d <= edge_limit) & (color_distance <= 120.0)
            for a, b in zip(first[keep], second[keep]):
                union(int(a), int(b))

    roots = np.fromiter(
        (find(index) for index in range(n)),
        count=n,
        dtype=np.int32,
    )
    _unique_roots, component_for_node = np.unique(
        roots,
        return_inverse=True,
    )
    component_for_node = component_for_node.astype(np.int32, copy=False)
    labels = np.full((h, w), -1, dtype=np.int32)
    labels[yy, xx] = component_for_node
    return labels, yy, xx, component_for_node


def _component_near_click(
    mask: np.ndarray,
    click_u: int,
    click_v: int,
    *,
    world_points: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    focal_px: float,
    merge_gap_3d_m: float = 0.010,
    merge_gap_px: float = 5.0,
) -> np.ndarray:
    """Return the clicked RGB/depth component using local 3D continuity.

    A plain 2D component merges visually crossing branches. Edges here require
    neighboring pixels to also be close in 3D, which keeps the foreground
    branch connected while leaving an occluded crossing branch separate.
    """
    raw = np.asarray(mask, dtype=bool)
    if not raw.any():
        return raw
    h, w = raw.shape
    u = int(np.clip(click_u, 0, w - 1))
    v = int(np.clip(click_v, 0, h - 1))
    _labels, yy, xx, component_for_node = _depth_component_labels(
        raw,
        world_points=world_points,
        rgb=rgb,
        depth=depth,
        focal_px=focal_px,
    )
    points = np.asarray(world_points, dtype=np.float64)
    nearest = int(np.argmin((xx - u) ** 2 + (yy - v) ** 2))
    selected_components = {int(component_for_node[nearest])}

    # Thin branches often disappear for a few anti-aliased pixels or behind a
    # crossing limb. Reattach only components that are close in both image and
    # 3D space. The dual gate avoids merging another branch that merely crosses
    # in the image at a different depth.
    unique_components = np.unique(component_for_node)
    while True:
        selected_nodes = np.isin(
            component_for_node,
            list(selected_components),
        )
        selected_world = points[yy[selected_nodes], xx[selected_nodes]]
        selected_uv = np.column_stack((xx[selected_nodes], yy[selected_nodes]))
        world_tree = cKDTree(selected_world)
        added: List[int] = []
        for candidate_component in unique_components:
            component_value = int(candidate_component)
            if component_value in selected_components:
                continue
            candidate_nodes = component_for_node == component_value
            candidate_world = points[yy[candidate_nodes], xx[candidate_nodes]]
            distance_3d, nearest_selected = world_tree.query(candidate_world, k=1)
            best = int(np.argmin(distance_3d))
            if float(distance_3d[best]) > float(merge_gap_3d_m):
                continue
            candidate_uv = np.array(
                [xx[candidate_nodes][best], yy[candidate_nodes][best]],
                dtype=np.float64,
            )
            paired_uv = selected_uv[int(nearest_selected[best])]
            if float(np.linalg.norm(candidate_uv - paired_uv)) <= float(merge_gap_px):
                added.append(component_value)
        if not added:
            break
        selected_components.update(added)

    selected_nodes = np.isin(
        component_for_node,
        list(selected_components),
    )
    selected = np.zeros_like(raw)
    selected[yy[selected_nodes], xx[selected_nodes]] = True
    return selected


def _mesh_from_skeleton_tubes(
    selected_mask: np.ndarray,
    world_points: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera_pos: np.ndarray,
    focal_px: float,
    voxel_m: float,
    radius_scale: float,
    radius_min_m: float,
    radius_max_m: float,
    centerline_offset_scale: float,
) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Turn RGBD-connected silhouettes into a watertight union of local tubes.

    Each 3D-connected component is skeletonized independently. This prevents
    branches that only overlap in image space from becoming a false, thick 2D
    junction whose volume grows quadratically with the estimated radius.
    """
    selected = np.asarray(selected_mask, dtype=bool)
    labels, _yy, _xx, component_for_node = _depth_component_labels(
        selected,
        world_points=world_points,
        rgb=rgb,
        depth=depth,
        focal_px=focal_px,
    )
    camera = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    points = np.asarray(world_points, dtype=np.float64)
    dep = np.asarray(depth, dtype=np.float64)
    centerline_parts: List[np.ndarray] = []
    radius_parts: List[np.ndarray] = []
    raw_radius_parts: List[np.ndarray] = []
    component_sizes = np.bincount(component_for_node).astype(np.int64)
    skeleton_sizes: List[int] = []
    for component in range(int(len(component_sizes))):
        component_mask = labels == component
        skeleton = skeletonize(component_mask)
        sy, sx = np.nonzero(skeleton)
        if len(sx) == 0:
            continue
        surface = points[sy, sx]
        source_depth = dep[sy, sx]
        radius_px = ndimage.distance_transform_edt(component_mask)[sy, sx]
        raw_radii = (
            np.maximum(radius_px - 0.25, 0.25)
            * source_depth
            / max(float(focal_px), 1e-9)
        )
        radii = np.clip(
            raw_radii * float(radius_scale),
            float(radius_min_m),
            float(radius_max_m),
        )

        view_ray = surface - camera
        view_ray /= np.linalg.norm(view_ray, axis=1, keepdims=True) + 1e-12
        # Radius calibration compensates raster quantization. It must not also
        # move the centerline by the same factor, especially at projected
        # crossings. Use a separately calibrated, bounded camera-ray offset.
        centerline_offset = np.minimum(
            raw_radii * float(centerline_offset_scale),
            radii,
        )
        centerline_parts.append(
            surface + view_ray * centerline_offset[:, None]
        )
        radius_parts.append(radii)
        raw_radius_parts.append(raw_radii)
        skeleton_sizes.append(int(len(sx)))

    if not centerline_parts:
        raise ValueError("clicked RGBD components have no skeleton points")
    centerline = np.concatenate(centerline_parts, axis=0)
    radii = np.concatenate(radius_parts, axis=0)
    raw_radii = np.concatenate(raw_radius_parts, axis=0)
    if len(centerline) < 3:
        raise ValueError(
            f"clicked component skeleton is too small: {len(centerline)}"
        )

    margin = float(radii.max()) + 2.0 * voxel_m
    lower = np.floor((centerline.min(axis=0) - margin) / voxel_m) * voxel_m
    upper = np.ceil((centerline.max(axis=0) + margin) / voxel_m) * voxel_m
    shape = np.rint((upper - lower) / voxel_m).astype(np.int64) + 1
    if np.any(shape <= 2) or int(np.prod(shape)) > 80_000_000:
        raise ValueError(f"invalid tube voxel grid: {shape.tolist()}")
    occupancy = np.zeros(tuple(int(v) for v in shape), dtype=bool)

    for center, radius in zip(centerline, radii):
        center_index = np.rint((center - lower) / voxel_m).astype(np.int64)
        reach = int(math.ceil(float(radius) / voxel_m)) + 1
        axes = [
            np.arange(
                max(0, int(center_index[axis]) - reach),
                min(int(shape[axis]), int(center_index[axis]) + reach + 1),
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
        raise ValueError(f"tube occupancy too small: {int(occupancy.sum())} voxels")
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(
        occupancy,
        pitch=float(voxel_m),
    )
    mesh.vertices += lower
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    meta = {
        "grid_shape": [int(v) for v in shape],
        "occupied_voxels": int(occupancy.sum()),
        "skeleton_points": int(len(centerline)),
        "rgbd_components": int(len(component_sizes)),
        "rgbd_component_pixels_desc": sorted(
            [int(value) for value in component_sizes],
            reverse=True,
        ),
        "rgbd_component_skeleton_points": skeleton_sizes,
        "raw_radius_min_mm": float(raw_radii.min() * 1000.0),
        "raw_radius_median_mm": float(np.median(raw_radii) * 1000.0),
        "raw_radius_max_mm": float(raw_radii.max() * 1000.0),
        "tube_radius_min_mm": float(radii.min() * 1000.0),
        "tube_radius_median_mm": float(np.median(radii) * 1000.0),
        "tube_radius_max_mm": float(radii.max() * 1000.0),
    }
    return mesh, centerline, radii, meta


def reconstruct_supported_object_mesh(
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
    voxel_mm: float = 1.25,
    min_height_mm: float = 3.0,
    strong_height_mm: float = 7.0,
    max_height_mm: float = 120.0,
    min_rgb_delta: float = 16.0,
    tube_radius_scale: float = 1.04,
    tube_radius_min_mm: float = 1.55,
    tube_radius_max_mm: float = 3.5,
    centerline_offset_scale: float = 0.75,
) -> DepthMeshReconstruction:
    """Reconstruct the clicked thin object without segmentation."""
    color = np.asarray(rgb)
    if color.ndim != 3 or color.shape[2] < 3:
        raise ValueError(f"rgb must be HxWx3, got {color.shape}")
    color = color[..., :3].astype(np.float64)
    dep = np.asarray(depth)
    h, w = dep.shape[:2]
    if color.shape[:2] != (h, w):
        raise ValueError(f"rgb/depth size mismatch: {color.shape[:2]} vs {(h, w)}")

    u = int(np.clip(click_u, 0, w - 1))
    v = int(np.clip(click_v, 0, h - 1))
    radius = max(20, int(roi_radius_px))
    x0, x1 = max(0, u - radius), min(w, u + radius + 1)
    y0, y1 = max(0, v - radius), min(h, v + radius + 1)

    world, valid = backproject_depth_world(
        dep,
        camera_pos=camera_pos,
        camera_quat_xyzw=camera_quat_xyzw,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
    )
    plane = fit_dominant_support_plane(
        world,
        valid,
        roi=(x0, y0, x1, y1),
    )
    height = plane.signed_height(world)

    floor_pixels = valid & (np.abs(height) <= 0.003)
    floor_pixels[:y0, :] = False
    floor_pixels[y1:, :] = False
    floor_pixels[:, :x0] = False
    floor_pixels[:, x1:] = False
    if int(floor_pixels.sum()) < 100:
        floor_pixels = valid & (np.abs(height) <= 0.005)
    floor_rgb = np.median(color[floor_pixels], axis=0)
    rgb_delta = np.linalg.norm(color - floor_rgb.reshape(1, 1, 3), axis=-1)

    min_h = float(min_height_mm) / 1000.0
    strong_h = float(strong_height_mm) / 1000.0
    max_h = float(max_height_mm) / 1000.0
    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[y0:y1, x0:x1] = True
    candidate = (
        valid
        & roi_mask
        & (height >= min_h)
        & (height <= max_h)
        & ((height >= strong_h) | (rgb_delta >= float(min_rgb_delta)))
    )
    fx = float(focal_length) / float(horizontal_aperture) * float(w)
    selected = _component_near_click(
        candidate,
        u,
        v,
        world_points=world,
        rgb=color,
        depth=dep,
        focal_px=fx,
    )
    if int(selected.sum()) < 12:
        raise ValueError(
            f"clicked protrusion has too few depth pixels: {int(selected.sum())}"
        )

    selected_points = world[selected]
    selected_heights = height[selected]
    mesh, centerline, centerline_radii, grid_meta = _mesh_from_skeleton_tubes(
        selected,
        world,
        color,
        dep,
        camera_pos=np.asarray(camera_pos, dtype=np.float64),
        focal_px=fx,
        voxel_m=float(voxel_mm) / 1000.0,
        radius_scale=float(tube_radius_scale),
        radius_min_m=float(tube_radius_min_mm) / 1000.0,
        radius_max_m=float(tube_radius_max_mm) / 1000.0,
        centerline_offset_scale=float(centerline_offset_scale),
    )
    metadata: Dict[str, Any] = {
        "build": DEPTH_MESH_BUILD,
        "click_native": [u, v],
        "roi_native": [x0, y0, x1, y1],
        "selected_pixels": int(selected.sum()),
        "candidate_pixels": int(candidate.sum()),
        "floor_pixels": int(floor_pixels.sum()),
        "floor_normal": plane.normal.tolist(),
        "floor_offset": float(plane.offset),
        "floor_inliers": int(plane.inlier_count),
        "floor_residual_median_mm": float(plane.residual_median_mm),
        "floor_residual_p95_mm": float(plane.residual_p95_mm),
        "floor_rgb_median": floor_rgb.tolist(),
        "voxel_mm": float(voxel_mm),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_volume_cm3": float(abs(mesh.volume) * 1e6),
        "tube_radius_scale": float(tube_radius_scale),
        "tube_radius_min_config_mm": float(tube_radius_min_mm),
        "tube_radius_max_config_mm": float(tube_radius_max_mm),
        "centerline_offset_scale": float(centerline_offset_scale),
        **grid_meta,
    }
    return DepthMeshReconstruction(
        mesh=mesh,
        plane=plane,
        selected_mask=selected,
        height_image_m=height,
        selected_surface_points=selected_points,
        selected_heights_m=selected_heights,
        centerline_points=centerline,
        centerline_radii_m=centerline_radii,
        rgb_floor_median=floor_rgb,
        metadata=metadata,
    )


def select_evaluation_centers(
    reconstruction: DepthMeshReconstruction,
    reference_mesh: trimesh.Trimesh,
    *,
    n_centers: int = 6,
    reference_surface_tol_mm: float = 4.0,
    min_separation_mm: float = 18.0,
    sphere_radius_mm: float = 20.0,
    eval_voxel_mm: float = 1.5,
    min_reference_overlap_cm3: float = 0.25,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Choose spatially separated branch centers from reconstructed depth points.

    The target mesh is used only here to reject pixels belonging to another
    crossing branch or the floor. Reconstruction itself never sees the target
    mesh or segmentation.
    """
    points = reconstruction.centerline_points
    radii = reconstruction.centerline_radii_m
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64), {"error": "no surface points"}

    closest, distance, _tri = trimesh.proximity.closest_point(reference_mesh, points)
    distance = np.asarray(distance, dtype=np.float64)
    eligible = (
        np.isfinite(distance)
        & (
            distance
            <= radii + float(reference_surface_tol_mm) / 1000.0
        )
    )
    candidates = points[eligible]
    candidate_radii = radii[eligible]
    candidate_distance = distance[eligible]
    if len(candidates) == 0:
        return np.zeros((0, 3), dtype=np.float64), {
            "error": "no reconstructed points close to reference mesh",
            "nearest_reference_mm": float(np.min(distance) * 1000.0),
        }

    # Build a compact candidate pool before the more expensive reference-volume
    # check. This rejects endpoints where a 2 cm sphere contains almost no
    # target material and relative volume error would be numerically unstable.
    pool: List[int] = []
    for index in np.argsort(candidate_distance):
        if all(
            np.linalg.norm(candidates[index] - candidates[other]) > 0.006
            for other in pool
        ):
            pool.append(int(index))
        if len(pool) >= 50:
            break

    voxel_m = float(eval_voxel_mm) / 1000.0
    voxel_cm3 = voxel_m ** 3 * 1e6
    reference_volume: Dict[int, float] = {}
    for index in pool:
        query = sphere_query_points(
            candidates[index],
            radius_m=float(sphere_radius_mm) / 1000.0,
            voxel_m=voxel_m,
        )
        occupied, _method = _mesh_occupancy(
            reference_mesh,
            query,
            voxel_m=voxel_m,
        )
        reference_volume[index] = float(int(occupied.sum()) * voxel_cm3)
    viable = [
        index
        for index in pool
        if reference_volume[index] >= float(min_reference_overlap_cm3)
    ]
    if len(viable) < 3:
        return np.zeros((0, 3), dtype=np.float64), {
            "error": "too few sphere centers with stable reference volume",
            "candidate_pool": int(len(pool)),
            "viable_centers": int(len(viable)),
            "min_reference_overlap_cm3": float(min_reference_overlap_cm3),
        }

    chosen: List[int] = [
        max(viable, key=lambda index: reference_volume[index])
    ]
    min_sep = float(min_separation_mm) / 1000.0
    while len(chosen) < int(n_centers):
        viable_arr = np.asarray(viable, dtype=np.int64)
        chosen_pts = candidates[np.asarray(chosen, dtype=np.int64)]
        dist_to_chosen = np.linalg.norm(
            candidates[viable_arr, None, :] - chosen_pts[None, :, :],
            axis=-1,
        ).min(axis=1)
        dist_to_chosen[np.isin(viable_arr, chosen)] = -1.0
        next_pos = int(np.argmax(dist_to_chosen))
        if float(dist_to_chosen[next_pos]) < min_sep:
            break
        chosen.append(int(viable_arr[next_pos]))

    selected_centers = candidates[np.asarray(chosen, dtype=np.int64)]
    return selected_centers, {
        "eligible_points": int(len(candidates)),
        "candidate_pool": int(len(pool)),
        "viable_centers": int(len(viable)),
        "selected_centers": int(len(selected_centers)),
        "reference_surface_tol_mm": float(reference_surface_tol_mm),
        "min_separation_mm": float(min_separation_mm),
        "min_reference_overlap_cm3": float(min_reference_overlap_cm3),
        "selected_tube_radius_mm": (
            candidate_radii[np.asarray(chosen)] * 1000.0
        ).tolist(),
        "selected_reference_distance_mm": (
            candidate_distance[np.asarray(chosen)] * 1000.0
        ).tolist(),
        "selected_reference_overlap_cm3": [
            float(reference_volume[index]) for index in chosen
        ],
    }


def sphere_query_points(
    center: Sequence[float],
    *,
    radius_m: float,
    voxel_m: float,
) -> np.ndarray:
    center_arr = np.asarray(center, dtype=np.float64).reshape(3)
    axis = np.arange(
        -float(radius_m) + 0.5 * float(voxel_m),
        float(radius_m),
        float(voxel_m),
        dtype=np.float64,
    )
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    local = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    local = local[np.einsum("ij,ij->i", local, local) <= float(radius_m) ** 2]
    return local + center_arr


def _mesh_occupancy(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    *,
    voxel_m: float,
) -> Tuple[np.ndarray, str]:
    if bool(mesh.is_watertight):
        try:
            return np.asarray(mesh.contains(points), dtype=bool), "contains"
        except Exception:
            pass

    voxels = mesh.voxelized(pitch=float(voxel_m))
    try:
        voxels = voxels.fill()
    except Exception:
        pass
    voxel_points = np.asarray(voxels.points, dtype=np.float64)
    if len(voxel_points) == 0:
        return np.zeros(len(points), dtype=bool), "empty_voxelized"
    tree = cKDTree(voxel_points)
    distance, _ = tree.query(points, k=1)
    threshold = float(voxel_m) * math.sqrt(3.0) * 0.55
    return np.asarray(distance <= threshold, dtype=bool), "filled_voxel_nearest"


def compare_sphere_mesh_overlaps(
    centers: Iterable[Sequence[float]],
    reference_mesh: trimesh.Trimesh,
    reconstructed_mesh: trimesh.Trimesh,
    *,
    sphere_radius_mm: float = 20.0,
    eval_voxel_mm: float = 1.5,
) -> List[Dict[str, Any]]:
    radius_m = float(sphere_radius_mm) / 1000.0
    voxel_m = float(eval_voxel_mm) / 1000.0
    voxel_cm3 = voxel_m ** 3 * 1e6
    rows: List[Dict[str, Any]] = []
    for index, center in enumerate(centers):
        query = sphere_query_points(center, radius_m=radius_m, voxel_m=voxel_m)
        ref_occ, ref_method = _mesh_occupancy(
            reference_mesh,
            query,
            voxel_m=voxel_m,
        )
        rec_occ, rec_method = _mesh_occupancy(
            reconstructed_mesh,
            query,
            voxel_m=voxel_m,
        )
        ref_n = int(ref_occ.sum())
        rec_n = int(rec_occ.sum())
        intersection = int(np.count_nonzero(ref_occ & rec_occ))
        union = int(np.count_nonzero(ref_occ | rec_occ))
        ref_vol = float(ref_n * voxel_cm3)
        rec_vol = float(rec_n * voxel_cm3)
        rows.append(
            {
                "index": int(index),
                "center_world": np.asarray(center, dtype=np.float64).tolist(),
                "sphere_radius_mm": float(sphere_radius_mm),
                "eval_voxel_mm": float(eval_voxel_mm),
                "sphere_query_voxels": int(len(query)),
                "reference_method": ref_method,
                "reconstruction_method": rec_method,
                "reference_overlap_voxels": ref_n,
                "reconstructed_overlap_voxels": rec_n,
                "reference_overlap_cm3": ref_vol,
                "reconstructed_overlap_cm3": rec_vol,
                "signed_error_cm3": float(rec_vol - ref_vol),
                "absolute_error_cm3": float(abs(rec_vol - ref_vol)),
                "relative_error": float(abs(rec_vol - ref_vol) / max(ref_vol, voxel_cm3)),
                "intersection_voxels": intersection,
                "union_voxels": union,
                "iou": float(intersection / max(union, 1)),
                "recall": float(intersection / max(ref_n, 1)),
                "precision": float(intersection / max(rec_n, 1)),
            }
        )
    return rows


def aggregate_overlap_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "sphere_count": 0,
            "median_relative_error": float("inf"),
            "p80_relative_error": float("inf"),
            "p90_relative_error": float("inf"),
            "median_iou": 0.0,
            "median_recall": 0.0,
            "median_precision": 0.0,
        }
    relative = np.asarray([float(row["relative_error"]) for row in rows])
    iou = np.asarray([float(row["iou"]) for row in rows])
    recall = np.asarray([float(row["recall"]) for row in rows])
    precision = np.asarray([float(row["precision"]) for row in rows])
    reference_volume = np.asarray(
        [float(row["reference_overlap_cm3"]) for row in rows]
    )
    reconstructed_volume = np.asarray(
        [float(row["reconstructed_overlap_cm3"]) for row in rows]
    )
    signed_error = reconstructed_volume - reference_volume
    return {
        "sphere_count": int(len(rows)),
        "median_relative_error": float(np.median(relative)),
        "p80_relative_error": float(np.percentile(relative, 80.0)),
        "p90_relative_error": float(np.percentile(relative, 90.0)),
        "max_relative_error": float(np.max(relative)),
        "within_10_percent": int(np.count_nonzero(relative <= 0.10)),
        "within_15_percent": int(np.count_nonzero(relative <= 0.15)),
        "within_30_percent": int(np.count_nonzero(relative <= 0.30)),
        "within_40_percent": int(np.count_nonzero(relative <= 0.40)),
        "within_50_percent": int(np.count_nonzero(relative <= 0.50)),
        "median_iou": float(np.median(iou)),
        "min_iou": float(np.min(iou)),
        "median_recall": float(np.median(recall)),
        "median_precision": float(np.median(precision)),
        "mean_reference_overlap_cm3": float(np.mean(reference_volume)),
        "mean_reconstructed_overlap_cm3": float(
            np.mean(reconstructed_volume)
        ),
        "mean_absolute_error_cm3": float(np.mean(np.abs(signed_error))),
        "rmse_cm3": float(np.sqrt(np.mean(signed_error ** 2))),
        "mean_signed_error_cm3": float(np.mean(signed_error)),
        "volume_correlation": float(
            np.corrcoef(reference_volume, reconstructed_volume)[0, 1]
        ),
    }
