"""Project the plan-grasp opening wedge into a wrist camera.

For each wrist-image pixel covered by the strict two-finger opening, the
projection stores the nearest and farthest camera-forward z-depth where the
pixel ray intersects the closed opening surface.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np


WRIST_GRASP_ZONE_BUILD = (
    "dynamic_continuous_finger_corridor_near_far_eef_calibrated_v7"
)
_MIN_CAMERA_DEPTH_M = 1e-4


def _quat_to_mat(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in np.asarray(quat_xyzw, dtype=np.float64).reshape(4)]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _section_corners(x_half: float, y_lo: float, y_hi: float, z: float) -> np.ndarray:
    return np.array([
        [-x_half, y_lo, z],
        [+x_half, y_lo, z],
        [+x_half, y_hi, z],
        [-x_half, y_hi, z],
    ], dtype=np.float64)


def opening_surface_triangles_eef(gripper_qpos=None) -> np.ndarray:
    """Return the closed outer surface of the strict plan-grasp opening."""
    from behavior_interface.skills.plan_grasp_gripper_geom import (
        normalize_gripper_qpos,
    )

    q1, q2 = normalize_gripper_qpos(gripper_qpos)
    return _opening_surface_triangles_eef_cached(q1, q2)


@lru_cache(maxsize=64)
def _opening_surface_triangles_eef_cached(q1: float, q2: float) -> np.ndarray:
    from behavior_interface.skills.plan_grasp_gripper_geom import (
        gap_opening_strict_mask_from_lut,
        get_wrist_opening_lut,
    )

    gripper_qpos = (q1, q2)
    lut = get_wrist_opening_lut(gripper_qpos)
    z_lo = float(lut["z_grasp_lo"])
    z_hi = float(lut["z_grasp_hi"])
    if not np.isfinite(z_lo) or not np.isfinite(z_hi) or z_hi < z_lo:
        return np.zeros((0, 3, 3), dtype=np.float64)
    z_samples = np.unique(np.concatenate([
        np.array([z_lo, z_hi], dtype=np.float64),
        np.asarray(lut["z"], dtype=np.float64),
    ]))
    z_samples = z_samples[(z_samples >= z_lo) & (z_samples <= z_hi)]

    sections = []
    for z in z_samples:
        y_lo = float(np.interp(z, lut["z"], lut["y_inner_lo"]))
        y_hi = float(np.interp(z, lut["z"], lut["y_inner_hi"]))
        x_half = float(np.interp(z, lut["z"], lut["x_half_gap"]))
        center = np.array([[0.0, 0.5 * (y_lo + y_hi), z]], dtype=np.float64)
        is_open = bool(
            gap_opening_strict_mask_from_lut(center, lut)[0]
        )
        sections.append((is_open, _section_corners(x_half, y_lo, y_hi, float(z))))

    groups = []
    current = []
    for is_open, corners in sections:
        if is_open:
            current.append(corners)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    triangles = []
    for group in groups:
        if len(group) < 2:
            continue
        first = group[0]
        last = group[-1]
        triangles.extend([
            first[[0, 2, 1]],
            first[[0, 3, 2]],
            last[[0, 1, 2]],
            last[[0, 2, 3]],
        ])
        for lower, upper in zip(group[:-1], group[1:]):
            for edge in range(4):
                nxt = (edge + 1) % 4
                triangles.extend([
                    np.array([lower[edge], lower[nxt], upper[nxt]]),
                    np.array([lower[edge], upper[nxt], upper[edge]]),
                ])

    if not triangles:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.asarray(triangles, dtype=np.float64)


def _rasterize_depth_bounds(
    uv: np.ndarray,
    depth: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    near = np.full((image_height, image_width), np.nan, dtype=np.float32)
    far = np.full((image_height, image_width), np.nan, dtype=np.float32)
    for tri_uv, tri_depth in zip(uv, depth):
        if not np.all(np.isfinite(tri_uv)) or not np.all(np.isfinite(tri_depth)):
            continue
        if np.any(tri_depth <= _MIN_CAMERA_DEPTH_M):
            continue

        min_u = max(0, int(np.floor(np.min(tri_uv[:, 0]))))
        max_u = min(image_width - 1, int(np.ceil(np.max(tri_uv[:, 0]))))
        min_v = max(0, int(np.floor(np.min(tri_uv[:, 1]))))
        max_v = min(image_height - 1, int(np.ceil(np.max(tri_uv[:, 1]))))
        if min_u > max_u or min_v > max_v:
            continue

        u0, v0 = tri_uv[0]
        u1, v1 = tri_uv[1]
        u2, v2 = tri_uv[2]
        denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if abs(float(denom)) < 1e-10:
            continue

        xs = np.arange(min_u, max_u + 1, dtype=np.float64) + 0.5
        ys = np.arange(min_v, max_v + 1, dtype=np.float64) + 0.5
        xx, yy = np.meshgrid(xs, ys)
        w0 = ((v1 - v2) * (xx - u2) + (u2 - u1) * (yy - v2)) / denom
        w1 = ((v2 - v0) * (xx - u2) + (u0 - u2) * (yy - v2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        if not np.any(inside):
            continue

        inv_depth = (
            w0 / float(tri_depth[0])
            + w1 / float(tri_depth[1])
            + w2 / float(tri_depth[2])
        )
        tri_z = np.divide(
            1.0,
            inv_depth,
            out=np.full_like(inv_depth, np.nan),
            where=np.abs(inv_depth) > 1e-12,
        )
        near_patch = near[min_v:max_v + 1, min_u:max_u + 1]
        far_patch = far[min_v:max_v + 1, min_u:max_u + 1]
        valid = inside & np.isfinite(tri_z)
        near_update = valid & (
            ~np.isfinite(near_patch) | (tri_z < near_patch)
        )
        far_update = valid & (
            ~np.isfinite(far_patch) | (tri_z > far_patch)
        )
        near_patch[near_update] = tri_z[near_update].astype(np.float32)
        far_patch[far_update] = tri_z[far_update].astype(np.float32)
    return near, far


def project_opening_depth_bounds(
    *,
    eef_pos_world: np.ndarray,
    eef_quat_world: np.ndarray,
    camera_pos_world: np.ndarray,
    camera_quat_world: np.ndarray,
    focal_length: float,
    horizontal_aperture: float,
    image_width: int,
    image_height: int,
    gripper_qpos=None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Project the strict opening surface and return near/far z-depth per pixel."""
    from behavior_interface.skills.plan_grasp_gripper_geom import (
        normalize_gripper_qpos,
    )

    normalized_qpos = normalize_gripper_qpos(gripper_qpos)
    triangles_eef = opening_surface_triangles_eef(normalized_qpos)
    if len(triangles_eef) == 0:
        empty = np.full(
            (int(image_height), int(image_width)),
            np.nan,
            dtype=np.float32,
        )
        return empty.copy(), empty, {
            "build": WRIST_GRASP_ZONE_BUILD,
            "definition": "central opening between the two gripper fingers",
            "gripper_qpos_m": [float(v) for v in normalized_qpos],
            "depth_semantics": "empty opening; no valid near/far interval",
            "zone_pixel_n": 0,
            "uv_bbox": None,
            "depth_near_min_m": None,
            "depth_near_max_m": None,
            "depth_far_min_m": None,
            "depth_far_max_m": None,
            "depth_bianjie_min_m": None,
            "depth_bianjie_max_m": None,
            "surface_triangle_n": 0,
            "eef_z_lo_m": None,
            "eef_z_hi_m": None,
        }

    eef_pos = np.asarray(eef_pos_world, dtype=np.float64).reshape(3)
    camera_pos = np.asarray(camera_pos_world, dtype=np.float64).reshape(3)
    r_eef = _quat_to_mat(eef_quat_world)
    r_camera = _quat_to_mat(camera_quat_world)

    points_world = np.einsum("ij,tkj->tki", r_eef, triangles_eef) + eef_pos
    points_camera = np.einsum(
        "ij,tkj->tki",
        r_camera.T,
        points_world - camera_pos,
    )
    z_depth = -points_camera[..., 2]

    fx = float(focal_length) / float(horizontal_aperture) * int(image_width)
    fy = fx
    cx = float(image_width) / 2.0
    cy = float(image_height) / 2.0
    uv = np.empty((*points_camera.shape[:2], 2), dtype=np.float64)
    uv[..., 0] = cx + fx * points_camera[..., 0] / z_depth
    uv[..., 1] = cy - fy * points_camera[..., 1] / z_depth

    depth_near, depth_far = _rasterize_depth_bounds(
        uv,
        z_depth,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    zone = (
        np.isfinite(depth_near)
        & (depth_near > 0.0)
        & np.isfinite(depth_far)
        & (depth_far >= depth_near)
    )
    depth_near = np.where(zone, depth_near, np.nan).astype(np.float32)
    depth_far = np.where(zone, depth_far, np.nan).astype(np.float32)
    if not np.any(zone):
        raise RuntimeError("两爪中间严格开口未投影到 wrist camera 画面")

    vv, uu = np.nonzero(zone)
    return depth_near, depth_far, {
        "build": WRIST_GRASP_ZONE_BUILD,
        "definition": "central opening between the two gripper fingers",
        "gripper_qpos_m": [float(v) for v in normalized_qpos],
        "depth_semantics": (
            "camera forward z-depth interval from the nearest to the farthest "
            "intersection with the closed opening surface"
        ),
        "zone_pixel_n": int(zone.sum()),
        "uv_bbox": [int(uu.min()), int(vv.min()), int(uu.max()), int(vv.max())],
        "depth_near_min_m": float(np.nanmin(depth_near)),
        "depth_near_max_m": float(np.nanmax(depth_near)),
        "depth_far_min_m": float(np.nanmin(depth_far)),
        "depth_far_max_m": float(np.nanmax(depth_far)),
        # Backward-compatible metadata names: depth_bianjie remains the far bound.
        "depth_bianjie_min_m": float(np.nanmin(depth_far)),
        "depth_bianjie_max_m": float(np.nanmax(depth_far)),
        "surface_triangle_n": int(len(triangles_eef)),
        "eef_z_lo_m": float(np.min(triangles_eef[..., 2])),
        "eef_z_hi_m": float(np.max(triangles_eef[..., 2])),
    }


def project_opening_depth_bianjie(
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Backward-compatible wrapper returning only the far depth boundary."""
    _depth_near, depth_far, meta = project_opening_depth_bounds(**kwargs)
    return depth_far, meta


def visible_depth_inside_opening_mask(
    depth_linear: np.ndarray,
    depth_near: np.ndarray,
    depth_far: np.ndarray,
    *,
    eef_pos_world: np.ndarray,
    eef_quat_world: np.ndarray,
    camera_pos_world: np.ndarray,
    camera_quat_world: np.ndarray,
    focal_length: float,
    horizontal_aperture: float,
    gripper_qpos=None,
) -> np.ndarray:
    """Return visible pixels whose reconstructed 3D point is in the strict opening."""
    from behavior_interface.skills.plan_grasp_gripper_geom import (
        gap_opening_strict_mask_from_lut,
        get_wrist_opening_lut,
    )

    depth = np.asarray(depth_linear, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    near = np.asarray(depth_near, dtype=np.float32)
    far = np.asarray(depth_far, dtype=np.float32)
    if depth.shape != near.shape or depth.shape != far.shape:
        raise ValueError(
            f"depth/depth_near/depth_far shape 不一致: "
            f"{depth.shape} / {near.shape} / {far.shape}"
        )

    candidates = (
        np.isfinite(near)
        & (near > 0.0)
        & np.isfinite(far)
        & (far >= near)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (depth >= near)
        & (depth <= far)
    )
    if not np.any(candidates):
        return candidates

    image_height, image_width = depth.shape
    vv, uu = np.nonzero(candidates)
    d = depth[vv, uu].astype(np.float64)
    fx = float(focal_length) / float(horizontal_aperture) * int(image_width)
    fy = fx
    cx = float(image_width) / 2.0
    cy = float(image_height) / 2.0
    points_camera = np.column_stack([
        (uu.astype(np.float64) - cx) / fx * d,
        -(vv.astype(np.float64) - cy) / fy * d,
        -d,
    ])

    camera_pos = np.asarray(camera_pos_world, dtype=np.float64).reshape(3)
    eef_pos = np.asarray(eef_pos_world, dtype=np.float64).reshape(3)
    r_camera = _quat_to_mat(camera_quat_world)
    r_eef = _quat_to_mat(eef_quat_world)
    points_world = points_camera @ r_camera.T + camera_pos
    points_eef = (points_world - eef_pos) @ r_eef
    lut = get_wrist_opening_lut(gripper_qpos)
    strict_inside = gap_opening_strict_mask_from_lut(points_eef, lut)

    mask = np.zeros_like(candidates)
    mask[vv[strict_inside], uu[strict_inside]] = True
    return mask


def overlay_depth_inside_grasp_zone(
    rgb: np.ndarray,
    depth_linear: np.ndarray,
    depth_near: np.ndarray,
    depth_far: np.ndarray,
    *,
    alpha: float = 0.58,
    visible_inside_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add red over valid pixels whose depth lies within the zone bounds."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"RGB shape 非法: {image.shape}")
    image = image[..., :3].astype(np.uint8, copy=False)

    depth = np.asarray(depth_linear, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    near = np.asarray(depth_near, dtype=np.float32)
    far = np.asarray(depth_far, dtype=np.float32)
    if (
        depth.shape != image.shape[:2]
        or near.shape != image.shape[:2]
        or far.shape != image.shape[:2]
    ):
        raise ValueError(
            f"RGB/depth/depth_near/depth_far shape 不一致: "
            f"{image.shape[:2]} / {depth.shape} / {near.shape} / {far.shape}"
        )

    zone = (
        np.isfinite(near)
        & (near > 0.0)
        & np.isfinite(far)
        & (far >= near)
    )
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    depth_inside_bounds = (
        zone
        & valid_depth
        & (depth >= near)
        & (depth <= far)
    )
    if visible_inside_mask is None:
        red_mask = depth_inside_bounds
    else:
        inside = np.asarray(visible_inside_mask, dtype=bool)
        if inside.shape != depth.shape:
            raise ValueError(
                f"visible_inside_mask shape 不一致: {inside.shape} / {depth.shape}"
            )
        red_mask = depth_inside_bounds & inside

    out = image.astype(np.float32)
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    out[red_mask] = (1.0 - a) * out[red_mask] + a * red
    return np.rint(out).astype(np.uint8), red_mask
