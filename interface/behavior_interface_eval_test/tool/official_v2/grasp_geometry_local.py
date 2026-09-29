"""Submission-local geometry helpers for the strict RGB-D grasp planner."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


_ASSET_PATH = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "r1pro_gripper_geometry_3mm.npz",
)
_MIN_GAP_WIDTH_M = 0.002
_RY_PI_MAT = np.diag([-1.0, 1.0, -1.0])
_PALM_T_EEF = np.array([0.0, 0.0, -0.06000], dtype=np.float64)
_RIGHT_REALSENSE_ORIGIN_GRIPPER = np.array(
    [0.05051, 0.0028934, 0.0051317],
    dtype=np.float64,
)
_APPROACH_AXIS_EEF = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@lru_cache(maxsize=1)
def _asset_arrays() -> Dict[str, np.ndarray]:
    with np.load(_ASSET_PATH, allow_pickle=False) as archive:
        return {
            str(name): np.asarray(archive[name]).copy()
            for name in archive.files
        }


def gripper_geometry(voxel_m: float = 0.003) -> Dict[str, Any]:
    arrays = _asset_arrays()
    stored_voxel = float(arrays["voxel_m"].reshape(()))
    if abs(float(voxel_m) - stored_voxel) > 1e-12:
        raise ValueError(
            f"only the frozen {stored_voxel:g} m gripper geometry is available"
        )
    return {
        "voxel_m": stored_voxel,
        "axial_z_m": np.asarray(arrays["axial_z_m"], dtype=np.float64).copy(),
        "gripper_voxels": np.asarray(
            arrays["gripper_voxels"],
            dtype=np.float64,
        ).reshape(-1, 3).copy(),
        "opening_voxels": np.asarray(
            arrays["opening_voxels"],
            dtype=np.float64,
        ).reshape(-1, 3).copy(),
        "components": {
            name: np.asarray(
                arrays[f"component_{name}"],
                dtype=np.float64,
            ).reshape(-1, 3).copy()
            for name in (
                "palm",
                "finger_positive_y",
                "finger_negative_y",
                "camera",
            )
        },
    }


def get_wrist_opening_lut(
    gripper_qpos: Sequence[float] = (0.05, 0.05),
) -> Dict[str, Any]:
    requested = np.asarray(gripper_qpos, dtype=np.float64).reshape(2)
    arrays = _asset_arrays()
    frozen = np.asarray(arrays["lut_gripper_qpos_m"], dtype=np.float64).reshape(2)
    if not np.allclose(requested, frozen, atol=1e-12, rtol=0.0):
        raise ValueError(
            "strict planner contains only the frozen fully-open gripper LUT"
        )
    return {
        key: np.asarray(arrays[f"lut_{key}"], dtype=np.float64).copy()
        for key in (
            "gripper_qpos_m",
            "z",
            "y_inner_lo",
            "y_inner_hi",
            "y_outer_f1",
            "y_outer_f2",
            "x_half_gap",
            "x_half_finger",
            "gap_width",
            "z_grasp_lo",
            "z_grasp_hi",
            "gap_center_local",
            "finger_inner_y",
            "finger_z_tip",
            "finger_z_open",
        )
    }


def _interp_cols(
    z: np.ndarray,
    lut: Dict[str, Any],
    keys: Tuple[str, ...],
) -> Tuple[np.ndarray, ...]:
    z_bins = np.asarray(lut["z"], dtype=np.float64)
    return tuple(
        np.interp(z, z_bins, np.asarray(lut[key], dtype=np.float64))
        for key in keys
    )


def gap_opening_strict_mask_from_lut(
    q: np.ndarray,
    lut: Dict[str, Any],
) -> np.ndarray:
    points = np.asarray(q, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    z = points[:, 2]
    y_lo, y_hi, x_half, gap_width = _interp_cols(
        z,
        lut,
        ("y_inner_lo", "y_inner_hi", "x_half_gap", "gap_width"),
    )
    z_lo = float(np.asarray(lut["z_grasp_lo"]).reshape(()))
    z_hi = float(np.asarray(lut["z_grasp_hi"]).reshape(()))
    valid_z_band = math.isfinite(z_lo) and math.isfinite(z_hi) and z_hi >= z_lo
    in_z = (
        (z >= z_lo) & (z <= z_hi)
        if valid_z_band
        else np.zeros(len(points), dtype=bool)
    )
    in_y = (points[:, 1] > y_lo) & (points[:, 1] < y_hi)
    in_x = np.abs(points[:, 0]) <= x_half
    return in_z & in_y & in_x & (gap_width >= _MIN_GAP_WIDTH_M)


def quat_to_mat_xyzw(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.asarray(
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


def mat_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quat = np.array(
            [
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        quat = np.array(
            [
                0.25 * scale,
                (m[0, 1] + m[1, 0]) / scale,
                (m[0, 2] + m[2, 0]) / scale,
                (m[2, 1] - m[1, 2]) / scale,
            ],
            dtype=np.float64,
        )
    elif m[1, 1] > m[2, 2]:
        scale = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        quat = np.array(
            [
                (m[0, 1] + m[1, 0]) / scale,
                0.25 * scale,
                (m[1, 2] + m[2, 1]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        scale = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        quat = np.array(
            [
                (m[0, 2] + m[2, 0]) / scale,
                (m[1, 2] + m[2, 1]) / scale,
                0.25 * scale,
                (m[1, 0] - m[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def eef_frame_from_approach_roll(
    approach: np.ndarray,
    roll: float,
    ref_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    z = np.asarray(approach, dtype=np.float64).reshape(3)
    z /= np.linalg.norm(z) + 1e-9
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if ref_y is not None:
        y0 = np.asarray(ref_y, dtype=np.float64).reshape(3)
        y0 = y0 - np.dot(y0, z) * z
    else:
        y0 = np.cross(z, up)
        if np.linalg.norm(y0) < 1e-3:
            y0 = np.cross(z, np.array([1.0, 0.0, 0.0]))
    y0 /= np.linalg.norm(y0) + 1e-9
    x0 = np.cross(y0, z)
    cosine, sine = math.cos(roll), math.sin(roll)
    x = cosine * x0 + sine * y0
    y = -sine * x0 + cosine * y0
    return np.column_stack([x, y, z])


def icosa_face_normals() -> np.ndarray:
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    b, c = 1.0 / phi, phi
    vertices = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                vertices.append((sx, sy, sz))
    for sy in (1.0, -1.0):
        for sz in (1.0, -1.0):
            vertices.append((0.0, sy * b, sz * c))
    for sx in (1.0, -1.0):
        for sz in (1.0, -1.0):
            vertices.append((sx * b, sz * c, 0.0))
    for sx in (1.0, -1.0):
        for sz in (1.0, -1.0):
            vertices.append((sx * c, 0.0, sz * b))
    result = np.asarray(vertices, dtype=np.float64)
    result /= np.linalg.norm(result, axis=1, keepdims=True) + 1e-12
    return result


def map_uv_to_depth(
    u: int,
    v: int,
    width: int,
    height: int,
    depth: np.ndarray,
) -> Tuple[int, int, int, int]:
    depth_height, depth_width = int(depth.shape[0]), int(depth.shape[1])
    if depth_width == width and depth_height == height:
        return int(u), int(v), width, height
    depth_u = int(
        round(np.clip(u * depth_width / max(width, 1), 0, depth_width - 1))
    )
    depth_v = int(
        round(np.clip(v * depth_height / max(height, 1), 0, depth_height - 1))
    )
    return depth_u, depth_v, depth_width, depth_height


def depth_backproject_uv(
    depth: np.ndarray,
    u: int,
    v: int,
    camera_pos: np.ndarray,
    camera_quat: np.ndarray,
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
) -> Optional[np.ndarray]:
    depth_u, depth_v, width_use, height_use = map_uv_to_depth(
        u,
        v,
        width,
        height,
        depth,
    )
    if (
        depth_v < 0
        or depth_v >= depth.shape[0]
        or depth_u < 0
        or depth_u >= depth.shape[1]
    ):
        return None
    value = float(depth[int(depth_v), int(depth_u)])
    if not (np.isfinite(value) and 0.05 < value < 50.0):
        return None
    focal_px = focal_length / horizontal_aperture * width_use
    center_x, center_y = width_use / 2.0, height_use / 2.0
    camera_point = np.array(
        [
            (depth_u - center_x) / focal_px * value,
            -(depth_v - center_y) / focal_px * value,
            -value,
        ],
        dtype=np.float64,
    )
    return (
        np.asarray(camera_pos, dtype=np.float64).reshape(3)
        + quat_to_mat_xyzw(camera_quat) @ camera_point
    )


def _normal_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(3)
    b = np.asarray(right, dtype=np.float64).reshape(3)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return float("inf")
    return float(
        math.degrees(
            math.acos(float(np.clip((a / a_norm) @ (b / b_norm), -1.0, 1.0)))
        )
    )


def _fit_outward_normal_scale(
    depth: np.ndarray,
    u: int,
    v: int,
    camera_pos: np.ndarray,
    camera_quat: np.ndarray,
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
    *,
    half_window_px: int,
    center_point: np.ndarray,
    center_depth_m: float,
    depth_jump_m: float,
    radius_m: float,
    min_points: int,
) -> Dict[str, Any]:
    samples = []
    seen_depth_pixels = set()
    half = max(1, int(half_window_px))
    for vv in range(max(0, int(v) - half), min(int(height), int(v) + half + 1)):
        for uu in range(max(0, int(u) - half), min(int(width), int(u) + half + 1)):
            depth_u, depth_v, _, _ = map_uv_to_depth(
                uu,
                vv,
                int(width),
                int(height),
                depth,
            )
            key = (int(depth_u), int(depth_v))
            if key in seen_depth_pixels:
                continue
            seen_depth_pixels.add(key)
            value = float(depth[int(depth_v), int(depth_u)])
            if not (math.isfinite(value) and 0.05 < value < 50.0):
                continue
            if abs(value - float(center_depth_m)) > float(depth_jump_m):
                continue
            point = depth_backproject_uv(
                depth,
                uu,
                vv,
                camera_pos,
                camera_quat,
                width,
                height,
                focal_length,
                horizontal_aperture,
            )
            if point is None:
                continue
            point = np.asarray(point, dtype=np.float64).reshape(3)
            if float(np.linalg.norm(point - center_point)) > float(radius_m):
                continue
            samples.append((point, float(uu - u), float(vv - v)))
    result: Dict[str, Any] = {
        "half_window_px": half,
        "support_points": int(len(samples)),
        "ok": False,
    }
    if len(samples) < int(min_points):
        result["reason"] = "too_few_points"
        return result

    points = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    delta_uv = np.asarray(
        [[sample[1], sample[2]] for sample in samples],
        dtype=np.float64,
    )
    sigma_px = max(1.0, float(half) * 0.65)
    weights = np.exp(
        -0.5 * np.sum(delta_uv * delta_uv, axis=1) / (sigma_px * sigma_px)
    )

    def weighted_plane_fit(
        values: np.ndarray,
        sample_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        weight_sum = float(sample_weights.sum())
        centroid = (
            np.sum(values * sample_weights[:, None], axis=0)
            / max(weight_sum, 1e-12)
        )
        centered = values - centroid
        covariance = (
            centered.T @ (centered * sample_weights[:, None])
        ) / max(weight_sum, 1e-12)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        return centroid, eigenvalues, eigenvectors

    centroid, eigenvalues, eigenvectors = weighted_plane_fit(points, weights)
    normal = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    residuals = np.abs((points - centroid) @ normal)
    residual_median = float(np.median(residuals))
    residual_mad = float(np.median(np.abs(residuals - residual_median)))
    residual_limit_m = max(
        0.0015,
        residual_median + 3.0 * 1.4826 * residual_mad,
    )
    keep = residuals <= residual_limit_m
    if int(keep.sum()) >= int(min_points) and int(keep.sum()) < len(points):
        points = points[keep]
        weights = weights[keep]
        centroid, eigenvalues, eigenvectors = weighted_plane_fit(points, weights)
        normal = np.asarray(eigenvectors[:, 0], dtype=np.float64)

    normal /= float(np.linalg.norm(normal)) + 1e-12
    view = np.asarray(camera_pos, dtype=np.float64).reshape(3) - center_point
    view_norm = float(np.linalg.norm(view))
    if float(normal @ view) < 0.0:
        normal = -normal
    signed_residual = (points - centroid) @ normal
    rms_m = float(np.sqrt(np.average(signed_residual ** 2, weights=weights)))
    eigen_sum = float(np.sum(eigenvalues))
    planarity = (
        float(eigenvalues[0]) / eigen_sum
        if eigen_sum > 1e-15
        else float("inf")
    )
    view_alignment = (
        float(normal @ (view / view_norm))
        if view_norm > 1e-12
        else 0.0
    )
    result.update(
        {
            "ok": True,
            "reason": "fit",
            "support_points": int(len(points)),
            "normal_world": normal.tolist(),
            "centroid_world": centroid.tolist(),
            "eigenvalues": [float(value) for value in eigenvalues.tolist()],
            "planarity": float(planarity),
            "rms_m": float(rms_m),
            "view_alignment": float(view_alignment),
            "residual_limit_m": float(residual_limit_m),
        }
    )
    return result


def estimate_outward_normal_from_depth(
    depth: np.ndarray,
    u: int,
    v: int,
    camera_pos: np.ndarray,
    camera_quat: np.ndarray,
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
    *,
    half_windows_px: Tuple[int, ...] = (4, 6, 8),
    depth_jump_m: float = 0.02,
    radius_m: float = 0.04,
    min_points: int = 20,
    max_planarity: float = 0.08,
    max_rms_m: float = 0.003,
    max_scale_angle_deg: float = 12.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    depth_array = np.asarray(depth, dtype=np.float64)
    if depth_array.ndim == 3:
        depth_array = depth_array[..., 0]
    audit: Dict[str, Any] = {
        "source": "organized_depth_pointcloud_multiscale_pca",
        "uses_segmentation": False,
        "uses_gt_mesh": False,
        "pixel": [int(u), int(v)],
        "half_windows_px": [int(value) for value in half_windows_px],
        "depth_jump_m": float(depth_jump_m),
        "radius_m": float(radius_m),
        "min_points": int(min_points),
        "max_planarity": float(max_planarity),
        "max_rms_m": float(max_rms_m),
        "max_allowed_scale_angle_deg": float(max_scale_angle_deg),
    }
    if depth_array.ndim != 2 or depth_array.size == 0:
        audit["reason"] = "invalid_depth_shape"
        return None, None, audit
    if not (0 <= int(u) < int(width) and 0 <= int(v) < int(height)):
        audit["reason"] = "pixel_out_of_bounds"
        return None, None, audit

    center_point = depth_backproject_uv(
        depth_array,
        int(u),
        int(v),
        camera_pos,
        camera_quat,
        width,
        height,
        focal_length,
        horizontal_aperture,
    )
    if center_point is None:
        audit["reason"] = "center_depth_invalid"
        return None, None, audit
    center_point = np.asarray(center_point, dtype=np.float64).reshape(3)
    depth_u, depth_v, _, _ = map_uv_to_depth(
        int(u),
        int(v),
        int(width),
        int(height),
        depth_array,
    )
    center_depth_m = float(depth_array[int(depth_v), int(depth_u)])
    scales = [
        _fit_outward_normal_scale(
            depth_array,
            int(u),
            int(v),
            np.asarray(camera_pos, dtype=np.float64),
            np.asarray(camera_quat, dtype=np.float64),
            int(width),
            int(height),
            float(focal_length),
            float(horizontal_aperture),
            half_window_px=int(half),
            center_point=center_point,
            center_depth_m=center_depth_m,
            depth_jump_m=float(depth_jump_m),
            radius_m=float(radius_m),
            min_points=int(min_points),
        )
        for half in half_windows_px
    ]
    passing = [
        scale
        for scale in scales
        if scale.get("ok")
        and float(scale.get("planarity", float("inf"))) <= float(max_planarity)
        and float(scale.get("rms_m", float("inf"))) <= float(max_rms_m)
    ]
    audit["point_world"] = center_point.tolist()
    audit["center_depth_m"] = center_depth_m
    audit["scales"] = scales
    if len(passing) < 2:
        audit["reason"] = "insufficient_quality_scales"
        audit["passing_scales"] = int(len(passing))
        return center_point, None, audit

    maximum_angle = 0.0
    for left_index in range(len(passing)):
        for right_index in range(left_index + 1, len(passing)):
            maximum_angle = max(
                maximum_angle,
                _normal_angle_deg(
                    passing[left_index]["normal_world"],
                    passing[right_index]["normal_world"],
                ),
            )
    audit["max_scale_angle_deg"] = float(maximum_angle)
    if maximum_angle > float(max_scale_angle_deg):
        audit["reason"] = "normal_unstable_across_scales"
        return center_point, None, audit

    selected = min(
        passing,
        key=lambda scale: (
            float(scale["planarity"]),
            float(scale["rms_m"]),
            -int(scale["support_points"]),
        ),
    )
    normal = np.asarray(selected["normal_world"], dtype=np.float64).reshape(3)
    normal /= float(np.linalg.norm(normal)) + 1e-12
    audit.update(
        {
            "ok": True,
            "reason": "ok",
            "normal_world": normal.tolist(),
            "selected_half_window_px": int(selected["half_window_px"]),
            "selected_support_points": int(selected["support_points"]),
            "selected_planarity": float(selected["planarity"]),
            "selected_rms_m": float(selected["rms_m"]),
            "view_alignment": float(selected["view_alignment"]),
        }
    )
    return center_point, normal, audit


def _quat_normalize(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _quat_mul_xyzw(
    left: Sequence[float],
    right: Sequence[float],
) -> np.ndarray:
    x1, y1, z1, w1 = [float(value) for value in _quat_normalize(left)]
    x2, y2, z2, w2 = [float(value) for value in _quat_normalize(right)]
    return _quat_normalize(
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float64,
        )
    )


def _horizontal_unit(vector: Sequence[float]) -> Optional[np.ndarray]:
    value = np.asarray(vector, dtype=np.float64).reshape(3).copy()
    value[2] = 0.0
    norm = float(np.linalg.norm(value))
    return None if norm < 1e-9 else value / norm


def wrist_camera_normal_eef() -> np.ndarray:
    camera_eef = (
        _RY_PI_MAT @ _RIGHT_REALSENSE_ORIGIN_GRIPPER
        + _PALM_T_EEF
    )
    normal = (
        camera_eef
        - _APPROACH_AXIS_EEF
        * float(np.dot(camera_eef, _APPROACH_AXIS_EEF))
    )
    if float(np.linalg.norm(normal)) < 1e-9:
        normal = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    return normal / float(np.linalg.norm(normal))


def ensure_camera_face_forward(
    quat: Sequence[float],
    *,
    forward: Sequence[float] = (1.0, 0.0, 0.0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    original = _quat_normalize(quat)
    forward_xy = _horizontal_unit(forward)
    normal_xy = _horizontal_unit(
        quat_to_mat_xyzw(original) @ wrist_camera_normal_eef()
    )
    if forward_xy is None or normal_xy is None:
        return original, {
            "build": "gripper_camera_face_v1_forward_dot_roll180",
            "skipped": True,
            "flipped": False,
            "reason": "missing_forward_or_camera_normal",
        }
    dot_before = float(np.dot(normal_xy, forward_xy))
    flipped = dot_before < 0.0
    roll_pi = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    selected = _quat_mul_xyzw(original, roll_pi) if flipped else original
    selected_normal = _horizontal_unit(
        quat_to_mat_xyzw(selected) @ wrist_camera_normal_eef()
    )
    dot_after = (
        float(np.dot(selected_normal, forward_xy))
        if selected_normal is not None
        else float("nan")
    )
    return selected, {
        "build": "gripper_camera_face_v1_forward_dot_roll180",
        "skipped": False,
        "flipped": bool(flipped),
        "dot_before": dot_before,
        "dot_after": dot_after,
        "angle_before_deg": float(
            math.degrees(math.acos(float(np.clip(dot_before, -1.0, 1.0))))
        ),
        "angle_after_deg": (
            float(
                math.degrees(
                    math.acos(float(np.clip(dot_after, -1.0, 1.0)))
                )
            )
            if math.isfinite(dot_after)
            else None
        ),
        "robot_forward_xy": forward_xy.tolist(),
        "camera_normal_xy_before": normal_xy.tolist(),
        "camera_normal_xy_after": (
            selected_normal.tolist()
            if selected_normal is not None
            else None
        ),
    }


def world_to_pixel(
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    point_world: Sequence[float],
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
) -> Optional[Tuple[int, int]]:
    focal_px = focal_length / horizontal_aperture * width
    point_camera = quat_to_mat_xyzw(camera_quat_xyzw).T @ (
        np.asarray(point_world, dtype=np.float64).reshape(3)
        - np.asarray(camera_pos, dtype=np.float64).reshape(3)
    )
    if point_camera[2] >= -1e-6:
        return None
    u = focal_px * point_camera[0] / (-point_camera[2]) + width / 2.0
    v = height / 2.0 - focal_px * point_camera[1] / (-point_camera[2])
    return int(round(u)), int(round(v))
