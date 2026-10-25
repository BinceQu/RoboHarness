"""Red depth-occluded preview of the selected official X5 gripper pose."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools.util.storage import _encode_png_stdlib
from tools.util.x5_kinematics import quaternion_to_matrix
from tools.util.x5_grasp_geometry import (
    ROBODOJO_X5_GRIPPER_SCALE_M,
    VOXEL_M,
    points_inside_x5_grasp_volume,
    x5_dynamic_opening_bounds,
    x5_dynamic_opening_mesh,
    x5_dynamic_opening_voxels,
    x5_dynamic_visual_vertices,
    x5_gripper_command_from_joint_position_m,
    x5_gripper_command_fraction,
    x5_gripper_geometry,
    x5_gripper_joint_position_m,
    x5_gripper_open_fraction,
)
from tools.util.x5_grasp_planner import DepthRayOccupancy
from tools.util.rgbd_scene_mesh_robodojo_v1 import (
    MeshSceneOccupancy,
    RGBDSceneMeshResult,
    reconstruct_rgbd_scene_mesh,
)
from tools.util.x5_exact_opening import (
    OPENING_QUERY_VERSION,
    exact_x5_planning_geometry,
)


DEPTH_OCCLUDE_EPS_M = 0.003
PLAN_GRIPPER_ALPHA = 0.78
PLAN_GRIPPER_BASE_RGB = np.array([150.0, 18.0, 18.0], dtype=np.float32)
WRIST_GRASP_ALPHA = 0.58
WRIST_GRASP_RGB = np.array([220.0, 28.0, 36.0], dtype=np.float32)
WRIST_SELF_DEPTH_TOLERANCE_M = 0.008
WRIST_SELF_UV_RADIUS_PX = 1
WRIST_OVERLAY_MAX_GRIPPER_QVEL_M_S = 0.005
WRIST_REGISTRATION_COARSE_STEP_M = 0.004
WRIST_REGISTRATION_REFINE_STEP_M = 0.001
WRIST_REGISTRATION_REFINE_RADIUS_M = 0.004
WRIST_REGISTRATION_MIN_COVERAGE = 0.20
WRIST_REGISTRATION_MIN_MATCHED_PIXELS = 512
WRIST_REGISTRATION_SURFACE_SAMPLE_M = 0.002
PLAN_THREE_VIEW_SCENE_MARGIN_M = 0.055
PLAN_THREE_VIEW_SCENE_STRIDE_PX = 2
PLAN_DEPTH_SHELL_BACK_EXTRUSION_M = 0.060


def _project_world(
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    camera_from_world_rotation = extrinsic[:3, :3].T
    points_usd = (points - extrinsic[:3, 3]) @ camera_from_world_rotation.T
    optical_depth = -points_usd[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(optical_depth) & (optical_depth > 1e-5)
    uv[valid, 0] = (
        float(intrinsic[0, 2])
        + float(intrinsic[0, 0]) * points_usd[valid, 0] / optical_depth[valid]
    )
    uv[valid, 1] = (
        float(intrinsic[1, 2])
        - float(intrinsic[1, 1]) * points_usd[valid, 1] / optical_depth[valid]
    )
    return uv, optical_depth


def _face_shading(
    triangles_world: np.ndarray,
    camera_position: np.ndarray,
) -> np.ndarray:
    edge_one = triangles_world[:, 1] - triangles_world[:, 0]
    edge_two = triangles_world[:, 2] - triangles_world[:, 0]
    normal = np.cross(edge_one, edge_two)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    view = camera_position.reshape(1, 3) - triangles_world.mean(axis=1)
    view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
    lambert = np.abs(np.einsum("ij,ij->i", normal, view))
    return 0.4 + 0.6 * np.clip(lambert, 0.0, 1.0)


def _rasterize_triangle(
    triangle_uv: np.ndarray,
    triangle_depth: np.ndarray,
    color: np.ndarray,
    color_buffer: np.ndarray,
    depth_buffer: np.ndarray,
) -> None:
    height, width = depth_buffer.shape
    min_u = max(0, int(np.floor(np.min(triangle_uv[:, 0]))))
    max_u = min(width - 1, int(np.ceil(np.max(triangle_uv[:, 0]))))
    min_v = max(0, int(np.floor(np.min(triangle_uv[:, 1]))))
    max_v = min(height - 1, int(np.ceil(np.max(triangle_uv[:, 1]))))
    if min_u > max_u or min_v > max_v:
        return

    u0, v0 = triangle_uv[0]
    u1, v1 = triangle_uv[1]
    u2, v2 = triangle_uv[2]
    denominator = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
    if abs(float(denominator)) < 1e-10:
        return
    pixel_u, pixel_v = np.meshgrid(
        np.arange(min_u, max_u + 1, dtype=np.float64) + 0.5,
        np.arange(min_v, max_v + 1, dtype=np.float64) + 0.5,
    )
    weight_zero = (
        (v1 - v2) * (pixel_u - u2) + (u2 - u1) * (pixel_v - v2)
    ) / denominator
    weight_one = ((v2 - v0) * (pixel_u - u2) + (u0 - u2) * (pixel_v - v2)) / denominator
    weight_two = 1.0 - weight_zero - weight_one
    inside = (
        (weight_zero >= -1e-6)
        & (weight_one >= -1e-6)
        & (weight_two >= -1e-6)
        & (weight_zero <= 1.0 + 1e-6)
        & (weight_one <= 1.0 + 1e-6)
        & (weight_two <= 1.0 + 1e-6)
    )
    if not np.any(inside):
        return
    local_v, local_u = np.nonzero(inside)
    inverse_depth = (
        weight_zero[inside] / triangle_depth[0]
        + weight_one[inside] / triangle_depth[1]
        + weight_two[inside] / triangle_depth[2]
    )
    valid_depth = inverse_depth > 1e-12
    if not np.any(valid_depth):
        return
    pixel_depth = (1.0 / inverse_depth[valid_depth]).astype(np.float32)
    rows = local_v + min_v
    columns = local_u + min_u
    rows = rows[valid_depth]
    columns = columns[valid_depth]
    closer = pixel_depth < depth_buffer[rows, columns]
    if not np.any(closer):
        return
    rows = rows[closer]
    columns = columns[closer]
    depth_buffer[rows, columns] = pixel_depth[closer]
    color_buffer[rows, columns] = color


def render_x5_gripper_overlay(
    *,
    rgb: np.ndarray,
    depth_linear: np.ndarray | None,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    eef_pose: dict[str, Any],
    output_path: str | Path,
    alpha: float = PLAN_GRIPPER_ALPHA,
) -> dict[str, Any]:
    """Rasterize the frozen open X5 mesh with a local z-buffer."""
    target = Path(output_path)
    try:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"RGB image must be HxWx3, got {image.shape}")
        image = image[:, :, :3].copy()
        height, width = image.shape[:2]
        intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
        extrinsic = np.asarray(extrinsic_matrix, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(extrinsic)):
            raise ValueError("camera matrices must be finite")
        position = np.asarray(eef_pose["pos"], dtype=np.float64).reshape(3)
        quaternion = np.asarray(eef_pose["quat"], dtype=np.float64).reshape(4)
        geometry = x5_gripper_geometry()
        vertices_eef = geometry["visual_vertices"]
        faces = geometry["visual_faces"]
        rotation = quaternion_to_matrix(quaternion)
        vertices_world = vertices_eef @ rotation.T + position.reshape(1, 3)
        uv, vertex_depth = _project_world(vertices_world, intrinsic, extrinsic)
        triangle_uv = uv[faces]
        triangle_depth = vertex_depth[faces]
        triangle_world = vertices_world[faces]
        valid = (
            np.all(np.isfinite(triangle_uv), axis=(1, 2))
            & np.all(np.isfinite(triangle_depth), axis=1)
            & np.all(triangle_depth > 1e-5, axis=1)
        )
        near_image = np.any(
            (
                (triangle_uv[:, :, 0] > -float(width))
                & (triangle_uv[:, :, 0] < 2.0 * float(width))
                & (triangle_uv[:, :, 1] > -float(height))
                & (triangle_uv[:, :, 1] < 2.0 * float(height))
            ),
            axis=1,
        )
        keep = valid & near_image
        if not np.any(keep):
            raise ValueError("selected X5 gripper mesh is outside the camera")
        triangle_uv = triangle_uv[keep]
        triangle_depth = triangle_depth[keep]
        triangle_world = triangle_world[keep]
        selected_components = geometry["visual_face_component"][keep]
        shade = _face_shading(triangle_world, extrinsic[:3, 3])

        color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
        for index in np.argsort(-triangle_depth.mean(axis=1)):
            color = np.floor(PLAN_GRIPPER_BASE_RGB * float(shade[index])).astype(
                np.uint8
            )
            _rasterize_triangle(
                triangle_uv[index],
                triangle_depth[index],
                color,
                color_buffer,
                depth_buffer,
            )

        rasterized = np.isfinite(depth_buffer)
        visible = rasterized.copy()
        scene_depth_available = False
        if depth_linear is not None:
            scene = np.asarray(depth_linear, dtype=np.float32).squeeze()
            if scene.shape == (height, width):
                scene_depth_available = True
                valid_scene = np.isfinite(scene) & (scene > 0.02)
                visible &= ~(
                    valid_scene & (scene < depth_buffer - float(DEPTH_OCCLUDE_EPS_M))
                )
        blend = float(np.clip(alpha, 0.0, 1.0))
        output = image.astype(np.float32)
        output[visible] = (
            output[visible] * (1.0 - blend)
            + color_buffer[visible].astype(np.float32) * blend
        )
        output = np.rint(np.clip(output, 0.0, 255.0)).astype(np.uint8)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_encode_png_stdlib(output))
        return {
            "ok": True,
            "path": str(target),
            "build": "official_robodojo_x5_open_mesh_zbuffer_v1",
            "color": "red",
            "geometry_source": "policy_local_frozen_official_x5_open_visual_mesh",
            "input_vertex_count": int(len(vertices_eef)),
            "input_triangle_count": int(len(faces)),
            "projected_triangle_count": int(len(triangle_uv)),
            "projected_component_count": int(len(np.unique(selected_components))),
            "component_names": list(geometry["component_names"]),
            "face_shading": "abs_view_lambert_0.4_plus_0.6",
            "shade_min": float(shade.min()),
            "shade_max": float(shade.max()),
            "base_color_rgb": PLAN_GRIPPER_BASE_RGB.astype(int).tolist(),
            "alpha": blend,
            "rasterized_pixel_count": int(np.count_nonzero(rasterized)),
            "visible_pixel_count": int(np.count_nonzero(visible)),
            "depth_occlusion": scene_depth_available,
            "depth_occlude_eps_m": float(DEPTH_OCCLUDE_EPS_M),
            "segmentation_used": False,
            "simulator_geometry_used": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "path": str(target),
        }


def _nearby_depth_match_mask(
    scene_depth: np.ndarray,
    model_depth: np.ndarray,
    *,
    radius_px: int,
    tolerance_m: float,
) -> np.ndarray:
    scene = np.asarray(scene_depth, dtype=np.float32)
    model = np.asarray(model_depth, dtype=np.float32)
    if scene.shape != model.shape:
        raise ValueError("scene and model depth shapes do not match")
    height, width = scene.shape
    best_delta = np.full(scene.shape, np.inf, dtype=np.float32)
    radius = max(0, int(radius_px))
    for delta_v in range(-radius, radius + 1):
        target_v0 = max(0, -delta_v)
        target_v1 = min(height, height - delta_v)
        source_v0 = target_v0 + delta_v
        source_v1 = target_v1 + delta_v
        for delta_u in range(-radius, radius + 1):
            target_u0 = max(0, -delta_u)
            target_u1 = min(width, width - delta_u)
            source_u0 = target_u0 + delta_u
            source_u1 = target_u1 + delta_u
            target = np.s_[target_v0:target_v1, target_u0:target_u1]
            source = np.s_[source_v0:source_v1, source_u0:source_u1]
            candidate = np.abs(scene[target] - model[source])
            np.minimum(best_delta[target], candidate, out=best_delta[target])
    valid_scene = np.isfinite(scene) & (scene > 0.02)
    return valid_scene & (best_delta <= float(tolerance_m))


def _x5_finger_self_depth_mask(
    *,
    scene_depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    eef_pose: dict[str, Any],
    gripper_qpos: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    scene = np.asarray(scene_depth, dtype=np.float32)
    height, width = scene.shape
    geometry = x5_gripper_geometry()
    vertices_eef = x5_dynamic_visual_vertices(gripper_qpos)
    faces = np.asarray(geometry["visual_faces"], dtype=np.int64)
    components = np.asarray(geometry["visual_face_component"], dtype=np.int64)
    finger_faces = faces[np.isin(components, [1, 2])]
    rotation = quaternion_to_matrix(eef_pose["quat"])
    position = np.asarray(eef_pose["pos"], dtype=np.float64).reshape(3)
    vertices_world = vertices_eef @ rotation.T + position.reshape(1, 3)
    uv, vertex_depth = _project_world(
        vertices_world,
        np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3),
        np.asarray(extrinsic_matrix, dtype=np.float64).reshape(4, 4),
    )
    triangle_uv = uv[finger_faces]
    triangle_depth = vertex_depth[finger_faces]
    valid = (
        np.all(np.isfinite(triangle_uv), axis=(1, 2))
        & np.all(np.isfinite(triangle_depth), axis=1)
        & np.all(triangle_depth > 1e-5, axis=1)
    )
    near_image = np.any(
        (
            (triangle_uv[:, :, 0] > -float(width))
            & (triangle_uv[:, :, 0] < 2.0 * float(width))
            & (triangle_uv[:, :, 1] > -float(height))
            & (triangle_uv[:, :, 1] < 2.0 * float(height))
        ),
        axis=1,
    )
    keep = valid & near_image
    if not np.any(keep):
        return np.zeros(scene.shape, dtype=bool), {
            "applied": False,
            "reason": "finger mesh is outside the wrist camera",
            "projected_finger_pixel_count": 0,
        }

    triangle_uv = triangle_uv[keep]
    triangle_depth = triangle_depth[keep]
    model_depth = np.full(scene.shape, np.inf, dtype=np.float32)
    color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    for index in np.argsort(-triangle_depth.mean(axis=1)):
        _rasterize_triangle(
            triangle_uv[index],
            triangle_depth[index],
            np.zeros(3, dtype=np.uint8),
            color_buffer,
            model_depth,
        )
    self_mask = _nearby_depth_match_mask(
        scene,
        model_depth,
        radius_px=WRIST_SELF_UV_RADIUS_PX,
        tolerance_m=WRIST_SELF_DEPTH_TOLERANCE_M,
    )
    return self_mask, {
        "applied": True,
        "source": "policy_local_frozen_dynamic_official_x5_finger_visual_mesh",
        "projected_finger_pixel_count": int(np.isfinite(model_depth).sum()),
        "matched_scene_pixel_count": int(np.count_nonzero(self_mask)),
        "uv_radius_px": int(WRIST_SELF_UV_RADIUS_PX),
        "depth_tolerance_m": float(WRIST_SELF_DEPTH_TOLERANCE_M),
        "component_ids": [1, 2],
    }


@lru_cache(maxsize=1)
def _x5_finger_registration_samples() -> tuple[np.ndarray, np.ndarray]:
    """Return dense official finger surface samples and their closure signs."""
    geometry = x5_gripper_geometry()
    vertices = np.asarray(geometry["visual_vertices"], dtype=np.float64)
    faces = np.asarray(geometry["visual_faces"], dtype=np.int64)
    components = np.asarray(geometry["visual_face_component"], dtype=np.int64)
    samples: list[np.ndarray] = []
    closure_signs: list[np.ndarray] = []
    for component_index, closure_sign in ((1, -1.0), (2, 1.0)):
        component_faces = faces[components == component_index]
        component_sample_groups: list[np.ndarray] = []
        for triangle in vertices[component_faces]:
            edge_lengths = (
                np.linalg.norm(triangle[1] - triangle[0]),
                np.linalg.norm(triangle[2] - triangle[1]),
                np.linalg.norm(triangle[0] - triangle[2]),
            )
            divisions = max(
                1,
                int(np.ceil(max(edge_lengths) / WRIST_REGISTRATION_SURFACE_SAMPLE_M)),
            )
            rows = []
            for first_index in range(divisions + 1):
                second_index = np.arange(
                    divisions - first_index + 1,
                    dtype=np.float64,
                )
                first_weight = first_index / divisions
                second_weight = second_index / divisions
                zero_weight = 1.0 - first_weight - second_weight
                rows.append(
                    zero_weight[:, None] * triangle[0]
                    + first_weight * triangle[1]
                    + second_weight[:, None] * triangle[2]
                )
            component_sample_groups.append(np.concatenate(rows, axis=0))
        component_samples = np.concatenate(component_sample_groups, axis=0)
        samples.append(component_samples)
        closure_signs.append(
            np.full(len(component_samples), closure_sign, dtype=np.float64)
        )
    points = np.concatenate(samples, axis=0)
    signs = np.concatenate(closure_signs, axis=0)
    points.setflags(write=False)
    signs.setflags(write=False)
    return points, signs


def _x5_finger_sparse_registration_score(
    *,
    scene_depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    eef_pose: dict[str, Any],
    joint_position_m: float,
) -> dict[str, Any]:
    """Score one opening using projected official mesh samples, without rasterizing."""
    scene = np.asarray(scene_depth, dtype=np.float32)
    height, width = scene.shape
    geometry = x5_gripper_geometry()
    samples_open, closure_signs = _x5_finger_registration_samples()
    closure = float(geometry["metadata"]["gripper_open_m"]) - float(joint_position_m)
    samples_eef = samples_open.copy()
    samples_eef[:, 1] += closure_signs * closure
    rotation = quaternion_to_matrix(eef_pose["quat"])
    position = np.asarray(eef_pose["pos"], dtype=np.float64).reshape(3)
    samples_world = samples_eef @ rotation.T + position.reshape(1, 3)
    uv, sample_depth = _project_world(
        samples_world,
        np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3),
        np.asarray(extrinsic_matrix, dtype=np.float64).reshape(4, 4),
    )
    valid = (
        np.all(np.isfinite(uv), axis=1)
        & np.isfinite(sample_depth)
        & (sample_depth > 1e-5)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    if not np.any(valid):
        return {
            "joint_position_m": float(joint_position_m),
            "matched_model_pixel_count": 0,
            "projected_model_pixel_count": 0,
            "coverage": 0.0,
        }

    columns = np.rint(uv[valid, 0]).astype(np.int64)
    rows = np.rint(uv[valid, 1]).astype(np.int64)
    columns = np.clip(columns, 0, width - 1)
    rows = np.clip(rows, 0, height - 1)
    flat_pixels = rows * width + columns
    unique_pixels, inverse = np.unique(flat_pixels, return_inverse=True)
    nearest_model_depth = np.full(len(unique_pixels), np.inf, dtype=np.float64)
    np.minimum.at(nearest_model_depth, inverse, sample_depth[valid])
    model_rows = unique_pixels // width
    model_columns = unique_pixels % width
    best_delta = np.full(len(unique_pixels), np.inf, dtype=np.float64)
    radius = max(0, int(WRIST_SELF_UV_RADIUS_PX))
    for delta_v in range(-radius, radius + 1):
        sample_rows = model_rows + delta_v
        valid_rows = (sample_rows >= 0) & (sample_rows < height)
        for delta_u in range(-radius, radius + 1):
            sample_columns = model_columns + delta_u
            inside = valid_rows & (sample_columns >= 0) & (sample_columns < width)
            if not np.any(inside):
                continue
            sampled_scene = scene[sample_rows[inside], sample_columns[inside]]
            valid_scene = np.isfinite(sampled_scene) & (sampled_scene > 0.02)
            if not np.any(valid_scene):
                continue
            selected = np.flatnonzero(inside)[valid_scene]
            delta = np.abs(
                sampled_scene[valid_scene].astype(np.float64)
                - nearest_model_depth[selected]
            )
            best_delta[selected] = np.minimum(best_delta[selected], delta)
    matched = int(np.count_nonzero(best_delta <= float(WRIST_SELF_DEPTH_TOLERANCE_M)))
    projected = int(len(unique_pixels))
    return {
        "joint_position_m": float(joint_position_m),
        "matched_model_pixel_count": matched,
        "projected_model_pixel_count": projected,
        "coverage": float(matched / max(1, projected)),
    }


def _estimate_x5_gripper_joint_position_from_depth(
    *,
    scene_depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    eef_pose: dict[str, Any],
    gripper_qpos: Any,
) -> tuple[list[float], np.ndarray, dict[str, Any], dict[str, Any]]:
    """Register the dynamic official finger mesh to the evaluator wrist depth."""
    _scale_low, scale_high = ROBODOJO_X5_GRIPPER_SCALE_M
    nominal_joint_m = x5_gripper_joint_position_m(gripper_qpos)
    evaluated: dict[float, tuple[list[float], dict[str, Any]]] = {}

    def evaluate(position_m: float):
        limited = float(np.clip(position_m, 0.0, scale_high))
        key = round(limited, 9)
        cached = evaluated.get(key)
        if cached is not None:
            return cached
        geometry_qpos = [x5_gripper_command_from_joint_position_m(limited)]
        score = _x5_finger_sparse_registration_score(
            scene_depth=scene_depth,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            eef_pose=eef_pose,
            joint_position_m=limited,
        )
        cached = (geometry_qpos, score)
        evaluated[key] = cached
        return cached

    coarse_positions = np.arange(
        0.0,
        scale_high + WRIST_REGISTRATION_COARSE_STEP_M * 0.25,
        WRIST_REGISTRATION_COARSE_STEP_M,
        dtype=np.float64,
    )
    for position_m in (*coarse_positions.tolist(), scale_high, nominal_joint_m):
        evaluate(float(position_m))

    def rank(item):
        score = item[1]
        return (
            float(score["coverage"]),
            int(score["matched_model_pixel_count"]),
        )

    best = max(evaluated.values(), key=rank)
    best_position_m = float(best[1]["joint_position_m"])
    refine_low = max(0.0, best_position_m - WRIST_REGISTRATION_REFINE_RADIUS_M)
    refine_high = min(
        scale_high,
        best_position_m + WRIST_REGISTRATION_REFINE_RADIUS_M,
    )
    refine_positions = np.arange(
        refine_low,
        refine_high + WRIST_REGISTRATION_REFINE_STEP_M * 0.25,
        WRIST_REGISTRATION_REFINE_STEP_M,
        dtype=np.float64,
    )
    for position_m in refine_positions:
        evaluate(float(position_m))
    best = max(evaluated.values(), key=rank)
    best_score = best[1]
    registration_safe = bool(
        int(best_score["matched_model_pixel_count"])
        >= WRIST_REGISTRATION_MIN_MATCHED_PIXELS
        and float(best_score["coverage"]) >= WRIST_REGISTRATION_MIN_COVERAGE
    )
    selected = best if registration_safe else evaluate(nominal_joint_m)
    selected_score = selected[1]
    self_mask, self_mask_meta = _x5_finger_self_depth_mask(
        scene_depth=scene_depth,
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=extrinsic_matrix,
        eef_pose=eef_pose,
        gripper_qpos=selected[0],
    )
    if registration_safe and not self_mask_meta.get("applied"):
        selected = evaluate(nominal_joint_m)
        selected_score = selected[1]
        self_mask, self_mask_meta = _x5_finger_self_depth_mask(
            scene_depth=scene_depth,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            eef_pose=eef_pose,
            gripper_qpos=selected[0],
        )
        registration_safe = False
    registration = {
        "applied": registration_safe,
        "source": "evaluator_wrist_depth_registered_official_x5_finger_mesh",
        "search_mode": "projected_official_mesh_two_mm_triangle_surface_samples",
        "candidate_count": int(len(evaluated)),
        "coarse_step_m": float(WRIST_REGISTRATION_COARSE_STEP_M),
        "refine_step_m": float(WRIST_REGISTRATION_REFINE_STEP_M),
        "nominal_command_joint_position_m": nominal_joint_m,
        "selected_joint_position_m": float(selected_score["joint_position_m"]),
        "estimated_command_to_physical_lag_m": float(
            nominal_joint_m - float(selected_score["joint_position_m"])
        ),
        "search_matched_model_pixel_count": int(
            selected_score["matched_model_pixel_count"]
        ),
        "search_projected_model_pixel_count": int(
            selected_score["projected_model_pixel_count"]
        ),
        "search_coverage": float(selected_score["coverage"]),
        "matched_scene_pixel_count": int(
            self_mask_meta.get("matched_scene_pixel_count", 0)
        ),
        "projected_finger_pixel_count": int(
            self_mask_meta.get("projected_finger_pixel_count", 0)
        ),
        "minimum_coverage": float(WRIST_REGISTRATION_MIN_COVERAGE),
        "minimum_matched_pixels": int(WRIST_REGISTRATION_MIN_MATCHED_PIXELS),
        "fallback": None if registration_safe else "normalized_command_mapping",
        "segmentation_used": False,
        "simulator_state_used": False,
    }
    self_mask_meta["depth_registration"] = registration
    return selected[0], self_mask, self_mask_meta, registration


def render_x5_wrist_grasp_volume_overlay(
    *,
    rgb: np.ndarray,
    depth_linear: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    eef_pose: dict[str, Any],
    gripper_qpos: Any,
    gripper_qvel: Any | None = None,
    output_path: str | Path,
    mask_path: str | Path | None = None,
    alpha: float = WRIST_GRASP_ALPHA,
) -> dict[str, Any]:
    """Highlight visible wrist-camera depth points inside the X5 opening."""
    target = Path(output_path)
    mask_target = None if mask_path is None else Path(mask_path)
    try:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"RGB image must be HxWx3, got {image.shape}")
        image = image[:, :, :3].copy()
        height, width = image.shape[:2]
        depth = np.asarray(depth_linear, dtype=np.float32).squeeze()
        if depth.shape != (height, width):
            raise ValueError(
                f"wrist RGB/depth mismatch: {(height, width)} vs {depth.shape}"
            )
        intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
        extrinsic = np.asarray(extrinsic_matrix, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(extrinsic)):
            raise ValueError("camera matrices must be finite")
        geometry = x5_gripper_geometry()
        position = np.asarray(eef_pose["pos"], dtype=np.float64).reshape(3)
        rotation = quaternion_to_matrix(eef_pose["quat"])
        command_fraction = x5_gripper_command_fraction(gripper_qpos)
        command_joint_position_m = x5_gripper_joint_position_m(gripper_qpos)
        command_open_fraction = x5_gripper_open_fraction(gripper_qpos)

        valid = np.isfinite(depth) & (depth > 0.02)
        pixel_v, pixel_u = np.nonzero(valid)
        distance = depth[pixel_v, pixel_u].astype(np.float64)
        optical = np.column_stack(
            (
                (pixel_u.astype(np.float64) - intrinsic[0, 2])
                / intrinsic[0, 0]
                * distance,
                (pixel_v.astype(np.float64) - intrinsic[1, 2])
                / intrinsic[1, 1]
                * distance,
                distance,
            )
        )
        points_usd = optical * np.array([1.0, -1.0, -1.0])
        points_world = points_usd @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        points_eef = (points_world - position.reshape(1, 3)) @ rotation

        qvel = np.asarray(
            [] if gripper_qvel is None else gripper_qvel,
            dtype=np.float64,
        ).reshape(-1)
        scale_low, scale_high = ROBODOJO_X5_GRIPPER_SCALE_M
        normalized_command_range_m = scale_high - scale_low
        max_abs_qvel_m_s = (
            None
            if qvel.size == 0 or not np.all(np.isfinite(qvel))
            else float(np.max(np.abs(qvel)) * normalized_command_range_m)
        )
        motion_safe = bool(
            max_abs_qvel_m_s is None
            or max_abs_qvel_m_s <= WRIST_OVERLAY_MAX_GRIPPER_QVEL_M_S
        )
        if motion_safe:
            (
                geometry_qpos,
                self_mask,
                self_mask_meta,
                depth_registration,
            ) = _estimate_x5_gripper_joint_position_from_depth(
                scene_depth=depth,
                intrinsic_matrix=intrinsic,
                extrinsic_matrix=extrinsic,
                eef_pose=eef_pose,
                gripper_qpos=gripper_qpos,
            )
        else:
            geometry_qpos = [command_fraction]
            self_mask = np.zeros((height, width), dtype=bool)
            self_mask_meta = {
                "applied": False,
                "reason": "gripper motion exceeds overlay qvel limit",
                "projected_finger_pixel_count": 0,
            }
            depth_registration = {
                "applied": False,
                "source": ("evaluator_wrist_depth_registered_official_x5_finger_mesh"),
                "fallback": "suppressed_during_gripper_motion",
                "segmentation_used": False,
                "simulator_state_used": False,
            }

        geometry_command_fraction = x5_gripper_command_fraction(geometry_qpos)
        joint_position_m = x5_gripper_joint_position_m(geometry_qpos)
        open_fraction = x5_gripper_open_fraction(geometry_qpos)
        bounds = x5_dynamic_opening_bounds(geometry_qpos)
        inside = points_inside_x5_grasp_volume(points_eef, geometry_qpos)
        candidate_mask = np.zeros((height, width), dtype=bool)
        candidate_mask[pixel_v[inside], pixel_u[inside]] = True
        red_mask = (
            candidate_mask & ~self_mask
            if motion_safe
            else np.zeros((height, width), dtype=bool)
        )

        blend = float(np.clip(alpha, 0.0, 1.0))
        output = image.astype(np.float32)
        output[red_mask] = output[red_mask] * (1.0 - blend) + WRIST_GRASP_RGB * blend
        output = np.rint(np.clip(output, 0.0, 255.0)).astype(np.uint8)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_encode_png_stdlib(output))
        if mask_target is not None:
            mask_target.parent.mkdir(parents=True, exist_ok=True)
            np.save(mask_target, red_mask)
        opening_voxels = x5_dynamic_opening_voxels(geometry_qpos)
        opening_mesh = x5_dynamic_opening_mesh(geometry_qpos)
        return {
            "ok": True,
            "path": str(target),
            "mask_path": None if mask_target is None else str(mask_target),
            "build": "official_robodojo_x5_wrist_exact_inner_triangles_v4",
            "opening_query_version": OPENING_QUERY_VERSION,
            "red_pixel_semantics": "visible_raw_depth_samples_not_material_volume",
            "color": "red",
            "definition": (
                "visible evaluator depth points inside the current official X5 "
                "two-finger opening, excluding depth-matched finger surfaces"
            ),
            "red_pixel_count": int(np.count_nonzero(red_mask)),
            "red_candidate_pixel_count": int(np.count_nonzero(candidate_mask)),
            "self_rejected_pixel_count": int(
                np.count_nonzero(candidate_mask & self_mask)
            ),
            "valid_depth_pixel_count": int(np.count_nonzero(valid)),
            "gripper_open_fraction": open_fraction,
            "gripper_joint_position_m": [joint_position_m],
            "gripper_geometry_qpos_normalized": [geometry_command_fraction],
            "gripper_command_joint_position_m": [command_joint_position_m],
            "gripper_command_open_fraction": command_open_fraction,
            "gripper_qpos_normalized": [command_fraction],
            "gripper_qvel_normalized_s": (
                None if qvel.size == 0 else qvel.astype(float).tolist()
            ),
            "max_abs_gripper_qvel_m_s": max_abs_qvel_m_s,
            "gripper_motion_safe": motion_safe,
            "overlay_suppressed": not motion_safe,
            "overlay_suppressed_reason": (
                None
                if motion_safe
                else "gripper image/proprio alignment is unsafe during motion"
            ),
            "opening_bounds_eef_m": bounds.tolist(),
            "opening_voxel_count": int(len(opening_voxels)),
            "grasp_zone_capacity_cm3": float(opening_mesh["volume_cm3"]),
            "opening_quadrature_capacity_cm3": float(
                len(opening_voxels) * (VOXEL_M**3) * 1e6
            ),
            "depth_source": "frozen_evaluator_depth_linear",
            "camera_pose_source": "evaluator_camera_extrinsic_matrix",
            "eef_pose_source": "capture_time_evaluator_proprioception",
            "opening_geometry_source": (
                "policy_local_frozen_official_x5_two_finger_inner_surface_corridor"
            ),
            "opening_geometry": geometry["metadata"].get("opening_geometry"),
            "opening_projection": geometry["metadata"].get("opening_projection"),
            "opening_column_count": int(len(geometry["opening_columns"])),
            "exact_opening_component_selection": geometry["metadata"].get(
                "exact_opening_component_selection"
            ),
            "exact_opening_mesh_volume_cm3": float(opening_mesh["volume_cm3"]),
            "exact_opening_mesh_exact": bool(opening_mesh["exact"]),
            "exact_opening_mesh_source": opening_mesh["source"],
            "exact_opening_inner_y_bounds_m": np.asarray(
                opening_mesh["inner_y_bounds_m"], dtype=np.float64
            ).tolist(),
            "exact_opening_y_boundary": opening_mesh.get("boundary_definition"),
            "exact_opening_metric_independent_of_voxel": True,
            "robot_mask_used": bool(self_mask_meta.get("applied")),
            "robot_mask_source": self_mask_meta.get("source"),
            "robot_self_mask": self_mask_meta,
            "gripper_depth_registration": depth_registration,
            "segmentation_used": False,
            "simulator_geometry_used": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "path": str(target),
            "mask_path": None if mask_target is None else str(mask_target),
        }


def _projected_opening_mesh_mask(
    projected_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    panel_width: int,
    panel_height: int,
    cv2: Any,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rasterize the exact projected opening triangle mesh.

    Every source triangle is filled independently at 4x supersampling and the
    union is reduced to panel resolution.  This preserves the actual tapered
    mesh boundary (including holes/notches) and avoids both voxel-sized
    rectangles and convex-hull completion.
    """
    return _projected_triangle_mesh_mask(
        projected_vertices,
        faces,
        panel_width=panel_width,
        panel_height=panel_height,
        cv2=cv2,
        supersample=4,
    )


def _opening_boundary_metadata(
    contours: list[np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    """Describe the exact mesh silhouette used in one rendered panel.

    The returned points are image-pixel contour coordinates from the union of
    projected source triangles.  Keeping them in the result makes the visible
    boundary independently auditable without exposing any simulator state.
    """
    normalized: list[list[list[int]]] = []
    all_points: list[np.ndarray] = []
    for contour in contours:
        points = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
        if len(points) < 2:
            continue
        # OpenCV can repeat the first point at the end.  Preserve the actual
        # raster contour, but remove only consecutive duplicates introduced by
        # antialiasing so the serialized boundary stays compact.
        if len(points) > 1:
            keep = np.ones(len(points), dtype=bool)
            keep[1:] = np.any(points[1:] != points[:-1], axis=1)
            points = points[keep]
        if len(points) < 2:
            continue
        normalized.append(points.tolist())
        all_points.append(points)
    if all_points:
        joined = np.concatenate(all_points, axis=0)
        bbox = [
            int(joined[:, 0].min()),
            int(joined[:, 1].min()),
            int(joined[:, 0].max()),
            int(joined[:, 1].max()),
        ]
    else:
        bbox = None
    return {
        "source": "projected_exact_opening_mesh_union_contours",
        "rasterization": "4x_supersampled_source_triangle_union",
        "contour_count": int(len(normalized)),
        "contour_vertex_count": int(sum(len(points) for points in normalized)),
        "bbox_px": bbox,
        "mask_pixel_count": int(np.count_nonzero(mask)),
        "contours_px": normalized,
    }


def _projected_triangle_mesh_mask(
    projected_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    panel_width: int,
    panel_height: int,
    cv2: Any,
    supersample: int = 2,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rasterize a projected triangle union without polygon/hull completion.

    This is used for both the red opening and the gray official gripper.  Each
    source STL face is filled independently at supersampled resolution, so
    concavities and genuine mesh holes survive the projection.  The operation
    is a display rasterization only; all metric membership and volume values
    continue to come from the planner's 3 mm voxel contract.
    """
    mask = np.zeros((panel_height, panel_width), dtype=np.uint8)
    vertices = np.asarray(projected_vertices, dtype=np.float64).reshape(-1, 2)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(triangles) == 0:
        return mask, []
    supersample = max(1, int(supersample))
    high = np.zeros(
        (panel_height * supersample, panel_width * supersample),
        dtype=np.uint8,
    )
    for face in triangles:
        points = vertices[face]
        if not np.all(np.isfinite(points)):
            continue
        pixels = np.rint(points * supersample).astype(np.int32)
        # Avoid pathological coordinates from an out-of-frame triangle while
        # retaining triangles that intersect the panel.
        if (
            np.max(pixels[:, 0]) < -supersample
            or np.min(pixels[:, 0]) >= panel_width * supersample + supersample
            or np.max(pixels[:, 1]) < -supersample
            or np.min(pixels[:, 1]) >= panel_height * supersample + supersample
        ):
            continue
        cv2.fillConvexPoly(high, pixels, 255, lineType=cv2.LINE_AA)
    mask = cv2.resize(
        high,
        (panel_width, panel_height),
        interpolation=cv2.INTER_AREA,
    )
    binary = np.where(mask >= 32, 255, 0).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    return mask, list(contours)


def _blend_opening_mask(
    panel: np.ndarray,
    mask: np.ndarray,
    *,
    cv2: Any,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Blend an exact projected mesh union onto a diagnostic panel."""
    if not np.any(mask):
        return
    layer = panel.copy()
    layer[mask > 0] = color
    clipped_alpha = float(np.clip(alpha, 0.0, 1.0))
    panel[:] = cv2.addWeighted(
        layer,
        clipped_alpha,
        panel,
        1.0 - clipped_alpha,
        0.0,
    )


def render_x5_grasp_volume_three_views(
    *,
    gripper_qpos: Any,
    output_path: str | Path,
    command_qpos: Any | None = None,
) -> dict[str, Any]:
    """Render dynamic X5 views from the official mesh-derived opening volume."""
    target = Path(output_path)
    try:
        import cv2

        geometry = x5_gripper_geometry()
        vertices = x5_dynamic_visual_vertices(gripper_qpos)
        faces = np.asarray(geometry["visual_faces"], dtype=np.int64)
        face_components = np.asarray(geometry["visual_face_component"], dtype=np.int64)
        bounds = x5_dynamic_opening_bounds(gripper_qpos)
        geometry_command_fraction = x5_gripper_command_fraction(gripper_qpos)
        observed_command_fraction = x5_gripper_command_fraction(
            gripper_qpos if command_qpos is None else command_qpos
        )
        joint_position_m = x5_gripper_joint_position_m(gripper_qpos)
        open_fraction = x5_gripper_open_fraction(gripper_qpos)
        opening_voxels = x5_dynamic_opening_voxels(gripper_qpos)
        opening_mesh = x5_dynamic_opening_mesh(gripper_qpos)
        opening_mesh_vertices = np.asarray(opening_mesh["vertices"], dtype=np.float64)
        opening_mesh_faces = np.asarray(opening_mesh["faces"], dtype=np.int64)

        canvas_height, canvas_width = 470, 1200
        canvas = np.full((canvas_height, canvas_width, 3), 246, dtype=np.uint8)
        panel_width, panel_height = 370, 350
        panel_y = 58
        view_specs = (
            ("TOP", "X", "Y", 0, 1, 2),
            ("SIDE", "X", "Z", 0, 2, 1),
            ("FRONT", "Y", "Z", 1, 2, 0),
        )
        view_metadata = []
        for view_index, (
            name,
            horizontal_name,
            vertical_name,
            horizontal,
            vertical,
            depth_axis,
        ) in enumerate(view_specs):
            panel_x = 20 + view_index * 390
            panel = canvas[
                panel_y : panel_y + panel_height,
                panel_x : panel_x + panel_width,
            ]
            panel[:] = 255
            groups = [vertices]
            if len(opening_mesh_vertices):
                groups.append(opening_mesh_vertices)
            if len(opening_voxels):
                groups.append(opening_voxels)
            all_points = np.concatenate(groups, axis=0)
            low = all_points[:, [horizontal, vertical]].min(axis=0)
            high = all_points[:, [horizontal, vertical]].max(axis=0)
            center = 0.5 * (low + high)
            span = np.maximum(high - low, 1e-6)
            scale = min(
                (panel_width - 56.0) / span[0],
                (panel_height - 72.0) / span[1],
            )

            def project(points: np.ndarray) -> np.ndarray:
                values = np.asarray(points, dtype=np.float64)
                u = (values[:, horizontal] - center[0]) * scale + panel_width / 2
                v = panel_height / 2 - (values[:, vertical] - center[1]) * scale
                return np.column_stack((u, v))

            projected_vertices = project(vertices)
            opening_mask, opening_contours = (
                _projected_opening_mesh_mask(
                    project(opening_mesh_vertices),
                    opening_mesh_faces,
                    panel_width=panel_width,
                    panel_height=panel_height,
                    cv2=cv2,
                )
                if len(opening_mesh_vertices)
                else (
                    np.zeros((panel_height, panel_width), dtype=np.uint8),
                    [],
                )
            )

            component_depths = []
            for component_index in np.unique(face_components):
                component_mask = face_components == component_index
                component_depths.append(
                    (
                        float(vertices[faces[component_mask], depth_axis].mean()),
                        int(component_index),
                    )
                )
            component_colors = (
                (73, 84, 96),
                (82, 95, 108),
                (88, 102, 116),
            )
            for _depth, component_index in sorted(component_depths):
                component_faces = faces[face_components == component_index]
                component_render_mask, contours = _projected_triangle_mesh_mask(
                    projected_vertices,
                    component_faces,
                    panel_width=panel_width,
                    panel_height=panel_height,
                    cv2=cv2,
                    supersample=2,
                )
                visible = component_render_mask > 0
                if np.any(visible):
                    # Keep the historical per-link palette while deriving
                    # every filled pixel from actual projected STL triangles.
                    color = np.asarray(
                        component_colors[component_index % len(component_colors)],
                        dtype=np.uint8,
                    )
                    panel[visible] = color
                    cv2.drawContours(
                        panel,
                        contours,
                        -1,
                        (48, 58, 68),
                        1,
                        cv2.LINE_AA,
                    )

            _blend_opening_mask(
                panel,
                opening_mask,
                cv2=cv2,
                color=(222, 36, 45),
                alpha=0.30,
            )
            if opening_contours:
                cv2.drawContours(
                    panel,
                    opening_contours,
                    -1,
                    (196, 24, 34),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            cv2.rectangle(
                panel,
                (0, 0),
                (panel_width - 1, panel_height - 1),
                (205, 211, 218),
                thickness=1,
            )
            cv2.putText(
                panel,
                f"{name}  {horizontal_name}-{vertical_name}",
                (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (28, 34, 41),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                "GRASP VOLUME",
                (14, panel_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (196, 24, 34),
                1,
                cv2.LINE_AA,
            )
            if horizontal == 0:
                arrow_y = panel_height - 38
                cv2.arrowedLine(
                    panel,
                    (28, arrow_y),
                    (100, arrow_y),
                    (31, 111, 235),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.16,
                )
                cv2.putText(
                    panel,
                    "+X APPROACH",
                    (108, arrow_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (31, 84, 180),
                    1,
                    cv2.LINE_AA,
                )
            view_metadata.append(
                {
                    "name": name.lower(),
                    "horizontal_axis": horizontal_name.lower(),
                    "vertical_axis": vertical_name.lower(),
                    "opening_boundary": _opening_boundary_metadata(
                        opening_contours, opening_mask
                    ),
                }
            )

        cv2.putText(
            canvas,
            "OFFICIAL ROBODOJO X5 - DYNAMIC GRASP VOLUME",
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (20, 27, 34),
            2,
            cv2.LINE_AA,
        )
        footer = (
            f"open={open_fraction:.3f} command={observed_command_fraction:.3f}  "
            "red=exact STL free volume  "
            "gray=official link6/link7/link8  frame=link6 EEF"
        )
        cv2.putText(
            canvas,
            footer,
            (24, 444),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (62, 70, 79),
            1,
            cv2.LINE_AA,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_encode_png_stdlib(canvas))
        return {
            "ok": True,
            "path": str(target),
            "build": "official_robodojo_x5_grasp_volume_three_views_exact_v4",
            "opening_query_version": OPENING_QUERY_VERSION,
            "views": view_metadata,
            "color": "red",
            "gripper_open_fraction": open_fraction,
            "gripper_joint_position_m": [joint_position_m],
            "gripper_geometry_qpos_normalized": [geometry_command_fraction],
            "gripper_qpos_normalized": [observed_command_fraction],
            "opening_bounds_eef_m": bounds.tolist(),
            "opening_voxel_count": int(len(opening_voxels)),
            "grasp_zone_capacity_cm3": float(opening_mesh["volume_cm3"]),
            "opening_quadrature_capacity_cm3": float(
                len(opening_voxels) * (VOXEL_M**3) * 1e6
            ),
            "image_width": canvas_width,
            "image_height": canvas_height,
            "geometry_source": (
                "policy_local_frozen_official_x5_two_finger_inner_surface_corridor"
            ),
            "opening_geometry": geometry["metadata"].get("opening_geometry"),
            "opening_projection": geometry["metadata"].get("opening_projection"),
            "opening_column_count": int(len(geometry["opening_columns"])),
            # Keep the historical key for clients that only understand the
            # voxel diagnostic, and expose the actual mesh renderer explicitly.
            "opening_render": "exact_projected_opening_voxel_union",
            "opening_mesh_render": "exact_projected_opening_triangle_mesh_union",
            "opening_boundary_source": ("projected_exact_opening_mesh_union_contours"),
            "exact_opening_component_selection": geometry["metadata"].get(
                "exact_opening_component_selection"
            ),
            "exact_opening_mesh_volume_cm3": float(opening_mesh["volume_cm3"]),
            "exact_opening_mesh_exact": bool(opening_mesh["exact"]),
            "exact_opening_mesh_source": opening_mesh["source"],
            "exact_opening_geometry": geometry["metadata"].get(
                "exact_opening_geometry"
            ),
            "exact_opening_generation": geometry["metadata"].get(
                "exact_opening_generation"
            ),
            "exact_opening_integrated_volume_cm3": float(
                geometry["metadata"].get("exact_opening_integrated_volume_cm3", 0.0)
            ),
            "exact_opening_inner_y_bounds_m": np.asarray(
                opening_mesh["inner_y_bounds_m"], dtype=np.float64
            ).tolist(),
            "exact_opening_y_boundary": opening_mesh.get("boundary_definition"),
            "exact_opening_metric_independent_of_voxel": True,
            "render_mode": (
                "official_mesh_component_projected_triangle_union_plus_"
                "projected_opening_triangle_mesh"
            ),
            "simulator_geometry_used": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "path": str(target),
        }


def _depth_shell_occupancy_mask(
    points_eef: np.ndarray,
    *,
    depth_linear: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    eef_pose: dict[str, Any],
    support_plane_z: float | None = None,
    surface_only: bool = False,
    target_anchor_world: np.ndarray | None = None,
    target_radius_m: float | None = None,
    occupancy_override: Any | None = None,
) -> np.ndarray:
    """Evaluate the planner's organized-depth shell, including its filters.

    The three-view artifact must report the same occupancy that selected the
    executable pose.  Defaults preserve the legacy diagnostic behavior.
    """
    depth = np.asarray(depth_linear, dtype=np.float64).squeeze()
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(extrinsic_matrix, dtype=np.float64).reshape(4, 4)
    rotation = quaternion_to_matrix(eef_pose["quat"])
    position = np.asarray(eef_pose["pos"], dtype=np.float64).reshape(3)
    occupancy = occupancy_override
    if occupancy is None:
        occupancy = DepthRayOccupancy(
            depth,
            intrinsic,
            extrinsic,
            support_z=support_plane_z,
            target_anchor=target_anchor_world,
            target_radius_m=(
                float(target_radius_m) if target_radius_m is not None else 0.110
            ),
        )
    return occupancy.occupied_mask(
        points_eef,
        rotation,
        position,
        surface_only=bool(surface_only),
        target_anchor=target_anchor_world,
        target_radius_m=target_radius_m,
    )


def _local_depth_points_eef(
    *,
    depth_linear: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    eef_pose: dict[str, Any],
    model_points_eef: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(depth_linear, dtype=np.float64).squeeze()
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    extrinsic = np.asarray(extrinsic_matrix, dtype=np.float64).reshape(4, 4)
    height, width = depth.shape
    stride = max(1, int(PLAN_THREE_VIEW_SCENE_STRIDE_PX))
    rows, columns = np.meshgrid(
        np.arange(0, height, stride, dtype=np.int64),
        np.arange(0, width, stride, dtype=np.int64),
        indexing="ij",
    )
    distances = depth[rows, columns]
    valid = np.isfinite(distances) & (distances > 0.02)
    rows = rows[valid].astype(np.float64)
    columns = columns[valid].astype(np.float64)
    distances = distances[valid]
    optical = np.column_stack(
        (
            (columns - intrinsic[0, 2]) * distances / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * distances / intrinsic[1, 1],
            distances,
        )
    )
    points_usd = optical * np.array([1.0, -1.0, -1.0])
    points_world = points_usd @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    rotation = quaternion_to_matrix(eef_pose["quat"])
    position = np.asarray(eef_pose["pos"], dtype=np.float64).reshape(3)
    points_eef = (points_world - position.reshape(1, 3)) @ rotation
    model = np.asarray(model_points_eef, dtype=np.float64).reshape(-1, 3)
    low = model.min(axis=0) - float(PLAN_THREE_VIEW_SCENE_MARGIN_M)
    high = model.max(axis=0) + float(PLAN_THREE_VIEW_SCENE_MARGIN_M)
    local = np.all(
        np.isfinite(points_eef)
        & (points_eef >= low.reshape(1, 3))
        & (points_eef <= high.reshape(1, 3)),
        axis=1,
    )
    return points_eef[local]


def render_x5_grasp_plan_three_views(
    *,
    depth_linear: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    eef_pose: dict[str, Any],
    hit_world: Any,
    grasp_vol_cm3: float,
    overlap_vol_cm3: float,
    inflated_overlap_vol_cm3: float,
    arm: str,
    axial_point: str,
    output_path: str | Path,
    support_plane_z: float | None = None,
    opening_surface_only: bool = False,
    opening_target_world: np.ndarray | None = None,
    opening_target_radius_m: float | None = None,
    rgb: np.ndarray | None = None,
    scene_occupancy_route: str | None = None,
    scene_occupancy_fallback_reason: str | None = None,
    scene_mesh_result: RGBDSceneMeshResult | None = None,
    opening_query_version: str | None = None,
    surface_query_radius_cells: int | None = None,
) -> dict[str, Any]:
    """Render selected-plan diagnostics with the exact triangular opening projection."""
    target = Path(output_path)
    try:
        import cv2

        geometry = x5_gripper_geometry()
        opening_contract = (
            {}
            if scene_mesh_result is None
            else scene_mesh_result.metadata.get("planner_opening_contract", {})
        )
        query_version = opening_query_version or opening_contract.get(
            "opening_query_version"
        )
        surface_query_cells = (
            surface_query_radius_cells
            if surface_query_radius_cells is not None
            else opening_contract.get("surface_query_radius_cells")
        )
        if query_version == OPENING_QUERY_VERSION:
            geometry = exact_x5_planning_geometry()
        vertices = x5_dynamic_visual_vertices([1.0])
        opening_mesh = x5_dynamic_opening_mesh([1.0])
        opening_mesh_vertices = np.asarray(opening_mesh["vertices"], dtype=np.float64)
        opening_mesh_faces = np.asarray(opening_mesh["faces"], dtype=np.int64)
        faces = np.asarray(geometry["visual_faces"], dtype=np.int64)
        face_components = np.asarray(geometry["visual_face_component"], dtype=np.int64)
        gripper_voxels = np.asarray(geometry["gripper_voxels"], dtype=np.float64)
        opening_voxels = np.asarray(geometry["opening_voxels"], dtype=np.float64)
        inflated_voxels = np.asarray(
            geometry["inflated_gripper_voxels"], dtype=np.float64
        )
        model_points = np.concatenate(
            (vertices, opening_mesh_vertices, opening_voxels), axis=0
        )
        scene_points = _local_depth_points_eef(
            depth_linear=depth_linear,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            eef_pose=eef_pose,
            model_points_eef=model_points,
        )
        scene_occupancy = None
        mesh_route = "v4"
        mesh_fallback_reason = scene_occupancy_fallback_reason
        mesh_reused_from_planner = False
        if rgb is not None or scene_mesh_result is not None:
            requested_route = (
                str(
                    scene_occupancy_route
                    or os.environ.get(
                        "ROBODOJO_INTERFACE_RGBD_MESH_VERSION", "robodojo_v1"
                    )
                )
                .strip()
                .lower()
            )
            continuous_route = requested_route in {
                "v2_continuous",
                "robodojo_v2_continuous",
            }
            if requested_route in {"v1", "robodojo_v1", "latest"} or continuous_route:
                # Use the exact, already partitioned result scored by the
                # planner. Reconstructing only v1 would silently replace v2's
                # continuous collision cover (and any robot withholding).
                expected_version = (
                    "robodojo_v2_continuous" if continuous_route else "robodojo_v1"
                )
                if continuous_route and scene_mesh_result is None:
                    raise ValueError(
                        "continuous plan visualization requires the planner's frozen scene_mesh_result"
                    )
                if scene_mesh_result is not None:
                    actual_version = scene_mesh_result.metadata.get("effective_version")
                    if actual_version != expected_version:
                        raise ValueError(
                            f"plan visualization mesh route mismatch: {expected_version} != {actual_version}"
                        )
                try:
                    mesh_result = scene_mesh_result
                    if mesh_result is None:
                        mesh_result = reconstruct_rgbd_scene_mesh(
                            np.asarray(rgb),
                            depth_linear,
                            intrinsic_matrix=intrinsic_matrix,
                            extrinsic_matrix=extrinsic_matrix,
                            pixel_stride=1,
                            min_depth_m=0.03,
                            max_depth_m=8.0,
                            collision_extrusion_mm=12.0,
                            analytic_support_plane=True,
                        )
                    else:
                        mesh_reused_from_planner = True
                    scene_occupancy = MeshSceneOccupancy(
                        mesh_result,
                        support_z=support_plane_z,
                        target_anchor=opening_target_world,
                        target_radius_m=(
                            0.110
                            if opening_target_radius_m is None
                            else float(opening_target_radius_m)
                        ),
                        surface_query_radius_cells=surface_query_cells,
                    )
                    mesh_route = str(
                        mesh_result.metadata.get("effective_version", "robodojo_v1")
                    )
                except (OSError, TypeError, ValueError, RuntimeError) as error:
                    if scene_mesh_result is not None:
                        # A failed query on a supplied plan must not be drawn
                        # with another occupancy under the original numbers.
                        raise
                    # The planner uses the same explicit v4 fallback when mesh
                    # construction fails.  Keep the diagnostic image aligned
                    # with that effective route instead of mixing metrics.
                    scene_occupancy = None
                    mesh_route = "v4_fallback"
                    mesh_fallback_reason = (
                        mesh_fallback_reason or f"{type(error).__name__}: {error}"
                    )
            elif requested_route in {
                "v4",
                "legacy",
                "organized_shell",
                "robodojo_v4",
                "v4_fallback",
            }:
                mesh_route = requested_route
            else:
                mesh_route = "v4_fallback"
                mesh_fallback_reason = (
                    mesh_fallback_reason
                    or f"unsupported scene occupancy route: {requested_route}"
                )
        grasp_occupied = _depth_shell_occupancy_mask(
            opening_voxels,
            depth_linear=depth_linear,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            eef_pose=eef_pose,
            support_plane_z=support_plane_z,
            surface_only=opening_surface_only,
            target_anchor_world=opening_target_world,
            target_radius_m=opening_target_radius_m,
            occupancy_override=scene_occupancy,
        )
        overlap_occupied = _depth_shell_occupancy_mask(
            gripper_voxels,
            depth_linear=depth_linear,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            eef_pose=eef_pose,
            occupancy_override=scene_occupancy,
        )
        inflated_occupied = _depth_shell_occupancy_mask(
            inflated_voxels,
            depth_linear=depth_linear,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            eef_pose=eef_pose,
            occupancy_override=scene_occupancy,
        )
        rotation = quaternion_to_matrix(eef_pose["quat"])
        position = np.asarray(eef_pose["pos"], dtype=np.float64).reshape(3)
        hit_eef = (
            np.asarray(hit_world, dtype=np.float64).reshape(3) - position
        ) @ rotation

        canvas_height, canvas_width = 540, 1400
        canvas = np.full((canvas_height, canvas_width, 3), 246, dtype=np.uint8)
        panel_width, panel_height = 440, 390
        panel_y = 58
        view_specs = (
            ("FRONT", "X", "Y", 0, 1, 2),
            ("SIDE", "X", "Z", 0, 2, 1),
            ("TOP", "Y", "Z", 1, 2, 0),
        )
        view_metadata = []
        for view_index, spec in enumerate(view_specs):
            name, horizontal_name, vertical_name, horizontal, vertical, depth_axis = (
                spec
            )
            panel_x = 20 + view_index * 460
            panel = canvas[
                panel_y : panel_y + panel_height,
                panel_x : panel_x + panel_width,
            ]
            panel[:] = 255
            plotted = [vertices, opening_mesh_vertices, opening_voxels]
            if len(scene_points):
                plotted.append(scene_points)
            all_points = np.concatenate(plotted, axis=0)
            low = all_points[:, [horizontal, vertical]].min(axis=0)
            high = all_points[:, [horizontal, vertical]].max(axis=0)
            center = 0.5 * (low + high)
            span = np.maximum(high - low, 1e-6)
            scale = min(
                (panel_width - 54.0) / span[0],
                (panel_height - 78.0) / span[1],
            )

            def project(points: np.ndarray) -> np.ndarray:
                values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
                u = (values[:, horizontal] - center[0]) * scale + panel_width / 2
                v = panel_height / 2 - (values[:, vertical] - center[1]) * scale
                return np.column_stack((u, v))

            if len(scene_points):
                scene_pixels = np.rint(project(scene_points)).astype(np.int32)
                inside = (
                    (scene_pixels[:, 0] >= 0)
                    & (scene_pixels[:, 0] < panel_width)
                    & (scene_pixels[:, 1] >= 0)
                    & (scene_pixels[:, 1] < panel_height)
                )
                scene_pixels = scene_pixels[inside]
                panel[scene_pixels[:, 1], scene_pixels[:, 0]] = [128, 175, 202]

            opening_mask, opening_contours = _projected_opening_mesh_mask(
                project(opening_mesh_vertices),
                opening_mesh_faces,
                panel_width=panel_width,
                panel_height=panel_height,
                cv2=cv2,
            )

            projected_vertices = project(vertices)
            component_colors = (
                (76, 88, 100),
                (84, 98, 112),
                (91, 106, 120),
            )
            component_depths = []
            for component_index in np.unique(face_components):
                component_mask = face_components == component_index
                component_depths.append(
                    (
                        float(vertices[faces[component_mask], depth_axis].mean()),
                        int(component_index),
                    )
                )
            for _depth, component_index in sorted(component_depths):
                component_faces = faces[face_components == component_index]
                component_render_mask, contours = _projected_triangle_mesh_mask(
                    projected_vertices,
                    component_faces,
                    panel_width=panel_width,
                    panel_height=panel_height,
                    cv2=cv2,
                    supersample=2,
                )
                visible = component_render_mask > 0
                if np.any(visible):
                    color = np.asarray(
                        component_colors[component_index % len(component_colors)],
                        dtype=np.uint8,
                    )
                    panel[visible] = color
                    cv2.drawContours(panel, contours, -1, (47, 57, 67), 1, cv2.LINE_AA)

            _blend_opening_mask(
                panel,
                opening_mask,
                cv2=cv2,
                color=(245, 154, 158),
                alpha=0.22,
            )
            empty_grasp_pixels = np.rint(project(opening_voxels[::10])).astype(int)
            empty_valid = (
                (empty_grasp_pixels[:, 0] >= 0)
                & (empty_grasp_pixels[:, 0] < panel_width)
                & (empty_grasp_pixels[:, 1] >= 0)
                & (empty_grasp_pixels[:, 1] < panel_height)
            )
            empty_grasp_pixels = empty_grasp_pixels[empty_valid]
            panel[empty_grasp_pixels[:, 1], empty_grasp_pixels[:, 0]] = [225, 94, 101]

            point_layers = (
                (inflated_voxels[inflated_occupied], (243, 183, 45), 1),
                (gripper_voxels[overlap_occupied], (242, 117, 38), 2),
                (opening_voxels[grasp_occupied], (198, 24, 34), 2),
            )
            for points, color, radius in point_layers:
                if not len(points):
                    continue
                pixels = np.unique(np.rint(project(points)).astype(int), axis=0)
                for pixel in pixels:
                    if 0 <= pixel[0] < panel_width and 0 <= pixel[1] < panel_height:
                        cv2.circle(
                            panel,
                            tuple(pixel),
                            radius,
                            color,
                            thickness=-1,
                            lineType=cv2.LINE_AA,
                        )
            hit_pixel = np.rint(project(hit_eef.reshape(1, 3))[0]).astype(int)
            if 0 <= hit_pixel[0] < panel_width and 0 <= hit_pixel[1] < panel_height:
                cv2.circle(panel, tuple(hit_pixel), 6, (34, 197, 94), -1, cv2.LINE_AA)
                cv2.circle(panel, tuple(hit_pixel), 7, (20, 93, 52), 1, cv2.LINE_AA)
            if opening_contours:
                cv2.drawContours(
                    panel,
                    opening_contours,
                    -1,
                    (196, 24, 34),
                    2,
                    cv2.LINE_AA,
                )
            cv2.rectangle(
                panel,
                (0, 0),
                (panel_width - 1, panel_height - 1),
                (205, 211, 218),
                1,
            )
            cv2.putText(
                panel,
                f"{name}  {horizontal_name}-{vertical_name}",
                (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (28, 34, 41),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"grasp occupied: {int(np.count_nonzero(grasp_occupied))} vox",
                (14, panel_height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (190, 24, 34),
                1,
                cv2.LINE_AA,
            )
            view_metadata.append(
                {
                    "name": name.lower(),
                    "horizontal_axis": horizontal_name.lower(),
                    "vertical_axis": vertical_name.lower(),
                    "opening_boundary": _opening_boundary_metadata(
                        opening_contours, opening_mask
                    ),
                }
            )

        cv2.putText(
            canvas,
            "ROBODOJO X5 RGB-D LITE - SELECTED GRASP THREE VIEWS",
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (20, 27, 34),
            2,
            cv2.LINE_AA,
        )
        footer_one = (
            f"grasp={float(grasp_vol_cm3):.3f} cm3  "
            f"overlap={float(overlap_vol_cm3):.3f} cm3  "
            f"inflated={float(inflated_overlap_vol_cm3):.3f} cm3  "
            f"arm={arm}  axial={axial_point}"
        )
        footer_two = (
            "blue=observed RGB-D  gray=official X5 STL  red=exact opening/grasp  "
            "dark red=occupied grasp  orange=overlap  yellow=inflated overlap"
        )
        cv2.putText(
            canvas,
            footer_one,
            (24, 480),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (48, 57, 66),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            footer_two,
            (24, 512),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (72, 80, 88),
            1,
            cv2.LINE_AA,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_encode_png_stdlib(canvas))
        voxel_cm3 = VOXEL_M**3 * 1e6
        measured = {
            "grasp_vol_cm3": float(np.count_nonzero(grasp_occupied) * voxel_cm3),
            "overlap_vol_cm3": float(np.count_nonzero(overlap_occupied) * voxel_cm3),
            "inflated_overlap_vol_cm3": float(
                np.count_nonzero(inflated_occupied) * voxel_cm3
            ),
        }
        expected = {
            "grasp_vol_cm3": float(grasp_vol_cm3),
            "overlap_vol_cm3": float(overlap_vol_cm3),
            "inflated_overlap_vol_cm3": float(inflated_overlap_vol_cm3),
        }
        return {
            "ok": True,
            "path": str(target),
            "build": "official_robodojo_x5_selected_grasp_three_views_query_contract_v3",
            "opening_query_version": query_version or "legacy_voxel_columns",
            "surface_query_radius_cells": surface_query_cells,
            "views": view_metadata,
            "frame": "selected_link6_eef",
            "geometry_source": (
                "policy_local_frozen_official_x5_two_finger_inner_surface_corridor"
            ),
            "opening_geometry": geometry["metadata"].get("opening_geometry"),
            "opening_projection": geometry["metadata"].get("opening_projection"),
            "opening_column_count": int(len(geometry["opening_columns"])),
            "opening_render": "exact_projected_opening_voxel_union",
            "opening_occupancy_filter": {
                "surface_only": bool(opening_surface_only),
                "support_plane_z_m": (
                    None if support_plane_z is None else float(support_plane_z)
                ),
                "target_anchor_world": (
                    None
                    if opening_target_world is None
                    else np.asarray(opening_target_world, dtype=np.float64)
                    .reshape(3)
                    .tolist()
                ),
                "target_radius_m": (
                    None
                    if opening_target_radius_m is None
                    else float(opening_target_radius_m)
                ),
            },
            "opening_mesh_render": "exact_projected_opening_triangle_mesh_union",
            "opening_boundary_source": ("projected_exact_opening_mesh_union_contours"),
            "exact_opening_component_selection": geometry["metadata"].get(
                "exact_opening_component_selection"
            ),
            "exact_opening_mesh_volume_cm3": float(opening_mesh["volume_cm3"]),
            "exact_opening_mesh_exact": bool(opening_mesh["exact"]),
            "exact_opening_mesh_source": opening_mesh["source"],
            "exact_opening_geometry": geometry["metadata"].get(
                "exact_opening_geometry"
            ),
            "exact_opening_generation": geometry["metadata"].get(
                "exact_opening_generation"
            ),
            "exact_opening_integrated_volume_cm3": float(
                geometry["metadata"].get("exact_opening_integrated_volume_cm3", 0.0)
            ),
            "exact_opening_inner_y_bounds_m": np.asarray(
                opening_mesh["inner_y_bounds_m"], dtype=np.float64
            ).tolist(),
            "exact_opening_y_boundary": opening_mesh.get("boundary_definition"),
            "exact_opening_metric_independent_of_voxel": True,
            "render_mode": (
                "official_mesh_component_projected_triangle_union_plus_"
                "projected_opening_triangle_mesh"
            ),
            "scene_source": "frozen_evaluator_depth_linear",
            "scene_occupancy_route": mesh_route,
            "scene_occupancy_fallback_reason": mesh_fallback_reason,
            "scene_mesh_reused_from_planner": mesh_reused_from_planner,
            "scene_point_count": int(len(scene_points)),
            "grasp_occupied_voxel_count": int(np.count_nonzero(grasp_occupied)),
            "overlap_occupied_voxel_count": int(np.count_nonzero(overlap_occupied)),
            "inflated_overlap_occupied_voxel_count": int(
                np.count_nonzero(inflated_occupied)
            ),
            "expected_metrics_cm3": expected,
            "rendered_metrics_cm3": measured,
            "metrics_match_planner": bool(
                all(abs(measured[key] - expected[key]) <= 1e-9 for key in expected)
            ),
            "image_width": canvas_width,
            "image_height": canvas_height,
            "segmentation_used": False,
            "simulator_geometry_used": False,
            "simulator_state_used": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "path": str(target),
        }


__all__ = [
    "DEPTH_OCCLUDE_EPS_M",
    "PLAN_GRIPPER_ALPHA",
    "PLAN_GRIPPER_BASE_RGB",
    "WRIST_GRASP_ALPHA",
    "WRIST_GRASP_RGB",
    "WRIST_OVERLAY_MAX_GRIPPER_QVEL_M_S",
    "WRIST_SELF_DEPTH_TOLERANCE_M",
    "WRIST_SELF_UV_RADIUS_PX",
    "render_x5_grasp_plan_three_views",
    "render_x5_grasp_volume_three_views",
    "render_x5_gripper_overlay",
    "render_x5_wrist_grasp_volume_overlay",
]
