"""Submission-local RGB-D visualizations for the strict official_v2 tools.

All geometry is derived from frozen evaluator observations and static assets
shipped in this package.  The module has no simulator, scene, segmentation, or
legacy Behavior Interface dependency.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .grasp_geometry_local import (
    gap_opening_strict_mask_from_lut,
    get_wrist_opening_lut,
    quat_to_mat_xyzw,
)
from .base_path_overlay_local import (
    render_base_forward_path_overlay_frame_v2,
    render_base_forward_path_overlay_v2,
)


DEPTH_OCCLUDE_EPS_M = 0.003
PLAN_GRIPPER_ALPHA = 0.78
PLAN_GRIPPER_BASE_RGB = np.array([150.0, 18.0, 18.0], dtype=np.float32)
WRIST_SELF_DEPTH_TOLERANCE_M = 0.00075
WRIST_SELF_UV_RADIUS_PX = 2
_GRIPPER_VISUAL_MESH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "r1pro_gripper_visual_mesh_open.npz",
)

_WRIST_RED_BGR = np.array([0.0, 0.0, 255.0], dtype=np.float32)

WRIST_OPENING_REFERENCE_PIXELS = 480 * 480
WRIST_OPENING_MIN_COMPONENT_POINTS = 512
WRIST_OPENING_MIN_SIDE_POINTS = 64
WRIST_OPENING_MIN_CENTER_POINTS = 128
WRIST_OPENING_SIDE_BAND_FRACTION = 0.20
WRIST_OPENING_CENTER_LO_FRACTION = 0.35
WRIST_OPENING_CENTER_HI_FRACTION = 0.65
WRIST_OPENING_BOUNDARY_BAND_FRACTION = 0.10
WRIST_OPENING_OCCLUSION_MIN_DOMINANT_FRACTION = 0.65


def _camera_values(
    camera: Dict[str, Any],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    position = np.asarray(camera["pos"], dtype=np.float64).reshape(3)
    quaternion = np.asarray(camera["quat"], dtype=np.float64).reshape(4)
    if camera.get("fx") is not None:
        fx = float(camera["fx"])
    else:
        fx = (
            float(camera["focal_length"])
            / float(camera["horizontal_aperture"])
            * int(width)
        )
    fy = float(camera.get("fy", fx))
    cx = float(camera.get("cx", float(width) * 0.5))
    cy = float(camera.get("cy", float(height) * 0.5))
    return position, quaternion, fx, fy, cx, cy


def _scene_depth(
    depth_linear: Optional[np.ndarray],
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    if depth_linear is None:
        return None
    depth = np.asarray(depth_linear, dtype=np.float32).squeeze()
    if depth.shape != (int(height), int(width)):
        return None
    return depth


def _project_world_points(
    points_world: np.ndarray,
    camera: Dict[str, Any],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    camera_pos, camera_quat, fx, fy, cx, cy = _camera_values(
        camera,
        width,
        height,
    )
    rotation = quat_to_mat_xyzw(camera_quat)
    points_camera = (
        rotation.T
        @ (
            np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
            - camera_pos.reshape(1, 3)
        ).T
    ).T
    depth = -points_camera[:, 2]
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 1e-5)
    uv[valid, 0] = cx + fx * points_camera[valid, 0] / depth[valid]
    uv[valid, 1] = cy - fy * points_camera[valid, 1] / depth[valid]
    return uv, depth


def _composite(
    image_bgr: np.ndarray,
    color_bgr: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)[..., None]
    output = image.astype(np.float32) * (1.0 - a)
    output += np.asarray(color_bgr, dtype=np.float32).reshape(1, 1, 3) * a
    return np.rint(np.clip(output, 0.0, 255.0)).astype(np.uint8)


def _visible_alpha(
    layer_depth: np.ndarray,
    layer_alpha: np.ndarray,
    scene_depth: Optional[np.ndarray],
) -> np.ndarray:
    alpha = np.asarray(layer_alpha, dtype=np.float32).copy()
    present = np.isfinite(layer_depth) & (alpha > 0.0)
    alpha[~present] = 0.0
    if scene_depth is None:
        return alpha
    scene = np.asarray(scene_depth, dtype=np.float32)
    valid_scene = np.isfinite(scene) & (scene > 0.02)
    occluded = (
        present
        & valid_scene
        & (scene < layer_depth - float(DEPTH_OCCLUDE_EPS_M))
    )
    alpha[occluded] = 0.0
    return alpha


def _composite_color_layer(
    image_bgr: np.ndarray,
    color_layer_bgr: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    layer = np.asarray(color_layer_bgr, dtype=np.uint8)
    if layer.shape != image.shape:
        raise ValueError("color layer shape does not match RGB image")
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)[..., None]
    output = image.astype(np.float32) * (1.0 - a)
    output += layer.astype(np.float32) * a
    return np.rint(np.clip(output, 0.0, 255.0)).astype(np.uint8)


@lru_cache(maxsize=1)
def _gripper_visual_mesh() -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tuple[str, ...],
]:
    with np.load(_GRIPPER_VISUAL_MESH_PATH, allow_pickle=False) as asset:
        vertices = np.asarray(
            asset["vertices_eef"],
            dtype=np.float64,
        ).reshape(-1, 3)
        faces = np.asarray(asset["faces"], dtype=np.int64).reshape(-1, 3)
        face_component = np.asarray(
            asset["face_component"],
            dtype=np.int64,
        ).reshape(-1)
        component_names = tuple(
            str(value) for value in asset["component_names"].tolist()
        )
        gripper_q_m = float(
            np.asarray(asset["gripper_q_m"], dtype=np.float64).reshape(-1)[0]
        )
    if vertices.shape[0] < 3 or faces.shape[0] < 1:
        raise ValueError("frozen gripper visual mesh is empty")
    if len(face_component) != len(faces):
        raise ValueError("gripper mesh face components do not match faces")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("gripper mesh face index is out of range")
    if abs(gripper_q_m - 0.05) > 1e-6:
        raise ValueError("gripper visual mesh is not the v2 open geometry")
    vertices.setflags(write=False)
    faces.setflags(write=False)
    face_component.setflags(write=False)
    return vertices, faces, face_component, component_names


def _prepare_gripper_triangles(
    *,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    camera: Dict[str, Any],
    width: int,
    height: int,
    gripper_qpos: Optional[Sequence[float]] = None,
    component_ids: Optional[Sequence[int]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    vertices_eef, faces, face_component, _component_names = (
        _gripper_visual_mesh()
    )
    selected = np.ones(len(faces), dtype=bool)
    if component_ids is not None:
        selected &= np.isin(
            face_component,
            np.asarray(component_ids, dtype=np.int64).reshape(-1),
        )
    if not np.any(selected):
        return None
    selected_faces = faces[selected]
    selected_components = face_component[selected]
    selected_face_indices = np.flatnonzero(selected)
    triangle_eef = vertices_eef[selected_faces].copy()
    if gripper_qpos is not None:
        values = _normalized_gripper_qpos(gripper_qpos)
        positive_finger = selected_components == 1
        negative_finger = selected_components == 2
        triangle_eef[positive_finger, :, 1] -= 0.05 - values[0]
        triangle_eef[negative_finger, :, 1] += 0.05 - values[1]
    eef_rotation = quat_to_mat_xyzw(eef_quat)
    triangle_world = (
        triangle_eef @ eef_rotation.T + eef_pos.reshape(1, 1, 3)
    )
    uv, depth = _project_world_points(
        triangle_world.reshape(-1, 3),
        camera,
        width,
        height,
    )
    triangle_uv = uv.reshape(-1, 3, 2)
    triangle_depth = depth.reshape(-1, 3)
    valid = (
        np.all(np.isfinite(triangle_uv), axis=(1, 2))
        & np.all(np.isfinite(triangle_depth), axis=1)
        & np.all(triangle_depth > 1e-6, axis=1)
    )
    in_bounds = np.any(
        (
            (triangle_uv[:, :, 0] > -float(width))
            & (triangle_uv[:, :, 0] < 2.0 * float(width))
            & (triangle_uv[:, :, 1] > -float(height))
            & (triangle_uv[:, :, 1] < 2.0 * float(height))
        ),
        axis=1,
    )
    keep = valid & in_bounds
    if not np.any(keep):
        return None
    return (
        triangle_world[keep],
        triangle_uv[keep],
        triangle_depth[keep],
        selected_components[keep],
        selected_face_indices[keep],
    )


def _shade_gripper_faces(
    triangle_world: np.ndarray,
    camera_position: np.ndarray,
) -> np.ndarray:
    edge_1 = triangle_world[:, 1] - triangle_world[:, 0]
    edge_2 = triangle_world[:, 2] - triangle_world[:, 0]
    normals = np.cross(edge_1, edge_2)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    centroids = triangle_world.mean(axis=1)
    view = camera_position.reshape(1, 3) - centroids
    view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
    facing = np.abs(np.einsum("ij,ij->i", normals, view))
    return 0.4 + 0.6 * np.clip(facing, 0.0, 1.0)


def _rasterize_gripper_triangle(
    *,
    triangle_uv: np.ndarray,
    triangle_depth: np.ndarray,
    color_bgr: np.ndarray,
    color_buffer: np.ndarray,
    depth_buffer: np.ndarray,
    mask_buffer: np.ndarray,
) -> None:
    height, width = depth_buffer.shape
    xs = triangle_uv[:, 0]
    ys = triangle_uv[:, 1]
    min_u = max(0, int(np.floor(float(xs.min()))))
    max_u = min(width - 1, int(np.ceil(float(xs.max()))))
    min_v = max(0, int(np.floor(float(ys.min()))))
    max_v = min(height - 1, int(np.ceil(float(ys.max()))))
    if min_u > max_u or min_v > max_v:
        return
    pixel_v_int, pixel_u_int = np.mgrid[
        min_v : max_v + 1,
        min_u : max_u + 1,
    ]
    pixel_u = pixel_u_int.astype(np.float64) + 0.5
    pixel_v = pixel_v_int.astype(np.float64) + 0.5
    u0, v0 = triangle_uv[0]
    u1, v1 = triangle_uv[1]
    u2, v2 = triangle_uv[2]
    denominator = (
        (v1 - v2) * (u0 - u2)
        + (u2 - u1) * (v0 - v2)
    )
    if abs(float(denominator)) < 1e-12:
        return
    weight_0 = (
        (v1 - v2) * (pixel_u - u2)
        + (u2 - u1) * (pixel_v - v2)
    ) / denominator
    weight_1 = (
        (v2 - v0) * (pixel_u - u2)
        + (u0 - u2) * (pixel_v - v2)
    ) / denominator
    weight_2 = 1.0 - weight_0 - weight_1
    inside = (
        (weight_0 >= -1e-6)
        & (weight_1 >= -1e-6)
        & (weight_2 >= -1e-6)
        & (weight_0 <= 1.0 + 1e-6)
        & (weight_1 <= 1.0 + 1e-6)
        & (weight_2 <= 1.0 + 1e-6)
    )
    if not np.any(inside):
        return
    pixel_u = pixel_u[inside]
    pixel_v = pixel_v[inside]
    pixel_u_int = pixel_u_int[inside]
    pixel_v_int = pixel_v_int[inside]
    weight_0 = weight_0[inside]
    weight_1 = weight_1[inside]
    weight_2 = weight_2[inside]
    inverse_depth = (
        weight_0 / triangle_depth[0]
        + weight_1 / triangle_depth[1]
        + weight_2 / triangle_depth[2]
    )
    valid_depth = inverse_depth > 1e-12
    if not np.any(valid_depth):
        return
    pixel_u = pixel_u[valid_depth]
    pixel_v = pixel_v[valid_depth]
    pixel_u_int = pixel_u_int[valid_depth]
    pixel_v_int = pixel_v_int[valid_depth]
    inverse_depth = inverse_depth[valid_depth]
    pixel_depth = (1.0 / inverse_depth).astype(np.float32)
    closer = (
        pixel_depth
        < depth_buffer[pixel_v_int, pixel_u_int]
    )
    if not np.any(closer):
        return
    visible_u = pixel_u_int[closer]
    visible_v = pixel_v_int[closer]
    depth_buffer[visible_v, visible_u] = pixel_depth[closer]
    mask_buffer[visible_v, visible_u] = 255
    color_buffer[visible_v, visible_u] = color_bgr


def _rasterize_gripper_buffers(
    *,
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    camera: Dict[str, Any],
    width: int,
    height: int,
    gripper_qpos: Optional[Sequence[float]] = None,
    component_ids: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, Any]]:
    prepared = _prepare_gripper_triangles(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        camera=camera,
        width=width,
        height=height,
        gripper_qpos=gripper_qpos,
        component_ids=component_ids,
    )
    if prepared is None:
        return None
    (
        triangle_world,
        triangle_uv,
        triangle_depth,
        face_component,
        face_indices,
    ) = prepared
    camera_position, _camera_quat, _fx, _fy, _cx, _cy = _camera_values(
        camera,
        width,
        height,
    )
    shade = _shade_gripper_faces(triangle_world, camera_position)
    color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    mask_buffer = np.zeros((height, width), dtype=np.uint8)
    base_bgr = PLAN_GRIPPER_BASE_RGB[::-1]
    order = np.argsort(-triangle_depth.mean(axis=1))
    for index in order:
        color_bgr = np.floor(
            base_bgr * float(shade[index])
        ).astype(np.uint8)
        _rasterize_gripper_triangle(
            triangle_uv=triangle_uv[index],
            triangle_depth=triangle_depth[index],
            color_bgr=color_bgr,
            color_buffer=color_buffer,
            depth_buffer=depth_buffer,
            mask_buffer=mask_buffer,
        )
    return {
        "color": color_buffer,
        "depth": depth_buffer,
        "mask": mask_buffer,
        "shade": shade,
        "face_component": face_component,
        "face_indices": face_indices,
    }


def render_plan_gripper_overlay(
    *,
    rgb_path: str,
    depth_linear: Optional[np.ndarray],
    camera: Dict[str, Any],
    eef_pose: Dict[str, Any],
    output_path: str,
    gripper_qpos: Optional[Sequence[float]] = None,
    alpha: float = PLAN_GRIPPER_ALPHA,
) -> Dict[str, Any]:
    """Render the v2 visual mesh with face shading and depth occlusion."""
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "ok": False,
            "error": f"failed to read RGB image: {rgb_path}",
            "path": output_path,
        }
    height, width = image.shape[:2]
    try:
        eef_pos = np.asarray(eef_pose["pos"], dtype=np.float64).reshape(3)
        eef_quat = np.asarray(eef_pose["quat"], dtype=np.float64).reshape(4)
        buffers = _rasterize_gripper_buffers(
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            camera=camera,
            width=width,
            height=height,
            gripper_qpos=gripper_qpos,
        )
        if buffers is None:
            raise ValueError("open gripper visual mesh is outside the camera")
        color_buffer = np.asarray(buffers["color"], dtype=np.uint8)
        layer_depth = np.asarray(buffers["depth"], dtype=np.float32)
        mask_buffer = np.asarray(buffers["mask"], dtype=np.uint8)
        layer_alpha = np.where(
            mask_buffer > 0,
            float(np.clip(alpha, 0.0, 1.0)),
            0.0,
        ).astype(np.float32)
        rasterized_pixel_count = int(np.isfinite(layer_depth).sum())
        if rasterized_pixel_count == 0:
            raise ValueError(
                "open gripper visual mesh projected outside the RGB image"
            )
        scene = _scene_depth(depth_linear, width, height)
        visible_alpha = _visible_alpha(layer_depth, layer_alpha, scene)
        output = _composite_color_layer(
            image,
            color_buffer,
            visible_alpha,
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if not cv2.imwrite(output_path, output):
            raise RuntimeError(f"failed to write {output_path}")
        vertices, faces, _face_component, component_names = (
            _gripper_visual_mesh()
        )
        shade = np.asarray(buffers["shade"], dtype=np.float64)
        projected_components = np.asarray(
            buffers["face_component"],
            dtype=np.int64,
        )
        return {
            "ok": True,
            "path": output_path,
            "build": "official_v2_local_gripper_visual_mesh_zbuffer_v2",
            "color": "red",
            "geometry_source": (
                "submission_local_frozen_v2_dynamic_finger_visual_triangle_mesh"
                if gripper_qpos is not None
                else "submission_local_frozen_v2_open_visual_triangle_mesh"
            ),
            "gripper_qpos_m": _normalized_gripper_qpos(
                gripper_qpos
            ).astype(float).tolist(),
            "input_vertex_count": int(len(vertices)),
            "input_triangle_count": int(len(faces)),
            "projected_triangle_count": int(len(shade)),
            "projected_component_count": int(
                len(np.unique(projected_components))
            ),
            "component_names": list(component_names),
            "face_shading": "v2_abs_view_lambert_0.4_plus_0.6",
            "shade_min": float(shade.min()),
            "shade_max": float(shade.max()),
            "shade_level_count_1e3": int(
                len(np.unique(np.rint(shade * 1000.0).astype(np.int32)))
            ),
            "base_color_rgb": PLAN_GRIPPER_BASE_RGB.astype(int).tolist(),
            "alpha": float(np.clip(alpha, 0.0, 1.0)),
            "rasterized_pixel_count": rasterized_pixel_count,
            "visible_pixel_count": int(np.count_nonzero(visible_alpha > 0.0)),
            "depth_occlusion": scene is not None,
            "depth_occlude_eps_m": float(DEPTH_OCCLUDE_EPS_M),
            "segmentation_used": False,
            "simulator_geometry_used": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "path": output_path,
        }


def render_head_path_overlay(
    *,
    rgb_path: str,
    depth_linear: Optional[np.ndarray],
    camera: Dict[str, Any],
    robot: Dict[str, Any],
    output_path: str,
) -> Dict[str, Any]:
    """Render the frozen v2 metric base path HUD."""
    return render_base_forward_path_overlay_v2(
        rgb_path=rgb_path,
        depth_linear=depth_linear,
        camera=camera,
        robot=robot,
        output_path=output_path,
    )


def render_head_path_overlay_frame(
    *,
    image_bgr: np.ndarray,
    depth_linear: Optional[np.ndarray],
    camera: Dict[str, Any],
    robot: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Render the frozen v2 metric base path HUD without filesystem I/O."""
    return render_base_forward_path_overlay_frame_v2(
        image_bgr=image_bgr,
        depth_linear=depth_linear,
        camera=camera,
        robot=robot,
    )


def _dynamic_opening_lut(
    gripper_qpos: Optional[Sequence[float]],
) -> Dict[str, np.ndarray | float]:
    frozen = get_wrist_opening_lut((0.05, 0.05))
    values = _normalized_gripper_qpos(gripper_qpos)
    frozen_q = np.asarray(frozen["gripper_qpos_m"], dtype=np.float64).reshape(2)
    y_inner_lo = np.asarray(
        frozen["y_inner_lo"],
        dtype=np.float64,
    ) + (frozen_q[1] - values[1])
    y_inner_hi = np.asarray(
        frozen["y_inner_hi"],
        dtype=np.float64,
    ) - (frozen_q[0] - values[0])
    return {
        "gripper_qpos_m": values,
        "z": np.asarray(frozen["z"], dtype=np.float64),
        "y_inner_lo": y_inner_lo,
        "y_inner_hi": y_inner_hi,
        "x_half_gap": np.asarray(frozen["x_half_gap"], dtype=np.float64),
        "gap_width": y_inner_hi - y_inner_lo,
        "z_grasp_lo": float(np.asarray(frozen["z_grasp_lo"]).reshape(())),
        "z_grasp_hi": float(np.asarray(frozen["z_grasp_hi"]).reshape(())),
    }


def _normalized_gripper_qpos(
    gripper_qpos: Optional[Sequence[float]],
) -> np.ndarray:
    values = np.asarray(
        [0.05, 0.05] if gripper_qpos is None else gripper_qpos,
        dtype=np.float64,
    ).reshape(-1)
    if len(values) < 2:
        values = np.pad(values, (0, 2 - len(values)), constant_values=0.05)
    return np.clip(values[:2], 0.0, 0.05)


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
            scene_window = scene[target]
            model_window = model[source]
            comparable = np.isfinite(scene_window) & np.isfinite(model_window)
            candidate = np.full(scene_window.shape, np.inf, dtype=np.float32)
            np.subtract(
                scene_window,
                model_window,
                out=candidate,
                where=comparable,
            )
            np.abs(candidate, out=candidate)
            np.minimum(best_delta[target], candidate, out=best_delta[target])
    valid_scene = np.isfinite(scene) & (scene > 0.02)
    return valid_scene & (best_delta <= float(tolerance_m))


def _gripper_self_depth_mask(
    *,
    scene_depth: np.ndarray,
    camera: Dict[str, Any],
    eef_pose: Dict[str, Any],
    gripper_qpos: Optional[Sequence[float]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scene = np.asarray(scene_depth, dtype=np.float32)
    height, width = scene.shape
    buffers = _rasterize_gripper_buffers(
        eef_pos=np.asarray(eef_pose["pos"], dtype=np.float64).reshape(3),
        eef_quat=np.asarray(eef_pose["quat"], dtype=np.float64).reshape(4),
        camera=camera,
        width=width,
        height=height,
        gripper_qpos=gripper_qpos,
        component_ids=(0, 1, 2, 3),
    )
    if buffers is None:
        return np.zeros(scene.shape, dtype=bool), {
            "applied": False,
            "reason": "gripper mesh is outside the wrist camera",
            "projected_gripper_pixel_count": 0,
        }
    model_depth = np.asarray(buffers["depth"], dtype=np.float32)
    self_mask = _nearby_depth_match_mask(
        scene,
        model_depth,
        radius_px=WRIST_SELF_UV_RADIUS_PX,
        tolerance_m=WRIST_SELF_DEPTH_TOLERANCE_M,
    )
    return self_mask, {
        "applied": True,
        "source": "submission_local_frozen_dynamic_gripper_visual_mesh",
        "projected_gripper_pixel_count": int(np.isfinite(model_depth).sum()),
        "matched_scene_pixel_count": int(self_mask.sum()),
        "uv_radius_px": int(WRIST_SELF_UV_RADIUS_PX),
        "depth_tolerance_m": float(WRIST_SELF_DEPTH_TOLERANCE_M),
        "component_ids": [0, 1, 2, 3],
    }


def _points_inside_opening(
    points_eef: np.ndarray,
    lut: Dict[str, np.ndarray | float],
) -> np.ndarray:
    return gap_opening_strict_mask_from_lut(points_eef, lut)


def measure_wrist_opening_depth_evidence(
    *,
    depth_linear: np.ndarray,
    camera: Dict[str, Any],
    eef_pose: Dict[str, Any],
    gripper_qpos: Optional[Sequence[float]],
) -> Dict[str, Any]:
    """Measure a continuous observed surface spanning the finger opening.

    This uses only evaluator RGB-D geometry and proprioceptive poses.  It does
    not identify an object or observe contact / assisted-grasp simulator state.
    """
    depth = np.asarray(depth_linear, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        return {
            "ok": False,
            "error": f"wrist depth_linear must be HxW, got {depth.shape}",
            "bilateral_depth_span_observed": False,
            "occlusion_limited_bilateral_blockage_observed": False,
        }
    height, width = depth.shape
    try:
        camera_pos, camera_quat, fx, fy, cx, cy = _camera_values(
            camera,
            width,
            height,
        )
        valid = np.isfinite(depth) & (depth > 0.02)
        vv, uu = np.nonzero(valid)
        if len(vv) == 0:
            return {
                "ok": True,
                "bilateral_depth_span_observed": False,
                "occlusion_limited_bilateral_blockage_observed": False,
                "inside_point_count": 0,
                "connected_component_count": 0,
                "reason": "no_valid_wrist_depth",
            }

        distances = depth[vv, uu].astype(np.float64)
        points_camera = np.column_stack(
            [
                (uu.astype(np.float64) + 0.5 - cx) / fx * distances,
                -(vv.astype(np.float64) + 0.5 - cy) / fy * distances,
                -distances,
            ]
        )
        camera_rotation = quat_to_mat_xyzw(camera_quat)
        points_world = points_camera @ camera_rotation.T + camera_pos
        eef_pos = np.asarray(eef_pose["pos"], dtype=np.float64).reshape(3)
        eef_rotation = quat_to_mat_xyzw(eef_pose["quat"])
        points_eef = (points_world - eef_pos) @ eef_rotation

        lut = _dynamic_opening_lut(gripper_qpos)
        inside = _points_inside_opening(points_eef, lut)
        inside_count = int(np.count_nonzero(inside))
        if inside_count == 0:
            return {
                "ok": True,
                "bilateral_depth_span_observed": False,
                "occlusion_limited_bilateral_blockage_observed": False,
                "inside_point_count": 0,
                "connected_component_count": 0,
                "reason": "opening_volume_empty",
            }

        points_inside = points_eef[inside]
        pixels_v = vv[inside]
        pixels_u = uu[inside]
        z = points_inside[:, 2]
        z_samples = np.asarray(lut["z"], dtype=np.float64)
        y_lo = np.interp(
            z,
            z_samples,
            np.asarray(lut["y_inner_lo"], dtype=np.float64),
        )
        y_hi = np.interp(
            z,
            z_samples,
            np.asarray(lut["y_inner_hi"], dtype=np.float64),
        )
        normalized_y = (points_inside[:, 1] - y_lo) / np.maximum(
            y_hi - y_lo,
            1e-9,
        )

        opening_mask = np.zeros((height, width), dtype=np.uint8)
        opening_mask[pixels_v, pixels_u] = 1
        component_count, labels, _stats, _centroids = (
            cv2.connectedComponentsWithStats(opening_mask, connectivity=8)
        )
        image_scale = max(
            0.05,
            float(height * width) / float(WRIST_OPENING_REFERENCE_PIXELS),
        )
        min_component_points = max(
            32,
            int(round(WRIST_OPENING_MIN_COMPONENT_POINTS * image_scale)),
        )
        min_side_points = max(
            8,
            int(round(WRIST_OPENING_MIN_SIDE_POINTS * image_scale)),
        )
        min_center_points = max(
            16,
            int(round(WRIST_OPENING_MIN_CENTER_POINTS * image_scale)),
        )

        components: list[Dict[str, Any]] = []
        point_labels = labels[pixels_v, pixels_u]
        for label in range(1, int(component_count)):
            selected = point_labels == label
            component_y = normalized_y[selected]
            point_count = int(len(component_y))
            if point_count == 0:
                continue
            side_0_count = int(
                np.count_nonzero(
                    component_y < WRIST_OPENING_SIDE_BAND_FRACTION
                )
            )
            center_count = int(
                np.count_nonzero(
                    (component_y >= WRIST_OPENING_CENTER_LO_FRACTION)
                    & (component_y <= WRIST_OPENING_CENTER_HI_FRACTION)
                )
            )
            side_1_count = int(
                np.count_nonzero(
                    component_y > 1.0 - WRIST_OPENING_SIDE_BAND_FRACTION
                )
            )
            q01, q05, q95, q99 = np.quantile(
                component_y,
                [0.01, 0.05, 0.95, 0.99],
            )
            spans = bool(
                point_count >= min_component_points
                and side_0_count >= min_side_points
                and center_count >= min_center_points
                and side_1_count >= min_side_points
                and float(q05) <= WRIST_OPENING_SIDE_BAND_FRACTION
                and float(q95) >= 1.0 - WRIST_OPENING_SIDE_BAND_FRACTION
            )
            candidate = {
                "label": int(label),
                "point_count": point_count,
                "side_0_point_count": side_0_count,
                "center_point_count": center_count,
                "side_1_point_count": side_1_count,
                "normalized_lateral_q01": float(q01),
                "normalized_lateral_q05": float(q05),
                "normalized_lateral_q95": float(q95),
                "normalized_lateral_q99": float(q99),
                "spans_opening": spans,
            }
            components.append(candidate)

        spanning_components = [
            component
            for component in components
            if bool(component["spans_opening"])
        ]
        evidence = bool(spanning_components)
        best_pool = spanning_components if spanning_components else components
        best = (
            max(best_pool, key=lambda component: int(component["point_count"]))
            if best_pool
            else None
        )

        # From an oblique wrist view, the grasped surface can be visible from
        # the center to one finger while the other endpoint is hidden behind
        # the opposing finger. Preserve this as a separate, weaker signal: the
        # caller must confirm it across multiple frames and proprioceptive
        # close actions before treating it as bilateral blockage.
        occlusion_candidate: Optional[Dict[str, Any]] = None
        for bridge in sorted(
            components,
            key=lambda component: int(component["point_count"]),
            reverse=True,
        ):
            bridge_fraction = float(bridge["point_count"]) / max(
                1.0,
                float(inside_count),
            )
            if (
                int(bridge["point_count"]) < min_component_points
                or int(bridge["center_point_count"]) < min_center_points
                or bridge_fraction
                < WRIST_OPENING_OCCLUSION_MIN_DOMINANT_FRACTION
            ):
                continue

            visible_side: Optional[int] = None
            if (
                int(bridge["side_0_point_count"]) >= min_side_points
                and int(bridge["side_1_point_count"]) < min_side_points
            ):
                visible_side = 0
            elif (
                int(bridge["side_1_point_count"]) >= min_side_points
                and int(bridge["side_0_point_count"]) < min_side_points
            ):
                visible_side = 1
            if visible_side is None:
                continue

            opposing_side = 1 - visible_side
            boundary_matches = []
            for boundary in components:
                if boundary is bridge:
                    continue
                boundary_count = int(
                    boundary[f"side_{opposing_side}_point_count"]
                )
                near_boundary = (
                    float(boundary["normalized_lateral_q05"])
                    >= 1.0 - WRIST_OPENING_BOUNDARY_BAND_FRACTION
                    if opposing_side == 1
                    else float(boundary["normalized_lateral_q95"])
                    <= WRIST_OPENING_BOUNDARY_BAND_FRACTION
                )
                if (
                    int(boundary["point_count"]) >= min_component_points
                    and boundary_count >= min_component_points
                    and near_boundary
                ):
                    boundary_matches.append(boundary)
            if not boundary_matches:
                continue

            opposing_boundary = max(
                boundary_matches,
                key=lambda component: int(component["point_count"]),
            )
            occlusion_candidate = {
                "visible_side": f"side_{visible_side}",
                "opposing_side": f"side_{opposing_side}",
                "bridge_component": bridge,
                "opposing_boundary_component": opposing_boundary,
                "bridge_inside_fraction": bridge_fraction,
            }
            break

        values = np.asarray(lut["gripper_qpos_m"], dtype=np.float64)
        return {
            "ok": True,
            "bilateral_depth_span_observed": evidence,
            "occlusion_limited_bilateral_blockage_observed": bool(
                occlusion_candidate
            ),
            "occlusion_limited_candidate": occlusion_candidate,
            "inside_point_count": inside_count,
            "connected_component_count": max(0, int(component_count) - 1),
            "spanning_component": best,
            "thresholds": {
                "min_component_points": min_component_points,
                "min_side_points": min_side_points,
                "min_center_points": min_center_points,
                "side_band_fraction": WRIST_OPENING_SIDE_BAND_FRACTION,
                "boundary_band_fraction": (
                    WRIST_OPENING_BOUNDARY_BAND_FRACTION
                ),
                "occlusion_min_dominant_fraction": (
                    WRIST_OPENING_OCCLUSION_MIN_DOMINANT_FRACTION
                ),
                "center_band_fraction": [
                    WRIST_OPENING_CENTER_LO_FRACTION,
                    WRIST_OPENING_CENTER_HI_FRACTION,
                ],
            },
            "gripper_qpos_m": values.astype(float).tolist(),
            "depth_source": "live_evaluator_depth_linear",
            "camera_pose_source": "evaluator_camera_relative_pose",
            "eef_pose_source": "evaluator_proprioception",
            "opening_geometry_source": "submission_local_frozen_gripper_lut",
            "segmentation_used": False,
            "robot_mask_used": False,
            "simulator_state_used": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "bilateral_depth_span_observed": False,
            "occlusion_limited_bilateral_blockage_observed": False,
        }


def render_wrist_grasp_volume_overlay(
    *,
    rgb_path: str,
    depth_linear: np.ndarray,
    camera: Dict[str, Any],
    eef_pose: Dict[str, Any],
    gripper_qpos: Optional[Sequence[float]],
    gripper_qvel: Optional[Sequence[float]] = None,
    output_path: str,
    mask_path: Optional[str] = None,
    alpha: float = 0.58,
) -> Dict[str, Any]:
    """Highlight visible depth points inside the local two-finger opening."""
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "ok": False,
            "error": f"failed to read RGB image: {rgb_path}",
            "path": output_path,
        }
    height, width = image.shape[:2]
    depth = _scene_depth(depth_linear, width, height)
    if depth is None:
        return {
            "ok": False,
            "error": "wrist depth_linear shape does not match RGB",
            "path": output_path,
        }
    try:
        camera_pos, camera_quat, fx, fy, cx, cy = _camera_values(
            camera,
            width,
            height,
        )
        valid = np.isfinite(depth) & (depth > 0.02)
        vv, uu = np.nonzero(valid)
        distances = depth[vv, uu].astype(np.float64)
        points_camera = np.column_stack(
            [
                (uu.astype(np.float64) + 0.5 - cx) / fx * distances,
                -(vv.astype(np.float64) + 0.5 - cy) / fy * distances,
                -distances,
            ]
        )
        camera_rotation = quat_to_mat_xyzw(camera_quat)
        points_world = points_camera @ camera_rotation.T + camera_pos
        eef_pos = np.asarray(eef_pose["pos"], dtype=np.float64).reshape(3)
        eef_rotation = quat_to_mat_xyzw(eef_pose["quat"])
        points_eef = (points_world - eef_pos) @ eef_rotation
        lut = _dynamic_opening_lut(gripper_qpos)
        inside = _points_inside_opening(points_eef, lut)
        candidate_mask = np.zeros((height, width), dtype=bool)
        candidate_mask[vv[inside], uu[inside]] = True
        qvel = np.asarray(
            [] if gripper_qvel is None else gripper_qvel,
            dtype=np.float64,
        ).reshape(-1)
        max_abs_qvel = (
            None
            if qvel.size == 0 or not np.all(np.isfinite(qvel))
            else float(np.max(np.abs(qvel)))
        )
        self_mask, self_mask_meta = _gripper_self_depth_mask(
            scene_depth=depth,
            camera=camera,
            eef_pose=eef_pose,
            gripper_qpos=gripper_qpos,
        )
        red_mask = candidate_mask & ~self_mask
        overlay_alpha = np.zeros((height, width), dtype=np.float32)
        overlay_alpha[red_mask] = float(np.clip(alpha, 0.0, 1.0))
        output = _composite(image, _WRIST_RED_BGR, overlay_alpha)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if not cv2.imwrite(output_path, output):
            raise RuntimeError(f"failed to write {output_path}")
        if mask_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(mask_path)),
                exist_ok=True,
            )
            np.save(mask_path, red_mask)
        values = np.asarray(lut["gripper_qpos_m"], dtype=np.float64)
        return {
            "ok": True,
            "path": output_path,
            "mask_path": mask_path,
            "build": "official_v2_local_wrist_depth_opening_volume_v3",
            "color": "red",
            "definition": (
                "visible depth points strictly inside the current two-finger "
                "opening, excluding depth-matched complete gripper surfaces"
            ),
            "red_condition": (
                "strict_dynamic_opening_membership AND NOT "
                "depth_coincident_complete_gripper_surface"
            ),
            "red_pixel_count": int(red_mask.sum()),
            "red_candidate_pixel_count": int(candidate_mask.sum()),
            "self_rejected_pixel_count": int(
                np.count_nonzero(candidate_mask & self_mask)
            ),
            "valid_depth_pixel_count": int(valid.sum()),
            "gripper_qpos_m": values.astype(float).tolist(),
            "gripper_qvel_m_s": (
                None if qvel.size == 0 else qvel.astype(float).tolist()
            ),
            "max_abs_gripper_qvel_m_s": max_abs_qvel,
            "gripper_motion_safe": True,
            "overlay_suppressed": False,
            "overlay_suppressed_reason": None,
            "motion_suppression_used": False,
            "depth_source": "frozen_evaluator_depth_linear",
            "camera_pose_source": "evaluator_camera_relative_pose",
            "eef_pose_source": "same_frozen_evaluator_observation_proprioception",
            "observation_alignment_required": "single_evaluator_observation",
            "opening_geometry_source": "submission_local_frozen_gripper_lut",
            "segmentation_used": False,
            "robot_mask_used": bool(self_mask_meta.get("applied")),
            "robot_mask_source": self_mask_meta.get("source"),
            "robot_self_mask": self_mask_meta,
            "simulator_geometry_used": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "path": output_path,
            "mask_path": mask_path,
        }


__all__ = [
    "measure_wrist_opening_depth_evidence",
    "render_head_path_overlay",
    "render_head_path_overlay_frame",
    "render_plan_gripper_overlay",
    "render_wrist_grasp_volume_overlay",
]
