"""Submission-local replica of the v2 head-view base path HUD.

The geometry and styling are frozen from the current v2 implementation.
Runtime inputs are limited to evaluator-provided RGB, metric depth, camera
calibration / pose, and the observation-backed local base pose.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .grasp_geometry_local import quat_to_mat_xyzw


PATH_LENGTH_M = 3.0
PATH_WIDTH_M = 0.8
PATH_SIDE_INSET_M = 0.0874765
PATH_Y_CENTER_M = -5.000000000143778e-07
PATH_HEIGHT_OFFSET_M = 0.035
PATH_SIDE_MARGIN_M = 0.0
PATH_FADE_START_M = 2.15
DEPTH_OCCLUDE_EPS_M = 0.003
# 底盘直行提示只看蓝线走廊的前 2m，不看整条 3m HUD。
CHASSIS_FORWARD_HINT_M = 2.0
CHASSIS_FORWARD_2M_CLEAR = (
    "No object on the chassis-forward path within 2m"
)
# 场景点低于此机体系高度视为地板。
CHASSIS_SELF_GROUND_Z_M = 0.05
# 底盘占用盒：略超出前碰撞沿，盖住轮罩/保险杠，但不吃前方真实物体。
CHASSIS_SELF_X_MIN_M = -0.25
CHASSIS_SELF_X_PAST_FRONT_M = 0.10
CHASSIS_SELF_HALF_WIDTH_M = 0.42
CHASSIS_SELF_Z_MAX_M = 0.55
# 贴前沿的矮体积：车头自遮挡常出现 0.00m，不是前方物体。
CHASSIS_SELF_NEAR_FRONT_M = 0.12
CHASSIS_SELF_NEAR_Z_MAX_M = 0.38
# 持物/长刀具：从夹爪/EEF 种子在「比蓝线近」的像素上深度连续生长。
# 不用地板/底盘/大臂当种子，避免顺着地面或墙把走廊吃掉。
# 3D 距夹爪超过此值就停，避免空爪子贴着垃圾桶时把桶也连进去。
CHASSIS_HELD_DEPTH_CONT_M = 0.04
CHASSIS_HELD_GRIPPER_SEED_M = 0.16
CHASSIS_HELD_MAX_M = 0.28
# 手部近距：0.16m 球内先抠掉持物，剩下的 3D 分团每团一个最近点。
# 持物 = depth 3D 生长 ∪ SAM（扣相机）。SAM 近处够完整时，只留贴着
# SAM 的 3D 沿，把外侧第二物体剥开；斧刃/宽盒 SAM 盖不住则整团 3D。
# 夹爪/指尖网格当本体扣掉。无点时不写提示。
EEF_NEAR_INNER_M = 0.05
EEF_NEAR_SPHERE_M = 0.16
EEF_NEAR_SAM2_POS_POINTS = 8
EEF_NEAR_SAM2_BOX_PAD_PX = 40
EEF_NEAR_SAM2_NEAR_M = 0.10
EEF_NEAR_SAM2_FAR_M = 0.13
EEF_NEAR_SAM2_SEED_COV = 0.70
EEF_NEAR_SAM2_NEAR_COV = 0.55
EEF_NEAR_SAM2_FAR_COV = 0.60
# 种子很多说明持物已经占满开口，不再剥，避免斧刃/盒沿被当成障碍。
EEF_NEAR_SAM2_MAX_SEED = 60
EEF_NEAR_SAM2_PEEL_X_M = 0.08
# 闭运算只填 SAM/3D 团内部麻点，不外扩。
EEF_NEAR_SAM2_CLOSE_PX = 9
EEF_NEAR_GAP_PALM_Z_M = -0.02
EEF_NEAR_GAP_FACE_PEEL_M = 0.003
EEF_NEAR_GAP_MIN_POINTS = 8
# 全开爪持物：沿开口 x 外扩、只取指尖附近的近点。
EEF_NEAR_SLIT_X_PAD_M = 0.012
EEF_NEAR_SLIT_Z_LO_M = -0.008
EEF_NEAR_SLIT_Z_HI_M = 0.025
# 指前近距透视：椅背会填满这段；罐子会挡住，1m 外背景不算。
EEF_NEAR_SLIT_FAR_Z_LO_M = 0.04
EEF_NEAR_DEPTH_CONT_M = 0.04
# 开口 LUT 经常吃不到盆壁。夹爪中心小盒子当核心种子，柜门/空爪进不去。
EEF_NEAR_CORE_SEED_DIST_M = 0.04
EEF_NEAR_CORE_SEED_X_M = 0.04
EEF_NEAR_CORE_SEED_Y_M = 0.035
EEF_NEAR_CORE_SEED_Z_LO_M = -0.01
EEF_NEAR_CORE_SEED_Z_HI_M = 0.035
EEF_NEAR_CORE_SEED_MIN = 8
# 开口容器近壁/远壁中间是空腔，4cm 体会拆成两团；6cm 仍不会把空爪误连。
EEF_NEAR_CLUSTER_VOXEL_M = 0.06
# _grow_held_by_3d 的体素是 radius 的一半；0.12 → 6cm。
EEF_NEAR_HELD_GROW_RADIUS_M = 0.12
EEF_NEAR_GROW_MAX_M = 0.20
# 开爪无持物种子时只扣手指/夹爪本体，不扣 16cm 大球。
EEF_NEAR_LINK_EXCLUDE_M = 0.03
# 腕 RealSense 在 gripper +X / EEF -X，离 EEF 约 8cm；3cm 球盖不住壳体。
EEF_NEAR_CAMERA_EXCLUDE_M = 0.06
# 相机网格 + 手背/指根靠相机一侧。z 接近 0 的指前接触（椅背/柜门）不扣。
EEF_NEAR_CAMERA_BOX_EEF = (
    (-0.085, -0.015),
    (-0.085, 0.025),
    (-0.115, -0.050),
)
# 0.16m 球会扫到腕后/指后壳体。指前接触（z≥-0.035）不进这个盒子。
EEF_NEAR_WRIST_BOX_EEF = (
    (-0.10, 0.032),
    (-0.08, 0.08),
    (-0.16, -0.035),
)
# 指前 8mm；指后 20mm，避免把夹爪黑壳标成障碍。
EEF_NEAR_SELF_MESH_M = 0.008
EEF_NEAR_SELF_MESH_REAR_M = 0.020
EEF_NEAR_SELF_MESH_FINGER_Z = -0.03
# URDF left/right_realsense 在 gripper 系同一偏移，变到 EEF 系。
EEF_NEAR_CAMERA_T_EEF = np.array([-0.05051, 0.0028934, -0.0651317], dtype=np.float64)
EEF_NEAR_MIN_BLOB = 32
EEF_NEAR_CLEAR = ""
DISTANCE_TICK_STEP_M = 0.5
LATERAL_REFERENCE_OFFSET_M = 0.2
LATERAL_REFERENCE_OUTER_OFFSET_M = 0.4
NEAR_EDGE_SIDE_MARK_LENGTH_M = 0.09
NEAR_EDGE_CENTER_MARK_LENGTH_M = 0.12
NEAR_EDGE_LABEL_FORWARD_OFFSET_M = 0.18
NEAR_EDGE_OUTER_LABEL_OUTSET_M = 0.12
LABEL_COLOR_RGB = (12, 92, 205)
LABEL_STROKE_WIDTH_PX = 0
# Maximum +X extent of the R1Pro base collision footprint in base_link:
# front wheel center x (0.168970 m) + collision radius (0.07045779099844562 m).
BASE_FRONT_OFFSET_M = 0.23942779099844563
PATH_START_X_M = BASE_FRONT_OFFSET_M
PATH_END_X_M = PATH_START_X_M + PATH_LENGTH_M
_LABEL_FONT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "DejaVuSans-Bold.ttf",
)


def _fade_alpha_from_forward_x(x_m: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x_m, dtype=np.float64)
    denominator = max(PATH_LENGTH_M - PATH_FADE_START_M, 1e-6)
    distance_from_front = x - BASE_FRONT_OFFSET_M
    t = np.clip(
        (distance_from_front - PATH_FADE_START_M) / denominator,
        0.0,
        1.0,
    )
    smooth = t * t * (3.0 - 2.0 * t)
    return np.clip(255.0 * (1.0 - smooth), 0.0, 255.0)


def _base_local_x_from_front_distance(distance_m: float) -> float:
    return BASE_FRONT_OFFSET_M + float(distance_m)


def _near_edge_lateral_mark_specs() -> Tuple[
    Tuple[float, float, Optional[str]], ...
]:
    return (
        (-LATERAL_REFERENCE_OUTER_OFFSET_M, NEAR_EDGE_SIDE_MARK_LENGTH_M, "0.4m"),
        (-LATERAL_REFERENCE_OFFSET_M, NEAR_EDGE_SIDE_MARK_LENGTH_M, "0.2m"),
        (0.0, NEAR_EDGE_CENTER_MARK_LENGTH_M, None),
        (LATERAL_REFERENCE_OFFSET_M, NEAR_EDGE_SIDE_MARK_LENGTH_M, "0.2m"),
        (LATERAL_REFERENCE_OUTER_OFFSET_M, NEAR_EDGE_SIDE_MARK_LENGTH_M, "0.4m"),
    )


def _base_pose_values(robot: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    base = dict(robot.get("base_pose") or robot)
    position = np.asarray(
        base.get(
            "pos",
            [
                float(base.get("x", 0.0)),
                float(base.get("y", 0.0)),
                float(base.get("z", 0.0)),
            ],
        ),
        dtype=np.float64,
    ).reshape(3)
    if base.get("yaw_deg") is not None:
        yaw_deg = float(base["yaw_deg"])
    elif base.get("quat") is not None:
        rotation = quat_to_mat_xyzw(
            np.asarray(base["quat"], dtype=np.float64).reshape(4)
        )
        yaw_deg = math.degrees(
            math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        )
    else:
        yaw_deg = math.degrees(float(base.get("yaw", 0.0)))
    return position, yaw_deg


def _robot_local_points_to_world(
    robot: Dict[str, Any],
    points_local: np.ndarray,
) -> np.ndarray:
    base_position, yaw_deg = _base_pose_values(robot)
    yaw = math.radians(yaw_deg)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    local = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
    world = np.empty_like(local)
    world[:, 0] = (
        base_position[0] + cosine * local[:, 0] - sine * local[:, 1]
    )
    world[:, 1] = (
        base_position[1] + sine * local[:, 0] + cosine * local[:, 1]
    )
    world[:, 2] = base_position[2] + local[:, 2]
    return world


def _robot_local_path_corners(
    robot: Dict[str, Any],
    *,
    height_offset_m: float,
) -> np.ndarray:
    half_width = 0.5 * PATH_WIDTH_M
    y0 = PATH_Y_CENTER_M - half_width
    y1 = PATH_Y_CENTER_M + half_width
    local = np.array(
        [
            [PATH_START_X_M, y0, float(height_offset_m)],
            [PATH_END_X_M, y0, float(height_offset_m)],
            [PATH_END_X_M, y1, float(height_offset_m)],
            [PATH_START_X_M, y1, float(height_offset_m)],
        ],
        dtype=np.float64,
    )
    return _robot_local_points_to_world(robot, local)


def _camera_projection_values(
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
            float(camera.get("focal_length", 17.0))
            / float(camera.get("horizontal_aperture", 40.0))
            * int(width)
        )
    fy = float(camera.get("fy", fx))
    cx = float(camera.get("cx", float(width) * 0.5))
    cy = float(camera.get("cy", float(height) * 0.5))
    return position, quaternion, fx, fy, cx, cy


def _project_points(
    points_world: np.ndarray,
    camera: Dict[str, Any],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    camera_position, camera_quaternion, fx, fy, cx, cy = (
        _camera_projection_values(camera, width, height)
    )
    rotation = quat_to_mat_xyzw(camera_quaternion)
    camera_points = (
        rotation.T
        @ (
            np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
            - camera_position.reshape(1, 3)
        ).T
    ).T
    z_camera = camera_points[:, 2]
    denominator = np.where(z_camera < -1e-9, -z_camera, np.nan)
    u = fx * camera_points[:, 0] / denominator + cx
    v = cy - fy * camera_points[:, 1] / denominator
    return np.stack([u, v], axis=1), z_camera


def _rasterize_triangle_zbuffer(
    triangle_uv: np.ndarray,
    triangle_depth: np.ndarray,
    color_rgb: Tuple[int, int, int],
    color_buffer: np.ndarray,
    depth_buffer: np.ndarray,
    mask_buffer: np.ndarray,
    triangle_alpha: Optional[np.ndarray] = None,
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

    u0, v0 = triangle_uv[0]
    u1, v1 = triangle_uv[1]
    u2, v2 = triangle_uv[2]
    denominator = (
        (v1 - v2) * (u0 - u2)
        + (u2 - u1) * (v0 - v2)
    )
    if abs(float(denominator)) < 1e-12:
        return

    yy, xx = np.mgrid[min_v : max_v + 1, min_u : max_u + 1]
    pixel_u = xx.astype(np.float64) + 0.5
    pixel_v = yy.astype(np.float64) + 0.5
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
    )
    if not np.any(inside):
        return

    inverse_depth = (
        weight_0 / float(triangle_depth[0])
        + weight_1 / float(triangle_depth[1])
        + weight_2 / float(triangle_depth[2])
    )
    valid_depth = inverse_depth > 1e-9
    pixel_depth = np.full_like(inverse_depth, np.inf, dtype=np.float64)
    pixel_depth[valid_depth] = 1.0 / inverse_depth[valid_depth]

    depth_patch = depth_buffer[min_v : max_v + 1, min_u : max_u + 1]
    closer = (
        inside
        & np.isfinite(pixel_depth)
        & (pixel_depth.astype(np.float32) < depth_patch)
    )
    if not np.any(closer):
        return
    if triangle_alpha is None:
        alpha_pixel = np.full_like(pixel_depth, 255.0, dtype=np.float64)
    else:
        alpha = np.asarray(triangle_alpha, dtype=np.float64).reshape(3)
        alpha_pixel = np.zeros_like(pixel_depth, dtype=np.float64)
        alpha_pixel[valid_depth] = (
            weight_0[valid_depth]
            * alpha[0]
            / float(triangle_depth[0])
            + weight_1[valid_depth]
            * alpha[1]
            / float(triangle_depth[1])
            + weight_2[valid_depth]
            * alpha[2]
            / float(triangle_depth[2])
        ) / inverse_depth[valid_depth]
        closer &= alpha_pixel > 1.0
        if not np.any(closer):
            return

    depth_patch[closer] = pixel_depth.astype(np.float32)[closer]
    mask_patch = mask_buffer[min_v : max_v + 1, min_u : max_u + 1]
    color_patch = color_buffer[min_v : max_v + 1, min_u : max_u + 1]
    mask_patch[closer] = np.clip(
        alpha_pixel[closer],
        0.0,
        255.0,
    ).astype(np.uint8)
    color_patch[closer] = color_rgb


def _path_fill_buffers(
    robot: Dict[str, Any],
    *,
    height_offset_m: float,
    camera: Dict[str, Any],
    width: int,
    height: int,
    color_rgb: Tuple[int, int, int] = (42, 145, 255),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    mask_buffer = np.zeros((height, width), dtype=np.uint8)
    half_width = 0.5 * PATH_WIDTH_M
    y0 = PATH_Y_CENTER_M - half_width
    y1 = PATH_Y_CENTER_M + half_width
    z = float(height_offset_m)
    segment_xs = [
        PATH_START_X_M,
        _base_local_x_from_front_distance(PATH_FADE_START_M),
        PATH_END_X_M,
    ]
    for start_x, end_x in zip(segment_xs[:-1], segment_xs[1:]):
        local = np.array(
            [
                [start_x, y0, z],
                [end_x, y0, z],
                [end_x, y1, z],
                [start_x, y1, z],
            ],
            dtype=np.float64,
        )
        world = _robot_local_points_to_world(robot, local)
        uv, z_camera = _project_points(world, camera, width, height)
        alpha = _fade_alpha_from_forward_x(local[:, 0])
        for triangle in ((0, 1, 2), (0, 2, 3)):
            indices = list(triangle)
            triangle_uv = uv[indices]
            triangle_z = z_camera[indices]
            if not (
                np.all(np.isfinite(triangle_uv))
                and np.all(triangle_z < -1e-6)
            ):
                continue
            _rasterize_triangle_zbuffer(
                triangle_uv,
                (-triangle_z).astype(np.float32),
                color_rgb,
                color_buffer,
                depth_buffer,
                mask_buffer,
                triangle_alpha=alpha[indices],
            )
    corners_world = _robot_local_path_corners(
        robot,
        height_offset_m=height_offset_m,
    )
    corners_uv, _ = _project_points(corners_world, camera, width, height)
    return color_buffer, mask_buffer, depth_buffer, corners_uv


def _draw_depth_disk(
    center_u: float,
    center_v: float,
    depth: float,
    radius_px: float,
    color_rgb: Tuple[int, int, int],
    color_buffer: np.ndarray,
    depth_buffer: np.ndarray,
    mask_buffer: np.ndarray,
    alpha: int = 255,
) -> None:
    height, width = depth_buffer.shape
    radius = max(1.0, float(radius_px))
    min_u = max(0, int(np.floor(center_u - radius)))
    max_u = min(width - 1, int(np.ceil(center_u + radius)))
    min_v = max(0, int(np.floor(center_v - radius)))
    max_v = min(height - 1, int(np.ceil(center_v + radius)))
    if min_u > max_u or min_v > max_v:
        return
    yy, xx = np.mgrid[min_v : max_v + 1, min_u : max_u + 1]
    inside = (
        (xx.astype(np.float64) - float(center_u)) ** 2
        + (yy.astype(np.float64) - float(center_v)) ** 2
        <= radius**2
    )
    depth_patch = depth_buffer[min_v : max_v + 1, min_u : max_u + 1]
    closer = inside & (float(depth) < depth_patch)
    if not np.any(closer):
        return
    alpha_value = int(np.clip(alpha, 0, 255))
    if alpha_value <= 0:
        return
    depth_patch[closer] = float(depth)
    mask_patch = mask_buffer[min_v : max_v + 1, min_u : max_u + 1]
    color_patch = color_buffer[min_v : max_v + 1, min_u : max_u + 1]
    mask_patch[closer] = alpha_value
    color_patch[closer] = color_rgb


def _draw_local_polyline(
    robot: Dict[str, Any],
    points_xy: np.ndarray,
    *,
    height_offset_m: float,
    camera: Dict[str, Any],
    width: int,
    height: int,
    color_rgb: Tuple[int, int, int],
    radius_px: float,
    step_m: float,
    color_buffer: np.ndarray,
    depth_buffer: np.ndarray,
    mask_buffer: np.ndarray,
) -> None:
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points_xy) < 2:
        return
    for start, end in zip(points_xy[:-1], points_xy[1:]):
        segment_length = float(np.linalg.norm(end - start))
        sample_count = max(
            2,
            int(
                np.ceil(
                    segment_length / max(float(step_m), 1e-3)
                )
            )
            + 1,
        )
        t = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
        xy = (
            start.reshape(1, 2) * (1.0 - t[:, None])
            + end.reshape(1, 2) * t[:, None]
        )
        local = np.column_stack(
            [
                xy[:, 0],
                xy[:, 1],
                np.full(
                    sample_count,
                    float(height_offset_m),
                    dtype=np.float64,
                ),
            ]
        )
        world = _robot_local_points_to_world(robot, local)
        uv, z_camera = _project_points(world, camera, width, height)
        alphas = _fade_alpha_from_forward_x(xy[:, 0])
        for (u, v), z, alpha in zip(uv, z_camera, alphas):
            if not (
                np.isfinite(u)
                and np.isfinite(v)
                and np.isfinite(z)
                and z < -1e-6
            ):
                continue
            if u < -10 or u > width + 10 or v < -10 or v > height + 10:
                continue
            if alpha <= 1.0:
                continue
            _draw_depth_disk(
                float(u),
                float(v),
                float(-z),
                radius_px,
                color_rgb,
                color_buffer,
                depth_buffer,
                mask_buffer,
                alpha=int(round(float(alpha))),
            )


def _render_label_texture(
    text: str,
    color_rgb: Tuple[int, int, int],
) -> np.ndarray:
    texture_width, texture_height = 192, 72
    image = Image.new(
        "RGBA",
        (texture_width, texture_height),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(_LABEL_FONT_PATH, 42)
    except Exception:
        font = ImageFont.load_default()
    bounds = draw.textbbox(
        (0, 0),
        text,
        font=font,
        stroke_width=LABEL_STROKE_WIDTH_PX,
    )
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = (texture_width - text_width) * 0.5 - bounds[0]
    y = (texture_height - text_height) * 0.5 - bounds[1] - 1
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(*color_rgb, 255),
        stroke_width=LABEL_STROKE_WIDTH_PX,
    )
    return np.asarray(image, dtype=np.uint8)


def _rasterize_textured_triangle(
    triangle_uv: np.ndarray,
    triangle_depth: np.ndarray,
    triangle_texture_uv: np.ndarray,
    triangle_alpha: np.ndarray,
    texture_rgba: np.ndarray,
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

    u0, v0 = triangle_uv[0]
    u1, v1 = triangle_uv[1]
    u2, v2 = triangle_uv[2]
    denominator = (
        (v1 - v2) * (u0 - u2)
        + (u2 - u1) * (v0 - v2)
    )
    if abs(float(denominator)) < 1e-12:
        return

    yy, xx = np.mgrid[min_v : max_v + 1, min_u : max_u + 1]
    pixel_u = xx.astype(np.float64) + 0.5
    pixel_v = yy.astype(np.float64) + 0.5
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
    )
    if not np.any(inside):
        return

    inverse_depth = (
        weight_0 / float(triangle_depth[0])
        + weight_1 / float(triangle_depth[1])
        + weight_2 / float(triangle_depth[2])
    )
    valid_depth = inverse_depth > 1e-9
    pixel_depth = np.full_like(inverse_depth, np.inf, dtype=np.float64)
    pixel_depth[valid_depth] = 1.0 / inverse_depth[valid_depth]
    depth_patch = depth_buffer[min_v : max_v + 1, min_u : max_u + 1]
    closer = (
        inside
        & np.isfinite(pixel_depth)
        & (pixel_depth.astype(np.float32) < depth_patch)
    )
    if not np.any(closer):
        return

    texture_coordinates = np.asarray(
        triangle_texture_uv,
        dtype=np.float64,
    ).reshape(3, 2)
    alpha_vertices = np.asarray(
        triangle_alpha,
        dtype=np.float64,
    ).reshape(3)
    texture_u = np.zeros_like(pixel_depth, dtype=np.float64)
    texture_v = np.zeros_like(pixel_depth, dtype=np.float64)
    ground_alpha = np.zeros_like(pixel_depth, dtype=np.float64)
    texture_u[valid_depth] = (
        weight_0[valid_depth]
        * texture_coordinates[0, 0]
        / float(triangle_depth[0])
        + weight_1[valid_depth]
        * texture_coordinates[1, 0]
        / float(triangle_depth[1])
        + weight_2[valid_depth]
        * texture_coordinates[2, 0]
        / float(triangle_depth[2])
    ) / inverse_depth[valid_depth]
    texture_v[valid_depth] = (
        weight_0[valid_depth]
        * texture_coordinates[0, 1]
        / float(triangle_depth[0])
        + weight_1[valid_depth]
        * texture_coordinates[1, 1]
        / float(triangle_depth[1])
        + weight_2[valid_depth]
        * texture_coordinates[2, 1]
        / float(triangle_depth[2])
    ) / inverse_depth[valid_depth]
    ground_alpha[valid_depth] = (
        weight_0[valid_depth]
        * alpha_vertices[0]
        / float(triangle_depth[0])
        + weight_1[valid_depth]
        * alpha_vertices[1]
        / float(triangle_depth[1])
        + weight_2[valid_depth]
        * alpha_vertices[2]
        / float(triangle_depth[2])
    ) / inverse_depth[valid_depth]

    texture_height, texture_width = texture_rgba.shape[:2]
    texture_x = np.clip(
        np.rint(texture_u * (texture_width - 1)).astype(np.int32),
        0,
        texture_width - 1,
    )
    texture_y = np.clip(
        np.rint(texture_v * (texture_height - 1)).astype(np.int32),
        0,
        texture_height - 1,
    )
    sample = texture_rgba[texture_y, texture_x]
    texture_alpha = sample[..., 3].astype(np.float64)
    alpha_pixel = np.clip(
        texture_alpha * np.clip(ground_alpha, 0.0, 255.0) / 255.0,
        0.0,
        255.0,
    )
    draw = closer & (alpha_pixel > 3.0)
    if not np.any(draw):
        return

    depth_patch[draw] = pixel_depth.astype(np.float32)[draw]
    mask_patch = mask_buffer[min_v : max_v + 1, min_u : max_u + 1]
    color_patch = color_buffer[min_v : max_v + 1, min_u : max_u + 1]
    mask_patch[draw] = np.maximum(
        mask_patch[draw],
        alpha_pixel.astype(np.uint8)[draw],
    )
    color_patch[draw] = sample[..., :3][draw]


def _draw_ground_label(
    robot: Dict[str, Any],
    *,
    text: str,
    center_x_m: float,
    center_y_m: float,
    width_y_m: float,
    height_x_m: float,
    height_offset_m: float,
    camera: Dict[str, Any],
    width: int,
    height: int,
    color_rgb: Tuple[int, int, int],
    color_buffer: np.ndarray,
    depth_buffer: np.ndarray,
    mask_buffer: np.ndarray,
) -> None:
    half_width = 0.5 * float(width_y_m)
    half_height = 0.5 * float(height_x_m)
    x = float(center_x_m)
    y = float(center_y_m)
    z = float(height_offset_m)
    local = np.array(
        [
            [x + half_height, y + half_width, z],
            [x + half_height, y - half_width, z],
            [x - half_height, y - half_width, z],
            [x - half_height, y + half_width, z],
        ],
        dtype=np.float64,
    )
    world = _robot_local_points_to_world(robot, local)
    uv, z_camera = _project_points(world, camera, width, height)
    if not (np.all(np.isfinite(uv)) and np.all(z_camera < -1e-6)):
        return
    texture_uv = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    alpha = _fade_alpha_from_forward_x(local[:, 0])
    if float(np.max(alpha)) <= 1.0:
        return
    texture = _render_label_texture(text, color_rgb)
    for triangle in ((0, 1, 2), (0, 2, 3)):
        indices = list(triangle)
        _rasterize_textured_triangle(
            uv[indices],
            (-z_camera[indices]).astype(np.float32),
            texture_uv[indices],
            alpha[indices],
            texture,
            color_buffer,
            depth_buffer,
            mask_buffer,
        )


def _hud_path_buffers(
    robot: Dict[str, Any],
    *,
    height_offset_m: float,
    camera: Dict[str, Any],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    mask_buffer = np.zeros((height, width), dtype=np.uint8)

    half_width = 0.5 * PATH_WIDTH_M
    y0 = PATH_Y_CENTER_M - half_width
    y1 = PATH_Y_CENTER_M + half_width
    lane_color = (20, 125, 230)
    arrow_color = (30, 145, 245)
    label_color = LABEL_COLOR_RGB

    def draw(
        polyline,
        *,
        color_rgb,
        radius_px=2.0,
        step_m=0.004,
    ) -> None:
        _draw_local_polyline(
            robot,
            np.asarray(polyline, dtype=np.float64),
            height_offset_m=height_offset_m,
            camera=camera,
            width=width,
            height=height,
            color_rgb=color_rgb,
            radius_px=radius_px,
            step_m=step_m,
            color_buffer=color_buffer,
            depth_buffer=depth_buffer,
            mask_buffer=mask_buffer,
        )

    draw(
        [[PATH_START_X_M, y0], [PATH_END_X_M, y0]],
        color_rgb=lane_color,
        radius_px=3.4,
    )
    draw(
        [[PATH_START_X_M, y1], [PATH_END_X_M, y1]],
        color_rgb=lane_color,
        radius_px=3.4,
    )
    draw(
        [[PATH_START_X_M, y0], [PATH_START_X_M, y1]],
        color_rgb=lane_color,
        radius_px=3.4,
    )

    for lateral_offset, mark_length, label in _near_edge_lateral_mark_specs():
        marker_y = PATH_Y_CENTER_M + lateral_offset
        draw(
            [
                [PATH_START_X_M, marker_y],
                [PATH_START_X_M + mark_length, marker_y],
            ],
            color_rgb=lane_color,
            radius_px=3.0,
        )
        if label is not None:
            label_y = marker_y
            if abs(lateral_offset) == LATERAL_REFERENCE_OUTER_OFFSET_M:
                label_y += (
                    -NEAR_EDGE_OUTER_LABEL_OUTSET_M
                    if lateral_offset < 0.0
                    else NEAR_EDGE_OUTER_LABEL_OUTSET_M
                )
            _draw_ground_label(
                robot,
                text=label,
                center_x_m=PATH_START_X_M + NEAR_EDGE_LABEL_FORWARD_OFFSET_M,
                center_y_m=label_y,
                width_y_m=0.16,
                height_x_m=0.075,
                height_offset_m=height_offset_m,
                camera=camera,
                width=width,
                height=height,
                color_rgb=label_color,
                color_buffer=color_buffer,
                depth_buffer=depth_buffer,
                mask_buffer=mask_buffer,
            )

    draw(
        [
            [PATH_START_X_M, PATH_Y_CENTER_M],
            [PATH_START_X_M + 0.001, PATH_Y_CENTER_M],
        ],
        color_rgb=lane_color,
        radius_px=4.2,
    )

    tick_length = 0.13
    for distance in np.arange(
        DISTANCE_TICK_STEP_M,
        PATH_LENGTH_M + 1e-6,
        DISTANCE_TICK_STEP_M,
    ):
        value = float(distance)
        local_x = _base_local_x_from_front_distance(value)
        draw(
            [[local_x, y0], [local_x, y0 - tick_length]],
            color_rgb=lane_color,
            radius_px=2.5,
        )
        draw(
            [[local_x, y1], [local_x, y1 + tick_length]],
            color_rgb=lane_color,
            radius_px=2.5,
        )
        label = f"{value:.1f}m"
        _draw_ground_label(
            robot,
            text=label,
            center_x_m=local_x,
            center_y_m=y0 - tick_length - 0.14,
            width_y_m=0.28,
            height_x_m=0.12,
            height_offset_m=height_offset_m,
            camera=camera,
            width=width,
            height=height,
            color_rgb=label_color,
            color_buffer=color_buffer,
            depth_buffer=depth_buffer,
            mask_buffer=mask_buffer,
        )
        _draw_ground_label(
            robot,
            text=label,
            center_x_m=local_x,
            center_y_m=y1 + tick_length + 0.14,
            width_y_m=0.28,
            height_x_m=0.12,
            height_offset_m=height_offset_m,
            camera=camera,
            width=width,
            height=height,
            color_rgb=label_color,
            color_buffer=color_buffer,
            depth_buffer=depth_buffer,
            mask_buffer=mask_buffer,
        )

    for arrow_distance in (0.45, 0.9, 1.35, 1.8, 2.25, 2.7):
        arrow_x = _base_local_x_from_front_distance(arrow_distance)
        tail_x = arrow_x - 0.15
        tip_x = arrow_x + 0.17
        wing = min(0.27, half_width * 0.70)
        draw(
            [
                [tail_x, PATH_Y_CENTER_M - wing],
                [tip_x, PATH_Y_CENTER_M],
            ],
            color_rgb=arrow_color,
            radius_px=3.0,
        )
        draw(
            [
                [tail_x, PATH_Y_CENTER_M + wing],
                [tip_x, PATH_Y_CENTER_M],
            ],
            color_rgb=arrow_color,
            radius_px=3.0,
        )

    return color_buffer, mask_buffer, depth_buffer


def _apply_scene_occlusion(
    mask: np.ndarray,
    layer_depth: np.ndarray,
    scene_depth: Optional[np.ndarray],
) -> np.ndarray:
    visible = np.asarray(mask, dtype=np.uint8).copy()
    if scene_depth is None:
        return visible
    scene = np.asarray(scene_depth, dtype=np.float32)
    if scene.shape != layer_depth.shape:
        return visible
    valid_scene = np.isfinite(scene) & (scene > 0.05)
    present = visible > 0
    occluded = (
        present
        & valid_scene
        & (scene < layer_depth - float(DEPTH_OCCLUDE_EPS_M))
    )
    visible[occluded] = 0
    return visible


def chassis_forward_2m_sentence(
    points: Optional[list[tuple[int, int, float]]] = None,
) -> str:
    """模型可见的一句提示：先总述，再按距离列出每片物体的最近点。"""
    if not points:
        return CHASSIS_FORWARD_2M_CLEAR
    parts: list[str] = []
    for index, (u, v, distance_m) in enumerate(points):
        meters = f"{max(0.0, float(distance_m)):.2f}"
        parts.append(
            f"nearest point {index + 1} ({int(u)},{int(v)}), "
            f"distance {meters}m"
        )
    return (
        "Some kind of object is on the chassis-forward path: "
        + "; ".join(parts)
    )


def compose_nearby_object_warning(
    chassis_forward_2m: str = "",
    eef_near_0_1m: str = "",
) -> str:
    """合成给模型的近物预警：目前只报成熟的底盘走廊。

    `eef_near_0.1m` 仍留给 overlay / 核对，先不并进这句。
    """
    del eef_near_0_1m
    chassis = str(chassis_forward_2m or "").strip()
    if chassis and chassis != CHASSIS_FORWARD_2M_CLEAR:
        return chassis
    return ""


def _unproject_path_pixels_to_world(
    *,
    pixel_u: np.ndarray,
    pixel_v: np.ndarray,
    path_depth: np.ndarray,
    camera: Dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    """用蓝线平面深度把像素反投到世界系。path_depth 是 -z_camera。"""
    camera_position, camera_quaternion, fx, fy, cx, cy = (
        _camera_projection_values(camera, width, height)
    )
    depth = np.asarray(path_depth, dtype=np.float64)
    x_cam = (np.asarray(pixel_u, dtype=np.float64) - cx) / float(fx) * depth
    y_cam = (float(cy) - np.asarray(pixel_v, dtype=np.float64)) / float(fy) * depth
    z_cam = -depth
    camera_points = np.stack([x_cam, y_cam, z_cam], axis=-1)
    rotation = quat_to_mat_xyzw(camera_quaternion)
    return (rotation @ camera_points.T).T + camera_position.reshape(1, 3)


def _world_points_to_base_local(
    robot: Dict[str, Any],
    points_world: np.ndarray,
) -> np.ndarray:
    """世界点变到底盘坐标系（+X 前，+Y 左，+Z 上）。"""
    base_position, yaw_deg = _base_pose_values(robot)
    yaw = math.radians(yaw_deg)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    delta = np.asarray(points_world, dtype=np.float64).reshape(-1, 3) - (
        base_position.reshape(1, 3)
    )
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    local_z = delta[:, 2]
    return np.stack([local_x, local_y, local_z], axis=1)


def _world_points_to_front_distance_m(
    robot: Dict[str, Any],
    points_world: np.ndarray,
) -> np.ndarray:
    """世界点到底盘前碰撞沿的直行距离（机体系 +X）。"""
    local = _world_points_to_base_local(robot, points_world)
    return local[:, 0] - float(BASE_FRONT_OFFSET_M)


def _link_self_radius_m(link_name: str) -> float:
    """连杆原点附近的自遮挡半径，只盖本体，不上完整 mesh。"""
    name = str(link_name).lower()
    if "finger" in name or "gripper" in name:
        return 0.14
    if "realsense" in name or "zed" in name:
        return 0.08
    if "arm_link" in name or "arm_base" in name:
        return 0.12
    if "torso" in name:
        return 0.18
    if "base_link" in name or "steer" in name or "wheel" in name:
        return 0.28
    return 0.10


def _consecutive_link_bones(
    link_names: list[str],
) -> list[tuple[str, str]]:
    """臂/躯干相邻连杆，用来做胶囊而不是只打原点球。"""
    pairs: list[tuple[str, str]] = []
    name_set = set(link_names)
    for prefix in ("left_arm_link", "right_arm_link", "torso_link"):
        chain = [
            name
            for name in link_names
            if name.startswith(prefix) and name[len(prefix) :].isdigit()
        ]
        chain.sort(key=lambda name: int(name[len(prefix) :]))
        pairs.extend(zip(chain, chain[1:]))
        side = prefix.split("_", 1)[0]
        gripper = f"{side}_gripper_link"
        if chain and gripper in name_set:
            pairs.append((chain[-1], gripper))
    return pairs


def _points_near_link_capsules(
    points_world: np.ndarray,
    transforms: Dict[str, np.ndarray],
) -> np.ndarray:
    """点落在连杆原点球或相邻连杆胶囊内则视为本体。"""
    names = list(transforms)
    if not names:
        return np.zeros((int(points_world.shape[0]),), dtype=bool)
    centers = np.stack(
        [np.asarray(transforms[name][:3, 3], dtype=np.float64) for name in names],
        axis=0,
    )
    radii = np.array(
        [_link_self_radius_m(name) for name in names],
        dtype=np.float64,
    )
    delta = points_world[:, None, :] - centers[None, :, :]
    near_origin = (np.linalg.norm(delta, axis=2) < radii[None, :]).any(axis=1)
    bones = _consecutive_link_bones(names)
    if not bones:
        return near_origin
    start = np.stack(
        [np.asarray(transforms[a][:3, 3], dtype=np.float64) for a, _b in bones],
        axis=0,
    )
    end = np.stack(
        [np.asarray(transforms[b][:3, 3], dtype=np.float64) for _a, b in bones],
        axis=0,
    )
    bone_radius = np.array(
        [
            max(_link_self_radius_m(start_name), _link_self_radius_m(end_name))
            for start_name, end_name in bones
        ],
        dtype=np.float64,
    )
    axis = end - start
    length2 = np.sum(axis * axis, axis=1)
    length2 = np.maximum(length2, 1e-12)
    offset = points_world[:, None, :] - start[None, :, :]
    t = np.clip(
        np.sum(offset * axis[None, :, :], axis=2) / length2[None, :],
        0.0,
        1.0,
    )
    closest = start[None, :, :] + t[:, :, None] * axis[None, :, :]
    near_bone = (
        np.linalg.norm(points_world[:, None, :] - closest, axis=2)
        < bone_radius[None, :]
    ).any(axis=1)
    return near_origin | near_bone


def _chassis_and_ground_mask(
    *,
    points_world: np.ndarray,
    robot: Dict[str, Any],
    front_m: np.ndarray,
) -> np.ndarray:
    """地板、底盘盒、贴前沿矮体积。不当持物生长的种子。"""
    local = _world_points_to_base_local(robot, points_world)
    local_x = local[:, 0]
    local_y = local[:, 1]
    local_z = local[:, 2]
    return (
        (local_z < float(CHASSIS_SELF_GROUND_Z_M))
        | (
            (local_x >= float(CHASSIS_SELF_X_MIN_M))
            & (
                local_x
                <= float(BASE_FRONT_OFFSET_M) + float(CHASSIS_SELF_X_PAST_FRONT_M)
            )
            & (np.abs(local_y) <= float(CHASSIS_SELF_HALF_WIDTH_M))
            & (local_z >= -0.05)
            & (local_z <= float(CHASSIS_SELF_Z_MAX_M))
        )
        | (
            (front_m < float(CHASSIS_SELF_NEAR_FRONT_M))
            & (local_z < float(CHASSIS_SELF_NEAR_Z_MAX_M))
            & (np.abs(local_y) <= float(CHASSIS_SELF_HALF_WIDTH_M))
        )
    )


def _held_body_seed_mask(
    points_world: np.ndarray,
    robot: Dict[str, Any],
) -> np.ndarray:
    """臂/爪 FK 胶囊和 EEF 球，静态滤本体，不当持物生长的唯一种子。"""
    n_points = int(np.asarray(points_world).reshape(-1, 3).shape[0])
    seed = np.zeros((n_points,), dtype=bool)
    qpos = robot.get("arm_left_qpos")
    if qpos is None or len(qpos) < 6:
        return seed
    try:
        from .grasp_kinematics_local import LocalRobotState, link_transforms

        state = LocalRobotState.from_capture(robot, arm_dof=int(len(qpos)))
        transforms = link_transforms(state)
        seed = seed | _points_near_link_capsules(points_world, transforms)
        eef_centers = []
        for key in ("eef_left", "eef_right"):
            pos = (robot.get(key) or {}).get("pos")
            if pos is None:
                continue
            eef_centers.append(np.asarray(pos, dtype=np.float64).reshape(3))
        if eef_centers:
            centers = np.stack(eef_centers, axis=0)
            seed = seed | (
                np.linalg.norm(
                    np.asarray(points_world, dtype=np.float64).reshape(-1, 3)[
                        :, None, :
                    ]
                    - centers[None, :, :],
                    axis=2,
                )
                < float(CHASSIS_HELD_GRIPPER_SEED_M)
            ).any(axis=1)
    except Exception:
        return seed
    return seed


def _gripper_anchor_distance_m(
    points_world: np.ndarray,
    robot: Dict[str, Any],
) -> np.ndarray:
    """到左右 EEF / 夹爪 / 手指原点的最近距离。无关节角则为 inf。"""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    anchors: list[np.ndarray] = []
    for key in ("eef_left", "eef_right"):
        pos = (robot.get(key) or {}).get("pos")
        if pos is None:
            continue
        anchors.append(np.asarray(pos, dtype=np.float64).reshape(3))
    qpos = robot.get("arm_left_qpos")
    if qpos is not None and len(qpos) >= 6:
        try:
            from .grasp_kinematics_local import LocalRobotState, link_transforms

            state = LocalRobotState.from_capture(robot, arm_dof=int(len(qpos)))
            transforms = link_transforms(state)
            for name, transform in transforms.items():
                lower = str(name).lower()
                if "gripper" in lower or "finger" in lower:
                    anchors.append(
                        np.asarray(transform[:3, 3], dtype=np.float64).reshape(3)
                    )
        except Exception:
            pass
    if not anchors:
        return np.full((points.shape[0],), np.inf, dtype=np.float64)
    centers = np.stack(anchors, axis=0)
    return np.min(
        np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2),
        axis=1,
    )


def _robot_self_or_ground_mask(
    *,
    points_world: np.ndarray,
    robot: Dict[str, Any],
    front_m: np.ndarray,
) -> np.ndarray:
    """地板/底盘加上臂爪本体，不含持物连通。"""
    return _chassis_and_ground_mask(
        points_world=points_world,
        robot=robot,
        front_m=front_m,
    ) | _held_body_seed_mask(points_world, robot)


def _shift_mask_and_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    drow: int,
    dcol: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """把图沿 (drow, dcol) 平移，空位为 0 / nan。"""
    height, width = mask.shape
    shifted_mask = np.zeros_like(mask)
    shifted_depth = np.full(depth.shape, np.nan, dtype=np.float32)
    src_r0 = max(0, drow)
    src_c0 = max(0, dcol)
    src_r1 = height + min(0, drow)
    src_c1 = width + min(0, dcol)
    dst_r0 = max(0, -drow)
    dst_c0 = max(0, -dcol)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return shifted_mask, shifted_depth
    shifted_mask[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    shifted_depth[dst_r0:dst_r1, dst_c0:dst_c1] = depth[src_r0:src_r1, src_c0:src_c1]
    return shifted_mask, shifted_depth


def _grow_held_by_depth(
    *,
    height: int,
    width: int,
    rows: np.ndarray,
    cols: np.ndarray,
    scene_on_points: np.ndarray,
    seed_on_points: np.ndarray,
    anchor_dist_on_points: Optional[np.ndarray] = None,
    max_anchor_m: float = CHASSIS_HELD_MAX_M,
    depth_eps_m: float = CHASSIS_HELD_DEPTH_CONT_M,
) -> np.ndarray:
    """从夹爪种子在遮挡像素上 8 连通生长，邻点深度差不超过阈值。

    距夹爪超过 max_anchor_m 的点不长进去，避免把贴着爪子的地面物体整片吃掉。
    """
    attached = np.zeros((int(rows.shape[0]),), dtype=bool)
    if not np.any(seed_on_points):
        return attached
    allowed = np.ones((int(rows.shape[0]),), dtype=bool)
    if anchor_dist_on_points is not None:
        allowed = np.isfinite(anchor_dist_on_points) & (
            anchor_dist_on_points <= float(max_anchor_m)
        )
        seed_on_points = seed_on_points & allowed
        if not np.any(seed_on_points):
            return attached
    occluded = np.zeros((height, width), dtype=np.uint8)
    occluded[rows, cols] = 1
    if anchor_dist_on_points is not None:
        occluded[rows[~allowed], cols[~allowed]] = 0
    grown = np.zeros((height, width), dtype=np.uint8)
    grown[rows[seed_on_points], cols[seed_on_points]] = 1
    depth_map = np.full((height, width), np.nan, dtype=np.float32)
    depth_map[rows, cols] = np.asarray(scene_on_points, dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    max_iter = int(height + width)
    for _ in range(max_iter):
        candidates = cv2.dilate(grown, kernel) & occluded
        new_pixels = candidates & (1 - grown)
        if not np.any(new_pixels):
            break
        min_diff = np.full((height, width), np.inf, dtype=np.float32)
        for drow, dcol in neighbors:
            neigh_mask, neigh_depth = _shift_mask_and_depth(
                grown,
                depth_map,
                drow,
                dcol,
            )
            valid = (
                (new_pixels > 0)
                & (neigh_mask > 0)
                & np.isfinite(neigh_depth)
                & np.isfinite(depth_map)
            )
            diff = np.abs(depth_map - neigh_depth)
            min_diff = np.where(valid, np.minimum(min_diff, diff), min_diff)
        add = (new_pixels > 0) & (min_diff <= float(depth_eps_m))
        if not np.any(add):
            break
        grown[add] = 1
    attached[:] = grown[rows, cols] > 0
    return attached


def list_chassis_forward_2m_hits(
    *,
    path_mask: np.ndarray,
    path_depth: np.ndarray,
    scene_depth: Optional[np.ndarray],
    camera: Dict[str, Any],
    robot: Dict[str, Any],
) -> list[tuple[int, int, float]]:
    """2m 蓝线走廊上、滤掉地板/本体后，每片连续遮挡一个代表点。"""
    if scene_depth is None:
        return []
    scene = np.asarray(scene_depth, dtype=np.float32)
    if scene.shape != path_depth.shape:
        return []
    present = (
        (np.asarray(path_mask) > 0)
        & np.isfinite(path_depth)
        & (path_depth > 0.05)
        & np.isfinite(scene)
        & (scene > 0.05)
    )
    occluded = present & (scene < path_depth - float(DEPTH_OCCLUDE_EPS_M))
    if not np.any(occluded):
        return []
    rows, cols = np.nonzero(occluded)
    pixel_u = cols.astype(np.float64) + 0.5
    pixel_v = rows.astype(np.float64) + 0.5
    height, width = path_depth.shape
    world_path = _unproject_path_pixels_to_world(
        pixel_u=pixel_u,
        pixel_v=pixel_v,
        path_depth=path_depth[rows, cols],
        camera=camera,
        width=width,
        height=height,
    )
    front_m = _world_points_to_front_distance_m(robot, world_path)
    world_scene = _unproject_path_pixels_to_world(
        pixel_u=pixel_u,
        pixel_v=pixel_v,
        path_depth=scene[rows, cols],
        camera=camera,
        width=width,
        height=height,
    )
    chassis_or_ground = _chassis_and_ground_mask(
        points_world=world_scene,
        robot=robot,
        front_m=front_m,
    )
    body_seed = _held_body_seed_mask(world_scene, robot)
    gripper_dist = _gripper_anchor_distance_m(world_scene, robot)
    gripper_seed = np.isfinite(gripper_dist) & (
        gripper_dist <= float(CHASSIS_HELD_GRIPPER_SEED_M)
    )
    held_attached = _grow_held_by_depth(
        height=height,
        width=width,
        rows=rows,
        cols=cols,
        scene_on_points=scene[rows, cols],
        seed_on_points=gripper_seed,
        anchor_dist_on_points=gripper_dist,
        max_anchor_m=float(CHASSIS_HELD_MAX_M),
    )
    self_or_ground = chassis_or_ground | body_seed | held_attached
    in_window = (
        np.isfinite(front_m)
        & (front_m >= 0.0)
        & (front_m <= float(CHASSIS_FORWARD_HINT_M))
        & ~self_or_ground
    )
    if not np.any(in_window):
        return []
    # 同一片连续遮挡只取一个代表点（该片里离底盘最近的像素）。
    hit_mask = np.zeros(path_depth.shape, dtype=np.uint8)
    hit_mask[rows[in_window], cols[in_window]] = 255
    hit_mask = cv2.morphologyEx(
        hit_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        hit_mask,
        connectivity=8,
    )
    front_map = np.full(path_depth.shape, np.inf, dtype=np.float64)
    front_map[rows[in_window], cols[in_window]] = front_m[in_window]
    points: list[tuple[float, int, int]] = []
    for label in range(1, int(component_count)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 16:
            continue
        component = labels == label
        nearest = np.argmin(
            np.where(component, front_map, np.inf)
        )
        row, col = np.unravel_index(int(nearest), front_map.shape)
        distance = float(front_map[row, col])
        if not np.isfinite(distance):
            continue
        points.append((distance, int(col), int(row)))
    points.sort()
    # 持物裁掉近端后，同一物体可能裂成两片，近且像素靠近的只留一个。
    merged: list[tuple[float, int, int]] = []
    for distance, u, v in points:
        too_close = False
        for keep_distance, keep_u, keep_v in merged:
            pixel_d2 = (u - keep_u) ** 2 + (v - keep_v) ** 2
            if pixel_d2 <= 40 * 40 and abs(distance - keep_distance) <= 0.15:
                too_close = True
                break
        if not too_close:
            merged.append((distance, u, v))
    return [(u, v, distance) for distance, u, v in merged]


def describe_chassis_forward_2m(
    *,
    path_mask: np.ndarray,
    path_depth: np.ndarray,
    scene_depth: Optional[np.ndarray],
    camera: Dict[str, Any],
    robot: Dict[str, Any],
) -> str:
    """从 depth 与蓝线几何生成 `chassis_forward_2m` 一句提示。"""
    return chassis_forward_2m_sentence(
        list_chassis_forward_2m_hits(
            path_mask=path_mask,
            path_depth=path_depth,
            scene_depth=scene_depth,
            camera=camera,
            robot=robot,
        )
    )


def eef_near_sentence(
    hits: Optional[list[tuple[str, int, int, float]]] = None,
) -> str:
    """有近距团才写一句；不分持物/障碍，没有就空串。"""
    if not hits:
        return ""
    by_side: dict[str, list[tuple[int, int, float]]] = {
        "left": [],
        "right": [],
    }
    for side, u, v, distance_m in hits:
        key = str(side).strip().lower()
        if key not in by_side:
            continue
        by_side[key].append((int(u), int(v), float(distance_m)))
    parts: list[str] = []
    for side in ("left", "right"):
        points = by_side[side]
        if not points:
            continue
        inner = "; ".join(
            f"nearest point {index + 1} ({u},{v}), "
            f"distance {max(0.0, distance_m):.2f}m"
            for index, (u, v, distance_m) in enumerate(points)
        )
        parts.append(f"object near {side} eef：{inner}")
    return "; ".join(parts)


def _eef_side_link_centers(
    robot: Dict[str, Any],
    side: str,
) -> list[np.ndarray]:
    """同侧 EEF / 夹爪 / 手指原点，只用来扣本体，不当持物种子。"""
    return [center for center, _radius in _eef_side_body_excludes(robot, side)]


def _eef_side_body_excludes(
    robot: Dict[str, Any],
    side: str,
) -> list[tuple[np.ndarray, float]]:
    """同侧本体球：手指/EEF 3cm，腕相机 5.5cm。"""
    items: list[tuple[np.ndarray, float]] = []
    pose = robot.get(f"eef_{side}") or {}
    pos = pose.get("pos")
    if pos is not None:
        eef = np.asarray(pos, dtype=np.float64).reshape(3)
        items.append((eef, float(EEF_NEAR_LINK_EXCLUDE_M)))
        if pose.get("quat") is not None:
            rotation = quat_to_mat_xyzw(
                np.asarray(pose["quat"], dtype=np.float64).reshape(4)
            )
            items.append(
                (
                    eef + rotation @ EEF_NEAR_CAMERA_T_EEF,
                    float(EEF_NEAR_CAMERA_EXCLUDE_M),
                )
            )
    qpos = robot.get("arm_left_qpos")
    if qpos is None or len(qpos) < 6:
        return items
    try:
        from .grasp_kinematics_local import LocalRobotState, link_transforms

        state = LocalRobotState.from_capture(robot, arm_dof=int(len(qpos)))
        transforms = link_transforms(state)
        for name, transform in transforms.items():
            lower = str(name).lower()
            if side not in lower:
                continue
            center = np.asarray(transform[:3, 3], dtype=np.float64).reshape(3)
            if "realsense" in lower or "zed" in lower:
                items.append((center, float(EEF_NEAR_CAMERA_EXCLUDE_M)))
            elif "finger" in lower or "gripper" in lower:
                items.append((center, float(EEF_NEAR_LINK_EXCLUDE_M)))
    except Exception:
        return items
    return items


def _inside_eef_box(
    local: np.ndarray,
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = box
    return (
        (local[:, 0] >= x_lo)
        & (local[:, 0] <= x_hi)
        & (local[:, 1] >= y_lo)
        & (local[:, 1] <= y_hi)
        & (local[:, 2] >= z_lo)
        & (local[:, 2] <= z_hi)
    )


def _points_in_eef_camera_box(points_eef: Optional[np.ndarray]) -> np.ndarray:
    """EEF 系里腕相机和手背壳体，避免把头图里的 RealSense 报成障碍。"""
    if points_eef is None:
        return np.zeros((0,), dtype=bool)
    local = np.asarray(points_eef, dtype=np.float64).reshape(-1, 3)
    return _inside_eef_box(local, EEF_NEAR_CAMERA_BOX_EEF) | _inside_eef_box(
        local,
        EEF_NEAR_WRIST_BOX_EEF,
    )


def _wrist_self_mesh_cloud(gripper_qpos: Optional[Sequence[float]]) -> np.ndarray:
    """夹爪壳体 + RealSense + 整根手指（含指尖，EEF 系）。"""
    from .visualization_local import _gripper_visual_mesh, _normalized_gripper_qpos

    vertices, faces, face_component, _names = _gripper_visual_mesh()
    values = _normalized_gripper_qpos(gripper_qpos)
    parts: list[np.ndarray] = []
    for component in (0, 3):
        parts.append(vertices[np.unique(faces[face_component == component])])
    finger_shift = {
        1: -(0.05 - float(values[0])),
        2: (0.05 - float(values[1])),
    }
    for component, shift_y in finger_shift.items():
        points = vertices[np.unique(faces[face_component == component])].copy()
        points[:, 1] += shift_y
        parts.append(points)
    return np.vstack(parts)


def _points_on_wrist_self_mesh(
    points_eef: Optional[np.ndarray],
    gripper_qpos: Optional[Sequence[float]],
) -> np.ndarray:
    """贴近夹爪/指尖网格的点当本体，避免空爪把指垫报成物体。"""
    if points_eef is None:
        return np.zeros((0,), dtype=bool)
    local = np.asarray(points_eef, dtype=np.float64).reshape(-1, 3)
    if local.size == 0:
        return np.zeros((0,), dtype=bool)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(_wrist_self_mesh_cloud(gripper_qpos))
        distance, _index = tree.query(local, k=1)
    except Exception:
        return np.zeros((local.shape[0],), dtype=bool)
    local_z = local[:, 2]
    limit = np.where(
        local_z <= float(EEF_NEAR_SELF_MESH_FINGER_Z),
        float(EEF_NEAR_SELF_MESH_REAR_M),
        float(EEF_NEAR_SELF_MESH_M),
    )
    return np.asarray(distance, dtype=np.float64) <= limit


def _world_points_to_eef_local(
    eef_pose: Dict[str, Any],
    points_world: np.ndarray,
) -> Optional[np.ndarray]:
    """世界点变到该侧 EEF 系。缺位姿则无法判指缝。"""
    pos = eef_pose.get("pos")
    quat = eef_pose.get("quat")
    if pos is None or quat is None:
        return None
    origin = np.asarray(pos, dtype=np.float64).reshape(3)
    rotation = quat_to_mat_xyzw(np.asarray(quat, dtype=np.float64).reshape(4))
    return (np.asarray(points_world, dtype=np.float64).reshape(-1, 3) - origin) @ rotation


def _opening_band_on_points(
    points_eef: np.ndarray,
    gripper_qpos: Optional[Sequence[float]],
    *,
    x_pad_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """开口（可沿 x 外扩）内、剥掉贴指面后的点掩码，以及对应 z。"""
    from .visualization_local import _dynamic_opening_lut

    local = np.asarray(points_eef, dtype=np.float64).reshape(-1, 3)
    empty = np.zeros((local.shape[0],), dtype=bool)
    if local.size == 0:
        return empty, local[:, 2] if local.ndim == 2 else np.zeros((0,), dtype=np.float64)
    lut = _dynamic_opening_lut(gripper_qpos)
    z = local[:, 2]
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
    x_half = np.interp(
        z,
        z_samples,
        np.asarray(lut["x_half_gap"], dtype=np.float64),
    )
    peel = float(EEF_NEAR_GAP_FACE_PEEL_M)
    band = (
        (local[:, 1] > y_lo + peel)
        & (local[:, 1] < y_hi - peel)
        & (np.abs(local[:, 0]) <= x_half + float(x_pad_m))
    )
    return band, z


def _fingertip_slit_held_seed_on_points(
    points_eef: np.ndarray,
    gripper_qpos: Optional[Sequence[float]],
) -> np.ndarray:
    """全开爪时，指尖缝近点且指前无透视，才当持物种子。

    罐沿常落在严格开口 x 盒外；椅背会在指前 4–20cm 露出一大片远点。
    """
    band, z = _opening_band_on_points(
        points_eef,
        gripper_qpos,
        x_pad_m=float(EEF_NEAR_SLIT_X_PAD_M),
    )
    seed = np.zeros((band.shape[0],), dtype=bool)
    near = band & (z >= float(EEF_NEAR_SLIT_Z_LO_M)) & (
        z <= float(EEF_NEAR_SLIT_Z_HI_M)
    )
    far = band & (z > float(EEF_NEAR_SLIT_FAR_Z_LO_M)) & (
        z <= float(EEF_NEAR_GROW_MAX_M)
    )
    if (
        int(np.count_nonzero(near)) < int(EEF_NEAR_GAP_MIN_POINTS)
        or int(np.count_nonzero(far)) > 0
    ):
        return seed
    seed[near] = True
    return seed


def _opening_held_seed_on_points(
    points_eef: np.ndarray,
    gripper_qpos: Optional[Sequence[float]],
) -> np.ndarray:
    """开口盒子内、去掉掌心和贴指面后的持物种子。

    贴墙会铺满开口，但点都贴在指内侧面；剥 3mm 后盒子就空了。
    真夹住的斧柄/罐子在两指中间，剥完还在。全开爪持罐时再加指尖缝。
    """
    from .visualization_local import _dynamic_opening_lut, _points_inside_opening

    local = np.asarray(points_eef, dtype=np.float64).reshape(-1, 3)
    seed = np.zeros((local.shape[0],), dtype=bool)
    if local.size == 0:
        return seed
    lut = _dynamic_opening_lut(gripper_qpos)
    inside = _points_inside_opening(local, lut) & (
        local[:, 2] >= float(EEF_NEAR_GAP_PALM_Z_M)
    )
    if np.any(inside):
        z = local[inside, 2]
        y = local[inside, 1]
        y_lo = np.interp(
            z,
            np.asarray(lut["z"], dtype=np.float64),
            np.asarray(lut["y_inner_lo"], dtype=np.float64),
        )
        y_hi = np.interp(
            z,
            np.asarray(lut["z"], dtype=np.float64),
            np.asarray(lut["y_inner_hi"], dtype=np.float64),
        )
        peel = float(EEF_NEAR_GAP_FACE_PEEL_M)
        interior = (y > y_lo + peel) & (y < y_hi - peel)
        if int(np.count_nonzero(interior)) >= int(EEF_NEAR_GAP_MIN_POINTS):
            inside_idx = np.flatnonzero(inside)
            seed[inside_idx[interior]] = True
    seed |= _fingertip_slit_held_seed_on_points(local, gripper_qpos)
    return seed


def _grasp_core_held_seed_on_points(
    points_eef: np.ndarray,
    allowed: np.ndarray,
) -> np.ndarray:
    """夹爪中心小盒子里的近点当持物种子。

    开口 LUT 经常吃不到宽盆壁；柜门/空爪的点在指后或 |y| 太大，进不来。
    """
    local = np.asarray(points_eef, dtype=np.float64).reshape(-1, 3)
    seed = np.zeros((local.shape[0],), dtype=bool)
    if local.size == 0:
        return seed
    dist = np.linalg.norm(local, axis=1)
    seed[
        np.asarray(allowed, dtype=bool)
        & (dist <= float(EEF_NEAR_CORE_SEED_DIST_M))
        & (np.abs(local[:, 0]) <= float(EEF_NEAR_CORE_SEED_X_M))
        & (np.abs(local[:, 1]) <= float(EEF_NEAR_CORE_SEED_Y_M))
        & (local[:, 2] >= float(EEF_NEAR_CORE_SEED_Z_LO_M))
        & (local[:, 2] <= float(EEF_NEAR_CORE_SEED_Z_HI_M))
    ] = True
    return seed


def _grow_held_by_3d(
    points_world: np.ndarray,
    seed_on_points: np.ndarray,
    allowed: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    """从指缝种子在 3D 体素里连通外扩。

    头图上细长柄会被更远的地面像素掐断 8 连通；3D 邻接能顺着柄走，
    又不会跳到 0.67m 外的地板。密深度不用 query_pairs，以免太慢。
    """
    attached = np.zeros((int(points_world.shape[0]),), dtype=bool)
    grow = np.asarray(allowed, dtype=bool)
    seed = np.asarray(seed_on_points, dtype=bool)
    if not np.any(grow & seed):
        return attached
    idx = np.flatnonzero(grow)
    local_seed = np.flatnonzero(seed[idx])
    if local_seed.size == 0:
        return attached
    voxel_m = max(float(radius_m) * 0.5, 0.01)
    coords = np.floor(
        np.asarray(points_world, dtype=np.float64).reshape(-1, 3)[idx] / voxel_m
    ).astype(np.int32)
    occupied: dict[tuple[int, int, int], list[int]] = {}
    for local_i, ijk in enumerate(coords):
        key = (int(ijk[0]), int(ijk[1]), int(ijk[2]))
        bucket = occupied.get(key)
        if bucket is None:
            occupied[key] = [local_i]
        else:
            bucket.append(local_i)
    frontier = [
        (int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]))
        for i in local_seed
    ]
    seen = set(frontier)
    neighbors = (
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if dx or dy or dz
    )
    neighbor_list = tuple(neighbors)
    stack = list(seen)
    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in neighbor_list:
            nxt = (x + dx, y + dy, z + dz)
            if nxt in occupied and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    attached_local = np.zeros((idx.shape[0],), dtype=bool)
    for key in seen:
        attached_local[occupied[key]] = True
    attached[idx] = attached_local
    return attached


def _cluster_nearest_hits(
    *,
    height: int,
    width: int,
    rows: np.ndarray,
    cols: np.ndarray,
    keep: np.ndarray,
    distance_m: np.ndarray,
    min_blob: int,
) -> list[tuple[int, int, float]]:
    """连续遮挡每片取离锚点最近的一个像素。"""
    if not np.any(keep):
        return []
    hit_mask = np.zeros((height, width), dtype=np.uint8)
    hit_mask[rows[keep], cols[keep]] = 255
    hit_mask = cv2.morphologyEx(
        hit_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        hit_mask,
        connectivity=8,
    )
    dist_map = np.full((height, width), np.inf, dtype=np.float64)
    dist_map[rows[keep], cols[keep]] = distance_m[keep]
    points: list[tuple[float, int, int]] = []
    for label in range(1, int(component_count)):
        if int(stats[label, cv2.CC_STAT_AREA]) < int(min_blob):
            continue
        nearest = np.argmin(np.where(labels == label, dist_map, np.inf))
        row, col = np.unravel_index(int(nearest), dist_map.shape)
        distance = float(dist_map[row, col])
        if not np.isfinite(distance):
            continue
        points.append((distance, int(col), int(row)))
    points.sort()
    merged: list[tuple[float, int, int]] = []
    for distance, u, v in points:
        too_close = False
        for keep_distance, keep_u, keep_v in merged:
            pixel_d2 = (u - keep_u) ** 2 + (v - keep_v) ** 2
            if pixel_d2 <= 40 * 40 and abs(distance - keep_distance) <= 0.15:
                too_close = True
                break
        if not too_close:
            merged.append((distance, u, v))
    return [(u, v, distance) for distance, u, v in merged]


def _cluster_nearest_hits_3d(
    *,
    points_world: np.ndarray,
    keep: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    distance_m: np.ndarray,
    voxel_m: float,
    min_points: int,
    report: Optional[np.ndarray] = None,
    points_eef: Optional[np.ndarray] = None,
    points_bgr: Optional[np.ndarray] = None,
) -> list[tuple[int, int, float]]:
    """按 3D 体素连通分团，每团报最近点。持物已在外层抠掉。"""
    connect = np.asarray(keep, dtype=bool)
    if report is None:
        report_mask = connect
    else:
        report_mask = np.asarray(report, dtype=bool)
    idx = np.flatnonzero(connect)
    if idx.size == 0:
        return []
    voxel = max(float(voxel_m), 0.01)
    coords = np.floor(
        np.asarray(points_world, dtype=np.float64).reshape(-1, 3)[idx] / voxel
    ).astype(np.int32)
    occupied: dict[tuple[int, int, int], list[int]] = {}
    for local_i, ijk in enumerate(coords):
        key = (int(ijk[0]), int(ijk[1]), int(ijk[2]))
        bucket = occupied.get(key)
        if bucket is None:
            occupied[key] = [local_i]
        else:
            bucket.append(local_i)
    seen: set[tuple[int, int, int]] = set()
    neighbors = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if dx or dy or dz
    )
    hits: list[tuple[float, int, int]] = []
    for start in occupied:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members: list[int] = []
        while stack:
            x, y, z = stack.pop()
            members.extend(occupied[(x, y, z)])
            for dx, dy, dz in neighbors:
                nxt = (x + dx, y + dy, z + dz)
                if nxt in occupied and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        loc = idx[np.asarray(members, dtype=np.int64)]
        ring = loc[report_mask[loc]]
        if ring.size < int(min_points):
            continue
        nearest = ring[int(np.argmin(distance_m[ring]))]
        hits.append(
            (
                float(distance_m[nearest]),
                int(cols[nearest]),
                int(rows[nearest]),
            )
        )
    hits.sort()
    return [(u, v, distance) for distance, u, v in hits]


def _predict_sam2_held_mask(
    *,
    seed: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    scene_bgr: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """从夹爪种子扩一张 SAM 持物 mask，给抠除和核对图共用。"""
    if scene_bgr is None or not np.any(seed):
        return None
    try:
        from .eef_near_sam2_local import predict_held_mask_bgr
    except Exception:
        return None
    return predict_held_mask_bgr(
        scene_bgr,
        rows[seed],
        cols[seed],
        point_count=int(EEF_NEAR_SAM2_POS_POINTS),
        box_pad_px=int(EEF_NEAR_SAM2_BOX_PAD_PX),
    )


def _close_bool_mask(mask: np.ndarray, size_px: int) -> np.ndarray:
    """闭运算填内部麻点；核太小则原样返回。"""
    on = np.asarray(mask, dtype=bool)
    kernel = int(size_px)
    if kernel < 3 or not np.any(on):
        return on
    if kernel % 2 == 0:
        kernel += 1
    closed = cv2.morphologyEx(
        on.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((kernel, kernel), dtype=np.uint8),
    )
    return closed > 0


def _drop_sam_edge_peel(
    peel: np.ndarray,
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    sam_mask: np.ndarray,
) -> np.ndarray:
    """贴着 SAM 边的剥离子团是持物沿空洞，不当第二物体。"""
    if not np.any(peel):
        return peel
    height, width = np.asarray(sam_mask, dtype=bool).shape[:2]
    peel_img = np.zeros((height, width), dtype=np.uint8)
    peel_img[rows[peel], cols[peel]] = 255
    sam_edge = cv2.dilate(
        np.asarray(sam_mask, dtype=np.uint8) * 255,
        np.ones((3, 3), dtype=np.uint8),
    )
    _nlab, labels, _stats, _cent = cv2.connectedComponentsWithStats(peel_img, 8)
    keep_peel = np.zeros((height, width), dtype=bool)
    for lab in range(1, int(_nlab)):
        on_lab = labels == lab
        if np.any(sam_edge[on_lab] > 0):
            continue
        keep_peel[on_lab] = True
    out = np.zeros_like(peel)
    out[peel] = keep_peel[rows[peel], cols[peel]]
    return out


def _sam_agrees_with_held(
    held: np.ndarray,
    *,
    seed: np.ndarray,
    in_sphere: np.ndarray,
    eef_dist: np.ndarray,
    sam_pts: np.ndarray,
) -> bool:
    """SAM 近处盖住持物时才用它剥第二物体；否则只信 3D。"""
    seed_n = int(np.count_nonzero(seed))
    if seed_n <= 0 or seed_n > int(EEF_NEAR_SAM2_MAX_SEED):
        return False
    near = held & (eef_dist <= float(EEF_NEAR_SAM2_NEAR_M))
    far = held & in_sphere & (eef_dist >= float(EEF_NEAR_SAM2_FAR_M))
    if not np.any(near) or not np.any(far):
        return False
    seed_cov = float(np.mean(sam_pts[seed]))
    near_cov = float(np.mean(sam_pts[near]))
    far_cov = float(np.mean(sam_pts[far]))
    return (
        seed_cov >= float(EEF_NEAR_SAM2_SEED_COV)
        and near_cov >= float(EEF_NEAR_SAM2_NEAR_COV)
        and far_cov >= float(EEF_NEAR_SAM2_FAR_COV)
    )


def _fuse_held_by_depth_and_sam(
    held_3d: np.ndarray,
    *,
    in_sphere: np.ndarray,
    seed: np.ndarray,
    eef_dist: np.ndarray,
    eef_local: Optional[np.ndarray],
    rows: np.ndarray,
    cols: np.ndarray,
    grow: np.ndarray,
    camera_pts: np.ndarray,
    sam_mask: Optional[np.ndarray],
) -> np.ndarray:
    """depth 3D 团盖满，SAM 补外形并剥第二物体。检测和核对图共用。"""
    fused = np.asarray(held_3d, dtype=bool).copy()
    if sam_mask is None:
        return fused
    on_raw = np.asarray(sam_mask, dtype=bool)
    if on_raw.shape[0] <= int(np.max(rows)) or on_raw.shape[1] <= int(np.max(cols)):
        return fused
    on_closed = _close_bool_mask(on_raw, int(EEF_NEAR_SAM2_CLOSE_PX))
    sam_pts = on_closed[rows, cols] & np.asarray(grow, dtype=bool) & ~camera_pts
    fused = fused | sam_pts
    raw_pts = on_raw[rows, cols]
    if eef_local is None or not _sam_agrees_with_held(
        held_3d,
        seed=seed,
        in_sphere=in_sphere,
        eef_dist=eef_dist,
        sam_pts=raw_pts,
    ):
        return fused
    sam_touch = (
        cv2.dilate(
            on_closed.astype(np.uint8) * 255,
            np.ones((3, 3), dtype=np.uint8),
        )
        > 0
    )
    touch_pts = sam_touch[rows, cols]
    peel = (
        np.asarray(held_3d, dtype=bool)
        & in_sphere
        & ~touch_pts
        & (eef_dist >= float(EEF_NEAR_SAM2_FAR_M))
        & (np.abs(eef_local[:, 0]) >= float(EEF_NEAR_SAM2_PEEL_X_M))
    )
    peel = _drop_sam_edge_peel(
        peel,
        rows=rows,
        cols=cols,
        sam_mask=on_closed,
    )
    if np.any(peel):
        fused = (fused & ~peel) | sam_pts
    return fused


def _peel_held_by_sam2(
    held: np.ndarray,
    *,
    in_sphere: np.ndarray,
    seed: np.ndarray,
    eef_dist: np.ndarray,
    eef_local: Optional[np.ndarray],
    rows: np.ndarray,
    cols: np.ndarray,
    sam_mask: Optional[np.ndarray],
    keep_edge_holes: bool = True,
) -> np.ndarray:
    """3D 持物团里，用 SAM 把侧向贴着的第二物体剥出来。

    只在种子少、近处 mask 够完整、远处也大部分仍是持物时启用。
    斧刃/宽盒远处覆盖低或种子太多，保持整团抠掉。
    贴 SAM 边的空洞仍算持物，避免盆沿白斑被报成障碍。
    `keep_edge_holes=False` 只给核对图画旧逼近点，不当官方提示。
    """
    if sam_mask is None or eef_local is None:
        return held
    seed_n = int(np.count_nonzero(seed))
    if seed_n <= 0 or seed_n > int(EEF_NEAR_SAM2_MAX_SEED):
        return held
    on_raw = np.asarray(sam_mask, dtype=bool)
    if on_raw.shape[0] <= int(np.max(rows)) or on_raw.shape[1] <= int(np.max(cols)):
        return held
    raw_pts = on_raw[rows, cols]
    on_closed = _close_bool_mask(on_raw, int(EEF_NEAR_SAM2_CLOSE_PX))
    # 官方提示用闭运算后的 SAM；核对旧逼近点用裸 mask，才能看到原来的点。
    on_pts = raw_pts if not keep_edge_holes else on_closed[rows, cols]
    near = held & (eef_dist <= float(EEF_NEAR_SAM2_NEAR_M))
    far = held & in_sphere & (eef_dist >= float(EEF_NEAR_SAM2_FAR_M))
    if not np.any(near) or not np.any(far):
        return held
    # 覆盖门控看裸 SAM，闭运算不拿来放宽启用条件。
    seed_cov = float(np.mean(raw_pts[seed]))
    near_cov = float(np.mean(raw_pts[near]))
    far_cov = float(np.mean(raw_pts[far]))
    if (
        seed_cov < float(EEF_NEAR_SAM2_SEED_COV)
        or near_cov < float(EEF_NEAR_SAM2_NEAR_COV)
        or far_cov < float(EEF_NEAR_SAM2_FAR_COV)
    ):
        return held
    peel = (
        held
        & in_sphere
        & ~on_pts
        & (eef_dist >= float(EEF_NEAR_SAM2_FAR_M))
        & (np.abs(eef_local[:, 0]) >= float(EEF_NEAR_SAM2_PEEL_X_M))
    )
    if keep_edge_holes:
        peel = _drop_sam_edge_peel(
            peel,
            rows=rows,
            cols=cols,
            sam_mask=on_closed,
        )
    if not np.any(peel):
        return held
    return held & ~peel


def _compose_held_review_mask(
    *,
    height: int,
    width: int,
    rows: np.ndarray,
    cols: np.ndarray,
    held: np.ndarray,
    camera_pts: np.ndarray,
    sam_mask: Optional[np.ndarray],
) -> np.ndarray:
    """核对图持物层：融合后的 3D∪SAM，闭运算填洞，再扣相机。不进官方 RGB。"""
    canvas = np.zeros((int(height), int(width)), dtype=bool)
    if np.any(held):
        canvas[rows[held], cols[held]] = True
    if sam_mask is not None:
        on = np.asarray(sam_mask, dtype=bool)
        if on.shape[:2] == canvas.shape:
            canvas |= on
    canvas = _close_bool_mask(canvas, int(EEF_NEAR_SAM2_CLOSE_PX))
    if np.any(camera_pts):
        canvas[rows[camera_pts], cols[camera_pts]] = False
    return canvas


def _analyze_eef_near(
    *,
    scene_depth: Optional[np.ndarray],
    camera: Dict[str, Any],
    robot: Dict[str, Any],
    scene_bgr: Optional[np.ndarray] = None,
) -> tuple[
    list[tuple[str, int, int, float]],
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, np.ndarray],
    list[tuple[str, int, int, float]],
]:
    """近距团代表点。有持物种子则 depth 3D 与 SAM 融合成一层再抠掉。

    第三项是核对图持物层，与抠除共用融合结果。
    第四项是旧剥法逼近点，只给核对图，不当官方提示。
    """
    empty: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    empty_masks: dict[str, np.ndarray] = {}
    if scene_depth is None:
        return [], empty, empty_masks, []
    scene = np.asarray(scene_depth, dtype=np.float32)
    if scene.ndim != 2:
        return [], empty, empty_masks, []
    height, width = scene.shape
    valid = np.isfinite(scene) & (scene > 0.05)
    if not np.any(valid):
        return [], empty, empty_masks, []

    roi = np.zeros((height, width), dtype=bool)
    side_eef: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for side in ("left", "right"):
        pos = (robot.get(f"eef_{side}") or {}).get("pos")
        if pos is None:
            continue
        eef = np.asarray(pos, dtype=np.float64).reshape(3)
        uv, z_cam = _project_points(eef.reshape(1, 3), camera, width, height)
        u = float(uv[0, 0])
        v = float(uv[0, 1])
        z_eef = float(-z_cam[0]) if np.isfinite(z_cam[0]) else float("nan")
        if not np.isfinite(z_eef) or z_eef <= 0.05:
            continue
        fx = float(camera.get("fx") or camera.get("fy") or width)
        radius_px = int(
            math.ceil(
                float(EEF_NEAR_GROW_MAX_M) * float(fx) / max(z_eef, 0.05) * 1.4
            )
        )
        radius_px = max(24, min(radius_px, max(height, width)))
        u0 = max(0, int(math.floor(u)) - radius_px)
        u1 = min(width, int(math.ceil(u)) + radius_px + 1)
        v0 = max(0, int(math.floor(v)) - radius_px)
        v1 = min(height, int(math.ceil(v)) + radius_px + 1)
        # 夹爪中心出画也扫：0.16m 球仍可能落在画面里。
        if u1 <= 0 or v1 <= 0 or u0 >= width or v0 >= height:
            continue
        roi[v0:v1, u0:u1] = True
        side_eef[side] = (eef, np.array([u, v], dtype=np.float64), z_eef)
    if not side_eef:
        return [], empty, empty_masks, []

    rows, cols = np.nonzero(valid & roi)
    if rows.size == 0:
        return [], empty, empty_masks, []
    world = _unproject_path_pixels_to_world(
        pixel_u=cols.astype(np.float64) + 0.5,
        pixel_v=rows.astype(np.float64) + 0.5,
        path_depth=scene[rows, cols],
        camera=camera,
        width=width,
        height=height,
    )
    local = _world_points_to_base_local(robot, world)
    floor = local[:, 2] < float(CHASSIS_SELF_GROUND_Z_M)
    hits: list[tuple[str, int, int, float]] = []
    probe_hits: list[tuple[str, int, int, float]] = []
    seeds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sam_masks: dict[str, np.ndarray] = {}
    for side, (eef, _uv, _z_eef) in side_eef.items():
        eef_dist = np.linalg.norm(world - eef.reshape(1, 3), axis=1)
        pose = robot.get(f"eef_{side}") or {}
        eef_local = _world_points_to_eef_local(pose, world)
        excludes = _eef_side_body_excludes(robot, side)
        near_body = np.zeros((rows.shape[0],), dtype=bool)
        if eef_local is not None:
            near_body |= _points_in_eef_camera_box(eef_local)
            near_sphere = eef_dist <= float(EEF_NEAR_SPHERE_M)
            if np.any(near_sphere):
                mesh_hit = _points_on_wrist_self_mesh(
                    eef_local[near_sphere],
                    robot.get(f"gripper_{side}_qpos"),
                )
                near_body[near_sphere] |= mesh_hit
        if excludes:
            centers = np.stack([center for center, _radius in excludes], axis=0)
            radii = np.array(
                [radius for _center, radius in excludes],
                dtype=np.float64,
            )
            link_dist = np.linalg.norm(
                world[:, None, :] - centers[None, :, :],
                axis=2,
            )
            near_body = near_body | (link_dist <= radii[None, :]).any(axis=1)
        in_sphere = (eef_dist <= float(EEF_NEAR_SPHERE_M)) & ~near_body & ~floor
        grow = (eef_dist <= float(EEF_NEAR_GROW_MAX_M)) & ~near_body & ~floor
        seed = np.zeros((rows.shape[0],), dtype=bool)
        if eef_local is not None:
            seed |= _grasp_core_held_seed_on_points(eef_local, in_sphere)
            seed |= _opening_held_seed_on_points(
                eef_local,
                robot.get(f"gripper_{side}_qpos"),
            ) & grow
        if int(np.count_nonzero(seed & grow)) >= int(EEF_NEAR_CORE_SEED_MIN):
            held = _grow_held_by_3d(
                world,
                seed,
                grow,
                float(EEF_NEAR_HELD_GROW_RADIUS_M),
            )
            seeds[side] = (rows[seed], cols[seed])
            sam_mask = _predict_sam2_held_mask(
                seed=seed,
                rows=rows,
                cols=cols,
                scene_bgr=scene_bgr,
            )
            camera_pts = (
                _points_in_eef_camera_box(eef_local)
                if eef_local is not None
                else np.zeros((rows.shape[0],), dtype=bool)
            )
            held_probe = _peel_held_by_sam2(
                held,
                in_sphere=in_sphere,
                seed=seed,
                eef_dist=eef_dist,
                eef_local=eef_local,
                rows=rows,
                cols=cols,
                sam_mask=sam_mask,
                keep_edge_holes=False,
            )
            held = _fuse_held_by_depth_and_sam(
                held,
                in_sphere=in_sphere,
                seed=seed,
                eef_dist=eef_dist,
                eef_local=eef_local,
                rows=rows,
                cols=cols,
                grow=grow,
                camera_pts=camera_pts,
                sam_mask=sam_mask,
            )
            sam_masks[side] = _compose_held_review_mask(
                height=height,
                width=width,
                rows=rows,
                cols=cols,
                held=held,
                camera_pts=camera_pts,
                sam_mask=sam_mask,
            )
            probe_sphere = in_sphere & ~held_probe
            in_sphere = in_sphere & ~held
        else:
            probe_sphere = in_sphere
        for u, v, distance_m in _cluster_nearest_hits_3d(
            points_world=world,
            keep=in_sphere,
            rows=rows,
            cols=cols,
            distance_m=eef_dist,
            voxel_m=float(EEF_NEAR_CLUSTER_VOXEL_M),
            min_points=int(EEF_NEAR_MIN_BLOB),
        ):
            hits.append((side, u, v, distance_m))
        for u, v, distance_m in _cluster_nearest_hits_3d(
            points_world=world,
            keep=probe_sphere,
            rows=rows,
            cols=cols,
            distance_m=eef_dist,
            voxel_m=float(EEF_NEAR_CLUSTER_VOXEL_M),
            min_points=int(EEF_NEAR_MIN_BLOB),
        ):
            probe_hits.append((side, u, v, distance_m))
    return hits, seeds, sam_masks, probe_hits


def list_eef_near_hits(
    *,
    scene_depth: Optional[np.ndarray],
    camera: Dict[str, Any],
    robot: Dict[str, Any],
    scene_bgr: Optional[np.ndarray] = None,
) -> list[tuple[str, int, int, float]]:
    """左右 EEF 0.16m 球内、抠掉持物后的近距团代表点。"""
    hits, _seeds, _sam_masks, _probe_hits = _analyze_eef_near(
        scene_depth=scene_depth,
        camera=camera,
        robot=robot,
        scene_bgr=scene_bgr,
    )
    return hits


def describe_eef_near_0_1m(
    *,
    scene_depth: Optional[np.ndarray],
    camera: Dict[str, Any],
    robot: Dict[str, Any],
    scene_bgr: Optional[np.ndarray] = None,
) -> str:
    """从 depth 与 EEF 位姿生成 `eef_near_0.1m` 一句提示；无障碍为空串。"""
    return eef_near_sentence(
        list_eef_near_hits(
            scene_depth=scene_depth,
            camera=camera,
            robot=robot,
            scene_bgr=scene_bgr,
        )
    )


def _composite_path_hud(
    rgb: np.ndarray,
    color_buffer: np.ndarray,
    visible_mask: np.ndarray,
    hud_color_buffer: np.ndarray,
    hud_visible_mask: np.ndarray,
) -> np.ndarray:
    output = rgb.astype(np.float32)
    fill_mask = (visible_mask.astype(np.float32) / 255.0)[:, :, None]
    fill_alpha = 0.18
    output = (
        output * (1.0 - fill_alpha * fill_mask)
        + color_buffer.astype(np.float32) * (fill_alpha * fill_mask)
    )
    line_mask = (hud_visible_mask.astype(np.float32) / 255.0)[:, :, None]
    line_alpha = 0.92
    output = (
        output * (1.0 - line_alpha * line_mask)
        + hud_color_buffer.astype(np.float32) * (line_alpha * line_mask)
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def render_base_forward_path_overlay_frame_v2(
    *,
    image_bgr: np.ndarray,
    depth_linear: Optional[np.ndarray],
    camera: Dict[str, Any],
    robot: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Render the frozen v2 metric base path HUD into an in-memory BGR frame."""
    try:
        source = np.asarray(image_bgr)
        if source.ndim != 3 or source.shape[2] < 3 or source.size == 0:
            raise ValueError("image_bgr must be a non-empty HxWx3 image")
        source = np.clip(source[..., :3], 0, 255).astype(
            np.uint8,
            copy=False,
        )
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scene_depth = None
        if depth_linear is not None:
            candidate = np.asarray(depth_linear, dtype=np.float32).squeeze()
            if candidate.shape == (height, width):
                scene_depth = candidate

        corners_world = _robot_local_path_corners(
            robot,
            height_offset_m=PATH_HEIGHT_OFFSET_M,
        )
        color_buffer, path_mask, path_depth, corners_uv = (
            _path_fill_buffers(
                robot,
                height_offset_m=PATH_HEIGHT_OFFSET_M,
                camera=camera,
                width=width,
                height=height,
            )
        )
        visible_mask = _apply_scene_occlusion(
            path_mask,
            path_depth,
            scene_depth,
        )
        hud_color_buffer, hud_mask, hud_depth = _hud_path_buffers(
            robot,
            height_offset_m=PATH_HEIGHT_OFFSET_M,
            camera=camera,
            width=width,
            height=height,
        )
        hud_visible_mask = _apply_scene_occlusion(
            hud_mask,
            hud_depth,
            scene_depth,
        )
        output_rgb = _composite_path_hud(
            rgb,
            color_buffer,
            visible_mask,
            hud_color_buffer,
            hud_visible_mask,
        )
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        labels = [
            f"{float(distance):.1f}m"
            for distance in np.arange(
                DISTANCE_TICK_STEP_M,
                PATH_LENGTH_M + 1e-6,
                DISTANCE_TICK_STEP_M,
            )
        ]
        chassis_forward_2m = describe_chassis_forward_2m(
            path_mask=path_mask,
            path_depth=path_depth,
            scene_depth=scene_depth,
            camera=camera,
            robot=robot,
        )
        eef_near_0_1m = describe_eef_near_0_1m(
            scene_depth=scene_depth,
            camera=camera,
            robot=robot,
            scene_bgr=source,
        )
        overlay_payload = {
            "ok": True,
            "build": "official_v2_local_width_080_lateral_020_040_label_no_outline_v2",
            "chassis_forward_2m": chassis_forward_2m,
            "visualization_source": (
                "submission_local_copy_of_v3_base_front_path_overlay"
            ),
            "geometry_source": (
                "evaluator_rgb_depth_camera_pose_intrinsics_and_local_base_pose"
            ),
            "path_length_m": float(PATH_LENGTH_M),
            "path_width_m": float(PATH_WIDTH_M),
            "path_side_inset_m": float(PATH_SIDE_INSET_M),
            "path_side_margin_m": float(PATH_SIDE_MARGIN_M),
            "distance_origin": "r1pro_base_front_collision_edge",
            "distance_origin_base_local_x_m": float(BASE_FRONT_OFFSET_M),
            "path_start_base_local_x_m": float(PATH_START_X_M),
            "path_end_base_local_x_m": float(PATH_END_X_M),
            "path_height_offset_m": float(PATH_HEIGHT_OFFSET_M),
            "distance_scale_enabled": True,
            "distance_tick_step_m": float(DISTANCE_TICK_STEP_M),
            "distance_labels": labels,
            "distance_tick_count_per_side": len(labels),
            "direction_arrows_enabled": True,
            "direction_arrow_count": 6,
            "near_edge_border_enabled": True,
            "near_edge_border_base_local_x_m": float(PATH_START_X_M),
            "near_edge_center_marker_enabled": True,
            "near_edge_lateral_marker_offsets_m": [
                -float(LATERAL_REFERENCE_OUTER_OFFSET_M),
                -float(LATERAL_REFERENCE_OFFSET_M),
                0.0,
                float(LATERAL_REFERENCE_OFFSET_M),
                float(LATERAL_REFERENCE_OUTER_OFFSET_M),
            ],
            "near_edge_lateral_distance_m": float(
                LATERAL_REFERENCE_OFFSET_M
            ),
            "near_edge_lateral_distances_m": [
                float(LATERAL_REFERENCE_OFFSET_M),
                float(LATERAL_REFERENCE_OUTER_OFFSET_M),
            ],
            "near_edge_lateral_labels": [
                "0.4m",
                "0.2m",
                "0.2m",
                "0.4m",
            ],
            "near_edge_outer_label_outset_m": float(
                NEAR_EDGE_OUTER_LABEL_OUTSET_M
            ),
            "label_color_rgb": list(LABEL_COLOR_RGB),
            "label_outline_enabled": False,
            "label_stroke_width_px": int(LABEL_STROKE_WIDTH_PX),
            "v2_margin_rails_enabled": False,
            "yellow_overlay_enabled": False,
            "yellow_overlay_semantics": "removed",
            "margin_rails_depth_occluded": False,
            "corners_world": corners_world.round(6).tolist(),
            "corners_uv": corners_uv.round(2).tolist(),
            "rasterized_pixel_count": int(np.count_nonzero(path_mask)),
            "visible_pixel_count": int(np.count_nonzero(visible_mask)),
            "hud_rasterized_pixel_count": int(np.count_nonzero(hud_mask)),
            "hud_visible_pixel_count": int(
                np.count_nonzero(hud_visible_mask)
            ),
            "depth_occlusion": scene_depth is not None,
            "depth_occlude_eps_m": float(DEPTH_OCCLUDE_EPS_M),
            "segmentation_used": False,
            "scene_geometry_used": False,
            "simulator_state_used": False,
        }
        if eef_near_0_1m:
            overlay_payload["eef_near_0.1m"] = eef_near_0_1m
        nearby_object_warning = compose_nearby_object_warning(
            chassis_forward_2m,
            eef_near_0_1m,
        )
        if nearby_object_warning:
            overlay_payload["nearby_object_warning"] = nearby_object_warning
        return output_bgr, overlay_payload
    except Exception as exc:
        return None, {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def render_base_forward_path_overlay_v2(
    *,
    rgb_path: str,
    depth_linear: Optional[np.ndarray],
    camera: Dict[str, Any],
    robot: Dict[str, Any],
    output_path: str,
) -> Dict[str, Any]:
    """Render the frozen v2 metric base path HUD from legal observations."""
    try:
        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"failed to read RGB image: {rgb_path}")
        output_bgr, result = render_base_forward_path_overlay_frame_v2(
            image_bgr=image_bgr,
            depth_linear=depth_linear,
            camera=camera,
            robot=robot,
        )
        if output_bgr is None or not bool(result.get("ok")):
            return {
                **result,
                "path": output_path,
            }
        os.makedirs(
            os.path.dirname(os.path.abspath(output_path)),
            exist_ok=True,
        )
        if not cv2.imwrite(output_path, output_bgr):
            raise RuntimeError(f"failed to write {output_path}")
        return {
            **result,
            "path": output_path,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "path": output_path,
        }


__all__ = [
    "BASE_FRONT_OFFSET_M",
    "CHASSIS_FORWARD_2M_CLEAR",
    "CHASSIS_FORWARD_HINT_M",
    "CHASSIS_HELD_DEPTH_CONT_M",
    "CHASSIS_HELD_GRIPPER_SEED_M",
    "CHASSIS_HELD_MAX_M",
    "DEPTH_OCCLUDE_EPS_M",
    "chassis_forward_2m_sentence",
    "compose_nearby_object_warning",
    "describe_chassis_forward_2m",
    "describe_eef_near_0_1m",
    "eef_near_sentence",
    "list_chassis_forward_2m_hits",
    "list_eef_near_hits",
    "EEF_NEAR_CLEAR",
    "EEF_NEAR_INNER_M",
    "EEF_NEAR_SPHERE_M",
    "DISTANCE_TICK_STEP_M",
    "LATERAL_REFERENCE_OFFSET_M",
    "LATERAL_REFERENCE_OUTER_OFFSET_M",
    "LABEL_COLOR_RGB",
    "LABEL_STROKE_WIDTH_PX",
    "NEAR_EDGE_OUTER_LABEL_OUTSET_M",
    "PATH_FADE_START_M",
    "PATH_HEIGHT_OFFSET_M",
    "PATH_LENGTH_M",
    "PATH_END_X_M",
    "PATH_SIDE_INSET_M",
    "PATH_SIDE_MARGIN_M",
    "PATH_START_X_M",
    "PATH_WIDTH_M",
    "PATH_Y_CENTER_M",
    "render_base_forward_path_overlay_frame_v2",
    "render_base_forward_path_overlay_v2",
]
