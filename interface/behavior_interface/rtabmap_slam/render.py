"""Heading-up minimap renderer for RTAB-Map occupancy and height layers."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import os

from PIL import Image, ImageDraw, ImageFont

from .official import SE2Pose
from .protocol import MapResult, PoseRecord


UNKNOWN_RGBA = (214, 216, 220, 220)
FREE_RGBA = (255, 255, 255, 248)
FURNITURE_RGBA = (168, 173, 180, 245)
WALL_RGBA = (18, 22, 26, 255)
TRAIL_RGBA = (25, 99, 220, 255)
NAVIGATION_ROUTE_RGBA = (238, 126, 132, 224)
CONE_RGBA = (151, 211, 250, 92)
START_RGBA = (15, 72, 166, 255)
AGENT_RGBA = (27, 104, 226, 255)
# 与 EgoMap 地名同色：菱形+描边字，避免被轨迹/机器人蓝点盖掉
PLACE_RGBA = (12, 48, 150, 255)
PLACE_HALO_RGBA = (230, 232, 236, 240)
AUTO_VIEW_MIN_SPAN_M = 16.0
AUTO_VIEW_PADDING_M = 0.75
_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

def obstacle_layers(result: MapResult) -> tuple[np.ndarray, np.ndarray]:
    occupied = np.asarray(result.occupancy) >= 65
    low = np.asarray(result.low_obstacles, dtype=bool)
    high = np.asarray(result.high_obstacles, dtype=bool)
    walls = low
    detail = (occupied | high) & ~walls
    return walls, detail


def _sample_base_layer(
    result: MapResult,
    pose: SE2Pose,
    size_px: int,
    span_m: float,
) -> np.ndarray:
    output = np.empty((size_px, size_px, 4), dtype=np.uint8)
    output[...] = UNKNOWN_RGBA
    if result.occupancy.size == 0:
        return output
    scale = float(size_px) / float(span_m)
    center = (float(size_px) - 1.0) * 0.5
    py, px = np.indices((size_px, size_px), dtype=np.float64)
    up = (center - py) / scale
    left = (center - px) / scale
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    world_x = pose.x_m + cosine * up - sine * left
    world_y = pose.y_m + sine * up + cosine * left
    columns = np.floor((world_x - result.x_min_m) / result.cell_size_m).astype(np.int64)
    rows = np.floor((world_y - result.y_min_m) / result.cell_size_m).astype(np.int64)
    valid = (
        (rows >= 0)
        & (columns >= 0)
        & (rows < result.occupancy.shape[0])
        & (columns < result.occupancy.shape[1])
    )
    sampled_occupancy = np.full((size_px, size_px), -1, dtype=np.int8)
    sampled_occupancy[valid] = result.occupancy[rows[valid], columns[valid]]
    walls, detail = obstacle_layers(result)
    sampled_walls = np.zeros((size_px, size_px), dtype=bool)
    sampled_detail = np.zeros((size_px, size_px), dtype=bool)
    sampled_walls[valid] = walls[rows[valid], columns[valid]]
    sampled_detail[valid] = detail[rows[valid], columns[valid]]
    output[sampled_occupancy == 0] = FREE_RGBA
    output[sampled_detail] = FURNITURE_RGBA
    output[sampled_walls] = WALL_RGBA
    return output


def render_floorplan(
    result: MapResult,
    *,
    size_px: int = 720,
    padding_m: float = 0.75,
) -> Image.Image:
    """渲染自动取景、无轨迹遮挡的静态平面图。"""

    if size_px < 64 or not math.isfinite(padding_m) or padding_m < 0.0:
        raise ValueError("invalid floorplan dimensions")
    center, span_m = _floorplan_view(result, padding_m=padding_m)
    return Image.fromarray(
        _sample_base_layer(result, center, int(size_px), float(span_m)),
        mode="RGBA",
    )


def _floorplan_view(
    result: MapResult,
    *,
    padding_m: float = 0.75,
) -> tuple[SE2Pose, float]:
    """返回未经墙线旋转校正的地图坐标取景范围。"""

    if not math.isfinite(padding_m) or padding_m < 0.0:
        raise ValueError("invalid floorplan padding")
    known = (
        (np.asarray(result.occupancy) >= 0)
        | np.asarray(result.low_obstacles, dtype=bool)
        | np.asarray(result.high_obstacles, dtype=bool)
    )
    rows, columns = np.nonzero(known)
    if rows.size == 0:
        center = result.current_pose
        span_m = 4.0
    else:
        min_row, max_row = int(rows.min()), int(rows.max())
        min_column, max_column = int(columns.min()), int(columns.max())
        center_x = result.x_min_m + (
            0.5 * (min_column + max_column) + 0.5
        ) * result.cell_size_m
        center_y = result.y_min_m + (
            0.5 * (min_row + max_row) + 0.5
        ) * result.cell_size_m
        center = SE2Pose(center_x, center_y, 0.0)
        world_x = result.x_min_m + (columns.astype(np.float64) + 0.5) * result.cell_size_m
        world_y = result.y_min_m + (rows.astype(np.float64) + 0.5) * result.cell_size_m
        dx = world_x - center_x
        dy = world_y - center_y
        up = dx
        left_offset = dy
        span_m = max(
            4.0,
            float(up.max() - up.min()) + result.cell_size_m + 2.0 * padding_m,
            float(left_offset.max() - left_offset.min())
            + result.cell_size_m
            + 2.0 * padding_m,
        )
    return center, float(span_m)


def _to_pixel(
    x_m: float,
    y_m: float,
    pose: SE2Pose,
    size_px: int,
    span_m: float,
) -> tuple[float, float]:
    dx = float(x_m) - pose.x_m
    dy = float(y_m) - pose.y_m
    up = math.cos(pose.yaw_rad) * dx + math.sin(pose.yaw_rad) * dy
    left = -math.sin(pose.yaw_rad) * dx + math.cos(pose.yaw_rad) * dy
    scale = float(size_px) / float(span_m)
    center = (float(size_px) - 1.0) * 0.5
    return center - left * scale, center - up * scale


def _auto_map_view(
    result: MapResult,
    *,
    yaw_rad: float,
    padding_m: float = AUTO_VIEW_PADDING_M,
    minimum_span_m: float = AUTO_VIEW_MIN_SPAN_M,
    extra_points_xy_m: Sequence[tuple[float, float]] = (),
) -> tuple[SE2Pose, float]:
    """Fit known cells, agent and optional vector overlays in one square."""

    values = (yaw_rad, padding_m, minimum_span_m, result.cell_size_m)
    if (
        not all(math.isfinite(float(value)) for value in values)
        or padding_m < 0.0
        or minimum_span_m <= 0.0
        or result.cell_size_m <= 0.0
    ):
        raise ValueError("invalid automatic minimap view")

    agent = result.current_pose
    cosine = math.cos(float(yaw_rad))
    sine = math.sin(float(yaw_rad))
    agent_up = cosine * agent.x_m + sine * agent.y_m
    agent_left = -sine * agent.x_m + cosine * agent.y_m
    known = (
        (np.asarray(result.occupancy) >= 0)
        | np.asarray(result.low_obstacles, dtype=bool)
        | np.asarray(result.high_obstacles, dtype=bool)
    )
    rows, columns = np.nonzero(known)
    up_values = [float(agent_up)]
    left_values = [float(agent_left)]
    if rows.size:
        world_x = result.x_min_m + (
            columns.astype(np.float64) + 0.5
        ) * result.cell_size_m
        world_y = result.y_min_m + (
            rows.astype(np.float64) + 0.5
        ) * result.cell_size_m
        up = cosine * world_x + sine * world_y
        left = -sine * world_x + cosine * world_y
        cell_half_extent = (
            0.5 * result.cell_size_m * (abs(cosine) + abs(sine))
        )
        up_values.extend(
            [float(up.min()) - cell_half_extent, float(up.max()) + cell_half_extent]
        )
        left_values.extend(
            [
                float(left.min()) - cell_half_extent,
                float(left.max()) + cell_half_extent,
            ]
        )
    for x_m, y_m in extra_points_xy_m:
        up_values.append(cosine * float(x_m) + sine * float(y_m))
        left_values.append(-sine * float(x_m) + cosine * float(y_m))

    min_up = min(up_values)
    max_up = max(up_values)
    min_left = min(left_values)
    max_left = max(left_values)
    center_up = 0.5 * (min_up + max_up)
    center_left = 0.5 * (min_left + max_left)
    center_x = cosine * center_up - sine * center_left
    center_y = sine * center_up + cosine * center_left
    span_m = max(
        float(minimum_span_m),
        max_up - min_up + 2.0 * padding_m,
        max_left - min_left + 2.0 * padding_m,
    )
    return SE2Pose(center_x, center_y, float(yaw_rad)), float(span_m)


def render_heading_up(
    result: MapResult,
    *,
    size_px: int = 490,
    span_m: Optional[float] = None,
    heading_up: bool = True,
    fov_deg: float = 99.2,
    cone_range_m: float = 3.5,
    places: Optional[Iterable[tuple[float, float, str]]] = None,
    route_xy_m: Optional[Iterable[tuple[float, float]]] = None,
) -> Image.Image:
    """Render the complete known map while preserving the requested orientation.

    The live default fits the known grid instead of pinning the agent to the
    canvas center. Passing ``span_m`` retains the old fixed, agent-centered view
    for reproducible offline diagnostics.
    """

    if size_px < 64 or (
        span_m is not None
        and (not math.isfinite(float(span_m)) or float(span_m) <= 0.0)
    ):
        raise ValueError("invalid minimap dimensions")
    try:
        raw_route_points = () if route_xy_m is None else tuple(route_xy_m)
        route_points = tuple(
            (float(point[0]), float(point[1]))
            for point in raw_route_points
            if len(point) == 2
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("invalid navigation route coordinates") from exc
    if (
        route_xy_m is not None and len(route_points) != len(raw_route_points)
    ) or not all(
        math.isfinite(value) for point in route_points for value in point
    ):
        raise ValueError("invalid navigation route coordinates")
    agent_pose = result.current_pose
    view_yaw = agent_pose.yaw_rad if heading_up else 0.0
    if span_m is None:
        view_pose, render_span_m = _auto_map_view(
            result,
            yaw_rad=view_yaw,
            extra_points_xy_m=route_points,
        )
    else:
        view_pose = SE2Pose(agent_pose.x_m, agent_pose.y_m, view_yaw)
        render_span_m = float(span_m)
    base = _sample_base_layer(result, view_pose, int(size_px), render_span_m)
    image = Image.fromarray(base, mode="RGBA")
    cone_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(cone_overlay, "RGBA")
    center = (float(size_px) - 1.0) * 0.5
    scale = float(size_px) / render_span_m
    agent_px, agent_py = _to_pixel(
        agent_pose.x_m,
        agent_pose.y_m,
        view_pose,
        int(size_px),
        render_span_m,
    )
    relative_heading = agent_pose.yaw_rad - view_pose.yaw_rad
    half_angle = math.radians(float(fov_deg) * 0.5)
    cone_radius = min(float(cone_range_m) * scale, center * 0.95)
    cone = [
        (agent_px, agent_py),
        (
            agent_px - math.sin(relative_heading - half_angle) * cone_radius,
            agent_py - math.cos(relative_heading - half_angle) * cone_radius,
        ),
        (
            agent_px - math.sin(relative_heading + half_angle) * cone_radius,
            agent_py - math.cos(relative_heading + half_angle) * cone_radius,
        ),
    ]
    draw.polygon(cone, fill=CONE_RGBA)
    cone_pixels = np.asarray(cone_overlay).copy()
    obstacle_pixels = np.all(base == WALL_RGBA, axis=2) | np.all(base == FURNITURE_RGBA, axis=2)
    cone_pixels[obstacle_pixels] = 0
    image.alpha_composite(Image.fromarray(cone_pixels, mode="RGBA"))

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    if len(route_points) >= 2:
        route_pixels = [
            _to_pixel(x_m, y_m, view_pose, size_px, render_span_m)
            for x_m, y_m in route_points
        ]
        draw.line(
            route_pixels,
            fill=NAVIGATION_ROUTE_RGBA,
            width=max(3, int(round(size_px / 150.0))),
            joint="curve",
        )

    points = [
        _to_pixel(item.x_m, item.y_m, view_pose, size_px, render_span_m)
        for item in result.poses
    ]
    if len(points) >= 2:
        draw.line(points, fill=TRAIL_RGBA, width=max(2, int(round(size_px / 245.0))), joint="curve")
    if points:
        sx, sy = points[0]
        radius = max(4, int(round(size_px / 100.0)))
        draw.ellipse(
            (sx - radius, sy - radius, sx + radius, sy + radius),
            fill=START_RGBA,
            outline=WALL_RGBA,
            width=1,
        )

    radius = max(5, int(round(size_px / 85.0)))
    draw.ellipse(
        (
            agent_px - radius,
            agent_py - radius,
            agent_px + radius,
            agent_py + radius,
        ),
        fill=AGENT_RGBA,
        outline=WALL_RGBA,
        width=max(1, int(round(size_px / 300.0))),
    )
    # 地名画在机器人之后，脚下标记和标签都不会被蓝点盖住
    _draw_places(draw, view_pose, int(size_px), render_span_m, places)
    image.alpha_composite(overlay)
    return image


def _place_font_size(canvas: int) -> int:
    """按画布比例取字号：叠到 head 右上角后字还要能认。"""
    return max(22, int(round(int(canvas) * 0.055)))


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _text_with_halo(draw: ImageDraw.ImageDraw, xy, text: str, fill, font) -> None:
    x, y = xy
    for ox, oy in (
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-1, -1), (1, -1), (-1, 1), (1, 1),
        (-1, 0), (1, 0), (0, -1), (0, 1),
    ):
        draw.text((x + ox, y + oy), text, fill=PLACE_HALO_RGBA, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def _clamp_label_xy(
    draw: ImageDraw.ImageDraw,
    font,
    text: str,
    x: float,
    y: float,
    canvas: int,
) -> tuple[float, float]:
    """把标签留在画布里，避免「garage」被裁成半截。"""
    try:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    except Exception:
        return x, y
    pad = 4
    dx = 0.0
    dy = 0.0
    if right > canvas - pad:
        dx = (canvas - pad) - right
    if left + dx < pad:
        dx = pad - left
    if top < pad:
        dy = pad - top
    if bottom + dy > canvas - pad:
        dy = (canvas - pad) - bottom
    return x + dx, y + dy


def _draw_places(
    draw: ImageDraw.ImageDraw,
    pose: SE2Pose,
    size_px: int,
    span_m: float,
    places: Optional[Iterable[tuple[float, float, str]]],
) -> None:
    """自己标的地点：菱形 + 描边字，这是小地图上唯一保留的文字。"""
    if not places:
        return
    font = _load_font(_place_font_size(size_px))
    radius = max(7, int(round(size_px * 0.012)))
    for x_m, y_m, name in places:
        px, py = _to_pixel(float(x_m), float(y_m), pose, size_px, span_m)
        draw.polygon(
            [
                (px, py - radius),
                (px + radius, py),
                (px, py + radius),
                (px - radius, py),
            ],
            fill=PLACE_RGBA,
            outline=WALL_RGBA,
        )
        label = str(name or "").strip()
        if not label:
            continue
        tx, ty = _clamp_label_xy(
            draw, font, label, px + radius + 4, py - radius - 2, size_px
        )
        _text_with_halo(draw, (tx, ty), label, PLACE_RGBA, font)
