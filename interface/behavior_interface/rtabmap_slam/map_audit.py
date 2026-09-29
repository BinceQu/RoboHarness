"""独立检查结构地图，并在不修改地图的前提下叠加完整轨迹。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .official import SE2Pose
from .protocol import MapResult
from .render import (
    TRAIL_RGBA,
    _floorplan_view,
    _to_pixel,
    obstacle_layers,
    render_floorplan,
)


BOUNDARY_RADIUS_CELLS = 2
TRAJECTORY_CLEARANCE_CELLS = 1
MIN_BOUNDARY_WALL_COVERAGE = 0.75
MIN_WALL_AXIS_COHERENCE = 0.70
MAX_WALL_AXIS_P90_ERROR_DEG = 10.0


@dataclass(frozen=True)
class MapAudit:
    passed: bool
    reasons: tuple[str, ...]
    trajectory_pose_count: int
    trajectory_poses_in_grid: int
    trajectory_path_cells: int
    trajectory_free_ratio: float
    trajectory_wall_intersections: int
    trajectory_detail_intersections: int
    trajectory_wall_clearance_intersections: int
    wall_cells: int
    free_boundary_cells: int
    boundary_wall_coverage: float
    wall_axis_coherence: float
    wall_axis_p90_error_deg: float
    map_was_modified: bool = False
    forbidden_inputs_consumed: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["schema_version"] = "behavior.rtabmap_map_audit.v2"
        return output


def _grid_point(result: MapResult, pose: SE2Pose) -> tuple[int, int]:
    column = int(math.floor((pose.x_m - result.x_min_m) / result.cell_size_m))
    row = int(math.floor((pose.y_m - result.y_min_m) / result.cell_size_m))
    return column, row


def _trajectory_mask(
    result: MapResult, poses: Sequence[SE2Pose]
) -> tuple[np.ndarray, int]:
    height, width = result.occupancy.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    points = [_grid_point(result, pose) for pose in poses]
    in_grid = sum(0 <= x < width and 0 <= y < height for x, y in points)
    if len(points) == 1:
        x, y = points[0]
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = 1
    for first, second in zip(points, points[1:]):
        visible, begin, end = cv2.clipLine((0, 0, width, height), first, second)
        if visible:
            cv2.line(mask, begin, end, 1, thickness=1, lineType=cv2.LINE_8)
    return mask.astype(bool), in_grid


def _wall_axis_metrics(
    walls: np.ndarray, cell_size_m: float
) -> tuple[float, float]:
    minimum = max(4, int(math.ceil(0.50 / cell_size_m)))
    lines = cv2.HoughLinesP(
        np.asarray(walls, dtype=np.uint8),
        1.0,
        math.pi / 180.0,
        threshold=minimum,
        minLineLength=minimum,
        maxLineGap=max(1, minimum // 2),
    )
    if lines is None or not len(lines):
        return 0.0, float("inf")
    angles = []
    weights = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < minimum:
            continue
        angles.append(math.atan2(dy, dx))
        weights.append(length)
    if not angles:
        return 0.0, float("inf")
    angle_array = np.asarray(angles, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    vector = np.sum(weight_array * np.exp(4.0j * angle_array))
    coherence = float(abs(vector) / max(float(weight_array.sum()), 1e-12))
    axis = math.atan2(float(vector.imag), float(vector.real)) / 4.0
    errors = np.abs(
        (np.degrees(angle_array - axis) + 45.0) % 90.0 - 45.0
    )
    order = np.argsort(errors)
    cumulative = np.cumsum(weight_array[order])
    target = 0.90 * float(weight_array.sum())
    index = min(len(order) - 1, int(np.searchsorted(cumulative, target)))
    return coherence, float(errors[order[index]])


def audit_map(result: MapResult, poses: Sequence[SE2Pose]) -> MapAudit:
    """只读检查输出栅格；轨迹绝不反馈到墙、自由区或位姿。"""

    occupancy = np.asarray(result.occupancy)
    if occupancy.ndim != 2:
        raise ValueError("occupancy must be a two-dimensional grid")
    walls, detail = obstacle_layers(result)
    native_free = occupancy == 0
    visible_free = native_free & ~walls & ~detail
    unknown = occupancy < 0
    route, poses_in_grid = _trajectory_mask(result, poses)
    route_cells = int(np.count_nonzero(route))
    route_free = int(np.count_nonzero(route & visible_free))
    route_free_ratio = 0.0 if route_cells == 0 else route_free / route_cells
    wall_intersections = int(np.count_nonzero(route & walls))
    detail_intersections = int(np.count_nonzero(route & detail))
    clearance_kernel = np.ones(
        (2 * TRAJECTORY_CLEARANCE_CELLS + 1,) * 2, dtype=np.uint8
    )
    wall_clearance = cv2.dilate(
        walls.astype(np.uint8), clearance_kernel
    ).astype(bool)
    clearance_intersections = int(np.count_nonzero(route & wall_clearance))

    boundary_kernel = np.ones(
        (2 * BOUNDARY_RADIUS_CELLS + 1,) * 2, dtype=np.uint8
    )
    near_unknown = cv2.dilate(
        unknown.astype(np.uint8), boundary_kernel
    ).astype(bool)
    free_boundary = native_free & near_unknown
    wall_support = cv2.dilate(
        walls.astype(np.uint8), boundary_kernel
    ).astype(bool)
    boundary_cells = int(np.count_nonzero(free_boundary))
    boundary_coverage = (
        0.0
        if boundary_cells == 0
        else int(np.count_nonzero(free_boundary & wall_support)) / boundary_cells
    )
    axis_coherence, axis_p90 = _wall_axis_metrics(walls, result.cell_size_m)

    reasons = []
    if not poses:
        reasons.append("trajectory is empty")
    if poses_in_grid != len(poses):
        reasons.append(
            f"trajectory poses in grid {poses_in_grid}/{len(poses)}"
        )
    if route_free_ratio < 1.0:
        reasons.append(f"trajectory free ratio {route_free_ratio:.6f} < 1.0")
    if wall_intersections:
        reasons.append(f"trajectory intersects {wall_intersections} wall cells")
    if detail_intersections:
        reasons.append(
            f"trajectory intersects {detail_intersections} gray detail cells"
        )
    if clearance_intersections:
        reasons.append(
            f"trajectory enters wall clearance at {clearance_intersections} cells"
        )
    if boundary_coverage < MIN_BOUNDARY_WALL_COVERAGE:
        reasons.append(
            f"boundary wall coverage {boundary_coverage:.3f} < "
            f"{MIN_BOUNDARY_WALL_COVERAGE:.2f}"
        )
    if axis_coherence < MIN_WALL_AXIS_COHERENCE:
        reasons.append(
            f"wall axis coherence {axis_coherence:.3f} < "
            f"{MIN_WALL_AXIS_COHERENCE:.2f}"
        )
    if axis_p90 > MAX_WALL_AXIS_P90_ERROR_DEG:
        reasons.append(
            f"wall axis p90 error {axis_p90:.2f}deg > "
            f"{MAX_WALL_AXIS_P90_ERROR_DEG:.1f}deg"
        )
    return MapAudit(
        passed=not reasons,
        reasons=tuple(reasons),
        trajectory_pose_count=len(poses),
        trajectory_poses_in_grid=poses_in_grid,
        trajectory_path_cells=route_cells,
        trajectory_free_ratio=route_free_ratio,
        trajectory_wall_intersections=wall_intersections,
        trajectory_detail_intersections=detail_intersections,
        trajectory_wall_clearance_intersections=clearance_intersections,
        wall_cells=int(np.count_nonzero(walls)),
        free_boundary_cells=boundary_cells,
        boundary_wall_coverage=boundary_coverage,
        wall_axis_coherence=axis_coherence,
        wall_axis_p90_error_deg=axis_p90,
    )


def render_trajectory_audit(
    result: MapResult,
    poses: Sequence[SE2Pose],
    *,
    size_px: int = 720,
    padding_m: float = 0.75,
) -> Image.Image:
    """先渲染只读地图，再画蓝色轨迹；此函数不返回或修改任何地图层。"""

    image = render_floorplan(
        result, size_px=int(size_px), padding_m=float(padding_m)
    )
    if not poses:
        return image
    center, span_m = _floorplan_view(result, padding_m=float(padding_m))
    points = [
        _to_pixel(pose.x_m, pose.y_m, center, int(size_px), span_m)
        for pose in poses
    ]
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    if len(points) >= 2:
        draw.line(
            points,
            fill=TRAIL_RGBA,
            width=max(2, int(round(size_px / 360.0))),
            joint="curve",
        )
    else:
        x, y = points[0]
        draw.point((x, y), fill=TRAIL_RGBA)
    image.alpha_composite(overlay)
    return image


__all__ = ["MapAudit", "audit_map", "render_trajectory_audit"]
