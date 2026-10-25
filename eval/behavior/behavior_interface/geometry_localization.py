"""跨场景 GPU 几何闭环验证。

外观特征只负责指出“可能回到了哪张历史子图”。本模块只看深度几何，
用多个滚动时间窗独立验证该候选是否具有唯一、稳定的刚体校正峰。搜索主体
在 CUDA 上批量完成；CUDA 不可用时拒绝校正，不启动高开销 CPU 回退。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

import numpy as np


# 这些门槛由四组随机合成几何固定：48 个唯一结构均通过，12 个单墙、
# 重复角和走廊反例均被拒绝。它们是搜索分布上的无量纲量，不含任务或房型。
UNIQUE_MIN_PROMINENCE_ROBUST = 1.20
UNIQUE_MIN_TOP_CLUSTER_RATIO = 0.45
CONSENSUS_MIN_WINDOWS = 2
CONSENSUS_MAX_TRANSLATION_M = 0.10
CONSENSUS_MAX_YAW_DEG = 1.5
DEFAULT_WINDOWS = (16, 32, 48)

# 单次滚动窗只能证明一次“看见了旧结构”。修改全局位姿还要求两个时间上
# 不重叠、视角有基线的观测段，经相邻里程传播到同一时刻后仍指向同一 SE(2)
# 位姿。尺度来自 5cm 栅格和 RGB-D 视角变化，不依赖房型、任务或录制帧号。
TEMPORAL_POSE_MIN_EPISODES = 2
TEMPORAL_POSE_MAX_TRANSLATION_M = 0.15
TEMPORAL_POSE_MAX_YAW_DEG = 2.5
TEMPORAL_POSE_MIN_BASELINE_M = 0.25
TEMPORAL_POSE_MIN_BASELINE_YAW_DEG = 8.0
TEMPORAL_POSE_MIN_GAP_FRAMES = min(DEFAULT_WINDOWS)
TEMPORAL_POSE_MAX_EVIDENCE = 256

# 两种传感器给出的校正必须彼此一致。视觉先选候选，深度负责否决歧义；
# 通过后才融合二者，任何单一传感器都不能独自写回。
VISUAL_GEOMETRY_MAX_TRANSLATION_DISAGREEMENT_M = 0.12
VISUAL_GEOMETRY_MAX_YAW_DISAGREEMENT_DEG = 2.0

# 当占据几何在走廊切向退化时，低协方差 RGB-D 特征可以补足该自由度；
# 最长滚动窗仍必须证明这个视觉位姿落在几何高似然平台内。阈值来自 5cm
# 栅格分辨能力、图像覆盖和 SE(2) 信息矩阵，不依赖房型或轨迹标注。
VISUAL_SUPPORT_MIN_INDEPENDENT_CANDIDATES = 2
VISUAL_SUPPORT_MAX_TRANSLATION_STD_M = 0.025
VISUAL_SUPPORT_MAX_YAW_STD_DEG = 0.5
VISUAL_SUPPORT_MIN_MINOR_SPAN_M = 0.20
VISUAL_SUPPORT_MIN_PIXEL_COVERAGE_X = 0.15
VISUAL_SUPPORT_MIN_PIXEL_COVERAGE_Y = 0.12
VISUAL_SUPPORT_MAX_SCORE_DROP_ROBUST = 1.0
VISUAL_SUPPORT_MIN_GEOMETRY_SCORE = 0.0

DEFAULT_SEARCH_SPAN_M = 0.60
DEFAULT_SEARCH_STEP_M = 0.05
DEFAULT_SEARCH_SPAN_DEG = 8.0
DEFAULT_SEARCH_STEP_DEG = 1.0
DEFAULT_GPU_BATCH = 128
GEOMETRY_DEVICE_ENV = "BEHAVIOR_SLAM_GEOMETRY_DEVICE"

# 局部几何候选只应和它能实际看见的冻结结构比较。整张地图里远离候选的
# 墙不会提供支持，却会在反向冲突项中稀释分数。边距覆盖平移搜索、绕机器人
# 旋转时的最大弦长和两个栅格的离散误差；它只由传感器视域和搜索尺度决定。
TARGET_SUPPORT_GRID_MARGIN_CELLS = 2

# 相关搜索只负责落入正确吸引域；最终位姿由 GPU 刚体细化给出。限制细化
# 跨度可以防止错误粗峰借 ICP 沿重复墙面滑到另一个位置。
ICP_ITERATIONS = 6
ICP_MAX_SOURCE_POINTS = 1024
ICP_MAX_TARGET_POINTS = 2048
ICP_MAX_CORRESPONDENCE_M = 0.20
ICP_TRIM_QUANTILE = 0.75
ICP_MIN_CORRESPONDENCES = 24
ICP_NORMAL_NEIGHBORS = 24
ICP_MIN_NORMAL_ANISOTROPY = 0.35
ICP_ORIENTATION_STEP_DEG = 0.25
ICP_ORIENTATION_BLUR_DEG = 1.0
ICP_MAX_INCREMENT_TRANSLATION_M = 0.08
ICP_MAX_INCREMENT_YAW_DEG = 1.5
ICP_MAX_REFINEMENT_TRANSLATION_M = 0.15
ICP_MAX_REFINEMENT_YAW_DEG = 4.0


def wrap_deg(value: float) -> float:
    """把角度包到 [-180, 180)。"""
    return (float(value) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class GridSpec:
    """与占据栅格一致的世界坐标画布。"""

    resolution_m: float
    half_span_m: float

    @property
    def size(self) -> int:
        return int(round(2.0 * float(self.half_span_m) / float(self.resolution_m)))


@dataclass(frozen=True)
class GeometryConfig:
    """只含传感器和搜索尺度，不含场景、任务或帧号。"""

    windows: Tuple[int, ...] = DEFAULT_WINDOWS
    search_span_m: float = DEFAULT_SEARCH_SPAN_M
    search_step_m: float = DEFAULT_SEARCH_STEP_M
    search_span_deg: float = DEFAULT_SEARCH_SPAN_DEG
    search_step_deg: float = DEFAULT_SEARCH_STEP_DEG
    gpu_batch: int = DEFAULT_GPU_BATCH
    min_prominence_robust: float = UNIQUE_MIN_PROMINENCE_ROBUST
    min_top_cluster_ratio: float = UNIQUE_MIN_TOP_CLUSTER_RATIO
    consensus_min_windows: int = CONSENSUS_MIN_WINDOWS
    consensus_max_translation_m: float = CONSENSUS_MAX_TRANSLATION_M
    consensus_max_yaw_deg: float = CONSENSUS_MAX_YAW_DEG
    visual_max_translation_disagreement_m: float = (
        VISUAL_GEOMETRY_MAX_TRANSLATION_DISAGREEMENT_M
    )
    visual_max_yaw_disagreement_deg: float = (
        VISUAL_GEOMETRY_MAX_YAW_DISAGREEMENT_DEG
    )
    visual_support_min_independent_candidates: int = (
        VISUAL_SUPPORT_MIN_INDEPENDENT_CANDIDATES
    )
    visual_support_max_translation_std_m: float = (
        VISUAL_SUPPORT_MAX_TRANSLATION_STD_M
    )
    visual_support_max_yaw_std_deg: float = VISUAL_SUPPORT_MAX_YAW_STD_DEG
    visual_support_min_minor_span_m: float = VISUAL_SUPPORT_MIN_MINOR_SPAN_M
    visual_support_min_pixel_coverage_x: float = (
        VISUAL_SUPPORT_MIN_PIXEL_COVERAGE_X
    )
    visual_support_min_pixel_coverage_y: float = (
        VISUAL_SUPPORT_MIN_PIXEL_COVERAGE_Y
    )
    visual_support_max_score_drop_robust: float = (
        VISUAL_SUPPORT_MAX_SCORE_DROP_ROBUST
    )
    visual_support_min_geometry_score: float = (
        VISUAL_SUPPORT_MIN_GEOMETRY_SCORE
    )
    icp_iterations: int = ICP_ITERATIONS
    icp_max_source_points: int = ICP_MAX_SOURCE_POINTS
    icp_max_target_points: int = ICP_MAX_TARGET_POINTS
    icp_max_correspondence_m: float = ICP_MAX_CORRESPONDENCE_M
    icp_trim_quantile: float = ICP_TRIM_QUANTILE
    icp_min_correspondences: int = ICP_MIN_CORRESPONDENCES
    icp_normal_neighbors: int = ICP_NORMAL_NEIGHBORS
    icp_min_normal_anisotropy: float = ICP_MIN_NORMAL_ANISOTROPY
    icp_orientation_step_deg: float = ICP_ORIENTATION_STEP_DEG
    icp_orientation_blur_deg: float = ICP_ORIENTATION_BLUR_DEG
    icp_max_increment_translation_m: float = ICP_MAX_INCREMENT_TRANSLATION_M
    icp_max_increment_yaw_deg: float = ICP_MAX_INCREMENT_YAW_DEG
    icp_max_refinement_translation_m: float = ICP_MAX_REFINEMENT_TRANSLATION_M
    icp_max_refinement_yaw_deg: float = ICP_MAX_REFINEMENT_YAW_DEG


@dataclass(frozen=True)
class TemporalPoseConfig:
    """跨时段位姿授权门；参数只描述观测独立性和 SE(2) 一致性。"""

    min_episodes: int = TEMPORAL_POSE_MIN_EPISODES
    max_translation_m: float = TEMPORAL_POSE_MAX_TRANSLATION_M
    max_yaw_deg: float = TEMPORAL_POSE_MAX_YAW_DEG
    min_baseline_m: float = TEMPORAL_POSE_MIN_BASELINE_M
    min_baseline_yaw_deg: float = TEMPORAL_POSE_MIN_BASELINE_YAW_DEG
    min_gap_frames: int = TEMPORAL_POSE_MIN_GAP_FRAMES
    max_evidence: int = TEMPORAL_POSE_MAX_EVIDENCE


@dataclass(frozen=True)
class TemporalPoseEvidence:
    """一次视觉和深度共同通过后的只读证据。"""

    serial: int
    frame_index: int
    first_frame: int
    last_frame: int
    target_group: Tuple[int, ...]
    predicted_pose: Tuple[float, float, float]
    corrected_pose: Tuple[float, float, float]


@dataclass(frozen=True)
class ColumnScan:
    """一帧深度中的竖直墙柱和每个方位最近的射线端点。"""

    wall: np.ndarray
    rays: np.ndarray


def _empty_xy() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _unique_rows(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.empty((0,), dtype=np.int64)
    _, indices = np.unique(values, axis=0, return_index=True)
    return np.sort(indices)


def unique_cells(points: np.ndarray, cell_m: float) -> np.ndarray:
    """每个米制网格只保留一个点，控制 GPU 内存且不改变覆盖范围。"""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return _empty_xy()
    bins = np.rint(points / float(cell_m)).astype(np.int32)
    _, indices = np.unique(bins, axis=0, return_index=True)
    return points[np.sort(indices)].astype(np.float32, copy=False)


def crop_target_to_source_support(
    target_wall_points: np.ndarray,
    target_free_points: np.ndarray,
    source_wall: np.ndarray,
    source_free: np.ndarray,
    *,
    pivot: Tuple[float, float],
    config: GeometryConfig,
    resolution_m: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """把冻结目标裁到当前滚动观测可触达的米制范围。

    地点候选已经把 ``source_*`` 预对齐到历史位置。这里不选择候选，也不
    使用任务先验，只去掉传感器视域之外的目标点。所有候选仍由调用方并列
    验证，因此重复房间会继续形成竞争假设并失败关闭。
    """
    walls = np.asarray(target_wall_points, dtype=np.float32).reshape(-1, 2)
    frees = np.asarray(target_free_points, dtype=np.float32).reshape(-1, 2)
    source_parts = [
        np.asarray(points, dtype=np.float32).reshape(-1, 2)
        for points in (source_wall, source_free)
        if len(points)
    ]
    if not source_parts:
        return walls, frees, {
            "applied": False,
            "reason": "empty_source_support",
            "wall_before": int(len(walls)),
            "wall_after": int(len(walls)),
            "free_before": int(len(frees)),
            "free_after": int(len(frees)),
        }
    support = np.concatenate(source_parts, axis=0)
    pivot_xy = np.asarray(pivot, dtype=np.float32)
    radius = float(np.max(np.linalg.norm(support - pivot_xy, axis=1)))
    yaw_rad = math.radians(abs(float(config.search_span_deg)))
    yaw_swing = 2.0 * radius * math.sin(0.5 * yaw_rad)
    margin = (
        float(config.search_span_m)
        + yaw_swing
        + TARGET_SUPPORT_GRID_MARGIN_CELLS * float(resolution_m)
    )
    lower = np.min(support, axis=0) - margin
    upper = np.max(support, axis=0) + margin

    def inside(points: np.ndarray) -> np.ndarray:
        if not len(points):
            return np.zeros((0,), dtype=bool)
        return np.all((points >= lower) & (points <= upper), axis=1)

    cropped_walls = walls[inside(walls)]
    cropped_frees = frees[inside(frees)]
    return cropped_walls, cropped_frees, {
        "applied": True,
        "reason": "sensor_support_extent",
        "margin_m": float(margin),
        "source_radius_m": radius,
        "bounds_m": [
            float(lower[0]), float(lower[1]),
            float(upper[0]), float(upper[1]),
        ],
        "wall_before": int(len(walls)),
        "wall_after": int(len(cropped_walls)),
        "free_before": int(len(frees)),
        "free_after": int(len(cropped_frees)),
    }


def transform_points(
    points: np.ndarray,
    pose: Tuple[float, float, float],
) -> np.ndarray:
    """把局部 XY 按 (x, y, yaw_deg) 刚体变换到地图系。"""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return _empty_xy()
    angle = math.radians(float(pose[2]))
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.column_stack((
        float(pose[0]) + cosine * points[:, 0] - sine * points[:, 1],
        float(pose[1]) + sine * points[:, 0] + cosine * points[:, 1],
    )).astype(np.float32)


def apply_map_correction(
    points: np.ndarray,
    pivot: Tuple[float, float],
    correction: Tuple[float, float, float],
) -> np.ndarray:
    """绕机器人当前位置施加地图系 ``(dx, dy, dyaw)`` 校正。"""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return _empty_xy()
    angle = math.radians(float(correction[2]))
    cosine, sine = math.cos(angle), math.sin(angle)
    centered = points - np.asarray(pivot, dtype=np.float32)
    return np.column_stack((
        cosine * centered[:, 0] - sine * centered[:, 1],
        sine * centered[:, 0] + cosine * centered[:, 1],
    )).astype(np.float32) + np.asarray(
        (
            float(pivot[0]) + float(correction[0]),
            float(pivot[1]) + float(correction[1]),
        ),
        dtype=np.float32,
    )


def compose_map_corrections(
    base: Tuple[float, float, float],
    residual: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """组合候选预对齐和其新枢轴处的局部残差校正。"""
    return (
        float(base[0]) + float(residual[0]),
        float(base[1]) + float(residual[1]),
        wrap_deg(float(base[2]) + float(residual[2])),
    )


def geometry_search_center(
    visual: Optional[Tuple[float, float, float]],
    config: GeometryConfig = GeometryConfig(),
) -> Tuple[Tuple[float, float, float], str]:
    """选择独立零中心搜索或由长期地点候选引导的粗到细搜索。"""
    if visual is None:
        return (0.0, 0.0, 0.0), "independent_without_visual"
    # 如果视觉候选连同允许的跨传感器分歧都落在深度搜索域内，深度应从
    # 零开始独立估计，而不是重复使用视觉先验。只有真正的大闭环才需要
    # 视觉预对齐；这个判定只依赖搜索尺度和传感器一致性门。
    independent = (
        max(abs(float(visual[0])), abs(float(visual[1])))
        + float(config.visual_max_translation_disagreement_m)
        <= float(config.search_span_m)
        and abs(wrap_deg(float(visual[2])))
        + float(config.visual_max_yaw_disagreement_deg)
        <= float(config.search_span_deg)
    )
    if independent:
        return (0.0, 0.0, 0.0), "independent_origin"
    return tuple(float(value) for value in visual), "visual_candidate"


def _search_report_in_map_frame(
    report: Dict[str, Any],
    center: Tuple[float, float, float],
    mode: str,
) -> Dict[str, Any]:
    """把零中心局部搜索结果换成原地图系校正，同时保留残差诊断。"""
    converted = dict(report)
    converted["search_center_correction"] = [float(value) for value in center]
    converted["search_center_mode"] = str(mode)
    converted["search_coordinates"] = "residual_about_search_center"
    if not report.get("valid"):
        return converted

    residual_peaks = [dict(peak) for peak in report.get("peaks", [])]
    absolute_peaks = []
    for peak in residual_peaks:
        absolute = compose_map_corrections(
            center,
            (
                float(peak["dx_m"]),
                float(peak["dy_m"]),
                float(peak["dyaw_deg"]),
            ),
        )
        absolute_peaks.append({
            **peak,
            "dx_m": absolute[0],
            "dy_m": absolute[1],
            "dyaw_deg": absolute[2],
        })
    converted["residual_peaks"] = residual_peaks
    converted["peaks"] = absolute_peaks

    coarse = report.get("coarse_best")
    if isinstance(coarse, dict):
        residual_coarse = dict(coarse)
        absolute = compose_map_corrections(
            center,
            (
                float(coarse["dx_m"]),
                float(coarse["dy_m"]),
                float(coarse["dyaw_deg"]),
            ),
        )
        converted["residual_coarse_best"] = residual_coarse
        converted["coarse_best"] = {
            **coarse,
            "dx_m": absolute[0],
            "dy_m": absolute[1],
            "dyaw_deg": absolute[2],
        }

    refinement = report.get("icp_refinement")
    if isinstance(refinement, dict):
        converted_refinement = dict(refinement)
        correction = refinement.get("correction")
        if isinstance(correction, (list, tuple)) and len(correction) == 3:
            residual = tuple(float(value) for value in correction)
            converted_refinement["residual_correction"] = list(residual)
            converted_refinement["correction"] = list(
                compose_map_corrections(center, residual)
            )
        converted["icp_refinement"] = converted_refinement
    return converted


def extract_column_scan(
    points_xyz: np.ndarray,
    *,
    column_m: float,
    min_range_m: float,
    max_range_m: float,
    obstacle_z_min_m: float,
    obstacle_z_max_m: float,
    chassis_block_z_max_m: float,
    wall_band_count: int,
    wall_band_min_run: int,
    self_clear_forward_m: float,
    self_clear_half_width_m: float,
) -> ColumnScan:
    """从机体系点云提取无语义的墙柱和自由射线端点。"""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return ColumnScan(_empty_xy(), _empty_xy())
    radial = np.hypot(points[:, 0], points[:, 1])
    usable = (
        np.isfinite(points).all(axis=1)
        & (radial >= float(min_range_m))
        & (radial <= float(max_range_m))
        & (points[:, 2] >= float(obstacle_z_min_m))
        & (points[:, 2] <= float(obstacle_z_max_m))
    )
    usable &= ~(
        (points[:, 0] <= float(self_clear_forward_m))
        & (np.abs(points[:, 1]) <= float(self_clear_half_width_m))
    )
    points = points[usable]
    if len(points) == 0:
        return ColumnScan(_empty_xy(), _empty_xy())

    xy_bins = np.rint(points[:, :2] / float(column_m)).astype(np.int32)
    low_mask = points[:, 2] <= float(chassis_block_z_max_m)
    high_mask = ~low_mask
    low_indices = _unique_rows(xy_bins[low_mask])
    high_indices = _unique_rows(xy_bins[high_mask])
    low = points[low_mask][low_indices, :2]
    high = points[high_mask][high_indices, :2]

    wall = _empty_xy()
    if np.any(low_mask):
        low_points = points[low_mask]
        low_xy_bins = xy_bins[low_mask]
        unique_bins, inverse = np.unique(low_xy_bins, axis=0, return_inverse=True)
        band_span = (
            float(chassis_block_z_max_m) - float(obstacle_z_min_m)
        ) / max(1, int(wall_band_count))
        bands = np.clip(
            ((low_points[:, 2] - float(obstacle_z_min_m)) / band_span).astype(
                np.int32
            ),
            0,
            int(wall_band_count) - 1,
        )
        bitmasks = np.zeros(len(unique_bins), dtype=np.uint16)
        np.bitwise_or.at(bitmasks, inverse, np.left_shift(np.uint16(1), bands))
        contiguous = (bitmasks & (bitmasks + np.uint16(1))) == 0
        minimum = np.uint16((1 << int(wall_band_min_run)) - 1)
        wall_bins = unique_bins[contiguous & (bitmasks >= minimum)]
        if len(wall_bins):
            wall = np.asarray(wall_bins, dtype=np.float32) * float(column_m)

    endpoint_xy = np.concatenate((low, high), axis=0)
    if len(endpoint_xy):
        angles = np.arctan2(endpoint_xy[:, 1], endpoint_xy[:, 0])
        angle_bins = np.rint(np.degrees(angles) / 2.0).astype(np.int32)
        distance = np.linalg.norm(endpoint_xy, axis=1)
        order = np.lexsort((distance, angle_bins))
        sorted_bins = angle_bins[order]
        keep = np.r_[True, sorted_bins[1:] != sorted_bins[:-1]]
        rays = endpoint_xy[order[keep]].astype(np.float32, copy=False)
    else:
        rays = _empty_xy()
    return ColumnScan(wall.astype(np.float32, copy=False), rays)


def rasterize(spec: GridSpec, points: np.ndarray) -> np.ndarray:
    """把地图系点落到与主地图同规格的布尔画布。"""
    mask = np.zeros((spec.size, spec.size), dtype=bool)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return mask
    columns = np.floor(
        (points[:, 0] + float(spec.half_span_m)) / float(spec.resolution_m)
    ).astype(np.int64)
    rows = np.floor(
        (points[:, 1] + float(spec.half_span_m)) / float(spec.resolution_m)
    ).astype(np.int64)
    inside = (
        (columns >= 0)
        & (rows >= 0)
        & (columns < spec.size)
        & (rows < spec.size)
    )
    mask[rows[inside], columns[inside]] = True
    return mask


def uniqueness_gate(
    result: Dict[str, Any],
    config: GeometryConfig = GeometryConfig(),
) -> Tuple[bool, list[str]]:
    """判断相关峰是否唯一；可独立单测，不依赖 CUDA。"""
    blockers: list[str] = []
    if not result.get("valid"):
        blockers.append(str(result.get("reason") or "invalid_geometry"))
        return False, blockers
    if result.get("best_at_search_boundary"):
        blockers.append("search_boundary")
    if (
        float(result.get("best_prominence_robust") or 0.0)
        < float(config.min_prominence_robust)
    ):
        blockers.append("low_peak_prominence")
    if (
        float(result.get("top_one_percent_cluster_ratio") or 0.0)
        < float(config.min_top_cluster_ratio)
    ):
        blockers.append("diffuse_or_multimodal_peak")
    return not blockers, blockers


def _peaks_agree(
    first: Dict[str, Any],
    second: Dict[str, Any],
    config: GeometryConfig,
) -> bool:
    return (
        math.hypot(
            float(first["dx_m"]) - float(second["dx_m"]),
            float(first["dy_m"]) - float(second["dy_m"]),
        ) <= float(config.consensus_max_translation_m)
        and abs(wrap_deg(
            float(first["dyaw_deg"]) - float(second["dyaw_deg"])
        )) <= float(config.consensus_max_yaw_deg)
    )


def select_consensus_rows(
    rows: Iterable[Dict[str, Any]],
    config: GeometryConfig = GeometryConfig(),
) -> Tuple[list[Dict[str, Any]], str]:
    """返回唯一的最大相容窗簇；并列的冲突簇必须失败关闭。"""
    candidates = [
        row
        for row in rows
        if row.get("unique_geometry") and row.get("peaks")
    ]
    if len(candidates) < int(config.consensus_min_windows):
        return [], "insufficient_unique_windows"

    clusters: set[Tuple[int, ...]] = set()
    for anchor_index, anchor in enumerate(candidates):
        anchor_peak = anchor["peaks"][0]
        members = tuple(
            index
            for index, row in enumerate(candidates)
            if _peaks_agree(anchor_peak, row["peaks"][0], config)
        )
        if anchor_index in members:
            clusters.add(members)
    largest_size = max((len(cluster) for cluster in clusters), default=0)
    if largest_size < int(config.consensus_min_windows):
        return [], "multi_scale_disagreement"
    largest = [cluster for cluster in clusters if len(cluster) == largest_size]
    memberships = {frozenset(cluster) for cluster in largest}
    if len(memberships) != 1:
        return [], "ambiguous_consensus_clusters"
    selected = [candidates[index] for index in largest[0]]
    # 锚点邻域不是传递闭包；最终仍要求簇内两两相容。
    if any(
        not _peaks_agree(first["peaks"][0], second["peaks"][0], config)
        for index, first in enumerate(selected)
        for second in selected[index + 1:]
    ):
        return [], "multi_scale_disagreement"
    return selected, "consensus"


def select_visual_confirmed_long_window(
    rows: Iterable[Dict[str, Any]],
    visual: Tuple[float, float, float],
    config: GeometryConfig = GeometryConfig(),
) -> list[Dict[str, Any]]:
    """多窗冲突时，只允许视觉确认唯一的最长深度窗。"""
    longest = max(int(value) for value in config.windows)
    confirmed = []
    for row in rows:
        if (
            not row.get("unique_geometry")
            or int(row.get("window_frames", -1)) != longest
            or not row.get("peaks")
        ):
            continue
        peak = row["peaks"][0]
        translation = math.hypot(
            float(peak["dx_m"]) - float(visual[0]),
            float(peak["dy_m"]) - float(visual[1]),
        )
        yaw = abs(wrap_deg(float(peak["dyaw_deg"]) - float(visual[2])))
        if (
            translation
            <= float(config.visual_max_translation_disagreement_m)
            and yaw <= float(config.visual_max_yaw_disagreement_deg)
        ):
            confirmed.append(row)
    return confirmed if len(confirmed) == 1 else []


def _visual_registration_is_observable(
    observability: Optional[Dict[str, float]],
    config: GeometryConfig,
) -> bool:
    """Return whether one metric RGB-D registration is full-rank enough."""
    metrics = dict(observability or {})
    finite = all(math.isfinite(float(metrics.get(key, math.inf))) for key in (
        "translation_std_m",
        "yaw_std_deg",
        "normal_matrix_condition",
    ))
    return bool(
        finite
        and float(metrics.get("translation_std_m", math.inf))
        <= float(config.visual_support_max_translation_std_m)
        and float(metrics.get("yaw_std_deg", math.inf))
        <= float(config.visual_support_max_yaw_std_deg)
        and float(metrics.get("geometry_minor_span_m", 0.0))
        >= float(config.visual_support_min_minor_span_m)
        and float(metrics.get("pixel_coverage_x", 0.0))
        >= float(config.visual_support_min_pixel_coverage_x)
        and float(metrics.get("pixel_coverage_y", 0.0))
        >= float(config.visual_support_min_pixel_coverage_y)
    )


def select_per_query_visual_supported_long_window(
    rows: Iterable[Dict[str, Any]],
    observability: Optional[Dict[str, float]],
    config: GeometryConfig = GeometryConfig(),
) -> list[Dict[str, Any]]:
    """选择支持一次低协方差 RGB-D 注册的最长几何窗。

    这只说明当前查询的一个完整 RGB-D 注册既可观测、又位于长时占据证据
    的高似然平台。它不是跨查询或跨历史视点的独立性授权，不能单独令
    ``accepted``、``revisit_evidence_safe`` 或 ``pose_correction_safe`` 为真。
    """
    if not _visual_registration_is_observable(observability, config):
        return []
    longest = max(int(value) for value in config.windows)
    supported = []
    for row in rows:
        probe = row.get("probe_hypothesis") or {}
        if (
            int(row.get("window_frames", -1)) == longest
            and row.get("valid")
            and not row.get("best_at_search_boundary")
            and probe.get("inside_search")
            and probe.get("at_or_above_p90")
            and float(probe.get("score", -math.inf))
            >= float(config.visual_support_min_geometry_score)
            and float(probe.get("score_drop_robust", math.inf))
            <= float(config.visual_support_max_score_drop_robust)
        ):
            supported.append(row)
    return supported if len(supported) == 1 else []


def select_observable_visual_supported_long_window(
    rows: Iterable[Dict[str, Any]],
    observability: Optional[Dict[str, float]],
    config: GeometryConfig = GeometryConfig(),
) -> list[Dict[str, Any]]:
    """选择由多个独立历史 RGB-D 视点授权的最长几何窗。"""
    metrics = dict(observability or {})
    if (
        int(metrics.get("independent_candidates", 0))
        < int(config.visual_support_min_independent_candidates)
    ):
        return []
    return select_per_query_visual_supported_long_window(
        rows, metrics, config
    )


def _per_query_visual_support_report(
    rows: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    selected = list(rows)
    intervals = [
        [int(row["first_frame"]), int(row["last_frame"])]
        for row in selected
        if "first_frame" in row and "last_frame" in row
    ]
    return {
        "per_query_visual_support_safe": bool(selected),
        "per_query_visual_support_mode": (
            "single_observable_registration_supported_by_long_geometry"
            if selected else "none"
        ),
        "per_query_visual_support_window_count": len(selected),
        "per_query_visual_support_rows": selected,
        "per_query_visual_support_evidence_interval": (
            intervals[0] if len(intervals) == 1 else None
        ),
    }


def select_temporally_independent_rows(
    rows: Iterable[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """选择数量最多的时间不重叠窗口，嵌套多尺度窗口只算一份证据。"""
    candidates = [
        row
        for row in rows
        if "first_frame" in row and "last_frame" in row
    ]
    selected: list[Dict[str, Any]] = []
    last_frame = -math.inf
    # 最早结束优先是区间调度的最优贪心解。
    for row in sorted(
        candidates,
        key=lambda value: (
            int(value["last_frame"]),
            int(value["first_frame"]),
        ),
    ):
        if int(row["first_frame"]) <= last_frame:
            continue
        selected.append(row)
        last_frame = int(row["last_frame"])
    return selected


def _circular_mean_deg(values: Iterable[float]) -> float:
    radians = np.radians(np.asarray(list(values), dtype=np.float64))
    if not len(radians):
        return 0.0
    return wrap_deg(math.degrees(math.atan2(
        float(np.sin(radians).mean()),
        float(np.cos(radians).mean()),
    )))


def fuse_corrections(
    visual: Tuple[float, float, float],
    geometry: Tuple[float, float, float],
    *,
    visual_weight: float = 0.5,
) -> Tuple[float, float, float]:
    """融合两个已通过一致性门的 SE(2) 校正，角度使用最短圆弧。"""
    weight = min(1.0, max(0.0, float(visual_weight)))
    return (
        weight * float(visual[0]) + (1.0 - weight) * float(geometry[0]),
        weight * float(visual[1]) + (1.0 - weight) * float(geometry[1]),
        wrap_deg(
            float(geometry[2])
            + weight * wrap_deg(float(visual[2]) - float(geometry[2]))
        ),
    )


def _relative_pose(
    origin: Tuple[float, float, float],
    target: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    angle = math.radians(float(origin[2]))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    return (
        cos_a * dx + sin_a * dy,
        -sin_a * dx + cos_a * dy,
        wrap_deg(float(target[2]) - float(origin[2])),
    )


def _compose_pose(
    origin: Tuple[float, float, float],
    relative: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    angle = math.radians(float(origin[2]))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy = float(relative[0]), float(relative[1])
    return (
        float(origin[0]) + cos_a * dx - sin_a * dy,
        float(origin[1]) + sin_a * dx + cos_a * dy,
        wrap_deg(float(origin[2]) + float(relative[2])),
    )


def _mean_pose(
    poses: Iterable[Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    values = list(poses)
    return (
        float(np.median([pose[0] for pose in values])),
        float(np.median([pose[1] for pose in values])),
        _circular_mean_deg(pose[2] for pose in values),
    )


class TemporalPoseHypothesisBank:
    """保留竞争假设，只让唯一的跨时段共识拥有位姿写权限。

    每条证据先在自己的观测时刻得到一个校正后绝对位姿。评估时用相邻
    里程把它传播到当前时刻，再比较不同观测段是否仍落在同一位姿簇。
    两条证据的历史子图可以不同；能在不同地图区域给出同一全局校正，反而
    是比反复匹配同一面墙更强的约束。
    """

    def __init__(
        self,
        config: TemporalPoseConfig = TemporalPoseConfig(),
    ) -> None:
        self.config = config
        self._evidence: list[TemporalPoseEvidence] = []
        self._next_serial = 0

    @property
    def evidence_count(self) -> int:
        return len(self._evidence)

    def clear(self) -> None:
        self._evidence.clear()

    def _intervals_disjoint(
        self,
        first: TemporalPoseEvidence,
        second: TemporalPoseEvidence,
    ) -> bool:
        if int(first.last_frame) < int(second.first_frame):
            gap = int(second.first_frame) - int(first.last_frame) - 1
        elif int(second.last_frame) < int(first.first_frame):
            gap = int(first.first_frame) - int(second.last_frame) - 1
        else:
            return False
        # 刚好切成前后两半的连续视野仍共享运动和传感器误差。至少空出
        # 一个最短滚动窗，才把它们当成两次独立地点观测。
        return gap >= int(self.config.min_gap_frames)

    def _viewpoints_independent(
        self,
        first: TemporalPoseEvidence,
        second: TemporalPoseEvidence,
    ) -> bool:
        relative = _relative_pose(first.predicted_pose, second.predicted_pose)
        return (
            math.hypot(relative[0], relative[1])
            >= float(self.config.min_baseline_m)
            or abs(relative[2])
            >= float(self.config.min_baseline_yaw_deg)
        )

    def _poses_agree(
        self,
        first: Tuple[float, float, float],
        second: Tuple[float, float, float],
    ) -> bool:
        return (
            math.hypot(first[0] - second[0], first[1] - second[1])
            <= float(self.config.max_translation_m)
            and abs(wrap_deg(first[2] - second[2]))
            <= float(self.config.max_yaw_deg)
        )

    @staticmethod
    def _propagate(
        evidence: TemporalPoseEvidence,
        current_predicted_pose: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        odometry_delta = _relative_pose(
            evidence.predicted_pose,
            current_predicted_pose,
        )
        return _compose_pose(evidence.corrected_pose, odometry_delta)

    def observe(
        self,
        *,
        target_group: Iterable[int],
        predicted_pose: Tuple[float, float, float],
        corrected_pose: Tuple[float, float, float],
        evidence_interval: Tuple[int, int],
        frame_index: int,
    ) -> Dict[str, Any]:
        """加入一次传感器共识，并返回当前时刻是否可安全校正。"""
        first_frame, last_frame = (int(value) for value in evidence_interval)
        if last_frame < first_frame or last_frame > int(frame_index):
            return {
                "pose_correction_safe": False,
                "reason": "invalid_or_future_evidence_interval",
                "evidence_count": self.evidence_count,
                "hypothesis_count": 0,
            }
        group = tuple(sorted({int(value) for value in target_group}))
        if not group:
            return {
                "pose_correction_safe": False,
                "reason": "missing_target_group",
                "evidence_count": self.evidence_count,
                "hypothesis_count": 0,
            }
        self._next_serial += 1
        latest = TemporalPoseEvidence(
            serial=int(self._next_serial),
            frame_index=int(frame_index),
            first_frame=first_frame,
            last_frame=last_frame,
            target_group=group,
            predicted_pose=tuple(float(value) for value in predicted_pose),
            corrected_pose=tuple(float(value) for value in corrected_pose),
        )
        # 同一帧的重复调用只保留最后一条，避免 API 重试制造票数。
        self._evidence = [
            row for row in self._evidence
            if not (
                row.frame_index == latest.frame_index
                and row.target_group == latest.target_group
            )
        ]
        self._evidence.append(latest)
        maximum = max(2, int(self.config.max_evidence))
        if len(self._evidence) > maximum:
            del self._evidence[:-maximum]
        return self.evaluate(
            current_predicted_pose=latest.predicted_pose,
            latest_serial=latest.serial,
        )

    def evaluate(
        self,
        *,
        current_predicted_pose: Tuple[float, float, float],
        latest_serial: Optional[int] = None,
    ) -> Dict[str, Any]:
        """列出全部有独立双证据支持的假设；有竞争簇时失败关闭。"""
        evidence = list(self._evidence)
        expected = [
            self._propagate(row, current_predicted_pose) for row in evidence
        ]
        pairs: list[Dict[str, Any]] = []
        for first_index, first in enumerate(evidence):
            for second_index in range(first_index + 1, len(evidence)):
                second = evidence[second_index]
                if not self._intervals_disjoint(first, second):
                    continue
                if not self._viewpoints_independent(first, second):
                    continue
                if not self._poses_agree(
                    expected[first_index], expected[second_index]
                ):
                    continue
                pairs.append({
                    "serials": {int(first.serial), int(second.serial)},
                    "pose": _mean_pose(
                        (expected[first_index], expected[second_index])
                    ),
                    "target_groups": {
                        tuple(first.target_group), tuple(second.target_group)
                    },
                })

        # 成对假设只有在和簇内每一项都一致时才合并。桥接峰不会把两个
        # 冲突峰串成一个大簇；多留一个竞争簇只会拒绝，不会误授权。
        clusters: list[Dict[str, Any]] = []
        for pair in pairs:
            placed = False
            for cluster in clusters:
                if all(
                    self._poses_agree(pair["pose"], pose)
                    for pose in cluster["pair_poses"]
                ):
                    cluster["serials"].update(pair["serials"])
                    cluster["pair_poses"].append(pair["pose"])
                    cluster["target_groups"].update(pair["target_groups"])
                    placed = True
                    break
            if not placed:
                clusters.append({
                    "serials": set(pair["serials"]),
                    "pair_poses": [pair["pose"]],
                    "target_groups": set(pair["target_groups"]),
                })

        summaries = []
        for cluster in clusters:
            pose = _mean_pose(cluster["pair_poses"])
            summaries.append({
                "pose": [float(value) for value in pose],
                "evidence_serials": sorted(cluster["serials"]),
                "evidence_count": len(cluster["serials"]),
                "pair_count": len(cluster["pair_poses"]),
                "target_groups": [
                    list(group) for group in sorted(cluster["target_groups"])
                ],
            })

        base = {
            "evidence_count": len(evidence),
            "hypothesis_count": len(summaries),
            "hypotheses": summaries,
            "latest_serial": latest_serial,
            "minimum_independent_episodes": int(self.config.min_episodes),
            "minimum_interval_gap_frames": int(self.config.min_gap_frames),
        }
        if not summaries:
            return {
                **base,
                "pose_correction_safe": False,
                "reason": "insufficient_independent_temporal_evidence",
            }
        if len(summaries) != 1:
            return {
                **base,
                "pose_correction_safe": False,
                "reason": "ambiguous_temporal_pose_hypotheses",
            }
        winner = summaries[0]
        if (
            latest_serial is not None
            and int(latest_serial) not in winner["evidence_serials"]
        ):
            return {
                **base,
                "pose_correction_safe": False,
                "reason": "latest_evidence_not_in_supported_hypothesis",
            }
        if winner["evidence_count"] < int(self.config.min_episodes):
            return {
                **base,
                "pose_correction_safe": False,
                "reason": "insufficient_independent_temporal_evidence",
            }
        corrected = tuple(float(value) for value in winner["pose"])
        correction = (
            corrected[0] - float(current_predicted_pose[0]),
            corrected[1] - float(current_predicted_pose[1]),
            wrap_deg(corrected[2] - float(current_predicted_pose[2])),
        )
        return {
            **base,
            "pose_correction_safe": True,
            "reason": "unique_temporal_pose_hypothesis",
            "corrected_pose": list(corrected),
            "correction": [float(value) for value in correction],
            "independent_episode_count": int(winner["evidence_count"]),
        }


def _concat_nonempty(parts: Iterable[np.ndarray]) -> np.ndarray:
    values = [
        np.asarray(part, dtype=np.float32).reshape(-1, 2)
        for part in parts
        if part is not None and len(part)
    ]
    return np.concatenate(values, axis=0) if values else _empty_xy()


class GpuGeometryValidator:
    """维护滚动深度证据，并在 CUDA 上验证历史子图候选。"""

    def __init__(
        self,
        grid: GridSpec,
        *,
        config: GeometryConfig = GeometryConfig(),
        device: Optional[str] = None,
    ) -> None:
        self.grid = grid
        self.config = config
        self.device_name = str(
            device or os.environ.get(GEOMETRY_DEVICE_ENV, "cuda:0")
        )
        self._samples: Deque[Dict[str, Any]] = deque(
            maxlen=max(int(value) for value in config.windows)
        )
        self.last_backend_reason = "not_checked"

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples.clear()

    def remember(
        self,
        scan: ColumnScan,
        pose: Tuple[float, float, float],
        frame_index: int,
    ) -> bool:
        """按当前地图位姿保存一帧；图优化后调用方必须清空。"""
        wall = transform_points(scan.wall, pose)
        if len(scan.rays):
            fractions = np.asarray([0.20, 0.40, 0.60, 0.80], dtype=np.float32)
            free_local = (
                scan.rays[None, :, :] * fractions[:, None, None]
            ).reshape(-1, 2)
            free = transform_points(free_local, pose)
        else:
            free = _empty_xy()
        self._samples.append({
            "frame_index": int(frame_index),
            "wall": wall,
            "free": free,
            "pose": tuple(float(value) for value in pose),
        })
        return bool(len(wall) or len(free))

    def backend_info(self) -> Dict[str, Any]:
        return {
            "backend": "torch_cuda_correlation_point_to_plane_icp",
            "device": self.device_name,
            "cpu_fallback": False,
            "last_reason": self.last_backend_reason,
            "rolling_samples": self.sample_count,
        }

    def rolling_source(self, window: Optional[int] = None) -> Dict[str, Any]:
        """Return one read-only rolling geometry window for global seeding.

        The points stay in the caller-provided odometry/map coordinates.  A
        global candidate generator may propose a rigid correction, but cannot
        modify this validator's evidence ledger.
        """
        samples = list(self._samples)
        requested = max(self.config.windows) if window is None else int(window)
        selected = samples[-max(1, requested)::2]
        if not selected:
            return {
                "wall": _empty_xy(),
                "free": _empty_xy(),
                "first_frame": None,
                "last_frame": None,
                "sample_count": 0,
            }
        return {
            "wall": unique_cells(
                _concat_nonempty(row["wall"] for row in selected),
                self.grid.resolution_m,
            ),
            "free": unique_cells(
                _concat_nonempty(row["free"] for row in selected),
                self.grid.resolution_m,
            ),
            "first_frame": int(selected[0]["frame_index"]),
            "last_frame": int(selected[-1]["frame_index"]),
            "sample_count": int(len(selected)),
        }

    def score_pair(
        self,
        source_wall: np.ndarray,
        source_free: np.ndarray,
        target_wall_points: np.ndarray,
        target_free_points: np.ndarray,
        *,
        pivot: Tuple[float, float],
    ) -> Dict[str, Any]:
        """对一对静态几何运行生产搜索，供随机化回归和诊断复用。"""
        torch, device, backend_reason = self._cuda()
        self.last_backend_reason = backend_reason
        if torch is None or device is None:
            return {"valid": False, "reason": backend_reason}
        target_wall_points = unique_cells(
            target_wall_points, self.grid.resolution_m
        )
        target_free_points = unique_cells(
            target_free_points, self.grid.resolution_m
        )
        target_wall = rasterize(self.grid, target_wall_points)
        target_free = rasterize(self.grid, target_free_points) & ~target_wall
        return self._search(
            unique_cells(source_wall, self.grid.resolution_m),
            unique_cells(source_free, self.grid.resolution_m),
            target_wall_points,
            target_wall,
            target_free,
            pivot,
            torch=torch,
            device=device,
        )

    def _cuda(self) -> Tuple[Optional[Any], Optional[Any], str]:
        try:
            import torch
        except Exception as exc:
            return None, None, f"torch_import_failed:{type(exc).__name__}"
        try:
            device = torch.device(self.device_name)
            if device.type != "cuda":
                return None, None, "non_cuda_device_rejected"
            if not torch.cuda.is_available():
                return None, None, "cuda_unavailable"
            # 提前触发设备索引检查，避免在大张量分配中才失败。
            torch.empty(1, device=device)
            return torch, device, "cuda_ready"
        except Exception as exc:
            return None, None, f"cuda_device_failed:{type(exc).__name__}"

    def validate(
        self,
        target_wall_points: np.ndarray,
        target_free_points: np.ndarray,
        *,
        pivot: Tuple[float, float],
        visual_correction: Optional[Tuple[float, float, float]],
        visual_metric: bool = True,
        visual_observability: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """返回多窗口验证报告；accepted 为真时才给出可记账校正。

        ``visual_metric=False`` 表示输入只是长期外观序列提出的地点搜索中心，
        不是由 RGB-D 对应求出的米制位姿。此时它不能参与传感器融合，也不能
        用来挽救单个深度窗；只有深度自身形成唯一的多窗口共识才会接受。
        """
        torch, device, backend_reason = self._cuda()
        self.last_backend_reason = backend_reason
        if torch is None or device is None:
            return {
                "accepted": False,
                "reason": backend_reason,
                "revisit_evidence_safe": False,
                "pose_correction_safe": False,
                **_per_query_visual_support_report([]),
                "geometry_backend": self.backend_info(),
                "rows": [],
            }

        visual = (
            None
            if visual_correction is None
            else tuple(float(value) for value in visual_correction)
        )
        search_center, search_center_mode = geometry_search_center(
            visual, self.config
        )
        search_pivot = (
            float(pivot[0]) + float(search_center[0]),
            float(pivot[1]) + float(search_center[1]),
        )
        target_wall_points = unique_cells(
            target_wall_points, self.grid.resolution_m
        )
        target_free_points = unique_cells(
            target_free_points, self.grid.resolution_m
        )
        samples = list(self._samples)
        rows: list[Dict[str, Any]] = []
        for window in self.config.windows:
            selected = samples[-int(window)::2]
            if len(selected) < max(4, int(window) // 4):
                rows.append({
                    "window_frames": int(window),
                    "valid": False,
                    "unique_geometry": False,
                    "uniqueness_blockers": ["insufficient_distinct_frames"],
                })
                continue
            source_wall = unique_cells(
                _concat_nonempty(row["wall"] for row in selected),
                self.grid.resolution_m,
            )
            source_free = unique_cells(
                _concat_nonempty(row["free"] for row in selected),
                self.grid.resolution_m,
            )
            # 地点检索负责给出任意尺度的全局候选；CUDA 相关搜索只估计
            # 该候选附近的小残差。这样搜索尺度仍由传感器噪声决定，而不随
            # 房屋大小、路径长度或某个 benchmark 的回环位移改变。
            source_wall = apply_map_correction(
                source_wall, pivot, search_center
            )
            source_free = apply_map_correction(
                source_free, pivot, search_center
            )
            local_wall_points, local_free_points, support = (
                crop_target_to_source_support(
                    target_wall_points,
                    target_free_points,
                    source_wall,
                    source_free,
                    pivot=search_pivot,
                    config=self.config,
                    resolution_m=self.grid.resolution_m,
                )
            )
            target_wall = rasterize(self.grid, local_wall_points)
            target_free = (
                rasterize(self.grid, local_free_points) & ~target_wall
            )
            search = self._search(
                source_wall,
                source_free,
                local_wall_points,
                target_wall,
                target_free,
                search_pivot,
                torch=torch,
                device=device,
                probe_correction=(
                    None
                    if visual is None
                    else (
                        float(visual[0]) - float(search_center[0]),
                        float(visual[1]) - float(search_center[1]),
                        wrap_deg(
                            float(visual[2]) - float(search_center[2])
                        ),
                    )
                ),
            )
            search = _search_report_in_map_frame(
                search, search_center, search_center_mode
            )
            unique, blockers = uniqueness_gate(search, self.config)
            rows.append({
                "window_frames": int(window),
                "first_frame": int(selected[0]["frame_index"]),
                "last_frame": int(selected[-1]["frame_index"]),
                "target_support": support,
                "valid": bool(search.get("valid")),
                "unique_geometry": bool(unique),
                "uniqueness_blockers": blockers,
                **search,
            })

        # 起点相同的窗口实际是同一份证据，不能重复投票。
        independent = {
            int(row["first_frame"]): row
            for row in rows
            if row.get("valid") and "first_frame" in row
        }
        unique_rows = [
            row for row in independent.values() if row.get("unique_geometry")
        ]
        per_query_supported_rows = (
            select_per_query_visual_supported_long_window(
                rows, visual_observability, self.config
            )
            if visual is not None and visual_metric
            else []
        )
        per_query_support_report = _per_query_visual_support_report(
            per_query_supported_rows
        )
        accepted_rows, consensus_reason = select_consensus_rows(
            unique_rows, self.config
        )
        consensus_mode = "multi_window_geometry_consensus"
        if not accepted_rows and visual is not None and visual_metric:
            accepted_rows = select_visual_confirmed_long_window(
                unique_rows, visual, self.config
            )
            if accepted_rows:
                consensus_mode = "visual_confirmed_long_geometry_window"
        if not accepted_rows and visual is not None and visual_metric:
            accepted_rows = select_observable_visual_supported_long_window(
                rows, visual_observability, self.config
            )
            if accepted_rows:
                consensus_mode = "observable_visual_supported_long_window"
        if not accepted_rows:
            return {
                "accepted": False,
                "reason": consensus_reason,
                "revisit_evidence_safe": False,
                "pose_correction_safe": False,
                **per_query_support_report,
                "visual_correction": (
                    None if visual is None else list(visual)
                ),
                "visual_observability": dict(visual_observability or {}),
                "independent_window_count": len(independent),
                "unique_window_count": len(unique_rows),
                "consensus_window_count": 0,
                "consensus_mode": "none",
                "geometry_backend": self.backend_info(),
                "rows": rows,
            }
        peaks = [row["peaks"][0] for row in accepted_rows]
        accepted_intervals = [
            [int(row["first_frame"]), int(row["last_frame"])]
            for row in accepted_rows
            if "first_frame" in row and "last_frame" in row
        ]
        evidence_interval = (
            [
                min(interval[0] for interval in accepted_intervals),
                max(interval[1] for interval in accepted_intervals),
            ]
            if accepted_intervals else None
        )
        geometry = (
            float(np.median([peak["dx_m"] for peak in peaks])),
            float(np.median([peak["dy_m"] for peak in peaks])),
            _circular_mean_deg(peak["dyaw_deg"] for peak in peaks),
        )
        if visual is None:
            return {
                "accepted": False,
                "reason": "missing_visual_candidate",
                "revisit_evidence_safe": False,
                "pose_correction_safe": False,
                **per_query_support_report,
                "geometry_correction": list(geometry),
                "independent_window_count": len(independent),
                "unique_window_count": len(unique_rows),
                "consensus_window_count": len(accepted_rows),
                "consensus_mode": consensus_mode,
                "accepted_intervals": accepted_intervals,
                "evidence_interval": evidence_interval,
                "geometry_backend": self.backend_info(),
                "rows": rows,
            }
        translation_disagreement = math.hypot(
            visual[0] - geometry[0], visual[1] - geometry[1]
        )
        yaw_disagreement = abs(wrap_deg(visual[2] - geometry[2]))
        temporal_independent = select_temporally_independent_rows(accepted_rows)
        if consensus_mode == "observable_visual_supported_long_window":
            return {
                "accepted": True,
                "reason": "observable_visual_supported_by_long_geometry",
                "visual_correction": list(visual),
                "visual_observability": dict(visual_observability or {}),
                "geometry_correction": list(geometry),
                # 退化几何只做一致性检验，不能沿不可观测方向拉动视觉解。
                "fused_correction": list(visual),
                "revisit_evidence_safe": True,
                "pose_correction_safe": False,
                **per_query_support_report,
                "translation_disagreement_m": float(
                    translation_disagreement
                ),
                "yaw_disagreement_deg": float(yaw_disagreement),
                "independent_window_count": len(independent),
                "unique_window_count": len(unique_rows),
                "consensus_window_count": len(accepted_rows),
                "temporal_independent_window_count": len(
                    temporal_independent
                ),
                "consensus_mode": consensus_mode,
                "accepted_intervals": accepted_intervals,
                "evidence_interval": evidence_interval,
                "geometry_backend": self.backend_info(),
                "rows": rows,
            }
        if not visual_metric:
            # 外观序列只负责把全局搜索缩到某个历史子图附近。最终校正完全
            # 取深度几何峰；外观中心与细化峰的残差仅供审计，不能被平均进
            # 位姿，也不能冒充第二个传感器投票。
            pose_correction_safe = (
                consensus_mode == "multi_window_geometry_consensus"
                and len(temporal_independent)
                >= int(self.config.consensus_min_windows)
            )
            return {
                "accepted": True,
                "reason": "appearance_guided_geometry_consensus",
                "visual_correction": list(visual),
                "visual_metric": False,
                "geometry_correction": list(geometry),
                "fused_correction": list(geometry),
                "revisit_evidence_safe": True,
                "pose_correction_safe": bool(pose_correction_safe),
                **per_query_support_report,
                "translation_disagreement_m": float(translation_disagreement),
                "yaw_disagreement_deg": float(yaw_disagreement),
                "independent_window_count": len(independent),
                "unique_window_count": len(unique_rows),
                "consensus_window_count": len(accepted_rows),
                "temporal_independent_window_count": len(temporal_independent),
                "consensus_mode": consensus_mode,
                "accepted_intervals": accepted_intervals,
                "evidence_interval": evidence_interval,
                "geometry_backend": self.backend_info(),
                "rows": rows,
            }
        sensors_agree = (
            translation_disagreement
            <= float(self.config.visual_max_translation_disagreement_m)
            and yaw_disagreement
            <= float(self.config.visual_max_yaw_disagreement_deg)
        )
        fused = fuse_corrections(visual, geometry)
        # 单个长时间窗与视觉一致，足以证明机器人回到了旧区域；嵌套的
        # 多尺度尾窗仍共享同一批深度，不能据此旋转位姿图或让当前位姿跳变。
        # 位姿校正必须由至少两个时间不重叠的观测段共同授权。
        pose_correction_safe = (
            sensors_agree
            and consensus_mode == "multi_window_geometry_consensus"
            and len(temporal_independent)
            >= int(self.config.consensus_min_windows)
        )
        return {
            "accepted": bool(sensors_agree),
            "reason": (
                "visual_depth_consensus"
                if sensors_agree
                else "visual_depth_disagreement"
            ),
            "visual_correction": list(visual),
            "geometry_correction": list(geometry),
            "fused_correction": list(fused),
            "revisit_evidence_safe": bool(sensors_agree),
            "pose_correction_safe": bool(pose_correction_safe),
            **per_query_support_report,
            "translation_disagreement_m": float(translation_disagreement),
            "yaw_disagreement_deg": float(yaw_disagreement),
            "independent_window_count": len(independent),
            "unique_window_count": len(unique_rows),
            "consensus_window_count": len(accepted_rows),
            "temporal_independent_window_count": len(temporal_independent),
            "consensus_mode": consensus_mode,
            "accepted_intervals": accepted_intervals,
            "evidence_interval": evidence_interval,
            "geometry_backend": self.backend_info(),
            "rows": rows,
        }

    def _refine_peak_icp(
        self,
        source_wall: np.ndarray,
        target_wall_points: np.ndarray,
        pivot: Tuple[float, float],
        coarse_peak: Dict[str, float],
        *,
        torch: Any,
        device: Any,
    ) -> Dict[str, Any]:
        """在 CUDA 上用截断互近邻 ICP 细化粗相关峰。"""

        def subsample(points: np.ndarray, maximum: int) -> np.ndarray:
            if len(points) <= maximum:
                return points
            indices = np.linspace(0, len(points) - 1, maximum).astype(np.int64)
            return points[indices]

        source = subsample(
            np.asarray(source_wall, dtype=np.float32),
            max(1, int(self.config.icp_max_source_points)),
        )
        target = subsample(
            np.asarray(target_wall_points, dtype=np.float32),
            max(1, int(self.config.icp_max_target_points)),
        )
        if (
            len(source) < int(self.config.icp_min_correspondences)
            or len(target) < int(self.config.icp_min_correspondences)
        ):
            return {"accepted": False, "reason": "insufficient_icp_points"}

        source_gpu = torch.from_numpy(source).to(device=device)
        target_gpu = torch.from_numpy(target).to(device=device)
        pivot_gpu = torch.tensor(pivot, device=device, dtype=torch.float32)
        delta = torch.tensor(
            [float(coarse_peak["dx_m"]), float(coarse_peak["dy_m"])],
            device=device,
            dtype=torch.float32,
        )

        # 厚占据墙的最近点会让点到点 ICP 偏向某个栅格边缘。局部 PCA 法向
        # 把约束恢复成点到墙面距离，因而 yaw 不依赖墙有一格还是三格厚。
        def local_frames(points: Any) -> Tuple[Any, Any, Any]:
            pairwise = torch.cdist(points, points)
            neighbor_count = min(
                len(points), max(4, int(self.config.icp_normal_neighbors))
            )
            neighbor_indices = torch.topk(
                pairwise,
                k=neighbor_count,
                dim=1,
                largest=False,
            ).indices
            neighborhoods = points[neighbor_indices]
            neighborhood_zero = (
                neighborhoods - neighborhoods.mean(dim=1, keepdim=True)
            )
            covariance = (
                neighborhood_zero.transpose(1, 2) @ neighborhood_zero
            ) / float(max(1, neighbor_count - 1))
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            anisotropy = 1.0 - (
                eigenvalues[:, 0] / eigenvalues[:, 1].clamp_min(1e-8)
            )
            return eigenvectors[:, :, 0], eigenvectors[:, :, 1], anisotropy

        target_normals, target_tangents, normal_anisotropy = local_frames(
            target_gpu
        )
        _, source_tangents, source_anisotropy = local_frames(source_gpu)

        def orientation_histogram(tangents: Any, anisotropy: Any) -> Any:
            step_deg = float(self.config.icp_orientation_step_deg)
            bin_count = max(180, int(round(180.0 / step_deg)))
            angles = torch.remainder(
                torch.atan2(tangents[:, 1], tangents[:, 0]), math.pi
            )
            positions = angles * (float(bin_count) / math.pi)
            lower = torch.floor(positions).long() % bin_count
            fraction = positions - torch.floor(positions)
            weights = torch.clamp(anisotropy, min=0.0, max=1.0)
            histogram = torch.zeros(
                bin_count, device=device, dtype=torch.float32
            )
            histogram.scatter_add_(0, lower, weights * (1.0 - fraction))
            histogram.scatter_add_(
                0, (lower + 1) % bin_count, weights * fraction
            )
            sigma_bins = max(
                0.5,
                float(self.config.icp_orientation_blur_deg) / step_deg,
            )
            radius = max(1, int(math.ceil(3.0 * sigma_bins)))
            offsets = range(-radius, radius + 1)
            kernel = [math.exp(-0.5 * (value / sigma_bins) ** 2) for value in offsets]
            blurred = sum(
                weight * torch.roll(histogram, shifts=value)
                for value, weight in zip(offsets, kernel)
            )
            return blurred / torch.linalg.vector_norm(blurred).clamp_min(1e-8)

        target_orientation = orientation_histogram(
            target_tangents, normal_anisotropy
        )
        source_orientation = orientation_histogram(
            source_tangents, source_anisotropy
        )
        orientation_offsets = np.arange(
            -float(self.config.search_span_deg),
            float(self.config.search_span_deg)
            + 0.5 * float(self.config.icp_orientation_step_deg),
            float(self.config.icp_orientation_step_deg),
            dtype=np.float32,
        )
        orientation_scores = torch.stack([
            torch.dot(
                target_orientation,
                torch.roll(
                    source_orientation,
                    shifts=int(round(
                        float(offset) / float(
                            self.config.icp_orientation_step_deg
                        )
                    )),
                ),
            )
            for offset in orientation_offsets
        ])
        orientation_index = int(torch.argmax(orientation_scores).item())
        orientation_yaw = float(orientation_offsets[orientation_index])
        orientation_quantiles = torch.quantile(
            orientation_scores, torch.tensor(
                [0.5, 0.9], device=device, dtype=torch.float32
            )
        )
        orientation_spread = float(
            (orientation_quantiles[1] - orientation_quantiles[0])
            .clamp_min(1e-8).item()
        )
        orientation_prominence = (
            float(
                orientation_scores[orientation_index].item()
                - orientation_quantiles[1].item()
            ) / orientation_spread
        )
        orientation_delta = abs(wrap_deg(
            orientation_yaw - float(coarse_peak["dyaw_deg"])
        ))
        orientation_used = (
            orientation_delta <= float(self.config.icp_max_refinement_yaw_deg)
        )
        start_yaw_deg = (
            wrap_deg(
                float(coarse_peak["dyaw_deg"])
                + 0.5 * wrap_deg(
                    orientation_yaw - float(coarse_peak["dyaw_deg"])
                )
            )
            if orientation_used
            else float(coarse_peak["dyaw_deg"])
        )
        yaw = math.radians(start_yaw_deg)
        rotation = torch.tensor(
            [[math.cos(yaw), -math.sin(yaw)],
             [math.sin(yaw), math.cos(yaw)]],
            device=device,
            dtype=torch.float32,
        )
        translation = pivot_gpu + delta - rotation @ pivot_gpu
        moved = source_gpu @ rotation.T + translation
        coarse_yaw = math.radians(float(coarse_peak["dyaw_deg"]))
        coarse_rotation = torch.tensor(
            [[math.cos(coarse_yaw), -math.sin(coarse_yaw)],
             [math.sin(coarse_yaw), math.cos(coarse_yaw)]],
            device=device,
            dtype=torch.float32,
        )
        coarse_translation = pivot_gpu + delta - coarse_rotation @ pivot_gpu
        coarse_moved = source_gpu @ coarse_rotation.T + coarse_translation

        def correspondences(points: Any) -> Tuple[Any, Any, Any, Any, Any]:
            distances = torch.cdist(points, target_gpu)
            nearest_distance, nearest_target = torch.min(distances, dim=1)
            normals = target_normals[nearest_target]
            fixed = target_gpu[nearest_target]
            residual = torch.sum(normals * (points - fixed), dim=1)
            eligible = (
                (nearest_distance <= float(self.config.icp_max_correspondence_m))
                & (
                    normal_anisotropy[nearest_target]
                    >= float(self.config.icp_min_normal_anisotropy)
                )
            )
            eligible_residuals = torch.abs(residual[eligible])
            if len(eligible_residuals) >= int(self.config.icp_min_correspondences):
                trim = torch.quantile(
                    eligible_residuals,
                    float(self.config.icp_trim_quantile),
                )
                eligible &= torch.abs(residual) <= trim
            return nearest_target, normals, residual, eligible, nearest_distance

        with torch.inference_mode():
            _, _, initial_residual, initial_mask, _ = correspondences(
                coarse_moved
            )
            if int(initial_mask.sum().item()) < int(
                self.config.icp_min_correspondences
            ):
                return {
                    "accepted": False,
                    "reason": "insufficient_planar_correspondences",
                    "correspondences": int(initial_mask.sum().item()),
                }
            initial_rmse = float(torch.sqrt(torch.mean(
                torch.square(initial_residual[initial_mask])
            )).item())
            _, _, _, start_mask, _ = correspondences(moved)
            if int(start_mask.sum().item()) < int(
                self.config.icp_min_correspondences
            ):
                orientation_used = False
                start_yaw_deg = float(coarse_peak["dyaw_deg"])
                rotation = coarse_rotation
                translation = coarse_translation
                moved = coarse_moved
            iterations = 0
            singular_ratio = 0.0
            for iteration in range(max(1, int(self.config.icp_iterations))):
                _, normals, residual, mask, _ = correspondences(moved)
                if int(mask.sum().item()) < int(
                    self.config.icp_min_correspondences
                ):
                    break
                moving_points = moved[mask]
                selected_normals = normals[mask]
                selected_residual = residual[mask]
                moving_center = moving_points.mean(dim=0)
                centered = moving_points - moving_center
                yaw_jacobian = (
                    -selected_normals[:, 0] * centered[:, 1]
                    + selected_normals[:, 1] * centered[:, 0]
                )
                jacobian = torch.column_stack((selected_normals, yaw_jacobian))
                scale = torch.median(torch.abs(selected_residual)).clamp_min(0.01)
                weights = torch.clamp(
                    scale / torch.abs(selected_residual).clamp_min(1e-6),
                    max=1.0,
                )
                weighted = torch.sqrt(weights)[:, None]
                normal_matrix = (jacobian * weighted).T @ (jacobian * weighted)
                singular = torch.linalg.eigvalsh(normal_matrix)
                singular_ratio = float(
                    (singular[0] / singular[-1].clamp_min(1e-8)).item()
                )
                solution = torch.linalg.lstsq(
                    jacobian * weighted,
                    -selected_residual[:, None] * weighted,
                ).solution[:, 0]
                increment_translation_local = solution[:2]
                translation_norm = torch.linalg.vector_norm(
                    increment_translation_local
                ).clamp_min(1e-8)
                max_translation = float(
                    self.config.icp_max_increment_translation_m
                )
                increment_translation_local *= torch.clamp(
                    max_translation / translation_norm, max=1.0
                )
                max_yaw = math.radians(float(
                    self.config.icp_max_increment_yaw_deg
                ))
                increment_yaw_rad = torch.clamp(
                    solution[2], min=-max_yaw, max=max_yaw
                )
                cosine = torch.cos(increment_yaw_rad)
                sine = torch.sin(increment_yaw_rad)
                increment_rotation = torch.stack((
                    torch.stack((cosine, -sine)),
                    torch.stack((sine, cosine)),
                ))
                increment_translation = (
                    moving_center
                    + increment_translation_local
                    - increment_rotation @ moving_center
                )
                moved = moved @ increment_rotation.T + increment_translation
                rotation = increment_rotation @ rotation
                translation = (
                    increment_rotation @ translation + increment_translation
                )
                iterations = iteration + 1
                if (
                    float(torch.linalg.vector_norm(
                        increment_translation_local
                    ).item())
                    < 1e-4
                    and abs(math.degrees(float(increment_yaw_rad.item()))) < 0.02
                ):
                    break

            _, _, final_residual, final_mask, _ = correspondences(moved)
            final_count = int(final_mask.sum().item())
            if final_count < int(self.config.icp_min_correspondences):
                return {
                    "accepted": False,
                    "reason": "icp_lost_correspondences",
                    "iterations": iterations,
                    "correspondences": final_count,
                }
            final_rmse = float(torch.sqrt(torch.mean(
                torch.square(final_residual[final_mask])
            )).item())
            refined_delta = translation - pivot_gpu + rotation @ pivot_gpu
            refined = (
                float(refined_delta[0].item()),
                float(refined_delta[1].item()),
                wrap_deg(math.degrees(math.atan2(
                    float(rotation[1, 0].item()),
                    float(rotation[0, 0].item()),
                ))),
            )

        refinement_translation = math.hypot(
            refined[0] - float(coarse_peak["dx_m"]),
            refined[1] - float(coarse_peak["dy_m"]),
        )
        refinement_yaw = abs(wrap_deg(
            refined[2] - float(coarse_peak["dyaw_deg"])
        ))
        within_refinement_basin = (
            refinement_translation
            <= float(self.config.icp_max_refinement_translation_m)
            and refinement_yaw
            <= float(self.config.icp_max_refinement_yaw_deg)
        )
        within_search = (
            abs(refined[0])
            <= float(self.config.search_span_m + self.config.search_step_m)
            and abs(refined[1])
            <= float(self.config.search_span_m + self.config.search_step_m)
            and abs(refined[2])
            <= float(self.config.search_span_deg + self.config.search_step_deg)
        )
        improved = final_rmse <= initial_rmse + 1e-6
        accepted = bool(within_refinement_basin and within_search and improved)
        reason = "icp_refined" if accepted else "icp_refinement_rejected"
        return {
            "accepted": accepted,
            "reason": reason,
            "correction": list(refined),
            "iterations": int(iterations),
            "correspondences": int(final_count),
            "initial_rmse_m": initial_rmse,
            "final_rmse_m": final_rmse,
            "refinement_translation_m": float(refinement_translation),
            "refinement_yaw_deg": float(refinement_yaw),
            "singular_ratio": float(singular_ratio),
            "orientation_yaw_deg": float(orientation_yaw),
            "orientation_fused_start_yaw_deg": float(start_yaw_deg),
            "orientation_delta_from_coarse_deg": float(orientation_delta),
            "orientation_prominence_robust": float(orientation_prominence),
            "orientation_used": bool(orientation_used),
            "within_refinement_basin": bool(within_refinement_basin),
            "within_search": bool(within_search),
            "improved": bool(improved),
        }

    def _search(
        self,
        source_wall: np.ndarray,
        source_free: np.ndarray,
        target_wall_points: np.ndarray,
        target_wall: np.ndarray,
        target_free: np.ndarray,
        pivot: Tuple[float, float],
        *,
        torch: Any,
        device: Any,
        probe_correction: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, Any]:
        """CUDA 双向相关搜索；这里只做一次小型单线程距离变换。"""
        if len(source_wall) < 16 or len(target_wall_points) < 16:
            return {"valid": False, "reason": "insufficient_wall_points"}
        try:
            import cv2

            cv2.setNumThreads(1)
            distance = cv2.distanceTransform(
                (~target_wall).astype(np.uint8),
                cv2.DIST_L2,
                cv2.DIST_MASK_PRECISE,
            ).astype(np.float32) * np.float32(self.grid.resolution_m)
        except Exception as exc:
            return {
                "valid": False,
                "reason": f"distance_transform_failed:{type(exc).__name__}",
            }

        source_free_mask = rasterize(self.grid, source_free)
        target_known = target_wall | target_free
        offsets = np.arange(
            -self.config.search_span_m,
            self.config.search_span_m + 0.5 * self.config.search_step_m,
            self.config.search_step_m,
            dtype=np.float32,
        )
        yaw_offsets = np.arange(
            -self.config.search_span_deg,
            self.config.search_span_deg + 0.5 * self.config.search_step_deg,
            self.config.search_step_deg,
            dtype=np.float32,
        )
        dx, dy, dyaw = np.meshgrid(offsets, offsets, yaw_offsets, indexing="ij")
        candidates = np.column_stack((dx.ravel(), dy.ravel(), dyaw.ravel())).astype(
            np.float32
        )

        reverse_points = target_wall_points
        if len(reverse_points) > 768:
            selection = np.linspace(0, len(reverse_points) - 1, 768).astype(
                np.int64
            )
            reverse_points = reverse_points[selection]
        height, width = target_wall.shape
        try:
            with torch.inference_mode():
                distance_gpu = torch.from_numpy(distance).to(device=device)
                target_wall_gpu = torch.from_numpy(target_wall).to(device=device)
                target_free_gpu = torch.from_numpy(target_free).to(device=device)
                target_known_gpu = torch.from_numpy(target_known).to(device=device)
                source_free_mask_gpu = torch.from_numpy(source_free_mask).to(
                    device=device
                )
                source_wall_gpu = torch.from_numpy(source_wall).to(device=device)
                source_free_gpu = torch.from_numpy(source_free).to(device=device)
                reverse_gpu = torch.from_numpy(reverse_points).to(device=device)
                candidates_gpu = torch.from_numpy(candidates).to(device=device)
                pivot_gpu = torch.tensor(
                    pivot, device=device, dtype=torch.float32
                )
                centered_wall = source_wall_gpu - pivot_gpu
                centered_free = source_free_gpu - pivot_gpu
                scores = torch.empty(
                    len(candidates), device=device, dtype=torch.float32
                )

                def indices(points: Any) -> Tuple[Any, Any, Any]:
                    columns = torch.floor(
                        (points[..., 0] + self.grid.half_span_m)
                        / self.grid.resolution_m
                    ).long()
                    rows = torch.floor(
                        (points[..., 1] + self.grid.half_span_m)
                        / self.grid.resolution_m
                    ).long()
                    valid = (
                        (columns >= 0)
                        & (rows >= 0)
                        & (columns < width)
                        & (rows < height)
                    )
                    return (
                        rows.clamp(0, height - 1),
                        columns.clamp(0, width - 1),
                        valid,
                    )

                batch_size = max(1, int(self.config.gpu_batch))
                for begin in range(0, len(candidates), batch_size):
                    current = candidates_gpu[begin:begin + batch_size]
                    yaw = torch.deg2rad(current[:, 2])
                    cosine, sine = torch.cos(yaw), torch.sin(yaw)
                    moved_wall = torch.stack((
                        cosine[:, None] * centered_wall[None, :, 0]
                        - sine[:, None] * centered_wall[None, :, 1],
                        sine[:, None] * centered_wall[None, :, 0]
                        + cosine[:, None] * centered_wall[None, :, 1],
                    ), dim=-1) + pivot_gpu + current[:, None, :2]
                    wall_rows, wall_columns, wall_valid = indices(moved_wall)
                    wall_distance = distance_gpu[wall_rows, wall_columns]
                    wall_known = (
                        target_known_gpu[wall_rows, wall_columns] & wall_valid
                    )
                    wall_affinity = torch.exp(
                        -0.5 * torch.square(wall_distance / 0.10)
                    )
                    wall_hit = (
                        (wall_affinity * wall_known).sum(dim=1)
                        / wall_known.sum(dim=1).clamp_min(1)
                    )
                    wall_known_ratio = wall_known.float().mean(dim=1)
                    wall_free_conflict = (
                        target_free_gpu[wall_rows, wall_columns] & wall_valid
                    ).float().mean(dim=1)

                    if len(source_free):
                        moved_free = torch.stack((
                            cosine[:, None] * centered_free[None, :, 0]
                            - sine[:, None] * centered_free[None, :, 1],
                            sine[:, None] * centered_free[None, :, 0]
                            + cosine[:, None] * centered_free[None, :, 1],
                        ), dim=-1) + pivot_gpu + current[:, None, :2]
                        free_rows, free_columns, free_valid = indices(moved_free)
                        free_agreement = (
                            target_free_gpu[free_rows, free_columns] & free_valid
                        ).float().mean(dim=1)
                        free_wall_conflict = (
                            target_wall_gpu[free_rows, free_columns] & free_valid
                        ).float().mean(dim=1)
                    else:
                        free_agreement = torch.zeros_like(wall_hit)
                        free_wall_conflict = torch.zeros_like(wall_hit)

                    shifted = (
                        reverse_gpu[None, :, :]
                        - pivot_gpu
                        - current[:, None, :2]
                    )
                    inverse = torch.stack((
                        cosine[:, None] * shifted[..., 0]
                        + sine[:, None] * shifted[..., 1],
                        -sine[:, None] * shifted[..., 0]
                        + cosine[:, None] * shifted[..., 1],
                    ), dim=-1) + pivot_gpu
                    reverse_rows, reverse_columns, reverse_valid = indices(inverse)
                    reverse_free_conflict = (
                        source_free_mask_gpu[reverse_rows, reverse_columns]
                        & reverse_valid
                    ).float().mean(dim=1)
                    scores[begin:begin + len(current)] = (
                        0.58 * wall_hit * wall_known_ratio
                        + 0.22 * free_agreement
                        - 0.65 * wall_free_conflict
                        - 1.20 * free_wall_conflict
                        - 0.55 * reverse_free_conflict
                    )
                values = scores.detach().cpu().numpy()
        except Exception as exc:
            self.last_backend_reason = f"cuda_search_failed:{type(exc).__name__}"
            return {"valid": False, "reason": self.last_backend_reason}

        order = np.argsort(values)[::-1]
        peaks: list[Dict[str, float]] = []
        for candidate_index in order:
            candidate = candidates[int(candidate_index)]
            peak = {
                "score": float(values[int(candidate_index)]),
                "dx_m": float(candidate[0]),
                "dy_m": float(candidate[1]),
                "dyaw_deg": float(candidate[2]),
            }
            if all(
                math.hypot(
                    peak["dx_m"] - prior["dx_m"],
                    peak["dy_m"] - prior["dy_m"],
                ) >= 0.20
                or abs(wrap_deg(
                    peak["dyaw_deg"] - prior["dyaw_deg"]
                )) >= 2.0
                for prior in peaks
            ):
                peaks.append(peak)
            if len(peaks) >= 8:
                break
        if not peaks:
            return {"valid": False, "reason": "empty_score_peaks"}
        zero = int(np.argmin(np.sum(np.square(candidates), axis=1)))
        best = peaks[0]
        top_count = min(
            len(candidates), max(16, int(math.ceil(0.01 * len(candidates))))
        )
        top = candidates[order[:top_count]]
        top_near_best = (
            np.hypot(
                top[:, 0] - best["dx_m"], top[:, 1] - best["dy_m"]
            ) <= 0.15
        ) & (
            np.abs([
                wrap_deg(float(value) - best["dyaw_deg"])
                for value in top[:, 2]
            ]) <= 2.0
        )
        quantiles = np.quantile(values, [0.1, 0.5, 0.9])
        robust_spread = max(1e-6, float(quantiles[2] - quantiles[1]))
        probe_report = None
        if probe_correction is not None:
            probe = np.asarray(probe_correction, dtype=np.float32)
            inside_probe = (
                abs(float(probe[0])) <= float(self.config.search_span_m)
                and abs(float(probe[1])) <= float(self.config.search_span_m)
                and abs(float(probe[2])) <= float(self.config.search_span_deg)
            )
            normalized = np.column_stack((
                (candidates[:, 0] - probe[0])
                / max(1e-9, float(self.config.search_step_m)),
                (candidates[:, 1] - probe[1])
                / max(1e-9, float(self.config.search_step_m)),
                (candidates[:, 2] - probe[2])
                / max(1e-9, float(self.config.search_step_deg)),
            ))
            probe_index = int(np.argmin(np.sum(np.square(normalized), axis=1)))
            probe_score = float(values[probe_index])
            probe_report = {
                "inside_search": bool(inside_probe),
                "requested_residual_correction": [
                    float(value) for value in probe
                ],
                "sampled_residual_correction": [
                    float(value) for value in candidates[probe_index]
                ],
                "score": probe_score,
                "score_drop_robust": float(
                    (float(best["score"]) - probe_score) / robust_spread
                ),
                "at_or_above_p90": bool(probe_score >= float(quantiles[2])),
            }
        separated_margin = (
            float(peaks[0]["score"] - peaks[1]["score"])
            if len(peaks) > 1
            else None
        )
        boundary = (
            abs(best["dx_m"])
            >= self.config.search_span_m - 0.25 * self.config.search_step_m
            or abs(best["dy_m"])
            >= self.config.search_span_m - 0.25 * self.config.search_step_m
            or abs(best["dyaw_deg"])
            >= self.config.search_span_deg - 0.25 * self.config.search_step_deg
        )
        coarse_best = dict(best)
        refinement = self._refine_peak_icp(
            source_wall,
            target_wall_points,
            pivot,
            coarse_best,
            torch=torch,
            device=device,
        )
        if refinement.get("accepted"):
            correction = refinement["correction"]
            peaks[0] = {
                **best,
                "dx_m": float(correction[0]),
                "dy_m": float(correction[1]),
                "dyaw_deg": float(correction[2]),
            }
        return {
            "valid": True,
            "source_wall_points": int(len(source_wall)),
            "source_free_points": int(len(source_free)),
            "target_wall_points": int(len(target_wall_points)),
            "baseline_score": float(values[zero]),
            "score_quantiles": [float(value) for value in quantiles],
            "probe_hypothesis": probe_report,
            "best_at_search_boundary": bool(boundary),
            "coarse_best": coarse_best,
            "icp_refinement": refinement,
            "separated_peak_margin": separated_margin,
            "separated_peak_margin_robust": (
                None
                if separated_margin is None
                else separated_margin / robust_spread
            ),
            "best_prominence_robust": float(
                (best["score"] - quantiles[2]) / robust_spread
            ),
            "top_one_percent_cluster_ratio": float(np.mean(top_near_best)),
            "peaks": peaks,
        }


__all__ = [
    "ColumnScan",
    "CONSENSUS_MAX_TRANSLATION_M",
    "CONSENSUS_MAX_YAW_DEG",
    "CONSENSUS_MIN_WINDOWS",
    "DEFAULT_WINDOWS",
    "GEOMETRY_DEVICE_ENV",
    "GeometryConfig",
    "GpuGeometryValidator",
    "GridSpec",
    "TemporalPoseConfig",
    "TemporalPoseEvidence",
    "TemporalPoseHypothesisBank",
    "crop_target_to_source_support",
    "UNIQUE_MIN_PROMINENCE_ROBUST",
    "UNIQUE_MIN_TOP_CLUSTER_RATIO",
    "extract_column_scan",
    "fuse_corrections",
    "rasterize",
    "select_consensus_rows",
    "select_observable_visual_supported_long_window",
    "select_per_query_visual_supported_long_window",
    "select_temporally_independent_rows",
    "select_visual_confirmed_long_window",
    "transform_points",
    "unique_cells",
    "uniqueness_gate",
    "wrap_deg",
]
