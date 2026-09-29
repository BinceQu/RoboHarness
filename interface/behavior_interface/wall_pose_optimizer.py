"""用跨帧直墙重观测离线校正 RGB-D 关键帧位姿。

严格官方录制只保留工具边界 RGB-D，但每帧都带策略层 local odometry。
这里把 local odometry 当主链，只在证据充分的 Manhattan 场景中，用同侧重复
墙面的法向重合和墙方向约束做小幅全局修正。任何证据不足、修正过大或验收不达标
的结果都会被拒绝，调用方应继续使用原始里程。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


MIN_FRAMES = 8
MIN_LINES = 16
MIN_LINE_FRAMES = 6
MIN_LINE_SUPPORT_M = 12.0
MIN_AXIS_COHERENCE = 0.80
MIN_AXIS_INLIER_SUPPORT_RATIO = 0.75
AXIS_INLIER_MAX_DEG = 6.0
ASSOC_NORMAL_M = 0.18
ASSOC_MIN_OVERLAP_M = 0.20
MIN_WALL_PAIRS = 8
MIN_PAIR_OVERLAP_M = 5.0
MAX_POSITION_CORRECTION_M = 0.15
MAX_YAW_CORRECTION_DEG = 4.0
MAX_ORIENTATION_P90_DEG = 1.25
MAX_ORIENTATION_ERROR_DEG = 1.25
MAX_PAIR_P90_CELL_FRACTION = 0.25
MAX_PAIR_ERROR_CELL_FRACTION = 0.50


@dataclass
class LineModel:
    frame: int
    center: np.ndarray
    start: np.ndarray
    end: np.ndarray
    angle_deg: float
    length_m: float
    rms_m: float


@dataclass
class WallObservation:
    line: LineModel
    family: int
    side: int
    normal_offset_m: float
    along_lo_m: float
    along_hi_m: float


@dataclass
class WallPair:
    left: int
    right: int
    overlap_m: float


@dataclass
class PoseOptimizationResult:
    poses: np.ndarray
    accepted: bool
    applied: bool
    reason: str
    stats: Dict[str, Any]


def _wrap_90(value: float) -> float:
    return (float(value) + 45.0) % 90.0 - 45.0


def _weighted_quantile(
    values: Sequence[float], weights: Sequence[float], quantile: float
) -> float:
    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if value.size == 0 or weight.size != value.size or float(weight.sum()) <= 0.0:
        return float("nan")
    order = np.argsort(value)
    cumulative = np.cumsum(weight[order])
    index = int(np.searchsorted(cumulative, float(quantile) * cumulative[-1]))
    return float(value[order[min(index, value.size - 1)]])


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    return _weighted_quantile(values, weights, 0.5)


def _relative_recorded_poses(sm, bundles: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
    from behavior_interface.pose_graph import relative_pose

    absolute = [sm.capture_local_odometry_pose(bundle) for bundle in bundles]
    if not absolute or any(pose is None for pose in absolute):
        return None
    anchor = absolute[0]
    return np.asarray([relative_pose(anchor, pose) for pose in absolute], dtype=np.float64)


def _scan_lines(sm, points: np.ndarray, frame: int) -> List[LineModel]:
    if points is None or points.ndim != 2 or points.shape[0] < sm.SEG_MIN_POINTS:
        return []
    z = points[:, 2]
    radial = np.hypot(points[:, 0], points[:, 1])
    points = points[
        (z >= sm.OBSTACLE_Z_MIN_M)
        & (z <= sm.CHASSIS_BLOCK_Z_MAX_M)
        & (radial <= sm.SCAN_MATCH_MAX_RANGE_M)
    ]
    if points.shape[0] < sm.SEG_MIN_POINTS:
        return []

    # 位姿约束只能信竖直墙面。仅取 15--80cm 的所有点会把沙发、箱子和踢脚线
    # 也拟合成“墙”；同一帧里这些边本来就不互相垂直，任何 yaw 都不可能把它们
    # 同时扶正。这里逐 XY 栅格做与 OccupancyGrid.wall_face_mask 相同的高度带
    # 判定：必须从最低带起连续命中至少两带。
    span = (
        sm.CHASSIS_BLOCK_Z_MAX_M - sm.OBSTACLE_Z_MIN_M
    ) / sm.WALL_BAND_COUNT
    band = np.clip(
        ((points[:, 2] - sm.OBSTACLE_Z_MIN_M) / span).astype(np.int64),
        0,
        sm.WALL_BAND_COUNT - 1,
    )
    cells = np.floor(points[:, :2] / sm.GRID_RES_M).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    bits = np.zeros(int(inverse.max()) + 1, dtype=np.uint16)
    np.bitwise_or.at(bits, inverse, (1 << band).astype(np.uint16))
    contiguous = (bits & (bits + np.uint16(1))) == 0
    face = contiguous & (
        bits >= np.uint16((1 << sm.WALL_BAND_MIN_RUN) - 1)
    )
    points = points[face[inverse]]
    if points.shape[0] < sm.SEG_MIN_POINTS:
        return []

    x = np.asarray(points[:, 0], dtype=np.float64)
    y = np.asarray(points[:, 1], dtype=np.float64)
    radial = np.hypot(x, y)
    bins = np.floor(
        np.degrees(np.arctan2(y, x)) / sm.SEG_ANGLE_BIN_DEG
    ).astype(np.int64)
    order = np.lexsort((radial, bins))
    sorted_bins = bins[order]
    first = np.ones(sorted_bins.shape[0], dtype=bool)
    first[1:] = sorted_bins[1:] != sorted_bins[:-1]
    indices = order[first]
    xs, ys = x[indices], y[indices]
    bs, rs = bins[indices], radial[indices]
    if xs.shape[0] < sm.SEG_MIN_POINTS:
        return []
    cut = np.ones(xs.shape[0], dtype=bool)
    cut[1:] = (np.abs(np.diff(rs)) > sm.SEG_BREAK_JUMP_M) | (np.diff(bs) > 2)
    edges = list(np.flatnonzero(cut)) + [xs.shape[0]]
    found: List[LineModel] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < sm.SEG_MIN_POINTS:
            continue
        px, py = xs[lo:hi], ys[lo:hi]
        for start, end in sm._iepf_splits(px, py):
            sample = np.column_stack([px[start:end + 1], py[start:end + 1]])
            if sample.shape[0] < sm.SEG_MIN_POINTS:
                continue
            center = sample.mean(axis=0)
            covariance = np.cov((sample - center).T)
            if not np.all(np.isfinite(covariance)):
                continue
            _, vectors = np.linalg.eigh(covariance)
            axis = vectors[:, 1]
            along = (sample - center) @ axis
            span = float(along.max() - along.min())
            if span < sm.SEG_MIN_LENGTH_M:
                continue
            normal = np.array([-axis[1], axis[0]], dtype=np.float64)
            rms = float(np.sqrt(np.mean(((sample - center) @ normal) ** 2)))
            if rms > sm.SEG_FIT_TOL_M:
                continue
            found.append(LineModel(
                frame=frame,
                center=center,
                start=center + float(along.min()) * axis,
                end=center + float(along.max()) * axis,
                angle_deg=math.degrees(math.atan2(axis[1], axis[0])) % 180.0,
                length_m=span,
                rms_m=rms,
            ))
    return found


def _axis(lines: Sequence[LineModel], poses: np.ndarray) -> Tuple[float, float]:
    angles = np.asarray([
        (line.angle_deg + poses[line.frame, 2]) % 90.0 for line in lines
    ], dtype=np.float64)
    weights = np.asarray([line.length_m for line in lines], dtype=np.float64)
    phase = np.radians(4.0 * angles)
    vector = np.sum(weights * np.exp(1j * phase))
    coherence = float(abs(vector) / max(float(weights.sum()), 1e-12))
    axis_deg = float((math.degrees(math.atan2(vector.imag, vector.real)) / 4.0) % 90.0)
    return axis_deg, coherence


def _orientation_report(
    lines: Sequence[LineModel], poses: np.ndarray, axis_deg: float
) -> Dict[str, float]:
    errors = [
        abs(_wrap_90(line.angle_deg + poses[line.frame, 2] - axis_deg))
        for line in lines
    ]
    weights = [line.length_m for line in lines]
    return {
        "weighted_median_deg": _weighted_quantile(errors, weights, 0.5),
        "weighted_p90_deg": _weighted_quantile(errors, weights, 0.9),
        "max_deg": float(max(errors)) if errors else float("nan"),
    }


def _corrected_yaws(
    lines: Sequence[LineModel], poses: np.ndarray, axis_deg: float
) -> np.ndarray:
    corrected = poses[:, 2].copy()
    by_frame: Dict[int, List[LineModel]] = {}
    for line in lines:
        by_frame.setdefault(line.frame, []).append(line)
    for frame, own in by_frame.items():
        errors = [
            _wrap_90(axis_deg - line.angle_deg - poses[frame, 2])
            for line in own
        ]
        corrected[frame] += _weighted_median(
            errors, [line.length_m for line in own]
        )
    return corrected


def _wall_observations(
    lines: Sequence[LineModel], poses: np.ndarray, axis_deg: float
) -> List[WallObservation]:
    found: List[WallObservation] = []
    for line in lines:
        x, y, yaw_deg = poses[line.frame]
        angle = math.radians(yaw_deg)
        rotation = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ])
        center = np.array([x, y]) + rotation @ line.center
        start = np.array([x, y]) + rotation @ line.start
        end = np.array([x, y]) + rotation @ line.end
        world_angle = (line.angle_deg + yaw_deg) % 180.0
        family = int(round(((world_angle - axis_deg) % 180.0) / 90.0)) % 2
        direction_angle = math.radians(axis_deg + 90.0 * family)
        direction = np.array([math.cos(direction_angle), math.sin(direction_angle)])
        normal = np.array([-direction[1], direction[0]])
        offset = float(normal @ center)
        robot_offset = float(normal @ np.array([x, y]))
        along = sorted([float(direction @ start), float(direction @ end)])
        found.append(WallObservation(
            line=line,
            family=family,
            side=1 if robot_offset >= offset else -1,
            normal_offset_m=offset,
            along_lo_m=along[0],
            along_hi_m=along[1],
        ))
    return found


def _overlap(left: WallObservation, right: WallObservation) -> float:
    return max(
        0.0,
        min(left.along_hi_m, right.along_hi_m)
        - max(left.along_lo_m, right.along_lo_m),
    )


def _wall_pairs(observations: Sequence[WallObservation]) -> List[WallPair]:
    frames = sorted({observation.line.frame for observation in observations})
    found: List[WallPair] = []
    for frame_index, left_frame in enumerate(frames):
        left_ids = [
            index for index, observation in enumerate(observations)
            if observation.line.frame == left_frame
        ]
        for right_frame in frames[frame_index + 1:]:
            right_ids = [
                index for index, observation in enumerate(observations)
                if observation.line.frame == right_frame
            ]
            candidates: List[Tuple[float, float, int, int]] = []
            for left in left_ids:
                a = observations[left]
                for right in right_ids:
                    b = observations[right]
                    if a.family != b.family or a.side != b.side:
                        continue
                    normal_delta = abs(a.normal_offset_m - b.normal_offset_m)
                    overlap = _overlap(a, b)
                    if normal_delta > ASSOC_NORMAL_M or overlap < ASSOC_MIN_OVERLAP_M:
                        continue
                    candidates.append((normal_delta, -overlap, left, right))
            nearest_left: Dict[int, Tuple[float, float, int, int]] = {}
            nearest_right: Dict[int, Tuple[float, float, int, int]] = {}
            for candidate in sorted(candidates):
                nearest_left.setdefault(candidate[2], candidate)
                nearest_right.setdefault(candidate[3], candidate)
            for candidate in candidates:
                _, negative_overlap, left, right = candidate
                if nearest_left.get(left) != candidate or nearest_right.get(right) != candidate:
                    continue
                found.append(WallPair(left, right, -negative_overlap))
    return found


def _line_normal(axis_deg: float, family: int) -> np.ndarray:
    angle = math.radians(axis_deg + 90.0 * int(family))
    return np.array([-math.sin(angle), math.cos(angle)], dtype=np.float64)


def _line_offset(line: LineModel, pose: np.ndarray, normal: np.ndarray) -> float:
    yaw = math.radians(float(pose[2]))
    rotation = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw), math.cos(yaw)],
    ])
    return float(normal @ (pose[:2] + rotation @ line.center))


def _pair_report(
    observations: Sequence[WallObservation],
    pairs: Sequence[WallPair],
    poses: np.ndarray,
    axis_deg: float,
) -> Dict[str, float]:
    errors: List[float] = []
    weights: List[float] = []
    for pair in pairs:
        left = observations[pair.left]
        right = observations[pair.right]
        normal = _line_normal(axis_deg, left.family)
        errors.append(abs(
            _line_offset(left.line, poses[left.line.frame], normal)
            - _line_offset(right.line, poses[right.line.frame], normal)
        ))
        weights.append(pair.overlap_m)
    if not errors:
        return {"pairs": 0, "overlap_support_m": 0.0}
    values = np.asarray(errors, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    return {
        "pairs": len(pairs),
        "overlap_support_m": float(weight.sum()),
        "weighted_rms_m": float(np.sqrt(np.average(values ** 2, weights=weight))),
        "weighted_p90_m": _weighted_quantile(errors, weights, 0.9),
        "max_m": float(values.max()),
    }


def _solve(
    initial: np.ndarray,
    lines: Sequence[LineModel],
    observations: Sequence[WallObservation],
    pairs: Sequence[WallPair],
    axis_deg: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from scipy.optimize import least_squares
    from behavior_interface.pose_graph import relative_pose, wrap_deg

    odometry = [
        relative_pose(initial[index], initial[index + 1])
        for index in range(initial.shape[0] - 1)
    ]

    def residual(flat: np.ndarray) -> np.ndarray:
        correction = flat.reshape(-1, 3)
        poses = initial + correction
        values: List[float] = []
        values.extend((correction[0] / np.array([0.001, 0.001, 0.02])).tolist())
        for item in correction[1:]:
            values.extend((item / np.array([0.35, 0.35, 5.0])).tolist())
        for index, measured in enumerate(odometry):
            predicted = relative_pose(poses[index], poses[index + 1])
            travel = math.hypot(float(measured[0]), float(measured[1]))
            turn = abs(float(measured[2]))
            sigma_xy = 0.035 + 0.015 * travel
            sigma_yaw = 0.65 + 0.01 * turn
            values.extend([
                (predicted[0] - measured[0]) / sigma_xy,
                (predicted[1] - measured[1]) / sigma_xy,
                wrap_deg(predicted[2] - measured[2]) / sigma_yaw,
            ])
        for line in lines:
            sigma = 0.70 / math.sqrt(max(0.50, line.length_m))
            values.append(
                _wrap_90(line.angle_deg + poses[line.frame, 2] - axis_deg) / sigma
            )
        for pair in pairs:
            left = observations[pair.left]
            right = observations[pair.right]
            normal = _line_normal(axis_deg, left.family)
            delta = (
                _line_offset(left.line, poses[left.line.frame], normal)
                - _line_offset(right.line, poses[right.line.frame], normal)
            )
            sigma = 0.025 / math.sqrt(max(ASSOC_MIN_OVERLAP_M, pair.overlap_m))
            values.append(delta / sigma)
        return np.asarray(values, dtype=np.float64)

    zero = np.zeros_like(initial).reshape(-1)
    lower = np.tile(np.array([-0.50, -0.50, -8.0]), initial.shape[0])
    upper = np.tile(np.array([0.50, 0.50, 8.0]), initial.shape[0])
    before = residual(zero)
    solved = least_squares(
        residual,
        zero,
        bounds=(lower, upper),
        loss="huber",
        f_scale=1.5,
        max_nfev=300,
        x_scale="jac",
    )
    correction = solved.x.reshape(-1, 3)
    after = residual(solved.x)
    return initial + correction, {
        "success": bool(solved.success),
        "status": int(solved.status),
        "message": str(solved.message),
        "function_evaluations": int(solved.nfev),
        "residual_rms_before": float(np.sqrt(np.mean(before ** 2))),
        "residual_rms_after": float(np.sqrt(np.mean(after ** 2))),
        "position_correction_median_m": float(np.median(np.linalg.norm(
            correction[:, :2], axis=1
        ))),
        "position_correction_max_m": float(np.max(np.linalg.norm(
            correction[:, :2], axis=1
        ))),
        "yaw_correction_median_abs_deg": float(np.median(np.abs(correction[:, 2]))),
        "yaw_correction_max_abs_deg": float(np.max(np.abs(correction[:, 2]))),
        "per_frame_correction": correction.astype(float).tolist(),
    }


def _rejected(
    poses: np.ndarray, reason: str, stats: Optional[Dict[str, Any]] = None
) -> PoseOptimizationResult:
    report = dict(stats or {})
    report.update({"accepted": False, "applied": False, "reason": reason})
    return PoseOptimizationResult(poses, False, False, reason, report)


def _relative_to_first(poses: np.ndarray) -> np.ndarray:
    """去掉求解器残留的微小全局规范偏移，让首帧严格等于零。"""
    from behavior_interface.pose_graph import relative_pose

    if poses.shape[0] == 0:
        return poses.copy()
    anchor = poses[0]
    return np.asarray(
        [relative_pose(anchor, pose) for pose in poses], dtype=np.float64
    )


def optimize_capture_poses(
    bundles: Sequence[Dict[str, Any]],
) -> PoseOptimizationResult:
    """返回相对首帧的优化位姿；不满足严格门槛时原样返回官方里程。"""
    from behavior_interface import spatial_map as sm

    poses = _relative_recorded_poses(sm, bundles)
    if poses is None:
        return _rejected(np.zeros((0, 3)), "missing_recorded_local_odometry")
    if len(bundles) < MIN_FRAMES:
        return _rejected(poses, "too_few_frames", {"frames": len(bundles)})

    lines: List[LineModel] = []
    for frame, bundle in enumerate(bundles):
        points = sm._depth_to_robot_points(bundle)
        if points is not None:
            lines.extend(_scan_lines(sm, points, frame))
    line_support = float(sum(line.length_m for line in lines))
    line_frames = len({line.frame for line in lines})
    base_stats: Dict[str, Any] = {
        "frames": len(bundles),
        "line_models": len(lines),
        "line_frames": line_frames,
        "line_support_m": line_support,
    }
    if (
        len(lines) < MIN_LINES
        or line_frames < MIN_LINE_FRAMES
        or line_support < MIN_LINE_SUPPORT_M
    ):
        return _rejected(poses, "insufficient_wall_lines", base_stats)

    axis_deg, coherence = _axis(lines, poses)
    residuals = [
        abs(_wrap_90(line.angle_deg + poses[line.frame, 2] - axis_deg))
        for line in lines
    ]
    inlier_lines = [
        line for line, error in zip(lines, residuals)
        if error <= AXIS_INLIER_MAX_DEG
    ]
    inlier_support = float(sum(line.length_m for line in inlier_lines))
    inlier_ratio = inlier_support / max(line_support, 1e-12)
    base_stats.update({
        "axis_deg_mod_90": axis_deg,
        "axis_coherence": coherence,
        "axis_inlier_support_ratio": inlier_ratio,
        "orientation_before": _orientation_report(inlier_lines, poses, axis_deg),
    })
    if coherence < MIN_AXIS_COHERENCE or inlier_ratio < MIN_AXIS_INLIER_SUPPORT_RATIO:
        return _rejected(poses, "not_manhattan_enough", base_stats)

    association_poses = poses.copy()
    association_poses[:, 2] = _corrected_yaws(inlier_lines, poses, axis_deg)
    observations = _wall_observations(inlier_lines, association_poses, axis_deg)
    pairs = _wall_pairs(observations)
    pair_before = _pair_report(observations, pairs, poses, axis_deg)
    base_stats["wall_pairs_before"] = pair_before
    if (
        len(pairs) < MIN_WALL_PAIRS
        or pair_before.get("overlap_support_m", 0.0) < MIN_PAIR_OVERLAP_M
    ):
        return _rejected(poses, "insufficient_wall_reobservations", base_stats)

    try:
        optimized, solver = _solve(
            poses, inlier_lines, observations, pairs, axis_deg
        )
    except Exception as exc:
        return _rejected(poses, f"solver_failed:{type(exc).__name__}", base_stats)
    pair_after = _pair_report(observations, pairs, optimized, axis_deg)
    orientation_after = _orientation_report(inlier_lines, optimized, axis_deg)
    grid_res = float(sm.GRID_RES_M)
    gates = {
        "solver_success": bool(solver.get("success")),
        "finite_solution": bool(np.all(np.isfinite(optimized))),
        "pair_p90_within_quarter_cell": (
            pair_after.get("weighted_p90_m", math.inf)
            <= MAX_PAIR_P90_CELL_FRACTION * grid_res
        ),
        "pair_max_within_half_cell": (
            pair_after.get("max_m", math.inf)
            <= MAX_PAIR_ERROR_CELL_FRACTION * grid_res
        ),
        "pair_rms_improved": (
            pair_after.get("weighted_rms_m", math.inf)
            <= 0.5 * pair_before.get("weighted_rms_m", 0.0)
        ),
        "orientation_p90_straight": (
            orientation_after.get("weighted_p90_deg", math.inf)
            <= MAX_ORIENTATION_P90_DEG
        ),
        "orientation_max_straight": (
            orientation_after.get("max_deg", math.inf)
            <= MAX_ORIENTATION_ERROR_DEG
        ),
        "position_correction_bounded": (
            solver.get("position_correction_max_m", math.inf)
            <= MAX_POSITION_CORRECTION_M
        ),
        "yaw_correction_bounded": (
            solver.get("yaw_correction_max_abs_deg", math.inf)
            <= MAX_YAW_CORRECTION_DEG
        ),
    }
    accepted = all(gates.values())
    reason = "accepted" if accepted else "post_optimization_gate_failed"
    stats = {
        **base_stats,
        "constraints": {
            "odometry": max(0, len(bundles) - 1),
            "wall_directions": len(inlier_lines),
            "wall_pairs": len(pairs),
        },
        "solver": solver,
        "wall_pairs_after": pair_after,
        "orientation_after": orientation_after,
        "gates": gates,
        "accepted": accepted,
        "applied": accepted,
        "reason": reason,
    }
    return PoseOptimizationResult(
        _relative_to_first(optimized) if accepted else poses,
        accepted,
        accepted,
        reason,
        stats,
    )
