"""Measured three-turn stability gate for live RTAB-Map sessions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .official import SE2Pose
from .protocol import MapResult


@dataclass(frozen=True)
class TurnMetrics:
    turn: int
    closure_translation_m: float
    closure_yaw_deg: float
    edge_p95_m: float
    width_cells: int
    height_cells: int
    loop_count: int


@dataclass(frozen=True)
class ThreeTurnReport:
    passed: bool
    complete_in_place_sequence_observed: bool
    turns: tuple[TurnMetrics, ...]
    max_tracking_loss_s: float
    accepted_loops_after_first_turn: int
    session_reasons: tuple[str, ...]
    reasons: tuple[str, ...]


def _wrap_radians(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _occupied_world_points(result: MapResult) -> np.ndarray:
    occupied = (
        (np.asarray(result.occupancy) >= 65)
        | np.asarray(result.low_obstacles, dtype=bool)
        | np.asarray(result.high_obstacles, dtype=bool)
    )
    rows, columns = np.nonzero(occupied)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack(
        (
            result.x_min_m + (columns.astype(np.float64) + 0.5) * result.cell_size_m,
            result.y_min_m + (rows.astype(np.float64) + 0.5) * result.cell_size_m,
        )
    )


def _symmetric_edge_p95(first: np.ndarray, second: np.ndarray) -> float:
    if first.size == 0 or second.size == 0:
        return math.inf
    first_tree = cKDTree(first)
    second_tree = cKDTree(second)
    first_to_second = second_tree.query(first, k=1, workers=1)[0]
    second_to_first = first_tree.query(second, k=1, workers=1)[0]
    return float(np.percentile(np.concatenate((first_to_second, second_to_first)), 95.0))


class ThreeTurnValidator:
    """Collect release metrics while actual body odometry completes three turns."""

    MAX_IN_PLACE_PATH_M = 0.25
    DIRECTION_LOCK_RAD = math.radians(10.0)
    DIRECTION_REVERSAL_RAD = math.radians(10.0)

    def __init__(self) -> None:
        self._start_map_pose: Optional[SE2Pose] = None
        self._last_odom_pose: Optional[SE2Pose] = None
        self._unwrapped_yaw = 0.0
        self._translation_path_m = 0.0
        self._reverse_yaw = 0.0
        self._direction = 0
        self._turns: list[TurnMetrics] = []
        self._previous_edges: Optional[np.ndarray] = None
        self._first_turn_loop_count = 0
        self._tracking_loss_started: Optional[float] = None
        self._max_tracking_loss_s = 0.0
        self._last_map_shape: Optional[tuple[int, int]] = None
        self._closure_dimension_jump = False

    def _restart_turn_candidate(self, map_pose: SE2Pose) -> None:
        """从当前位置重新寻找连续、近似原地的三圈旋转。"""

        self._start_map_pose = map_pose
        self._unwrapped_yaw = 0.0
        self._translation_path_m = 0.0
        self._reverse_yaw = 0.0
        self._direction = 0
        self._turns.clear()
        self._previous_edges = None
        self._first_turn_loop_count = 0

    def observe(
        self,
        result: MapResult,
        actual_odometry: SE2Pose,
        timestamp_s: float,
    ) -> Optional[TurnMetrics]:
        stamp = float(timestamp_s)
        shape = tuple(int(value) for value in result.occupancy.shape)
        if result.loop_closed and self._last_map_shape is not None:
            self._closure_dimension_jump |= any(
                abs(current - previous) > 2
                for current, previous in zip(shape, self._last_map_shape)
            )
        self._last_map_shape = shape
        if result.tracking_ok:
            if self._tracking_loss_started is not None:
                self._max_tracking_loss_s = max(
                    self._max_tracking_loss_s, stamp - self._tracking_loss_started
                )
                self._tracking_loss_started = None
        elif self._tracking_loss_started is None:
            self._tracking_loss_started = stamp

        if self._last_odom_pose is None:
            self._last_odom_pose = actual_odometry
            self._restart_turn_candidate(result.current_pose)
            return None
        previous_odometry = self._last_odom_pose
        self._last_odom_pose = actual_odometry
        delta = _wrap_radians(
            actual_odometry.yaw_rad - previous_odometry.yaw_rad
        )
        self._translation_path_m += math.hypot(
            actual_odometry.x_m - previous_odometry.x_m,
            actual_odometry.y_m - previous_odometry.y_m,
        )
        self._unwrapped_yaw += delta

        # 普通绕屋导航也可能累计超过 1080 度。只允许低平移、同向的连续
        # 旋转进入三圈门槛，避免把整段导航伪装成原地闭环测试。
        if self._translation_path_m > self.MAX_IN_PLACE_PATH_M:
            self._restart_turn_candidate(result.current_pose)
            return None
        if self._direction == 0 and abs(self._unwrapped_yaw) >= self.DIRECTION_LOCK_RAD:
            self._direction = 1 if self._unwrapped_yaw > 0.0 else -1
        if self._direction != 0:
            if self._direction * delta < 0.0:
                self._reverse_yaw += abs(delta)
            else:
                self._reverse_yaw = 0.0
            if self._reverse_yaw >= self.DIRECTION_REVERSAL_RAD:
                self._restart_turn_candidate(result.current_pose)
                return None
        # Repeated wrapped additions can land a few ulps below exactly 2*pi,
        # otherwise a full turn would be recorded one camera frame late.
        completed = int((abs(self._unwrapped_yaw) + 1e-9) // (2.0 * math.pi))
        if completed <= len(self._turns) or completed > 3:
            return None

        assert self._start_map_pose is not None
        dx = result.current_pose.x_m - self._start_map_pose.x_m
        dy = result.current_pose.y_m - self._start_map_pose.y_m
        translation = math.hypot(dx, dy)
        yaw_error = abs(math.degrees(_wrap_radians(
            result.current_pose.yaw_rad - self._start_map_pose.yaw_rad
        )))
        edges = _occupied_world_points(result)
        edge_p95 = 0.0 if self._previous_edges is None else _symmetric_edge_p95(
            self._previous_edges, edges
        )
        metrics = TurnMetrics(
            completed,
            translation,
            yaw_error,
            edge_p95,
            int(result.occupancy.shape[1]),
            int(result.occupancy.shape[0]),
            int(result.loop_count),
        )
        self._turns.append(metrics)
        self._previous_edges = edges
        if completed == 1:
            self._first_turn_loop_count = result.loop_count
        return metrics

    def report(self, final_timestamp_s: Optional[float] = None) -> ThreeTurnReport:
        max_loss = self._max_tracking_loss_s
        if self._tracking_loss_started is not None and final_timestamp_s is not None:
            max_loss = max(max_loss, float(final_timestamp_s) - self._tracking_loss_started)
        turn_reasons: list[str] = []
        if len(self._turns) != 3:
            turn_reasons.append(
                "no complete three-turn in-place sequence observed "
                f"({len(self._turns)} turns before reset; path must stay <= "
                f"{self.MAX_IN_PLACE_PATH_M:.2f}m)"
            )
        for turn in self._turns:
            if turn.closure_translation_m > 0.05:
                turn_reasons.append(
                    f"turn {turn.turn} translation closure "
                    f"{turn.closure_translation_m:.3f}m > 0.05m"
                )
            if turn.closure_yaw_deg > 1.0:
                turn_reasons.append(
                    f"turn {turn.turn} yaw closure "
                    f"{turn.closure_yaw_deg:.2f}deg > 1deg"
                )
            if turn.turn >= 2 and turn.edge_p95_m > 0.05 + 1e-6:
                turn_reasons.append(
                    f"turn {turn.turn} edge p95 "
                    f"{turn.edge_p95_m:.3f}m > 0.05m"
                )
        session_reasons: list[str] = []
        if max_loss > 0.5:
            session_reasons.append(f"tracking loss lasted {max_loss:.2f}s > 0.5s")
        loops_after_first = 0
        if self._turns:
            loops_after_first = self._turns[-1].loop_count - self._first_turn_loop_count
        # Once a purely in-place trajectory is connected, RTAB-Map may reject
        # later links as topologically redundant. Require evidence that loop
        # detection worked, then judge repeated turns by the pose and map-edge
        # invariants above instead of rewarding duplicate graph constraints.
        if len(self._turns) == 3 and self._turns[-1].loop_count < 1:
            turn_reasons.append("no loop closure accepted across three turns")
        if self._closure_dimension_jump:
            session_reasons.append(
                "map dimensions jumped by more than two cells at a loop closure"
            )
        reasons = turn_reasons + session_reasons
        return ThreeTurnReport(
            not reasons,
            len(self._turns) == 3,
            tuple(self._turns),
            max_loss,
            loops_after_first,
            tuple(session_reasons),
            tuple(reasons),
        )
