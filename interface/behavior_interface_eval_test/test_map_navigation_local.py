"""Pure-grid tests for the official-v2 map-independent path planner."""

from __future__ import annotations

import math
import unittest

import numpy as np

from behavior_interface_eval_test.tool.official_v2.map_navigation_local import (
    START_EGRESS_SAFETY_MARGIN_M,
    _segment_clearance_certificate,
    _traversed_raster_route_between,
    certify_fixed_yaw_local_translation,
    plan_clearance_path,
)
from behavior_interface_eval_test.tool.official_v2.navigation_footprint_local import (
    base_navigation_polygon,
)


UNKNOWN = -1
FREE = 0
OCCUPIED = 100
GOAL_NAME = "marked_goal"


def _cell_xy(
    row: int,
    column: int,
    *,
    resolution: float,
    origin: tuple[float, float],
) -> list[float]:
    return [
        float(origin[0] + (column + 0.5) * resolution),
        float(origin[1] + (row + 0.5) * resolution),
    ]


def _snapshot(
    occupancy: np.ndarray,
    *,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    resolution: float = 0.1,
    origin: tuple[float, float] = (0.0, 0.0),
    start_xy: list[float] | None = None,
    goal_xy: list[float] | None = None,
) -> dict:
    """Build the backend-neutral snapshot consumed by the planner."""

    start = start_xy or _cell_xy(
        *start_cell,
        resolution=resolution,
        origin=origin,
    )
    goal = goal_xy or _cell_xy(
        *goal_cell,
        resolution=resolution,
        origin=origin,
    )
    return {
        "occupancy": np.asarray(occupancy, dtype=np.int16).tolist(),
        "origin": [float(origin[0]), float(origin[1])],
        "resolution": float(resolution),
        "pose": {
            "x": float(start[0]),
            "y": float(start[1]),
            "yaw_deg": 0.0,
        },
        "places": [
            {
                "name": GOAL_NAME,
                "x": float(goal[0]),
                "y": float(goal[1]),
            }
        ],
        "map_version": "fixture-map-v1",
        "episode_id": "fixture-episode",
    }


def _path_xy(result: dict) -> np.ndarray:
    path = np.asarray(result.get("path_xy_m", []), dtype=np.float64)
    if path.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return path.reshape(-1, 2)


def _assert_polyline_only_uses_known_free(
    case: unittest.TestCase,
    snapshot: dict,
    path: np.ndarray,
) -> None:
    """Resample every simplified segment; unknown is deliberately non-free."""

    occupancy = np.asarray(snapshot["occupancy"], dtype=np.int16)
    resolution = float(snapshot["resolution"])
    origin_x, origin_y = (float(value) for value in snapshot["origin"])
    case.assertGreaterEqual(len(path), 2)
    for start, end in zip(path[:-1], path[1:]):
        distance = float(np.linalg.norm(end - start))
        sample_count = max(2, int(math.ceil(distance / (0.1 * resolution))) + 1)
        for alpha in np.linspace(0.0, 1.0, sample_count):
            point = start + float(alpha) * (end - start)
            column = int(math.floor((float(point[0]) - origin_x) / resolution))
            row = int(math.floor((float(point[1]) - origin_y) / resolution))
            case.assertGreaterEqual(row, 0)
            case.assertGreaterEqual(column, 0)
            case.assertLess(row, occupancy.shape[0])
            case.assertLess(column, occupancy.shape[1])
            case.assertEqual(
                int(occupancy[row, column]),
                FREE,
                msg=(
                    f"simplified segment {start.tolist()} -> {end.tolist()} "
                    f"entered non-free cell {(row, column)}"
                ),
            )


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return math.dist(point, start)
    projection = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_sq
    projection = min(1.0, max(0.0, projection))
    closest = (
        start[0] + projection * dx,
        start[1] + projection * dy,
    )
    return math.dist(point, closest)


def _point_to_aabb_distance(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> float:
    x_min, x_max, y_min, y_max = bounds
    dx = max(x_min - point[0], 0.0, point[0] - x_max)
    dy = max(y_min - point[1], 0.0, point[1] - y_max)
    return math.hypot(dx, dy)


def _segment_intersects_aabb(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    """Liang-Barsky intersection against a closed axis-aligned box."""

    x_min, x_max, y_min, y_max = bounds
    t_min, t_max = 0.0, 1.0
    for coordinate, delta, lower, upper in (
        (start[0], end[0] - start[0], x_min, x_max),
        (start[1], end[1] - start[1], y_min, y_max),
    ):
        if abs(delta) <= 1e-15:
            if coordinate < lower or coordinate > upper:
                return False
            continue
        enter = (lower - coordinate) / delta
        leave = (upper - coordinate) / delta
        if enter > leave:
            enter, leave = leave, enter
        t_min = max(t_min, enter)
        t_max = min(t_max, leave)
        if t_min > t_max:
            return False
    return True


def _segment_to_aabb_distance(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> float:
    if _segment_intersects_aabb(start, end, bounds):
        return 0.0
    x_min, x_max, y_min, y_max = bounds
    corners = (
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_min),
        (x_max, y_max),
    )
    return min(
        _point_to_aabb_distance(start, bounds),
        _point_to_aabb_distance(end, bounds),
        *(_point_to_segment_distance(corner, start, end) for corner in corners),
    )


def _segment_clearances_to_occupied_cells(
    snapshot: dict,
    path: np.ndarray,
) -> np.ndarray:
    """Exact distance from each polyline segment to occupied cell squares."""

    occupancy = np.asarray(snapshot["occupancy"], dtype=np.int16)
    resolution = float(snapshot["resolution"])
    origin_x, origin_y = (float(value) for value in snapshot["origin"])
    occupied_rows, occupied_columns = np.nonzero(occupancy >= 50)
    clearances = []
    for start_value, end_value in zip(path[:-1], path[1:]):
        start = (float(start_value[0]), float(start_value[1]))
        end = (float(end_value[0]), float(end_value[1]))
        minimum = math.inf
        for row, column in zip(occupied_rows, occupied_columns):
            x_min = origin_x + float(column) * resolution
            y_min = origin_y + float(row) * resolution
            bounds = (
                x_min,
                x_min + resolution,
                y_min,
                y_min + resolution,
            )
            minimum = min(
                minimum,
                _segment_to_aabb_distance(start, end, bounds),
            )
        clearances.append(minimum)
    return np.asarray(clearances, dtype=np.float64)


def _polyline_clearance_to_occupied_cells(
    snapshot: dict,
    path: np.ndarray,
) -> float:
    """Exact minimum distance from a polyline to occupied cell squares."""

    clearances = _segment_clearances_to_occupied_cells(snapshot, path)
    return float(np.min(clearances, initial=math.inf))


def _assert_segment_clearance_certificates(
    case: unittest.TestCase,
    snapshot: dict,
    result: dict,
) -> None:
    path = _path_xy(result)
    certificates = np.asarray(
        result.get("path_segment_clearance_m"), dtype=np.float64
    ).reshape(-1)
    actual = _segment_clearances_to_occupied_cells(snapshot, path)
    case.assertEqual(certificates.shape, (len(path) - 1,))
    case.assertTrue(np.all(np.isfinite(certificates)))
    start_egress_count = int(result.get("start_egress_segment_count", 0))
    segment_requirements = np.asarray(
        [
            (
                float(result["start_egress_required_clearance_m"])
                if index < start_egress_count
                else float(result["required_clearance_m"])
            )
            for index in range(len(path) - 1)
        ],
        dtype=np.float64,
    )
    case.assertTrue(
        np.all(certificates + 1e-9 >= segment_requirements),
        msg=(
            f"segment certificate falls below required clearance: "
            f"certificates={certificates.tolist()}, "
            f"requirements={segment_requirements.tolist()}"
        ),
    )
    np.testing.assert_array_less(
        certificates,
        actual + 2e-9,
        err_msg=(
            "reported segment clearance must not overstate the exact "
            "segment-to-occupied-AABB distance"
        ),
    )


class MapNavigationLocalTest(unittest.TestCase):
    def test_fixed_yaw_local_translation_certifies_only_monotone_egress(
        self,
    ) -> None:
        occupancy = np.full((100, 100), FREE, dtype=np.int16)
        snapshot = _snapshot(
            occupancy,
            start_cell=(40, 40),
            goal_cell=(70, 70),
            resolution=0.05,
        )
        polygon = base_navigation_polygon()
        start = np.asarray(
            [snapshot["pose"]["x"], snapshot["pose"]["y"]],
            dtype=np.float64,
        )
        low_vertex = polygon[int(np.argmin(polygon[:, 1]))]
        point_below = start + low_vertex + [0.0, -0.01]
        reference = {"traversed_route_evidence_used": False}

        separating = certify_fixed_yaw_local_translation(
            snapshot,
            reference,
            polygon,
            start,
            start + [0.0, 0.15],
            0.0,
            np.asarray([point_below]),
        )
        approaching = certify_fixed_yaw_local_translation(
            snapshot,
            reference,
            polygon,
            start,
            start + [0.0, -0.15],
            0.0,
            np.asarray([point_below]),
        )

        self.assertIsNotNone(separating)
        self.assertTrue(separating["monotone_egress"])
        self.assertAlmostEqual(
            separating["initial_depth_clearance_m"], 0.01, places=6
        )
        self.assertIsNone(approaching)

    def test_relocalization_lifecycle_blocks_planning(self) -> None:
        occupancy = np.full((20, 24), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(5, 5),
            goal_cell=(14, 18),
        )
        snapshot["pose"]["global_confident"] = False
        snapshot["lifecycle"] = {
            "pose_confident": False,
            "recovery_hold": True,
        }

        with self.assertRaisesRegex(ValueError, "relocalizing"):
            plan_clearance_path(snapshot, GOAL_NAME)

    def test_advisory_confidence_miss_without_hold_still_plans(self) -> None:
        occupancy = np.full((20, 24), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(5, 5),
            goal_cell=(14, 18),
        )
        snapshot["pose"]["global_confident"] = False
        snapshot["lifecycle"] = {
            "pose_confident": False,
            "recovery_hold": False,
            "uncertain_hold": False,
        }

        result = plan_clearance_path(snapshot, GOAL_NAME)

        self.assertTrue(result["ok"], result)

    def test_clearance_weight_prefers_wider_corridor_over_shorter_corridor(
        self,
    ) -> None:
        occupancy = np.full((45, 70), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        # The upper route is shorter but has a 1.0 m wall-to-wall width.  The
        # lower route is wider, so a strong clearance cost should choose it.
        occupancy[11:24, 20:50] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(15, 7),
            goal_cell=(15, 62),
        )

        shortest = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.10,
            safety_margin_m=0.05,
            clearance_weight=0.0,
        )
        safest = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.10,
            safety_margin_m=0.05,
            clearance_weight=8.0,
        )

        self.assertTrue(shortest["ok"], shortest)
        self.assertTrue(safest["ok"], safest)
        shortest_path = _path_xy(shortest)
        safest_path = _path_xy(safest)
        obstacle_top_y = 11 * snapshot["resolution"]
        obstacle_bottom_y = 24 * snapshot["resolution"]
        self.assertLess(float(np.min(shortest_path[:, 1])), obstacle_top_y)
        self.assertGreater(float(np.max(safest_path[:, 1])), obstacle_bottom_y)
        self.assertGreater(
            float(safest["minimum_clearance_m"]),
            float(shortest["minimum_clearance_m"]),
        )
        self.assertGreater(
            float(safest["path_length_m"]),
            float(shortest["path_length_m"]),
        )

    def test_polyline_simplification_preserves_clearance_detour(self) -> None:
        occupancy = np.full((50, 80), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(5, 8),
            goal_cell=(5, 71),
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.10,
            safety_margin_m=0.05,
            clearance_weight=12.0,
        )

        self.assertTrue(result["ok"], result)
        grid_rows = [cell[0] for cell in result["grid_path"]]
        corner_path = np.asarray(result["corner_path_xy_m"])
        # The direct row is only 0.45 m from the boundary wall.  A* moves into
        # the room for clearance, and simplification must not pull it back.
        self.assertGreater(max(grid_rows), 10)
        self.assertGreater(float(np.max(corner_path[:, 1])), 1.0)
        self.assertGreater(float(result["mean_clearance_m"]), 0.70)

    def test_footprint_inflation_rejects_narrow_gate_and_uses_wide_gate(
        self,
    ) -> None:
        occupancy = np.full((32, 52), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        barrier_column = 26
        occupancy[1:-1, barrier_column] = OCCUPIED
        occupancy[6:11, barrier_column] = FREE
        occupancy[18:29, barrier_column] = FREE
        snapshot = _snapshot(
            occupancy,
            start_cell=(8, 5),
            goal_cell=(8, 46),
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.06,
            clearance_weight=2.0,
        )

        self.assertTrue(result["ok"], result)
        path = _path_xy(result)
        self.assertGreater(float(np.max(path[:, 1])), 1.8)
        self.assertGreaterEqual(float(result["minimum_clearance_m"]), 0.26 - 1e-6)
        _assert_polyline_only_uses_known_free(self, snapshot, path)

    def test_unknown_cells_block_the_direct_shortcut(self) -> None:
        occupancy = np.full((28, 36), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[1:23, 16:19] = UNKNOWN
        snapshot = _snapshot(
            occupancy,
            start_cell=(5, 5),
            goal_cell=(5, 30),
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.0,
            safety_margin_m=0.0,
            clearance_weight=0.0,
        )

        self.assertTrue(result["ok"], result)
        path = _path_xy(result)
        self.assertGreater(float(np.max(path[:, 1])), 2.3)
        _assert_polyline_only_uses_known_free(self, snapshot, path)

    def test_polyline_simplification_never_cuts_an_obstacle_corner(self) -> None:
        occupancy = np.full((24, 24), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[8:16, 8:16] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(5, 5),
            goal_cell=(18, 18),
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.10,
            safety_margin_m=0.05,
            clearance_weight=2.0,
        )

        self.assertTrue(result["ok"], result)
        path = _path_xy(result)
        self.assertGreaterEqual(len(path), 3)
        self.assertGreater(len(result["grid_path"]), len(path))
        _assert_polyline_only_uses_known_free(self, snapshot, path)
        # A direct start-to-goal simplification would cross the square.
        direct_midpoint = 0.5 * (path[0] + path[-1])
        self.assertTrue(0.8 <= direct_midpoint[0] < 1.6)
        self.assertTrue(0.8 <= direct_midpoint[1] < 1.6)

    def test_simplified_polyline_preserves_continuous_clearance(self) -> None:
        occupancy = np.full((40, 40), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[20, 20] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(14, 20),
            goal_cell=(24, 19),
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.02,
            safety_margin_m=0.0,
            clearance_weight=0.0,
            arrival_tolerance_m=0.05,
        )

        self.assertTrue(result["ok"], result)
        path = np.asarray(result["corner_path_xy_m"], dtype=np.float64)
        actual_clearance = _polyline_clearance_to_occupied_cells(snapshot, path)
        self.assertGreaterEqual(
            actual_clearance + 1e-9,
            float(result["required_clearance_m"]),
            msg=(
                f"continuous polyline clearance {actual_clearance:.9f} m is below "
                f"required {result['required_clearance_m']:.9f} m: {path.tolist()}"
            ),
        )
        self.assertLessEqual(
            float(result["minimum_clearance_m"]),
            actual_clearance + 1e-9,
            msg="reported minimum clearance must not overstate the executed polyline",
        )
        _assert_segment_clearance_certificates(self, snapshot, result)

    def test_goal_is_snapped_to_nearby_navigable_cell(self) -> None:
        occupancy = np.full((24, 30), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        start_cell = (5, 5)
        goal_cell = (17, 24)
        occupancy[goal_cell] = UNKNOWN
        snapshot = _snapshot(
            occupancy,
            start_cell=start_cell,
            goal_cell=goal_cell,
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.0,
            safety_margin_m=0.0,
            clearance_weight=1.0,
            arrival_tolerance_m=0.25,
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["goal"]["name"], GOAL_NAME)
        requested = np.asarray(result["goal"]["requested_xy_m"], dtype=np.float64)
        planned = np.asarray(result["goal"]["planned_xy_m"], dtype=np.float64)
        self.assertFalse(np.allclose(requested, planned))
        self.assertTrue(result["snap"]["goal"]["applied"])
        self.assertGreater(result["snap"]["goal"]["distance_m"], 0.0)
        self.assertLessEqual(result["snap"]["goal"]["distance_m"], 0.25 + 1e-9)
        self.assertFalse(result["snap"]["start"]["applied"])
        _assert_polyline_only_uses_known_free(self, snapshot, _path_xy(result))

    def test_goal_snap_selects_nearest_metric_cell_not_row_order(self) -> None:
        occupancy = np.full((20, 24), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[5, 5] = UNKNOWN
        snapshot = _snapshot(
            occupancy,
            start_cell=(12, 16),
            goal_cell=(5, 5),
            goal_xy=[0.599, 0.55],
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.0,
            safety_margin_m=0.0,
            clearance_weight=0.0,
            arrival_tolerance_m=0.06,
        )

        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["planned_goal_xy_m"][0], 0.65)
        self.assertAlmostEqual(result["planned_goal_xy_m"][1], 0.55)
        self.assertAlmostEqual(result["goal_standoff_m"], 0.051)

    def test_goal_snap_cannot_exceed_arrival_tolerance(self) -> None:
        occupancy = np.full((24, 30), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        goal_cell = (12, 23)
        # Every free cell is at least 0.30 m from the requested goal cell.
        occupancy[10:15, 21:26] = UNKNOWN
        snapshot = _snapshot(
            occupancy,
            start_cell=(12, 5),
            goal_cell=goal_cell,
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.0,
            safety_margin_m=0.0,
            clearance_weight=1.0,
            arrival_tolerance_m=0.25,
        )

        self.assertFalse(result["ok"], result)
        self.assertEqual(result.get("failure_stage"), "planning")
        self.assertEqual(result.get("path_xy_m"), [])
        self.assertTrue(str(result.get("error", "")).strip())

    def test_exact_goal_near_cell_edge_is_snapped_to_safe_center(self) -> None:
        occupancy = np.full((24, 30), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[1:-1, 9] = OCCUPIED
        # Cell (12, 11) is traversable at its center, but this exact point is
        # only 10.1 cm from the occupied cell boundary. Retaining it would
        # spend the caller's requested 11 cm clearance margin.
        snapshot = _snapshot(
            occupancy,
            start_cell=(12, 20),
            goal_cell=(12, 11),
            goal_xy=[1.101, 1.25],
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.0,
            safety_margin_m=0.11,
            clearance_weight=1.0,
            arrival_tolerance_m=0.08,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["snap"]["goal"]["applied"])
        self.assertAlmostEqual(result["planned_goal_xy_m"][0], 1.15)
        self.assertAlmostEqual(result["planned_goal_xy_m"][1], 1.25)
        self.assertAlmostEqual(result["goal_standoff_m"], 0.049)
        self.assertGreaterEqual(
            result["minimum_clearance_m"],
            result["required_clearance_m"] - 1e-9,
        )

    def test_goal_standoff_uses_tolerance_to_shorten_a_safe_route(self) -> None:
        occupancy = np.full((40, 80), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(20, 10),
            goal_cell=(20, 65),
        )

        exact = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.10,
            safety_margin_m=0.05,
            arrival_tolerance_m=0.45,
        )
        standoff = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.10,
            safety_margin_m=0.05,
            arrival_tolerance_m=0.45,
            optimize_goal_standoff=True,
        )

        self.assertTrue(exact["ok"], exact)
        self.assertTrue(standoff["ok"], standoff)
        self.assertGreater(standoff["goal_standoff_m"], 0.20)
        self.assertLessEqual(standoff["goal_standoff_m"], 0.37 + 1e-9)
        self.assertLess(standoff["path_length_m"], exact["path_length_m"])
        self.assertEqual(
            standoff["path_xy_m"][-1], standoff["planned_goal_xy_m"]
        )
        _assert_segment_clearance_certificates(self, snapshot, standoff)

    def test_safe_exact_start_does_not_keep_a_grid_center_micro_segment(self) -> None:
        occupancy = np.full((40, 80), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        start_xy = [1.02, 2.03]
        start_cell = (20, 10)
        snapshot = _snapshot(
            occupancy,
            start_cell=start_cell,
            goal_cell=(20, 65),
            start_xy=start_xy,
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.10,
            safety_margin_m=0.05,
            clearance_weight=1.0,
            arrival_tolerance_m=0.25,
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["start_snap_m"], 0.0)
        np.testing.assert_allclose(result["corner_path_xy_m"][0], start_xy)
        grid_center = _cell_xy(
            *start_cell,
            resolution=float(snapshot["resolution"]),
            origin=tuple(snapshot["origin"]),
        )
        self.assertFalse(
            any(
                np.allclose(point, grid_center, atol=1.0e-12, rtol=0.0)
                for point in result["corner_path_xy_m"][1:]
            )
        )
        np.testing.assert_allclose(result["start_safe_xy_m"], start_xy)
        self.assertEqual(result["start_egress_segment_count"], 0)
        _assert_segment_clearance_certificates(self, snapshot, result)

    def test_safe_nonzero_start_snap_is_certified_and_explicit(self) -> None:
        occupancy = np.full((50, 50), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[:, :11] = OCCUPIED
        # This pose has the reduced egress margin, not the full route margin.
        # It must first reach a point with the full margin explicitly.
        start_xy = [1.59, 2.55]
        snapshot = _snapshot(
            occupancy,
            start_cell=(25, 15),
            goal_cell=(25, 35),
            start_xy=start_xy,
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.43,
            safety_margin_m=0.12,
            clearance_weight=2.0,
            arrival_tolerance_m=0.25,
        )

        self.assertTrue(result["ok"], result)
        self.assertGreater(result["start_snap_m"], 0.0)
        self.assertTrue(result["start_egress_certified"])
        self.assertTrue(result["start"]["egress_certified"])
        self.assertTrue(result["snap"]["start"]["egress_certified"])
        self.assertTrue(result["snap"]["start"]["applied"])

        path = _path_xy(result)
        safe_xy = np.asarray(result["start_safe_xy_m"], dtype=np.float64)
        self.assertEqual(safe_xy.shape, (2,))
        self.assertTrue(np.all(np.isfinite(safe_xy)))
        np.testing.assert_allclose(path[0], start_xy, atol=1e-12)
        np.testing.assert_allclose(path[1], safe_xy, atol=1e-12)
        np.testing.assert_allclose(
            result["start"]["safe_grid_xy_m"], safe_xy, atol=1e-12
        )
        self.assertAlmostEqual(
            math.dist(start_xy, safe_xy), result["start_snap_m"]
        )

        egress_clearance = _polyline_clearance_to_occupied_cells(
            snapshot, path[:2]
        )
        self.assertEqual(result["start_egress_segment_count"], 1)
        self.assertAlmostEqual(
            result["start_egress_required_clearance_m"],
            0.43 + START_EGRESS_SAFETY_MARGIN_M,
        )
        self.assertGreaterEqual(
            egress_clearance + 1e-9,
            result["start_egress_required_clearance_m"],
            msg=(
                f"certified start egress clearance {egress_clearance:.9f} m is "
                "below its start-egress requirement "
                f"{result['start_egress_required_clearance_m']:.9f} m"
            ),
        )
        self.assertTrue(
            all(
                certificate + 1e-9 >= result["required_clearance_m"]
                for certificate in result["path_segment_clearance_m"][1:]
            )
        )
        _assert_segment_clearance_certificates(self, snapshot, result)

    def test_grid_snap_cannot_invent_a_turn_when_direct_egress_is_safe(self) -> None:
        occupancy = np.zeros((50, 50), dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[:, :11] = OCCUPIED
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                start, goal = [1.69, 2.53], [3.55, 4.05]
                for _ in range(rotation):
                    start, goal = [start[1], 5.0 - start[0]], [goal[1], 5.0 - goal[0]]
                snapshot = _snapshot(
                    np.rot90(occupancy, rotation), start_cell=(0, 0), goal_cell=(0, 0),
                    start_xy=start, goal_xy=goal,
                )
                result = plan_clearance_path(
                    snapshot, GOAL_NAME, robot_radius_m=0.43, safety_margin_m=0.12,
                    clearance_weight=2.0, require_turning_pockets=True,
                )
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["start_snap_m"], 0.0)
                self.assertEqual(result["start_egress_segment_count"], 0)
                np.testing.assert_allclose(result["start_safe_xy_m"], start)
                _assert_segment_clearance_certificates(self, snapshot, result)

    def test_unsafe_start_is_not_snapped_across_unverified_egress(self) -> None:
        occupancy = np.full((24, 30), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[1:-1, 10] = OCCUPIED
        # The requested pose is 1 mm from the wall. Its cell center is safe for
        # the 2 cm footprint, but moving there would begin with an unsafe segment.
        snapshot = _snapshot(
            occupancy,
            start_cell=(12, 11),
            goal_cell=(12, 20),
            start_xy=[1.101, 1.25],
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.02,
            safety_margin_m=0.0,
            clearance_weight=1.0,
            arrival_tolerance_m=0.08,
        )

        self.assertFalse(result["ok"], result)
        self.assertEqual(result.get("failure_stage"), "planning")
        self.assertEqual(result.get("path_xy_m"), [])

    def test_zero_radius_start_cannot_touch_closed_obstacle_boundary(self) -> None:
        occupancy = np.full((24, 30), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[1:-1, 10] = OCCUPIED
        # x=1.1 is both the wall's closed right edge and the free cell's left
        # edge. A zero-radius point still collides when it touches occupied space.
        snapshot = _snapshot(
            occupancy,
            start_cell=(12, 11),
            goal_cell=(12, 20),
            start_xy=[1.1, 1.25],
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.0,
            safety_margin_m=0.0,
            clearance_weight=0.0,
            arrival_tolerance_m=0.08,
        )

        self.assertFalse(result["ok"], result)
        self.assertEqual(result.get("failure_stage"), "planning")
        self.assertEqual(result.get("path_xy_m"), [])

    def test_coarsening_does_not_promote_unknown_source_cells(self) -> None:
        occupancy = np.full((28, 44), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        # A two-fine-cell-thick unknown barrier has one unknown in each 2x2
        # coarse block. A 75%-free vote used to erase it completely.
        occupancy[1:-1:2, 20:22] = UNKNOWN
        snapshot = _snapshot(
            occupancy,
            start_cell=(14, 8),
            goal_cell=(14, 36),
            resolution=0.05,
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.0,
            safety_margin_m=0.0,
            clearance_weight=0.0,
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["unknown_space_policy"], "blocked")
        self.assertTrue(result["planning_resolution_fallback_used"])
        _assert_polyline_only_uses_known_free(
            self, snapshot, _path_xy(result)
        )

    def test_native_resolution_fallback_preserves_a_known_narrow_passage(
        self,
    ) -> None:
        # Nineteen 5 cm free rows form a 0.95 m observed corridor.  Conservative
        # 2x2 aggregation leaves only nine 10 cm rows and its cell-area
        # clearance certificate closes the route.  The unchanged native grid
        # still certifies the 0.84 m base with a 4 cm safety margin.
        occupancy = np.full((21, 120), OCCUPIED, dtype=np.int16)
        occupancy[1:-1, 1:-1] = FREE
        snapshot = _snapshot(
            occupancy,
            start_cell=(10, 10),
            goal_cell=(10, 109),
            resolution=0.05,
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.42,
            safety_margin_m=0.08,
            safety_margin_fallbacks_m=(0.06, 0.04, 0.02),
            clearance_weight=0.0,
            arrival_tolerance_m=0.45,
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["planning_resolution_trials_m"], [0.10, 0.05])
        self.assertEqual(result["planning_resolution_trial_index"], 1)
        self.assertTrue(result["planning_resolution_fallback_used"])
        self.assertAlmostEqual(result["planning_resolution_m"], 0.05)
        self.assertAlmostEqual(result["effective_safety_margin_m"], 0.04)
        _assert_polyline_only_uses_known_free(
            self, snapshot, _path_xy(result)
        )

    def test_diagonal_obstacle_clearance_uses_cell_area_not_cell_center(
        self,
    ) -> None:
        occupancy = np.full((16, 20), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[7, 9] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(8, 10),
            goal_cell=(8, 16),
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.08,
            safety_margin_m=0.0,
            clearance_weight=0.0,
            arrival_tolerance_m=0.05,
        )

        # The start center is diagonally adjacent to the occupied 10 cm cell:
        # its true distance to that square is sqrt(2) * 5 cm, below 8 cm.
        self.assertFalse(result["ok"], result)
        self.assertTrue(
            "footprint-safe" in result["error"]
            or "snap beyond" in result["error"]
        )

    def test_requested_safety_margin_is_not_silently_relaxed(self) -> None:
        occupancy = np.full((7, 40), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(3, 4),
            goal_cell=(3, 35),
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.10,
            clearance_weight=2.0,
        )

        self.assertFalse(result["ok"], result)
        self.assertEqual(result.get("failure_stage"), "planning")
        self.assertEqual(result.get("path_xy_m"), [])

    def test_explicit_margin_fallback_uses_the_widest_feasible_policy(self) -> None:
        occupancy = np.full((7, 40), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(3, 4),
            goal_cell=(3, 35),
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.10,
            safety_margin_fallbacks_m=(0.06, 0.02),
            clearance_weight=2.0,
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["safety_margin_trials_m"], [0.10, 0.06, 0.02])
        self.assertEqual(result["safety_margin_trial_index"], 2)
        self.assertAlmostEqual(result["effective_safety_margin_m"], 0.02)
        self.assertTrue(result["safety_margin_relaxed"])
        self.assertGreaterEqual(
            result["minimum_clearance_m"], result["required_clearance_m"]
        )

    def test_long_search_calls_progress_check_and_propagates_abort(self) -> None:
        occupancy = np.full((100, 100), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[1:90, 50] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(50, 10),
            goal_cell=(50, 90),
        )
        calls = 0

        class PlanningAborted(RuntimeError):
            pass

        def progress_check() -> None:
            nonlocal calls
            calls += 1
            if calls >= 6:
                raise PlanningAborted("cancelled by execution deadline")

        with self.assertRaisesRegex(PlanningAborted, "execution deadline"):
            plan_clearance_path(
                snapshot,
                GOAL_NAME,
                robot_radius_m=0.0,
                safety_margin_m=0.0,
                clearance_weight=2.0,
                progress_check=progress_check,
            )
        self.assertGreaterEqual(calls, 6)

    def test_unreachable_goal_returns_structured_planning_failure(self) -> None:
        occupancy = np.full((22, 32), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[1:-1, 16] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(10, 5),
            goal_cell=(10, 26),
        )

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.0,
            safety_margin_m=0.0,
            clearance_weight=2.0,
        )

        self.assertFalse(result["ok"], result)
        self.assertEqual(result.get("failure_stage"), "planning")
        self.assertEqual(result.get("path_xy_m"), [])
        self.assertEqual(result.get("grid_path"), [])
        self.assertIn("unreachable", str(result.get("error", "")).lower())

    def test_traversed_footprint_repairs_only_its_unknown_route(self) -> None:
        occupancy = np.full((80, 80), UNKNOWN, dtype=np.int16)
        snapshot = _snapshot(
            occupancy,
            start_cell=(40, 10),
            goal_cell=(40, 60),
        )
        snapshot["traversed_paths_xy_m"] = [
            [_cell_xy(40, column, resolution=0.1, origin=(0.0, 0.0))
             for column in range(10, 61, 5)]
        ]

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.42,
            safety_margin_m=0.08,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=2.0,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_evidence_used"])
        self.assertTrue(result["planning_evidence_fallback_used"])
        self.assertEqual(result["planning_evidence_trial_index"], 1)
        self.assertAlmostEqual(result["effective_safety_margin_m"], 0.0)
        self.assertGreater(result["traversed_route_promoted_cells"], 0)
        self.assertGreaterEqual(
            min(result["path_segment_clearance_m"]), 0.42
        )

        blocked = np.asarray(snapshot["occupancy"], dtype=np.int16)
        blocked[:, 35] = OCCUPIED
        snapshot["occupancy"] = blocked.tolist()
        nonwall_override = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.42,
            safety_margin_m=0.08,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=2.0,
        )
        self.assertTrue(nonwall_override["ok"], nonwall_override)
        self.assertEqual(
            nonwall_override["planning_evidence_trial_index"], 2
        )
        self.assertGreater(
            nonwall_override[
                "traversed_route_overridden_nonwall_obstacle_cells"
            ],
            0,
        )

        wall_mask = np.zeros_like(blocked, dtype=bool)
        wall_mask[:, 35] = True
        snapshot["wall_mask"] = wall_mask.tolist()
        wall_override = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.42,
            safety_margin_m=0.08,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=2.0,
        )
        self.assertTrue(wall_override["ok"], wall_override)
        self.assertEqual(wall_override["planning_evidence_trial_index"], 3)
        self.assertGreater(
            wall_override["traversed_route_overridden_wall_obstacle_cells"],
            0,
        )
        self.assertLessEqual(wall_override["max_execution_segment_m"], 0.5)

        snapshot["traversed_paths_xy_m"] = [
            [_cell_xy(40, column, resolution=0.1, origin=(0.0, 0.0))
             for column in range(10, 31, 5)],
            [_cell_xy(40, column, resolution=0.1, origin=(0.0, 0.0))
             for column in range(40, 61, 5)],
        ]
        disconnected = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.42,
            safety_margin_m=0.08,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=2.0,
        )
        self.assertFalse(disconnected["ok"], disconnected)

    def test_large_static_detour_prefers_a_shorter_traversed_corridor(self) -> None:
        occupancy = np.full((120, 120), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        occupancy[20:100, 60] = OCCUPIED
        snapshot = _snapshot(
            occupancy,
            start_cell=(60, 15),
            goal_cell=(60, 105),
        )
        snapshot["wall_mask"] = (occupancy == OCCUPIED).tolist()
        snapshot["traversed_paths_xy_m"] = [
            [
                _cell_xy(row, column, resolution=0.1, origin=(0.0, 0.0))
                for column in range(15, 106, 5)
            ]
            for row in (59, 61)
        ]

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.42,
            safety_margin_m=0.08,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=2.0,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_preference_evaluated"])
        self.assertTrue(result["traversed_route_preference_selected"])
        self.assertEqual(
            result["planning_evidence_trials"][
                result["planning_evidence_trial_index"]
            ],
            "traversed_swept_space_override",
        )
        self.assertGreater(
            result["ordinary_path_length_m"] - result["path_length_m"],
            0.5,
        )
        self.assertGreater(
            result["traversed_route_overridden_wall_obstacle_cells"], 0
        )

    def test_missing_wall_shortcut_prefers_smoothed_traversed_topology(self) -> None:
        occupancy = np.full((140, 140), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED

        def line(start, end):
            count = int(math.ceil(math.dist(start, end) / 0.10)) + 1
            return [
                (
                    start[0] + alpha * (end[0] - start[0]),
                    start[1] + alpha * (end[1] - start[1]),
                )
                for alpha in np.linspace(0.0, 1.0, count)
            ]

        start = (2.05, 7.05)
        goal = (11.05, 7.05)
        taught: list[tuple[float, float]] = []
        for first, second in zip(
            (start, (2.05, 3.05), (11.05, 3.05)),
            ((2.05, 3.05), (11.05, 3.05), goal),
        ):
            segment = line(first, second)
            taught.extend(segment if not taught else segment[1:])
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.10,
            start_xy=list(start),
            goal_xy=list(goal),
        )
        snapshot["traversed_paths_xy_m"] = [taught]

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.05,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=0.0,
            require_turning_pockets=True,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_preference_selected"])
        self.assertTrue(result["traversed_route_guidance_selected"])
        self.assertTrue(result["traversed_route_graph_used"])
        self.assertFalse(result["traversed_route_raster_graph_used"])
        self.assertFalse(result["traversed_route_evidence_used"])
        self.assertEqual(
            result["traversed_route_preference_reason"],
            "ordinary_route_leaves_traversed_corridor",
        )
        self.assertGreater(result["ordinary_route_unsupported_by_history_m"], 5.0)
        self.assertGreaterEqual(result["turning_count"], 2)
        path = _path_xy(result)
        self.assertLess(float(path[:, 1].min()), 3.20)
        self.assertGreater(result["path_length_m"], 15.0)

    def test_nearby_history_branches_do_not_form_missing_wall_shortcut(self) -> None:
        occupancy = np.full((300, 300), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED

        def line(start, end):
            count = int(math.ceil(math.dist(start, end) / 0.04)) + 1
            return [
                (
                    start[0] + alpha * (end[0] - start[0]),
                    start[1] + alpha * (end[1] - start[1]),
                )
                for alpha in np.linspace(0.0, 1.0, count)
            ]

        # The two arms are close enough for 12 cm raster guidance tubes to
        # overlap, but they are separate moments on a U-shaped traversal.  A
        # missing wall pixel must not turn that overlap into a direct edge.
        start = (2.00, 4.00)
        goal = (2.18, 4.00)
        taught: list[tuple[float, float]] = []
        for first, second in zip(
            (start, (2.00, 1.00), (2.18, 1.00)),
            ((2.00, 1.00), (2.18, 1.00), goal),
        ):
            segment = line(first, second)
            taught.extend(segment if not taught else segment[1:])
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.02,
            start_xy=list(start),
            goal_xy=list(goal),
        )
        snapshot["traversed_paths_xy_m"] = [taught]

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.04,
            safety_margin_m=0.0,
            clearance_weight=0.0,
            require_traversed_route=True,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_graph_used"])
        self.assertFalse(result["traversed_route_raster_graph_used"])
        self.assertGreater(result["path_length_m"], 5.5)
        path = _path_xy(result)
        self.assertLess(float(path[:, 1].min()), 1.10)
        self.assertLess(float(path[1, 1]), float(path[0, 1]))

    def test_committed_history_route_keeps_the_same_geometric_branch(self) -> None:
        occupancy = np.full((140, 140), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        start = (2.05, 7.05)
        goal = (11.05, 7.05)

        def route(via_y: float) -> list[tuple[float, float]]:
            points: list[tuple[float, float]] = []
            for first, second in zip(
                (start, (2.05, via_y), (11.05, via_y)),
                ((2.05, via_y), (11.05, via_y), goal),
            ):
                segment = [
                    (
                        first[0] + alpha * (second[0] - first[0]),
                        first[1] + alpha * (second[1] - first[1]),
                    )
                    for alpha in np.linspace(0.0, 1.0, 41)
                ]
                points.extend(segment if not points else segment[1:])
            return points

        upper = route(10.05)
        lower = route(4.05)
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.10,
            start_xy=list(start),
            goal_xy=list(goal),
        )
        snapshot["traversed_paths_xy_m"] = [upper, lower]
        committed = [lower[index] for index in range(0, len(lower), 20)]
        if committed[-1] != lower[-1]:
            committed.append(lower[-1])

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.05,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=0.0,
            require_turning_pockets=True,
            require_traversed_route=True,
            committed_traversed_path_xy_m=committed,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["committed_traversed_route_requested"])
        self.assertTrue(result["committed_traversed_route_selected"])
        self.assertTrue(result["traversed_route_guidance_selected"])
        path = _path_xy(result)
        self.assertLess(float(path[:, 1].min()), 4.20)
        self.assertLess(float(path[:, 1].max()), 7.20)

    def test_history_guidance_uses_image_pick_arrival_disk(self) -> None:
        occupancy = np.full((110, 140), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        start = (2.05, 5.05)
        marked_goal = (10.05, 5.95)
        taught = [
            (x, 5.05) for x in np.linspace(start[0], 10.05, 81)
        ]
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.10,
            start_xy=list(start),
            goal_xy=list(marked_goal),
        )
        snapshot["traversed_paths_xy_m"] = [taught]

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.05,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=0.0,
            arrival_tolerance_m=1.0,
            optimize_goal_standoff=True,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_guidance_selected"])
        self.assertGreaterEqual(result["goal_standoff_m"], 0.79)
        self.assertLessEqual(result["goal_standoff_m"], 0.92 + 1.0e-9)
        self.assertLessEqual(
            result["traversed_route_planned_goal_distance_m"], 0.12
        )

    def test_live_depth_can_veto_history_guidance(self) -> None:
        occupancy = np.full((140, 140), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        start = (2.05, 7.05)
        goal = (11.05, 7.05)
        taught = [
            (2.05, y) for y in np.linspace(7.05, 3.05, 41)
        ]
        taught += [
            (x, 3.05) for x in np.linspace(2.15, 11.05, 90)
        ]
        taught += [
            (11.05, y) for y in np.linspace(3.15, 7.05, 40)
        ]
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.10,
            start_xy=list(start),
            goal_xy=list(goal),
        )
        snapshot["traversed_paths_xy_m"] = [taught]
        live_depth = np.zeros_like(occupancy, dtype=bool)
        live_depth[27:35, 62:70] = True
        snapshot["navigation_obstacle_mask"] = live_depth

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.05,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=0.0,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_preference_evaluated"])
        self.assertFalse(result["traversed_route_preference_selected"])
        self.assertFalse(result["traversed_route_guidance_selected"])
        self.assertEqual(
            result["traversed_route_preference_reason"],
            "history_route_failed_safety_validation",
        )

        committed = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.05,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=0.0,
            require_traversed_route=True,
        )

        self.assertFalse(committed["ok"], committed)
        self.assertTrue(committed["traversed_route_commitment_requested"])
        self.assertEqual(
            committed["traversed_route_preference_reason"],
            "history_route_commitment_temporarily_blocked",
        )
        self.assertIn("footprint-safe", committed["error"])

    def test_curved_traversed_route_survives_raster_centerline_gaps(self) -> None:
        occupancy = np.full((200, 200), UNKNOWN, dtype=np.int16)
        taught_path = [
            (
                5.0 + 2.0 * math.cos(theta),
                5.0 + 2.0 * math.sin(theta),
            )
            for theta in np.linspace(0.0, math.pi, 101)
        ]
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.05,
            start_xy=list(taught_path[0]),
            goal_xy=list(taught_path[-1]),
        )
        snapshot["traversed_paths_xy_m"] = [taught_path]

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.42,
            safety_margin_m=0.08,
            safety_margin_fallbacks_m=(0.06, 0.04, 0.02, 0.0),
            clearance_weight=2.0,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_direct_replay"])
        self.assertEqual(result["planner"], "certified_traversed_polyline")
        self.assertLessEqual(result["max_execution_segment_m"], 1.0)
        path = _path_xy(result)
        self.assertLessEqual(
            float(np.linalg.norm(np.diff(path, axis=0), axis=1).max()),
            result["max_execution_segment_m"] + 1.0e-9,
        )
        self.assertGreaterEqual(
            min(result["path_segment_clearance_m"]),
            result["required_clearance_m"],
        )
        self.assertLessEqual(result["goal_standoff_m"], 0.25)

    def test_history_raster_route_prefers_middle_of_taught_corridor(self) -> None:
        shape = (60, 120)
        support = np.zeros(shape, dtype=bool)
        support[20:40, 10:110] = True
        blocked = np.zeros(shape, dtype=bool)
        clearance = np.full(shape, 5.0, dtype=np.float64)
        start = (1.05, 2.15)
        goal = (10.95, 2.15)

        result = _traversed_raster_route_between(
            support,
            start,
            goal,
            0.01,
            0.01,
            blocked,
            clearance,
            (0.0, 0.0),
            0.10,
            0.25,
            None,
        )

        self.assertIsNotNone(result)
        route, _, _ = result
        middle = route[len(route) // 2]
        self.assertGreater(middle[1], 2.75)
        self.assertLess(middle[1], 3.25)

    def test_traversed_graph_shortcuts_temporal_detours_at_revisits(self) -> None:
        occupancy = np.full((240, 240), UNKNOWN, dtype=np.int16)

        def line(start: tuple[float, float], end: tuple[float, float]):
            distance = math.dist(start, end)
            count = int(math.ceil(distance / 0.1)) + 1
            return [
                (
                    start[0] + alpha * (end[0] - start[0]),
                    start[1] + alpha * (end[1] - start[1]),
                )
                for alpha in np.linspace(0.0, 1.0, count)
            ]

        goal = (1.0, 1.0)
        revisit = (1.0, 5.0)
        current = (9.0, 5.0)
        waypoints = [
            goal,
            revisit,
            (5.0, 5.0),
            (5.0, 9.0),
            (1.0, 9.0),
            revisit,
            current,
        ]
        taught_path: list[tuple[float, float]] = []
        for segment_start, segment_end in zip(waypoints[:-1], waypoints[1:]):
            segment = line(segment_start, segment_end)
            taught_path.extend(segment if not taught_path else segment[1:])
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.05,
            start_xy=list(current),
            goal_xy=list(goal),
        )
        snapshot["traversed_paths_xy_m"] = [taught_path]

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.42,
            safety_margin_m=0.08,
            safety_margin_fallbacks_m=(0.06, 0.04, 0.02, 0.0),
            clearance_weight=2.0,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_graph_used"])
        self.assertLess(result["path_length_m"], 14.0)
        self.assertGreater(math.dist(current, goal), 8.0)

    def test_full_base_uses_shortest_certified_union_of_repeated_tracks(self) -> None:
        occupancy = np.full((120, 120), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        # This stale wall makes the ordinary map route detour.  An old trip
        # crossed it, returned to the lower end, and then approached the same
        # corridor on a pose-corrected parallel track.  Temporal replay would
        # traverse the entire lap; the footprint-supported union should join
        # the two observations locally.
        occupancy[60:112, 16] = OCCUPIED

        def line(start, end):
            count = int(math.ceil(math.dist(start, end) / 0.05)) + 1
            return [
                (
                    start[0] + alpha * (end[0] - start[0]),
                    start[1] + alpha * (end[1] - start[1]),
                )
                for alpha in np.linspace(0.0, 1.0, count)
            ]

        goal = (1.00, 9.00)
        current = (2.18, 8.00)
        waypoints = [
            goal,
            (2.00, 8.10),
            (2.00, 1.00),
            (2.18, 1.00),
            current,
        ]
        taught: list[tuple[float, float]] = []
        for first, second in zip(waypoints[:-1], waypoints[1:]):
            segment = line(first, second)
            taught.extend(segment if not taught else segment[1:])
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.10,
            start_xy=list(current),
            goal_xy=list(goal),
        )
        snapshot["wall_mask"] = (occupancy == OCCUPIED).tolist()
        snapshot["traversed_paths_xy_m"] = [taught]

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.42,
            safety_margin_m=0.08,
            safety_margin_fallbacks_m=(0.06, 0.04, 0.02, 0.0),
            clearance_weight=2.0,
            require_turning_pockets=True,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_preference_selected"])
        self.assertTrue(result["traversed_route_raster_graph_used"])
        self.assertFalse(result["traversed_route_graph_used"])
        self.assertGreater(result["ordinary_path_length_m"], 8.0)
        self.assertLess(result["path_length_m"], 2.0)

    def test_blocked_committed_branch_reselects_only_from_history(self) -> None:
        occupancy = np.full((140, 140), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        start = (2.05, 7.05)
        goal = (11.05, 7.05)

        def route(via_y: float) -> list[tuple[float, float]]:
            points: list[tuple[float, float]] = []
            for first, second in zip(
                (start, (2.05, via_y), (11.05, via_y)),
                ((2.05, via_y), (11.05, via_y), goal),
            ):
                segment = [
                    (
                        first[0] + alpha * (second[0] - first[0]),
                        first[1] + alpha * (second[1] - first[1]),
                    )
                    for alpha in np.linspace(0.0, 1.0, 41)
                ]
                points.extend(segment if not points else segment[1:])
            return points

        upper = route(10.05)
        lower = route(4.05)
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.10,
            start_xy=list(start),
            goal_xy=list(goal),
        )
        snapshot["traversed_paths_xy_m"] = [upper, lower]
        live_depth = np.zeros_like(occupancy, dtype=bool)
        live_depth[37:45, 63:68] = True
        snapshot["navigation_obstacle_mask"] = live_depth
        committed = [lower[index] for index in range(0, len(lower), 20)]
        if committed[-1] != lower[-1]:
            committed.append(lower[-1])

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.05,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=0.0,
            require_turning_pockets=True,
            require_traversed_route=True,
            committed_traversed_path_xy_m=committed,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["committed_traversed_route_requested"])
        self.assertFalse(result["committed_traversed_route_selected"])
        self.assertTrue(
            result["committed_traversed_branch_reselection_evaluated"]
        )
        self.assertTrue(
            result["committed_traversed_branch_reselection_selected"]
        )
        path = _path_xy(result)
        self.assertGreater(float(path[:, 1].max()), 9.80)
        self.assertGreater(float(path[:, 1].min()), 6.90)

    def test_history_raster_normalizes_submillimetre_start_snap(self) -> None:
        occupancy = np.full((140, 140), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        trail_start = (2.05, 7.05)
        start = (trail_start[0] + 0.000065, trail_start[1])
        goal = (11.05, 7.05)
        taught = [
            (2.05, y) for y in np.linspace(7.05, 3.05, 41)
        ]
        taught += [
            (x, 3.05) for x in np.linspace(2.15, 11.05, 90)
        ]
        taught += [
            (11.05, y) for y in np.linspace(3.15, 7.05, 40)
        ]
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.10,
            start_xy=list(start),
            goal_xy=list(goal),
        )
        snapshot["traversed_paths_xy_m"] = [taught]

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.05,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=0.0,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_guidance_selected"])
        self.assertEqual(result["start_snap_m"], 0.0)
        self.assertEqual(result["start_egress_segment_count"], 0)
        self.assertLess(math.dist(result["path_xy_m"][0], start), 1.0e-9)

    def test_history_replay_collapses_short_start_connector(self) -> None:
        occupancy = np.full((140, 140), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED
        trail_start = (2.05, 7.05)
        start = (trail_start[0] - 0.02, trail_start[1] + 0.02)
        goal = (11.05, 7.05)
        taught = [
            (2.05, y) for y in np.linspace(7.05, 3.05, 41)
        ]
        taught += [
            (x, 3.05) for x in np.linspace(2.15, 11.05, 90)
        ]
        taught += [
            (11.05, y) for y in np.linspace(3.15, 7.05, 40)
        ]
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.10,
            start_xy=list(start),
            goal_xy=list(goal),
        )
        snapshot["traversed_paths_xy_m"] = [taught]

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.05,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=0.0,
            require_turning_pockets=True,
            require_traversed_route=True,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_direct_replay"])
        self.assertEqual(result["start_snap_m"], 0.0)
        self.assertEqual(result["start_egress_segment_count"], 0)
        self.assertLess(math.dist(result["path_xy_m"][0], start), 1.0e-9)
        self.assertGreater(
            math.dist(result["path_xy_m"][0], result["path_xy_m"][1]),
            0.10,
        )

    def test_history_raster_removes_chronological_revisit_loops(self) -> None:
        occupancy = np.full((240, 240), FREE, dtype=np.int16)
        occupancy[[0, -1], :] = OCCUPIED
        occupancy[:, [0, -1]] = OCCUPIED

        def line(start, end):
            count = int(math.ceil(math.dist(start, end) / 0.1)) + 1
            return [
                (
                    start[0] + alpha * (end[0] - start[0]),
                    start[1] + alpha * (end[1] - start[1]),
                )
                for alpha in np.linspace(0.0, 1.0, count)
            ]

        goal = (1.0, 1.0)
        revisit = (1.0, 5.0)
        current = (9.0, 5.0)
        waypoints = [
            goal,
            revisit,
            (5.0, 5.0),
            (5.0, 9.0),
            (1.0, 9.0),
            revisit,
            current,
        ]
        taught_path: list[tuple[float, float]] = []
        for segment_start, segment_end in zip(waypoints[:-1], waypoints[1:]):
            segment = line(segment_start, segment_end)
            taught_path.extend(segment if not taught_path else segment[1:])
        snapshot = _snapshot(
            occupancy,
            start_cell=(0, 0),
            goal_cell=(0, 0),
            resolution=0.05,
            start_xy=list(current),
            goal_xy=list(goal),
        )
        snapshot["traversed_paths_xy_m"] = [taught_path]

        result = plan_clearance_path(
            snapshot,
            GOAL_NAME,
            robot_radius_m=0.20,
            safety_margin_m=0.05,
            safety_margin_fallbacks_m=(0.0,),
            clearance_weight=0.0,
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["traversed_route_preference_selected"])
        self.assertTrue(result["traversed_route_graph_used"])
        self.assertFalse(result["traversed_route_raster_graph_used"])
        self.assertLess(result["path_length_m"], 14.0)
        self.assertGreater(result["path_length_m"], math.dist(current, goal))


class ExecutionClearanceMeasurementTest(unittest.TestCase):
    def test_spare_room_is_measured_without_reducing_required_radius(self):
        blocked = np.zeros((60, 80), dtype=bool)
        blocked[:, 20] = True
        lower_bound = np.zeros_like(blocked, dtype=float)
        arguments = ((1.49, 1.0), (1.49, 2.0), blocked, lower_bound,
                     (0.0, 0.0), 0.05, 0.42)
        self.assertEqual(_segment_clearance_certificate(*arguments), (True, 0.42))
        safe, measured = _segment_clearance_certificate(
            *arguments, measure_reserve_m=0.30,
        )
        self.assertTrue(safe)
        self.assertAlmostEqual(measured, 0.44)
        unsafe, distance = _segment_clearance_certificate(
            (1.46, 1.0), (1.46, 2.0), blocked, lower_bound,
            (0.0, 0.0), 0.05, 0.42, measure_reserve_m=0.30,
        )
        self.assertFalse(unsafe)
        self.assertAlmostEqual(distance, 0.41)

    def test_expanded_measurement_never_overstates_exact_clearance(self):
        rng = np.random.default_rng(17)
        blocked = rng.random((24, 24)) < 0.03
        occupancy = np.where(blocked, OCCUPIED, FREE)
        snapshot = {"occupancy": occupancy, "resolution": 0.1, "origin": [0.0, 0.0]}
        accepted = 0
        for _ in range(80):
            path = rng.uniform(0.15, 2.25, (2, 2))
            safe, measured = _segment_clearance_certificate(
                tuple(path[0]), tuple(path[1]), blocked,
                np.zeros_like(blocked, dtype=float), (0.0, 0.0), 0.1,
                0.07, measure_reserve_m=0.30,
            )
            exact = min(float(_segment_clearances_to_occupied_cells(snapshot, path)[0]),
                        float(path.min()), float((2.4 - path).min()))
            self.assertLessEqual(measured, exact + 1.0e-9)
            if safe:
                accepted += 1
                self.assertGreaterEqual(measured + 1.0e-9, 0.07)
        self.assertGreater(accepted, 5)


if __name__ == "__main__":
    unittest.main()
