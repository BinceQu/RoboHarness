"""Round metric route buffers and original-map overlap costs, without a scene."""

from __future__ import annotations

import math
import unittest

import numpy as np

from behavior_interface_eval_test import navigation_route_overlay as overlay
from behavior_interface_eval_test.tool.official_v2 import map_navigation_local as nav


class RouteSweepGeometryTest(unittest.TestCase):
    shape = (240, 240)
    origin = (-3.0, -3.0)
    resolution = 0.025

    def sweep(self, path):
        return nav.polyline_swept_mask(
            path, self.shape, self.origin, self.resolution
        )

    def cell(self, point):
        return nav._cell(point, self.origin, self.resolution)

    def test_caps_and_outer_corner_are_round_not_square(self):
        mask = self.sweep([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)])
        for point in [(-0.30, -0.20), (2.30, -0.20), (2.20, 2.30)]:
            with self.subTest(point=point):
                self.assertTrue(mask[self.cell(point)])
        for point in [(-0.35, -0.35), (2.35, -0.35), (2.35, 2.35)]:
            with self.subTest(point=point):
                self.assertFalse(mask[self.cell(point)])

    def test_continuous_segment_hits_cell_even_when_cell_center_is_outside(self):
        mask = nav.polyline_swept_mask(
            [(0.0, 0.0), (2.0, 0.0)], (100, 100), (-1.0, -1.0), 0.05
        )
        # This square touches y=0.4, but its center at y=0.425 is outside.
        self.assertTrue(mask[28, 40])
        self.assertFalse(mask[29, 40])

    def test_union_does_not_count_shared_corners_or_retracing_twice(self):
        first = self.sweep([(0.0, 0.0), (2.0, 0.0)])
        second = self.sweep([(2.0, 0.0), (2.0, 2.0)])
        whole = self.sweep([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)])
        np.testing.assert_array_equal(whole, first | second)
        np.testing.assert_array_equal(
            whole,
            self.sweep([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0),
                        (2.0, 0.0), (0.0, 0.0)]),
        )

    def test_rotation_and_translation_do_not_depend_on_a_room_direction(self):
        path = [(-1.5, -1.0), (0.5, 0.5), (1.5, -1.0)]
        mask = self.sweep(path)
        rotated = self.sweep([(-y, x) for x, y in path])
        np.testing.assert_array_equal(rotated, np.rot90(mask, -1))
        moved = self.sweep([(x + 0.5, y + 0.25) for x, y in path])
        np.testing.assert_array_equal(moved, np.roll(mask, (10, 20), (0, 1)))

    def test_audit_separates_original_obstacles_unknown_and_fresh_depth(self):
        occupancy = np.zeros(self.shape, dtype=np.int8)
        occupancy[self.cell((0.5, 0.3))] = 100
        occupancy[self.cell((1.0, -0.3))] = -1
        protected = np.zeros(self.shape, dtype=bool)
        protected[self.cell((1.5, 0.2))] = True
        execution_stall = np.zeros(self.shape, dtype=bool)
        execution_stall[self.cell((1.75, -0.2))] = True
        snapshot = {
            "occupancy": occupancy,
            "resolution_m": self.resolution,
            "origin_xy_m": self.origin,
            "navigation_obstacle_mask": protected,
            "navigation_stall_obstacle_mask": execution_stall,
        }
        before = occupancy.copy()
        audit = nav.route_sweep_audit(snapshot, [(0, 0), (2, 0)])
        self.assertEqual(audit["radius_m"], 0.4)
        self.assertEqual(audit["occupied_overlap_cells"], 1)
        self.assertEqual(audit["unknown_overlap_cells"], 1)
        self.assertEqual(audit["live_depth_overlap_cells"], 1)
        self.assertEqual(audit["execution_stall_overlap_cells"], 1)
        self.assertAlmostEqual(
            audit["map_overlap_area_upper_bound_m2"], 2 * self.resolution ** 2
        )
        np.testing.assert_array_equal(occupancy, before)

    def test_invalid_geometry_is_rejected(self):
        for radius in [-0.1, math.nan, math.inf, 3.0]:
            with self.subTest(radius=radius), self.assertRaises(nav.NavigationPlanError):
                nav.polyline_swept_mask([(0, 0)], (4, 4), (0, 0), 0.05, radius)


class RouteSweepCostTest(unittest.TestCase):
    def test_fast_disk_cost_matches_full_convolution(self):
        from scipy import ndimage

        blocked = np.random.default_rng(3).random((40, 50)) < 0.15
        for resolution in [0.025, 0.05, 0.1]:
            with self.subTest(resolution=resolution):
                extent = int(math.ceil(0.4 / resolution + 0.5))
                axis = np.maximum(np.abs(np.arange(-extent, extent + 1)) - 0.5, 0)
                kernel = np.hypot(axis[:, None], axis[None, :]) * resolution <= 0.4 + 1e-9
                expected = ndimage.convolve(
                    blocked.astype(np.float64), kernel.astype(float),
                    mode="constant", cval=1.0,
                ) * nav.SWEEP_OVERLAP_WEIGHT / kernel.sum()
                np.testing.assert_allclose(
                    nav._sweep_overlap_cost(blocked, resolution), expected,
                    rtol=1e-6, atol=1e-6,
                )

    def test_search_and_simplification_avoid_overridden_grey_black_cells(self):
        shape = (100, 100)
        resolution = 0.05
        original_blocked = np.zeros(shape, dtype=bool)
        original_blocked[45:56, 45:56] = True
        overlap_cost = nav._sweep_overlap_cost(original_blocked, resolution)
        traversable = np.ones(shape, dtype=bool)
        clearance = np.full(shape, 2.0)
        args = (traversable, clearance, (50, 15), (50, 85), resolution, 0.42, 0.0)
        straight, _ = nav._astar(*args)
        detour, _ = nav._astar(*args, overlap_cost=overlap_cost)
        self.assertGreater(nav._path_overlap_cost(straight, overlap_cost, resolution), 1.0)
        self.assertAlmostEqual(nav._path_overlap_cost(detour, overlap_cost, resolution), 0.0)
        simplified = nav._simplify(
            detour, traversable, ~traversable, clearance, resolution, (0, 0),
            0.42, 0.0, [0.42] * (len(detour) - 1), overlap_cost,
        )
        points = [nav._xy(cell, (0, 0), resolution) for cell in simplified]
        sweep = nav.polyline_swept_mask(points, shape, (0, 0), resolution)
        self.assertEqual(np.count_nonzero(sweep & original_blocked), 0)
        self.assertLess(len(simplified), len(detour))

    def test_successful_plan_reports_zero_overlap_without_changing_robot_radius(self):
        occupancy = np.zeros((80, 120), dtype=np.int8)
        occupancy[0, :] = occupancy[-1, :] = 100
        occupancy[:, 0] = occupancy[:, -1] = 100
        snapshot = {
            "occupancy": occupancy, "resolution_m": 0.05, "origin_xy_m": (0, 0),
            "pose": {"x_m": 1, "y_m": 2, "yaw_deg": 0},
            "places": [{"name": "goal", "x_m": 5, "y_m": 2}],
        }
        result = nav.plan_clearance_path(snapshot, "goal")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["robot_radius_m"], 0.42)
        self.assertEqual(result["route_sweep"]["radius_m"], 0.4)
        self.assertEqual(result["route_sweep"]["map_overlap_fraction"], 0.0)
        self.assertEqual(result["route_sweep"]["live_depth_overlap_cells"], 0)

    def test_invocation_stall_obstacle_forces_a_new_collision_free_route(self):
        occupancy = np.zeros((80, 120), dtype=np.int8)
        occupancy[0, :] = occupancy[-1, :] = 100
        occupancy[:, 0] = occupancy[:, -1] = 100
        stall = np.zeros_like(occupancy, dtype=bool)
        stall[40, 50] = True
        snapshot = {
            "occupancy": occupancy,
            "resolution_m": 0.05,
            "origin_xy_m": (0, 0),
            "pose": {"x_m": 1.0, "y_m": 2.025, "yaw_deg": 0},
            "places": [{"name": "goal", "x_m": 5.0, "y_m": 2.025}],
            "navigation_stall_obstacle_mask": stall,
            # Old traversals may relax stale SLAM cells, but never this layer.
            "traversed_paths_xy_m": [[(1.0, 2.025), (5.0, 2.025)]],
        }
        before = occupancy.copy()
        result = nav.plan_clearance_path(snapshot, "goal")
        self.assertTrue(result["ok"], result)
        self.assertTrue(
            any(abs(point[1] - 2.025) > 0.2 for point in result["path_xy_m"]),
            result["path_xy_m"],
        )
        self.assertEqual(
            result["route_sweep"]["execution_stall_overlap_cells"], 0
        )
        np.testing.assert_array_equal(occupancy, before)


class RouteSweepRenderTest(unittest.TestCase):
    def test_metric_width_and_round_corner_match_planner(self):
        self.assertEqual(overlay.ROUTE_SWEEP_RADIUS_M, nav.DEFAULT_ROUTE_SWEEP_RADIUS_M)
        source = np.full((400, 400, 4), 255, dtype=np.uint8)
        path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        image = overlay._composite_route_rgba(
            source, path, lambda x, y: (100 + 100*x, 200 - 100*y), width=3
        )
        self.assertFalse(np.array_equal(image[228, 228], source[228, 228]))
        np.testing.assert_array_equal(image[232, 232], source[232, 232])
        # Side distance 0.39m is inside, 0.42m is outside, at a 100px/m view.
        self.assertFalse(np.array_equal(image[239, 150], source[239, 150]))
        np.testing.assert_array_equal(image[242, 150], source[242, 150])
        # Union drawing has the same alpha at the shared corner as at a side.
        np.testing.assert_array_equal(image[228, 228], image[239, 150])


if __name__ == "__main__":
    unittest.main()
