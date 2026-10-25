"""Rotation clearance must certify the swept base disk exactly once."""

import unittest

import numpy as np
from scipy import ndimage

from behavior_interface_eval_test.tool.official_v2.map_navigation_local import (
    _repair_turning_pockets,
    _segment_clearance_certificate,
    _turning_pocket_certificate,
)


class NavigationTurningPocketTest(unittest.TestCase):
    def _certificate(self, blocked, points, required=0.50):
        clearance = ndimage.distance_transform_edt(~blocked) * 0.05
        return _turning_pocket_certificate(
            points, blocked, clearance, (0.0, 0.0), 0.05, required
        )

    def test_rotation_disk_is_not_inflated_at_offset_centers(self):
        blocked = np.zeros((80, 80), dtype=bool)
        blocked[:, 0:24] = True  # wall x <= 1.20, corner at x=2.0
        blocked[:, 56:] = True  # wall x >= 2.80
        result = self._certificate(blocked, [(2.0, 1.0), (2.0, 2.0), (2.6, 2.0)])
        self.assertTrue(result[0], result)
        self.assertEqual(result[2], 1)
        self.assertGreaterEqual(result[1][0], 0.62)

    def test_reserve_still_rejects_a_genuinely_tight_turn(self):
        blocked = np.zeros((80, 80), dtype=bool)
        blocked[:, :29] = True  # only 0.55m to the wall
        result = self._certificate(blocked, [(2.0, 1.0), (2.0, 2.0), (3.0, 2.0)])
        self.assertFalse(result[0], result)
        self.assertAlmostEqual(result[1][0], 0.55)

    def test_checks_cell_edges_at_subcell_corner_position(self):
        blocked = np.zeros((80, 80), dtype=bool)
        blocked[:, :28] = True
        result = self._certificate(blocked, [(2.01, 1.0), (2.01, 2.0), (3.0, 2.0)])
        self.assertFalse(result[0], result)
        self.assertAlmostEqual(result[1][0], 0.61)

    def test_all_directions_and_grid_boundary_remain_blocked(self):
        points = [(2.0, 1.0), (2.0, 2.0), (3.0, 2.0)]
        for cell in ((47, 47), (47, 32), (32, 47), (32, 32)):
            with self.subTest(cell=cell):
                blocked = np.zeros((80, 80), dtype=bool)
                blocked[cell] = True
                self.assertFalse(self._certificate(blocked, points)[0])
        self.assertFalse(self._certificate(
            np.zeros((80, 80), dtype=bool),
            [(0.4, 1.0), (0.4, 2.0), (1.4, 2.0)],
        )[0])

    def test_relocates_a_turn_and_certifies_connectors_in_rotated_maps(self):
        original = np.zeros((80, 80), dtype=bool)
        original[:, :29] = True
        points = [(2.0, 1.0), (2.0, 2.0), (3.0, 2.0)]
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                blocked = np.rot90(original, rotation)
                route = list(points)
                for _ in range(rotation):
                    route = [(y, 4.0 - x) for x, y in route]
                clearance = ndimage.distance_transform_edt(~blocked) * 0.05
                result, count = _repair_turning_pockets(
                    route, blocked, clearance, (0.0, 0.0), 0.05, 0.50
                )
                self.assertEqual(count, 1, result)
                self.assertEqual(result[0], route[0])
                self.assertEqual(result[-1], route[-1])
                self.assertTrue(self._certificate(blocked, result)[0], result)
                for start, end in zip(result, result[1:]):
                    self.assertTrue(_segment_clearance_certificate(
                        start, end, blocked, clearance, (0.0, 0.0), 0.05, 0.5,
                    )[0], result)

    def test_no_room_does_not_relax_turn_reserve_or_move_protected_start(self):
        blocked = np.zeros((80, 80), dtype=bool)
        blocked[:, :29] = True
        route = [(2.0, 1.0), (2.0, 2.0), (3.0, 2.0)]
        clearance = ndimage.distance_transform_edt(~blocked) * 0.05
        result, count = _repair_turning_pockets(
            route, blocked, clearance, (0.0, 0.0), 0.05, 0.50,
            protected_points=[route[1]],
        )
        self.assertEqual(result, route)
        self.assertEqual(count, 0)
        blocked[:, 51:] = True
        clearance = ndimage.distance_transform_edt(~blocked) * 0.05
        result, count = _repair_turning_pockets(
            route, blocked, clearance, (0.0, 0.0), 0.05, 0.50,
        )
        self.assertEqual(result, route)
        self.assertEqual(count, 0)
        self.assertFalse(self._certificate(blocked, result)[0])


if __name__ == "__main__":
    unittest.main()
