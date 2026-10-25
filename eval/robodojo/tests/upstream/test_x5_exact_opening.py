from __future__ import annotations

import unittest
import numpy as np

from tools.util.x5_exact_opening import (
    exact_opening_query, exact_x5_opening_voxels, exact_x5_planning_geometry,
    points_inside_exact_x5_opening,
)
from tools.util.x5_grasp_geometry import (
    x5_gripper_geometry, x5_dynamic_opening_mesh, points_inside_x5_grasp_volume,
    points_inside_legacy_x5_grasp_volume, x5_gripper_joint_position_m,
)


class ExactOpeningTest(unittest.TestCase):
    def test_tip_is_not_eroded_by_legacy_six_mm_margin(self):
        points = np.array([[x, 0., 0.] for x in (.150, .152, .155, .157)])
        np.testing.assert_array_equal(points_inside_legacy_x5_grasp_volume(points, [1.]), [True, False, False, False])
        np.testing.assert_array_equal(points_inside_x5_grasp_volume(points, [1.]), [True] * 4)

    def test_nonfinite_and_outside_projection_are_rejected(self):
        points = [[np.nan, 0., 0.], [np.inf, 0., 0.], [.16, 0., 0.], [.155, 0., .03], [.1, .2, 0.]]
        self.assertFalse(points_inside_exact_x5_opening(points, [1.]).any())
        self.assertEqual(points_inside_exact_x5_opening([], [1.]).shape, (0,))

    def test_classification_uses_each_inner_triangle_plane(self):
        g = x5_gripper_geometry()
        vertices, faces, sides = [g[key] for key in ("exact_opening_mesh_vertices", "exact_opening_mesh_faces", "exact_opening_mesh_vertex_side")]
        lower = faces[np.all(sides[faces] < 0, axis=1)]
        centers = vertices[lower].mean(axis=1)
        q = exact_opening_query()
        low, high = q.y_bounds(centers, [1.])
        np.testing.assert_allclose(low, centers[:, 1], atol=2e-10)
        middle = centers.copy()
        middle[:, 1] = .5 * (low + high)
        self.assertTrue(q.contains(middle, [1.]).all())
        for boundary, sign in ((low, -1), (high, 1)):
            outside = centers.copy()
            outside[:, 1] = boundary + sign * 1e-6
            self.assertFalse(q.contains(outside, [1.]).any())

    def test_bucket_lookup_matches_independent_all_triangle_query(self):
        rng = np.random.default_rng(891)
        g = x5_gripper_geometry()
        v, f, side = [g[key] for key in ("exact_opening_mesh_vertices", "exact_opening_mesh_faces", "exact_opening_mesh_vertex_side")]
        q = exact_opening_query()
        points = rng.uniform(q.bounds[0] - .002, q.bounds[1] + .002, (600, 3))
        # Brute-force intersections with every lower and upper triangle; no
        # bucket, voxel lookup, or production barycentric helper is reused.
        for command in (1., .5, 0.):
            closure = .044 - x5_gripper_joint_position_m([command])
            bounds = {s: np.full(len(points), np.nan) for s in (-1, 1)}
            for s in (-1, 1):
                for face in f[np.all(side[f] == s, axis=1)]:
                    tri = v[face]
                    matrix = np.vstack((tri[:, [0, 2]].T, np.ones(3)))
                    if abs(np.linalg.det(matrix)) < 1e-18:
                        continue
                    weights = np.column_stack((points[:, [0, 2]], np.ones(len(points)))) @ np.linalg.inv(matrix).T
                    hit = np.all(weights >= -1e-9, axis=1)
                    bounds[s][hit] = weights[hit] @ tri[:, 1] - s * closure
            expected = (points[:, 1] > bounds[-1]) & (points[:, 1] < bounds[1])
            np.testing.assert_array_equal(q.contains(points, [command]), expected)

    def test_no_staircase_at_voxel_bucket_edge(self):
        points = np.array([[.15299999, 0., .001], [.15300001, 0., .001]])
        low, high = exact_opening_query().y_bounds(points, [1.])
        self.assertTrue(np.isfinite(low).all())
        self.assertLess(abs(low[0] - low[1]), 1e-6)
        self.assertLess(abs(high[0] - high[1]), 1e-6)

    def test_closure_uses_prismatic_triangle_translation_not_empty_cube(self):
        points = np.array([[.11, 0., 0.], [.11, .03, 0.]])
        full = exact_opening_query().y_bounds(points, [1.])
        closed = exact_opening_query().y_bounds(points, [0.])
        np.testing.assert_allclose(closed[0] - full[0], .044, atol=1e-12)
        np.testing.assert_allclose(closed[1] - full[1], -.044, atol=1e-12)
        np.testing.assert_array_equal(points_inside_x5_grasp_volume(points, [0.]), [True, False])

    def test_voxel_centers_match_continuous_query_and_capacity(self):
        counts = []
        for command in (0., .5, 1.):
            voxels = exact_x5_opening_voxels([command])
            self.assertTrue(points_inside_exact_x5_opening(voxels, [command]).all())
            self.assertFalse(voxels.flags.writeable)
            counts.append(len(voxels))
        self.assertTrue(counts[0] < counts[1] < counts[2])
        # Coarse quadrature is named an approximation, not the exact volume.
        mesh_volume = x5_dynamic_opening_mesh([1.])["volume_cm3"]
        self.assertLess(abs(counts[2] * .027 - mesh_volume) / mesh_volume, .025)

    def test_planning_geometry_does_not_mutate_rollback_asset(self):
        old = x5_gripper_geometry()
        before = old["opening_voxels"].copy()
        new = exact_x5_planning_geometry()
        np.testing.assert_array_equal(old["opening_voxels"], before)
        self.assertGreater(new["opening_voxels"][:, 0].max(), old["opening_voxels"][:, 0].max())
        self.assertEqual(len(new["axial_x_m"]), 7)
        self.assertNotIn("opening_query_version", old["metadata"])


if __name__ == "__main__":
    unittest.main()
