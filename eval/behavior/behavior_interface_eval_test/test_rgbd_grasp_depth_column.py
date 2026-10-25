from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from behavior_interface_eval_test.tool.official_v2 import rgbd_grasp_planner


class RGBDGraspDepthColumnTest(unittest.TestCase):
    @staticmethod
    def _hit(depth: np.ndarray, *, u: int, v: int, fl: float, ha: float):
        return rgbd_grasp_planner.depth_hit_from_pixel(
            depth,
            u=u,
            v=v,
            camera_pos=[1.0, 2.0, 3.0],
            camera_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            focal_length=fl,
            horizontal_aperture=ha,
        )

    def test_front_point_inside_column_replaces_center_pixel(self) -> None:
        depth = np.full((6, 6), np.nan, dtype=np.float64)
        depth[3, 3] = 1.0
        depth[3, 4] = 0.8

        hit, audit = self._hit(depth, u=3, v=3, fl=100.0, ha=1.0)

        np.testing.assert_allclose(hit, [1.0 + 0.8 / 600.0, 2.0, 2.2])
        self.assertEqual(
            audit["method"], "depth_column_first_observed_surface"
        )
        self.assertEqual(
            audit["selection"],
            "minimum_positive_axial_distance_in_click_ray_column",
        )
        self.assertEqual(audit["source"], "raw_observed_depth")
        self.assertEqual(audit["depth_pixel"], [4, 3])
        self.assertEqual(audit["candidate_count"], 2)
        self.assertAlmostEqual(audit["column_radius_mm"], 3.0)
        self.assertAlmostEqual(audit["column_diameter_mm"], 6.0)
        self.assertGreater(audit["selected_lead_vs_center_mm"], 199.0)

    def test_first_hit_beats_a_later_point_closer_to_column_axis(self) -> None:
        depth = np.full((8, 8), np.nan, dtype=np.float64)
        depth[4, 6] = 0.800
        depth[4, 5] = 0.801

        _, audit = self._hit(depth, u=4, v=4, fl=100.0, ha=1.0)

        self.assertEqual(audit["depth_pixel"], [6, 4])
        self.assertEqual(audit["candidate_count"], 2)
        self.assertGreater(audit["selected_axis_offset_mm"], 1.9)

    def test_invalid_center_pixel_can_hit_neighbor_inside_column(self) -> None:
        depth = np.full((6, 6), np.nan, dtype=np.float64)
        depth[3, 4] = 0.8

        hit, audit = self._hit(depth, u=3, v=3, fl=100.0, ha=1.0)

        np.testing.assert_allclose(hit, [1.0 + 0.8 / 600.0, 2.0, 2.2])
        self.assertEqual(audit["depth_pixel"], [4, 3])
        self.assertIsNone(audit["center_ray_axial_distance_m"])
        self.assertIsNone(audit["selected_lead_vs_center_mm"])

    def test_front_point_outside_three_mm_column_is_ignored(self) -> None:
        depth = np.full((6, 6), np.nan, dtype=np.float64)
        depth[3, 3] = 1.0
        depth[3, 5] = 0.2

        _, audit = self._hit(depth, u=3, v=3, fl=10.0, ha=1.0)

        self.assertEqual(audit["depth_pixel"], [3, 3])
        self.assertEqual(audit["candidate_count"], 1)
        self.assertAlmostEqual(audit["selected_axis_offset_mm"], 0.0)

    def test_empty_column_raises(self) -> None:
        depth = np.full((6, 6), np.nan, dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "3.0mm.*no valid observed depth"):
            self._hit(depth, u=3, v=3, fl=100.0, ha=1.0)

    def test_surface_normal_uses_selected_depth_pixel(self) -> None:
        depth = np.full((6, 6), np.nan, dtype=np.float64)
        depth[3, 3] = 1.0
        depth[3, 4] = 0.8
        normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

        with mock.patch.object(
            rgbd_grasp_planner,
            "estimate_grasp_alignment_normal",
            return_value=(normal, {"point_world": None}),
        ) as estimate:
            target = rgbd_grasp_planner.prepare_rgbd_grasp_target(
                depth,
                u=3,
                v=3,
                cam_pos=np.asarray([1.0, 2.0, 3.0]),
                cam_quat=np.asarray([0.0, 0.0, 0.0, 1.0]),
                w=6,
                h=6,
                fl=100.0,
                ha=1.0,
            )

        self.assertEqual(target["hit_audit"]["depth_pixel"], [4, 3])
        self.assertEqual(estimate.call_args.kwargs["u"], 4)
        self.assertEqual(estimate.call_args.kwargs["v"], 3)


if __name__ == "__main__":
    unittest.main()
