from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools.util import (
    x5_move_tracked_overlap as move_overlap,
)
from tools.util.x5_grasp_planner import (
    DepthRayOccupancy,
    _volume_validation,
    estimate_depth_edge_frame,
    estimate_support_plane_z,
)


class X5RgbdFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.height = 64
        self.width = 80
        self.depth = np.full((self.height, self.width), 0.800, dtype=np.float64)
        # A raised strip is the public-depth analogue of a garment edge.
        self.depth[:, 40:] = 0.900
        self.intrinsic = np.array(
            [[300.0, 0.0, 39.5], [0.0, 300.0, 31.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        # Match the evaluator optical/USD axis convention used by the adapter.
        self.extrinsic = np.eye(4, dtype=np.float64)
        self.extrinsic[:3, :3] = np.diag([1.0, -1.0, -1.0])

    def test_support_plane_comes_from_dominant_world_z_mode(self) -> None:
        support_z, audit = estimate_support_plane_z(
            self.depth, self.intrinsic, self.extrinsic
        )
        self.assertAlmostEqual(float(support_z), 0.800, places=6)
        self.assertEqual(audit["reason"], "dominant_world_z_mode")
        self.assertFalse(audit["uses_segmentation"])
        self.assertFalse(audit["uses_simulator_geometry"])

    def test_edge_frame_points_from_support_toward_raised_strip(self) -> None:
        approach, audit = estimate_depth_edge_frame(
            self.depth,
            self.intrinsic,
            self.extrinsic,
            40,
            32,
            support_z=0.800,
        )
        self.assertIsNotNone(approach)
        self.assertTrue(audit["usable"])
        self.assertGreaterEqual(float(audit["best"]["contrast_m"]), 0.099)
        # The strip is to the +u side, which maps to +world x here.
        self.assertGreater(float(approach[0]), 0.95)
        self.assertAlmostEqual(float(approach[2]), 0.0, places=9)

    def test_surface_only_opening_excludes_support_but_solid_keeps_it(self) -> None:
        occupancy = DepthRayOccupancy(
            self.depth,
            self.intrinsic,
            self.extrinsic,
            support_z=0.800,
            target_anchor=np.array([0.0, 0.0, 0.900]),
            target_radius_m=0.110,
        )
        # Both points project to valid depth pixels; only the second is above
        # the support plane.  A rotation identity means EEF == world here.
        points = np.array(
            [[-0.012, 0.0, 0.800], [0.015, 0.0, 0.900]], dtype=np.float64
        )
        rotation = np.eye(3, dtype=np.float64)
        position = np.zeros(3, dtype=np.float64)
        solid = occupancy.occupied_mask(points, rotation, position)
        surface = occupancy.occupied_mask(
            points,
            rotation,
            position,
            surface_only=True,
            target_anchor=np.array([0.0, 0.0, 0.900]),
            target_radius_m=0.110,
        )
        np.testing.assert_array_equal(solid, [True, True])
        np.testing.assert_array_equal(surface, [False, True])

    @staticmethod
    def _grasp_metrics(
        *,
        grasp_vox: int,
        overlap_vox: int = 0,
        inflated_overlap_vox: int = 0,
    ) -> dict:
        voxel_cm3 = 0.027
        return {
            "grasp_vox": grasp_vox,
            "overlap_vox": overlap_vox,
            "inflated_overlap_vox": inflated_overlap_vox,
            "grasp_vol_cm3": grasp_vox * voxel_cm3,
            "overlap_vol_cm3": overlap_vox * voxel_cm3,
            "inflated_overlap_vol_cm3": inflated_overlap_vox * voxel_cm3,
            "overlap_frac": 0.0,
            "opening_axial_min_m": 0.10 if grasp_vox else None,
            "opening_axial_max_m": 0.12 if grasp_vox else None,
            "opening_axial_span_m": 0.02 if grasp_vox else 0.0,
            "opening_tip_fraction": 0.0,
            "tip_only": not bool(grasp_vox),
        }

    def test_final_q_fk_volume_is_a_required_second_gate(self) -> None:
        ideal = self._grasp_metrics(grasp_vox=160, overlap_vox=5)
        displaced_fk = self._grasp_metrics(grasp_vox=0, overlap_vox=5)
        validation = _volume_validation(
            ideal,
            displaced_fk,
            edge_usable=True,
        )
        self.assertFalse(validation["ok"])
        self.assertTrue(validation["ideal_ok"])
        self.assertFalse(validation["final_q_fk_ok"])
        self.assertEqual(
            validation["final_q_fk_reason"], "opening_geometry"
        )
        self.assertEqual(validation["robust"]["grasp_vox"], 0)
        self.assertEqual(validation["robust"]["grasp_vol_cm3"], 0.0)

    def test_candidate_ranking_volume_is_minimum_of_ideal_and_fk(self) -> None:
        ideal = self._grasp_metrics(grasp_vox=160, overlap_vox=4)
        realized = self._grasp_metrics(grasp_vox=120, overlap_vox=5)
        validation = _volume_validation(ideal, realized, edge_usable=True)
        self.assertTrue(validation["ok"])
        self.assertEqual(validation["robust"]["grasp_vox"], 120)
        self.assertAlmostEqual(
            validation["robust"]["grasp_vol_cm3"], 3.24
        )
        self.assertEqual(validation["robust"]["overlap_vox"], 5)

    def test_move_overlap_reuses_one_frozen_scene_across_frontiers(self) -> None:
        class EmptyOccupancy:
            @staticmethod
            def occupied_mask(points, _rotation, _position):
                return np.zeros(len(points), dtype=bool)

        snapshot = SimpleNamespace(
            depth=np.ones((4, 5), dtype=np.float32),
            intrinsic_matrix=np.eye(3, dtype=np.float64),
            extrinsic_matrix=np.eye(4, dtype=np.float64),
        )
        mesh_result = SimpleNamespace(metadata={"source": "frozen_rgbd"})
        geometry = {
            "gripper_voxels": np.zeros((2, 3), dtype=np.float64),
            "inflated_gripper_voxels": np.zeros((3, 3), dtype=np.float64),
        }
        context = {"base_rotation": np.eye(3, dtype=np.float64).tolist()}
        poses = [{"eef_pos": [0.0, 0.0, 0.8], "quat": [0.0, 0.0, 0.0, 1.0]}]

        with (
            mock.patch.object(move_overlap, "_load_snapshot", return_value=snapshot) as load,
            mock.patch.object(
                move_overlap,
                "estimate_support_plane_z",
                return_value=(0.75, {"source": "frozen_rgbd"}),
            ) as support,
            mock.patch.object(
                move_overlap,
                "_build_scene_occupancy",
                return_value=(EmptyOccupancy(), mesh_result, "robodojo_v1", None),
            ) as build,
            mock.patch.object(
                move_overlap, "x5_gripper_geometry", return_value=geometry
            ),
        ):
            first = move_overlap.evaluate_x5_move_endpoint_overlap(context, poses)
            second = move_overlap.evaluate_x5_move_endpoint_overlap(context, poses)

        self.assertFalse(first["scene_reconstruction_reused"])
        self.assertTrue(second["scene_reconstruction_reused"])
        self.assertEqual(first["poses"], second["poses"])
        load.assert_called_once_with(context)
        support.assert_called_once()
        build.assert_called_once()


if __name__ == "__main__":
    unittest.main()
