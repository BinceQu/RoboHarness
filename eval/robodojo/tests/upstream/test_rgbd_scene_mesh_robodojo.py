from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from tools.util import rgbd_scene_mesh_robodojo_v1 as mesh


def _camera_fixture(height: int = 24, width: int = 32):
    rows, cols = np.mgrid[:height, :width]
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (cols * 7) % 255
    rgb[..., 1] = (rows * 11) % 255
    rgb[..., 2] = 96
    depth = 0.80 + 0.0005 * rows + 0.0002 * cols
    intrinsic = np.array(
        [[240.0, 0.0, (width - 1) / 2.0],
         [0.0, 240.0, (height - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    extrinsic = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    return rgb, depth.astype(np.float64), intrinsic, extrinsic


class RoboDojoRgbdSceneMeshTest(unittest.TestCase):
    def test_grasp_can_disable_neighbor_dilation_without_weakening_collision(self):
        occupied = mesh.VoxelOccupancy(voxel_size_m=.003, origin_world=np.zeros(3),
                                      occupied_indices=np.array([[0, 0, 0]]),
                                      source="fixture", query_radius_cells=1)
        result = SimpleNamespace(surface_occupancy=occupied, collision_occupancy=occupied)
        exact = mesh.MeshSceneOccupancy(result, surface_query_radius_cells=0)
        legacy = mesh.MeshSceneOccupancy(result)
        points = np.array([[.0045, .0015, .0015]])
        self.assertFalse(exact.occupied_mask(points, np.eye(3), np.zeros(3), surface_only=True)[0])
        self.assertTrue(legacy.occupied_mask(points, np.eye(3), np.zeros(3), surface_only=True)[0])
        self.assertTrue(exact.occupied_mask(points, np.eye(3), np.zeros(3))[0])

    def test_analytic_support_removes_source_band_not_visible_geometry(self) -> None:
        rgb, depth, intrinsic, extrinsic = _camera_fixture(40, 48)
        depth[:] = .75
        depth[16:24, 20:28] = .80
        result = mesh.reconstruct_rgbd_scene_mesh(
            rgb, depth, intrinsic_matrix=intrinsic, extrinsic_matrix=extrinsic,
            analytic_support_plane=True,
        )
        self.assertTrue(result.metadata["support_separation"]["applied"])
        self.assertEqual(len(result.observed_points_world), depth.size)
        query = np.array([[-.05, 0., .752], [0., 0., .80]])
        np.testing.assert_array_equal(result.surface_occupancy.contains_world(query), [False, True])
        np.testing.assert_array_equal(result.collision_occupancy.contains_world(query), [False, True])
        self.assertEqual(result.metadata["forbidden_inputs_used"], [])

    def test_roundtrip_and_dual_occupancy_contract(self) -> None:
        rgb, depth, intrinsic, extrinsic = _camera_fixture()
        result = mesh.reconstruct_rgbd_scene_mesh(
            rgb,
            depth,
            intrinsic_matrix=intrinsic,
            extrinsic_matrix=extrinsic,
            collision_extrusion_mm=9.0,
        )

        self.assertEqual(result.metadata["mesh_version"], "robodojo_v1")
        self.assertEqual(result.metadata["requested_version"], "robodojo_v1")
        self.assertEqual(result.metadata["effective_version"], "robodojo_v1")
        self.assertEqual(result.metadata["forbidden_inputs_used"], [])
        self.assertFalse(result.metadata["used_click"])
        self.assertFalse(result.metadata["used_segmentation"])
        self.assertGreater(result.metadata["valid_depth_fraction"], 0.99)
        self.assertGreater(result.metadata["surface_mesh"]["faces_count"], 0)
        residual = result.metadata["projection_residual_stats"]
        self.assertLess(float(residual["reprojection_error_p95_px"]), 1e-8)
        self.assertEqual(
            result.metadata["uncertainty"]["thickness_status"],
            "unknown_single_view",
        )
        self.assertNotEqual(
            result.surface_occupancy.source,
            result.collision_occupancy.source,
        )
        self.assertGreaterEqual(
            result.collision_occupancy.volume_cm3,
            result.surface_occupancy.volume_cm3,
        )
        self.assertEqual(
            result.metadata["surface_occupancy"]["voxel_size_m"],
            result.metadata["collision_occupancy"]["voxel_size_m"],
        )

    def test_flat_support_plane_is_retained_and_audited(self) -> None:
        rgb, depth, intrinsic, extrinsic = _camera_fixture(20, 28)
        depth[...] = 0.75
        result = mesh.reconstruct_rgbd_scene_mesh(
            rgb,
            depth,
            intrinsic_matrix=intrinsic,
            extrinsic_matrix=extrinsic,
            collision_extrusion_mm=0.0,
        )
        support = result.metadata["support_plane"]
        self.assertIsNotNone(result.metadata["support_plane_z_m"])
        self.assertIn("normal_world", support)
        self.assertAlmostEqual(float(support["z_m"]), 0.75, places=3)
        self.assertLess(float(support["residual_p95_mm"]), 1e-6)

    def test_steep_wall_repair_keeps_click_surface(self) -> None:
        # This fixture produces the same failure shape as the 15062 report:
        # vertical/diagonal organized edges exceed the strict 35 mm gate, but
        # all four points lie on one RGB-continuous plane.
        height, width = 12, 16
        fx = fy = 500.0
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        intrinsic = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        extrinsic = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
        rows = np.arange(height, dtype=np.float64)[:, None]
        slope = 40.0
        depth = 0.5 / (1.0 + slope * (rows - cy) / fy)
        depth = np.broadcast_to(depth, (height, width)).copy()
        rgb = np.full((height, width, 3), 128, dtype=np.uint8)
        result = mesh.reconstruct_rgbd_scene_mesh(
            rgb,
            depth,
            intrinsic_matrix=intrinsic,
            extrinsic_matrix=extrinsic,
            pixel_stride=1,
        )
        edge = result.metadata["edge_rejection_counts"]
        self.assertGreater(edge["vertical_rejected"], 0)
        self.assertGreater(edge["diagonal_rejected"], 0)
        self.assertGreater(edge["triangle_repaired"], 0)
        self.assertGreater(result.metadata["surface_mesh"]["faces_count"], 0)
        clicked = np.array(
            [
                (7.0 - cx) * depth[6, 7] / fx,
                (6.0 - cy) * depth[6, 7] / fy,
                depth[6, 7],
            ],
            dtype=np.float64,
        )
        audit = result.anchor_audit(clicked, max_distance_m=0.006)
        self.assertEqual(audit["status"], "ok")
        self.assertLessEqual(float(audit["nearest_surface_distance_mm"]), 1e-6)

    def test_visible_free_space_guard_distinguishes_front_and_back(self) -> None:
        height = width = 20
        intrinsic = np.array(
            [[100.0, 0.0, 9.5], [0.0, 100.0, 9.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        extrinsic = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
        rows, cols = np.mgrid[:height, :width]
        depth = np.full((height, width), 0.5, dtype=np.float64)

        def points_at(value: float) -> np.ndarray:
            return np.stack(
                ((cols - 9.5) * value / 100.0,
                 -(rows - 9.5) * value / 100.0,
                 np.full_like(rows, value, dtype=np.float64)),
                axis=-1,
            ).reshape(-1, 3)

        front = mesh.visible_free_space_guard(
            points_at(0.4), depth, intrinsic, extrinsic, minimum_violation_pixels=1
        )
        back = mesh.visible_free_space_guard(
            points_at(0.6), depth, intrinsic, extrinsic, minimum_violation_pixels=1
        )
        self.assertFalse(front["ok"])
        self.assertGreater(front["violation_pixels"], 0)
        self.assertTrue(back["ok"])
        self.assertEqual(back["violation_pixels"], 0)

    def test_guard_rejection_falls_back_to_observed_band(self) -> None:
        rgb, depth, intrinsic, extrinsic = _camera_fixture(12, 16)
        rejected = {
            "ok": False,
            "violation_fraction": 1.0,
            "violation_pixels": 100,
            "projected_candidate_count": 100,
            "reason": "observed_free_space_occlusion",
        }
        with patch.object(mesh, "visible_free_space_guard", return_value=rejected):
            result = mesh.reconstruct_rgbd_scene_mesh(
                rgb,
                depth,
                intrinsic_matrix=intrinsic,
                extrinsic_matrix=extrinsic,
                collision_extrusion_mm=12.0,
            )
        self.assertEqual(
            result.collision_occupancy.source,
            "visibility_guard_fallback_observed_surface_band",
        )
        candidate = result.metadata["completion_candidates"][0]
        self.assertFalse(candidate["applied"])
        self.assertEqual(
            candidate["fallback_reason"],
            "completion_rejected_by_visible_free_space_guard",
        )

    def test_surface_opening_filters_support_on_either_plane_side(self) -> None:
        occupied = mesh.VoxelOccupancy(
            voxel_size_m=1.0,
            origin_world=np.zeros(3, dtype=np.float64),
            occupied_indices=np.array(
                [[0, 0, -1], [0, 0, 1]], dtype=np.int64
            ),
            source="observed_surface_fixture",
            query_radius_cells=0,
        )
        result = SimpleNamespace(
            surface_occupancy=occupied,
            collision_occupancy=occupied,
        )
        points = np.array(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        rotation = np.eye(3, dtype=np.float64)
        position = np.zeros(3, dtype=np.float64)

        near_support = mesh.MeshSceneOccupancy(
            result,
            support_z=0.0,
            target_anchor=np.array([0.0, 0.0, 0.05]),
            target_radius_m=3.0,
            support_margin_m=0.1,
        )
        self.assertEqual(
            near_support.occupied_mask(
                points, rotation, position, surface_only=True
            ).tolist(),
            [False, False],
        )

        below_support = mesh.MeshSceneOccupancy(
            result,
            support_z=0.0,
            target_anchor=np.array([0.0, 0.0, -1.0]),
            target_radius_m=3.0,
            support_margin_m=0.1,
        )
        above_support = mesh.MeshSceneOccupancy(
            result,
            support_z=0.0,
            target_anchor=np.array([0.0, 0.0, 1.0]),
            target_radius_m=3.0,
            support_margin_m=0.1,
        )
        self.assertEqual(
            below_support.occupied_mask(
                points, rotation, position, surface_only=True
            ).tolist(),
            [True, False],
        )
        self.assertEqual(
            above_support.occupied_mask(
                points, rotation, position, surface_only=True
            ).tolist(),
            [False, True],
        )


if __name__ == "__main__":
    unittest.main()
