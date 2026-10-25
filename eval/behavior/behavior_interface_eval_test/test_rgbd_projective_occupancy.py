from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from behavior_interface_eval_test.tool.official_v2 import rgbd_grasp_lite
from behavior_interface_eval_test.tool.official_v2 import rgbd_grasp_planner
from behavior_interface_eval_test.tool.official_v2 import rgbd_projective_occupancy
from behavior_interface_eval_test.tool.official_v2.grasp_geometry_local import (
    gripper_geometry,
)


class RGBDProjectiveOccupancyTest(unittest.TestCase):
    def _scene(self, depth: np.ndarray) -> rgbd_projective_occupancy.ProjectiveDepthScene:
        return rgbd_projective_occupancy.ProjectiveDepthScene(
            depth=np.asarray(depth, dtype=np.float32),
            camera_pos=np.zeros(3, dtype=np.float64),
            camera_quat_xyzw=np.asarray(
                [0.0, 0.0, 0.0, 1.0],
                dtype=np.float64,
            ),
            focal_length=17.0,
            horizontal_aperture=20.0,
            scene_key="unit-flat",
            metadata={},
        )

    @staticmethod
    def _at(grid, point) -> bool:
        index = np.rint(
            (np.asarray(point, dtype=np.float64) - grid.origin)
            / float(grid.voxel_m)
        ).astype(np.int64)
        return bool(grid.occupancy[tuple(index)])

    def _build(self, scene, *, hit=(0.0, 0.0, -1.0), max_voxels=None):
        offsets = np.asarray(
            [
                [-0.08, 0.0, 0.0],
                [0.08, 0.0, 0.0],
                [0.0, 0.0, -0.08],
                [0.0, 0.0, 0.08],
            ],
            dtype=np.float64,
        )
        return rgbd_projective_occupancy.build_projective_local_occupancy(
            scene,
            hit=hit,
            query_offsets=[offsets],
            axial_z_m=np.asarray([0.0], dtype=np.float64),
            anchor_radius_m=0.0,
            voxel_m=0.01,
            requested_device="cpu",
            allow_cpu_reference=True,
            max_grid_voxels=max_voxels,
        )

    def test_flat_depth_builds_only_surface_and_back_shell(self) -> None:
        grid = self._build(self._scene(np.ones((21, 21), dtype=np.float32)))

        self.assertFalse(self._at(grid, [0.0, 0.0, -0.97]))
        self.assertTrue(self._at(grid, [0.0, 0.0, -1.00]))
        self.assertTrue(self._at(grid, [0.0, 0.0, -1.05]))
        self.assertFalse(self._at(grid, [0.0, 0.0, -1.08]))
        self.assertTrue(grid.metadata["hit_contract"]["ok"])
        self.assertEqual(grid.metadata["mesh_split_calls"], 0)
        self.assertEqual(grid.metadata["trimesh_contains_calls"], 0)
        self.assertFalse(grid.metadata["pyembree_used"])

    def test_valid_center_depth_is_not_replaced_across_an_edge(self) -> None:
        depth = np.ones((21, 21), dtype=np.float32)
        depth[10, 11] = 0.94
        grid = self._build(self._scene(depth))

        self.assertFalse(self._at(grid, [0.0, 0.0, -0.95]))
        self.assertTrue(self._at(grid, [0.0, 0.0, -1.00]))
        self.assertIn(
            "no_cross_edge_union",
            grid.metadata["pixel_neighborhood_policy"],
        )

    def test_invalid_center_uses_nearest_valid_depth_only_as_fallback(self) -> None:
        depth = np.ones((21, 21), dtype=np.float32)
        depth[10, 10] = np.nan
        grid = self._build(self._scene(depth))

        self.assertTrue(self._at(grid, [0.0, 0.0, -1.00]))
        self.assertTrue(grid.metadata["hit_contract"]["ok"])

    def test_runtime_cpu_backend_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            rgbd_projective_occupancy.ProjectiveOccupancyUnavailable,
            "CPU projective occupancy is disabled",
        ):
            rgbd_projective_occupancy.build_projective_local_occupancy(
                self._scene(np.ones((9, 9), dtype=np.float32)),
                hit=[0.0, 0.0, -1.0],
                query_offsets=[np.zeros((1, 3), dtype=np.float64)],
                axial_z_m=np.asarray([0.0]),
                anchor_radius_m=0.0,
                voxel_m=0.01,
                requested_device="cpu",
                allow_cpu_reference=False,
            )

    def test_grid_limit_fails_before_backend_initialization(self) -> None:
        with self.assertRaisesRegex(
            rgbd_projective_occupancy.ProjectiveOccupancyContractError,
            "limit is 100",
        ):
            self._build(
                self._scene(np.ones((21, 21), dtype=np.float32)),
                max_voxels=100,
            )

    def test_pose_count_contract_cannot_fall_back_to_cpu(self) -> None:
        occupancy = rgbd_grasp_planner.DenseSceneOccupancy(
            origin=np.zeros(3, dtype=np.float64),
            occupancy=np.ones((3, 3, 3), dtype=bool),
            voxel_m=0.01,
            metadata={"pose_count_device_required": "cuda:0"},
        )
        pose = {
            "eef_pos": np.zeros(3, dtype=np.float64),
            "R": np.eye(3, dtype=np.float64),
        }
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(
                RuntimeError,
                "CPU fallback is disabled",
            ):
                occupancy.counts_for_poses(
                    [pose],
                    np.zeros((1, 3), dtype=np.float64),
                )

    def test_scene_adapter_needs_depth_but_not_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            depth_path = os.path.join(directory, "depth.npy")
            np.save(depth_path, np.ones((7, 9), dtype=np.float32))
            scene, rgb, depth, metadata = (
                rgbd_projective_occupancy.prepare_projective_scene_from_session(
                    {"depth_path": depth_path},
                    camera_pos=[0.0, 0.0, 0.0],
                    camera_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                    focal_length=17.0,
                    horizontal_aperture=20.0,
                    scene_key="adapter",
                )
            )

        self.assertIsNone(rgb)
        self.assertEqual(depth.shape, (7, 9))
        self.assertIs(scene.depth, depth)
        self.assertFalse(metadata["scene_mesh_built"])
        self.assertTrue(metadata["whole_scene_mesh_skipped"])

    def test_lite_projective_path_never_calls_legacy_mesh_occupancy(self) -> None:
        scene = self._scene(np.ones((21, 21), dtype=np.float32))
        kwargs = {
            "hit": np.asarray([0.0, 0.0, -1.0]),
            "query_offsets": [
                np.asarray(
                    [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]],
                    dtype=np.float64,
                )
            ],
            "axial_z_m": np.asarray([0.0]),
            "anchor_radius_m": 0.0,
            "voxel_m": 0.01,
        }
        with mock.patch.object(
            rgbd_grasp_lite.production,
            "build_local_scene_occupancy",
            side_effect=AssertionError("legacy mesh occupancy called"),
        ), mock.patch.dict(
            os.environ,
            {
                "OFFICIAL_V2_RGBD_LITE_TEST_MODE": "1",
                "OFFICIAL_V2_LITE_ALLOW_CPU_REFERENCE": "1",
                "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE": "cpu",
                "OFFICIAL_V2_LITE_OCCUPANCY_CACHE": "0",
            },
        ):
            result = rgbd_grasp_lite._build_local_scene_occupancy_lite(
                scene,
                **kwargs,
            )

        self.assertEqual(
            result.metadata["occupancy_method"],
            "torch_cpu_projective_depth_shell_reference",
        )
        self.assertEqual(result.metadata["mesh_split_calls"], 0)
        self.assertEqual(result.metadata["trimesh_contains_calls"], 0)

    def test_5012_fixture_hit_and_full_candidate_envelope_contract(self) -> None:
        fixture = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "unittest"
            / "rgbd_scene_reconstruction_5010_5014"
            / "5012"
        )
        metadata = json.loads(
            (fixture / "img_0001.meta.json").read_text(encoding="utf-8")
        )
        camera = metadata["camera"]
        depth = np.load(fixture / "img_0001.depth.npy").astype(np.float32)
        target = rgbd_grasp_planner.prepare_rgbd_grasp_target(
            depth,
            u=431,
            v=361,
            cam_pos=np.asarray(camera["pos"]),
            cam_quat=np.asarray(camera["quat"]),
            w=int(camera["image_width"]),
            h=int(camera["image_height"]),
            fl=float(camera["focal_length"]),
            ha=float(camera["horizontal_aperture"]),
        )
        scene = rgbd_projective_occupancy.ProjectiveDepthScene(
            depth=depth,
            camera_pos=np.asarray(camera["pos"], dtype=np.float64),
            camera_quat_xyzw=np.asarray(camera["quat"], dtype=np.float64),
            focal_length=float(camera["focal_length"]),
            horizontal_aperture=float(camera["horizontal_aperture"]),
            scene_key="fixture-5012",
            metadata={},
        )
        local = rgbd_projective_occupancy.build_projective_local_occupancy(
            scene,
            hit=target["hit"],
            query_offsets=[np.zeros((1, 3), dtype=np.float64)],
            axial_z_m=np.asarray([0.0]),
            anchor_radius_m=0.009,
            voxel_m=rgbd_grasp_planner.VOXEL_M,
            requested_device="cpu",
            allow_cpu_reference=True,
        )
        self.assertTrue(local.metadata["hit_contract"]["ok"])
        self.assertLess(
            local.metadata["hit_contract"]["nearest_occupied_distance_mm"],
            3.0,
        )

        geometry = gripper_geometry(rgbd_grasp_planner.VOXEL_M)
        inflated, _regions, _audit = (
            rgbd_grasp_planner.inflated_gripper_voxels_eef(
                geometry["gripper_voxels"],
                component_voxels=geometry["components"],
                voxel_m=rgbd_grasp_planner.VOXEL_M,
                camera_inflate_m=rgbd_grasp_planner.CAMERA_INFLATE_M,
                finger_inward_inflate_m=(
                    rgbd_grasp_planner.FINGER_INWARD_INFLATE_M
                ),
                return_regions=True,
            )
        )
        _origin, _upper, shape, _radius = (
            rgbd_projective_occupancy.local_grid_spec(
                hit=target["hit"],
                query_offsets=[
                    geometry["gripper_voxels"],
                    geometry["opening_voxels"],
                    inflated,
                ],
                axial_z_m=rgbd_grasp_planner.AXIAL_Z_M,
                anchor_radius_m=(
                    rgbd_grasp_planner.RESCUE_OCCUPANCY_MARGIN_M
                ),
                voxel_m=rgbd_grasp_planner.VOXEL_M,
            )
        )
        self.assertEqual(tuple(shape), (132, 132, 132))
        self.assertEqual(int(np.prod(shape)), 2_299_968)


if __name__ == "__main__":
    unittest.main()
