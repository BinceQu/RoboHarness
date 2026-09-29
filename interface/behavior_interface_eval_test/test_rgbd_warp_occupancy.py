from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np
import trimesh

from behavior_interface_eval_test.tool.official_v2 import rgbd_grasp_lite
from behavior_interface_eval_test.tool.official_v2 import rgbd_grasp_planner
from behavior_interface_eval_test.tool.official_v2.rgbd_projective_occupancy import (
    ProjectiveOccupancyContractError,
    ProjectiveOccupancyUnavailable,
)
from behavior_interface_eval_test.tool.official_v2.rgbd_warp_occupancy import (
    build_warp_local_occupancy,
    clear_warp_mesh_cache,
)


class WarpMeshOccupancyTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_warp_mesh_cache()

    @staticmethod
    def _box(center=(0.0, 0.0, 0.05)):
        mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        mesh.apply_translation(center)
        return mesh

    def _build(self, mesh, *, hit=(0.0, 0.0, 0.1)):
        return build_warp_local_occupancy(
            mesh,
            hit=hit,
            query_offsets=[np.asarray([[0.0, 0.0, 0.0]])],
            axial_z_m=np.asarray([0.0]),
            anchor_radius_m=0.08,
            voxel_m=0.01,
            requested_device="cpu",
            allow_cpu_reference=True,
        )

    @staticmethod
    def _sample(result, point) -> bool:
        index = np.rint(
            (np.asarray(point, dtype=np.float64) - result.origin)
            / result.voxel_m
        ).astype(np.int64)
        return bool(result.occupancy[tuple(index)])

    def test_closed_box_inside_outside_and_audit(self) -> None:
        result = self._build(self._box())
        self.assertTrue(self._sample(result, [0.0, 0.0, 0.05]))
        self.assertFalse(self._sample(result, [0.08, 0.0, 0.05]))
        self.assertFalse(self._sample(result, [0.0, 0.0, 0.13]))
        self.assertTrue(result.metadata["hit_contract"]["ok"])
        self.assertEqual(result.metadata["mesh_split_calls"], 0)
        self.assertEqual(result.metadata["trimesh_contains_calls"], 0)
        self.assertFalse(result.metadata["pyembree_used"])
        self.assertEqual(
            result.metadata["topology_validation"],
            "closed_edge_incidence_gpu_component_outward_orientation",
        )
        self.assertEqual(
            result.metadata["topology_edge_incidence_failures"],
            0,
        )

    def test_inward_component_is_oriented_before_winding_query(self) -> None:
        outward = self._box(center=(-0.03, 0.0, 0.05))
        inward = self._box(center=(0.03, 0.0, 0.05))
        inward.invert()
        result = self._build(trimesh.util.concatenate((outward, inward)))

        self.assertTrue(self._sample(result, [-0.03, 0.0, 0.05]))
        self.assertTrue(self._sample(result, [0.03, 0.0, 0.05]))
        self.assertEqual(
            result.metadata["topology_inward_components_flipped"],
            1,
        )

    def test_closed_mesh_with_local_face_winding_errors_is_repaired(self) -> None:
        mesh = self._box()
        faces = np.asarray(mesh.faces, dtype=np.int64).copy()
        faces[[0, 3, 7]] = faces[[0, 3, 7]][:, (0, 2, 1)]
        inconsistent = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=faces,
            process=False,
        )

        result = self._build(inconsistent)

        self.assertTrue(self._sample(result, [0.0, 0.0, 0.05]))
        self.assertFalse(self._sample(result, [0.08, 0.0, 0.05]))
        self.assertGreater(
            result.metadata["topology_edge_orientation_failures"],
            0,
        )
        self.assertEqual(
            result.metadata[
                "topology_edge_orientation_failures_after_repair"
            ],
            0,
        )
        self.assertEqual(
            result.metadata["topology_orientation_repair_components"],
            1,
        )
        self.assertEqual(
            result.metadata["topology_orientation_repair_faces_flipped"],
            3,
        )

    def test_open_mesh_fails_closed_as_geometry_contract_error(self) -> None:
        mesh = self._box()
        mesh.update_faces(np.arange(len(mesh.faces) - 1))
        mesh.remove_unreferenced_vertices()

        with self.assertRaisesRegex(
            ProjectiveOccupancyContractError,
            "failed closed-topology validation",
        ):
            self._build(mesh)

    def test_winding_number_treats_overlapping_components_as_union(self) -> None:
        first = self._box(center=(-0.02, 0.0, 0.05))
        second = self._box(center=(0.02, 0.0, 0.05))
        mesh = trimesh.util.concatenate((first, second))
        result = self._build(mesh)
        self.assertTrue(self._sample(result, [-0.045, 0.0, 0.05]))
        self.assertTrue(self._sample(result, [0.0, 0.0, 0.05]))
        self.assertTrue(self._sample(result, [0.045, 0.0, 0.05]))
        self.assertFalse(self._sample(result, [0.09, 0.0, 0.05]))

    def test_second_query_reuses_warp_bvh(self) -> None:
        mesh = self._box()
        first = self._build(mesh)
        second = self._build(mesh, hit=(0.01, 0.0, 0.1))
        self.assertFalse(first.metadata["mesh_bvh_cache_hit"])
        self.assertTrue(second.metadata["mesh_bvh_cache_hit"])
        self.assertEqual(second.metadata["mesh_bvh_build_elapsed_s"], 0.0)

    def test_runtime_cpu_fallback_is_disabled(self) -> None:
        with self.assertRaisesRegex(
            ProjectiveOccupancyUnavailable,
            "CPU Warp occupancy is disabled",
        ):
            build_warp_local_occupancy(
                self._box(),
                hit=[0.0, 0.0, 0.1],
                query_offsets=[np.asarray([[0.0, 0.0, 0.0]])],
                axial_z_m=np.asarray([0.0]),
                anchor_radius_m=0.02,
                voxel_m=0.01,
                requested_device="cpu",
                allow_cpu_reference=False,
            )

    def test_lite_warp_path_never_calls_legacy_occupancy(self) -> None:
        mesh = self._box()
        setattr(mesh, "_official_v2_lite_scene_key", "warp-lite-test")
        with mock.patch.dict(
            os.environ,
            {
                rgbd_grasp_lite.LITE_GEOMETRY_BACKEND_ENV: (
                    rgbd_grasp_lite.LITE_GEOMETRY_BACKEND_WARP
                ),
                "OFFICIAL_V2_RGBD_LITE_TEST_MODE": "1",
                "OFFICIAL_V2_LITE_ALLOW_CPU_REFERENCE": "1",
                "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE": "cpu",
                "OFFICIAL_V2_LITE_OCCUPANCY_CACHE": "0",
            },
        ), mock.patch.object(
            rgbd_grasp_planner,
            "build_local_scene_occupancy",
            side_effect=AssertionError("legacy occupancy must not run"),
        ):
            result = rgbd_grasp_lite._build_local_scene_occupancy_lite(
                mesh,
                hit=np.asarray([0.0, 0.0, 0.1]),
                query_offsets=[np.asarray([[0.0, 0.0, 0.0]])],
                axial_z_m=np.asarray([0.0]),
                anchor_radius_m=0.08,
                voxel_m=0.01,
            )
        self.assertIsInstance(result, rgbd_grasp_planner.DenseSceneOccupancy)
        self.assertIn("warp_cpu", result.metadata["occupancy_method"])
        self.assertFalse(result.metadata["pyembree_used"])

    def test_warp_is_default_and_projective_requires_opt_in(self) -> None:
        with mock.patch.dict(
            os.environ,
            {},
            clear=False,
        ):
            os.environ.pop(rgbd_grasp_lite.LITE_GEOMETRY_BACKEND_ENV, None)
            self.assertEqual(
                rgbd_grasp_lite._lite_geometry_backend(),
                rgbd_grasp_lite.LITE_GEOMETRY_BACKEND_WARP,
            )
        with mock.patch.dict(
            os.environ,
            {
                rgbd_grasp_lite.LITE_GEOMETRY_BACKEND_ENV: (
                    rgbd_grasp_lite.LITE_GEOMETRY_BACKEND_PROJECTIVE
                )
            },
        ):
            self.assertEqual(
                rgbd_grasp_lite._lite_geometry_backend(),
                rgbd_grasp_lite.LITE_GEOMETRY_BACKEND_PROJECTIVE,
            )

    def test_deferred_topology_log_does_not_touch_trimesh_property(self) -> None:
        class DeferredMesh:
            @property
            def is_watertight(self):
                raise AssertionError("deferred log triggered CPU topology")

        value, backend = rgbd_grasp_planner._mesh_topology_diagnostic(
            DeferredMesh(),
            {"mesh_topology_validation": "deferred_to_warp_cuda"},
        )

        self.assertEqual(value, "deferred_to_warp_cuda")
        self.assertEqual(backend, "deferred_to_warp_cuda")


if __name__ == "__main__":
    unittest.main()
