from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.util.x5_grasp_geometry import (
    ASSET_PATH,
    ROBODOJO_X5_GRIPPER_SCALE_M,
    VOXEL_M,
    icosa_face_normals,
    points_inside_x5_grasp_volume,
    points_inside_legacy_x5_grasp_volume,
    x5_legacy_opening_bounds,
    x5_legacy_opening_voxels,
    x5_dynamic_opening_bounds,
    x5_dynamic_opening_mesh,
    x5_dynamic_opening_voxels,
    x5_dynamic_visual_vertices,
    x5_eef_frame_from_approach_roll,
    x5_gripper_command_fraction,
    x5_gripper_command_from_joint_position_m,
    x5_gripper_geometry,
    x5_gripper_joint_position_m,
    x5_gripper_open_fraction,
)
from tools.util.x5_gripper_visualization import (
    render_x5_grasp_volume_three_views,
    render_x5_gripper_overlay,
    render_x5_wrist_grasp_volume_overlay,
)


class X5GripperGeometryTest(unittest.TestCase):
    def test_asset_is_frozen_from_the_official_robodojo_x5(self) -> None:
        geometry = x5_gripper_geometry()
        metadata = geometry["metadata"]
        self.assertTrue(ASSET_PATH.is_file())
        self.assertEqual(metadata["eef_link"], "link6")
        self.assertEqual(metadata["schema_version"], 3)
        self.assertEqual(
            metadata["opening_geometry"],
            "two_finger_inner_surface_corridor",
        )
        self.assertEqual(
            metadata["opening_projection"],
            "xz_triangular_common_finger_silhouette",
        )
        self.assertEqual(metadata["approach_axis_local"], [1.0, 0.0, 0.0])
        self.assertEqual(metadata["gripper_open_m"], 0.044)
        self.assertEqual(metadata["finger_inflation_m"], 0.015)
        self.assertFalse(metadata["camera_geometry_included"])
        self.assertFalse(metadata["runtime_simulator_access_required"])
        self.assertEqual(
            metadata["exact_opening_y_boundary"],
            "official_link8_to_link7_inner_triangle_planes_per_xz_cell",
        )
        self.assertEqual(
            metadata["exact_opening_geometry"],
            "variable_free_volume_between_official_link7_link8_inner_triangle_surfaces",
        )
        self.assertEqual(
            metadata["exact_opening_generation"],
            "xz_triangle_arrangement_inner_surface_ray_intervals_closed_mesh",
        )
        self.assertTrue(metadata["exact_opening_mesh_watertight"])
        self.assertGreater(metadata["exact_opening_mesh_vertex_count"], 1000)
        self.assertGreater(metadata["exact_opening_mesh_triangle_count"], 2000)
        self.assertAlmostEqual(
            metadata["exact_opening_mesh_volume_cm3"],
            metadata["exact_opening_integrated_volume_cm3"],
            places=5,
        )
        self.assertEqual(
            len(metadata["exact_opening_inner_y_bounds_reference_m"]), 2
        )
        self.assertEqual(
            metadata["source_sha256"]["X5A.urdf"],
            "a31a0d3a957196ea42d216036db2fffdf7678f4f36f65f90a0b69c58daa8d7d0",
        )

    def test_opening_is_disjoint_and_inflation_is_a_solid_superset(self) -> None:
        geometry = x5_gripper_geometry()

        def keys(name: str) -> set[tuple[int, int, int]]:
            return {
                tuple(row)
                for row in np.rint(geometry[name] / VOXEL_M).astype(np.int32)
            }

        solid = keys("gripper_voxels")
        opening = keys("opening_voxels")
        inflated = keys("inflated_gripper_voxels")
        self.assertGreater(len(solid), 10_000)
        self.assertGreater(len(opening), 5_000)
        self.assertLess(len(opening), 10_000)
        self.assertFalse(solid & opening)
        self.assertLessEqual(solid, inflated)
        self.assertGreater(len(inflated), len(solid))
        self.assertEqual(len(geometry["axial_x_m"]), 7)
        self.assertTrue(np.all(np.diff(geometry["axial_x_m"]) > 0.0))

        # The X-Z footprint must follow the common two-finger silhouette.  A
        # rectangular AABB would have the same number of z columns at every x.
        columns = np.asarray(geometry["opening_columns"], dtype=np.int64)
        self.assertEqual(
            len(np.unique(columns[:, :2], axis=0)),
            int(geometry["metadata"]["opening_column_count"]),
        )
        x_keys = sorted(set(columns[:, 0].tolist()))
        spans = [int(np.count_nonzero(columns[:, 0] == ix)) for ix in x_keys]
        self.assertGreater(len(x_keys), 3)
        self.assertLess(min(spans), max(spans))
        self.assertLess(spans[-1], max(spans))

        # Pick a grid cell inside the metadata AABB but outside the shared
        # projection; it must not be classified as graspable.
        bounds = np.asarray(geometry["metadata"]["opening_bounds_eef_m"])
        projected_keys = {tuple(row[:2]) for row in columns}
        outside_projection = None
        for ix in range(
            int(np.ceil(bounds[0, 0] / VOXEL_M)),
            int(np.floor(bounds[1, 0] / VOXEL_M)) + 1,
        ):
            for iz in range(
                int(np.ceil(bounds[0, 2] / VOXEL_M)),
                int(np.floor(bounds[1, 2] / VOXEL_M)) + 1,
            ):
                if (ix, iz) not in projected_keys:
                    outside_projection = np.array(
                        [[ix * VOXEL_M, 0.0, iz * VOXEL_M]]
                    )
                    break
            if outside_projection is not None:
                break
        self.assertIsNotNone(outside_projection)
        np.testing.assert_array_equal(
            points_inside_legacy_x5_grasp_volume(outside_projection, [1.0]),
            [False],
        )

    def test_candidate_frames_use_x5_local_positive_x_approach(self) -> None:
        normals = icosa_face_normals()
        self.assertEqual(normals.shape, (20, 3))
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)
        for normal in normals:
            for roll_index in range(8):
                approach = -normal
                rotation = x5_eef_frame_from_approach_roll(
                    approach, 2.0 * np.pi * roll_index / 8.0
                )
                np.testing.assert_allclose(rotation[:, 0], approach, atol=1e-9)
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
                self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)

    def test_dynamic_grasp_volume_tracks_the_normalized_x5_opening(self) -> None:
        geometry = x5_gripper_geometry()
        full_bounds = x5_dynamic_opening_bounds([1.0])
        half_bounds = x5_dynamic_opening_bounds([0.5])
        closed_bounds = x5_dynamic_opening_bounds([0.0])
        np.testing.assert_allclose(
            full_bounds,
            [geometry["exact_opening_mesh_vertices"].min(axis=0),
             geometry["exact_opening_mesh_vertices"].max(axis=0)],
        )
        self.assertLess(full_bounds[0, 1], half_bounds[0, 1])
        self.assertGreater(full_bounds[1, 1], half_bounds[1, 1])
        self.assertLess(closed_bounds[0, 1], closed_bounds[1, 1])
        legacy_closed = x5_legacy_opening_bounds([0.0])
        self.assertGreater(legacy_closed[0, 1], legacy_closed[1, 1])

        full_voxels = x5_dynamic_opening_voxels([1.0])
        half_voxels = x5_dynamic_opening_voxels([0.5])
        closed_voxels = x5_dynamic_opening_voxels([0.0])
        self.assertGreater(len(full_voxels), len(half_voxels))
        self.assertGreater(len(half_voxels), 0)
        self.assertGreater(len(closed_voxels), 0)
        self.assertEqual(len(x5_legacy_opening_voxels([0.])), 0)
        fixed_keys = {
            tuple(row)
            for row in np.rint(geometry["fixed_solid_voxels"] / VOXEL_M).astype(
                np.int32
            )
        }
        self.assertFalse(
            fixed_keys
            & {
                tuple(row)
                for row in np.rint(x5_legacy_opening_voxels([.5]) / VOXEL_M).astype(np.int32)
            }
        )
        # Surface-voxel shells round both sides of a boundary into the same
        # cell. Exact quadrature may share that key without lying in link6.
        # Validate against its actual projected triangle union, not its
        # expanded raster shell. This is the projection removed by the exact
        # opening asset builder.
        fixed_faces = geometry["visual_faces"][geometry["visual_face_component"] == 0]
        triangles = geometry["visual_vertices"][fixed_faces][:, :, [0, 2]]
        footprint = half_voxels[:, [0, 2]]
        strictly_in_solid_projection = np.zeros(len(footprint), dtype=bool)
        for triangle in triangles:
            matrix = np.column_stack((triangle[1]-triangle[0], triangle[2]-triangle[0]))
            if abs(np.linalg.det(matrix)) < 1e-18:
                continue
            bc = (footprint-triangle[0]) @ np.linalg.inv(matrix).T
            weights = np.column_stack((1-bc.sum(axis=1), bc))
            strictly_in_solid_projection |= np.all(weights > 1e-7, axis=1)
        self.assertFalse(strictly_in_solid_projection.any())
        probes = np.array(
            [
                [0.11, 0.00, 0.00],
                [0.11, 0.03, 0.00],
                [0.02, 0.00, 0.00],
            ]
        )
        np.testing.assert_array_equal(
            points_inside_x5_grasp_volume(probes, [1.0]),
            [True, True, False],
        )
        self.assertFalse(
            bool(points_inside_x5_grasp_volume([[0.11, 0.03, 0.0]], [0.5])[0])
        )
        np.testing.assert_array_equal(
            points_inside_x5_grasp_volume(probes, [0.0]),
            [True, False, False],
        )

        full_vertices = x5_dynamic_visual_vertices([1.0])
        closed_vertices = x5_dynamic_visual_vertices([0.0])
        faces = geometry["visual_faces"]
        components = geometry["visual_face_component"]
        positive_finger = np.unique(faces[components == 1])
        negative_finger = np.unique(faces[components == 2])
        np.testing.assert_allclose(
            closed_vertices[positive_finger, 1]
            - full_vertices[positive_finger, 1],
            -0.044,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            closed_vertices[negative_finger, 1]
            - full_vertices[negative_finger, 1],
            0.044,
            atol=1e-9,
        )

    def test_exact_opening_mesh_is_closed_and_independent_of_voxel_empty_cases(self) -> None:
        geometry = x5_gripper_geometry()
        full = x5_dynamic_opening_mesh([1.0])
        closed = x5_dynamic_opening_mesh([0.0])
        self.assertTrue(full["exact"])
        self.assertTrue(closed["exact"])
        self.assertGreater(full["volume_cm3"], closed["volume_cm3"])
        # The official inner triangles have a varying Y gap; this cannot be a
        # constant-width cube or slab.
        vertex_count = len(full["vertices"]) // 2
        gaps = (
            full["vertices"][vertex_count:, 1]
            - full["vertices"][:vertex_count, 1]
        )
        self.assertGreater(float(gaps.max() - gaps.min()), 1e-3)
        self.assertGreater(closed["volume_cm3"], 0.0)
        self.assertEqual(len(full["vertices"]), len(closed["vertices"]))
        self.assertEqual(len(full["faces"]), len(closed["faces"]))
        self.assertGreater(len(x5_dynamic_opening_voxels([0.0])), 0)
        self.assertEqual(len(x5_legacy_opening_voxels([0.0])), 0)
        reference_y = np.asarray(
            geometry["metadata"]["exact_opening_inner_y_bounds_reference_m"],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            full["inner_y_bounds_m"], reference_y, atol=2e-7
        )
        # The mesh bounds are extrema of the official variable inner
        # triangle surfaces, not an exterior construction slab.
        np.testing.assert_allclose(
            [full["vertices"][:, 1].min(), full["vertices"][:, 1].max()],
            reference_y,
            atol=2e-7,
        )
        closed_reference = np.array(
            [
                reference_y[0]
                + float(geometry["metadata"]["gripper_open_m"]),
                reference_y[1]
                - float(geometry["metadata"]["gripper_open_m"]),
            ]
        )
        np.testing.assert_allclose(
            closed["inner_y_bounds_m"], closed_reference, atol=2e-7
        )

        # Every exact mesh edge is shared by two triangles, i.e. the Boolean
        # result has no open boundary introduced by projection/extrusion.
        edges = np.sort(
            np.concatenate(
                [
                    full["faces"][:, [0, 1]],
                    full["faces"][:, [1, 2]],
                    full["faces"][:, [2, 0]],
                ],
                axis=0,
            ),
            axis=1,
        )
        _unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, np.full(len(counts), 2))

    def test_command_state_maps_through_official_x5_gripper_scale(self) -> None:
        self.assertEqual(ROBODOJO_X5_GRIPPER_SCALE_M, (-0.01, 0.044))
        self.assertEqual(x5_gripper_command_fraction([1.2]), 1.0)
        self.assertEqual(x5_gripper_command_fraction([-0.2]), 0.0)
        self.assertAlmostEqual(x5_gripper_joint_position_m([1.0]), 0.044)
        self.assertAlmostEqual(x5_gripper_joint_position_m([0.5]), 0.017)
        self.assertAlmostEqual(x5_gripper_joint_position_m([0.0]), 0.0)
        self.assertAlmostEqual(x5_gripper_open_fraction([0.5]), 0.017 / 0.044)
        reconstructed = x5_gripper_command_from_joint_position_m(0.019)
        self.assertAlmostEqual(reconstructed, (0.019 + 0.01) / 0.054)
        self.assertAlmostEqual(x5_gripper_joint_position_m([reconstructed]), 0.019)

    def test_x5_three_views_mark_the_grasp_volume_red(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "x5_grasp_three_views.png"
            result = render_x5_grasp_volume_three_views(
                gripper_qpos=[x5_gripper_command_from_joint_position_m(0.019)],
                command_qpos=[0.87],
                output_path=output,
            )
            self.assertTrue(result["ok"], result)
            self.assertEqual(
                [view["name"] for view in result["views"]],
                ["top", "side", "front"],
            )
            self.assertGreater(result["opening_voxel_count"], 0)
            self.assertEqual(
                result["opening_geometry"],
                "two_finger_inner_surface_corridor",
            )
            self.assertEqual(
                result["opening_projection"],
                "xz_triangular_common_finger_silhouette",
            )
            self.assertEqual(
                result["opening_render"],
                "exact_projected_opening_voxel_union",
            )
            self.assertTrue(result["exact_opening_mesh_exact"])
            self.assertEqual(
                result["exact_opening_y_boundary"],
                "official_link8_to_link7_inner_triangle_planes_per_xz_cell",
            )
            self.assertEqual(
                result["exact_opening_mesh_source"],
                "frozen_official_x5_inner_surface_triangle_mesh_prismatic_translation",
            )
            self.assertEqual(
                result["opening_boundary_source"],
                "projected_exact_opening_mesh_union_contours",
            )
            for view in result["views"]:
                boundary = view["opening_boundary"]
                self.assertGreaterEqual(boundary["contour_count"], 1)
                self.assertGreaterEqual(boundary["contour_vertex_count"], 3)
                self.assertEqual(
                    boundary["contour_vertex_count"],
                    sum(len(contour) for contour in boundary["contours_px"]),
                )
                self.assertEqual(len(boundary["bbox_px"]), 4)
            self.assertEqual(len(result["exact_opening_inner_y_bounds_m"]), 2)
            self.assertAlmostEqual(result["gripper_joint_position_m"][0], 0.019)
            self.assertAlmostEqual(result["gripper_qpos_normalized"][0], 0.87)
            self.assertNotEqual(
                result["gripper_geometry_qpos_normalized"],
                result["gripper_qpos_normalized"],
            )
            image = cv2.imread(str(output), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image)
            red = (
                (image[..., 2] > 130)
                & (image[..., 2] > image[..., 1] + 60)
                & (image[..., 2] > image[..., 0] + 60)
            )
            self.assertGreater(int(np.count_nonzero(red)), 500)

    def test_wrist_overlay_marks_only_depth_points_inside_x5_grasp_volume(self) -> None:
        rgb = np.full((160, 160, 3), 70, dtype=np.uint8)
        depth = np.full((160, 160), 0.115, dtype=np.float32)
        intrinsic = np.array(
            [[400.0, 0.0, 79.5], [0.0, 400.0, 79.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = np.array(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        eef_pose = {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}
        with tempfile.TemporaryDirectory() as directory:
            overlay = Path(directory) / "wrist_grasp.png"
            mask = Path(directory) / "wrist_grasp.npy"
            result = render_x5_wrist_grasp_volume_overlay(
                rgb=rgb,
                depth_linear=depth,
                intrinsic_matrix=intrinsic,
                extrinsic_matrix=extrinsic,
                eef_pose=eef_pose,
                gripper_qpos=[1.0],
                output_path=overlay,
                mask_path=mask,
            )
            self.assertTrue(result["ok"], result)
            self.assertGreater(result["red_pixel_count"], 0)
            self.assertTrue(result["robot_mask_used"])
            self.assertIn("self_rejected_pixel_count", result)
            self.assertIn("gripper_geometry_qpos_normalized", result)
            registration = result["gripper_depth_registration"]
            self.assertEqual(
                registration["source"],
                "evaluator_wrist_depth_registered_official_x5_finger_mesh",
            )
            self.assertFalse(registration["segmentation_used"])
            self.assertFalse(registration["simulator_state_used"])
            self.assertLessEqual(
                result["red_pixel_count"], result["red_candidate_pixel_count"]
            )
            self.assertFalse(result["segmentation_used"])
            self.assertFalse(result["simulator_geometry_used"])
            saved_mask = np.load(mask)
            self.assertTrue(bool(saved_mask[80, 80]))

    def test_red_mesh_renderer_writes_visible_depth_aware_png(self) -> None:
        rgb = np.full((96, 128, 3), 80, dtype=np.uint8)
        depth = np.full((96, 128), 2.0, dtype=np.float32)
        intrinsic = np.array(
            [[220.0, 0.0, 63.5], [0.0, 220.0, 47.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        extrinsic = np.eye(4, dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "x5_red.png"
            result = render_x5_gripper_overlay(
                rgb=rgb,
                depth_linear=depth,
                intrinsic_matrix=intrinsic,
                extrinsic_matrix=extrinsic,
                eef_pose={
                    "pos": [-0.08, 0.0, -0.75],
                    "quat": [1.0, 0.0, 0.0, 0.0],
                },
                output_path=output,
            )
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["color"], "red")
            self.assertTrue(result["depth_occlusion"])
            self.assertGreater(result["rasterized_pixel_count"], 0)
            self.assertGreater(result["visible_pixel_count"], 0)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
