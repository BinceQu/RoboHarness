from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import trimesh
from scipy import ndimage
from skimage.morphology import skeletonize

from behavior_interface_eval_test.tool.official_v2 import (
    rgbd_scene_mesh_v12,
    rgbd_scene_mesh_v53,
)
from behavior_interface_eval_test.tool.official_v2.depth_mesh_reconstruction import (
    backproject_depth_world,
)
from behavior_interface_eval_test.tool.official_v2.rgbd_scene_mesh_v12 import (
    RGBDSceneMeshResult,
    _elevated_planar_prism_meshes,
    _planar_prism_visible_free_space_audit,
)


ROOT = Path(__file__).resolve().parents[1]
FALSE_PRISM_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "rgbd_planar_prism_free_space_15065"
)
SIX_SCENE_FIXTURE = (
    ROOT
    / "assets"
    / "unittest"
    / "rgbd_scene_reconstruction_5010_5014"
)


def _world_from_fixture(
    depth_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    camera = metadata["camera"]
    depth = np.load(depth_path)
    world, valid = backproject_depth_world(
        depth,
        camera_pos=camera["pos"],
        camera_quat_xyzw=camera["quat"],
        focal_length=camera["focal_length"],
        horizontal_aperture=camera["horizontal_aperture"],
    )
    return world, valid, camera


class PlanarPrismVisibilityAuditTest(unittest.TestCase):
    def test_synthetic_observed_surface_passes_and_free_space_fails(self) -> None:
        candidate = trimesh.creation.box(extents=[0.4, 0.4, 0.2])
        coordinates = np.linspace(-0.15, 0.15, 20)
        xx, yy = np.meshgrid(coordinates, coordinates)
        visible_surface = np.stack(
            (xx, yy, np.full_like(xx, 0.1)),
            axis=-1,
        )
        hidden_surface = np.stack(
            (xx, yy, np.full_like(xx, -0.2)),
            axis=-1,
        )
        valid = np.ones(xx.shape, dtype=bool)
        common = {
            "camera_pos": [0.0, 0.0, 1.0],
            "focal_px": 300.0,
        }

        accepted = _planar_prism_visible_free_space_audit(
            candidate,
            visible_surface,
            valid,
            **common,
        )
        rejected = _planar_prism_visible_free_space_audit(
            candidate,
            hidden_surface,
            valid,
            **common,
        )

        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["violation_pixels"], 0)
        self.assertFalse(rejected["accepted"])
        self.assertGreater(rejected["violation_fraction"], 0.95)

    def test_15065_false_prism_is_rejected_from_rgbd_only(self) -> None:
        metadata_path = FALSE_PRISM_FIXTURE / "fixture.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        world, valid, camera = _world_from_fixture(
            FALSE_PRISM_FIXTURE / metadata["depth_file"],
            metadata_path,
        )
        _mesh, audit = _elevated_planar_prism_meshes(
            world,
            valid,
            enforce_visible_free_space=True,
            camera_pos=camera["pos"],
            focal_px=(
                camera["focal_length"]
                / camera["horizontal_aperture"]
                * camera["image_width"]
            ),
        )

        self.assertEqual(audit["planar_prism_generated_components"], 0)
        self.assertEqual(
            audit["planar_prism_rejected_visible_free_space_components"],
            1,
        )
        visibility_rows = [
            row["visible_free_space_audit"]
            for row in audit["planar_prism_components"]
            if "visible_free_space_audit" in row
        ]
        self.assertEqual(len(visibility_rows), 1)
        self.assertGreater(visibility_rows[0]["violation_fraction"], 0.80)

    def test_5014_real_box_prism_remains_enabled(self) -> None:
        scene = SIX_SCENE_FIXTURE / "5014"
        world, valid, camera = _world_from_fixture(
            scene / "img_0001.depth.npy",
            scene / "img_0001.meta.json",
        )
        mesh, audit = _elevated_planar_prism_meshes(
            world,
            valid,
            enforce_visible_free_space=True,
            camera_pos=camera["pos"],
            focal_px=(
                camera["focal_length"]
                / camera["horizontal_aperture"]
                * camera["image_width"]
            ),
        )

        self.assertIsNotNone(mesh)
        self.assertEqual(audit["planar_prism_generated_components"], 1)
        self.assertEqual(
            audit["planar_prism_rejected_visible_free_space_components"],
            0,
        )
        visibility = next(
            row["visible_free_space_audit"]
            for row in audit["planar_prism_components"]
            if "visible_free_space_audit" in row
        )
        self.assertTrue(visibility["accepted"])
        self.assertLess(visibility["violation_fraction"], 0.01)


class OfficialRGBDV53RuntimeTest(unittest.TestCase):
    def tearDown(self) -> None:
        rgbd_scene_mesh_v12.clear_support_structure_mask_cache()

    def test_support_structure_cache_normalizes_depth_dtype(self) -> None:
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        depth32 = np.ones((4, 5), dtype=np.float32)
        expected = (
            np.ones((4, 5), dtype=bool),
            object(),
            np.ones((4, 5, 3), dtype=np.float64),
            {"support_plane_detected": True},
        )
        camera = {
            "camera_pos": [0.0, 0.0, 0.0],
            "camera_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "focal_length": 17.0,
            "horizontal_aperture": 40.0,
        }
        with mock.patch.dict(
            "os.environ",
            {"OFFICIAL_V2_V53_SUPPORT_STRUCTURE_CACHE": "1"},
        ), mock.patch.object(
            rgbd_scene_mesh_v12,
            "_compute_support_structure_mask",
            return_value=expected,
        ) as compute:
            first = rgbd_scene_mesh_v12._support_structure_mask(
                rgb,
                depth32,
                **camera,
            )
            second = rgbd_scene_mesh_v12._support_structure_mask(
                rgb.copy(),
                depth32.astype(np.float64),
                **camera,
            )

        self.assertEqual(compute.call_count, 1)
        self.assertFalse(first[3]["support_structure_cache_hit"])
        self.assertTrue(second[3]["support_structure_cache_hit"])

    def test_component_bbox_metrics_equal_full_image_reference(self) -> None:
        random = np.random.default_rng(20260901)
        raw = random.random((96, 128)) < 0.06
        raw[20:36, 40:55] = True
        labels, count = ndimage.label(
            raw,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        slices = ndimage.find_objects(labels)
        for component_id in range(1, int(count) + 1):
            full = labels == component_id
            component_slice = slices[component_id - 1]
            self.assertIsNotNone(component_slice)
            local = labels[component_slice] == component_id
            padded = np.pad(local, 1, constant_values=False)

            self.assertEqual(
                int(skeletonize(padded).sum()),
                int(skeletonize(full).sum()),
            )
            self.assertEqual(
                float(ndimage.distance_transform_edt(padded).max()),
                float(ndimage.distance_transform_edt(full).max()),
            )

    def test_visibility_options_do_not_leak_into_nonstructural_base(self) -> None:
        base = RGBDSceneMeshResult(
            mesh=trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
            metadata={"build": "v11-base"},
        )
        guard_options = {
            key: value
            for key, value in rgbd_scene_mesh_v53.V13_RECONSTRUCTION_OVERRIDES.items()
        }
        with mock.patch.object(
            rgbd_scene_mesh_v12,
            "_v11_base_reconstruct_rgbd_scene_mesh",
            return_value=base,
        ) as reconstruct_v11:
            result = rgbd_scene_mesh_v12.reconstruct_rgbd_scene_mesh(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.ones((8, 8), dtype=np.float64),
                completion_method="organized_shell",
                **guard_options,
            )

        self.assertIs(result, base)
        forwarded = reconstruct_v11.call_args.kwargs
        self.assertEqual(forwarded["completion_method"], "organized_shell")
        for key in guard_options:
            self.assertNotIn(key, forwarded)

    def test_v53_enables_visibility_guard_without_changing_v52(self) -> None:
        base = RGBDSceneMeshResult(
            mesh=trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
            metadata={
                "build": "frozen-v13",
                "forbidden_inputs_used": [],
            },
        )
        with mock.patch.object(
            rgbd_scene_mesh_v53._v52,
            "_reconstruct_v13_scene_mesh",
            return_value=base,
        ) as reconstruct_v13, mock.patch.object(
            rgbd_scene_mesh_v53._v52,
            "_reconstruct_candidates",
            side_effect=ValueError("not applicable"),
        ):
            result = rgbd_scene_mesh_v53.reconstruct_rgbd_scene_mesh(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.ones((8, 8), dtype=np.float64),
                camera_pos=[0.0, 0.0, 0.0],
                camera_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                focal_length=17.0,
                horizontal_aperture=40.0,
            )

        kwargs = reconstruct_v13.call_args.kwargs
        for key, value in (
            rgbd_scene_mesh_v53.V13_RECONSTRUCTION_OVERRIDES.items()
        ):
            self.assertEqual(kwargs[key], value)
        self.assertEqual(result.metadata["mesh_version"], "v53")
        self.assertEqual(
            result.metadata["effective_mesh_version"],
            "v13_visibility_guarded_fallback",
        )
        self.assertEqual(
            result.metadata["fallback"],
            "visibility_guarded_v13_mesh",
        )
        self.assertEqual(result.metadata["forbidden_inputs_used"], [])

    def test_v53_can_defer_final_topology_checks_to_warp_cuda(self) -> None:
        base = RGBDSceneMeshResult(
            mesh=trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
            metadata={"build": "frozen-v13"},
        )
        with mock.patch.object(
            rgbd_scene_mesh_v53._v52,
            "_reconstruct_v13_scene_mesh",
            return_value=base,
        ) as reconstruct_v13, mock.patch.object(
            rgbd_scene_mesh_v53._v52,
            "_reconstruct_candidates",
            side_effect=ValueError("not applicable"),
        ):
            result = rgbd_scene_mesh_v53.reconstruct_rgbd_scene_mesh(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.ones((8, 8), dtype=np.float64),
                camera_pos=[0.0, 0.0, 0.0],
                camera_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                focal_length=17.0,
                horizontal_aperture=40.0,
                defer_topology_validation_to_warp_cuda=True,
            )

        self.assertTrue(
            reconstruct_v13.call_args.kwargs[
                "_defer_topology_validation_to_warp_cuda"
            ]
        )
        self.assertIsNone(result.metadata["mesh_watertight"])
        self.assertEqual(
            result.metadata["mesh_topology_validation"],
            "deferred_to_warp_cuda",
        )


if __name__ == "__main__":
    unittest.main()
