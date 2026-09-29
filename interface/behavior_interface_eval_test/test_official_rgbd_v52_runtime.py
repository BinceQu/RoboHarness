from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh
from PIL import Image

from behavior_interface_eval_test.tool.official_v2 import rgbd_grasp_lite
from behavior_interface_eval_test.tool.official_v2 import rgbd_grasp_planner
from behavior_interface_eval_test.tool.official_v2 import rgbd_scene_mesh_v52
from behavior_interface_eval_test.tool.official_v2.rgbd_scene_mesh_v12 import (
    RGBDSceneMeshResult,
)


class OfficialRGBDV52RuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        self.depth = np.full((8, 8), 0.8, dtype=np.float64)
        self.camera = {
            "camera_pos": [0.0, 0.0, 0.0],
            "camera_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "focal_length": 17.0,
            "horizontal_aperture": 40.0,
        }

    def test_v52_falls_back_only_to_frozen_v13(self) -> None:
        base_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        base = RGBDSceneMeshResult(
            mesh=base_mesh,
            metadata={"build": "frozen-v13", "forbidden_inputs_used": []},
        )
        with mock.patch.object(
            rgbd_scene_mesh_v52,
            "_reconstruct_v13_scene_mesh",
            return_value=base,
        ) as reconstruct_v13, mock.patch.object(
            rgbd_scene_mesh_v52,
            "_reconstruct_candidates",
            side_effect=ValueError("not applicable"),
        ):
            result = rgbd_scene_mesh_v52.reconstruct_rgbd_scene_mesh(
                self.rgb,
                self.depth,
                **self.camera,
            )

        kwargs = reconstruct_v13.call_args.kwargs
        self.assertEqual(kwargs["completion_method"], "structural_hybrid")
        self.assertEqual(kwargs["pixel_stride"], 3)
        self.assertNotIn(
            "structural_planar_prism_visibility_guard",
            kwargs,
        )
        self.assertEqual(result.metadata["mesh_version"], "v52")
        self.assertEqual(
            result.metadata["effective_mesh_version"],
            "v13_fallback",
        )
        self.assertFalse(result.metadata["expert_applied"])
        self.assertEqual(result.metadata["fallback"], "frozen_v13_mesh")
        self.assertEqual(result.metadata["forbidden_inputs_used"], [])
        np.testing.assert_allclose(result.mesh.vertices, base_mesh.vertices)

    def test_v52_falls_back_when_tmpdir_stays_missing(self) -> None:
        base_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        base = RGBDSceneMeshResult(
            mesh=base_mesh,
            metadata={"build": "frozen-v13", "forbidden_inputs_used": []},
        )
        with mock.patch.object(
            rgbd_scene_mesh_v52,
            "_reconstruct_v13_scene_mesh",
            return_value=base,
        ), mock.patch.object(
            rgbd_scene_mesh_v52.tempfile,
            "TemporaryDirectory",
            side_effect=FileNotFoundError("missing parent"),
        ):
            result = rgbd_scene_mesh_v52.reconstruct_rgbd_scene_mesh(
                self.rgb,
                self.depth,
                **self.camera,
            )
        self.assertEqual(result.metadata["effective_mesh_version"], "v13_fallback")
        self.assertIn("FileNotFoundError", result.metadata["reason"])

    def test_v52_retries_after_missing_runtime_tmp(self) -> None:
        base = RGBDSceneMeshResult(
            mesh=trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
            metadata={"build": "frozen-v13", "forbidden_inputs_used": []},
        )
        candidate_mesh = trimesh.creation.icosphere(radius=0.05)
        attempts = {"n": 0}

        def reconstruct_candidates(**kwargs):
            mesh_path = kwargs["output_dir"] / "p90.ply"
            candidate_mesh.export(mesh_path)
            return [
                {
                    "name": "v52_boundary_uncertainty_p90_band",
                    "mesh_path": str(mesh_path),
                    "runtime_eligible": True,
                }
            ]

        real_td = tempfile.TemporaryDirectory

        def flaky_td(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise FileNotFoundError("missing parent")
            return real_td(*args, **kwargs)

        with mock.patch.object(
            rgbd_scene_mesh_v52,
            "_reconstruct_v13_scene_mesh",
            return_value=base,
        ), mock.patch.object(
            rgbd_scene_mesh_v52,
            "_reconstruct_candidates",
            side_effect=reconstruct_candidates,
        ), mock.patch.object(
            rgbd_scene_mesh_v52.tempfile,
            "TemporaryDirectory",
            side_effect=flaky_td,
        ):
            result = rgbd_scene_mesh_v52.reconstruct_rgbd_scene_mesh(
                self.rgb,
                self.depth,
                **self.camera,
            )

        self.assertEqual(attempts["n"], 2)
        self.assertTrue(result.metadata["expert_applied"])
        self.assertEqual(result.metadata["effective_mesh_version"], "v52")

    def test_stale_temp_cache_recovers_without_original_runtime_import(self) -> None:
        from behavior_interface_eval_test.tool.official_v2 import rgbd_scene_mesh_v53
        base = RGBDSceneMeshResult(
            mesh=trimesh.creation.box(extents=[0.1, 0.1, 0.1]), metadata={}
        )
        with tempfile.TemporaryDirectory() as root:
            for module in (rgbd_scene_mesh_v52, rgbd_scene_mesh_v53):
                with self.subTest(version=module.__name__), mock.patch.object(
                    rgbd_scene_mesh_v52, "_reconstruct_v13_scene_mesh", return_value=base
                ), mock.patch.object(
                    rgbd_scene_mesh_v52, "_reconstruct_candidates", side_effect=ValueError("not applicable")
                ), mock.patch.object(tempfile, "tempdir", os.path.join(root, "missing")):
                    result = module.reconstruct_rgbd_scene_mesh(self.rgb, self.depth, **self.camera)
                    self.assertFalse(result.metadata["expert_applied"])
                    self.assertNotIn("FileNotFoundError", result.metadata["reason"])
                    self.assertTrue(os.path.isdir(tempfile.gettempdir()))

    def test_v52_loads_the_canonical_p90_candidate(self) -> None:
        base = RGBDSceneMeshResult(
            mesh=trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
            metadata={"build": "frozen-v13", "forbidden_inputs_used": []},
        )
        candidate_mesh = trimesh.creation.icosphere(radius=0.05)

        def reconstruct_candidates(**kwargs):
            mesh_path = kwargs["output_dir"] / "p90.ply"
            candidate_mesh.export(mesh_path)
            return [
                {
                    "name": "v52_boundary_uncertainty_p90_band",
                    "mesh_path": str(mesh_path),
                    "runtime_eligible": True,
                }
            ]

        with mock.patch.object(
            rgbd_scene_mesh_v52,
            "_reconstruct_v13_scene_mesh",
            return_value=base,
        ), mock.patch.object(
            rgbd_scene_mesh_v52,
            "_reconstruct_candidates",
            side_effect=reconstruct_candidates,
        ):
            result = rgbd_scene_mesh_v52.reconstruct_rgbd_scene_mesh(
                self.rgb,
                self.depth,
                **self.camera,
            )

        self.assertTrue(result.metadata["expert_applied"])
        self.assertEqual(result.metadata["effective_mesh_version"], "v52")
        self.assertIsNone(result.metadata["fallback"])
        self.assertEqual(
            result.metadata["selected_candidate"]["name"],
            "v52_boundary_uncertainty_p90_band",
        )
        self.assertNotIn(
            "mesh_path",
            result.metadata["selected_candidate"],
        )
        self.assertEqual(len(result.mesh.faces), len(candidate_mesh.faces))

    def test_session_adapter_reports_top_level_elapsed_time(self) -> None:
        reconstructed = RGBDSceneMeshResult(
            mesh=trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
            metadata={"mesh_version": "v52"},
        )
        rgb_image = Image.fromarray(self.rgb)
        with mock.patch.object(
            rgbd_scene_mesh_v52.os.path,
            "isfile",
            return_value=True,
        ), mock.patch(
            "PIL.Image.open",
            return_value=rgb_image,
        ), mock.patch.object(
            rgbd_scene_mesh_v52.np,
            "load",
            return_value=self.depth,
        ), mock.patch.object(
            rgbd_scene_mesh_v52,
            "reconstruct_rgbd_scene_mesh",
            return_value=reconstructed,
        ), mock.patch.object(
            rgbd_scene_mesh_v52.time,
            "perf_counter",
            side_effect=[10.0, 12.5],
        ):
            _mesh, _rgb, _depth, metadata = (
                rgbd_scene_mesh_v52.reconstruct_scene_mesh_from_session(
                    {"rgb_path": "/tmp/rgb.png", "depth_path": "/tmp/depth.npy"},
                    **self.camera,
                )
            )

        self.assertEqual(metadata["elapsed_s"], 2.5)

    def test_production_planner_does_not_require_reconstruction_timing(self) -> None:
        class LogObserved(RuntimeError):
            pass

        class StopAfterFirstLog:
            def raise_if_cancelled(self, _label: str) -> None:
                return None

            def log(self, message: str) -> None:
                raise LogObserved(message)

        prepared_scene = (
            object(),
            self.rgb,
            self.depth,
            {
                "build": "rgbd_representative_v52_runtime",
                "mesh_version": "v52",
            },
        )
        prepared_target = {
            "hit": np.asarray([0.0, 0.0, 0.8]),
            "hit_audit": {"method": "test_depth_hit"},
            "outward_normal": None,
            "normal_audit": {"confidence": "none"},
        }

        with self.assertRaisesRegex(LogObserved, "reconstruct=unknown"):
            rgbd_grasp_planner.plan_grasp_point_filter_rgbd(
                world=object(),
                session={"session_id": "test"},
                u=1,
                v=2,
                cam_pos=np.zeros(3),
                cam_quat=np.asarray([0.0, 0.0, 0.0, 1.0]),
                w=8,
                h=8,
                fl=17.0,
                ha=40.0,
                plan_arm="any",
                seed=42,
                ctx=StopAfterFirstLog(),
                prepared_scene=prepared_scene,
                prepared_target=prepared_target,
                render_debug=False,
            )

    def test_lite_reconstructs_once_and_reuses_scene_for_full_fallback(self) -> None:
        geometry_version = rgbd_grasp_lite.RGBD_LITE_GEOMETRY_VERSION
        projective_scene = rgbd_grasp_lite.ProjectiveDepthScene(
            depth=self.depth.astype(np.float32),
            camera_pos=np.zeros(3, dtype=np.float64),
            camera_quat_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
            focal_length=17.0,
            horizontal_aperture=40.0,
            scene_key="fallback-test",
            metadata={},
        )
        prepared_scene = (
            projective_scene,
            None,
            self.depth,
            {
                "build": "rgbd_lite_projective_cuda_depth_shell_v1",
                "geometry_version": geometry_version,
                "effective_geometry_version": geometry_version,
                "scene_mesh_built": False,
            },
        )
        seen = {}

        def lite_planner(**kwargs):
            seen["lite"] = kwargs["prepared_scene"]
            raise rgbd_grasp_planner.GraspObjPlanningError("lite failed")

        def full_planner(**kwargs):
            seen["full"] = kwargs["prepared_scene"]
            return {
                "ok": True,
                "candidates": [],
                "plan_audit": {
                    "build": rgbd_grasp_planner.BUILD,
                    "reconstruction": dict(prepared_scene[3]),
                },
                "grip_fit": {
                    "build": rgbd_grasp_planner.BUILD,
                    "volume_source": "rgbd_v8_dense_occupancy",
                },
            }

        call_state = {
            "skipped_se3_ik_stages": [],
            "ik_worker_policies": [],
            "phase_timings": {},
            "rank_calls": 0,
            "micro_rank_calls": 0,
            "warm_solver_prepare_requested": False,
            "warm_solver_prepare_elapsed_s": 0.0,
        }
        with mock.patch.object(
            rgbd_grasp_lite,
            "reconstruct_v52_scene_from_session",
            return_value=prepared_scene,
        ) as reconstruct, mock.patch.object(
            rgbd_grasp_lite,
            "_ensure_lite_gripper_geometry_cache",
            return_value={"cache_hit": True},
        ), mock.patch.object(
            rgbd_grasp_lite,
            "_production_planner_clone_with_lite_ranker",
            return_value=(lite_planner, call_state),
        ), mock.patch.object(
            rgbd_grasp_lite.production,
            "plan_grasp_point_filter_rgbd",
            side_effect=full_planner,
        ), mock.patch.object(
            rgbd_grasp_lite,
            "_production_planner_clone_with_exact_runtime_reuse",
            return_value=full_planner,
        ), mock.patch.object(
            rgbd_grasp_lite,
            "prepare_external_ik",
        ), mock.patch.object(
            rgbd_grasp_lite,
            "_attach_lite_candidate_diagnostics",
        ), mock.patch.dict(
            os.environ,
            {
                rgbd_grasp_lite.LITE_FALLBACK_ENV: "1",
                "OFFICIAL_V2_LITE_SCENE_CACHE": "0",
                rgbd_grasp_lite.LITE_GEOMETRY_BACKEND_ENV: (
                    rgbd_grasp_lite.LITE_GEOMETRY_BACKEND_PROJECTIVE
                ),
            },
        ):
            payload = rgbd_grasp_lite.plan_grasp_point_filter_rgbd_lite(
                world=object(),
                session={"session_id": "test"},
                u=1,
                v=2,
                cam_pos=np.zeros(3),
                cam_quat=np.asarray([0.0, 0.0, 0.0, 1.0]),
                w=8,
                h=8,
                fl=17.0,
                ha=40.0,
                plan_arm="any",
                seed=42,
            )

        reconstruct.assert_called_once()
        self.assertIs(seen["lite"], prepared_scene)
        self.assertIs(seen["full"], prepared_scene)
        lite_audit = payload["plan_audit"]["lite_planner"]
        self.assertTrue(lite_audit["fallback_used"])
        self.assertTrue(lite_audit["scene_reconstruction_prepared_once"])
        self.assertTrue(lite_audit["scene_reconstruction_reused_by_fallback"])
        self.assertIn(
            "projective_cuda_depth_shell_v1",
            payload["plan_audit"]["build"],
        )
        self.assertEqual(
            payload["grip_fit"]["volume_source"],
            "rgbd_projective_cuda_v1_dense_occupancy_experimental",
        )
        self.assertIsNone(lite_audit["scene_mesh_requested_version"])
        self.assertFalse(lite_audit["scene_mesh_built"])


if __name__ == "__main__":
    unittest.main()
