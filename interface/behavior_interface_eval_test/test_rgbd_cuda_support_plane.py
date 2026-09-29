from __future__ import annotations

import unittest

import numpy as np

from behavior_interface_eval_test.tool.official_v2.depth_mesh_reconstruction import (
    backproject_depth_world,
    fit_dominant_support_plane,
)
from behavior_interface_eval_test.tool.official_v2.rgbd_cuda_support_plane import (
    backproject_and_fit_support_plane_cuda,
)


class CudaSupportPlaneTest(unittest.TestCase):
    @staticmethod
    def _synthetic_depth() -> np.ndarray:
        yy, xx = np.indices((80, 96), dtype=np.float64)
        return 1.2 + xx * 2.0e-5 + yy * 1.0e-5

    def test_cpu_reference_matches_original_plane_fit(self) -> None:
        depth = self._synthetic_depth()
        camera = {
            "camera_pos": [0.3, -0.2, 1.0],
            "camera_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "focal_length": 17.0,
            "horizontal_aperture": 40.0,
        }
        world, valid = backproject_depth_world(depth, **camera)
        expected = fit_dominant_support_plane(
            world,
            valid,
            roi=(0, 0, depth.shape[1], depth.shape[0]),
        )
        actual_world, actual_valid, actual, metadata = (
            backproject_and_fit_support_plane_cuda(
                depth,
                **camera,
                requested_device="cpu",
                allow_cpu_reference=True,
            )
        )

        np.testing.assert_allclose(actual_world, world, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(actual_valid, valid)
        np.testing.assert_allclose(actual.normal, expected.normal, atol=1e-10)
        self.assertAlmostEqual(actual.offset, expected.offset, places=10)
        self.assertEqual(actual.inlier_count, expected.inlier_count)
        self.assertEqual(metadata["support_plane_device"], "cpu")

    def test_runtime_cpu_fallback_is_disabled(self) -> None:
        with self.assertRaisesRegex(
            Exception,
            "CPU V53 support-plane fitting is disabled",
        ):
            backproject_and_fit_support_plane_cuda(
                self._synthetic_depth(),
                camera_pos=[0.0, 0.0, 0.0],
                camera_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                focal_length=17.0,
                horizontal_aperture=40.0,
                requested_device="cpu",
                allow_cpu_reference=False,
            )


if __name__ == "__main__":
    unittest.main()
