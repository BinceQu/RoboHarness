from __future__ import annotations

import unittest

import numpy as np

from behavior_interface_eval_test.tool.official_v2.rgbd_cuda_mesh_ops import (
    split_nonmanifold_vertex_fans_cuda,
)
from behavior_interface_eval_test.tool.official_v2.rgbd_scene_mesh_v12 import (
    _split_nonmanifold_vertex_fans_cpu_reference,
)


class CudaMeshOpsTest(unittest.TestCase):
    def test_corner_components_match_cpu_fan_ordering(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [
                [0, 1, 2],
                [0, 2, 5],
                [0, 3, 4],
            ],
            dtype=np.int64,
        )

        expected = _split_nonmanifold_vertex_fans_cpu_reference(
            points,
            faces,
        )
        actual = split_nonmanifold_vertex_fans_cuda(
            points,
            faces,
            requested_device="cpu",
            allow_cpu_reference=True,
        )

        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(
            actual[2]["split_nonmanifold_vertices"],
            expected[2]["split_nonmanifold_vertices"],
        )
        self.assertEqual(
            actual[2]["added_vertex_fans"],
            expected[2]["added_vertex_fans"],
        )

    def test_runtime_cpu_fallback_is_disabled(self) -> None:
        with self.assertRaisesRegex(
            Exception,
            "CPU V53 mesh operations are disabled",
        ):
            split_nonmanifold_vertex_fans_cuda(
                np.eye(3, dtype=np.float64),
                np.asarray([[0, 1, 2]], dtype=np.int64),
                requested_device="cpu",
                allow_cpu_reference=False,
            )

    def test_random_corner_graphs_match_historical_cpu_output(self) -> None:
        random = np.random.default_rng(20260901)
        for _case in range(20):
            point_count = int(random.integers(6, 24))
            face_count = int(random.integers(2, 48))
            points = random.normal(size=(point_count, 3))
            faces = np.stack(
                [
                    random.choice(point_count, size=3, replace=False)
                    for _ in range(face_count)
                ]
            ).astype(np.int64)
            expected = _split_nonmanifold_vertex_fans_cpu_reference(
                points,
                faces,
            )
            actual = split_nonmanifold_vertex_fans_cuda(
                points,
                faces,
                requested_device="cpu",
                allow_cpu_reference=True,
            )
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            self.assertEqual(
                actual[2]["split_nonmanifold_vertices"],
                expected[2]["split_nonmanifold_vertices"],
            )
            self.assertEqual(
                actual[2]["added_vertex_fans"],
                expected[2]["added_vertex_fans"],
            )


if __name__ == "__main__":
    unittest.main()
