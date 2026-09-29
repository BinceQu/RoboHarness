from __future__ import annotations

import math
import unittest

import numpy as np

from behavior_interface_eval_test.official_action_world import (
    ObservationBackedActionWorld,
)
from behavior_interface_eval_test.tool.official_v2.tools import (
    _camera_intrinsics,
    _point_from_current_depth_robot_base,
    _point_in_robot_frame,
    _r1pro_chest_pose_robot,
    _r1pro_pitch_for_aligned_target,
    _r1pro_shoulder_positions_robot,
    _unproject_pixel,
)


class OfficialGeometryAccuracyTest(unittest.TestCase):
    @staticmethod
    def _quat_to_matrix(quat) -> np.ndarray:
        x, y, z, w = np.asarray(quat, dtype=np.float64)
        x, y, z, w = np.asarray([x, y, z, w]) / np.linalg.norm(quat)
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _yaw_matrix(yaw_deg: float) -> np.ndarray:
        yaw = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        return np.array(
            [
                [cos_y, -sin_y, 0.0],
                [sin_y, cos_y, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def test_intrinsics_match_official_principal_points(self) -> None:
        head = _camera_intrinsics("head", 720, 720)
        wrist = _camera_intrinsics("left_wrist", 480, 480)

        self.assertEqual((head["fx"], head["fy"]), (306.0, 306.0))
        self.assertEqual((head["cx"], head["cy"]), (360.0, 360.0))
        self.assertAlmostEqual(wrist["fx"], 388.6639, places=6)
        self.assertEqual((wrist["cx"], wrist["cy"]), (240.0, 240.0))

    def test_5021_depth_pose_matches_direct_simulator_oracle(self) -> None:
        # Frozen from 5021 img_0622. Simulator values are test-only oracle
        # data; the production calculation below receives only depth and the
        # evaluator-supplied camera pose relative to robot base.
        depth = np.full((720, 720), np.nan, dtype=np.float32)
        depth[362, 523] = np.float32(1.6056783199310303)
        camera_relative_pose = {
            "pos": [
                -0.0033800285623804918,
                0.0024444942770438685,
                1.6149000525474548,
            ],
            "quat": [
                0.40509029142836034,
                -0.40596543345638814,
                -0.5789574050998321,
                0.5795707426269221,
            ],
        }
        direct_target_robot = np.array(
            [1.501775292426781, -0.8537109559697593, 1.056888022274115],
            dtype=np.float64,
        )

        point, meta = _point_from_current_depth_robot_base(
            depth=depth,
            u=728,
            v=504,
            role="head",
            camera_relative_pose=camera_relative_pose,
        )

        self.assertEqual(meta["pixel_uv"], [523, 362])
        self.assertEqual(meta["frame"], "robot_base")
        self.assertLess(float(np.linalg.norm(point - direct_target_robot)), 5e-5)

    def test_old_half_pixel_principal_point_explains_5021_error(self) -> None:
        relative = {
            "pos": [
                -0.0033800285623804918,
                0.0024444942770438685,
                1.6149000525474548,
            ],
            "quat": [
                0.40509029142836034,
                -0.40596543345638814,
                -0.5789574050998321,
                0.5795707426269221,
            ],
        }
        direct_target_robot = np.array(
            [1.501775292426781, -0.8537109559697593, 1.056888022274115],
            dtype=np.float64,
        )
        old = _unproject_pixel(
            px=523,
            py=362,
            depth_m=1.6056783199310303,
            camera={
                **relative,
                "fx": 306.0,
                "fy": 306.0,
                "cx": 359.5,
                "cy": 359.5,
            },
        )

        self.assertGreater(float(np.linalg.norm(old - direct_target_robot)), 0.003)

    def test_current_observation_geometry_ignores_false_command_odometry(self) -> None:
        world = ObservationBackedActionWorld(
            proprio_provider=lambda: np.zeros(61, dtype=np.float32),
            hold_action_provider=lambda: np.zeros(23, dtype=np.float32),
            eef_pose_provider=lambda: {},
        )
        world._mock_base = np.array([0.2, 0.0, 0.0], dtype=np.float64)
        stale_local_target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        stale_robot = _point_in_robot_frame(world, stale_local_target)

        depth = np.full((720, 720), np.nan, dtype=np.float32)
        depth[360, 513] = np.float32(1.0)
        current_robot, _ = _point_from_current_depth_robot_base(
            depth=depth,
            u=714,
            v=501,
            role="head",
            camera_relative_pose={
                "pos": [0.0, 0.0, 0.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
            },
        )

        self.assertAlmostEqual(float(stale_robot[0]), 0.8, places=8)
        self.assertGreater(abs(float(current_robot[0]) - float(stale_robot[0])), 0.2)

    def test_upright_shoulder_fk_matches_5021_direct_chest_oracle(self) -> None:
        trunk_q = np.array([0.0, 0.001, 0.0, 0.0], dtype=np.float64)
        shoulders = _r1pro_shoulder_positions_robot(trunk_q)
        shoulder_mid = 0.5 * (shoulders["left"] + shoulders["right"])

        direct_chest_robot = np.array(
            [-0.07835424, 0.00172660, 1.14235157],
            dtype=np.float64,
        )
        direct_pitch = math.atan2(0.0001410979, 0.99999999)
        direct_rotation = np.array(
            [
                [math.cos(direct_pitch), 0.0, math.sin(direct_pitch)],
                [0.0, 1.0, 0.0],
                [-math.sin(direct_pitch), 0.0, math.cos(direct_pitch)],
            ],
            dtype=np.float64,
        )
        direct_shoulder_mid = direct_chest_robot + direct_rotation @ np.array(
            [-0.00048618, 0.0, 0.30302],
            dtype=np.float64,
        )

        self.assertLess(
            float(np.linalg.norm(shoulder_mid - direct_shoulder_mid)),
            0.003,
        )

    def test_pitched_urdf_fk_matches_5021_direct_simulator_oracles(self) -> None:
        # The fixed chest-to-camera rotation is calibrated once from the
        # upright frame. Each pitched sample then derives the full base
        # orientation from the direct camera orientation, independently of
        # the torso translation being tested.
        upright_trunk = np.array([0.0, 0.001, 0.0, 0.0])
        _, upright_rotation = _r1pro_chest_pose_robot(upright_trunk)
        upright_base_rotation = self._yaw_matrix(72.71617454974611)
        upright_camera_rotation = self._quat_to_matrix(
            [
                0.5668988823890686,
                -0.08678554743528366,
                -0.12266353517770767,
                0.8099676370620728,
            ]
        )
        chest_to_camera_rotation = (
            upright_base_rotation @ upright_rotation
        ).T @ upright_camera_rotation

        samples = [
            {
                "q3": -math.radians(30.0),
                "base": [5.663436412811279, -0.5029170513153076, 0.011310398578643799],
                "chest": [5.600672721862793, -0.5120795965194702, 1.1388803720474243],
                "camera_quat": [
                    0.31944215297698975,
                    -0.1458464413881302,
                    -0.34361177682876587,
                    0.8709859251976013,
                ],
                "forward": [
                    0.6178878399617969,
                    0.6201871659212318,
                    -0.48330373105732954,
                ],
            },
            {
                "q3": -math.radians(50.0),
                "base": [5.661981582641602, -0.5038514137268066, 0.007931709289550781],
                "chest": [5.641071796417236, -0.5002009868621826, 1.1144702434539795],
                "camera_quat": [
                    0.16216082870960236,
                    -0.07341031730175018,
                    -0.3645785450935364,
                    0.9140006899833679,
                ],
                "forward": [
                    0.4570241501419725,
                    0.46138893894723737,
                    -0.7604269676991671,
                ],
            },
        ]
        for sample in samples:
            trunk_q = np.array([0.0, 0.001, sample["q3"], 0.0])
            model_position, model_rotation = _r1pro_chest_pose_robot(trunk_q)
            camera_world_rotation = self._quat_to_matrix(
                sample["camera_quat"]
            )
            chest_world_rotation = (
                camera_world_rotation @ chest_to_camera_rotation.T
            )
            base_world_rotation = (
                chest_world_rotation @ model_rotation.T
            )
            direct_position = base_world_rotation.T @ (
                np.asarray(sample["chest"]) - np.asarray(sample["base"])
            )
            direct_forward = base_world_rotation.T @ np.asarray(
                sample["forward"]
            )

            self.assertLess(
                float(np.linalg.norm(model_position - direct_position)),
                0.002,
            )
            forward_error = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            np.dot(model_rotation[:, 0], direct_forward),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            self.assertLess(forward_error, 0.06)

    def test_pitch_solver_accounts_for_moving_shoulder_midpoint(self) -> None:
        expected_pitch = math.radians(37.0)
        q1, q2 = 0.12, -0.08
        q3 = q1 + q2 - expected_pitch
        q = np.array([q1, q2, q3, 0.0], dtype=np.float64)
        shoulders = _r1pro_shoulder_positions_robot(q)
        shoulder_mid = 0.5 * (shoulders["left"] + shoulders["right"])
        _, rotation = _r1pro_chest_pose_robot(q)
        target = shoulder_mid + 0.85 * rotation[:, 0]

        solution = _r1pro_pitch_for_aligned_target(
            target,
            np.array([q1, q2, 0.0, 0.0], dtype=np.float64),
        )

        self.assertAlmostEqual(
            solution["chest_pitch_deg"],
            37.0,
            places=6,
        )
        self.assertLess(solution["ray_residual_m"], 1e-6)


if __name__ == "__main__":
    unittest.main()
