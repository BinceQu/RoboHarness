from __future__ import annotations

import math
import unittest

import numpy as np

from tools.util.geometry import quaternion_angle_deg
from tools.util.x5_kinematics import (
    calibrated_pose_for_joints,
    solve_calibrated_x5_ik,
    transform_to_pose,
    x5_forward_kinematics,
    x5_spatial_jacobian,
)


class X5KinematicsTest(unittest.TestCase):
    def test_zero_configuration_matches_public_x5_chain(self) -> None:
        transform = x5_forward_kinematics(np.zeros(6))
        np.testing.assert_allclose(
            transform[:3, 3],
            [0.09770, -0.00050, 0.15650],
            atol=2e-6,
        )
        np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=2e-6)

    def test_position_jacobian_matches_finite_difference(self) -> None:
        joints = np.array([0.2, -0.3, 0.4, -0.2, 0.1, -0.15])
        analytic = x5_spatial_jacobian(joints)[:3]
        epsilon = 1e-6
        current = x5_forward_kinematics(joints)[:3, 3]
        numeric = np.zeros((3, 6), dtype=np.float64)
        for index in range(6):
            perturbed = joints.copy()
            perturbed[index] += epsilon
            numeric[:, index] = (
                x5_forward_kinematics(perturbed)[:3, 3] - current
            ) / epsilon
        np.testing.assert_allclose(analytic, numeric, atol=2e-6)

    def test_observation_calibrated_ik_reaches_cartesian_target(self) -> None:
        current_joints = np.array(
            [0.00025, 0.35144, -0.03644, 0.47016, -0.00006, 0.0]
        )
        yaw = math.pi / 2.0
        world_from_model = np.eye(4, dtype=np.float64)
        world_from_model[:3, :3] = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        world_from_model[:3, 3] = [0.3, -0.45, 0.765]
        current_pose = transform_to_pose(
            world_from_model @ x5_forward_kinematics(current_joints)
        )
        target_pose = current_pose.copy()
        target_pose[:3] += [0.0, 0.015, -0.025980762]

        solution = solve_calibrated_x5_ik(
            current_joints,
            current_pose,
            target_pose,
        )
        predicted = calibrated_pose_for_joints(
            current_joints,
            current_pose,
            solution.joint_target,
        )

        self.assertTrue(solution.converged, solution)
        self.assertLessEqual(
            float(np.linalg.norm(predicted[:3] - target_pose[:3])), 0.0006
        )
        self.assertLessEqual(
            quaternion_angle_deg(predicted[3:], target_pose[3:]), 0.16
        )
        np.testing.assert_allclose(
            solution.calibration_translation_m,
            [0.3, -0.45, 0.765],
            atol=1e-8,
        )

    def test_paired_ik_can_warm_start_without_changing_calibration(self) -> None:
        current_joints = np.zeros(6, dtype=np.float64)
        current_pose = np.array([0.25, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0])
        target_joints = np.array([0.08, -0.15, 0.12, 0.04, -0.06, 0.03])
        target_pose = calibrated_pose_for_joints(
            current_joints,
            current_pose,
            target_joints,
        )
        solution = solve_calibrated_x5_ik(
            current_joints,
            current_pose,
            target_pose,
            initial_joints=target_joints,
        )
        self.assertTrue(solution.converged, solution)
        self.assertEqual(solution.iterations, 1)
        np.testing.assert_allclose(solution.joint_target, target_joints, atol=1e-12)
        np.testing.assert_allclose(
            solution.calibration_translation_m,
            [0.1523, 0.0005, 0.7435],
            atol=2e-6,
        )


if __name__ == "__main__":
    unittest.main()
