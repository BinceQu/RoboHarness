"""Unit tests for symbolic tracked-point rigid-pose family enumeration."""

from __future__ import annotations

import unittest

import numpy as np

from behavior_interface_eval_test.tool.official_v2.grasp_geometry_local import (
    quat_to_mat_xyzw,
)
from behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local import (
    WHOLE_BODY_ENDPOINT_FRONTIER_CANDIDATES,
    WHOLE_BODY_PRECISION_SHELL_MAX_POSES,
    WHOLE_BODY_PRECISION_SHELL_TRANSLATION_OFFSETS_MM,
    _affine_pose_family_targets,
    _whole_body_endpoint_frontier,
    evaluate_target_constraints,
)


class TrackedPointPoseFamilyTest(unittest.TestCase):
    def test_precision_shell_preserves_cardinals_and_adds_face_diagonals(self) -> None:
        offsets = {
            tuple(float(value) for value in offset)
            for offset in WHOLE_BODY_PRECISION_SHELL_TRANSLATION_OFFSETS_MM
        }
        # Preserve every old candidate so an existing overlap-recovery case
        # cannot lose its successful endpoint.
        for magnitude in (1.0, 2.0, 3.0):
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    expected = [0.0, 0.0, 0.0]
                    expected[axis] = sign * magnitude
                    self.assertIn(tuple(expected), offsets)

        # Crossing a quantized depth boundary can require simultaneous motion
        # in two coordinates.  Cover all signs and all coordinate planes.
        face_diagonals = {
            offset
            for offset in offsets
            if sum(abs(value) > 0.0 for value in offset) == 2
        }
        self.assertEqual(len(face_diagonals), 12)
        self.assertIn((4.0, 0.0, 4.0), face_diagonals)
        self.assertGreaterEqual(
            WHOLE_BODY_PRECISION_SHELL_MAX_POSES,
            len(offsets) + 6,
        )
        self.assertGreaterEqual(
            WHOLE_BODY_ENDPOINT_FRONTIER_CANDIDATES,
            8 + WHOLE_BODY_PRECISION_SHELL_MAX_POSES,
        )

    def test_family_uses_eef_anchor_axis_and_keeps_half_turn(self) -> None:
        # The captured EEF is rotated in the robot-base frame.  The source
        # points therefore are not themselves EEF-frame anchors; using them
        # as a rotation axis would miss the exact half-turn member.
        start_position = np.asarray([0.30, -0.10, 0.40], dtype=np.float64)
        start_rotation = quat_to_mat_xyzw([0.0, 0.0, 0.382683432365, 0.923879532511])
        anchors = np.asarray(
            [
                [0.0, 0.05, 0.02],
                [0.0, -0.05, 0.02],
                [0.0, 0.0, -0.06],
            ],
            dtype=np.float64,
        )
        source = start_position + (start_rotation @ anchors.T).T
        base_position = np.asarray([0.50, 0.20, 0.50], dtype=np.float64)
        # Map the captured anchor line (EEF Y) onto robot-base X, matching the
        # shared-Y target equations below.
        base_rotation = quat_to_mat_xyzw(
            [0.0, 0.0, -0.7071067811865, 0.7071067811865]
        )
        fixed_point = np.asarray([[0.50, 0.20, 0.57]], dtype=np.float64)
        target_points = [
            {"name": "p0", "target_xyz_m": ["a", "y", "z-0.05"]},
            {"name": "p1", "target_xyz_m": ["c", "y", "z-0.05"]},
            {"name": "p2", "target_xyz_m": ["x", "y", "e"]},
        ]
        fixed_targets = [
            {"name": "fixed", "target_xyz_m": ["x", "y", "z"]}
        ]

        poses = _affine_pose_family_targets(
            source_points_robot_base_m=source,
            anchors_eef_m=anchors,
            base_rotation=base_rotation,
            base_position=base_position,
            target_points=target_points,
            fixed_points_robot_base_m=fixed_point,
            fixed_target_points=fixed_targets,
            angular_offsets_deg=(0.0, 180.0),
        )

        self.assertEqual(
            {int(round(float(pose["pose_family_angle_deg"]))) for pose in poses},
            {0, 180},
        )
        for pose in poses:
            rotation = quat_to_mat_xyzw(pose["final_eef_quaternion_xyzw"])
            position = np.asarray(pose["final_eef_position_m"], dtype=np.float64)
            points = position + (rotation @ anchors.T).T
            report = evaluate_target_constraints(
                points,
                target_points,
                tolerance_m=1.0e-7,
                fixed_points_robot_base_m=fixed_point,
                fixed_target_points=fixed_targets,
            )
            self.assertTrue(report["ok"], report)
            self.assertLessEqual(report["max_constraint_error_m"], 1.0e-8)

    def test_family_axis_follows_affine_line_instead_of_longest_pair(self) -> None:
        # At a partially closed asymmetric gripper opening, one fingertip can
        # be farther from the slide centre than from the other fingertip.  The
        # longest geometric pair is then not the wrist-flip axis.  The shared
        # target forms (same y and z, independent x) identify the fingertip
        # line without relying on point names or a particular opening width.
        anchors = np.asarray(
            [
                [0.0, 0.050364, 0.018355],
                [0.0, -0.035685, 0.018355],
                [0.0, 0.0, -0.062],
            ],
            dtype=np.float64,
        )
        self.assertGreater(
            np.linalg.norm(anchors[0] - anchors[2]),
            np.linalg.norm(anchors[0] - anchors[1]),
        )
        start_position = np.asarray([0.30, -0.10, 0.40], dtype=np.float64)
        start_rotation = np.eye(3, dtype=np.float64)
        source = start_position + (start_rotation @ anchors.T).T
        base_rotation = quat_to_mat_xyzw(
            [0.0, 0.0, -0.7071067811865, 0.7071067811865]
        )
        fixed_point = np.asarray([[0.50, 0.20, 0.57]], dtype=np.float64)
        base_position = np.asarray(
            [0.50, 0.20, 0.57 - 0.05 - anchors[0, 2]],
            dtype=np.float64,
        )
        target_points = [
            {"name": "p0", "target_xyz_m": ["a", "y", "z-0.05"]},
            {"name": "p1", "target_xyz_m": ["c", "y", "z-0.05"]},
            {"name": "p2", "target_xyz_m": ["x", "y", "e"]},
        ]
        fixed_targets = [
            {"name": "fixed", "target_xyz_m": ["x", "y", "z"]}
        ]

        poses = _affine_pose_family_targets(
            source_points_robot_base_m=source,
            anchors_eef_m=anchors,
            base_rotation=base_rotation,
            base_position=base_position,
            target_points=target_points,
            fixed_points_robot_base_m=fixed_point,
            fixed_target_points=fixed_targets,
            angular_offsets_deg=(0.0, 180.0),
        )

        self.assertEqual(len(poses), 2)
        self.assertEqual(
            {tuple(pose["pose_family_axis_source_pair"]) for pose in poses},
            {(0, 1)},
        )
        self.assertTrue(
            all(
                pose["pose_family_axis_shared_coordinate_count"] == 2
                for pose in poses
            )
        )
        for pose in poses:
            rotation = quat_to_mat_xyzw(pose["final_eef_quaternion_xyzw"])
            position = np.asarray(pose["final_eef_position_m"], dtype=np.float64)
            points = position + (rotation @ anchors.T).T
            report = evaluate_target_constraints(
                points,
                target_points,
                tolerance_m=1.0e-7,
                fixed_points_robot_base_m=fixed_point,
                fixed_target_points=fixed_targets,
            )
            self.assertTrue(report["ok"], report)

    def test_frontier_retains_family_members_after_primary_candidates(self) -> None:
        def candidate(index: int, *, family: bool) -> tuple:
            selected = np.zeros(10, dtype=np.float64)
            selected[0] = float(index)
            q = np.zeros(7, dtype=np.float64)
            trunk = np.zeros(4, dtype=np.float64)
            report = {
                "mode": (
                    "whole_body_affine_pose_family"
                    if family
                    else "whole_body_geometry_only_direct"
                ),
                "precise": True,
                "accuracy_cost_mm_plus_deg": 0.0,
                "position_error_mm": 0.0,
                "selection_orientation_error_deg": 0.0,
                "motion_score": float(index),
                "pose_family_angle_deg": 180.0 if family else 0.0,
                "pose_family_base_index": 0,
                "pose_family_seed_index": index,
            }
            return selected, q, trunk, report

        candidates = [
            candidate(0, family=False),
            candidate(1, family=False),
            candidate(2, family=True),
            candidate(3, family=True),
        ]
        frontier = _whole_body_endpoint_frontier(
            candidates,
            primary_limit=2,
            frontier_limit=4,
        )
        self.assertEqual(len(frontier), 4)
        self.assertEqual(
            sum(
                report.get("mode") == "whole_body_affine_pose_family"
                for _selected, _q, _trunk, report in frontier
            ),
            2,
        )

    def test_family_accepts_solver_scale_residual_but_rejects_real_conflict(self) -> None:
        """A few micrometres of centre-solve error must not erase a flip.

        The pose-family constructor is fed an endpoint returned by a bounded
        nonlinear solve, rather than an exact symbolic pose.  A tiny rotation
        perturbation therefore produces a small affine least-squares residual.
        It is safe to retain that member because the downstream endpoint
        checker still applies the 3 mm/5 degree precision gate.  A genuinely
        inconsistent pose remains excluded when the residual exceeds the
        construction tolerance.
        """

        anchors = np.asarray(
            [
                [0.0, 0.05, 0.02],
                [0.0, -0.05, 0.02],
                [0.0, 0.0, -0.06],
            ],
            dtype=np.float64,
        )
        source = np.asarray([0.3, -0.1, 0.4]) + anchors
        fixed = np.asarray([[0.5, 0.2, 0.57]], dtype=np.float64)
        target_points = [
            {"name": "p0", "target_xyz_m": ["a", "y", "z-0.05"]},
            {"name": "p1", "target_xyz_m": ["c", "y", "z-0.05"]},
            {"name": "p2", "target_xyz_m": ["x", "y", "e"]},
        ]
        fixed_targets = [
            {"name": "fixed", "target_xyz_m": ["x", "y", "z"]}
        ]
        base_rotation = quat_to_mat_xyzw(
            [0.0, 0.0, -0.7071067811865, 0.7071067811865]
        )

        # A 10 microradian centre error is larger than the old exact-zero
        # check (~0.5 micrometre affine residual) but far below the endpoint
        # precision envelope.
        tiny_error_rotation = np.asarray(
            [
                [0.99999999995, 0.0, 0.00001],
                [0.0, 1.0, 0.0],
                [-0.00001, 0.0, 0.99999999995],
            ],
            dtype=np.float64,
        ) @ base_rotation
        retained = _affine_pose_family_targets(
            source_points_robot_base_m=source,
            anchors_eef_m=anchors,
            base_rotation=tiny_error_rotation,
            base_position=[0.5, 0.2, 0.5],
            target_points=target_points,
            fixed_points_robot_base_m=fixed,
            fixed_target_points=fixed_targets,
            angular_offsets_deg=(0.0, 180.0),
        )
        self.assertEqual(len(retained), 2)
        self.assertTrue(
            all(
                float(pose["affine_equation_error_m"]) <= 1.0e-4
                for pose in retained
            )
        )

        # A 1 mrad perturbation gives roughly 50 micrometres of inconsistency
        # for this bundle and is still a plausible nonlinear endpoint; a 1 cm
        # perturbation is a real conflict for the exact family constructor.
        large_error_rotation = np.asarray(
            [
                [0.99995, 0.0, 0.01],
                [0.0, 1.0, 0.0],
                [-0.01, 0.0, 0.99995],
            ],
            dtype=np.float64,
        ) @ base_rotation
        rejected = _affine_pose_family_targets(
            source_points_robot_base_m=source,
            anchors_eef_m=anchors,
            base_rotation=large_error_rotation,
            base_position=[0.5, 0.2, 0.5],
            target_points=target_points,
            fixed_points_robot_base_m=fixed,
            fixed_target_points=fixed_targets,
            angular_offsets_deg=(0.0, 180.0),
        )
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
