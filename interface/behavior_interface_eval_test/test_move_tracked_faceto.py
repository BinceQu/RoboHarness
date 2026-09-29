"""Interface-free regression tests for the camera-facing move preset."""

from __future__ import annotations

import math
import unittest

import numpy as np

from behavior_interface_eval_test.tool.official_v2.contract import (
    validate_move_tracked_point_args,
)
from behavior_interface_eval_test.tool.official_v2.grasp_geometry_local import (
    quat_to_mat_xyzw,
)
from behavior_interface_eval_test.tool.official_v2.tracked_point_constraints_local import (
    evaluate_quick_constraint,
    quick_constraint_relation,
)
from behavior_interface_eval_test.tool.official_v2 import tools as official_tools


class MoveTrackedFacetoTest(unittest.TestCase):
    camera_position = np.zeros(3, dtype=np.float64)
    camera_quaternion = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    @staticmethod
    def request(
        quick_type: str = "faceto",
        *,
        names: tuple[str, str, str] = ("p1", "p2", "p3"),
        roles: tuple[str, str, str] = ("on_hand", "on_hand", "on_hand"),
    ) -> dict:
        points = [
            {"name": name, "role": role}
            for name, role in zip(names, roles)
        ]
        return {
            "mode": "quick_constraint",
            "points": points,
            "quick_constraint": {
                "type": quick_type,
                "on_hand_points": [
                    name for name, role in zip(names, roles) if role == "on_hand"
                ],
                "off_hand_points": [
                    name for name, role in zip(names, roles) if role == "off_hand"
                ],
            },
        }

    def identity_triangle(self) -> np.ndarray:
        # In the optical camera convention (+u right, +v down, -Z forward),
        # this row order is clockwise in the displayed image and has a -Z
        # outward normal.
        return np.asarray(
            [
                [-0.1, 0.1, -1.0],
                [0.1, 0.1, -1.0],
                [0.1, -0.1, -1.0],
            ],
            dtype=np.float64,
        )

    def evaluate(self, quick_type: str, points: np.ndarray) -> dict:
        return evaluate_quick_constraint(
            points,
            None,
            {
                "type": quick_type,
                "on_hand_points": ["p1", "p2", "p3"],
                "off_hand_points": [],
            },
            position_tolerance_m=0.003,
            orientation_tolerance_deg=5.0,
            camera_position_robot_base_m=self.camera_position,
            camera_quaternion_xyzw=self.camera_quaternion,
        )

    def test_contract_accepts_faceto_and_aliases(self) -> None:
        for alias, canonical in (
            ("faceto", "faceto"),
            ("face-to", "faceto"),
            ("face_to", "faceto"),
            ("reverse_faceto", "reverse_faceto"),
            ("reverse-face-to", "reverse_faceto"),
            ("reverse face to", "reverse_faceto"),
        ):
            with self.subTest(alias=alias):
                normalized = validate_move_tracked_point_args(self.request(alias))
                self.assertEqual(normalized["quick_constraint"]["type"], canonical)
                self.assertEqual(
                    normalized["quick_constraint"]["on_hand_points"],
                    ["p1", "p2", "p3"],
                )
                self.assertEqual(normalized["quick_constraint"]["off_hand_points"], [])

    def test_contract_requires_three_ordered_on_hand_points(self) -> None:
        for request in (
            self.request(names=("p1", "p2", "p3"), roles=("on_hand", "on_hand", "off_hand")),
            self.request(names=("p1", "p2"), roles=("on_hand", "on_hand", "on_hand")),
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_move_tracked_point_args(request)

        reordered = self.request()
        reordered["quick_constraint"]["on_hand_points"] = ["p3", "p2", "p1"]
        with self.assertRaises(ValueError):
            validate_move_tracked_point_args(reordered)

    def test_faceto_relation_and_live_metrics_are_clockwise(self) -> None:
        points = self.identity_triangle()
        relation = quick_constraint_relation(
            {
                "type": "faceto",
                "on_hand_points": ["p1", "p2", "p3"],
                "off_hand_points": [],
            },
            on_hand_points_robot_base_m=points,
            camera_position_robot_base_m=self.camera_position,
            camera_quaternion_xyzw=self.camera_quaternion,
        )
        self.assertEqual(relation["type"], "oriented_plane_normal")
        np.testing.assert_allclose(relation["normal_robot_base"], [0.0, 0.0, -1.0])
        self.assertEqual(relation["mode"], "same")
        report = self.evaluate("faceto", points)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["projected_winding"], "clockwise")
        self.assertTrue(report["winding_ok"])
        self.assertAlmostEqual(report["acute_normal_camera_angle_deg"], 0.0)

    def test_reverse_faceto_requires_counterclockwise_and_opposite_normal(self) -> None:
        points = self.identity_triangle()[[0, 2, 1]]
        relation = quick_constraint_relation(
            {
                "type": "reverse_faceto",
                "on_hand_points": ["p1", "p2", "p3"],
                "off_hand_points": [],
            },
            on_hand_points_robot_base_m=points,
            camera_position_robot_base_m=self.camera_position,
            camera_quaternion_xyzw=self.camera_quaternion,
        )
        np.testing.assert_allclose(relation["normal_robot_base"], [0.0, 0.0, 1.0])
        report = self.evaluate("reverse_faceto", points)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["projected_winding"], "counterclockwise")
        self.assertTrue(report["winding_ok"])

    def test_wrong_winding_is_rejected_even_when_acute_angle_is_zero(self) -> None:
        # Reversing the rows flips the signed normal but leaves the acute angle
        # unchanged.  Winding is therefore an independent acceptance gate.
        report = self.evaluate("faceto", self.identity_triangle()[[0, 2, 1]])
        self.assertFalse(report["ok"], report)
        self.assertEqual(report["error"], "faceto_winding_not_satisfied")
        self.assertAlmostEqual(report["acute_normal_camera_angle_deg"], 0.0)

    def test_camera_rotation_is_used_for_normal_and_projection(self) -> None:
        angle = math.pi / 2.0
        camera_quaternion = np.asarray(
            [0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0)],
            dtype=np.float64,
        )
        camera_rotation = quat_to_mat_xyzw(camera_quaternion)
        camera_points = self.identity_triangle()
        robot_points = (camera_rotation @ camera_points.T).T
        relation = quick_constraint_relation(
            {
                "type": "faceto",
                "on_hand_points": ["p1", "p2", "p3"],
                "off_hand_points": [],
            },
            on_hand_points_robot_base_m=robot_points,
            camera_position_robot_base_m=[0.0, 0.0, 0.0],
            camera_quaternion_xyzw=camera_quaternion,
        )
        np.testing.assert_allclose(relation["normal_robot_base"], [-1.0, 0.0, 0.0], atol=1e-12)
        report = evaluate_quick_constraint(
            robot_points,
            None,
            {"type": "faceto", "on_hand_points": ["p1", "p2", "p3"], "off_hand_points": []},
            camera_position_robot_base_m=[0.0, 0.0, 0.0],
            camera_quaternion_xyzw=camera_quaternion,
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["projected_winding"], "clockwise")

    def test_missing_camera_and_degenerate_triangle_fail_closed(self) -> None:
        quick = {
            "type": "faceto",
            "on_hand_points": ["p1", "p2", "p3"],
            "off_hand_points": [],
        }
        with self.assertRaises(ValueError):
            quick_constraint_relation(
                quick,
                on_hand_points_robot_base_m=self.identity_triangle(),
            )
        missing = evaluate_quick_constraint(
            self.identity_triangle(),
            None,
            quick,
        )
        self.assertFalse(missing["ok"])
        self.assertEqual(missing["error"], "faceto_camera_pose_unavailable")
        degenerate = evaluate_quick_constraint(
            np.asarray([[0.0, 0.0, -1.0], [0.1, 0.0, -1.0], [0.2, 0.0, -1.0]]),
            None,
            quick,
            camera_position_robot_base_m=self.camera_position,
            camera_quaternion_xyzw=self.camera_quaternion,
        )
        self.assertFalse(degenerate["ok"])
        self.assertEqual(degenerate["error"], "faceto_triangle_degenerate")

    def test_official_tool_wrapper_preserves_camera_winding_gate(self) -> None:
        """The execution-facing aggregate must pass the frozen camera through."""

        points = self.identity_triangle()
        quick = {
            "type": "faceto",
            "on_hand_points": ["p1", "p2", "p3"],
            "off_hand_points": [],
        }
        report = official_tools._move_tracked_evaluate_quick_constraints(
            points,
            None,
            [quick],
            controlled_names=["p1", "p2", "p3"],
            reference_names=[],
            position_tolerance_m=0.003,
            orientation_tolerance_deg=5.0,
            camera_position_robot_base_m=self.camera_position,
            camera_quaternion_xyzw=self.camera_quaternion,
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["group_count"], 1)
        self.assertEqual(report["groups"][0]["projected_winding"], "clockwise")


if __name__ == "__main__":
    unittest.main()
