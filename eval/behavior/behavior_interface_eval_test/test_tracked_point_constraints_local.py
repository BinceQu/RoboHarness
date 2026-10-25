from __future__ import annotations

import unittest

import numpy as np

from behavior_interface_eval_test.tool.official_v2.contract import (
    validate_move_tracked_point_args,
)
from behavior_interface_eval_test.tool.official_v2.tracked_point_constraints_local import (
    COLLINEAR_POSITION_TOLERANCE_M,
    TrackedPointConstraintSet,
    evaluate_relations,
    geometry_preflight,
    geometry_rank,
    geometry_report_difference,
    kabsch_rigid_transform,
    rigid_geometry_report,
)
from behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local import (
    evaluate_target_constraints,
)


class TrackedPointConstraintsLocalTest(unittest.TestCase):
    @staticmethod
    def _source_points(count: int) -> np.ndarray:
        points = np.asarray(
            [
                [0.00, 0.00, 0.00],
                [0.10, 0.00, 0.00],
                [0.00, 0.08, 0.00],
                [0.00, 0.00, 0.07],
            ],
            dtype=np.float64,
        )
        return points[:count].copy()

    @staticmethod
    def _rotation() -> np.ndarray:
        axis = np.asarray([0.3, -0.4, 0.8], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        skew = np.asarray(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        angle = 0.37
        return (
            np.eye(3)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew)
        )

    def test_geometry_report_scales_from_three_to_four_points(self) -> None:
        source3 = self._source_points(3)
        source4 = self._source_points(4)
        self.assertEqual(geometry_rank(source3), 2)
        self.assertEqual(geometry_rank(source4), 3)
        report3 = rigid_geometry_report(source3, source3 + [0.02, -0.01, 0.03])
        report4 = rigid_geometry_report(source4, source4 + [0.02, -0.01, 0.03])
        self.assertEqual(len(report3["pair_distances"]), 3)
        self.assertEqual(len(report3["triangle_areas"]), 1)
        self.assertEqual(len(report4["pair_distances"]), 6)
        self.assertEqual(len(report4["triangle_areas"]), 4)
        self.assertEqual(len(report4["tetrahedron_volumes"]), 1)
        self.assertIsNone(geometry_report_difference(report4, report4))

    def test_kabsch_and_preflight_accept_a_proper_four_point_transform(self) -> None:
        source = self._source_points(4)
        target = (self._rotation() @ source.T).T + [0.21, -0.12, 0.31]
        result = kabsch_rigid_transform(source, target)
        self.assertLess(result["max_error_m"], 1.0e-10)
        self.assertTrue(result["proper_rotation"])
        self.assertFalse(result["reflection_detected"])
        preflight = geometry_preflight(source, target, tolerance_m=0.012)
        self.assertTrue(preflight["ok"], preflight)

    def test_preflight_rejects_an_explicit_four_point_mirror(self) -> None:
        source = self._source_points(4)
        target = source.copy()
        target[:, 0] *= -1.0
        preflight = geometry_preflight(source, target, tolerance_m=0.012)
        self.assertFalse(preflight["ok"], preflight)
        self.assertIn("improper mirror", preflight["reason"])

    def test_typed_plane_and_vector_relations_use_one_evaluator(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.40], [0.10, 0.0, 0.40], [0.0, 0.08, 0.40]],
            dtype=np.float64,
        )
        relations = [
            {
                "type": "common_plane",
                "point_names": ["p1", "p2", "p3"],
                "normal_robot_base": [0.0, 0.0, 1.0],
                "offset_m": {"free": True},
            },
            {
                "type": "oriented_plane_normal",
                "point_names": ["p1", "p2", "p3"],
                "normal_robot_base": [0.0, 0.0, 1.0],
                "mode": "same",
            },
            {
                "type": "align_vector",
                "point_names": ["p1", "p2"],
                "direction_robot_base": [1.0, 0.0, 0.0],
                "mode": "same",
            },
        ]
        evaluated = evaluate_relations(
            points,
            relations,
            point_names=["p1", "p2", "p3"],
            position_tolerance_m=1.0e-5,
            orientation_tolerance_deg=0.1,
        )
        self.assertTrue(evaluated["ok"], evaluated)
        moved = points.copy()
        moved[2, 2] += 0.01
        self.assertFalse(
            evaluate_relations(
                moved,
                relations,
                point_names=["p1", "p2", "p3"],
                position_tolerance_m=1.0e-5,
                orientation_tolerance_deg=0.1,
            )["ok"]
        )

    def test_collinear_relation_tolerance_is_one_mm_only_for_line_relations(self) -> None:
        relation = {
            "type": "line_segment_contains",
            "point_names": ["held_a", "held_b"],
            "segment_start_robot_base_m": [0.05, 0.0, 0.0],
            "segment_end_robot_base_m": [0.15, 0.0, 0.0],
        }
        names = ["held_a", "held_b"]
        within = np.asarray(
            [[0.0, 0.0008, 0.0], [0.2, 0.0008, 0.0]], dtype=np.float64
        )
        outside = np.asarray(
            [[0.0, 0.0012, 0.0], [0.2, 0.0012, 0.0]], dtype=np.float64
        )
        within_report = evaluate_relations(
            within,
            [relation],
            point_names=names,
            position_tolerance_m=0.03,
            orientation_tolerance_deg=20.0,
        )
        outside_report = evaluate_relations(
            outside,
            [relation],
            point_names=names,
            position_tolerance_m=0.03,
            orientation_tolerance_deg=20.0,
        )
        self.assertTrue(within_report["ok"], within_report)
        self.assertFalse(outside_report["ok"], outside_report)
        self.assertAlmostEqual(
            outside_report["collinear_position_tolerance_m"],
            COLLINEAR_POSITION_TOLERANCE_M,
        )
        self.assertGreater(outside_report["max_collinear_error_m"], 0.001)

        target_points = [
            {"name": name, "target_xyz_m": ["?", "?", "?"]}
            for name in names
        ]
        target_report = evaluate_target_constraints(
            outside,
            target_points,
            tolerance_m=0.03,
            relations=[relation],
            orientation_tolerance_deg=20.0,
        )
        self.assertFalse(target_report["ok"], target_report)
        constraint_set = TrackedPointConstraintSet(
            point_names=tuple(names),
            target_points=tuple(target_points),
            relations=(relation,),
        )
        set_report = constraint_set.evaluate(
            outside,
            position_tolerance_m=0.03,
            orientation_tolerance_deg=20.0,
        )
        self.assertFalse(set_report["ok"], set_report)
        self.assertAlmostEqual(
            set_report["collinear_position_tolerance_m"],
            COLLINEAR_POSITION_TOLERANCE_M,
        )

        point_relation = {
            "type": "point_at_position",
            "point_names": ["held_a"],
            "target_position_robot_base_m": [0.0, 0.0, 0.0],
        }
        non_collinear = evaluate_relations(
            [[0.0, 0.0012, 0.0]],
            [point_relation],
            point_names=["held_a"],
            position_tolerance_m=0.03,
        )
        self.assertTrue(non_collinear["ok"], non_collinear)
        self.assertIsNone(non_collinear["collinear_position_tolerance_m"])

    def test_constraint_set_resolves_repeated_affine_variables(self) -> None:
        constraint_set = TrackedPointConstraintSet(
            point_names=("a", "b", "c"),
            target_points=(
                {"name": "a", "target_xyz_m": ["shared_x", "h", "?"]},
                {"name": "b", "target_xyz_m": ["shared_x", "h", "zb"]},
                {"name": "c", "target_xyz_m": ["shared_x+0.02", "h", "zc"]},
            ),
        )
        measured = np.asarray(
            [[0.30, 0.45, 0.10], [0.30, 0.45, 0.20], [0.32, 0.45, 0.25]],
            dtype=np.float64,
        )
        resolved = constraint_set.resolve_variables(measured)
        self.assertAlmostEqual(resolved["shared_x"], 0.30, places=8)
        self.assertAlmostEqual(resolved["h"], 0.45, places=8)
        self.assertAlmostEqual(resolved["zb"], 0.20, places=8)
        self.assertAlmostEqual(resolved["zc"], 0.25, places=8)

    def test_on_and_off_hand_coordinates_share_one_affine_variable_model(self) -> None:
        on_target = [
            {
                "name": "held",
                "target_xyz_m": ["contact_x", "contact_y", "contact_z"],
            }
        ]
        off_target = [
            {
                "name": "lock",
                "target_xyz_m": ["contact_x", "contact_y", "contact_z"],
            }
        ]
        satisfied = evaluate_target_constraints(
            [[0.2, -0.1, 0.4]],
            on_target,
            tolerance_m=0.001,
            fixed_points_robot_base_m=[[0.2, -0.1, 0.4]],
            fixed_target_points=off_target,
        )
        self.assertTrue(satisfied["ok"], satisfied)
        self.assertEqual(
            set(satisfied["resolved_variables"]),
            {"contact_x", "contact_y", "contact_z"},
        )

        violated = evaluate_target_constraints(
            [[0.24, -0.1, 0.4]],
            on_target,
            tolerance_m=0.005,
            fixed_points_robot_base_m=[[0.2, -0.1, 0.4]],
            fixed_target_points=off_target,
        )
        self.assertFalse(violated["ok"], violated)
        self.assertGreater(violated["max_coordinate_constraint_error_m"], 0.019)

    def test_constraint_set_defaults_and_strict_point_binding(self) -> None:
        points = np.asarray(
            [[0.0, 0.0, 0.4], [0.1, 0.0, 0.4], [0.0, 0.1, 0.4]],
            dtype=np.float64,
        )
        constraint_set = TrackedPointConstraintSet(
            point_names=("a", "b", "c"),
            relations=(
                {
                    "type": "common_plane",
                    "point_names": ["a", "b", "c"],
                    "normal": [0.0, 0.0, 1.0],
                    "offset": {"free": True},
                },
            ),
        )
        self.assertTrue(constraint_set.evaluate(points)["ok"])
        self.assertTrue(
            constraint_set.geometry_preflight(points)["ok"]
        )
        with self.assertRaisesRegex(ValueError, "point_names count"):
            constraint_set.evaluate(points[:2])
        with self.assertRaisesRegex(ValueError, "point_names count"):
            constraint_set.geometry_preflight(points[:2])

    def test_target_constraint_evaluation_rejects_extra_points(self) -> None:
        from behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local import (
            evaluate_target_constraints,
        )

        target = [
            {"name": "a", "target_xyz_m": [0.0, 0.0, 0.0]},
            {"name": "b", "target_xyz_m": [0.1, 0.0, 0.0]},
            {"name": "c", "target_xyz_m": [0.0, 0.1, 0.0]},
        ]
        with self.assertRaisesRegex(ValueError, "point counts"):
            evaluate_target_constraints(
                np.vstack([self._source_points(3), [[0.2, 0.2, 0.0]]]),
                target,
                tolerance_m=0.012,
            )

    def test_constraint_set_evaluate_combines_affine_and_typed_relations(self) -> None:
        target_points = (
            {"name": "p1", "target_xyz_m": ["x", "h", "z1"]},
            {"name": "p2", "target_xyz_m": ["x", "h", "z2"]},
            {"name": "p3", "target_xyz_m": ["x+0.10", "h", "z3"]},
        )
        constraint_set = TrackedPointConstraintSet(
            point_names=("p1", "p2", "p3"),
            target_points=target_points,
            relations=(
                {
                    "type": "common_plane",
                    "point_names": ["p1", "p2", "p3"],
                    "normal_robot_base": [0.0, 0.0, 1.0],
                    "offset_m": {"free": True},
                },
            ),
        )
        points = np.asarray(
            [[0.20, 0.40, 0.30], [0.20, 0.40, 0.30], [0.30, 0.40, 0.30]],
            dtype=np.float64,
        )
        report = constraint_set.evaluate(
            points,
            position_tolerance_m=1.0e-8,
            orientation_tolerance_deg=1.0,
        )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["point_count"], 3)
        self.assertIn("x", report["resolved_variables"])
        self.assertIn("h", report["resolved_variables"])
        self.assertTrue(
            any(block["type"] == "affine_coordinates" for block in report["residual_blocks"])
        )
        moved = points.copy()
        moved[2, 0] += 0.02
        failed = constraint_set.evaluate(moved, position_tolerance_m=1.0e-3)
        self.assertFalse(failed["ok"], failed)
        self.assertGreater(failed["max_coordinate_constraint_error_m"], 1.0e-3)

    def test_relation_and_geometry_apis_reject_bad_cardinality_and_flattened_arrays(self) -> None:
        points = self._source_points(3)
        with self.assertRaisesRegex(ValueError, "requires exactly 3 points"):
            evaluate_relations(
                points,
                [
                    {
                        "type": "oriented_plane_normal",
                        "point_names": ["p1", "p2"],
                        "normal_robot_base": [0.0, 0.0, 1.0],
                    }
                ],
                point_names=["p1", "p2", "p3"],
            )
        with self.assertRaisesRegex(ValueError, "N x 3"):
            geometry_rank(points.reshape(-1))

    def test_constraint_set_rejects_ambiguous_point_bindings(self) -> None:
        with self.assertRaisesRegex(ValueError, "point_names must be unique"):
            TrackedPointConstraintSet(point_names=("a", "a"))
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            TrackedPointConstraintSet(point_names=("a", ""))
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            TrackedPointConstraintSet(point_names=("a", 2))
        with self.assertRaisesRegex(ValueError, "one to 6"):
            TrackedPointConstraintSet(point_names=())

    def test_contract_rejects_conflicting_relation_aliases(self) -> None:
        points = [
            {"name": name, "target_xyz_m": ["x", "y", "z"]}
            for name in ("p1", "p2", "p3")
        ]
        with self.assertRaisesRegex(ValueError, "only one of"):
            validate_move_tracked_point_args(
                {
                    "points": points,
                    "relations": [
                        {
                            "type": "common_plane",
                            "point_names": ["p1", "p2", "p3"],
                            "normal_robot_base": [0, 0, 1],
                            "normal": [0, 0, 1],
                        }
                    ],
                }
            )

        with self.assertRaisesRegex(ValueError, "only one of"):
            validate_move_tracked_point_args(
                {
                    "points": points,
                    "relations": [
                        {
                            "type": "align_vector",
                            "point_names": ["p1", "p2"],
                            "direction_robot_base": [1, 0, 0],
                            "mode": "same",
                            "sense": "same",
                        }
                    ],
                }
            )
        with self.assertRaisesRegex(ValueError, "unsupported fields.*mode"):
            validate_move_tracked_point_args(
                {
                    "points": points,
                    "relations": [
                        {
                            "type": "common_plane",
                            "point_names": ["p1", "p2", "p3"],
                            "normal_robot_base": [0, 0, 1],
                            "offset_m": {"free": True},
                            "mode": "same",
                        }
                    ],
                }
            )

    def test_contract_and_relation_api_reject_non_string_marker_names(self) -> None:
        points = [
            {"name": name, "target_xyz_m": ["x", "y", "z"]}
            for name in ("p1", "p2", "p3")
        ]
        malformed_points = [dict(points[0], name=1), *points[1:]]
        with self.assertRaisesRegex(ValueError, "name must be a non-empty string"):
            validate_move_tracked_point_args({"points": malformed_points})

        with self.assertRaisesRegex(ValueError, "point_names entries must be non-empty strings"):
            validate_move_tracked_point_args(
                {
                    "points": points,
                    "relations": [
                        {
                            "type": "common_plane",
                            "point_names": ["p1", 2, "p3"],
                            "normal_robot_base": [0, 0, 1],
                        }
                    ],
                }
            )

        with self.assertRaisesRegex(ValueError, "point_names must contain only non-empty strings"):
            evaluate_relations(
                np.asarray(
                    [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
                    dtype=np.float64,
                ),
                [],
                point_names=["p1", 2, "p3"],
            )

    def test_default_geometry_preflight_and_variable_json_types_are_strict(self) -> None:
        source = self._source_points(4)
        # The local API has the same default public position tolerance as the
        # motion tool; callers do not need to duplicate it for a basic check.
        self.assertTrue(geometry_preflight(source, source)["ok"])
        with self.assertRaisesRegex(ValueError, r"\.var must be a string"):
            validate_move_tracked_point_args(
                {
                    "points": [
                        {
                            "name": "p1",
                            "target_xyz_m": [
                                {"var": True},
                                "y",
                                "z",
                            ],
                        }
                    ]
                }
            )

    def test_n_point_kabsch_bundle_rejects_one_redundant_marker_drift(self) -> None:
        from behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local import (
            evaluate_rigid_geometry_consistency,
        )

        source = self._source_points(4)
        target = (self._rotation() @ source.T).T + [0.20, -0.10, 0.30]
        target[3, 0] += 0.03
        report = evaluate_rigid_geometry_consistency(
            target,
            source,
            tolerance_m=0.005,
        )
        self.assertFalse(report["ok"], report)
        self.assertFalse(report["kabsch_bundle_ok"])
        self.assertGreater(report["max_kabsch_bundle_residual_m"], 0.005)

    def test_near_coplanar_volume_sign_is_reported_ambiguous(self) -> None:
        from behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local import (
            evaluate_rigid_geometry_consistency,
        )

        source = np.asarray(
            [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.0, 0.08, 0.0], [0.01, 0.02, 1.0e-8]],
            dtype=np.float64,
        )
        measured = source.copy()
        measured[3, 2] *= -1.0
        report = evaluate_rigid_geometry_consistency(
            measured,
            source,
            tolerance_m=0.012,
        )
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["volume_sign_ambiguous"], report)

    def test_geometry_report_requires_nested_point_matrices(self) -> None:
        source = self._source_points(4)
        with self.assertRaisesRegex(ValueError, "N x 3"):
            rigid_geometry_report(source.reshape(-1), source)

    def test_contract_bounds_one_to_six_points_and_typed_relations(self) -> None:
        points = [
            {"name": name, "target_xyz_m": ["x", "h", f"z_{name}"]}
            for name in ("p1", "p2", "p3", "p4")
        ]
        normalized = validate_move_tracked_point_args(
            {
                "points": points,
                "relations": [
                    {
                        "type": "common_plane",
                        "point_names": ["p1", "p2", "p3", "p4"],
                        "normal_robot_base": [0, 0, 1],
                        "offset_m": {"free": True},
                    }
                ],
            }
        )
        self.assertEqual(len(normalized["points"]), 4)
        self.assertEqual(normalized["relations"][0]["type"], "common_plane")
        six_points = points + [
            {"name": "p5", "target_xyz_m": ["x5", "h", "z5"]},
            {"name": "p6", "target_xyz_m": ["x6", "h", "z6"]},
        ]
        self.assertEqual(
            len(validate_move_tracked_point_args({"points": six_points})["points"]),
            6,
        )
        with self.assertRaisesRegex(ValueError, "more than 6"):
            validate_move_tracked_point_args(
                {
                    "points": six_points
                    + [{"name": "p7", "target_xyz_m": [0, 0, 0]}]
                }
            )
        with self.assertRaisesRegex(ValueError, "coplanar is not a motion constraint"):
            validate_move_tracked_point_args(
                {
                    "points": points[:3],
                    "relations": [
                        {
                            "type": "coplanar",
                            "point_names": ["p1", "p2", "p3"],
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
