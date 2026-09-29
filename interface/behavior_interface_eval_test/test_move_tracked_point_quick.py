from __future__ import annotations

import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from flask import Flask

import behavior_interface_eval_test.tool.official_v2.tools as official_tools
import behavior_interface_eval_test.test_official_move_tracked_point as official_test
from behavior_interface_eval_test.robot_contract import ACTION_DIM, ACTION_SLICES, ARM_DOF
from behavior_interface_eval_test.test_official_move_tracked_point import (
    _RigidTrackedManager,
)
from behavior_interface_eval_test.tool.official_v2.contract import (
    MOVE_TRACKED_POINT_MAX_POINTS,
    MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS,
    MOVE_TRACKED_POINT_QUICK_MAX_POINTS,
    validate_move_tracked_point_args,
)
from behavior_interface_eval_test.tool.official_v2.grasp_geometry_local import (
    quat_to_mat_xyzw,
)
from behavior_interface_eval_test.tool.official_v2.registry import build_registry
from behavior_interface_eval_test.tool.official_v2.tracked_point_constraints_local import (
    COLLINEAR_POSITION_TOLERANCE_M,
    evaluate_quick_constraint,
    quick_constraint_relation,
    relation_residual_blocks,
)
from behavior_interface_eval_test.official_policy_interface import (
    install_official_move_tracked_point_routes,
)


class _QuickTrackedManager(_RigidTrackedManager):
    """Rigid on-hand markers plus frozen off-hand observation references."""

    def __init__(
        self,
        world,
        adapter,
        controlled_points: dict[str, np.ndarray],
        reference_points: dict[str, np.ndarray],
    ) -> None:
        super().__init__(world, adapter, {**controlled_points, **reference_points})
        self._reference_points = {
            str(name): np.asarray(point, dtype=np.float64).copy()
            for name, point in reference_points.items()
        }
        self.reference_outlier_after_arm_motion_rad: float | None = None
        self.reference_outlier_offsets: dict[str, np.ndarray] = {}

    def observed_active_points_snapshot(self, names, *, episode_id):
        sequence = int(self.adapter.status()["sequence"])
        position, quaternion = self._eef_pose()
        rotation = quat_to_mat_xyzw(quaternion)
        arm_motion = float(
            np.linalg.norm(
                np.asarray(self.world.arm_qpos_list(self.arm), dtype=np.float64)
                - self.initial_arm_q,
                ord=np.inf,
            )
        )
        entries = {}
        for raw_name in names:
            name = str(raw_name)
            if name in self._reference_points:
                point = self._reference_points[name].copy()
                if (
                    self.reference_outlier_after_arm_motion_rad is not None
                    and arm_motion >= self.reference_outlier_after_arm_motion_rad
                    and name in self.reference_outlier_offsets
                ):
                    point = point + np.asarray(
                        self.reference_outlier_offsets[name], dtype=np.float64
                    )
            else:
                point = position + rotation @ self.anchors[name]
            entries[name] = {
                "name": name,
                "status": "observed",
                "depth_m": 1.0,
                "xyz_in_robot_base_coord_m": np.asarray(point, dtype=float).tolist(),
                "observation_sequence": sequence,
            }
        return {
            "ok": True,
            "reason": None,
            "entries": entries,
            "unavailable": {},
            "observation_sequence": sequence,
            "episode_id": str(episode_id),
            "session_id": self.session_id,
            "image_id": self.image_id,
            "frame": "current_robot_base",
        }


class MoveTrackedPointQuickTest(unittest.TestCase):
    @staticmethod
    def _request(
        quick_type: str,
        *,
        on_count: int,
        off_count: int,
        with_coordinates: bool = False,
    ) -> dict:
        points = []
        for role, count in (("on_hand", on_count), ("off_hand", off_count)):
            prefix = "held" if role == "on_hand" else "scene"
            for index in range(count):
                point = {"name": f"{prefix}_{index + 1}", "role": role}
                if with_coordinates:
                    point["target_xyz_m"] = [
                        "shared_x",
                        f"{prefix}_y_{index + 1}",
                        {"free": True},
                    ]
                points.append(point)
        if quick_type == "collinear" and on_count == 2 and off_count == 2:
            # Four-point collinear uses the actual UI row order as its directed
            # axial order: held[0] -> scene[0] -> scene[1] -> held[1].
            points = [points[0], points[2], points[3], points[1]]
        return {
            "mode": "quick_constraint",
            "points": points,
            "quick_constraint": {
                "type": quick_type,
                "on_hand_points": [
                    point["name"] for point in points if point["role"] == "on_hand"
                ],
                "off_hand_points": [
                    point["name"] for point in points if point["role"] == "off_hand"
                ],
            },
        }

    @staticmethod
    def _axis_request() -> dict:
        return {
            "mode": "quick_constraint",
            "points": [
                {"name": "stick_a", "role": "on_hand"},
                {"name": "stick_b", "role": "on_hand"},
                {"name": "table_a", "role": "off_hand"},
                {"name": "table_b", "role": "off_hand"},
                {"name": "table_c", "role": "off_hand"},
            ],
            "quick_constraint": {
                "type": "line_vertical_to_plane",
                "on_hand_points": ["stick_a", "stick_b"],
                "off_hand_points": ["table_a", "table_b", "table_c"],
            },
        }

    @staticmethod
    def _plane_request() -> dict:
        return {
            "mode": "quick_constraint",
            "points": [
                {"name": "face_a", "role": "on_hand"},
                {"name": "face_b", "role": "on_hand"},
                {"name": "face_c", "role": "on_hand"},
                {"name": "table_a", "role": "off_hand"},
                {"name": "table_b", "role": "off_hand"},
                {"name": "table_c", "role": "off_hand"},
            ],
            "quick_constraint": {
                "type": "plane_parallel",
                "on_hand_points": ["face_a", "face_b", "face_c"],
                "off_hand_points": ["table_a", "table_b", "table_c"],
            },
        }

    @staticmethod
    def _two_collinear_groups_request(*, with_coordinates: bool = False) -> dict:
        points = [
            {"name": "held_a", "role": "on_hand"},
            {"name": "held_b", "role": "on_hand"},
            {"name": "held_c", "role": "on_hand"},
            {"name": "held_d", "role": "on_hand"},
            {"name": "scene_a", "role": "off_hand"},
            {"name": "scene_b", "role": "off_hand"},
        ]
        if with_coordinates:
            points[0]["target_xyz_m"] = ["shared_x", "ya", "za"]
            points[2]["target_xyz_m"] = ["shared_x", "yc", "zc"]
        return {
            "mode": "quick_constraint",
            "points": points,
            "quick_constraints": [
                {
                    "type": "collinear",
                    "on_hand_points": ["held_a", "held_b"],
                    "off_hand_points": ["scene_a"],
                    "axial_mode": "line_only",
                },
                {
                    "type": "collinear",
                    "on_hand_points": ["held_c", "held_d"],
                    "off_hand_points": ["scene_b"],
                    "axial_mode": "line_only",
                },
            ],
        }

    def test_contract_accepts_all_six_presets_with_optional_coordinates(self) -> None:
        cases = (
            ("touch", 1, 1),
            ("flatwise", 3, 0),
            ("plane_parallel", 3, 3),
            ("collinear", 2, 1),
            ("collinear", 2, 2),
            ("line_vertical_to_plane", 2, 3),
            ("vertical_to_ground", 2, 0),
            ("vertical_to_ground", 3, 0),
        )
        for quick_type, on_count, off_count in cases:
            with self.subTest(
                quick_type=quick_type,
                on_count=on_count,
                off_count=off_count,
            ):
                request = self._request(
                    quick_type,
                    on_count=on_count,
                    off_count=off_count,
                    with_coordinates=True,
                )
                normalized = validate_move_tracked_point_args(request)
                self.assertEqual(normalized["mode"], "quick_constraint")
                self.assertEqual(
                    normalized["quick_constraint"]["type"], quick_type
                )
                if quick_type == "collinear":
                    expected_axial_mode = "ordered"
                    self.assertEqual(
                        normalized["quick_constraint"]["axial_mode"],
                        expected_axial_mode,
                    )
                self.assertEqual(len(normalized["points"]), on_count + off_count)
                self.assertTrue(
                    all("target_xyz_m" in point for point in normalized["points"])
                )
                self.assertEqual(
                    normalized["points"][0]["target_xyz_m"]["x"],
                    {"var": "shared_x"},
                )

        axis = validate_move_tracked_point_args(self._axis_request())
        self.assertEqual(axis["mode"], "quick_constraint")
        self.assertEqual(axis["quick_constraint"]["type"], "line_vertical_to_plane")
        self.assertEqual(len(axis["points"]), 5)

        plane = validate_move_tracked_point_args(self._plane_request())
        self.assertEqual(plane["quick_constraint"]["type"], "plane_parallel")
        self.assertEqual(len(plane["points"]), MOVE_TRACKED_POINT_QUICK_MAX_POINTS)

    def test_contract_accepts_multiple_named_constraint_groups(self) -> None:
        request = self._two_collinear_groups_request(with_coordinates=True)
        normalized = validate_move_tracked_point_args(request)

        self.assertEqual(normalized["mode"], "quick_constraint")
        self.assertNotIn("quick_constraint", normalized)
        self.assertEqual(len(normalized["quick_constraints"]), 2)
        self.assertEqual(
            normalized["quick_constraints"][0]["axial_point_order"],
            ["held_a", "held_b", "scene_a"],
        )
        self.assertEqual(
            normalized["quick_constraints"][1]["axial_point_order"],
            ["held_c", "held_d", "scene_b"],
        )
        self.assertEqual(
            normalized["points"][0]["target_xyz_m"]["x"],
            {"var": "shared_x"},
        )

        shared = self._two_collinear_groups_request()
        shared["quick_constraints"][1]["on_hand_points"] = [
            "held_a",
            "held_c",
        ]
        normalized_shared = validate_move_tracked_point_args(shared)
        self.assertEqual(
            normalized_shared["quick_constraints"][1]["on_hand_points"],
            ["held_a", "held_c"],
        )

    def test_contract_rejects_invalid_multiple_constraint_group_envelopes(self) -> None:
        both_forms = self._two_collinear_groups_request()
        both_forms["quick_constraint"] = both_forms["quick_constraints"][0]
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_move_tracked_point_args(both_forms)

        empty = self._two_collinear_groups_request()
        empty["quick_constraints"] = []
        empty.pop("mode", None)
        normalized_empty = validate_move_tracked_point_args(empty)
        self.assertNotIn("quick_constraints", normalized_empty)
        self.assertEqual(normalized_empty["relations"], [])

        too_many = self._two_collinear_groups_request()
        too_many["quick_constraints"] = [
            dict(too_many["quick_constraints"][0])
            for _ in range(MOVE_TRACKED_POINT_MAX_QUICK_CONSTRAINTS + 1)
        ]
        with self.assertRaisesRegex(ValueError, "more than 6"):
            validate_move_tracked_point_args(too_many)

        unknown = self._two_collinear_groups_request()
        unknown["quick_constraints"][1]["off_hand_points"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "unknown point"):
            validate_move_tracked_point_args(unknown)

        wrong_role = self._two_collinear_groups_request()
        wrong_role["quick_constraints"][1]["off_hand_points"] = ["held_a"]
        with self.assertRaisesRegex(ValueError, "must have role off_hand"):
            validate_move_tracked_point_args(wrong_role)

        # The singular field is a compatibility contract: it still binds all
        # submitted rows and therefore cannot silently acquire plural semantics.
        legacy = self._request("collinear", on_count=2, off_count=1)
        legacy["points"].append({"name": "unused", "role": "on_hand"})
        with self.assertRaisesRegex(ValueError, "every submitted point"):
            validate_move_tracked_point_args(legacy)

    def test_contract_derives_role_lists_and_ignores_null_optional_quick_field(self) -> None:
        compact = self._axis_request()
        compact["quick_constraint"].pop("on_hand_points")
        compact["quick_constraint"].pop("off_hand_points")
        normalized = validate_move_tracked_point_args(compact)
        self.assertEqual(
            normalized["quick_constraint"]["on_hand_points"],
            ["stick_a", "stick_b"],
        )
        self.assertEqual(
            normalized["quick_constraint"]["off_hand_points"],
            ["table_a", "table_b", "table_c"],
        )

        legacy = {
            "points": [{"name": "p", "target_xyz_m": [0.0, 0.0, 0.0]}],
            "quick_constraint": None,
        }
        self.assertNotIn("quick_constraint", validate_move_tracked_point_args(legacy))

        mode_only = self._axis_request()
        mode_only.pop("quick_constraint")
        mode_only["mode"] = "axis_perpendicular_to_plane"
        normalized_mode_only = validate_move_tracked_point_args(mode_only)
        self.assertEqual(
            normalized_mode_only["quick_constraint"]["type"],
            "line_vertical_to_plane",
        )

        empty_mode = self._axis_request()
        empty_mode["mode"] = ""
        self.assertEqual(
            validate_move_tracked_point_args(empty_mode)["mode"],
            "quick_constraint",
        )

        conflicting_mode = self._axis_request()
        conflicting_mode["mode"] = "plane_parallel"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            validate_move_tracked_point_args(conflicting_mode)

    def test_collinear_contract_defaults_to_spatial_order_and_preserves_legacy_modes(self) -> None:
        request = self._request("collinear", on_count=2, off_count=2)
        normalized = validate_move_tracked_point_args(request)
        self.assertEqual(
            normalized["quick_constraint"]["axial_mode"], "ordered"
        )
        self.assertEqual(
            normalized["quick_constraint"]["axial_point_order"],
            ["held_1", "scene_1", "scene_2", "held_2"],
        )

        request["quick_constraint"]["axial_mode"] = "segment_overlap"
        normalized = validate_move_tracked_point_args(request)
        self.assertEqual(
            normalized["quick_constraint"]["axial_mode"], "segment_overlap"
        )

        request["quick_constraint"]["axial_mode"] = "line_only"
        normalized = validate_move_tracked_point_args(request)
        self.assertEqual(normalized["quick_constraint"]["axial_mode"], "line_only")

        request["quick_constraint"]["axial_mode"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "ordered_containment"):
            validate_move_tracked_point_args(request)

        one_reference = self._request("collinear", on_count=2, off_count=1)
        one_reference["quick_constraint"]["axial_mode"] = "segment_overlap"
        with self.assertRaisesRegex(ValueError, "requires two off-hand"):
            validate_move_tracked_point_args(one_reference)

    def test_four_point_collinear_uses_and_validates_exact_input_row_order(self) -> None:
        request = {
            "mode": "quick_constraint",
            "points": [
                {"name": "1", "role": "on_hand"},
                {"name": "3", "role": "off_hand"},
                {"name": "4", "role": "off_hand"},
                {"name": "2", "role": "on_hand"},
            ],
            "quick_constraint": {
                "type": "collinear",
                "on_hand_points": ["1", "2"],
                "off_hand_points": ["3", "4"],
                "axial_point_order": ["1", "3", "4", "2"],
            },
        }
        normalized = validate_move_tracked_point_args(request)
        self.assertEqual(
            normalized["quick_constraint"]["axial_point_order"],
            ["1", "3", "4", "2"],
        )

        grouped = {
            **request,
            "points": [
                {"name": "1", "role": "on_hand"},
                {"name": "2", "role": "on_hand"},
                {"name": "3", "role": "off_hand"},
                {"name": "4", "role": "off_hand"},
            ],
            "quick_constraint": {
                **request["quick_constraint"],
                "axial_point_order": ["1", "2", "3", "4"],
            },
        }
        self.assertEqual(validate_move_tracked_point_args(grouped)["quick_constraint"]["axial_point_order"],
                         ["1", "2", "3", "4"])
        grouped["quick_constraint"]["axial_mode"] = "ordered_containment"
        with self.assertRaisesRegex(ValueError, "point rows in axial order"):
            validate_move_tracked_point_args(grouped)

        reordered_role_lists = {
            **request,
            "quick_constraint": {
                **request["quick_constraint"],
                "on_hand_points": ["2", "1"],
            },
        }
        with self.assertRaisesRegex(ValueError, "preserve their point-row order"):
            validate_move_tracked_point_args(reordered_role_lists)

        mismatched_axial_order = {
            **request,
            "quick_constraint": {
                **request["quick_constraint"],
                "axial_point_order": ["2", "4", "3", "1"],
            },
        }
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_move_tracked_point_args(mismatched_axial_order)

    def test_direct_tool_preserves_conflicting_mode_for_validation(self) -> None:
        """The Python entry point must not bypass the shared quick contract."""

        _adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        ctx, result = official_test.OfficialMoveTrackedPointTest._ctx(world)
        request = self._axis_request()
        generator = official_tools.move_tracked_point(
            ctx,
            points=request["points"],
            quick_constraint=request["quick_constraint"],
            mode="plane_parallel",
        )
        next(generator)
        with self.assertRaises(StopIteration):
            next(generator)
        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_stage"], "input validation")
        self.assertIn("conflicts", result["error"])

    def test_contract_rejects_wrong_counts_roles_orientation_and_extra_points(self) -> None:
        axis = self._axis_request()
        axis["points"] = axis["points"][:4]
        axis["quick_constraint"].pop("on_hand_points")
        axis["quick_constraint"].pop("off_hand_points")
        with self.assertRaisesRegex(ValueError, "exactly 2 on-hand and 3 off-hand"):
            validate_move_tracked_point_args(axis)

        axis = self._axis_request()
        axis["points"][0]["role"] = "off_hand"
        with self.assertRaisesRegex(ValueError, "must have role on_hand"):
            validate_move_tracked_point_args(axis)

        axis = self._axis_request()
        axis["points"][0]["target_xyz_m"] = [0, "shared_y", {"free": True}]
        self.assertIn(
            "target_xyz_m", validate_move_tracked_point_args(axis)["points"][0]
        )

        axis = self._axis_request()
        axis["points"].append({"name": "extra", "role": "off_hand"})
        with self.assertRaisesRegex(ValueError, "point lists|exactly 3 off-hand"):
            validate_move_tracked_point_args(axis)

        for field in ("normal_mode", "direction_mode", "mode", "sense"):
            wrong_mode_field = self._axis_request()
            wrong_mode_field["quick_constraint"][field] = "same"
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "unoriented"
            ):
                validate_move_tracked_point_args(wrong_mode_field)

        too_many = [
            {"name": f"p{index}", "role": "on_hand"}
            for index in range(MOVE_TRACKED_POINT_MAX_POINTS + 1)
        ]
        with self.assertRaisesRegex(ValueError, "more than 6"):
            validate_move_tracked_point_args({"points": too_many})

    def test_http_route_canonicalizes_role_only_quick_points(self) -> None:
        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-quick-http"),
            wait_for_skill_result=mock.Mock(
                return_value={"ok": False, "tool": "move_tracked_point"}
            ),
        )
        app = Flask(__name__)
        install_official_move_tracked_point_routes(
            app,
            SimpleNamespace(server=server),
        )
        response = app.test_client().post(
            "/api/v2/move_tracked_point",
            json={
                "session_id": "web-quick",
                "mode": "quick_constraint",
                "points": [
                    {"name": "a", "role": "on_hand"},
                    {"name": "b", "role": "on_hand"},
                    {"name": "p", "role": "off_hand"},
                    {"name": "q", "role": "off_hand"},
                    {"name": "r", "role": "off_hand"},
                ],
                "quick_constraint": {
                    "type": "line_vertical_to_plane",
                },
            },
        )
        self.assertEqual(response.status_code, 400)
        submitted = next(
            call
            for call in server.submit_skill.call_args_list
            if call.args and call.args[0] == "move_tracked_point"
        )
        self.assertEqual(submitted.args[0], "move_tracked_point")
        submitted_args = submitted.args[1]
        self.assertEqual(
            submitted_args["quick_constraint"]["on_hand_points"], ["a", "b"]
        )
        self.assertEqual(
            submitted_args["quick_constraint"]["off_hand_points"],
            ["p", "q", "r"],
        )

    def test_http_route_forwards_multiple_constraint_groups_canonically(self) -> None:
        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-multi-quick-http"),
            wait_for_skill_result=mock.Mock(
                return_value={"ok": True, "tool": "move_tracked_point"}
            ),
        )
        app = Flask(__name__)
        install_official_move_tracked_point_routes(
            app,
            SimpleNamespace(server=server),
        )
        request = self._two_collinear_groups_request(with_coordinates=True)
        request["session_id"] = "web-multi-quick"
        response = app.test_client().post(
            "/api/v2/move_tracked_point",
            json=request,
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        submitted = next(
            call
            for call in server.submit_skill.call_args_list
            if call.args and call.args[0] == "move_tracked_point"
        )
        self.assertEqual(submitted.args[0], "move_tracked_point")
        submitted_args = submitted.args[1]
        self.assertNotIn("quick_constraint", submitted_args)
        self.assertEqual(len(submitted_args["quick_constraints"]), 2)
        self.assertEqual(
            submitted_args["quick_constraints"][1]["axial_point_order"],
            ["held_c", "held_d", "scene_b"],
        )

    def test_off_hand_reference_snapshot_rejects_drift(self) -> None:
        class _DriftingManager:
            def observed_active_points_snapshot(self, names, *, episode_id):
                entries = {
                    str(name): {
                        "xyz_in_robot_base_coord_m": [
                            0.2 + (0.1 if str(name) == "p" else 0.0),
                            0.2,
                            0.3,
                        ]
                    }
                    for name in names
                }
                return {
                    "ok": True,
                    "entries": entries,
                    "observation_sequence": 7,
                }

        report = official_tools._move_tracked_reference_snapshot_check(
            _DriftingManager(),
            ["p", "q", "r"],
            np.asarray(
                [[0.2, 0.2, 0.3], [0.3, 0.2, 0.3], [0.2, 0.3, 0.3]],
                dtype=np.float64,
            ),
            episode_id="episode-1",
            expected_sequence=7,
            tolerance_m=0.02,
        )
        self.assertFalse(report["ok"], report)
        self.assertEqual(report["reason"], "off_hand_points_moved")
        self.assertGreater(report["max_drift_m"], 0.09)

    def test_off_hand_consensus_suppresses_one_of_two_association_jump(self) -> None:
        frozen = np.asarray(
            [[0.60, 0.12, 0.07], [0.64, 0.09, 0.09]], dtype=np.float64
        )
        measured = frozen.copy()
        measured[0] = [0.66, 0.14, -0.01]

        report = official_tools._move_tracked_off_hand_reference_consensus(
            ["wood_a", "wood_b"],
            measured,
            frozen,
            tolerance_m=0.024,
            allow_visual_outlier_suppression=True,
        )

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["tracking_outlier_suppressed"], report)
        self.assertFalse(report["all_points_verified"], report)
        self.assertEqual(report["stable_names"], ["wood_b"])
        self.assertEqual(report["outlier_names"], ["wood_a"])
        self.assertEqual(
            report["constraint_reference_source"],
            "frozen_off_hand_reference_due_visual_outlier",
        )
        self.assertFalse(report["rigid_geometry_consistent"], report)

    def test_verified_live_off_hand_points_do_not_redefine_frozen_goal(self) -> None:
        frozen = np.asarray(
            [[0.494844, 0.129715, 0.073980], [0.510219, 0.091586, 0.091019]],
            dtype=np.float64,
        )
        measured = frozen + np.asarray(
            [[0.00010, -0.00004, 0.00002], [-0.00006, 0.00003, -0.00001]],
            dtype=np.float64,
        )

        report = official_tools._move_tracked_off_hand_reference_consensus(
            ["wood_a", "wood_b"],
            measured,
            frozen,
            tolerance_m=0.024,
            allow_visual_outlier_suppression=True,
        )

        self.assertTrue(report["ok"], report)
        self.assertTrue(report["all_points_verified"], report)
        self.assertFalse(report["tracking_outlier_suppressed"], report)
        self.assertEqual(
            report["constraint_reference_source"],
            "frozen_off_hand_reference_verified_by_live_rgbd",
        )

    def test_off_hand_consensus_rejects_coherent_two_point_motion(self) -> None:
        frozen = np.asarray(
            [[0.60, 0.12, 0.07], [0.64, 0.09, 0.09]], dtype=np.float64
        )
        measured = frozen + np.asarray([0.05, -0.02, 0.0])

        report = official_tools._move_tracked_off_hand_reference_consensus(
            ["wood_a", "wood_b"],
            measured,
            frozen,
            tolerance_m=0.024,
            allow_visual_outlier_suppression=True,
        )

        self.assertFalse(report["ok"], report)
        self.assertFalse(report["tracking_outlier_suppressed"], report)
        self.assertTrue(report["rigid_geometry_consistent"], report)
        self.assertEqual(report["reason"], "off_hand_points_moved")

    def test_off_hand_consensus_rejects_majority_outliers(self) -> None:
        frozen = np.asarray(
            [[0.60, 0.12, 0.07], [0.64, 0.09, 0.09], [0.62, 0.16, 0.08]],
            dtype=np.float64,
        )
        measured = frozen.copy()
        measured[0] += [0.08, 0.0, 0.0]
        measured[1] += [0.0, -0.08, 0.0]

        report = official_tools._move_tracked_off_hand_reference_consensus(
            ["wood_a", "wood_b", "wood_c"],
            measured,
            frozen,
            tolerance_m=0.024,
            allow_visual_outlier_suppression=True,
        )

        self.assertFalse(report["ok"], report)
        self.assertEqual(report["stable_count"], 1)
        self.assertEqual(report["required_stable_count"], 2)
        self.assertEqual(report["reason"], "off_hand_points_moved")

    def test_quick_geometry_evaluator_checks_axis_plane_and_degeneracy(self) -> None:
        off = np.asarray(
            [[0.2, 0.2, 0.3], [0.3, 0.2, 0.3], [0.2, 0.3, 0.3]],
            dtype=np.float64,
        )
        axis = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]], dtype=np.float64)
        report = evaluate_quick_constraint(
            axis,
            off,
            self._axis_request()["quick_constraint"],
            orientation_tolerance_deg=1.0,
        )
        self.assertTrue(report["ok"], report)
        self.assertLess(report["max_error_deg"], 1.0)

        wrong_axis = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
        wrong = evaluate_quick_constraint(
            wrong_axis,
            off,
            self._axis_request()["quick_constraint"],
            orientation_tolerance_deg=1.0,
        )
        self.assertFalse(wrong["ok"], wrong)

        plane = np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            dtype=np.float64,
        )
        plane_report = evaluate_quick_constraint(
            plane,
            off,
            self._plane_request()["quick_constraint"],
            orientation_tolerance_deg=1.0,
        )
        self.assertTrue(plane_report["ok"], plane_report)

        degenerate = evaluate_quick_constraint(
            axis,
            np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64),
            self._axis_request()["quick_constraint"],
        )
        self.assertFalse(degenerate["ok"])
        self.assertEqual(degenerate["error"], "off_hand_plane_degenerate")

    def test_multiple_constraint_evaluator_uses_each_named_subset_and_max_error(self) -> None:
        request = validate_move_tracked_point_args(
            self._two_collinear_groups_request()
        )
        controlled = np.asarray(
            [
                [-0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, -0.1, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        references = np.asarray(
            [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0]],
            dtype=np.float64,
        )
        report = official_tools._move_tracked_evaluate_quick_constraints(
            controlled,
            references,
            request["quick_constraints"],
            controlled_names=["held_a", "held_b", "held_c", "held_d"],
            reference_names=["scene_a", "scene_b"],
            position_tolerance_m=0.03,
            orientation_tolerance_deg=20.0,
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["type"], "quick_constraint_group_set")
        self.assertEqual(report["group_count"], 2)
        self.assertEqual(
            report["groups"][0]["on_hand_point_names"],
            ["held_a", "held_b"],
        )
        self.assertEqual(
            report["groups"][1]["off_hand_point_names"],
            ["scene_b"],
        )

        perturbed = controlled.copy()
        perturbed[3, 0] = 0.02
        failed = official_tools._move_tracked_evaluate_quick_constraints(
            perturbed,
            references,
            request["quick_constraints"],
            controlled_names=["held_a", "held_b", "held_c", "held_d"],
            reference_names=["scene_a", "scene_b"],
            position_tolerance_m=0.03,
            orientation_tolerance_deg=20.0,
        )
        self.assertFalse(failed["ok"], failed)
        self.assertTrue(failed["groups"][0]["ok"], failed)
        self.assertFalse(failed["groups"][1]["ok"], failed)
        self.assertGreater(failed["max_error_m"], 0.001)
        self.assertIn("quick_constraint_group_2", failed["error"])

    def test_all_quick_geometry_modes_accept_and_reject_expected_shapes(self) -> None:
        horizontal_plane = np.asarray(
            [[0.0, 0.0, 0.2], [0.1, 0.0, 0.2], [0.0, 0.1, 0.2]],
            dtype=np.float64,
        )
        vertical_plane = np.asarray(
            [[0.0, 0.0, 0.2], [0.1, 0.0, 0.2], [0.0, 0.0, 0.3]],
            dtype=np.float64,
        )

        touch = self._request("touch", on_count=1, off_count=1)[
            "quick_constraint"
        ]
        self.assertTrue(
            evaluate_quick_constraint([[0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3]], touch)[
                "ok"
            ]
        )
        self.assertFalse(
            evaluate_quick_constraint([[0.1, 0.2, 0.3]], [[0.2, 0.2, 0.3]], touch)[
                "ok"
            ]
        )

        flatwise = self._request("flatwise", on_count=3, off_count=0)[
            "quick_constraint"
        ]
        self.assertTrue(evaluate_quick_constraint(horizontal_plane, None, flatwise)["ok"])
        self.assertTrue(
            evaluate_quick_constraint(horizontal_plane[[0, 2, 1]], None, flatwise)[
                "ok"
            ]
        )
        self.assertFalse(evaluate_quick_constraint(vertical_plane, None, flatwise)["ok"])

        plane_parallel = self._request(
            "plane_parallel", on_count=3, off_count=3
        )["quick_constraint"]
        self.assertTrue(
            evaluate_quick_constraint(
                horizontal_plane[[0, 2, 1]], horizontal_plane, plane_parallel
            )["ok"]
        )
        self.assertFalse(
            evaluate_quick_constraint(vertical_plane, horizontal_plane, plane_parallel)[
                "ok"
            ]
        )

        collinear_one = self._request("collinear", on_count=2, off_count=1)[
            "quick_constraint"
        ]
        on_line = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
        self.assertTrue(
            evaluate_quick_constraint(on_line, [[0.1, 0.0, 0.0]], collinear_one)[
                "ok"
            ]
        )
        self.assertFalse(
            evaluate_quick_constraint(on_line, [[0.1, 0.1, 0.0]], collinear_one)[
                "ok"
            ]
        )
        collinear_two = self._request("collinear", on_count=2, off_count=2)[
            "quick_constraint"
        ]
        contained = evaluate_quick_constraint(
            on_line, [[0.05, 0.0, 0.0], [0.15, 0.0, 0.0]], collinear_two
        )
        self.assertTrue(contained["ok"], contained)
        self.assertEqual(contained["axial_mode"], "ordered")
        self.assertEqual(
            contained["ordered_point_sequence"],
            ["held_1", "scene_1", "scene_2", "held_2"],
        )
        self.assertTrue(contained["reference_inside_controlled"])
        self.assertFalse(
            evaluate_quick_constraint(
                on_line + [0.0, 0.1, 0.0],
                [[0.05, 0.0, 0.0], [0.15, 0.0, 0.0]],
                collinear_two,
            )["ok"]
        )
        reversed_controlled = evaluate_quick_constraint(
            on_line[::-1], [[0.05, 0.0, 0.0], [0.15, 0.0, 0.0]], collinear_two
        )
        self.assertFalse(reversed_controlled["ok"], reversed_controlled)
        self.assertAlmostEqual(reversed_controlled["max_error_deg"], 180.0)
        wrong_order = evaluate_quick_constraint(
            [[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            collinear_two,
            position_tolerance_m=0.01,
        )
        self.assertFalse(wrong_order["ok"], wrong_order)
        self.assertEqual(
            wrong_order["error"],
            "collinear_axial_order_not_satisfied",
        )
        separated = evaluate_quick_constraint(
            on_line,
            [[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
            collinear_two,
            position_tolerance_m=0.01,
        )
        self.assertFalse(separated["ok"], separated)
        self.assertEqual(
            separated["error"],
            "collinear_axial_order_not_satisfied",
        )
        self.assertAlmostEqual(separated["segment_gap_m"], 0.1)

        overlap = dict(collinear_two)
        overlap["axial_mode"] = "segment_overlap"
        legacy_overlap = evaluate_quick_constraint(
            on_line,
            [[-0.1, 0.0, 0.0], [0.3, 0.0, 0.0]],
            overlap,
        )
        self.assertTrue(legacy_overlap["ok"], legacy_overlap)
        self.assertEqual(legacy_overlap["axial_mode"], "segment_overlap")

        line_only = dict(collinear_two)
        line_only["axial_mode"] = "line_only"
        legacy = evaluate_quick_constraint(
            on_line,
            [[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
            line_only,
            position_tolerance_m=0.01,
        )
        self.assertTrue(legacy["ok"], legacy)
        self.assertEqual(legacy["axial_mode"], "line_only")
        self.assertAlmostEqual(legacy["segment_gap_m"], 0.1)

        vertical = self._request(
            "line_vertical_to_plane", on_count=2, off_count=3
        )["quick_constraint"]
        self.assertTrue(
            evaluate_quick_constraint(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
                horizontal_plane,
                vertical,
            )["ok"]
        )
        self.assertTrue(
            evaluate_quick_constraint(
                [[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]],
                horizontal_plane,
                vertical,
            )["ok"]
        )
        self.assertFalse(
            evaluate_quick_constraint(on_line, horizontal_plane, vertical)["ok"]
        )

    def test_collinear_uses_one_mm_without_tightening_other_quick_presets(self) -> None:
        collinear = self._request("collinear", on_count=2, off_count=2)[
            "quick_constraint"
        ]
        reference = np.asarray(
            [[0.05, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=np.float64
        )
        within = evaluate_quick_constraint(
            [[0.0, 0.0008, 0.0], [0.2, 0.0008, 0.0]],
            reference,
            collinear,
            position_tolerance_m=0.03,
            orientation_tolerance_deg=20.0,
        )
        outside = evaluate_quick_constraint(
            [[0.0, 0.0012, 0.0], [0.2, 0.0012, 0.0]],
            reference,
            collinear,
            position_tolerance_m=0.03,
            orientation_tolerance_deg=20.0,
        )
        self.assertTrue(within["ok"], within)
        self.assertFalse(outside["ok"], outside)
        self.assertEqual(outside["error"], "collinear_not_satisfied")
        self.assertAlmostEqual(
            outside["position_tolerance_m"], COLLINEAR_POSITION_TOLERANCE_M
        )
        self.assertAlmostEqual(outside["requested_position_tolerance_m"], 0.03)

        touch = self._request("touch", on_count=1, off_count=1)[
            "quick_constraint"
        ]
        touch_report = evaluate_quick_constraint(
            [[0.0, 0.0, 0.0]],
            [[0.0012, 0.0, 0.0]],
            touch,
            position_tolerance_m=0.03,
        )
        self.assertTrue(touch_report["ok"], touch_report)
        self.assertAlmostEqual(touch_report["position_tolerance_m"], 0.03)

    def test_quick_planning_relation_is_derived_from_frozen_reference_plane(self) -> None:
        off = np.asarray(
            [[0.2, 0.2, 0.3], [0.3, 0.2, 0.3], [0.2, 0.3, 0.3]],
            dtype=np.float64,
        )
        axis_relation = quick_constraint_relation(
            self._axis_request()["quick_constraint"],
            off_hand_points_robot_base_m=off,
        )
        self.assertEqual(axis_relation["type"], "align_vector")
        self.assertEqual(axis_relation["point_names"], ["stick_a", "stick_b"])
        np.testing.assert_allclose(axis_relation["direction_robot_base"], [0, 0, 1])

        plane_relation = quick_constraint_relation(
            self._plane_request()["quick_constraint"],
            off_hand_points_robot_base_m=off,
        )
        self.assertEqual(plane_relation["type"], "oriented_plane_normal")
        self.assertEqual(
            plane_relation["point_names"], ["face_a", "face_b", "face_c"]
        )

        touch = self._request("touch", on_count=1, off_count=1)
        touch_relation = quick_constraint_relation(
            touch["quick_constraint"],
            off_hand_points_robot_base_m=[[0.2, 0.3, 0.4]],
        )
        self.assertEqual(touch_relation["type"], "point_at_position")
        np.testing.assert_allclose(
            touch_relation["target_position_robot_base_m"], [0.2, 0.3, 0.4]
        )

        flatwise = self._request("flatwise", on_count=3, off_count=0)
        flatwise_relation = quick_constraint_relation(
            flatwise["quick_constraint"], off_hand_points_robot_base_m=None
        )
        self.assertEqual(flatwise_relation["type"], "oriented_plane_normal")
        self.assertEqual(flatwise_relation["mode"], "parallel")

        collinear_one = self._request("collinear", on_count=2, off_count=1)
        self.assertEqual(
            quick_constraint_relation(
                collinear_one["quick_constraint"],
                off_hand_points_robot_base_m=[[0.1, 0.0, 0.0]],
            )["type"],
            "ordered_collinear",
        )
        collinear_two = self._request("collinear", on_count=2, off_count=2)
        containment_relation = quick_constraint_relation(
            collinear_two["quick_constraint"],
            off_hand_points_robot_base_m=[[-0.1, 0.0, 0.0], [0.3, 0.0, 0.0]],
        )
        self.assertEqual(containment_relation["type"], "ordered_collinear")
        collinear_two["quick_constraint"]["axial_mode"] = "ordered_containment"
        containment_relation = quick_constraint_relation(
            collinear_two["quick_constraint"],
            off_hand_points_robot_base_m=[[-0.1, 0.0, 0.0], [0.3, 0.0, 0.0]],
        )
        self.assertEqual(
            containment_relation["segment_start_robot_base_m"], [-0.1, 0.0, 0.0]
        )
        self.assertEqual(
            containment_relation["segment_end_robot_base_m"], [0.3, 0.0, 0.0]
        )

        collinear_two["quick_constraint"]["axial_mode"] = "segment_overlap"
        self.assertEqual(
            quick_constraint_relation(
                collinear_two["quick_constraint"],
                off_hand_points_robot_base_m=[[-0.1, 0.0, 0.0], [0.3, 0.0, 0.0]],
            )["type"],
            "line_segment_overlap",
        )

        collinear_two["quick_constraint"]["axial_mode"] = "line_only"
        self.assertEqual(
            quick_constraint_relation(
                collinear_two["quick_constraint"],
                off_hand_points_robot_base_m=[[-0.1, 0.0, 0.0], [0.3, 0.0, 0.0]],
            )["type"],
            "line_coincident",
        )

    def test_segment_overlap_relation_drives_axis_gap_and_stacks_with_coordinates(self) -> None:
        relation = {
            "type": "line_segment_overlap",
            "point_names": ["held_a", "held_b"],
            "segment_start_robot_base_m": [0.3, 0.0, 0.0],
            "segment_end_robot_base_m": [0.4, 0.0, 0.0],
        }
        separated = relation_residual_blocks(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [relation],
            point_names=["held_a", "held_b"],
        )[0]
        self.assertEqual(separated["type"], "line_segment_overlap")
        self.assertEqual(len(separated["residual"]), 7)
        self.assertAlmostEqual(separated["segment_gap_m"], 0.1)
        self.assertAlmostEqual(separated["residual"][-1], 0.1)
        self.assertAlmostEqual(separated["max_error"], 0.1)

        touching = relation_residual_blocks(
            [[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]],
            [relation],
            point_names=["held_a", "held_b"],
        )[0]
        self.assertAlmostEqual(touching["segment_gap_m"], 0.0)
        self.assertAlmostEqual(touching["max_error"], 0.0)

        request = self._request(
            "collinear", on_count=2, off_count=2, with_coordinates=True
        )
        normalized = validate_move_tracked_point_args(request)
        self.assertTrue(
            all("target_xyz_m" in point for point in normalized["points"])
        )
        self.assertEqual(
            normalized["quick_constraint"]["axial_mode"], "ordered"
        )

    def test_ordered_containment_relation_enforces_direction_and_both_endpoints(self) -> None:
        relation = {
            "type": "line_segment_contains",
            "point_names": ["held_start", "held_end"],
            "segment_start_robot_base_m": [0.1, 0.0, 0.0],
            "segment_end_robot_base_m": [0.2, 0.0, 0.0],
        }
        exact = relation_residual_blocks(
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
            [relation],
            point_names=["held_start", "held_end"],
        )[0]
        self.assertEqual(exact["type"], "line_segment_contains")
        self.assertEqual(len(exact["residual"]), 8)
        self.assertAlmostEqual(exact["max_error"], 0.0)
        self.assertAlmostEqual(exact["orientation_error_deg"], 0.0)
        self.assertTrue(exact["reference_inside_controlled"])
        self.assertEqual(
            exact["ordered_point_sequence"],
            ["held_start", "segment_start", "segment_end", "held_end"],
        )

        after_reference = relation_residual_blocks(
            [[0.2, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [relation],
            point_names=["held_start", "held_end"],
        )[0]
        self.assertAlmostEqual(after_reference["containment_violation_m"], 0.1)
        self.assertAlmostEqual(after_reference["max_error"], 0.106)
        self.assertFalse(after_reference["reference_inside_controlled"])

        reversed_segment = relation_residual_blocks(
            [[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [relation],
            point_names=["held_start", "held_end"],
        )[0]
        self.assertAlmostEqual(reversed_segment["orientation_error_deg"], 180.0)
        self.assertGreater(float(np.linalg.norm(reversed_segment["orientation_residual"])), 1.9)

    def test_plan_0135_tiny_line_error_is_rejected_for_wrong_axial_order(self) -> None:
        controlled = np.asarray(
            [
                [0.5187201238204809, 0.014113727691664475, 0.08404208505043043],
                [0.5541936660832922, -0.0832910068387843, 0.08048015236974079],
            ],
            dtype=np.float64,
        )
        reference = np.asarray(
            [
                [0.506854480786034, 0.04670071030278718, 0.08522950282031738],
                [0.5187215160235084, 0.01411600379276613, 0.08403803913486418],
            ],
            dtype=np.float64,
        )
        quick = {
            "type": "collinear",
            "on_hand_points": ["1", "2"],
            "off_hand_points": ["3", "4"],
        }

        report = evaluate_quick_constraint(
            controlled,
            reference,
            quick,
            position_tolerance_m=0.03,
            orientation_tolerance_deg=20.0,
        )

        self.assertFalse(report["ok"], report)
        self.assertLess(report["line_error_m"], 5.0e-6)
        self.assertLess(report["max_error_deg"], 3.0e-4)
        self.assertGreater(report["containment_violation_m"], 0.034)
        self.assertEqual(
            report["ordered_point_sequence"], ["1", "3", "4", "2"]
        )

    def test_collinear_reports_axis_angle_separately_from_line_distance(self) -> None:
        reference = np.asarray(
            [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64
        )
        angle_deg = 18.0
        # Keep the perpendicular displacement below the dedicated 1 mm
        # collinearity gate so this case isolates the angular acceptance path.
        half_axis = 0.002 * np.asarray(
            [
                math.cos(math.radians(angle_deg)),
                math.sin(math.radians(angle_deg)),
                0.0,
            ],
            dtype=np.float64,
        )
        controlled = np.asarray([0.025, 0.0, 0.0])[None, :] + np.vstack(
            [-half_axis, half_axis]
        )
        quick = {
            "type": "collinear",
            "on_hand_points": ["held_a", "held_b"],
            "off_hand_points": ["scene_a", "scene_b"],
            "axial_mode": "segment_overlap",
        }

        broad = evaluate_quick_constraint(
            controlled,
            reference,
            quick,
            position_tolerance_m=0.03,
            orientation_tolerance_deg=20.0,
        )
        precise = evaluate_quick_constraint(
            controlled,
            reference,
            quick,
            position_tolerance_m=0.03,
            orientation_tolerance_deg=5.0,
        )
        relation = quick_constraint_relation(
            quick,
            off_hand_points_robot_base_m=reference,
        )
        block = relation_residual_blocks(
            controlled,
            [relation],
            point_names=["held_a", "held_b"],
        )[0]

        self.assertTrue(broad["ok"], broad)
        self.assertFalse(precise["ok"], precise)
        self.assertEqual(precise["error"], "collinear_orientation_not_satisfied")
        self.assertAlmostEqual(broad["max_error_deg"], angle_deg, places=9)
        self.assertAlmostEqual(block["orientation_error_deg"], angle_deg, places=9)
        self.assertGreater(block["max_error"], 0.0)
        self.assertLess(block["max_error"], 0.03)
        self.assertGreater(np.linalg.norm(block["orientation_residual"]), 0.0)

    def test_local_quick_helpers_accept_compact_mode_aliases_and_reject_conflicts(self) -> None:
        off = np.asarray(
            [[0.2, 0.2, 0.3], [0.3, 0.2, 0.3], [0.2, 0.3, 0.3]],
            dtype=np.float64,
        )
        axis = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]], dtype=np.float64)
        compact = {
            "type": "axis_perpendicular_to_plane",
            "on_hand_points": ["a", "b"],
            "mode": "parallel",
        }
        relation = quick_constraint_relation(
            compact,
            off_hand_points_robot_base_m=off,
        )
        self.assertEqual(relation["mode"], "parallel")
        report = evaluate_quick_constraint(
            axis,
            off,
            compact,
            orientation_tolerance_deg=1.0,
        )
        self.assertTrue(report["ok"], report)

        conflicting = dict(compact)
        conflicting["sense"] = "parallel"
        with self.assertRaisesRegex(ValueError, "only one"):
            quick_constraint_relation(
                conflicting,
                off_hand_points_robot_base_m=off,
            )

        wrong_typed_field = dict(compact)
        wrong_typed_field["mode"] = "same"
        with self.assertRaisesRegex(ValueError, "unoriented"):
            quick_constraint_relation(
                wrong_typed_field,
                off_hand_points_robot_base_m=off,
            )

    def _run_quick_case(
        self,
        *,
        plane: bool,
        reference_outlier: bool = False,
        coherent_reference_motion: bool = False,
        subthreshold_reference_jitter: bool = False,
    ) -> tuple[dict, list[np.ndarray]]:
        adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        _state, (eef_position, eef_quaternion) = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        eef_position = np.asarray(eef_position, dtype=np.float64)
        rotation = quat_to_mat_xyzw(eef_quaternion)
        if plane:
            offsets = np.asarray(
                [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.0, 0.10, 0.0]],
                dtype=np.float64,
            )
            names = ["face_a", "face_b", "face_c"]
            request = self._plane_request()
        else:
            offsets = np.asarray(
                [[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]],
                dtype=np.float64,
            )
            names = ["stick_a", "stick_b"]
            request = self._axis_request()
        controlled = {
            name: eef_position + rotation @ offset
            for name, offset in zip(names, offsets)
        }
        references = {
            "table_a": np.asarray([0.20, 0.20, 0.30], dtype=np.float64),
            "table_b": np.asarray([0.30, 0.20, 0.30], dtype=np.float64),
            "table_c": np.asarray([0.20, 0.30, 0.30], dtype=np.float64),
        }
        manager = _QuickTrackedManager(world, adapter, controlled, references)
        if reference_outlier:
            manager.reference_outlier_after_arm_motion_rad = 1.0e-4
            manager.reference_outlier_offsets = {
                "table_a": np.asarray([0.08, 0.04, -0.08], dtype=np.float64)
            }
        if coherent_reference_motion:
            manager.reference_outlier_after_arm_motion_rad = 1.0e-4
            manager.reference_outlier_offsets = {
                name: np.asarray([0.05, -0.02, 0.0], dtype=np.float64)
                for name in references
            }
        if subthreshold_reference_jitter:
            manager.reference_outlier_after_arm_motion_rad = 1.0e-4
            manager.reference_outlier_offsets = {
                "table_a": np.asarray([0.00010, -0.00004, 0.00002]),
                "table_b": np.asarray([-0.00006, 0.00003, -0.00001]),
                "table_c": np.asarray([0.00002, 0.00005, -0.00003]),
            }
        world._official_tracked_object_distances = manager
        ctx, result = official_test.OfficialMoveTrackedPointTest._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            actions = official_test.OfficialMoveTrackedPointTest._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    **request,
                    max_steps=360,
                    timeout_s=90.0,
                ),
            )
        return result, actions

    def test_execution_survives_one_off_hand_reacquisition_outlier(self) -> None:
        result, actions = self._run_quick_case(
            plane=False,
            reference_outlier=True,
        )

        self.assertTrue(result["ok"], result)
        continuity = result["tracking_continuity"]
        self.assertGreater(continuity["off_hand_tracking_outlier_steps"], 0)
        self.assertEqual(
            continuity["off_hand_tracking_outlier_names"], ["table_a"]
        )
        self.assertTrue(continuity["off_hand_reference_substitution_used"])
        final_reference = result["off_hand_final_validation"]
        self.assertTrue(final_reference["ok"], final_reference)
        self.assertTrue(
            final_reference["tracking_outlier_suppressed"], final_reference
        )
        self.assertFalse(final_reference["all_points_verified"], final_reference)
        self.assertGreater(len(actions), 0)

    def test_execution_still_rejects_coherent_off_hand_motion(self) -> None:
        result, actions = self._run_quick_case(
            plane=False,
            coherent_reference_motion=True,
        )

        self.assertFalse(result["ok"], result)
        self.assertEqual(result["motion"]["reason"], "off_hand_points_moved")
        last_reference = result["motion"]["last_step_monitor"][
            "off_hand_reference"
        ]
        self.assertFalse(last_reference["ok"], last_reference)
        self.assertFalse(
            last_reference["tracking_outlier_suppressed"], last_reference
        )
        self.assertGreater(len(actions), 0)

    def test_execution_uses_frozen_goal_after_subthreshold_rgbd_jitter(self) -> None:
        result, actions = self._run_quick_case(
            plane=False,
            subthreshold_reference_jitter=True,
        )

        self.assertTrue(result["ok"], result)
        final_reference = result["off_hand_final_validation"]
        self.assertTrue(final_reference["all_points_verified"], final_reference)
        self.assertEqual(
            final_reference["constraint_reference_source"],
            "frozen_off_hand_reference_verified_by_live_rgbd",
        )
        constraint_points = result["post_execution_track_validation"][
            "off_hand_constraint_reference_points_robot_base_m"
        ]
        live_points = result["final_live_points_robot_base_m"]
        for name in result["off_hand_point_names"]:
            np.testing.assert_allclose(
                constraint_points[name],
                result["source_points_robot_base_m"][name],
                atol=1.0e-12,
            )
        self.assertGreater(
            max(
                np.linalg.norm(
                    np.asarray(live_points[name], dtype=np.float64)
                    - np.asarray(constraint_points[name], dtype=np.float64)
                )
                for name in result["off_hand_point_names"]
            ),
            0.0,
        )
        self.assertGreater(len(actions), 0)

    def _run_satisfied_preset_case(
        self,
        quick_type: str,
        *,
        off_count: int,
        couple_coordinates: bool = False,
    ) -> tuple[dict, list[np.ndarray]]:
        """Exercise signing, loading, execution, and final live validation."""

        counts = {
            "touch": 1,
            "flatwise": 3,
            "plane_parallel": 3,
            "collinear": 2,
            "line_vertical_to_plane": 2,
        }
        on_count = counts[quick_type]
        request = self._request(
            quick_type,
            on_count=on_count,
            off_count=off_count,
        )
        adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        _state, (eef_position, _eef_quaternion) = (
            official_test.OfficialMoveTrackedPointTest._left_eef(world)
        )
        origin = np.asarray(eef_position, dtype=np.float64)
        on_names = request["quick_constraint"]["on_hand_points"]
        off_names = request["quick_constraint"]["off_hand_points"]

        if quick_type in {"flatwise", "plane_parallel"}:
            on_values = np.asarray(
                [origin, origin + [0.08, 0.0, 0.0], origin + [0.0, 0.08, 0.0]]
            )
        elif quick_type == "line_vertical_to_plane":
            on_values = np.asarray([origin, origin + [0.0, 0.0, 0.08]])
        elif quick_type == "collinear":
            on_values = np.asarray(
                [origin + [-0.06, 0.0, 0.0], origin + [0.06, 0.0, 0.0]]
            )
        else:
            on_values = np.asarray([origin])

        if quick_type in {"plane_parallel", "line_vertical_to_plane"}:
            off_values = np.asarray(
                [
                    origin + [0.15, 0.0, 0.0],
                    origin + [0.23, 0.0, 0.0],
                    origin + [0.15, 0.08, 0.0],
                ]
            )
        elif quick_type == "collinear" and off_count == 2:
            # Keep the finite reference segment deliberately disjoint.  The
            # default quick preset must drive the axial gap to zero; merely
            # placing both controlled points on the same infinite line is not
            # an executable placement goal.
            off_values = np.asarray(
                [origin + [0.10, 0.0, 0.0], origin + [0.18, 0.0, 0.0]]
            )
        elif quick_type == "collinear":
            off_values = np.asarray([origin])
        elif quick_type == "touch":
            # This is deliberately outside the public tolerance, so the shared
            # affine variables and the touch preset must jointly cause motion.
            off_values = np.asarray([origin + [0.04, 0.0, 0.0]])
        else:
            off_values = np.empty((0, 3), dtype=np.float64)

        if couple_coordinates:
            self.assertEqual(quick_type, "touch")
            for point in request["points"]:
                point["target_xyz_m"] = ["contact_x", "contact_y", "contact_z"]

        controlled = {
            name: on_values[index] for index, name in enumerate(on_names)
        }
        references = {
            name: off_values[index] for index, name in enumerate(off_names)
        }
        manager = _QuickTrackedManager(world, adapter, controlled, references)
        world._official_tracked_object_distances = manager
        ctx, result = official_test.OfficialMoveTrackedPointTest._ctx(world)
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            actions = official_test.OfficialMoveTrackedPointTest._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    **request,
                    max_steps=420,
                    timeout_s=90.0,
                ),
            )
        return result, actions

    def test_all_five_presets_complete_official_execution_and_live_validation(self) -> None:
        cases = (
            ("touch", 1, True),
            ("flatwise", 0, False),
            ("plane_parallel", 3, False),
            ("collinear", 1, False),
            ("collinear", 2, False),
            ("line_vertical_to_plane", 3, False),
        )
        for quick_type, off_count, couple_coordinates in cases:
            with self.subTest(quick_type=quick_type, off_count=off_count):
                result, actions = self._run_satisfied_preset_case(
                    quick_type,
                    off_count=off_count,
                    couple_coordinates=couple_coordinates,
                )
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["quick_constraint"]["type"], quick_type)
                self.assertTrue(result["final_constraints"]["ok"], result)
                self.assertTrue(
                    result["final_constraints"]["quick_constraint"]["ok"], result
                )
                if quick_type == "collinear" and off_count == 2:
                    quick_report = result["final_constraints"]["quick_constraint"]
                    self.assertEqual(
                        quick_report["axial_mode"], "ordered"
                    )
                    self.assertTrue(
                        quick_report["reference_inside_controlled"], quick_report
                    )
                    self.assertEqual(
                        quick_report["ordered_point_sequence"],
                        ["held_1", "scene_1", "scene_2", "held_2"],
                    )
                self.assertTrue(result["post_execution_track_validation"]["ok"], result)
                if off_count:
                    self.assertTrue(result["off_hand_final_validation"]["ok"], result)
                self.assertGreater(len(actions), 0)
                for action in actions:
                    action = np.asarray(action, dtype=np.float64)
                    self.assertEqual(action.shape, (ACTION_DIM,))
                    self.assertTrue(np.isfinite(action).all())
                    np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
                    if ARM_DOF == 8:
                        self.assertEqual(
                            float(action[ACTION_SLICES["arm_left"]][7]), 0.0
                        )

    def test_axis_perpendicular_quick_case_executes_and_validates_all_points(self) -> None:
        result, actions = self._run_quick_case(plane=False)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["quick_constraint"]["type"], "line_vertical_to_plane")
        self.assertEqual(result["on_hand_point_names"], ["stick_a", "stick_b"])
        self.assertEqual(
            result["off_hand_point_names"], ["table_a", "table_b", "table_c"]
        )
        self.assertTrue(result["off_hand_final_validation"]["ok"], result)
        self.assertTrue(result["post_execution_track_validation"]["ok"], result)
        self.assertGreater(len(result["final_live_points_robot_base_m"]), 0)
        self.assertGreater(len(actions), 2)
        for action in actions:
            action = np.asarray(action, dtype=np.float64)
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
            if ARM_DOF == 8:
                self.assertEqual(float(action[ACTION_SLICES["arm_left"]][7]), 0.0)

    def test_two_planes_parallel_quick_case_executes_without_changing_legacy_path(self) -> None:
        result, actions = self._run_quick_case(plane=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["quick_constraint"]["type"], "plane_parallel")
        self.assertTrue(result["final_constraints"]["ok"], result)
        self.assertTrue(result["off_hand_final_validation"]["ok"], result)
        self.assertEqual(result["point_count"], 6)
        self.assertEqual(result["trajectory"]["schema_version"] if "trajectory" in result else result["trajectory_schema_version"], official_tools.MOVE_TRACKED_POINT_TRAJECTORY_SCHEMA_VERSION)
        for action in actions:
            action = np.asarray(action, dtype=np.float64)
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())

    def test_two_collinear_groups_share_one_signed_plan_and_joint_validation(self) -> None:
        adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        _state, (eef_position, eef_quaternion) = (
            official_test.OfficialMoveTrackedPointTest._left_eef(world)
        )
        origin = np.asarray(eef_position, dtype=np.float64)
        rotation = quat_to_mat_xyzw(eef_quaternion)
        controlled_offsets = {
            "held_a": np.asarray([-0.04, 0.0, 0.0]),
            "held_b": np.asarray([0.04, 0.0, 0.0]),
            "held_c": np.asarray([0.0, -0.04, 0.0]),
            "held_d": np.asarray([0.0, 0.04, 0.0]),
        }
        controlled = {
            name: origin + rotation @ offset
            for name, offset in controlled_offsets.items()
        }
        references = {
            "scene_a": origin + rotation @ np.asarray([0.01, 0.0, 0.0]),
            "scene_b": origin + rotation @ np.asarray([0.0, 0.01, 0.0]),
        }
        manager = _QuickTrackedManager(world, adapter, controlled, references)
        world._official_tracked_object_distances = manager
        ctx, result = official_test.OfficialMoveTrackedPointTest._ctx(world)
        request = self._two_collinear_groups_request()
        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
        ):
            actions = official_test.OfficialMoveTrackedPointTest._drive(
                adapter,
                world,
                build_registry(adapter)["move_tracked_point"].fn(
                    ctx,
                    **request,
                    pos_tol=0.03,
                    ori_tol_deg=20.0,
                    max_steps=360,
                    timeout_s=90.0,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertIsNone(result["quick_constraint"])
        self.assertEqual(len(result["quick_constraints"]), 2)
        final_quick = result["final_constraints"]["quick_constraint"]
        self.assertTrue(final_quick["ok"], final_quick)
        self.assertEqual(final_quick["group_count"], 2)
        self.assertTrue(all(group["ok"] for group in final_quick["groups"]))
        self.assertEqual(
            [group["off_hand_point_names"] for group in final_quick["groups"]],
            [["scene_a"], ["scene_b"]],
        )
        self.assertTrue(result["post_execution_track_validation"]["ok"], result)
        self.assertGreater(len(actions), 0)
        for action in actions:
            action = np.asarray(action, dtype=np.float64)
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
            if ARM_DOF == 8:
                self.assertEqual(float(action[ACTION_SLICES["arm_left"]][7]), 0.0)


if __name__ == "__main__":
    unittest.main()
