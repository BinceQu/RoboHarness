from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from flask import Flask, jsonify

from behavior_interface_eval_test.official_v2_hot_reload import _make_tools_metadata_view
from behavior_interface_eval_test.robot_contract import ACTION_DIM, ACTION_SLICES, ARM_DOF
import behavior_interface_eval_test.test_move_tracked_point_quick as quick_test
import behavior_interface_eval_test.test_official_move_tracked_point as official_test
from behavior_interface_eval_test.tool.official_v2 import tools
from behavior_interface_eval_test.tool.official_v2.contract import validate_move_tracked_point_args
from behavior_interface_eval_test.tool.official_v2.grasp_geometry_local import quat_to_mat_xyzw
from behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local import eef_pose
from behavior_interface_eval_test.tool.official_v2.registry import build_registry
from behavior_interface_eval_test.tool.official_v2.tracked_point_constraints_local import (
    evaluate_quick_constraint,
    evaluate_relations,
    quick_constraint_relation,
    relation_residual_blocks,
)


class MoveTrackedVerticalToGroundTest(unittest.TestCase):
    @staticmethod
    def request(count=2, *, off_count=0):
        return quick_test.MoveTrackedPointQuickTest._request(
            "vertical_to_ground", on_count=count, off_count=off_count
        )

    def evaluate(self, points, **kwargs):
        return evaluate_quick_constraint(
            points, None, self.request(len(points))["quick_constraint"], **kwargs
        )

    def test_public_aliases_counts_and_optional_coordinates(self):
        for count in (2, 3):
            for alias in ("vertical_to_ground", "vertical-to-ground", "vertical to ground"):
                request = self.request(count)
                request["quick_constraint"]["type"] = alias
                request["points"][0]["target_xyz_m"] = [0.4, "shared_y", "z+0.07"]
                result = validate_move_tracked_point_args(request)
                self.assertEqual(result["quick_constraint"]["type"], "vertical_to_ground")
                self.assertEqual(result["points"][0]["target_xyz_m"]["x"], 0.4)
                self.assertEqual(result["points"][0]["target_xyz_m"]["y"], {"var": "shared_y"})
                self.assertEqual(result["points"][0]["target_xyz_m"]["z"], {"expr": "z+0.07"})

    def test_invalid_counts_roles_orientation_and_fields_are_rejected(self):
        bad_requests = [self.request(n) for n in (0, 1, 4, 5, 6)]
        bad_requests.extend(self.request(n, off_count=1) for n in (2, 3))
        for key, value in (("sense", "same"), ("sense", "opposite"), ("axial_mode", "line_only")):
            request = self.request()
            request["quick_constraint"][key] = value
            bad_requests.append(request)
        for request in bad_requests:
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_move_tracked_point_args(request)

    def test_two_points_require_xy_equality_but_no_height_or_direction(self):
        for first_z, second_z in ((-0.4, 0.7), (0.7, -0.4)):
            report = self.evaluate([[0.3, -0.2, first_z], [0.3, -0.2, second_z]])
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["xy_difference_m"], [0.0, 0.0])
            self.assertEqual(report["max_error_deg"], 0.0)
        report = self.evaluate([[0, 0, 0], [0.04, 0, 2.0]], position_tolerance_m=0.03)
        self.assertFalse(report["ok"], report)
        self.assertAlmostEqual(report["max_error_m"], 0.04)
        self.assertLess(report["max_error_deg"], 5.0)

    def test_two_point_short_axis_also_enforces_orientation(self):
        report = self.evaluate([[0, 0, 0], [0.001, 0, 0]], position_tolerance_m=0.03)
        self.assertFalse(report["ok"], report)
        self.assertAlmostEqual(report["max_error_m"], 0.001)
        self.assertAlmostEqual(report["max_error_deg"], 90.0)

    def test_three_points_allow_every_horizontal_normal_and_point_order(self):
        origin = np.array([0.3, -0.2, 0.7])
        for heading in np.linspace(-math.pi, math.pi, 9):
            tangent = 0.1 * np.array([math.cos(heading), math.sin(heading), 0.0])
            points = np.array([origin, origin + tangent, origin + [0, 0, 0.1]])
            for order in ((0, 1, 2), (0, 2, 1), (2, 0, 1)):
                report = self.evaluate(points[list(order)])
                self.assertTrue(report["ok"], report)
                self.assertAlmostEqual(report["normal_z"], 0.0)
                self.assertAlmostEqual(report["max_error_deg"], 0.0)

    def test_horizontal_plane_is_not_vertical_and_tilt_is_measured_in_degrees(self):
        horizontal = self.evaluate([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
        self.assertFalse(horizontal["ok"], horizontal)
        self.assertAlmostEqual(horizontal["max_error_deg"], 90.0)
        for angle_deg in (-27.0, -4.0, 4.0, 27.0):
            angle = math.radians(angle_deg)
            points = [[0, 0, 0], [0.1, 0, 0], [0, 0.1 * math.sin(angle), 0.1 * math.cos(angle)]]
            report = self.evaluate(points, orientation_tolerance_deg=5.0)
            self.assertEqual(report["ok"], abs(angle_deg) <= 5.0)
            self.assertAlmostEqual(report["max_error_deg"], abs(angle_deg))

    def test_degenerate_or_nonfinite_geometry_cannot_pass(self):
        for points in (
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]],
            [[0, 0, 0], [0.1, 0, 0], [0.2, 1e-14, 0]],
        ):
            report = self.evaluate(points)
            self.assertFalse(report["ok"], report)
            self.assertTrue(report["degenerate"])
            json.dumps(report, allow_nan=False)
        with self.assertRaises(ValueError):
            self.evaluate([[0, 0, 0], [math.nan, 0, 0.1]])
        with np.errstate(over="ignore", invalid="ignore"):
            for points in ([[0, 0, 0], [1e200, 0, 0]], [[0, 0, 0], [1e200, 0, 0], [0, 1e200, 0]]):
                with self.assertRaises(ValueError):
                    self.evaluate(points)

    def test_planning_and_live_metrics_match_without_fixing_heading(self):
        for points in (
            [[0, 0, 0], [0.005, 0.01, 0.1]],
            [[0, 0, 0], [0.1, 0, 0], [0, 0.02, 0.1]],
        ):
            quick = self.request(len(points))["quick_constraint"]
            relation = quick_constraint_relation(quick)
            self.assertEqual(set(relation), {"type", "point_names"})
            planned = evaluate_relations(points, [relation], point_names=quick["on_hand_points"])
            live = self.evaluate(points)
            self.assertEqual(planned["ok"], live["ok"])
            self.assertAlmostEqual(planned["max_relation_error_m"], live["max_error_m"])
            self.assertAlmostEqual(planned["max_relation_error_deg"], live["max_error_deg"])

    def test_invalid_typed_relation_and_off_hand_data_are_rejected(self):
        for count in (1, 4):
            names = [str(i) for i in range(count)]
            with self.assertRaises(ValueError):
                relation_residual_blocks(np.zeros((count, 3)), [{
                    "type": "vertical_to_ground", "point_names": names,
                }], point_names=names)
        quick = self.request()["quick_constraint"]
        with self.assertRaises(ValueError):
            quick_constraint_relation(quick, off_hand_points_robot_base_m=[[0, 0, 0]])
        with self.assertRaises(ValueError):
            evaluate_quick_constraint([[0, 0, 0], [0, 0, 0.1]], [[0, 0, 0]], quick)

    def test_vertical_preset_and_other_groups_use_their_own_point_subsets(self):
        request = self.request(3, off_count=1)
        request.pop("quick_constraint")
        request["quick_constraints"] = [
            {"type": "vertical_to_ground", "on_hand_points": ["held_1", "held_2"], "off_hand_points": []},
            {"type": "touch", "on_hand_points": ["held_3"], "off_hand_points": ["scene_1"]},
        ]
        normalized = validate_move_tracked_point_args(request)
        controlled = np.array([[0, 0, 0], [0, 0, 0.1], [0.1, 0, 0]])
        references = np.array([[0.1, 0, 0]])
        for delta, expected in ((0.0, True), (0.1, False)):
            points = controlled.copy()
            points[1, 0] += delta
            report = tools._move_tracked_evaluate_quick_constraints(
                points, references, normalized["quick_constraints"],
                controlled_names=["held_1", "held_2", "held_3"],
                reference_names=["scene_1"],
                position_tolerance_m=0.03, orientation_tolerance_deg=20.0,
            )
            self.assertEqual(report["ok"], expected, report)
            self.assertTrue(report["groups"][1]["ok"], report)

    def test_hot_reload_replaces_old_quick_preset_metadata(self):
        import threading

        app = Flask(__name__)
        runtime = SimpleNamespace(
            _official_reload_lock=threading.RLock(),
            tool_registry=build_registry(None),
            official_v2_reload_status=lambda: {"generation": 1},
        )

        def old_metadata():
            return jsonify({"tools": [{"name": "move_tracked_point", "args": [
                {"name": "points", "widget": "tracked_target_points", "quick_presets": []},
                {"name": "quick_constraint", "enum_types": ["touch"]},
                {"name": "quick_constraints", "items": {"properties": {"type": {"enum": ["touch"]}}}},
            ]}]})

        view = _make_tools_metadata_view(runtime, old_metadata)
        with app.test_request_context():
            args = {arg["name"]: arg for arg in view().get_json()["tools"][0]["args"]}
        vertical = next(item for item in args["points"]["quick_presets"] if item["type"] == "vertical_to_ground")
        self.assertEqual(vertical, {"type": "vertical_to_ground", "on_hand_count": [2, 3], "off_hand_count": 0})
        self.assertEqual(args["points"]["widget"], "tracked_target_points")
        self.assertIn("vertical_to_ground", args["quick_constraint"]["enum_types"])
        self.assertIn("vertical_to_ground", args["quick_constraints"]["items"]["properties"]["type"]["enum"])

    @staticmethod
    def reachable_case(count):
        adapter, world = official_test.OfficialMoveTrackedPointTest._ready_world()
        state, (start_position, start_quaternion) = official_test.OfficialMoveTrackedPointTest._left_eef(world)
        goal_q = np.asarray(world.arm_qpos_list("left"), dtype=float).copy()
        goal_q[:3] += [0.2, -0.12, 0.1]
        goal_position, goal_quaternion = eef_pose(state, "left", goal_q)
        offsets = np.array([[0, 0, 0], [0, 0, 0.1]] if count == 2 else [
            [0, 0, 0], [0.08, 0.06, 0], [0, 0, 0.1],
        ])
        anchors = offsets @ quat_to_mat_xyzw(goal_quaternion)
        source = np.asarray(start_position)[None, :] + anchors @ quat_to_mat_xyzw(start_quaternion).T
        return adapter, world, source, np.asarray(goal_position) + offsets

    def test_known_reachable_axis_and_plane_execute_and_validate_live_points(self):
        for count in (2, 3):
            with self.subTest(count=count):
                adapter, world, source, goal_points = self.reachable_case(count)
                request = self.request(count)
                self.assertFalse(self.evaluate(source, orientation_tolerance_deg=1.0)["ok"])
                request["points"][0]["target_xyz_m"] = goal_points[0].tolist()
                names = request["quick_constraint"]["on_hand_points"]
                manager = quick_test._QuickTrackedManager(world, adapter, dict(zip(names, source)), {})
                world._official_tracked_object_distances = manager
                ctx, result = official_test.OfficialMoveTrackedPointTest._ctx(world)
                initial_right = np.array(world.arm_qpos_list("right"))
                with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": root}):
                    actions = official_test.OfficialMoveTrackedPointTest._drive(adapter, world, build_registry(adapter)["move_tracked_point"].fn(
                        ctx, **request, max_steps=420, timeout_s=90.0,
                    ))
                self.assertTrue(result["ok"], result)
                self.assertTrue(result["final_constraints"]["quick_constraint"]["ok"], result)
                self.assertTrue(result["post_execution_track_validation"]["ok"], result)
                final = np.array([result["final_live_points_robot_base_m"][name] for name in names])
                self.assertTrue(self.evaluate(final, position_tolerance_m=0.003, orientation_tolerance_deg=5.0)["ok"])
                self.assertLess(np.max(np.abs(final[0] - goal_points[0])), 0.003)
                self.assertGreater(np.max(np.abs(final - source)), 0.005)
                for action in actions:
                    self.assertEqual(np.shape(action), (ACTION_DIM,))
                    self.assertTrue(np.isfinite(action).all())
                    np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
                    np.testing.assert_allclose(action[ACTION_SLICES["arm_right"]], initial_right)
                    if ARM_DOF == 8:
                        self.assertEqual(float(action[ACTION_SLICES["arm_left"]][7]), 0.0)

    def test_javascript_serializes_and_validates_new_and_existing_presets(self):
        node = os.environ.get("BEHAVIOR_TEST_NODE") or shutil.which("node") or shutil.which("nodejs")
        if not node:
            self.skipTest("Node.js is required for the frontend execution test")
        asset = Path(tools.__file__).parent / "assets" / "move_tracked_point_human_ui.js"
        script = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const context = {
  V2: {tool: {args: [{name: 'points', widget: 'tracked_target_points', min_points: 1, max_points: 6}]}},
  window: {}, console, setTimeout: () => 0,
  renderV2Args: () => {}, _v2CollectArgs: () => ({}),
  _v2ValidateBody: () => ({ok: true}), escapeHtml: String,
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
assert.strictEqual(context.V2.moveTrackedPointUiBuild, 'tracked_target_rows_v17_inequalities');
const cases = [
  ['touch', 1, 1, true], ['flatwise', 3, 0, true],
  ['plane_parallel', 3, 3, true], ['collinear', 2, 1, true],
  ['line_vertical_to_plane', 2, 3, true],
  ['vertical_to_ground', 2, 0, true], ['vertical_to_ground', 3, 0, true],
  ['faceto', 3, 0, true], ['reverse_faceto', 3, 0, true],
  ['vertical_to_ground', 1, 0, false], ['vertical_to_ground', 4, 0, false],
  ['vertical_to_ground', 2, 1, false],
];
for (const [type, onCount, offCount, valid] of cases) {
  const rows = Array.from({length: onCount + offCount}, (_, i) => ({
    name: `p${i}`, role: i < onCount ? 'on_hand' : 'off_hand',
    coordinates: i === 0 ? '0.4 y z+0.07' : '',
  }));
  context.V2.trackedConstraintBlocks = [{quickType: type, rows}];
  const body = context._v2CollectArgs();
  assert.strictEqual(context._v2ValidateBody(context.V2.tool, body).ok, valid, JSON.stringify(body));
  assert.strictEqual(body.quick_constraints[0].type, type);
  assert.strictEqual(body.points[0].target_xyz_m.x, 0.4);
  assert.strictEqual(body.points[0].target_xyz_m.y.var, 'y');
  assert.strictEqual(body.points[0].target_xyz_m.z.expr, 'z+0.07');
}
console.log('frontend preset round trips passed');
"""
        completed = subprocess.run([node, "-e", script, str(asset)], text=True, capture_output=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_plan_preview_overlap_fallback_and_exec_plan_pose_share_vertical_goal(self):
        adapter, world, source, goal = self.reachable_case(3)
        request = self.request(3)
        request["points"][0]["target_xyz_m"] = goal[0].tolist()
        names = request["quick_constraint"]["on_hand_points"]
        manager = quick_test._QuickTrackedManager(world, adapter, dict(zip(names, source)), {})
        world._official_tracked_object_distances = manager
        initial_q = np.array(world.arm_qpos_list("left"))
        capture = tools._FrozenCapture(
            session_id=manager.session_id, image_id=manager.image_id, role="head",
            depth=np.ones((24, 32), dtype=np.float32),
            camera={"robot_relative_pose": {"pos": [0.45, 0, 0.6], "quat": [0, 0, 0, 1]}},
            evaluator_sequence=int(adapter.status()["sequence"]),
        )

        def overlap_evaluation(*, poses, **kwargs):
            report = official_test.OfficialMoveTrackedPointTest._safe_overlap_evaluation(poses=poses, **kwargs)
            for pose in report["poses"]:
                pose.update({
                    "ok": False, "overlap_vox": 6, "overlap_vol_cm3": 0.162,
                    "original_overlap_ok": False, "inflated_overlap_vox": None,
                    "inflated_overlap_vol_cm3": None, "inflated_overlap_ok": False,
                    "inflated_query_skipped": True,
                    "inflated_query_skipped_reason": "original_overlap_failed",
                })
            report.update({"passing_pose_count": 0, "rejected_pose_count": len(poses), "passing_pose_indices": []})
            return report

        def render(**kwargs):
            path = Path(kwargs["output_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG\r\n\x1a\nvertical-plan-test")
            return {"ok": True, "path": str(path), "visible_pixel_count": 64}

        ctx, result = official_test.OfficialMoveTrackedPointTest._ctx(world)
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": root}), mock.patch.object(
            tools, "_run_tracked_plan_worker",
            side_effect=lambda kwargs, **options: tools._move_tracked_plan_frozen(**kwargs),
        ), mock.patch.object(
            tools, "_move_tracked_freeze_plan_capture",
            return_value={"image_id": "img_plan_frozen", "observation_sequence": capture.evaluator_sequence},
        ), mock.patch.object(
            tools, "_load_frozen_capture", return_value=capture
        ), mock.patch.object(
            tools, "_move_tracked_plan_overlap_context",
            return_value=official_test.OfficialMoveTrackedPointTest._safe_overlap_context(),
        ), mock.patch.object(
            tools, "_evaluate_rgbd_lite_pose_overlaps", side_effect=overlap_evaluation
        ), mock.patch.object(tools, "render_plan_gripper_overlay", side_effect=render) as overlay:
            official_test.OfficialMoveTrackedPointTest._drive(adapter, world, build_registry(adapter)["move_tracked_point"].fn(
                ctx, **request, execution_mode="plan", max_steps=420,
            ))
            self.assertTrue(result["ok"], result)
            self.assertTrue(result["overlap_fallback_applied"], result)
            self.assertEqual(overlay.call_args.kwargs["eef_pose"], result["eef_pose"])
            np.testing.assert_allclose(world.arm_qpos_list("left"), initial_q, atol=1e-6)
            trajectory, _, record = tools._load_plan_trajectory(manager.session_id, result["plan_id"], include_record=True)
            self.assertEqual(record["move_tracked_trajectory"]["target"]["quick_constraint"]["type"], "vertical_to_ground")
            self.assertTrue(trajectory["safety_validation"]["overlap_fallback_applied"])
            exec_ctx, exec_result = official_test.OfficialMoveTrackedPointTest._ctx(world)
            official_test.OfficialMoveTrackedPointTest._drive(adapter, world, build_registry(adapter)["exec_plan_pose"].fn(
                exec_ctx, session_id=manager.session_id, plan_id=result["plan_id"],
            ))
            self.assertTrue(exec_result["ok"], exec_result)
            snapshot = manager.observed_active_points_snapshot(names, episode_id=world.episode_id)
            final_points = [snapshot["entries"][name]["xyz_in_robot_base_coord_m"] for name in names]
            self.assertTrue(self.evaluate(final_points, orientation_tolerance_deg=5.0)["ok"])


if __name__ == "__main__":
    unittest.main()
