"""Spatial row-order regression, independent of the live interface/simulator."""

import itertools
import json
import os
import shutil
import subprocess
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from behavior_interface_eval_test.tool.official_v2 import tools
from behavior_interface_eval_test.tool.official_v2.contract import (
    MOVE_TRACKED_POINT_ORDER_DESCRIPTION,
    coordinate_collinear_relations,
    validate_move_tracked_point_args,
)
from behavior_interface_eval_test.tool.official_v2.tracked_point_constraints_local import (
    evaluate_quick_constraint,
    evaluate_relations,
    quick_constraint_relation,
)
from behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local import (
    evaluate_target_constraints,
    plan_endpoint,
)
from behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local import LocalRobotState
from behavior_interface_eval_test.robot_contract import ARM_DOF


class MoveTrackedAxialOrderTest(unittest.TestCase):
    @staticmethod
    def request(order=("1", "3", "2")):
        return {"points": [
            {"name": name, "role": "off_hand" if name == "3" else "on_hand",
             "target_xyz_m": ["x", "y", f"z{name}"]}
            for name in order
        ]}

    def test_coordinate_mode_infers_order_without_changing_names_or_variables(self):
        for order in itertools.permutations(("1", "3", "2")):
            result = validate_move_tracked_point_args(self.request(order))
            self.assertEqual(result["relations"], [{"type": "ordered_collinear", "point_names": list(order)}])
            self.assertEqual(validate_move_tracked_point_args(result), result)

    def test_general_affine_line_inference_not_symbol_name_matching(self):
        points = [{"name": str(i), "target_xyz_m": [f"x+2*t{i}", f"y-3*t{i}", f"z+4*t{i}"]} for i in range(4)]
        self.assertEqual(coordinate_collinear_relations(points)[0]["point_names"], ["0", "1", "2", "3"])
        points[2]["target_xyz_m"][2] = "z+5*t2"
        self.assertEqual(coordinate_collinear_relations(points)[0]["point_names"], ["0", "1", "3"])
        self.assertEqual(coordinate_collinear_relations([
            {"name": str(i), "target_xyz_m": ["?", "?", "?"]} for i in range(3)
        ]), [])

    def test_plan_0165_was_collinear_but_not_in_requested_order(self):
        args = validate_move_tracked_point_args(self.request())
        targets = [{key: value for key, value in point.items() if key != "role"}
                   for point in args["points"] if point["role"] == "on_hand"]
        fixed = [{key: value for key, value in point.items() if key != "role"}
                 for point in args["points"] if point["role"] == "off_hand"]
        report = evaluate_target_constraints(
            [[.496169296539, -.010964459096, .466098082821],
             [.496169348029, -.010965568871, .572204681468]],
            targets, tolerance_m=.03, relations=args["relations"],
            fixed_points_robot_base_m=[[.49617095449, -.010961125073, .373599519808]],
            fixed_target_points=fixed,
        )
        self.assertFalse(report["ok"])
        self.assertFalse(report["axial_order_satisfied"])
        self.assertEqual(report["relations"][0]["observed_axial_order"], ["2", "1", "3"])
        self.assertGreater(report["relations"][0]["order_violation_m"], .19)

    def test_all_three_and_four_point_permutations_are_actually_checked(self):
        for count in (3, 4):
            names = [str(i) for i in range(count)]
            points = np.array([[.5, .1, .6 - .04*i] for i in range(count)])
            for order in itertools.permutations(names):
                report = evaluate_relations(points, [{"type": "ordered_collinear", "point_names": list(order)}], point_names=names)
                self.assertEqual(report["ok"], list(order) == names, report)

    def test_robot_reading_directions_and_ties(self):
        for direction, label in (([0, -1, 0], "y_left_to_right"), ([0, 0, -1], "z_top_to_bottom"),
                                 ([1, 0, 0], "x_near_to_far"), ([0, -1, 1], "y_left_to_right")):
            points = np.array([np.array(direction)*i*.02 for i in range(3)])
            report = evaluate_relations(points, [{"type": "ordered_collinear", "point_names": ["a", "b", "c"]}], point_names=["a", "b", "c"])
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["relations"][0]["reading_axis"], label)

    def test_order_is_categorical_even_with_large_position_tolerance(self):
        report = evaluate_relations([[0, 0, .1], [0, 0, .0499], [0, 0, .05]],
            [{"type": "ordered_collinear", "point_names": ["1", "3", "2"]}],
            point_names=["1", "3", "2"], position_tolerance_m=.1)
        self.assertFalse(report["ok"])
        self.assertFalse(report["axial_order_satisfied"])
        json.dumps(report, allow_nan=False)

    def test_quick_and_coordinate_modes_have_identical_order_semantics(self):
        request = self.request()
        request["quick_constraint"] = {"type": "collinear"}
        args = validate_move_tracked_point_args(request)
        quick = args["quick_constraint"]
        self.assertEqual(quick["axial_point_order"], ["1", "3", "2"])
        relation = quick_constraint_relation(quick, off_hand_points_robot_base_m=[[0, 0, .1]])
        self.assertEqual(relation["point_names"], ["1", "3", "2"])
        for on, expected in (([[0, 0, .15], [0, 0, .05]], True),
                             ([[0, 0, .05], [0, 0, .15]], False),
                             ([[0, 0, .25], [0, 0, .15]], False)):
            self.assertEqual(evaluate_quick_constraint(on, [[0, 0, .1]], quick)["ok"], expected)

    def test_group_order_is_not_reconstructed_from_roles(self):
        request = self.request()
        request["quick_constraints"] = [{"type": "collinear", "on_hand_points": ["1", "2"],
            "off_hand_points": ["3"], "axial_point_order": ["3", "2", "1"]}]
        self.assertEqual(validate_move_tracked_point_args(request)["quick_constraints"][0]["axial_point_order"], ["3", "2", "1"])

    def test_saved_endpoint_rechecks_order_even_with_forged_success_report(self):
        args = validate_move_tracked_point_args(self.request())
        target = {"requested_points": args["points"], "relations": args["relations"],
                  "on_hand_point_names": ["1", "2"], "off_hand_point_names": ["3"],
                  "resolved_points_robot_base_m": [[0, 0, .15], [0, 0, .05]],
                  "off_hand_source_points_robot_base_m": [[0, 0, .1]],
                  "constraints": {"ok": True, "axial_order_satisfied": True}}
        tools._validate_move_tracked_endpoint_order(target)
        target["resolved_points_robot_base_m"].reverse()
        with self.assertRaisesRegex(ValueError, "axial order"):
            tools._validate_move_tracked_endpoint_order(target)
        with self.assertRaisesRegex(ValueError, "axial order"):
            tools._compile_move_tracked_exec_plan_trajectory({
                "active_arm": "right", "start_state": {}, "target": target})
        target["relations"] = []
        with self.assertRaisesRegex(ValueError, "axial order"):
            tools._validate_move_tracked_endpoint_order(target)

    def test_mcp_catalog_description_is_refreshed_from_current_registry(self):
        from flask import Flask, jsonify
        from behavior_interface_eval_test.official_v2_hot_reload import _make_tools_metadata_view

        app = Flask(__name__)
        runtime = SimpleNamespace(
            _official_reload_lock=threading.RLock(),
            tool_registry={"move_tracked_point": SimpleNamespace(
                params=[], description=MOVE_TRACKED_POINT_ORDER_DESCRIPTION)},
            official_v2_reload_status=lambda: {"generation": 2, "ui": {}},
        )
        with app.test_request_context():
            view = _make_tools_metadata_view(runtime, lambda: jsonify({
                "tools": [{"name": "move_tracked_point", "desc": "stale description"}]}))
            self.assertEqual(view().get_json()["tools"][0]["desc"], MOVE_TRACKED_POINT_ORDER_DESCRIPTION)

    def test_legacy_mcp_metadata_refresh_uses_same_generation_and_is_idempotent(self):
        from flask import Flask, jsonify
        from behavior_interface_eval_test.live_move_tracked_point_test import (
            _install_committed_move_tools_description,
        )
        from behavior_interface_eval_test.tool.official_v2.registry import build_registry

        spec = build_registry(None)["move_tracked_point"]
        app = Flask(__name__)

        @app.get("/api/v2/tools")
        def metadata():
            return jsonify({"tools": [{"name": spec.name, "desc": "stale", "args": spec.params}]})

        with app.test_request_context():
            self.assertTrue(_install_committed_move_tools_description())
            self.assertTrue(_install_committed_move_tools_description())
        self.assertEqual(len(app.after_request_funcs[None]), 1)
        payload = app.test_client().get("/api/v2/tools").get_json()
        self.assertEqual(payload["tools"][0]["desc"], spec.description)
        self.assertIn(MOVE_TRACKED_POINT_ORDER_DESCRIPTION, spec.description)

    def test_frontend_preserves_each_groups_exact_rows(self):
        node = os.environ.get("BEHAVIOR_TEST_NODE") or shutil.which("node")
        if not node:
            self.skipTest("Node.js is required")
        source = Path(tools.__file__).parent / "assets" / "move_tracked_point_human_ui.js"
        script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const ctx={V2:{tool:{args:[{name:'points',widget:'tracked_target_points',min_points:1,max_points:6}]}},
window:{},console,setTimeout:()=>0,renderV2Args:()=>{},_v2CollectArgs:()=>({}),
_v2ValidateBody:()=>({ok:true}),escapeHtml:String};
vm.createContext(ctx); vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);
for(const order of [['1','3','2'],['3','1','2'],['2','4','3','1']]){
 ctx.V2.trackedConstraintBlocks=[{quickType:'collinear',rows:order.map(name=>({name,role:['1','2'].includes(name)?'on_hand':'off_hand',coordinates:''}))}];
 const body=ctx._v2CollectArgs();
 assert.strictEqual(JSON.stringify(body.quick_constraints[0].axial_point_order),JSON.stringify(order));
 assert(ctx._v2ValidateBody(ctx.V2.tool,body).ok);
}
"""
        result = subprocess.run([node, "-e", script, str(source)], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tv_door_coordinate_blocks_share_variables_and_preserve_axial_order(self):
        node = os.environ.get("BEHAVIOR_TEST_NODE") or shutil.which("node")
        if not node:
            self.skipTest("Node.js is required")
        source = Path(tools.__file__).parent / "assets" / "move_tracked_point_human_ui.js"
        script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const ctx={V2:{tool:{args:[{name:'points',widget:'tracked_target_points',min_points:1,max_points:6}]}},
window:{},console,setTimeout:()=>0,renderV2Args:()=>{},_v2CollectArgs:()=>({}),
_v2ValidateBody:()=>({ok:true}),escapeHtml:String};
vm.createContext(ctx); vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);
ctx.V2.trackedExecutionMode='plan';
ctx.V2.trackedConstraintBlocks=[
 {quickType:'',rows:[
  {name:'1',role:'on_hand',coordinates:'x y z'},
  {name:'4',role:'off_hand',coordinates:'x y b'},
  {name:'2',role:'on_hand',coordinates:'x y c'}]},
 {quickType:'',rows:[
  {name:'3',role:'on_hand',coordinates:'d e b'},
  {name:'4',role:'off_hand',coordinates:'x y b'}]}
];
const body=ctx._v2CollectArgs();
assert(ctx._v2ValidateBody(ctx.V2.tool,body).ok);
assert.strictEqual(JSON.stringify(body.points.map(p=>p.name)),JSON.stringify(['1','4','2','3']));
console.log(JSON.stringify(body));
"""
        output = subprocess.run([node, "-e", script, str(source)], capture_output=True,
                                text=True, timeout=15, check=True)
        normalized = validate_move_tracked_point_args(json.loads(output.stdout))
        self.assertEqual(normalized["execution_mode"], "plan")
        self.assertEqual(normalized["relations"], [
            {"type": "ordered_collinear", "point_names": ["1", "4", "2"]},
        ])
        by_name = {point["name"]: point for point in normalized["points"]}
        self.assertEqual(by_name["4"]["role"], "off_hand")
        self.assertEqual(by_name["3"]["target_xyz_m"]["z"], {"var": "b"})
        self.assertEqual(by_name["4"]["target_xyz_m"]["z"], {"var": "b"})

    @unittest.skipUnless(ARM_DOF == 8, "recorded evaluator fixture uses the 8-DOF robot")
    def test_recorded_132_request_has_a_precise_reachable_correctly_ordered_endpoint(self):
        q = np.array([-.152395814657, .163238376379, .28880366683, -2.073270320892,
                      .488339602947, -.845126926899, -.252543151379, 0.0])
        state = LocalRobotState(
            arm_dof=8, base_pos=np.zeros(3), base_quat=np.array([0., 0., 0., 1.]),
            trunk_q=np.array([-.99149876833, 2.530690193177, .322886526585, 0.]),
            arm_left_q=np.array([0., 0., 0., -2.0943956375, 0., -1.047198057, 0., 0.]),
            arm_right_q=q, gripper_left_q=np.array([.04752751, .04849127]),
            gripper_right_q=np.array([.04890143, .04999947]), robot_forward=np.array([1., 0., 0.]),
        )
        args = validate_move_tracked_point_args(self.request())
        targets = tools._move_tracked_planning_targets(args["points"], ["1", "2"])
        fixed = tools._move_tracked_planning_targets(args["points"], ["3"])
        result = plan_endpoint(
            state=state, arm="right", q_start=q,
            source_points_robot_base_m=[[.434132352695, -.099795635965, .522675662623],
                                       [.433669890427, -.205752407137, .517057918489]],
            target_points=targets, relations=args["relations"],
            fixed_target_points=fixed,
            fixed_points_robot_base_m=[[.49617095449, -.010961125073, .373599519808]],
            pos_tol_m=.03, ori_tol_deg=20., planning_deadline_monotonic=time.monotonic()+45,
        )
        report = result["constraints"]
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["axial_order_satisfied"], report)
        self.assertEqual(report["relations"][0]["observed_axial_order"], ["1", "3", "2"])
        self.assertLessEqual(report["max_collinear_error_m"], .001)


if __name__ == "__main__":
    unittest.main()
