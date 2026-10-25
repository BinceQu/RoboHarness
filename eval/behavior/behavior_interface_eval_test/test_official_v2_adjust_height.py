from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import numpy as np

from behavior_interface_eval_test.official_action_world import (
    ObservationBackedActionWorld,
)
from behavior_interface_eval_test.official_policy_interface import (
    ObservationActionAdapter,
)
from behavior_interface_eval_test.robot_contract import (
    ACTION_DIM,
    ACTION_SLICES,
    PROPRIO_DIM,
    PROPRIO_SLICES,
)
from behavior_interface_eval_test.tool.official_v2 import build_registry
from behavior_interface_eval_test.tool.official_v2 import (
    trunk_vertical_lift_local as vertical,
)
from behavior_interface_eval_test.tool.official_v2.tools import (
    MOVE_TO_REACH_PRE_LIFT_IMPLEMENTATION_VERSION,
    PUBLIC_TOOL_FUNCTIONS,
    _reach_pre_lift_request,
    _r1pro_shoulder_positions_robot,
    _yield_adjust_height_motion,
)


class OfficialV2AdjustHeightTest(unittest.TestCase):
    @staticmethod
    def _adapter_world(trunk_q):
        adapter = ObservationActionAdapter()
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["trunk_qpos"]] = np.asarray(
            trunk_q, dtype=np.float32
        )
        proprio[PROPRIO_SLICES["trunk_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        world = ObservationBackedActionWorld(
            proprio_provider=adapter.proprio_vector,
            hold_action_provider=adapter.hold_action,
            eef_pose_provider=adapter.eef_pose,
        )
        adapter.set_final_action_overlay(world.enforce_gripper_close_keepalive)
        world._official_adapter = adapter
        world.set_episode_initialized(True)
        return adapter, world

    @staticmethod
    def _ctx(world):
        result = {}
        ctx = SimpleNamespace(
            world=world,
            task_name="make_microwave_popcorn",
            set_result=lambda value: result.update(value),
            raise_if_cancelled=lambda where="": None,
        )
        return ctx, result

    @staticmethod
    def _drive_trunk(adapter, generator, *, follow_actions=True):
        actions = []
        previous = adapter.proprio_vector()[
            PROPRIO_SLICES["trunk_qpos"]
        ].astype(np.float64)
        for raw_action in generator:
            action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            if follow_actions:
                command = action[ACTION_SLICES["trunk"]].astype(np.float64)
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["trunk_qpos"]] = command
                proprio[PROPRIO_SLICES["trunk_qvel"]] = (
                    command - previous
                ) * 30.0
                if np.max(np.abs(command - previous)) <= 1e-7:
                    proprio[PROPRIO_SLICES["trunk_qvel"]] = 0.0
                adapter.update({"robot_r1::proprio": proprio})
                previous = command
        return actions

    @staticmethod
    def _row_at(z_target_m):
        rows = vertical.load_reverse_upward_lut()["rows"]
        return min(
            rows,
            key=lambda row: abs(float(row["z_target_m"]) - z_target_m),
        )

    def test_static_lut_has_validated_two_phase_workspace(self) -> None:
        lut = vertical.load_reverse_upward_lut()
        self.assertEqual(lut["source"], "submission_static_validated_tables")
        self.assertEqual(lut["n_phase1"], 43)
        self.assertEqual(lut["n_phase2"], 6)
        self.assertAlmostEqual(lut["z_max_m"], 1.15)
        self.assertAlmostEqual(lut["z_min_m"], 0.67)
        self.assertAlmostEqual(lut["z_boundary_m"], 0.73)
        self.assertEqual(lut["phase2_limited_by"], "q2_limit")
        self.assertEqual(len(lut["asset_digests"]), 2)

    def test_downward_plan_crosses_phase_boundary_and_preserves_verticality(
        self,
    ) -> None:
        start = np.asarray(self._row_at(1.15)["trunk_q"], dtype=np.float64)
        waypoints, meta = vertical.plan_absolute_height_trajectory(start, -0.45)

        self.assertEqual(meta["direction"], "down")
        self.assertFalse(meta["z_clamped"])
        self.assertEqual(sorted(set(meta["phases_used"])), [1, 2])
        self.assertAlmostEqual(meta["z_tgt_m"], 0.70)
        self.assertGreater(len(waypoints), 40)
        start_theta = vertical.torso_theta_z_deg(start)
        for waypoint in waypoints:
            self.assertTrue(
                bool(np.all(waypoint >= vertical.TRUNK_JOINT_LIMITS[:, 0]))
            )
            self.assertTrue(
                bool(np.all(waypoint <= vertical.TRUNK_JOINT_LIMITS[:, 1]))
            )
            self.assertEqual(float(waypoint[3]), 0.0)
            self.assertLess(
                abs(vertical.torso_theta_z_deg(waypoint) - start_theta),
                0.5,
            )

    def test_execution_sampling_uses_18_source_lut_waypoints(self) -> None:
        start = np.asarray(self._row_at(1.15)["trunk_q"], dtype=np.float64)
        waypoints, plan = vertical.plan_absolute_height_trajectory(start, -0.45)
        sampled, sampling = vertical.sample_lut_execution_waypoints(
            waypoints,
            plan["phases_used"],
        )

        self.assertGreater(len(waypoints), 18)
        self.assertEqual(len(sampled), 18)
        self.assertEqual(sampling["execution_waypoint_count"], 18)
        self.assertEqual(
            sampling["max_execution_waypoints"],
            vertical.TRUNK_LUT_MAX_EXEC_WAYPOINTS,
        )
        self.assertTrue(sampling["sampled"])
        self.assertTrue(sampling["preserves_phase_boundaries"])
        indices = sampling["source_indices"]
        self.assertEqual(indices, sorted(set(indices)))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], len(waypoints) - 1)
        for sampled_waypoint, source_index in zip(sampled, indices):
            np.testing.assert_allclose(
                sampled_waypoint,
                waypoints[source_index],
                atol=0.0,
            )
        for transition in sampling["phase_transition_source_indices"]:
            self.assertIn(transition - 1, indices)
            self.assertIn(transition, indices)
        np.testing.assert_allclose(sampled[-1], waypoints[-1], atol=0.0)
        sampled_commands = np.vstack(sampled)
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(sampled_commands, axis=0)))),
            0.161,
        )
        self.assertAlmostEqual(
            sampling["execution_max_joint_step_rad"],
            float(np.max(np.abs(np.diff(sampled_commands, axis=0)))),
        )

    def test_out_of_workspace_request_clamps_to_v2_lut_limit(self) -> None:
        start = np.array(
            [1.025012, -1.450305, -0.641406, 0.0],
            dtype=np.float64,
        )
        waypoints, meta = vertical.plan_absolute_height_trajectory(start, -0.40)

        self.assertTrue(meta["z_clamped"])
        self.assertAlmostEqual(meta["z_tgt_m"], 0.67)
        self.assertEqual(meta["phase2_limited_by"], "q2_limit")
        self.assertGreater(len(waypoints), 1)
        self.assertTrue(all(float(waypoint[3]) == 0.0 for waypoint in waypoints))
        self.assertTrue(meta["absolute_lut"])
        self.assertFalse(meta["relative_lut"])
        self.assertNotIn("lut_branch", meta)
        np.testing.assert_allclose(
            waypoints[0], meta["q_lut_start"], atol=1e-9
        )
        np.testing.assert_allclose(
            waypoints[-1], meta["q_lut_end"], atol=1e-9
        )
        self.assertLess(meta["predicted_upward_m"], 0.0)
        self.assertAlmostEqual(
            vertical.chest_height_lut_frame_m(waypoints[-1]),
            meta["z_predicted_m"],
            delta=1e-9,
        )
        self.assertLess(abs(meta["z_predicted_m"] - 0.67), 0.01)

    def test_live_off_manifold_start_executes_absolute_v2_lut(self) -> None:
        start = np.array(
            [1.1724210978, -1.1224188805, -0.4838452041, 0.0],
            dtype=np.float64,
        )
        waypoints, meta = vertical.plan_absolute_height_trajectory(start, -100.0)

        self.assertTrue(meta["z_clamped"])
        self.assertEqual(meta["direction"], "down")
        self.assertTrue(meta["absolute_lut"])
        self.assertFalse(meta["relative_lut"])
        self.assertNotIn("lut_branch", meta)
        self.assertAlmostEqual(meta["z_tgt_m"], 0.67)
        np.testing.assert_allclose(
            waypoints[-1],
            [-0.9915, 2.53068, 1.544, 0.0],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            waypoints[0], meta["q_lut_start"], atol=1e-9
        )
        self.assertLess(meta["predicted_upward_m"], -0.1)
        self.assertLess(float(waypoints[-1][0]), float(start[0]))
        self.assertGreater(float(waypoints[-1][1]), float(start[1]))
        self.assertGreater(float(waypoints[-1][2]), float(start[2]))

    def test_zero_height_request_holds_observed_posture(self) -> None:
        start = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        waypoints, meta = vertical.plan_absolute_height_trajectory(start, 0.0)

        self.assertEqual(len(waypoints), 1)
        np.testing.assert_allclose(waypoints[0], start, atol=1e-9)
        self.assertLessEqual(float(np.max(np.abs(waypoints[0] - start))), 5e-5)
        self.assertLess(abs(meta["z_tgt_m"] - meta["z_curr_m"]), 0.01)
        self.assertTrue(meta["absolute_lut"])
        self.assertFalse(meta["relative_lut"])
        self.assertTrue(meta["holds_observed_pose"])

    def test_upright_descent_uses_only_canonical_reverse_fold(self) -> None:
        start = np.zeros(4, dtype=np.float64)
        waypoints, meta = vertical.plan_absolute_height_trajectory(start, -1.0)

        self.assertNotIn("lut_branch", meta)
        self.assertLess(float(waypoints[-1][0]), 0.0)
        self.assertGreater(float(waypoints[-1][1]), 0.0)
        self.assertGreater(float(waypoints[-1][2]), 0.0)
        self.assertLess(meta["predicted_upward_m"], -0.44)

    def test_short_height_request_uses_exact_v2_absolute_lut_rows(self) -> None:
        start = np.array(
            [0.211126, 0.774916, 0.427086, 0.0],
            dtype=np.float64,
        )
        waypoints, meta = vertical.plan_absolute_height_trajectory(start, -0.01)

        self.assertGreaterEqual(len(waypoints), 2)
        np.testing.assert_allclose(
            waypoints[0], meta["q_lut_start"], atol=1e-9
        )
        np.testing.assert_allclose(
            waypoints[-1], meta["q_lut_end"], atol=1e-9
        )

    def test_upward_plan_replays_lut_in_reverse(self) -> None:
        start = np.asarray(self._row_at(0.70)["trunk_q"], dtype=np.float64)
        waypoints, meta = vertical.plan_absolute_height_trajectory(start, 0.30)

        self.assertEqual(meta["direction"], "up")
        self.assertEqual(sorted(set(meta["phases_used"])), [1, 2])
        self.assertAlmostEqual(meta["z_tgt_m"], 1.0)
        self.assertLess(meta["i_end"], meta["i_start"])
        self.assertGreater(len(waypoints), 20)

    def test_tool_executes_lut_and_requires_stable_proprio_convergence(self) -> None:
        start = np.asarray(self._row_at(1.15)["trunk_q"], dtype=np.float64)
        adapter, world = self._adapter_world(start)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)["adjust_height"].fn(
            ctx,
            upward=-0.45,
            timeout_s=30.0,
        )
        actions = self._drive_trunk(adapter, generator)

        self.assertTrue(result["ok"], result)
        self.assertEqual(
            result["execution"],
            "official_action_two_phase_absolute_lut",
        )
        self.assertEqual(
            sorted(set(result["planner"]["phases_used"])),
            [1, 2],
        )
        self.assertGreaterEqual(
            result["stable_steps"], result["required_stable_steps"]
        )
        self.assertLessEqual(
            result["max_abs_error_rad"], result["tolerance_rad"]
        )
        self.assertAlmostEqual(result["actual_upward_m"], -0.45, delta=0.02)
        self.assertTrue(result["trunk_yaw_locked"])
        self.assertTrue(actions)
        waypoints, plan = vertical.plan_absolute_height_trajectory(
            start,
            -0.45,
        )
        sampled, sampling = vertical.sample_lut_execution_waypoints(
            waypoints,
            plan["phases_used"],
        )
        self.assertEqual(len(sampled), 18)
        self.assertEqual(result["lut_action_steps"], len(sampled))
        self.assertEqual(result["settle_steps"], 5)
        self.assertEqual(result["action_steps"], 23)
        self.assertEqual(result["waypoint_hold_counts"], [1] * len(sampled))
        self.assertEqual(result["full_lut_waypoint_count"], len(waypoints))
        self.assertEqual(result["execution_lut_waypoint_count"], len(sampled))
        self.assertEqual(
            result["skipped_lut_waypoint_count"],
            len(waypoints) - len(sampled),
        )
        self.assertEqual(
            result["waypoint_sampling"]["source_indices"],
            sampling["source_indices"],
        )
        self.assertEqual(
            result["fixed_hold_equivalent_lut_action_steps"],
            len(sampled) * vertical.TRUNK_LUT_HOLD_ACTIONS,
        )
        self.assertEqual(
            result["saved_lut_action_steps"],
            len(sampled) * (vertical.TRUNK_LUT_HOLD_ACTIONS - 1),
        )
        self.assertEqual(
            result["full_fixed_hold_equivalent_lut_action_steps"],
            len(waypoints) * vertical.TRUNK_LUT_HOLD_ACTIONS,
        )
        self.assertEqual(
            result["saved_vs_full_fixed_hold_action_steps"],
            len(waypoints) * vertical.TRUNK_LUT_HOLD_ACTIONS - len(sampled),
        )
        commanded_waypoints = [
            action[ACTION_SLICES["trunk"]].astype(np.float64)
            for action in actions[: len(sampled)]
        ]
        for commanded, expected in zip(commanded_waypoints, sampled):
            np.testing.assert_allclose(commanded, expected, atol=1e-6)
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertEqual(float(action[ACTION_SLICES["trunk"]][3]), 0.0)

    def test_move_to_reach_pre_lift_reuses_adjust_height_action_sequence(
        self,
    ) -> None:
        start = np.asarray(self._row_at(1.15)["trunk_q"], dtype=np.float64)
        shoulders = _r1pro_shoulder_positions_robot(start)
        shoulder_mid = 0.5 * (shoulders["left"] + shoulders["right"])
        target_z = float(shoulder_mid[2]) - 0.80
        upward_m, pre_lift = _reach_pre_lift_request(target_z, start)

        self.assertTrue(pre_lift["applied"])
        self.assertAlmostEqual(upward_m, -0.40, places=7)
        self.assertEqual(
            pre_lift["planner"],
            MOVE_TO_REACH_PRE_LIFT_IMPLEMENTATION_VERSION,
        )

        private_adapter, private_world = self._adapter_world(start)
        private_ctx, _ = self._ctx(private_world)
        private_actions = self._drive_trunk(
            private_adapter,
            _yield_adjust_height_motion(
                private_ctx,
                upward=upward_m,
                timeout_s=30.0,
                tolerance=0.025,
                stage_prefix="move-to-reach pre-lift",
            ),
        )
        private_target = private_world.trunk_qpos()

        public_adapter, public_world = self._adapter_world(start)
        public_ctx, public_result = self._ctx(public_world)
        public_actions = self._drive_trunk(
            public_adapter,
            build_registry(public_adapter)["adjust_height"].fn(
                public_ctx,
                upward=upward_m,
                timeout_s=30.0,
                tol=0.025,
            ),
        )

        self.assertTrue(public_result["ok"], public_result)
        self.assertEqual(len(public_actions), len(private_actions) + 1)
        np.testing.assert_allclose(
            np.asarray(private_actions),
            np.asarray(public_actions[:-1]),
            atol=0.0,
        )
        np.testing.assert_allclose(
            private_target,
            public_result["target_trunk_q"],
            atol=1e-6,
        )
        self.assertLess(float(private_target[0]), float(start[0]))
        self.assertGreater(float(private_target[1]), float(start[1]))
        self.assertGreater(float(private_target[2]), float(start[2]))

        move_source = inspect.getsource(
            PUBLIC_TOOL_FUNCTIONS["move_to_reach_point"]
        )
        self.assertIn("_yield_adjust_height_motion", move_source)
        self.assertIn("_reach_pre_lift_request", move_source)
        self.assertNotIn("_reach_pre_lift_target", move_source)

    def test_move_to_reach_pre_lift_regression_for_15065_posture(self) -> None:
        start = np.array(
            [
                1.0251178741455078,
                -1.4501429796218872,
                -0.47001510858535767,
                0.0,
            ],
            dtype=np.float64,
        )
        target_z = 0.05311440934371059
        upward_m, pre_lift = _reach_pre_lift_request(target_z, start)
        waypoints, plan = vertical.plan_absolute_height_trajectory(
            start,
            upward_m,
        )
        target = waypoints[-1]

        self.assertTrue(pre_lift["applied"])
        self.assertTrue(plan["absolute_lut"])
        self.assertFalse(plan["relative_lut"])
        self.assertTrue(plan["z_clamped"])
        np.testing.assert_allclose(
            target,
            [-0.9915, 2.53068, 1.544, 0.0],
            atol=1e-9,
        )
        self.assertLess(plan["predicted_upward_m"], -0.2)
        self.assertGreater(float(target[1]), float(start[1]))
        self.assertLess(float(target[0]), float(start[0]))
        self.assertGreater(float(target[2]), float(start[2]))
        self.assertGreater(
            abs(float(target[0] + target[1] - start[0] - start[1])),
            0.5,
        )
        self.assertAlmostEqual(
            vertical.chest_height_lut_frame_m(target),
            plan["z_predicted_m"],
            delta=1e-9,
        )
        self.assertLess(abs(plan["z_predicted_m"] - 0.67), 0.01)

    def test_15065_repeated_pre_lift_holds_at_phase2_lower_limit(self) -> None:
        # Proprioception from job-1787916581696: the robot was already at the
        # final v2 LUT row, whose local-FK height is closer to the 0.68 label.
        start = np.array(
            [
                -0.9915049076080322,
                2.5306777954101562,
                1.5439999103546143,
                1.8189894035458565e-12,
            ],
            dtype=np.float64,
        )
        waypoints, plan = vertical.plan_absolute_height_trajectory(
            start,
            -0.5761566477280218,
        )

        self.assertEqual(plan["i_start_height_nearest"], 47)
        self.assertEqual(plan["i_start_joint_nearest"], 48)
        self.assertEqual(plan["i_start"], 48)
        self.assertEqual(plan["i_end"], 48)
        self.assertTrue(plan["starts_on_lut"])
        self.assertEqual(plan["requested_direction"], "down")
        self.assertEqual(plan["direction"], "hold")
        self.assertTrue(plan["z_clamped"])
        self.assertTrue(plan["saturated_hold"])
        self.assertEqual(len(waypoints), 1)
        np.testing.assert_allclose(waypoints[0], start, atol=1e-10)
        self.assertAlmostEqual(plan["predicted_upward_m"], 0.0, delta=1e-9)

    def test_tool_accepts_repeated_down_request_at_phase2_lower_limit(
        self,
    ) -> None:
        start = np.array(
            [-0.9915049, 2.5306778, 1.5439999, 0.0],
            dtype=np.float64,
        )
        adapter, world = self._adapter_world(start)
        ctx, result = self._ctx(world)
        actions = self._drive_trunk(
            adapter,
            build_registry(adapter)["adjust_height"].fn(
                ctx,
                upward=-1.0,
                timeout_s=30.0,
            ),
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["workspace_saturated"])
        self.assertTrue(result["planner"]["saturated_hold"])
        self.assertEqual(result["planner"]["direction"], "hold")
        self.assertTrue(result["direction_verified"])
        self.assertFalse(result["opposite_direction"])
        self.assertAlmostEqual(result["actual_upward_m"], 0.0, delta=1e-6)
        self.assertEqual(result["lut_action_steps"], 1)
        self.assertEqual(result["settle_steps"], 5)
        self.assertEqual(len(actions), 7)

    def test_tool_executes_large_downward_request_to_saturated_target(self) -> None:
        start = np.asarray(self._row_at(1.15)["trunk_q"], dtype=np.float64)
        adapter, world = self._adapter_world(start)
        ctx, result = self._ctx(world)
        actions = self._drive_trunk(
            adapter,
            build_registry(adapter)["adjust_height"].fn(
                ctx,
                upward=-100.0,
                timeout_s=30.0,
            ),
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["planner"]["z_clamped"])
        self.assertTrue(result["planner"]["absolute_lut"])
        self.assertFalse(result["planner"]["relative_lut"])
        self.assertNotIn("lut_branch", result["planner"])
        self.assertAlmostEqual(result["planner"]["z_tgt_m"], 0.67)
        self.assertLessEqual(result["z_end_m"], 0.69)
        self.assertLess(result["actual_upward_m"], -0.47)
        self.assertTrue(result["direction_verified"])
        self.assertTrue(result["height_converged"])
        self.assertEqual(
            result["lut_action_steps"],
            vertical.TRUNK_LUT_MAX_EXEC_WAYPOINTS,
        )
        self.assertGreater(
            result["planner"]["n_waypoints"],
            result["execution_lut_waypoint_count"],
        )
        self.assertEqual(result["execution_lut_waypoint_count"], 18)
        self.assertTrue(result["waypoint_sampling"]["preserves_source_endpoint"])
        self.assertTrue(actions)

    def test_tool_reports_stalled_proprio_instead_of_false_success(self) -> None:
        start = np.asarray(self._row_at(1.15)["trunk_q"], dtype=np.float64)
        adapter, world = self._adapter_world(start)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)["adjust_height"].fn(
            ctx,
            upward=-0.20,
            timeout_s=30.0,
        )
        actions = self._drive_trunk(
            adapter,
            generator,
            follow_actions=False,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["tracking_stalled"])
        self.assertFalse(result["timed_out"])
        self.assertIn("stalled", result["error"])
        self.assertEqual(
            result["lut_action_steps"],
            result["execution_lut_waypoint_count"],
        )
        self.assertEqual(
            result["waypoint_hold_counts"],
            [1] * result["execution_lut_waypoint_count"],
        )
        self.assertEqual(
            result["saved_lut_action_steps"],
            result["execution_lut_waypoint_count"]
            * (vertical.TRUNK_LUT_HOLD_ACTIONS - 1),
        )
        self.assertLess(len(actions), 100)

    def test_tool_honors_timeout_budget(self) -> None:
        start = np.asarray(self._row_at(1.15)["trunk_q"], dtype=np.float64)
        adapter, world = self._adapter_world(start)
        ctx, result = self._ctx(world)
        actions = self._drive_trunk(
            adapter,
            build_registry(adapter)["adjust_height"].fn(
                ctx,
                upward=-0.45,
                timeout_s=0.1,
            ),
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertLessEqual(result["action_steps"], 3)
        self.assertLessEqual(len(actions), 4)

    def test_local_planner_has_no_original_interface_dependency(self) -> None:
        source = inspect.getsource(vertical)
        self.assertNotIn("from behavior_interface", source)
        self.assertNotIn("import behavior_interface", source)
        self.assertNotIn("world.robot", source)
        self.assertNotIn("og.sim", source)


if __name__ == "__main__":
    unittest.main()
