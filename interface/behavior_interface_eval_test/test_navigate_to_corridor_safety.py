"""Independent corridor and certificate regressions for ``navigate_to``."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import behavior_interface_eval_test.tool.official_v2.tools as official_v2_tools
import behavior_interface_eval_test.test_navigate_to_execution as _execution_test
from behavior_interface_eval_test.robot_contract import ACTION_SLICES
from behavior_interface_eval_test.tool.official_v2 import build_registry


GOAL_NAME = _execution_test.GOAL_NAME


class NavigateToCorridorSafetyTest(unittest.TestCase):
    """Exercise signed-corridor checks through the public tool boundary."""

    def test_new_geometry_revision_does_not_hide_slam_cross_track_error(self) -> None:
        _adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(world)
        ctx, _results, _setter = _execution_test.NavigateToExecutionTest._ctx(world)
        x, y, _yaw = official_v2_tools._navigate_to_pose(snapshot)
        trajectory = {
            "path_xy_m": [[x, y], [x + 2.0, y]],
            "path_segment_clearance_m": [0.9],
            "clearance": {"required_clearance_m": 0.5, "robot_radius_m": 0.42},
        }
        world._mock_base[:] = [0.0, 0.0, 0.0]
        monitor = official_v2_tools._navigate_to_new_segment_monitor(ctx, snapshot)
        recovery_monitor = official_v2_tools._navigate_to_new_segment_monitor(
            ctx,
            snapshot,
            policy_relative_motion=True,
        )
        snapshot["map_version"] = "new-occupancy-and-trail"
        snapshot["pose"].update(x=x + 0.5, y=y + 0.31)
        world._mock_base[:] = [0.5, 0.0, 0.0]
        report = official_v2_tools._navigate_to_monitored_corridor(
            ctx, trajectory, snapshot, 1, monitor,
        )
        self.assertFalse(report["ok"])
        self.assertTrue(report["slam_pose_checked"])
        self.assertTrue(report["map_version_changed"])
        self.assertEqual(report["monitor_source"], "slam")
        self.assertAlmostEqual(report["max_cross_track_m"], 0.31)

        recovery_report = official_v2_tools._navigate_to_monitored_corridor(
            ctx, trajectory, snapshot, 1, recovery_monitor,
        )
        self.assertTrue(recovery_report["ok"], recovery_report)
        self.assertEqual(
            recovery_report["monitor_policy"],
            "policy_relative_recovery",
        )
        self.assertEqual(recovery_report["monitor_source"], "policy_odometry")
        self.assertFalse(recovery_report["slam_corridor_ok"])
        self.assertAlmostEqual(recovery_report["slam_deviation_m"], 0.31)
        self.assertAlmostEqual(recovery_report["max_cross_track_m"], 0.0)

    def test_direct_history_replay_uses_atomic_policy_motion_frame(self) -> None:
        _adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(world)
        ctx, _results, _setter = _execution_test.NavigateToExecutionTest._ctx(world)
        trajectory = {
            "planner": {"traversed_route_direct_replay": True},
        }
        self.assertTrue(
            official_v2_tools._navigate_to_uses_traversed_motion_frame(
                trajectory
            )
        )
        monitor = official_v2_tools._navigate_to_new_segment_monitor(
            ctx,
            snapshot,
            policy_relative_motion=True,
        )
        snapshot["pose"]["x"] = float(snapshot["pose"]["x"]) + 0.10
        self.assertTrue(
            official_v2_tools._navigate_to_policy_pose_matches_monitor(
                ctx, monitor
            )
        )

        world._mock_base[0] += 0.03
        self.assertFalse(
            official_v2_tools._navigate_to_policy_pose_matches_monitor(
                ctx, monitor
            )
        )

    def _assert_action_contract(self, actions) -> None:
        _execution_test.NavigateToExecutionTest._assert_action_contract(
            self, actions
        )

    @staticmethod
    def _conservative_plan(real_planner, snapshot, name, **kwargs):
        plan = deepcopy(real_planner(snapshot, name, **kwargs))
        if plan.get("ok") is True:
            plan["path_segment_clearance_m"] = [
                float(plan["required_clearance_m"])
                for _ in plan["path_segment_clearance_m"]
            ]
        return plan

    @staticmethod
    def _exact_controller(state, calls, *, after_move=None):
        """Move the fixture pose by the exact body-frame command after one action."""

        def adjust(
            ctx,
            forward=0.0,
            translation=0.0,
            spin=0.0,
            **kwargs,
        ):
            call_index = len(calls)
            call = {
                "forward": float(forward),
                "translation": float(translation),
                "spin": float(spin),
                "length_m": math.hypot(float(forward), float(translation)),
                "map_version": str(state["snapshot"]["map_version"]),
                "kwargs": dict(kwargs),
                "entered": False,
            }
            calls.append(call)

            def phase():
                call["entered"] = True
                base = [
                    float(np.clip(float(forward), -0.2, 0.2)),
                    float(np.clip(float(translation), -0.2, 0.2)),
                    float(np.clip(math.radians(float(spin)), -0.2, 0.2)),
                ]
                yield ctx.world.make_action(base=base)

                pose = state["snapshot"]["pose"]
                yaw_rad = math.radians(float(pose.get("yaw_deg", 0.0)))
                pose["x"] = float(pose["x"]) + (
                    math.cos(yaw_rad) * float(forward)
                    - math.sin(yaw_rad) * float(translation)
                )
                pose["y"] = float(pose["y"]) + (
                    math.sin(yaw_rad) * float(forward)
                    + math.cos(yaw_rad) * float(translation)
                )
                pose["yaw_deg"] = float(pose.get("yaw_deg", 0.0)) + float(spin)
                if after_move is not None:
                    after_move(call_index, state, ctx.world)
                _execution_test.NavigateToExecutionTest._advance_snapshot_pose(
                    state["snapshot"]
                )
                return {
                    "ok": True,
                    "action_steps": 1,
                    "linear_remaining_m": 0.0,
                    "obstacle_stop_reason": None,
                }

            return phase()

        return adjust

    def test_certificate_alignment_and_floor_fail_before_motion(self) -> None:
        real_planner = official_v2_tools.plan_clearance_path
        cases = (
            ("wrong_length", "must align"),
            ("below_required", "below required clearance"),
            ("relaxed_margin", "fixed clearance policy"),
        )
        for mutation, expected_error in cases:
            with self.subTest(mutation=mutation):
                adapter, world = (
                    _execution_test.NavigateToExecutionTest._adapter_world()
                )
                snapshot = _execution_test.NavigateToExecutionTest._snapshot(
                    world, goal_cell=(30, 15)
                )
                _execution_test.NavigateToExecutionTest._install_snapshot_provider(
                    adapter, snapshot
                )
                ctx, results, setter = (
                    _execution_test.NavigateToExecutionTest._ctx(world)
                )
                controller = mock.Mock()

                def invalid_certificate(current, name, **kwargs):
                    plan = deepcopy(real_planner(current, name, **kwargs))
                    self.assertTrue(plan["ok"], plan)
                    if mutation == "wrong_length":
                        plan["path_segment_clearance_m"] = []
                    else:
                        if mutation == "below_required":
                            plan["path_segment_clearance_m"] = [
                                float(plan["required_clearance_m"]) - 0.001
                                for _ in plan["path_segment_clearance_m"]
                            ]
                        else:
                            plan["effective_safety_margin_m"] = 0.02
                            plan["required_clearance_m"] = (
                                float(plan["robot_radius_m"]) + 0.02
                            )
                            plan["safety_margin_relaxed"] = True
                    return plan

                with tempfile.TemporaryDirectory() as temp_root:
                    with mock.patch.dict(
                        os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
                    ):
                        with mock.patch.object(
                            official_v2_tools,
                            "plan_clearance_path",
                            side_effect=invalid_certificate,
                        ):
                            with mock.patch.object(
                                official_v2_tools,
                                "_yield_adjust_chassis_controller",
                                side_effect=controller,
                            ):
                                actions = list(
                                    build_registry(adapter)["navigate_to"].fn(
                                        ctx,
                                        name=GOAL_NAME,
                                        session_id=f"corridor_bad_{mutation}",
                                        timeout_s=5.0,
                                    )
                                )

                self.assertEqual(setter.call_count, 1)
                self.assertEqual(len(results), 1)
                self.assertFalse(results[0]["ok"], results[0])
                self.assertEqual(results[0]["failure_stage"], "planning")
                self.assertIn(expected_error, results[0]["error"])
                controller.assert_not_called()
                self.assertEqual(len(actions), 1)
                self._assert_action_contract(actions)
                np.testing.assert_allclose(
                    actions[-1][ACTION_SLICES["base"]], 0.0
                )

    def test_success_record_contains_aligned_finite_signed_certificates(self) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world,
            goal_cell=(30, 80),
            shape=(64, 96),
        )
        state = (
            _execution_test.NavigateToExecutionTest._install_snapshot_provider(
                adapter, snapshot
            )
        )
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        calls: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=self._exact_controller(state, calls),
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="corridor_signed_certificates",
                            timeout_s=10.0,
                        )
                    )
            record = json.loads(
                Path(results[0]["plan_record_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        trajectory = record["trajectory"]
        certificates = trajectory["path_segment_clearance_m"]
        self.assertEqual(len(certificates), len(trajectory["path_xy_m"]) - 1)
        required = float(trajectory["clearance"]["required_clearance_m"])
        self.assertTrue(certificates)
        self.assertTrue(
            all(math.isfinite(float(value)) and float(value) >= required for value in certificates)
        )
        self.assertEqual(results[0]["path_segment_clearance_m"], certificates)
        self.assertTrue(
            all(
                call["length_m"]
                <= official_v2_tools.NAVIGATE_TO_MAX_COMMAND_SEGMENT_M + 1.0e-9
                for call in calls
            ),
            calls,
        )
        self._assert_action_contract(actions)

    def test_signed_clearance_certificate_tamper_is_rejected_before_motion(self) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world, goal_cell=(30, 15)
        )
        _execution_test.NavigateToExecutionTest._install_snapshot_provider(
            adapter, snapshot
        )
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        controller = mock.Mock()
        real_loader = official_v2_tools._navigate_to_load_plan
        tampered = False

        with tempfile.TemporaryDirectory() as temp_root:

            def tampering_loader(session_id, plan_id):
                nonlocal tampered
                path = Path(temp_root) / session_id / "plans" / f"{plan_id}.json"
                if not tampered:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    record["trajectory"]["path_segment_clearance_m"][0] += 0.01
                    path.write_text(json.dumps(record), encoding="utf-8")
                    tampered = True
                return real_loader(session_id, plan_id)

            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_navigate_to_load_plan",
                    side_effect=tampering_loader,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=controller,
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id="corridor_certificate_tamper",
                                timeout_s=5.0,
                            )
                        )

        self.assertTrue(tampered)
        self.assertEqual(setter.call_count, 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "start-state validation")
        self.assertIn("digest mismatch", results[0]["error"])
        controller.assert_not_called()
        self.assertEqual(len(actions), 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_same_version_pose_outside_tube_holds_and_replans_before_next_motion(
        self,
    ) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world,
            goal_cell=(30, 80),
            shape=(64, 96),
        )
        state = (
            _execution_test.NavigateToExecutionTest._install_snapshot_provider(
                adapter, snapshot
            )
        )
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        real_planner = official_v2_tools.plan_clearance_path
        calls: list[dict] = []
        events: list[str] = []
        displaced = False
        plan_count = 0

        def conservative_plan(current, name, **kwargs):
            nonlocal plan_count
            plan_count += 1
            events.append(f"plan:{plan_count}")
            return self._conservative_plan(
                real_planner, current, name, **kwargs
            )

        def leave_tube(call_index, current_state, _world):
            nonlocal displaced
            if call_index == 0 and not displaced:
                displaced = True
                current_state["snapshot"]["pose"]["y"] += 0.08
                events.append("outside-tube")

        base_controller = self._exact_controller(
            state, calls, after_move=leave_tube
        )

        def controller(inner_ctx, **kwargs):
            events.append(f"controller:{len(calls) + 1}")
            return base_controller(inner_ctx, **kwargs)

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=conservative_plan,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=controller,
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id="corridor_outside_tube",
                                timeout_s=10.0,
                            )
                        )

        self.assertTrue(displaced)
        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 1)
        self.assertEqual(plan_count, 2)
        self.assertLess(events.index("outside-tube"), events.index("plan:2"))
        self.assertLess(events.index("plan:2"), events.index("controller:2"))
        self.assertGreaterEqual(len(actions), 3)
        self.assertTrue(np.any(np.abs(actions[0][ACTION_SLICES["base"]]) > 0.0))
        np.testing.assert_allclose(actions[1][ACTION_SLICES["base"]], 0.0)
        self.assertTrue(
            all(
                call["kwargs"].get("_allow_reverse_recovery") is False
                for call in calls
            ),
            calls,
        )
        self._assert_action_contract(actions)

    def test_same_version_pose_inside_tube_keeps_residual_margin(self) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world,
            goal_cell=(30, 80),
            shape=(64, 96),
        )
        state = (
            _execution_test.NavigateToExecutionTest._install_snapshot_provider(
                adapter, snapshot
            )
        )
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        real_planner = official_v2_tools.plan_clearance_path
        calls: list[dict] = []
        offset_applied = False
        plan_count = 0

        def conservative_plan(current, name, **kwargs):
            nonlocal plan_count
            plan_count += 1
            return self._conservative_plan(
                real_planner, current, name, **kwargs
            )

        def remain_inside(call_index, current_state, _world):
            nonlocal offset_applied
            if call_index == 0 and not offset_applied:
                offset_applied = True
                current_state["snapshot"]["pose"]["y"] += 0.03

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=conservative_plan,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=self._exact_controller(
                            state, calls, after_move=remain_inside
                        ),
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id="corridor_inside_tube",
                                timeout_s=10.0,
                            )
                        )

        self.assertTrue(offset_applied)
        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 0)
        self.assertEqual(plan_count, 1)
        self.assertGreaterEqual(len(calls), 2)
        first_corridor = results[0]["segments"][0]["corridor"]
        self.assertAlmostEqual(first_corridor["deviation_m"], 0.03, places=6)
        self.assertTrue(first_corridor["projection_in_range"])
        self.assertTrue(first_corridor["command_length_in_range"])
        self.assertGreaterEqual(
            first_corridor["residual_certified_clearance_m"] + 1.0e-9,
            first_corridor["required_runtime_clearance_m"],
        )
        self.assertTrue(
            all(
                call["length_m"]
                <= official_v2_tools.NAVIGATE_TO_MAX_COMMAND_SEGMENT_M + 1.0e-9
                for call in calls
            ),
            calls,
        )
        self.assertTrue(
            all(
                call["kwargs"].get("_allow_reverse_recovery") is False
                for call in calls
            ),
            calls,
        )
        self._assert_action_contract(actions)

    def test_heading_error_at_45_degrees_translates_without_spin(self) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world, goal_cell=(30, 15)
        )
        snapshot["pose"]["yaw_deg"] = 45.0
        state = (
            _execution_test.NavigateToExecutionTest._install_snapshot_provider(
                adapter, snapshot
            )
        )
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        calls: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=self._exact_controller(state, calls),
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="corridor_heading_45_no_spin",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertTrue(calls)
        self.assertFalse(
            any(abs(call["spin"]) > 1.0e-9 for call in calls), calls
        )
        translations = [call for call in calls if call["length_m"] > 1.0e-9]
        self.assertEqual(len(translations), 1, calls)
        self.assertGreater(abs(translations[0]["translation"]), 0.1)
        self._assert_action_contract(actions)

    def test_heading_error_over_45_degrees_spins_once_then_translates(self) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world, goal_cell=(30, 15)
        )
        snapshot["pose"]["yaw_deg"] = 46.0
        state = (
            _execution_test.NavigateToExecutionTest._install_snapshot_provider(
                adapter, snapshot
            )
        )
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        calls: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=self._exact_controller(state, calls),
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="corridor_heading_46_single_spin",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        spins = [call for call in calls if abs(call["spin"]) > 1.0e-9]
        translations = [call for call in calls if call["length_m"] > 1.0e-9]
        self.assertEqual(len(spins), 1, calls)
        self.assertEqual(len(translations), 1, calls)
        self.assertAlmostEqual(spins[0]["forward"], 0.0, places=9)
        self.assertAlmostEqual(spins[0]["translation"], 0.0, places=9)
        self.assertAlmostEqual(spins[0]["spin"], -46.0, places=6)
        self.assertGreater(translations[0]["forward"], 0.0)
        self.assertAlmostEqual(translations[0]["translation"], 0.0, places=6)
        self.assertLess(calls.index(spins[0]), calls.index(translations[0]))
        self._assert_action_contract(actions)

    def test_real_spin_only_controller_emits_spin_without_linear_presolve(self) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        ctx, _results, _setter = _execution_test.NavigateToExecutionTest._ctx(world)

        phase = official_v2_tools._yield_adjust_chassis_controller(
            ctx,
            forward=0.0,
            translation=0.0,
            spin=30.0,
            timeout_s=5.0,
            _require_linear_target=False,
            _commit_result=False,
            _emit_terminal_hold=False,
            _allow_reverse_recovery=False,
            _skip_linear_settle_for_spin=True,
            _result_tool_name="navigate_to",
        )
        try:
            first = next(phase)
        finally:
            phase.close()

        self._assert_action_contract([first])
        base = first[ACTION_SLICES["base"]]
        np.testing.assert_allclose(base[:2], 0.0)
        self.assertGreater(float(base[2]), 0.0)

    def test_stale_source_pose_revision_cannot_reanchor_away_odometry_drift(
        self,
    ) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world, goal_cell=(30, 15)
        )
        state = (
            _execution_test.NavigateToExecutionTest._install_snapshot_provider(
                adapter, snapshot
            )
        )
        policy_pose = np.zeros(3, dtype=np.float64)
        world.policy_local_base_pose = lambda: policy_pose.copy()
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        real_planner = official_v2_tools.plan_clearance_path
        plan_count = 0
        calls: list[dict] = []
        first_phase_requested_second_action = False

        def tracked_plan(current, name, **kwargs):
            nonlocal plan_count
            plan_count += 1
            return real_planner(current, name, **kwargs)

        exact_controller = self._exact_controller(state, calls)

        def controller(inner_ctx, forward=0.0, translation=0.0, spin=0.0, **kwargs):
            nonlocal first_phase_requested_second_action
            if calls:
                return exact_controller(
                    inner_ctx,
                    forward=forward,
                    translation=translation,
                    spin=spin,
                    **kwargs,
                )
            calls.append(
                {
                    "forward": float(forward),
                    "translation": float(translation),
                    "spin": float(spin),
                    "length_m": math.hypot(float(forward), float(translation)),
                    "map_version": str(state["snapshot"]["map_version"]),
                    "kwargs": dict(kwargs),
                    "entered": True,
                }
            )

            def phase():
                nonlocal first_phase_requested_second_action
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                first_phase_requested_second_action = True
                yield inner_ctx.world.make_action(base=[0.9, 0.0, 0.0])
                self.fail("stale-source odometry drift resumed the old segment")

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=tracked_plan,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=controller,
                    ):
                        navigation = build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="corridor_stale_source_no_reanchor",
                            timeout_s=10.0,
                        )
                        first = next(navigation)

                        pose = state["snapshot"]["pose"]
                        pose["x"] += 0.10
                        _execution_test.NavigateToExecutionTest._advance_snapshot_pose(
                            state["snapshot"]
                        )
                        # The new producer revision was derived from observation 1,
                        # the same observation at which the prior motion was issued.
                        state["snapshot"][
                            "pose_source_observation_sequence"
                        ] = 1
                        policy_pose[:] = [0.10, 0.301, 0.0]

                        stop = next(navigation)
                        self.assertEqual(plan_count, 1)
                        self.assertFalse(first_phase_requested_second_action)
                        np.testing.assert_allclose(
                            stop[ACTION_SLICES["base"]], 0.0
                        )

                        # Let the stop settle against a producer revision that now
                        # causally covers the discarded segment, then finish plan 2.
                        first_settle = next(navigation)
                        state["snapshot"][
                            "pose_source_observation_sequence"
                        ] = int(state["snapshot"]["observation_sequence"])
                        policy_pose[:] = [0.10, 0.0, 0.0]
                        actions = [first, stop, first_settle, *list(navigation)]

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 1)
        self.assertEqual(plan_count, 2)
        self.assertFalse(first_phase_requested_second_action)
        self.assertTrue(np.any(np.abs(first[ACTION_SLICES["base"]]) > 0.0))
        self.assertFalse(
            any(
                math.isclose(
                    float(action[ACTION_SLICES["base"]][0]),
                    0.9,
                    abs_tol=1.0e-9,
                )
                for action in actions
            )
        )
        self._assert_action_contract(actions)

    def test_cross_track_at_030_m_remains_inside_active_segment(self) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world, goal_cell=(30, 15)
        )
        state = (
            _execution_test.NavigateToExecutionTest._install_snapshot_provider(
                adapter, snapshot
            )
        )
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        controller_calls = 0

        def boundary_controller(inner_ctx, **kwargs):
            nonlocal controller_calls
            controller_calls += 1
            self.assertAlmostEqual(float(kwargs.get("spin", 0.0)), 0.0)

            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                pose = state["snapshot"]["pose"]
                pose["x"] += 0.10
                pose["y"] += 0.30
                _execution_test.NavigateToExecutionTest._advance_snapshot_pose(
                    state["snapshot"]
                )
                yield inner_ctx.world.make_action(base=[0.2, 0.0, 0.0])
                goal = state["snapshot"]["places"][0]
                pose["x"] = float(goal["x"])
                pose["y"] = float(goal["y"])
                _execution_test.NavigateToExecutionTest._advance_snapshot_pose(
                    state["snapshot"]
                )
                return {
                    "ok": True,
                    "action_steps": 2,
                    "linear_remaining_m": 0.0,
                    "obstacle_stop_reason": None,
                }

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=boundary_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="corridor_cross_track_boundary",
                            timeout_s=10.0,
                            arrival_tolerance_m=0.08,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 0)
        self.assertEqual(controller_calls, 1)
        self.assertGreaterEqual(len(actions), 2)
        self.assertAlmostEqual(
            float(actions[0][ACTION_SLICES["base"]][0]), 0.1
        )
        self.assertAlmostEqual(
            float(actions[1][ACTION_SLICES["base"]][0]), 0.2
        )
        self._assert_action_contract(actions)

    def test_cross_track_over_030_stops_then_causally_replans(self) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world, goal_cell=(30, 15)
        )
        state = (
            _execution_test.NavigateToExecutionTest._install_snapshot_provider(
                adapter, snapshot
            )
        )
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        initial_motion_epoch = world.motion_epoch()
        real_planner = official_v2_tools.plan_clearance_path
        plan_sources: list[tuple[bool, int | None]] = []
        calls: list[dict] = []
        used_deviation_phase = False

        def tracked_plan(current, name, **kwargs):
            known = bool(current["pose_source_sequence_known"])
            source = current.get("pose_source_observation_sequence")
            plan_sources.append((known, int(source) if known else None))
            return real_planner(current, name, **kwargs)

        exact_controller = self._exact_controller(state, calls)

        def controller(inner_ctx, forward=0.0, translation=0.0, spin=0.0, **kwargs):
            nonlocal used_deviation_phase
            if used_deviation_phase or abs(float(spin)) > 1.0e-9:
                return exact_controller(
                    inner_ctx,
                    forward=forward,
                    translation=translation,
                    spin=spin,
                    **kwargs,
                )
            used_deviation_phase = True
            calls.append(
                {
                    "forward": float(forward),
                    "translation": float(translation),
                    "spin": float(spin),
                    "length_m": math.hypot(float(forward), float(translation)),
                    "map_version": str(state["snapshot"]["map_version"]),
                    "kwargs": dict(kwargs),
                    "entered": True,
                }
            )

            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                pose = state["snapshot"]["pose"]
                pose["x"] += 0.10
                pose["y"] += 0.301
                _execution_test.NavigateToExecutionTest._advance_snapshot_pose(
                    state["snapshot"]
                )
                state["snapshot"]["pose_source_sequence_known"] = False
                state["snapshot"].pop(
                    "pose_source_observation_sequence", None
                )
                yield inner_ctx.world.make_action(base=[0.9, 0.0, 0.0])
                self.fail("cross-track guard resumed a discarded motion phase")

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=tracked_plan,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=controller,
                    ):
                        navigation = build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="corridor_cross_track_causal_replan",
                            timeout_s=10.0,
                        )
                        first = next(navigation)
                        stop = next(navigation)
                        self.assertEqual(len(plan_sources), 1)
                        np.testing.assert_allclose(
                            stop[ACTION_SLICES["base"]], 0.0
                        )
                        np.testing.assert_allclose(
                            world.policy_local_base_pose(),
                            [0.1 * 0.75 / 30.0, 0.0, 0.0],
                            atol=1.0e-12,
                        )
                        self.assertEqual(
                            world.motion_epoch(), initial_motion_epoch + 1
                        )
                        np.testing.assert_allclose(
                            world._pending_base_odometry["controller_command"],
                            [0.0, 0.0, 0.0],
                            atol=0.0,
                        )
                        causal_hold = next(navigation)
                        self.assertEqual(len(plan_sources), 1)
                        np.testing.assert_allclose(
                            causal_hold[ACTION_SLICES["base"]], 0.0
                        )
                        state["snapshot"]["pose_source_sequence_known"] = True
                        state["snapshot"][
                            "pose_source_observation_sequence"
                        ] = int(state["snapshot"]["observation_sequence"])
                        actions = [first, stop, causal_hold, *list(navigation)]

        self.assertTrue(used_deviation_phase)
        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 1)
        self.assertEqual(len(plan_sources), 2, plan_sources)
        self.assertTrue(plan_sources[1][0], plan_sources)
        self.assertGreater(plan_sources[1][1], plan_sources[0][1])
        self.assertTrue(np.any(np.abs(first[ACTION_SLICES["base"]]) > 0.0))
        self.assertFalse(
            any(
                math.isclose(
                    float(action[ACTION_SLICES["base"]][0]),
                    0.9,
                    abs_tol=1.0e-9,
                )
                for action in actions
            )
        )
        self.assertTrue(
            all(
                call["kwargs"].get("_allow_reverse_recovery") is False
                for call in calls
            ),
            calls,
        )
        self._assert_action_contract(actions)

    def test_arrival_settle_coalesces_continuous_map_updates_without_replan(
        self,
    ) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world,
            start_cell=(30, 10),
            goal_cell=(30, 10),
        )
        state = (
            _execution_test.NavigateToExecutionTest._install_snapshot_provider(
                adapter, snapshot
            )
        )
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        controller = mock.Mock()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=controller,
                ):
                    navigation = build_registry(adapter)["navigate_to"].fn(
                        ctx,
                        name=GOAL_NAME,
                        session_id="corridor_settle_continuous_map_updates",
                        timeout_s=10.0,
                    )
                    actions = []
                    while True:
                        try:
                            action = next(navigation)
                        except StopIteration:
                            break
                        actions.append(action)
                        if results:
                            continue
                        self.assertLessEqual(
                            len(actions),
                            2 * official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS,
                            "continuous map updates caused arrival-settle livelock",
                        )
                        current = state["snapshot"]
                        current["map_version"] = f"map-settle-{len(actions)}"
                        current["pose"]["x"] += 0.001
                        current["places"][0]["x"] += 0.001
                        _execution_test.NavigateToExecutionTest._advance_snapshot_pose(
                            current
                        )

        controller.assert_not_called()
        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 0)
        self.assertEqual(len(results[0]["plan_records"]), 1)
        self.assertEqual(
            results[0]["arrival_settle_steps"],
            official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS,
        )
        self.assertEqual(
            len(actions),
            official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS + 1,
        )
        latest = state["snapshot"]
        self.assertEqual(results[0]["map_version"], latest["map_version"])
        self.assertEqual(
            results[0]["observation_sequence"], latest["observation_sequence"]
        )
        self.assertAlmostEqual(
            results[0]["final_map_pose"]["x_m"], latest["pose"]["x"]
        )
        self.assertAlmostEqual(
            results[0]["final_map_pose"]["y_m"], latest["pose"]["y"]
        )
        self.assertEqual(
            results[0]["marked_goal_xy_m"],
            [latest["places"][0]["x"], latest["places"][0]["y"]],
        )
        self.assertAlmostEqual(results[0]["marked_goal_error_m"], 0.0)
        self._assert_action_contract(actions)
        for action in actions:
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)

    def test_live_target_chord_bound_closes_endpoint_margin_loophole(self) -> None:
        _adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(world)
        start = np.asarray(
            [snapshot["pose"]["x"], snapshot["pose"]["y"]],
            dtype=np.float64,
        )
        end = start + np.asarray(
            [official_v2_tools.NAVIGATE_TO_MAX_PLANNED_SEGMENT_M, 0.0],
            dtype=np.float64,
        )
        trajectory = {
            "path_xy_m": [start.tolist(), end.tolist()],
            "path_segment_clearance_m": [0.51],
            "clearance": {
                "required_clearance_m": 0.50,
                "robot_radius_m": official_v2_tools.BASE_FOOTPRINT_RADIUS_M,
            },
        }

        # The pose remains inside the signed endpoint/tube margins, but its
        # diagonal live chord to the target is just over the bounded command length.
        snapshot["pose"]["x"] = float(start[0] - 0.049999)
        snapshot["pose"]["y"] = float(start[1] + 0.03)
        corridor = official_v2_tools._navigate_to_segment_corridor(
            trajectory, snapshot, 1
        )

        self.assertTrue(corridor["projection_in_range"], corridor)
        self.assertGreaterEqual(
            corridor["residual_certified_clearance_m"] + 1.0e-9,
            corridor["required_runtime_clearance_m"],
        )
        self.assertGreater(
            corridor["live_target_distance_m"],
            official_v2_tools.NAVIGATE_TO_MAX_COMMAND_SEGMENT_M,
        )
        self.assertFalse(corridor["command_length_in_range"])
        self.assertFalse(corridor["ok"])

        snapshot["pose"]["x"] = float(start[0])
        snapshot["pose"]["y"] = float(start[1])
        nominal = official_v2_tools._navigate_to_segment_corridor(
            trajectory, snapshot, 1
        )
        self.assertLessEqual(
            nominal["live_target_distance_m"],
            official_v2_tools.NAVIGATE_TO_MAX_PLANNED_SEGMENT_M,
        )
        self.assertTrue(nominal["command_length_in_range"])
        self.assertTrue(nominal["ok"])

    def test_corridor_uses_signed_whole_body_radius(self) -> None:
        _adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(world)
        start = np.asarray(
            [snapshot["pose"]["x"], snapshot["pose"]["y"]],
            dtype=np.float64,
        )
        end = start + np.asarray([0.70, 0.0], dtype=np.float64)
        trajectory = {
            "path_xy_m": [start.tolist(), end.tolist()],
            "path_segment_clearance_m": [0.90],
            "clearance": {
                "required_clearance_m": 0.78,
                "robot_radius_m": 0.70,
            },
        }

        # The signed radius, rather than a module-level footprint default,
        # determines how much of this segment certificate remains available.
        snapshot["pose"]["y"] = float(start[1] + 0.21)
        outside = official_v2_tools._navigate_to_segment_corridor(
            trajectory, snapshot, 1
        )
        self.assertAlmostEqual(outside["required_runtime_clearance_m"], 0.70)
        self.assertAlmostEqual(outside["effective_cross_track_limit_m"], 0.20)
        self.assertFalse(outside["cross_track_in_range"], outside)
        self.assertFalse(outside["ok"], outside)

        snapshot["pose"]["y"] = float(start[1] + 0.19)
        inside = official_v2_tools._navigate_to_segment_corridor(
            trajectory, snapshot, 1
        )
        self.assertTrue(inside["cross_track_in_range"], inside)
        self.assertTrue(inside["ok"], inside)

    def test_map_change_inside_atomic_segment_keeps_signed_route(self) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world, goal_cell=(30, 15)
        )
        state = (
            _execution_test.NavigateToExecutionTest._install_snapshot_provider(
                adapter, snapshot
            )
        )
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        calls: list[dict] = []

        def atomic_controller(
            inner_ctx,
            forward=0.0,
            translation=0.0,
            spin=0.0,
            **kwargs,
        ):
            calls.append(
                {
                    "forward": float(forward),
                    "translation": float(translation),
                    "spin": float(spin),
                    "kwargs": dict(kwargs),
                }
            )

            def move_fraction(fraction):
                pose = state["snapshot"]["pose"]
                yaw = math.radians(float(pose.get("yaw_deg", 0.0)))
                pose["x"] += fraction * (
                    math.cos(yaw) * float(forward)
                    - math.sin(yaw) * float(translation)
                )
                pose["y"] += fraction * (
                    math.sin(yaw) * float(forward)
                    + math.cos(yaw) * float(translation)
                )
                _execution_test.NavigateToExecutionTest._advance_snapshot_pose(
                    state["snapshot"]
                )

            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                move_fraction(0.5)
                state["snapshot"]["map_version"] = "map-v2"
                yield inner_ctx.world.make_action(base=[0.2, 0.0, 0.0])
                move_fraction(0.5)
                return {
                    "ok": True,
                    "action_steps": 2,
                    "linear_remaining_m": 0.0,
                    "obstacle_stop_reason": None,
                }

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=atomic_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="corridor_atomic_map_change",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 0)
        self.assertEqual(
            [item["map_version"] for item in results[0]["plan_records"]],
            ["map-v1"],
        )
        self.assertEqual(len(calls), 1)
        self.assertLess(len(actions), 30, "map refresh must not livelock")
        self.assertTrue(np.any(np.abs(actions[0][ACTION_SLICES["base"]]) > 0.0))
        self.assertTrue(np.any(np.abs(actions[1][ACTION_SLICES["base"]]) > 0.0))
        np.testing.assert_allclose(actions[2][ACTION_SLICES["base"]], 0.0)
        self.assertIs(calls[0]["kwargs"].get("_allow_reverse_recovery"), False)
        self._assert_action_contract(actions)

    def test_continuous_map_revisions_do_not_livelock_alignment(self) -> None:
        adapter, world = _execution_test.NavigateToExecutionTest._adapter_world()
        snapshot = _execution_test.NavigateToExecutionTest._snapshot(
            world, goal_cell=(30, 15)
        )
        snapshot["pose"]["yaw_deg"] = 46.0
        state = {"snapshot": snapshot, "reads": 0}

        def current_snapshot():
            state["reads"] += 1
            _execution_test.NavigateToExecutionTest._advance_snapshot_pose(
                state["snapshot"]
            )
            sequence = int(state["snapshot"]["observation_sequence"])
            state["snapshot"]["map_version"] = f"live-map-{sequence}"
            return deepcopy(state["snapshot"])

        adapter.navigation_map_snapshot = current_snapshot
        ctx, results, setter = _execution_test.NavigateToExecutionTest._ctx(world)
        calls: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=self._exact_controller(state, calls),
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="corridor_continuous_map_revision",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 0)
        self.assertEqual(len(results[0]["plan_records"]), 1)
        spins = [call for call in calls if abs(call["spin"]) > 1.0e-9]
        translations = [call for call in calls if call["length_m"] > 1.0e-9]
        self.assertEqual(len(spins), 1, calls)
        self.assertEqual(len(translations), 1, calls)
        self.assertLess(calls.index(spins[0]), calls.index(translations[0]))
        self.assertGreater(state["reads"], 10)
        self.assertLess(len(actions), 40, "map revisions must not cause livelock")
        self._assert_action_contract(actions)


if __name__ == "__main__":
    unittest.main()
