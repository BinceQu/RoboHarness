"""Execution-contract tests for the official-v2 ``navigate_to`` tool."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import behavior_interface_eval_test.tool.official_v2.tools as official_v2_tools
from behavior_interface_eval_test import navigation_route_overlay
from behavior_interface_eval_test.official_action_world import (
    ACTION_DIM,
    ObservationBackedActionWorld,
)
from behavior_interface_eval_test.official_policy_interface import (
    ObservationActionAdapter,
)
from behavior_interface_eval_test.robot_contract import (
    ACTION_SLICES,
    ARM_DOF,
    PROPRIO_DIM,
    PROPRIO_SLICES,
)
from behavior_interface_eval_test.tool.official_v2 import (
    PUBLIC_TOOLS,
    OfficialToolBoundaryError,
    build_registry,
    validate_submission,
)


GOAL_NAME = "marked_goal"


class NavigateToExecutionTest(unittest.TestCase):
    def test_stalled_map_producer_is_a_localization_hold(self) -> None:
        for field in ("worker_stalled", "worker_timed_out"):
            with self.subTest(field=field):
                snapshot = {"lifecycle": {field: True}}
                self.assertEqual(
                    official_v2_tools._navigate_to_localization_hold_reason(
                        snapshot
                    ),
                    "navigation map producer is stalled",
                )
        self.assertEqual(
            official_v2_tools._navigate_to_localization_hold_reason(
                {"lifecycle": {"worker_healthy": False}}
            ),
            "navigation map producer is stalled",
        )

    @staticmethod
    def _adapter_world():
        adapter = ObservationActionAdapter()
        adapter.update(
            {"robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32)}
        )
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
    def _snapshot(
        world,
        *,
        start_cell: tuple[int, int] = (30, 10),
        goal_cell: tuple[int, int] = (30, 48),
        shape: tuple[int, int] = (64, 64),
        map_version: str = "map-v1",
        map_epoch: str = "map-epoch-1",
        include_goal: bool = True,
        blocked_column: int | None = None,
    ) -> dict:
        resolution = 0.1
        occupancy = np.zeros(shape, dtype=np.int8)
        occupancy[[0, -1], :] = 100
        occupancy[:, [0, -1]] = 100
        if blocked_column is not None:
            occupancy[1:-1, int(blocked_column)] = 100

        def xy(cell: tuple[int, int]) -> tuple[float, float]:
            row, column = cell
            return (
                (float(column) + 0.5) * resolution,
                (float(row) + 0.5) * resolution,
            )

        start_x, start_y = xy(start_cell)
        goal_x, goal_y = xy(goal_cell)
        pose_version = f"{map_epoch}|fixture-pose:1"
        return {
            "schema": "behavior.official.navigation_map.v1",
            "schema_version": 1,
            "backend": "unit-test-map",
            "frame": "fixture-map",
            "episode_id": world.episode_id(),
            "map_epoch": map_epoch,
            "map_version": map_version,
            "frame_version": pose_version,
            "pose_version": pose_version,
            "observation_sequence": 1,
            "captured_ts": 100.0,
            "pose_observed_sequence": 1,
            "pose_observed_ts": 100.0,
            "pose_age_observations": 0,
            "pose_age_s": 0.0,
            "pose_source_sequence_known": True,
            "pose_source_observation_sequence": 1,
            "occupancy": occupancy,
            "origin": [0.0, 0.0],
            "resolution": resolution,
            "pose": {
                "x": start_x,
                "y": start_y,
                "yaw_deg": 0.0,
                "global_confident": True,
            },
            "places": (
                [{"name": GOAL_NAME, "x": goal_x, "y": goal_y}]
                if include_goal
                else []
            ),
            "lifecycle": {
                "pose_confident": True,
                "recovery_hold": False,
            },
        }

    @staticmethod
    def _install_snapshot_provider(adapter, snapshot: dict | None) -> dict:
        state = {"snapshot": snapshot}

        def current_snapshot():
            value = state["snapshot"]
            return None if value is None else deepcopy(value)

        adapter.navigation_map_snapshot = current_snapshot
        return state

    @staticmethod
    def _advance_snapshot_pose(snapshot: dict) -> None:
        sequence = int(snapshot["observation_sequence"]) + 1
        captured_ts = float(snapshot["captured_ts"]) + 0.1
        pose_version = f"{snapshot['map_epoch']}|fixture-pose:{sequence}"
        snapshot["observation_sequence"] = sequence
        snapshot["captured_ts"] = captured_ts
        snapshot["frame_version"] = pose_version
        snapshot["pose_version"] = pose_version
        snapshot["pose_observed_sequence"] = sequence
        snapshot["pose_observed_ts"] = captured_ts
        snapshot["pose_age_observations"] = 0
        snapshot["pose_age_s"] = 0.0
        snapshot["pose_source_sequence_known"] = True
        snapshot["pose_source_observation_sequence"] = sequence

    @staticmethod
    def _advance_snapshot_observation(snapshot: dict) -> None:
        snapshot["observation_sequence"] = int(
            snapshot["observation_sequence"]
        ) + 1
        snapshot["captured_ts"] = float(snapshot["captured_ts"]) + 0.1
        snapshot["pose_age_observations"] = (
            int(snapshot["observation_sequence"])
            - int(snapshot["pose_observed_sequence"])
        )
        snapshot["pose_age_s"] = max(
            0.0,
            float(snapshot["captured_ts"])
            - float(snapshot["pose_observed_ts"]),
        )

    @staticmethod
    def _ctx(world):
        results: list[dict] = []
        setter = mock.Mock(side_effect=lambda value: results.append(value))
        return (
            SimpleNamespace(
                world=world,
                task_name="make_microwave_popcorn",
                set_result=setter,
                raise_if_cancelled=lambda where="": None,
            ),
            results,
            setter,
        )

    @staticmethod
    def _controller_stub(state, calls, *, after_phase=None):
        """Return a deterministic action-only chassis phase implementation."""

        def adjust(ctx, forward=0.0, translation=0.0, spin=0.0, **kwargs):
            kind = "spin" if abs(float(spin)) > 1e-9 else "forward"
            amount = float(
                spin
                if kind == "spin"
                else math.hypot(float(forward), float(translation))
            )
            calls.append(
                {
                    "kind": kind,
                    "amount": amount,
                    "forward": float(forward),
                    "translation": float(translation),
                    "spin": float(spin),
                    "map_version": state["snapshot"]["map_version"],
                    "kwargs": dict(kwargs),
                }
            )

            def phase():
                base = (
                    [0.0, 0.0, math.copysign(0.2, amount)]
                    if kind == "spin"
                    else [
                        math.copysign(0.2, float(forward))
                        if abs(float(forward)) > 1e-9
                        else 0.0,
                        math.copysign(0.2, float(translation))
                        if abs(float(translation)) > 1e-9
                        else 0.0,
                        0.0,
                    ]
                )
                yield ctx.world.make_action(base=base)

                pose = state["snapshot"]["pose"]
                if kind == "spin":
                    pose["yaw_deg"] = float(pose.get("yaw_deg", 0.0)) + amount
                else:
                    yaw_rad = math.radians(float(pose.get("yaw_deg", 0.0)))
                    pose["x"] = float(pose["x"]) + (
                        math.cos(yaw_rad) * float(forward)
                        - math.sin(yaw_rad) * float(translation)
                    )
                    pose["y"] = float(pose["y"]) + (
                        math.sin(yaw_rad) * float(forward)
                        + math.cos(yaw_rad) * float(translation)
                    )
                NavigateToExecutionTest._advance_snapshot_pose(state["snapshot"])
                if after_phase is not None:
                    after_phase(kind, state, ctx.world)
                return {
                    "ok": True,
                    "action_steps": 1,
                    "linear_remaining_m": 0.0,
                    "obstacle_stop_reason": None,
                }

            return phase()

        return adjust

    def _assert_action_contract(self, actions) -> None:
        self.assertGreaterEqual(len(actions), 1)
        for raw in actions:
            action = np.asarray(raw, dtype=np.float32).reshape(-1)
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())

    def test_public_registry_and_argument_contract(self) -> None:
        self.assertIn("navigate_to", PUBLIC_TOOLS)
        spec = build_registry(None)["navigate_to"]
        self.assertIs(spec.fn, official_v2_tools.PUBLIC_TOOL_FUNCTIONS["navigate_to"])
        self.assertEqual(
            [parameter["name"] for parameter in spec.params],
            ["name", "session_id", "timeout_s", "arrival_tolerance_m"],
        )
        self.assertEqual(
            validate_submission("navigate_to", {"name": "  kitchen mark  "}),
            {
                "name": "kitchen mark",
                "session_id": "",
                "timeout_s": 180.0,
                "arrival_tolerance_m": 0.25,
            },
        )
        invalid = (
            ({"name": ""}, "name is required"),
            ({"name": "bad\nmark"}, "control characters"),
            ({"name": GOAL_NAME, "timeout_s": 0.0}, "timeout_s"),
            (
                {"name": GOAL_NAME, "arrival_tolerance_m": 1.01},
                "arrival_tolerance_m",
            ),
            ({"name": GOAL_NAME, "x": 1.0}, "unsupported arguments"),
        )
        for arguments, message in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(OfficialToolBoundaryError, message):
                    validate_submission("navigate_to", arguments)

    def test_missing_map_mark_and_unreachable_are_structured_failures(self) -> None:
        scenarios = ("missing_map", "missing_mark", "unreachable")
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                adapter, world = self._adapter_world()
                if scenario == "missing_map":
                    snapshot = None
                    expected_stage = "observation validation"
                elif scenario == "missing_mark":
                    snapshot = self._snapshot(world, include_goal=False)
                    expected_stage = "planning"
                else:
                    snapshot = self._snapshot(
                        world,
                        start_cell=(30, 10),
                        goal_cell=(30, 52),
                        blocked_column=32,
                    )
                    expected_stage = "planning"
                self._install_snapshot_provider(adapter, snapshot)
                ctx, results, setter = self._ctx(world)
                with tempfile.TemporaryDirectory() as temp_root:
                    with mock.patch.dict(
                        os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id=f"navigate_{scenario}",
                                timeout_s=5.0,
                            )
                        )

                self.assertEqual(setter.call_count, 1)
                self.assertEqual(len(results), 1)
                result = results[0]
                self.assertFalse(result["ok"], result)
                self.assertEqual(result["failure_stage"], expected_stage)
                self.assertTrue(str(result["error"]).strip())
                self.assertEqual(result["execution"], "official_action_only")
                self.assertFalse(result["direct_simulator_mutation"])
                self.assertEqual(len(actions), 1)
                self._assert_action_contract(actions)
                np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_zero_extra_margin_is_a_last_resort_base_clearance_plan(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(10, 10),
            goal_cell=(50, 10),
            shape=(64, 21),
        )
        occupancy = np.full((64, 21), 100, dtype=np.int8)
        occupancy[1:-1, 6:15] = 0
        snapshot["occupancy"] = occupancy
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=self._controller_stub(state, calls),
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_zero_margin_fallback",
                            timeout_s=20.0,
                        )
                    )
            record = json.loads(
                Path(results[0]["plan_record_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        trajectory = record["trajectory"]
        self.assertEqual(
            trajectory["planner"]["safety_margin_trials_m"],
            list(official_v2_tools.NAVIGATE_TO_SAFETY_MARGIN_TRIALS_M),
        )
        self.assertEqual(
            trajectory["planner"]["safety_margin_trial_index"],
            len(official_v2_tools.NAVIGATE_TO_SAFETY_MARGIN_TRIALS_M) - 1,
        )
        self.assertAlmostEqual(
            trajectory["clearance"]["effective_safety_margin_m"], 0.0
        )
        self.assertAlmostEqual(
            trajectory["clearance"]["required_clearance_m"],
            official_v2_tools.BASE_FOOTPRINT_RADIUS_M,
        )
        self.assertTrue(calls)
        self._assert_action_contract(actions)

    def test_transient_initial_recovery_hold_waits_then_plans(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        state = {"snapshot": snapshot, "reads": 0}

        def current_snapshot():
            state["reads"] += 1
            if state["reads"] == 1:
                snapshot["pose"]["global_confident"] = False
                snapshot["lifecycle"]["pose_confident"] = False
                snapshot["lifecycle"]["recovery_hold"] = True
            else:
                snapshot["pose"]["global_confident"] = True
                snapshot["lifecycle"]["pose_confident"] = True
                snapshot["lifecycle"]["recovery_hold"] = False
                if state["reads"] <= 3:
                    self._advance_snapshot_pose(snapshot)
            return deepcopy(snapshot)

        adapter.navigation_map_snapshot = current_snapshot
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []
        controller = self._controller_stub(state, calls)

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_transient_recovery_hold",
                            timeout_s=20.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["localization_recovery_steps"], 1)
        self.assertGreaterEqual(state["reads"], 3)
        self.assertTrue(calls)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[0][ACTION_SLICES["base"]], 0.0)
        self.assertTrue(
            any(
                np.any(np.abs(action[ACTION_SLICES["base"]]) > 1.0e-9)
                for action in actions[1:-1]
            )
        )

    def test_localization_preflight_uses_the_shared_timeout_budget(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        state = {"snapshot": snapshot, "reads": 0}
        clock = {"now": 0.0}

        def current_snapshot():
            state["reads"] += 1
            if state["reads"] <= 9:
                clock["now"] = float(2 * (state["reads"] - 1))
                snapshot["pose"]["global_confident"] = False
                snapshot["lifecycle"]["pose_confident"] = False
                snapshot["lifecycle"]["recovery_hold"] = True
            else:
                clock["now"] = 18.0
                snapshot["pose"]["global_confident"] = True
                snapshot["lifecycle"]["pose_confident"] = True
                snapshot["lifecycle"]["recovery_hold"] = False
                if state["reads"] == 10:
                    self._advance_snapshot_pose(snapshot)
            return deepcopy(snapshot)

        adapter.navigation_map_snapshot = current_snapshot
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools.time,
                    "monotonic",
                    side_effect=lambda: clock["now"],
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=self._controller_stub(state, calls),
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id="navigate_long_localization_preflight",
                                timeout_s=180.0,
                            )
                        )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertGreater(clock["now"], 15.0)
        self.assertGreaterEqual(results[0]["localization_preflight_steps"], 9)
        self.assertTrue(calls)
        self._assert_action_contract(actions)
        for action in actions[:9]:
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)

    def test_preflight_reuses_exact_witness_after_transient_hold(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["backend"] = "behavior_interface.rtabmap_slam.live.LiveMapper"
        snapshot["source_frame"] = {
            "backend_frame_id": 49,
            "lag_known": True,
            "ordered_pose_producer": {
                "schema": (
                    "behavior.official.navigation_map."
                    "ordered_pose_producer.v1"
                ),
                "enqueued_frame_count": 49,
                "processed_frame_count": 49,
                "pose_frame_id": 49,
                "last_enqueued_observation_sequence": 1,
            },
        }
        snapshot["lifecycle"]["uncertain_hold"] = True
        state = {"snapshot": snapshot, "reads": 0}

        def current_snapshot():
            state["reads"] += 1
            if state["reads"] == 2:
                self._advance_snapshot_observation(snapshot)
                snapshot["pose_source_sequence_known"] = False
                snapshot.pop("pose_source_observation_sequence", None)
                snapshot["lifecycle"]["uncertain_hold"] = False
                progress = snapshot["source_frame"]["ordered_pose_producer"]
                progress.update(
                    {
                        "enqueued_frame_count": 100,
                        "processed_frame_count": 50,
                        "pose_frame_id": 50,
                        "last_enqueued_observation_sequence": 2,
                    }
                )
                snapshot["source_frame"]["backend_frame_id"] = 50
                if hasattr(world, official_v2_tools.NAVIGATE_TO_POSE_FUSION_ATTR):
                    delattr(
                        world,
                        official_v2_tools.NAVIGATE_TO_POSE_FUSION_ATTR,
                    )
            return deepcopy(snapshot)

        adapter.navigation_map_snapshot = current_snapshot
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=self._controller_stub(state, calls),
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_reuse_invocation_witness",
                            timeout_s=20.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["localization_preflight_steps"], 1)
        self.assertEqual(
            results[0]["initial_causal_localization"]["assurance"],
            "legacy_rtab_fifo_after_exact_source",
        )
        self.assertTrue(calls)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[0][ACTION_SLICES["base"]], 0.0)

    def test_transient_hold_after_planning_waits_and_replans_without_motion(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(
            adapter, self._snapshot(world, goal_cell=(30, 15))
        )
        ctx, results, setter = self._ctx(world)
        real_planner = official_v2_tools.plan_clearance_path
        plan_calls = 0
        held_reads = 0

        def current_snapshot():
            nonlocal held_reads
            current = state["snapshot"]
            if current["lifecycle"].get("uncertain_hold", False):
                held_reads += 1
                if held_reads >= 2:
                    current["lifecycle"]["uncertain_hold"] = False
                    current["map_version"] = "map-v2"
                    current["places"] = [
                        {
                            "name": GOAL_NAME,
                            "x": float(current["pose"]["x"]),
                            "y": float(current["pose"]["y"]),
                        }
                    ]
            return deepcopy(current)

        adapter.navigation_map_snapshot = current_snapshot

        def plan_then_hold(snapshot, name, **kwargs):
            nonlocal plan_calls
            plan_calls += 1
            plan = real_planner(snapshot, name, **kwargs)
            if plan_calls == 1:
                state["snapshot"]["lifecycle"]["uncertain_hold"] = True
            return plan

        controller = mock.Mock()
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=plan_then_hold,
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
                                session_id="navigate_post_plan_hold",
                                timeout_s=20.0,
                            )
                        )

        self.assertEqual(plan_calls, 2)
        self.assertGreaterEqual(held_reads, 2)
        controller.assert_not_called()
        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 1)
        self.assertGreaterEqual(results[0]["localization_recovery_steps"], 1)
        self._assert_action_contract(actions)
        for action in actions:
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)

    def test_advisory_global_confidence_miss_does_not_block_pose(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"]["global_confident"] = False
        snapshot["lifecycle"]["pose_confident"] = False
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []
        controller = self._controller_stub(state, calls)

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_advisory_confidence_miss",
                            timeout_s=20.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["localization_recovery_steps"], 0)
        self.assertTrue(calls)
        self._assert_action_contract(actions)

    def test_successful_multisegment_execution_commits_one_result(self) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(
            adapter,
            self._snapshot(
                world,
                goal_cell=(30, 80),
                shape=(64, 96),
            ),
        )
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []
        controller = self._controller_stub(state, calls)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_success",
                            timeout_s=20.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result["ok"], result)
        self.assertGreaterEqual(result["segment_count"], 2)
        self.assertEqual(result["segment_count"], len(result["segments"]))
        self.assertEqual(result["replan_count"], 0)
        self.assertNotIn("base_qvel", vars(world))
        self.assertEqual(result["action_steps"], len(actions) - 1)
        self.assertEqual(
            result["clearance"]["robot_radius_m"],
            official_v2_tools.BASE_FOOTPRINT_RADIUS_M,
        )
        self.assertTrue(all(segment["ok"] for segment in result["segments"]))
        forward_calls = [call for call in calls if call["kind"] == "forward"]
        self.assertGreaterEqual(len(forward_calls), 2)
        self.assertTrue(
            all(call["kwargs"].get("_skip_linear_settle") is True for call in forward_calls)
        )
        self.assertTrue(
            all(
                call["kwargs"].get("vmax")
                == official_v2_tools.ADJUST_CHASSIS_BASE_MAX_LIN_MPS
                for call in forward_calls
            )
        )
        self.assertTrue(
            all(
                call["kwargs"].get("_linear_deceleration_mps2")
                == official_v2_tools.NAVIGATE_TO_LINEAR_DECEL_MPS2
                for call in forward_calls
            )
        )
        self.assertTrue(
            all(
                call["kwargs"].get("_linear_near_speed_mps")
                == official_v2_tools.NAVIGATE_TO_LINEAR_NEAR_SPEED_MPS
                and call["kwargs"].get("_linear_final_speed_mps")
                == official_v2_tools.NAVIGATE_TO_LINEAR_FINAL_SPEED_MPS
                for call in forward_calls
            )
        )
        self._assert_action_contract(actions)
        self.assertTrue(
            any(
                np.any(np.abs(action[ACTION_SLICES["base"]]) > 0.0)
                for action in actions[:-1]
            )
        )
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_navigation_route_overlay_shrinks_and_clears_on_success(self) -> None:
        session_id = "navigate_overlay_success"
        navigation_route_overlay.clear_all_routes()
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []
        controller = self._controller_stub(state, calls)
        visible = []

        try:
            with tempfile.TemporaryDirectory() as temp_root:
                with mock.patch.dict(
                    os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=controller,
                    ):
                        navigation = build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id=session_id,
                            timeout_s=20.0,
                        )
                        while True:
                            try:
                                next(navigation)
                            except StopIteration:
                                break
                            snapshot = navigation_route_overlay.get_route_snapshot(
                                session_id
                            )
                            if snapshot is not None:
                                visible.append(snapshot)

            self.assertEqual(setter.call_count, 1)
            self.assertTrue(results[0]["ok"], results[0])
            self.assertGreaterEqual(len(visible), 2)
            first = visible[0]
            self.assertGreaterEqual(len(first.points_xy_m), 2)
            self.assertTrue(
                any(
                    snapshot.revision > first.revision
                    and (
                        len(snapshot.points_xy_m) < len(first.points_xy_m)
                        or math.dist(
                            snapshot.points_xy_m[0], first.points_xy_m[0]
                        )
                        > 1.0e-6
                    )
                    for snapshot in visible[1:]
                )
            )
            self.assertIsNone(
                navigation_route_overlay.get_route_snapshot(session_id)
            )
        finally:
            navigation_route_overlay.clear_all_routes()

    def test_navigation_route_overlay_clears_on_execution_failure(self) -> None:
        session_id = "navigate_overlay_failure"
        navigation_route_overlay.clear_all_routes()
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)

        def failing_controller(inner_ctx, **_kwargs):
            def phase():
                yield inner_ctx.world.make_action(base=[0.2, 0.0, 0.0])
                raise official_v2_tools._NavigateToFailure(
                    "tracking", "fixture chassis failure"
                )

            return phase()

        try:
            with tempfile.TemporaryDirectory() as temp_root:
                with mock.patch.dict(
                    os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=failing_controller,
                    ):
                        navigation = build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id=session_id,
                            timeout_s=20.0,
                        )
                        first_action = next(navigation)
                        self.assertTrue(
                            np.any(
                                np.abs(first_action[ACTION_SLICES["base"]]) > 0.0
                            )
                        )
                        self.assertIsNotNone(
                            navigation_route_overlay.get_route_snapshot(
                                session_id
                            )
                        )
                        snapshot["pose"]["x"] += 0.1
                        self._advance_snapshot_pose(snapshot)
                        remaining_actions = list(navigation)

            self.assertEqual(setter.call_count, 1)
            self.assertFalse(results[0]["ok"], results[0])
            self.assertEqual(results[0]["failure_stage"], "tracking")
            self.assertEqual(len(remaining_actions), 1)
            self.assertIsNone(
                navigation_route_overlay.get_route_snapshot(session_id)
            )
        finally:
            navigation_route_overlay.clear_all_routes()

    def test_navigation_route_overlay_clears_when_generator_is_closed(self) -> None:
        session_id = "navigate_overlay_close"
        navigation_route_overlay.clear_all_routes()
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []

        try:
            with tempfile.TemporaryDirectory() as temp_root:
                with mock.patch.dict(
                    os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=self._controller_stub(state, calls),
                    ):
                        navigation = build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id=session_id,
                            timeout_s=20.0,
                        )
                        next(navigation)
                        self.assertIn("base_qvel", vars(world))
                        self.assertIsNotNone(
                            navigation_route_overlay.get_route_snapshot(
                                session_id
                            )
                        )
                        navigation.close()
                        self.assertNotIn("base_qvel", vars(world))

            self.assertEqual(setter.call_count, 0)
            self.assertEqual(results, [])
            self.assertIsNone(
                navigation_route_overlay.get_route_snapshot(session_id)
            )
        finally:
            navigation_route_overlay.clear_all_routes()

    def test_arm_posture_does_not_change_base_navigation_radius(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector()
        proprio[PROPRIO_SLICES["arm_left_qpos"]][0] = -math.pi / 2.0
        if ARM_DOF == 8:
            proprio[PROPRIO_SLICES["arm_left_qpos"]][7] = 0.63
            proprio[PROPRIO_SLICES["arm_right_qpos"]][7] = -0.47
        adapter.update({"robot_r1::proprio": proprio.astype(np.float32)})
        snapshot = self._snapshot(world)
        self._advance_snapshot_pose(snapshot)
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []
        controller = self._controller_stub(state, calls)
        planner_radii: list[float] = []
        real_planner = official_v2_tools.plan_clearance_path

        def recording_planner(snapshot, name, **kwargs):
            planner_radii.append(float(kwargs["robot_radius_m"]))
            return real_planner(snapshot, name, **kwargs)

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=recording_planner,
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
                                session_id="navigate_base_footprint",
                                timeout_s=20.0,
                            )
                        )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(len(planner_radii), 1)
        self.assertAlmostEqual(
            planner_radii[0], official_v2_tools.BASE_FOOTPRINT_RADIUS_M
        )
        self.assertAlmostEqual(
            results[0]["clearance"]["robot_radius_m"],
            planner_radii[0],
            places=9,
        )
        self._assert_action_contract(actions)
        if ARM_DOF == 8:
            for action in actions:
                self.assertAlmostEqual(
                    float(action[ACTION_SLICES["arm_left"]][7]), 0.63, places=6
                )
                self.assertAlmostEqual(
                    float(action[ACTION_SLICES["arm_right"]][7]), -0.47, places=6
                )

    def test_planning_repins_limbs_to_the_same_observed_posture(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector()
        observed_trunk = np.asarray([0.08, -0.04, 0.06, 0.0])
        observed_left = np.zeros(ARM_DOF, dtype=np.float64)
        observed_left[0] = -0.35
        observed_right = np.zeros(ARM_DOF, dtype=np.float64)
        observed_right[1] = 0.22
        proprio[PROPRIO_SLICES["trunk_qpos"]] = observed_trunk
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = observed_left
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = observed_right
        adapter.update({"robot_r1::proprio": proprio.astype(np.float32)})
        world.make_action(
            trunk=[-0.2, 0.1, -0.1, 0.0],
            arm_left=[0.25] + [0.0] * (ARM_DOF - 1),
            arm_right=[-0.2] + [0.0] * (ARM_DOF - 1),
        )

        contract, hold = (
            official_v2_tools._navigate_to_capture_inactive_limb_contract(
                SimpleNamespace(world=world), initialize_pins=True
            )
        )

        self.assertIsNotNone(hold)
        np.testing.assert_allclose(
            contract["resolved_pins"]["trunk"], observed_trunk, atol=1.0e-7
        )
        np.testing.assert_allclose(
            contract["resolved_pins"]["arm_left"], observed_left, atol=1.0e-7
        )
        np.testing.assert_allclose(
            contract["resolved_pins"]["arm_right"], observed_right, atol=1.0e-7
        )
        channels = official_v2_tools._navigate_to_inactive_action_channels(hold)
        np.testing.assert_allclose(channels["trunk"], observed_trunk, atol=1.0e-7)
        np.testing.assert_allclose(
            channels["arm_left"], observed_left, atol=1.0e-7
        )
        np.testing.assert_allclose(
            channels["arm_right"], observed_right, atol=1.0e-7
        )

    def test_nonzero_start_snap_requires_certified_egress(self) -> None:
        adapter, world = self._adapter_world()
        self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        real_planner = official_v2_tools.plan_clearance_path
        controller = mock.Mock()

        def snapped_start(snapshot, name, **kwargs):
            self.assertEqual(
                kwargs["robot_radius_m"],
                official_v2_tools.BASE_FOOTPRINT_RADIUS_M,
            )
            self.assertTrue(callable(kwargs["progress_check"]))
            plan = dict(real_planner(snapshot, name, **kwargs))
            self.assertTrue(plan["ok"], plan)
            plan["start_snap_m"] = 0.01
            plan["start_egress_certified"] = False
            return plan

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=snapped_start,
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
                                session_id="navigate_start_snap",
                                timeout_s=5.0,
                            )
                        )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "planning")
        self.assertIn("did not certify", results[0]["error"])
        controller.assert_not_called()
        self.assertEqual(len(actions), 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_certified_nonzero_start_egress_is_signed_and_executed(self) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        real_planner = official_v2_tools.plan_clearance_path
        calls: list[dict] = []

        def certified_start(snapshot, name, **kwargs):
            plan = deepcopy(real_planner(snapshot, name, **kwargs))
            self.assertTrue(plan["ok"], plan)
            start = np.asarray(plan["path_xy_m"][0], dtype=np.float64)
            toward = np.asarray(plan["path_xy_m"][1], dtype=np.float64) - start
            safe = start + 0.01 * toward / np.linalg.norm(toward)
            safe_xy = safe.astype(float).tolist()
            plan["path_xy_m"].insert(1, safe_xy)
            plan["path_segment_clearance_m"].insert(
                0, plan["path_segment_clearance_m"][0]
            )
            plan["corner_path_xy_m"].insert(1, safe_xy)
            plan["start_snap_m"] = 0.01
            plan["start_safe_xy_m"] = safe_xy
            plan["start_egress_certified"] = True
            plan["start_egress_segment_count"] = 1
            plan["start_egress_required_clearance_m"] = (
                float(kwargs["robot_radius_m"])
                + min(
                    float(plan["effective_safety_margin_m"]),
                    official_v2_tools.NAVIGATE_TO_START_EGRESS_SAFETY_MARGIN_M,
                )
            )
            plan["start"]["safe_grid_xy_m"] = safe_xy
            plan["start"]["egress_certified"] = True
            plan["snap"]["start"].update(
                {"applied": True, "distance_m": 0.01, "egress_certified": True}
            )
            return plan

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=certified_start,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=self._controller_stub(state, calls),
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id="navigate_certified_start_egress",
                                timeout_s=20.0,
                            )
                        )

            record = json.loads(
                Path(results[0]["plan_record_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        trajectory = record["trajectory"]
        self.assertEqual(trajectory["clearance"]["start_snap_m"], 0.01)
        self.assertTrue(
            trajectory["safety_validation"]["start_egress_certified"]
        )
        self.assertEqual(trajectory["start_state"]["start_safe_path_index"], 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_short_egress_is_not_skipped_outside_next_corridor(self) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        real_planner = official_v2_tools.plan_clearance_path
        calls: list[dict] = []

        def narrow_after_egress(snapshot, name, **kwargs):
            plan = deepcopy(real_planner(snapshot, name, **kwargs))
            self.assertTrue(plan["ok"], plan)
            radius = float(kwargs["robot_radius_m"])
            start = np.asarray(plan["path_xy_m"][0], dtype=np.float64)
            toward = np.asarray(plan["path_xy_m"][1], dtype=np.float64) - start
            safe = start + 0.049 * toward / np.linalg.norm(toward)
            safe_xy = safe.astype(float).tolist()
            plan["path_xy_m"].insert(1, safe_xy)
            plan["corner_path_xy_m"].insert(1, safe_xy)
            plan["path_segment_clearance_m"].insert(
                0, float(plan["path_segment_clearance_m"][0])
            )
            plan["path_segment_clearance_m"][1] = radius + 0.04
            plan["path_segment_required_clearance_m"] = [
                radius + 0.04
            ] * (len(plan["path_xy_m"]) - 1)
            plan["requested_safety_margin_m"] = 0.08
            plan["effective_safety_margin_m"] = 0.04
            plan["required_clearance_m"] = radius + 0.04
            plan["minimum_clearance_m"] = radius + 0.04
            plan["minimum_execution_clearance_m"] = radius + 0.04
            plan["safety_margin_relaxed"] = True
            plan["safety_margin_trial_index"] = 2
            plan["start_snap_m"] = 0.049
            plan["start_safe_xy_m"] = safe_xy
            plan["start_egress_certified"] = True
            plan["start_egress_segment_count"] = 1
            plan["start_egress_required_clearance_m"] = radius + 0.02
            plan["start"]["safe_grid_xy_m"] = safe_xy
            plan["start"]["egress_certified"] = True
            plan["snap"]["start"].update(
                {
                    "applied": True,
                    "distance_m": 0.049,
                    "egress_certified": True,
                }
            )
            return plan

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=narrow_after_egress,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=self._controller_stub(state, calls),
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id="navigate_short_egress",
                                timeout_s=20.0,
                            )
                        )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 0)
        forward_calls = [call for call in calls if call["kind"] == "forward"]
        self.assertGreaterEqual(len(forward_calls), 2)
        self.assertAlmostEqual(forward_calls[0]["amount"], 0.049, places=6)
        self.assertAlmostEqual(
            forward_calls[0]["kwargs"]["_linear_tolerance_m"],
            0.02,
            places=6,
        )
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_goal_standoff_reserves_marked_goal_tolerance(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        planned_goal_x = float(snapshot["places"][0]["x"])
        marked_goal_x = planned_goal_x + 0.20
        snapshot["places"][0]["x"] = marked_goal_x
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        real_planner = official_v2_tools.plan_clearance_path
        calls: list[dict] = []

        def standoff_plan(current, name, **kwargs):
            planning_snapshot = deepcopy(current)
            planning_snapshot["places"][0]["x"] = planned_goal_x
            kwargs["optimize_goal_standoff"] = False
            plan = dict(real_planner(planning_snapshot, name, **kwargs))
            self.assertTrue(plan["ok"], plan)
            plan["marked_goal_xy_m"] = [
                marked_goal_x,
                float(current["places"][0]["y"]),
            ]
            plan["goal"] = {
                **dict(plan["goal"]),
                "requested_xy_m": list(plan["marked_goal_xy_m"]),
            }
            plan["goal_standoff_m"] = 0.20
            plan["snap"] = {
                **dict(plan["snap"]),
                "goal": {"applied": True, "distance_m": 0.20},
            }
            return plan

        controller = self._controller_stub(state, calls)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=standoff_plan,
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
                                session_id="navigate_goal_standoff",
                                timeout_s=20.0,
                                arrival_tolerance_m=0.25,
                            )
                        )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["planned_goal_tolerance_m"], 0.05)
        self.assertLessEqual(result["planned_goal_error_m"], 0.05 + 1e-9)
        self.assertLessEqual(result["marked_goal_error_m"], 0.25 + 1e-9)
        self.assertAlmostEqual(result["marked_goal_error_m"], 0.20)
        self.assertTrue(
            all(item["map_frame_waypoint_verified"] for item in result["segments"])
        )
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_image_pick_always_uses_footprint_standoff(
        self,
    ) -> None:
        for goal_cell in ((30, 48), (2, 48)):
            with self.subTest(goal_cell=goal_cell):
                adapter, world = self._adapter_world()
                snapshot = self._snapshot(world, goal_cell=goal_cell)
                snapshot["places"][0]["source"] = "image_pick"
                state = self._install_snapshot_provider(adapter, snapshot)
                ctx, results, setter = self._ctx(world)
                calls: list[dict] = []
                planner_tolerances: list[float] = []
                real_planner = official_v2_tools.plan_clearance_path

                def recording_planner(planner_snapshot, *args, **kwargs):
                    planner_tolerances.append(
                        float(kwargs["arrival_tolerance_m"])
                    )
                    return real_planner(planner_snapshot, *args, **kwargs)

                with tempfile.TemporaryDirectory() as temp_root:
                    with mock.patch.dict(
                        os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
                    ):
                        with mock.patch.object(
                            official_v2_tools,
                            "plan_clearance_path",
                            side_effect=recording_planner,
                        ):
                            with mock.patch.object(
                                official_v2_tools,
                                "_yield_adjust_chassis_controller",
                                side_effect=self._controller_stub(state, calls),
                            ):
                                actions = list(
                                    build_registry(adapter)["navigate_to"].fn(
                                        ctx,
                                        name=GOAL_NAME,
                                        session_id="navigate_image_pick_standoff",
                                        timeout_s=30.0,
                                        arrival_tolerance_m=0.25,
                                    )
                                )

                self.assertEqual(setter.call_count, 1)
                result = results[0]
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["arrival_tolerance_m"], 0.25)
                self.assertEqual(
                    result["goal_arrival_contract"],
                    official_v2_tools.NAVIGATE_TO_IMAGE_PICK_ARRIVAL_CONTRACT,
                )
                expected_effective = (
                    official_v2_tools._navigate_to_image_pick_standoff_tolerance(
                        0.25,
                        official_v2_tools.BASE_FOOTPRINT_RADIUS_M,
                    )
                )
                self.assertAlmostEqual(
                    result["effective_arrival_tolerance_m"],
                    expected_effective,
                )
                self.assertEqual(planner_tolerances, [expected_effective])
                self.assertLessEqual(
                    result["marked_goal_error_m"],
                    expected_effective + 1.0e-9,
                )
                self.assertGreater(result["marked_goal_error_m"], 0.25)
                self.assertTrue(
                    result["planner"]["image_pick_standoff_fallback_used"]
                )
                self.assertTrue(
                    result["planner"]["image_pick_approach_certificate"]["ok"]
                )
                self._assert_action_contract(actions)
                np.testing.assert_allclose(
                    actions[-1][ACTION_SLICES["base"]], 0.0
                )

    def test_image_pick_standoff_does_not_regress_to_exact_on_replan(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(30, 10),
            goal_cell=(2, 48),
        )
        snapshot["places"][0]["source"] = "image_pick"
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []
        planner_tolerances: list[tuple[str, float]] = []
        changed = False
        real_planner = official_v2_tools.plan_clearance_path

        def recording_planner(planner_snapshot, *args, **kwargs):
            planner_tolerances.append(
                (
                    str(planner_snapshot["map_version"]),
                    float(kwargs["arrival_tolerance_m"]),
                )
            )
            return real_planner(planner_snapshot, *args, **kwargs)

        def correct_map_after_segment(kind, current_state, _world):
            nonlocal changed
            if kind != "forward" or changed:
                return
            changed = True
            current = current_state["snapshot"]
            current["map_version"] = "map-v2"
            current["occupancy"][0, :] = 0
            current["places"][0]["y"] += 0.20

        controller = self._controller_stub(
            state,
            calls,
            after_phase=correct_map_after_segment,
        )
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=recording_planner,
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
                                session_id="navigate_image_pick_replan_floor",
                                timeout_s=30.0,
                                arrival_tolerance_m=0.25,
                            )
                        )

        self.assertTrue(changed)
        self.assertEqual(setter.call_count, 1)
        result = results[0]
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["replan_count"], 1)
        expected_standoff = (
            official_v2_tools._navigate_to_image_pick_standoff_tolerance(
                0.25,
                official_v2_tools.BASE_FOOTPRINT_RADIUS_M,
            )
        )
        self.assertEqual(planner_tolerances[:1], [("map-v1", expected_standoff)])
        map_v2_tolerances = [
            tolerance
            for version, tolerance in planner_tolerances
            if version == "map-v2"
        ]
        self.assertTrue(map_v2_tolerances, planner_tolerances)
        self.assertTrue(
            all(
                abs(tolerance - expected_standoff) <= 1.0e-9
                for tolerance in map_v2_tolerances
            ),
            planner_tolerances,
        )
        self.assertEqual(
            [record["goal_arrival_contract"] for record in result["plan_records"]],
            [
                official_v2_tools.NAVIGATE_TO_IMAGE_PICK_ARRIVAL_CONTRACT,
                official_v2_tools.NAVIGATE_TO_IMAGE_PICK_ARRIVAL_CONTRACT,
            ],
        )
        self.assertTrue(
            result["planner"]["image_pick_standoff_preserved_across_replan"]
        )
        self._assert_action_contract(actions)

    def test_episode_and_map_epoch_changes_mid_execution_fail_closed(self) -> None:
        for changed_identity in ("episode", "map_epoch"):
            with self.subTest(changed_identity=changed_identity):
                adapter, world = self._adapter_world()
                snapshot = self._snapshot(
                    world,
                    start_cell=(20, 30),
                    goal_cell=(38, 30),
                )
                state = self._install_snapshot_provider(adapter, snapshot)
                ctx, results, setter = self._ctx(world)
                calls: list[dict] = []
                changed = False

                def change_after_segment(kind, current_state, current_world):
                    nonlocal changed
                    if kind != "forward" or changed:
                        return
                    changed = True
                    if changed_identity == "episode":
                        current_world.reset_observation_state()
                        current_world.set_episode_initialized(True)
                    else:
                        current_state["snapshot"]["map_epoch"] = "map-epoch-2"

                controller = self._controller_stub(
                    state,
                    calls,
                    after_phase=change_after_segment,
                )
                with tempfile.TemporaryDirectory() as temp_root:
                    with mock.patch.dict(
                        os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
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
                                    session_id=f"navigate_changed_{changed_identity}",
                                    timeout_s=10.0,
                                )
                            )

                self.assertTrue(changed)
                self.assertEqual(setter.call_count, 1)
                self.assertEqual(len(results), 1)
                self.assertFalse(results[0]["ok"], results[0])
                self.assertEqual(
                    results[0]["failure_stage"], "start-state validation"
                )
                self.assertEqual(
                    [call["kind"] for call in calls], ["spin", "forward"]
                )
                self.assertGreaterEqual(len(actions), 3)
                self._assert_action_contract(actions)
                np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_tampered_signed_plan_is_rejected_before_motion(self) -> None:
        adapter, world = self._adapter_world()
        self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        real_loader = official_v2_tools._navigate_to_load_plan
        tampered = False

        with tempfile.TemporaryDirectory() as temp_root:

            def tampering_loader(session_id, plan_id):
                nonlocal tampered
                path = Path(temp_root) / session_id / "plans" / f"{plan_id}.json"
                if not tampered:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    record["trajectory"]["path_xy_m"][-1][0] += 0.01
                    path.write_text(json.dumps(record), encoding="utf-8")
                    tampered = True
                return real_loader(session_id, plan_id)

            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_navigate_to_load_plan",
                    side_effect=tampering_loader,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_tamper",
                            timeout_s=10.0,
                        )
                    )

        self.assertTrue(tampered)
        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "start-state validation")
        self.assertIn("digest mismatch", results[0]["error"])
        self.assertEqual(len(actions), 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_map_version_change_replans_before_old_forward_segment(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(30, 10),
            goal_cell=(30, 80),
            shape=(64, 96),
        )
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []
        changed = False

        def replace_map_after_segment(kind, current_state, _world):
            nonlocal changed
            if kind != "forward" or changed:
                return
            changed = True
            current = current_state["snapshot"]
            current["map_version"] = "map-v2"
            current["places"][0]["x"] += 0.20

        controller = self._controller_stub(
            state,
            calls,
            after_phase=replace_map_after_segment,
        )
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_replan",
                            timeout_s=10.0,
                        )
                    )

        self.assertTrue(changed)
        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["replan_count"], 1)
        self.assertEqual(len(result["plan_records"]), 2)
        self.assertEqual(
            [record["map_version"] for record in result["plan_records"]],
            ["map-v1", "map-v2"],
        )
        # The first bounded chord completes, then its causal hold observes
        # map-v2. The remaining chord must come from the replacement plan.
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0]["kind"], "forward")
        self.assertEqual(calls[0]["map_version"], "map-v1")
        self.assertTrue(
            all(call["map_version"] == "map-v2" for call in calls[1:]),
            calls,
        )
        self.assertGreaterEqual(len(actions), 18)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[1][ACTION_SLICES["base"]], 0.0)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_map_version_change_discards_prefetched_child_action(self) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        controller_calls = 0

        def changing_controller(inner_ctx, **_kwargs):
            nonlocal controller_calls
            controller_calls += 1

            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                current = state["snapshot"]
                current["pose"]["x"] = float(current["pose"]["x"]) + 0.01
                NavigateToExecutionTest._advance_snapshot_pose(current)
                current["map_version"] = "map-v2"
                current["places"] = [
                    {
                        "name": GOAL_NAME,
                        "x": float(current["pose"]["x"]),
                        "y": float(current["pose"]["y"]),
                    }
                ]
                # The second command is part of the already-started bounded
                # chord transaction, so it completes before boundary replan.
                yield inner_ctx.world.make_action(base=[0.2, 0.0, 0.0])
                current["pose"]["x"] = float(current["pose"]["x"]) + 0.01
                NavigateToExecutionTest._advance_snapshot_pose(current)
                return {"ok": True, "action_steps": 2}

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=changing_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_mid_action_replan",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(controller_calls, 1)
        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 1)
        self.assertEqual(results[0]["action_steps"], 18)
        self.assertEqual(len(actions), 19)
        self._assert_action_contract(actions)
        self.assertAlmostEqual(float(actions[0][ACTION_SLICES["base"]][0]), 0.1)
        self.assertAlmostEqual(float(actions[1][ACTION_SLICES["base"]][0]), 0.2)
        np.testing.assert_allclose(actions[2][ACTION_SLICES["base"]], 0.0)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_new_depth_wall_discards_motion_without_overwriting_map(self) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        original_grid = state["snapshot"]["occupancy"].copy()
        ctx, results, setter = self._ctx(world)
        visible = False
        closed = False

        def observed_points(_ctx, _snapshot):
            if not visible:
                return np.empty((0, 3))
            return np.column_stack((np.full(141, 0.52), np.linspace(-3.5, 3.5, 141),
                                    np.full(141, 0.25)))

        def controller(inner_ctx, **_kwargs):
            def phase():
                nonlocal visible, closed
                try:
                    visible = True
                    yield inner_ctx.world.make_action(base=[0.2, 0.0, 0.0])
                    self.fail("a depth-blocked child action must be discarded")
                finally:
                    closed = True
            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(official_v2_tools, "_navigate_to_depth_obstacles",
                                       side_effect=observed_points):
                    with mock.patch.object(official_v2_tools, "_yield_adjust_chassis_controller",
                                           side_effect=controller):
                        actions = list(build_registry(adapter)["navigate_to"].fn(
                            ctx, name=GOAL_NAME, session_id="navigate_new_depth_wall",
                            timeout_s=10.0,
                        ))
                details = results[0]["failure_details"]
                with np.load(details["planning_replay_path"], allow_pickle=False) as replay:
                    np.testing.assert_array_equal(replay["occupancy"], original_grid)
                    self.assertTrue(replay["navigation_obstacle_mask"].any())

        self.assertTrue(closed)
        self.assertEqual(setter.call_count, 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 1)
        self.assertEqual(details["planning_failure_code"], "live_depth_blocks_map_route")
        self.assertTrue(details["map_only_route_available"])
        self.assertGreater(details["live_depth_in_mapped_free_cells"], 0)
        self._assert_action_contract(actions)
        for action in actions:
            np.testing.assert_array_equal(action[ACTION_SLICES["base"]], 0.0)
        np.testing.assert_array_equal(state["snapshot"]["occupancy"], original_grid)
        self.assertFalse(hasattr(world, official_v2_tools.NAVIGATE_TO_DEPTH_GUARD_ATTR))

    def test_high_depth_returns_do_not_block_chassis_navigation(self) -> None:
        ctx = SimpleNamespace(world=SimpleNamespace())
        setattr(ctx.world, official_v2_tools.NAVIGATE_TO_DEPTH_GUARD_ATTR,
                {"stops": 0})
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [0.4, 0.0, 0.0]
        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_depth_obstacles",
            return_value=np.asarray([[0.2, 0.0, 1.2]]),
        ):
            self.assertTrue(
                official_v2_tools._navigate_to_depth_motion_clear(
                    ctx, {}, action
                )
            )
        self.assertEqual(
            getattr(ctx.world, official_v2_tools.NAVIGATE_TO_DEPTH_GUARD_ATTR)[
                "stops"
            ],
            0,
        )

    def test_runtime_depth_guard_uses_oriented_base_polygon(self) -> None:
        ctx = SimpleNamespace(world=SimpleNamespace())
        guard = {"stops": 0}
        setattr(
            ctx.world,
            official_v2_tools.NAVIGATE_TO_DEPTH_GUARD_ATTR,
            guard,
        )
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [0.4, 0.0, 0.0]
        taught_trajectory = {
            "planner": {"traversed_route_direct_replay": True}
        }

        # This return lies inside the old 0.42 m circular guard, but remains
        # more than the RGB-D margin outside the actual side of the base.
        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_depth_obstacles",
            return_value=np.asarray([[0.10, 0.41, 0.25]]),
        ):
            self.assertTrue(
                official_v2_tools._navigate_to_depth_motion_clear(
                    ctx, {}, action, taught_trajectory
                )
            )

        # A low return in the actual forward sweep must still veto motion.
        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_depth_obstacles",
            return_value=np.asarray([[0.30, 0.0, 0.25]]),
        ):
            self.assertFalse(
                official_v2_tools._navigate_to_depth_motion_clear(
                    ctx, {}, action, taught_trajectory
                )
            )
        self.assertEqual(guard["stops"], 1)
        self.assertEqual(guard["last_stop_layer"], "oriented_base_depth_sweep")
        self.assertEqual(
            guard["last_stop"]["layer"], "oriented_base_depth_sweep"
        )
        self.assertEqual(guard["last_stop"]["waypoint_index"], None)
        self.assertGreater(guard["last_stop"]["margin_deficit_m"], 0.0)
        self.assertEqual(
            guard["last_stop"]["point_robot_xyz_m"], [0.3, 0.0, 0.25]
        )

    def test_direct_history_depth_guard_has_only_bounded_numeric_hysteresis(
        self,
    ) -> None:
        ctx = SimpleNamespace(world=SimpleNamespace())
        guard = {"stops": 0}
        setattr(
            ctx.world,
            official_v2_tools.NAVIGATE_TO_DEPTH_GUARD_ATTR,
            guard,
        )
        action = np.zeros(ACTION_DIM)
        action[ACTION_SLICES["base"]] = [.4, 0.0, 0.0]
        taught = {"planner": {"traversed_route_direct_replay": True}}
        point = np.asarray([[.7, 0.0, .25]])

        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_depth_obstacles",
            return_value=point,
        ), mock.patch.object(
            official_v2_tools,
            "convex_sweep_distances",
            side_effect=(np.asarray([.0197]), np.asarray([.10])),
        ):
            self.assertTrue(official_v2_tools._navigate_to_depth_motion_clear(
                ctx, {}, action, taught
            ))

        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_depth_obstacles",
            return_value=point,
        ), mock.patch.object(
            official_v2_tools,
            "convex_sweep_distances",
            side_effect=(np.asarray([.014]), np.asarray([.10])),
        ):
            self.assertFalse(official_v2_tools._navigate_to_depth_motion_clear(
                ctx, {}, action, taught
            ))
        self.assertAlmostEqual(
            guard["last_stop"]["numerical_hysteresis_m"], .005
        )
        self.assertAlmostEqual(
            guard["last_stop"]["enforced_margin_m"], .015
        )

    def test_committed_history_route_survives_transient_depth_plan_layer(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(50, 10),
            goal_cell=(50, 90),
            shape=(100, 100),
        )
        start = (1.05, 5.05)
        goal = (9.05, 5.05)
        snapshot["traversed_paths_xy_m"] = [[
            (x, 5.05) for x in np.linspace(start[0], goal[0], 161)
        ]]
        protected = np.zeros((100, 100), dtype=bool)
        protected[:, 50] = True
        planning_snapshot = deepcopy(snapshot)
        planning_snapshot["navigation_obstacle_mask"] = protected
        planning_snapshot["navigation_depth_points_xy_m"] = np.asarray(
            [[5.05, 5.05]], dtype=np.float64
        )
        ctx, _results, _setter = self._ctx(world)
        args = {
            "name": GOAL_NAME,
            "session_id": "committed_history_depth_layer",
            "timeout_s": 720.0,
            "arrival_tolerance_m": 0.25,
        }

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                official_v2_tools._ensure_session(args["session_id"])
                with mock.patch.object(
                    official_v2_tools,
                    "_navigate_to_depth_planning_snapshot",
                    return_value=planning_snapshot,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_navigate_to_traversed_thin_barrier_recovery",
                        return_value=(
                            None,
                            {
                                "detected": False,
                                "planned": False,
                                "bound": False,
                                "evaluated": True,
                            },
                        ),
                    ):
                        trajectory, _path = official_v2_tools._navigate_to_new_plan(
                            ctx,
                            args=args,
                            storage_session_id=args["session_id"],
                            snapshot=snapshot,
                            deadline=float("inf"),
                            require_traversed_route=True,
                        )

        self.assertTrue(trajectory["planner"]["traversed_route_direct_replay"])
        self.assertTrue(
            trajectory["planner"]["committed_history_runtime_guard_selected"]
        )
        self.assertNotIn("depth_refinement", trajectory)
        self.assertGreater(
            trajectory["planner"]["live_depth_obstacle_cells"], 0
        )

    def test_verified_history_stall_forces_barrier_recovery_on_clear_costmap(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(50, 10),
            goal_cell=(50, 90),
            shape=(100, 100),
        )
        snapshot["traversed_paths_xy_m"] = [[
            (x, 5.05) for x in np.linspace(1.05, 9.05, 161)
        ]]
        ctx, _results, _setter = self._ctx(world)
        args = {
            "name": GOAL_NAME,
            "session_id": "forced_history_barrier_recovery",
            "timeout_s": 720.0,
            "arrival_tolerance_m": 0.25,
        }

        def recover(_ctx, *, map_only_plan, **_kwargs):
            return dict(map_only_plan), {
                "detected": True,
                "planned": True,
                "bound": True,
                "full_observation_certified": True,
            }

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                official_v2_tools._ensure_session(args["session_id"])
                with mock.patch.object(
                    official_v2_tools,
                    "_navigate_to_depth_planning_snapshot",
                    return_value=snapshot,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_navigate_to_traversed_thin_barrier_recovery",
                        side_effect=recover,
                    ) as recovery:
                        trajectory, _path = official_v2_tools._navigate_to_new_plan(
                            ctx,
                            args=args,
                            storage_session_id=args["session_id"],
                            snapshot=snapshot,
                            deadline=float("inf"),
                            require_traversed_route=True,
                            force_traversed_thin_barrier_recovery=True,
                            verified_stall_xy_m=[4.8, 5.05],
                        )

        recovery.assert_called_once()
        self.assertTrue(
            trajectory["planner"]["traversed_thin_barrier_recovery_forced"]
        )
        barrier = trajectory["planner"]["traversed_thin_barrier_recovery"]
        self.assertTrue(barrier["evaluated"])
        self.assertTrue(barrier["forced_after_verified_stall"])
        self.assertTrue(barrier["selected"])
        self.assertFalse(
            trajectory["planner"].get(
                "committed_history_runtime_guard_selected", False
            )
        )

    def test_verified_stall_does_not_replay_same_history_leaf_contact(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(50, 10),
            goal_cell=(50, 90),
            shape=(100, 100),
        )
        snapshot["traversed_paths_xy_m"] = [[
            (x, 5.05) for x in np.linspace(1.05, 9.05, 161)
        ]]
        ctx, _results, _setter = self._ctx(world)
        args = {
            "name": GOAL_NAME,
            "session_id": "forced_history_contact_rejected",
            "timeout_s": 720.0,
            "arrival_tolerance_m": 0.25,
        }

        def recover(_ctx, *, map_only_plan, **_kwargs):
            recovered = dict(map_only_plan)
            recovered["traversed_thin_barrier"] = {
                "route_source": "traversed_history",
            }
            return recovered, {
                "detected": True,
                "planned": True,
                "bound": True,
                "full_observation_certified": False,
            }

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                official_v2_tools._ensure_session(args["session_id"])
                with mock.patch.object(
                    official_v2_tools,
                    "_navigate_to_depth_planning_snapshot",
                    return_value=snapshot,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_navigate_to_traversed_thin_barrier_recovery",
                        side_effect=recover,
                    ):
                        trajectory, _path = official_v2_tools._navigate_to_new_plan(
                            ctx,
                            args=args,
                            storage_session_id=args["session_id"],
                            snapshot=snapshot,
                            deadline=float("inf"),
                            require_traversed_route=True,
                            force_traversed_thin_barrier_recovery=True,
                            verified_stall_xy_m=[4.8, 5.05],
                        )

        barrier = trajectory["planner"]["traversed_thin_barrier_recovery"]
        self.assertTrue(barrier["evaluated"])
        self.assertTrue(barrier["forced_after_verified_stall"])
        self.assertFalse(barrier["selected"])
        self.assertIsNone(trajectory.get("traversed_thin_barrier"))
        self.assertFalse(
            trajectory["planner"]["depth_detour_recovery_selected"]
        )

    def test_transient_uncertain_hold_discards_prefetched_motion_and_replans(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(
            adapter, self._snapshot(world, goal_cell=(30, 15))
        )
        ctx, results, setter = self._ctx(world)
        controller_calls = 0
        held_reads = 0

        def current_snapshot():
            nonlocal held_reads
            current = state["snapshot"]
            if current["lifecycle"].get("uncertain_hold", False):
                held_reads += 1
                if held_reads >= 3:
                    current["lifecycle"]["uncertain_hold"] = False
                    current["map_version"] = "map-v2"
                    current["places"] = [
                        {
                            "name": GOAL_NAME,
                            "x": float(current["pose"]["x"]),
                            "y": float(current["pose"]["y"]),
                        }
                    ]
            return deepcopy(current)

        adapter.navigation_map_snapshot = current_snapshot

        def held_controller(inner_ctx, **_kwargs):
            nonlocal controller_calls
            controller_calls += 1

            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                current = state["snapshot"]
                current["pose"]["x"] = float(current["pose"]["x"]) + 0.05
                self._advance_snapshot_pose(current)
                current["lifecycle"]["uncertain_hold"] = True
                # This action is generated inside the child but must never be
                # emitted after the observation reports localization hold.
                yield inner_ctx.world.make_action(base=[0.9, 0.0, 0.0])
                return {"ok": True, "action_steps": 2}

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=held_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_transient_uncertain_hold",
                            timeout_s=20.0,
                        )
                    )

        self.assertEqual(controller_calls, 1)
        self.assertGreaterEqual(held_reads, 3)
        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 1)
        self._assert_action_contract(actions)
        self.assertAlmostEqual(float(actions[0][ACTION_SLICES["base"]][0]), 0.1)
        self.assertFalse(
            any(
                abs(float(action[ACTION_SLICES["base"]][0]) - 0.9) < 1.0e-6
                for action in actions
            )
        )
        np.testing.assert_allclose(actions[1][ACTION_SLICES["base"]], 0.0)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_motion_requires_a_new_navigation_observation(self) -> None:
        adapter, world = self._adapter_world()
        self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)

        def stale_controller(inner_ctx, **_kwargs):
            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                # A second command without a newer map pose must be discarded.
                yield inner_ctx.world.make_action(base=[0.8, 0.0, 0.0])
                return {"ok": True, "action_steps": 2}

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=stale_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_stale_observation",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "observation validation")
        self.assertIn("did not advance", results[0]["error"])
        self.assertEqual(results[0]["action_steps"], 1)
        self.assertEqual(len(actions), 2)
        self._assert_action_contract(actions)
        self.assertAlmostEqual(float(actions[0][ACTION_SLICES["base"]][0]), 0.1)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_localization_preflight_waits_for_source_to_cover_invocation(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        snapshot["pose_source_sequence_known"] = False
        snapshot.pop("pose_source_observation_sequence", None)
        reads = 0

        def current_snapshot():
            nonlocal reads
            reads += 1
            if reads == 2:
                self._advance_snapshot_observation(snapshot)
                snapshot["pose_source_sequence_known"] = True
                snapshot["pose_source_observation_sequence"] = int(
                    snapshot["observation_sequence"]
                )
            return deepcopy(snapshot)

        adapter.navigation_map_snapshot = current_snapshot
        state = {"snapshot": snapshot}
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=self._controller_stub(state, calls),
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_localization_preflight",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["localization_preflight_steps"], 1)
        self.assertEqual(results[0]["invocation_observation_sequence"], 1)
        self.assertGreaterEqual(len(calls), 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[0][ACTION_SLICES["base"]], 0.0)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_low_frequency_rtab_pose_waits_past_wall_age_limit(self) -> None:
        for ordered_progress in (False, True):
            with self.subTest(ordered_progress=ordered_progress):
                adapter, world = self._adapter_world()
                snapshot = self._snapshot(world, goal_cell=(30, 15))
                snapshot.update(
                    {
                        "backend": (
                            "behavior_interface.rtabmap_slam.live.LiveMapper"
                        ),
                        "observation_sequence": 25,
                        "captured_ts": 102.161,
                        "pose_observed_sequence": 1,
                        "pose_observed_ts": 100.0,
                        "pose_age_observations": 24,
                        "pose_age_s": 2.161,
                        "pose_source_sequence_known": False,
                        "source_frame": {
                            "backend_frame_id": 1,
                            "lag_known": False,
                        },
                    }
                )
                snapshot.pop("pose_source_observation_sequence", None)
                if ordered_progress:
                    snapshot["source_frame"]["ordered_pose_producer"] = {
                        "schema": (
                            "behavior.official.navigation_map."
                            "ordered_pose_producer.v1"
                        ),
                        "enqueued_frame_count": 2,
                        "processed_frame_count": 1,
                        "pose_frame_id": 1,
                        "last_enqueued_observation_sequence": 25,
                    }
                reads = 0

                def current_snapshot():
                    nonlocal reads
                    reads += 1
                    if reads == 2:
                        snapshot["pose_version"] = "map-epoch-1|fixture-pose:2"
                        snapshot["frame_version"] = snapshot["pose_version"]
                        snapshot["pose_observed_sequence"] = 26
                        snapshot["pose_observed_ts"] = 102.261
                        snapshot["observation_sequence"] = 26
                        snapshot["captured_ts"] = 102.261
                        snapshot["pose_age_observations"] = 0
                        snapshot["pose_age_s"] = 0.0
                        if ordered_progress:
                            progress = snapshot["source_frame"][
                                "ordered_pose_producer"
                            ]
                            progress["enqueued_frame_count"] = 3
                            progress["processed_frame_count"] = 2
                            progress["pose_frame_id"] = 2
                            progress[
                                "last_enqueued_observation_sequence"
                            ] = 26
                        else:
                            snapshot["pose_source_sequence_known"] = True
                            snapshot[
                                "pose_source_observation_sequence"
                            ] = 26
                            snapshot["source_frame"]["lag_known"] = True
                    elif reads == 3 and ordered_progress:
                        snapshot["pose_source_sequence_known"] = True
                        snapshot["pose_source_observation_sequence"] = 26
                    return deepcopy(snapshot)

                adapter.navigation_map_snapshot = current_snapshot
                state = {"snapshot": snapshot}
                ctx, results, setter = self._ctx(world)
                calls: list[dict] = []
                with tempfile.TemporaryDirectory() as temp_root:
                    with mock.patch.dict(
                        os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
                    ):
                        with mock.patch.object(
                            official_v2_tools,
                            "_yield_adjust_chassis_controller",
                            side_effect=self._controller_stub(state, calls),
                        ):
                            actions = list(
                                build_registry(adapter)["navigate_to"].fn(
                                    ctx,
                                    name=GOAL_NAME,
                                    session_id=(
                                        "navigate_slow_ordered"
                                        if ordered_progress
                                        else "navigate_slow_legacy_rtab"
                                    ),
                                    timeout_s=10.0,
                                )
                            )

                self.assertEqual(setter.call_count, 1)
                self.assertTrue(results[0]["ok"], results[0])
                self.assertEqual(results[0]["localization_preflight_steps"], 0)
                self.assertGreaterEqual(len(calls), 1)
                self._assert_action_contract(actions)
                self.assertTrue(
                    any(
                        np.any(
                            np.abs(action[ACTION_SLICES["base"]]) > 1.0e-9
                        )
                        for action in actions[:-1]
                    )
                )

    def test_async_slam_pose_is_continuously_propagated_by_policy_odometry(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        snapshot.update(
            {
                "backend": "behavior_interface.rtabmap_slam.LiveMapper",
                "observation_sequence": 25,
                "captured_ts": 102.161,
                "pose_observed_sequence": 1,
                "pose_observed_ts": 100.0,
                "pose_age_observations": 24,
                "pose_age_s": 2.161,
                "pose_source_sequence_known": False,
                "source_frame": {
                    "backend_frame_id": 1,
                    "lag_known": False,
                },
            }
        )
        snapshot.pop("pose_source_observation_sequence", None)
        ctx, _results, _setter = self._ctx(world)
        world._mock_base[:] = [1.0, 2.0, 0.0]
        cleanup: dict = {}
        anchored = official_v2_tools._navigate_to_begin_pose_fusion(
            ctx, snapshot, cleanup
        )
        start_pose = official_v2_tools._navigate_to_pose(anchored)

        advanced = deepcopy(snapshot)
        advanced["observation_sequence"] = 26
        advanced["captured_ts"] = 102.261
        advanced["pose_age_observations"] = 25
        advanced["pose_age_s"] = 2.261
        world._mock_base[:] = [1.4, 2.0, 0.1]
        adapter.navigation_map_snapshot = lambda: deepcopy(advanced)
        fused = official_v2_tools._navigate_to_snapshot(
            ctx,
            adapter,
            require_localization_ready=False,
        )
        fused_pose = official_v2_tools._navigate_to_pose(fused)

        self.assertAlmostEqual(fused_pose[0], start_pose[0] + 0.4, places=6)
        self.assertAlmostEqual(fused_pose[1], start_pose[1], places=6)
        self.assertAlmostEqual(fused_pose[2], start_pose[2] + 0.1, places=6)
        self.assertEqual(fused["pose_source_observation_sequence"], 26)
        self.assertEqual(fused["pose_age_s"], 0.0)
        self.assertEqual(
            fused["source_frame"]["navigation_pose_fusion"]["schema"],
            official_v2_tools.NAVIGATE_TO_POSE_FUSION_SCHEMA,
        )
        official_v2_tools._navigate_to_clear_pose_fusion(ctx, cleanup)
        self.assertFalse(
            hasattr(world, official_v2_tools.NAVIGATE_TO_POSE_FUSION_ATTR)
        )

    def test_async_map_export_lag_is_time_aligned_to_current_observation(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        for _ in range(3):
            adapter.update(
                {"robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32)}
            )
        current_status = adapter.status()
        current_sequence = int(current_status["sequence"])
        current_ts = float(current_status["received_ts"])

        snapshot = self._snapshot(world, goal_cell=(30, 15))
        snapshot.update(
            {
                "backend": "behavior_interface.rtabmap_slam.live.LiveMapper",
                "captured_ts": current_ts - 0.8,
                "pose_observed_ts": current_ts - 0.8,
                "policy_local_pose": {
                    "x": 1.0,
                    "y": 2.0,
                    "yaw_rad": 0.0,
                    "frame": "policy_local_odometry",
                },
                "source_frame": {
                    "backend_frame_id": 1,
                    "lag_known": True,
                    "ordered_pose_producer": {
                        "schema": (
                            "behavior.official.navigation_map."
                            "ordered_pose_producer.v1"
                        ),
                        "enqueued_frame_count": 1,
                        "processed_frame_count": 1,
                        "pose_frame_id": 1,
                        "last_enqueued_observation_sequence": 1,
                    },
                },
            }
        )
        adapter.navigation_map_snapshot = lambda: deepcopy(snapshot)
        ctx, _results, _setter = self._ctx(world)
        world._mock_base[:] = [1.4, 2.0, 0.1]

        accepted = official_v2_tools._navigate_to_snapshot(
            ctx,
            adapter,
            require_localization_ready=False,
        )
        transport = accepted["source_frame"]["navigation_snapshot_transport"]
        self.assertEqual(transport["snapshot_observation_sequence"], 1)
        self.assertEqual(transport["current_observation_sequence"], current_sequence)

        cleanup: dict = {}
        fused = official_v2_tools._navigate_to_begin_pose_fusion(
            ctx, accepted, cleanup
        )
        try:
            raw_pose = official_v2_tools._navigate_to_pose(snapshot)
            fused_pose = official_v2_tools._navigate_to_pose(fused)
            self.assertAlmostEqual(fused_pose[0], raw_pose[0] + 0.4, places=6)
            self.assertAlmostEqual(fused_pose[1], raw_pose[1], places=6)
            self.assertAlmostEqual(fused_pose[2], raw_pose[2] + 0.1, places=6)
            self.assertEqual(fused["observation_sequence"], current_sequence)
            self.assertEqual(
                fused["pose_source_observation_sequence"], current_sequence
            )
            map_snapshot = fused["source_frame"]["navigation_map_snapshot"]
            self.assertEqual(map_snapshot["observation_sequence"], 1)
            self.assertEqual(
                map_snapshot["lag_observations"], current_sequence - 1
            )
            self.assertFalse(
                fused["source_frame"]["navigation_pose_fusion"][
                    "backend_pose_caught_up"
                ]
            )
        finally:
            official_v2_tools._navigate_to_clear_pose_fusion(ctx, cleanup)

    def test_stationary_single_observation_survives_slow_planning_wall_time(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        adapter.update(
            {"robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32)}
        )
        current_status = adapter.status()
        current_sequence = int(current_status["sequence"])
        current_ts = float(current_status["received_ts"])
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        snapshot.update(
            {
                "backend": "behavior_interface.rtabmap_slam.live.LiveMapper",
                "observation_sequence": current_sequence - 1,
                "captured_ts": current_ts - 6.02,
                "pose_observed_sequence": current_sequence - 1,
                "pose_observed_ts": current_ts - 6.02,
                "pose_age_observations": 0,
                "pose_age_s": 0.0,
                "pose_source_sequence_known": True,
                "pose_source_observation_sequence": current_sequence - 1,
                "policy_local_pose": {
                    "x": 0.0,
                    "y": 0.0,
                    "yaw_rad": 0.0,
                    "frame": "policy_local_odometry",
                },
                "source_frame": {
                    "backend_frame_id": 1,
                    "lag_known": True,
                    "ordered_pose_producer": {
                        "schema": (
                            "behavior.official.navigation_map."
                            "ordered_pose_producer.v1"
                        ),
                        "enqueued_frame_count": 1,
                        "processed_frame_count": 1,
                        "pose_frame_id": 1,
                        "last_enqueued_observation_sequence": (
                            current_sequence - 1
                        ),
                    },
                },
            }
        )
        adapter.navigation_map_snapshot = lambda: deepcopy(snapshot)
        ctx, _results, _setter = self._ctx(world)

        accepted = official_v2_tools._navigate_to_snapshot(
            ctx, adapter, require_localization_ready=False
        )

        transport = accepted["source_frame"]["navigation_snapshot_transport"]
        self.assertEqual(transport["lag_observations"], 1)
        self.assertGreater(transport["lag_s"], 5.0)
        self.assertEqual(
            transport["lag_bound"], "stationary_single_observation"
        )

    def test_active_pose_fusion_bridges_old_anchor_with_one_frame_transport_lag(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        for _ in range(20):
            adapter.update(
                {"robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32)}
            )
        current_status = adapter.status()
        current_sequence = int(current_status["sequence"])
        current_ts = float(current_status["received_ts"])
        initial = self._snapshot(world, goal_cell=(30, 15))
        initial.update(
            {
                "backend": "behavior_interface.rtabmap_slam.live.LiveMapper",
                "observation_sequence": current_sequence - 2,
                "captured_ts": current_ts - 0.2,
                "pose_observed_sequence": current_sequence - 2,
                "pose_observed_ts": current_ts - 0.2,
                "pose_age_observations": 0,
                "pose_age_s": 0.0,
                "pose_source_sequence_known": True,
                "pose_source_observation_sequence": current_sequence - 2,
                "policy_local_pose": {
                    "x": 0.0,
                    "y": 0.0,
                    "yaw_rad": 0.0,
                    "frame": "policy_local_odometry",
                },
                "source_frame": {"backend_frame_id": 1},
            }
        )
        ctx, _results, _setter = self._ctx(world)
        cleanup: dict = {}
        anchored = official_v2_tools._navigate_to_begin_pose_fusion(
            ctx, initial, cleanup
        )
        anchor_pose = official_v2_tools._navigate_to_pose(anchored)

        delayed = deepcopy(initial)
        delayed.update(
            {
                "observation_sequence": current_sequence - 1,
                "captured_ts": current_ts - 0.1,
                "pose_observed_sequence": 1,
                "pose_observed_ts": current_ts - 14.83,
                "pose_age_observations": current_sequence - 2,
                "pose_age_s": 14.73,
                "pose_source_sequence_known": False,
                "policy_local_pose": {
                    "x": 0.0,
                    "y": 0.0,
                    "yaw_rad": 0.0,
                    "frame": "policy_local_odometry",
                },
                "source_frame": {"backend_frame_id": 2},
            }
        )
        delayed.pop("pose_source_observation_sequence", None)
        adapter.navigation_map_snapshot = lambda: deepcopy(delayed)
        world._mock_base[:] = [0.012, 0.0, 0.0]
        try:
            accepted = official_v2_tools._navigate_to_snapshot(
                ctx, adapter, require_localization_ready=False
            )
        finally:
            official_v2_tools._navigate_to_clear_pose_fusion(ctx, cleanup)

        accepted_pose = official_v2_tools._navigate_to_pose(accepted)
        self.assertAlmostEqual(accepted_pose[0], anchor_pose[0] + 0.012, places=6)
        self.assertAlmostEqual(accepted_pose[1], anchor_pose[1], places=6)
        self.assertEqual(accepted["pose_age_s"], 0.0)
        self.assertEqual(accepted["observation_sequence"], current_sequence)

    def test_slow_async_snapshot_still_rejects_unaccounted_chassis_motion(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        adapter.update(
            {"robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32)}
        )
        current_status = adapter.status()
        current_sequence = int(current_status["sequence"])
        current_ts = float(current_status["received_ts"])
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        snapshot.update(
            {
                "backend": "behavior_interface.rtabmap_slam.live.LiveMapper",
                "observation_sequence": current_sequence - 1,
                "captured_ts": current_ts - 6.02,
                "pose_observed_sequence": current_sequence - 1,
                "pose_observed_ts": current_ts - 6.02,
                "pose_age_observations": 0,
                "pose_age_s": 0.0,
                "pose_source_sequence_known": True,
                "pose_source_observation_sequence": current_sequence - 1,
                "policy_local_pose": {
                    "x": 0.0,
                    "y": 0.0,
                    "yaw_rad": 0.0,
                    "frame": "policy_local_odometry",
                },
                "source_frame": {"backend_frame_id": 1},
            }
        )
        adapter.navigation_map_snapshot = lambda: deepcopy(snapshot)
        world._mock_base[:] = [0.03, 0.0, 0.0]
        ctx, _results, _setter = self._ctx(world)

        with self.assertRaisesRegex(
            official_v2_tools._NavigateToFailure,
            "snapshot is stale",
        ):
            official_v2_tools._navigate_to_snapshot(
                ctx, adapter, require_localization_ready=False
            )

    def test_navigation_waits_for_retryable_snapshot_transport_lag(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        ctx, _results, _setter = self._ctx(world)
        stale = official_v2_tools._NavigateToFailure(
            "observation validation",
            "navigation map snapshot is stale for the current observation",
            details={
                "transport_catchup_retryable": True,
                "observation_lag": 1,
            },
        )

        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_snapshot",
            side_effect=[stale, deepcopy(snapshot)],
        ):
            waiter = official_v2_tools._navigate_to_wait_for_transport_snapshot(
                ctx,
                adapter,
                active_trajectory=None,
                deadline=float("inf"),
                remaining_steps=10,
                require_localization_ready=False,
            )
            hold = next(waiter)
            with self.assertRaises(StopIteration) as stopped:
                next(waiter)

        self.assertFalse(
            official_v2_tools._navigate_to_action_has_base_motion(hold)
        )
        outcome = stopped.exception.value
        self.assertEqual(
            outcome["snapshot"]["map_version"], snapshot["map_version"]
        )
        np.testing.assert_array_equal(
            outcome["snapshot"]["occupancy"], snapshot["occupancy"]
        )
        self.assertEqual(outcome["action_steps"], 1)
        self.assertEqual(outcome["transport_recovery_steps"], 1)
        self.assertEqual(outcome["last_transport_lag"]["observation_lag"], 1)

    def test_navigation_does_not_retry_unclassified_snapshot_failure(self) -> None:
        adapter, world = self._adapter_world()
        ctx, _results, _setter = self._ctx(world)
        invalid = official_v2_tools._NavigateToFailure(
            "observation validation",
            "navigation map snapshot has an unsupported schema",
        )

        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_snapshot",
            side_effect=invalid,
        ):
            waiter = official_v2_tools._navigate_to_wait_for_transport_snapshot(
                ctx,
                adapter,
                active_trajectory=None,
                deadline=float("inf"),
                remaining_steps=10,
                require_localization_ready=False,
            )
            with self.assertRaisesRegex(
                official_v2_tools._NavigateToFailure,
                "unsupported schema",
            ):
                next(waiter)

    def test_lagging_snapshot_requires_async_contract_and_source_odometry(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        adapter.update(
            {"robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32)}
        )
        current_ts = float(adapter.status()["received_ts"])
        base = self._snapshot(world, goal_cell=(30, 15))
        base["captured_ts"] = current_ts - 0.2
        base["pose_observed_ts"] = current_ts - 0.2
        ctx, _results, _setter = self._ctx(world)

        with self.assertRaisesRegex(
            official_v2_tools._NavigateToFailure,
            "snapshot is stale",
        ):
            adapter.navigation_map_snapshot = lambda: deepcopy(base)
            official_v2_tools._navigate_to_snapshot(
                ctx, adapter, require_localization_ready=False
            )

        async_without_source_pose = deepcopy(base)
        async_without_source_pose.update(
            {
                "backend": "behavior_interface.rtabmap_slam.live.LiveMapper",
                "source_frame": {"backend_frame_id": 1},
            }
        )
        adapter.navigation_map_snapshot = lambda: deepcopy(
            async_without_source_pose
        )
        with self.assertRaisesRegex(
            official_v2_tools._NavigateToFailure,
            "snapshot is stale",
        ):
            official_v2_tools._navigate_to_snapshot(
                ctx, adapter, require_localization_ready=False
            )

    def test_delayed_slam_correction_uses_odometry_at_source_sequence(self) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["backend"] = "behavior_interface.rtabmap_slam.LiveMapper"
        snapshot["source_frame"] = {"backend_frame_id": 1}
        snapshot["pose"].update(x=2.0, y=3.0, yaw_deg=0.0)
        ctx, _results, _setter = self._ctx(world)
        world._mock_base[:] = [0.0, 0.0, 0.0]
        cleanup = {}
        official_v2_tools._navigate_to_begin_pose_fusion(ctx, snapshot, cleanup)
        try:
            snapshot["observation_sequence"] = 2
            world._mock_base[:] = [1.0, 0.0, 0.0]
            official_v2_tools._navigate_to_apply_pose_fusion(ctx, snapshot)

            snapshot["observation_sequence"] = 3
            snapshot["pose_source_observation_sequence"] = 2
            snapshot["pose_version"] = "delayed-correction-2"
            snapshot["pose"].update(x=3.0, y=3.4, yaw_deg=90.0)
            world._mock_base[:] = [2.0, 0.0, 0.0]
            fused = official_v2_tools._navigate_to_apply_pose_fusion(ctx, snapshot)
            np.testing.assert_allclose(
                official_v2_tools._navigate_to_pose(fused),
                [3.0, 4.4, math.pi / 2.0], atol=1e-9,
            )
            diagnostics = fused["source_frame"]["navigation_pose_fusion"]
            self.assertFalse(diagnostics["backend_pose_caught_up"])
            self.assertTrue(diagnostics["backend_pose_history_matched"])
            self.assertEqual(diagnostics["anchor_observation_sequence"], 2)
            self.assertEqual(diagnostics["slam_corrections"], 1)

            for source_sequence in (1, 5):
                with self.subTest(older_or_missing_source=source_sequence):
                    snapshot["observation_sequence"] = 6
                    snapshot["pose_source_observation_sequence"] = source_sequence
                    snapshot["pose_version"] = f"unusable-{source_sequence}"
                    snapshot["pose"].update(x=-99.0, y=-99.0)
                    fused = official_v2_tools._navigate_to_apply_pose_fusion(
                        ctx, snapshot,
                    )
                    np.testing.assert_allclose(
                        official_v2_tools._navigate_to_pose(fused),
                        [3.0, 4.4, math.pi / 2.0], atol=1e-9,
                    )
        finally:
            official_v2_tools._navigate_to_clear_pose_fusion(ctx, cleanup)

    def test_pose_fusion_history_is_bounded(self) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["backend"] = "behavior_interface.rtabmap_slam.LiveMapper"
        snapshot["source_frame"] = {"backend_frame_id": 1}
        ctx, _results, _setter = self._ctx(world)
        cleanup = {}
        official_v2_tools._navigate_to_begin_pose_fusion(ctx, snapshot, cleanup)
        try:
            with mock.patch.object(official_v2_tools, "NAVIGATE_TO_POSE_HISTORY_SIZE", 2):
                for sequence in range(2, 10):
                    snapshot["observation_sequence"] = sequence
                    official_v2_tools._navigate_to_apply_pose_fusion(ctx, snapshot)
            state = getattr(world, official_v2_tools.NAVIGATE_TO_POSE_FUSION_ATTR)
            self.assertEqual(list(state["policy_pose_history"]), [8, 9])
        finally:
            official_v2_tools._navigate_to_clear_pose_fusion(ctx, cleanup)

    def test_slow_ordered_pose_producer_uses_policy_odometry(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        snapshot.update(
            {
                "backend": "behavior_interface.rtabmap_slam.live.LiveMapper",
                "observation_sequence": 25,
                "captured_ts": 102.161,
                "pose_observed_sequence": 1,
                "pose_observed_ts": 100.0,
                "pose_age_observations": 24,
                "pose_age_s": 2.161,
                "pose_source_sequence_known": False,
                "source_frame": {
                    "backend_frame_id": 1,
                    "lag_known": False,
                    "ordered_pose_producer": {
                        "schema": (
                            "behavior.official.navigation_map."
                            "ordered_pose_producer.v1"
                        ),
                        "enqueued_frame_count": 2,
                        "processed_frame_count": 1,
                        "pose_frame_id": 1,
                        "last_enqueued_observation_sequence": 25,
                    },
                },
            }
        )
        snapshot.pop("pose_source_observation_sequence", None)
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=self._controller_stub(state, calls),
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_slow_ordered_policy_odometry",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["localization_preflight_steps"], 0)
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual(
            results[0]["initial_causal_localization"]["assurance"],
            "slam_anchor_plus_policy_odometry",
        )
        self._assert_action_contract(actions)
        self.assertTrue(
            any(
                np.any(np.abs(action[ACTION_SLICES["base"]]) > 1.0e-9)
                for action in actions[:-1]
            )
        )

    def test_arrival_does_not_reuse_a_witness_from_an_earlier_motion(self) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        snapshot["observation_sequence"] = 20
        snapshot["pose_source_sequence_known"] = False
        snapshot.pop("pose_source_observation_sequence", None)
        snapshot["source_frame"] = {
            "backend_frame_id": 8,
            "lag_known": False,
            "ordered_pose_producer": {
                "schema": (
                    "behavior.official.navigation_map."
                    "ordered_pose_producer.v1"
                ),
                "enqueued_frame_count": 8,
                "processed_frame_count": 8,
                "pose_frame_id": 8,
                "last_enqueued_observation_sequence": 20,
            },
        }
        stale_witness = {
            "assurance": "ordered_producer_frame",
            "required_observation_sequence": 10,
            "strictly_after": True,
            "ordered_target_frame": 4,
            "processed_frame_count": 4,
            "pose_version": snapshot["pose_version"],
            "map_epoch": snapshot["map_epoch"],
        }
        ctx, _results, _setter = self._ctx(world)
        hold = world.hold_action()

        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_active_snapshot",
            return_value=(deepcopy(snapshot), False),
        ), mock.patch.object(
            official_v2_tools,
            "_navigate_to_validate_inactive_limb_contract",
            return_value=hold,
        ), mock.patch.object(
            official_v2_tools,
            "_navigate_to_base_is_settled",
            return_value=True,
        ):
            settle = official_v2_tools._navigate_to_wait_for_arrival_settle(
                ctx,
                object(),
                active_trajectory={},
                deadline=float("inf"),
                remaining_steps=100,
                minimum_source_observation_sequence=15,
                causal_localization=stale_witness,
            )
            actions = []
            while True:
                try:
                    actions.append(next(settle))
                except StopIteration as stopped:
                    outcome = stopped.value
                    break

        self.assertEqual(len(actions), official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS)
        self.assertEqual(
            outcome["causal_localization"]["ordered_target_frame"], 8
        )
        self.assertEqual(
            outcome["causal_localization"]["required_observation_sequence"],
            15,
        )

    def test_frozen_pose_with_advancing_evaluator_sequence_never_continues(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        state = {"snapshot": snapshot, "motion_prefetched": False}

        def current_snapshot():
            if state["motion_prefetched"]:
                self._advance_snapshot_observation(snapshot)
            return deepcopy(snapshot)

        adapter.navigation_map_snapshot = current_snapshot
        ctx, results, setter = self._ctx(world)
        controller_calls = 0

        def frozen_controller(inner_ctx, **_kwargs):
            nonlocal controller_calls
            controller_calls += 1

            def phase():
                state["motion_prefetched"] = True
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                return {"ok": True, "action_steps": 1, "linear_remaining_m": 0.0}

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=frozen_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_frozen_pose",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(controller_calls, 1)
        self.assertEqual(setter.call_count, 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "observation validation")
        self.assertIn("pose producer is stale", results[0]["error"])
        moving = [
            action
            for action in actions
            if np.any(np.abs(action[ACTION_SLICES["base"]]) > 1.0e-9)
        ]
        self.assertEqual(len(moving), 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_new_pose_revision_with_old_source_sequence_never_continues(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        state = {"motion_prefetched": False, "queued_revision_published": False}

        def current_snapshot():
            if state["motion_prefetched"]:
                self._advance_snapshot_observation(snapshot)
                if not state["queued_revision_published"]:
                    state["queued_revision_published"] = True
                    queued_version = (
                        f"{snapshot['map_epoch']}|fixture-queued-pose:2"
                    )
                    snapshot["frame_version"] = queued_version
                    snapshot["pose_version"] = queued_version
                    snapshot["pose_observed_sequence"] = int(
                        snapshot["observation_sequence"]
                    )
                    snapshot["pose_observed_ts"] = float(snapshot["captured_ts"])
                    snapshot["pose_age_observations"] = 0
                    snapshot["pose_age_s"] = 0.0
                    snapshot["pose_source_sequence_known"] = True
                    snapshot["pose_source_observation_sequence"] = 1
            return deepcopy(snapshot)

        adapter.navigation_map_snapshot = current_snapshot
        ctx, results, setter = self._ctx(world)
        controller_calls = 0

        def queued_controller(inner_ctx, **_kwargs):
            nonlocal controller_calls
            controller_calls += 1

            def phase():
                state["motion_prefetched"] = True
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                return {"ok": True, "action_steps": 1, "linear_remaining_m": 0.0}

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=queued_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_queued_pose",
                            timeout_s=10.0,
                        )
                    )

        self.assertTrue(state["queued_revision_published"])
        # A producer revision with the same geometric pose does not invalidate
        # the prefetched chord. Its stale source sequence must still prevent any
        # subsequent segment or success after that one emitted motion.
        self.assertEqual(controller_calls, 1)
        self.assertEqual(setter.call_count, 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "observation validation")
        self.assertIn("pose producer is stale", results[0]["error"])
        moving = [
            action
            for action in actions
            if np.any(np.abs(action[ACTION_SLICES["base"]]) > 1.0e-9)
        ]
        self.assertEqual(len(moving), 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_delayed_causal_pose_confirmation_holds_then_succeeds(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        state = {
            "snapshot": snapshot,
            "pending": False,
            "pending_reads": 0,
            "distance_m": 0.0,
        }

        def current_snapshot():
            if state["pending"]:
                state["pending_reads"] += 1
                self._advance_snapshot_observation(snapshot)
                if state["pending_reads"] == 6:
                    yaw_rad = math.radians(float(snapshot["pose"]["yaw_deg"]))
                    snapshot["pose"]["x"] += state["distance_m"] * math.cos(yaw_rad)
                    snapshot["pose"]["y"] += state["distance_m"] * math.sin(yaw_rad)
                    self._advance_snapshot_pose(snapshot)
                    state["pending"] = False
            return deepcopy(snapshot)

        adapter.navigation_map_snapshot = current_snapshot
        ctx, results, setter = self._ctx(world)

        def delayed_controller(inner_ctx, forward=0.0, **_kwargs):
            def phase():
                state["distance_m"] = float(forward)
                state["pending"] = True
                state["pending_reads"] = 0
                yield inner_ctx.world.make_action(base=[0.2, 0.0, 0.0])
                return {"ok": True, "action_steps": 1, "linear_remaining_m": 0.0}

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=delayed_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_delayed_pose",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertGreaterEqual(results[0]["segments"][0]["pose_wait_steps"], 1)
        freshness = results[0]["final_pose_freshness"]
        self.assertTrue(freshness["source_sequence_known"])
        self.assertEqual(freshness["assurance"], "source_observation_sequence")
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_fifo_worker_confirms_motion_without_draining_newer_frames(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        snapshot["source_frame"] = {
            "backend_frame_id": 1,
            "lag_known": True,
            "ordered_pose_producer": {
                "schema": (
                    "behavior.official.navigation_map."
                    "ordered_pose_producer.v1"
                ),
                "enqueued_frame_count": 1,
                "processed_frame_count": 1,
                "pose_frame_id": 1,
                "last_enqueued_observation_sequence": 1,
            },
        }
        state = {
            "snapshot": snapshot,
            "pending": False,
            "pending_reads": 0,
            "distance_m": 0.0,
        }

        def current_snapshot():
            if state["pending"]:
                state["pending_reads"] += 1
                self._advance_snapshot_observation(snapshot)
                progress = snapshot["source_frame"][
                    "ordered_pose_producer"
                ]
                progress["enqueued_frame_count"] += 1
                progress["last_enqueued_observation_sequence"] = int(
                    snapshot["observation_sequence"]
                )
                if state["pending_reads"] == 6:
                    yaw_rad = math.radians(float(snapshot["pose"]["yaw_deg"]))
                    snapshot["pose"]["x"] += state["distance_m"] * math.cos(
                        yaw_rad
                    )
                    snapshot["pose"]["y"] += state["distance_m"] * math.sin(
                        yaw_rad
                    )
                    processed = int(progress["enqueued_frame_count"]) - 1
                    progress["processed_frame_count"] = processed
                    progress["pose_frame_id"] = processed
                    snapshot["source_frame"]["backend_frame_id"] = processed
                    pose_version = (
                        f"{snapshot['map_epoch']}|fixture-pose:{processed}"
                    )
                    snapshot["frame_version"] = pose_version
                    snapshot["pose_version"] = pose_version
                    snapshot["pose_observed_sequence"] = int(
                        snapshot["observation_sequence"]
                    )
                    snapshot["pose_observed_ts"] = float(
                        snapshot["captured_ts"]
                    )
                    snapshot["pose_age_observations"] = 0
                    snapshot["pose_age_s"] = 0.0
                    snapshot["pose_source_sequence_known"] = False
                    snapshot.pop("pose_source_observation_sequence", None)
                    snapshot["source_frame"]["lag_known"] = False
                    state["pending"] = False
            return deepcopy(snapshot)

        adapter.navigation_map_snapshot = current_snapshot
        ctx, results, setter = self._ctx(world)

        def slow_fifo_controller(inner_ctx, forward=0.0, **_kwargs):
            def phase():
                state["distance_m"] = float(forward)
                state["pending"] = True
                state["pending_reads"] = 0
                yield inner_ctx.world.make_action(base=[0.2, 0.0, 0.0])
                return {
                    "ok": True,
                    "action_steps": 1,
                    "linear_remaining_m": 0.0,
                }

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=slow_fifo_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_slow_fifo_motion",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(
            results[0]["final_causal_localization"]["assurance"],
            "ordered_producer_frame",
        )
        final_proof = results[0]["final_causal_localization"]
        final_progress = results[0]["final_pose_freshness"][
            "ordered_pose_producer"
        ]
        self.assertEqual(
            final_progress["processed_frame_count"],
            final_proof["ordered_target_frame"],
        )
        self.assertGreater(
            final_progress["enqueued_frame_count"],
            final_progress["processed_frame_count"],
        )
        self.assertGreaterEqual(results[0]["segments"][0]["pose_wait_steps"], 2)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_stale_initial_pose_is_rejected_before_planning(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["observation_sequence"] = 2
        snapshot["captured_ts"] = 102.0
        snapshot["pose_age_observations"] = 1
        snapshot["pose_age_s"] = 2.0
        self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        planner = mock.Mock()

        with mock.patch.object(
            official_v2_tools, "plan_clearance_path", side_effect=planner
        ):
            actions = list(
                build_registry(adapter)["navigate_to"].fn(
                    ctx,
                    name=GOAL_NAME,
                    session_id="navigate_stale_initial_pose",
                    timeout_s=5.0,
                )
            )

        self.assertEqual(setter.call_count, 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "observation validation")
        self.assertIn("pose producer is stale", results[0]["error"])
        planner.assert_not_called()
        self.assertEqual(len(actions), 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_deadline_discards_child_action_generated_after_timeout(self) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        clock = {"now": 0.0}

        def late_controller(inner_ctx, **_kwargs):
            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                NavigateToExecutionTest._advance_snapshot_observation(
                    state["snapshot"]
                )
                clock["now"] = 10.0
                yield inner_ctx.world.make_action(base=[0.8, 0.0, 0.0])
                return {"ok": True, "action_steps": 2}

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools.time,
                    "monotonic",
                    side_effect=lambda: clock["now"],
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=late_controller,
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id="navigate_mid_action_timeout",
                                timeout_s=5.0,
                            )
                        )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "timeout")
        self.assertEqual(results[0]["action_steps"], 1)
        self.assertEqual(len(actions), 2)
        self._assert_action_contract(actions)
        self.assertAlmostEqual(float(actions[0][ACTION_SLICES["base"]][0]), 0.1)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_cancellation_discards_prefetched_child_action(self) -> None:
        class SkillCancelled(RuntimeError):
            pass

        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        cancellation = {"raised": False}

        def raise_if_cancelled(_where=""):
            if cancellation["raised"]:
                raise SkillCancelled("cancelled between child actions")

        ctx.raise_if_cancelled = raise_if_cancelled

        def cancelled_controller(inner_ctx, **_kwargs):
            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                NavigateToExecutionTest._advance_snapshot_observation(
                    state["snapshot"]
                )
                cancellation["raised"] = True
                yield inner_ctx.world.make_action(base=[0.8, 0.0, 0.0])
                return {"ok": True, "action_steps": 2}

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=cancelled_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_mid_action_cancel",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "cancellation")
        self.assertEqual(results[0]["action_steps"], 1)
        self.assertEqual(len(actions), 2)
        self._assert_action_contract(actions)
        self.assertAlmostEqual(float(actions[0][ACTION_SLICES["base"]][0]), 0.1)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_large_heading_error_aligns_once_before_polyline_translation(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(20, 30),
            goal_cell=(38, 30),
        )
        snapshot["pose"]["yaw_deg"] = 27.0
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=self._controller_stub(state, calls),
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_heading_preserved",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertTrue(calls)
        spin_calls = [call for call in calls if call["kind"] == "spin"]
        forward_calls = [call for call in calls if call["kind"] == "forward"]
        self.assertEqual(len(spin_calls), 1, calls)
        self.assertTrue(forward_calls)
        self.assertAlmostEqual(spin_calls[0]["spin"], 63.0, places=6)
        self.assertEqual(
            spin_calls[0]["kwargs"].get("_spin_gain"),
            official_v2_tools.NAVIGATE_TO_SPIN_GAIN,
        )
        self.assertLess(calls.index(spin_calls[0]), calls.index(forward_calls[0]))
        self.assertAlmostEqual(state["snapshot"]["pose"]["yaw_deg"], 90.0)
        self.assertFalse(results[0]["segments"][0]["heading_preserved"])
        self.assertTrue(
            all(segment["heading_preserved"] for segment in results[0]["segments"][1:])
        )
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_post_spin_pose_correction_replans_instead_of_failing(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(20, 30),
            goal_cell=(38, 30),
        )
        snapshot["pose"]["yaw_deg"] = 27.0
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []
        corrected = {"done": False}

        def correct_pose_after_first_spin(kind, current_state, _world):
            if kind != "spin" or corrected["done"]:
                return
            corrected["done"] = True
            # A map-frame translation correction changes the bearing of the
            # already-signed chord after its single alignment spin.
            current_state["snapshot"]["pose"]["x"] += 2.0

        always_inside_corridor = {
            "ok": True,
            "effective_cross_track_limit_m": 1.0,
        }
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=self._controller_stub(
                        state,
                        calls,
                        after_phase=correct_pose_after_first_spin,
                    ),
                ), mock.patch.object(
                    official_v2_tools,
                    "_navigate_to_segment_corridor",
                    return_value=always_inside_corridor,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_post_spin_pose_correction",
                            timeout_s=30.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertGreaterEqual(results[0]["replan_count"], 1)
        self.assertGreaterEqual(
            len([call for call in calls if call["kind"] == "spin"]),
            2,
        )
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_map_chord_is_converted_to_holonomic_body_frame(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(20, 30),
            goal_cell=(30, 30),
        )
        # The segment points at 90 degrees, so a 50-degree heading leaves a
        # 40-degree error: below the strict alignment-spin threshold.
        snapshot["pose"]["yaw_deg"] = 50.0
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []

        def residual_heading_controller(
            inner_ctx,
            forward=0.0,
            translation=0.0,
            spin=0.0,
            **_kwargs,
        ):
            kind = "spin" if abs(float(spin)) > 1e-9 else "forward"
            calls.append(
                {
                    "kind": kind,
                    "forward": float(forward),
                    "translation": float(translation),
                    "spin": float(spin),
                }
            )

            def phase():
                self.assertEqual(kind, "forward")
                yield inner_ctx.world.make_action(base=[0.2, 0.1, 0.0])
                yaw = math.radians(state["snapshot"]["pose"]["yaw_deg"])
                state["snapshot"]["pose"]["x"] += (
                    math.cos(yaw) * float(forward)
                    - math.sin(yaw) * float(translation)
                )
                state["snapshot"]["pose"]["y"] += (
                    math.sin(yaw) * float(forward)
                    + math.cos(yaw) * float(translation)
                )
                NavigateToExecutionTest._advance_snapshot_pose(state["snapshot"])
                return {"ok": True, "action_steps": 1}

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=residual_heading_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_body_frame_vector",
                            timeout_s=10.0,
                            arrival_tolerance_m=0.08,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["ok"], results[0])
        forward_calls = [call for call in calls if call["kind"] == "forward"]
        self.assertGreaterEqual(len(forward_calls), 1, calls)
        self.assertTrue(
            all(abs(call["translation"]) > 0.005 for call in forward_calls),
            calls,
        )
        self.assertAlmostEqual(
            sum(
                math.hypot(call["forward"], call["translation"])
                for call in forward_calls
            ),
            1.0,
            places=6,
        )
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_low_clearance_segment_uses_tight_passage_motion_policy(self) -> None:
        trajectory = {
            "path_segment_clearance_m": [0.44, 0.54],
            "path_segment_required_clearance_m": [0.42, 0.42],
        }

        threshold, speed_cap, tight, reserve = (
            official_v2_tools._navigate_to_segment_motion_policy(
                trajectory, 0, depth_refined=False
            )
        )
        self.assertTrue(tight)
        self.assertAlmostEqual(reserve, 0.02)
        self.assertEqual(
            threshold,
            official_v2_tools.NAVIGATE_TO_TIGHT_PASSAGE_ALIGN_THRESHOLD_DEG,
        )
        self.assertEqual(
            speed_cap,
            official_v2_tools.NAVIGATE_TO_TIGHT_PASSAGE_MAX_SPEED_MPS,
        )

        threshold, speed_cap, tight, reserve = (
            official_v2_tools._navigate_to_segment_motion_policy(
                trajectory, 1, depth_refined=False
            )
        )
        self.assertFalse(tight)
        self.assertAlmostEqual(reserve, 0.12)
        self.assertEqual(
            threshold, official_v2_tools.NAVIGATE_TO_ALIGN_SPIN_THRESHOLD_DEG
        )
        self.assertIsNone(speed_cap)

        _threshold, speed_cap, tight, _reserve = (
            official_v2_tools._navigate_to_segment_motion_policy(
                trajectory, 0, depth_refined=True
            )
        )
        self.assertFalse(tight)
        self.assertIsNone(speed_cap)

        traversed_trajectory = {
            "path_segment_clearance_m": [0.72],
            "path_segment_required_clearance_m": [0.42],
            "planner": {"traversed_route_direct_replay": True},
        }
        threshold, speed_cap, tight, reserve = (
            official_v2_tools._navigate_to_segment_motion_policy(
                traversed_trajectory, 0, depth_refined=False
            )
        )
        self.assertFalse(tight)
        self.assertAlmostEqual(reserve, 0.30)
        self.assertEqual(
            threshold,
            official_v2_tools.NAVIGATE_TO_TRAVERSED_ALIGN_THRESHOLD_DEG,
        )
        self.assertEqual(
            speed_cap,
            official_v2_tools.NAVIGATE_TO_TRAVERSED_MAX_SPEED_MPS,
        )

        moderate_traversed_trajectory = {
            "path_segment_clearance_m": [0.60],
            "path_segment_required_clearance_m": [0.42],
            "planner": {"traversed_route_direct_replay": True},
        }
        threshold, speed_cap, tight, reserve = (
            official_v2_tools._navigate_to_segment_motion_policy(
                moderate_traversed_trajectory, 0, depth_refined=False
            )
        )
        self.assertFalse(tight)
        self.assertAlmostEqual(reserve, 0.18)
        self.assertEqual(
            threshold,
            official_v2_tools.NAVIGATE_TO_TRAVERSED_ALIGN_THRESHOLD_DEG,
        )
        self.assertEqual(
            speed_cap,
            official_v2_tools.NAVIGATE_TO_TRAVERSED_MODERATE_MAX_SPEED_MPS,
        )

        threshold, speed_cap, tight, _reserve = (
            official_v2_tools._navigate_to_segment_motion_policy(
                traversed_trajectory, 0, depth_refined=True
            )
        )
        self.assertFalse(tight)
        self.assertEqual(
            threshold, official_v2_tools.NAVIGATE_TO_ALIGN_SPIN_THRESHOLD_DEG
        )
        self.assertIsNone(speed_cap)

    def test_direct_history_depth_stop_gets_bounded_same_route_retry(self) -> None:
        self.assertGreater(
            official_v2_tools.NAVIGATE_TO_TRAVERSED_DEPTH_RETRY_LIMIT,
            official_v2_tools.NAVIGATE_TO_DEPTH_SAMPLE_TICKS,
        )
        trajectory = {
            "planner": {"traversed_route_direct_replay": True},
        }
        safe_stop = {
            "depth_obstacle_stop": True,
            "cross_track_stop": False,
            "command_pose_changed": False,
            "localization_hold_recovered": False,
            "corridor": {"ok": True},
        }

        self.assertTrue(
            official_v2_tools._navigate_to_should_retry_traversed_depth_stop(
                trajectory, safe_stop, 0
            )
        )
        self.assertFalse(
            official_v2_tools._navigate_to_should_retry_traversed_depth_stop(
                trajectory,
                safe_stop,
                official_v2_tools.NAVIGATE_TO_TRAVERSED_DEPTH_RETRY_LIMIT,
            )
        )
        self.assertFalse(
            official_v2_tools._navigate_to_should_retry_traversed_depth_stop(
                {**trajectory, "depth_refinement": {"schema": "test"}},
                safe_stop,
                0,
            )
        )
        self.assertFalse(
            official_v2_tools._navigate_to_should_retry_traversed_depth_stop(
                trajectory,
                {**safe_stop, "corridor": {"ok": False}},
                0,
            )
        )

    def test_child_success_requires_map_frame_waypoint_confirmation(self) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)

        def false_success(inner_ctx, **_kwargs):
            def phase():
                yield inner_ctx.world.make_action(base=[0.2, 0.0, 0.0])
                NavigateToExecutionTest._advance_snapshot_pose(state["snapshot"])
                # Fresh timestamp alone is insufficient: pose never moved.
                return {
                    "ok": True,
                    "action_steps": 1,
                    "linear_remaining_m": 0.0,
                }

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "NAVIGATE_TO_MAX_REPLANS",
                    0,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=false_success,
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id="navigate_bad_waypoint",
                                timeout_s=10.0,
                            )
                        )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["failure_stage"], "tracking")
        self.assertIn("waypoint verification", result["error"])
        self.assertEqual(result["segment_count"], 1)
        self.assertFalse(result["segments"][0]["map_frame_waypoint_verified"])
        self.assertEqual(len(actions), 3)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_non_mapping_child_report_is_a_structured_failure(self) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)

        def malformed_controller(inner_ctx, **_kwargs):
            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                NavigateToExecutionTest._advance_snapshot_observation(
                    state["snapshot"]
                )
                return ["truthy", "but", "not", "a", "mapping"]

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=malformed_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_bad_report",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "tracking")
        self.assertIn("structured phase report", results[0]["error"])
        self.assertEqual(len(actions), 2)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_mark_change_without_map_version_never_reports_arrival(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 20))
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []

        def move_mark(kind, current_state, _world):
            if kind == "forward":
                current_state["snapshot"]["places"][0]["x"] += 0.1

        controller = self._controller_stub(state, calls, after_phase=move_mark)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_changed_mark",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual([call["kind"] for call in calls], ["forward"])
        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "start-state validation")
        self.assertIn("marked goal changed", results[0]["error"])
        self.assertEqual(len(actions), 3)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_uncertain_localization_hold_blocks_planning(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["lifecycle"]["uncertain_hold"] = True
        self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        controller = mock.Mock()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_uncertain_hold",
                            timeout_s=5.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "timeout")
        controller.assert_not_called()
        self.assertGreater(len(actions), 1)
        self._assert_action_contract(actions)
        for action in actions:
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)

    def test_rtab_map_write_hold_is_advisory_for_monitored_navigation(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 15))
        snapshot["backend"] = "behavior_interface.rtabmap_slam.live.LiveMapper"
        snapshot["pose"]["global_confident"] = False
        snapshot["lifecycle"]["pose_confident"] = False
        snapshot["lifecycle"]["uncertain_hold"] = True
        snapshot["lifecycle"]["tracking_ok"] = True
        snapshot["pose_source_sequence_known"] = False
        snapshot.pop("pose_source_observation_sequence", None)
        snapshot["source_frame"] = {
            "backend_frame_id": 1,
            "lag_known": False,
            "ordered_pose_producer": {
                "schema": (
                    "behavior.official.navigation_map."
                    "ordered_pose_producer.v1"
                ),
                "enqueued_frame_count": 1,
                "processed_frame_count": 1,
                "pose_frame_id": 1,
                "last_enqueued_observation_sequence": 1,
            },
        }
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        calls: list[dict] = []

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=self._controller_stub(state, calls),
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_rtab_map_write_hold",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertTrue(results[0]["pose_fusion_started"])
        self.assertEqual(results[0]["localization_preflight_steps"], 0)
        self.assertTrue(calls)
        self._assert_action_contract(actions)

    def test_shared_timeout_fails_before_starting_an_over_budget_segment(self) -> None:
        adapter, world = self._adapter_world()
        self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                actions = list(
                    build_registry(adapter)["navigate_to"].fn(
                        ctx,
                        name=GOAL_NAME,
                        session_id="navigate_timeout",
                        timeout_s=0.1,
                    )
                )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "timeout")
        self.assertEqual(results[0]["action_steps"], 0)
        self.assertEqual(len(actions), 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_shared_deadline_interrupts_planning(self) -> None:
        adapter, world = self._adapter_world()
        self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        clock = {"now": 0.0}

        def slow_planner(_snapshot, _name, *, progress_check, **_kwargs):
            clock["now"] = 6.0
            progress_check()
            self.fail("expired planner callback returned")

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools.time,
                    "monotonic",
                    side_effect=lambda: clock["now"],
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "plan_clearance_path",
                        side_effect=slow_planner,
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id="navigate_planning_timeout",
                                timeout_s=5.0,
                            )
                        )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "timeout")
        self.assertEqual(results[0]["action_steps"], 0)
        self.assertEqual(len(actions), 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_cancellation_after_planning_commits_one_safe_failure(self) -> None:
        class SkillCancelled(RuntimeError):
            pass

        adapter, world = self._adapter_world()
        self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        ctx.raise_if_cancelled = mock.Mock(
            side_effect=SkillCancelled("cancelled by evaluator")
        )

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                actions = list(
                    build_registry(adapter)["navigate_to"].fn(
                        ctx,
                        name=GOAL_NAME,
                        session_id="navigate_cancelled",
                        timeout_s=5.0,
                    )
                )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "cancellation")
        self.assertEqual(results[0]["action_steps"], 0)
        self.assertNotIn("base_qvel", vars(world))
        self.assertEqual(len(actions), 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_cancellation_during_terminal_hold_cannot_commit_success(self) -> None:
        class SkillCancelled(RuntimeError):
            pass

        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(30, 10),
            goal_cell=(30, 10),
        )
        self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        cancelled = False
        result_assembly_started = False
        real_validate = (
            official_v2_tools._navigate_to_validate_inactive_limb_contract
        )
        real_result_base = official_v2_tools._navigate_to_result_base

        def raise_if_cancelled(_where=""):
            if cancelled:
                raise SkillCancelled("cancelled during terminal hold validation")

        def cancel_after_terminal_hold(*args, **kwargs):
            nonlocal cancelled
            hold = real_validate(*args, **kwargs)
            if result_assembly_started:
                cancelled = True
            return hold

        def mark_result_assembly():
            nonlocal result_assembly_started
            result_assembly_started = True
            return real_result_base()

        ctx.raise_if_cancelled = raise_if_cancelled
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_navigate_to_validate_inactive_limb_contract",
                    side_effect=cancel_after_terminal_hold,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_navigate_to_result_base",
                        side_effect=mark_result_assembly,
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id="navigate_terminal_hold_cancel",
                                timeout_s=5.0,
                            )
                        )

        self.assertEqual(setter.call_count, 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "cancellation")
        self.assertTrue(result_assembly_started)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_navigation_uses_raw_lateral_velocity_and_restores_world_binding(self) -> None:
        for instance_binding in (False, True):
            with self.subTest(instance_binding=instance_binding):
                adapter, world = self._adapter_world()
                if instance_binding:
                    world.base_qvel = world.base_qvel
                previous = vars(world).get("base_qvel")
                ctx, _results, _setter = self._ctx(world)
                world.set_base_velocity(0.5, 0.0, 0.0)
                proprio = adapter.proprio_vector()
                proprio[PROPRIO_SLICES["base_qvel"]] = [0.0, 0.1, 0.0]
                adapter.update({"robot_r1::proprio": proprio})
                np.testing.assert_allclose(world.base_qvel(), [0.1, 0.0, 0.0], atol=1e-8)
                cleanup = {}
                official_v2_tools._navigate_to_begin_pose_fusion(ctx, self._snapshot(world), cleanup)
                try:
                    np.testing.assert_allclose(world.base_qvel(), [0.0, 0.1, 0.0], atol=1e-8)
                    world.reconcile_base_odometry_from_proprio(1.0 / 30.0)
                    np.testing.assert_allclose(world.policy_local_base_pose(),
                                               [0.0, 0.1 / 30.0, 0.0], atol=1e-8)
                    self.assertEqual(official_v2_tools._navigate_to_pose_fusion_diagnostics(cleanup)[
                        "velocity_source"], "raw_evaluator_body_qvel")
                finally:
                    official_v2_tools._navigate_to_clear_pose_fusion(ctx, cleanup)
                self.assertEqual("base_qvel" in vars(world), instance_binding)
                self.assertIs(vars(world).get("base_qvel"), previous)

    def test_navigation_low_but_real_progress_is_not_a_stall(self) -> None:
        for absolute_only, observed_speed, expected in (
            (False, 0.10, False), (True, 0.10, True), (True, 0.0, False),
        ):
            with self.subTest(absolute_only=absolute_only, speed=observed_speed):
                adapter, world = self._adapter_world()
                ctx, _results, _setter = self._ctx(world)
                controller = official_v2_tools._yield_adjust_chassis_controller(
                    ctx, forward=0.2, timeout_s=8.0, vmax=0.75,
                    _linear_tolerance_m=0.05, _require_linear_target=True,
                    _commit_result=False, _emit_terminal_hold=False,
                    _allow_reverse_recovery=False, _skip_linear_settle=True,
                    _linear_stall_absolute_only=absolute_only,
                    _linear_deceleration_mps2=official_v2_tools.NAVIGATE_TO_LINEAR_DECEL_MPS2,
                    _linear_near_speed_mps=official_v2_tools.NAVIGATE_TO_LINEAR_NEAR_SPEED_MPS,
                )
                actions = []
                while True:
                    try:
                        action = next(controller)
                    except StopIteration as stopped:
                        result = stopped.value
                        break
                    actions.append(action)
                    proprio = adapter.proprio_vector()
                    base = np.asarray(action)[ACTION_SLICES["base"]]
                    proprio[PROPRIO_SLICES["base_qvel"]] = [
                        observed_speed if base[0] > 0.0 else 0.0, 0.0, 0.0,
                    ]
                    adapter.update({"robot_r1::proprio": proprio})
                self.assertEqual(result["ok"], expected, result)
                self.assertEqual(result["linear_tracking"]["stall_policy"],
                                 "absolute_progress" if absolute_only else "command_ratio")
                if expected:
                    self.assertGreaterEqual(result["actual"]["forward_m"], 0.15)
                else:
                    self.assertEqual(result["obstacle_stop_reason"], "base_qvel_stall")
                self._assert_action_contract(actions)

    def test_body_velocity_feedback_keeps_line_despite_uncommanded_yaw(self) -> None:
        adapter, world = self._adapter_world()
        world.base_qvel = world.raw_base_qvel
        ctx, _results, _setter = self._ctx(world)
        controller = official_v2_tools._yield_adjust_chassis_controller(
            ctx, forward=1.0, timeout_s=10.0, vmax=0.75,
            _linear_tolerance_m=0.05, _require_linear_target=True,
            _commit_result=False, _emit_terminal_hold=False,
            _allow_reverse_recovery=False, _skip_linear_settle=True,
            _linear_stall_absolute_only=True, _linear_body_frame_feedback=True,
        )
        position = np.zeros(2)
        yaw = 0.0
        actions = []
        dt = 1.0 / official_v2_tools.CONTROL_HZ
        while True:
            try:
                action = next(controller)
            except StopIteration as stopped:
                result = stopped.value
                break
            actions.append(action)
            body = np.asarray(action)[ACTION_SLICES["base"]][:2] * 0.75
            yaw_rate = 0.35 if np.linalg.norm(body) > 0.0 else 0.0
            mid_yaw = yaw + yaw_rate * dt / 2.0
            c, s = math.cos(mid_yaw), math.sin(mid_yaw)
            position += np.array([c * body[0] - s * body[1], s * body[0] + c * body[1]]) * dt
            yaw += yaw_rate * dt
            proprio = adapter.proprio_vector()
            proprio[PROPRIO_SLICES["base_qvel"]] = [*body, yaw_rate]
            adapter.update({"robot_r1::proprio": proprio})
        self.assertTrue(result["ok"], result)
        self.assertLess(np.linalg.norm(position - [1.0, 0.0]), 0.05)
        self.assertLess(abs(position[1]), 0.01)
        self.assertGreater(abs(result["linear_yaw_drift_deg"]), 20.0)
        self._assert_action_contract(actions)

    def test_coupled_partial_turn_tracks_xy_and_yaw_together(self) -> None:
        adapter, world = self._adapter_world()
        world.base_qvel = world.raw_base_qvel
        ctx, _results, _setter = self._ctx(world)
        target = np.array([.25, .10])
        target_yaw = math.radians(-7.5)
        controller = official_v2_tools._yield_adjust_chassis_controller(
            ctx,
            forward=float(target[0]),
            translation=float(target[1]),
            spin=math.degrees(target_yaw),
            timeout_s=10.0,
            vmax=.15,
            _linear_tolerance_m=.015,
            _require_linear_target=True,
            _commit_result=False,
            _emit_terminal_hold=False,
            _allow_reverse_recovery=False,
            _skip_linear_settle=True,
            _linear_stall_absolute_only=True,
            _linear_body_frame_feedback=True,
            _coupled_linear_spin=True,
            _coupled_heading_tolerance_deg=2.0,
        )
        position = np.zeros(2)
        yaw = 0.0
        actions = []
        simultaneous = False
        dt = 1.0 / official_v2_tools.CONTROL_HZ
        while True:
            try:
                action = next(controller)
            except StopIteration as stopped:
                result = stopped.value
                break
            actions.append(action)
            normalized = np.asarray(action)[ACTION_SLICES["base"]]
            body = normalized[:2] * official_v2_tools.ADJUST_CHASSIS_BASE_MAX_LIN_MPS
            yaw_rate = (
                normalized[2]
                * official_v2_tools.ADJUST_CHASSIS_BASE_MAX_ANG_RADPS
            )
            simultaneous |= bool(
                np.linalg.norm(body) > 1.0e-9 and abs(yaw_rate) > 1.0e-9
            )
            mid_yaw = yaw + yaw_rate * dt / 2.0
            c, s = math.cos(mid_yaw), math.sin(mid_yaw)
            position += np.array([
                c * body[0] - s * body[1],
                s * body[0] + c * body[1],
            ]) * dt
            yaw += yaw_rate * dt
            proprio = adapter.proprio_vector()
            proprio[PROPRIO_SLICES["base_qvel"]] = [*body, yaw_rate]
            adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["coupled_linear_spin"])
        self.assertTrue(simultaneous)
        np.testing.assert_allclose(position, target, atol=.02)
        self.assertLess(abs(yaw - target_yaw), math.radians(.8))
        self.assertLessEqual(result["coupled_schedule_error_max_deg"], 2.5)
        self._assert_action_contract(actions)

    def test_segment_tracking_stall_is_reported_and_held(self) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)

        def stalled_controller(inner_ctx, **_kwargs):
            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                NavigateToExecutionTest._advance_snapshot_pose(
                    state["snapshot"]
                )
                return {
                    "ok": False,
                    "error": "base motion stalled",
                    "action_steps": 1,
                    "linear_remaining_m": 0.5,
                    "obstacle_stop_reason": "base_qvel_stall",
                    "linear_tracking": {"observed_along_command_mps": 0.0},
                }

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=stalled_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_stalled",
                            timeout_s=20.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"], results[0])
        self.assertEqual(results[0]["failure_stage"], "tracking")
        self.assertIn("stall", results[0]["error"])
        self.assertEqual(results[0]["failure_details"]["controller_failure"]["linear_tracking"],
                         {"observed_along_command_mps": 0.0})
        self.assertGreaterEqual(results[0]["action_steps"], 2)
        self.assertEqual(len(actions), results[0]["action_steps"] + 1)
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_zero_progress_stall_retreat_replans_without_false_obstacle(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        calls = []
        normal_controller = self._controller_stub(state, calls)
        stalled_once = False

        def recovering_controller(
            inner_ctx, forward=0.0, translation=0.0, spin=0.0, **kwargs
        ):
            nonlocal stalled_once
            if stalled_once or abs(float(spin)) > 1.0e-9:
                return normal_controller(
                    inner_ctx,
                    forward=forward,
                    translation=translation,
                    spin=spin,
                    **kwargs,
                )
            stalled_once = True
            calls.append(
                {
                    "kind": "recovered_stall",
                    "forward": float(forward),
                    "translation": float(translation),
                    "kwargs": dict(kwargs),
                }
            )

            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                self._advance_snapshot_pose(state["snapshot"])
                return {
                    "ok": False,
                    "error": "base motion stalled after a verified retreat",
                    "action_steps": 1,
                    "requested": {
                        "forward_m": float(forward),
                        "translation_m": float(translation),
                        "spin_deg": float(spin),
                    },
                    "actual": {
                        "forward_m": 0.0,
                        "translation_m": 0.0,
                        "spin_deg": 0.0,
                    },
                    "linear_remaining_m": max(0.0, float(forward)),
                    "obstacle_stop_reason": "base_qvel_stall",
                }

            return phase()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=recovering_controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_recovered_stall",
                            timeout_s=20.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["stall_recovery_count"], 1)
        self.assertEqual(
            results[0]["stall_recoveries"][0]["stalled_progress_m"], 0.0
        )
        self.assertEqual(
            results[0]["stall_recoveries"][0]["recovery_mode"],
            "certified_start_clearance_ball_backtrack",
        )
        self.assertEqual(
            results[0]["execution_stall_obstacle_evidence_count"], 0
        )
        self.assertEqual(results[0]["execution_stall_obstacles"], [])
        self.assertIsNone(
            results[0]["stall_recoveries"][0]["collision_evidence"]
        )
        self.assertEqual(
            results[0]["stall_recoveries"][0][
                "collision_evidence_suppressed_reason"
            ],
            "subcell_progress_cannot_localize_ahead_obstacle",
        )
        self.assertEqual(
            results[0]["plan_records"][-1][
                "execution_stall_obstacle_evidence_count"
            ],
            0,
        )
        self.assertEqual(
            results[0]["planner"][
                "execution_stall_obstacle_evidence_count"
            ],
            0,
        )
        self.assertNotIn(
            "navigation_stall_obstacle_mask", state["snapshot"]
        )
        self.assertGreaterEqual(
            results[0]["stall_recoveries"][0]["retreat_m"],
            official_v2_tools.NAVIGATE_TO_STALL_RECOVERY_MIN_RETREAT_M,
        )
        self.assertLessEqual(
            results[0]["stall_recoveries"][0]["retreat_m"],
            official_v2_tools.NAVIGATE_TO_STALL_RECOVERY_MAX_RETREAT_M,
        )
        self.assertGreaterEqual(results[0]["replan_count"], 1)
        self.assertTrue(
            any(
                report.get("stall_forced_alignment") is True
                for report in results[0]["segments"]
            ),
            results[0]["segments"],
        )
        self.assertTrue(
            all(
                call["kwargs"].get("_allow_reverse_recovery") is False
                for call in calls
            ),
            calls,
        )
        self._assert_action_contract(actions)

    def test_stall_backtrack_then_corridor_centering_replans_to_goal(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(adapter, self._snapshot(world))
        ctx, results, setter = self._ctx(world)
        calls = []
        normal_controller = self._controller_stub(state, calls)
        stalled_once = False

        def recovering_controller(
            inner_ctx, forward=0.0, translation=0.0, spin=0.0, **kwargs
        ):
            nonlocal stalled_once
            if stalled_once or abs(float(spin)) > 1.0e-9:
                return normal_controller(
                    inner_ctx,
                    forward=forward,
                    translation=translation,
                    spin=spin,
                    **kwargs,
                )
            stalled_once = True
            calls.append(
                {
                    "kind": "stalled",
                    "forward": float(forward),
                    "translation": float(translation),
                    "kwargs": dict(kwargs),
                }
            )

            def phase():
                yield inner_ctx.world.make_action(base=[0.1, 0.0, 0.0])
                self._advance_snapshot_pose(state["snapshot"])
                return {
                    "ok": False,
                    "error": "base motion stalled",
                    "action_steps": 1,
                    "requested": {
                        "forward_m": float(forward),
                        "translation_m": float(translation),
                        "spin_deg": float(spin),
                    },
                    "actual": {
                        "forward_m": 0.0,
                        "translation_m": 0.0,
                        "spin_deg": 0.0,
                    },
                    "linear_remaining_m": max(0.0, float(forward)),
                    "obstacle_stop_reason": "base_qvel_stall",
                }

            return phase()

        centering_evidence = {
            "schema": "official_v2_depth_corridor_centering_v1",
            "side": "left",
            "direction_map_xy": [0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=recovering_controller,
                ), mock.patch.object(
                    official_v2_tools,
                    "_navigate_to_depth_corridor_centering",
                    return_value=centering_evidence,
                ), mock.patch.object(
                    official_v2_tools,
                    "_navigate_to_uses_traversed_motion_frame",
                    return_value=True,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_stall_centering",
                            timeout_s=20.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["stall_recovery_count"], 1)
        centering = results[0]["stall_recoveries"][0][
            "corridor_centering"
        ]
        self.assertTrue(centering["attempted"], centering)
        self.assertTrue(centering["accepted"], centering)
        self.assertEqual(centering["evidence"]["side"], "left")
        centering_calls = [
            call
            for call in calls
            if call.get("kind") == "forward"
            and abs(float(call.get("forward", 0.0))) < 1.0e-8
            and float(call.get("translation", 0.0)) > 0.03
        ]
        self.assertEqual(len(centering_calls), 1, calls)
        self.assertTrue(
            centering_calls[0]["kwargs"].get(
                "_linear_body_frame_feedback"
            )
        )
        self._assert_action_contract(actions)

    def test_stall_obstacle_requires_a_current_depth_collision_witness(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.0,
            "y": 2.0,
            "yaw_deg": 0.0,
            "global_confident": True,
        }
        trajectory = {
            "plan_id": "stall-source",
            "path_xy_m": [[2.0, 2.0], [3.0, 2.0]],
            "start_state": {"base_footprint": {"radius_m": 0.42}},
        }
        snapshot["navigation_depth_points_xy_m"] = np.asarray(
            [[2.55, 2.05], [2.90, 2.0], [1.80, 2.0]], dtype=np.float64
        )
        witness = official_v2_tools._navigate_to_stall_depth_witness(
            trajectory, snapshot, 1
        )
        np.testing.assert_allclose(witness, [2.55, 2.05])
        record = official_v2_tools._navigate_to_stall_obstacle_record(
            trajectory,
            snapshot,
            1,
            1,
            observed_point_xy_m=witness,
        )
        self.assertEqual(
            record["inference"], "current_depth_confirmed_base_qvel_stall"
        )
        np.testing.assert_allclose(record["map_xy_m"], witness)

        snapshot["navigation_depth_points_xy_m"] = np.asarray(
            [[2.90, 2.0], [1.80, 2.0]], dtype=np.float64
        )
        self.assertIsNone(
            official_v2_tools._navigate_to_stall_depth_witness(
                trajectory, snapshot, 1
            )
        )

    def test_physical_stall_extracts_transverse_vertical_barrier_plane(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.0,
            "y": 2.0,
            "yaw_deg": 0.0,
            "global_confident": True,
        }
        trajectory = {
            "plan_id": "stall-plane",
            "path_xy_m": [[2.0, 2.0], [3.0, 2.0]],
            "start_state": {"base_footprint": {"radius_m": 0.42}},
        }
        door = np.asarray(
            [
                [0.28, lateral, height]
                for lateral in np.linspace(-0.42, 0.42, 15)
                for height in np.linspace(0.82, 1.72, 9)
            ],
            dtype=np.float64,
        )
        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_latest_depth_obstacles",
            return_value=door,
        ):
            evidence = (
                official_v2_tools._navigate_to_stall_vertical_barrier_evidence(
                    SimpleNamespace(world=world), trajectory, snapshot, 1
                )
            )

        self.assertIsNotNone(evidence)
        self.assertGreaterEqual(evidence["tangent_span_m"], 0.75)
        self.assertGreater(evidence["travel_normal_cos"], 0.99)
        self.assertAlmostEqual(evidence["crossing_distance_m"], 0.28, places=2)
        map_points = np.asarray(evidence["map_points_xy_m"], dtype=np.float64)
        self.assertGreaterEqual(len(map_points), 10)
        np.testing.assert_allclose(map_points[:, 0], 2.28, atol=0.01)

    def test_physical_stall_does_not_turn_parallel_side_wall_into_barrier(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.0,
            "y": 2.0,
            "yaw_deg": 0.0,
            "global_confident": True,
        }
        trajectory = {
            "path_xy_m": [[2.0, 2.0], [3.0, 2.0]],
            "start_state": {"base_footprint": {"radius_m": 0.42}},
        }
        side_wall = np.asarray(
            [
                [forward, 0.28, height]
                for forward in np.linspace(0.05, 0.55, 15)
                for height in np.linspace(0.82, 1.72, 9)
            ],
            dtype=np.float64,
        )
        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_latest_depth_obstacles",
            return_value=side_wall,
        ):
            evidence = (
                official_v2_tools._navigate_to_stall_vertical_barrier_evidence(
                    SimpleNamespace(world=world), trajectory, snapshot, 1
                )
            )

        self.assertIsNone(evidence)

    def test_stall_escape_prefers_reverse_when_depth_separation_is_tied(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.0,
            "y": 2.0,
            "yaw_deg": 0.0,
            "global_confident": True,
        }
        trajectory = {
            "path_xy_m": [[2.0, 2.0], [3.0, 2.0]],
        }
        points = np.tile(
            np.asarray([[0.34, 0.22, 0.96]], dtype=np.float64),
            (32, 1),
        )
        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_latest_depth_obstacles",
            return_value=points,
        ):
            escape = official_v2_tools._navigate_to_depth_guided_stall_escape(
                SimpleNamespace(world=world), trajectory, snapshot, 1
            )
        self.assertIsNotNone(escape)
        direction = np.asarray(escape["direction_robot_xy"], dtype=np.float64)
        self.assertLess(direction[0], -0.9)
        self.assertAlmostEqual(direction[1], 0.0, places=6)
        before = np.linalg.norm(points[0, :2])
        after = np.linalg.norm(
            points[0, :2]
            - official_v2_tools.NAVIGATE_TO_STALL_RECOVERY_MAX_RETREAT_M
            * direction
        )
        self.assertGreater(after, before)
        self.assertEqual(escape["candidate_count"], 13)
        self.assertEqual(escape["relative_angle_deg"], 180.0)
        self.assertEqual(escape["depth_source"], "latest_depth_frame")
        self.assertEqual(
            escape["selection_policy"],
            "max_depth_separation_then_nearest_reverse",
        )

    def test_depth_corridor_centering_moves_away_from_one_sided_wall(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.0,
            "y": 2.0,
            "yaw_deg": 0.0,
            "global_confident": True,
        }
        trajectory = {"path_xy_m": [[2.0, 2.0], [3.0, 2.0]]}
        wall = np.asarray(
            [
                [0.34, -0.26, z]
                for z in np.linspace(0.25, 1.45, 32)
            ],
            dtype=np.float64,
        )
        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_latest_depth_obstacles",
            return_value=wall,
        ):
            centering = (
                official_v2_tools._navigate_to_depth_corridor_centering(
                    SimpleNamespace(world=world), trajectory, snapshot, 1
                )
            )

        self.assertIsNotNone(centering)
        self.assertEqual(centering["side"], "left")
        np.testing.assert_allclose(
            centering["direction_robot_xy"], [0.0, 1.0], atol=1e-9
        )
        self.assertGreater(centering["side_advantage_m"], 0.03)
        self.assertGreater(centering["clearance_gain_m"], 0.02)
        self.assertEqual(
            centering["selection_policy"],
            "max_lateral_clearance_from_vertical_rgbd_columns",
        )

    def test_depth_corridor_centering_rejects_symmetric_corridor(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.0,
            "y": 2.0,
            "yaw_deg": 0.0,
            "global_confident": True,
        }
        trajectory = {"path_xy_m": [[2.0, 2.0], [3.0, 2.0]]}
        walls = np.asarray(
            [
                [0.34, side * 0.45, z]
                for side in (-1.0, 1.0)
                for z in np.linspace(0.25, 1.45, 32)
            ],
            dtype=np.float64,
        )
        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_latest_depth_obstacles",
            return_value=walls,
        ):
            centering = (
                official_v2_tools._navigate_to_depth_corridor_centering(
                    SimpleNamespace(world=world), trajectory, snapshot, 1
                )
            )

        self.assertIsNone(centering)

    def test_signed_corridor_centering_stays_inside_clearance_ball(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.25,
            "y": 2.0,
            "yaw_deg": 0.0,
            "global_confident": True,
        }
        trajectory = {
            "plan_id": "door-backtrack",
            "path_xy_m": [[2.40, 2.0], [2.25, 2.0]],
            "path_segment_clearance_m": [0.61],
            "path_segment_required_clearance_m": [0.42],
            "clearance": {
                "robot_radius_m": 0.42,
                "required_clearance_m": 0.42,
            },
        }
        centering = (
            official_v2_tools._navigate_to_signed_corridor_centering(
                trajectory,
                snapshot,
                1,
                1,
                direction_map_xy=[0.0, 1.0],
            )
        )

        self.assertIsNotNone(centering)
        self.assertAlmostEqual(centering["shift_m"], 0.15)
        self.assertAlmostEqual(centering["forward_m"], 0.0)
        self.assertAlmostEqual(centering["translation_m"], 0.15)
        self.assertAlmostEqual(centering["centering_clearance_m"], 0.46)
        endpoint = deepcopy(snapshot)
        endpoint["pose"]["x"] = centering["target_xy_m"][0]
        endpoint["pose"]["y"] = centering["target_xy_m"][1]
        corridor = official_v2_tools._navigate_to_segment_corridor(
            centering["trajectory"], endpoint, 1
        )
        self.assertTrue(corridor["ok"], corridor)

    def test_signed_corridor_centering_accepts_exact_separating_sweep(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.10,
            "y": 1.43,
            "yaw_deg": 28.0,
            "global_confident": True,
        }
        direction = np.asarray([-0.47, 0.88], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        start = np.asarray([2.10, 1.43], dtype=np.float64)
        end = start + 0.15 * direction
        trajectory = {
            "plan_id": "tight-door-backtrack",
            "path_xy_m": [[2.12, 1.44], [2.10, 1.43]],
            "path_segment_clearance_m": [0.449],
            "path_segment_required_clearance_m": [0.44],
            "clearance": {
                "robot_radius_m": 0.42,
                "required_clearance_m": 0.44,
            },
        }
        certificate = {
            "schema": "official_v2_fixed_yaw_local_translation_v1",
            "start_xy_m": start.tolist(),
            "end_xy_m": end.tolist(),
            "fixed_yaw_rad": math.radians(28.0),
            "translation_m": 0.15,
            "static_clearance_m": 0.031,
            "static_margin_m": 0.02,
            "initial_depth_clearance_m": 0.012,
            "sweep_depth_clearance_m": 0.012,
            "depth_margin_m": 0.02,
            "monotone_egress": True,
            "tight_point_count": 8,
            "depth_point_count": 120,
            "traversed_evidence": None,
        }

        centering = official_v2_tools._navigate_to_signed_corridor_centering(
            trajectory,
            snapshot,
            1,
            1,
            direction_map_xy=direction,
            translation_certificate=certificate,
        )

        self.assertIsNotNone(centering)
        self.assertAlmostEqual(centering["shift_m"], 0.15)
        self.assertAlmostEqual(centering["centering_clearance_m"], 0.451)
        self.assertEqual(centering["translation_certificate"], certificate)
        self.assertIsNotNone(
            centering["trajectory"]["depth_refinement"][
                "fixed_yaw_translation_certificate"
            ]
        )

    def test_history_centering_offset_is_kept_through_contact_zone(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.20,
            "y": 2.14,
            "yaw_deg": 0.0,
            "global_confident": True,
        }
        trajectory = {
            "plan_id": "committed-door-route",
            "path_xy_m": [[2.0, 2.0], [3.0, 2.0]],
            "path_segment_clearance_m": [0.60],
            "path_segment_required_clearance_m": [0.42],
            "clearance": {
                "robot_radius_m": 0.42,
                "required_clearance_m": 0.42,
                "minimum_execution_clearance_m": 0.60,
            },
            "planner": {
                "traversed_route_direct_replay": True,
                "committed_history_runtime_guard_selected": True,
                "execution_stall_depth_point_count": 0,
                "execution_stall_obstacle_evidence_count": 0,
                "route_sweep": {
                    "occupied_overlap_cells": 0,
                    "unknown_overlap_cells": 0,
                    "live_depth_overlap_cells": 0,
                    "execution_stall_overlap_cells": 0,
                },
            },
        }

        advance = (
            official_v2_tools._navigate_to_signed_corridor_offset_advance(
                trajectory, snapshot, 1, 2
            )
        )

        self.assertIsNotNone(advance)
        self.assertAlmostEqual(advance["advance_m"], 0.40)
        np.testing.assert_allclose(advance["target_xy_m"], [2.60, 2.14])
        self.assertAlmostEqual(advance["advance_clearance_m"], 0.46)
        self.assertAlmostEqual(advance["forward_m"], 0.40)
        self.assertAlmostEqual(advance["translation_m"], 0.0)
        endpoint = deepcopy(snapshot)
        endpoint["pose"]["x"] = advance["target_xy_m"][0]
        endpoint["pose"]["y"] = advance["target_xy_m"][1]
        corridor = official_v2_tools._navigate_to_segment_corridor(
            advance["trajectory"], endpoint, 1
        )
        self.assertTrue(corridor["ok"], corridor)

    def test_offset_advance_requires_committed_history_and_clearance(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.2,
            "y": 2.15,
            "yaw_deg": 0.0,
            "global_confident": True,
        }
        trajectory = {
            "plan_id": "ordinary-route",
            "path_xy_m": [[2.0, 2.0], [3.0, 2.0]],
            "path_segment_clearance_m": [0.60],
            "path_segment_required_clearance_m": [0.42],
            "clearance": {
                "robot_radius_m": 0.42,
                "required_clearance_m": 0.42,
            },
            "planner": {},
        }
        self.assertIsNone(
            official_v2_tools._navigate_to_signed_corridor_offset_advance(
                trajectory, snapshot, 1, 1
            )
        )
        trajectory["planner"] = {
            "traversed_route_direct_replay": True,
            "committed_history_runtime_guard_selected": True,
        }
        snapshot["pose"]["y"] = 2.18
        self.assertIsNone(
            official_v2_tools._navigate_to_signed_corridor_offset_advance(
                trajectory, snapshot, 1, 1
            )
        )

    def test_stall_recovery_retraces_observed_segment_before_depth_escape(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.350873881929821,
            "y": 1.293657330608161,
            "yaw_deg": math.degrees(-1.9402),
            "global_confident": True,
        }
        start = np.asarray(
            [2.603241084673837, 1.5530561349623777], dtype=np.float64
        )
        end = np.asarray(
            [2.0250001661479473, 1.0250001512467861], dtype=np.float64
        )
        trajectory = {
            "plan_id": "live-door-stall",
            "path_xy_m": [start.tolist(), end.tolist()],
            "path_segment_clearance_m": [0.5811260588228495],
            "clearance": {
                "robot_radius_m": 0.42,
                "required_clearance_m": 0.42,
            },
        }
        recovery = official_v2_tools._navigate_to_signed_stall_recovery(
            trajectory,
            snapshot,
            1,
            1,
            # This is the lateral direction chosen by the failed live run.
            escape_direction_map_xy=[-0.674, 0.738],
        )

        self.assertEqual(
            recovery["recovery_mode"], "certified_segment_backtrack"
        )
        delta = end - start
        unit = delta / np.linalg.norm(delta)
        current = np.asarray(
            [snapshot["pose"]["x"], snapshot["pose"]["y"]],
            dtype=np.float64,
        )
        expected = (
            current
            - official_v2_tools.NAVIGATE_TO_STALL_RECOVERY_MAX_RETREAT_M
            * unit
        )
        np.testing.assert_allclose(recovery["target_xy_m"], expected)
        self.assertLess(recovery["forward_m"], 0.0)
        self.assertGreater(recovery["translation_m"], 0.0)

    def test_stall_escape_uses_recent_fused_depth_between_sample_ticks(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.0,
            "y": 2.0,
            "yaw_deg": 0.0,
            "global_confident": True,
        }
        trajectory = {"path_xy_m": [[2.0, 2.0], [3.0, 2.0]]}
        points = np.tile(
            np.asarray([[0.34, 0.22, 0.96]], dtype=np.float64),
            (32, 1),
        )
        with mock.patch.object(
            official_v2_tools,
            "_navigate_to_latest_depth_obstacles",
            return_value=np.empty((0, 3), dtype=np.float64),
        ), mock.patch.object(
            official_v2_tools,
            "_navigate_to_depth_obstacles",
            return_value=points,
        ):
            escape = official_v2_tools._navigate_to_depth_guided_stall_escape(
                SimpleNamespace(world=world), trajectory, snapshot, 1
            )

        self.assertIsNotNone(escape)
        self.assertEqual(
            escape["depth_source"], "recent_fused_depth_fallback"
        )

    def test_depth_guided_stall_escape_keeps_monitoring_clearance_reserve(
        self,
    ) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.0,
            "y": 2.0,
            "yaw_deg": 0.0,
            "global_confident": True,
        }
        trajectory = {
            "plan_id": "stall-source",
            "path_xy_m": [[2.0, 2.0], [3.0, 2.0]],
            "path_segment_clearance_m": [0.50],
            "clearance": {
                "robot_radius_m": 0.42,
                "required_clearance_m": 0.42,
            },
        }
        recovery = official_v2_tools._navigate_to_signed_stall_recovery(
            trajectory,
            snapshot,
            1,
            1,
            escape_direction_map_xy=[0.0, -1.0],
        )

        self.assertEqual(
            recovery["recovery_mode"],
            "depth_guided_local_clearance_escape",
        )
        self.assertAlmostEqual(recovery["retreat_m"], 0.07)
        self.assertGreaterEqual(
            recovery["recovery_clearance_m"],
            0.42
            + official_v2_tools.NAVIGATE_TO_STALL_RECOVERY_CERTIFICATE_RESERVE_M,
        )
        endpoint_snapshot = deepcopy(snapshot)
        endpoint_snapshot["pose"]["x"] = recovery["target_xy_m"][0]
        endpoint_snapshot["pose"]["y"] = (
            recovery["target_xy_m"][1] + 0.001
        )
        corridor = official_v2_tools._navigate_to_segment_corridor(
            recovery["trajectory"], endpoint_snapshot, 1
        )
        self.assertTrue(corridor["ok"], corridor)
        self.assertGreater(corridor["effective_cross_track_limit_m"], 0.0)

    def test_safe_directional_partial_stall_recovery_is_accepted(self) -> None:
        recovery_spec = {"forward_m": -0.01, "translation_m": -0.16}
        outcome = {
            "status": "completed",
            "base_motion_emitted": True,
            "corridor": {"ok": True},
            "report": {
                "ok": False,
                "obstacle_stop_reason": "base_qvel_stall",
                "timed_out": False,
                "actual": {"forward_m": -0.007, "translation_m": -0.104},
            },
        }

        acceptance = official_v2_tools._navigate_to_stall_recovery_acceptance(
            recovery_spec, outcome
        )

        self.assertTrue(acceptance["accepted"], acceptance)
        self.assertTrue(acceptance["partial"], acceptance)
        self.assertGreater(acceptance["projected_progress_m"], 0.10)
        self.assertGreater(acceptance["direction_cosine"], 0.99)

    def test_stall_recovery_budget_is_local_to_each_obstruction(self) -> None:
        recoveries = [
            {
                "stalled_map_pose": {
                    "x_m": float(index) * 0.08,
                    "y_m": 0.0,
                }
            }
            for index in range(6)
        ]
        recoveries.extend(
            {
                "stalled_map_pose": {
                    "x_m": 2.0 + float(index) * 0.08,
                    "y_m": 0.0,
                }
            }
            for index in range(4)
        )

        self.assertEqual(
            official_v2_tools._navigate_to_local_stall_recovery_count(
                recoveries, (0.20, 0.0)
            ),
            6,
        )
        self.assertEqual(
            official_v2_tools._navigate_to_local_stall_recovery_count(
                recoveries, (2.12, 0.0)
            ),
            4,
        )

    def test_partial_stall_recovery_rejects_tiny_or_wrong_way_motion(self) -> None:
        recovery_spec = {"forward_m": 0.0, "translation_m": -0.16}
        base_outcome = {
            "status": "completed",
            "base_motion_emitted": True,
            "corridor": {"ok": True},
            "report": {
                "ok": False,
                "obstacle_stop_reason": "base_qvel_stall",
                "timed_out": False,
                "actual": {"forward_m": 0.0, "translation_m": -0.01},
            },
        }
        tiny = official_v2_tools._navigate_to_stall_recovery_acceptance(
            recovery_spec, base_outcome
        )
        self.assertFalse(tiny["accepted"], tiny)
        self.assertEqual(tiny["reason"], "insufficient_partial_progress")

        wrong_way_outcome = deepcopy(base_outcome)
        wrong_way_outcome["report"]["actual"] = {
            "forward_m": 0.0,
            "translation_m": 0.10,
        }
        wrong_way = official_v2_tools._navigate_to_stall_recovery_acceptance(
            recovery_spec, wrong_way_outcome
        )
        self.assertFalse(wrong_way["accepted"], wrong_way)
        self.assertEqual(wrong_way["reason"], "insufficient_partial_progress")

        unsafe_outcome = deepcopy(base_outcome)
        unsafe_outcome["report"]["actual"] = {
            "forward_m": 0.0,
            "translation_m": -0.10,
        }
        unsafe_outcome["corridor"] = {"ok": False}
        unsafe = official_v2_tools._navigate_to_stall_recovery_acceptance(
            recovery_spec, unsafe_outcome
        )
        self.assertFalse(unsafe["accepted"], unsafe)
        self.assertEqual(unsafe["reason"], "recovery_corridor_not_safe")

    def test_stall_obstacle_follows_policy_odometry_across_map_correction(self) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.0,
            "y": 2.0,
            "yaw_deg": 90.0,
            "global_confident": True,
        }
        snapshot["policy_local_pose"] = {
            "x": 1.0,
            "y": 1.0,
            "yaw_rad": 0.0,
        }
        trajectory = {
            "plan_id": "stall-source",
            "path_xy_m": [[2.0, 2.0], [2.0, 3.0]],
            "start_state": {"base_footprint": {"radius_m": 0.42}},
        }
        record = official_v2_tools._navigate_to_stall_obstacle_record(
            trajectory, snapshot, 1, 1
        )
        np.testing.assert_allclose(record["map_xy_m"], [2.0, 2.6])
        np.testing.assert_allclose(record["policy_xy_m"], [1.6, 1.0])

        corrected = deepcopy(snapshot)
        corrected["pose"] = {
            "x": 4.0,
            "y": 3.0,
            "yaw_deg": 180.0,
            "global_confident": True,
        }
        corrected["policy_local_pose"] = {
            "x": 2.0,
            "y": 1.0,
            "yaw_rad": 0.0,
        }
        planning = official_v2_tools._navigate_to_apply_stall_obstacles(
            corrected, [record]
        )
        np.testing.assert_allclose(
            planning["navigation_stall_obstacle_points_xy_m"], [[4.4, 3.0]]
        )
        expected_cell = (30, 44)
        self.assertTrue(
            planning["navigation_stall_obstacle_mask"][expected_cell]
        )
        self.assertNotIn("navigation_stall_obstacle_mask", corrected)

    def test_stall_plane_uses_exact_depth_and_follows_policy_odometry(self) -> None:
        _adapter, world = self._adapter_world()
        snapshot = self._snapshot(world)
        snapshot["pose"] = {
            "x": 2.0,
            "y": 2.0,
            "yaw_deg": 90.0,
            "global_confident": True,
        }
        snapshot["policy_local_pose"] = {
            "x": 1.0,
            "y": 1.0,
            "yaw_rad": 0.0,
        }
        trajectory = {
            "plan_id": "stall-plane-source",
            "path_xy_m": [[2.0, 2.0], [2.0, 3.0]],
            "start_state": {"base_footprint": {"radius_m": 0.42}},
        }
        record = official_v2_tools._navigate_to_stall_obstacle_record(
            trajectory,
            snapshot,
            1,
            1,
            observed_points_xy_m=[[2.0, 2.4], [2.4, 2.4]],
        )
        self.assertEqual(record["representation"], "exact_depth_plane")
        np.testing.assert_allclose(
            record["policy_points_xy_m"], [[1.4, 1.0], [1.4, 0.6]], atol=1e-9
        )

        corrected = deepcopy(snapshot)
        corrected["pose"] = {
            "x": 4.0,
            "y": 3.0,
            "yaw_deg": 180.0,
            "global_confident": True,
        }
        corrected["policy_local_pose"] = {
            "x": 2.0,
            "y": 1.0,
            "yaw_rad": 0.0,
        }
        planning = official_v2_tools._navigate_to_apply_stall_obstacles(
            corrected, [record]
        )
        np.testing.assert_allclose(
            planning["navigation_stall_depth_points_xy_m"],
            [[4.6, 3.0], [4.6, 3.4]],
            atol=1e-9,
        )
        self.assertNotIn("navigation_stall_obstacle_mask", planning)
        self.assertEqual(planning["navigation_stall_obstacle_cell_count"], 2)
        self.assertEqual(
            planning["navigation_stall_obstacle_evidence_count"], 1
        )

    def test_already_at_goal_requires_fifteen_settled_holds(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(30, 10),
            goal_cell=(30, 10),
        )
        self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        controller = mock.Mock()

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=controller,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_already_arrived",
                            timeout_s=5.0,
                        )
                    )

        controller.assert_not_called()
        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(
            results[0]["arrival_settle_steps"],
            official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS,
        )
        self.assertEqual(results[0]["action_steps"], len(actions) - 1)
        self.assertEqual(
            len(actions),
            official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS + 1,
        )
        self._assert_action_contract(actions)
        for action in actions:
            np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)

    def test_terminal_accepts_benign_joint_tracking_residual(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(30, 10),
            goal_cell=(30, 10),
        )
        self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                controller = build_registry(adapter)["navigate_to"].fn(
                    ctx,
                    name=GOAL_NAME,
                    session_id="navigate_benign_joint_residual",
                    timeout_s=5.0,
                )
                actions = [next(controller)]
                proprio = adapter.proprio_vector()
                proprio[PROPRIO_SLICES["trunk_qpos"]][0] = 1.0e-4
                adapter.update(
                    {"robot_r1::proprio": proprio.astype(np.float32)}
                )
                self._advance_snapshot_observation(snapshot)
                actions.extend(list(controller))

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(
            results[0]["arrival_settle_steps"],
            official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS,
        )
        self._assert_action_contract(actions)
        np.testing.assert_allclose(actions[-1][ACTION_SLICES["base"]], 0.0)

    def test_arrival_settle_requires_consecutive_low_qvel_samples(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(30, 10),
            goal_cell=(30, 10),
        )
        self._install_snapshot_provider(adapter, snapshot)
        qvel_state = {"value": np.zeros(3, dtype=np.float64)}
        world.raw_base_qvel = lambda: qvel_state["value"].copy()
        ctx, results, setter = self._ctx(world)

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                controller = build_registry(adapter)["navigate_to"].fn(
                    ctx,
                    name=GOAL_NAME,
                    session_id="navigate_consecutive_settle",
                    timeout_s=5.0,
                )
                actions = []
                while True:
                    try:
                        action = next(controller)
                    except StopIteration:
                        break
                    actions.append(action)
                    if not results and len(actions) <= 8:
                        qvel_state["value"] = np.asarray(
                            [0.1 if len(actions) % 2 else 0.0, 0.0, 0.0],
                            dtype=np.float64,
                        )
                    else:
                        qvel_state["value"] = np.zeros(3, dtype=np.float64)

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertGreater(
            results[0]["arrival_settle_steps"],
            official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS,
        )
        self.assertEqual(results[0]["action_steps"], len(actions) - 1)
        self._assert_action_contract(actions)

    def test_inactive_pin_or_motion_epoch_drift_fails_before_chassis_action(
        self,
    ) -> None:
        for mutation in ("arm_pin", "gripper_pin", "motion_epoch"):
            with self.subTest(mutation=mutation):
                adapter, world = self._adapter_world()
                state = self._install_snapshot_provider(
                    adapter, self._snapshot(world)
                )
                ctx, results, setter = self._ctx(world)
                calls: list[dict] = []
                real_new_plan = official_v2_tools._navigate_to_new_plan

                def mutate_after_plan(*args, **kwargs):
                    planned = real_new_plan(*args, **kwargs)
                    if mutation == "arm_pin":
                        world.set_arm_pin_qpos(
                            "left", [0.2] + [0.0] * (ARM_DOF - 1)
                        )
                    elif mutation == "gripper_pin":
                        world.latch_gripper_close_keepalive("left")
                    else:
                        world.set_base_velocity(0.2, 0.0, 0.0)
                    return planned

                with tempfile.TemporaryDirectory() as temp_root:
                    with mock.patch.dict(
                        os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}
                    ):
                        with mock.patch.object(
                            official_v2_tools,
                            "_navigate_to_new_plan",
                            side_effect=mutate_after_plan,
                        ):
                            with mock.patch.object(
                                official_v2_tools,
                                "_yield_adjust_chassis_controller",
                                side_effect=self._controller_stub(state, calls),
                            ):
                                actions = list(
                                    build_registry(adapter)["navigate_to"].fn(
                                        ctx,
                                        name=GOAL_NAME,
                                        session_id=f"navigate_drift_{mutation}",
                                        timeout_s=5.0,
                                    )
                                )

                self.assertEqual(setter.call_count, 1)
                self.assertFalse(results[0]["ok"], results[0])
                self.assertEqual(
                    results[0]["failure_stage"], "start-state validation"
                )
                self.assertEqual(calls, [])
                self.assertEqual(len(actions), 1)
                self._assert_action_contract(actions)
                np.testing.assert_allclose(
                    actions[-1][ACTION_SLICES["base"]], 0.0
                )

    def test_same_version_pose_departure_replans_outside_signed_corridor(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(world, goal_cell=(30, 20))
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)
        real_planner = official_v2_tools.plan_clearance_path
        calls: list[dict] = []
        displaced = False

        def conservative_certificates(current, name, **kwargs):
            plan = deepcopy(real_planner(current, name, **kwargs))
            plan["path_segment_clearance_m"] = [
                float(plan["required_clearance_m"])
                for _ in plan["path_segment_clearance_m"]
            ]
            return plan

        def leave_corridor(kind, current_state, _world):
            nonlocal displaced
            if kind != "forward" or displaced:
                return
            displaced = True
            current_state["snapshot"]["pose"]["y"] += 0.09
            self._advance_snapshot_pose(current_state["snapshot"])

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "plan_clearance_path",
                    side_effect=conservative_certificates,
                ):
                    with mock.patch.object(
                        official_v2_tools,
                        "_yield_adjust_chassis_controller",
                        side_effect=self._controller_stub(
                            state, calls, after_phase=leave_corridor
                        ),
                    ):
                        actions = list(
                            build_registry(adapter)["navigate_to"].fn(
                                ctx,
                                name=GOAL_NAME,
                                session_id="navigate_corridor_replan",
                                timeout_s=10.0,
                            )
                        )

        self.assertTrue(displaced)
        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 1)
        self.assertEqual(len(results[0]["plan_records"]), 2)
        self.assertEqual(
            {record["map_version"] for record in results[0]["plan_records"]},
            {"map-v1"},
        )
        self.assertTrue(
            all(
                segment["residual_certified_clearance_m"]
                + 1.0e-9
                >= official_v2_tools.BASE_FOOTPRINT_RADIUS_M
                + official_v2_tools.NAVIGATE_TO_RUNTIME_CLEARANCE_MARGIN_M
                for segment in results[0]["segments"]
            )
        )
        self._assert_action_contract(actions)

    def test_signed_path_disables_uncertified_reverse_recovery(self) -> None:
        adapter, world = self._adapter_world()
        ctx, _results, _setter = self._ctx(world)
        sequence = {"value": 0}
        qvel_reads = {"value": 0}

        def status():
            sequence["value"] += 1
            return {"sequence": sequence["value"]}

        def qvel():
            qvel_reads["value"] += 1
            if qvel_reads["value"] <= official_v2_tools.ADJUST_CHASSIS_STALL_STEPS:
                return np.zeros(3, dtype=np.float64)
            return np.asarray([0.1, 0.0, 0.0], dtype=np.float64)

        adapter.status = status
        world.base_qvel = qvel
        phase = official_v2_tools._yield_adjust_chassis_controller(
            ctx,
            forward=1.0,
            timeout_s=10.0,
            _linear_tolerance_m=0.05,
            _require_linear_target=True,
            _commit_result=False,
            _emit_terminal_hold=False,
            _allow_reverse_recovery=False,
            _result_tool_name="navigate_to",
        )
        actions = []
        while True:
            try:
                actions.append(next(phase))
            except StopIteration as stopped:
                report = stopped.value
                break

        self.assertFalse(report["ok"], report)
        self.assertEqual(
            report["recovery_trigger"],
            "reverse_recovery_disabled_for_signed_path",
        )
        self.assertTrue(actions)
        self._assert_action_contract(actions)
        along_commands = [
            float(action[ACTION_SLICES["base"]][0]) for action in actions
        ]
        self.assertGreater(max(along_commands), 0.0)
        self.assertGreaterEqual(min(along_commands), 0.0)

    def test_navigation_linear_phase_defers_intermediate_settle(self) -> None:
        adapter, world = self._adapter_world()
        ctx, _results, _setter = self._ctx(world)
        sequence = {"value": 0}

        def status():
            sequence["value"] += 1
            return {"sequence": sequence["value"]}

        adapter.status = status
        world.base_qvel = lambda: np.asarray([0.75, 0.0, 0.0])
        phase = official_v2_tools._yield_adjust_chassis_controller(
            ctx,
            forward=0.20,
            timeout_s=2.0,
            _linear_tolerance_m=0.05,
            _require_linear_target=True,
            _commit_result=False,
            _emit_terminal_hold=False,
            _allow_reverse_recovery=False,
            _skip_linear_settle=True,
            _result_tool_name="navigate_to",
            vmax=official_v2_tools.ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
        )
        actions = []
        while True:
            try:
                actions.append(next(phase))
            except StopIteration as stopped:
                report = stopped.value
                break

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["settled_steps"], 0)
        self.assertEqual(
            report["landing_verification"],
            "deferred_to_navigation_causal_hold",
        )
        self.assertLess(len(actions), official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS)
        self.assertTrue(actions)
        self.assertTrue(
            np.any(np.abs(actions[-1][ACTION_SLICES["base"]]) > 0.0)
        )
        self._assert_action_contract(actions)

    def test_navigation_diagonal_uses_each_official_linear_axis_limit(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        ctx, _results, _setter = self._ctx(world)
        sequence = {"value": 0}
        observed_qvel = np.zeros(3, dtype=np.float64)

        def status():
            sequence["value"] += 1
            return {"sequence": sequence["value"]}

        adapter.status = status
        world.base_qvel = lambda: observed_qvel.copy()
        phase = official_v2_tools._yield_adjust_chassis_controller(
            ctx,
            forward=0.50,
            translation=0.50,
            timeout_s=3.0,
            _linear_tolerance_m=0.05,
            _require_linear_target=True,
            _commit_result=False,
            _emit_terminal_hold=False,
            _allow_reverse_recovery=False,
            _skip_linear_settle=True,
            _componentwise_linear_limit=True,
            _result_tool_name="navigate_to",
            vmax=official_v2_tools.ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
            _linear_deceleration_mps2=(
                official_v2_tools.NAVIGATE_TO_LINEAR_DECEL_MPS2
            ),
        )
        actions = []
        while True:
            try:
                action = next(phase)
            except StopIteration as stopped:
                report = stopped.value
                break
            actions.append(action)
            base = np.asarray(action[ACTION_SLICES["base"]], dtype=np.float64)
            observed_qvel[:] = base * np.asarray([0.75, 0.75, 1.0])

        self.assertTrue(report["ok"], report)
        self.assertTrue(actions)
        first_base = np.asarray(
            actions[0][ACTION_SLICES["base"]], dtype=np.float64
        )
        np.testing.assert_allclose(first_base[:2], [1.0, 1.0], atol=1e-9)
        self.assertGreater(float(np.linalg.norm(first_base[:2])), 1.0)
        self._assert_action_contract(actions)

    def test_navigation_alignment_uses_only_its_private_short_settle(self) -> None:
        def run(spin_settle_steps):
            adapter, world = self._adapter_world()
            ctx, _results, _setter = self._ctx(world)
            sequence = {"value": 0}
            observed_qvel = np.zeros(3, dtype=np.float64)

            def status():
                sequence["value"] += 1
                return {"sequence": sequence["value"]}

            adapter.status = status
            world.base_qvel = lambda: observed_qvel.copy()
            kwargs = {}
            if spin_settle_steps is not None:
                kwargs["_spin_settle_steps"] = spin_settle_steps
            phase = official_v2_tools._yield_adjust_chassis_controller(
                ctx,
                spin=30.0,
                timeout_s=5.0,
                _require_linear_target=False,
                _commit_result=False,
                _emit_terminal_hold=False,
                _allow_reverse_recovery=False,
                _skip_linear_settle_for_spin=True,
                _result_tool_name="navigate_to",
                _spin_gain=official_v2_tools.NAVIGATE_TO_SPIN_GAIN,
                **kwargs,
            )
            actions = []
            while True:
                try:
                    action = next(phase)
                except StopIteration as stopped:
                    return actions, stopped.value
                actions.append(action)
                base = np.asarray(
                    action[ACTION_SLICES["base"]], dtype=np.float64
                )
                observed_qvel[:] = base * np.asarray(
                    [
                        official_v2_tools.ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
                        official_v2_tools.ADJUST_CHASSIS_BASE_MAX_LIN_MPS,
                        official_v2_tools.ADJUST_CHASSIS_BASE_MAX_ANG_RADPS,
                    ],
                    dtype=np.float64,
                )

        short_actions, short_report = run(
            official_v2_tools.NAVIGATE_TO_ALIGNMENT_SETTLE_STEPS
        )
        strict_actions, strict_report = run(None)

        self.assertTrue(short_report["ok"], short_report)
        self.assertTrue(strict_report["ok"], strict_report)
        self.assertEqual(
            short_report["settled_steps"],
            official_v2_tools.NAVIGATE_TO_ALIGNMENT_SETTLE_STEPS,
        )
        self.assertEqual(
            short_report["spin_settle_steps_required"],
            official_v2_tools.NAVIGATE_TO_ALIGNMENT_SETTLE_STEPS,
        )
        self.assertEqual(
            strict_report["settled_steps"],
            official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS,
        )
        self.assertEqual(
            len(strict_actions) - len(short_actions),
            official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS
            - official_v2_tools.NAVIGATE_TO_ALIGNMENT_SETTLE_STEPS,
        )
        self._assert_action_contract(short_actions)
        self._assert_action_contract(strict_actions)

    def test_pose_yaw_race_discards_body_frame_action_before_emission(self) -> None:
        adapter, world = self._adapter_world()
        state = self._install_snapshot_provider(
            adapter, self._snapshot(world, goal_cell=(30, 18))
        )
        ctx, results, setter = self._ctx(world)
        normal_controller = self._controller_stub(state, [])
        factory_calls = 0
        stale_phase_entered = False

        def racing_factory(inner_ctx, **kwargs):
            nonlocal factory_calls, stale_phase_entered
            factory_calls += 1
            if factory_calls == 1:
                state["snapshot"]["pose"]["yaw_deg"] += 45.0
                self._advance_snapshot_pose(state["snapshot"])

                def stale_phase():
                    nonlocal stale_phase_entered
                    stale_phase_entered = True
                    yield inner_ctx.world.make_action(base=[0.2, 0.0, 0.0])
                    return {"ok": True, "action_steps": 1}

                return stale_phase()
            return normal_controller(inner_ctx, **kwargs)

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_yield_adjust_chassis_controller",
                    side_effect=racing_factory,
                ):
                    actions = list(
                        build_registry(adapter)["navigate_to"].fn(
                            ctx,
                            name=GOAL_NAME,
                            session_id="navigate_yaw_race",
                            timeout_s=10.0,
                        )
                    )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertFalse(stale_phase_entered)
        self.assertEqual(results[0]["replan_count"], 1)
        self.assertEqual(len(results[0]["plan_records"]), 2)
        np.testing.assert_allclose(actions[0][ACTION_SLICES["base"]], 0.0)
        self._assert_action_contract(actions)

    def test_arrival_settle_accepts_latest_consistent_map_version(self) -> None:
        adapter, world = self._adapter_world()
        snapshot = self._snapshot(
            world,
            start_cell=(30, 10),
            goal_cell=(30, 10),
        )
        state = self._install_snapshot_provider(adapter, snapshot)
        ctx, results, setter = self._ctx(world)

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                controller = build_registry(adapter)["navigate_to"].fn(
                    ctx,
                    name=GOAL_NAME,
                    session_id="navigate_settle_map_updates",
                    timeout_s=10.0,
                )
                actions = []
                while True:
                    try:
                        action = next(controller)
                    except StopIteration:
                        break
                    actions.append(action)
                    if (
                        not results
                        and len(actions)
                        <= official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS
                    ):
                        state["snapshot"]["map_version"] = (
                            f"map-settle-{len(actions)}"
                        )

        self.assertEqual(setter.call_count, 1)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["replan_count"], 0)
        self.assertEqual(len(results[0]["plan_records"]), 1)
        self.assertEqual(
            results[0]["arrival_settle_steps"],
            official_v2_tools.ADJUST_CHASSIS_SETTLE_STEPS,
        )
        self.assertFalse(results[0]["planned_goal_frame_current"])
        self.assertEqual(results[0]["action_steps"], len(actions) - 1)
        self._assert_action_contract(actions)


if __name__ == "__main__":
    unittest.main()
