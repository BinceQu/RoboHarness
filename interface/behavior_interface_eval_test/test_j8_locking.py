from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from behavior_interface_eval_test.episode_initialization import (
    EpisodeGraspPrepController,
)
from behavior_interface_eval_test.official_action_world import (
    ObservationBackedActionWorld,
)
from behavior_interface_eval_test.official_policy_interface import (
    ObservationActionAdapter,
    OfficialPolicyRuntime,
)
from behavior_interface_eval_test.robot_contract import (
    ACTION_DIM,
    ACTION_SLICES,
    ARM_DOF,
    PROPRIO_DIM,
    PROPRIO_SLICES,
)
from behavior_interface_eval_test.tool.official_v2 import build_registry
from behavior_interface_eval_test.tool.official_v2.capabilities import (
    WRIST_ROLL_TEST_TOOL_ENABLED,
    OfficialToolBoundaryError,
    validate_submission,
)
from behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local import (
    LocalRobotState,
    arm_joint_limits,
)
from behavior_interface_eval_test.tool.official_v2.ik_filter_worker import (
    _lock_custom_j8_in_kinematics,
    _set_urdf_arm_only_kinematics,
)
from behavior_interface_eval_test.tool.official_v2.tools import (
    _FrozenCapture,
    _compile_joint_trajectory,
    _load_plan_trajectory,
    _trajectory_digest,
    control_wrist_roll,
)


@unittest.skipUnless(ARM_DOF == 8, "requires r1pro_8dof_hf250 profile")
class J8LockingTest(unittest.TestCase):
    @staticmethod
    def _adapter_world(proprio: np.ndarray | None = None):
        adapter = ObservationActionAdapter()
        adapter.update(
            {
                "robot_r1::proprio": (
                    np.zeros(PROPRIO_DIM, dtype=np.float32)
                    if proprio is None
                    else np.asarray(proprio, dtype=np.float32)
                )
            }
        )
        world = ObservationBackedActionWorld(
            proprio_provider=adapter.proprio_vector,
            hold_action_provider=adapter.hold_action,
            eef_pose_provider=adapter.eef_pose,
        )
        world._official_adapter = adapter
        return adapter, world

    @staticmethod
    def _ctx(world):
        result = {}
        return (
            SimpleNamespace(
                world=world,
                task_name="test_task",
                set_result=lambda value: result.update(value),
                raise_if_cancelled=lambda where="": None,
                is_cancelled=lambda: False,
            ),
            result,
        )

    @staticmethod
    def _follow_action(adapter, world, action) -> None:
        proprio = adapter.proprio_vector().copy()
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                world.controller_action_idx(f"arm_{side}")
            ]
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})

    @staticmethod
    def _assert_action_j8_zero(action) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        for side in ("left", "right"):
            assert float(action[ACTION_SLICES[f"arm_{side}"]][-1]) == 0.0

    def test_adapter_and_world_zero_j8_from_every_action_source(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["arm_left_qpos"]][-1] = 0.75
        proprio[PROPRIO_SLICES["arm_right_qpos"]][-1] = -1.25
        adapter, world = self._adapter_world(proprio)

        self._assert_action_j8_zero(adapter.hold_action())

        raw_official = np.arange(ACTION_DIM, dtype=np.float32)
        self._assert_action_j8_zero(adapter.record_action(raw_official))
        self._assert_action_j8_zero(adapter.adapt_legacy_action(raw_official))

        legacy_27 = np.arange(27, dtype=np.float32)
        legacy_27[25:] = [2.1, -2.2]
        self._assert_action_j8_zero(adapter.adapt_legacy_action(legacy_27))

        world_action = world.make_action(
            arm_left=[0.1] * 7 + [1.7],
            arm_right=[-0.1] * 7 + [-1.8],
        )
        self._assert_action_j8_zero(world_action)
        self.assertEqual(world.arm_pin_qpos_list("left")[-1], 0.0)
        self.assertEqual(world.arm_pin_qpos_list("right")[-1], 0.0)

    @unittest.skipUnless(
        WRIST_ROLL_TEST_TOOL_ENABLED,
        "requires BEHAVIOR_EVAL_TEST_ENABLE_WRIST_ROLL_TOOL=1",
    )
    def test_control_wrist_roll_is_selected_only_cumulative_and_resettable(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        left_entry = np.array(
            [0.2, 0.1, -0.2, -0.6, 0.25, -0.1, 0.15, 0.0],
            dtype=np.float32,
        )
        right_entry = np.array(
            [-0.1, -0.2, 0.3, -0.7, -0.15, 0.2, -0.25, 0.0],
            dtype=np.float32,
        )
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = left_entry
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = right_entry
        adapter, world = self._adapter_world(proprio)
        adapter.set_final_action_overlay(
            world.enforce_official_action,
            owns_locked_joints=True,
        )

        def execute(mode: str, angle_deg: float):
            ctx, result = self._ctx(world)
            actions = []
            for raw_action in control_wrist_roll(
                ctx,
                arm="right",
                mode=mode,
                angle_deg=angle_deg,
                timeout_s=5.0,
            ):
                action = adapter.record_action(raw_action)
                actions.append(action.copy())
                self._follow_action(adapter, world, action)
            return result, actions

        first, first_actions = execute("relative", 45.0)
        self.assertTrue(first["ok"], first)
        self.assertGreater(len(first_actions), 2)
        self.assertAlmostEqual(
            world.tool_roll_pin_qpos("right"),
            math.radians(45.0),
            places=6,
        )
        for action in first_actions:
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_right"]][:7],
                right_entry[:7],
                atol=1e-7,
            )
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_left"]],
                left_entry,
                atol=1e-7,
            )

        second, _second_actions = execute("relative", 30.0)
        self.assertTrue(second["ok"], second)
        self.assertAlmostEqual(second["before_deg"], 45.0, places=4)
        self.assertAlmostEqual(second["target_deg"], 75.0, places=4)
        self.assertAlmostEqual(
            world.tool_roll_pin_qpos("right"),
            math.radians(75.0),
            places=6,
        )

        unauthorized = np.asarray(
            world.arm_qpos_list("right"),
            dtype=np.float64,
        )
        unauthorized[7] = -1.0
        locked_action = world.make_action(arm_right=unauthorized.tolist())
        self.assertAlmostEqual(
            float(locked_action[ACTION_SLICES["arm_right"]][-1]),
            math.radians(75.0),
            places=6,
        )

        reset, _reset_actions = execute("reset", 123.0)
        self.assertTrue(reset["ok"], reset)
        self.assertAlmostEqual(reset["target_deg"], 0.0, places=7)
        self.assertAlmostEqual(world.tool_roll_pin_qpos("right"), 0.0, places=7)
        self.assertAlmostEqual(
            float(adapter.proprio_vector()[PROPRIO_SLICES["arm_right_qpos"]][-1]),
            0.0,
            places=7,
        )

    @unittest.skipUnless(
        WRIST_ROLL_TEST_TOOL_ENABLED,
        "requires BEHAVIOR_EVAL_TEST_ENABLE_WRIST_ROLL_TOOL=1",
    )
    def test_control_wrist_roll_boundary_validation(self) -> None:
        normalized = validate_submission(
            "control_wrist_roll",
            {
                "arm": "LEFT",
                "mode": "reset",
                "angle_deg": 90,
                "timeout_s": 4,
            },
        )
        self.assertEqual(normalized["arm"], "left")
        self.assertEqual(normalized["mode"], "reset")
        self.assertEqual(normalized["angle_deg"], 0.0)
        with self.assertRaises(OfficialToolBoundaryError):
            validate_submission(
                "control_wrist_roll",
                {"arm": "both", "mode": "relative", "angle_deg": 10},
            )
        with self.assertRaises(OfficialToolBoundaryError):
            validate_submission(
                "control_wrist_roll",
                {"arm": "right", "mode": "spin", "angle_deg": 10},
            )

    def test_episode_initializer_uses_observation_action_loop_and_resets(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = np.array(
            [0.4, 0.2, -0.1, -0.5, 0.3, 0.2, -0.2, 0.9],
            dtype=np.float32,
        )
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = np.array(
            [-0.3, -0.2, 0.1, -0.4, -0.2, 0.3, 0.2, -0.8],
            dtype=np.float32,
        )
        adapter, world = self._adapter_world(proprio)
        initializer = EpisodeGraspPrepController(
            arm_dof=8,
            max_step_rad=0.15,
            tolerance_rad=0.01,
            timeout_s=5.0,
        )

        actions = []
        for _ in range(150):
            action = initializer.step(world)
            actions.append(np.asarray(action).copy())
            self._assert_action_j8_zero(action)
            self._follow_action(adapter, world, action)
            if initializer.ready():
                break

        self.assertTrue(initializer.ready(), initializer.status())
        self.assertGreater(len(actions), 3)
        self.assertFalse(initializer.status()["timed_out"])
        np.testing.assert_allclose(
            adapter.proprio_vector()[PROPRIO_SLICES["arm_left_qpos"]],
            initializer.target,
            atol=1e-7,
        )

        initializer.reset()
        world.reset_observation_state()
        self.assertFalse(initializer.ready())
        self.assertFalse(world.episode_initialized())

    def test_runtime_preempts_tools_until_startup_initialization_converges(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["arm_left_qpos"]][3] = -0.4
        proprio[PROPRIO_SLICES["arm_left_qpos"]][7] = 0.8
        proprio[PROPRIO_SLICES["arm_right_qpos"]][3] = -0.3
        proprio[PROPRIO_SLICES["arm_right_qpos"]][7] = -0.9
        adapter, world = self._adapter_world(proprio)
        tool_calls = []
        logs = []
        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime._step_lock = threading.RLock()
        runtime.adapter = adapter
        runtime.server = SimpleNamespace(world=world, log=logs.append)
        runtime.episode_initializer = EpisodeGraspPrepController(
            arm_dof=8,
            max_step_rad=0.15,
            tolerance_rad=0.01,
            timeout_s=5.0,
        )
        runtime.downstream = None
        runtime._source = "hold"
        runtime._last_error = ""
        runtime._update_ui = lambda: None
        runtime._tool_action = lambda: tool_calls.append(True)

        for _ in range(150):
            action = runtime.act({"robot_r1::proprio": proprio})
            self._assert_action_j8_zero(action)
            self.assertEqual(tool_calls, [])
            for side in ("left", "right"):
                proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                    ACTION_SLICES[f"arm_{side}"]
                ]
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            if runtime.episode_initializer.ready():
                break

        self.assertTrue(runtime.episode_initializer.ready())
        self.assertTrue(world.episode_initialized())
        runtime.act({"robot_r1::proprio": proprio})
        self.assertEqual(tool_calls, [True])

    def test_planning_and_exec_are_blocked_before_initialization(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)

        plan_ctx, plan_result = self._ctx(world)
        plan_actions = list(
            registry["plan_grasp_point_filter_rgbd_lite"].fn(
                plan_ctx,
                session_id="blocked",
                image_id="img_0001",
                u=500,
                v=500,
            )
        )
        self.assertFalse(plan_result["ok"])
        self.assertIn("initialization is incomplete", plan_result["error"])
        self.assertEqual(len(plan_actions), 1)

        exec_ctx, exec_result = self._ctx(world)
        exec_actions = list(
            registry["exec_plan_pose"].fn(
                exec_ctx,
                session_id="blocked",
                plan_id="plan_0001",
            )
        )
        self.assertFalse(exec_result["ok"])
        self.assertIn("initialization is incomplete", exec_result["error"])
        self.assertEqual(len(exec_actions), 1)
        self._assert_action_j8_zero(exec_actions[0])

    def test_grasp_prep_reports_real_convergence_and_timeout(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["arm_left_qpos"]][-1] = 0.6
        proprio[PROPRIO_SLICES["arm_right_qpos"]][-1] = -0.7
        adapter, world = self._adapter_world(proprio)
        world.set_episode_initialized(True)
        registry = build_registry(adapter)
        ctx, result = self._ctx(world)

        actions = []
        for action in registry["set_arm_to_grasp_position"].fn(
            ctx,
            arm="both",
            tol=0.01,
            timeout_s=8.0,
            max_dq_per_step=0.15,
        ):
            action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
            actions.append(action.copy())
            self._assert_action_j8_zero(action)
            self._follow_action(adapter, world, action)

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["converged"])
        self.assertLessEqual(result["max_abs_error_rad"], 0.01)
        self.assertEqual(len(result["target_qpos"]["left"]), 8)
        self.assertEqual(result["target_qpos"]["left"][-1], 0.0)

        stalled_proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        stalled_proprio[PROPRIO_SLICES["arm_right_qpos"]][3] = 0.2
        stalled_adapter, stalled_world = self._adapter_world(stalled_proprio)
        stalled_world.set_episode_initialized(True)
        stalled_ctx, stalled_result = self._ctx(stalled_world)
        stalled_registry = build_registry(stalled_adapter)
        stalled_actions = list(
            stalled_registry["set_arm_to_grasp_position"].fn(
                stalled_ctx,
                arm="right",
                tol=0.001,
                timeout_s=0.2,
            )
        )
        self.assertFalse(stalled_result["ok"])
        self.assertFalse(stalled_result["converged"])
        self.assertIn("before timeout", stalled_result["error"])
        self.assertLessEqual(stalled_result["action_steps"], 6)
        for action in stalled_actions:
            self._assert_action_j8_zero(action)

    def test_grasp_prep_resets_and_locks_both_j8_pins(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["arm_left_qpos"]][7] = -0.6
        proprio[PROPRIO_SLICES["arm_right_qpos"]][7] = 0.8
        adapter, world = self._adapter_world(proprio)
        world.set_tool_roll_pin_qpos("left", -0.6)
        world.set_tool_roll_pin_qpos("right", 0.8)
        registry = build_registry(adapter)
        ctx, result = self._ctx(world)

        actions = []
        for raw_action in registry["set_arm_to_grasp_position"].fn(
            ctx,
            arm="right",
            tol=0.01,
            timeout_s=8.0,
            max_dq_per_step=0.15,
        ):
            action = np.asarray(raw_action, dtype=np.float32).reshape(ACTION_DIM)
            actions.append(action.copy())
            self.assertAlmostEqual(
                float(action[ACTION_SLICES["arm_right"]][7]),
                0.0,
                places=6,
            )
            self.assertAlmostEqual(
                float(action[ACTION_SLICES["arm_left"]][7]),
                0.0,
                places=6,
            )
            self._follow_action(adapter, world, action)

        self.assertTrue(result["ok"], result)
        self.assertNotIn("j8_resets", result)
        self.assertAlmostEqual(
            result["target_qpos"]["right"][7],
            0.0,
            places=6,
        )
        self.assertAlmostEqual(world.tool_roll_pin_qpos("right"), 0.0)
        self.assertAlmostEqual(world.tool_roll_pin_qpos("left"), 0.0)

        unauthorized = np.asarray(
            world.arm_qpos_list("right"),
            dtype=np.float64,
        )
        unauthorized[7] = 1.2
        locked_action = world.make_action(arm_right=unauthorized.tolist())
        self.assertAlmostEqual(
            float(locked_action[ACTION_SLICES["arm_right"]][7]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(locked_action[ACTION_SLICES["arm_left"]][7]),
            0.0,
            places=6,
        )

    def test_local_ik_contract_removes_j8_from_active_cspace(self) -> None:
        names = [f"right_arm_joint{index}" for index in range(1, 9)]
        kin = {
            "lock_joints": {},
            "cspace": {
                "joint_names": list(names),
                "cspace_distance_weight": list(range(8)),
                "retract_config": [0.1] * 8,
            },
        }
        request = {
            "arm_dof": 8,
            "q_by_name": {name: 0.5 for name in names},
        }

        _lock_custom_j8_in_kinematics(kin, request, "right")

        self.assertEqual(kin["lock_joints"]["right_arm_joint8"], 0.0)
        self.assertNotIn("right_arm_joint8", kin["cspace"]["joint_names"])
        self.assertEqual(len(kin["cspace"]["retract_config"]), 7)
        self.assertEqual(request["q_by_name"]["right_arm_joint8"], 0.0)

        state = LocalRobotState(
            arm_dof=8,
            base_pos=np.zeros(3),
            base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            trunk_q=np.zeros(4),
            arm_left_q=np.ones(8),
            arm_right_q=-np.ones(8),
            gripper_left_q=np.zeros(2),
            gripper_right_q=np.zeros(2),
            robot_forward=np.array([1.0, 0.0, 0.0]),
        )
        self.assertEqual(state.q_by_name()["left_arm_joint8"], 0.0)
        lower, upper = arm_joint_limits(state, "right")
        self.assertEqual(float(lower[7]), 0.0)
        self.assertEqual(float(upper[7]), 0.0)

    def test_single_arm_ik_config_only_locks_selected_arm_j8(self) -> None:
        names = [
            f"{side}_arm_joint{index}"
            for side in ("left", "right")
            for index in range(1, 9)
        ]
        robot_cfg = {
            "kinematics": {
                "lock_joints": {
                    "left_arm_joint8": 0.0,
                    "right_arm_joint8": 0.0,
                },
                "cspace": {
                    "joint_names": list(names),
                    "cspace_distance_weight": [1.0] * len(names),
                    "retract_config": [0.0] * len(names),
                },
            }
        }
        request = {
            "arm_dof": 8,
            "q_by_name": {name: 0.0 for name in names},
            "eef_link_names": {"left": "left_eef_link"},
            "eef_extra_links": {
                "left": {
                    "parent_link_name": "left_gripper_link",
                    "link_name": "left_eef_link",
                    "fixed_transform": [0.0, 0.0, -0.06, 0.0, 0.0, 1.0, 0.0],
                    "joint_type": "FIXED",
                    "joint_name": "left_gripper_to_eef_fixed_joint",
                }
            },
        }
        urdf_path = str(
            Path(__file__).parent
            / "tool"
            / "official_v2"
            / "assets"
            / "r1pro_8dof_hf250_kinematics.urdf"
        )

        _set_urdf_arm_only_kinematics(
            robot_cfg,
            request,
            "left",
            "left_eef_link",
            urdf_path,
        )
        _lock_custom_j8_in_kinematics(
            robot_cfg["kinematics"],
            request,
            "left",
        )

        self.assertEqual(
            robot_cfg["kinematics"]["lock_joints"],
            {"left_arm_joint8": 0.0},
        )
        self.assertEqual(
            robot_cfg["kinematics"]["cspace"]["joint_names"],
            [f"left_arm_joint{index}" for index in range(1, 8)],
        )

    def _compiled_trajectory(self, world):
        start = np.zeros(8, dtype=np.float64)
        start[7] = 0.01
        robot = {
            "arm_left_qpos": start.tolist(),
            "arm_right_qpos": start.tolist(),
            "arm_left_qvel": np.zeros(8).tolist(),
            "arm_right_qvel": np.zeros(8).tolist(),
            "trunk_qpos": np.zeros(4).tolist(),
            "gripper_left_qpos": np.zeros(2).tolist(),
            "gripper_right_qpos": np.zeros(2).tolist(),
            "motion_epoch": world.motion_epoch(),
            "episode_id": world.episode_id(),
        }
        capture = _FrozenCapture(
            session_id="j8",
            image_id="img_0001",
            role="head",
            depth=np.ones((2, 2), dtype=np.float32),
            camera={},
            robot=robot,
        )
        safe_q = np.zeros(8, dtype=np.float64)
        safe_q[0] = -0.08
        safe_q[7] = 0.7
        final_q = np.zeros(8, dtype=np.float64)
        final_q[0] = -0.16
        final_q[7] = 1.9
        payload = {
            "recommended_arm": "right",
            "selected_pose_ik_q": {"right": final_q.tolist()},
            "candidates": [
                {
                    "arm": "right",
                    "meta": {
                        "planned_safe": {
                            "ok": True,
                            "active_back_m": 0.10,
                            "safe_q": safe_q.tolist(),
                        }
                    },
                }
            ],
        }
        ctx, _ = self._ctx(world)
        trajectory = _compile_joint_trajectory(
            ctx=ctx,
            plan_id="plan_0001",
            session_id="j8",
            image_id="img_0001",
            capture=capture,
            payload=payload,
        )
        return trajectory, payload

    @staticmethod
    def _write_plan(root: str, trajectory: dict) -> None:
        plan_dir = Path(root) / "j8" / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "plan_0001.json").write_text(
            json.dumps(
                {
                    "plan_id": "plan_0001",
                    "session_id": "j8",
                    "tool": "plan_grasp_point_filter_rgbd_lite",
                    "trajectory": trajectory,
                }
            ),
            encoding="utf-8",
        )

    def test_trajectory_zeroes_j8_and_loader_rejects_nonzero_j8(self) -> None:
        adapter, world = self._adapter_world()
        world.set_episode_initialized(True)
        trajectory, payload = self._compiled_trajectory(world)

        self.assertEqual(payload["selected_pose_ik_q"]["right"][7], 0.0)
        self.assertEqual(
            payload["candidates"][0]["meta"]["planned_safe"]["safe_q"][7],
            0.0,
        )
        for side in ("left", "right"):
            self.assertEqual(trajectory["start_state"]["arm_qpos"][side][7], 0.0)
        for segment in trajectory["segments"]:
            self.assertEqual(segment["endpoint_q"][7], 0.0)
            self.assertTrue(all(q[7] == 0.0 for q in segment["waypoints"]))

        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": root}):
                self._write_plan(root, trajectory)
                loaded, _ = _load_plan_trajectory("j8", "plan_0001")
                self.assertEqual(loaded["locked_arm_joints"], {"J8": 0.0})

                trajectory["start_state"]["arm_qpos"]["left"][7] = 0.4
                trajectory["integrity"]["digest"] = _trajectory_digest(trajectory)
                self._write_plan(root, trajectory)
                with self.assertRaisesRegex(ValueError, "non-zero locked J8"):
                    _load_plan_trajectory("j8", "plan_0001")

    def test_exec_resets_only_execution_arm_j8_before_trajectory(self) -> None:
        adapter, world = self._adapter_world()
        world.set_episode_initialized(True)
        trajectory, _ = self._compiled_trajectory(world)
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_left_qpos"]][7] = -0.4
        proprio[PROPRIO_SLICES["arm_right_qpos"]][7] = 0.25
        adapter.update({"robot_r1::proprio": proprio})
        world.set_tool_roll_pin_qpos("left", -0.4)
        world.set_tool_roll_pin_qpos("right", 0.25)
        registry = build_registry(adapter)

        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": root}):
                self._write_plan(root, trajectory)
                ctx, result = self._ctx(world)
                actions = []
                for raw_action in registry["exec_plan_pose"].fn(
                    ctx,
                    session_id="j8",
                    plan_id="plan_0001",
                ):
                    action = np.asarray(raw_action, dtype=np.float32).reshape(
                        ACTION_DIM
                    )
                    actions.append(action.copy())
                    self.assertEqual(
                        float(action[ACTION_SLICES["arm_right"]][7]),
                        0.0,
                    )
                    self.assertAlmostEqual(
                        float(action[ACTION_SLICES["arm_left"]][7]),
                        -0.4,
                        places=6,
                    )
                    self._follow_action(adapter, world, action)

        self.assertTrue(result["ok"], result)
        self.assertGreater(len(actions), 3)
        self.assertTrue(result["j8_reset"]["ok"])
        self.assertAlmostEqual(
            result["j8_reset"]["before_deg"],
            math.degrees(0.25),
            places=4,
        )
        self.assertAlmostEqual(world.tool_roll_pin_qpos("right"), 0.0)
        self.assertAlmostEqual(world.tool_roll_pin_qpos("left"), -0.4)


if __name__ == "__main__":
    unittest.main()
