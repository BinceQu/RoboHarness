from __future__ import annotations

import ast
import concurrent.futures
import hashlib
import inspect
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
from flask import Flask, jsonify

import behavior_interface_eval_test.official_policy_interface as official_policy_interface
import behavior_interface_eval_test.tool.official_v2.tools as official_v2_tools
import behavior_interface_eval_test.tool.official_v2.visualization_local as visualization_local
from behavior_interface.tool.v2 import V2_TOOL_NAMES
from behavior_interface_eval_test.official_action_world import (
    ACTION_DIM,
    ObservationBackedActionWorld,
)
from behavior_interface_eval_test.robot_contract import (
    ACTION_SLICES,
    ARM_DOF,
    PROPRIO_DIM,
    PROPRIO_SLICES,
)
from behavior_interface_eval_test.official_policy_interface import (
    OFFICIAL_V2_CAPTURE_POLICIES,
    ObservationActionAdapter,
    OfficialPolicyRuntime,
    install_official_adjust_height_route,
    install_official_head_adjust_routes,
    install_official_set_arm_route,
    install_official_surface_facing_route,
    install_official_track_object_distance_routes,
    install_official_v2_failure_capture_contract,
    install_official_wrist_roll_test_route,
    install_strict_http_control_boundary,
)
from behavior_interface_eval_test.tool.official_v2 import (
    PUBLIC_TOOLS,
    WRIST_ROLL_TEST_TOOL_ENABLED,
    OfficialToolBoundaryError,
    build_registry,
    capability_report,
    ensure_profile_installed,
    install_profile,
    translate_submission,
    validate_submission,
)
from behavior_interface_eval_test.tool.official_v2.tools import (
    ADJUST_CHASSIS_FORWARD_TOL_M,
    ADJUST_EEF_LOCAL_BUILD,
    DEFAULT_TRAJECTORY_STEP_RAD,
    GRASP_PREP_Q,
    GRIPPER_OPEN_EFFORT_N,
    PUBLIC_TOOL_FUNCTIONS,
    REACH_BASE_COMMAND_SPEED_SCALE,
    REACH_BASE_MAX_SPEED_MPS,
    REACH_BASE_MAX_YAW_RATE_RADPS,
    REACH_BASE_OFFICIAL_MAX_SPEED_MPS,
    REACH_BASE_OFFICIAL_MAX_YAW_RATE_RADPS,
    TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD,
    TRUNK_LIMITS,
    _ARM_LIMITS,
    _compile_joint_trajectory,
    _require_signed_trunk_q,
    _trajectory_digest,
    _base_action_from_physical_velocity,
    _interpolate_joint_segment,
    _load_frozen_capture,
    _load_plan_trajectory,
    _pick_reach_base_goal,
    _r1pro_arm_fk_robot,
    _reach_chord_plan,
    _reach_pitch_interpolation_steps,
    _reach_same_target_measurement,
    _solve_j567_keep_orientation,
    _v2_local_base_command,
    _yield_reach_compensation,
)
from behavior_interface_eval_test.tool.official_v2.grasp_geometry_local import (
    gripper_geometry,
    mat_to_quat_xyzw,
)
from behavior_interface_eval_test.tool.official_v2.eef_adjustment_local import (
    ADJUST_POSE_MAX_JOINT_STEP_RAD,
)
from behavior_interface_eval_test.tool.official_v2.base_path_overlay_local import (
    BASE_FRONT_OFFSET_M,
    CHASSIS_FORWARD_2M_CLEAR,
    compose_nearby_object_warning,
    LABEL_COLOR_RGB,
    LABEL_STROKE_WIDTH_PX,
    LATERAL_REFERENCE_OFFSET_M,
    LATERAL_REFERENCE_OUTER_OFFSET_M,
    NEAR_EDGE_OUTER_LABEL_OUTSET_M,
    PATH_END_X_M,
    PATH_SIDE_INSET_M,
    PATH_START_X_M,
    PATH_WIDTH_M,
    PATH_Y_CENTER_M,
)
from behavior_interface_eval_test.tool.official_v2.eef_adjustment_local import (
    camera_frame_pose_target,
    local_robot_state,
    orientation_error_deg,
    robot_base_frame_pose_target,
)
from behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local import (
    LocalRobotState,
    eef_pose as local_eef_pose,
)
from behavior_interface_eval_test.tool.official_v2.rgbd_grasp_planner import (
    AXIAL_Z_M,
    N_ROLL,
    generate_rgbd_filter_poses,
)
from behavior_interface_eval_test.tool.official_v2.surface_facing_local import (
    SURFACE_FACING_STANDOFF_M,
    SurfaceFacingSolution,
    rebase_robot_points,
    solve_surface_facing_target,
)
from behavior_interface_eval_test.tool.official_v2.visualization_local import (
    render_head_path_overlay,
    render_plan_gripper_overlay,
    render_wrist_grasp_volume_overlay,
)


class OfficialV2ToolsTest(unittest.TestCase):
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

    def test_uncommanded_arm_action_replaces_zero_effort_with_qpos_hold(self) -> None:
        adapter, world = self._adapter_world()
        if not world.gripper_uses_effort("right"):
            self.skipTest("independent effort profile is required")
        proprio = adapter.proprio_vector().copy()
        entry_qpos = np.asarray([0.0497, 0.0499], dtype=np.float32)
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = entry_qpos
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})

        # Reproduce the old open -> zero effort -> arm-only exec sequence.
        zero = world.make_action(gripper_effort_right=[0.0, 0.0])
        adapter.record_action(zero)
        self.assertIsNone(world.gripper_hold_qpos_list("right"))
        first = world.make_action(arm_right=world.arm_qpos_list("right"))
        np.testing.assert_allclose(
            world.gripper_hold_qpos_list("right"),
            entry_qpos,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            first[ACTION_SLICES["gripper_right"]],
            [0.0, 0.0],
            atol=1e-7,
        )
        adapter.record_action(first)

        disturbed = adapter.proprio_vector().copy()
        disturbed[PROPRIO_SLICES["gripper_right_qpos"]] = [0.045, 0.047]
        disturbed[PROPRIO_SLICES["gripper_right_qvel"]] = [-0.020, -0.010]
        adapter.update({"robot_r1::proprio": disturbed})
        corrected = world.make_action(arm_right=world.arm_qpos_list("right"))
        correction = corrected[ACTION_SLICES["gripper_right"]]
        self.assertTrue(bool(np.all(correction > 0.0)), correction.tolist())
        self.assertLessEqual(float(np.max(correction)), 0.5 + 1e-7)
        self.assertNotAlmostEqual(float(correction[0]), float(correction[1]))

        # A raw policy action has an explicit gripper slice and must not be
        # replaced by a tool-local qpos hold at the final boundary.
        explicit_policy = np.zeros(ACTION_DIM, dtype=np.float32)
        explicit_policy[ACTION_SLICES["gripper_right"]] = [-0.2, -0.3]
        recorded = adapter.record_action(explicit_policy)
        np.testing.assert_allclose(
            recorded[ACTION_SLICES["gripper_right"]],
            [-0.2, -0.3],
            atol=1e-7,
        )

    def test_open_gripper_finishes_in_observed_qpos_hold(self) -> None:
        adapter, world = self._adapter_world()
        if not world.gripper_uses_effort("left"):
            self.skipTest("independent effort profile is required")
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["gripper_left_qpos"]] = [0.020, 0.021]
        proprio[PROPRIO_SLICES["gripper_left_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)
        actions = []

        for index, action in enumerate(
            build_registry(adapter)["open_gripper"].fn(ctx, arm="left")
        ):
            recorded = adapter.record_action(action)
            actions.append(recorded.copy())
            proprio = adapter.proprio_vector().copy()
            if index < 6:
                opening = min(0.0498, 0.025 + 0.005 * index)
                proprio[PROPRIO_SLICES["gripper_left_qpos"]] = [
                    opening,
                    min(0.0496, opening + 0.0004),
                ]
                proprio[PROPRIO_SLICES["gripper_left_qvel"]] = [0.015, 0.014]
                adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["post_open_hold_mode"], "qpos_effort_servo")
        self.assertFalse(world.gripper_hold_allows_close_effort("left"))
        np.testing.assert_allclose(
            result["post_open_hold_qpos"],
            [0.0498, 0.0496],
            atol=1e-6,
        )
        self.assertEqual(len(actions), 7)
        self.assertEqual(result["open_action_steps"], 6)
        self.assertTrue(result["open_target_reached"])
        np.testing.assert_allclose(
            actions[-1][ACTION_SLICES["gripper_left"]],
            [0.0, 0.0],
            atol=1e-7,
        )

    def test_open_gripper_already_open_uses_one_release_frame(self) -> None:
        adapter, world = self._adapter_world()
        if not world.gripper_uses_effort("right"):
            self.skipTest("independent effort profile is required")
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.0498, 0.0497]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        actions = list(
            build_registry(adapter)["open_gripper"].fn(ctx, arm="right")
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["open_action_steps"], 1)
        self.assertTrue(result["open_target_reached"])
        self.assertEqual(len(actions), 2)
        np.testing.assert_allclose(
            actions[0][ACTION_SLICES["gripper_right"]],
            [3.0, 3.0],
            atol=1e-7,
        )

    def test_close_keepalive_overrides_qpos_hold(self) -> None:
        adapter, world = self._adapter_world()
        if not world.gripper_uses_effort("right"):
            self.skipTest("independent effort profile is required")
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.030, 0.031]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        world.capture_gripper_hold_qpos("right")
        world.latch_gripper_close_keepalive("right", effort=[-0.1, -0.1])

        action = world.make_action(arm_right=world.arm_qpos_list("right"))
        self.assertIsNone(world.gripper_hold_qpos_list("right"))
        np.testing.assert_allclose(
            action[ACTION_SLICES["gripper_right"]],
            [-0.1, -0.1],
            atol=1e-7,
        )

    def test_close_gripper_confirms_bilateral_stall_and_latches_carry(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.05, 0.05]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = [0.0, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        registry = build_registry(adapter)
        ctx, result = self._ctx(world)
        world.latch_gripper_close_keepalive(
            "right",
            effort=[-0.5, -0.5]
            if world.gripper_uses_effort("right")
            else None,
        )
        generator = registry["close_gripper"].fn(
            ctx,
            arm="right",
            timeout_s=2.0,
        )

        actions = []
        for index, action in enumerate(generator):
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            if index < 4:
                qpos = max(0.038, 0.05 - 0.003 * (index + 1))
                proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [qpos, qpos]
                proprio[PROPRIO_SLICES["gripper_right_qvel"]] = [-0.02, -0.02]
            else:
                proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.038, 0.038]
                proprio[PROPRIO_SLICES["gripper_right_qvel"]] = [0.0, 0.0]
            adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["grasp_confirmed"])
        self.assertFalse(result["constraint_directly_observed"])
        self.assertEqual(
            result["confirmation_source"],
            "legal_gripper_qpos_bilateral_stall_window",
        )
        self.assertGreaterEqual(
            result["confirmation_steps"],
            result["confirmation_steps_required"],
        )
        self.assertTrue(result["close_keepalive_active"])

        gripper_actions = np.asarray(
            [action[ACTION_SLICES["gripper_right"]] for action in actions],
            dtype=np.float64,
        )
        if world.gripper_uses_effort("right"):
            self.assertLessEqual(float(np.max(np.abs(gripper_actions))), 20.0)
            self.assertEqual(result["peak_seek_effort_n"], 0.5)
            self.assertEqual(result["peak_precontact_effort_n"], 0.5)
            self.assertEqual(result["peak_confirmation_effort_n"], 20.0)
            confirmation = -gripper_actions[
                np.all(gripper_actions <= -1.0 + 1e-7, axis=1)
            ][:12]
            self.assertEqual(len(confirmation), 12)
            np.testing.assert_allclose(confirmation[:, 0], confirmation[:, 1])
            self.assertAlmostEqual(confirmation[0, 0], 1.0)
            self.assertAlmostEqual(confirmation[-1, 0], 20.0)
            self.assertTrue(bool(np.all(np.diff(confirmation[:, 0]) >= 0.0)))
        else:
            np.testing.assert_allclose(gripper_actions, -1.0)

        overwritten = np.zeros(ACTION_DIM, dtype=np.float32)
        overwritten[ACTION_SLICES["gripper_right"]] = 1.0
        kept = adapter.record_action(overwritten)
        np.testing.assert_allclose(
            kept[ACTION_SLICES["gripper_right"]],
            [-20.0, -20.0]
            if world.gripper_uses_effort("right")
            else [-1.0],
        )
        if world.gripper_uses_effort("right"):
            self.assertEqual(result["carry_effort_n"], 20.0)

    def test_close_gripper_first_contact_finger_drops_to_hold_effort(self) -> None:
        adapter, world = self._adapter_world()
        if not world.gripper_uses_effort("right"):
            self.skipTest("independent effort profile is required")
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.05, 0.05]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = [0.0, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        registry = build_registry(adapter)
        ctx, result = self._ctx(world)
        actions = []

        for index, action in enumerate(
            registry["close_gripper"].fn(ctx, arm="right", timeout_s=4.0)
        ):
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            if index < 3:
                qpos = 0.05 - 0.003 * (index + 1)
                qvel = [-0.02, -0.02]
                finger_qpos = [qpos, qpos]
            elif index < 15:
                finger_qpos = [0.040, 0.041 - 0.001 * (index - 2)]
                qvel = [0.0, -0.02]
            else:
                finger_qpos = [0.040, 0.029]
                qvel = [0.0, 0.0]
            proprio[PROPRIO_SLICES["gripper_right_qpos"]] = finger_qpos
            proprio[PROPRIO_SLICES["gripper_right_qvel"]] = qvel
            adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(result["ok"], result)
        commands = np.asarray(
            [action[ACTION_SLICES["gripper_right"]] for action in actions]
        )
        unilateral_wait = (
            (np.abs(commands[:, 0]) > 0.5)
            & (np.abs(commands[:, 0]) <= 2.0)
            & np.isclose(np.abs(commands[:, 1]), 0.5)
        )
        self.assertTrue(bool(np.any(unilateral_wait)), commands.tolist())
        self.assertLessEqual(float(np.max(np.abs(commands))), 20.0 + 1e-7)
        self.assertGreater(result["peak_precontact_effort_n"], 0.5)
        self.assertLessEqual(result["peak_precontact_effort_n"], 2.0 + 1e-7)
        self.assertEqual(result["peak_confirmation_effort_n"], 20.0)

    def test_close_gripper_contact_compliance_does_not_reset_window(self) -> None:
        adapter, world = self._adapter_world()
        if not world.gripper_uses_effort("left"):
            self.skipTest("independent effort profile is required")
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["gripper_left_qpos"]] = [0.030, 0.050]
        proprio[PROPRIO_SLICES["gripper_left_qvel"]] = [0.0, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        registry = build_registry(adapter)
        ctx, result = self._ctx(world)
        actions = []

        for index, action in enumerate(
            registry["close_gripper"].fn(ctx, arm="left", timeout_s=3.5)
        ):
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            if index < 4:
                qpos = [0.030, 0.050 - 0.010 * (index + 1)]
                qvel = [0.0, -0.020]
            elif index < 12:
                qpos = [0.030, 0.008]
                qvel = [0.0, 0.0]
            else:
                # Matches the failed 15061 close: the first finger remained
                # far from its lower limit while bilateral preload compressed
                # both contacts beyond the 0.5 mm stationary tolerance.
                qpos = [0.0293, 0.0073]
                qvel = [-0.008, 0.0002]
            proprio[PROPRIO_SLICES["gripper_left_qpos"]] = qpos
            proprio[PROPRIO_SLICES["gripper_left_qvel"]] = qvel
            adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["assisted_grasp_window_completed"])
        self.assertGreaterEqual(
            result["confirmation_steps"],
            result["confirmation_steps_required"],
        )
        commands = np.asarray(
            [action[ACTION_SLICES["gripper_left"]] for action in actions]
        )
        self.assertLessEqual(float(np.max(np.abs(commands))), 20.0 + 1e-7)
        bilateral = np.flatnonzero(
            np.all(np.abs(commands) >= 1.0 - 1e-7, axis=1)
        )
        self.assertGreater(len(bilateral), 0)
        self.assertTrue(
            bool(np.all(np.abs(commands[bilateral[0] :]) >= 1.0 - 1e-7)),
            commands.tolist(),
        )
        self.assertGreaterEqual(result["confirmation_motion_restart_count"], 1)
        self.assertEqual(result["candidate_reset_count"], 0)

    def test_close_gripper_short_timeout_preserves_detected_blockage(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["gripper_left_qpos"]] = [0.05, 0.05]
        proprio[PROPRIO_SLICES["gripper_left_qvel"]] = [0.0, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        registry = build_registry(adapter)
        ctx, result = self._ctx(world)

        actions = list(
            registry["close_gripper"].fn(
                ctx,
                arm="left",
                timeout_s=0.3,
            )
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(
            result["failure_stage"],
            "bilateral_contact_confirmation",
        )
        self.assertTrue(result["bilateral_contact_ever_observed"])
        self.assertEqual(result["per_finger_blocked"], [True, True])
        self.assertEqual(
            result["per_finger_blocked_without_prior_travel"],
            [True, True],
        )
        self.assertEqual(result["action_steps"], 9)
        self.assertEqual(
            len(actions),
            11 if world.gripper_uses_effort("left") else 10,
        )
        self.assertTrue(result["close_keepalive_active"])
        self.assertTrue(result["close_intent_persists_until_explicit_open"])
        expected_close = (
            result["commanded_gripper_action"]
            if world.gripper_uses_effort("left")
            else [-1.0]
        )
        np.testing.assert_allclose(
            actions[-1][ACTION_SLICES["gripper_left"]],
            expected_close,
        )
        if world.gripper_uses_effort("left"):
            self.assertGreater(abs(float(expected_close[0])), 0.5)
            self.assertLessEqual(abs(float(expected_close[0])), 20.0)
            self.assertEqual(expected_close[0], expected_close[1])

        open_ctx, open_result = self._ctx(world)
        open_actions = list(
            registry["open_gripper"].fn(open_ctx, arm="left")
        )
        self.assertTrue(open_result["ok"])
        self.assertFalse(open_result["close_keepalive_active"])
        expected_open = (
            [3.0, 3.0]
            if world.gripper_uses_effort("left")
            else [1.0]
        )
        np.testing.assert_allclose(
            open_actions[0][ACTION_SLICES["gripper_left"]],
            expected_open,
        )

    def test_close_gripper_retry_cancellation_preserves_keepalive(self) -> None:
        class SkillCancelled(RuntimeError):
            pass

        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.038, 0.038]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = [0.0, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        initial_close = (
            [-1.0, -1.0]
            if world.gripper_uses_effort("right")
            else None
        )
        world.latch_gripper_close_keepalive("right", effort=initial_close)
        ctx, _result = self._ctx(world)

        def cancel_immediately(where=""):
            raise SkillCancelled(f"cancelled at {where}")

        ctx.raise_if_cancelled = cancel_immediately
        generator = build_registry(adapter)["close_gripper"].fn(
            ctx,
            arm="right",
            timeout_s=1.0,
        )
        with self.assertRaises(SkillCancelled):
            next(generator)

        self.assertTrue(world.gripper_close_keepalive_active("right"))
        held = adapter.record_action(np.zeros(ACTION_DIM, dtype=np.float32))
        expected = (
            [-1.0, -1.0]
            if world.gripper_uses_effort("right")
            else [-1.0]
        )
        np.testing.assert_allclose(
            held[ACTION_SLICES["gripper_right"]],
            expected,
        )

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
    def _drive_actions(adapter, world, generator):
        actions = []
        for action in generator:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            for side in ("left", "right"):
                action_idx = world.controller_action_idx(f"arm_{side}")
                proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                    action_idx
                ]
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})
        return actions

    @staticmethod
    def _drive_set_arm_dynamic(
        adapter,
        world,
        generator,
        *,
        momentum: float = 0.85,
        plateau_j6: float | None = None,
        joint_response_scale: dict[tuple[str, int], float] | None = None,
    ):
        """Run setarm against a delayed, proprio-only plant model.

        This deliberately does not touch a simulator.  It gives each yielded
        official action a first-order velocity response, so stale-waypoint
        regressions and entry handoff failures are observable in unit tests.
        """
        momentum = float(np.clip(momentum, 0.0, 0.99))
        q = {
            side: np.asarray(
                adapter.proprio_vector()[
                    PROPRIO_SLICES[f"arm_{side}_qpos"]
                ],
                dtype=np.float64,
            ).copy()
            for side in ("left", "right")
        }
        velocity = {
            side: np.asarray(
                adapter.proprio_vector()[
                    PROPRIO_SLICES[f"arm_{side}_qvel"]
                ],
                dtype=np.float64,
            ).copy()
            for side in ("left", "right")
        }
        actions = []
        trace = []
        for raw_action in generator:
            action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            trace.append(
                {
                    side: {
                        "q": q[side].copy(),
                        "command": action[
                            world.controller_action_idx(f"arm_{side}")
                        ].astype(np.float64),
                    }
                    for side in ("left", "right")
                }
            )
            proprio = adapter.proprio_vector().copy()
            for side in ("left", "right"):
                command = action[
                    world.controller_action_idx(f"arm_{side}")
                ].astype(np.float64)
                before = q[side].copy()
                acceleration = (command - before) * 30.0
                velocity[side] = (
                    momentum * velocity[side]
                    + (1.0 - momentum) * acceleration
                )
                displacement = velocity[side] / 30.0
                for index in range(ARM_DOF):
                    scale = 1.0 if joint_response_scale is None else float(
                        joint_response_scale.get((side, index), 1.0)
                    )
                    displacement[index] *= max(0.0, min(1.0, scale))
                q[side] = before + displacement
                velocity[side] = displacement * 30.0
                if plateau_j6 is not None:
                    # Simulate a static lower-limit plateau on J6.
                    if q[side][5] < float(plateau_j6):
                        q[side][5] = float(plateau_j6)
                        velocity[side][5] = 0.0
                if ARM_DOF == 8:
                    q[side][7] = 0.0
                    velocity[side][7] = 0.0
                proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = q[side]
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = velocity[side]
            adapter.update({"robot_r1::proprio": proprio})
        return actions, trace

    @staticmethod
    def _drive_trunk_actions(adapter, world, generator):
        actions = []
        trunk_action_idx = world.controller_action_idx("trunk")
        for action in generator:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["trunk_qpos"]] = action[trunk_action_idx]
            proprio[PROPRIO_SLICES["trunk_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})
        return actions

    @staticmethod
    def _sync_local_eef_proprio(proprio):
        state = local_robot_state(
            trunk_q=proprio[PROPRIO_SLICES["trunk_qpos"]],
            arm_left_q=proprio[PROPRIO_SLICES["arm_left_qpos"]],
            arm_right_q=proprio[PROPRIO_SLICES["arm_right_qpos"]],
            gripper_left_q=proprio[PROPRIO_SLICES["gripper_left_qpos"]],
            gripper_right_q=proprio[PROPRIO_SLICES["gripper_right_qpos"]],
        )
        for side in ("left", "right"):
            arm_q = proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]]
            position, quaternion = local_eef_pose(state, side, arm_q)
            proprio[PROPRIO_SLICES[f"eef_{side}_pos"]] = position
            proprio[PROPRIO_SLICES[f"eef_{side}_quat"]] = quaternion
        return proprio

    @classmethod
    def _drive_reset_body_actions(cls, adapter, world, generator):
        proprio = cls._sync_local_eef_proprio(
            adapter.proprio_vector().copy()
        )
        adapter.update({"robot_r1::proprio": proprio})
        actions = []
        trunk_action_idx = world.controller_action_idx("trunk")
        for action in generator:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["trunk_qpos"]] = action[trunk_action_idx]
            proprio[PROPRIO_SLICES["trunk_qvel"]] = 0.0
            for side in ("left", "right"):
                action_idx = world.controller_action_idx(f"arm_{side}")
                proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                    action_idx
                ]
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            proprio = cls._sync_local_eef_proprio(proprio)
            adapter.update({"robot_r1::proprio": proprio})
        return actions

    @classmethod
    def _drive_reset_body_actions_with_arm_lag(
        cls,
        adapter,
        world,
        generator,
        *,
        arm_response: float,
    ):
        """Apply trunk commands immediately but lag arm position targets."""
        response = float(np.clip(arm_response, 0.0, 1.0))
        proprio = cls._sync_local_eef_proprio(
            adapter.proprio_vector().copy()
        )
        adapter.update({"robot_r1::proprio": proprio})
        actions = []
        trunk_action_idx = world.controller_action_idx("trunk")
        for raw_action in generator:
            action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["trunk_qpos"]] = action[trunk_action_idx]
            proprio[PROPRIO_SLICES["trunk_qvel"]] = 0.0
            for side in ("left", "right"):
                action_idx = world.controller_action_idx(f"arm_{side}")
                before = np.asarray(
                    proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]],
                    dtype=np.float64,
                ).copy()
                command = np.asarray(action[action_idx], dtype=np.float64)
                after = before + response * (command - before)
                if ARM_DOF == 8:
                    after[7] = 0.0
                proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = after
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = (
                    after - before
                ) * 30.0
            proprio = cls._sync_local_eef_proprio(proprio)
            adapter.update({"robot_r1::proprio": proprio})
        return actions

    @staticmethod
    def _set_adjust_ready_observation(adapter):
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        arm_q = np.asarray(GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = arm_q
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            proprio[PROPRIO_SLICES[f"gripper_{side}_qpos"]] = [0.05, 0.05]
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [0.45, -0.4, 0.0, 0.0]
        camera_poses = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3,
            dtype=np.float32,
        )
        adapter.update(
            {
                "robot_r1::proprio": proprio,
                "robot_r1::cam_rel_poses": camera_poses,
            }
        )
        return arm_q, camera_poses

    @staticmethod
    def _set_move_point_ready_observation(
        adapter,
        *,
        depth=None,
        rgb=None,
        include_camera_pose=True,
    ):
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        arm_q = np.asarray(GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = arm_q
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            proprio[PROPRIO_SLICES[f"gripper_{side}_qpos"]] = [0.02, 0.02]
            proprio[PROPRIO_SLICES[f"gripper_{side}_qvel"]] = 0.0
        trunk_q = np.asarray([0.45, -0.4, 0.0, 0.0], dtype=np.float32)
        proprio[PROPRIO_SLICES["trunk_qpos"]] = trunk_q
        depth_frame = (
            np.full((12, 16), 0.2, dtype=np.float32)
            if depth is None
            else np.asarray(depth)
        )
        observation = {
            "robot_r1::proprio": proprio,
            "robot_r1::head::rgb": (
                np.zeros((12, 16, 3), dtype=np.uint8)
                if rgb is None
                else np.asarray(rgb)
            ),
            "robot_r1::head::depth_linear": depth_frame,
        }
        if include_camera_pose:
            state = official_v2_tools.local_robot_state(
                trunk_q=trunk_q,
                arm_left_q=arm_q,
                arm_right_q=arm_q,
                gripper_left_q=[0.02, 0.02],
                gripper_right_q=[0.02, 0.02],
            )
            left_eef, _left_quat = local_eef_pose(state, "left", arm_q)
            fixture_depth_m = 0.2
            head_pose = [
                float(left_eef[0]),
                float(left_eef[1]),
                float(left_eef[2] + fixture_depth_m),
                0.0,
                0.0,
                0.0,
                1.0,
            ]
            wrist_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            observation["robot_r1::cam_rel_poses"] = np.asarray(
                wrist_pose + wrist_pose + head_pose,
                dtype=np.float32,
            )
        adapter.update(observation)
        return arm_q

    def _capture_move_point(self, registry, world, session_id):
        ctx, result = self._ctx(world)
        actions = list(
            registry["capture_head_camera"].fn(
                ctx,
                session_id=session_id,
            )
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(actions), 1)
        return result

    def _write_exec_plan(
        self,
        *,
        world,
        run_root: str,
        session_id: str,
        plan_id: str,
        safe_q: np.ndarray,
        final_q: np.ndarray,
        active_arm: str = "right",
        alternate_final_q_by_arm: dict[str, np.ndarray] | None = None,
        record_payload: dict | None = None,
        trunk_qpos: np.ndarray | None = None,
    ):
        capture = SimpleNamespace(
            robot={
                "arm_left_qpos": world.arm_qpos_list("left"),
                "arm_left_qvel": world.arm_qvel_list("left"),
                "arm_right_qpos": world.arm_qpos_list("right"),
                "arm_right_qvel": world.arm_qvel_list("right"),
                "trunk_qpos": (
                    np.asarray(trunk_qpos, dtype=float).reshape(4).tolist()
                    if trunk_qpos is not None
                    else world.trunk_qpos().astype(float).tolist()
                ),
                "gripper_left_qpos": world.gripper_qpos_list("left"),
                "gripper_right_qpos": world.gripper_qpos_list("right"),
                "motion_epoch": world.motion_epoch(),
                "episode_id": world.episode_id(),
            }
        )
        compile_ctx, _ = self._ctx(world)
        selected_pose_ik_q = {
            active_arm: np.asarray(final_q, dtype=float).tolist()
        }
        for side, raw_q in (alternate_final_q_by_arm or {}).items():
            selected_pose_ik_q[side] = np.asarray(
                raw_q,
                dtype=float,
            ).tolist()
        compile_payload = {
            "recommended_arm": active_arm,
            "selected_pose_ik_q": selected_pose_ik_q,
            "candidates": [
                {
                    "arm": active_arm,
                    "meta": {
                        "planned_safe": {
                            "ok": True,
                            "active_back_m": 0.10,
                            "safe_q": np.asarray(
                                safe_q,
                                dtype=float,
                            ).tolist(),
                        }
                    },
                }
            ],
        }
        trajectory = _compile_joint_trajectory(
            ctx=compile_ctx,
            plan_id=plan_id,
            session_id=session_id,
            image_id="img_exec_fixture",
            capture=capture,
            payload=compile_payload,
        )
        plans_dir = Path(run_root) / session_id / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "plan_id": plan_id,
            "session_id": session_id,
            "tool": "plan_grasp_point_filter_rgbd_lite",
            "trajectory": trajectory,
        }
        if record_payload is not None:
            record["payload"] = record_payload
        (plans_dir / f"{plan_id}.json").write_text(
            json.dumps(record),
            encoding="utf-8",
        )
        return trajectory

    def test_registry_matches_exact_public_v2_surface(self) -> None:
        existing_tools = (
            V2_TOOL_NAMES[:4]
            + ("track_object_distance", "cut_object")
            + V2_TOOL_NAMES[4:]
        )
        surface_index = existing_tools.index("move_chassis_to_floor_point") + 1
        existing_tools = (
            existing_tools[:surface_index]
            + ("move_chassis_to_directly_facing_surface",)
            + existing_tools[surface_index:]
        )
        navigate_index = (
            existing_tools.index("move_chassis_to_directly_facing_surface") + 1
        )
        existing_tools = (
            existing_tools[:navigate_index]
            + ("navigate_to",)
            + existing_tools[navigate_index:]
        )
        move_point_index = existing_tools.index("move_point_to_point") + 1
        expected_tools = (
            existing_tools[:move_point_index]
            + ("move_tracked_point",)
            + existing_tools[move_point_index:]
            + (
                ("control_wrist_roll",)
                if WRIST_ROLL_TEST_TOOL_ENABLED
                else ()
            )
        )
        self.assertEqual(
            PUBLIC_TOOLS,
            expected_tools,
        )
        self.assertEqual(
            len(PUBLIC_TOOLS),
            32 + int(WRIST_ROLL_TEST_TOOL_ENABLED),
        )

        registry = build_registry(SimpleNamespace())

        self.assertEqual(tuple(registry), PUBLIC_TOOLS)
        self.assertTrue(
            all(
                spec.fn.__module__.startswith(
                    "behavior_interface_eval_test.tool.official_v2"
                )
                for spec in registry.values()
            )
        )
        self.assertEqual(tuple(PUBLIC_TOOL_FUNCTIONS), PUBLIC_TOOLS)
        self.assertTrue(
            all(
                inspect.isfunction(fn)
                and fn.__module__
                == "behavior_interface_eval_test.tool.official_v2.tools"
                for fn in PUBLIC_TOOL_FUNCTIONS.values()
            )
        )

    def test_installed_registry_rejects_legacy_same_name_overwrite(self) -> None:
        skills_module = SimpleNamespace()
        registry = install_profile(skills_module, SimpleNamespace())
        name = "adjust_right_eef_pose_in_wrist_frame"
        canonical = registry[name]
        legacy = SimpleNamespace(
            name=name,
            fn=lambda _ctx: None,
            description="simulator-backed legacy callable",
            params=[],
        )

        skills_module.SKILL_REGISTRY[name] = legacy
        skills_module.SKILL_REGISTRY.clear()

        self.assertIs(skills_module.SKILL_REGISTRY[name], canonical)
        self.assertEqual(registry.blocked_mutation_count, 2)

        skills_module.SKILL_REGISTRY = {name: legacy}
        restored = ensure_profile_installed(skills_module, registry)
        self.assertTrue(restored)
        self.assertIs(skills_module.SKILL_REGISTRY, registry)
        self.assertIs(skills_module.SKILL_REGISTRY[name], canonical)

    def test_capability_report_is_public_level(self) -> None:
        report = capability_report()

        self.assertEqual(report["tool_version"], "official_v2")
        self.assertEqual(report["registry_layer"], "Interface v2 public tools")
        self.assertEqual(tuple(report["tools"]), PUBLIC_TOOLS)
        self.assertEqual(
            report["counts"],
            {
                "supported": 8 + int(WRIST_ROLL_TEST_TOOL_ENABLED),
                "conditional": 19,
                "blocked": 5,
            },
        )
        self.assertEqual(
            report["feasibility_counts"],
            {
                "implemented_now": 27 + int(WRIST_ROLL_TEST_TOOL_ENABLED),
                "portable": 5,
            },
        )
        self.assertEqual(
            report["tools"]["exec_plan_pose"]["semantics"],
            "from-current action-only execution of a signed plan pose",
        )
        self.assertEqual(
            report["tools"]["capture_head_camera"]["semantics"],
            "observation-equivalent capture with the frozen v2 base-path HUD",
        )
        self.assertIn(
            "not collision truth",
            report["tools"]["capture_head_camera"]["constraints"],
        )
        self.assertEqual(
            report["tools"]["adjust_height"]["status"],
            "supported",
        )
        self.assertIn(
            "phase1/phase2",
            report["tools"]["adjust_height"]["semantics"],
        )
        self.assertEqual(
            report["tools"]["read_depth"]["semantics"],
            "exact frozen-capture depth lookup in meters",
        )
        self.assertEqual(
            report["tools"]["read_depth"]["observations"],
            ["*::depth_linear"],
        )
        for name in (
            "capture_left_wrist_camera",
            "capture_right_wrist_camera",
        ):
            self.assertEqual(
                report["tools"][name]["semantics"],
                "RGB-D capture with a local red grasp-volume overlay",
            )
            self.assertIn(
                "*::proprio",
                report["tools"][name]["observations"],
            )
        for name in (
            "adjust_left_eef_pose_in_head_frame",
            "adjust_right_eef_pose_in_head_frame",
            "adjust_left_eef_pose_in_wrist_frame",
            "adjust_right_eef_pose_in_wrist_frame",
        ):
            self.assertEqual(report["tools"][name]["status"], "conditional")
            self.assertEqual(
                report["tools"][name]["observations"],
                ["*::proprio", "*::cam_rel_poses"],
            )
        self.assertEqual(
            report["tools"]["move_point_to_point"]["status"],
            "conditional",
        )
        self.assertIn(
            "defaults to 0m",
            report["tools"]["move_point_to_point"]["constraints"],
        )
        self.assertEqual(
            report["tools"]["cut_object"]["semantics"],
            "single cutting-tool touch attempt verified by live tracked RGB-D point coincidence",
        )
        self.assertIn(
            "does not assert simulator contact",
            report["tools"]["cut_object"]["constraints"],
        )

    def test_exec_plan_pose_call_surface_is_observation_action_only(self) -> None:
        fn = PUBLIC_TOOL_FUNCTIONS["exec_plan_pose"]
        tree = ast.parse(inspect.getsource(fn))
        world_calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "ctx"
            and node.func.value.attr == "world"
        }

        self.assertEqual(
            world_calls,
            {
                "arm_qpos_list",
                "arm_qvel_list",
                "episode_id",
                "gripper_qpos_list",
                "hold_action",
                "make_action",
                "motion_epoch",
                "set_arm_pin_qpos",
                "set_trunk_pin_qpos",
                "trunk_qpos",
            },
        )
        source = inspect.getsource(fn)
        for forbidden in (
            "_r1pro_arm_fk_robot",
            "_solve_arm_ik",
            "collision",
            "ctx.world.env",
            "ctx.world.robot",
            "set_joint_positions",
        ):
            self.assertNotIn(forbidden, source)

    def test_legacy_web_dispatch_maps_to_public_names(self) -> None:
        cases = (
            ("capture", {}, "", "capture_head_camera"),
            (
                "capture_left_wrist_camera",
                {},
                "",
                "capture_left_wrist_camera",
            ),
            ("move_base_to_point", {}, "", "move_chassis_to_floor_point"),
            (
                "move_in_robot_coord",
                {"forward": 0.1, "spin": 2.0},
                "",
                "adjust_chassis",
            ),
            (
                "move_in_robot_coord",
                {"pitch": 0.0},
                "adjust_pitch",
                "adjust_pitch",
            ),
            (
                "move_in_robot_coord",
                {"upward": 0.0},
                "adjust_height",
                "adjust_height",
            ),
            ("face_to_point", {}, "", "spin_to_facing_point"),
            ("move_eef", {"gripper": "open"}, "", "open_gripper"),
            ("move_eef", {"gripper": "close"}, "", "close_gripper"),
            (
                "plan_move_eef",
                {"u": 500, "v": 500, "depth": 1.0},
                "",
                "plan_eef_translation_to_uvd_point",
            ),
            ("move_to_point_v3", {}, "", "move_to_reach_point"),
            (
                "mesure_shoulder_distance",
                {},
                "",
                "measure_shoulder_distance",
            ),
            (
                "plan_eef_v2",
                {"mode": "grasp_point_filter"},
                "",
                "plan_grasp_point_filter",
            ),
            (
                "plan_eef_v2",
                {"mode": "press_point"},
                "",
                "plan_press_point",
            ),
            (
                "plan_eef_rgbd_batch",
                {},
                "",
                "plan_grasp_point_filter_rgbd",
            ),
            (
                "plan_eef_rgbd_lite",
                {},
                "",
                "plan_grasp_point_filter_rgbd_lite",
            ),
            ("exec_eef_pose_v2", {}, "", "exec_plan_pose"),
            (
                "set_arm_to_grasp_position",
                {},
                "",
                "set_arm_to_grasp_position",
            ),
            ("reset_body", {}, "", "reset_body"),
        )
        for legacy, args, hint, expected in cases:
            with self.subTest(legacy=legacy, expected=expected):
                actual, _ = translate_submission(
                    legacy,
                    args,
                    public_hint=hint,
                )
                self.assertEqual(actual, expected)

    def test_nonpublic_legacy_routes_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "not in the public official_v2 surface",
        ):
            translate_submission(
                "plan_eef_v2",
                {"mode": "grasp_obj"},
            )
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "no public official_v2 equivalent",
        ):
            translate_submission(
                "move_eef",
                {"gripper": "keep", "forward": 1.0},
            )
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "no public official_v2 tool mapping",
        ):
            translate_submission("diag_reset_object_diff", {})

    def test_all_blocked_public_tools_fail_before_queueing(self) -> None:
        report = capability_report()
        blocked = [
            name
            for name, capability in report["tools"].items()
            if capability["status"] == "blocked"
        ]
        self.assertEqual(len(blocked), 5)
        for name in blocked:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    OfficialToolBoundaryError,
                    f"blocks {name}",
                ):
                    validate_submission(name, {})

    def test_adjust_height_accepts_finite_requests_for_lut_saturation(self) -> None:
        normalized = validate_submission("adjust_height", {"upward": -100.0})
        self.assertEqual(normalized["upward"], -100.0)
        for invalid in (float("nan"), float("inf"), -float("inf"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(OfficialToolBoundaryError):
                    validate_submission("adjust_height", {"upward": invalid})

    def test_adjust_height_http_route_passes_large_finite_request(self) -> None:
        class FakeServer:
            def __init__(self):
                self.submissions = []

            def submit_skill(self, name, args):
                self.submissions.append((name, dict(args)))
                return f"job-{len(self.submissions)}"

            def wait_for_skill_result(
                self,
                name,
                timeout_s=120.0,
                request_id=None,
            ):
                del timeout_s, request_id
                if name == "adjust_height":
                    return {"ok": True, "upward_m": -100.0}
                return {
                    "ok": True,
                    "image_id": "img_test",
                    "rgb_main": "/tmp/img_test.png",
                }

        app = Flask(__name__)

        def legacy_rejection():
            return jsonify({"ok": False, "error": "legacy limit"}), 400

        app.add_url_rule(
            "/api/v2/adjust_height",
            endpoint="api_v2_adjust_hight",
            view_func=legacy_rejection,
            methods=["POST"],
        )
        server = FakeServer()
        runtime = SimpleNamespace(server=server)
        install_official_adjust_height_route(app, runtime)

        response = app.test_client().post(
            "/api/v2/adjust_height",
            json={"session_id": "session", "upward": -100.0},
        )
        payload = response.get_json()

        self.assertEqual(response.status_code, 200, payload)
        self.assertTrue(payload["ok"])
        self.assertEqual(
            server.submissions[0],
            ("adjust_height", {"upward": -100.0}),
        )
        self.assertEqual(server.submissions[1][0], "capture_head_camera")

    def test_conditional_parameter_boundaries(self) -> None:
        normalized_move_point = validate_submission(
            "move_point_to_point",
            {
                "session_id": "session",
                "image_id": "img_0001",
                "points": [[100, 200], {"u": 300, "v": 400}],
            },
        )
        self.assertEqual(normalized_move_point["above_target_point_m"], 0.0)
        self.assertEqual(
            normalized_move_point["points"],
            [{"u": 100.0, "v": 200.0}, {"u": 300.0, "v": 400.0}],
        )
        normalized_move_point = validate_submission(
            "move_point_to_point",
            {
                "session_id": "session",
                "image_id": "img_0001",
                "points": [[100, 200], [300, 400]],
                "above_target_point_m": "0.05",
            },
        )
        self.assertEqual(normalized_move_point["above_target_point_m"], 0.05)
        for invalid_above in (-0.01, float("nan"), float("inf"), True, 0.751):
            with self.subTest(invalid_above=invalid_above):
                with self.assertRaises(OfficialToolBoundaryError):
                    validate_submission(
                        "move_point_to_point",
                        {
                            "session_id": "session",
                            "image_id": "img_0001",
                            "points": [[100, 200], [300, 400]],
                            "above_target_point_m": invalid_above,
                        },
                    )
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "exactly two points",
        ):
            validate_submission(
                "move_point_to_point",
                {
                    "session_id": "session",
                    "image_id": "img_0001",
                    "points": [[100, 200]],
                },
            )
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "unsupported arguments",
        ):
            validate_submission(
                "move_point_to_point",
                {
                    "session_id": "session",
                    "image_id": "img_0001",
                    "points": [[100, 200], [300, 400]],
                    "object_id": 7,
                },
            )
        normalized_adjust = validate_submission(
            "adjust_right_eef_pose_in_wrist_frame",
            {
                "forward": "0.03",
                "yaw": "5",
                "max_steps": "120",
                "timeout_s": "30",
            },
        )
        self.assertEqual(normalized_adjust["forward"], 0.03)
        self.assertEqual(normalized_adjust["yaw"], 5.0)
        self.assertEqual(normalized_adjust["max_steps"], 120)
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "max_steps must be an integer",
        ):
            validate_submission(
                "adjust_left_eef_pose_in_head_frame",
                {"max_steps": 1.5},
            )
        validate_submission(
            "move_chassis_to_floor_point",
            {
                "session_id": "session",
                "image_id": "image",
                "u": 500,
                "v": 500,
            },
        )
        normalized_depth = validate_submission(
            "read_depth",
            {
                "session_id": "session",
                "image_id": "img_0001",
                "u": "250",
                "v": 750,
            },
        )
        self.assertEqual(normalized_depth["u"], 250.0)
        self.assertEqual(normalized_depth["v"], 750.0)
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "image_id is required",
        ):
            validate_submission(
                "read_depth",
                {"session_id": "session", "u": 500, "v": 500},
            )
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "u must be in 0..1000",
        ):
            validate_submission(
                "read_depth",
                {
                    "session_id": "session",
                    "image_id": "img_0001",
                    "u": 1001,
                    "v": 500,
                },
            )
        validate_submission(
            "measure_shoulder_distance",
            {
                "session_id": "session",
                "image_id": "image",
                "u": 500,
                "v": 500,
            },
        )
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "cannot resolve object_name",
        ):
            validate_submission(
                "measure_shoulder_distance",
                {"session_id": "session", "object_name": "mousetrap.n.01_1"},
            )
        normalized = validate_submission(
            "move_to_reach_point",
            {
                "session_id": "session",
                "image_id": "image",
                "u": 500,
                "v": 500,
                "reach": 0.6,
                "keep_ori_arm": "both",
            },
        )
        self.assertEqual(normalized["keep_ori_arm"], "both")
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "reach must be",
        ):
            validate_submission(
                "move_to_reach_point",
                {
                    "session_id": "session",
                    "image_id": "image",
                    "u": 500,
                    "v": 500,
                    "reach": 1.2,
                },
            )
        for keep_ori_arm in ("left", "right", "both"):
            normalized = validate_submission(
                "reset_body",
                {"keep_ori_arm": keep_ori_arm},
            )
            self.assertEqual(normalized["keep_ori_arm"], keep_ori_arm)
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "keep_ori_arm must be",
        ):
            validate_submission("reset_body", {"keep_ori_arm": "center"})
        normalized = validate_submission(
            "plan_grasp_point_filter_rgbd_lite",
            {
                "session_id": "session",
                "image_id": "image",
                "u": 500,
                "v": 500,
                "plan_arm": "any",
            },
        )
        self.assertEqual(normalized["plan_arm"], "any")
        normalized = validate_submission(
            "exec_plan_pose",
            {
                "session_id": "session",
                "plan_id": "plan_0001",
                "arm": "right",
                "back_m": 0.1,
                "stop_after_safe": "false",
                "reset_tool_roll_at_start": "true",
            },
        )
        self.assertEqual(normalized["arm"], "right")
        self.assertFalse(normalized["stop_after_safe"])
        self.assertTrue(normalized["reset_tool_roll_at_start"])
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "plan_arm must be",
        ):
            validate_submission(
                "plan_grasp_point_filter_rgbd_lite",
                {
                    "session_id": "session",
                    "image_id": "image",
                    "u": 500,
                    "v": 500,
                    "plan_arm": "both",
                },
            )

    def test_head_adjust_translation_families_are_explicit_and_exclusive(self) -> None:
        for name in (
            "adjust_left_eef_pose_in_head_frame",
            "adjust_right_eef_pose_in_head_frame",
        ):
            with self.subTest(name=name, frame="robot_base"):
                base = validate_submission(
                    name,
                    {"x": "0.03", "y": -0.02, "z": 0.01},
                )
                self.assertEqual(base["x"], 0.03)
                self.assertEqual(base["y"], -0.02)
                self.assertEqual(base["z"], 0.01)
                self.assertNotIn("forward", base)
                self.assertNotIn("leftward", base)
                self.assertNotIn("upward", base)

            with self.subTest(name=name, frame="head_camera"):
                camera = validate_submission(
                    name,
                    {"forward": 0.03, "leftward": -0.01},
                )
                self.assertEqual(camera["forward"], 0.03)
                self.assertEqual(camera["leftward"], -0.01)
                self.assertNotIn("x", camera)
                self.assertNotIn("y", camera)
                self.assertNotIn("z", camera)

            with self.subTest(name=name, frame="mixed_even_when_zero"):
                with self.assertRaisesRegex(
                    OfficialToolBoundaryError,
                    "mutually exclusive",
                ):
                    validate_submission(name, {"x": 0.0, "forward": 0.0})

            with self.subTest(name=name, unsupported=True):
                with self.assertRaisesRegex(
                    OfficialToolBoundaryError,
                    "unsupported arguments",
                ):
                    validate_submission(name, {"world_x": 0.1})

    def test_head_adjust_registry_describes_both_coordinate_families(self) -> None:
        spec = build_registry(None)["adjust_right_eef_pose_in_head_frame"]
        params = {item["name"]: item for item in spec.params}
        self.assertTrue(
            {"x", "y", "z", "forward", "leftward", "upward"}
            <= set(params)
        )
        self.assertEqual(params["x"]["coordinate_frame"], "robot_base")
        self.assertEqual(params["y"]["positive_direction"], "chassis_left")
        self.assertEqual(
            params["leftward"]["coordinate_frame"],
            "starting_head_camera",
        )
        self.assertEqual(
            params["x"]["mutually_exclusive_with"],
            "head_camera_forward_leftward_upward",
        )
        self.assertIn("same positive-left sign", params["y"]["description"])
        self.assertIn("never mix", spec.description)

    def test_supported_public_actions_emit_profile_actions_and_public_results(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)

        for name, kwargs in (
            ("adjust_chassis", {"forward": 0.0}),
            ("adjust_pitch", {"degree": 0.0}),
            ("adjust_height", {"upward": 0.0}),
            ("spin_to_facing_point", {"u": 500, "v": 500}),
            ("open_gripper", {"arm": "left"}),
            ("close_gripper", {"arm": "right"}),
            ("set_arm_to_grasp_position", {"arm": "right"}),
            ("reset_body", {}),
        ):
            with self.subTest(name=name):
                ctx, result = self._ctx(world)
                actions = list(registry[name].fn(ctx, **kwargs))
                self.assertGreater(len(actions), 0)
                for action in actions:
                    self.assertEqual(np.asarray(action).shape, (ACTION_DIM,))
                self.assertEqual(result["tool"], name)
                self.assertEqual(result["tool_version"], "official_v2")
                if name == "set_arm_to_grasp_position":
                    self.assertEqual(
                        len(result["target_qpos"]["right"]),
                        ARM_DOF,
                    )

    def test_adjust_chassis_large_forward_recovers_from_loaded_wheels(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        ctx, result = self._ctx(world)
        generator = registry["adjust_chassis"].fn(
            ctx,
            forward=10.0,
            timeout_s=10.0,
        )
        base_idx = world.controller_action_idx("base")
        actions = []
        forward_motion_m = 0.0
        reverse_motion_m = 0.0
        loaded = False
        unloaded = False

        for action in generator:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            self.assertEqual(action.shape, (ACTION_DIM,))
            physical_vx = float(action[base_idx][0]) * 0.75
            if physical_vx > 1e-9:
                if forward_motion_m < 0.08:
                    observed_vx = physical_vx
                    forward_motion_m += observed_vx / 30.0
                else:
                    loaded = True
                    observed_vx = 0.0
                observed_wz = 0.0
            elif physical_vx < -1e-9:
                observed_vx = (
                    physical_vx if abs(physical_vx) >= 0.15 else 0.0
                )
                if observed_vx < 0.0:
                    reverse_motion_m += abs(observed_vx) / 30.0
                    if reverse_motion_m >= 0.025:
                        unloaded = True
                observed_wz = 0.08 if loaded and not unloaded else 0.0
            else:
                observed_vx = 0.025 if loaded and not unloaded else 0.0
                observed_wz = 0.08 if loaded and not unloaded else 0.0
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["base_qvel"]] = [
                observed_vx,
                0.0,
                observed_wz,
            ]
            adapter.update({"robot_r1::proprio": proprio})
            self.assertLess(len(actions), 300)

        self.assertTrue(result["ok"])
        self.assertTrue(result["obstacle_limited"])
        self.assertEqual(result["obstacle_stop_reason"], "base_qvel_stall")
        self.assertTrue(result["recovery_attempted"])
        self.assertTrue(result["recovery_ok"])
        self.assertGreaterEqual(result["retreat_speed_escalations"], 2)
        self.assertGreaterEqual(result["retreat_speed_max_mps"], 0.15)
        self.assertGreaterEqual(
            result["retreat_m"] + 0.002,
            result["retreat_target_m"],
        )
        self.assertGreater(result["actual"]["forward_m"], 0.0)
        self.assertLess(result["actual"]["forward_m"], 0.10)
        self.assertEqual(
            result["verification"],
            "direct_evaluator_base_qvel_integration",
        )
        self.assertFalse(result["global_base_pose_truth"])
        self.assertFalse(result["dynamic_collision_truth"])
        self.assertEqual(result["forbidden_observation_reads"], [])
        self.assertTrue(np.allclose(actions[-1][base_idx], 0.0))

    def test_adjust_chassis_tracks_left_positive_translation(self) -> None:
        for requested in (0.12, -0.12):
            with self.subTest(requested=requested):
                adapter, world = self._adapter_world()
                registry = build_registry(adapter)
                ctx, result = self._ctx(world)
                base_idx = world.controller_action_idx("base")
                actions = []

                for action in registry["adjust_chassis"].fn(
                    ctx,
                    translation=requested,
                    timeout_s=8.0,
                ):
                    action = np.asarray(action, dtype=np.float32).reshape(-1)
                    actions.append(action.copy())
                    physical_vy = float(action[base_idx][1]) * 0.75
                    proprio = adapter.proprio_vector().copy()
                    proprio[PROPRIO_SLICES["base_qvel"]] = [
                        0.0,
                        physical_vy,
                        0.0,
                    ]
                    adapter.update({"robot_r1::proprio": proprio})

                lateral_actions = [
                    float(action[base_idx][1])
                    for action in actions
                    if abs(float(action[base_idx][1])) > 1e-9
                ]
                self.assertTrue(result["ok"])
                self.assertTrue(result["translation_ok"])
                self.assertTrue(lateral_actions)
                self.assertGreater(lateral_actions[0] * requested, 0.0)
                self.assertGreater(
                    result["actual"]["translation_m"] * requested,
                    0.0,
                )
                self.assertTrue(
                    all(abs(float(action[base_idx][0])) <= 1e-9 for action in actions)
                )
                self.assertTrue(
                    all(abs(float(action[base_idx][2])) <= 1e-9 for action in actions)
                )
                self.assertTrue(np.allclose(actions[-1][base_idx], 0.0))

    def test_adjust_chassis_forward_left_is_one_straight_vector(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)
        base_idx = world.controller_action_idx("base")
        diagonal = []

        for action in build_registry(adapter)["adjust_chassis"].fn(
            ctx,
            forward=0.12,
            translation=0.09,
            timeout_s=8.0,
        ):
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            physical_xy = action[base_idx][:2].astype(np.float64) * 0.75
            if float(np.linalg.norm(physical_xy)) > 1e-9:
                diagonal.append(physical_xy.copy())
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["base_qvel"]] = [
                float(physical_xy[0]),
                float(physical_xy[1]),
                0.0,
            ]
            adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(result["ok"])
        self.assertTrue(result["linear_target_reached"])
        self.assertEqual(
            result["linear_control"],
            "simultaneous_robot_frame_xy",
        )
        self.assertTrue(diagonal)
        self.assertTrue(
            all(abs(command[0]) > 1e-9 and abs(command[1]) > 1e-9
                for command in diagonal)
        )
        self.assertTrue(
            all(
                math.isclose(command[1] / command[0], 0.75, abs_tol=1e-6)
                for command in diagonal
            )
        )
        self.assertLessEqual(
            max(float(np.linalg.norm(command)) for command in diagonal),
            0.5 + 1e-6,
        )
        self.assertLessEqual(
            abs(result["actual"]["forward_m"] - 0.12),
            ADJUST_CHASSIS_FORWARD_TOL_M,
        )
        self.assertLessEqual(
            abs(result["actual"]["translation_m"] - 0.09),
            ADJUST_CHASSIS_FORWARD_TOL_M,
        )

    def test_floor_point_moves_one_direct_xy_vector_without_pre_spin(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)
        base_idx = world.controller_action_idx("base")
        capture = SimpleNamespace()
        target_robot = np.asarray(
            [
                BASE_FRONT_OFFSET_M + 0.16,
                PATH_Y_CENTER_M + 0.12,
                0.0,
            ],
            dtype=np.float64,
        )
        diagonal = []

        with (
            mock.patch.object(
                official_v2_tools,
                "_load_frozen_capture",
                return_value=capture,
            ),
            mock.patch.object(
                official_v2_tools,
                "_point_from_relative_uv",
                return_value=(
                    target_robot.copy(),
                    {"source": "frozen_evaluator_depth"},
                ),
            ),
            mock.patch.object(
                official_v2_tools,
                "_surface_normal_from_relative_uv",
                return_value=(
                    np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                    {"normal_abs_z": 1.0},
                ),
            ),
        ):
            for action in build_registry(adapter)[
                "move_chassis_to_floor_point"
            ].fn(
                ctx,
                session_id="test-session",
                image_id="img_test",
                u=500,
                v=500,
                nav_timeout_s=8.0,
                pos_tol_m=0.05,
            ):
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                self.assertEqual(action.shape, (ACTION_DIM,))
                physical_base = action[base_idx].astype(np.float64) * np.asarray(
                    [0.75, 0.75, 1.0],
                    dtype=np.float64,
                )
                if float(np.linalg.norm(physical_base[:2])) > 1e-9:
                    diagonal.append(physical_base.copy())
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["base_qvel"]] = physical_base
                adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["tool"], "move_chassis_to_floor_point")
        self.assertEqual(result["tool_version"], "official_v2")
        self.assertEqual(result["exec_order"], "direct_xy_translation")
        self.assertEqual(
            result["linear_control"],
            "simultaneous_robot_frame_xy",
        )
        self.assertEqual(result["spin_deg"], 0.0)
        self.assertAlmostEqual(result["forward_m"], 0.16)
        self.assertAlmostEqual(result["translation_m"], 0.12)
        self.assertEqual(
            result["target_reference"],
            "base_front_path_start_midpoint",
        )
        self.assertEqual(
            result["front_reference_robot_xy_m"],
            [BASE_FRONT_OFFSET_M, PATH_Y_CENTER_M],
        )
        self.assertAlmostEqual(result["target_distance_m"], 0.20)
        self.assertAlmostEqual(
            result["clicked_point_distance_from_base_center_m"],
            float(np.linalg.norm(target_robot[:2])),
        )
        self.assertAlmostEqual(result["linear_tolerance_m"], 0.12)
        self.assertTrue(result["linear_target_reached"])
        self.assertFalse(result["dynamic_collision_truth"])
        self.assertFalse(result["global_base_pose_truth"])
        self.assertEqual(
            result["arrival_verification"],
            "evaluator_base_qvel_proprioception",
        )
        self.assertTrue(diagonal)
        self.assertTrue(
            all(
                abs(command[0]) > 1e-9
                and abs(command[1]) > 1e-9
                and abs(command[2]) <= 1e-9
                for command in diagonal
            )
        )
        self.assertTrue(
            all(
                math.isclose(command[1] / command[0], 0.75, abs_tol=1e-6)
                for command in diagonal
            )
        )

    def test_surface_facing_geometry_is_order_invariant_and_camera_facing(self) -> None:
        points = np.asarray(
            [
                [1.0, -0.20, 0.20],
                [1.0, 0.20, 0.20],
                [1.0, 0.00, 0.60],
            ],
            dtype=np.float64,
        )
        first = solve_surface_facing_target(points, [1.0, 0.0, 0.0])
        reversed_order = solve_surface_facing_target(
            points[[0, 2, 1]], [1.0, 0.0, 0.0]
        )

        np.testing.assert_allclose(first.normal_toward_robot, [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            reversed_order.normal_toward_robot,
            first.normal_toward_robot,
        )
        np.testing.assert_allclose(first.stand_point_robot_base_m, [0.2, 0.0, 0.0])
        self.assertAlmostEqual(first.standoff_m, 0.8)
        np.testing.assert_allclose(
            first.desired_chassis_forward_robot_base,
            [1.0, 0.0, 0.0],
        )
        self.assertLess(first.camera_normal_dot, 0.0)
        self.assertAlmostEqual(first.desired_spin_deg, 0.0)

    def test_surface_facing_standoff_uses_3d_normal_before_ground_projection(self) -> None:
        centroid_seed = np.asarray([1.0, 0.1, 0.8], dtype=np.float64)
        edge_a = np.asarray([0.0, 0.2, 0.0], dtype=np.float64)
        edge_b = np.asarray([0.12, 0.0, -0.16], dtype=np.float64)
        points = np.stack(
            [centroid_seed, centroid_seed + edge_a, centroid_seed + edge_b]
        )
        solution = solve_surface_facing_target(points, [1.0, 0.0, 0.0])

        self.assertLess(float(solution.normal_toward_robot[0]), 0.0)
        np.testing.assert_allclose(
            solution.stand_point_robot_base_m[:2],
            solution.centroid_robot_base_m[:2]
            + SURFACE_FACING_STANDOFF_M * solution.normal_toward_robot[:2],
            atol=1e-12,
        )
        self.assertEqual(float(solution.stand_point_robot_base_m[2]), 0.0)
        self.assertAlmostEqual(
            float(
                np.linalg.norm(
                    solution.stand_point_robot_base_m[:2]
                    - solution.centroid_robot_base_m[:2]
                )
            ),
            0.64,
            places=12,
        )

    def test_surface_facing_rejects_degenerate_or_horizontal_surface(self) -> None:
        with self.assertRaisesRegex(ValueError, "degenerate|collinear"):
            solve_surface_facing_target(
                [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 0.2, 0.0]],
                [1.0, 0.0, 0.0],
            )
        with self.assertRaisesRegex(ValueError, "horizontal"):
            solve_surface_facing_target(
                [[0.0, 0.0, 0.5], [0.2, 0.0, 0.5], [0.0, 0.2, 0.5]],
                [0.0, 0.0, -1.0],
            )

    def test_surface_facing_rebases_capture_points_through_policy_odometry(self) -> None:
        rebased = rebase_robot_points(
            [[0.5, 0.0, 0.2]],
            source_base_position_policy_m=[1.0, 2.0, 0.0],
            source_base_yaw_rad=math.pi / 2.0,
            target_base_position_policy_m=[1.0, 1.5, 0.0],
            target_base_yaw_rad=0.0,
        )
        np.testing.assert_allclose(rebased, [[0.0, 1.0, 0.2]], atol=1e-12)

    def test_surface_facing_capture_backprojects_exact_frozen_head_rgbd(self) -> None:
        half_sqrt = math.sqrt(0.5)
        capture = official_v2_tools._FrozenCapture(
            session_id="session",
            image_id="img_surface",
            role="head",
            depth=np.ones((11, 11), dtype=np.float32),
            rgb=np.zeros((11, 11, 3), dtype=np.uint8),
            camera={
                "image_width": 11,
                "image_height": 11,
                "fx": 10.0,
                "fy": 10.0,
                "cx": 5.5,
                "cy": 5.5,
                "robot_relative_pose": {
                    "pos": [0.0, 0.0, 0.8],
                    "quat": [0.0, -half_sqrt, 0.0, half_sqrt],
                },
            },
            robot={},
            evaluator_sequence=3,
        )
        solution, metadata = official_v2_tools._surface_facing_capture_geometry(
            capture,
            [
                {"u": 200.0, "v": 250.0},
                {"u": 800.0, "v": 250.0},
                {"u": 500.0, "v": 750.0},
            ],
        )

        self.assertEqual(len(metadata), 3)
        np.testing.assert_allclose(solution.points_robot_base_m[:, 0], 1.0)
        np.testing.assert_allclose(solution.normal_toward_robot, [-1.0, 0.0, 0.0])
        self.assertTrue(all(item["depth_m"] == 1.0 for item in metadata))

    def test_surface_facing_rejects_incomplete_or_invalid_frozen_rgbd(self) -> None:
        camera = {
            "image_width": 11,
            "image_height": 11,
            "fx": 10.0,
            "fy": 10.0,
            "cx": 5.5,
            "cy": 5.5,
            "robot_relative_pose": {
                "pos": [0.0, 0.0, 0.8],
                "quat": [0.0, 0.0, 0.0, 1.0],
            },
        }
        points = [
            {"u": 200.0, "v": 250.0},
            {"u": 800.0, "v": 250.0},
            {"u": 500.0, "v": 750.0},
        ]

        def capture(*, depth, rgb, camera_metadata):
            return official_v2_tools._FrozenCapture(
                session_id="session",
                image_id="img_surface",
                role="head",
                depth=depth,
                rgb=rgb,
                camera=camera_metadata,
                robot={},
                evaluator_sequence=3,
            )

        with self.assertRaisesRegex(ValueError, "resolutions do not match"):
            official_v2_tools._surface_facing_capture_geometry(
                capture(
                    depth=np.ones((11, 11), dtype=np.float32),
                    rgb=np.zeros((10, 11, 3), dtype=np.uint8),
                    camera_metadata=camera,
                ),
                points,
            )
        missing_pose = dict(camera)
        missing_pose.pop("robot_relative_pose")
        with self.assertRaisesRegex(ValueError, "camera-relative pose"):
            official_v2_tools._surface_facing_capture_geometry(
                capture(
                    depth=np.ones((11, 11), dtype=np.float32),
                    rgb=np.zeros((11, 11, 3), dtype=np.uint8),
                    camera_metadata=missing_pose,
                ),
                points,
            )
        with self.assertRaisesRegex(ValueError, "no valid depth"):
            official_v2_tools._surface_facing_capture_geometry(
                capture(
                    depth=np.zeros((11, 11), dtype=np.float32),
                    rgb=np.zeros((11, 11, 3), dtype=np.uint8),
                    camera_metadata=camera,
                ),
                points,
            )

    def test_surface_facing_contract_and_registry_metadata(self) -> None:
        normalized = validate_submission(
            "move_chassis_to_directly_facing_surface",
            {
                "session_id": "surface-session",
                "image_id": "img_0001",
                "points": [[100, 200], {"u": 400, "v": 500}, [700, 800]],
            },
        )
        self.assertEqual(len(normalized["points"]), 3)
        self.assertEqual(normalized["nav_timeout_s"], 120.0)
        self.assertEqual(normalized["pos_tol_m"], 0.04)
        with self.assertRaisesRegex(OfficialToolBoundaryError, "exactly three"):
            validate_submission(
                "move_chassis_to_directly_facing_surface",
                {
                    "session_id": "surface-session",
                    "image_id": "img_0001",
                    "points": [[100, 200], [700, 800]],
                },
            )
        with self.assertRaisesRegex(OfficialToolBoundaryError, "unsupported fields"):
            validate_submission(
                "move_chassis_to_directly_facing_surface",
                {
                    "session_id": "surface-session",
                    "image_id": "img_0001",
                    "points": [
                        {"u": 100, "v": 200, "plan_arm": "any"},
                        [400, 500],
                        [700, 800],
                    ],
                },
            )
        with self.assertRaisesRegex(OfficialToolBoundaryError, "0..1000"):
            validate_submission(
                "move_chassis_to_directly_facing_surface",
                {
                    "session_id": "surface-session",
                    "image_id": "img_0001",
                    "points": [[-1, 200], [400, 500], [700, 800]],
                },
            )
        spec = build_registry(None)["move_chassis_to_directly_facing_surface"]
        parameters = {item["name"]: item for item in spec.params}
        self.assertEqual(parameters["image_id"]["widget"], "image")
        self.assertEqual(parameters["points"]["type"], "array")
        self.assertEqual(parameters["points"]["widget"], "multi_uv_arm")
        self.assertEqual(parameters["points"]["min_points"], 3)
        self.assertEqual(parameters["points"]["max_points"], 3)

    def test_surface_facing_executes_xy_then_yaw_with_complete_actions(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)
        base_idx = world.controller_action_idx("base")
        normal = np.asarray([-math.sqrt(0.5), -math.sqrt(0.5), 0.0])
        forward = -normal
        solution = SurfaceFacingSolution(
            points_robot_base_m=np.asarray(
                [[0.5, 0.4, 0.4], [0.4, 0.5, 0.4], [0.5, 0.4, 0.6]],
                dtype=np.float64,
            ),
            centroid_robot_base_m=np.asarray([0.5, 0.45, 0.5]),
            normal_toward_robot=normal,
            camera_forward_robot_base=np.asarray([1.0, 0.0, 0.0]),
            stand_point_robot_base_m=np.asarray([0.15, 0.10, 0.0]),
            desired_chassis_forward_robot_base=forward,
            desired_spin_deg=45.0,
            camera_normal_dot=float(normal[0]),
            longest_edge_m=0.2,
            normalized_double_area=0.5,
            standoff_m=SURFACE_FACING_STANDOFF_M,
        )
        capture = SimpleNamespace(
            robot={
                "episode_id": world.episode_id(),
                "base_pose": {
                    "pos": [0.0, 0.0, 0.0],
                    "quat": [0.0, 0.0, 0.0, 1.0],
                },
            }
        )
        actions = []
        with (
            mock.patch.object(
                official_v2_tools,
                "_load_frozen_capture",
                return_value=capture,
            ),
            mock.patch.object(
                official_v2_tools,
                "_surface_facing_capture_geometry",
                return_value=(solution, [{}, {}, {}]),
            ),
        ):
            generator = build_registry(adapter)[
                "move_chassis_to_directly_facing_surface"
            ].fn(
                ctx,
                session_id="surface-session",
                image_id="img_surface",
                points=[
                    {"u": 100, "v": 200},
                    {"u": 400, "v": 500},
                    {"u": 700, "v": 800},
                ],
                nav_timeout_s=12.0,
                pos_tol_m=0.02,
            )
            for raw_action in generator:
                action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
                self.assertEqual(action.shape, (ACTION_DIM,))
                self.assertTrue(np.all(np.isfinite(action)))
                if ARM_DOF == 8:
                    self.assertEqual(float(action[ACTION_SLICES["arm_left"]][-1]), 0.0)
                    self.assertEqual(float(action[ACTION_SLICES["arm_right"]][-1]), 0.0)
                actions.append(action.copy())
                physical_base = action[base_idx] * np.asarray([0.75, 0.75, 1.0])
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["base_qvel"]] = physical_base
                adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["tool"], "move_chassis_to_directly_facing_surface")
        self.assertTrue(result["translation_phase"]["linear_target_reached"])
        self.assertTrue(result["rotation_phase"]["ok"])
        self.assertTrue(result["final_position_ok"])
        self.assertTrue(result["final_heading_ok"])
        self.assertLessEqual(result["final_position_error_m"], 0.02)
        self.assertLessEqual(result["final_heading_error_deg"], 0.7)
        moving = [action[base_idx] for action in actions if np.linalg.norm(action[base_idx]) > 1e-9]
        self.assertTrue(any(np.linalg.norm(command[:2]) > 0.0 for command in moving))
        self.assertTrue(any(abs(float(command[2])) > 0.0 for command in moving))
        self.assertTrue(
            all(
                not (
                    np.linalg.norm(command[:2]) > 1e-9
                    and abs(float(command[2])) > 1e-9
                )
                for command in moving
            )
        )

    def test_surface_facing_http_route_and_ui_use_exactly_three_clicks(self) -> None:
        class FakeServer:
            def __init__(self):
                self.submissions = []

            def submit_skill(self, name, args):
                self.submissions.append((name, dict(args)))
                return "job-surface"

            def wait_for_skill_result(self, name, timeout_s=120.0, request_id=None):
                del timeout_s, request_id
                return {"ok": True, "tool": name, "action_steps": 42}

        app = Flask(__name__)

        @app.get("/api/v2/tools", endpoint="api_v2_tools")
        def base_tools():
            return jsonify({"tools": []})

        server = FakeServer()
        runtime = SimpleNamespace(server=server)
        install_official_surface_facing_route(app, runtime)
        client = app.test_client()
        tools_payload = client.get("/api/v2/tools").get_json()
        metadata = tools_payload["tools"][0]
        point_arg = next(item for item in metadata["args"] if item["name"] == "points")
        self.assertEqual(metadata["name"], "move_chassis_to_directly_facing_surface")
        self.assertEqual(point_arg["widget"], "multi_uv_arm")
        self.assertEqual(point_arg["min_points"], 3)
        self.assertEqual(point_arg["max_points"], 3)

        response = client.post(
            "/api/v2/move_chassis_to_directly_facing_surface",
            json={
                "session_id": "surface-session",
                "image_id": "img_0001",
                "points": [
                    {"u": 100, "v": 200},
                    {"u": 400, "v": 500},
                    {"u": 700, "v": 800},
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertTrue(response.get_json()["ok"])
        self.assertEqual(server.submissions[0][0], metadata["name"])

    def test_surface_facing_rejects_controller_false_success_without_arrival(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)
        solution = SurfaceFacingSolution(
            points_robot_base_m=np.asarray(
                [[0.5, 0.4, 0.4], [0.4, 0.5, 0.4], [0.5, 0.4, 0.6]],
                dtype=np.float64,
            ),
            centroid_robot_base_m=np.asarray([0.5, 0.45, 0.5]),
            normal_toward_robot=np.asarray([-1.0, 0.0, 0.0]),
            camera_forward_robot_base=np.asarray([1.0, 0.0, 0.0]),
            stand_point_robot_base_m=np.asarray([0.15, 0.10, 0.0]),
            desired_chassis_forward_robot_base=np.asarray([1.0, 0.0, 0.0]),
            desired_spin_deg=0.0,
            camera_normal_dot=-1.0,
            longest_edge_m=0.2,
            normalized_double_area=0.5,
            standoff_m=SURFACE_FACING_STANDOFF_M,
        )
        capture = SimpleNamespace(
            robot={
                "episode_id": world.episode_id(),
                "base_pose": {
                    "pos": [0.0, 0.0, 0.0],
                    "quat": [0.0, 0.0, 0.0, 1.0],
                },
            }
        )

        def false_success_controller(*_args, **_kwargs):
            if False:
                yield None
            return {
                "ok": True,
                "error": None,
                "failure_stage": None,
                "linear_target_reached": True,
                "linear_yaw_drift_deg": 0.0,
                "actual": {"spin_deg": 0.0},
            }

        with (
            mock.patch.object(
                official_v2_tools,
                "_load_frozen_capture",
                return_value=capture,
            ),
            mock.patch.object(
                official_v2_tools,
                "_surface_facing_capture_geometry",
                return_value=(solution, [{}, {}, {}]),
            ),
            mock.patch.object(
                official_v2_tools,
                "_yield_adjust_chassis_controller",
                side_effect=false_success_controller,
            ),
        ):
            list(
                build_registry(adapter)[
                    "move_chassis_to_directly_facing_surface"
                ].fn(
                    ctx,
                    session_id="surface-session",
                    image_id="img_surface",
                    points=[
                        {"u": 100, "v": 200},
                        {"u": 400, "v": 500},
                        {"u": 700, "v": 800},
                    ],
                    nav_timeout_s=12.0,
                    pos_tol_m=0.02,
                )
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["failure_stage"], "tracking")
        self.assertFalse(result["final_position_ok"])
        self.assertGreater(result["final_position_error_m"], 0.02)
        json.dumps(result, allow_nan=False)

    def test_surface_facing_public_stack_has_no_privileged_runtime_reads(self) -> None:
        source = "\n".join(
            (
                inspect.getsource(
                    PUBLIC_TOOL_FUNCTIONS[
                        "move_chassis_to_directly_facing_surface"
                    ]
                ),
                inspect.getsource(official_v2_tools._surface_facing_capture_geometry),
                inspect.getsource(
                    official_v2_tools._surface_facing_current_policy_base_pose
                ),
                inspect.getsource(official_v2_tools._yield_adjust_chassis_controller),
            )
        )
        for forbidden in (
            "ctx.world.robot",
            "ctx.world.env",
            "og.sim",
            "segmentation",
            "object_scope",
            ".aabb",
            "get_joint_positions",
            "set_joint_positions",
            "set_position_orientation",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("depth_linear", source)
        self.assertIn("policy_local_base_pose", source)
        self.assertIn("base_qvel", source)

    def test_floor_point_at_front_reference_requires_no_translation(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)
        base_idx = world.controller_action_idx("base")
        target_robot = np.asarray(
            [BASE_FRONT_OFFSET_M, PATH_Y_CENTER_M, 0.0],
            dtype=np.float64,
        )
        actions = []

        with (
            mock.patch.object(
                official_v2_tools,
                "_load_frozen_capture",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(
                official_v2_tools,
                "_point_from_relative_uv",
                return_value=(target_robot.copy(), {}),
            ),
            mock.patch.object(
                official_v2_tools,
                "_surface_normal_from_relative_uv",
                return_value=(
                    np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                    {"normal_abs_z": 1.0},
                ),
            ),
        ):
            for action in build_registry(adapter)[
                "move_chassis_to_floor_point"
            ].fn(
                ctx,
                session_id="test-session",
                image_id="img_test",
                u=500,
                v=500,
                nav_timeout_s=3.0,
            ):
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                actions.append(action.copy())
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["base_qvel"]] = 0.0
                adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["target_distance_m"], 0.0)
        self.assertEqual(result["requested"]["forward_m"], 0.0)
        self.assertEqual(result["requested"]["translation_m"], 0.0)
        self.assertTrue(actions)
        self.assertTrue(
            all(
                np.allclose(action[base_idx], np.zeros(3), atol=1e-9)
                for action in actions
            )
        )

    def test_floor_point_does_not_report_arrival_at_stalled_boundary(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)
        base_idx = world.controller_action_idx("base")
        target_robot = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        actions = []

        with (
            mock.patch.object(
                official_v2_tools,
                "_load_frozen_capture",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(
                official_v2_tools,
                "_point_from_relative_uv",
                return_value=(target_robot.copy(), {}),
            ),
            mock.patch.object(
                official_v2_tools,
                "_surface_normal_from_relative_uv",
                return_value=(
                    np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                    {"normal_abs_z": 1.0},
                ),
            ),
        ):
            for action in build_registry(adapter)[
                "move_chassis_to_floor_point"
            ].fn(
                ctx,
                session_id="test-session",
                image_id="img_test",
                u=500,
                v=500,
                nav_timeout_s=3.0,
            ):
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                actions.append(action.copy())
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["base_qvel"]] = 0.0
                adapter.update({"robot_r1::proprio": proprio})

        self.assertFalse(result["ok"])
        self.assertTrue(result["obstacle_limited"])
        self.assertFalse(result["near_target_ok"])
        self.assertTrue(result["require_linear_target"])
        self.assertEqual(result["obstacle_stop_reason"], "base_qvel_stall")
        self.assertIn("did not reach", result["error"])
        tracking = result["linear_tracking"]
        self.assertEqual(tracking["source"], "evaluator_base_qvel")
        self.assertEqual(tracking["observed_xy_mps"], [0.0, 0.0])
        self.assertEqual(tracking["low_speed_progress_m"], 0.0)
        self.assertEqual(tracking["consecutive_low_speed_observations"],
                         official_v2_tools.ADJUST_CHASSIS_STALL_STEPS)
        self.assertTrue(actions)
        self.assertTrue(
            all(abs(float(action[base_idx][2])) <= 1e-9 for action in actions)
        )

    def test_floor_point_direct_xy_path_has_no_privileged_runtime_reads(self) -> None:
        source = "\n".join(
            (
                inspect.getsource(
                    PUBLIC_TOOL_FUNCTIONS["move_chassis_to_floor_point"]
                ),
                inspect.getsource(
                    official_v2_tools._yield_adjust_chassis_controller
                ),
                inspect.getsource(official_v2_tools._point_in_robot_frame),
            )
        )
        for forbidden in (
            "ctx.world.robot",
            "ctx.world.env",
            "og.sim",
            "object_scope",
            ".aabb",
            "get_joint_positions",
            "set_joint_positions",
            "set_position_orientation",
            "camera_depth_frames",
            "contact",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("yield from adjust_chassis", source)
        self.assertIn("base_qvel", source)
        self.assertIn("set_base_velocity", source)

    def test_adjust_chassis_stable_boundary_is_idempotent(self) -> None:
        normalized = validate_submission(
            "adjust_chassis",
            {"forward": 10.0, "translation": 0.25, "spin": 0.0},
        )
        self.assertEqual(normalized["forward"], 10.0)
        self.assertEqual(normalized["translation"], 0.25)
        self.assertNotIn("nav_guard", normalized)

        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)
        base_idx = world.controller_action_idx("base")
        actions = []
        for action in build_registry(adapter)["adjust_chassis"].fn(
            ctx,
            forward=10.0,
            timeout_s=5.0,
        ):
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["base_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})
            self.assertLess(len(actions), 150)

        self.assertTrue(result["ok"])
        self.assertTrue(result["obstacle_limited"])
        self.assertEqual(result["obstacle_stop_reason"], "base_qvel_stall")
        self.assertFalse(result["recovery_attempted"])
        self.assertEqual(result["retreat_m"], 0.0)
        self.assertGreaterEqual(result["settled_steps"], 15)
        self.assertFalse(
            any(float(action[base_idx][0]) < 0.0 for action in actions)
        )

    def test_adjust_chassis_does_not_substitute_depth_for_base_qvel(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)
        base_idx = world.controller_action_idx("base")

        actions = []
        with mock.patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_depth_points",
            side_effect=AssertionError("adjust_chassis must not read RGB-D"),
        ):
            for action in build_registry(adapter)["adjust_chassis"].fn(
                ctx,
                forward=10.0,
                timeout_s=5.0,
            ):
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                actions.append(action.copy())
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["base_qvel"]] = 0.0
                adapter.update({"robot_r1::proprio": proprio})
                self.assertLess(len(actions), 150)

        self.assertTrue(result["ok"])
        self.assertTrue(result["obstacle_limited"])
        self.assertEqual(result["obstacle_stop_reason"], "base_qvel_stall")
        self.assertEqual(result["actual"]["forward_m"], 0.0)
        self.assertEqual(
            result["verification"],
            "direct_evaluator_base_qvel_integration",
        )
        self.assertFalse(result["recovery_attempted"])
        self.assertFalse(result["rgbd_motion_guard_used"])
        self.assertFalse(
            any(float(action[base_idx][0]) < 0.0 for action in actions)
        )

    def test_adjust_chassis_source_has_no_privileged_observation_reads(self) -> None:
        source = inspect.getsource(PUBLIC_TOOL_FUNCTIONS["adjust_chassis"])
        for forbidden in (
            "robot_pose",
            "chest_pose",
            "current_scene_graph",
            "ctx.world.env",
            "ctx.world.robot",
            "contact",
            "segmentation",
            "object_state",
            "camera_depth_frames",
            "_depth_points",
            "clearance_m",
        ):
            self.assertNotIn(forbidden, source)

    def test_adjust_chassis_does_not_forward_private_control_kwargs(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)

        list(
            build_registry(adapter)["adjust_chassis"].fn(
                ctx,
                forward=0.0,
                _result_tool_name="move_chassis_to_floor_point",
                _linear_tolerance_m=42.0,
                _require_linear_target=True,
            )
        )

        self.assertEqual(result["tool"], "adjust_chassis")
        self.assertEqual(
            result["linear_tolerance_m"],
            ADJUST_CHASSIS_FORWARD_TOL_M,
        )
        self.assertFalse(result["require_linear_target"])

    def test_policy_odometry_scales_normalized_base_actions_to_si(self) -> None:
        _, world = self._adapter_world()
        world.set_base_velocity(1.0, 0.0, 0.0)
        pose = world.robot_pose()
        self.assertAlmostEqual(float(pose.pos[0]), 0.75 / 30.0, places=8)
        self.assertAlmostEqual(float(pose.pos[1]), 0.0, places=8)

        world.set_base_velocity(0.0, 1.0, 0.0)
        pose = world.robot_pose()
        self.assertAlmostEqual(float(pose.pos[0]), 0.75 / 30.0, places=8)
        self.assertAlmostEqual(float(pose.pos[1]), 0.75 / 30.0, places=8)

    def test_policy_odometry_uses_qvel_magnitude_in_action_direction(self) -> None:
        adapter, world = self._adapter_world()
        world.set_base_velocity(0.0, 1.0, 0.0)
        proprio = adapter.proprio_vector().copy()
        # Virtual-base qvel can be expressed in a canonical frame whose x axis
        # is not the robot's current x axis.  Its planar speed is invariant.
        proprio[PROPRIO_SLICES["base_qvel"]] = [0.75, 0.0, 0.0]
        adapter.update({"robot_r1::proprio": proprio})

        np.testing.assert_allclose(
            world.base_qvel(),
            [0.0, 0.75, 0.0],
            atol=1e-9,
        )
        report = world.reconcile_base_odometry_from_proprio(1.0 / 30.0)
        pose = world.robot_pose()

        self.assertTrue(report["corrected"])
        self.assertEqual(
            report["raw_observed_base_qvel"],
            [0.75, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            report["observed_base_qvel"],
            [0.0, 0.75, 0.0],
            atol=1e-9,
        )
        self.assertAlmostEqual(float(pose.pos[0]), 0.0, places=8)
        self.assertAlmostEqual(float(pose.pos[1]), 0.75 / 30.0, places=8)

    def test_set_arm_respects_requested_step_and_converges(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)

        actions = self._drive_actions(
            adapter,
            world,
            build_registry(adapter)["set_arm_to_grasp_position"].fn(
                ctx,
                arm="left",
                max_dq_per_step=0.055,
            ),
        )

        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["requested_max_dq_per_step"], 0.055)
        self.assertAlmostEqual(result["effective_max_dq_per_step"], 0.055)
        self.assertLess(len(actions), 50)
        self.assertGreaterEqual(result["planned_interpolation_steps"], 38)
        arm_slice = world.controller_action_idx("arm_left")
        previous = np.zeros(ARM_DOF, dtype=np.float64)
        for action in actions[: result["interpolation_steps"]]:
            command = np.asarray(action[arm_slice], dtype=np.float64)
            self.assertLessEqual(
                float(np.max(np.abs(command - previous))),
                result["effective_max_dq_per_step"] + 1e-6,
            )
            previous = command

    def test_set_arm_respects_sub_minimum_requested_step_and_j8_lock(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)

        actions = self._drive_actions(
            adapter,
            world,
            build_registry(adapter)["set_arm_to_grasp_position"].fn(
                ctx,
                arm="right",
                max_dq_per_step=0.01,
                timeout_s=15.0,
            ),
        )

        self.assertTrue(result["ok"], result)
        self.assertAlmostEqual(result["effective_max_dq_per_step"], 0.01)
        arm_slice = world.controller_action_idx("arm_right")
        previous = np.zeros(ARM_DOF, dtype=np.float64)
        for action in actions[: result["action_steps"]]:
            command = np.asarray(action[arm_slice], dtype=np.float64)
            self.assertLessEqual(
                float(np.max(np.abs(command - previous))),
                0.01 + 1e-6,
            )
            if ARM_DOF == 8:
                self.assertAlmostEqual(float(command[7]), 0.0, places=7)
            previous = command

    def test_set_arm_snapshot_uses_one_atomic_evaluator_proprio_frame(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        expected_q = np.linspace(-0.4, 0.4, ARM_DOF, dtype=np.float32)
        expected_v = np.linspace(0.7, -0.7, ARM_DOF, dtype=np.float32)
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = expected_q
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = expected_v
        adapter.update({"robot_r1::proprio": proprio})
        ctx, _result = self._ctx(world)

        # A pair of independent world reads could observe different frames;
        # setarm must consume the adapter's single immutable proprio snapshot.
        with (
            mock.patch.object(
                world,
                "arm_qpos_list",
                side_effect=AssertionError("qpos was read outside the frame"),
            ),
            mock.patch.object(
                world,
                "arm_qvel_list",
                side_effect=AssertionError("qvel was read outside the frame"),
            ),
        ):
            qpos, qvel = official_v2_tools._grasp_prep_snapshot(
                ctx,
                ["right"],
            )

        np.testing.assert_allclose(qpos["right"], expected_q)
        np.testing.assert_allclose(qvel["right"], expected_v)

    def test_set_arm_dynamic_entry_handoff_does_not_reject_moving_arm(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        arm_q = np.asarray(GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = arm_q
        proprio[PROPRIO_SLICES["arm_right_qvel"]][4] = 3.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        actions, _trace = self._drive_set_arm_dynamic(
            adapter,
            world,
            build_registry(adapter)["set_arm_to_grasp_position"].fn(
                ctx,
                arm="right",
                tol=0.01,
                timeout_s=2.0,
            ),
            momentum=0.88,
        )

        self.assertTrue(result["ok"], result)
        self.assertGreater(result["entry_settle_steps"], 0)
        self.assertLessEqual(
            result["entry_settle_steps"], result["handoff_max_steps"]
        )
        self.assertGreater(len(actions), 3)
        self.assertFalse(result["tracking_stalled"])

    def test_set_arm_dynamic_commands_stay_target_directed_before_crossing(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        start = np.asarray(
            [-1.40, 0.17, 0.87, -1.19, -0.64, 0.71, -0.76, 0.0],
            dtype=np.float32,
        )[:ARM_DOF]
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = start
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        _actions, trace = self._drive_set_arm_dynamic(
            adapter,
            world,
            build_registry(adapter)["set_arm_to_grasp_position"].fn(
                ctx,
                arm="right",
                max_dq_per_step=0.25,
                tol=0.08,
                timeout_s=5.0,
            ),
            momentum=0.95,
        )

        self.assertTrue(result["ok"], result)
        target = np.asarray(GRASP_PREP_Q[:ARM_DOF], dtype=np.float64)
        active = np.arange(ARM_DOF - 1 if ARM_DOF == 8 else ARM_DOF)
        for frame in trace:
            measured = frame["right"]["q"]
            command = frame["right"]["command"]
            for index in active:
                direction = float(np.sign(target[index] - start[index]))
                if direction == 0.0:
                    continue
                crossed = (
                    (measured[index] - target[index]) * direction > 1e-6
                )
                if not crossed:
                    self.assertGreaterEqual(
                        float((command[index] - measured[index]) * direction),
                        -1e-7,
                        msg=(
                            f"J{index + 1} command retreated before crossing: "
                            f"q={measured[index]:.5f} cmd={command[index]:.5f}"
                        ),
                    )
        self.assertEqual(
            result["command_reversals_before_target_crossing"],
            0,
        )

    def test_set_arm_does_not_global_stall_while_one_joint_moves_slowly(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        start = np.asarray(GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        start[4] = -0.30  # J5 is deliberately slow, but not at a hard limit.
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = start
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        self._drive_set_arm_dynamic(
            adapter,
            world,
            build_registry(adapter)["set_arm_to_grasp_position"].fn(
                ctx,
                arm="right",
                max_dq_per_step=0.12,
                tol=0.01,
                timeout_s=8.0,
            ),
            momentum=0.0,
            joint_response_scale={("right", 4): 0.02},
        )

        self.assertTrue(result["ok"], result)
        self.assertFalse(result["tracking_stalled"])
        self.assertLessEqual(result["error_rad"]["right"][4], 0.01)

    def test_set_arm_accepts_only_stationary_static_limit_residual(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        start = np.asarray(
            [-1.40, 0.17, 0.87, -1.19, -0.64, 0.71, -0.76, 0.0],
            dtype=np.float32,
        )[:ARM_DOF]
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = start
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)
        limit_plateau = float(GRASP_PREP_Q[5]) + 0.105

        self._drive_set_arm_dynamic(
            adapter,
            world,
            build_registry(adapter)["set_arm_to_grasp_position"].fn(
                ctx,
                arm="right",
                max_dq_per_step=0.30,
                tol=0.08,
                timeout_s=5.0,
            ),
            momentum=0.75,
            plateau_j6=limit_plateau,
        )

        self.assertTrue(result["ok"], result)
        self.assertFalse(result["strict_position_converged"])
        self.assertTrue(result["accepted_position_converged"])
        self.assertEqual(result["limit_residual_joints_by_arm"]["right"], [5])
        self.assertFalse(result["tracking_stalled"])
        self.assertAlmostEqual(
            world.arm_pin_qpos_list("right")[5],
            limit_plateau,
            places=5,
        )

    def test_set_arm_progress_extends_soft_timeout_and_finishes(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        start = GRASP_PREP_Q[:ARM_DOF].astype(np.float32).copy()
        start[4] = -0.30
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = start
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        self._drive_actions(
            adapter,
            world,
            build_registry(adapter)["set_arm_to_grasp_position"].fn(
                ctx,
                arm="right",
                timeout_s=0.2,
                max_dq_per_step=0.05,
                tol=0.01,
            ),
        )

        self.assertTrue(result["ok"], result)
        self.assertGreaterEqual(result["progress_deadline_extensions"], 1)
        self.assertFalse(result["timed_out"])
        self.assertLessEqual(result["terminal_max_qvel_rad_s"], 0.35)

    def test_set_arm_carry_mode_caps_each_feedback_step(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        start = GRASP_PREP_Q[:ARM_DOF].astype(np.float32).copy()
        start[4] = -0.36
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = start
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        world.latch_gripper_close_keepalive(
            "right",
            effort=[-0.1, -0.1]
            if world.gripper_uses_effort("right")
            else None,
        )
        ctx, result = self._ctx(world)

        actions = self._drive_actions(
            adapter,
            world,
            build_registry(adapter)["set_arm_to_grasp_position"].fn(
                ctx,
                arm="right",
                max_dq_per_step=0.30,
                tol=0.01,
            ),
        )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["carry_aware_step_limit_active"])
        self.assertAlmostEqual(
            result["effective_max_dq_per_step_by_arm"]["right"],
            0.12,
        )
        arm_slice = world.controller_action_idx("arm_right")
        previous = start.astype(np.float64)
        for action in actions[: result["action_steps"]]:
            command = np.asarray(action[arm_slice], dtype=np.float64)
            self.assertLessEqual(
                float(np.max(np.abs(command - previous))),
                0.12 + 1e-6,
            )
            previous = command

    def test_set_arm_waits_for_entry_velocity_to_settle(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = (
            GRASP_PREP_Q[:ARM_DOF]
        )
        proprio[PROPRIO_SLICES["arm_right_qvel"]][4] = 2.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)["set_arm_to_grasp_position"].fn(
            ctx,
            arm="right",
            tol=0.01,
        )

        actions = []
        for index, raw_action in enumerate(generator):
            action = np.asarray(raw_action, dtype=np.float32)
            actions.append(action.copy())
            updated = adapter.proprio_vector().copy()
            updated[PROPRIO_SLICES["arm_right_qpos"]] = action[
                world.controller_action_idx("arm_right")
            ]
            if index >= 1:
                updated[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": updated})

        self.assertTrue(result["ok"], result)
        self.assertGreaterEqual(result["entry_settle_steps"], 3)
        self.assertTrue(result["entry_settled"])
        self.assertGreaterEqual(result["stable_steps"], 3)

    def test_set_arm_accepts_stationary_qpos_when_raw_qvel_is_aliased(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        raw_alias_by_arm = {"left": 2.56, "right": 4.0}
        for side in ("left", "right"):
            raw_qvel = np.zeros(ARM_DOF, dtype=np.float32)
            raw_qvel[0] = raw_alias_by_arm[side]
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = raw_qvel
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)["set_arm_to_grasp_position"].fn(
            ctx,
            arm="both",
            tol=0.01,
            timeout_s=5.0,
        )

        actions = []
        for index, raw_action in enumerate(generator):
            self.assertLess(index, 80, "aliased-qvel grasp prep did not finish")
            action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            updated = adapter.proprio_vector().copy()
            for side in ("left", "right"):
                updated[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                    world.controller_action_idx(f"arm_{side}")
                ]
                raw_qvel = np.zeros(ARM_DOF, dtype=np.float32)
                raw_qvel[0] = raw_alias_by_arm[side]
                updated[PROPRIO_SLICES[f"arm_{side}_qvel"]] = raw_qvel
            adapter.update({"robot_r1::proprio": updated})

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["raw_qvel_incoherence_detected"])
        self.assertEqual(
            result["terminal_stationarity_evidence"],
            "consecutive_qpos_delta",
        )
        self.assertGreaterEqual(result["terminal_max_qvel_rad_s"], 4.0)
        self.assertLessEqual(
            result["terminal_derived_max_qvel_rad_s"],
            result["velocity_tolerance_rad_s"],
        )
        self.assertFalse(result["tracking_stalled"])
        self.assertLess(len(actions), 80)

    def test_set_arm_failure_latches_latest_measured_pose(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        measured = GRASP_PREP_Q[:ARM_DOF].astype(np.float32).copy()
        measured[4] = -0.70
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = measured
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        actions = list(
            build_registry(adapter)["set_arm_to_grasp_position"].fn(
                ctx,
                arm="right",
                timeout_s=0.2,
                tol=0.01,
            )
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["terminal_hold_source"],
            "latest_evaluator_proprioception",
        )
        np.testing.assert_allclose(
            world.arm_pin_qpos_list("right"),
            measured,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            actions[-1][world.controller_action_idx("arm_right")],
            measured,
            atol=1e-7,
        )

    def test_set_arm_cancellation_latches_latest_measured_pose(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        start = GRASP_PREP_Q[:ARM_DOF].astype(np.float32).copy()
        start[4] = -0.70
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = start
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, _result = self._ctx(world)
        cancelled = False

        class SkillCancelled(RuntimeError):
            pass

        def raise_if_cancelled(where=""):
            if cancelled:
                raise SkillCancelled(f"cancelled at {where}")

        ctx.raise_if_cancelled = raise_if_cancelled
        generator = build_registry(adapter)["set_arm_to_grasp_position"].fn(
            ctx,
            arm="right",
        )
        first_action = np.asarray(next(generator), dtype=np.float32)
        measured = start.copy()
        measured[4] = -0.55
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = measured
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        self.assertGreater(
            float(
                first_action[world.controller_action_idx("arm_right")][4]
            ),
            float(start[4]),
        )

        cancelled = True
        with self.assertRaises(SkillCancelled):
            next(generator)
        np.testing.assert_allclose(
            world.arm_pin_qpos_list("right"),
            measured,
            atol=1e-7,
        )

    def test_set_arm_stalled_proprio_fails_fast_instead_of_1200_steps(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)

        actions = list(
            build_registry(adapter)["set_arm_to_grasp_position"].fn(
                ctx,
                arm="right",
                timeout_s=80.0,
            )
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["tracking_stalled"])
        self.assertFalse(result["timed_out"])
        self.assertIn("tracking stalled", result["error"])
        self.assertLess(len(actions), 50)
        self.assertEqual(result["stall_steps"], result["stall_limit"])

    def test_set_arm_aliased_qvel_does_not_mask_stalled_qpos(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        raw_qvel = np.zeros(ARM_DOF, dtype=np.float32)
        raw_qvel[0] = 2.56
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = raw_qvel
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)["set_arm_to_grasp_position"].fn(
            ctx,
            arm="right",
            timeout_s=80.0,
        )

        actions = []
        for index, raw_action in enumerate(generator):
            self.assertLess(index, 60, "stalled aliased-qvel arm timed out")
            actions.append(np.asarray(raw_action, dtype=np.float32).copy())
            # Advance the official observation sequence, but deliberately keep
            # qpos unchanged while preserving the impossible raw qvel.
            updated = adapter.proprio_vector().copy()
            updated[PROPRIO_SLICES["arm_right_qvel"]] = raw_qvel
            adapter.update({"robot_r1::proprio": updated})

        self.assertFalse(result["ok"])
        self.assertTrue(result["tracking_stalled"])
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["raw_qvel_incoherence_detected"])
        self.assertIn("tracking stalled", result["error"])
        self.assertLess(len(actions), 50)
        self.assertEqual(result["stall_steps"], result["stall_limit"])

    def test_reset_body_respects_requested_step_and_converges(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        start = np.asarray([0.0, 2.291, 0.0, 0.0], dtype=np.float32)
        proprio[PROPRIO_SLICES["trunk_qpos"]] = start
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        actions = self._drive_trunk_actions(
            adapter,
            world,
            build_registry(adapter)["reset_body"].fn(
                ctx,
                timeout_s=45.0,
                trunk_max_step=0.06,
            ),
        )

        self.assertTrue(result["ok"], result)
        expected_min = int(np.ceil(1.5 * 2.291 / 0.06))
        self.assertGreaterEqual(result["interpolation_steps"], expected_min)
        trunk_slice = world.controller_action_idx("trunk")
        commands = np.asarray(
            [
                action[trunk_slice]
                for action in actions[: result["interpolation_steps"]]
            ],
            dtype=np.float64,
        )
        path = np.vstack([start.astype(np.float64), commands])
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(path, axis=0)))),
            0.06 + 1e-6,
        )

    def test_reset_body_keep_right_uses_soft_position_leash_and_j567(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        start_trunk = np.asarray(
            [0.25, 0.10, -0.12, 0.0],
            dtype=np.float32,
        )
        arm_q = np.asarray(GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        proprio[PROPRIO_SLICES["trunk_qpos"]] = start_trunk
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = arm_q
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        actions = self._drive_reset_body_actions(
            adapter,
            world,
            build_registry(adapter)["reset_body"].fn(
                ctx,
                keep_ori_arm="right",
                timeout_s=45.0,
                trunk_max_step=0.06,
            ),
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(
            result["implementation_version"],
            "body_first_nonblocking_keep_ori_v4",
        )
        self.assertEqual(result["keep_ori_arm"], "right")
        self.assertEqual(
            result["keep_orientation"]["controller"],
            "body_first_nonblocking_keep_ori",
        )
        self.assertAlmostEqual(result["effective_max_step_rad"], 0.06)
        self.assertEqual(result["keep_recovery_steps"], 0)
        self.assertGreater(result["body_progress_steps"], 0)
        travel = float(np.max(np.abs(
            np.asarray(result["target_trunk_q"], dtype=np.float64)
            - start_trunk.astype(np.float64)
        )))
        old_plan = int(math.ceil(1.5 * travel / 0.025))
        self.assertLess(result["planned_interpolation_steps"], old_plan / 2)
        self.assertLessEqual(result["settle_steps"], 8)
        self.assertLess(result["action_steps"], 50)
        keep = result["keep_orientation"]
        self.assertTrue(keep["enabled"])
        self.assertEqual(set(keep["arms"]), {"right"})
        right = keep["arms"]["right"]
        self.assertGreater(right["solve_calls"], 0)
        self.assertLessEqual(
            right["max_command_q1234_delta_rad"],
            keep["q1234_max_delta_from_measured_per_action_rad"] + 1e-6,
        )
        self.assertLessEqual(
            right["max_command_j567_delta_rad"],
            keep["j567_max_delta_from_measured_per_action_rad"] + 1e-6,
        )
        final = right["final_observation"]
        self.assertLessEqual(
            final["effective_position_error_m"],
            keep["position_final_tolerance_m"],
        )
        self.assertLessEqual(
            final["effective_orientation_error_deg"],
            keep["orientation_final_tolerance_deg"],
        )
        self.assertIsNotNone(final["raw_evaluator_position_error_m"])
        if ARM_DOF == 8:
            self.assertAlmostEqual(final["observed_j8_rad"], 0.0, places=6)

        right_idx = world.controller_action_idx("arm_right")
        previous = arm_q.astype(np.float64)
        for action in actions:
            command = np.asarray(action[right_idx], dtype=np.float64)
            self.assertLessEqual(
                float(np.max(np.abs(command[:4] - previous[:4]))),
                keep["q1234_max_delta_from_measured_per_action_rad"] + 1e-6,
            )
            self.assertLessEqual(
                float(np.max(np.abs(command[4:7] - previous[4:7]))),
                keep["j567_max_delta_from_measured_per_action_rad"] + 1e-6,
            )
            previous = command

    def test_reset_body_keep_left_replays_15062_timeout_pose(self) -> None:
        if ARM_DOF != 8:
            self.skipTest("15062 timeout replay requires the 8-DoF robot profile")
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [
            1.8325188,
            -2.2576492,
            -1.5300280,
            0.0,
        ]
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = [
            -0.7606644,
            0.00807146,
            -0.43232054,
            -1.34808254,
            -2.20450592,
            0.74670720,
            -0.45001575,
            0.0,
        ]
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = [
            0.0,
            0.0,
            0.0,
            -2.09439707,
            0.0,
            -1.04719818,
            0.0,
            0.0,
        ]
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        self._drive_reset_body_actions(
            adapter,
            world,
            build_registry(adapter)["reset_body"].fn(
                ctx,
                keep_ori_arm="left",
                timeout_s=45.0,
                trunk_max_step=0.06,
            ),
        )

        self.assertTrue(result["ok"], result)
        self.assertFalse(result["timed_out"])
        self.assertLessEqual(result["body_progress_steps"], 33)
        self.assertLessEqual(result["settle_steps"], 2)
        self.assertLessEqual(result["action_steps"], 35)
        self.assertTrue(result["position_drift_warning"])
        left = result["keep_orientation"]["arms"]["left"]
        self.assertLessEqual(
            left["max_observed_orientation_error_deg"],
            result["keep_orientation"]["orientation_final_tolerance_deg"],
        )
        self.assertTrue(
            left["final_observation"]["within_orientation_final_tolerance"]
        )

    def test_reset_body_keep_left_does_not_pin_body_when_arm_lags(self) -> None:
        if ARM_DOF != 8:
            self.skipTest("arm-lag replay requires the 8-DoF robot profile")
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [
            -0.55,
            2.07,
            0.22,
            0.0,
        ]
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = [
            -0.7606644,
            0.00807146,
            -0.43232054,
            -1.34808254,
            -2.20450592,
            0.74670720,
            -0.45001575,
            0.0,
        ]
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = [
            0.0,
            0.0,
            0.0,
            -2.09439707,
            0.0,
            -1.04719818,
            0.0,
            0.0,
        ]
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        actions = self._drive_reset_body_actions_with_arm_lag(
            adapter,
            world,
            build_registry(adapter)["reset_body"].fn(
                ctx,
                keep_ori_arm="left",
                timeout_s=45.0,
            ),
            arm_response=0.25,
        )

        self.assertTrue(result["ok"], result)
        self.assertGreater(result["keep_tracking_guard_steps"], 3)
        self.assertEqual(result["stationary_recovery_steps_during_body_motion"], 0)
        self.assertTrue(result["defaulted_max_step"])
        self.assertAlmostEqual(result["effective_max_step_rad"], 0.08)
        self.assertGreater(result["body_progress_steps"], 0)
        self.assertLessEqual(result["body_progress_steps"], 35)
        self.assertLessEqual(result["action_steps"], 40)
        trunk_idx = world.controller_action_idx("trunk")
        moving_actions = actions[: result["body_progress_steps"]]
        self.assertTrue(moving_actions)
        for previous, current in zip(moving_actions, moving_actions[1:]):
            self.assertGreater(
                float(np.max(np.abs(current[trunk_idx] - previous[trunk_idx]))),
                1e-6,
            )
        final = result["keep_orientation"]["arms"]["left"][
            "final_observation"
        ]
        self.assertGreater(
            result["keep_orientation"]["arms"]["left"][
                "max_observed_orientation_error_deg"
            ],
            result["keep_orientation"]["orientation_tracking_guard_deg"],
        )
        self.assertLessEqual(
            result["keep_orientation"]["arms"]["left"][
                "max_observed_orientation_error_deg"
            ],
            result["keep_orientation"]["orientation_hard_limit_deg"],
        )
        self.assertLessEqual(
            result["keep_orientation"]["arms"]["left"][
                "max_observed_orientation_error_deg"
            ],
            12.0,
        )
        self.assertTrue(final["within_orientation_final_tolerance"], final)

    def test_reset_body_replays_15064_transient_without_arm_gating(self) -> None:
        if ARM_DOF != 8:
            self.skipTest("15064 replay requires the 8-DoF robot profile")
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [
            -0.9507288,
            2.4891582,
            0.9345989,
            0.0,
        ]
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = [
            -1.5791105,
            -0.1711374,
            -0.9479846,
            -0.8313239,
            1.0484877,
            0.5853702,
            0.6123332,
            0.0,
        ]
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = [
            0.0,
            0.0,
            0.0,
            -2.0943971,
            0.0,
            -1.0471982,
            0.0,
            0.0,
        ]
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        actions = self._drive_reset_body_actions_with_arm_lag(
            adapter,
            world,
            build_registry(adapter)["reset_body"].fn(
                ctx,
                keep_ori_arm="left",
                timeout_s=45.0,
                trunk_max_step=0.06,
            ),
            arm_response=0.10,
        )

        self.assertTrue(result["ok"], result)
        self.assertFalse(result["timed_out"])
        self.assertFalse(result["tracking_stalled"])
        self.assertFalse(result["keep_orientation_affects_trunk"])
        self.assertFalse(result["keep_orientation_affects_success"])
        self.assertGreater(result["keep_hard_limit_observation_steps"], 0)
        self.assertTrue(result["keep_orientation_degraded"])
        self.assertEqual(result["keep_plan_backoff_steps"], 0)
        trunk_idx = world.controller_action_idx("trunk")
        np.testing.assert_allclose(
            actions[-1][trunk_idx],
            result["target_trunk_q"],
            atol=1e-6,
        )

    def test_reset_body_keep_uses_raw_eef_error_as_local_solver_feedback(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [0.2, 0.1, -0.1, 0.0]
        arm_q = np.asarray(GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = arm_q
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
        proprio = self._sync_local_eef_proprio(proprio)
        adapter.update({"robot_r1::proprio": proprio})
        ctx, _result = self._ctx(world)
        keep_states = official_v2_tools._reset_body_init_keep_states(
            ctx,
            ("left",),
        )
        keeper = keep_states["left"]

        angle = math.radians(8.0)
        rotation_delta = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        proprio = adapter.proprio_vector().copy()
        raw_reference_rotation = official_v2_tools._quat_to_mat_xyzw(
            keeper.raw_reference_quaternion
        )
        proprio[PROPRIO_SLICES["eef_left_quat"]] = mat_to_quat_xyzw(
            rotation_delta @ raw_reference_rotation
        )
        adapter.update({"robot_r1::proprio": proprio})

        target, report = (
            official_v2_tools._reset_body_feedback_orientation_target(
                ctx,
                "left",
                keeper,
            )
        )

        self.assertTrue(report["raw_feedback_available"])
        self.assertAlmostEqual(report["raw_orientation_error_deg"], 8.0, places=4)
        self.assertAlmostEqual(
            report["raw_orientation_feedback_gain"],
            1.0 + 0.5 * (8.0 - 3.0) / (10.0 - 3.0),
            places=4,
        )
        self.assertAlmostEqual(
            orientation_error_deg(target, keeper.reference_quaternion),
            8.0 * report["raw_orientation_feedback_gain"],
            places=4,
        )
        self.assertEqual(
            report["source"],
            "evaluator_eef_delta_mapped_to_local_fk",
        )

    def test_reset_body_keep_both_controls_both_arms(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [0.40, -0.30, -0.05, 0.0]
        arm_q = np.asarray(GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = arm_q
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        self._drive_reset_body_actions(
            adapter,
            world,
            build_registry(adapter)["reset_body"].fn(
                ctx,
                keep_ori_arm="both",
                timeout_s=45.0,
            ),
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(set(result["keep_orientation"]["arms"]), {"left", "right"})
        for side in ("left", "right"):
            final = result["keep_orientation"]["arms"][side][
                "final_observation"
            ]
            self.assertTrue(final["within_final_tolerances"], final)

    def test_reset_body_keep_solve_failure_cannot_block_trunk(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        start_trunk = np.asarray([0.30, -0.10, -0.08, 0.0], dtype=np.float32)
        arm_q = np.asarray(GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        proprio[PROPRIO_SLICES["trunk_qpos"]] = start_trunk
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = arm_q
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        def fail_orientation_solve(_state, _arm, q_start, _target, **_kwargs):
            return np.asarray(q_start, dtype=np.float64).copy(), {
                "ok": False,
                "error": "test-unreachable",
            }

        with mock.patch.object(
            official_v2_tools,
            "solve_j567_orientation_target",
            side_effect=fail_orientation_solve,
        ):
            actions = self._drive_reset_body_actions(
                adapter,
                world,
                build_registry(adapter)["reset_body"].fn(
                    ctx,
                    keep_ori_arm="right",
                    timeout_s=45.0,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertFalse(result["keep_orientation_affects_trunk"])
        self.assertFalse(result["keep_orientation_affects_success"])
        self.assertTrue(result["keep_orientation_degraded"])
        self.assertGreater(result["keep_fallback_steps"], 0)
        self.assertEqual(result["keep_plan_backoff_steps"], 0)
        self.assertGreater(result["action_steps"], 0)
        trunk_idx = world.controller_action_idx("trunk")
        np.testing.assert_allclose(
            actions[-1][trunk_idx],
            result["target_trunk_q"],
            atol=1e-6,
        )

    def test_reset_body_keep_initialization_failure_cannot_block_trunk(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [0.1, 0.8, -0.2, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        with mock.patch.object(
            official_v2_tools,
            "_reset_body_init_keep_states",
            side_effect=AssertionError("test-missing-arm-kinematics"),
        ):
            actions = self._drive_reset_body_actions(
                adapter,
                world,
                build_registry(adapter)["reset_body"].fn(
                    ctx,
                    keep_ori_arm="left",
                    timeout_s=45.0,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["keep_orientation_degraded"])
        self.assertIn(
            "test-missing-arm-kinematics",
            result["keep_orientation"]["initialization_error"],
        )
        self.assertEqual(result["keep_orientation"]["initialized_arms"], [])
        trunk_idx = world.controller_action_idx("trunk")
        np.testing.assert_allclose(
            actions[-1][trunk_idx],
            result["target_trunk_q"],
            atol=1e-6,
        )

    def test_reset_body_rejected_arm_override_falls_back_to_trunk_only(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [0.1, 0.3, -0.1, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)
        real_make_action = world.make_action

        def reject_explicit_kept_arm(**overrides):
            if "arm_left" in overrides:
                raise KeyError("test-rejected-kept-arm-override")
            return real_make_action(**overrides)

        with mock.patch.object(
            world,
            "make_action",
            side_effect=reject_explicit_kept_arm,
        ):
            actions = self._drive_reset_body_actions(
                adapter,
                world,
                build_registry(adapter)["reset_body"].fn(
                    ctx,
                    keep_ori_arm="left",
                    timeout_s=45.0,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertGreater(result["keep_action_fallback_steps"], 0)
        self.assertTrue(
            any("trunk-only command" in item for item in result["warnings"])
        )
        trunk_idx = world.controller_action_idx("trunk")
        np.testing.assert_allclose(
            actions[-1][trunk_idx],
            result["target_trunk_q"],
            atol=1e-6,
        )

    def test_reset_body_keep_observation_exception_cannot_block_trunk(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [0.1, 0.3, -0.1, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        with mock.patch.object(
            official_v2_tools,
            "_reset_body_observed_keep_report",
            side_effect=KeyError("test-malformed-arm-observation"),
        ):
            actions = self._drive_reset_body_actions(
                adapter,
                world,
                build_registry(adapter)["reset_body"].fn(
                    ctx,
                    keep_ori_arm="left",
                    timeout_s=45.0,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["keep_orientation_degraded"])
        self.assertGreater(result["keep_observation_failure_steps"], 0)
        self.assertTrue(
            any("observation failed" in item for item in result["warnings"])
        )
        trunk_idx = world.controller_action_idx("trunk")
        np.testing.assert_allclose(
            actions[-1][trunk_idx],
            result["target_trunk_q"],
            atol=1e-6,
        )

    def test_reset_body_keep_trunk_command_ramps_then_holds_user_step(self) -> None:
        current = np.zeros(4, dtype=np.float64)
        target = np.asarray([0.45, -0.4, 0.0, 0.0], dtype=np.float64)
        _command0, step0, scale0 = official_v2_tools._reset_body_keep_trunk_command(
            current,
            target,
            0.06,
            0,
            0.0,
        )
        _command2, step2, scale2 = official_v2_tools._reset_body_keep_trunk_command(
            current,
            target,
            0.06,
            2,
            0.0,
        )
        _command_slow, step_slow, scale_slow = (
            official_v2_tools._reset_body_keep_trunk_command(
                current,
                target,
                0.06,
                5,
                official_v2_tools.RESET_BODY_KEEP_POSITION_HARD_LIMIT_M,
            )
        )
        command_over_limit, step_over_limit, scale_over_limit = (
            official_v2_tools._reset_body_keep_trunk_command(
                current,
                target,
                0.06,
                5,
                2.0 * official_v2_tools.RESET_BODY_KEEP_POSITION_HARD_LIMIT_M,
            )
        )
        self.assertAlmostEqual(step0, 0.02, places=6)
        self.assertAlmostEqual(scale0, 1.0, places=6)
        self.assertAlmostEqual(step2, 0.06, places=6)
        self.assertAlmostEqual(scale2, 1.0, places=6)
        self.assertAlmostEqual(step_slow, 0.06, places=6)
        self.assertAlmostEqual(scale_slow, 1.0, places=6)
        self.assertAlmostEqual(step_over_limit, 0.06, places=6)
        self.assertAlmostEqual(scale_over_limit, 1.0, places=6)
        self.assertLess(
            float(np.max(np.abs(target - command_over_limit))),
            float(np.max(np.abs(target - current))),
        )

    def test_reset_body_keep_step_budget_beats_old_smoothstep_plan(self) -> None:
        travel = 1.86
        old_plan = int(math.ceil(1.5 * travel / 0.025))
        new_plan = max(
            2,
            int(
                math.ceil(
                    travel / official_v2_tools.RESET_BODY_KEEP_TRUNK_MAX_STEP_RAD
                )
            ),
        )
        self.assertGreaterEqual(old_plan, 110)
        self.assertLessEqual(new_plan, 35)
        self.assertLess(new_plan, old_plan / 3)

    def test_reset_body_keep_rejected_plan_never_reduces_trunk_step(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        start_trunk = np.asarray(
            [0.25, 0.10, -0.12, 0.0],
            dtype=np.float32,
        )
        arm_q = np.asarray(GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        proprio[PROPRIO_SLICES["trunk_qpos"]] = start_trunk
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = arm_q
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)
        real_solve = official_v2_tools.solve_j567_orientation_target

        def reject_large_trunk_step(state, arm, q_start, target_quat, **kwargs):
            step = float(
                np.max(np.abs(state.trunk_q - ctx.world.trunk_qpos()))
            )
            if step > 0.031:
                return np.asarray(q_start, dtype=np.float64).copy(), {
                    "ok": False,
                    "error": "test-step-too-large",
                }
            return real_solve(state, arm, q_start, target_quat, **kwargs)

        with mock.patch.object(
            official_v2_tools,
            "solve_j567_orientation_target",
            side_effect=reject_large_trunk_step,
        ):
            actions = self._drive_reset_body_actions(
                adapter,
                world,
                build_registry(adapter)["reset_body"].fn(
                    ctx,
                    keep_ori_arm="right",
                    timeout_s=45.0,
                    trunk_max_step=0.06,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["keep_plan_backoff_steps"], 0)
        self.assertGreater(result["keep_fallback_steps"], 0)
        self.assertLess(result["action_steps"], 50)
        self.assertEqual(result["trunk_priority"], "absolute")
        trunk_idx = world.controller_action_idx("trunk")
        trunk_commands = np.asarray(
            [
                action[trunk_idx]
                for action in actions[: result["body_progress_steps"]]
            ],
            dtype=np.float64,
        )
        path = np.vstack([start_trunk.astype(np.float64), trunk_commands])
        step_sizes = np.max(np.abs(np.diff(path, axis=0)), axis=1)
        self.assertAlmostEqual(step_sizes[0], 0.02, places=6)
        self.assertAlmostEqual(step_sizes[1], 0.04, places=6)
        if step_sizes.size > 2:
            self.assertTrue(np.all(step_sizes[2:-1] >= 0.059))

    def test_reset_body_keep_emergency_orientation_is_warning_only(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        start_trunk = np.asarray([0.20, 0.00, -0.10, 0.0], dtype=np.float32)
        arm_q = np.asarray(GRASP_PREP_Q[:ARM_DOF], dtype=np.float32)
        proprio[PROPRIO_SLICES["trunk_qpos"]] = start_trunk
        for side in ("left", "right"):
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = arm_q
            proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        def always_over_hard_limit(_ctx, keep_states):
            arms = {}
            for arm in keep_states:
                arms[arm] = {
                    "effective_position_error_m": 0.04,
                    "effective_orientation_error_deg": 30.0,
                }
            return {
                "ok": False,
                "within_hard_limits": False,
                "within_position_hard_limits": True,
                "within_orientation_hard_limits": False,
                "within_position_final_tolerances": True,
                "within_orientation_final_tolerances": False,
                "within_final_tolerances": False,
                "arms": arms,
            }

        with mock.patch.object(
            official_v2_tools,
            "_reset_body_observed_keep_report",
            side_effect=always_over_hard_limit,
        ):
            actions = self._drive_reset_body_actions(
                adapter,
                world,
                build_registry(adapter)["reset_body"].fn(
                    ctx,
                    keep_ori_arm="left",
                    timeout_s=45.0,
                    trunk_max_step=0.06,
                    shoulder_iters=4,
                ),
            )

        self.assertTrue(result["ok"], result)
        self.assertFalse(result["keep_orientation_satisfied"])
        self.assertTrue(result["keep_orientation_degraded"])
        self.assertFalse(result["keep_orientation_affects_trunk"])
        self.assertGreater(result["keep_hard_limit_observation_steps"], 0)
        self.assertTrue(
            any("25 deg diagnostic threshold" in item for item in result["warnings"])
        )
        self.assertFalse(result["timed_out"])
        self.assertGreater(result["keep_recovery_steps"], 0)
        self.assertGreater(result["action_steps"], 0)
        trunk_idx = world.controller_action_idx("trunk")
        np.testing.assert_allclose(
            actions[-1][trunk_idx],
            result["target_trunk_q"],
            atol=1e-6,
        )

    def test_reset_body_stalled_proprio_fails_fast(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [0.0, 2.291, 0.0, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        ctx, result = self._ctx(world)

        actions = list(
            build_registry(adapter)["reset_body"].fn(
                ctx,
                timeout_s=45.0,
                trunk_max_step=0.06,
            )
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["tracking_stalled"])
        self.assertFalse(result["timed_out"])
        self.assertIn("tracking stalled", result["error"])
        self.assertLess(len(actions), 100)

    def test_all_adjust_tools_accept_zero_delta_from_evaluator_observation(self) -> None:
        for name in (
            "adjust_left_eef_pose_in_head_frame",
            "adjust_right_eef_pose_in_head_frame",
            "adjust_left_eef_pose_in_wrist_frame",
            "adjust_right_eef_pose_in_wrist_frame",
        ):
            with self.subTest(name=name):
                adapter, world = self._adapter_world()
                self._set_adjust_ready_observation(adapter)
                ctx, result = self._ctx(world)
                actions = list(build_registry(adapter)[name].fn(ctx))

                self.assertTrue(result["ok"], result)
                self.assertEqual(result["tool"], name)
                self.assertEqual(result["tool_version"], "official_v2")
                self.assertFalse(result["j8_participates"])
                self.assertEqual(result["camera_pose_source"], "evaluator_cam_rel_poses")
                self.assertEqual(len(actions), 1)
                self.assertEqual(np.asarray(actions[0]).shape, (ACTION_DIM,))

    def test_adjust_target_preserves_v2_camera_and_local_rpy_conventions(self) -> None:
        half_sqrt = float(np.sqrt(0.5))
        target = camera_frame_pose_target(
            eef_position=[1.0, 2.0, 3.0],
            eef_quaternion=[0.0, 0.0, 0.0, 1.0],
            camera_quaternion=[0.0, 0.0, half_sqrt, half_sqrt],
            forward=0.10,
            leftward=0.20,
            upward=0.30,
            roll=10.0,
            pitch=0.0,
            yaw=0.0,
        )

        np.testing.assert_allclose(
            target.delta_camera,
            [-0.20, 0.30, -0.10],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            target.delta_robot,
            [-0.30, -0.20, -0.10],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            target.position,
            [0.70, 1.80, 2.90],
            atol=1e-9,
        )
        self.assertAlmostEqual(
            orientation_error_deg(
                [0.0, 0.0, 0.0, 1.0],
                target.quaternion,
            ),
            10.0,
            places=6,
        )

    def test_adjust_target_robot_base_xyz_is_not_rotated_by_head_camera(self) -> None:
        half_sqrt = float(np.sqrt(0.5))
        target = robot_base_frame_pose_target(
            eef_position=[1.0, 2.0, 3.0],
            eef_quaternion=[0.0, 0.0, 0.0, 1.0],
            camera_quaternion=[0.0, 0.0, half_sqrt, half_sqrt],
            x=0.10,
            y=0.20,
            z=0.30,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        )

        np.testing.assert_allclose(
            target.delta_robot,
            [0.10, 0.20, 0.30],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            target.delta_camera,
            [0.20, -0.10, 0.30],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            target.position,
            [1.10, 2.20, 3.30],
            atol=1e-9,
        )

    def test_head_adjustment_metadata_distinguishes_horizontal_push_from_camera_forward(
        self,
    ) -> None:
        spec = build_registry(None)["adjust_left_eef_pose_in_head_frame"]
        params = {item["name"]: item for item in spec.params}

        self.assertIn("chassis-horizontal push", spec.description)
        self.assertIn("chassis-horizontal", params["x"]["description"])
        self.assertIn("not chassis-horizontal", params["forward"]["description"])
        self.assertEqual(params["x"]["coordinate_frame"], "robot_base")
        self.assertEqual(
            params["forward"]["coordinate_frame"],
            "starting_head_camera",
        )

    def test_head_adjustment_reports_robot_base_xyz_mode(self) -> None:
        adapter, world = self._adapter_world()
        self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_left_eef_pose_in_head_frame"
        ].fn(
            ctx,
            y=0.03,
            max_steps=120,
        )

        actions = self._drive_actions(adapter, world, generator)

        self.assertTrue(result["ok"], result)
        self.assertGreaterEqual(len(actions), 2)
        self.assertEqual(result["pose_control"]["settle_steps_executed"], 0)
        self.assertEqual(result["translation_input_mode"], "robot_base_xyz")
        self.assertEqual(result["translation_frame"], "robot_base")
        self.assertEqual(result["translation_m"], {"x": 0.0, "y": 0.03, "z": 0.0})
        np.testing.assert_allclose(
            result["delta_robot_base_m"],
            [0.0, 0.03, 0.0],
            atol=1e-9,
        )

    def test_head_adjustment_rejects_mixed_frames_without_motion(self) -> None:
        adapter, world = self._adapter_world()
        self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)

        actions = list(
            build_registry(adapter)["adjust_right_eef_pose_in_head_frame"].fn(
                ctx,
                x=0.0,
                forward=0.0,
            )
        )

        self.assertFalse(result["ok"])
        self.assertIn("mutually exclusive", result["error"])
        self.assertEqual(len(actions), 1)
        np.testing.assert_allclose(actions[0], world.hold_action(), atol=0.0)

    def test_right_wrist_adjustment_closes_observation_action_loop(self) -> None:
        adapter, world = self._adapter_world()
        _arm_start, _camera_poses = self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_right_eef_pose_in_wrist_frame"
        ].fn(
            ctx,
            forward=0.03,
            max_steps=120,
        )

        actions = self._drive_actions(adapter, world, generator)

        self.assertTrue(result["ok"], result)
        self.assertGreater(len(actions), 3)
        self.assertEqual(result["delta_camera_m"], [0.0, 0.0, -0.03])
        self.assertTrue(result["translation_control"]["observation_action_loop"])
        self.assertEqual(
            result["eef_after"]["source"],
            "submission_local_static_urdf_fk",
        )
        self.assertFalse(result["simulator_fk_ik_used"])
        self.assertFalse(result["simulator_eef_pose_used"])
        self.assertFalse(result["direct_simulator_mutation"])
        for action in actions:
            self.assertEqual(np.asarray(action).shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
            if ARM_DOF == 8:
                right = action[world.controller_action_idx("arm_right")]
                self.assertEqual(float(right[7]), 0.0)

    def test_left_head_adjustment_uses_local_fk_and_head_frame(self) -> None:
        adapter, world = self._adapter_world()
        self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_left_eef_pose_in_head_frame"
        ].fn(
            ctx,
            forward=0.03,
            max_steps=120,
        )

        actions = self._drive_actions(adapter, world, generator)

        self.assertTrue(result["ok"], result)
        self.assertGreater(len(actions), 3)
        self.assertEqual(result["delta_camera_m"], [0.0, 0.0, -0.03])
        self.assertEqual(
            result["kinematics_source"],
            "submission_local_static_urdf_fk",
        )
        self.assertEqual(len(result["eef_before"]), 3)
        self.assertEqual(len(result["quat_before"]), 4)
        self.assertEqual(result["delta_world_m"], result["delta_robot_m"])
        self.assertEqual(result["camera_before"]["source"], "evaluator_cam_rel_poses")
        control = result["pose_control"]
        self.assertAlmostEqual(ADJUST_POSE_MAX_JOINT_STEP_RAD, 0.028)
        self.assertEqual(
            control["relinearization_multiplier"],
            official_v2_tools.ADJUST_EEF_GUIDED_RELINEARIZATION_MULTIPLIER,
        )
        self.assertAlmostEqual(control["requested_distance_m"], 0.03, places=8)
        self.assertGreater(control["actual_distance_m"], 0.0)
        self.assertGreater(control["projected_progress_m"], 0.0)
        self.assertGreater(control["progress_ratio"], 0.0)
        self.assertLess(control["lateral_error_m"], 0.02)
        self.assertIsNone(result["failure_reason"])
        self.assertFalse(result["external_object_motion_checked"])
        for action in actions:
            if ARM_DOF == 8:
                left = action[world.controller_action_idx("arm_left")]
                self.assertEqual(float(left[7]), 0.0)

    def test_head_adjustment_adapts_step_only_after_good_tracking(self) -> None:
        adapter, world = self._adapter_world()
        arm_start, _camera_poses = self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_left_eef_pose_in_head_frame"
        ].fn(
            ctx,
            x=0.12,
            max_steps=120,
        )

        actions = self._drive_actions(adapter, world, generator)

        self.assertTrue(result["ok"], result)
        control = result["pose_control"]
        self.assertEqual(control["adaptive_step_scale_max"], 2.0)
        self.assertEqual(control["adaptive_step_scale_limit"], 2.0)
        self.assertEqual(control["settle_steps_executed"], 0)
        target_ik = control["target_ik"]
        self.assertEqual(target_ik["max_function_evaluations"], 192)
        self.assertLessEqual(
            target_ik["iterations"],
            target_ik["max_function_evaluations"],
        )
        previous = np.asarray(arm_start, dtype=np.float64)
        command_steps = []
        for action in actions:
            command = np.asarray(
                action[ACTION_SLICES["arm_left"]],
                dtype=np.float64,
            )
            command_steps.append(
                float(np.linalg.norm(command - previous, ord=np.inf))
            )
            previous = command
        self.assertGreater(
            command_steps[0],
            ADJUST_POSE_MAX_JOINT_STEP_RAD + 1e-3,
        )
        self.assertLessEqual(
            command_steps[0],
            2.0 * ADJUST_POSE_MAX_JOINT_STEP_RAD + 2e-6,
        )
        self.assertGreater(
            max(command_steps),
            ADJUST_POSE_MAX_JOINT_STEP_RAD + 1e-3,
        )
        self.assertLessEqual(
            max(command_steps),
            2.0 * ADJUST_POSE_MAX_JOINT_STEP_RAD + 2e-6,
        )

        good = {
            "command_progress_ratio": 0.95,
            "tracking_error_inf_rad": 0.002,
        }
        poor = {
            "command_progress_ratio": 0.30,
            "tracking_error_inf_rad": 0.040,
        }
        self.assertEqual(
            official_v2_tools._adjust_next_step_scale(
                1.0,
                good,
                near_target=False,
            ),
            1.5,
        )
        self.assertEqual(
            official_v2_tools._adjust_next_step_scale(
                2.0,
                poor,
                near_target=False,
            ),
            1.0,
        )
        self.assertEqual(
            official_v2_tools._adjust_next_step_scale(
                2.0,
                good,
                near_target=True,
            ),
            1.0,
        )

    @unittest.skipUnless(ARM_DOF == 8, "replay uses the official 8DOF profile")
    def test_right_head_adjustment_replays_job_1787489395353_efficiently(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        _arm_start, camera_poses = self._set_adjust_ready_observation(adapter)
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [
            1.8325998783111572,
            -2.2576241493225098,
            -1.8325997591018677,
            2.1841442432446456e-08,
        ]
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = [
            -7.954692904377225e-09,
            2.115194419616273e-08,
            -7.666495882574509e-09,
            -2.0943949222564697,
            -8.958492117017158e-07,
            -1.0471972227096558,
            6.350335013394215e-08,
            0.0,
        ]
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
        adapter.update(
            {
                "robot_r1::proprio": proprio,
                "robot_r1::cam_rel_poses": camera_poses,
            }
        )
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_right_eef_pose_in_head_frame"
        ].fn(
            ctx,
            x=0.10,
            y=0.10,
            pos_tol=0.012,
            ori_tol_deg=3.0,
            max_steps=240,
            timeout_s=60.0,
        )

        actions = []
        for action in generator:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            for side in ("left", "right"):
                q_slice = PROPRIO_SLICES[f"arm_{side}_qpos"]
                q_current = proprio[q_slice].copy()
                q_target = action[ACTION_SLICES[f"arm_{side}"]]
                proprio[q_slice] = q_current + 0.30 * (
                    q_target - q_current
                )
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})

        self.assertTrue(result["ok"], result)
        control = result["pose_control"]
        self.assertTrue(control["target_ik"]["ok"])
        self.assertEqual(
            control["target_ik"]["residual_weighting"],
            "equal_error_at_requested_tolerances",
        )
        self.assertLess(control["target_ik"]["pos_err_m"], 0.012)
        self.assertLess(control["target_ik"]["ori_err_deg"], 3.0)
        self.assertLessEqual(control["steps_executed"], 30)
        self.assertGreater(control["guided_steps_executed"], 0)
        self.assertEqual(control["tangent_steps_executed"], 0)
        self.assertLessEqual(control["final_position_error_m"], 0.012)
        self.assertLessEqual(control["final_orientation_error_deg"], 3.0)
        for action in actions:
            self.assertEqual(
                float(action[ACTION_SLICES["arm_right"]][7]),
                0.0,
            )

    def test_head_adjustment_stops_near_zero_solver_commands(self) -> None:
        adapter, world = self._adapter_world()
        self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)

        def no_terminal_solution(_state, _arm, q_seed, **_kwargs):
            return np.asarray(q_seed, dtype=np.float64), {
                "ok": False,
                "pos_err_m": 0.03,
                "ori_err_deg": 0.0,
            }

        def zero_tangent_step(*_args, **_kwargs):
            return np.zeros(7, dtype=np.float64), {
                "command_dq_inf_rad": 0.0,
                "target_ik_guidance_used": False,
            }

        with mock.patch.object(
            official_v2_tools,
            "solve_pose_target",
            no_terminal_solution,
        ), mock.patch.object(
            official_v2_tools,
            "bounded_pose_tangent_step",
            zero_tangent_step,
        ):
            self._drive_actions(
                adapter,
                world,
                build_registry(adapter)[
                    "adjust_left_eef_pose_in_head_frame"
                ].fn(
                    ctx,
                    x=0.03,
                    max_steps=120,
                ),
            )

        self.assertFalse(result["ok"], result)
        control = result["pose_control"]
        self.assertEqual(control["reason"], "near_zero_command_stall")
        self.assertEqual(control["steps_executed"], 3)
        self.assertEqual(control["near_zero_command_steps"], 3)
        self.assertEqual(control["pose_stall_limit"], 6)
        self.assertTrue(control["orientation_divergence_guard_active"])
        self.assertEqual(control["orientation_divergence_limit_deg"], 12.0)

    def test_head_adjustment_stops_unguided_orientation_divergence(self) -> None:
        adapter, world = self._adapter_world()
        self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        orientation_calls = 0

        def no_terminal_solution(_state, _arm, q_seed, **_kwargs):
            return np.asarray(q_seed, dtype=np.float64), {
                "ok": False,
                "pos_err_m": 0.03,
                "ori_err_deg": 0.0,
            }

        def small_tangent_step(*_args, **_kwargs):
            delta = np.zeros(7, dtype=np.float64)
            delta[0] = 0.004
            return delta, {
                "command_dq_inf_rad": 0.004,
                "target_ik_guidance_used": False,
            }

        def diverging_orientation(*_args, **_kwargs):
            nonlocal orientation_calls
            orientation_calls += 1
            return 0.0 if orientation_calls == 1 else 13.0

        with mock.patch.object(
            official_v2_tools,
            "solve_pose_target",
            no_terminal_solution,
        ), mock.patch.object(
            official_v2_tools,
            "bounded_pose_tangent_step",
            small_tangent_step,
        ), mock.patch.object(
            official_v2_tools,
            "orientation_error_deg",
            diverging_orientation,
        ):
            actions = self._drive_actions(
                adapter,
                world,
                build_registry(adapter)[
                    "adjust_left_eef_pose_in_head_frame"
                ].fn(
                    ctx,
                    x=0.03,
                    max_steps=120,
                ),
            )

        self.assertFalse(result["ok"], result)
        control = result["pose_control"]
        self.assertEqual(control["reason"], "orientation_error_diverged")
        self.assertEqual(control["steps_executed"], 1)
        self.assertEqual(control["tangent_steps_executed"], 1)
        self.assertEqual(control["settle_steps_executed"], 2)
        self.assertEqual(control["orientation_divergence_limit_deg"], 12.0)
        self.assertEqual(control["max_observed_orientation_error_deg"], 13.0)
        self.assertEqual(len(actions), 4)

    def test_head_adjustment_rejects_gross_terminal_ik_before_arm_motion(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        arm_start, _camera_poses = self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)

        def grossly_unreachable(_state, _arm, q_seed, **_kwargs):
            return np.asarray(q_seed, dtype=np.float64), {
                "ok": False,
                "pos_err_m": 0.080,
                "ori_err_deg": 15.0,
            }

        with mock.patch.object(
            official_v2_tools,
            "solve_pose_target",
            grossly_unreachable,
        ), mock.patch.object(
            official_v2_tools,
            "bounded_pose_tangent_step",
        ) as tangent_step:
            actions = self._drive_actions(
                adapter,
                world,
                build_registry(adapter)[
                    "adjust_right_eef_pose_in_head_frame"
                ].fn(
                    ctx,
                    x=0.40,
                    y=0.20,
                    z=0.20,
                    max_steps=240,
                    timeout_s=60.0,
                ),
            )

        self.assertFalse(result["ok"], result)
        control = result["pose_control"]
        self.assertEqual(control["reason"], "terminal_ik_grossly_unreachable")
        self.assertTrue(control["gross_preflight_rejection"])
        self.assertTrue(control["target_ik"]["motion_rejected"])
        self.assertEqual(control["steps_planned"], 0)
        self.assertEqual(control["steps_executed"], 0)
        self.assertEqual(control["settle_steps_executed"], 2)
        tangent_step.assert_not_called()
        self.assertEqual(len(actions), 3)
        for action in actions:
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_right"]],
                arm_start,
                atol=1e-7,
            )
        self.assertIn("rejected before arm motion", result["error"])

    def test_head_adjustment_gross_preflight_threshold_keeps_near_miss_fallback(
        self,
    ) -> None:
        limit = official_v2_tools.ADJUST_EEF_GROSS_PREFLIGHT_MAX_NORMALIZED_ERROR
        self.assertEqual(limit, 4.0)
        self.assertLess(14.5 / 12.0, limit)
        self.assertLess(0.17 / 3.0, limit)

    @unittest.skipUnless(ARM_DOF == 8, "replay uses the official 8DOF profile")
    def test_right_head_adjustment_rejects_15061_unreachable_large_move(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        _arm_start, camera_poses = self._set_adjust_ready_observation(adapter)
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [
            0.45003819465637207,
            -0.39992764592170715,
            -2.8531416319310665e-05,
            0.0,
        ]
        right_start = np.asarray(
            [
                3.901009768014774e-05,
                0.00014160506543703377,
                -0.0007428707904182374,
                -2.0943989753723145,
                -0.044697608798742294,
                -1.0472099781036377,
                0.004513951018452644,
                0.0,
            ],
            dtype=np.float32,
        )
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = right_start
        proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [
            0.049786169081926346,
            0.04926072806119919,
        ]
        camera_poses[-7:] = [
            0.2134416103363037,
            9.5367431640625e-06,
            1.5701589584350586,
            0.3909298777580261,
            -0.3909412920475006,
            -0.5892155170440674,
            0.5892060995101929,
        ]
        adapter.update(
            {
                "robot_r1::proprio": proprio,
                "robot_r1::cam_rel_poses": camera_poses,
            }
        )
        ctx, result = self._ctx(world)

        actions = self._drive_actions(
            adapter,
            world,
            build_registry(adapter)[
                "adjust_right_eef_pose_in_head_frame"
            ].fn(
                ctx,
                forward=0.4,
                leftward=0.2,
                upward=0.2,
                pos_tol=0.012,
                ori_tol_deg=3.0,
                max_steps=240,
                timeout_s=60.0,
            ),
        )

        self.assertFalse(result["ok"], result)
        control = result["pose_control"]
        self.assertEqual(control["reason"], "terminal_ik_grossly_unreachable")
        self.assertTrue(control["gross_preflight_rejection"])
        self.assertGreater(
            control["target_ik"]["candidate_normalized_max_error"],
            control["gross_preflight_normalized_error_limit"],
        )
        self.assertEqual(control["steps_executed"], 0)
        self.assertEqual(control["tangent_steps_executed"], 0)
        self.assertEqual(len(actions), 3)
        for action in actions:
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_right"]],
                right_start,
                atol=1e-7,
            )

    def test_wrist_translation_and_rotation_share_one_deadline(self) -> None:
        adapter, world = self._adapter_world()
        self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        deadlines = []
        skip_success_settle = []

        def fake_pose_controller(*_args, **kwargs):
            deadlines.append(kwargs["deadline_monotonic"])
            skip_success_settle.append(kwargs["skip_success_settle"])
            if False:
                yield None
            return {
                "ok": False,
                "reason": "test_pose_stop",
                "last_observation_sequence": kwargs["observation_sequence"],
            }

        def fake_orientation_controller(*_args, **kwargs):
            deadlines.append(kwargs["deadline_monotonic"])
            if False:
                yield None
            return {
                "ok": False,
                "reason": "test_orientation_stop",
                "last_observation_sequence": kwargs["observation_sequence"],
            }

        with mock.patch.object(
            official_v2_tools,
            "_execute_local_pose_closed_loop",
            fake_pose_controller,
        ), mock.patch.object(
            official_v2_tools,
            "_execute_local_j567_closed_loop",
            fake_orientation_controller,
        ):
            actions = self._drive_actions(
                adapter,
                world,
                build_registry(adapter)[
                    "adjust_right_eef_pose_in_wrist_frame"
                ].fn(
                    ctx,
                    forward=0.02,
                    yaw=4.0,
                    timeout_s=10.0,
                ),
            )

        self.assertEqual(len(actions), 1)
        self.assertEqual(len(deadlines), 2)
        self.assertEqual(deadlines[0], deadlines[1])
        self.assertEqual(skip_success_settle, [True])
        self.assertTrue(result["translation_and_orientation_share_timeout"])
        self.assertEqual(result["shared_timeout_budget_s"], 10.0)

    def test_head_adjustment_failure_reports_progress_and_fixed_pose_reason(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_left_eef_pose_in_head_frame"
        ].fn(
            ctx,
            x=0.10,
            max_steps=1,
        )

        self._drive_actions(adapter, world, generator)

        self.assertFalse(result["ok"], result)
        self.assertEqual(result["pose_control"]["steps_executed"], 1)
        self.assertIn("measured requested-direction progress", result["error"])
        self.assertIn("final position error", result["error"])
        self.assertEqual(result["failure_reason"], result["error"])
        self.assertFalse(result["external_object_motion_checked"])

    def test_head_adjustment_releases_blocked_arm_after_three_observations(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        arm_start, _camera_poses = self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_left_eef_pose_in_head_frame"
        ].fn(
            ctx,
            forward=0.03,
            max_steps=120,
        )

        actions = []
        for action in generator:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["arm_left_qpos"]] = arm_start
            proprio[PROPRIO_SLICES["arm_left_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})

        control = result["pose_control"]
        self.assertFalse(result["ok"], result)
        self.assertEqual(control["reason"], "proprio_stall")
        self.assertEqual(control["steps_executed"], 3)
        self.assertLessEqual(control["settle_steps_executed"], 2)
        self.assertEqual(control["safety"]["proprio_stall_steps"], 3)
        for action in actions[-3:]:
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_left"]],
                arm_start,
                atol=1e-7,
            )

    def test_head_adjustment_treats_finite_impedance_lag_as_diagnostic(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_right_eef_pose_in_head_frame"
        ].fn(
            ctx,
            forward=0.03,
            max_steps=120,
        )

        actions = []
        # A finite-effort position controller normally realizes only part of a
        # one-frame target. Lowering the diagnostic threshold makes this test
        # exercise the exact old three-frame false-abort path deterministically.
        with mock.patch.object(
            official_v2_tools,
            "ADJUST_EEF_MAX_TRACKING_ERROR_RAD",
            0.001,
        ):
            for action in generator:
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                actions.append(action.copy())
                proprio = adapter.proprio_vector().copy()
                for side in ("left", "right"):
                    q_slice = PROPRIO_SLICES[f"arm_{side}_qpos"]
                    q_current = proprio[q_slice].copy()
                    q_target = action[ACTION_SLICES[f"arm_{side}"]]
                    proprio[q_slice] = q_current + 0.80 * (
                        q_target - q_current
                    )
                    proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
                adapter.update({"robot_r1::proprio": proprio})

        control = result["pose_control"]
        self.assertTrue(result["ok"], result)
        self.assertGreater(control["steps_executed"], 3)
        self.assertGreater(control["max_tracking_error_inf_rad"], 0.001)
        self.assertTrue(
            control["safety"]["tracking_error_is_diagnostic_only"]
        )
        self.assertNotEqual(control["reason"], "proprio_tracking_error")
        self.assertGreater(len(actions), 3)

    def test_adjust_tracking_guard_replays_job_1786428495683(self) -> None:
        adapter, world = self._adapter_world()
        self._set_adjust_ready_observation(adapter)
        ctx, _result = self._ctx(world)
        before = official_v2_tools._adjust_kinematic_observation(ctx, "right")
        monitor = official_v2_tools._adjust_safety_monitor(ctx, before)
        command = before["q"].copy()
        command[0] += 0.011798067576574534
        after = dict(before)
        after["q"] = before["q"].copy()
        after["q"][0] += 0.011798067576574534 * 0.21808166531992718
        after["q"][1] -= 0.018230124003576487

        reason = None
        observation = {}
        for _ in range(3):
            reason, observation = official_v2_tools._adjust_safety_after_action(
                ctx,
                monitor,
                before=before,
                command=command,
                after=after,
            )

        self.assertIsNone(reason)
        self.assertEqual(monitor.tracking_error_steps, 3)
        self.assertEqual(monitor.proprio_stall_steps, 0)
        self.assertAlmostEqual(
            observation["command_progress_ratio"],
            0.21808166531992718,
        )
        self.assertAlmostEqual(
            observation["tracking_error_inf_rad"],
            0.018230124003576487,
        )

    def test_head_adjustment_stops_when_arm_reaction_moves_base(self) -> None:
        adapter, world = self._adapter_world()
        self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_left_eef_pose_in_head_frame"
        ].fn(
            ctx,
            forward=0.03,
            max_steps=120,
        )

        actions = []
        for action in generator:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            for side in ("left", "right"):
                proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                    ACTION_SLICES[f"arm_{side}"]
                ]
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            proprio[PROPRIO_SLICES["base_qvel"]] = [0.006, 0.0, 0.0]
            adapter.update({"robot_r1::proprio": proprio})

        control = result["pose_control"]
        self.assertFalse(result["ok"], result)
        self.assertEqual(control["reason"], "base_disturbance")
        self.assertEqual(control["steps_executed"], 3)
        self.assertEqual(control["safety"]["base_disturbance_steps"], 3)
        self.assertGreaterEqual(
            control["safety"]["max_base_linear_speed_mps"],
            0.0059,
        )
        np.testing.assert_allclose(
            actions[-1][ACTION_SLICES["arm_left"]],
            adapter.proprio_vector()[PROPRIO_SLICES["arm_left_qpos"]],
            atol=1e-7,
        )

    def test_head_adjustment_stops_on_trunk_drift(self) -> None:
        adapter, world = self._adapter_world()
        self._set_adjust_ready_observation(adapter)
        trunk_start = adapter.proprio_vector()[
            PROPRIO_SLICES["trunk_qpos"]
        ].copy()
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_left_eef_pose_in_head_frame"
        ].fn(
            ctx,
            forward=0.03,
            max_steps=120,
        )

        actions = []
        for action in generator:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            for side in ("left", "right"):
                proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = action[
                    ACTION_SLICES[f"arm_{side}"]
                ]
                proprio[PROPRIO_SLICES[f"arm_{side}_qvel"]] = 0.0
            if len(actions) <= 2:
                trunk = trunk_start.copy()
                trunk[0] += 0.003 * len(actions)
                proprio[PROPRIO_SLICES["trunk_qpos"]] = trunk
            adapter.update({"robot_r1::proprio": proprio})

        control = result["pose_control"]
        self.assertFalse(result["ok"], result)
        self.assertEqual(control["reason"], "trunk_disturbance")
        self.assertEqual(control["steps_executed"], 2)
        self.assertGreater(
            control["safety"]["max_trunk_q123_drift_inf_rad"],
            0.005,
        )

    def test_right_wrist_rotation_commands_only_j567_and_locks_j8(self) -> None:
        adapter, world = self._adapter_world()
        arm_start, _camera_poses = self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_right_eef_pose_in_wrist_frame"
        ].fn(
            ctx,
            yaw=5.0,
            max_steps=120,
        )

        actions = self._drive_actions(adapter, world, generator)

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["j567_only_orientation_control"])
        self.assertTrue(result["orientation_solve"]["ok"])
        self.assertEqual(result["build"], ADJUST_EEF_LOCAL_BUILD)
        self.assertEqual(
            result["orientation_solve"]["solver"],
            "submission_local_static_urdf_j567_v29",
        )
        self.assertEqual(
            result["orientation_solve"]["only_optimized_joints"],
            [5, 6, 7],
        )
        self.assertGreater(result["orientation_solve"]["probe_count"], 0)
        self.assertLessEqual(
            float(
                np.max(
                    np.abs(
                        np.asarray(
                            result["orientation_solve"]["j567_rad"],
                            dtype=np.float64,
                        )
                        - np.asarray(arm_start[4:7], dtype=np.float64)
                    )
                )
            ),
            0.20,
        )
        self.assertEqual(
            result["orientation_solve"]["commanded_joint_numbers"],
            [5, 6, 7],
        )
        self.assertTrue(result["j1234_held"])
        self.assertEqual(result["j1234_measured_drift_inf_rad"], 0.0)
        for action in actions:
            right = action[world.controller_action_idx("arm_right")]
            np.testing.assert_allclose(right[:4], arm_start[:4], atol=1e-6)
            if ARM_DOF == 8:
                self.assertEqual(float(right[7]), 0.0)

    def test_wrist_j567_releases_blocked_arm_after_three_observations(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        arm_start, _camera_poses = self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        generator = build_registry(adapter)[
            "adjust_right_eef_pose_in_wrist_frame"
        ].fn(
            ctx,
            yaw=5.0,
            max_steps=120,
        )

        actions = []
        for action in generator:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            actions.append(action.copy())
            proprio = adapter.proprio_vector().copy()
            proprio[PROPRIO_SLICES["arm_right_qpos"]] = arm_start
            proprio[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": proprio})

        orientation = result["wrist_joint_motion"]["arm_joints"]
        self.assertFalse(result["ok"], result)
        self.assertEqual(orientation["reason"], "proprio_stall")
        self.assertEqual(orientation["controller_steps"], 3)
        self.assertLessEqual(orientation["settle_steps"], 2)
        self.assertEqual(orientation["safety"]["proprio_stall_steps"], 3)
        for action in actions[-3:]:
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_right"]],
                arm_start,
                atol=1e-7,
            )

    def test_adjustment_requires_camera_pose_and_new_observations(self) -> None:
        adapter, world = self._adapter_world()
        ctx, result = self._ctx(world)
        missing_camera_actions = list(
            build_registry(adapter)[
                "adjust_right_eef_pose_in_wrist_frame"
            ].fn(ctx, forward=0.03)
        )
        self.assertFalse(result["ok"])
        self.assertIn("camera-relative pose is unavailable", result["error"])
        self.assertEqual(len(missing_camera_actions), 1)

        self._set_adjust_ready_observation(adapter)
        ctx, result = self._ctx(world)
        stale_actions = list(
            build_registry(adapter)[
                "adjust_right_eef_pose_in_wrist_frame"
            ].fn(ctx, forward=0.03, max_steps=20)
        )
        self.assertFalse(result["ok"])
        self.assertEqual(
            result["translation_control"]["reason"],
            "observation_not_advanced",
        )
        self.assertGreaterEqual(len(stale_actions), 1)

    def test_capture_is_owned_and_persisted_by_official_v2(self) -> None:
        adapter, world = self._adapter_world()
        adapter.update(
            {
                "robot_r1::proprio": np.zeros(
                    PROPRIO_DIM,
                    dtype=np.float32,
                ),
                "robot_r1::cam_rel_poses": np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ]
                    * 3,
                    dtype=np.float32,
                ),
                "robot_r1::head::rgb": np.zeros(
                    (12, 16, 3),
                    dtype=np.uint8,
                ),
                "robot_r1::head::depth_linear": np.ones(
                    (12, 16),
                    dtype=np.float32,
                ),
            }
        )
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ), mock.patch(
                "behavior_interface_eval_test.tool.official_v2.tools."
                "_stamp_selected_minimap_on_bgr",
                side_effect=lambda image: np.full_like(image, 91),
            ):
                ctx, result = self._ctx(world)
                actions = list(
                    registry["capture_head_camera"].fn(
                        ctx,
                        session_id="self_contained",
                    )
                )
                self.assertEqual(len(actions), 1)
                self.assertTrue(result["ok"])
                self.assertTrue(Path(result["rgb_path"]).is_file())
                self.assertTrue(Path(result["depth_path"]).is_file())
                self.assertTrue(
                    (
                        Path(temp_root)
                        / "self_contained"
                        / "images"
                        / f"{result['image_id']}.meta.json"
                    ).is_file()
                )
                meta_path = (
                    Path(temp_root)
                    / "self_contained"
                    / "images"
                    / f"{result['image_id']}.meta.json"
                )
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    meta["camera"]["robot_relative_pose"]["frame"],
                    "robot_base",
                )
                self.assertEqual(
                    meta["camera"]["robot_relative_pose"]["pos"],
                    [0.0, 0.0, 0.0],
                )
                self.assertTrue(result["base_path_overlay"]["ok"])
                self.assertFalse(
                    result["base_path_overlay"]["yellow_overlay_enabled"]
                )
                self.assertEqual(
                    result["base_path_overlay"]["yellow_overlay_semantics"],
                    "removed",
                )
                self.assertTrue(
                    result["base_path_overlay"]["distance_scale_enabled"]
                )
                self.assertTrue(
                    result["base_path_overlay"]["direction_arrows_enabled"]
                )
                self.assertTrue(Path(result["rgb_overlay_path"]).is_file())
                self.assertNotEqual(result["rgb_overlay_path"], result["rgb_path"])
                self.assertTrue(result["spatial_map_overlay_applied"])
                raw = cv2.imread(result["rgb_path"], cv2.IMREAD_COLOR)
                display = cv2.imread(result["rgb_overlay_path"], cv2.IMREAD_COLOR)
                self.assertTrue(np.all(raw == 0))
                self.assertTrue(np.all(display == 91))
                memory = result["memory"]
                self.assertEqual(memory["task"], "make_microwave_popcorn")
                self.assertIn("popcorn bag", memory["task_objective"])
                self.assertIn("Task objective:", memory["instruction"])
                self.assertEqual(
                    memory["bddl_conditions"],
                    [
                        "cooked popcorn 1 exists",
                        "popcorn bag 1 contains cooked popcorn 1",
                    ],
                )
                self.assertFalse(memory["bddl_live_state_available"])

    def test_detached_capture_releases_evaluator_generator_before_media_finishes(
        self,
    ) -> None:
        """The official callback must not wait on the expensive artifact worker."""

        _adapter, world = self._adapter_world()
        world._official_async_capture_artifacts = True
        world._official_detached_capture_results = True
        ctx, result = self._ctx(world)
        worker_future = concurrent.futures.Future()
        worker = lambda: {
            "ok": True,
            "feed": "head",
            "image_id": "img_detached",
            "rgb_main_path": "/tmp/img_detached.png",
        }

        with mock.patch.object(
            official_v2_tools,
            "_submit_capture_artifact_worker",
            return_value=(worker_future, False),
        ):
            generator = official_v2_tools._yield_capture_artifact(
                ctx,
                "capture_head_camera",
                worker,
            )
            next(generator)
            token = result.get("_official_capture_pending_token")
            self.assertTrue(token)
            # A second next() must finish the skill even while the renderer is
            # deliberately unresolved.
            with self.assertRaises(StopIteration):
                next(generator)
            self.assertEqual(result["tool"], "capture_head_camera")

            worker_future.set_result(worker())
            final = official_v2_tools.wait_for_capture_artifact(token, 1.0)

        self.assertTrue(final["ok"], final)
        self.assertEqual(final["image_id"], "img_detached")
        self.assertNotIn("_official_capture_pending_token", final)
        self.assertEqual(result["image_id"], "img_detached")
        with official_v2_tools._CAPTURE_ARTIFACT_PENDING_LOCK:
            official_v2_tools._CAPTURE_ARTIFACT_PENDING.pop(token, None)

    def test_capture_artifact_pruning_retires_only_stale_completed_orphan(
        self,
    ) -> None:
        completed = concurrent.futures.Future()
        active = concurrent.futures.Future()
        worker = lambda: {"ok": True}
        completed_token = official_v2_tools._register_capture_artifact(
            completed,
            isolated=False,
            worker=worker,
            tool_name="capture_head_camera",
            finalize=lambda result: result,
        )
        active_token = official_v2_tools._register_capture_artifact(
            active,
            isolated=False,
            worker=worker,
            tool_name="capture_head_camera",
            finalize=lambda result: result,
        )
        completed.set_result(worker())
        stale_age = official_v2_tools._CAPTURE_ARTIFACT_PENDING_TTL_S + 1.0
        with official_v2_tools._CAPTURE_ARTIFACT_PENDING_LOCK:
            completed_entry = official_v2_tools._CAPTURE_ARTIFACT_PENDING[
                completed_token
            ]
            active_entry = official_v2_tools._CAPTURE_ARTIFACT_PENDING[
                active_token
            ]
            completed_entry.created_mono -= stale_age
            active_entry.created_mono -= stale_age
        try:
            self.assertEqual(
                official_v2_tools.capture_artifact_pending_count(),
                1,
            )
            with official_v2_tools._CAPTURE_ARTIFACT_PENDING_LOCK:
                self.assertNotIn(
                    completed_token,
                    official_v2_tools._CAPTURE_ARTIFACT_PENDING,
                )
                self.assertIn(
                    active_token,
                    official_v2_tools._CAPTURE_ARTIFACT_PENDING,
                )
        finally:
            active.cancel()
            with official_v2_tools._CAPTURE_ARTIFACT_PENDING_LOCK:
                official_v2_tools._CAPTURE_ARTIFACT_PENDING.pop(
                    completed_token,
                    None,
                )
                official_v2_tools._CAPTURE_ARTIFACT_PENDING.pop(
                    active_token,
                    None,
                )

    def test_capture_status_reports_worker_priority_override(self) -> None:
        world = SimpleNamespace(
            _official_async_capture_artifacts=True,
            _official_detached_capture_results=True,
        )
        with mock.patch.dict(
            os.environ,
            {"BEHAVIOR_OFFICIAL_CAPTURE_NICE": "13"},
        ):
            status = official_policy_interface._capture_artifact_status(world)
        self.assertEqual(status["worker_nice_target"], 13)
        self.assertEqual(
            status["worker_policy"],
            "spawn_process_nice13_cv2_threads1",
        )

    def test_presentation_worker_priority_is_thread_local(self) -> None:
        getpriority = getattr(os, "getpriority", None)
        if not callable(getpriority):
            self.skipTest("native per-thread priority query is unavailable")
        which = getattr(os, "PRIO_PROCESS", 0)
        main_before = int(getpriority(which, 0))
        observed: dict[str, int] = {}

        def initialize_worker() -> None:
            observed["before"] = int(getpriority(which, 0))
            official_policy_interface._presentation_worker_initializer()
            observed["after"] = int(getpriority(which, 0))

        with mock.patch.dict(
            os.environ,
            {"BEHAVIOR_OFFICIAL_PRESENTATION_NICE": "13"},
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(initialize_worker).result(timeout=5.0)

        self.assertEqual(int(getpriority(which, 0)), main_before)
        self.assertGreaterEqual(observed["after"], observed["before"])
        self.assertEqual(observed["after"], 13)

    def test_move_point_to_point_default_and_above_target_offset(self) -> None:
        for above_target_point_m in (0.0, 0.02):
            with self.subTest(above_target_point_m=above_target_point_m):
                adapter, world = self._adapter_world()
                self._set_move_point_ready_observation(adapter)
                registry = build_registry(adapter)
                with tempfile.TemporaryDirectory() as temp_root:
                    with mock.patch.dict(
                        os.environ,
                        {"BEHAVIOR_AGENT_RUNS": temp_root},
                    ):
                        capture = self._capture_move_point(
                            registry,
                            world,
                            "move_point_offset",
                        )
                        ctx, result = self._ctx(world)
                        actions = self._drive_actions(
                            adapter,
                            world,
                            registry["move_point_to_point"].fn(
                                ctx,
                                session_id="move_point_offset",
                                image_id=capture["image_id"],
                                points=[[500, 500], [500, 500]],
                                above_target_point_m=above_target_point_m,
                                max_steps=120,
                            ),
                        )

                        self.assertTrue(result["ok"], result)
                        self.assertEqual(
                            result["above_target_point_m"],
                            above_target_point_m,
                        )
                        self.assertAlmostEqual(
                            result["delta_robot_base_m"][2],
                            above_target_point_m,
                            places=7,
                        )
                        self.assertAlmostEqual(
                            result["translation_m"],
                            above_target_point_m,
                            places=7,
                        )
                        self.assertTrue(result["point_reached"])
                        self.assertEqual(
                            result["trajectory_schema"],
                            "official_v2_move_point_trajectory",
                        )
                        self.assertTrue(Path(result["plan_record_path"]).is_file())
                        self.assertEqual(len(result["trajectory_digest"]), 64)
                        self.assertEqual(result["stability"]["stable_steps"], 2)
                        self.assertEqual(
                            result["stability"]["arm_qvel_threshold_rad_s"],
                            0.03,
                        )
                        self.assertFalse(result["direct_simulator_mutation"])
                        self.assertFalse(result["grasp_truth_used"])
                        self.assertFalse(result["collision_checked"])
                        self.assertGreaterEqual(len(actions), 4)
                        for action in actions:
                            self.assertEqual(action.shape, (ACTION_DIM,))
                            self.assertTrue(np.isfinite(action).all())
                            np.testing.assert_allclose(
                                action[ACTION_SLICES["base"]],
                                0.0,
                            )
                            if ARM_DOF == 8:
                                for side in ("left", "right"):
                                    self.assertEqual(
                                        float(
                                            action[
                                                ACTION_SLICES[f"arm_{side}"]
                                            ][7]
                                        ),
                                        0.0,
                                    )

    def test_move_point_to_point_rejects_stale_episode_and_tampered_plan(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        self._set_move_point_ready_observation(adapter)
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture = self._capture_move_point(
                    registry,
                    world,
                    "move_point_stale",
                )
                world.reset_observation_state()
                world.set_episode_initialized(True)
                ctx, stale = self._ctx(world)
                actions = list(
                    registry["move_point_to_point"].fn(
                        ctx,
                        session_id="move_point_stale",
                        image_id=capture["image_id"],
                        points=[[500, 500], [500, 500]],
                        max_steps=120,
                    )
                )
                self.assertFalse(stale["ok"])
                self.assertEqual(
                    stale["failure_stage"],
                    "start-state validation",
                )
                self.assertEqual(len(actions), 1)

                world.reset_observation_state()
                world.set_episode_initialized(True)
                self._set_move_point_ready_observation(adapter)
                capture = self._capture_move_point(
                    registry,
                    world,
                    "move_point_tamper",
                )
                real_loader = official_v2_tools._move_point_load_trajectory

                def tampering_loader(session_id, plan_id):
                    path = Path(temp_root) / session_id / "plans" / f"{plan_id}.json"
                    record = json.loads(path.read_text(encoding="utf-8"))
                    record["trajectory"]["waypoints"][0][0] += 0.01
                    path.write_text(json.dumps(record), encoding="utf-8")
                    return real_loader(session_id, plan_id)

                ctx, tampered = self._ctx(world)
                with mock.patch.object(
                    official_v2_tools,
                    "_move_point_load_trajectory",
                    side_effect=tampering_loader,
                ):
                    actions = list(
                        registry["move_point_to_point"].fn(
                            ctx,
                            session_id="move_point_tamper",
                            image_id=capture["image_id"],
                            points=[[500, 500], [500, 500]],
                            max_steps=120,
                        )
                    )
                self.assertFalse(tampered["ok"])
                self.assertEqual(tampered["failure_stage"], "planning")
                self.assertIn("digest mismatch", tampered["error"])
                self.assertEqual(len(actions), 1)

    def test_move_point_to_point_rejects_invalid_frozen_rgbd(self) -> None:
        cases = {
            "missing_camera": {
                "include_camera_pose": False,
            },
            "resolution_mismatch": {
                "rgb": np.zeros((11, 16, 3), dtype=np.uint8),
            },
            "invalid_depth": {
                "depth": np.full((12, 16), np.nan, dtype=np.float32),
            },
        }
        for name, observation_kwargs in cases.items():
            with self.subTest(name=name):
                adapter, world = self._adapter_world()
                self._set_move_point_ready_observation(
                    adapter,
                    **observation_kwargs,
                )
                registry = build_registry(adapter)
                with tempfile.TemporaryDirectory() as temp_root:
                    with mock.patch.dict(
                        os.environ,
                        {"BEHAVIOR_AGENT_RUNS": temp_root},
                    ):
                        capture = self._capture_move_point(
                            registry,
                            world,
                            f"move_point_{name}",
                        )
                        ctx, result = self._ctx(world)
                        actions = list(
                            registry["move_point_to_point"].fn(
                                ctx,
                                session_id=f"move_point_{name}",
                                image_id=capture["image_id"],
                                points=[[500, 500], [500, 500]],
                                max_steps=120,
                            )
                        )
                        self.assertFalse(result["ok"])
                        self.assertEqual(
                            result["failure_stage"],
                            "observation validation",
                        )
                        self.assertEqual(len(actions), 1)

    def test_move_point_to_point_detects_tracking_stall(self) -> None:
        adapter, world = self._adapter_world()
        self._set_move_point_ready_observation(adapter)
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture = self._capture_move_point(
                    registry,
                    world,
                    "move_point_stall",
                )
                ctx, result = self._ctx(world)
                actions = []
                for action in registry["move_point_to_point"].fn(
                    ctx,
                    session_id="move_point_stall",
                    image_id=capture["image_id"],
                    points=[[500, 500], [500, 500]],
                    above_target_point_m=0.02,
                    max_steps=120,
                ):
                    actions.append(
                        np.asarray(action, dtype=np.float32).reshape(-1).copy()
                    )
                    adapter.update(
                        {"robot_r1::proprio": adapter.proprio_vector().copy()}
                    )

                self.assertFalse(result["ok"], result)
                self.assertEqual(result["failure_stage"], "tracking")
                self.assertEqual(result["motion"]["reason"], "proprio_stall")
                self.assertEqual(
                    result["motion"]["safety"]["proprio_stall_steps"],
                    3,
                )
                self.assertGreaterEqual(len(actions), 4)
                for action in actions:
                    self.assertEqual(action.shape, (ACTION_DIM,))
                    self.assertTrue(np.isfinite(action).all())
                    np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
                    if ARM_DOF == 8:
                        for side in ("left", "right"):
                            self.assertEqual(
                                float(action[ACTION_SLICES[f"arm_{side}"]][7]),
                                0.0,
                            )

    def test_move_point_to_point_times_out_and_holds(self) -> None:
        adapter, world = self._adapter_world()
        self._set_move_point_ready_observation(adapter)
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture = self._capture_move_point(
                    registry,
                    world,
                    "move_point_timeout",
                )
                ctx, result = self._ctx(world)
                clock_calls = 0

                def fake_monotonic():
                    nonlocal clock_calls
                    clock_calls += 1
                    return 0.0 if clock_calls <= 2 else 1.0

                with mock.patch.object(
                    official_v2_tools.time,
                    "monotonic",
                    side_effect=fake_monotonic,
                ):
                    actions = list(
                        registry["move_point_to_point"].fn(
                            ctx,
                            session_id="move_point_timeout",
                            image_id=capture["image_id"],
                            points=[[500, 500], [500, 500]],
                            above_target_point_m=0.02,
                            max_steps=120,
                            timeout_s=0.1,
                        )
                    )

                self.assertFalse(result["ok"], result)
                self.assertEqual(result["failure_stage"], "timeout")
                self.assertEqual(result["motion"]["reason"], "timeout")
                self.assertEqual(result["stability"]["reason"], "timeout")
                self.assertEqual(len(actions), 1)
                action = np.asarray(actions[0], dtype=np.float32).reshape(-1)
                self.assertEqual(action.shape, (ACTION_DIM,))
                self.assertTrue(np.isfinite(action).all())
                if ARM_DOF == 8:
                    for side in ("left", "right"):
                        self.assertEqual(
                            float(action[ACTION_SLICES[f"arm_{side}"]][7]),
                            0.0,
                        )

    def test_move_point_to_point_cancellation_holds_observed_state(self) -> None:
        class SkillCancelled(RuntimeError):
            pass

        adapter, world = self._adapter_world()
        self._set_move_point_ready_observation(adapter)
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture = self._capture_move_point(
                    registry,
                    world,
                    "move_point_cancel",
                )
                result = {}

                def raise_cancelled(where=""):
                    raise SkillCancelled(f"cancelled at {where}")

                ctx = SimpleNamespace(
                    world=world,
                    task_name="make_microwave_popcorn",
                    set_result=lambda value: result.update(value),
                    raise_if_cancelled=raise_cancelled,
                )
                actions = list(
                    registry["move_point_to_point"].fn(
                        ctx,
                        session_id="move_point_cancel",
                        image_id=capture["image_id"],
                        points=[[500, 500], [500, 500]],
                        above_target_point_m=0.02,
                        max_steps=120,
                    )
                )

                self.assertFalse(result["ok"], result)
                self.assertEqual(result["failure_stage"], "cancellation")
                self.assertIn("cancelled at", result["error"])
                self.assertEqual(
                    result["cancellation_hold_source"],
                    "subsequent_proprioception_measured_arm_qpos",
                )
                self.assertEqual(len(actions), 1)
                action = np.asarray(actions[0], dtype=np.float32).reshape(-1)
                self.assertEqual(action.shape, (ACTION_DIM,))
                self.assertTrue(np.isfinite(action).all())
                np.testing.assert_allclose(action[ACTION_SLICES["base"]], 0.0)
                if ARM_DOF == 8:
                    for side in ("left", "right"):
                        self.assertEqual(
                            float(action[ACTION_SLICES[f"arm_{side}"]][7]),
                            0.0,
                        )

    def test_read_depth_uses_exact_pixel_from_frozen_capture(self) -> None:
        adapter, world = self._adapter_world()
        frozen_depth = np.full((12, 16), 9.0, dtype=np.float32)
        frozen_depth[6, 8] = 1.25
        adapter.update(
            {
                "robot_r1::proprio": np.zeros(
                    PROPRIO_DIM,
                    dtype=np.float32,
                ),
                "robot_r1::head::rgb": np.zeros(
                    (12, 16, 3),
                    dtype=np.uint8,
                ),
                "robot_r1::head::depth_linear": frozen_depth,
            }
        )
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture_ctx, capture_result = self._ctx(world)
                list(
                    registry["capture_head_camera"].fn(
                        capture_ctx,
                        session_id="depth_lookup",
                    )
                )
                self.assertTrue(capture_result["ok"], capture_result)

                live_depth = np.full((12, 16), 7.5, dtype=np.float32)
                adapter.update(
                    {"robot_r1::head::depth_linear": live_depth}
                )
                ctx, result = self._ctx(world)
                actions = list(
                    registry["read_depth"].fn(
                        ctx,
                        session_id="depth_lookup",
                        image_id=capture_result["image_id"],
                        u=500,
                        v=500,
                    )
                )

        self.assertEqual(len(actions), 1)
        self.assertEqual(np.asarray(actions[0]).shape, (ACTION_DIM,))
        self.assertTrue(np.isfinite(np.asarray(actions[0])).all())
        if ARM_DOF == 8:
            for side in ("left", "right"):
                arm_action = np.asarray(actions[0])[
                    world.controller_action_idx(f"arm_{side}")
                ]
                self.assertEqual(float(arm_action[7]), 0.0)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["depth_m"], 1.25)
        self.assertEqual(result["unit"], "m")
        self.assertEqual(result["execution"], "official_action_only")
        self.assertEqual(
            result["observation_contract"],
            ["frozen *::depth_linear"],
        )
        self.assertEqual(
            result["source"],
            "policy_owned_frozen_evaluator_depth_linear",
        )
        self.assertFalse(result["camera_pose_used"])
        self.assertFalse(result["neighborhood_interpolation"])
        self.assertFalse(result["direct_simulator_mutation"])

    def test_read_depth_rejects_missing_invalid_and_mismatched_depth(self) -> None:
        adapter, world = self._adapter_world()
        adapter.update(
            {
                "robot_r1::head::rgb": np.zeros(
                    (12, 16, 3),
                    dtype=np.uint8,
                ),
                "robot_r1::head::depth_linear": np.ones(
                    (12, 16),
                    dtype=np.float32,
                ),
            }
        )
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture_ctx, capture_result = self._ctx(world)
                list(
                    registry["capture_head_camera"].fn(
                        capture_ctx,
                        session_id="depth_failures",
                    )
                )
                self.assertTrue(capture_result["ok"], capture_result)
                image_id = capture_result["image_id"]
                depth_path = Path(capture_result["depth_path"])

                np.save(depth_path, np.ones((11, 16), dtype=np.float32))
                ctx, mismatch = self._ctx(world)
                list(
                    registry["read_depth"].fn(
                        ctx,
                        session_id="depth_failures",
                        image_id=image_id,
                        u=500,
                        v=500,
                    )
                )
                self.assertFalse(mismatch["ok"])
                self.assertEqual(
                    mismatch["failure_stage"],
                    "observation validation",
                )
                self.assertIn("resolution mismatch", mismatch["error"])

                invalid_depth = np.ones((12, 16), dtype=np.float32)
                invalid_depth[6, 8] = np.nan
                np.save(depth_path, invalid_depth)
                ctx, invalid = self._ctx(world)
                list(
                    registry["read_depth"].fn(
                        ctx,
                        session_id="depth_failures",
                        image_id=image_id,
                        u=500,
                        v=500,
                    )
                )
                self.assertFalse(invalid["ok"])
                self.assertIn("invalid at selected pixel", invalid["error"])

                ctx, missing = self._ctx(world)
                list(
                    registry["read_depth"].fn(
                        ctx,
                        session_id="depth_failures",
                        image_id="img_9999",
                        u=500,
                        v=500,
                    )
                )
                self.assertFalse(missing["ok"])
                self.assertEqual(
                    missing["failure_stage"],
                    "observation validation",
                )
                self.assertIn("metadata does not exist", missing["error"])

    def test_head_path_overlay_replicates_v2_hud_and_depth_layers(self) -> None:
        camera_rotation = np.array(
            [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        camera = {
            "pos": [0.0, 0.0, 1.0],
            "quat": mat_to_quat_xyzw(camera_rotation).tolist(),
            "fx": 100.0,
            "fy": 100.0,
            "cx": 100.0,
            "cy": 80.0,
        }
        robot = {
            "base_pose": {
                "pos": [0.0, 0.0, 0.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
            }
        }
        with tempfile.TemporaryDirectory() as temp_root:
            rgb_path = str(Path(temp_root) / "head.png")
            overlay_path = str(Path(temp_root) / "head.path.png")
            occluded_path = str(Path(temp_root) / "head.path.occluded.png")
            cv2.imwrite(
                rgb_path,
                np.zeros((160, 200, 3), dtype=np.uint8),
            )
            result = render_head_path_overlay(
                rgb_path=rgb_path,
                depth_linear=np.full(
                    (160, 200),
                    10.0,
                    dtype=np.float32,
                ),
                camera=camera,
                robot=robot,
                output_path=overlay_path,
            )

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["path_length_m"], 3.0)
            self.assertEqual(result["path_width_m"], 0.8)
            self.assertEqual(result["path_width_m"], PATH_WIDTH_M)
            self.assertAlmostEqual(result["path_side_inset_m"], 0.0874765)
            self.assertAlmostEqual(PATH_SIDE_INSET_M, 0.0874765)
            self.assertEqual(result["path_side_margin_m"], 0.0)
            self.assertAlmostEqual(
                BASE_FRONT_OFFSET_M,
                0.168970 + 0.07045779099844562,
                places=12,
            )
            self.assertEqual(
                result["distance_origin"],
                "r1pro_base_front_collision_edge",
            )
            self.assertAlmostEqual(
                result["distance_origin_base_local_x_m"],
                BASE_FRONT_OFFSET_M,
            )
            self.assertAlmostEqual(
                result["path_start_base_local_x_m"],
                PATH_START_X_M,
            )
            self.assertAlmostEqual(
                result["path_end_base_local_x_m"],
                PATH_END_X_M,
            )
            np.testing.assert_allclose(
                np.asarray(result["corners_world"])[:, 0],
                [PATH_START_X_M, PATH_END_X_M, PATH_END_X_M, PATH_START_X_M],
                atol=1e-6,
            )
            self.assertTrue(result["near_edge_border_enabled"])
            self.assertAlmostEqual(
                result["near_edge_border_base_local_x_m"],
                PATH_START_X_M,
            )
            self.assertTrue(result["near_edge_center_marker_enabled"])
            self.assertEqual(
                result["near_edge_lateral_marker_offsets_m"],
                [
                    -LATERAL_REFERENCE_OUTER_OFFSET_M,
                    -LATERAL_REFERENCE_OFFSET_M,
                    0.0,
                    LATERAL_REFERENCE_OFFSET_M,
                    LATERAL_REFERENCE_OUTER_OFFSET_M,
                ],
            )
            self.assertEqual(
                result["near_edge_lateral_distances_m"],
                [LATERAL_REFERENCE_OFFSET_M, LATERAL_REFERENCE_OUTER_OFFSET_M],
            )
            self.assertEqual(
                result["near_edge_lateral_labels"],
                ["0.4m", "0.2m", "0.2m", "0.4m"],
            )
            self.assertEqual(
                result["near_edge_outer_label_outset_m"],
                NEAR_EDGE_OUTER_LABEL_OUTSET_M,
            )
            self.assertEqual(result["label_color_rgb"], list(LABEL_COLOR_RGB))
            self.assertFalse(result["label_outline_enabled"])
            self.assertEqual(
                result["label_stroke_width_px"],
                LABEL_STROKE_WIDTH_PX,
            )
            self.assertEqual(result["distance_tick_step_m"], 0.5)
            self.assertEqual(result["distance_tick_count_per_side"], 6)
            self.assertEqual(
                result["distance_labels"],
                ["0.5m", "1.0m", "1.5m", "2.0m", "2.5m", "3.0m"],
            )
            self.assertTrue(result["distance_scale_enabled"])
            self.assertTrue(result["direction_arrows_enabled"])
            self.assertEqual(result["direction_arrow_count"], 6)
            self.assertFalse(result["v2_margin_rails_enabled"])
            self.assertFalse(result["yellow_overlay_enabled"])
            self.assertEqual(
                result["yellow_overlay_semantics"],
                "removed",
            )
            self.assertFalse(result["margin_rails_depth_occluded"])
            self.assertFalse(result["segmentation_used"])
            self.assertFalse(result["scene_geometry_used"])
            self.assertFalse(result["simulator_state_used"])
            self.assertGreater(result["visible_pixel_count"], 0)
            self.assertGreater(result["hud_visible_pixel_count"], 0)
            self.assertEqual(result["chassis_forward_2m"], CHASSIS_FORWARD_2M_CLEAR)
            overlay = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
            blue = (
                (overlay[..., 0] > overlay[..., 2] + 50)
                & (overlay[..., 0] > overlay[..., 1] + 20)
            )
            yellow = (
                (overlay[..., 2] > 120)
                & (overlay[..., 1] > 100)
                & (overlay[..., 0] < 100)
            )
            self.assertGreater(int(blue.sum()), 0)
            self.assertEqual(int(yellow.sum()), 0)

            occluded_result = render_head_path_overlay(
                rgb_path=rgb_path,
                depth_linear=np.full(
                    (160, 200),
                    0.1,
                    dtype=np.float32,
                ),
                camera=camera,
                robot=robot,
                output_path=occluded_path,
            )
            self.assertTrue(occluded_result["ok"], occluded_result)
            self.assertEqual(occluded_result["visible_pixel_count"], 0)
            self.assertEqual(occluded_result["hud_visible_pixel_count"], 0)
            self.assertRegex(
                str(occluded_result["chassis_forward_2m"]),
                r"^Some kind of object is on the chassis-forward path: nearest point 1 \(\d+,\d+\), distance \d+\.\d{2}m$",
            )
            occluded = cv2.imread(occluded_path, cv2.IMREAD_COLOR)
            occluded_blue = (
                (occluded[..., 0] > occluded[..., 2] + 50)
                & (occluded[..., 0] > occluded[..., 1] + 20)
            )
            occluded_yellow = (
                (occluded[..., 2] > 120)
                & (occluded[..., 1] > 100)
                & (occluded[..., 0] < 100)
            )
            self.assertEqual(int(occluded_blue.sum()), 0)
            self.assertEqual(int(occluded_yellow.sum()), 0)

    def test_chassis_forward_2m_ignores_depth_hits_beyond_two_meters(self) -> None:
        from behavior_interface_eval_test.tool.official_v2.base_path_overlay_local import (
            PATH_HEIGHT_OFFSET_M,
            _path_fill_buffers,
            _world_points_to_front_distance_m,
            _unproject_path_pixels_to_world,
            describe_chassis_forward_2m,
        )

        camera_rotation = np.array(
            [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        camera = {
            "pos": [0.0, 0.0, 1.0],
            "quat": mat_to_quat_xyzw(camera_rotation).tolist(),
            "fx": 100.0,
            "fy": 100.0,
            "cx": 100.0,
            "cy": 80.0,
        }
        robot = {
            "base_pose": {
                "pos": [0.0, 0.0, 0.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
            }
        }
        _, path_mask, path_depth, _ = _path_fill_buffers(
            robot,
            height_offset_m=PATH_HEIGHT_OFFSET_M,
            camera=camera,
            width=200,
            height=160,
        )
        rows, cols = np.nonzero(path_mask > 0)
        self.assertGreater(len(rows), 0)
        world = _unproject_path_pixels_to_world(
            pixel_u=cols.astype(np.float64) + 0.5,
            pixel_v=rows.astype(np.float64) + 0.5,
            path_depth=path_depth[rows, cols],
            camera=camera,
            width=200,
            height=160,
        )
        front_m = _world_points_to_front_distance_m(robot, world)
        beyond = front_m > 2.2
        within = (front_m >= 0.2) & (front_m <= 1.5)
        self.assertTrue(np.any(beyond))
        self.assertTrue(np.any(within))

        far_only = np.full(path_depth.shape, 10.0, dtype=np.float32)
        far_only[rows[beyond], cols[beyond]] = (
            path_depth[rows[beyond], cols[beyond]] - 0.2
        )
        self.assertEqual(
            describe_chassis_forward_2m(
                path_mask=path_mask,
                path_depth=path_depth,
                scene_depth=far_only,
                camera=camera,
                robot=robot,
            ),
            CHASSIS_FORWARD_2M_CLEAR,
        )

        near_hit = np.full(path_depth.shape, 10.0, dtype=np.float32)
        near_hit[rows[within], cols[within]] = (
            path_depth[rows[within], cols[within]] - 0.2
        )
        self.assertRegex(
            describe_chassis_forward_2m(
                path_mask=path_mask,
                path_depth=path_depth,
                scene_depth=near_hit,
                camera=camera,
                robot=robot,
            ),
            r"^Some kind of object is on the chassis-forward path: nearest point 1 \(\d+,\d+\), distance \d+\.\d{2}m$",
        )

        # 同一片连续遮挡只出一点；两片分开的遮挡才出两个点。
        one_blob = describe_chassis_forward_2m(
            path_mask=path_mask,
            path_depth=path_depth,
            scene_depth=near_hit,
            camera=camera,
            robot=robot,
        )
        self.assertEqual(one_blob.count("("), 1)

        split_hit = np.full(path_depth.shape, 10.0, dtype=np.float32)
        near_band = (front_m >= 0.2) & (front_m <= 0.4)
        far_band = (front_m >= 1.2) & (front_m <= 1.6)
        self.assertTrue(np.any(near_band))
        self.assertTrue(np.any(far_band))
        split_hit[rows[near_band], cols[near_band]] = (
            path_depth[rows[near_band], cols[near_band]] - 0.2
        )
        split_hit[rows[far_band], cols[far_band]] = (
            path_depth[rows[far_band], cols[far_band]] - 0.2
        )
        split_text = describe_chassis_forward_2m(
            path_mask=path_mask,
            path_depth=path_depth,
            scene_depth=split_hit,
            camera=camera,
            robot=robot,
        )
        self.assertRegex(
            split_text,
            r"^Some kind of object is on the chassis-forward path: nearest point 1 \(\d+,\d+\), distance \d+\.\d{2}m; nearest point 2 \(\d+,\d+\), distance \d+\.\d{2}m$",
        )

    def test_promote_copies_chassis_forward_2m(self) -> None:
        from behavior_interface.web import _promote_post_action_observation_fields

        result = {
            "ok": True,
            "observation": {
                "image_id": "img_0028",
                "chassis_forward_2m": CHASSIS_FORWARD_2M_CLEAR,
            },
        }
        _promote_post_action_observation_fields(result)
        self.assertEqual(result["chassis_forward_2m"], CHASSIS_FORWARD_2M_CLEAR)

    def test_promote_copies_eef_near_0_1m(self) -> None:
        from behavior_interface.web import _promote_post_action_observation_fields

        result = {
            "ok": True,
            "observation": {
                "image_id": "img_0056",
                "eef_near_0.1m": (
                    "非加持物体object near left eef：nearest point 1 (216,488), distance 0.07m"
                ),
            },
        }
        _promote_post_action_observation_fields(result)
        self.assertEqual(
            result["eef_near_0.1m"],
            "非加持物体object near left eef：nearest point 1 (216,488), distance 0.07m",
        )

    def test_compose_nearby_object_warning_is_chassis_only(self) -> None:
        chassis = (
            "Some kind of object is on the chassis-forward path: "
            "nearest point 1 (360,770), distance 0.80m"
        )
        eef = "非加持物体object near left eef：nearest point 1 (216,488), distance 0.07m"
        self.assertEqual(compose_nearby_object_warning(chassis, eef), chassis)
        self.assertEqual(compose_nearby_object_warning(CHASSIS_FORWARD_2M_CLEAR, ""), "")
        self.assertEqual(compose_nearby_object_warning(CHASSIS_FORWARD_2M_CLEAR, eef), "")
        self.assertEqual(compose_nearby_object_warning(chassis, ""), chassis)

    def test_promote_copies_nearby_object_warning(self) -> None:
        from behavior_interface.web import _promote_post_action_observation_fields

        warning = (
            "Some kind of object is on the chassis-forward path: "
            "nearest point 1 (360,770), distance 0.80m"
        )
        result = {
            "ok": True,
            "observation": {
                "image_id": "img_0028",
                "nearby_object_warning": warning,
            },
        }
        _promote_post_action_observation_fields(result)
        self.assertEqual(result["nearby_object_warning"], warning)

    def test_eef_near_sentence_lists_each_hand(self) -> None:
        from behavior_interface_eval_test.tool.official_v2.base_path_overlay_local import (
            eef_near_sentence,
        )

        self.assertEqual(eef_near_sentence([]), "")
        self.assertEqual(
            eef_near_sentence([("left", 216, 488, 0.074)]),
            "object near left eef：nearest point 1 (216,488), distance 0.07m",
        )
        self.assertEqual(
            eef_near_sentence(
                [("left", 216, 488, 0.074), ("right", 500, 400, 0.08)]
            ),
            (
                "object near left eef：nearest point 1 (216,488), "
                "distance 0.07m; object near right eef：nearest point 1 (500,400), "
                "distance 0.08m"
            ),
        )

    def test_plan_gripper_overlay_is_red_and_respects_depth(self) -> None:
        camera = {
            "pos": [0.0, 0.0, 0.0],
            "quat": [0.0, 0.0, 0.0, 1.0],
            "fx": 160.0,
            "fy": 160.0,
            "cx": 80.0,
            "cy": 80.0,
        }
        eef_pose = {
            "pos": [0.0, 0.0, -0.6],
            "quat": [0.0, 0.0, 0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as temp_root:
            rgb_path = str(Path(temp_root) / "plan.png")
            visible_path = str(Path(temp_root) / "plan.visible.png")
            hidden_path = str(Path(temp_root) / "plan.hidden.png")
            offscreen_path = str(Path(temp_root) / "plan.offscreen.png")
            cv2.imwrite(
                rgb_path,
                np.zeros((160, 160, 3), dtype=np.uint8),
            )
            visible = render_plan_gripper_overlay(
                rgb_path=rgb_path,
                depth_linear=np.full(
                    (160, 160),
                    2.0,
                    dtype=np.float32,
                ),
                camera=camera,
                eef_pose=eef_pose,
                output_path=visible_path,
            )
            hidden = render_plan_gripper_overlay(
                rgb_path=rgb_path,
                depth_linear=np.full(
                    (160, 160),
                    0.2,
                    dtype=np.float32,
                ),
                camera=camera,
                eef_pose=eef_pose,
                output_path=hidden_path,
            )
            offscreen = render_plan_gripper_overlay(
                rgb_path=rgb_path,
                depth_linear=np.full(
                    (160, 160),
                    2.0,
                    dtype=np.float32,
                ),
                camera=camera,
                eef_pose={
                    "pos": [0.8, 0.0, -0.6],
                    "quat": [0.0, 0.0, 0.0, 1.0],
                },
                output_path=offscreen_path,
            )

            self.assertTrue(visible["ok"], visible)
            self.assertTrue(hidden["ok"], hidden)
            self.assertEqual(
                visible["build"],
                "official_v2_local_gripper_visual_mesh_zbuffer_v2",
            )
            self.assertEqual(
                visible["geometry_source"],
                "submission_local_frozen_v2_open_visual_triangle_mesh",
            )
            self.assertEqual(visible["input_vertex_count"], 13011)
            self.assertEqual(visible["input_triangle_count"], 26129)
            self.assertEqual(visible["projected_component_count"], 4)
            self.assertEqual(
                visible["face_shading"],
                "v2_abs_view_lambert_0.4_plus_0.6",
            )
            self.assertGreater(
                visible["shade_max"] - visible["shade_min"],
                0.5,
            )
            self.assertGreater(visible["shade_level_count_1e3"], 100)
            self.assertGreater(visible["visible_pixel_count"], 0)
            self.assertEqual(hidden["visible_pixel_count"], 0)
            self.assertFalse(offscreen["ok"], offscreen)
            self.assertIn("outside the RGB image", offscreen["error"])
            self.assertFalse(Path(offscreen_path).exists())
            overlay = cv2.imread(visible_path, cv2.IMREAD_COLOR)
            red = (
                (overlay[..., 2] > 100)
                & (overlay[..., 2] > overlay[..., 1] + 80)
                & (overlay[..., 2] > overlay[..., 0] + 80)
            )
            self.assertGreater(int(red.sum()), 0)
            changed = np.any(overlay != 0, axis=2)
            self.assertGreater(
                len(np.unique(overlay[changed], axis=0)),
                20,
            )

    def test_wrist_grasp_volume_marks_depth_points_red(self) -> None:
        camera = {
            "pos": [0.0, 0.0, 0.0],
            "quat": [0.0, 0.0, 0.0, 1.0],
            "fx": 200.0,
            "fy": 200.0,
            "cx": 80.0,
            "cy": 80.0,
        }
        eef_pose = {
            "pos": [0.0, 0.0, -0.5],
            "quat": [0.0, 0.0, 0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as temp_root:
            rgb_path = str(Path(temp_root) / "wrist.png")
            overlay_path = str(Path(temp_root) / "wrist.grasp.png")
            mask_path = str(Path(temp_root) / "wrist.red_mask.npy")
            cv2.imwrite(
                rgb_path,
                np.zeros((160, 160, 3), dtype=np.uint8),
            )
            result = render_wrist_grasp_volume_overlay(
                rgb_path=rgb_path,
                depth_linear=np.full(
                    (160, 160),
                    0.5,
                    dtype=np.float32,
                ),
                camera=camera,
                eef_pose=eef_pose,
                gripper_qpos=[0.05, 0.05],
                output_path=overlay_path,
                mask_path=mask_path,
            )

            self.assertTrue(result["ok"], result)
            self.assertGreater(result["red_pixel_count"], 0)
            self.assertFalse(result["segmentation_used"])
            self.assertTrue(result["robot_mask_used"])
            self.assertEqual(
                result["robot_mask_source"],
                "submission_local_frozen_dynamic_gripper_visual_mesh",
            )
            self.assertGreaterEqual(
                result["red_candidate_pixel_count"],
                result["red_pixel_count"],
            )
            mask = np.load(mask_path)
            self.assertTrue(bool(mask[80, 80]))
            overlay = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
            red = (
                (overlay[..., 2] > 100)
                & (overlay[..., 2] > overlay[..., 1] + 80)
                & (overlay[..., 2] > overlay[..., 0] + 80)
            )
            self.assertEqual(int(red.sum()), result["red_pixel_count"])

            moving_path = str(Path(temp_root) / "wrist.moving.grasp.png")
            moving_mask_path = str(
                Path(temp_root) / "wrist.moving.red_mask.npy"
            )
            moving = render_wrist_grasp_volume_overlay(
                rgb_path=rgb_path,
                depth_linear=np.full(
                    (160, 160),
                    0.5,
                    dtype=np.float32,
                ),
                camera=camera,
                eef_pose=eef_pose,
                gripper_qpos=[0.05, 0.05],
                gripper_qvel=[0.02, 0.0],
                output_path=moving_path,
                mask_path=moving_mask_path,
            )
            self.assertTrue(moving["ok"], moving)
            self.assertGreater(moving["red_candidate_pixel_count"], 0)
            self.assertEqual(
                moving["red_pixel_count"],
                result["red_pixel_count"],
            )
            self.assertFalse(moving["overlay_suppressed"])
            self.assertFalse(moving["motion_suppression_used"])
            self.assertAlmostEqual(moving["max_abs_gripper_qvel_m_s"], 0.02)
            np.testing.assert_array_equal(np.load(moving_mask_path), mask)

            half_open = render_wrist_grasp_volume_overlay(
                rgb_path=rgb_path,
                depth_linear=np.full(
                    (160, 160),
                    0.5,
                    dtype=np.float32,
                ),
                camera=camera,
                eef_pose=eef_pose,
                gripper_qpos=[0.025, 0.025],
                output_path=str(
                    Path(temp_root) / "wrist.half_open.grasp.png"
                ),
            )
            self.assertTrue(half_open["ok"], half_open)
            self.assertGreater(half_open["red_candidate_pixel_count"], 0)
            self.assertLess(
                half_open["red_candidate_pixel_count"],
                result["red_candidate_pixel_count"],
            )

            closed = render_wrist_grasp_volume_overlay(
                rgb_path=rgb_path,
                depth_linear=np.full(
                    (160, 160),
                    0.5,
                    dtype=np.float32,
                ),
                camera=camera,
                eef_pose=eef_pose,
                gripper_qpos=[0.0, 0.0],
                output_path=str(Path(temp_root) / "wrist.closed.grasp.png"),
            )
            self.assertTrue(closed["ok"], closed)
            self.assertEqual(closed["red_candidate_pixel_count"], 0)
            self.assertEqual(closed["red_pixel_count"], 0)

    def test_wrist_grasp_volume_rejects_only_gripper_surface(self) -> None:
        camera = {
            "pos": [0.0, 0.0, 0.0],
            "quat": [0.0, 0.0, 0.0, 1.0],
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
        }
        eef_pose = {
            "pos": [0.0, 0.0, -0.5],
            "quat": [0.0, 0.0, 0.0, 1.0],
        }
        buffers = visualization_local._rasterize_gripper_buffers(
            eef_pos=np.asarray(eef_pose["pos"], dtype=np.float64),
            eef_quat=np.asarray(eef_pose["quat"], dtype=np.float64),
            camera=camera,
            width=640,
            height=480,
            gripper_qpos=[0.05, 0.05],
        )
        self.assertIsNotNone(buffers)
        model_depth = np.asarray(buffers["depth"], dtype=np.float32)
        scene_depth = np.full(model_depth.shape, np.nan, dtype=np.float32)
        projected = np.isfinite(model_depth)
        scene_depth[projected] = model_depth[projected]

        with tempfile.TemporaryDirectory() as temp_root:
            rgb_path = str(Path(temp_root) / "self.png")
            overlay_path = str(Path(temp_root) / "self.overlay.png")
            cv2.imwrite(rgb_path, np.zeros((480, 640, 3), dtype=np.uint8))
            result = render_wrist_grasp_volume_overlay(
                rgb_path=rgb_path,
                depth_linear=scene_depth,
                camera=camera,
                eef_pose=eef_pose,
                gripper_qpos=[0.05, 0.05],
                output_path=overlay_path,
            )

        self.assertTrue(result["ok"], result)
        self.assertGreater(result["red_candidate_pixel_count"], 0)
        self.assertEqual(result["red_pixel_count"], 0)
        self.assertEqual(
            result["self_rejected_pixel_count"],
            result["red_candidate_pixel_count"],
        )

        foreground_depth = scene_depth.copy()
        foreground_depth[projected] -= 0.001
        with tempfile.TemporaryDirectory() as temp_root:
            rgb_path = str(Path(temp_root) / "foreground.png")
            overlay_path = str(Path(temp_root) / "foreground.overlay.png")
            cv2.imwrite(rgb_path, np.zeros((480, 640, 3), dtype=np.uint8))
            foreground = render_wrist_grasp_volume_overlay(
                rgb_path=rgb_path,
                depth_linear=foreground_depth,
                camera=camera,
                eef_pose=eef_pose,
                gripper_qpos=[0.05, 0.05],
                output_path=overlay_path,
            )
        self.assertTrue(foreground["ok"], foreground)
        self.assertGreater(foreground["red_pixel_count"], 0)

    def test_wrist_grasp_volume_does_not_mark_points_outside_opening(self) -> None:
        camera = {
            "pos": [0.0, 0.0, 0.0],
            "quat": [0.0, 0.0, 0.0, 1.0],
            "fx": 200.0,
            "fy": 200.0,
            "cx": 80.0,
            "cy": 80.0,
        }
        eef_pose = {
            "pos": [0.0, 0.0, -0.5],
            "quat": [0.0, 0.0, 0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as temp_root:
            rgb_path = str(Path(temp_root) / "outside.png")
            overlay_path = str(Path(temp_root) / "outside.overlay.png")
            cv2.imwrite(rgb_path, np.zeros((160, 160, 3), dtype=np.uint8))
            result = render_wrist_grasp_volume_overlay(
                rgb_path=rgb_path,
                depth_linear=np.full((160, 160), 0.8, dtype=np.float32),
                camera=camera,
                eef_pose=eef_pose,
                gripper_qpos=[0.05, 0.05],
                output_path=overlay_path,
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["red_candidate_pixel_count"], 0)
        self.assertEqual(result["red_pixel_count"], 0)

    def test_left_wrist_capture_publishes_red_grasp_overlay(self) -> None:
        adapter, world = self._adapter_world()
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["eef_left_pos"]] = [0.0, 0.0, -0.5]
        proprio[PROPRIO_SLICES["eef_left_quat"]] = [0.0, 0.0, 0.0, 1.0]
        proprio[PROPRIO_SLICES["gripper_left_qpos"]] = [0.05, 0.05]
        adapter.update(
            {
                "robot_r1::proprio": proprio,
                "robot_r1::cam_rel_poses": np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3,
                    dtype=np.float32,
                ),
                "robot_r1::left_wrist::rgb": np.zeros(
                    (160, 160, 3),
                    dtype=np.uint8,
                ),
                "robot_r1::left_wrist::depth_linear": np.full(
                    (160, 160),
                    0.5,
                    dtype=np.float32,
                ),
            }
        )
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                ctx, result = self._ctx(world)
                actions = list(
                    registry["capture_left_wrist_camera"].fn(
                        ctx,
                        session_id="wrist_overlay",
                    )
                )

                self.assertEqual(len(actions), 1)
                self.assertTrue(result["ok"], result)
                self.assertTrue(result["capture_settle"]["settled"])
                self.assertFalse(result["capture_settle"]["settle_required"])
                self.assertEqual(result["capture_settle"]["action_steps"], 0)
                self.assertEqual(
                    result["capture_settle"]["alignment"],
                    "single_evaluator_observation",
                )
                self.assertEqual(
                    result["capture_settle"]["observation_sequence"],
                    result["robot"]["observation_sequence"],
                )
                self.assertTrue(result["grasp_zone_overlay"]["ok"])
                self.assertGreater(
                    result["grasp_zone_overlay"]["red_pixel_count"],
                    0,
                )
                self.assertTrue(Path(result["rgb_overlay_path"]).is_file())
                self.assertTrue(Path(result["red_mask_path"]).is_file())
                self.assertNotEqual(result["rgb_overlay_path"], result["rgb_path"])

    def test_wrist_capture_uses_one_atomic_evaluator_observation(self) -> None:
        adapter, world = self._adapter_world()
        first = np.zeros(PROPRIO_DIM, dtype=np.float32)
        first[PROPRIO_SLICES["eef_left_pos"]] = [0.0, 0.0, -0.5]
        first[PROPRIO_SLICES["eef_left_quat"]] = [0.0, 0.0, 0.0, 1.0]
        first[PROPRIO_SLICES["gripper_left_qpos"]] = [0.041, 0.037]
        first[PROPRIO_SLICES["gripper_left_qvel"]] = [0.021, -0.013]
        observation = {
            "robot_r1::proprio": first,
            "robot_r1::cam_rel_poses": np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3,
                dtype=np.float32,
            ),
            "robot_r1::left_wrist::rgb": np.zeros(
                (160, 160, 3),
                dtype=np.uint8,
            ),
            "robot_r1::left_wrist::depth_linear": np.full(
                (160, 160),
                0.5,
                dtype=np.float32,
            ),
        }
        first_snapshot = adapter.update(observation)
        second = first.copy()
        second[PROPRIO_SLICES["eef_left_pos"]] = [0.3, 0.2, -0.9]
        second[PROPRIO_SLICES["gripper_left_qpos"]] = [0.004, 0.006]
        second[PROPRIO_SLICES["gripper_left_qvel"]] = [0.0, 0.0]
        advanced = {**observation, "robot_r1::proprio": second}
        real_borrow = adapter.borrow_snapshot_for_reading

        def borrow_then_advance():
            snapshot = real_borrow()
            adapter.update(advanced)
            return snapshot

        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    adapter,
                    "borrow_snapshot_for_reading",
                    side_effect=borrow_then_advance,
                ), mock.patch.object(
                    adapter,
                    "camera_frame",
                    side_effect=AssertionError("non-atomic RGB read"),
                ), mock.patch.object(
                    adapter,
                    "camera_depth_frame",
                    side_effect=AssertionError("non-atomic depth read"),
                ), mock.patch.object(
                    adapter,
                    "camera_relative_poses",
                    side_effect=AssertionError("non-atomic camera pose read"),
                ):
                    ctx, result = self._ctx(world)
                    actions = list(
                        registry["capture_left_wrist_camera"].fn(
                            ctx,
                            session_id="atomic_wrist",
                        )
                    )

        self.assertEqual(len(actions), 1)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["robot"]["observation_sequence"], first_snapshot.sequence)
        np.testing.assert_allclose(
            result["robot"]["gripper_left_qpos"],
            [0.041, 0.037],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            result["robot"]["gripper_left_qvel"],
            [0.021, -0.013],
            atol=1e-7,
        )
        self.assertEqual(
            result["capture_settle"]["observation_sequence"],
            first_snapshot.sequence,
        )

    def test_rgbd_lite_public_tool_owns_plan_record(self) -> None:
        adapter, world = self._adapter_world()
        adapter.update(
            {
                "robot_r1::proprio": np.zeros(
                    PROPRIO_DIM,
                    dtype=np.float32,
                ),
                "robot_r1::cam_rel_poses": np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3,
                    dtype=np.float32,
                ),
                "robot_r1::head::rgb": np.zeros(
                    (24, 32, 3),
                    dtype=np.uint8,
                ),
                "robot_r1::head::depth_linear": np.ones(
                    (24, 32),
                    dtype=np.float32,
                ),
            }
        )
        registry = build_registry(adapter)
        payload = {
            "ok": True,
            "recommended_arm": "right",
            "eef_pose": {
                "pos": [0.0, 0.0, -0.6],
                "quat": [0.0, 0.0, 0.0, 1.0],
            },
            "next_eef_move": [0.0, 0.0, 0.1],
            "hit_world": [0.5, 0.0, 0.7],
            "candidate_count": 1,
            "selected_pose_ik": {"solution": "right"},
            "selected_pose_ik_q": {"right": [0.0] * ARM_DOF},
            "ik_solution": "right",
            "candidates": [
                {
                    "arm": "right",
                    "meta": {
                        "planned_safe": {
                            "ok": True,
                            "active_back_m": 0.10,
                            "safe_q": [0.0] * ARM_DOF,
                        }
                    },
                }
            ],
            "grasp_vol_cm3": 5.0,
            "overlap_vol_cm3": 0.0,
            "inflated_overlap_vol_cm3": 0.0,
            "grip_fit": {},
            "plan_audit": {"validation": "submission_local_fk"},
            "debug_images": {},
        }
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture_ctx, capture_result = self._ctx(world)
                list(
                    registry["capture_head_camera"].fn(
                        capture_ctx,
                        session_id="rgbd_lite",
                    )
                )
                plan_ctx, plan_result = self._ctx(world)
                with (
                    mock.patch(
                        "behavior_interface_eval_test.tool.official_v2.tools."
                        "_plan_rgbd_lite_local",
                        return_value=payload,
                    ) as planner,
                    mock.patch.object(
                        official_v2_tools,
                        "_r1pro_shoulder_positions_robot",
                        return_value={
                            "left": np.array([0.0, 0.0, -1.0]),
                            "right": np.array([0.0, 0.0, -1.0]),
                        },
                    ),
                ):
                    actions = list(
                        registry[
                            "plan_grasp_point_filter_rgbd_lite"
                        ].fn(
                            plan_ctx,
                            session_id="rgbd_lite",
                            image_id=capture_result["image_id"],
                            u=500,
                            v=500,
                            plan_arm="right",
                        )
                    )

                self.assertEqual(len(actions), 2)
                planner.assert_called_once()
                self.assertIn(
                    "prepared_target",
                    planner.call_args.kwargs,
                )
                self.assertEqual(
                    planner.call_args.kwargs["prepared_target"][
                        "hit_audit"
                    ]["column_radius_m"],
                    0.003,
                )
                self.assertTrue(plan_result["ok"])
                self.assertTrue(plan_result["shoulder_reach"]["ok"])
                self.assertLess(
                    plan_result["shoulder_reach"]["right_m"],
                    0.7,
                )
                self.assertFalse(plan_result["segmentation_used"])
                self.assertFalse(plan_result["simulator_fk_ik_used"])
                self.assertFalse(plan_result["direct_simulator_mutation"])
                self.assertTrue(plan_result["gripper_visualization"]["ok"])
                self.assertGreater(
                    plan_result["gripper_visualization"][
                        "visible_pixel_count"
                    ],
                    0,
                )
                self.assertTrue(Path(plan_result["render_image_path"]).is_file())
                trajectory, path = _load_plan_trajectory(
                    "rgbd_lite",
                    plan_result["plan_id"],
                )
                self.assertEqual(path, plan_result["plan_record_path"])
                self.assertEqual(trajectory["active_arm"], "right")
                self.assertEqual(
                    trajectory["integrity"]["digest"],
                    plan_result["trajectory_digest"],
                )
                record_path = Path(plan_result["plan_record_path"])
                self.assertTrue(record_path.is_file())
                record = json.loads(record_path.read_text(encoding="utf-8"))
                self.assertEqual(record["robot_model"]["arm_dof"], ARM_DOF)
                self.assertTrue(record["visualization"]["ok"])
                self.assertEqual(
                    record["payload"]["plan_audit"]["validation"],
                    "submission_local_fk",
                )
                self.assertTrue(record["shoulder_reach"]["ok"])

    def test_rgbd_lite_shoulder_gate_is_strict_and_arm_aware(self) -> None:
        target = np.zeros(3, dtype=np.float64)
        trunk = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float64)
        shoulders = {
            "left": np.array([0.69, 0.0, 0.0], dtype=np.float64),
            "right": np.array([0.71, 0.0, 0.0], dtype=np.float64),
        }
        with mock.patch.object(
            official_v2_tools,
            "_r1pro_shoulder_positions_robot",
            return_value=shoulders,
        ):
            any_report = official_v2_tools._plan_grasp_shoulder_reach_report(
                target_robot=target,
                trunk_qpos=trunk,
                plan_arm="any",
            )
            left_report = official_v2_tools._plan_grasp_shoulder_reach_report(
                target_robot=target,
                trunk_qpos=trunk,
                plan_arm="left",
            )
            right_report = official_v2_tools._plan_grasp_shoulder_reach_report(
                target_robot=target,
                trunk_qpos=trunk,
                plan_arm="right",
            )

        self.assertTrue(any_report["ok"])
        self.assertTrue(left_report["ok"])
        self.assertFalse(right_report["ok"])
        self.assertEqual(any_report["checked_distance_m"], 0.69)
        self.assertEqual(right_report["checked_distance_m"], 0.71)

        boundary_shoulders = {
            "left": np.array([0.7, 0.0, 0.0], dtype=np.float64),
            "right": np.array([0.9, 0.0, 0.0], dtype=np.float64),
        }
        with mock.patch.object(
            official_v2_tools,
            "_r1pro_shoulder_positions_robot",
            return_value=boundary_shoulders,
        ):
            boundary = official_v2_tools._plan_grasp_shoulder_reach_report(
                target_robot=target,
                trunk_qpos=trunk,
                plan_arm="any",
            )
        self.assertFalse(boundary["ok"])
        self.assertEqual(boundary["comparison"], "strict_less_than")
        self.assertEqual(boundary["threshold_m"], 0.7)

    def test_rgbd_lite_shoulder_gate_reprojects_selected_hit_pixel(self) -> None:
        half_sqrt = float(np.sqrt(0.5))
        capture = official_v2_tools._FrozenCapture(
            session_id="selected-hit",
            image_id="img_0001",
            role="head",
            depth=np.ones((720, 720), dtype=np.float32),
            camera={
                "robot_relative_pose": {
                    "pos": [0.2, -0.1, 1.3],
                    "quat": [0.0, 0.0, half_sqrt, half_sqrt],
                }
            },
        )
        target_robot, audit = (
            official_v2_tools._prepared_rgbd_target_robot_point(
                prepared_target={
                    "hit": np.array([9.0, 9.0, 9.0]),
                    "hit_audit": {
                        "depth_pixel": [390, 330],
                        "depth_m": 0.8,
                    },
                },
                capture=capture,
                focal_length=17.0,
                horizontal_aperture=40.0,
                world=None,
            )
        )
        camera_xy = 30.0 / 306.0 * 0.8
        expected = np.array(
            [0.2 - camera_xy, -0.1 + camera_xy, 0.5],
            dtype=np.float64,
        )
        np.testing.assert_allclose(target_robot, expected, atol=1e-12)
        self.assertEqual(audit["depth_pixel"], [390, 330])
        self.assertEqual(
            audit["source"],
            "planner_depth_column_hit_and_cam_rel_pose",
        )

    def test_rgbd_lite_far_depth_column_hit_returns_before_planner(self) -> None:
        adapter, world = self._adapter_world()
        trunk = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["trunk_qpos"]] = trunk
        adapter.update(
            {
                "robot_r1::proprio": proprio,
                "robot_r1::cam_rel_poses": np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3,
                    dtype=np.float32,
                ),
                "robot_r1::head::rgb": np.zeros(
                    (24, 32, 3),
                    dtype=np.uint8,
                ),
                "robot_r1::head::depth_linear": np.ones(
                    (24, 32),
                    dtype=np.float32,
                ),
            }
        )
        registry = build_registry(adapter)
        prepared_target = {
            "hit": np.array([0.0, 0.0, -1.0], dtype=np.float64),
            "hit_audit": {
                "method": "depth_column_first_observed_surface",
                "depth_pixel": [16, 12],
                "depth_m": 1.0,
                "column_radius_m": 0.003,
            },
            "outward_normal": None,
            "normal_audit": {"confidence": "none"},
        }
        shoulders = {
            "left": np.array([0.0, 0.0, -0.3], dtype=np.float64),
            "right": np.array([0.0, 0.0, -0.2], dtype=np.float64),
        }
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture_ctx, capture_result = self._ctx(world)
                list(
                    registry["capture_head_camera"].fn(
                        capture_ctx,
                        session_id="rgbd_lite_far",
                    )
                )
                plan_ctx, plan_result = self._ctx(world)
                with (
                    mock.patch.object(
                        official_v2_tools,
                        "_prepare_rgbd_grasp_target",
                        return_value=prepared_target,
                    ) as target_preparer,
                    mock.patch.object(
                        official_v2_tools,
                        "_r1pro_shoulder_positions_robot",
                        return_value=shoulders,
                    ) as shoulder_fk,
                    mock.patch.object(
                        official_v2_tools,
                        "_plan_rgbd_lite_local",
                    ) as planner,
                ):
                    actions = list(
                        registry[
                            "plan_grasp_point_filter_rgbd_lite"
                        ].fn(
                            plan_ctx,
                            session_id="rgbd_lite_far",
                            image_id=capture_result["image_id"],
                            u=500,
                            v=500,
                            plan_arm="any",
                        )
                    )

        self.assertEqual(len(actions), 1)
        target_preparer.assert_called_once()
        planner.assert_not_called()
        np.testing.assert_allclose(
            shoulder_fk.call_args.args[0],
            trunk,
            atol=0.0,
        )
        self.assertFalse(plan_result["ok"])
        self.assertEqual(
            plan_result["error"],
            official_v2_tools.PLAN_GRASP_POINT_TOO_FAR_ERROR,
        )
        self.assertEqual(
            plan_result["failure_stage"],
            "projected_point_shoulder_distance_gate",
        )
        self.assertEqual(plan_result["hit_observation"]["depth_pixel"], [16, 12])
        self.assertEqual(
            plan_result["target_robot_observation"]["source"],
            "planner_depth_column_hit_and_cam_rel_pose",
        )
        self.assertEqual(plan_result["shoulder_reach"]["left_m"], 0.7)
        self.assertEqual(plan_result["shoulder_reach"]["right_m"], 0.8)
        self.assertEqual(
            plan_result["shoulder_reach"]["trunk_source"],
            "frozen_capture",
        )

    def test_exec_plan_pose_skips_open_action_when_selected_gripper_is_open(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        proprio = adapter.proprio_vector().copy()
        current_right = np.zeros(ARM_DOF, dtype=np.float32)
        current_right[0] = 0.12
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = current_right
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.0498, 0.0497]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        safe_q[0] = -0.10
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q[0] = -0.20

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_already_open",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                )
                exec_ctx, exec_result = self._ctx(world)
                actions = self._drive_actions(
                    adapter,
                    world,
                    registry["exec_plan_pose"].fn(
                        exec_ctx,
                        session_id="exec_already_open",
                        plan_id="plan_0001",
                    ),
                )

        self.assertTrue(exec_result["ok"], exec_result)
        open_report = exec_result["pre_exec_gripper_open"]
        self.assertTrue(open_report["checked_before_motion"])
        self.assertTrue(open_report["already_open"])
        self.assertFalse(open_report["open_invoked"])
        self.assertEqual(open_report["open_action_steps"], 0)
        self.assertTrue(open_report["open_target_reached"])
        self.assertEqual(len(actions), exec_result["action_steps"] + 1)
        if world.gripper_uses_effort("right"):
            self.assertFalse(
                any(
                    np.allclose(
                        action[ACTION_SLICES["gripper_right"]],
                        [GRIPPER_OPEN_EFFORT_N, GRIPPER_OPEN_EFFORT_N],
                        rtol=0.0,
                        atol=1e-7,
                    )
                    for action in actions
                )
            )

    def test_exec_plan_pose_opens_selected_gripper_before_arm_motion(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        proprio = adapter.proprio_vector().copy()
        current_right = np.zeros(ARM_DOF, dtype=np.float32)
        current_right[0] = 0.12
        proprio[PROPRIO_SLICES["arm_right_qpos"]] = current_right
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.020, 0.021]
        proprio[PROPRIO_SLICES["gripper_right_qvel"]] = 0.0
        adapter.update({"robot_r1::proprio": proprio})
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        safe_q[0] = -0.10
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q[0] = -0.20

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_preopen",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                )
                exec_ctx, exec_result = self._ctx(world)
                generator = registry["exec_plan_pose"].fn(
                    exec_ctx,
                    session_id="exec_preopen",
                    plan_id="plan_0001",
                )
                actions = []
                for raw_action in generator:
                    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
                    actions.append(action.copy())
                    next_proprio = adapter.proprio_vector().copy()
                    for side in ("left", "right"):
                        next_proprio[
                            PROPRIO_SLICES[f"arm_{side}_qpos"]
                        ] = action[world.controller_action_idx(f"arm_{side}")]
                        next_proprio[
                            PROPRIO_SLICES[f"arm_{side}_qvel"]
                        ] = 0.0
                    if len(actions) == 1:
                        next_proprio[
                            PROPRIO_SLICES["gripper_right_qpos"]
                        ] = [0.0498, 0.0497]
                        next_proprio[
                            PROPRIO_SLICES["gripper_right_qvel"]
                        ] = 0.0
                    adapter.update({"robot_r1::proprio": next_proprio})

        self.assertTrue(exec_result["ok"], exec_result)
        open_report = exec_result["pre_exec_gripper_open"]
        self.assertFalse(open_report["already_open"])
        self.assertTrue(open_report["open_invoked"])
        self.assertEqual(open_report["open_action_steps"], 1)
        self.assertTrue(open_report["open_target_reached"])
        np.testing.assert_allclose(
            actions[0][ACTION_SLICES["arm_right"]],
            current_right,
            atol=1e-7,
        )
        expected_open = (
            [GRIPPER_OPEN_EFFORT_N, GRIPPER_OPEN_EFFORT_N]
            if world.gripper_uses_effort("right")
            else [1.0]
        )
        np.testing.assert_allclose(
            actions[0][ACTION_SLICES["gripper_right"]],
            expected_open,
            atol=1e-7,
        )
        self.assertEqual(len(actions), exec_result["action_steps"] + 2)

    def test_exec_plan_pose_continues_when_open_threshold_is_not_observed(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        proprio = adapter.proprio_vector().copy()
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.020, 0.021]
        adapter.update({"robot_r1::proprio": proprio})
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        safe_q[0] = -0.10
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q[0] = -0.20

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_preopen_no_confirmation",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                )
                exec_ctx, exec_result = self._ctx(world)
                actions = self._drive_actions(
                    adapter,
                    world,
                    registry["exec_plan_pose"].fn(
                        exec_ctx,
                        session_id="exec_preopen_no_confirmation",
                        plan_id="plan_0001",
                    ),
                )

        self.assertTrue(exec_result["ok"], exec_result)
        open_report = exec_result["pre_exec_gripper_open"]
        self.assertTrue(open_report["open_invoked"])
        self.assertEqual(
            open_report["open_action_steps"],
            official_v2_tools.GRIPPER_OPEN_MAX_STEPS,
        )
        self.assertFalse(open_report["open_target_reached"])
        self.assertTrue(open_report["execution_continues"])
        self.assertEqual(
            len(actions),
            official_v2_tools.GRIPPER_OPEN_MAX_STEPS
            + exec_result["action_steps"]
            + 1,
        )

    def test_exec_plan_pose_replays_only_compiled_arm_actions(self) -> None:
        adapter, world = self._adapter_world()
        adapter.update(
            {
                "robot_r1::proprio": np.zeros(
                    PROPRIO_DIM,
                    dtype=np.float32,
                ),
                "robot_r1::cam_rel_poses": np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3,
                    dtype=np.float32,
                ),
                "robot_r1::head::rgb": np.zeros(
                    (24, 32, 3),
                    dtype=np.uint8,
                ),
                "robot_r1::head::depth_linear": np.ones(
                    (24, 32),
                    dtype=np.float32,
                ),
            }
        )
        registry = build_registry(adapter)
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        safe_q[0] = -0.08
        final_q[0] = -0.16
        payload = {
            "ok": True,
            "recommended_arm": "right",
            "eef_pose": {
                "pos": [0.0, 0.0, -0.6],
                "quat": [0.0, 0.0, 0.0, 1.0],
            },
            "next_eef_move": [0.0, 0.0, 0.1],
            "hit_world": [0.5, 0.0, 0.7],
            "candidate_count": 1,
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
            "selected_pose_ik": {"solution": "right"},
            "selected_pose_ik_q": {"right": final_q.tolist()},
            "ik_solution": "right",
            "grasp_vol_cm3": 5.0,
            "overlap_vol_cm3": 0.0,
            "inflated_overlap_vol_cm3": 0.0,
            "grip_fit": {},
            "plan_audit": {"validation": "submission_local_fk"},
            "debug_images": {},
        }
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture_ctx, capture_result = self._ctx(world)
                list(
                    registry["capture_head_camera"].fn(
                        capture_ctx,
                        session_id="exec_plan",
                    )
                )
                plan_ctx, plan_result = self._ctx(world)
                with (
                    mock.patch(
                        "behavior_interface_eval_test.tool.official_v2.tools."
                        "_plan_rgbd_lite_local",
                        return_value=payload,
                    ),
                    mock.patch.object(
                        official_v2_tools,
                        "_r1pro_shoulder_positions_robot",
                        return_value={
                            "left": np.array([0.0, 0.0, -1.0]),
                            "right": np.array([0.0, 0.0, -1.0]),
                        },
                    ),
                ):
                    list(
                        registry[
                            "plan_grasp_point_filter_rgbd_lite"
                        ].fn(
                            plan_ctx,
                            session_id="exec_plan",
                            image_id=capture_result["image_id"],
                            u=500,
                            v=500,
                            plan_arm="right",
                        )
                    )
                exec_ctx, exec_result = self._ctx(world)
                actions = self._drive_actions(
                    adapter,
                    world,
                    registry["exec_plan_pose"].fn(
                        exec_ctx,
                        session_id="exec_plan",
                        plan_id=plan_result["plan_id"],
                        arm="right",
                        reset_tool_roll_at_start=True,
                    ),
                )
                world.make_action(base=[0.01, 0.0, 0.0])
                stale_ctx, stale_result = self._ctx(world)
                stale_actions = list(
                    registry["exec_plan_pose"].fn(
                        stale_ctx,
                        session_id="exec_plan",
                        plan_id=plan_result["plan_id"],
                        arm="right",
                        reset_tool_roll_at_start=True,
                    )
                )

        self.assertTrue(exec_result["ok"], exec_result)
        self.assertEqual(
            exec_result["execution"],
            "official_action_joint_trajectory_replay",
        )
        self.assertEqual(
            exec_result["segments_executed"],
            ["runtime_current_to_safe", "safe_to_final"],
        )
        self.assertGreater(len(actions), 2)
        np.testing.assert_allclose(
            adapter.proprio_vector()[
                PROPRIO_SLICES["arm_right_qpos"]
            ],
            final_q,
            atol=1e-6,
        )
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
        self.assertFalse(stale_result["ok"])
        self.assertIn("base or trunk motion", stale_result["error"])
        self.assertEqual(len(stale_actions), 1)

    def test_exec_plan_pose_restarts_from_matching_signed_waypoint(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        safe_q[0] = -0.80
        final_q[0] = -1.00

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                trajectory = self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_resume",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                )
                paused_index = 9
                paused_q = np.asarray(
                    trajectory["segments"][1]["waypoints"][paused_index],
                    dtype=np.float32,
                )
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["arm_right_qpos"]] = paused_q
                proprio[PROPRIO_SLICES["gripper_right_qpos"]] = 0.05
                adapter.update({"robot_r1::proprio": proprio})

                exec_ctx, exec_result = self._ctx(world)
                actions = self._drive_actions(
                    adapter,
                    world,
                    registry["exec_plan_pose"].fn(
                        exec_ctx,
                        session_id="exec_resume",
                        plan_id="plan_0001",
                    ),
                )

        self.assertTrue(exec_result["ok"], exec_result)
        self.assertEqual(exec_result["start_mode"], "fresh_current_to_safe")
        self.assertEqual(
            exec_result["start_resolution"]["start_source"],
            "evaluator_proprioception",
        )
        self.assertFalse(exec_result["start_resolution"]["signed_prefix_replay"])
        self.assertFalse(exec_result["start_resolution"]["signed_waypoint_resume"])
        self.assertEqual(
            exec_result["segments_executed"],
            ["runtime_current_to_safe", "safe_to_final"],
        )
        expected_first = np.asarray(
            _interpolate_joint_segment(paused_q, safe_q)[0],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            actions[0][ACTION_SLICES["arm_right"]],
            expected_first,
            atol=1e-6,
        )
        self.assertFalse(
            np.allclose(
                actions[0][ACTION_SLICES["arm_right"]],
                paused_q,
                rtol=0.0,
                atol=1e-7,
            )
        )
        np.testing.assert_allclose(
            adapter.proprio_vector()[PROPRIO_SLICES["arm_right_qpos"]],
            final_q,
            atol=1e-6,
        )
        self.assertTrue(
            all(
                np.allclose(
                    action[ACTION_SLICES["gripper_right"]],
                    [0.0, 0.0]
                    if world.gripper_uses_effort("right")
                    else [1.0],
                )
                for action in actions
            )
        )
        if ARM_DOF == 8:
            self.assertTrue(
                all(
                    abs(float(action[ACTION_SLICES["arm_right"]][7])) < 1e-9
                    for action in actions
                )
            )

    def test_exec_plan_pose_bridges_from_arbitrary_current_arm_state(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        safe_q[0] = -0.50
        final_q[0] = -0.75

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_bridge",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                )
                # A historical close command is not a carry contract. Exec
                # must replace it with the controller's open hold.
                world.make_action(gripper_right=[-1.0])
                current_right = np.zeros(ARM_DOF, dtype=np.float32)
                current_right[0] = 0.30
                current_right[1] = -0.25
                current_right[3] = -0.40
                current_right[4] = 0.90
                current_left = np.zeros(ARM_DOF, dtype=np.float32)
                current_left[0] = 0.60
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["arm_right_qpos"]] = current_right
                proprio[PROPRIO_SLICES["arm_left_qpos"]] = current_left
                proprio[PROPRIO_SLICES["gripper_right_qpos"]] = 0.05
                adapter.update({"robot_r1::proprio": proprio})

                exec_ctx, exec_result = self._ctx(world)
                actions = self._drive_actions(
                    adapter,
                    world,
                    registry["exec_plan_pose"].fn(
                        exec_ctx,
                        session_id="exec_bridge",
                        plan_id="plan_0001",
                    ),
                )

        self.assertTrue(exec_result["ok"], exec_result)
        self.assertEqual(
            exec_result["start_mode"],
            "fresh_current_to_safe",
        )
        self.assertEqual(
            exec_result["segments_executed"],
            ["runtime_current_to_safe", "safe_to_final"],
        )
        bridge = exec_result["start_resolution"]["runtime_bridge"]
        self.assertTrue(bridge["current_start_within_local_limits"])
        self.assertFalse(bridge["current_start_joint_limits_enforced"])
        self.assertTrue(bridge["signed_safe_joint_limits_checked"])
        self.assertTrue(bridge["monotonic_to_safe_checked"])
        self.assertTrue(bridge["local_fk_finite_checked"])
        self.assertFalse(bridge["frozen_rgbd_collision_checked"])
        commanded = [
            action[ACTION_SLICES["arm_right"]].astype(np.float64)
            for action in actions[: exec_result["action_steps"]]
        ]
        previous = current_right.astype(np.float64)
        for target in commanded:
            self.assertLessEqual(
                float(np.linalg.norm(target - previous, ord=np.inf)),
                DEFAULT_TRAJECTORY_STEP_RAD + 2e-6,
            )
            previous = target
        np.testing.assert_allclose(previous, final_q, atol=1e-6)
        self.assertEqual(
            exec_result["gripper_holds"]["right"]["mode"],
            (
                "qpos_effort_servo"
                if world.gripper_uses_effort("right")
                else "position_controller_pin"
            ),
        )
        for action in actions:
            np.testing.assert_allclose(
                action[ACTION_SLICES["arm_left"]],
                current_left,
                atol=1e-7,
            )
            self.assertAlmostEqual(
                float(action[ACTION_SLICES["gripper_right"]][0]),
                0.0 if world.gripper_uses_effort("right") else 1.0,
                places=6,
            )
        if ARM_DOF == 8:
            self.assertTrue(
                all(
                    abs(float(action[ACTION_SLICES["arm_right"]][7])) < 1e-9
                    and abs(float(action[ACTION_SLICES["arm_left"]][7])) < 1e-9
                    for action in actions
                )
            )

    def test_exec_plan_pose_uses_qpos_tracking_when_raw_qvel_is_aliased(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        safe_q[0] = -0.10
        final_q[0] = -0.20

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_aliased_qvel",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                )
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["arm_left_qvel"]] = 2.5624
                proprio[PROPRIO_SLICES["arm_right_qvel"]] = -3.9970
                proprio[PROPRIO_SLICES["gripper_right_qpos"]] = 0.05
                adapter.update({"robot_r1::proprio": proprio})

                exec_ctx, exec_result = self._ctx(world)
                actions = []
                for raw_action in registry["exec_plan_pose"].fn(
                    exec_ctx,
                    session_id="exec_aliased_qvel",
                    plan_id="plan_0001",
                    arm="right",
                ):
                    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
                    actions.append(action.copy())
                    next_proprio = adapter.proprio_vector().copy()
                    for side in ("left", "right"):
                        next_proprio[
                            PROPRIO_SLICES[f"arm_{side}_qpos"]
                        ] = action[world.controller_action_idx(f"arm_{side}")]
                    # Preserve the impossible instantaneous velocity residual
                    # while sampled qpos follows every official action.
                    next_proprio[PROPRIO_SLICES["arm_left_qvel"]] = 2.5624
                    next_proprio[PROPRIO_SLICES["arm_right_qvel"]] = -3.9970
                    adapter.update({"robot_r1::proprio": next_proprio})

        self.assertTrue(exec_result["ok"], exec_result)
        self.assertEqual(
            len(actions),
            exec_result["action_steps"]
            + exec_result["j8_reset"]["action_steps"]
            + 1,
        )
        for side in ("left", "right"):
            velocity = exec_result["start_arm_velocity_observation"][side]
            self.assertEqual(velocity["policy"], "diagnostic_only")
            self.assertTrue(velocity["exceeds_legacy_stationary_tolerance"])
        self.assertTrue(exec_result["j8_reset"]["ok"])
        if ARM_DOF == 8:
            self.assertEqual(
                exec_result["j8_reset"]["stationarity_evidence"],
                "consecutive_qpos_delta",
            )
            self.assertTrue(
                exec_result["j8_reset"]["raw_qvel_incoherence_detected"]
            )
            self.assertGreaterEqual(
                exec_result["j8_reset"]["stable_steps"],
                official_v2_tools.WRIST_ROLL_REQUIRED_STABLE_STEPS,
            )
        np.testing.assert_allclose(
            adapter.proprio_vector()[PROPRIO_SLICES["arm_right_qpos"]],
            final_q,
            atol=1e-6,
        )

    def test_exec_plan_pose_switches_to_signed_alternate_arm_final_q(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        planned_safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        planned_safe_q[0] = -0.20
        right_final_q = np.zeros(ARM_DOF, dtype=np.float64)
        right_final_q[0] = -0.40
        left_final_q = np.zeros(ARM_DOF, dtype=np.float64)
        left_final_q[0] = 0.40
        left_safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        left_safe_q[0] = 0.20

        def fake_local_eef_pose(_state, arm, q_arm):
            q = np.asarray(q_arm, dtype=np.float64).reshape(ARM_DOF)
            position = (
                np.array([0.0, 0.0, -0.10], dtype=np.float64)
                if arm == "left"
                and np.allclose(q, left_safe_q, rtol=0.0, atol=1e-9)
                else np.zeros(3, dtype=np.float64)
            )
            return position, np.array([0.0, 0.0, 0.0, 1.0])

        def fake_ik(_state, poses, **kwargs):
            self.assertEqual(kwargs["active_arms"], ("left",))
            self.assertAlmostEqual(float(poses[0]["eef_pos"][2]), -0.10)
            left = [
                {
                    "ok": index == 0,
                    "q_arm": left_safe_q.tolist() if index == 0 else None,
                    "pos_err_m": 0.0 if index == 0 else 1.0,
                    "ori_err_deg": 0.0 if index == 0 else 180.0,
                }
                for index, _pose in enumerate(poses)
            ]
            inactive = [
                {"ok": False, "q_arm": None, "error": "inactive"}
                for _pose in poses
            ]
            return left, inactive, {"solver": "test_local_ik"}

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                with mock.patch.object(
                    official_v2_tools,
                    "_local_eef_pose",
                    side_effect=fake_local_eef_pose,
                ):
                    trajectory = self._write_exec_plan(
                        world=world,
                        run_root=temp_root,
                        session_id="exec_switch_signed",
                        plan_id="plan_0001",
                        safe_q=planned_safe_q,
                        final_q=right_final_q,
                        alternate_final_q_by_arm={"left": left_final_q},
                    )
                    exec_ctx, exec_result = self._ctx(world)
                    with mock.patch.object(
                        official_v2_tools,
                        "_run_external_ik",
                        side_effect=fake_ik,
                    ):
                        actions = self._drive_actions(
                            adapter,
                            world,
                            registry["exec_plan_pose"].fn(
                                exec_ctx,
                                session_id="exec_switch_signed",
                                plan_id="plan_0001",
                                arm="left",
                            ),
                        )

        self.assertIn("left", trajectory["final_q_by_arm"])
        self.assertTrue(exec_result["ok"], exec_result)
        self.assertEqual(exec_result["planned_arm"], "right")
        self.assertEqual(exec_result["arm"], "left")
        self.assertTrue(exec_result["arm_switched"])
        self.assertEqual(
            exec_result["pre_exec_gripper_open"]["arm"],
            "left",
        )
        self.assertTrue(
            exec_result["pre_exec_gripper_open"]["open_invoked"]
        )
        self.assertEqual(
            exec_result["final_q_resolution"]["source"],
            "signed_final_q_by_arm",
        )
        self.assertEqual(
            exec_result["segments_executed"],
            ["runtime_current_to_safe", "safe_to_final"],
        )
        np.testing.assert_allclose(
            adapter.proprio_vector()[PROPRIO_SLICES["arm_left_qpos"]],
            left_final_q,
            atol=1e-6,
        )
        self.assertTrue(
            any(
                np.allclose(
                    action[ACTION_SLICES["arm_left"]],
                    left_safe_q,
                    rtol=0.0,
                    atol=1e-6,
                )
                for action in actions
            )
        )

    def test_exec_plan_pose_switches_arm_for_existing_dual_ik_plan(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        planned_safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        planned_safe_q[0] = -0.20
        right_final_q = np.zeros(ARM_DOF, dtype=np.float64)
        right_final_q[0] = -0.40
        left_final_q = np.zeros(ARM_DOF, dtype=np.float64)
        left_final_q[0] = 0.40
        left_safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        left_safe_q[0] = 0.20
        legacy_payload = {
            "selected_pose_ik": {
                "solution": "both",
                "left_ok": True,
                "right_ok": True,
                "both_ok": True,
                "left_q_arm": left_final_q.tolist(),
                "right_q_arm": right_final_q.tolist(),
            },
            "selected_pose_ik_q": {
                "left": left_final_q.tolist(),
                "right": right_final_q.tolist(),
            },
        }

        def fake_local_eef_pose(_state, arm, q_arm):
            q = np.asarray(q_arm, dtype=np.float64).reshape(ARM_DOF)
            position = (
                np.array([0.0, 0.0, -0.10], dtype=np.float64)
                if arm == "left"
                and np.allclose(q, left_safe_q, rtol=0.0, atol=1e-9)
                else np.zeros(3, dtype=np.float64)
            )
            return position, np.array([0.0, 0.0, 0.0, 1.0])

        def fake_ik(_state, poses, **_kwargs):
            left = [
                {
                    "ok": index == 0,
                    "q_arm": left_safe_q.tolist() if index == 0 else None,
                    "pos_err_m": 0.0 if index == 0 else 1.0,
                    "ori_err_deg": 0.0 if index == 0 else 180.0,
                }
                for index, _pose in enumerate(poses)
            ]
            inactive = [
                {"ok": False, "q_arm": None, "error": "inactive"}
                for _pose in poses
            ]
            return left, inactive, {"solver": "test_local_ik"}

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_switch_legacy",
                    plan_id="plan_0001",
                    safe_q=planned_safe_q,
                    final_q=right_final_q,
                    record_payload=legacy_payload,
                )
                exec_ctx, exec_result = self._ctx(world)
                with mock.patch.object(
                    official_v2_tools,
                    "_local_eef_pose",
                    side_effect=fake_local_eef_pose,
                ), mock.patch.object(
                    official_v2_tools,
                    "_run_external_ik",
                    side_effect=fake_ik,
                ):
                    self._drive_actions(
                        adapter,
                        world,
                        registry["exec_plan_pose"].fn(
                            exec_ctx,
                            session_id="exec_switch_legacy",
                            plan_id="plan_0001",
                            arm="left",
                        ),
                    )

        self.assertTrue(exec_result["ok"], exec_result)
        self.assertTrue(exec_result["arm_switched"])
        self.assertTrue(
            exec_result["final_q_resolution"]["source"].startswith(
                "legacy_plan_record_locally_verified:"
            )
        )
        self.assertTrue(
            exec_result["final_q_resolution"]["submission_local_fk_checked"]
        )
        np.testing.assert_allclose(
            adapter.proprio_vector()[PROPRIO_SLICES["arm_left_qpos"]],
            left_final_q,
            atol=1e-6,
        )

    def test_exec_plan_pose_rejects_arm_without_stored_final_q(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q[0] = -0.40

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_switch_missing",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                )
                exec_ctx, exec_result = self._ctx(world)
                actions = list(
                    registry["exec_plan_pose"].fn(
                        exec_ctx,
                        session_id="exec_switch_missing",
                        plan_id="plan_0001",
                        arm="left",
                    )
                )

        self.assertFalse(exec_result["ok"])
        self.assertIn(
            "no stored final joint solution for requested arm 'left'",
            exec_result["error"],
        )
        self.assertNotIn("does not match planned arm", exec_result["error"])
        self.assertEqual(exec_result["execution"], "rejected_before_action")
        self.assertEqual(len(actions), 1)

    def test_exec_plan_pose_replans_mismatched_back_m_locally(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        planned_safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        planned_safe_q[0] = -0.25
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q[0] = -0.50
        runtime_safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        runtime_safe_q[0] = -0.35

        def fake_local_eef_pose(_state, _arm, q_arm):
            q = np.asarray(q_arm, dtype=np.float64).reshape(ARM_DOF)
            position = (
                np.array([0.0, 0.0, 0.0], dtype=np.float64)
                if np.allclose(q, final_q, rtol=0.0, atol=1e-9)
                else np.array([0.0, 0.0, -0.12], dtype=np.float64)
            )
            return position, np.array([0.0, 0.0, 0.0, 1.0])

        def fake_ik(_state, poses, **_kwargs):
            self.assertAlmostEqual(float(poses[0]["eef_pos"][2]), -0.12)
            inactive = [
                {"ok": False, "q_arm": None, "error": "inactive"}
                for _ in poses
            ]
            right = [
                {
                    "ok": index == 0,
                    "q_arm": runtime_safe_q.tolist() if index == 0 else None,
                    "pos_err_m": 0.0 if index == 0 else 1.0,
                    "ori_err_deg": 0.0 if index == 0 else 180.0,
                }
                for index, _pose in enumerate(poses)
            ]
            return inactive, right, {"solver": "test_local_ik"}

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_back_m_replan",
                    plan_id="plan_0001",
                    safe_q=planned_safe_q,
                    final_q=final_q,
                )
                exec_ctx, exec_result = self._ctx(world)
                with mock.patch.object(
                    official_v2_tools,
                    "_local_eef_pose",
                    side_effect=fake_local_eef_pose,
                ), mock.patch.object(
                    official_v2_tools,
                    "_run_external_ik",
                    side_effect=fake_ik,
                ) as worker:
                    actions = self._drive_actions(
                        adapter,
                        world,
                        registry["exec_plan_pose"].fn(
                            exec_ctx,
                            session_id="exec_back_m_replan",
                            plan_id="plan_0001",
                            back_m=0.12,
                        ),
                    )

        self.assertTrue(exec_result["ok"], exec_result)
        worker.assert_called_once()
        self.assertEqual(exec_result["requested_back_m"], 0.12)
        self.assertEqual(exec_result["active_back_m"], 0.12)
        self.assertFalse(exec_result["back_m_adapted"])
        resolution = exec_result["start_resolution"]["back_m_resolution"]
        self.assertEqual(resolution["mode"], "runtime_local_safe_replan")
        self.assertEqual(
            resolution["final_pose_source"],
            "signed_final_q_submission_local_fk",
        )
        self.assertEqual(resolution["ik_retract_source"], "signed_final_q")
        self.assertEqual(
            exec_result["segments_executed"],
            ["runtime_current_to_safe", "safe_to_final"],
        )
        np.testing.assert_allclose(
            adapter.proprio_vector()[PROPRIO_SLICES["arm_right_qpos"]],
            final_q,
            atol=1e-6,
        )
        self.assertTrue(
            any(
                np.allclose(
                    action[ACTION_SLICES["arm_right"]],
                    runtime_safe_q,
                    rtol=0.0,
                    atol=1e-6,
                )
                for action in actions
            )
        )

    def test_exec_plan_pose_adapts_back_m_after_requested_ik_fails(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        planned_safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        planned_safe_q[0] = -0.20
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q[0] = -0.45
        fallback_safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        fallback_safe_q[0] = -0.30

        def fake_local_eef_pose(_state, _arm, q_arm):
            q = np.asarray(q_arm, dtype=np.float64).reshape(ARM_DOF)
            position = (
                np.array([0.0, 0.0, 0.0], dtype=np.float64)
                if np.allclose(q, final_q, rtol=0.0, atol=1e-9)
                else np.array([0.0, 0.0, -0.11], dtype=np.float64)
            )
            return position, np.array([0.0, 0.0, 0.0, 1.0])

        def fake_ik(_state, poses, **_kwargs):
            self.assertAlmostEqual(float(poses[0]["eef_pos"][2]), -0.12)
            self.assertAlmostEqual(float(poses[1]["eef_pos"][2]), -0.11)
            inactive = [
                {"ok": False, "q_arm": None, "error": "inactive"}
                for _ in poses
            ]
            right = []
            for index, _pose in enumerate(poses):
                right.append(
                    {
                        "ok": index == 1,
                        "q_arm": (
                            fallback_safe_q.tolist() if index == 1 else None
                        ),
                        "pos_err_m": 0.0 if index == 1 else 1.0,
                        "ori_err_deg": 0.0 if index == 1 else 180.0,
                        "error": None if index == 1 else "requested failed",
                    }
                )
            return inactive, right, {"solver": "test_local_ik"}

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_back_m_fallback",
                    plan_id="plan_0001",
                    safe_q=planned_safe_q,
                    final_q=final_q,
                )
                exec_ctx, exec_result = self._ctx(world)
                with mock.patch.object(
                    official_v2_tools,
                    "_local_eef_pose",
                    side_effect=fake_local_eef_pose,
                ), mock.patch.object(
                    official_v2_tools,
                    "_run_external_ik",
                    side_effect=fake_ik,
                ):
                    self._drive_actions(
                        adapter,
                        world,
                        registry["exec_plan_pose"].fn(
                            exec_ctx,
                            session_id="exec_back_m_fallback",
                            plan_id="plan_0001",
                            back_m=0.12,
                        ),
                    )

        self.assertTrue(exec_result["ok"], exec_result)
        self.assertEqual(exec_result["requested_back_m"], 0.12)
        self.assertEqual(exec_result["active_back_m"], 0.11)
        self.assertTrue(exec_result["back_m_adapted"])
        resolution = exec_result["start_resolution"]["back_m_resolution"]
        self.assertEqual(resolution["candidate_index"], 1)
        self.assertEqual(resolution["attempts"][0]["back_m"], 0.12)
        self.assertFalse(resolution["attempts"][0]["solver_ok"])
        self.assertEqual(resolution["attempts"][1]["back_m"], 0.11)
        self.assertTrue(resolution["attempts"][1]["accepted"])

    def test_exec_plan_pose_rejects_only_after_all_back_m_ik_candidates_fail(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        planned_safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q[0] = -0.40

        def fake_ik(_state, poses, **_kwargs):
            failures = [
                {
                    "ok": False,
                    "q_arm": None,
                    "pos_err_m": 1.0,
                    "ori_err_deg": 180.0,
                    "error": "no solution",
                }
                for _ in poses
            ]
            return failures, failures, {"solver": "test_local_ik"}

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_back_m_failure",
                    plan_id="plan_0001",
                    safe_q=planned_safe_q,
                    final_q=final_q,
                )
                exec_ctx, exec_result = self._ctx(world)
                with mock.patch.object(
                    official_v2_tools,
                    "_run_external_ik",
                    side_effect=fake_ik,
                ) as worker:
                    actions = list(
                        registry["exec_plan_pose"].fn(
                            exec_ctx,
                            session_id="exec_back_m_failure",
                            plan_id="plan_0001",
                            back_m=0.12,
                        )
                    )

        self.assertFalse(exec_result["ok"])
        worker.assert_called_once()
        self.assertIn("no locally feasible safe pose", exec_result["error"])
        self.assertNotIn("does not match the immutable plan", exec_result["error"])
        self.assertEqual(exec_result["execution"], "rejected_before_action")
        self.assertEqual(len(actions), 1)

    def test_exec_plan_pose_replans_back_m_from_out_of_limit_observation(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        planned_safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        planned_safe_q[3] = -1.60
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q[3] = -1.75
        runtime_safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        runtime_safe_q[3] = -1.65
        current_right = np.zeros(ARM_DOF, dtype=np.float32)
        current_right[3] = -2.094413

        def fake_local_eef_pose(_state, _arm, q_arm):
            q = np.asarray(q_arm, dtype=np.float64).reshape(ARM_DOF)
            position = (
                np.array([0.0, 0.0, 0.0], dtype=np.float64)
                if np.allclose(q, final_q, rtol=0.0, atol=1e-9)
                else np.array([0.0, 0.0, -0.12], dtype=np.float64)
            )
            return position, np.array([0.0, 0.0, 0.0, 1.0])

        def fake_ik(state, poses, **_kwargs):
            # The active solver retract is signed final_q, not the observed
            # J4 value that lies just beyond the submission-local limit.
            np.testing.assert_allclose(state.arm_right_q, final_q, atol=1e-9)
            inactive = [
                {"ok": False, "q_arm": None, "error": "inactive"}
                for _ in poses
            ]
            right = [
                {
                    "ok": index == 0,
                    "q_arm": runtime_safe_q.tolist() if index == 0 else None,
                    "pos_err_m": 0.0 if index == 0 else 1.0,
                    "ori_err_deg": 0.0 if index == 0 else 180.0,
                }
                for index, _pose in enumerate(poses)
            ]
            return inactive, right, {"solver": "test_local_ik"}

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_back_m_limit_recovery",
                    plan_id="plan_0001",
                    safe_q=planned_safe_q,
                    final_q=final_q,
                )
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["arm_right_qpos"]] = current_right
                adapter.update({"robot_r1::proprio": proprio})
                exec_ctx, exec_result = self._ctx(world)
                with mock.patch.object(
                    official_v2_tools,
                    "_local_eef_pose",
                    side_effect=fake_local_eef_pose,
                ), mock.patch.object(
                    official_v2_tools,
                    "_run_external_ik",
                    side_effect=fake_ik,
                ):
                    self._drive_actions(
                        adapter,
                        world,
                        registry["exec_plan_pose"].fn(
                            exec_ctx,
                            session_id="exec_back_m_limit_recovery",
                            plan_id="plan_0001",
                            back_m=0.12,
                        ),
                    )

        self.assertTrue(exec_result["ok"], exec_result)
        bridge = exec_result["start_resolution"]["runtime_bridge"]
        self.assertFalse(bridge["current_start_within_local_limits"])
        self.assertEqual(bridge["current_start_outside_local_limit_joints"], ["J4"])
        self.assertTrue(bridge["monotonic_local_limit_recovery_checked"])

    def test_exec_plan_pose_recovers_from_observed_static_limit_overshoot(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        safe_q[3] = -1.60
        final_q[3] = -1.75

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": temp_root}):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="exec_limit_recovery",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                )
                current_right = np.zeros(ARM_DOF, dtype=np.float32)
                current_right[3] = -2.094413
                self.assertLess(current_right[3], _ARM_LIMITS["right"][3, 0])
                proprio = adapter.proprio_vector().copy()
                proprio[PROPRIO_SLICES["arm_right_qpos"]] = current_right
                adapter.update({"robot_r1::proprio": proprio})

                exec_ctx, exec_result = self._ctx(world)
                actions = self._drive_actions(
                    adapter,
                    world,
                    registry["exec_plan_pose"].fn(
                        exec_ctx,
                        session_id="exec_limit_recovery",
                        plan_id="plan_0001",
                    ),
                )

        self.assertTrue(exec_result["ok"], exec_result)
        self.assertEqual(exec_result["start_mode"], "fresh_current_to_safe")
        bridge = exec_result["start_resolution"]["runtime_bridge"]
        self.assertFalse(bridge["current_start_within_local_limits"])
        self.assertEqual(bridge["current_start_outside_local_limit_joints"], ["J4"])
        self.assertGreater(
            bridge["current_start_max_local_limit_violation_rad"],
            TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD,
        )
        self.assertTrue(bridge["monotonic_local_limit_recovery_checked"])
        bridge_targets = [
            action[ACTION_SLICES["arm_right"]].astype(np.float64)
            for action in actions[: bridge["waypoint_count"]]
        ]
        previous_distance = abs(float(safe_q[3] - current_right[3]))
        for target in bridge_targets:
            distance = abs(float(safe_q[3] - target[3]))
            self.assertLessEqual(distance, previous_distance + 1e-7)
            previous_distance = distance
        np.testing.assert_allclose(
            adapter.proprio_vector()[PROPRIO_SLICES["arm_right_qpos"]],
            final_q,
            atol=1e-6,
        )

    def test_smoothstep_joint_interpolation_respects_step_limit(self) -> None:
        start = np.zeros(ARM_DOF, dtype=np.float64)
        target = start.copy()
        target[3] = -1.875

        waypoints = _interpolate_joint_segment(start, target)
        previous = start
        max_step = 0.0
        for raw_q in waypoints:
            q = np.asarray(raw_q, dtype=np.float64)
            max_step = max(
                max_step,
                float(np.linalg.norm(q - previous, ord=np.inf)),
            )
            previous = q

        self.assertLessEqual(max_step, DEFAULT_TRAJECTORY_STEP_RAD + 1e-12)
        np.testing.assert_allclose(previous, target, atol=1e-12)

    def test_compile_normalizes_float32_joint_limit_roundoff(self) -> None:
        _, world = self._adapter_world()
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        # cuRobo exports float32: this exact conversion is 5.7e-8rad above
        # the local J6 upper bound even though both represent 1.047198rad.
        safe_q[5] = float(np.float32(1.047198))
        final_q[5] = 1.0

        with tempfile.TemporaryDirectory() as temp_root:
            trajectory = self._write_exec_plan(
                world=world,
                run_root=temp_root,
                session_id="limit_roundoff",
                plan_id="plan_0001",
                safe_q=safe_q,
                final_q=final_q,
            )

        safe_segment = trajectory["segments"][1]
        self.assertEqual(float(safe_segment["endpoint_q"][5]), 1.047198)
        self.assertTrue(
            all(
                float(waypoint[5]) <= 1.047198
                for segment in trajectory["segments"]
                for waypoint in segment["waypoints"]
            )
        )
        validation = trajectory["safety_validation"]
        self.assertEqual(
            validation["joint_limit_numeric_tolerance_rad"],
            TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD,
        )
        self.assertEqual(
            [
                (item["point"], item["joint"])
                for item in validation["joint_limit_normalizations"]
            ],
            [("safe_q", "J6")],
        )

    def test_compile_recovers_observed_start_q_like_v2_plan(self) -> None:
        """v2 plan 不因冻结起点微越界失败；test 编译应对齐同一契约。"""
        _, world = self._adapter_world()
        observed_right_j4 = -2.0949196815490723  # 15063 / img_0018
        observed_left_j4 = -2.0944488048553467
        right_lower = float(_ARM_LIMITS["right"][3, 0])
        left_lower = float(_ARM_LIMITS["left"][3, 0])
        self.assertGreater(
            right_lower - observed_right_j4,
            TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD,
        )
        start = np.zeros(ARM_DOF, dtype=np.float64)
        start[3] = observed_right_j4
        left = np.zeros(ARM_DOF, dtype=np.float64)
        left[3] = observed_left_j4
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        safe_q[3] = -1.60
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q[3] = -1.75
        capture = SimpleNamespace(
            robot={
                "arm_left_qpos": left.tolist(),
                "arm_left_qvel": [0.0] * ARM_DOF,
                "arm_right_qpos": start.tolist(),
                "arm_right_qvel": [0.0] * ARM_DOF,
                "trunk_qpos": world.trunk_qpos().astype(float).tolist(),
                "gripper_left_qpos": world.gripper_qpos_list("left"),
                "gripper_right_qpos": world.gripper_qpos_list("right"),
                "motion_epoch": world.motion_epoch(),
                "episode_id": world.episode_id(),
            }
        )
        compile_ctx, _ = self._ctx(world)
        trajectory = _compile_joint_trajectory(
            ctx=compile_ctx,
            plan_id="plan_0001",
            session_id="img_0018_start_recovery",
            image_id="img_0018",
            capture=capture,
            payload={
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
            },
        )

        self.assertEqual(
            float(trajectory["start_state"]["arm_qpos"]["right"][3]),
            right_lower,
        )
        self.assertEqual(
            float(trajectory["start_state"]["arm_qpos"]["left"][3]),
            left_lower,
        )
        validation = trajectory["safety_validation"]
        self.assertFalse(validation["captured_start_joint_limits_enforced"])
        self.assertTrue(validation["captured_start_recovered_to_local_limits"])
        recovered = {
            (item["point"], item["joint"])
            for item in validation["joint_limit_normalizations"]
            if item.get("recovered_observed_overshoot")
        }
        self.assertEqual(recovered, {("start_q", "J4"), ("left_start_q", "J4")})
        limits = _ARM_LIMITS["right"][:ARM_DOF]
        for segment in trajectory["segments"]:
            for waypoint in segment["waypoints"]:
                q = np.asarray(waypoint, dtype=np.float64)
                self.assertEqual(q.size, ARM_DOF)
                self.assertTrue(np.all(q >= limits[:, 0] - 1e-12))
                self.assertTrue(np.all(q <= limits[:, 1] + 1e-12))

    def test_compile_rejects_real_joint_limit_violation(self) -> None:
        _, world = self._adapter_world()
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        safe_q[5] = (
            1.047198
            + 2.0 * TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
        )

        with tempfile.TemporaryDirectory() as temp_root:
            with self.assertRaisesRegex(
                ValueError,
                r"planned safe_q exceeds local arm joint limits:.*J6=",
            ):
                self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="real_limit_violation",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                )

    def test_signed_trunk_numeric_residual_matches_r5_plan(self) -> None:
        """r5 的 J4=6e-9 是锁死关节浮点残差，必须能过签名校验。"""
        r5_trunk = np.array(
            [
                -0.9915018677711487,
                2.5306808948516846,
                0.16532184183597565,
                6.059627466470374e-09,
            ],
            dtype=np.float64,
        )
        normalized = _require_signed_trunk_q(
            r5_trunk,
            label="r5 start trunk",
            point="signed_start_trunk",
        )
        self.assertEqual(float(normalized[3]), 0.0)
        np.testing.assert_array_equal(normalized[:3], r5_trunk[:3])

        over = r5_trunk.copy()
        over[3] = 2.0 * TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
        with self.assertRaisesRegex(
            ValueError,
            r"r5 start trunk exceeds local trunk joint limits:.*J4=",
        ):
            _require_signed_trunk_q(
                over,
                label="r5 start trunk",
                point="signed_start_trunk",
            )
        self.assertGreater(
            float(over[3]),
            float(TRUNK_LIMITS[3, 1]) + TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD,
        )

    def test_compile_and_load_accepts_r5_trunk_j4_numeric_residual(self) -> None:
        _, world = self._adapter_world()
        r5_trunk = np.array(
            [
                -0.9915018677711487,
                2.5306808948516846,
                0.16532184183597565,
                6.059627466470374e-09,
            ],
            dtype=np.float64,
        )
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                trajectory = self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="r5trunk",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                    trunk_qpos=r5_trunk,
                )
                signed_trunk = np.asarray(
                    trajectory["start_state"]["trunk_qpos"],
                    dtype=np.float64,
                )
                self.assertEqual(float(signed_trunk[3]), 0.0)
                np.testing.assert_array_equal(signed_trunk[:3], r5_trunk[:3])
                trunk_corrections = [
                    item
                    for item in trajectory["safety_validation"][
                        "joint_limit_normalizations"
                    ]
                    if item["point"] == "start_trunk"
                ]
                self.assertEqual(
                    [item["joint"] for item in trunk_corrections],
                    ["J4"],
                )
                self.assertNotIn(
                    "recovered_observed_overshoot",
                    trunk_corrections[0],
                )
                loaded, _ = _load_plan_trajectory("r5trunk", "plan_0001")
                self.assertEqual(
                    loaded["integrity"]["digest"],
                    trajectory["integrity"]["digest"],
                )

    def test_load_accepts_legacy_signed_trunk_j4_residual(self) -> None:
        """旧 plan 已把 6e-9 签进 HMAC，读回时不得再被零容差硬闸拒掉。"""
        _, world = self._adapter_world()
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        legacy_j4 = 6.059627466470374e-09
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                trajectory = self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="legacyj4",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                )
                trajectory["start_state"]["trunk_qpos"][3] = legacy_j4
                trajectory["integrity"] = {
                    "algorithm": "hmac-sha256",
                    "digest": _trajectory_digest(trajectory),
                }
                plan_path = (
                    Path(temp_root) / "legacyj4" / "plans" / "plan_0001.json"
                )
                plan_path.write_text(
                    json.dumps(
                        {
                            "plan_id": "plan_0001",
                            "session_id": "legacyj4",
                            "tool": "plan_grasp_point_filter_rgbd_lite",
                            "trajectory": trajectory,
                        }
                    ),
                    encoding="utf-8",
                )
                loaded, _ = _load_plan_trajectory("legacyj4", "plan_0001")
                self.assertEqual(
                    float(loaded["start_state"]["trunk_qpos"][3]),
                    legacy_j4,
                )

    def test_load_rejects_signed_trunk_beyond_numeric_tolerance(self) -> None:
        _, world = self._adapter_world()
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                trajectory = self._write_exec_plan(
                    world=world,
                    run_root=temp_root,
                    session_id="badtrunk",
                    plan_id="plan_0001",
                    safe_q=safe_q,
                    final_q=final_q,
                )
                trajectory["start_state"]["trunk_qpos"][3] = (
                    2.0 * TRAJECTORY_JOINT_LIMIT_NUMERIC_TOLERANCE_RAD
                )
                trajectory["integrity"] = {
                    "algorithm": "hmac-sha256",
                    "digest": _trajectory_digest(trajectory),
                }
                plan_path = (
                    Path(temp_root) / "badtrunk" / "plans" / "plan_0001.json"
                )
                plan_path.write_text(
                    json.dumps(
                        {
                            "plan_id": "plan_0001",
                            "session_id": "badtrunk",
                            "tool": "plan_grasp_point_filter_rgbd_lite",
                            "trajectory": trajectory,
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    r"trajectory start trunk_qpos exceeds local trunk joint limits",
                ):
                    _load_plan_trajectory("badtrunk", "plan_0001")

    def test_large_smoothstep_trajectory_round_trips_through_loader(self) -> None:
        adapter, world = self._adapter_world()
        adapter.update(
            {
                "robot_r1::proprio": np.zeros(
                    PROPRIO_DIM,
                    dtype=np.float32,
                ),
                "robot_r1::cam_rel_poses": np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3,
                    dtype=np.float32,
                ),
                "robot_r1::head::rgb": np.zeros(
                    (24, 32, 3),
                    dtype=np.uint8,
                ),
                "robot_r1::head::depth_linear": np.ones(
                    (24, 32),
                    dtype=np.float32,
                ),
            }
        )
        registry = build_registry(adapter)
        safe_q = np.zeros(ARM_DOF, dtype=np.float64)
        final_q = np.zeros(ARM_DOF, dtype=np.float64)
        safe_q[3] = -1.875
        final_q[3] = -1.414

        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture_ctx, capture_result = self._ctx(world)
                list(
                    registry["capture_head_camera"].fn(
                        capture_ctx,
                        session_id="large_trajectory",
                    )
                )
                capture = _load_frozen_capture(
                    "large_trajectory",
                    capture_result["image_id"],
                )
                compile_ctx, _ = self._ctx(world)
                trajectory = _compile_joint_trajectory(
                    ctx=compile_ctx,
                    plan_id="plan_0001",
                    session_id="large_trajectory",
                    image_id=capture_result["image_id"],
                    capture=capture,
                    payload={
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
                    },
                )
                plans_dir = Path(temp_root) / "large_trajectory" / "plans"
                plans_dir.mkdir(parents=True, exist_ok=True)
                (plans_dir / "plan_0001.json").write_text(
                    json.dumps(
                        {
                            "plan_id": "plan_0001",
                            "session_id": "large_trajectory",
                            "tool": "plan_grasp_point_filter_rgbd_lite",
                            "trajectory": trajectory,
                        }
                    ),
                    encoding="utf-8",
                )

                loaded, _ = _load_plan_trajectory(
                    "large_trajectory",
                    "plan_0001",
                )

        self.assertEqual(loaded["integrity"], trajectory["integrity"])
        self.assertGreater(
            len(loaded["segments"][1]["waypoints"]),
            int(abs(safe_q[3]) / DEFAULT_TRAJECTORY_STEP_RAD),
        )

    def test_exec_plan_pose_rejects_digest_tampering(self) -> None:
        adapter, world = self._adapter_world()
        adapter.update(
            {
                "robot_r1::proprio": np.zeros(
                    PROPRIO_DIM,
                    dtype=np.float32,
                ),
                "robot_r1::cam_rel_poses": np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3,
                    dtype=np.float32,
                ),
                "robot_r1::head::rgb": np.zeros(
                    (24, 32, 3),
                    dtype=np.uint8,
                ),
                "robot_r1::head::depth_linear": np.ones(
                    (24, 32),
                    dtype=np.float32,
                ),
            }
        )
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture_ctx, capture_result = self._ctx(world)
                list(
                    registry["capture_head_camera"].fn(
                        capture_ctx,
                        session_id="tamper",
                    )
                )
                capture = _load_frozen_capture(
                    "tamper",
                    capture_result["image_id"],
                )
                compile_ctx, _ = self._ctx(world)
                trajectory = _compile_joint_trajectory(
                    ctx=compile_ctx,
                    plan_id="plan_0001",
                    session_id="tamper",
                    image_id=capture_result["image_id"],
                    capture=capture,
                    payload={
                        "recommended_arm": "right",
                        "selected_pose_ik_q": {
                            "right": [-0.16] + [0.0] * (ARM_DOF - 1)
                        },
                        "candidates": [
                            {
                                "arm": "right",
                                "meta": {
                                    "planned_safe": {
                                        "ok": True,
                                        "active_back_m": 0.10,
                                        "safe_q": (
                                            [-0.08]
                                            + [0.0] * (ARM_DOF - 1)
                                        ),
                                    }
                                },
                            }
                        ],
                    },
                )
                trajectory["segments"][-1]["waypoints"][-1][0] -= 0.01
                unsigned = dict(trajectory)
                unsigned.pop("integrity", None)
                trajectory["integrity"]["digest"] = hashlib.sha256(
                    json.dumps(
                        unsigned,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    ).encode("ascii")
                ).hexdigest()
                plans_dir = Path(temp_root) / "tamper" / "plans"
                plans_dir.mkdir(parents=True, exist_ok=True)
                plan_path = plans_dir / "plan_0001.json"
                plan_path.write_text(
                    json.dumps(
                        {
                            "plan_id": "plan_0001",
                            "session_id": "tamper",
                            "tool": "plan_grasp_point_filter_rgbd_lite",
                            "trajectory": trajectory,
                        }
                    ),
                    encoding="utf-8",
                )
                exec_ctx, exec_result = self._ctx(world)
                actions = list(
                    registry["exec_plan_pose"].fn(
                        exec_ctx,
                        session_id="tamper",
                        plan_id="plan_0001",
                    )
                )

        self.assertFalse(exec_result["ok"])
        self.assertEqual(
            trajectory["integrity"]["algorithm"],
            "hmac-sha256",
        )
        self.assertIn("digest mismatch", exec_result["error"])
        self.assertEqual(exec_result["execution"], "rejected_before_action")
        self.assertEqual(len(actions), 1)

    def test_exec_plan_pose_tracking_stall_holds_observed_arm(self) -> None:
        adapter, world = self._adapter_world()
        adapter.update(
            {
                "robot_r1::proprio": np.zeros(
                    PROPRIO_DIM,
                    dtype=np.float32,
                ),
                "robot_r1::cam_rel_poses": np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] * 3,
                    dtype=np.float32,
                ),
                "robot_r1::head::rgb": np.zeros(
                    (24, 32, 3),
                    dtype=np.uint8,
                ),
                "robot_r1::head::depth_linear": np.ones(
                    (24, 32),
                    dtype=np.float32,
                ),
            }
        )
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture_ctx, capture_result = self._ctx(world)
                list(
                    registry["capture_head_camera"].fn(
                        capture_ctx,
                        session_id="tracking_stall",
                    )
                )
                capture = _load_frozen_capture(
                    "tracking_stall",
                    capture_result["image_id"],
                )
                compile_ctx, _ = self._ctx(world)
                trajectory = _compile_joint_trajectory(
                    ctx=compile_ctx,
                    plan_id="plan_0001",
                    session_id="tracking_stall",
                    image_id=capture_result["image_id"],
                    capture=capture,
                    payload={
                        "recommended_arm": "right",
                        "selected_pose_ik_q": {
                            "right": [-0.16] + [0.0] * (ARM_DOF - 1)
                        },
                        "candidates": [
                            {
                                "arm": "right",
                                "meta": {
                                    "planned_safe": {
                                        "ok": True,
                                        "active_back_m": 0.10,
                                        "safe_q": (
                                            [-0.08]
                                            + [0.0] * (ARM_DOF - 1)
                                        ),
                                    }
                                },
                            }
                        ],
                    },
                )
                plans_dir = Path(temp_root) / "tracking_stall" / "plans"
                plans_dir.mkdir(parents=True, exist_ok=True)
                (plans_dir / "plan_0001.json").write_text(
                    json.dumps(
                        {
                            "plan_id": "plan_0001",
                            "session_id": "tracking_stall",
                            "tool": "plan_grasp_point_filter_rgbd_lite",
                            "trajectory": trajectory,
                        }
                    ),
                    encoding="utf-8",
                )
                exec_ctx, exec_result = self._ctx(world)
                actions = [
                    np.asarray(action, dtype=np.float32).reshape(-1)
                    for action in registry["exec_plan_pose"].fn(
                        exec_ctx,
                        session_id="tracking_stall",
                        plan_id="plan_0001",
                    )
                ]

        arm_idx = world.controller_action_idx("arm_right")
        self.assertFalse(exec_result["ok"])
        self.assertEqual(
            exec_result["execution"],
            "stopped_on_tracking_stall",
        )
        self.assertGreater(len(actions), 2)
        self.assertTrue(
            any(abs(float(action[arm_idx][0])) > 0.01 for action in actions[:-1])
        )
        np.testing.assert_allclose(
            actions[-1][arm_idx],
            np.zeros(ARM_DOF, dtype=np.float32),
            atol=1e-7,
        )

    def test_exec_plan_pose_rejects_legacy_plan_without_trajectory(self) -> None:
        adapter, world = self._adapter_world()
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                source = Path(
                    "/tmp/behavior_agent_runs/web/plans/plan_0127.json"
                )
                if not source.is_file():
                    self.skipTest("reference lite plan is unavailable")
                record = json.loads(source.read_text(encoding="utf-8"))
                self.assertNotIn("trajectory", record)
                plans_dir = Path(temp_root) / "web" / "plans"
                plans_dir.mkdir(parents=True)
                (plans_dir / "plan_0127.json").write_text(
                    json.dumps(record),
                    encoding="utf-8",
                )
                exec_ctx, exec_result = self._ctx(world)
                actions = list(
                    registry["exec_plan_pose"].fn(
                        exec_ctx,
                        session_id="web",
                        plan_id="plan_0127",
                    )
                )

        self.assertFalse(exec_result["ok"])
        self.assertIn("no immutable joint trajectory", exec_result["error"])
        self.assertEqual(len(actions), 1)

    def test_rgbd_lite_static_geometry_and_candidate_shape(self) -> None:
        geometry = gripper_geometry(0.003)
        self.assertEqual(geometry["gripper_voxels"].shape, (12918, 3))
        self.assertEqual(geometry["opening_voxels"].shape, (4983, 3))
        self.assertEqual(len(AXIAL_Z_M), 7)
        poses = generate_rgbd_filter_poses(
            np.zeros((1, 3), dtype=np.float64)
        )
        self.assertEqual(len(poses), 20 * N_ROLL * len(AXIAL_Z_M))
        self.assertEqual(
            (poses[0]["pi"], poses[0]["ni"], poses[0]["ri"], poses[0]["ai"]),
            (0, 0, 0, 0),
        )

    def test_rgbd_lite_local_fk_supports_custom_8dof(self) -> None:
        state = LocalRobotState.from_capture(
            {
                "base_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "trunk_qpos": [0.45, -0.4, 0.0, 0.0],
                "arm_left_qpos": [0.0] * ARM_DOF,
                "arm_right_qpos": [0.0] * ARM_DOF,
                "gripper_left_qpos": [0.05, 0.05],
                "gripper_right_qpos": [0.05, 0.05],
            },
            arm_dof=ARM_DOF,
        )
        position, quaternion = local_eef_pose(
            state,
            "right",
            np.zeros(ARM_DOF, dtype=np.float64),
        )
        self.assertEqual(position.shape, (3,))
        self.assertEqual(quaternion.shape, (4,))
        self.assertTrue(np.isfinite(position).all())
        self.assertTrue(np.isfinite(quaternion).all())
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=8)

    def test_reach_base_command_is_signed_holonomic_like_v2(self) -> None:
        pose = SimpleNamespace(
            pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            yaw=0.0,
        )
        diagonal, _, _ = _v2_local_base_command(
            pose,
            bx=1.0,
            by=1.0,
            target_yaw_rad=0.5,
            pos_tol_m=0.01,
            yaw_tol_deg=1.0,
        )
        self.assertGreater(diagonal[0], 0.0)
        self.assertGreater(diagonal[1], 0.0)
        self.assertGreater(diagonal[2], 0.0)

        reverse, _, _ = _v2_local_base_command(
            pose,
            bx=-1.0,
            by=0.0,
            target_yaw_rad=0.0,
            pos_tol_m=0.01,
            yaw_tol_deg=1.0,
        )
        self.assertLess(reverse[0], 0.0)
        self.assertAlmostEqual(reverse[1], 0.0, places=8)
        self.assertAlmostEqual(reverse[2], 0.0, places=8)

    def test_reach_speed_request_is_30_percent_and_action_stays_official(self) -> None:
        self.assertAlmostEqual(REACH_BASE_COMMAND_SPEED_SCALE, 1.30)
        self.assertAlmostEqual(
            REACH_BASE_MAX_SPEED_MPS,
            1.30 * REACH_BASE_OFFICIAL_MAX_SPEED_MPS,
        )
        self.assertAlmostEqual(
            REACH_BASE_MAX_YAW_RATE_RADPS,
            1.30 * REACH_BASE_OFFICIAL_MAX_YAW_RATE_RADPS,
        )
        _, world = self._adapter_world()
        ctx, _ = self._ctx(world)
        action = _base_action_from_physical_velocity(
            ctx,
            [REACH_BASE_MAX_SPEED_MPS, 0.0, REACH_BASE_MAX_YAW_RATE_RADPS],
        )
        np.testing.assert_allclose(
            np.asarray(action)[ACTION_SLICES["base"]],
            [1.0, 0.0, 1.0],
            atol=1e-7,
        )

    def test_reach_pitch_zero_gap_emits_no_interpolation_steps(self) -> None:
        self.assertEqual(_reach_pitch_interpolation_steps(0.0), 0)
        self.assertEqual(_reach_pitch_interpolation_steps(5e-7), 0)
        self.assertEqual(_reach_pitch_interpolation_steps(2e-6), 2)

    def test_reach_base_plan_uses_nearest_of_thirteen_chord_slices(self) -> None:
        target = np.array([1.0, 0.25, 1.25], dtype=np.float64)
        chord = _reach_chord_plan(
            target,
            np.zeros(4, dtype=np.float64),
            0.60,
        )
        plan = _pick_reach_base_goal(
            target_policy_local=target,
            start_pose=SimpleNamespace(
                pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                yaw=0.0,
            ),
            chord_plan=chord,
        )
        self.assertEqual(plan["candidate_count"], 13)
        self.assertEqual(plan["clear_candidate_count"], 13)
        self.assertEqual(plan["pick_policy"], "nearest_chord_travel_first")
        self.assertEqual(
            plan["clearance_source"],
            "v2_optional_nav_clearance_unavailable",
        )
        self.assertAlmostEqual(
            float(plan["travel_m"]),
            min(float(item["travel_m"]) for item in plan["all_candidates"]),
            places=6,
        )

    def test_reach_floor_point_rejects_back_facing_q3_mirror(self) -> None:
        target = np.array(
            [0.7922463462243692, 0.5706552826193259, -0.003573206805472573],
            dtype=np.float64,
        )
        trunk_after_pre_lift = np.array(
            [-0.9915, 2.53068, 1.544, 0.0],
            dtype=np.float64,
        )

        chord = _reach_chord_plan(target, trunk_after_pre_lift, 0.60)
        _position, chest_rotation = official_v2_tools._r1pro_chest_pose_robot(
            chord["target_trunk_q"]
        )
        self.assertGreaterEqual(float(chest_rotation[0, 0]), -1e-6)
        self.assertGreater(float(chord["target_offset_robot_m"][0]), 0.0)
        for alpha in np.linspace(0.0, 1.0, 101):
            trunk_waypoint = trunk_after_pre_lift + alpha * (
                np.asarray(chord["target_trunk_q"], dtype=np.float64)
                - trunk_after_pre_lift
            )
            _position, waypoint_rotation = (
                official_v2_tools._r1pro_chest_pose_robot(trunk_waypoint)
            )
            self.assertGreaterEqual(float(waypoint_rotation[0, 0]), -1e-6)

        plan = _pick_reach_base_goal(
            target_policy_local=target,
            start_pose=SimpleNamespace(
                pos=np.zeros(3, dtype=np.float64),
                yaw=0.0,
            ),
            chord_plan=chord,
        )
        forward_xy = np.array(
            [math.cos(float(plan["yaw_rad"])), math.sin(float(plan["yaw_rad"]))],
            dtype=np.float64,
        )
        target_from_base = target[:2] - np.asarray(
            plan["base_xy"], dtype=np.float64
        )
        self.assertGreater(float(np.dot(forward_xy, target_from_base)), 0.0)
        self.assertGreater(math.degrees(float(plan["yaw_rad"])), 0.0)
        self.assertLess(math.degrees(float(plan["yaw_rad"])), 90.0)

    def test_move_to_reach_point_runs_from_clicked_rgbd_without_object_truth(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["eef_left_quat"]] = [0.0, 0.0, 0.0, 1.0]
        proprio[PROPRIO_SLICES["eef_right_quat"]] = [0.0, 0.0, 0.0, 1.0]
        depth = np.full((72, 72), 4.0, dtype=np.float32)
        depth[34:39, 34:39] = 1.0
        camera_poses = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.35,
                0.0,
                -np.sqrt(0.5),
                0.0,
                np.sqrt(0.5),
            ],
            dtype=np.float32,
        )
        adapter.update(
            {
                "robot_r1::proprio": proprio,
                "robot_r1::cam_rel_poses": camera_poses,
                "robot_r1::head::rgb": np.zeros(
                    (72, 72, 3),
                    dtype=np.uint8,
                ),
                "robot_r1::head::depth_linear": depth,
            }
        )
        registry = build_registry(adapter)
        with tempfile.TemporaryDirectory() as temp_root:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_AGENT_RUNS": temp_root},
            ):
                capture_ctx, capture_result = self._ctx(world)
                list(
                    registry["capture_head_camera"].fn(
                        capture_ctx,
                        session_id="reach",
                    )
                )
                image_id = capture_result["image_id"]
                move_ctx, move_result = self._ctx(world)
                actions = list(
                    registry["move_to_reach_point"].fn(
                        move_ctx,
                        session_id="reach",
                        image_id=image_id,
                        u=500,
                        v=500,
                        reach=0.60,
                        nav_timeout_s=12.0,
                        keep_ori_arm="right",
                    )
                )

        self.assertGreater(len(actions), 5)
        self.assertTrue(move_result["ok"], move_result)
        self.assertFalse(move_result["object_aabb_used"])
        self.assertFalse(move_result["segmentation_used"])
        self.assertFalse(move_result["global_pose_available"])
        self.assertTrue(move_result["keep_ori_ok"], move_result["keep_ori"])
        self.assertEqual(
            move_result["keep_ori"]["right"]["j8_policy"],
            "unchanged_from_pitch_entry",
        )
        self.assertEqual(
            move_result["target_envelope_policy_local_m"]["source"],
            "synthetic_envelope_around_clicked_point",
        )
        self.assertEqual(
            move_result["chord_plan"]["pitch_branch"],
            "forward_half_plane",
        )
        base_execution = move_result["base_execution"]
        self.assertEqual(
            base_execution["motion_model"],
            "v2_signed_holonomic_xy_yaw",
        )
        self.assertFalse(base_execution["lateral_action_locked"])
        self.assertFalse(base_execution["rgbd_motion_guard_used"])
        self.assertEqual(move_result["base_plan"]["candidate_count"], 13)
        self.assertEqual(
            move_result["base_plan"]["pick_policy"],
            "nearest_chord_travel_first",
        )
        base_actions = np.asarray(actions)[:, ACTION_SLICES["base"]]
        self.assertTrue(np.isfinite(base_actions).all())
        for action in actions:
            self.assertEqual(np.asarray(action).shape, (ACTION_DIM,))
            self.assertTrue(np.isfinite(action).all())
        self.assertEqual(
            move_result["reach_compensation"]["target_identity"],
            "immutable_frozen_RGB-D_click",
        )
        self.assertTrue(
            move_result["reach_compensation"]["observation_action_only"]
        )
        timing = move_result["stage_timing"]
        self.assertEqual(timing["clock"], "monotonic")
        self.assertEqual(
            set(timing["phases"]),
            {
                "capture_and_point",
                "pre_lift",
                "reach_planning",
                "base_motion",
                "final_pitch",
                "reach_compensation",
                "final_measurement",
            },
        )
        self.assertEqual(
            timing["total_action_steps_to_result"],
            sum(
                phase["action_steps"]
                for phase in timing["phases"].values()
            ),
        )
        self.assertAlmostEqual(
            move_result["base_execution"]["requested_speed_scale"],
            1.30,
        )
        self.assertTrue(
            move_result["base_execution"]["official_action_limit_respected"]
        )

    def test_reach_compensation_advances_same_target_in_five_cm_steps(
        self,
    ) -> None:
        _, world = self._adapter_world()
        ctx, _ = self._ctx(world)
        target_policy = np.array([0.9, 0.0, 1.445], dtype=np.float64)
        before = _reach_same_target_measurement(ctx, target_policy)
        self.assertGreater(before["minimum_m"], 0.70)

        generator = _yield_reach_compensation(
            ctx,
            target_policy_local=target_policy,
            keep_states={},
        )
        actions = []
        while True:
            try:
                actions.append(
                    np.asarray(next(generator), dtype=np.float32).reshape(-1)
                )
            except StopIteration as stop:
                report = stop.value
                break

        self.assertTrue(report["attempted"], report)
        self.assertTrue(report["ok"], report)
        self.assertEqual(
            report["phase1_forward"]["stopped_reason"],
            "distance_ok",
        )
        self.assertGreater(len(report["phase1_forward"]["steps"]), 0)
        self.assertEqual(report["phase2_pitch"]["steps"], [])
        self.assertLessEqual(report["min_distance_after_m"], 0.70)
        base_actions = np.asarray(actions)[:, ACTION_SLICES["base"]]
        np.testing.assert_allclose(base_actions[:, 1], 0.0, atol=0.0)
        np.testing.assert_allclose(base_actions[:, 2], 0.0, atol=0.0)

    def test_reach_compensation_falls_back_to_bounded_five_degree_pitch(
        self,
    ) -> None:
        adapter, world = self._adapter_world()
        ctx, _ = self._ctx(world)
        target_policy = np.array([0.8, 0.0, 1.0], dtype=np.float64)

        def stationary_base_step(local_ctx, **kwargs):
            yield local_ctx.world.set_base_velocity(0.0, 0.0, 0.0)
            return {
                "ok": True,
                "requested_signed_m": float(kwargs["signed_distance_m"]),
                "actual_signed_m": 0.0,
                "action_steps": 1,
                "safety_abort": None,
                "last_rgbd_guard": None,
            }

        actions = []
        with mock.patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_yield_reach_signed_base_step",
            side_effect=stationary_base_step,
        ):
            generator = _yield_reach_compensation(
                ctx,
                target_policy_local=target_policy,
                keep_states={},
            )
            while True:
                try:
                    action = np.asarray(
                        next(generator),
                        dtype=np.float32,
                    ).reshape(-1)
                    actions.append(action.copy())
                    proprio = adapter.proprio_vector().copy()
                    proprio[PROPRIO_SLICES["trunk_qpos"]] = action[
                        ACTION_SLICES["trunk"]
                    ]
                    proprio[PROPRIO_SLICES["trunk_qvel"]] = 0.0
                    adapter.update({"robot_r1::proprio": proprio})
                except StopIteration as stop:
                    report = stop.value
                    break

        self.assertTrue(report["attempted"], report)
        self.assertFalse(report["ok"], report)
        self.assertEqual(
            report["phase1_forward"]["stopped_reason"],
            "distance_not_decreasing",
        )
        self.assertTrue(report["phase1_forward"]["rolled_back_to_best"])
        pitch_steps = report["phase2_pitch"]["steps"]
        self.assertEqual(len(pitch_steps), 6)
        self.assertAlmostEqual(
            report["phase2_pitch"]["total_pitch_deg"],
            30.0,
            places=5,
        )
        self.assertEqual(
            report["phase2_pitch"]["stopped_reason"],
            "max_pitch",
        )
        self.assertTrue(all(step["decreased"] for step in pitch_steps))
        for action in actions:
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertEqual(float(action[ACTION_SLICES["base"]][1]), 0.0)
            self.assertEqual(float(action[ACTION_SLICES["trunk"]][3]), 0.0)
            if ARM_DOF == 8:
                self.assertEqual(
                    float(action[ACTION_SLICES["arm_left"]][-1]),
                    0.0,
                )
                self.assertEqual(
                    float(action[ACTION_SLICES["arm_right"]][-1]),
                    0.0,
                )

    def test_reach_compensation_chest_pitch_limit_is_165_degrees(self) -> None:
        adapter, world = self._adapter_world()
        proprio = adapter.proprio_vector().copy()
        trunk = np.array(
            [0.5, np.deg2rad(160.0) - 0.5, 0.0, 0.0],
            dtype=np.float32,
        )
        proprio[PROPRIO_SLICES["trunk_qpos"]] = trunk
        adapter.update({"robot_r1::proprio": proprio})
        ctx, _ = self._ctx(world)

        def stationary_base_step(local_ctx, **kwargs):
            yield local_ctx.world.set_base_velocity(0.0, 0.0, 0.0)
            return {
                "ok": True,
                "requested_signed_m": float(kwargs["signed_distance_m"]),
                "actual_signed_m": 0.0,
                "action_steps": 1,
                "safety_abort": None,
                "last_rgbd_guard": None,
            }

        measurements = [
            {"minimum_m": distance}
            for distance in (0.90, 0.90, 0.90, 0.89, 0.89)
        ]
        with mock.patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_yield_reach_signed_base_step",
            side_effect=stationary_base_step,
        ), mock.patch(
            "behavior_interface_eval_test.tool.official_v2.tools."
            "_reach_same_target_measurement",
            side_effect=measurements,
        ):
            generator = _yield_reach_compensation(
                ctx,
                target_policy_local=np.array([0.8, 0.0, 1.0]),
                keep_states={},
            )
            while True:
                try:
                    action = np.asarray(next(generator), dtype=np.float32)
                    proprio = adapter.proprio_vector().copy()
                    proprio[PROPRIO_SLICES["trunk_qpos"]] = action[
                        ACTION_SLICES["trunk"]
                    ]
                    proprio[PROPRIO_SLICES["trunk_qvel"]] = 0.0
                    adapter.update({"robot_r1::proprio": proprio})
                except StopIteration as stop:
                    report = stop.value
                    break

        pitch_steps = report["phase2_pitch"]["steps"]
        self.assertEqual(len(pitch_steps), 1, report)
        self.assertAlmostEqual(
            pitch_steps[0]["chest_pitch_target_deg"],
            165.0,
            places=4,
        )
        self.assertEqual(pitch_steps[0]["q3_floor_by"], "chest_pitch_165deg")
        self.assertEqual(
            report["phase2_pitch"]["stopped_reason"],
            "q3_at_chest_pitch_limit",
        )

    def test_local_j567_keep_orientation_freezes_j1234_and_j8(self) -> None:
        arm_q = np.zeros(ARM_DOF, dtype=np.float64)
        trunk_start = np.zeros(4, dtype=np.float64)
        _, target_rotation = _r1pro_arm_fk_robot(
            "right",
            trunk_start,
            arm_q,
        )
        trunk_changed = trunk_start.copy()
        trunk_changed[2] = -0.20

        command, report = _solve_j567_keep_orientation(
            arm="right",
            trunk_q=trunk_changed,
            fixed_arm_q=arm_q,
            seed_arm_q=arm_q,
            target_rotation=target_rotation,
            tolerance_deg=2.0,
        )

        self.assertTrue(report["ok"], report)
        np.testing.assert_allclose(command[:4], arm_q[:4], atol=1e-10)
        if ARM_DOF == 8:
            self.assertAlmostEqual(float(command[7]), float(arm_q[7]))

    def test_chord_plan_places_both_shoulders_on_requested_sphere(self) -> None:
        plan = _reach_chord_plan(
            np.array([1.0, 0.0, 1.25], dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            0.60,
        )

        self.assertAlmostEqual(plan["left_distance_m"], 0.60, places=6)
        self.assertAlmostEqual(plan["right_distance_m"], 0.60, places=6)

    def test_http_boundary_preserves_blocked_tool_status(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/blocked")
        def blocked():
            return jsonify(
                {
                    "ok": False,
                    "error": (
                        "official_v2 blocks adjust_plan_pose: "
                        "reads a live simulator camera pose"
                    ),
                }
            )

        install_strict_http_control_boundary(
            app,
            evaluator_connected=lambda: True,
        )
        server = SimpleNamespace(
            submit_skill=mock.Mock(),
            wait_for_skill_result=mock.Mock(),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )
        response = app.test_client().post(
            "/api/v2/blocked",
            json={"session_id": "boundary-test"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])
        server.submit_skill.assert_not_called()

    def test_head_adjust_http_route_forwards_xyz_without_camera_defaults(self) -> None:
        app = Flask(__name__)

        @app.post(
            "/api/v2/adjust_right_eef_pose_in_head_frame",
            endpoint="api_v2_adjust_right_eef_pose_in_head_frame",
        )
        def stale_legacy_route():
            return jsonify({"ok": False, "error": "legacy route was used"}), 500

        server = SimpleNamespace(
            submit_skill=mock.Mock(side_effect=["job-adjust", "job-capture"]),
            wait_for_skill_result=mock.Mock(
                side_effect=[
                    {"ok": True, "translation_frame": "robot_base"},
                    {
                        "ok": True,
                        "image_id": "img_xyz",
                        "rgb_main": "data:image/jpeg;base64,xyz",
                        "track_object_distance_binding": {
                            "ok": True,
                            "trackable": True,
                            "reason": None,
                            "session_id": "coordinate-test",
                            "image_id": "img_xyz",
                            "frame_digest": "a" * 64,
                        },
                    },
                ]
            ),
            cancel_current_skill=mock.Mock(),
        )
        install_official_head_adjust_routes(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/adjust_right_eef_pose_in_head_frame",
            json={
                "session_id": "coordinate-test",
                "x": 0.04,
                "z": -0.02,
                "roll": 5.0,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["translation_frame"], "robot_base")
        self.assertEqual(payload["image_id"], "img_xyz")
        self.assertTrue(
            payload["track_object_distance_binding"]["trackable"]
        )
        self.assertEqual(
            payload["track_object_distance_binding"],
            payload["observation"]["track_object_distance_binding"],
        )
        self.assertEqual(
            server.submit_skill.call_args_list,
            [
                mock.call(
                    "adjust_right_eef_pose_in_head_frame",
                    {"x": 0.04, "z": -0.02, "roll": 5.0},
                ),
                mock.call(
                    "capture_head_camera",
                    {"session_id": "coordinate-test"},
                ),
            ],
        )
        adjust_wait = server.wait_for_skill_result.call_args_list[0]
        self.assertEqual(adjust_wait.args[0], "adjust_right_eef_pose_in_head_frame")
        self.assertNotIn("forward", server.submit_skill.call_args_list[0].args[1])

    def test_head_adjust_timeout_returns_failed_result_with_head_capture(self) -> None:
        app = Flask(__name__)

        @app.post(
            "/api/v2/adjust_right_eef_pose_in_head_frame",
            endpoint="api_v2_adjust_right_eef_pose_in_head_frame",
        )
        def stale_legacy_route():
            return jsonify({"ok": False, "error": "legacy route was used"}), 500

        server = SimpleNamespace(
            submit_skill=mock.Mock(side_effect=["job-adjust", "job-capture"]),
            wait_for_skill_result=mock.Mock(
                side_effect=[
                    TimeoutError("adjust timed out"),
                    {
                        "ok": True,
                        "feed": "head",
                        "image_id": "img-timeout-exit",
                        "rgb_main": "data:image/jpeg;base64,timeout-exit",
                    },
                ]
            ),
            cancel_current_skill=mock.Mock(),
        )
        runtime = SimpleNamespace(
            server=server,
            cancel_public_job=mock.Mock(return_value=True),
        )
        install_official_head_adjust_routes(app, runtime)
        install_official_v2_failure_capture_contract(app, runtime)

        response = app.test_client().post(
            "/api/v2/adjust_right_eef_pose_in_head_frame",
            json={"session_id": "timeout-test", "x": 0.05, "timeout_s": 30.0},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["failure_http_status"], 504)
        self.assertEqual(payload["failure_transport"], "application_result")
        self.assertEqual(payload["image_id"], "img-timeout-exit")
        self.assertEqual(payload["observation"]["feed"], "head")
        self.assertEqual(payload["exit_capture_skill"], "capture_head_camera")
        runtime.cancel_public_job.assert_called_once_with("job-adjust")
        server.cancel_current_skill.assert_not_called()
        self.assertTrue(payload["request_scoped_cancellation"])
        self.assertEqual(
            server.submit_skill.call_args_list[-1],
            mock.call(
                "capture_head_camera",
                {"session_id": "timeout-test"},
            ),
        )

    def test_set_arm_http_route_waits_through_progress_hard_deadline(self) -> None:
        app = Flask(__name__)

        @app.post(
            "/api/v2/set_arm_to_grasp_position",
            endpoint="api_v2_set_arm_to_grasp_position",
        )
        def stale_legacy_route():
            return jsonify({"ok": False, "error": "legacy route was used"}), 500

        server = SimpleNamespace(
            submit_skill=mock.Mock(side_effect=["job-set-arm", "job-capture"]),
            wait_for_skill_result=mock.Mock(
                side_effect=[
                    {
                        "ok": True,
                        "converged": True,
                        "progress_deadline_extensions": 2,
                    },
                    {
                        "ok": True,
                        "feed": "head",
                        "image_id": "img-set-arm-exit",
                        "rgb_main": "data:image/jpeg;base64,set-arm-exit",
                    },
                ]
            ),
        )
        runtime = SimpleNamespace(server=server)
        install_official_set_arm_route(app, runtime)

        response = app.test_client().post(
            "/api/v2/set_arm_to_grasp_position",
            json={
                "session_id": "set-arm-test",
                "arm": "right",
                "keep_ori_arm": "none",
                "gripper": "keep",
                "timeout_s": 15.0,
                "max_dq_per_step": 0.3,
                "tol": 0.08,
            },
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["skill_soft_timeout_s"], 15.0)
        self.assertEqual(payload["skill_hard_timeout_s"], 45.0)
        self.assertEqual(payload["http_wait_timeout_s"], 75.0)
        self.assertEqual(payload["image_id"], "img-set-arm-exit")
        first_wait = server.wait_for_skill_result.call_args_list[0]
        self.assertEqual(first_wait.kwargs["timeout_s"], 75.0)
        self.assertEqual(first_wait.kwargs["request_id"], "job-set-arm")
        submitted = server.submit_skill.call_args_list[0]
        self.assertEqual(submitted.args[0], "set_arm_to_grasp_position")
        self.assertEqual(submitted.args[1]["timeout_s"], 15.0)
        self.assertEqual(submitted.args[1]["keep_ori_arm"], "none")

    def test_set_arm_http_timeout_cancels_named_job_and_captures_head(self) -> None:
        app = Flask(__name__)

        @app.post(
            "/api/v2/set_arm_to_grasp_position",
            endpoint="api_v2_set_arm_to_grasp_position",
        )
        def stale_legacy_route():
            return jsonify({"ok": False, "error": "legacy route was used"}), 500

        server = SimpleNamespace(
            submit_skill=mock.Mock(side_effect=["job-set-arm", "job-capture"]),
            wait_for_skill_result=mock.Mock(
                side_effect=[
                    TimeoutError("set arm transport wait timed out"),
                    {
                        "ok": True,
                        "feed": "head",
                        "image_id": "img-set-arm-timeout",
                        "rgb_main": "data:image/jpeg;base64,set-arm-timeout",
                    },
                ]
            ),
        )
        runtime = SimpleNamespace(
            server=server,
            cancel_public_job=mock.Mock(return_value=True),
        )
        install_official_set_arm_route(app, runtime)
        install_official_v2_failure_capture_contract(app, runtime)

        response = app.test_client().post(
            "/api/v2/set_arm_to_grasp_position",
            json={
                "session_id": "set-arm-timeout-test",
                "arm": "right",
                "timeout_s": 15.0,
            },
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["failure_http_status"], 504)
        self.assertEqual(payload["skill_hard_timeout_s"], 45.0)
        self.assertEqual(payload["image_id"], "img-set-arm-timeout")
        self.assertEqual(payload["observation"]["feed"], "head")
        runtime.cancel_public_job.assert_called_once_with("job-set-arm")
        self.assertEqual(
            server.submit_skill.call_args_list[-1],
            mock.call(
                "capture_head_camera",
                {"session_id": "set-arm-timeout-test"},
            ),
        )

    def test_set_arm_boundary_rejects_unimplemented_keep_orientation(self) -> None:
        with self.assertRaisesRegex(
            OfficialToolBoundaryError,
            "keep_ori_arm=none",
        ):
            validate_submission(
                "set_arm_to_grasp_position",
                {"arm": "right", "keep_ori_arm": "right"},
            )

    def test_failure_contract_exposes_existing_reset_body_capture(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/reset_body")
        def failed_reset_body():
            return jsonify(
                {
                    "ok": False,
                    "tool": "reset_body",
                    "job": "job-reset",
                    "timed_out": True,
                    "error": "reset_body timed out",
                    "exit_capture_skill": "capture",
                    "observation": {
                        "ok": True,
                        "feed": "head",
                        "image_id": "img-reset-exit",
                        "rgb_main": "data:image/jpeg;base64,reset-exit",
                    },
                }
            ), 504

        server = SimpleNamespace(
            submit_skill=mock.Mock(),
            wait_for_skill_result=mock.Mock(),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/reset_body",
            json={"session_id": "reset-test"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["failure_http_status"], 504)
        self.assertEqual(payload["image_id"], "img-reset-exit")
        self.assertEqual(payload["observation"]["image_id"], "img-reset-exit")
        server.submit_skill.assert_not_called()

    def test_failure_contract_preserves_close_gripper_wrist_capture(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/close_gripper")
        def failed_close_gripper():
            return jsonify(
                {
                    "ok": False,
                    "tool": "close_gripper",
                    "job": "job-close",
                    "failure_stage": "bilateral_contact_confirmation",
                    "error": "bilateral contact was not confirmed",
                    "exit_capture_skill": "capture_right_wrist_camera",
                    "observation": {
                        "ok": True,
                        "feed": "right_wrist",
                        "image_id": "img-close-wrist",
                        "rgb_main": "data:image/jpeg;base64,close-wrist",
                    },
                }
            )

        server = SimpleNamespace(
            submit_skill=mock.Mock(),
            wait_for_skill_result=mock.Mock(),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/close_gripper",
            json={"session_id": "close-test", "arm": "right"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["image_id"], "img-close-wrist")
        self.assertEqual(payload["observation"]["feed"], "right_wrist")
        self.assertEqual(
            payload["exit_capture_skill"],
            "capture_right_wrist_camera",
        )
        server.submit_skill.assert_not_called()
        server.wait_for_skill_result.assert_not_called()

    def test_capture_policy_covers_every_public_tool(self) -> None:
        self.assertEqual(
            set(OFFICIAL_V2_CAPTURE_POLICIES),
            set(PUBLIC_TOOLS),
        )
        self.assertEqual(
            OFFICIAL_V2_CAPTURE_POLICIES["close_gripper"],
            {
                "success": "request_arm_wrist_grasp",
                "failure": "request_arm_wrist_grasp",
            },
        )
        self.assertEqual(
            OFFICIAL_V2_CAPTURE_POLICIES[
                "plan_grasp_point_filter_rgbd_lite"
            ],
            {"success": "plan_gripper", "failure": "head_path"},
        )

    def test_failed_close_gripper_captures_requested_wrist(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/close_gripper")
        def failed_close_gripper():
            return jsonify({
                "ok": False,
                "tool": "close_gripper",
                "failure_stage": "bilateral_contact_confirmation",
                "error": "bilateral contact was not confirmed",
            })

        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-left-wrist"),
            wait_for_skill_result=mock.Mock(
                return_value={
                    "ok": True,
                    "feed": "left_wrist",
                    "image_id": "img-close-left-wrist",
                    "rgb_main": "data:image/jpeg;base64,left-wrist",
                    "grasp_zone_overlay": {"ok": True},
                }
            ),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/close_gripper",
            json={"session_id": "close-left", "arm": "left"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["observation"]["feed"], "left_wrist")
        self.assertEqual(
            payload["exit_capture_skill"],
            "capture_left_wrist_camera",
        )
        server.submit_skill.assert_called_once_with(
            "capture_left_wrist_camera",
            {"session_id": "close-left"},
        )

    def test_failed_wrist_adjust_captures_same_wrist(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/adjust_right_eef_pose_in_wrist_frame")
        def failed_wrist_adjust():
            return jsonify({
                "ok": False,
                "tool": "adjust_right_eef_pose_in_wrist_frame",
                "failure_stage": "pose_tracking",
                "error": "pose control did not converge",
            }), 504

        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-right-wrist"),
            wait_for_skill_result=mock.Mock(
                return_value={
                    "ok": True,
                    "feed": "right_wrist",
                    "image_id": "img-adjust-right-wrist",
                    "rgb_main": "data:image/jpeg;base64,right-wrist",
                }
            ),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/adjust_right_eef_pose_in_wrist_frame",
            json={"session_id": "adjust-right"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["failure_http_status"], 504)
        self.assertEqual(payload["observation"]["feed"], "right_wrist")
        server.submit_skill.assert_called_once_with(
            "capture_right_wrist_camera",
            {"session_id": "adjust-right"},
        )

    def test_failed_exec_plan_pose_uses_reported_execution_arm_wrist(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/exec_plan_pose")
        def failed_exec():
            return jsonify({
                "ok": False,
                "tool": "exec_plan_pose",
                "arm": "left",
                "failure_stage": "trajectory_tracking",
                "error": "tracking stalled",
            })

        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-exec-left-wrist"),
            wait_for_skill_result=mock.Mock(
                return_value={
                    "ok": True,
                    "feed": "left_wrist",
                    "image_id": "img-exec-left-wrist",
                    "rgb_main": "data:image/jpeg;base64,exec-left-wrist",
                }
            ),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/exec_plan_pose",
            json={
                "session_id": "exec-arm-test",
                "arm": "right",
                "plan_id": "plan-test",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["observation"]["feed"], "left_wrist")
        server.submit_skill.assert_called_once_with(
            "capture_left_wrist_camera",
            {"session_id": "exec-arm-test"},
        )

    def test_failed_plan_returns_head_path_without_red_gripper_media(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/plan")
        def failed_plan():
            return jsonify({
                "ok": False,
                "tool": "plan_grasp_point_filter_rgbd_lite",
                "failure_stage": "local_planning",
                "error": "no valid candidate",
                "marked_image_url": "/tmp/stale-red.png",
                "render_image_path": "/tmp/stale-red.png",
                "rgb_overlay_path": "/tmp/stale-red.png",
                "debug_images": {
                    "frozen_rgb_overlay": "/tmp/stale-red.png",
                    "mesh": "/tmp/mesh.png",
                },
                "terminal_result": {
                    "marked_image_url": "/tmp/stale-terminal-red.png",
                    "render_image_path": "/tmp/stale-terminal-red.png",
                    "debug_images": {
                        "frozen_rgb_overlay": "/tmp/stale-terminal-red.png",
                    },
                },
            }), 400

        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-plan-head"),
            wait_for_skill_result=mock.Mock(
                return_value={
                    "ok": True,
                    "feed": "head",
                    "image_id": "img-plan-failure-head",
                    "rgb_main": "data:image/jpeg;base64,head-path",
                    "rgb_main_path": "/tmp/head.path.png",
                    "rgb_overlay_path": "/tmp/head.path.png",
                    "rgb_path": "/tmp/head.png",
                    "base_path_overlay": {
                        "ok": True,
                        "path": "/tmp/head.path.png",
                    },
                }
            ),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/plan",
            json={
                "session_id": "plan-failure-test",
                "image_id": "img-input",
                "mode": "grasp_point_filter_rgbd_lite",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "no valid candidate")
        self.assertEqual(payload["failure_http_status"], 400)
        self.assertEqual(payload["failure_capture_overlay"], "blue_base_path")
        self.assertFalse(payload["plan_gripper_overlay_available"])
        self.assertEqual(payload["rgb_overlay_path"], "/tmp/head.path.png")
        self.assertNotIn("marked_image_url", payload)
        self.assertNotIn("render_image_path", payload)
        self.assertNotIn("frozen_rgb_overlay", payload["debug_images"])
        self.assertNotIn("marked_image_url", payload["terminal_result"])
        self.assertNotIn("render_image_path", payload["terminal_result"])
        self.assertNotIn(
            "frozen_rgb_overlay",
            payload["terminal_result"]["debug_images"],
        )
        server.submit_skill.assert_called_once_with(
            "capture_head_camera",
            {"session_id": "plan-failure-test"},
        )

    def test_failed_camera_capture_retries_the_same_camera(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/capture_left_wrist_camera")
        def failed_left_wrist_capture():
            return jsonify({
                "ok": False,
                "tool": "capture_left_wrist_camera",
                "failure_stage": "observation_capture",
                "error": "first wrist observation was stale",
            }), 504

        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-left-wrist-retry"),
            wait_for_skill_result=mock.Mock(
                return_value={
                    "ok": True,
                    "feed": "left_wrist",
                    "image_id": "img-left-wrist-retry",
                    "rgb_main": "data:image/jpeg;base64,left-wrist-retry",
                }
            ),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/capture_left_wrist_camera",
            json={"session_id": "left-wrist-retry"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["observation"]["feed"], "left_wrist")
        self.assertEqual(
            payload["exit_capture_skill"],
            "capture_left_wrist_camera",
        )
        server.submit_skill.assert_called_once_with(
            "capture_left_wrist_camera",
            {"session_id": "left-wrist-retry"},
        )

    def test_successful_cut_object_attaches_head_path_capture(self) -> None:
        app = Flask(__name__)
        server = SimpleNamespace(
            submit_skill=mock.Mock(side_effect=["job-cut", "job-cut-head"]),
            wait_for_skill_result=mock.Mock(
                side_effect=[
                    {"ok": True, "tool": "cut_object", "cuts_completed": 1},
                    {
                        "ok": True,
                        "feed": "head",
                        "image_id": "img-cut-head",
                        "rgb_main": "data:image/jpeg;base64,cut-head",
                        "rgb_overlay_path": "/tmp/cut-head.path.png",
                        "base_path_overlay": {
                            "ok": True,
                            "path": "/tmp/cut-head.path.png",
                        },
                    },
                ]
            ),
        )
        runtime = SimpleNamespace(server=server)
        install_official_track_object_distance_routes(app, runtime)

        response = app.test_client().post(
            "/api/v2/cut_object",
            json={"session_id": "cut-success", "timeout_s": 30.0},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["observation"]["feed"], "head")
        self.assertEqual(payload["image_id"], "img-cut-head")
        self.assertEqual(payload["exit_capture_skill"], "capture_head_camera")
        self.assertEqual(
            server.submit_skill.call_args_list,
            [
                mock.call("cut_object", {"session_id": "cut-success", "timeout_s": 30.0}),
                mock.call("capture_head_camera", {"session_id": "cut-success"}),
            ],
        )

    def test_successful_plan_exposes_red_gripper_as_primary_overlay(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/plan")
        def successful_plan():
            return jsonify({
                "ok": True,
                "tool": "plan_grasp_point_filter_rgbd_lite",
                "gripper_visualization": {
                    "ok": True,
                    "path": "/tmp/plan-red-gripper.png",
                },
                "render_image_path": "/tmp/plan-red-gripper.png",
                "observation": {
                    "ok": True,
                    "feed": "head",
                    "image_id": "img-exit-head",
                    "rgb_overlay_path": "/tmp/head.path.png",
                },
            })

        server = SimpleNamespace(
            submit_skill=mock.Mock(),
            wait_for_skill_result=mock.Mock(),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/plan",
            json={
                "session_id": "plan-success-test",
                "mode": "grasp_point_filter_rgbd_lite",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(
            payload["marked_image_url"],
            "/tmp/plan-red-gripper.png",
        )
        self.assertEqual(
            payload["plan_preview_overlay"],
            "red_gripper_on_frozen_head",
        )
        server.submit_skill.assert_not_called()

    def test_failure_contract_does_not_capture_for_tracking_result(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/track_object_distance")
        def failed_tracking():
            return jsonify(
                {
                    "ok": False,
                    "tool": "track_object_distance",
                    "job": "job-track",
                    "failure_stage": "tracking",
                    "error": "point is not observed",
                }
            ), 400

        server = SimpleNamespace(
            submit_skill=mock.Mock(),
            wait_for_skill_result=mock.Mock(),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/track_object_distance",
            json={"session_id": "track-test"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["failure_http_status"], 400)
        self.assertNotIn("observation", payload)
        server.submit_skill.assert_not_called()

    def test_failure_contract_does_not_capture_for_read_depth_result(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/read_depth")
        def failed_depth_read():
            return jsonify({
                "ok": False,
                "tool": "read_depth",
                "failure_stage": "observation validation",
                "error": "selected depth is invalid",
            }), 400

        server = SimpleNamespace(
            submit_skill=mock.Mock(),
            wait_for_skill_result=mock.Mock(),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/read_depth",
            json={
                "session_id": "depth-test",
                "image_id": "img-depth",
                "u": 500,
                "v": 500,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["failure_http_status"], 400)
        self.assertNotIn("observation", payload)
        server.submit_skill.assert_not_called()

    def test_failure_contract_preserves_request_validation_status(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/read_depth")
        def invalid_read_depth():
            return jsonify({"ok": False, "error": "image_id is required"}), 400

        server = SimpleNamespace(
            submit_skill=mock.Mock(),
            wait_for_skill_result=mock.Mock(),
        )
        install_official_v2_failure_capture_contract(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/read_depth",
            json={"session_id": "validation-test"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["ok"])
        server.submit_skill.assert_not_called()

    @unittest.skipUnless(
        WRIST_ROLL_TEST_TOOL_ENABLED,
        "requires BEHAVIOR_EVAL_TEST_ENABLE_WRIST_ROLL_TOOL=1",
    )
    def test_wrist_roll_http_route_and_metadata(self) -> None:
        app = Flask(__name__)

        @app.get("/api/v2/tools", endpoint="api_v2_tools")
        def base_tools():
            return jsonify({"ok": True, "tools": []})

        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-roll"),
            wait_for_skill_result=mock.Mock(
                return_value={
                    "ok": True,
                    "arm": "left",
                    "mode": "absolute",
                    "target_deg": -35.0,
                }
            ),
        )
        install_official_wrist_roll_test_route(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/control_wrist_roll",
            json={
                "arm": "left",
                "mode": "absolute",
                "angle_deg": -35.0,
                "timeout_s": 4.0,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])
        server.submit_skill.assert_called_once_with(
            "control_wrist_roll",
            {
                "arm": "left",
                "mode": "absolute",
                "angle_deg": -35.0,
                "timeout_s": 4.0,
            },
        )
        server.wait_for_skill_result.assert_called_once_with(
            "control_wrist_roll",
            timeout_s=14.0,
            request_id="job-roll",
        )

        tools_response = app.test_client().get("/api/v2/tools")
        metadata = tools_response.get_json()["tools"]
        self.assertEqual([item["name"] for item in metadata], ["control_wrist_roll"])
        self.assertTrue(metadata[0]["test_only"])
        self.assertEqual(metadata[0]["args"][1]["widget"], "select")
        self.assertEqual(
            metadata[0]["args"][1]["options"],
            ["relative", "absolute", "reset"],
        )

    def test_http_version_route_never_imports_legacy_adjust_builds(self) -> None:
        app = Flask(__name__)
        legacy_calls: list[bool] = []

        @app.get("/api/skills/version", endpoint="api_skills_version")
        def legacy_version():
            legacy_calls.append(True)
            return jsonify({"ok": True, "adjust_wrist_frame_build": "legacy"})

        install_strict_http_control_boundary(app)
        response = app.test_client().get("/api/skills/version")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(legacy_calls, [])
        self.assertEqual(
            response.json["adjust_wrist_frame_build"],
            ADJUST_EEF_LOCAL_BUILD,
        )
        self.assertEqual(
            response.json["registry_owner"],
            "behavior_interface_eval_test.tool.official_v2",
        )
        self.assertEqual(
            response.json["spatial_map_backend"],
            "behavior_interface.spatial_map.EgoMap",
        )
        self.assertEqual(
            response.json["spatial_map_build"],
            "egomap_self_rgbd_wallbands_v3_20260827",
        )
        self.assertFalse(response.json["simulator_handle"])

    def test_http_version_route_reports_selected_rtabmap_backend(self) -> None:
        from behavior_interface.rtabmap_slam.live import BUILD as RTABMAP_BUILD

        app = Flask(__name__)

        @app.get("/api/skills/version", endpoint="api_skills_version")
        def legacy_version():
            return jsonify({"ok": True, "spatial_map_backend": "legacy"})

        with mock.patch.dict(
            os.environ,
            {"BEHAVIOR_SPATIAL_MAP_BACKEND": "rtabmap"},
        ):
            install_strict_http_control_boundary(app)
            response = app.test_client().get("/api/skills/version")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json["spatial_map_backend"],
            "behavior_interface.rtabmap_slam.LiveMapper",
        )
        self.assertEqual(
            response.json["spatial_map_build"],
            RTABMAP_BUILD,
        )
        self.assertFalse(response.json["simulator_handle"])

    def test_official_spatial_frame_uses_selected_rtabmap_singleton(self) -> None:
        mapper = SimpleNamespace(
            disabled=False,
            odom_tick=mock.Mock(),
            map_tick=mock.Mock(),
        )
        world = SimpleNamespace()
        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime.server = SimpleNamespace(world=world, log=mock.Mock())
        runtime._spatial_live_mapper = None

        with mock.patch.dict(
            os.environ,
            {
                "BEHAVIOR_SPATIAL_MAP": "1",
                "BEHAVIOR_SPATIAL_MAP_BACKEND": "rtabmap",
            },
        ), mock.patch(
            "behavior_interface.rtabmap_slam.live.get_live_mapper",
            return_value=mapper,
        ) as get_mapper:
            runtime._spatial_map_frame(0.05)

        get_mapper.assert_called_once_with()
        self.assertIs(runtime._spatial_live_mapper, mapper)
        mapper.odom_tick.assert_called_once_with(world, 0.05)
        mapper.map_tick.assert_called_once_with(world)

    def test_official_v2_source_has_no_simulator_or_legacy_skill_access(self) -> None:
        profile_dir = Path(__file__).parent / "tool" / "official_v2"
        forbidden = (
            "import omnigibson",
            "from omnigibson",
            "behavior_interface.skills",
            "behavior_interface_eval_test.tool.official_v1",
            "from behavior_interface ",
            "import behavior_interface",
        )
        for path in sorted(profile_dir.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path.name} contains {token}")
            tree = ast.parse(source, filename=str(path))
            forbidden_calls = {
                "get_joint_positions",
                "set_joint_positions",
                "set_position_orientation",
                "_establish_grasp",
            }
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if isinstance(fn, ast.Attribute):
                    self.assertNotIn(
                        fn.attr,
                        forbidden_calls,
                        f"{path.name} calls forbidden method {fn.attr}",
                    )

    def test_official_v2_has_no_repository_import_outside_its_package(self) -> None:
        profile_dir = Path(__file__).parent / "tool" / "official_v2"
        repository_roots = ("behavior_interface", "behavior_interface_eval_test")
        for path in sorted(profile_dir.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            alias.name.startswith(repository_roots),
                            f"{path.name} imports repository module {alias.name}",
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        continue
                    module = str(node.module or "")
                    self.assertFalse(
                        module.startswith(repository_roots),
                        f"{path.name} imports repository module {module}",
                    )

    def test_all_public_tools_are_explicit_and_not_forwarded(self) -> None:
        tools_path = (
            Path(__file__).parent / "tool" / "official_v2" / "tools.py"
        )
        tree = ast.parse(
            tools_path.read_text(encoding="utf-8"),
            filename=str(tools_path),
        )
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertEqual(
            [name for name in PUBLIC_TOOLS if name in definitions],
            list(PUBLIC_TOOLS),
        )
        for name in PUBLIC_TOOLS:
            with self.subTest(tool=name):
                forwarded = [
                    node
                    for node in ast.walk(definitions[name])
                    if isinstance(node, ast.YieldFrom)
                ]
                self.assertEqual(
                    forwarded,
                    [],
                    f"{name} forwards to another generator implementation",
                )


if __name__ == "__main__":
    unittest.main()
