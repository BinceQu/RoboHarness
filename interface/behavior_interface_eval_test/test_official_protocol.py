from __future__ import annotations

import asyncio
import os
import queue
import tempfile
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from flask import Flask, jsonify

from behavior_interface_eval_test.operator_scene_control import write_reset_request
from behavior_interface_eval_test.official_policy_interface import (
    ACTION_DIM,
    ACTION_SLICES,
    ARM_DOF,
    EvaluatorConnectionState,
    ObservationActionAdapter,
    ObservationSnapshot,
    OfficialEvaluatorDisconnectedError,
    OfficialPolicyRuntime,
    PROPRIO_DIM,
    PROPRIO_SLICES,
    _PipelineTiming,
    _render_official_live_head_frame,
    _wait_for_skill_result_with_evaluator_liveness,
    filter_allowed_observation,
    install_official_live_frame_routes,
    install_strict_http_control_boundary,
    serve_policy,
)
from behavior_interface_eval_test.tool.official_v1 import (
    PUBLIC_SKILLS,
    build_registry,
)
from behavior_interface_eval_test.tool.official_v1.geometry import (
    relative_uv_to_pixel,
    r1pro_shoulder_positions_robot,
    unproject_pixel,
)
from behavior_interface_eval_test.official_action_world import (
    ObservationBackedActionWorld,
)
from behavior_interface_eval_test.official_bddl_progress import (
    UI_BDDL_PROGRESS_KEY,
    encode_bddl_progress,
)
from behavior_interface_eval_test.official_protocol import packb, unpackb
from behavior_interface_eval_test.official_tools import (
    StrictToolBoundaryError,
    capability_report,
    install_strict_tool_registry,
    validate_submission,
)
from behavior_interface_eval_test.tool.official_v2.grasp_geometry_local import (
    mat_to_quat_xyzw,
)


class OfficialProtocolTest(unittest.TestCase):
    @staticmethod
    def _world(adapter: ObservationActionAdapter) -> ObservationBackedActionWorld:
        world = ObservationBackedActionWorld(
            proprio_provider=adapter.proprio_vector,
            hold_action_provider=adapter.hold_action,
            eef_pose_provider=adapter.eef_pose,
        )
        adapter.set_final_action_overlay(world.enforce_gripper_close_keepalive)
        return world

    def test_numpy_msgpack_roundtrip(self) -> None:
        value = np.arange(12, dtype=np.float32).reshape(3, 4)
        decoded = unpackb(packb({"value": value}), strict_map_key=False)
        np.testing.assert_array_equal(decoded["value"], value)

    def test_evaluator_connection_state_counts_live_websockets(self) -> None:
        state = EvaluatorConnectionState()
        self.assertEqual(state.snapshot(), (False, 0))
        state.opened()
        state.opened()
        self.assertEqual(state.snapshot(), (True, 2))
        state.closed()
        self.assertEqual(state.snapshot(), (True, 1))
        state.closed()
        state.closed()
        self.assertEqual(state.snapshot(), (False, 0))

    def test_idle_probe_is_compact_and_tracks_skill_queue(self) -> None:
        runtime = object.__new__(OfficialPolicyRuntime)
        runtime._source = "hold"
        runtime._last_error = ""
        runtime.downstream = None
        runtime.episode_initializer = SimpleNamespace(ready=lambda: True)
        runtime._live_test_lock = threading.RLock()
        runtime._live_test_job = None
        runtime.server = SimpleNamespace(
            state_lock=threading.RLock(),
            skill_lock=threading.RLock(),
            reset_request=False,
            _vision_safe_state=False,
            _simulation_degraded=False,
            _pending_skill_hint=None,
            current_job=None,
            task_switch_request=None,
            task_switch_in_progress=False,
            skill_queue=queue.Queue(),
        )

        idle = runtime.idle_probe()
        self.assertTrue(idle["idle"])
        self.assertTrue(idle["diagnostic_ok"])
        self.assertEqual(idle["protocol"], "behavior-interface-idle-probe-v1")
        self.assertLess(len(repr(idle)), 1200)

        runtime.server.skill_queue.put(object())
        active = runtime.idle_probe()
        self.assertFalse(active["idle"])
        self.assertEqual(active["reason"], "skill_active_or_queued")

    def test_idle_probe_releases_for_operator_finish(self) -> None:
        from behavior_interface_eval_test.operator_scene_control import write_finish_request

        runtime = object.__new__(OfficialPolicyRuntime)
        runtime._source = "hold"
        runtime._last_error = ""
        runtime.downstream = None
        runtime.episode_initializer = SimpleNamespace(ready=lambda: True)
        runtime._live_test_lock = threading.RLock()
        runtime._live_test_job = None
        runtime.server = SimpleNamespace(
            state_lock=threading.RLock(),
            skill_lock=threading.RLock(),
            reset_request=False,
            _vision_safe_state=False,
            _simulation_degraded=False,
            _pending_skill_hint=None,
            current_job=None,
            task_switch_request=None,
            task_switch_in_progress=False,
            skill_queue=queue.Queue(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "BEHAVIOR_EVAL_OPERATOR_DIR": tmp,
                    "BEHAVIOR_EVAL_TEST_PORT": "15065",
                },
                clear=False,
            ):
                idle = runtime.idle_probe()
                self.assertTrue(idle["idle"])
                write_finish_request(15065, reason="model_done")
                pending = runtime.idle_probe()
                self.assertFalse(pending["idle"])
                self.assertEqual(pending["reason"], "finish_pending")

    def test_idle_probe_releases_for_operator_reset(self) -> None:
        runtime = object.__new__(OfficialPolicyRuntime)
        runtime._source = "hold"
        runtime._last_error = ""
        runtime.downstream = None
        runtime.episode_initializer = SimpleNamespace(ready=lambda: True)
        runtime._live_test_lock = threading.RLock()
        runtime._live_test_job = None
        runtime.server = SimpleNamespace(
            state_lock=threading.RLock(),
            skill_lock=threading.RLock(),
            reset_request=False,
            _vision_safe_state=False,
            _simulation_degraded=False,
            _pending_skill_hint=None,
            current_job=None,
            task_switch_request=None,
            task_switch_in_progress=False,
            skill_queue=queue.Queue(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "BEHAVIOR_EVAL_OPERATOR_DIR": tmp,
                    "BEHAVIOR_EVAL_TEST_PORT": "15062",
                },
                clear=False,
            ):
                idle = runtime.idle_probe()
                self.assertTrue(idle["idle"])
                write_reset_request(15062, 304)
                pending = runtime.idle_probe()
                self.assertFalse(pending["idle"])
                self.assertEqual(pending["reason"], "reset_pending")

    def test_policy_server_disables_keepalive_during_slow_evaluator_reset(self) -> None:
        class StopServing(Exception):
            pass

        class FakeServer:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def serve_forever(self):
                raise StopServing

        captured = {}

        def fake_serve(*args, **kwargs):
            captured.update(kwargs)
            return FakeServer()

        runtime = SimpleNamespace(tool_version="official_v2", ui_tool_version="v2")
        with mock.patch(
            "behavior_interface_eval_test.official_policy_interface."
            "websocket_server.serve",
            side_effect=fake_serve,
        ):
            with self.assertRaises(StopServing):
                asyncio.run(serve_policy(runtime, "127.0.0.1", 0))

        self.assertIn("ping_interval", captured)
        self.assertIsNone(captured["ping_interval"])

    def test_skill_wait_fails_fast_after_evaluator_disconnect(self) -> None:
        calls = []

        def wait_for_result(skill_name, **kwargs):
            calls.append((skill_name, kwargs))
            raise TimeoutError("chunk elapsed")

        with self.assertRaisesRegex(
            OfficialEvaluatorDisconnectedError,
            "disconnected while close_gripper was running",
        ):
            _wait_for_skill_result_with_evaluator_liveness(
                wait_for_result,
                lambda: (False, 0),
                "close_gripper",
                timeout_s=90.0,
                poll_s=0.25,
                request_id="job-close",
            )

        self.assertEqual(len(calls), 1)
        self.assertLessEqual(calls[0][1]["timeout_s"], 0.5)

    def test_skill_wait_returns_matching_result_while_connected(self) -> None:
        expected = {"ok": True, "job": "job-close"}
        attempts = 0

        def wait_for_result(_skill_name, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError("chunk elapsed")
            return expected

        result = _wait_for_skill_result_with_evaluator_liveness(
            wait_for_result,
            lambda: (True, 1),
            "close_gripper",
            timeout_s=90.0,
            poll_s=0.25,
            request_id="job-close",
        )

        self.assertEqual(result, expected)
        self.assertEqual(attempts, 2)

    def test_observation_freshness_requires_recent_nonempty_snapshot(self) -> None:
        adapter = ObservationActionAdapter()
        self.assertFalse(adapter.observation_is_fresh(5.0))

        with mock.patch(
            "behavior_interface_eval_test.official_policy_interface.time.time",
            return_value=100.0,
        ):
            adapter.update({"task_id": np.array([1], dtype=np.int64)})
            self.assertTrue(adapter.observation_is_fresh(5.0))

        with mock.patch(
            "behavior_interface_eval_test.official_policy_interface.time.time",
            return_value=106.0,
        ):
            self.assertFalse(adapter.observation_is_fresh(5.0))

    def test_hot_update_borrows_the_owned_snapshot_without_a_second_copy(self) -> None:
        adapter = ObservationActionAdapter()
        source = np.zeros((8, 8, 3), dtype=np.uint8)
        key = "robot_r1::zed_link::rgb"

        hot_snapshot = adapter.update(
            {key: source},
            copy_snapshot=False,
        )
        public_snapshot = adapter.snapshot()

        self.assertIsNot(hot_snapshot.observation[key], source)
        self.assertIs(hot_snapshot.observation[key], adapter._snapshot.observation[key])
        self.assertIsNot(public_snapshot.observation[key], hot_snapshot.observation[key])

    def test_internal_read_snapshot_borrows_but_public_snapshot_still_copies(self) -> None:
        adapter = ObservationActionAdapter()
        source = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        key = "robot_r1::zed_link::rgb"
        adapter.update({key: source}, copy_snapshot=False)

        borrowed = adapter.borrow_snapshot_for_reading()
        public = adapter.snapshot()

        self.assertIs(borrowed.observation[key], adapter._snapshot.observation[key])
        self.assertIsNot(public.observation[key], borrowed.observation[key])
        source.fill(0)
        self.assertGreater(int(borrowed.observation[key].sum()), 0)

    def test_msgpack_bytes_backed_arrays_are_shared_read_only(self) -> None:
        """The protocol decoder's immutable planes need no second memcpy."""

        payload = bytes(range(4 * 4 * 3))
        source = np.ndarray((4, 4, 3), dtype=np.uint8, buffer=payload)
        self.assertFalse(source.flags.writeable)
        adapter = ObservationActionAdapter()
        key = "robot_r1::zed_link::rgb"

        stored = adapter.update({key: source}, copy_snapshot=False)
        public = adapter.snapshot()

        self.assertIs(stored.observation[key], source)
        self.assertIs(public.observation[key], source)
        self.assertFalse(public.observation[key].flags.writeable)
        np.testing.assert_array_equal(public.observation[key], source)

    def test_read_only_view_with_mutable_owner_is_still_copied(self) -> None:
        """A read-only flag alone is not sufficient to prove ownership."""

        payload = bytearray(range(4 * 4 * 3))
        source = np.ndarray((4, 4, 3), dtype=np.uint8, buffer=payload)
        source.setflags(write=False)
        self.assertFalse(source.flags.writeable)
        adapter = ObservationActionAdapter()
        key = "robot_r1::zed_link::rgb"

        stored = adapter.update({key: source}, copy_snapshot=False)
        before = stored.observation[key].copy()
        payload[:] = b"\x00" * len(payload)

        self.assertIsNot(stored.observation[key], source)
        self.assertFalse(np.array_equal(stored.observation[key], np.zeros_like(source)))
        np.testing.assert_array_equal(stored.observation[key], before)

    def test_camera_frames_can_select_ui_roles_without_changing_default(self) -> None:
        adapter = ObservationActionAdapter()
        adapter.update(
            {
                "robot_r1::zed_link::rgb": np.full((4, 5, 3), 1, np.uint8),
                "robot_r1::left_realsense::rgb": np.full(
                    (4, 5, 3), 2, np.uint8
                ),
                "robot_r1::right_realsense::rgb": np.full(
                    (4, 5, 3), 3, np.uint8
                ),
            }
        )

        self.assertEqual(
            set(adapter.camera_frames()),
            {"head", "main", "left_wrist", "right_wrist"},
        )
        selected = adapter.camera_frames(
            roles={"left_wrist"},
            include_main=False,
        )
        self.assertEqual(set(selected), {"left_wrist"})

    def test_pipeline_timing_reports_stable_aggregates(self) -> None:
        timing = _PipelineTiming(ema_alpha=0.5)
        timing.record({"snapshot": 2.0, "total": 5.0})
        timing.record({"snapshot": 4.0, "total": 9.0})

        status = timing.status()

        self.assertEqual(status["steps"], 2)
        self.assertEqual(status["phases"]["snapshot"]["samples"], 2)
        self.assertEqual(status["phases"]["snapshot"]["last_ms"], 4.0)
        self.assertEqual(status["phases"]["snapshot"]["ema_ms"], 3.0)
        self.assertEqual(status["phases"]["snapshot"]["mean_ms"], 3.0)
        self.assertNotIn("total_ms", status["phases"]["snapshot"])

    def test_role_specific_camera_access_returns_only_requested_feed(self) -> None:
        adapter = ObservationActionAdapter()
        right_rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        right_rgb[..., :] = [1, 2, 3]
        right_depth = np.full((4, 5), 1.25, dtype=np.float32)
        adapter.update(
            {
                "robot_r1::right_realsense::rgb": right_rgb,
                "robot_r1::right_realsense::depth_linear": right_depth,
                "robot_r1::left_realsense::rgb": np.full(
                    (4, 5, 3), 9, dtype=np.uint8
                ),
                "robot_r1::left_realsense::depth_linear": np.full(
                    (4, 5), 9.0, dtype=np.float32
                ),
            }
        )

        frame = adapter.camera_frame("right_wrist")
        depth = adapter.camera_depth_frame("right_wrist")

        self.assertIsNotNone(frame)
        self.assertIsNotNone(depth)
        np.testing.assert_array_equal(frame[0, 0], [3, 2, 1])
        np.testing.assert_allclose(depth, 1.25)
        self.assertIsNone(adapter.camera_frame("head"))
        self.assertIsNone(adapter.camera_depth_frame("head"))

    def test_observation_allowlist_rejects_privileged_modalities(self) -> None:
        allowed, rejected = filter_allowed_observation(
            {
                "robot_r1::head::rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "robot_r1::head::depth_linear": np.ones((2, 2), dtype=np.float32),
                "robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32),
                "task_id": np.array([5], dtype=np.int64),
                "robot_r1::head::seg_instance_id": np.zeros((2, 2), dtype=np.int32),
                "object_pose": np.zeros(7, dtype=np.float32),
            }
        )
        self.assertEqual(
            set(allowed),
            {
                "robot_r1::head::rgb",
                "robot_r1::head::depth_linear",
                "robot_r1::proprio",
                "task_id",
            },
        )
        self.assertEqual(
            rejected,
            ["object_pose", "robot_r1::head::seg_instance_id"],
        )

    def test_bddl_envelope_is_ui_only_and_not_policy_observation(self) -> None:
        adapter = ObservationActionAdapter()
        envelope = encode_bddl_progress(
            {
                "items": [
                    {
                        "index": 0,
                        "label": "toy in toy box",
                        "full": "toy in toy box",
                        "satisfied": True,
                    }
                ],
                "ok": True,
            }
        )

        snapshot = adapter.update(
            {
                "robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32),
                UI_BDDL_PROGRESS_KEY: envelope,
            }
        )

        self.assertNotIn(UI_BDDL_PROGRESS_KEY, snapshot.observation)
        self.assertNotIn(UI_BDDL_PROGRESS_KEY, snapshot.rejected_keys)
        self.assertNotIn(UI_BDDL_PROGRESS_KEY, adapter.status()["keys"])
        self.assertTrue(adapter.ui_goal_progress()["complete"])
        self.assertTrue(adapter.ui_goal_status()["available"])
        self.assertFalse(adapter.ui_goal_status()["policy_visible"])

    def test_runtime_does_not_forward_bddl_envelope_downstream(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        adapter = ObservationActionAdapter()
        world = self._world(adapter)
        world.set_episode_initialized(True)
        downstream = SimpleNamespace(
            act=mock.Mock(return_value=np.zeros(ACTION_DIM, dtype=np.float32))
        )
        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime._step_lock = threading.RLock()
        runtime.adapter = adapter
        runtime.server = SimpleNamespace(world=world, log=lambda _: None)
        runtime.episode_initializer = SimpleNamespace(ready=lambda: True)
        runtime.downstream = downstream
        runtime._source = "hold"
        runtime._last_error = ""
        runtime._update_ui = lambda: None
        runtime._tool_action = mock.Mock(return_value=None)

        runtime.act(
            {
                "robot_r1::proprio": proprio,
                UI_BDDL_PROGRESS_KEY: encode_bddl_progress(
                    {
                        "items": [
                            {
                                "index": 0,
                                "label": "goal",
                                "full": "goal",
                                "satisfied": False,
                            }
                        ],
                        "ok": True,
                    }
                ),
            }
        )

        forwarded = downstream.act.call_args.args[0]
        self.assertIn("robot_r1::proprio", forwarded)
        self.assertNotIn(UI_BDDL_PROGRESS_KEY, forwarded)

    def test_runtime_ui_cache_receives_display_only_bddl_state(self) -> None:
        adapter = ObservationActionAdapter()
        adapter.update(
            {
                UI_BDDL_PROGRESS_KEY: encode_bddl_progress(
                    {
                        "items": [
                            {
                                "index": 0,
                                "label": "goal",
                                "full": "goal",
                                "satisfied": True,
                            }
                        ],
                        "ok": True,
                    }
                )
            }
        )
        server = SimpleNamespace(
            state_lock=threading.RLock(),
            world=SimpleNamespace(eef_pose=lambda arm: {}),
            tick=0,
            fps=0.0,
            _cached_goals={},
        )
        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime.adapter = adapter
        runtime.server = server
        runtime._last_obs_mono = None
        runtime._fps_ema = 0.0

        runtime._update_ui()

        self.assertTrue(server._cached_goals["complete"])
        self.assertEqual(server._cached_goals["satisfied"], 1)
        self.assertEqual(server._cached_goals["total"], 1)

    def test_runtime_ui_hot_path_skips_head_and_inactive_wrists(self) -> None:
        adapter = mock.Mock()
        adapter.camera_frames.return_value = {}
        adapter.ui_goal_progress.return_value = {}
        server = SimpleNamespace(
            state_lock=threading.RLock(),
            world=SimpleNamespace(eef_pose=lambda arm: {}),
            tick=0,
            fps=0.0,
            _cached_goals={},
            get_feed_active=lambda: {
                "head": True,
                "main": False,
                "left_wrist": False,
                "right_wrist": False,
            },
        )
        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime.adapter = adapter
        runtime.server = server
        runtime._last_obs_mono = None
        runtime._fps_ema = 0.0

        runtime._update_ui()

        adapter.camera_frames.assert_called_once_with(
            roles=set(),
            include_main=False,
        )

    def test_hold_action_uses_observed_absolute_joint_targets(self) -> None:
        proprio = np.arange(PROPRIO_DIM, dtype=np.float32)
        adapter = ObservationActionAdapter()
        adapter.update({"robot_r1::proprio": proprio})
        action = adapter.hold_action()

        self.assertEqual(action.shape, (ACTION_DIM,))
        np.testing.assert_array_equal(action[ACTION_SLICES["base"]], np.zeros(3))
        expected_trunk = proprio[PROPRIO_SLICES["trunk_qpos"]].copy()
        expected_trunk[3] = 0.0
        np.testing.assert_array_equal(
            action[ACTION_SLICES["trunk"]],
            expected_trunk,
        )
        for side in ("left", "right"):
            expected = proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]].copy()
            if ARM_DOF == 8:
                expected[7] = 0.0
            np.testing.assert_array_equal(
                action[ACTION_SLICES[f"arm_{side}"]],
                expected,
            )

    def test_runtime_idle_hold_keeps_fixed_arm_pins(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        adapter = ObservationActionAdapter()
        adapter.update({"robot_r1::proprio": proprio})
        world = self._world(adapter)
        pins = {
            "left": np.linspace(
                0.1,
                0.1 * ARM_DOF,
                ARM_DOF,
                dtype=np.float32,
            ),
            "right": np.linspace(
                -0.1,
                -0.1 * ARM_DOF,
                ARM_DOF,
                dtype=np.float32,
            ),
        }
        for side, pin in pins.items():
            if ARM_DOF == 8:
                pin[-1] = 0.0
            world.set_arm_pin_qpos(side, pin)
        adapter.record_action(world.hold_action())
        for side, pin in pins.items():
            observed = pin + 0.15
            if ARM_DOF == 8:
                observed[-1] = 0.0
            proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]] = observed

        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime._step_lock = threading.RLock()
        runtime.adapter = adapter
        runtime.server = SimpleNamespace(world=world, log=lambda _: None)
        runtime.episode_initializer = SimpleNamespace(ready=lambda: True)
        runtime.downstream = None
        runtime._source = "hold"
        runtime._last_error = ""
        runtime._update_ui = lambda: None
        runtime._tool_action = mock.Mock(return_value=None)
        world.set_episode_initialized(True)

        action = runtime.act({"robot_r1::proprio": proprio})

        for side, pin in pins.items():
            arm_action = action[ACTION_SLICES[f"arm_{side}"]]
            np.testing.assert_allclose(arm_action, pin, atol=1e-7)
            self.assertGreater(
                float(
                    np.max(
                        np.abs(
                            arm_action
                            - proprio[PROPRIO_SLICES[f"arm_{side}_qpos"]]
                        )
                    )
                ),
                0.05,
            )
        np.testing.assert_array_equal(
            action[ACTION_SLICES["base"]],
            np.zeros(3),
        )
        self.assertEqual(runtime._source, "hold")
        runtime._tool_action.assert_called_once_with()

    def test_runtime_idle_hold_recomputes_tool_gripper_qpos_servo(self) -> None:
        adapter = ObservationActionAdapter()
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.0495, 0.0497]
        adapter.update({"robot_r1::proprio": proprio})
        world = self._world(adapter)
        if not world.gripper_uses_effort("right"):
            self.skipTest("independent effort profile is required")
        world.capture_gripper_hold_qpos("right", allow_close_effort=False)
        adapter.record_action(world.hold_action())

        disturbed = proprio.copy()
        disturbed[PROPRIO_SLICES["gripper_right_qpos"]] = [0.044, 0.047]
        disturbed[PROPRIO_SLICES["gripper_right_qvel"]] = [-0.02, -0.01]
        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime._step_lock = threading.RLock()
        runtime.adapter = adapter
        runtime.server = SimpleNamespace(world=world, log=lambda _: None)
        runtime.episode_initializer = SimpleNamespace(ready=lambda: True)
        runtime.downstream = None
        runtime._source = "human/tool"
        runtime._last_error = ""
        runtime._update_ui = lambda: None
        runtime._tool_action = mock.Mock(return_value=None)
        world.set_episode_initialized(True)

        action = runtime.act({"robot_r1::proprio": disturbed})

        correction = action[ACTION_SLICES["gripper_right"]]
        self.assertTrue(bool(np.all(correction > 0.0)), correction.tolist())
        self.assertLessEqual(float(np.max(correction)), 0.5 + 1e-7)

    def test_successful_downstream_action_clears_tool_gripper_hold(self) -> None:
        adapter = ObservationActionAdapter()
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["gripper_right_qpos"]] = [0.040, 0.041]
        adapter.update({"robot_r1::proprio": proprio})
        world = self._world(adapter)
        if not world.gripper_uses_effort("right"):
            self.skipTest("independent effort profile is required")
        world.capture_gripper_hold_qpos("right")
        downstream_action = np.zeros(ACTION_DIM, dtype=np.float32)
        downstream_action[ACTION_SLICES["gripper_right"]] = [-0.2, -0.3]
        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime._step_lock = threading.RLock()
        runtime.adapter = adapter
        runtime.server = SimpleNamespace(world=world, log=lambda _: None)
        runtime.episode_initializer = SimpleNamespace(ready=lambda: True)
        runtime.downstream = SimpleNamespace(
            act=mock.Mock(return_value=downstream_action)
        )
        runtime._source = "hold"
        runtime._last_error = ""
        runtime._update_ui = lambda: None
        runtime._tool_action = mock.Mock(return_value=None)
        world.set_episode_initialized(True)

        action = runtime.act({"robot_r1::proprio": proprio})

        np.testing.assert_allclose(
            action[ACTION_SLICES["gripper_right"]],
            [-0.2, -0.3],
            atol=1e-7,
        )
        self.assertIsNone(world.gripper_hold_qpos_list("right"))
        self.assertEqual(runtime._source, "downstream-model")

    def test_runtime_error_hold_keeps_last_downstream_targets(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        adapter = ObservationActionAdapter()
        adapter.update({"robot_r1::proprio": proprio})
        world = self._world(adapter)
        last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        last_action[ACTION_SLICES["arm_left"]] = 0.25
        last_action[ACTION_SLICES["arm_right"]] = -0.35
        if ARM_DOF == 8:
            last_action[ACTION_SLICES["arm_left"]][-1] = 0.0
            last_action[ACTION_SLICES["arm_right"]][-1] = 0.0
        adapter.record_action(last_action)
        world.set_arm_pin_qpos("left", np.zeros(ARM_DOF))
        world.set_arm_pin_qpos("right", np.zeros(ARM_DOF))

        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime._step_lock = threading.RLock()
        runtime.adapter = adapter
        runtime.server = SimpleNamespace(world=world, log=lambda _: None)
        runtime.episode_initializer = SimpleNamespace(ready=lambda: True)
        runtime.downstream = SimpleNamespace(
            act=mock.Mock(side_effect=RuntimeError("downstream failed"))
        )
        runtime._source = "downstream-model"
        runtime._last_error = ""
        runtime._update_ui = lambda: None
        runtime._tool_action = mock.Mock(return_value=None)

        action = runtime.act({"robot_r1::proprio": proprio})

        np.testing.assert_array_equal(
            action[ACTION_SLICES["arm_left"]],
            last_action[ACTION_SLICES["arm_left"]],
        )
        np.testing.assert_array_equal(
            action[ACTION_SLICES["arm_right"]],
            last_action[ACTION_SLICES["arm_right"]],
        )
        np.testing.assert_array_equal(
            action[ACTION_SLICES["base"]],
            np.zeros(3),
        )
        self.assertEqual(runtime._source, "hold")
        self.assertIn("downstream failed", runtime._last_error)

    def test_legacy_action_layout_conversion(self) -> None:
        adapter = ObservationActionAdapter()
        legacy = np.zeros(25, dtype=np.float32)
        legacy[0:3] = [0.1, -0.2, 0.3]
        legacy[3:7] = [0.4, 0.5, 0.6, 0.7]
        legacy[7:14] = np.arange(1.0, 8.0)
        legacy[14:16] = [-0.2, -0.3]
        legacy[16:23] = np.arange(11.0, 18.0)
        legacy[23:25] = [0.4, 0.5]
        action = adapter.adapt_legacy_action(legacy)

        self.assertEqual(action.shape, (ACTION_DIM,))
        expected_base = legacy[0:3].copy()
        np.testing.assert_array_equal(action[ACTION_SLICES["base"]], expected_base)
        expected_trunk = legacy[3:7].copy()
        expected_trunk[3] = 0.0
        np.testing.assert_array_equal(
            action[ACTION_SLICES["trunk"]],
            expected_trunk,
        )
        np.testing.assert_array_equal(
            action[ACTION_SLICES["arm_left"]][:7],
            legacy[7:14],
        )
        np.testing.assert_array_equal(
            action[ACTION_SLICES["arm_right"]][:7],
            legacy[16:23],
        )
        left_gripper = action[ACTION_SLICES["gripper_left"]]
        right_gripper = action[ACTION_SLICES["gripper_right"]]
        if left_gripper.size == 2:
            np.testing.assert_array_equal(left_gripper, legacy[14:16])
            np.testing.assert_array_equal(right_gripper, legacy[23:25])
        else:
            self.assertEqual(float(left_gripper[0]), float(np.mean(legacy[14:16])))
            self.assertEqual(float(right_gripper[0]), float(np.mean(legacy[23:25])))
        if ARM_DOF == 8:
            self.assertEqual(float(action[ACTION_SLICES["arm_left"]][-1]), 0.0)
            self.assertEqual(float(action[ACTION_SLICES["arm_right"]][-1]), 0.0)

    def test_observation_backed_world_emits_profile_actions(self) -> None:
        proprio = np.arange(PROPRIO_DIM, dtype=np.float32)
        adapter = ObservationActionAdapter()
        adapter.update({"robot_r1::proprio": proprio})
        world = self._world(adapter)

        action = world.make_action(
            base=[0.2, -0.1, 0.3],
            trunk=[1.0, 2.0, 3.0, 4.0],
            arm_right=np.arange(10.0, 10.0 + ARM_DOF),
            gripper_left=[-1.0, -1.0],
        )

        self.assertEqual(action.shape, (ACTION_DIM,))
        np.testing.assert_allclose(action[ACTION_SLICES["base"]], [0.2, -0.1, 0.3])
        np.testing.assert_allclose(action[ACTION_SLICES["trunk"]], [1, 2, 3, 0])
        self.assertEqual(world.trunk_pin_qpos_list()[-1], 0.0)
        expected_left = proprio[PROPRIO_SLICES["arm_left_qpos"]].copy()
        expected_right = np.arange(10.0, 10.0 + ARM_DOF)
        if ARM_DOF == 8:
            expected_left[7] = 0.0
            expected_right[7] = 0.0
        np.testing.assert_array_equal(
            action[ACTION_SLICES["arm_left"]],
            expected_left,
        )
        np.testing.assert_allclose(
            action[ACTION_SLICES["arm_right"]],
            expected_right,
        )
        gripper_dim = ACTION_SLICES["gripper_left"].stop - ACTION_SLICES["gripper_left"].start
        np.testing.assert_allclose(
            action[ACTION_SLICES["gripper_left"]],
            [-0.1, -0.1] if gripper_dim == 2 else [-1.0],
        )
        np.testing.assert_allclose(
            action[ACTION_SLICES["gripper_right"]],
            [0.0, 0.0] if gripper_dim == 2 else [1.0],
        )

    def test_final_action_boundary_preserves_lateral_base_and_locks_trunk_yaw(self) -> None:
        adapter = ObservationActionAdapter()
        direct = np.zeros(ACTION_DIM, dtype=np.float32)
        direct[ACTION_SLICES["base"]] = [0.2, -0.7, 0.3]
        direct[ACTION_SLICES["trunk"]] = [0.1, 0.2, 0.3, 1.4]

        recorded = adapter.record_action(direct)
        legacy = np.zeros(25, dtype=np.float32)
        legacy[0:3] = [-0.2, 0.8, -0.3]
        legacy[3:7] = [-0.1, -0.2, -0.3, -1.5]
        adapted = adapter.adapt_legacy_action(legacy)

        np.testing.assert_allclose(
            recorded[ACTION_SLICES["base"]],
            [0.2, -0.7, 0.3],
        )
        np.testing.assert_allclose(
            adapted[ACTION_SLICES["base"]],
            [-0.2, 0.8, -0.3],
        )
        np.testing.assert_allclose(
            recorded[ACTION_SLICES["trunk"]],
            [0.1, 0.2, 0.3, 0.0],
        )
        np.testing.assert_allclose(
            adapted[ACTION_SLICES["trunk"]],
            [-0.1, -0.2, -0.3, 0.0],
        )

    def test_world_lateral_command_is_left_positive(self) -> None:
        adapter = ObservationActionAdapter()
        world = self._world(adapter)

        action = world.set_base_velocity(0.0, 0.5, 0.0)

        np.testing.assert_allclose(action[ACTION_SLICES["base"]], [0.0, 0.5, 0.0])
        self.assertGreater(float(world.robot_pose().pos[1]), 0.0)

    def test_observation_backed_world_pins_arm_and_holds_gripper_qpos_across_base_steps(
        self,
    ) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        adapter = ObservationActionAdapter()
        adapter.update({"robot_r1::proprio": proprio})
        world = self._world(adapter)
        target = np.linspace(0.1, 0.1 * ARM_DOF, ARM_DOF, dtype=np.float32)
        expected_target = target.copy()
        if ARM_DOF == 8:
            expected_target[7] = 0.0

        first = world.make_action(arm_left=target, gripper_right=[-1.0])
        adapter.record_action(first)
        following = world.set_base_velocity(0.25, 0.0, 0.0)

        np.testing.assert_allclose(
            following[ACTION_SLICES["arm_left"]],
            expected_target,
        )
        np.testing.assert_allclose(
            following[ACTION_SLICES["gripper_right"]],
            [0.0, 0.0]
            if ACTION_SLICES["gripper_right"].stop - ACTION_SLICES["gripper_right"].start == 2
            else [-1.0],
        )
        if world.gripper_uses_effort("right"):
            np.testing.assert_allclose(
                world.gripper_hold_qpos_list("right"),
                [0.0, 0.0],
            )
        np.testing.assert_allclose(
            following[ACTION_SLICES["base"]],
            [0.25, 0.0, 0.0],
        )
        self.assertGreater(float(world.robot_pose().pos[0]), 0.0)

    def test_final_action_boundary_keeps_close_until_explicit_open(self) -> None:
        adapter = ObservationActionAdapter()
        adapter.update(
            {"robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32)}
        )
        world = self._world(adapter)

        adapter.record_action(world.make_action(gripper_left=[-1.0]))
        world.latch_gripper_close_keepalive(
            "left",
            effort=[-1.0, -1.0]
            if world.gripper_uses_effort("left")
            else None,
        )
        unrelated = np.zeros(ACTION_DIM, dtype=np.float32)
        unrelated[ACTION_SLICES["gripper_left"]] = 1.0
        kept = adapter.record_action(unrelated)
        np.testing.assert_allclose(
            kept[ACTION_SLICES["gripper_left"]],
            [-1.0, -1.0]
            if world.gripper_uses_effort("left")
            else [-1.0],
        )

        adapter.record_action(world.make_action(gripper_left=[1.0]))
        released = adapter.record_action(unrelated)
        np.testing.assert_allclose(
            released[ACTION_SLICES["gripper_left"]],
            [1.0, 1.0]
            if world.gripper_uses_effort("left")
            else [1.0],
        )

    def test_profile_does_not_expose_assisted_grasp_truth(self) -> None:
        self.assertNotIn("grasp_left", PROPRIO_SLICES)
        self.assertNotIn("grasp_right", PROPRIO_SLICES)
        adapter = ObservationActionAdapter()
        world = self._world(adapter)
        self.assertFalse(hasattr(world, "gripper_grasp_state"))

    def test_base_odometry_reconciles_command_prediction_with_proprio(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        adapter = ObservationActionAdapter()
        adapter.update({"robot_r1::proprio": proprio})
        world = self._world(adapter)

        world.set_base_velocity(0.30, 0.0, 0.0)
        self.assertGreater(float(world.robot_pose().pos[0]), 0.0)
        report = world.reconcile_base_odometry_from_proprio(1.0 / 30.0)

        self.assertTrue(report["corrected"])
        self.assertAlmostEqual(float(world.robot_pose().pos[0]), 0.0)
        self.assertEqual(
            world.base_odometry_status()["source"],
            "proprio_base_qvel_magnitude_command_direction",
        )

        world.set_base_velocity(0.30, 0.0, 0.0)
        proprio[0] = 0.12
        adapter.update({"robot_r1::proprio": proprio})
        world.reconcile_base_odometry_from_proprio(1.0 / 30.0)

        self.assertAlmostEqual(
            float(world.robot_pose().pos[0]),
            0.12 / 30.0,
            places=8,
        )

    def test_base_odometry_rejects_qvel_outside_controller_envelope(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        adapter = ObservationActionAdapter()
        adapter.update({"robot_r1::proprio": proprio})
        world = self._world(adapter)

        world.set_base_velocity(0.0, 0.0, 0.0)
        pose_before = world.policy_local_base_pose()
        proprio[PROPRIO_SLICES["base_qvel"]] = [
            -0.251378,
            0.063917,
            -2.486668,
        ]
        adapter.update({"robot_r1::proprio": proprio})
        report = world.reconcile_base_odometry_from_proprio(1.0 / 30.0)

        self.assertFalse(report["corrected"])
        self.assertEqual(
            report["reason"],
            "base_qvel_outside_controller_envelope",
        )
        self.assertEqual(
            report["source"],
            "command_prediction_incoherent_qvel",
        )
        self.assertGreater(abs(report["raw_observed_base_qvel"][2]), 1.25)
        np.testing.assert_allclose(report["commanded_base_velocity"], 0.0)
        np.testing.assert_allclose(
            world.policy_local_base_pose(),
            pose_before,
            atol=0.0,
        )
        status = world.base_odometry_status()
        self.assertEqual(status["rejections"], 1)
        self.assertFalse(status["pending_command_prediction"])

    def test_base_odometry_keeps_qvel_reconciliation_during_active_command(
        self,
    ) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        adapter = ObservationActionAdapter()
        adapter.update({"robot_r1::proprio": proprio})
        world = self._world(adapter)

        world.set_base_velocity(0.5, 0.0, 0.0)
        proprio[PROPRIO_SLICES["base_qvel"]] = [1.2, 0.0, 0.0]
        adapter.update({"robot_r1::proprio": proprio})
        report = world.reconcile_base_odometry_from_proprio(1.0 / 30.0)

        self.assertTrue(report["corrected"])
        self.assertEqual(
            report["source"],
            "proprio_base_qvel_magnitude_command_direction",
        )
        self.assertAlmostEqual(
            float(world.policy_local_base_pose()[0]),
            1.2 / 30.0,
            places=7,
        )
        self.assertEqual(world.base_odometry_status()["rejections"], 0)

    def test_observation_backed_world_reset_clears_action_pins(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["arm_left_qpos"]] = 0.4
        adapter = ObservationActionAdapter()
        adapter.update({"robot_r1::proprio": proprio})
        world = self._world(adapter)
        adapter.record_action(
            world.make_action(arm_left=np.ones(ARM_DOF), gripper_left=[-1.0])
        )

        adapter.reset()
        world.reset_observation_state()
        action = world.hold_action()

        np.testing.assert_allclose(action[ACTION_SLICES["arm_left"]], 0.0)
        expected_gripper = (
            [0.0, 0.0]
            if ACTION_SLICES["gripper_left"].stop - ACTION_SLICES["gripper_left"].start == 2
            else [1.0]
        )
        np.testing.assert_allclose(
            action[ACTION_SLICES["gripper_left"]],
            expected_gripper,
        )
        self.assertFalse(adapter.observation_is_fresh(5.0))

    def test_observation_backed_eef_pose_declares_local_command_odometry(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["eef_left_pos"]] = [0.4, 0.2, 1.1]
        proprio[PROPRIO_SLICES["eef_left_quat"]] = [0.0, 0.0, 0.0, 1.0]
        adapter = ObservationActionAdapter()
        adapter.update({"robot_r1::proprio": proprio})

        pose = self._world(adapter).eef_pose("left")

        self.assertEqual(pose["frame"], "local_command_odometry")
        np.testing.assert_allclose(pose["pos"], [0.4, 0.2, 1.1])

    def test_camera_relative_pose_layout(self) -> None:
        adapter = ObservationActionAdapter()
        adapter.update(
            {
                "robot_r1::cam_rel_poses": np.arange(21, dtype=np.float32),
            }
        )

        poses = adapter.camera_relative_poses()

        self.assertEqual(set(poses), {"left_wrist", "right_wrist", "head"})
        np.testing.assert_allclose(poses["left_wrist"]["pos"], [0, 1, 2])
        np.testing.assert_allclose(poses["right_wrist"]["quat"], [10, 11, 12, 13])
        np.testing.assert_allclose(poses["head"]["pos"], [14, 15, 16])

    def test_official_live_head_renderer_uses_frozen_v2_path_hud(self) -> None:
        camera_rotation = np.array(
            [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        camera_quat = mat_to_quat_xyzw(camera_rotation).astype(np.float32)
        camera_poses = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
            * 2
            + [
                0.0,
                0.0,
                1.0,
                *camera_quat.tolist(),
            ],
            dtype=np.float32,
        )
        source_rgb = np.zeros((160, 200, 3), dtype=np.uint8)
        snapshot = ObservationSnapshot(
            observation={
                "robot_r1::head::rgb": source_rgb,
                "robot_r1::head::depth_linear": np.full(
                    (160, 200),
                    10.0,
                    dtype=np.float32,
                ),
                "robot_r1::cam_rel_poses": camera_poses,
            },
            rejected_keys=[],
            received_ts=1.0,
            sequence=7,
        )
        adapter = ObservationActionAdapter()
        world = self._world(adapter)

        output, status = _render_official_live_head_frame(snapshot, world)

        self.assertTrue(status["ok"], status)
        self.assertTrue(status["overlay_applied"])
        self.assertEqual(status["sequence"], 7)
        self.assertEqual(status["path_width_m"], 0.8)
        self.assertEqual(status["path_side_inset_m"], 0.0874765)
        self.assertEqual(status["path_side_margin_m"], 0.0)
        self.assertEqual(
            status["distance_origin"],
            "r1pro_base_front_collision_edge",
        )
        self.assertAlmostEqual(
            status["distance_origin_base_local_x_m"],
            0.23942779099844563,
        )
        self.assertTrue(status["near_edge_border_enabled"])
        self.assertTrue(status["near_edge_center_marker_enabled"])
        self.assertEqual(
            status["near_edge_lateral_marker_offsets_m"],
            [-0.4, -0.2, 0.0, 0.2, 0.4],
        )
        self.assertEqual(
            status["near_edge_lateral_distances_m"],
            [0.2, 0.4],
        )
        self.assertEqual(
            status["near_edge_lateral_labels"],
            ["0.4m", "0.2m", "0.2m", "0.4m"],
        )
        self.assertEqual(status["near_edge_outer_label_outset_m"], 0.12)
        self.assertEqual(status["label_color_rgb"], [12, 92, 205])
        self.assertFalse(status["label_outline_enabled"])
        self.assertEqual(status["label_stroke_width_px"], 0)
        self.assertEqual(status["direction_arrow_count"], 6)
        self.assertFalse(status["yellow_overlay_enabled"])
        self.assertGreater(status["visible_pixel_count"], 0)
        self.assertGreater(status["hud_visible_pixel_count"], 0)
        self.assertIsNotNone(output)
        blue = (
            (output[..., 0] > output[..., 2] + 50)
            & (output[..., 0] > output[..., 1] + 20)
        )
        yellow = (
            (output[..., 2] > 120)
            & (output[..., 1] > 100)
            & (output[..., 0] < 100)
        )
        self.assertGreater(int(blue.sum()), 0)
        self.assertEqual(int(yellow.sum()), 0)
        self.assertEqual(int(source_rgb.sum()), 0)

    def test_official_live_head_renderer_falls_back_without_depth(self) -> None:
        source_rgb = np.zeros((8, 10, 3), dtype=np.uint8)
        source_rgb[..., 0] = 11
        source_rgb[..., 1] = 22
        source_rgb[..., 2] = 33
        snapshot = ObservationSnapshot(
            observation={
                "robot_r1::head::rgb": source_rgb,
            },
            rejected_keys=[],
            received_ts=1.0,
            sequence=8,
        )

        output, status = _render_official_live_head_frame(
            snapshot,
            self._world(ObservationActionAdapter()),
        )

        self.assertFalse(status["ok"])
        self.assertFalse(status["overlay_applied"])
        self.assertIn("depth_linear", status["error"])
        np.testing.assert_array_equal(output[..., 0], 33)
        np.testing.assert_array_equal(output[..., 1], 22)
        np.testing.assert_array_equal(output[..., 2], 11)

    def test_live_head_hud_renders_async_and_keeps_only_latest_pending(self) -> None:
        adapter = ObservationActionAdapter()
        world = self._world(adapter)
        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime.adapter = adapter
        runtime.server = SimpleNamespace(
            world=world,
            main_w=24,
            main_h=16,
            _decorate=lambda frames: frames,
            frame_lock=threading.RLock(),
            frames={},
            frame_ids={},
            frame_id=0,
            frame_updated_ts={},
        )
        runtime._live_overlay_lock = threading.RLock()
        runtime._live_overlay_generation = 0
        runtime._live_overlay_future = None
        runtime._live_overlay_inflight_sequence = -1
        runtime._live_overlay_pending = None
        runtime._live_overlay_sequence = -1
        runtime._live_overlay_updated_ts = 0.0
        runtime._live_overlay_raw_head = None
        runtime._live_overlay_frames = {}
        runtime._live_overlay_preview_sequence = -1
        runtime._live_overlay_preview_updated_ts = 0.0
        runtime._live_overlay_preview_head = None
        runtime._live_overlay_preview_frames = {}
        runtime._live_overlay_dropped_requests = 0
        runtime._live_overlay_status = {}

        poses = np.zeros(21, dtype=np.float32)
        poses[[6, 13, 20]] = 1.0

        def publish(value: int) -> None:
            adapter.update(
                {
                    "robot_r1::proprio": np.zeros(
                        PROPRIO_DIM,
                        dtype=np.float32,
                    ),
                    "robot_r1::head::rgb": np.full(
                        (12, 16, 3),
                        value,
                        dtype=np.uint8,
                    ),
                    "robot_r1::head::depth_linear": np.full(
                        (12, 16),
                        2.0,
                        dtype=np.float32,
                    ),
                    "robot_r1::cam_rel_poses": poses,
                }
            )

        submitted = []
        futures = []

        def submit(_renderer, prepared):
            future = Future()
            submitted.append(prepared)
            futures.append(future)
            return future

        with mock.patch(
            "behavior_interface_eval_test.official_policy_interface."
            "_LIVE_OVERLAY_RENDER_EXECUTOR.submit",
            side_effect=submit,
        ):
            publish(1)
            preview, preview_id, _ = runtime.live_display_frame("head")
            self.assertIsNotNone(preview)
            self.assertLess(preview_id, 0)

            publish(2)
            runtime.live_display_frame("head")
            publish(3)
            runtime.live_display_frame("head")
            self.assertEqual([item.sequence for item in submitted], [1])
            self.assertEqual(runtime._live_overlay_dropped_requests, 1)

            rendered_one = np.full((12, 16, 3), 11, dtype=np.uint8)
            futures[0].set_result(
                (
                    rendered_one,
                    {"ok": True, "overlay_applied": True, "sequence": 1},
                )
            )
            self.assertEqual([item.sequence for item in submitted], [1, 3])

            rendered_three = np.full((12, 16, 3), 33, dtype=np.uint8)
            futures[1].set_result(
                (
                    rendered_three,
                    {"ok": True, "overlay_applied": True, "sequence": 3},
                )
            )
            head, frame_id, _ = runtime.live_display_frame("head")
            self.assertEqual(frame_id, 3)
            np.testing.assert_array_equal(head, rendered_three)
            self.assertEqual(set(runtime._live_overlay_frames), {"head"})
            main, main_id, _ = runtime.live_display_frame("main")
            self.assertEqual(main_id, 3)
            self.assertEqual(main.shape[:2], (16, 24))
            self.assertEqual(
                set(runtime._live_overlay_frames),
                {"head", "main"},
            )

    def test_official_live_frame_routes_replace_only_head_and_main(self) -> None:
        app = Flask(__name__)

        @app.get("/video/<feed>", endpoint="video")
        def video(feed):
            return f"original video {feed}"

        @app.get("/api/frame/<feed>.jpg", endpoint="api_frame_jpg")
        def frame_jpg(feed):
            return f"original frame {feed}"

        frame = np.full((12, 16, 3), 127, dtype=np.uint8)
        runtime = SimpleNamespace(
            live_display_frame=lambda feed: (frame, 42, 123.0),
        )
        install_official_live_frame_routes(app, runtime)
        client = app.test_client()

        for feed in ("head", "main"):
            response = client.get(f"/api/frame/{feed}.jpg")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.headers["X-Official-Overlay"],
                "frozen-v2-base-path-hud",
            )
            self.assertEqual(response.headers["X-Frame-Id"], "42")
            self.assertGreater(len(response.data), 0)
        self.assertEqual(
            client.get("/api/frame/left_wrist.jpg").data,
            b"original frame left_wrist",
        )

    def test_local_odometry_rotates_robot_relative_pose(self) -> None:
        adapter = ObservationActionAdapter()
        world = self._world(adapter)
        world._mock_base = np.array([1.0, 2.0, np.pi / 2.0])

        pose = world.local_pose_from_robot_relative(
            [1.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        )

        np.testing.assert_allclose(pose["pos"], [1.0, 3.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(
            pose["quat"],
            [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
            atol=1e-6,
        )

    def test_strict_tool_capability_validation(self) -> None:
        report = capability_report()
        self.assertEqual(set(report["tools"]), set(PUBLIC_SKILLS))
        self.assertEqual(len(report["tools"]), 23)
        self.assertEqual(
            sum(report["counts"].values()),
            23,
        )
        self.assertEqual(
            report["feasibility_counts"],
            {
                "implemented_now": 10,
                "portable": 11,
                "replacement_only": 1,
                "forbidden": 1,
            },
        )
        normalized = validate_submission(
            "move_in_robot_coord",
            {"forward": 0.1, "spin": 5.0, "pitch": 5.0, "upward": -0.02},
        )
        self.assertIs(normalized["nav_guard"], False)
        validate_submission("reset_body", {})
        with self.assertRaisesRegex(StrictToolBoundaryError, "world EEF orientation"):
            validate_submission("reset_body", {"keep_ori_arm": "right"})
        with self.assertRaisesRegex(StrictToolBoundaryError, "blocks diag_reset"):
            validate_submission("diag_reset_object_diff", {})
        validate_submission(
            "move_eef",
            {"gripper": "close", "arm": "left"},
        )
        validate_submission(
            "move_eef",
            {"gripper": "keep", "arm": "right"},
        )
        with self.assertRaisesRegex(StrictToolBoundaryError, "gripper-only"):
            validate_submission(
                "move_eef",
                {"gripper": "close", "forward": 1.0},
            )

    def test_official_v1_registry_has_no_legacy_skill_callable(self) -> None:
        registry = build_registry(SimpleNamespace())

        self.assertEqual(set(registry), set(PUBLIC_SKILLS))
        self.assertTrue(
            all(
                spec.fn.__module__.startswith(
                    "behavior_interface_eval_test.tool.official_v1"
                )
                for spec in registry.values()
            )
        )

    def test_official_v1_source_has_no_simulator_or_legacy_skill_access(self) -> None:
        profile_dir = Path(__file__).parent / "tool" / "official_v1"
        forbidden = (
            "import omnigibson",
            "from omnigibson",
            "behavior_interface.skills",
            ".get_joint_positions(",
            ".set_joint_positions(",
            ".set_position_orientation(",
            ".object_scope",
            "._establish_grasp(",
        )
        for path in profile_dir.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path.name} contains {token}")

    def test_official_v1_camera_and_static_robot_geometry(self) -> None:
        self.assertEqual(relative_uv_to_pixel(0, 0, 720, 720), (0, 0))
        self.assertEqual(relative_uv_to_pixel(1000, 1000, 720, 720), (719, 719))
        point = unproject_pixel(
            px=359,
            py=359,
            depth_m=2.0,
            camera={
                "fx": 306.0,
                "fy": 306.0,
                "cx": 359.0,
                "cy": 359.0,
                "pos": [0.0, 0.0, 1.0],
                "quat": [0.0, 0.0, 0.0, 1.0],
            },
        )
        np.testing.assert_allclose(point, [0.0, 0.0, -1.0])

        shoulders = r1pro_shoulder_positions_robot(np.zeros(4))
        self.assertGreater(shoulders["left"][1], 0.0)
        self.assertLess(shoulders["right"][1], 0.0)
        self.assertAlmostEqual(
            shoulders["left"][1] - shoulders["right"][1],
            0.341470,
            places=6,
        )

    def test_official_v1_reset_and_grasp_prep_emit_only_actions(self) -> None:
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        proprio[PROPRIO_SLICES["trunk_qpos"]] = [0.45, -0.4, 0.0, 0.0]
        adapter = ObservationActionAdapter()
        adapter.update({"robot_r1::proprio": proprio})
        world = self._world(adapter)
        registry = build_registry(adapter)
        result = {}
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda value: result.update(value),
            raise_if_cancelled=lambda where="": None,
        )

        reset_actions = list(registry["reset_body"].fn(ctx))
        self.assertGreater(len(reset_actions), 1)
        for action in reset_actions:
            self.assertEqual(np.asarray(action).shape, (ACTION_DIM,))
        self.assertTrue(result["ok"])
        self.assertTrue(result["converged"])
        np.testing.assert_allclose(
            result["target_trunk_q"],
            [0.45, -0.4, 0.0, 0.0],
        )
        self.assertFalse(result["direct_simulator_mutation"])

        result.clear()
        grasp_actions = []
        grasp_generator = registry["set_arm_to_grasp_position"].fn(
            ctx,
            arm="right",
            gripper="open",
        )
        for action in grasp_generator:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            grasp_actions.append(action.copy())
            observed = adapter.proprio_vector().copy()
            observed[PROPRIO_SLICES["arm_right_qpos"]] = action[
                ACTION_SLICES["arm_right"]
            ]
            observed[PROPRIO_SLICES["arm_right_qvel"]] = 0.0
            adapter.update({"robot_r1::proprio": observed})
        self.assertGreater(len(grasp_actions), 1)
        for action in grasp_actions:
            self.assertEqual(np.asarray(action).shape, (ACTION_DIM,))
        self.assertTrue(result["ok"])
        self.assertFalse(result["direct_simulator_mutation"])

    def test_small_base_motion_scales_the_last_control_step(self) -> None:
        adapter = ObservationActionAdapter()
        adapter.update(
            {"robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32)}
        )
        world = self._world(adapter)
        registry = build_registry(adapter)
        result = {}
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda value: result.update(value),
            raise_if_cancelled=lambda where="": None,
        )

        actions = list(
            registry["move_in_robot_coord"].fn(
                ctx,
                spin=0.14,
                timeout_s=5.0,
            )
        )

        spin_action = next(
            action
            for action in actions
            if abs(float(action[ACTION_SLICES["base"]][2])) > 1e-9
        )
        expected_wz = np.deg2rad(0.14) * 30.0
        self.assertAlmostEqual(
            float(spin_action[ACTION_SLICES["base"]][2]),
            expected_wz,
            places=6,
        )
        self.assertAlmostEqual(
            float(world.robot_pose().yaw),
            np.deg2rad(0.14),
            places=6,
        )
        self.assertLess(result["spin_execution"]["last_step_scale"], 1.0)

    def test_gripper_keep_emits_hold_without_closing(self) -> None:
        adapter = ObservationActionAdapter()
        adapter.update(
            {"robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32)}
        )
        world = self._world(adapter)
        registry = build_registry(adapter)
        result = {}
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda value: result.update(value),
        )

        actions = list(
            registry["move_eef"].fn(
                ctx,
                gripper="keep",
                arm="right",
            )
        )

        self.assertEqual(len(actions), 1)
        np.testing.assert_allclose(
            actions[0][ACTION_SLICES["gripper_right"]],
            [0.0, 0.0]
            if ACTION_SLICES["gripper_right"].stop - ACTION_SLICES["gripper_right"].start == 2
            else [1.0],
        )
        self.assertEqual(
            result["implementation"],
            "official_action_profile_gripper_hold",
        )

    def test_strict_gripper_registry_adapter_emits_official_action(self) -> None:
        adapter = ObservationActionAdapter()
        adapter.update(
            {"robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32)}
        )
        world = self._world(adapter)
        original = SimpleNamespace(
            fn=lambda ctx, **kwargs: iter([ctx.world.hold_action()])
        )
        registry = {
            "move_eef": SimpleNamespace(fn=None),
            "move_in_robot_coord": original,
        }
        install_strict_tool_registry(registry, adapter)
        result = {}
        ctx = SimpleNamespace(
            world=world,
            set_result=lambda value: result.update(value),
        )

        actions = list(
            registry["move_eef"].fn(
                ctx,
                gripper="close",
                arm="left",
            )
        )

        self.assertEqual(len(actions), 10)
        for action in actions:
            self.assertEqual(np.asarray(action).shape, (ACTION_DIM,))
        np.testing.assert_allclose(
            actions[0][ACTION_SLICES["gripper_left"]],
            [-0.1, -0.1]
            if ACTION_SLICES["gripper_left"].stop - ACTION_SLICES["gripper_left"].start == 2
            else [-1.0],
        )
        self.assertTrue(result["ok"])

    def test_strict_http_boundary_rejects_evaluator_owned_controls(self) -> None:
        app = Flask(__name__)

        @app.post("/api/reset", endpoint="api_reset")
        def reset():
            return jsonify({"ok": True})

        @app.post("/api/task/switch", endpoint="api_task_switch")
        def task_switch():
            return jsonify({"ok": True})

        @app.post("/api/skills/reload", endpoint="api_skills_reload")
        def reload_skills():
            return jsonify({"ok": True})

        @app.route("/api/camera", methods=["GET", "POST"], endpoint="api_camera")
        def camera():
            return jsonify({"ok": True, "method": "legacy"})

        install_strict_http_control_boundary(app)
        client = app.test_client()

        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "BEHAVIOR_EVAL_OPERATOR_DIR": tmp,
                "BEHAVIOR_EVAL_TEST_PORT": "15999",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                reset = client.post("/api/reset")
                self.assertEqual(reset.status_code, 409)
                self.assertEqual(reset.json["code"], "evaluator_operator_unavailable")
                for path in ("/api/task/switch", "/api/skills/reload"):
                    response = client.post(path)
                    self.assertEqual(response.status_code, 409)
                    self.assertEqual(response.json["code"], "evaluator_control_required")
                self.assertEqual(client.get("/api/camera").status_code, 200)
                self.assertEqual(client.post("/api/camera").status_code, 409)

    def test_strict_http_reset_forwards_to_evaluator_operator_channel(self) -> None:
        app = Flask(__name__)

        @app.post("/api/reset", endpoint="api_reset")
        def reset():
            return jsonify({"ok": True})

        install_strict_http_control_boundary(app)
        client = app.test_client()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "BEHAVIOR_EVAL_TEST_PORT": "15061",
                    "BEHAVIOR_EVAL_OPERATOR_DIR": tmp,
                },
                clear=False,
            ):
                from behavior_interface_eval_test.operator_scene_control import (
                    write_status,
                    LISTENER_NAME,
                    read_request,
                )

                write_status(15061, listener=LISTENER_NAME, state="idle")
                response = client.post("/api/reset", json={"instance_id": 304})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json["ok"])
                self.assertEqual(response.json["mode"], "evaluator_operator")
                self.assertEqual(response.json["instance_id"], 304)
                request = read_request(15061)
                self.assertEqual(request["op"], "reset")
                self.assertEqual(request["instance_id"], 304)

    def test_strict_http_boundary_rejects_tools_while_evaluator_offline(self) -> None:
        app = Flask(__name__)

        @app.post("/api/skill")
        def skill():
            return jsonify({"ok": True})

        @app.post("/api/v2/capture_head_camera")
        def capture():
            return jsonify({"ok": True})

        install_strict_http_control_boundary(
            app,
            evaluator_connected=lambda: False,
        )
        client = app.test_client()

        for path in ("/api/skill", "/api/v2/capture_head_camera"):
            response = client.post(path, json={})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json["code"], "evaluator_not_connected")


if __name__ == "__main__":
    unittest.main()
