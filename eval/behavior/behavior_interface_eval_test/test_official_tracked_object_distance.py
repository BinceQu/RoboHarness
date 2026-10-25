from __future__ import annotations

import ast
import json
import os
import tempfile
import threading
import unittest
import zlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
from flask import Flask, jsonify

from behavior_interface_eval_test.official_policy_interface import (
    ACTION_DIM,
    ObservationActionAdapter,
    ObservationSnapshot,
    OfficialPolicyRuntime,
    install_official_track_object_distance_routes,
)
from behavior_interface_eval_test.robot_contract import (
    PROPRIO_DIM,
    PROPRIO_SLICES,
)
from behavior_interface_eval_test.tool.official_v2.capabilities import (
    OfficialToolBoundaryError,
    validate_submission,
)
from behavior_interface_eval_test.tool.official_v2.tools import (
    capture_head_camera,
    track_object_distance,
)
from behavior_interface_eval_test.tool.official_v2.human_track_object_distance_ui import (
    install_track_object_distance_human_ui,
)
from behavior_interface_eval_test.tool.official_v2.dynamic_point_tracker import (
    DynamicPointTracker,
    DynamicPointTrackerConfig,
)
from behavior_interface_eval_test.tool.official_v2.eef_adjustment_local import (
    local_robot_state,
)
from behavior_interface_eval_test.tool.official_v2.grasp_geometry_local import (
    quat_to_mat_xyzw,
)
from behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local import (
    eef_pose,
)
from behavior_interface_eval_test.tool.official_v2.registry import build_registry
from behavior_interface_eval_test.tool.official_v2.tracked_object_distance import (
    TRACK_OBJECT_DISTANCE_MAX_REPLAY_WORK,
    TRACK_OBJECT_DISTANCE_REPLAY_MAX_FRAMES,
    TRACK_OBJECT_DISTANCE_REPLAY_TIMEOUT_S,
    TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_LEVEL,
    TrackObjectDistanceBindingError,
    TrackedObjectDistanceMemory,
    _compact_replay_frame,
    _store_replay_frame,
    _restore_replay_frame,
    project_registered_rigid_pair,
    tracker_frame_content_digest,
    tracker_frame_from_allowed_observation,
)


class _Context:
    def __init__(self, manager=None) -> None:
        self.world = SimpleNamespace(
            hold_action=lambda: np.zeros(ACTION_DIM, dtype=np.float32),
            _official_tracked_object_distances=manager,
        )
        self.result = None

    def set_result(self, result) -> None:
        self.result = result


class OfficialTrackedObjectDistanceTest(unittest.TestCase):
    WIDTH = 192
    HEIGHT = 144

    @classmethod
    def setUpClass(cls) -> None:
        cls.can_texture = cls._texture(17, "can")
        cls.bin_texture = cls._texture(29, "bin")

    @staticmethod
    def _texture(seed: int, kind: str) -> np.ndarray:
        rng = np.random.default_rng(seed)
        texture = rng.integers(25, 230, size=(43, 43), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.45)
        if kind == "can":
            cv2.circle(texture, (21, 21), 16, 250, 2)
            cv2.line(texture, (8, 12), (34, 30), 15, 2)
        else:
            cv2.rectangle(texture, (5, 5), (37, 37), 245, 2)
            cv2.line(texture, (5, 21), (37, 21), 10, 2)
        return texture

    @classmethod
    def _observation(
        cls,
        objects,
        *,
        camera_position_robot=(0.0, 0.0, 0.0),
        camera_quaternion_robot=(0.0, 0.0, 0.0, 1.0),
    ) -> dict:
        gray = np.full((cls.HEIGHT, cls.WIDTH), 8, dtype=np.uint8)
        depth = np.full((cls.HEIGHT, cls.WIDTH), 4.0, dtype=np.float32)
        for center, object_depth, texture in objects:
            center_x, center_y = (int(round(value)) for value in center)
            half = texture.shape[0] // 2
            left, top = center_x - half, center_y - half
            right, bottom = left + texture.shape[1], top + texture.shape[0]
            gray[top:bottom, left:right] = texture
            depth[top:bottom, left:right] = np.float32(object_depth)
        poses = np.zeros(21, dtype=np.float64)
        poses[[6, 13, 20]] = 1.0
        poses[14:17] = np.asarray(camera_position_robot, dtype=np.float64)
        poses[17:21] = np.asarray(
            camera_quaternion_robot,
            dtype=np.float64,
        )
        return {
            "robot_r1::zed_link::rgb": cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
            "robot_r1::zed_link::depth_linear": depth,
            "robot_r1::cam_rel_poses": poses,
            "robot_r1::proprio": np.zeros(PROPRIO_DIM, dtype=np.float32),
            "task_id": np.asarray([1], dtype=np.int64),
            "need_new_action": np.asarray([True]),
        }

    @classmethod
    def _frame(
        cls,
        sequence: int,
        objects,
        episode_id: str = "episode-a",
        *,
        base_xy_yaw=(0.0, 0.0, 0.0),
        camera_position_robot=(0.0, 0.0, 0.0),
        camera_quaternion_robot=(0.0, 0.0, 0.0, 1.0),
    ):
        return tracker_frame_from_allowed_observation(
            cls._observation(
                objects,
                camera_position_robot=camera_position_robot,
                camera_quaternion_robot=camera_quaternion_robot,
            ),
            sequence=sequence,
            timestamp_s=float(sequence) / 30.0,
            episode_id=episode_id,
            base_xy_yaw=base_xy_yaw,
        )

    @classmethod
    def _relative(cls, center) -> tuple[float, float]:
        return (
            float(center[0]) / float(cls.WIDTH - 1) * 1000.0,
            float(center[1]) / float(cls.HEIGHT - 1) * 1000.0,
        )

    def _register_capture(
        self,
        manager: TrackedObjectDistanceMemory,
        *,
        session_id: str = "unit-session",
        image_id: str = "img_0001",
    ) -> dict:
        status = manager.status()
        return manager.register_capture(
            session_id=session_id,
            image_id=image_id,
            observation_sequence=status["observation_sequence"],
            episode_id=status["episode_id"],
            image_shape=(self.HEIGHT, self.WIDTH),
        )

    @staticmethod
    def _binding_kwargs(binding: dict) -> dict:
        return {
            "session_id": binding["session_id"],
            "image_id": binding["image_id"],
            "capture_observation_sequence": binding["observation_sequence"],
            "capture_episode_id": binding["episode_id"],
            "capture_image_shape": tuple(binding["image_shape"]),
            "capture_frame_digest": binding["frame_digest"],
        }

    def _replace_points(
        self,
        manager: TrackedObjectDistanceMemory,
        points: list[dict],
        *,
        binding: dict | None = None,
    ) -> dict:
        selected_binding = binding or self._register_capture(manager)
        return manager.replace_points(
            points,
            **self._binding_kwargs(selected_binding),
        )

    def test_runtime_borrowed_frame_uses_one_ingest_ownership_copy(self) -> None:
        observation = self._observation([])
        rgb = observation["robot_r1::zed_link::rgb"]
        depth = observation["robot_r1::zed_link::depth_linear"]
        default_frame = tracker_frame_from_allowed_observation(
            observation,
            sequence=1,
            timestamp_s=1.0 / 30.0,
            episode_id="episode-a",
            base_xy_yaw=(0.0, 0.0, 0.0),
        )
        borrowed_frame = tracker_frame_from_allowed_observation(
            observation,
            sequence=1,
            timestamp_s=1.0 / 30.0,
            episode_id="episode-a",
            base_xy_yaw=(0.0, 0.0, 0.0),
            copy_arrays=False,
        )

        self.assertFalse(np.shares_memory(default_frame.rgb, rgb))
        self.assertFalse(np.shares_memory(default_frame.depth_linear, depth))
        self.assertTrue(np.shares_memory(borrowed_frame.rgb, rgb))
        self.assertTrue(np.shares_memory(borrowed_frame.depth_linear, depth))

        manager = TrackedObjectDistanceMemory()
        manager.ingest(borrowed_frame)
        frozen_rgb = manager._latest_frame.rgb.copy()
        frozen_depth = manager._latest_frame.depth_linear.copy()
        rgb.fill(255)
        depth.fill(9.0)
        np.testing.assert_array_equal(manager._latest_frame.rgb, frozen_rgb)
        np.testing.assert_array_equal(manager._latest_frame.depth_linear, frozen_depth)

    def test_official_borrowed_ingest_keeps_snapshot_arrays_without_copy(self) -> None:
        observation = self._observation([])
        frame = tracker_frame_from_allowed_observation(
            observation,
            sequence=1,
            timestamp_s=1.0 / 30.0,
            episode_id="episode-a",
            base_xy_yaw=(0.0, 0.0, 0.0),
            copy_arrays=False,
        )
        manager = TrackedObjectDistanceMemory()
        manager.ingest(frame, copy_frame=False)
        self.assertIs(manager._latest_frame, frame)
        self.assertTrue(np.shares_memory(manager._latest_frame.rgb, frame.rgb))
        self.assertTrue(
            np.shares_memory(manager._latest_frame.depth_linear, frame.depth_linear)
        )

        # The borrowed path is only valid for snapshot replacement.  The
        # manager must still reject a non-boolean opt-in rather than silently
        # changing ownership semantics.
        with self.assertRaises(TypeError):
            manager.ingest(frame, copy_frame=None)

    def test_replay_copy_elision_preserves_exact_compressed_payload(self) -> None:
        frame = self._frame(
            1,
            [((80, 70), 1.25, self.can_texture)],
        )
        compact = _compact_replay_frame(frame)
        stored = _store_replay_frame(frame)
        expected_gray = np.ascontiguousarray(compact.rgb, dtype=np.uint8)
        expected_depth = np.ascontiguousarray(compact.depth_linear, dtype="<f4")

        self.assertEqual(
            stored.gray_zlib,
            zlib.compress(
                expected_gray.tobytes(order="C"),
                TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_LEVEL,
            ),
        )
        self.assertEqual(
            stored.depth_zlib,
            zlib.compress(
                expected_depth.tobytes(order="C"),
                TRACK_OBJECT_DISTANCE_REPLAY_COMPRESSION_LEVEL,
            ),
        )
        expected_robot_base = stored.camera_to_robot_base.copy()
        expected_policy = stored.camera_to_policy.copy()
        frame.camera_to_robot_base.fill(7.0)
        frame.camera_to_policy.fill(8.0)
        np.testing.assert_array_equal(
            stored.camera_to_robot_base,
            expected_robot_base,
        )
        np.testing.assert_array_equal(stored.camera_to_policy, expected_policy)

    def test_replay_compression_overlaps_tracker_without_async_state_commit(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([96.0, 72.0])
        objects = [(center, 1.25, self.can_texture)]
        manager.ingest(self._frame(1, objects))
        binding = self._register_capture(manager)
        u, v = self._relative(center)
        self._replace_points(
            manager,
            [{"name": "can", "u": u, "v": v}],
            binding=binding,
        )
        compression_started = threading.Event()
        allow_compression = threading.Event()
        original_store = _store_replay_frame
        original_ingest = manager._tracker.ingest

        def delayed_store(frame):
            compression_started.set()
            if not allow_compression.wait(2.0):
                raise AssertionError("tracker did not overlap replay compression")
            return original_store(frame)

        def observed_ingest(frame):
            if not compression_started.wait(2.0):
                raise AssertionError("replay compression did not start")
            allow_compression.set()
            return original_ingest(frame)

        with mock.patch(
            "behavior_interface_eval_test.tool.official_v2."
            "tracked_object_distance._store_replay_frame",
            side_effect=delayed_store,
        ), mock.patch.object(
            manager._tracker,
            "ingest",
            side_effect=observed_ingest,
        ):
            manager.ingest(self._frame(2, objects))

        # Compression deliberately outlives ingest(); wait at the replay
        # barrier before asserting its committed history, not on scheduler luck.
        manager._flush_replay_through(2)
        status = manager.status()
        self.assertEqual(status["observation_sequence"], 2)
        self.assertEqual(status["replay_history_newest_sequence"], 2)

    def test_replay_without_tracked_points_is_async_and_skips_tracker(self) -> None:
        manager = TrackedObjectDistanceMemory()
        manager.ingest(self._frame(1, []))
        self._register_capture(manager)
        submitted_before = manager.status()["replay_async_submitted"]

        with mock.patch.object(manager._tracker, "ingest") as tracker_ingest:
            manager.ingest(self._frame(2, []))

        tracker_ingest.assert_not_called()
        manager._flush_replay_through(2)
        status = manager.status()
        self.assertEqual(status["replay_async_submitted"], submitted_before + 1)
        self.assertEqual(status["replay_pending_frames"], 0)
        self.assertTrue(status["idle_tracker_fast_path"])
        self.assertEqual(status["observation_sequence"], 2)
        self.assertEqual(status["replay_history_newest_sequence"], 2)

    def test_deferred_capture_registration_keeps_exact_replay_barrier(self) -> None:
        manager = TrackedObjectDistanceMemory()
        frame = self._frame(
            1,
            [(np.array([96.0, 72.0]), 1.25, self.can_texture)],
        )
        manager.ingest(frame)
        compression_started = threading.Event()
        allow_compression = threading.Event()
        original_store = _store_replay_frame

        def delayed_store(replay_frame):
            compression_started.set()
            if not allow_compression.wait(2.0):
                raise AssertionError("deferred replay compression did not release")
            return original_store(replay_frame)

        try:
            with mock.patch(
                "behavior_interface_eval_test.tool.official_v2."
                "tracked_object_distance._store_replay_frame",
                side_effect=delayed_store,
            ):
                binding = manager.register_capture_deferred(
                    session_id="unit-session",
                    image_id="img-deferred",
                    observation_sequence=1,
                    episode_id="episode-a",
                    image_shape=(self.HEIGHT, self.WIDTH),
                )
                self.assertTrue(compression_started.wait(2.0))
                self.assertEqual(
                    binding["frame_digest"],
                    tracker_frame_content_digest(frame),
                )
                pending = manager.status()
                self.assertGreaterEqual(pending["replay_pending_frames"], 1)
                self.assertEqual(pending["replay_history_frames"], 0)
        finally:
            allow_compression.set()

        manager._flush_replay_through(1)
        status = manager.status()
        self.assertEqual(status["replay_history_newest_sequence"], 1)
        self.assertEqual(status["replay_pending_frames"], 0)

    def test_deferred_capture_registration_does_not_join_pending_compression(
        self,
    ) -> None:
        """A second live capture must not wait for an older replay future."""

        manager = TrackedObjectDistanceMemory()
        frame_one = self._frame(1, [])
        frame_two = self._frame(2, [])
        manager.ingest(frame_one)
        compression_started = threading.Event()
        release_compression = threading.Event()
        original_store = _store_replay_frame

        def blocked_store(replay_frame):
            compression_started.set()
            if not release_compression.wait(2.0):
                raise AssertionError("pending replay compression did not release")
            return original_store(replay_frame)

        try:
            with mock.patch(
                "behavior_interface_eval_test.tool.official_v2."
                "tracked_object_distance._store_replay_frame",
                side_effect=blocked_store,
            ):
                first = manager.register_capture_deferred(
                    session_id="unit-session",
                    image_id="img-pending-1",
                    observation_sequence=1,
                    episode_id="episode-a",
                    image_shape=(self.HEIGHT, self.WIDTH),
                )
                self.assertTrue(first["ok"])
                self.assertTrue(compression_started.wait(2.0))
                manager.ingest(frame_two)
                with mock.patch.object(
                    manager,
                    "_flush_replay_through",
                    side_effect=AssertionError(
                        "deferred capture joined a pending replay future"
                    ),
                ):
                    second = manager.register_capture_deferred(
                        session_id="unit-session",
                        image_id="img-pending-2",
                        observation_sequence=2,
                        episode_id="episode-a",
                        image_shape=(self.HEIGHT, self.WIDTH),
                    )
                self.assertTrue(second["ok"])
        finally:
            release_compression.set()

        manager._flush_replay_through(2)
        status = manager.status()
        self.assertEqual(status["replay_history_newest_sequence"], 2)
        self.assertEqual(status["replay_pending_frames"], 0)

    def test_contract_accepts_multiple_named_points_and_rejects_bad_input(self) -> None:
        result = validate_submission(
            "track_object_distance",
            {
                "session_id": "unit-session",
                "image_id": "img_0001",
                "points": [
                    {"name": "can", "u": "250", "v": 400},
                    {"name": "bin", "u": 750, "v": 600},
                ]
            },
        )
        self.assertEqual(result["points"][0]["u"], 250.0)
        self.assertEqual(result["points"][1]["name"], "bin")
        for points, message in (
            ([], "at least one"),
            ([{"name": "can", "u": 1, "v": 1}, {"name": "can", "u": 2, "v": 2}], "duplicate"),
            ([{"name": "can", "u": 1001, "v": 2}], "0..1000"),
            ([{"name": "can", "u": 1, "v": 2, "object_id": 7}], "unsupported"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(OfficialToolBoundaryError, message):
                    validate_submission(
                        "track_object_distance",
                        {
                            "session_id": "unit-session",
                            "image_id": "img_0001",
                            "points": points,
                        },
                    )
        overlong = (
            "search-pick-place-15060-prompt-gates-sol-attempt12-continue-20260820"
        )
        with self.assertRaisesRegex(OfficialToolBoundaryError, "too long"):
            validate_submission(
                "track_object_distance",
                {
                    "session_id": overlong,
                    "image_id": "img_0001",
                    "points": [{"name": "can", "u": 1, "v": 2}],
                },
            )
        for incomplete, message in (
            ({"image_id": "img_0001", "points": [{"name": "can", "u": 1, "v": 2}]}, "session_id"),
            ({"session_id": "unit-session", "points": [{"name": "can", "u": 1, "v": 2}]}, "image_id"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(OfficialToolBoundaryError, message):
                    validate_submission("track_object_distance", incomplete)

    def test_registry_requires_session_image_and_points(self) -> None:
        params = build_registry(None)["track_object_distance"].params
        self.assertEqual(
            [item["name"] for item in params],
            ["session_id", "image_id", "points"],
        )
        self.assertTrue(all(item["required"] for item in params))

    def test_multiple_depths_update_temporary_loss_and_offscreen_deletion(self) -> None:
        manager = TrackedObjectDistanceMemory()
        can_1 = np.array([52.0, 60.0])
        bin_1 = np.array([142.0, 82.0])
        manager.ingest(
            self._frame(
                1,
                [
                    (can_1, 1.20, self.can_texture),
                    (bin_1, 2.10, self.bin_texture),
                ],
            )
        )
        can_uv = self._relative(can_1)
        bin_uv = self._relative(bin_1)
        started = self._replace_points(
            manager,
            [
                {"name": "can", "u": can_uv[0], "v": can_uv[1]},
                {"name": "bin", "u": bin_uv[0], "v": bin_uv[1]},
            ]
        )
        self.assertAlmostEqual(started["entries"]["can"]["depth_m"], 1.20, places=5)
        self.assertAlmostEqual(started["entries"]["bin"]["depth_m"], 2.10, places=5)
        self.assertEqual(
            len(started["entries"]["can"]["xyz_in_robot_base_coord_m"]),
            3,
        )

        can_2 = can_1 + [10.0, 4.0]
        bin_2 = bin_1 + [-8.0, -3.0]
        manager.ingest(
            self._frame(
                2,
                [
                    (can_2, 1.05, self.can_texture),
                    (bin_2, 2.35, self.bin_texture),
                ],
            )
        )
        memory = manager.memory_fields()["tracked_object_distances"]
        self.assertAlmostEqual(memory["can"]["depth_m"], 1.05, places=5)
        self.assertAlmostEqual(memory["bin"]["depth_m"], 2.35, places=5)
        self.assertNotEqual(
            memory["can"]["xyz_in_robot_base_coord_m"],
            started["entries"]["can"]["xyz_in_robot_base_coord_m"],
        )
        np.testing.assert_allclose(
            [memory["can"]["u"], memory["can"]["v"]],
            self._relative(can_2),
            atol=2.0,
        )

        manager.ingest(self._frame(3, [(bin_2 + [3.0, 2.0], 2.40, self.bin_texture)]))
        memory = manager.memory_fields()["tracked_object_distances"]
        self.assertIn("can", memory)
        self.assertIn("bin", memory)
        self.assertIsNone(memory["can"]["depth_m"])
        self.assertIsNone(memory["can"]["xyz_in_robot_base_coord_m"])
        self.assertEqual(memory["can"]["status"], "temporarily_unobserved")
        self.assertNotIn("predicted_depth_m", memory["can"])
        self.assertEqual(
            manager.status()["registered_names"],
            ["can", "bin"],
        )
        self.assertEqual(
            list(manager.status()["temporarily_unobserved"]),
            ["can"],
        )

        can_4 = can_2 + [2.0, -1.0]
        bin_4 = bin_2 + [3.0, 2.0]
        manager.ingest(
            self._frame(
                4,
                [
                    (can_4, 0.98, self.can_texture),
                    (bin_4, 2.40, self.bin_texture),
                ],
            )
        )
        memory = manager.memory_fields()["tracked_object_distances"]
        self.assertAlmostEqual(memory["can"]["depth_m"], 0.98, places=5)
        self.assertEqual(manager.status()["temporarily_unobserved"], {})

        offscreen = self._frame(5, [])
        camera_to_policy = np.eye(4, dtype=np.float64)
        camera_to_policy[0, 3] = -10.0
        manager.ingest(replace(offscreen, camera_to_policy=camera_to_policy))
        self.assertEqual(
            manager.memory_fields()["tracked_object_distances"],
            {},
        )
        self.assertEqual(manager.status()["registered"], 0)
        self.assertEqual(
            {item["name"] for item in manager.status()["last_deleted"]},
            {"can", "bin"},
        )
        self.assertTrue(
            all(
                item["reason"] == "out_of_view"
                for item in manager.status()["last_deleted"]
            )
        )

    def test_motion_retention_keeps_template_private_and_reacquires_after_lost(
        self,
    ) -> None:
        manager = TrackedObjectDistanceMemory(
            tracker_factory=lambda: DynamicPointTracker(
                config=DynamicPointTrackerConfig(max_occluded_steps=1)
            )
        )
        center = np.array([88.0, 68.0])
        manager.ingest(self._frame(1, [(center, 1.25, self.can_texture)]))
        uv = self._relative(center)
        self._replace_points(
            manager,
            [{"name": "can", "u": uv[0], "v": uv[1]}],
        )
        lease = manager.begin_motion_retention(
            ["can"],
            episode_id="episode-a",
            timeout_s=10.0,
        )

        manager.ingest(self._frame(2, []))
        manager.ingest(self._frame(3, []))
        status = manager.status()
        memory = manager.memory_fields()["tracked_object_distances"]
        self.assertEqual(status["registered_names"], ["can"])
        self.assertEqual(status["motion_retention"]["retained_names"], ["can"])
        self.assertEqual(memory["can"]["status"], "temporarily_unobserved")
        self.assertIsNone(memory["can"]["depth_m"])
        self.assertIsNone(memory["can"]["xyz_in_robot_base_coord_m"])
        self.assertFalse(
            status["motion_retention"][
                "coordinates_published_while_unobserved"
            ]
        )

        moved = center + [5.0, -2.0]
        manager.ingest(self._frame(4, [(moved, 1.10, self.can_texture)]))
        recovered = manager.observed_active_points_snapshot(
            ["can"],
            episode_id="episode-a",
        )
        self.assertTrue(recovered["ok"], recovered)
        self.assertEqual(recovered["entries"]["can"]["name"], "can")
        self.assertAlmostEqual(
            recovered["entries"]["can"]["depth_m"], 1.10, places=5
        )
        self.assertEqual(
            recovered["entries"]["can"]["observation_sequence"], 4
        )

        released = manager.end_motion_retention(lease["lease_id"])
        self.assertTrue(released["released"])
        self.assertEqual(manager.status()["motion_retention"]["active_leases"], 0)
        manager.ingest(self._frame(5, []))
        manager.ingest(self._frame(6, []))
        self.assertEqual(manager.status()["registered"], 0)
        self.assertNotIn(
            "can", manager.memory_fields()["tracked_object_distances"]
        )

    def test_motion_retention_rejects_wrong_episode_names_and_duration(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([88.0, 68.0])
        manager.ingest(self._frame(1, [(center, 1.25, self.can_texture)]))
        uv = self._relative(center)
        self._replace_points(
            manager,
            [{"name": "can", "u": uv[0], "v": uv[1]}],
        )
        for names, episode_id, timeout_s, message in (
            (["missing"], "episode-a", 1.0, "not registered"),
            (["can"], "episode-b", 1.0, "episode changed"),
            (["can"], "episode-a", 0.0, "timeout_s"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    manager.begin_motion_retention(
                        names,
                        episode_id=episode_id,
                        timeout_s=timeout_s,
                    )

    def test_same_name_motion_retention_atomically_supersedes_failed_attempt(self):
        manager = TrackedObjectDistanceMemory()
        center = np.array([88.0, 68.0])
        manager.ingest(self._frame(1, [(center, 1.25, self.can_texture)]))
        uv = self._relative(center)
        self._replace_points(
            manager,
            [{"name": "can", "u": uv[0], "v": uv[1]}],
        )

        first = manager.begin_motion_retention(
            ["can"], episode_id="episode-a", timeout_s=30.0
        )
        second = manager.begin_motion_retention(
            ["can"], episode_id="episode-a", timeout_s=30.0
        )

        self.assertEqual(second["superseded_lease_ids"], [first["lease_id"]])
        status = manager.status()
        self.assertEqual(status["motion_retention"]["active_leases"], 1)
        self.assertEqual(status["motion_retention"]["retained_names"], ["can"])
        self.assertFalse(manager.end_motion_retention(first["lease_id"])["released"])
        self.assertTrue(manager.end_motion_retention(second["lease_id"])["released"])
        self.assertEqual(
            manager.status()["motion_retention"]["active_leases"], 0
        )

    def test_motion_retention_preserves_registration_when_tracker_omits_snapshot(
        self,
    ) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([88.0, 68.0])
        manager.ingest(self._frame(1, [(center, 1.25, self.can_texture)]))
        uv = self._relative(center)
        self._replace_points(
            manager,
            [{"name": "can", "u": uv[0], "v": uv[1]}],
        )
        lease = manager.begin_motion_retention(
            ["can"], episode_id="episode-a", timeout_s=10.0
        )

        with mock.patch.object(manager._tracker, "ingest", return_value=()):
            manager.ingest(self._frame(2, []))
            status = manager.status()
            memory = manager.memory_fields()["tracked_object_distances"]
            self.assertEqual(status["registered_names"], ["can"])
            self.assertEqual(memory["can"]["status"], "temporarily_unobserved")
            self.assertIsNone(memory["can"]["depth_m"])
            self.assertIsNone(memory["can"]["xyz_in_robot_base_coord_m"])
            self.assertNotIn("can", {
                item["name"] for item in status["last_deleted"]
            })

            manager.end_motion_retention(lease["lease_id"])
            manager.ingest(self._frame(3, []))

        self.assertNotIn("can", manager.status()["registered_names"])
        self.assertEqual(manager.status()["last_deleted"][-1]["name"], "can")

    def test_registered_rigid_pair_projection_is_minimum_symmetric_correction(
        self,
    ) -> None:
        measured = np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]])
        report = project_registered_rigid_pair(
            measured,
            reference_distance_m=0.10,
        )
        self.assertTrue(report["ok"])
        fused = np.asarray(report["fused_points_robot_base_m"])
        np.testing.assert_allclose(np.mean(fused, axis=0), np.mean(measured, axis=0))
        np.testing.assert_allclose(fused, [[-0.01, 0.0, 0.0], [0.09, 0.0, 0.0]])
        self.assertAlmostEqual(np.linalg.norm(fused[1] - fused[0]), 0.10)
        self.assertAlmostEqual(report["max_point_correction_m"], 0.01)

    def test_registered_rigid_pair_projection_rejects_large_or_collapsed_input(
        self,
    ) -> None:
        excessive = project_registered_rigid_pair(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
            reference_distance_m=0.10,
            max_point_correction_m=0.02,
        )
        self.assertFalse(excessive["ok"])
        self.assertEqual(
            excessive["reason"], "rigid_pair_rgbd_correction_exceeded"
        )
        collapsed = project_registered_rigid_pair(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            reference_distance_m=0.10,
        )
        self.assertFalse(collapsed["ok"])
        self.assertEqual(collapsed["reason"], "measured_pair_collapsed")

    def test_rigid_pair_deactivation_preserves_registered_point_set(self) -> None:
        manager = TrackedObjectDistanceMemory()
        manager._name_by_track_id = {
            "distance_point_001": "1",
            "distance_point_002": "2",
            "distance_point_003": "scene",
        }
        manager._registration_entries = {
            "1": {"xyz_in_robot_base_coord_m": [0.0, 0.0, 0.0]},
            "2": {"xyz_in_robot_base_coord_m": [0.1, 0.0, 0.0]},
            "scene": {"xyz_in_robot_base_coord_m": [0.8, 0.2, 0.1]},
        }
        manager._entries = {
            "1": {"status": "observed"},
            "2": {"status": "observed"},
            "scene": {
                "status": "observed",
                "xyz_in_robot_base_coord_m": [0.8, 0.2, 0.1],
            },
        }
        manager._active_rigid_pair = {
            "names": ["1", "2"],
            "reference_distance_m": 0.1,
        }
        manager._active_on_hand_eef_prior = {
            "names": ["1", "2"],
            "lease_id": "pair-prior",
        }

        with mock.patch.object(
            manager._tracker,
            "deactivate_rigid_pair",
        ) as deactivate:
            report = manager.deactivate_rigid_pair(reason="unit_test_scope_end")

        self.assertTrue(report["ok"], report)
        self.assertIsNone(manager.status()["active_rigid_pair"])
        self.assertIsNone(manager.status()["active_on_hand_eef_observation_prior"])
        self.assertEqual(
            manager.status()["registered_names"], ["1", "2", "scene"]
        )
        self.assertEqual(set(manager._registration_entries), {"1", "2", "scene"})
        self.assertEqual(
            manager._entries["scene"]["xyz_in_robot_base_coord_m"],
            [0.8, 0.2, 0.1],
        )
        deactivate.assert_called_once_with()

    def test_hot_upgrade_replaces_stale_official_tracker_factory(self) -> None:
        manager = TrackedObjectDistanceMemory()
        stale_factory = type("DynamicPointTracker", (), {})
        stale_factory.__module__ = DynamicPointTracker.__module__
        manager._tracker_factory = stale_factory

        result = manager.upgrade_runtime_state()

        self.assertTrue(result["tracker_factory_upgraded"])
        self.assertIs(manager._tracker_factory, DynamicPointTracker)
        self.assertFalse(
            manager.upgrade_runtime_state()["tracker_factory_upgraded"]
        )

    def test_hot_upgrade_retires_pair_when_reactivation_fails(self) -> None:
        manager = TrackedObjectDistanceMemory()
        manager._name_by_track_id = {
            "distance_point_001": "1",
            "distance_point_002": "2",
        }
        manager._active_rigid_pair = {
            "names": ["1", "2"],
            "reference_distance_m": 0.1,
        }
        manager._active_on_hand_eef_prior = {
            "names": ["1", "2"],
            "lease_id": "stale-prior",
        }
        with mock.patch.object(
            manager._tracker,
            "list_points",
            return_value=(
                SimpleNamespace(track_id="distance_point_001"),
                SimpleNamespace(track_id="distance_point_002"),
            ),
        ), mock.patch.object(
            manager,
            "_activate_tracker_rigid_pair_locked",
            side_effect=ValueError("rigid-pair tracker ids are unavailable for 2"),
        ):
            report = manager.upgrade_runtime_state()

        self.assertIsNone(manager.status()["active_rigid_pair"])
        self.assertIsNone(
            manager.status()["active_on_hand_eef_observation_prior"]
        )
        self.assertEqual(
            report["tracker_rigid_pair"]["reason"],
            "rigid_pair_track_removed",
        )
        self.assertIn(
            "unavailable for 2",
            report["tracker_rigid_pair"]["activation_error"],
        )

    def test_active_rigid_pair_uses_registration_distance_across_updates(self) -> None:
        manager = TrackedObjectDistanceMemory()
        first_a = np.array([58.0, 62.0])
        first_b = np.array([124.0, 76.0])
        manager.ingest(
            self._frame(
                1,
                [
                    (first_a, 1.10, self.can_texture),
                    (first_b, 1.12, self.bin_texture),
                ],
            )
        )
        started = self._replace_points(
            manager,
            [
                {"name": "a", "u": self._relative(first_a)[0], "v": self._relative(first_a)[1]},
                {"name": "b", "u": self._relative(first_b)[0], "v": self._relative(first_b)[1]},
            ],
        )
        initial = np.asarray(
            [
                started["registration_entries"][name][
                    "xyz_in_robot_base_coord_m"
                ]
                for name in ("a", "b")
            ]
        )
        reference_distance = float(np.linalg.norm(initial[1] - initial[0]))
        activation = manager.activate_rigid_pair(
            ["a", "b"],
            episode_id="episode-a",
            max_point_correction_m=0.10,
        )
        self.assertTrue(activation["ok"])

        manager.ingest(
            self._frame(
                2,
                [
                    (first_a + [3.0, 1.0], 1.11, self.can_texture),
                    (first_b + [3.0, 1.0], 1.13, self.bin_texture),
                ],
            )
        )
        memory = manager.memory_fields()["tracked_object_distances"]
        fused = np.asarray(
            [memory["a"]["xyz_in_robot_base_coord_m"], memory["b"]["xyz_in_robot_base_coord_m"]]
        )
        raw = np.asarray(
            [memory["a"]["raw_xyz_in_robot_base_coord_m"], memory["b"]["raw_xyz_in_robot_base_coord_m"]]
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(fused[1] - fused[0])),
            reference_distance,
            places=10,
        )
        self.assertGreater(
            abs(float(np.linalg.norm(raw[1] - raw[0])) - reference_distance),
            1e-4,
        )
        snapshot = manager.observed_active_points_snapshot(
            ["a", "b"], episode_id="episode-a"
        )
        self.assertTrue(snapshot["ok"])
        self.assertTrue(snapshot["rigid_pair_fusion"]["ok"])
        self.assertTrue(
            snapshot["rigid_pair_fusion"]["visual_tracker_pair"]["ok"]
        )
        self.assertIn("registration_rigid_pair_projection", snapshot["measurement_source"])

        fused_before_retirement = fused.copy()
        raw_before_retirement = raw.copy()
        retirement = manager.deactivate_rigid_pair(
            reason="unit_test_request_scope_ended"
        )

        self.assertTrue(retirement["ok"], retirement)
        self.assertEqual(
            retirement["entry_retirement"]["restored_raw_names"],
            ["a", "b"],
        )
        self.assertEqual(
            retirement["entry_retirement"]["fail_closed_names"], []
        )
        self.assertTrue(
            retirement["retired_report"]["raw_rgbd_measurement_published"]
        )
        retired_memory = manager.memory_fields()["tracked_object_distances"]
        retired_xyz = np.asarray(
            [
                retired_memory["a"]["xyz_in_robot_base_coord_m"],
                retired_memory["b"]["xyz_in_robot_base_coord_m"],
            ]
        )
        np.testing.assert_allclose(retired_xyz, raw_before_retirement)
        self.assertGreater(
            float(np.max(np.abs(retired_xyz - fused_before_retirement))),
            1e-6,
        )
        for name in ("a", "b"):
            self.assertNotIn(
                "raw_xyz_in_robot_base_coord_m", retired_memory[name]
            )
            self.assertNotIn("raw_xyz_source", retired_memory[name])
            self.assertNotIn("rigid_pair_fusion", retired_memory[name])
            self.assertEqual(
                retired_memory[name]["xyz_source"],
                "current_evaluator_depth_linear_and_cam_rel_poses",
            )

        post_retirement_snapshot = manager.observed_active_points_snapshot(
            ["a", "b"], episode_id="episode-a"
        )
        self.assertTrue(post_retirement_snapshot["ok"], post_retirement_snapshot)
        self.assertEqual(post_retirement_snapshot["rigid_pair_fusion"], {})
        self.assertNotIn(
            "registration_rigid_pair_projection",
            post_retirement_snapshot["measurement_source"],
        )
        np.testing.assert_allclose(
            np.asarray(
                [
                    post_retirement_snapshot["entries"][name][
                        "xyz_in_robot_base_coord_m"
                    ]
                    for name in ("a", "b")
                ]
            ),
            raw_before_retirement,
        )

    def test_stale_rigid_pair_cannot_block_complete_reselection(self) -> None:
        manager = TrackedObjectDistanceMemory()
        centers = {
            "1": np.array([48.0, 56.0]),
            "2": np.array([104.0, 68.0]),
            "3": np.array([154.0, 88.0]),
        }
        objects = [
            (centers["1"], 1.10, self.can_texture),
            (centers["2"], 1.18, self.bin_texture),
            (centers["3"], 1.72, self.can_texture),
        ]
        manager.ingest(self._frame(1, objects))
        old_binding = self._register_capture(manager, image_id="img-old")
        old = manager.replace_points(
            [
                {
                    "name": name,
                    "u": self._relative(center)[0],
                    "v": self._relative(center)[1],
                }
                for name, center in centers.items()
            ],
            **self._binding_kwargs(old_binding),
        )
        manager.activate_rigid_pair(["1", "2"], episode_id="episode-a")

        # Reproduce a hot live manager where the underlying tracker correctly
        # retired point 2 and its pair, but the wrapper's name-based pair and
        # on-hand lease survived while points 1 and 3 remained registered.
        removed_track_id = old["entries"]["2"]["track_id"]
        with manager._lock:
            manager._tracker.remove_point(removed_track_id)
            manager._name_by_track_id.pop(removed_track_id)
            manager._entries.pop("2")
            manager._active_on_hand_eef_prior = {
                "names": ["1", "2"],
                "lease_id": "stale-on-hand-prior",
            }
        self.assertEqual(manager.status()["registered_names"], ["1", "3"])
        self.assertEqual(manager.status()["active_rigid_pair"]["names"], ["1", "2"])

        latest_binding = self._register_capture(manager, image_id="img-latest")
        replacement_names = ["2", "latest-target"]
        replaced = manager.replace_points(
            [
                {
                    "name": replacement_names[0],
                    "u": self._relative(centers["2"])[0],
                    "v": self._relative(centers["2"])[1],
                },
                {
                    "name": replacement_names[1],
                    "u": self._relative(centers["3"])[0],
                    "v": self._relative(centers["3"])[1],
                },
            ],
            **self._binding_kwargs(latest_binding),
        )

        self.assertEqual(list(replaced["entries"]), replacement_names)
        self.assertEqual(manager.status()["registered_names"], replacement_names)
        self.assertEqual(
            list(manager.memory_fields()["tracked_object_distances"]),
            replacement_names,
        )
        self.assertNotIn("1", manager.memory_fields()["tracked_object_distances"])
        self.assertIsNone(manager.status()["active_rigid_pair"])
        self.assertIsNone(
            manager.status()["active_on_hand_eef_observation_prior"]
        )

    def test_ingest_removal_retires_pair_when_another_name_survives(self) -> None:
        manager = TrackedObjectDistanceMemory()
        first_center = np.array([50.0, 56.0])
        second_center = np.array([112.0, 70.0])
        manager.ingest(
            self._frame(
                1,
                [
                    (first_center, 1.10, self.can_texture),
                    (second_center, 1.20, self.bin_texture),
                ],
            )
        )
        binding = self._register_capture(manager)
        started = manager.replace_points(
            [
                {
                    "name": "survivor",
                    "u": self._relative(first_center)[0],
                    "v": self._relative(first_center)[1],
                },
                {
                    "name": "removed",
                    "u": self._relative(second_center)[0],
                    "v": self._relative(second_center)[1],
                },
            ],
            **self._binding_kwargs(binding),
        )
        manager.activate_rigid_pair(["survivor", "removed"], episode_id="episode-a")

        survivor_id = started["entries"]["survivor"]["track_id"]
        removed_id = started["entries"]["removed"]["track_id"]
        survivor_snapshot = manager._tracker.get_point(survivor_id)
        removed_snapshot = manager._tracker.get_point(removed_id)
        removed_snapshot = replace(
            removed_snapshot,
            status="lost",
            uv=None,
            pixel_uv=None,
            depth_m=None,
            point_camera_m=None,
            point_robot_base_m=None,
            point_policy_local_m=None,
            predicted_uv=None,
            predicted_depth_m=None,
        )
        with mock.patch.object(
            manager._tracker,
            "ingest",
            return_value=(survivor_snapshot, removed_snapshot),
        ):
            manager.ingest(
                self._frame(
                    2,
                    [(first_center, 1.10, self.can_texture)],
                )
            )

        status = manager.status()
        self.assertEqual(status["registered_names"], ["survivor"])
        self.assertEqual(status["registration_reference_names"], ["survivor"])
        self.assertIsNone(status["active_rigid_pair"])
        self.assertEqual(
            status["active_rigid_pair_report"]["reason"],
            "rigid_pair_track_removed",
        )
        self.assertEqual(
            status["active_rigid_pair_report"]["unavailable"],
            ["removed"],
        )

    def test_successful_track_call_replaces_the_entire_previous_name_set(self) -> None:
        manager = TrackedObjectDistanceMemory()
        first_center = np.array([54.0, 58.0])
        second_center = np.array([136.0, 86.0])
        objects = [
            (first_center, 1.15, self.can_texture),
            (second_center, 1.85, self.bin_texture),
        ]
        manager.ingest(self._frame(1, objects))
        first_binding = self._register_capture(manager, image_id="img-first")
        manager.replace_points(
            [
                {
                    "name": "old-a",
                    "u": self._relative(first_center)[0],
                    "v": self._relative(first_center)[1],
                },
                {
                    "name": "old-b",
                    "u": self._relative(second_center)[0],
                    "v": self._relative(second_center)[1],
                },
            ],
            **self._binding_kwargs(first_binding),
        )
        latest_binding = self._register_capture(manager, image_id="img-latest")

        result = manager.replace_points(
            [
                {
                    "name": "latest-only",
                    "u": self._relative(second_center)[0],
                    "v": self._relative(second_center)[1],
                }
            ],
            **self._binding_kwargs(latest_binding),
        )

        self.assertEqual(result["replaced_names"], ["old-a", "old-b"])
        self.assertEqual(list(result["entries"]), ["latest-only"])
        self.assertEqual(manager.status()["registered_names"], ["latest-only"])
        self.assertEqual(
            list(manager.memory_fields()["tracked_object_distances"]),
            ["latest-only"],
        )

    def test_reselection_clear_keeps_capture_history_but_drops_live_points(
        self,
    ) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([62.0, 64.0])
        manager.ingest(self._frame(1, [(center, 1.20, self.can_texture)]))
        binding = self._register_capture(manager, image_id="img-reselect")
        self._replace_points(
            manager,
            [
                {
                    "name": "old",
                    "u": self._relative(center)[0],
                    "v": self._relative(center)[1],
                }
            ],
            binding=binding,
        )

        cleared = manager.clear_for_reselection()

        self.assertEqual(cleared["cleared_names"], ["old"])
        self.assertEqual(manager.status()["registered_names"], [])
        self.assertEqual(manager.status()["capture_bindings"], 1)
        self.assertEqual(manager.status()["observation_sequence"], 1)
        self.assertEqual(manager.memory_fields()["tracked_object_distances"], {})

    def test_visual_candidate_rejection_cannot_be_hidden_by_pair_projection(
        self,
    ) -> None:
        manager = TrackedObjectDistanceMemory()
        first_a = np.array([58.0, 62.0])
        first_b = np.array([124.0, 76.0])
        manager.ingest(
            self._frame(
                1,
                [
                    (first_a, 1.10, self.can_texture),
                    (first_b, 1.12, self.bin_texture),
                ],
            )
        )
        started = self._replace_points(
            manager,
            [
                {"name": "a", "u": self._relative(first_a)[0], "v": self._relative(first_a)[1]},
                {"name": "b", "u": self._relative(first_b)[0], "v": self._relative(first_b)[1]},
            ],
        )
        manager.activate_rigid_pair(["a", "b"], episode_id="episode-a")
        reference = float(manager._active_rigid_pair["reference_distance_m"])
        raw = np.asarray(
            [
                started["entries"]["a"]["xyz_in_robot_base_coord_m"],
                started["entries"]["b"]["xyz_in_robot_base_coord_m"],
            ],
            dtype=np.float64,
        )
        direction = raw[1] - raw[0]
        direction /= np.linalg.norm(direction)
        midpoint = np.mean(raw, axis=0)
        distorted_distance = reference + 0.0113
        distorted = np.asarray(
            [
                midpoint - 0.5 * distorted_distance * direction,
                midpoint + 0.5 * distorted_distance * direction,
            ]
        )
        for index, name in enumerate(("a", "b")):
            manager._entries[name]["raw_xyz_in_robot_base_coord_m"] = (
                distorted[index].astype(float).tolist()
            )
            manager._entries[name]["xyz_in_robot_base_coord_m"] = (
                distorted[index].astype(float).tolist()
            )

        visual_rejection = {
            "ok": False,
            "reason": "individual_candidate_gate_rejected",
            "per_point": {
                "distance_point_002": {
                    "ok": False,
                    "reason": "eef_prior_depth_layer_mismatch",
                    "pixel_error_px": 42.0,
                    "depth_layer_error_m": 0.039,
                }
            },
            "candidate_points_published": [],
            "prediction_values_published_as_observation": False,
        }
        with mock.patch.object(
            manager._tracker,
            "rigid_pair_report",
            return_value=visual_rejection,
        ):
            with manager._lock:
                report = manager._apply_active_rigid_pair_locked()

        self.assertFalse(report["ok"])
        self.assertFalse(report["fusion_applied"])
        self.assertEqual(
            report["reason"], "individual_candidate_gate_rejected"
        )
        self.assertFalse(report["raw_rgbd_measurement_published"])
        self.assertFalse(report["stale_depth_published"])
        self.assertFalse(report["stale_xyz_in_robot_base_coord_published"])
        for name in ("a", "b"):
            self.assertEqual(
                manager._entries[name]["status"], "temporarily_unobserved"
            )
            self.assertIsNone(manager._entries[name]["depth_m"])
            self.assertIsNone(
                manager._entries[name]["xyz_in_robot_base_coord_m"]
            )

        retirement = manager.deactivate_rigid_pair(
            reason="unit_test_rejected_pair_scope_ended"
        )
        self.assertEqual(
            retirement["entry_retirement"]["fail_closed_names"],
            ["a", "b"],
        )
        self.assertFalse(
            retirement["retired_report"]["raw_rgbd_measurement_published"]
        )
        self.assertEqual(manager.status()["registered_names"], ["a", "b"])
        retired_memory = manager.memory_fields()["tracked_object_distances"]
        for name in ("a", "b"):
            self.assertEqual(
                retired_memory[name]["status"], "temporarily_unobserved"
            )
            self.assertIsNone(retired_memory[name]["depth_m"])
            self.assertIsNone(
                retired_memory[name]["xyz_in_robot_base_coord_m"]
            )
            self.assertNotIn("raw_xyz_in_robot_base_coord_m", retired_memory[name])
            self.assertNotIn("rigid_pair_fusion", retired_memory[name])
        post_retirement_snapshot = manager.observed_active_points_snapshot(
            ["a", "b"], episode_id="episode-a"
        )
        self.assertFalse(post_retirement_snapshot["ok"])
        self.assertEqual(
            post_retirement_snapshot["reason"], "tracked_point_unavailable"
        )
        self.assertEqual(post_retirement_snapshot["entries"], {})

    def test_registered_points_snapshot_is_frozen_and_independent_of_live_entries(
        self,
    ) -> None:
        manager = TrackedObjectDistanceMemory()
        first_a = np.array([58.0, 62.0])
        first_b = np.array([124.0, 76.0])
        manager.ingest(
            self._frame(
                1,
                [
                    (first_a, 1.10, self.can_texture),
                    (first_b, 1.12, self.bin_texture),
                ],
            )
        )
        binding = self._register_capture(manager, image_id="img-registration")
        started = self._replace_points(
            manager,
            [
                {
                    "name": "a",
                    "u": self._relative(first_a)[0],
                    "v": self._relative(first_a)[1],
                },
                {
                    "name": "b",
                    "u": self._relative(first_b)[0],
                    "v": self._relative(first_b)[1],
                },
            ],
            binding=binding,
        )
        registered_before = {
            name: list(
                started["registration_entries"][name][
                    "xyz_in_robot_base_coord_m"
                ]
            )
            for name in ("a", "b")
        }
        manager._entries["a"]["xyz_in_robot_base_coord_m"] = [9.0, 9.0, 9.0]
        manager._entries["b"]["xyz_in_robot_base_coord_m"] = [8.0, 8.0, 8.0]

        snapshot = manager.registered_points_snapshot(
            ["a", "b"], episode_id="episode-a"
        )

        self.assertTrue(snapshot["ok"], snapshot)
        self.assertEqual(snapshot["session_id"], binding["session_id"])
        self.assertEqual(snapshot["image_id"], binding["image_id"])
        self.assertEqual(snapshot["registration_observation_sequence"], 1)
        self.assertFalse(snapshot["predictions_published"])
        for name in ("a", "b"):
            self.assertEqual(
                snapshot["entries"][name]["xyz_in_robot_base_coord_m"],
                registered_before[name],
            )
        snapshot["entries"]["a"]["xyz_in_robot_base_coord_m"][0] = -123.0
        self.assertEqual(
            manager._registration_entries["a"]["xyz_in_robot_base_coord_m"],
            registered_before["a"],
        )
        wrong_episode = manager.registered_points_snapshot(
            ["a", "b"], episode_id="episode-b"
        )
        self.assertFalse(wrong_episode["ok"])
        self.assertEqual(wrong_episode["reason"], "episode_changed")

    def test_on_hand_eef_prior_is_pair_scoped_and_uses_local_fk(self) -> None:
        manager = TrackedObjectDistanceMemory()
        centers = {
            "hand-a": np.array([54.0, 58.0]),
            "hand-b": np.array([112.0, 70.0]),
            "off-hand": np.array([157.0, 92.0]),
        }
        objects = [
            (centers["hand-a"], 1.10, self.can_texture),
            (centers["hand-b"], 1.13, self.bin_texture),
            (centers["off-hand"], 1.70, self.can_texture),
        ]
        manager.ingest(self._frame(1, objects))
        binding = self._register_capture(manager)
        started = manager.replace_points(
            [
                {
                    "name": name,
                    "u": self._relative(center)[0],
                    "v": self._relative(center)[1],
                }
                for name, center in centers.items()
            ],
            **self._binding_kwargs(binding),
        )
        manager.activate_rigid_pair(
            ["hand-a", "hand-b"], episode_id="episode-a"
        )

        proprio = np.asarray(manager._latest_frame.proprio, dtype=np.float64)
        state = local_robot_state(
            trunk_q=proprio[PROPRIO_SLICES["trunk_qpos"]],
            arm_left_q=proprio[PROPRIO_SLICES["arm_left_qpos"]],
            arm_right_q=proprio[PROPRIO_SLICES["arm_right_qpos"]],
            gripper_left_q=proprio[PROPRIO_SLICES["gripper_left_qpos"]],
            gripper_right_q=proprio[PROPRIO_SLICES["gripper_right_qpos"]],
        )
        q_left = proprio[PROPRIO_SLICES["arm_left_qpos"]]
        eef_position, eef_quaternion = eef_pose(state, "left", q_left)
        rotation = quat_to_mat_xyzw(eef_quaternion)
        expected = np.asarray(
            [
                started["entries"][name]["xyz_in_robot_base_coord_m"]
                for name in ("hand-a", "hand-b")
            ]
        )
        anchors = (rotation.T @ (expected - eef_position).T).T
        activation = manager.activate_on_hand_eef_observation_prior(
            ["hand-a", "hand-b"],
            episode_id="episode-a",
            session_id=binding["session_id"],
            image_id=binding["image_id"],
            arm="left",
            anchors_eef_m=anchors,
        )
        self.assertTrue(activation["ok"])
        self.assertAlmostEqual(activation["max_point_error_m"], 0.030)
        self.assertAlmostEqual(activation["max_depth_error_m"], 0.025)
        self.assertAlmostEqual(activation["max_pixel_error_px"], 24.0)
        with manager._lock:
            prior = manager._on_hand_eef_prior_for_frame_locked(
                manager._latest_frame
            )
        self.assertEqual(len(prior["track_ids"]), 2)
        np.testing.assert_allclose(prior["points_robot_base_m"], expected)
        self.assertNotIn(
            manager._entries["off-hand"]["track_id"], prior["track_ids"]
        )
        self.assertEqual(
            manager._entries["off-hand"]["status"], "observed"
        )

        manager.ingest(self._frame(2, objects))
        self.assertTrue(
            manager.observed_active_points_snapshot(
                ["hand-a", "hand-b"], episode_id="episode-a"
            )["ok"]
        )
        wrong_objects = [
            (centers["hand-a"], 1.10, self.can_texture),
            (centers["hand-b"] + [0.0, 42.0], 1.09, self.bin_texture),
            (centers["off-hand"], 1.70, self.can_texture),
        ]
        manager.ingest(self._frame(3, wrong_objects))
        rejected_memory = manager.memory_fields()["tracked_object_distances"]
        self.assertEqual(
            rejected_memory["hand-b"]["status"], "temporarily_unobserved"
        )
        self.assertIsNone(rejected_memory["hand-b"]["depth_m"])
        self.assertIsNone(
            rejected_memory["hand-b"]["xyz_in_robot_base_coord_m"]
        )
        self.assertEqual(rejected_memory["off-hand"]["status"], "observed")
        rejected_report = manager.status()["active_rigid_pair_report"]
        self.assertFalse(rejected_report["ok"])
        self.assertFalse(rejected_report["fusion_applied"])
        self.assertFalse(
            rejected_report["visual_tracker_pair"][
                "prediction_values_published_as_observation"
            ]
        )

        manager.ingest(self._frame(4, objects))
        recovered = manager.observed_active_points_snapshot(
            ["hand-a", "hand-b", "off-hand"], episode_id="episode-a"
        )
        self.assertTrue(recovered["ok"], recovered)
        self.assertEqual(recovered["entries"]["hand-b"]["status"], "observed")
        released = manager.deactivate_on_hand_eef_observation_prior(
            activation["lease_id"]
        )
        self.assertTrue(released["released"])
        self.assertIsNone(
            manager.status()["active_on_hand_eef_observation_prior"]
        )

    def test_on_hand_eef_prior_gates_three_points_without_touching_off_hand(
        self,
    ) -> None:
        manager = TrackedObjectDistanceMemory()
        third_texture = self._texture(47, "can")
        off_texture = self._texture(71, "bin")
        centers = {
            "hand-a": np.array([34.0, 45.0]),
            "hand-b": np.array([90.0, 48.0]),
            "hand-c": np.array([147.0, 51.0]),
            "off-hand": np.array([48.0, 112.0]),
        }
        objects = [
            (centers["hand-a"], 1.10, self.can_texture),
            (centers["hand-b"], 1.16, self.bin_texture),
            (centers["hand-c"], 1.22, third_texture),
            (centers["off-hand"], 1.65, off_texture),
        ]
        manager.ingest(self._frame(1, objects))
        binding = self._register_capture(manager)
        started = manager.replace_points(
            [
                {
                    "name": name,
                    "u": self._relative(center)[0],
                    "v": self._relative(center)[1],
                }
                for name, center in centers.items()
            ],
            **self._binding_kwargs(binding),
        )

        proprio = np.asarray(manager._latest_frame.proprio, dtype=np.float64)
        state = local_robot_state(
            trunk_q=proprio[PROPRIO_SLICES["trunk_qpos"]],
            arm_left_q=proprio[PROPRIO_SLICES["arm_left_qpos"]],
            arm_right_q=proprio[PROPRIO_SLICES["arm_right_qpos"]],
            gripper_left_q=proprio[PROPRIO_SLICES["gripper_left_qpos"]],
            gripper_right_q=proprio[PROPRIO_SLICES["gripper_right_qpos"]],
        )
        q_left = proprio[PROPRIO_SLICES["arm_left_qpos"]]
        eef_position, eef_quaternion = eef_pose(state, "left", q_left)
        rotation = quat_to_mat_xyzw(eef_quaternion)
        hand_names = ("hand-a", "hand-b", "hand-c")
        expected = np.asarray(
            [
                started["entries"][name]["xyz_in_robot_base_coord_m"]
                for name in hand_names
            ],
            dtype=np.float64,
        )
        anchors = (rotation.T @ (expected - eef_position).T).T
        activation = manager.activate_on_hand_eef_observation_prior(
            hand_names,
            episode_id="episode-a",
            session_id=binding["session_id"],
            image_id=binding["image_id"],
            arm="left",
            anchors_eef_m=anchors,
        )
        self.assertTrue(activation["ok"], activation)
        self.assertEqual(
            activation["tracking_mode"],
            "independent_npoint_eef_candidate_gates",
        )
        self.assertIsNone(manager.status()["active_rigid_pair"])
        with manager._lock:
            prior = manager._on_hand_eef_prior_for_frame_locked(
                manager._latest_frame
            )
        self.assertEqual(len(prior["track_ids"]), 3)
        np.testing.assert_allclose(prior["points_robot_base_m"], expected)
        off_track_id = manager._entries["off-hand"]["track_id"]
        self.assertNotIn(off_track_id, prior["track_ids"])

        manager.ingest(self._frame(2, objects))
        wrong_objects = [
            (centers["hand-a"], 1.10, self.can_texture),
            (centers["hand-b"], 1.16, self.bin_texture),
            (centers["hand-c"] + [0.0, 40.0], 1.22, third_texture),
            (centers["off-hand"], 1.65, off_texture),
        ]
        manager.ingest(self._frame(3, wrong_objects))
        rejected = manager.memory_fields()["tracked_object_distances"]
        self.assertEqual(rejected["hand-a"]["status"], "observed")
        self.assertEqual(rejected["hand-b"]["status"], "observed")
        self.assertEqual(
            rejected["hand-c"]["status"], "temporarily_unobserved"
        )
        self.assertIsNone(rejected["hand-c"]["xyz_in_robot_base_coord_m"])
        self.assertEqual(rejected["off-hand"]["status"], "observed")
        hand_c_track_id = manager._entries["hand-c"]["track_id"]
        rejection_reasons = {
            str(report.get("reason"))
            for report in manager._tracker._tracks[
                hand_c_track_id
            ].last_observation_rejections
        }
        self.assertTrue(
            rejection_reasons
            & {
                "eef_prior_pixel_mismatch",
                "eef_prior_depth_layer_mismatch",
                "eef_prior_point_mismatch",
            },
            rejection_reasons,
        )

        manager.ingest(self._frame(4, objects))
        recovered = manager.observed_active_points_snapshot(
            [*hand_names, "off-hand"], episode_id="episode-a"
        )
        self.assertTrue(recovered["ok"], recovered)
        np.testing.assert_allclose(
            [
                recovered["entries"][name]["xyz_in_robot_base_coord_m"]
                for name in hand_names
            ],
            expected,
            atol=1e-7,
        )
        released = manager.deactivate_on_hand_eef_observation_prior(
            activation["lease_id"]
        )
        self.assertTrue(released["released"], released)

    def test_frozen_image_uv_replays_to_moved_object_in_current_frame(self) -> None:
        manager = TrackedObjectDistanceMemory()
        capture_center = np.array([52.0, 60.0])
        current_center = np.array([70.0, 67.0])
        manager.ingest(
            self._frame(
                1,
                [(capture_center, 1.20, self.can_texture)],
            )
        )
        binding = self._register_capture(manager)
        manager.ingest(
            self._frame(
                2,
                [(current_center, 0.91, self.can_texture)],
            )
        )

        u, v = self._relative(capture_center)
        tracked = manager.replace_points(
            [{"name": "moving_can", "u": u, "v": v}],
            **self._binding_kwargs(binding),
        )

        entry = tracked["entries"]["moving_can"]
        self.assertAlmostEqual(entry["depth_m"], 0.91, places=5)
        self.assertNotAlmostEqual(entry["depth_m"], 4.0, places=3)
        np.testing.assert_allclose(
            [entry["u"], entry["v"]],
            self._relative(current_center),
            atol=2.0,
        )
        self.assertEqual(entry["source_image_id"], "img_0001")
        self.assertEqual(entry["source_observation_sequence"], 1)
        self.assertEqual(tracked["capture_observation_sequence"], 1)
        self.assertEqual(tracked["observation_sequence"], 2)
        self.assertEqual(tracked["replayed_observation_count"], 1)

    def test_replay_gap_fails_without_replacing_existing_tracker(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([52.0, 60.0])
        manager.ingest(self._frame(1, [(center, 1.20, self.can_texture)]))
        binding = self._register_capture(manager)
        u, v = self._relative(center)
        manager.replace_points(
            [{"name": "existing", "u": u, "v": v}],
            **self._binding_kwargs(binding),
        )
        manager.ingest(self._frame(3, [(center, 1.10, self.can_texture)]))

        with self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points(
                [{"name": "replacement", "u": u, "v": v}],
                **self._binding_kwargs(binding),
            )
        self.assertEqual(caught.exception.reason, "replay_gap")

        self.assertEqual(manager.status()["registered_names"], ["existing"])
        self.assertEqual(
            list(manager.memory_fields()["tracked_object_distances"]),
            ["existing"],
        )

    def test_replay_cancellation_is_atomic(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([52.0, 60.0])
        manager.ingest(self._frame(1, [(center, 1.20, self.can_texture)]))
        binding = self._register_capture(manager)
        u, v = self._relative(center)
        manager.replace_points(
            [{"name": "existing", "u": u, "v": v}],
            **self._binding_kwargs(binding),
        )
        manager.ingest(self._frame(2, [(center + [2.0, 1.0], 1.10, self.can_texture)]))

        calls = 0

        def cancel_before_swap() -> None:
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("request cancelled")

        with self.assertRaisesRegex(RuntimeError, "request cancelled"):
            manager.replace_points(
                [{"name": "replacement", "u": u, "v": v}],
                check_cancelled=cancel_before_swap,
                **self._binding_kwargs(binding),
            )

        self.assertEqual(manager.status()["registered_names"], ["existing"])
        self.assertEqual(
            list(manager.memory_fields()["tracked_object_distances"]),
            ["existing"],
        )

        @contextmanager
        def cancelled_commit():
            yield False

        with self.assertRaisesRegex(ValueError, "cancelled before commit"):
            manager.replace_points(
                [{"name": "replacement", "u": u, "v": v}],
                commit_guard=cancelled_commit,
                **self._binding_kwargs(binding),
            )
        self.assertEqual(manager.status()["registered_names"], ["existing"])

    def test_replay_workload_limit_fails_without_replacing_tracker(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([52.0, 60.0])
        manager.ingest(self._frame(1, [(center, 1.20, self.can_texture)]))
        binding = self._register_capture(manager)
        u, v = self._relative(center)
        manager.replace_points(
            [{"name": "existing", "u": u, "v": v}],
            **self._binding_kwargs(binding),
        )
        for sequence in range(2, 130):
            manager.ingest(
                self._frame(
                    sequence,
                    [(center, 1.20, self.can_texture)],
                )
            )

        # Exercise an explicitly restricted deployment budget; the default
        # now permits all public points throughout the retained history.
        with mock.patch(
            "behavior_interface_eval_test.tool.official_v2."
            "tracked_object_distance.TRACK_OBJECT_DISTANCE_MAX_REPLAY_WORK",
            4096,
        ), self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points(
                [
                    {"name": f"point-{index}", "u": u, "v": v}
                    for index in range(32)
                ],
                **self._binding_kwargs(binding),
            )
        self.assertEqual(caught.exception.reason, "replay_work_exceeded")

        self.assertEqual(manager.status()["registered_names"], ["existing"])

    def test_ten_minute_30hz_binding_replays_every_frame_from_cold_storage(self):
        manager = TrackedObjectDistanceMemory(replay_memory_max_bytes=1024)
        can = np.array([52.0, 60.0])
        bin_center = np.array([142.0, 82.0])
        objects = [(can, 1.20, self.can_texture), (bin_center, 2.10, self.bin_texture)]
        first = self._frame(1, objects)
        manager.ingest(first)
        binding = self._register_capture(manager)
        for sequence in range(2, TRACK_OBJECT_DISTANCE_REPLAY_MAX_FRAMES + 1):
            # Exact metadata for 30 Hz; no sleep, subsampling or tracker mock.
            frame = replace(first, sequence=sequence, timestamp_s=sequence / 30.0)
            if sequence == TRACK_OBJECT_DISTANCE_REPLAY_MAX_FRAMES:
                frame = self._frame(sequence, [
                    (can + [5.0, 2.0], 1.10, self.can_texture),
                    (bin_center, 2.10, self.bin_texture),
                ])
            manager.ingest(frame, copy_frame=False)
            if sequence in (1801, 2048, 2049, 2050, 3601, 7201, 18001):
                manager._flush_replay_through(sequence)
                with manager._lock:
                    _, replay = manager._bound_replay_frames_locked(**self._binding_kwargs(binding))
                self.assertEqual(len(replay), sequence)
                self.assertEqual(replay[0].sequence, 1)
                self.assertEqual(replay[-1].sequence, sequence)
                self.assertLessEqual(32 * len(replay), TRACK_OBJECT_DISTANCE_MAX_REPLAY_WORK)
                self.assertLessEqual(manager.status()["replay_memory_cache_bytes"], 1024)

        self.assertIsNone(manager._frame_history[0].archive_payload.cached)
        can_uv, bin_uv = self._relative(can), self._relative(bin_center)
        # Prove the unchanged tracker sees every archived sequence and reports
        # today's moving surface, not the capture's old depth / UV / XYZ.
        original_ingest = DynamicPointTracker.ingest
        observed_sequences = []
        def recording_ingest(tracker, frame, **kwargs):
            observed_sequences.append(frame.sequence)
            return original_ingest(tracker, frame, **kwargs)
        with mock.patch.object(DynamicPointTracker, "ingest", recording_ingest):
            tracked = manager.replace_points([
                {"name": "can", "u": can_uv[0], "v": can_uv[1]},
                {"name": "bin", "u": bin_uv[0], "v": bin_uv[1]},
            ], **self._binding_kwargs(binding))
        self.assertEqual(observed_sequences, list(range(1, 18002)))
        self.assertEqual(tracked["replayed_observation_count"], 18000)
        self.assertAlmostEqual(tracked["entries"]["can"]["depth_m"], 1.10, places=5)
        self.assertAlmostEqual(tracked["entries"]["bin"]["depth_m"], 2.10, places=5)
        self.assertAlmostEqual(tracked["registration_entries"]["can"]["depth_m"], 1.20, places=5)
        self.assertLess(np.linalg.norm(
            np.asarray([tracked["entries"]["can"][axis] for axis in ("u", "v")])
            - np.asarray(self._relative(can + [5.0, 2.0]))
        ), 3.0)
        self.assertLess(np.linalg.norm(
            np.asarray(tracked["entries"]["bin"]["xyz_in_robot_base_coord_m"])
            - np.asarray(tracked["registration_entries"]["bin"]["xyz_in_robot_base_coord_m"])
        ), 0.005)

    def test_cold_replay_matches_hot_replay_exactly_for_camera_and_object_motion(self):
        managers = [TrackedObjectDistanceMemory(replay_memory_max_bytes=size) for size in (0, 640*1024**2)]
        can, bin_center = np.array([60.0, 60.0]), np.array([140.0, 84.0])
        frames = [self._frame(
            seq, [(can + [seq-1, 0], 1.2 - 0.01*(seq-1), self.can_texture), (bin_center, 2.1, self.bin_texture)],
            camera_position_robot=(0.0, 0.001*seq, 1.0),
            base_xy_yaw=(0.001*seq, 0.0, 0.001*seq),
        ) for seq in range(1, 16)]
        results = []
        for manager in managers:
            manager.ingest(frames[0])
            binding = self._register_capture(manager)
            for frame in frames[1:]:
                manager.ingest(frame)
            u, v = self._relative(can)
            results.append(manager.replace_points([{"name": "can", "u": u, "v": v}], **self._binding_kwargs(binding)))
        self.assertEqual(results[0], results[1])
        restored = _restore_replay_frame(managers[0]._frame_history[-1])
        expected = _compact_replay_frame(frames[-1])
        for field in ("rgb", "depth_linear", "camera_to_robot_base", "camera_to_policy"):
            np.testing.assert_array_equal(getattr(restored, field), getattr(expected, field))
        self.assertEqual(restored.timestamp_s, expected.timestamp_s)

    def test_cold_replay_corruption_never_registers_stale_coordinates(self):
        manager = TrackedObjectDistanceMemory(replay_memory_max_bytes=0)
        manager.ingest(self._frame(1, [((52, 60), 1.2, self.can_texture)]))
        binding = self._register_capture(manager)
        payload = manager._frame_history[0].archive_payload
        os.pwrite(payload.chunk.stream.fileno(), b"corrupt", payload.offset)
        u, v = self._relative((52, 60))
        with self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points([{"name": "can", "u": u, "v": v}], **self._binding_kwargs(binding))
        self.assertEqual(caught.exception.reason, "replay_history_corrupt")
        self.assertEqual(manager.memory_fields()["tracked_object_distances"], {})

    def test_cold_archive_reset_releases_cache_and_rejects_previous_episode(self):
        manager = TrackedObjectDistanceMemory()
        manager.ingest(self._frame(1, [((52, 60), 1.2, self.can_texture)]))
        binding = self._register_capture(manager)
        self.assertGreater(manager.status()["replay_memory_cache_bytes"], 0)
        manager.reset()
        self.assertEqual(manager.status()["replay_memory_cache_bytes"], 0)
        manager.ingest(self._frame(1, [((52, 60), 1.1, self.can_texture)], episode_id="episode-b"))
        u, v = self._relative((52, 60))
        with self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points([{"name": "can", "u": u, "v": v}], **self._binding_kwargs(binding))
        self.assertEqual(caught.exception.reason, "episode_changed")
        self.assertEqual(manager.memory_fields()["tracked_object_distances"], {})

    def test_background_archive_disk_failure_invalidates_binding_without_xyz(self):
        manager = TrackedObjectDistanceMemory()
        frame = self._frame(1, [((52, 60), 1.2, self.can_texture)])
        manager.ingest(frame)
        binding = self._register_capture(manager)
        with mock.patch.object(manager._replay_archive, "store", side_effect=OSError("disk full")):
            manager.ingest(replace(frame, sequence=2))
            manager._flush_replay_through(2)
        u, v = self._relative((52, 60))
        with self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points([{"name": "can", "u": u, "v": v}], **self._binding_kwargs(binding))
        self.assertEqual(caught.exception.reason, "replay_storage_unavailable")
        self.assertEqual(manager.memory_fields()["tracked_object_distances"], {})

    def test_cold_replay_gap_and_digest_mismatch_remain_rejected(self):
        manager = TrackedObjectDistanceMemory(replay_memory_max_bytes=0)
        frame = self._frame(1, [((52, 60), 1.2, self.can_texture)])
        manager.ingest(frame)
        binding = self._register_capture(manager)
        manager.ingest(replace(frame, sequence=3))
        u, v = self._relative((52, 60))
        for digest, reason in (("tampered", "digest_mismatch"), (binding["frame_digest"], "replay_gap")):
            kwargs = self._binding_kwargs(binding)
            kwargs["capture_frame_digest"] = digest
            with self.assertRaises(TrackObjectDistanceBindingError) as caught:
                manager.replace_points([{"name": "can", "u": u, "v": v}], **kwargs)
            self.assertEqual(caught.exception.reason, reason)
            self.assertEqual(manager.memory_fields()["tracked_object_distances"], {})

    def test_evicted_capture_binding_fails_closed(self) -> None:
        manager = TrackedObjectDistanceMemory(replay_max_frames=2)
        center = np.array([52.0, 60.0])
        manager.ingest(self._frame(1, [(center, 1.20, self.can_texture)]))
        binding = self._register_capture(manager)
        manager.ingest(self._frame(2, [(center, 1.15, self.can_texture)]))
        manager.ingest(self._frame(3, [(center, 1.10, self.can_texture)]))
        u, v = self._relative(center)

        with self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points(
                [{"name": "can", "u": u, "v": v}],
                **self._binding_kwargs(binding),
            )
        self.assertEqual(caught.exception.reason, "replay_history_evicted")
        self.assertEqual(manager.status()["registered"], 0)

    def test_default_replay_has_no_179_180_181_frame_cliff(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([52.0, 60.0])
        manager.ingest(self._frame(1, [(center, 1.20, self.can_texture)]))
        bindings = [
            self._register_capture(manager, image_id="age-181"),
        ]
        for sequence in range(2, 183):
            manager.ingest(
                self._frame(
                    sequence,
                    [(center, 1.20, self.can_texture)],
                )
            )
            if sequence in (2, 3):
                bindings.append(
                    self._register_capture(
                        manager,
                        image_id=f"age-{182 - sequence}",
                    )
                )

        self.assertEqual(manager.status()["capture_bindings"], 3)
        u, v = self._relative(center)
        expected_ages = (181, 180, 179)
        for binding, expected_age in zip(bindings, expected_ages):
            with self.subTest(age=expected_age):
                tracked = manager.replace_points(
                    [{"name": f"can-{expected_age}", "u": u, "v": v}],
                    **self._binding_kwargs(binding),
                )
                self.assertEqual(
                    tracked["replayed_observation_count"],
                    expected_age,
                )
                self.assertAlmostEqual(
                    tracked["entries"][f"can-{expected_age}"]["depth_m"],
                    1.20,
                    places=5,
                )

    def test_replay_supports_60_and_120_plus_seconds_at_observed_rate(self) -> None:
        manager = TrackedObjectDistanceMemory()
        can = np.array([52.0, 60.0])
        bin_center = np.array([142.0, 82.0])
        objects = [
            (can, 1.20, self.can_texture),
            (bin_center, 2.10, self.bin_texture),
        ]
        first = replace(self._frame(1, objects), timestamp_s=0.0)
        manager.ingest(first)
        long_binding = self._register_capture(manager, image_id="age-124s")
        short_binding = None
        for sequence in range(2, 324):
            manager.ingest(
                replace(
                    self._frame(sequence, objects),
                    timestamp_s=float(sequence - 1) / 2.6,
                )
            )
            if sequence == 166:
                short_binding = self._register_capture(
                    manager,
                    image_id="age-60s",
                )

        self.assertIsNotNone(short_binding)
        can_u, can_v = self._relative(can)
        bin_u, bin_v = self._relative(bin_center)
        long_result = manager.replace_points(
            [
                {"name": "can", "u": can_u, "v": can_v},
                {"name": "bin", "u": bin_u, "v": bin_v},
            ],
            **self._binding_kwargs(long_binding),
        )
        self.assertEqual(long_result["replayed_observation_count"], 322)
        self.assertAlmostEqual(
            long_result["entries"]["can"]["depth_m"],
            1.20,
            places=5,
        )
        self.assertAlmostEqual(
            long_result["entries"]["bin"]["depth_m"],
            2.10,
            places=5,
        )

        short_result = manager.replace_points(
            [{"name": "can-short", "u": can_u, "v": can_v}],
            **self._binding_kwargs(short_binding),
        )
        self.assertEqual(short_result["replayed_observation_count"], 157)
        self.assertGreaterEqual(
            manager.status()["replay_history_max_bytes"],
            manager.status()["replay_history_bytes"],
        )

    def test_ninth_binding_evicts_only_oldest_with_stable_reason(self) -> None:
        manager = TrackedObjectDistanceMemory(max_capture_bindings=8)
        center = np.array([52.0, 60.0])
        manager.ingest(self._frame(1, [(center, 1.20, self.can_texture)]))
        bindings = [
            self._register_capture(manager, image_id=f"img-{index}")
            for index in range(9)
        ]
        self.assertEqual(manager.status()["capture_bindings"], 8)
        u, v = self._relative(center)
        with self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points(
                [{"name": "evicted", "u": u, "v": v}],
                **self._binding_kwargs(bindings[0]),
            )
        self.assertEqual(caught.exception.reason, "binding_evicted")

        tracked = manager.replace_points(
            [{"name": "retained", "u": u, "v": v}],
            **self._binding_kwargs(bindings[1]),
        )
        self.assertIn("retained", tracked["entries"])

    def test_binding_not_registered_and_session_mismatch_are_distinct(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([52.0, 60.0])
        manager.ingest(self._frame(1, [(center, 1.20, self.can_texture)]))
        binding = self._register_capture(
            manager,
            session_id="session-a",
            image_id="shared-image",
        )
        u, v = self._relative(center)
        wrong_session = dict(self._binding_kwargs(binding))
        wrong_session["session_id"] = "session-b"
        with self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points(
                [{"name": "can", "u": u, "v": v}],
                **wrong_session,
            )
        self.assertEqual(caught.exception.reason, "session_mismatch")

        never_registered = dict(self._binding_kwargs(binding))
        never_registered["image_id"] = "never-registered"
        with self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points(
                [{"name": "can", "u": u, "v": v}],
                **never_registered,
            )
        self.assertEqual(caught.exception.reason, "binding_not_registered")

    def test_other_session_does_not_consume_binding_quota(self) -> None:
        manager = TrackedObjectDistanceMemory(max_capture_bindings=8)
        center = np.array([52.0, 60.0])
        manager.ingest(self._frame(1, [(center, 1.20, self.can_texture)]))
        session_a = [
            self._register_capture(
                manager,
                session_id="session-a",
                image_id=f"img-a-{index}",
            )
            for index in range(8)
        ]
        self._register_capture(
            manager,
            session_id="session-b",
            image_id="img-b-0",
        )
        status = manager.status()
        self.assertEqual(status["capture_bindings"], 9)
        self.assertEqual(status["capture_bindings_by_session"]["session-a"], 8)
        self.assertEqual(status["capture_bindings_by_session"]["session-b"], 1)

        u, v = self._relative(center)
        tracked = manager.replace_points(
            [{"name": "retained-a", "u": u, "v": v}],
            **self._binding_kwargs(session_a[0]),
        )
        self.assertIn("retained-a", tracked["entries"])

    def test_byte_budget_eviction_is_bounded_and_machine_readable(self) -> None:
        center = np.array([52.0, 60.0])
        probe = TrackedObjectDistanceMemory()
        probe.ingest(self._frame(1, [(center, 1.20, self.can_texture)]))
        self._register_capture(probe)
        one_frame_bytes = probe.status()["replay_history_bytes"]

        manager = TrackedObjectDistanceMemory(
            replay_max_bytes=one_frame_bytes * 2 + 1,
        )
        manager.ingest(self._frame(1, [(center, 1.20, self.can_texture)]))
        binding = self._register_capture(manager)
        manager.ingest(self._frame(2, [(center, 1.20, self.can_texture)]))
        manager.ingest(self._frame(3, [(center, 1.20, self.can_texture)]))
        self.assertLessEqual(
            manager.status()["replay_history_bytes"],
            manager.status()["replay_history_max_bytes"],
        )
        u, v = self._relative(center)
        with self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points(
                [{"name": "can", "u": u, "v": v}],
                **self._binding_kwargs(binding),
            )
        self.assertEqual(caught.exception.reason, "replay_history_evicted")

    def test_capture_binding_rejects_wrong_episode_sequence_and_shape(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([52.0, 60.0])
        manager.ingest(self._frame(4, [(center, 1.20, self.can_texture)]))
        for overrides, message in (
            ({"observation_sequence": 3}, "sequence"),
            ({"episode_id": "episode-b"}, "episode"),
            ({"image_shape": (10, 10)}, "resolution"),
        ):
            kwargs = {
                "session_id": "unit-session",
                "image_id": "img_bad",
                "observation_sequence": 4,
                "episode_id": "episode-a",
                "image_shape": (self.HEIGHT, self.WIDTH),
                **overrides,
            }
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    manager.register_capture(**kwargs)

    def test_new_capture_does_not_rebind_an_active_track(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center_1 = np.array([52.0, 60.0])
        center_2 = center_1 + [6.0, 3.0]
        manager.ingest(self._frame(1, [(center_1, 1.20, self.can_texture)]))
        first_binding = self._register_capture(
            manager,
            image_id="img_first",
        )
        u, v = self._relative(center_1)
        manager.replace_points(
            [{"name": "can", "u": u, "v": v}],
            **self._binding_kwargs(first_binding),
        )
        manager.ingest(self._frame(2, [(center_2, 1.10, self.can_texture)]))

        second_binding = self._register_capture(
            manager,
            image_id="img_second",
        )

        memory = manager.memory_fields()["tracked_object_distances"]
        self.assertEqual(memory["can"]["source_image_id"], "img_first")
        self.assertEqual(manager.status()["source_image_id"], "img_first")
        self.assertEqual(second_binding["observation_sequence"], 2)
        self.assertEqual(manager.status()["capture_bindings"], 2)

    def test_multiple_featureless_points_register_as_persistent_read_depths(self) -> None:
        manager = TrackedObjectDistanceMemory()
        flat_chair = np.full((43, 43), 112, dtype=np.uint8)
        chair_1 = np.array([68.0, 62.0])
        chair_2 = np.array([138.0, 92.0])
        manager.ingest(
            self._frame(
                1,
                [
                    (chair_1, 1.34, flat_chair),
                    (chair_2, 2.18, flat_chair),
                ],
            )
        )
        chair_1_uv = self._relative(chair_1)
        chair_2_uv = self._relative(chair_2)

        started = self._replace_points(
            manager,
            [
                {
                    "name": "chair1",
                    "u": chair_1_uv[0],
                    "v": chair_1_uv[1],
                },
                {
                    "name": "chair2",
                    "u": chair_2_uv[0],
                    "v": chair_2_uv[1],
                },
            ]
        )

        self.assertEqual(sorted(started["entries"]), ["chair1", "chair2"])
        self.assertAlmostEqual(
            started["entries"]["chair1"]["depth_m"],
            1.34,
            places=5,
        )
        self.assertAlmostEqual(
            started["entries"]["chair2"]["depth_m"],
            2.18,
            places=5,
        )
        self.assertLess(
            started["entries"]["chair1"]["confidence"],
            1.0,
        )

    def test_xyz_in_current_robot_base_coord_matches_depth_and_camera_pose(
        self,
    ) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([70.0, 55.0])
        depth_m = 1.40
        camera_position_robot = np.array([0.30, -0.20, 1.10])
        half_sqrt = np.sqrt(0.5)
        camera_quaternion_robot = [0.0, 0.0, half_sqrt, half_sqrt]
        manager.ingest(
            self._frame(
                1,
                [(center, depth_m, self.can_texture)],
                base_xy_yaw=[5.0, -4.0, 0.7],
                camera_position_robot=camera_position_robot,
                camera_quaternion_robot=camera_quaternion_robot,
            )
        )
        u, v = self._relative(center)

        entry = self._replace_points(
            manager,
            [{"name": "can", "u": u, "v": v}],
        )["entries"]["can"]

        focal = 306.0 * self.WIDTH / 720.0
        point_camera = np.array(
            [
                (center[0] - self.WIDTH * 0.5) / focal * depth_m,
                -(center[1] - self.HEIGHT * 0.5) / focal * depth_m,
                -depth_m,
            ]
        )
        expected_robot_base = np.array(
            [
                camera_position_robot[0] - point_camera[1],
                camera_position_robot[1] + point_camera[0],
                camera_position_robot[2] + point_camera[2],
            ]
        )
        np.testing.assert_allclose(
            entry["xyz_in_robot_base_coord_m"],
            expected_robot_base,
            atol=1e-6,
            rtol=0.0,
        )
        self.assertEqual(entry["xyz_frame"], "current_robot_base")
        self.assertEqual(entry["xyz_axes"], "x_forward_y_left_z_up")
        self.assertEqual(
            entry["xyz_source"],
            "current_evaluator_depth_linear_and_cam_rel_poses",
        )
        text = manager.text()
        self.assertIn("xyz_in_robot_base_coord=", text)
        self.assertIn("XYZ in current robot base coord", text)

    def test_replace_is_atomic_when_any_point_is_invalid(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([52.0, 60.0])
        frame = self._frame(1, [(center, 1.2, self.can_texture)])
        invalid_depth = np.asarray(frame.depth_linear).copy()
        invalid_depth[0:3, 0:3] = np.nan
        manager.ingest(replace(frame, depth_linear=invalid_depth))
        u, v = self._relative(center)
        binding = self._register_capture(manager)
        manager.replace_points(
            [{"name": "can", "u": u, "v": v}],
            **self._binding_kwargs(binding),
        )

        with self.assertRaisesRegex(
            ValueError,
            "no valid evaluator depth",
        ):
            manager.replace_points(
                [
                    {"name": "replacement", "u": u, "v": v},
                    {"name": "invalid", "u": 0.0, "v": 0.0},
                ],
                **self._binding_kwargs(binding),
            )
        self.assertEqual(
            list(manager.memory_fields()["tracked_object_distances"]),
            ["can"],
        )

    def test_episode_change_and_reset_clear_memory(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([52.0, 60.0])
        manager.ingest(self._frame(1, [(center, 1.2, self.can_texture)]))
        binding = self._register_capture(manager, image_id="old-episode")
        u, v = self._relative(center)
        self._replace_points(
            manager,
            [{"name": "can", "u": u, "v": v}],
            binding=binding,
        )

        manager.ingest(
            self._frame(2, [(center, 1.2, self.can_texture)], episode_id="episode-b")
        )
        self.assertEqual(manager.memory_fields()["tracked_object_distances"], {})
        with self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points(
                [{"name": "stale", "u": u, "v": v}],
                **self._binding_kwargs(binding),
            )
        self.assertEqual(caught.exception.reason, "episode_changed")
        reset_binding = self._register_capture(
            manager,
            image_id="before-runtime-reset",
        )
        manager.reset()
        self.assertIsNone(manager.status()["observation_sequence"])
        with self.assertRaises(TrackObjectDistanceBindingError) as caught:
            manager.replace_points(
                [{"name": "after-reset", "u": u, "v": v}],
                **self._binding_kwargs(reset_binding),
            )
        self.assertEqual(caught.exception.reason, "episode_changed")

    def test_frame_builder_rejects_missing_mismatch_and_privileged_keys(self) -> None:
        good = self._observation([])
        cases = []
        missing_pose = dict(good)
        missing_pose.pop("robot_r1::cam_rel_poses")
        cases.append((missing_pose, "camera-relative"))
        mismatch = dict(good)
        mismatch["robot_r1::zed_link::depth_linear"] = np.ones((8, 9), dtype=np.float32)
        cases.append((mismatch, "resolutions"))
        privileged = dict(good)
        privileged["robot_r1::zed_link::seg_instance_id"] = np.zeros((self.HEIGHT, self.WIDTH))
        cases.append((privileged, "non-allowlisted"))
        invalid_pose = dict(good)
        invalid_pose["robot_r1::cam_rel_poses"] = np.asarray(
            good["robot_r1::cam_rel_poses"]
        ).copy()
        invalid_pose["robot_r1::cam_rel_poses"][17:21] = 0.0
        cases.append((invalid_pose, "zero norm"))
        for observation, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    tracker_frame_from_allowed_observation(
                        observation,
                        sequence=1,
                        timestamp_s=0.1,
                        episode_id="episode",
                        base_xy_yaw=[0.0, 0.0, 0.0],
                    )

    def test_tool_returns_hold_action_and_writes_structured_result(self) -> None:
        manager = TrackedObjectDistanceMemory()
        center = np.array([52.0, 60.0])
        frame = self._frame(1, [(center, 1.2, self.can_texture)])
        manager.ingest(frame)
        binding = self._register_capture(manager)
        u, v = self._relative(center)
        ctx = _Context(manager)

        frozen_capture = SimpleNamespace(
            rgb=cv2.cvtColor(
                np.asarray(frame.rgb),
                cv2.COLOR_RGB2BGR,
            ),
            depth=np.asarray(frame.depth_linear).copy(),
            evaluator_sequence=binding["observation_sequence"],
            robot={"episode_id": binding["episode_id"]},
            camera={
                "robot_relative_pose": {
                    "pos": [0.0, 0.0, 0.0],
                    "quat": [0.0, 0.0, 0.0, 1.0],
                }
            },
        )
        with mock.patch(
            "behavior_interface_eval_test.tool.official_v2.tools._load_frozen_capture",
            return_value=frozen_capture,
        ):
            actions = list(
                track_object_distance(
                    ctx,
                    binding["session_id"],
                    binding["image_id"],
                    [{"name": "can", "u": u, "v": v}],
                )
            )

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].shape, (ACTION_DIM,))
        self.assertTrue(np.all(np.isfinite(actions[0])))
        self.assertTrue(ctx.result["ok"])
        self.assertEqual(ctx.result["execution"], "official_action_only")
        self.assertFalse(ctx.result["direct_simulator_mutation"])
        self.assertEqual(
            ctx.result["replacement_policy"],
            "complete_named_set_on_success",
        )
        self.assertAlmostEqual(
            ctx.result["tracked_object_distances"]["can"]["depth_m"],
            1.2,
            places=5,
        )
        self.assertEqual(
            len(
                ctx.result["tracked_object_distances"]["can"][
                    "xyz_in_robot_base_coord_m"
                ]
            ),
            3,
        )
        self.assertEqual(
            ctx.result["xyz_in_robot_base_coord_frame"],
            "current_robot_base",
        )
        self.assertEqual(ctx.result["image_id"], binding["image_id"])
        self.assertEqual(ctx.result["capture_observation_sequence"], 1)
        self.assertEqual(
            ctx.result["tracked_object_distances"]["can"]["source_image_id"],
            binding["image_id"],
        )

    def test_capture_binds_displayed_image_to_exact_tracker_sequence(self) -> None:
        adapter = ObservationActionAdapter()
        center = np.array([52.0, 60.0])
        normalized_observation = self._observation(
            [(center, 1.20, self.can_texture)]
        )
        rgb_key = "robot_r1::zed_link::rgb"
        normalized_observation[rgb_key] = (
            np.asarray(normalized_observation[rgb_key], dtype=np.float32)
            / np.float32(255.0)
        )
        snapshot = adapter.update(normalized_observation)
        manager = TrackedObjectDistanceMemory()
        manager.ingest(
            tracker_frame_from_allowed_observation(
                snapshot.observation,
                sequence=snapshot.sequence,
                timestamp_s=snapshot.received_ts,
                episode_id="episode-a",
                base_xy_yaw=[0.0, 0.0, 0.0],
            )
        )
        world = SimpleNamespace(
            _official_adapter=adapter,
            _official_tracked_object_distances=manager,
            hold_action=lambda: np.zeros(ACTION_DIM, dtype=np.float32),
            local_pose_from_robot_relative=lambda pos, quat: {
                "pos": list(pos),
                "quat": list(quat),
                "frame": "local_command_odometry",
            },
        )
        ctx = _Context()
        ctx.world = world
        ctx.task_name = "make_microwave_popcorn"
        robot_state = {
            "episode_id": "episode-a",
            "motion_epoch": 0,
            "base_pose": {"pos": [0.0, 0.0, 0.0], "yaw": 0.0},
        }
        overlay = {"ok": False, "error": "disabled in unit test"}

        with tempfile.TemporaryDirectory() as temp_root, mock.patch.dict(
            os.environ,
            {"BEHAVIOR_AGENT_RUNS": temp_root},
        ), mock.patch(
            "behavior_interface_eval_test.tool.official_v2.tools._capture_robot_state",
            return_value=robot_state,
        ), mock.patch(
            "behavior_interface_eval_test.tool.official_v2.tools.render_head_path_overlay",
            return_value=overlay,
        ):
            list(capture_head_camera(ctx, session_id="unit-session"))
            capture_result = dict(ctx.result)
            binding = capture_result["track_object_distance_binding"]
            self.assertTrue(binding["ok"], binding)
            self.assertTrue(binding["trackable"], binding)
            self.assertIsNone(binding["reason"])
            self.assertEqual(binding["image_id"], capture_result["image_id"])
            self.assertEqual(binding["observation_sequence"], snapshot.sequence)
            self.assertEqual(binding["episode_id"], "episode-a")
            self.assertEqual(
                binding["image_shape"],
                [self.HEIGHT, self.WIDTH],
            )

            u, v = self._relative(center)
            list(
                track_object_distance(
                    ctx,
                    "unit-session",
                    capture_result["image_id"],
                    [{"name": "can", "u": u, "v": v}],
                )
            )
            self.assertTrue(ctx.result["ok"], ctx.result)
            self.assertEqual(
                ctx.result["tracked_object_distances"]["can"]["source_image_id"],
                capture_result["image_id"],
            )

            depth_path = capture_result["depth_path"]
            original_depth = np.load(depth_path, allow_pickle=False)
            tampered_depth = np.asarray(original_depth).copy()
            tampered_depth[0, 0] += np.float32(0.125)
            np.save(depth_path, tampered_depth)
            list(
                track_object_distance(
                    ctx,
                    "unit-session",
                    capture_result["image_id"],
                    [{"name": "tampered", "u": u, "v": v}],
                )
            )
            self.assertFalse(ctx.result["ok"])
            self.assertEqual(ctx.result["reason"], "digest_mismatch")
            self.assertIn("does not match", ctx.result["error"])
            self.assertFalse(ctx.result["stale_depth_published"])
            self.assertFalse(
                ctx.result["stale_xyz_in_robot_base_coord_published"]
            )
            self.assertNotIn("tracked_object_distances", ctx.result)
            self.assertEqual(ctx.result["cleared_names"], ["can"])
            self.assertEqual(manager.status()["registered_names"], [])
            np.save(depth_path, original_depth)

            rgb_path = capture_result["rgb_path"]
            original_rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            tampered_rgb = np.asarray(original_rgb).copy()
            tampered_rgb[0, 0, 1] ^= np.uint8(255)
            self.assertTrue(cv2.imwrite(rgb_path, tampered_rgb))
            list(
                track_object_distance(
                    ctx,
                    "unit-session",
                    capture_result["image_id"],
                    [{"name": "tampered-rgb", "u": u, "v": v}],
                )
            )
            self.assertFalse(ctx.result["ok"])
            self.assertIn("does not match", ctx.result["error"])
            self.assertEqual(ctx.result["cleared_names"], [])
            self.assertEqual(manager.status()["registered_names"], [])
            self.assertTrue(cv2.imwrite(rgb_path, original_rgb))

            meta_path = Path(rgb_path).with_suffix(".meta.json")
            original_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            tampered_meta = json.loads(json.dumps(original_meta))
            tampered_meta["camera"]["robot_relative_pose"]["pos"][0] += 0.1
            meta_path.write_text(
                json.dumps(tampered_meta),
                encoding="utf-8",
            )
            list(
                track_object_distance(
                    ctx,
                    "unit-session",
                    capture_result["image_id"],
                    [{"name": "tampered-pose", "u": u, "v": v}],
                )
            )
            self.assertFalse(ctx.result["ok"])
            self.assertIn("does not match", ctx.result["error"])
            self.assertEqual(ctx.result["cleared_names"], [])
            self.assertEqual(manager.status()["registered_names"], [])
            meta_path.write_text(
                json.dumps(original_meta),
                encoding="utf-8",
            )

            stale_snapshot = adapter.update(
                self._observation([(center, 1.10, self.can_texture)])
            )
            self.assertEqual(stale_snapshot.sequence, snapshot.sequence + 1)
            list(capture_head_camera(ctx, session_id="unit-session"))
            stale_capture = dict(ctx.result)
            self.assertFalse(
                stale_capture["track_object_distance_binding"]["ok"]
            )
            self.assertIn(
                "sequence",
                stale_capture["track_object_distance_binding"]["error"],
            )
            list(
                track_object_distance(
                    ctx,
                    "unit-session",
                    stale_capture["image_id"],
                    [{"name": "replacement", "u": u, "v": v}],
                )
            )
            self.assertFalse(ctx.result["ok"])
            self.assertEqual(ctx.result["cleared_names"], [])
            self.assertEqual(manager.status()["registered_names"], [])

    def test_runtime_hook_uses_one_filtered_snapshot_and_policy_odometry(self) -> None:
        manager = mock.Mock()
        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime.tracked_object_distances = manager
        runtime.server = SimpleNamespace(
            world=SimpleNamespace(
                episode_id=lambda: "episode-a",
                policy_local_base_pose=lambda: np.array([0.2, -0.1, 0.3]),
            )
        )
        snapshot = ObservationSnapshot(
            observation=self._observation([]),
            rejected_keys=[],
            received_ts=10.0,
            sequence=7,
        )

        runtime._update_tracked_object_distances(snapshot)

        frame = manager.ingest.call_args.args[0]
        self.assertEqual(frame.sequence, 7)
        self.assertEqual(frame.episode_id, "episode-a")
        self.assertEqual(frame.rgb.shape[:2], frame.depth_linear.shape)
        np.testing.assert_allclose(
            frame.camera_to_robot_base,
            np.eye(4),
            atol=0.0,
            rtol=0.0,
        )

    def test_runtime_hook_opts_into_borrowed_ingest_for_owned_adapter(self) -> None:
        manager = mock.Mock()
        manager._supports_borrowed_ingest = True
        runtime = OfficialPolicyRuntime.__new__(OfficialPolicyRuntime)
        runtime.adapter = SimpleNamespace(_official_camera_arrays_owned=True)
        runtime.tracked_object_distances = manager
        runtime.server = SimpleNamespace(
            world=SimpleNamespace(
                episode_id=lambda: "episode-a",
                policy_local_base_pose=lambda: np.array([0.0, 0.0, 0.0]),
            )
        )
        snapshot = ObservationSnapshot(
            observation=self._observation([]),
            rejected_keys=[],
            received_ts=10.0,
            sequence=8,
        )

        runtime._update_tracked_object_distances(snapshot)

        self.assertEqual(manager.ingest.call_args.kwargs, {"copy_frame": False})

    def test_test_only_http_route_submits_public_tool_and_lists_metadata(self) -> None:
        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-1"),
            wait_for_skill_result=mock.Mock(
                return_value={"ok": True, "tracked_names": ["can"]}
            ),
            cancel_current_skill=mock.Mock(),
        )
        runtime = SimpleNamespace(server=server)
        app = Flask(__name__)

        @app.get("/api/v2/tools")
        def api_v2_tools():
            return jsonify(
                {
                    "tool_version": "v2",
                    "tools": [
                        {
                            "name": "track_object_distance",
                            "args": [],
                            "desc": "stale generated metadata",
                        }
                    ],
                }
            )

        install_official_track_object_distance_routes(app, runtime)
        client = app.test_client()
        response = client.post(
            "/api/v2/track_object_distance",
            json={
                "session_id": "web",
                "image_id": "img_0007",
                "points": [{"name": "can", "u": 500, "v": 500}],
            },
        )
        self.assertEqual(response.status_code, 200)
        server.submit_skill.assert_called_once_with(
            "track_object_distance",
            {
                "session_id": "web",
                "image_id": "img_0007",
                "points": [{"name": "can", "u": 500, "v": 500}],
            },
        )
        server.wait_for_skill_result.assert_called_once_with(
            "track_object_distance",
            timeout_s=TRACK_OBJECT_DISTANCE_REPLAY_TIMEOUT_S,
            request_id="job-1",
        )
        metadata = client.get("/api/v2/tools").get_json()["tools"]
        self.assertEqual(
            sum(item["name"] == "track_object_distance" for item in metadata),
            1,
        )
        self.assertEqual(metadata[-1]["name"], "track_object_distance")
        right_adjust = next(
            item
            for item in metadata
            if item["name"] == "adjust_right_eef_pose_in_head_frame"
        )
        self.assertEqual(
            [arg["name"] for arg in right_adjust["args"][:6]],
            ["x", "y", "z", "forward", "leftward", "upward"],
        )
        right_args = {arg["name"]: arg for arg in right_adjust["args"]}
        self.assertEqual(right_args["x"]["coordinate_frame"], "robot_base")
        self.assertEqual(right_args["y"]["positive_direction"], "chassis_left")
        self.assertEqual(
            right_args["leftward"]["coordinate_frame"],
            "starting_head_camera",
        )
        self.assertTrue(
            right_adjust["translation_families_mutually_exclusive"]
        )
        self.assertIn("Never mix", right_adjust["desc"])
        image_arg = next(
            arg for arg in metadata[-1]["args"] if arg["name"] == "image_id"
        )
        self.assertEqual(image_arg["widget"], "image")
        self.assertTrue(image_arg["required"])
        point_arg = next(
            arg for arg in metadata[-1]["args"] if arg["name"] == "points"
        )
        self.assertEqual(point_arg["widget"], "named_multi_uv")
        self.assertEqual(point_arg["min_points"], 1)
        self.assertEqual(point_arg["max_points"], 32)
        self.assertEqual(point_arg["name_max_length"], 128)
        description = metadata[-1]["desc"]
        self.assertIn("xyz_in_robot_base_coord_m=[x, y, z]", description)
        self.assertIn("+X always points chassis-forward", description)
        self.assertIn("+Y chassis-left", description)
        self.assertIn("+Z chassis-up", description)

        cut_metadata = next(
            item for item in metadata if item["name"] == "cut_object"
        )
        cut_points = next(
            arg for arg in cut_metadata["args"] if arg["name"] == "points"
        )
        self.assertEqual(cut_points["min_points"], 2)
        self.assertEqual(cut_points["max_points"], 2)
        self.assertEqual(
            cut_points["fixed_names"],
            ["cutting_tool_point", "target_object_point"],
        )
        server.submit_skill.reset_mock()
        server.wait_for_skill_result.reset_mock()
        server.submit_skill.side_effect = ["job-cut", "job-cut-head"]
        server.wait_for_skill_result.side_effect = [
            {
                "ok": True,
                "point_coincidence_verified": True,
            },
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
        cut_response = client.post(
            "/api/v2/cut_object",
            json={
                "session_id": "web",
                "image_id": "img_0007",
                "points": [
                    {"name": "cutting_tool_point", "u": 450, "v": 500},
                    {"name": "target_object_point", "u": 550, "v": 500},
                ],
            },
        )
        self.assertEqual(cut_response.status_code, 200)
        self.assertEqual(
            server.submit_skill.call_args_list,
            [
                mock.call(
                    "cut_object",
                    {
                        "session_id": "web",
                        "image_id": "img_0007",
                        "points": [
                            {
                                "name": "cutting_tool_point",
                                "u": 450,
                                "v": 500,
                            },
                            {
                                "name": "target_object_point",
                                "u": 550,
                                "v": 500,
                            },
                        ],
                    },
                ),
                mock.call("capture_head_camera", {"session_id": "web"}),
            ],
        )
        self.assertEqual(cut_response.get_json()["image_id"], "img-cut-head")

    def test_action_return_images_publish_trackability_contract(self) -> None:
        app = Flask(__name__)

        @app.post("/api/v2/action_with_head_capture")
        def action_with_head_capture():
            return jsonify({
                "ok": True,
                "observation": {
                    "feed": "head",
                    "image_id": "img-head",
                    "rgb_main": "data:image/png;base64,AA==",
                    "track_object_distance_binding": {
                        "ok": True,
                        "session_id": "session-a",
                        "image_id": "img-head",
                        "frame_digest": "a" * 64,
                    },
                },
            })

        @app.post("/api/v2/action_with_wrist_capture")
        def action_with_wrist_capture():
            return jsonify({
                "ok": True,
                "observation": {
                    "feed": "left_wrist",
                    "image_id": "img-wrist",
                    "rgb_main": "data:image/png;base64,AA==",
                },
            })

        server = SimpleNamespace()
        install_official_track_object_distance_routes(
            app,
            SimpleNamespace(server=server),
        )
        client = app.test_client()

        head = client.post("/api/v2/action_with_head_capture").get_json()
        self.assertTrue(head["track_object_distance_binding"]["trackable"])
        self.assertIsNone(head["track_object_distance_binding"]["reason"])
        self.assertEqual(
            head["track_object_distance_binding"],
            head["observation"]["track_object_distance_binding"],
        )

        wrist = client.post("/api/v2/action_with_wrist_capture").get_json()
        self.assertFalse(wrist["track_object_distance_binding"]["trackable"])
        self.assertEqual(
            wrist["track_object_distance_binding"]["reason"],
            "unsupported_camera_role",
        )

    def test_http_binding_failure_is_400_and_never_publishes_stale_values(self) -> None:
        failure = {
            "ok": False,
            "reason": "replay_history_evicted",
            "failure_stage": "tracking",
            "stale_depth_published": False,
            "stale_xyz_in_robot_base_coord_published": False,
            "error": "capture replay history was evicted",
        }
        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-stale"),
            wait_for_skill_result=mock.Mock(return_value=failure),
        )
        app = Flask(__name__)
        install_official_track_object_distance_routes(
            app,
            SimpleNamespace(server=server),
        )

        response = app.test_client().post(
            "/api/v2/track_object_distance",
            json={
                "session_id": "web",
                "image_id": "img-old",
                "points": [{"name": "can", "u": 500, "v": 500}],
            },
        )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload["reason"], "replay_history_evicted")
        self.assertFalse(payload["stale_depth_published"])
        self.assertFalse(payload["stale_xyz_in_robot_base_coord_published"])
        self.assertNotIn("tracked_object_distances", payload)

    def test_http_long_replay_timeout_is_configurable_but_bounded(self):
        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-long"),
            wait_for_skill_result=mock.Mock(return_value={"ok": True}),
        )
        app = Flask(__name__)
        install_official_track_object_distance_routes(app, SimpleNamespace(server=server))
        client = app.test_client()
        body = {"session_id": "web", "image_id": "old-but-retained", "points": [{"name": "can", "u": 500, "v": 500}]}
        for timeout in (0.01, 60.0, 180.0, 600.0):
            response = client.post("/api/v2/track_object_distance", json={**body, "timeout_s": timeout})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(server.wait_for_skill_result.call_args.kwargs["timeout_s"], timeout)
        calls_before = server.submit_skill.call_count
        for timeout in (0, -1, 601, "nan", "inf", "invalid"):
            response = client.post("/api/v2/track_object_distance", json={**body, "timeout_s": timeout})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(server.submit_skill.call_count, calls_before)

    def test_http_timeout_cancels_only_the_submitted_request(self) -> None:
        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-track"),
            wait_for_skill_result=mock.Mock(
                side_effect=TimeoutError("track request timed out")
            ),
            cancel_current_skill=mock.Mock(),
        )
        runtime = SimpleNamespace(
            server=server,
            cancel_public_job=mock.Mock(return_value=True),
        )
        app = Flask(__name__)
        install_official_track_object_distance_routes(app, runtime)

        response = app.test_client().post(
            "/api/v2/track_object_distance",
            json={
                "session_id": "web",
                "image_id": "img_0007",
                "points": [{"name": "can", "u": 500, "v": 500}],
                "timeout_s": 0.01,
            },
        )

        self.assertEqual(response.status_code, 504)
        runtime.cancel_public_job.assert_called_once_with("job-track")
        server.cancel_current_skill.assert_not_called()

    def test_http_timeout_returns_late_committed_result_as_success(self) -> None:
        committed = {
            "ok": True,
            "job": "job-track",
            "tracked_names": ["can"],
        }
        server = SimpleNamespace(
            submit_skill=mock.Mock(return_value="job-track"),
            wait_for_skill_result=mock.Mock(
                side_effect=[
                    TimeoutError("boundary timeout"),
                    committed,
                ]
            ),
        )
        runtime = SimpleNamespace(
            server=server,
            cancel_public_job=mock.Mock(return_value=False),
        )
        app = Flask(__name__)
        install_official_track_object_distance_routes(app, runtime)

        response = app.test_client().post(
            "/api/v2/track_object_distance",
            json={
                "session_id": "web",
                "image_id": "img_0007",
                "points": [{"name": "can", "u": 500, "v": 500}],
                "timeout_s": 0.01,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), committed)
        runtime.cancel_public_job.assert_called_once_with("job-track")

    def test_human_ui_exposes_named_point_rows_and_builds_named_payload(self) -> None:
        app = Flask(__name__)

        @app.get("/")
        def index():
            return "<html><head></head><body>test interface</body></html>"

        install_track_object_distance_human_ui(app)
        client = app.test_client()
        page = client.get("/").get_data(as_text=True)
        self.assertIn("track_object_distance_human_ui.css", page)
        self.assertIn("track_object_distance_human_ui.js", page)

        script = client.get(
            "/__official__/assets/track_object_distance_human_ui.js"
        ).get_data(as_text=True)
        style = client.get(
            "/__official__/assets/track_object_distance_human_ui.css"
        ).get_data(as_text=True)
        self.assertIn('const WIDGET = "named_multi_uv"', script)
        self.assertIn('class="v2-named-row"', script)
        self.assertIn('data-v2-named-index="${index}"', script)
        self.assertIn("V2.multiPicks.push(point)", script)
        self.assertNotIn("V2.currentMedia", script)
        self.assertIn("V2.namedTrackImageId", script)
        self.assertIn("body.image_id = V2.namedTrackImageId", script)
        self.assertIn('mediaKind !== "head"', script)
        self.assertIn("dataset.loadedImageId", script)
        self.assertIn("dataset.loadedMediaKind", script)
        self.assertIn("mediaLoadToken", script)
        self.assertIn("track_object_distance_binding", script)
        self.assertIn("binding.ok !== true", script)
        self.assertIn(
            'name: String(point.name || "").trim()',
            script,
        )
        self.assertIn("名称不能重复", script)
        self.assertIn("请填写第 ${index + 1} 行名称", script)
        self.assertIn("const fixedNames", script)
        self.assertIn("spec.fixed_names", script)
        self.assertIn("该工具的点名称固定", script)
        self.assertIn(".v2-named-row", style)

    def test_module_has_no_simulator_or_production_tool_dependency(self) -> None:
        import behavior_interface_eval_test.tool.official_v2.tracked_object_distance as module

        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        forbidden = (
            "omnigibson",
            "behavior_interface.skills",
            "behavior_interface.tool",
            "behavior_interface.world_api",
            "behavior_interface.server",
        )
        self.assertFalse(
            any(name == prefix or name.startswith(prefix + ".") for name in imports for prefix in forbidden)
        )


if __name__ == "__main__":
    unittest.main()
